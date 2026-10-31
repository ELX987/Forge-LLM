"""PTQ tests: RTN, GPTQ (exactness properties), AWQ, SmoothQuant, FP4 formats, packed modules, mixed precision, serialisation."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from forge.compress import bitalloc
from forge.compress.calibrate import get_calibration_tokens
from forge.compress.packed import (
    Int8W8A8Linear,
    IntLinear,
    MXFP4Linear,
    NVFP4Linear,
    apply_block_hadamard,
    hadamard_matrix,
    packed_class,
)
from forge.compress.ptq import PTQConfig, gptq_core, quantize_model, quantize_weight, _int_quantizers
from forge.compress.serialize import load_quantized, save_quantized
from forge.core.errors import ConfigError
from forge.formats.int_quant import dequantize_int, quantize_int
from tests.compress_helpers import logits_of, make_lm, make_moe_lm, rand_tokens, rel_err


# ------------------------------------------------------------------------------------------------ packed modules
@pytest.mark.parametrize("bits,sym", [(2, True), (3, False), (4, True), (4, False), (8, True)])
def test_int_linear_roundtrip_and_registry(bits, sym):
    torch.manual_seed(0)
    lin = nn.Linear(96, 20, bias=True)
    m = IntLinear.from_dense(lin, bits=bits, group_size=32, sym=sym)
    assert type(m).kind == f"w{bits}a16"
    q, s, z = quantize_int(lin.weight.detach(), bits, 32, sym)
    ref = dequantize_int(q, s, z, 32)
    assert torch.allclose(m.dequantize(torch.float32), ref, atol=1e-6)
    x = torch.randn(3, 5, 96)
    assert torch.allclose(m(x), nn.functional.linear(x, ref, lin.bias), atol=1e-4)
    # empty module of the same spec can load the state dict (artifact round trip)
    m2 = type(m).from_shape(96, 20, True, dtype=torch.float32, **m.quant_spec())
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2(x), m(x))
    assert set(m.state_dict()) >= {"qweight", "scales", "bias"} and (("zeros" in m.state_dict()) == (not sym))
    from forge.models.linear import get_linear_cls
    assert get_linear_cls(type(m).kind) is type(m)


def test_int_linear_bytes_are_4bit():
    m = IntLinear.from_dense(nn.Linear(256, 64, bias=False), bits=4, group_size=128)
    assert m.qweight.shape == (64, 128) and m.qweight.dtype == torch.uint8
    assert m.packed_nbytes() == 64 * 128 + 64 * 2 * 4  # payload + fp32 scales (model dtype float32 here)


def test_input_scale_and_rotation_are_exact():
    torch.manual_seed(0)
    lin = nn.Linear(64, 16, bias=False)
    s = torch.rand(64) + 0.5
    # W' = W*s, input_scale = 1/s  =>  same function
    w2 = lin.weight.detach() * s
    q, sc, z = quantize_int(w2, 8, 64, True)
    m = IntLinear.from_codes(q, sc, None, bits=8, group_size=64, input_scale=1.0 / s)
    x = torch.randn(4, 64)
    assert rel_err(m(x), lin(x)) < 1e-4
    # block Hadamard
    h = hadamard_matrix(16)
    assert torch.allclose(h @ h.t(), torch.eye(16), atol=1e-6)
    xr = apply_block_hadamard(x, 16)
    assert torch.allclose(xr.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    wr = apply_block_hadamard(lin.weight.detach(), 16)
    q, sc, z = quantize_int(wr, 8, 64, True)
    mr = IntLinear.from_codes(q, sc, None, bits=8, group_size=64, rot_block=16)
    assert rel_err(mr(x), lin(x)) < 1e-4


def test_fp4_modules():
    torch.manual_seed(0)
    lin = nn.Linear(64, 24, bias=True)
    x = torch.randn(5, 64)
    ref = lin(x)
    nv = NVFP4Linear.from_dense(lin)
    mx = MXFP4Linear.from_dense(lin)
    assert nv.kind == "nvfp4" and mx.kind == "mxfp4"
    assert rel_err(nv(x), ref) < 0.03 and rel_err(mx(x), ref) < 0.05
    assert nv.block_scales.dtype == torch.float8_e4m3fn
    nv2 = NVFP4Linear.from_shape(64, 24, True, dtype=torch.float32, **nv.quant_spec())
    nv2.load_state_dict(nv.state_dict())
    assert torch.equal(nv2(x), nv(x))
    w4a4 = NVFP4Linear.from_dense(lin, act_quant="dynamic")
    assert rel_err(w4a4(x), ref) < 0.1
    assert nv.packed_nbytes() < lin.weight.numel()  # < 1 byte per weight


def test_int8_w8a8_module():
    lin = nn.Linear(64, 16)
    m = Int8W8A8Linear.from_dense(lin)
    x = torch.randn(6, 64)
    assert rel_err(m(x), lin(x)) < 1e-3


# ------------------------------------------------------------------------------------------------ GPTQ core properties
def _rand_hessian(k, n_tokens=512, seed=0, outliers=True):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_tokens, k, generator=g)
    if outliers:
        X[:, : k // 8] *= 6
    X = X @ torch.randn(k, k, generator=g) * 0.3 + X  # correlated
    return 2.0 / n_tokens * X.t() @ X, X


def _proxy(W, Wq, H):
    d = W - Wq
    return float(((d @ H) * d).sum())


def test_gptq_beats_rtn_on_proxy_loss():
    torch.manual_seed(0)
    W = torch.randn(48, 128)
    H, _ = _rand_hessian(128)
    cfg = PTQConfig(method="gptq", bits=3, group_size=32)
    rtn = quantize_weight(W, None, cfg.model_copy(update={"method": "rtn"}), sdtype=torch.float32)
    gp = quantize_weight(W, H, cfg, sdtype=torch.float32)
    lr, lg = _proxy(W, rtn.deq, H), _proxy(W, gp.deq, H)
    assert lg < 0.8 * lr, (lg, lr)
    cfg2 = cfg.model_copy(update={"act_order": True})
    ga = quantize_weight(W, H, cfg2, sdtype=torch.float32)
    assert _proxy(W, ga.deq, H) < 0.9 * lr
    assert ga.q.shape == W.shape and ga.scales.shape == (48, 4)
    # dequantising the stored parts reproduces .deq (act-order stores plain contiguous groups)
    assert torch.allclose(dequantize_int(ga.q, ga.scales, ga.zeros, 32, bits=3), ga.deq, atol=1e-5)


def test_gptq_identity_hessian_equals_rtn():
    """With H proportional to I there is no cross-column compensation: GPTQ must reproduce RTN codes (static group scales)."""
    torch.manual_seed(1)
    W = torch.randn(16, 64)
    cfg = PTQConfig(method="gptq", bits=4, group_size=32, static_groups=True)
    gp = quantize_weight(W, torch.eye(64), cfg, sdtype=torch.float32)
    q, s, z = quantize_int(W, 4, 32, True)
    assert torch.equal(gp.q, q)


@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_lazy_block_updates_are_equivalent(act_order):
    """Lazy batching only reorders arithmetic: block_size must not change the result (beyond float rounding)."""
    torch.manual_seed(2)
    W = torch.randn(24, 64)
    H, _ = _rand_hessian(64)
    outs = []
    for bs in (1, 8, 32, 64):
        pf, qf = _int_quantizers(4, True, torch.float32, False)
        codes, (s, z), deq, _ = gptq_core(W, H, group=32, block_size=bs, act_order=act_order, damp=0.01, static_groups=act_order, params_fn=pf, quant_fn=qf)
        outs.append((codes, deq))
    for codes, deq in outs[1:]:
        assert (codes != outs[0][0]).float().mean() < 0.01
        assert _proxy(W, deq, H) == pytest.approx(_proxy(W, outs[0][1], H), rel=0.02)


def test_gptq_dead_columns_and_damping_retry():
    torch.manual_seed(3)
    W = torch.randn(8, 32)
    H, _ = _rand_hessian(32)
    H[5, :] = 0
    H[:, 5] = 0
    cfg = PTQConfig(method="gptq", bits=4, group_size=32)
    gp = quantize_weight(W, H, cfg, sdtype=torch.float32)
    assert torch.all(gp.deq[:, 5] == 0)  # dead input column carries no signal -> zeroed
    # rank deficient Hessian (fewer tokens than columns) still factorises thanks to damping
    X = torch.randn(4, 32)
    Hr = 2 / 4 * X.t() @ X
    gp2 = quantize_weight(W, Hr, cfg, sdtype=torch.float32)
    assert torch.isfinite(gp2.deq).all()


def test_gptq_asym_and_clip_search():
    torch.manual_seed(4)
    W = torch.randn(16, 64) + 0.5
    H, _ = _rand_hessian(64)
    cfg = PTQConfig(method="gptq", bits=4, group_size=32, sym=False, clip_search=True)
    gp = quantize_weight(W, H, cfg, sdtype=torch.float32)
    assert gp.zeros is not None
    assert torch.allclose(dequantize_int(gp.q, gp.scales, gp.zeros, 32), gp.deq, atol=1e-5)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_fp4_gptq_beats_rtn(fmt):
    torch.manual_seed(5)
    W = torch.randn(32, 128)
    H, _ = _rand_hessian(128)
    cfg = PTQConfig(method="mrgptq", format=fmt)
    rtn = quantize_weight(W, None, cfg.model_copy(update={"method": "rtn"}))
    gp = quantize_weight(W, H, cfg)
    assert _proxy(W, gp.deq, H) < 0.85 * _proxy(W, rtn.deq, H)
    if fmt == "nvfp4":
        assert gp.nvfp4.block_scales_e4m3.dtype == torch.float8_e4m3fn
        assert torch.allclose(gp.nvfp4.dequantize(), gp.deq, atol=1e-5)
        f46 = quantize_weight(W, H, cfg.model_copy(update={"four_over_six": True}))
        assert torch.isfinite(f46.deq).all()


def test_fp4_rotation_effective_weight_matches_module():
    torch.manual_seed(6)
    W = torch.randn(16, 64)
    H, _ = _rand_hessian(64)
    lin = nn.Linear(64, 16, bias=False)
    lin.weight.data = W
    cfg = PTQConfig(method="mrgptq", format="nvfp4", rotate="block_hadamard")
    qw = quantize_weight(W, H, cfg)
    assert qw.rot_block == 16
    from forge.compress.ptq import build_module
    mod = build_module(qw, lin, cfg, torch.float32)
    x = torch.randn(5, 64)
    assert torch.allclose(mod(x), x @ qw.deq.t(), atol=1e-4)  # module == effective weight in the original input space


# ------------------------------------------------------------------------------------------------ model level
def test_rtn_model_report_and_forward():
    m = make_lm()
    t = rand_tokens()
    ref = logits_of(m, t)
    qm = quantize_model(m, "rtn", 8, 32)
    r = qm.report
    assert len(r.layers) == 14  # 2 blocks x 7 linears; lm_head excluded
    assert all(x.kind == "w8a16" for x in r.layers)
    assert r.compression > 2.0 and r.bytes_after < r.bytes_before
    assert rel_err(logits_of(m, t), ref) < 1e-3
    assert type(m.lm_head).__name__ == "DenseLinear"
    d = r.to_dict()
    assert d["n_layers"] == 14 and "layers" in d


def test_gptq_model_beats_rtn_and_is_sequential():
    t = rand_tokens(16, 32, seed=3)
    held = rand_tokens(8, 32, seed=9)
    ref = logits_of(make_lm(outliers=True), held)
    errs = {}
    for method in ("rtn", "gptq"):
        m = make_lm(outliers=True)
        quantize_model(m, method, 3, 32, calib=t, batch_size=4)
        errs[method] = rel_err(logits_of(m, held), ref)
    assert errs["gptq"] < errs["rtn"], errs
    # act-order variant also runs through the model driver
    m = make_lm(outliers=True)
    qm = quantize_model(m, "gptq", 3, 32, act_order=True, calib=t, true_sequential="block")
    assert rel_err(logits_of(m, held), ref) < errs["rtn"] * 1.2
    assert all(r.proxy_loss is not None for r in qm.report.layers)


def test_awq_improves_on_outlier_model_and_folds():
    t = rand_tokens(16, 32, seed=3)
    held = rand_tokens(8, 32, seed=9)
    ref = logits_of(make_lm(outliers=True), held)
    m0 = make_lm(outliers=True)
    quantize_model(m0, "rtn", 4, 32)
    e_rtn = rel_err(logits_of(m0, held), ref)
    m1 = make_lm(outliers=True)
    qm = quantize_model(m1, "awq", 4, 32, calib=t, awq_n_grid=10)
    e_awq = rel_err(logits_of(m1, held), ref)
    assert e_awq < e_rtn, (e_awq, e_rtn)
    assert any("folded" in n for r in qm.report.layers for n in r.notes)


def test_smoothquant_w8a8_folding_is_exact_without_act_quant():
    t = rand_tokens(16, 32, seed=3)
    held = rand_tokens(8, 32, seed=9)
    ref = logits_of(make_lm(outliers=True), held)
    m = make_lm(outliers=True)
    qm = quantize_model(m, "smoothquant", calib=t, act_quant="none")
    assert all(r.kind == "w8a8_int8" for r in qm.report.layers)
    notes = [n for r in qm.report.layers for n in r.notes]
    assert any("folded" in n for n in notes)
    assert rel_err(logits_of(m, held), ref) < 2e-3  # int8 weights only: scale migration (folded into norms) is exact
    # with dynamic activation quantisation the error stays small and is lower with smoothing than without
    m2 = make_lm(outliers=True)
    quantize_model(m2, "smoothquant", calib=t, sq_alpha=0.5)
    m3 = make_lm(outliers=True)
    quantize_model(m3, "smoothquant", calib=t, sq_alpha=0.0)
    assert rel_err(logits_of(m2, held), ref) <= rel_err(logits_of(m3, held), ref) * 1.05


def test_nvfp4_model_and_w4a4_emulation():
    t = rand_tokens(16, 32, seed=3)
    held = rand_tokens(8, 32, seed=9)
    ref = logits_of(make_lm(), held)
    m = make_lm()
    qm = quantize_model(m, "mrgptq", format="nvfp4", calib=t)
    assert all(r.kind == "nvfp4" for r in qm.report.layers)
    assert qm.report.avg_bits < 4.7
    assert rel_err(logits_of(m, held), ref) < 0.25


def test_fp8_format_uses_models_fp8linear():
    m = make_lm()
    t = rand_tokens()
    ref = logits_of(m, t)
    qm = quantize_model(m, "rtn", format="fp8")
    assert all(r.kind == "fp8" for r in qm.report.layers)
    assert rel_err(logits_of(m, t), ref) < 0.02


def test_per_layer_bits_override():
    m = make_lm()
    qm = quantize_model(m, "rtn", 4, 32, per_layer_bits={r"down_proj": 8})
    kinds = {r.name.split(".")[-1]: r.kind for r in qm.report.layers}
    assert kinds["down_proj"] == "w8a16" and kinds["q_proj"] == "w4a16"


def test_moe_model_gptq_routers_stay_dense_and_cold_experts_fall_back():
    m = make_moe_lm()
    t = rand_tokens(4, 16, seed=3)  # few tokens -> some experts get none
    ref = logits_of(m, t)
    qm = quantize_model(copy.deepcopy(m), "gptq", 4, 32, calib=t, batch_size=2)
    names = [r.name for r in qm.report.layers]
    assert any(".experts." in n for n in names)
    assert not any(n.endswith(".mlp.gate") for n in names)
    assert isinstance(qm.model.layers[0].mlp.gate, nn.Linear)  # router untouched
    out = logits_of(qm.model, t)
    assert torch.isfinite(out).all() and rel_err(out, ref) < 0.5


def test_calibration_sets_deterministic():
    a, _ = get_calibration_tokens("synthetic", None, vocab_size=128, n_seqs=4, seq_len=16, seed=3)
    b, _ = get_calibration_tokens("synthetic", None, vocab_size=128, n_seqs=4, seq_len=16, seed=3)
    assert torch.equal(a, b) and a.max() < 128
    c, labels = get_calibration_tokens("builtin-mix", None, tokenizer=None, vocab_size=256, n_seqs=10, seq_len=32, seed=0)
    assert c.shape == (10, 32) and len(set(labels)) == 4


# ------------------------------------------------------------------------------------------------ mixed precision
def test_bit_allocation_matches_brute_force():
    import random

    rnd = random.Random(0)
    names = [f"l{i}" for i in range(5)]
    npar = {n: rnd.choice([1000, 2000, 4000]) for n in names}
    sens = {n: {b: rnd.uniform(1, 5) * 4.0 ** (-b / 2) * (1 + i) for b in (2, 3, 4, 8)} for i, n in enumerate(names)}
    total = sum(npar.values())
    for avg in (3.2, 4.1, 5.0):
        budget = avg * total / 8
        opt = bitalloc.brute_force(sens, npar, budget, 128)
        bits, info = bitalloc.allocate_bits(sens, npar, budget, group_size=128, method="dp", resolution=20000)
        loss = lambda ch: sum(sens[n][ch[n]] for n in names)
        assert info["bytes"] <= budget + 1e-6
        assert loss(bits) <= loss(opt) * 1.001 + 1e-9
        g, ginfo = bitalloc.allocate_bits(sens, npar, budget, group_size=128, method="greedy")
        assert ginfo["bytes"] <= budget + 1e-6 and loss(g) <= loss(opt) * 1.4
    with pytest.raises(ConfigError):
        bitalloc.allocate_bits(sens, npar, 10.0)


def test_mixed_precision_model_respects_budget():
    t = rand_tokens(8, 32, seed=3)
    m = make_lm(outliers=True)
    qm = quantize_model(m, "rtn", group_size=32, mixed_precision=True, avg_bits=5.5, candidate_bits=[3, 4, 8], calib=t)
    alloc = qm.report.calib["bit_allocation"]
    assert alloc["avg_bits"] <= 5.5 + 1e-6 and len(alloc["histogram"]) >= 2
    bits_used = {r.bits for r in qm.report.layers}
    assert len(bits_used) >= 2


# ------------------------------------------------------------------------------------------------ serialisation
def test_save_load_roundtrip(tmp_path):
    t = rand_tokens(8, 32, seed=3)
    m = make_lm()
    qm = quantize_model(m, "gptq", 4, 32, calib=t)
    before = logits_of(m, t)
    out = save_quantized(m, tmp_path / "q", report=qm.report.to_dict())
    assert (out / "config.json").exists() and (out / "forge_compress.json").exists()
    m2, meta = load_quantized(out)
    assert torch.equal(logits_of(m2, t), before)
    assert meta["manifest"] and "w4a16" in {v["kind"] for v in meta["manifest"].values()}
    from forge.models.linear import quantization_manifest
    assert set(quantization_manifest(m)) <= set(meta["manifest"])


def test_save_load_tied_embeddings_and_shards(tmp_path):
    m = make_lm(tie_word_embeddings=True)
    quantize_model(m, "rtn", 8, 32)
    t = rand_tokens()
    before = logits_of(m, t)
    out = save_quantized(m, tmp_path / "q2", shard_bytes=20_000)
    assert (out / "model.safetensors.index.json").exists()
    m2, _ = load_quantized(out)
    assert torch.equal(logits_of(m2, t), before)
    assert m2.lm_head.weight is m2.embed_tokens.weight


def test_packed_class_lookup():
    assert packed_class("nvfp4") is NVFP4Linear
    with pytest.raises(ConfigError):
        packed_class("nope")
