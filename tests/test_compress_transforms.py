"""Rotation, KV quantisation, low-rank, delta compression, lossless coding."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from forge.compress import delta as D
from forge.compress import kvquant as KV
from forge.compress import lossless as LL
from forge.compress import lowrank as LR
from forge.compress import rotation as R
from forge.compress.packed import hadamard_matrix
from forge.compress.ptq import quantize_model
from forge.compress.serialize import load_quantized, save_quantized
from forge.core.errors import ConfigError
from forge.formats.int_quant import fake_quant_int
from tests.compress_helpers import logits_of, make_lm, make_moe_lm, rand_tokens, rel_err

TOK = rand_tokens(8, 24, seed=5)


# ------------------------------------------------------------------------------------------------ rotation
def test_fwht_matches_matrix_and_orthogonal_constructions():
    x = torch.randn(5, 64)
    assert torch.allclose(R.fwht(x), x @ hadamard_matrix(64), atol=1e-5)
    for n in (64, 48, 96):
        q = R.random_orthogonal_hadamard(n, seed=3, dtype=torch.float64)
        assert torch.allclose(q.t() @ q, torch.eye(n, dtype=torch.float64), atol=1e-10)
    assert not torch.equal(R.random_orthogonal_hadamard(64, 1), R.random_orthogonal_hadamard(64, 2))
    with pytest.raises(ConfigError):
        R.random_orthogonal_hadamard(7)
    a = torch.randn(16, 16) * 0.1
    qc = R.cayley(a)
    assert torch.allclose(qc.t() @ qc, torch.eye(16), atol=1e-4)


@pytest.mark.parametrize("kwargs", [{}, {"tie_word_embeddings": True}, {"num_layers": 3, "attention_bias": True}])
def test_rotation_preserves_function(kwargs):
    m = make_lm(outliers=True, **kwargs)
    ref = logits_of(m, TOK)
    rep = R.rotate_model(m, R.RotationConfig(seed=1))
    assert rep["orthogonality_error"] < 1e-12 and rep["norms_folded"] == 2 * len(m.layers) + 1
    assert rel_err(logits_of(m, TOK), ref) < 1e-9
    assert all((blk.input_layernorm.weight == 1).all() for blk in m.layers)  # norms became weightless
    if kwargs.get("tie_word_embeddings"):
        assert m.lm_head.weight is not m.embed_tokens.weight  # head had to be untied to absorb the final norm


def test_rotation_moe_and_online_hadamard():
    m = make_moe_lm(outliers=True)
    ref = logits_of(m, TOK)
    R.rotate_model(m, R.RotationConfig(seed=2))
    assert rel_err(logits_of(m, TOK), ref) < 1e-9
    m2 = make_lm(outliers=True)
    ref2 = logits_of(m2, TOK)
    rep = R.rotate_model(m2, R.RotationConfig(online_hadamard_down=True, online_block=32))
    assert rep["online_hadamard_layers"] == 2
    assert isinstance(m2.layers[0].mlp.down_proj, R.OnlineHadamardLinear)
    assert rel_err(logits_of(m2, TOK), ref2) < 1e-9
    # PTQ keeps the online rotation as rot_block of the packed module and quantises in the rotated basis
    qm = quantize_model(m2, "rtn", 8, 32)
    assert m2.layers[0].mlp.down_proj.rot_block == 32
    assert rel_err(logits_of(m2, TOK), ref2) < 1e-3
    assert any("online Hadamard" in n for r in qm.report.layers for n in r.notes)


def test_rotation_helps_int4_on_outlier_channels():
    """A weight with a few huge input columns quantises badly per row group; a Hadamard spreads the outliers."""
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    w[:, 3] *= 30
    q = R.random_orthogonal_hadamard(64, 0)
    e_plain = ((w - fake_quant_int(w, 4, 64, True)).pow(2).sum() / w.pow(2).sum()).item()
    wr = w @ q
    e_rot = ((wr - fake_quant_int(wr, 4, 64, True)).pow(2).sum() / wr.pow(2).sum()).item()
    assert e_rot < 0.7 * e_plain


def test_learned_rotation_reduces_objective_and_stays_orthogonal():
    torch.manual_seed(0)
    ws = [torch.randn(64, 64) * (1 + 6 * (torch.rand(64) > 0.9).float()) for _ in range(3)]
    q0 = R.random_orthogonal_hadamard(64, 0)
    q, info = R.learn_rotation(ws, q0, steps=40, lr=0.02, bits=4, group=32)
    assert info["after"] < info["before"]
    assert torch.allclose(q.t() @ q, torch.eye(64), atol=1e-4)
    m = make_lm(outliers=True)
    ref = logits_of(m, TOK)
    rep = R.rotate_model(m, R.RotationConfig(mode="learned", learn_steps=5, learn_group=32), TOK)
    assert "objective" in rep and rel_err(logits_of(m, TOK), ref) < 1e-6


# ------------------------------------------------------------------------------------------------ KV
def _kv(b=1, h=2, t=200, d=32, seed=0, key_outliers=True):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(b, h, t, d, generator=g)
    if key_outliers:
        k[..., [1, 7]] *= 12  # KIVI's premise: a few key channels carry outliers
    return k, torch.randn(b, h, t, d, generator=g)


def test_kivi_cache_window_and_error_ordering():
    k, v = _kv()
    c4 = KV.KiviKVCache(4, 32, 32, residual_window=64)
    for i in range(0, 200, 25):  # token-by-token-ish appends
        c4.append(k[:, :, i : i + 25], v[:, :, i : i + 25])
    assert c4.seq_len == 200 and 64 <= c4.residual_len < 96 and len(c4.chunks) == (200 - c4.residual_len) // 32
    kd, vd = c4.get()
    assert kd.shape == k.shape
    r = c4.residual_len
    assert torch.equal(kd[:, :, -r:], k[:, :, -r:]) and torch.equal(vd[:, :, -r:], v[:, :, -r:])  # window is exact
    e4 = ((kd - k).pow(2).sum() / k.pow(2).sum()).sqrt().item()
    c2 = KV.KiviKVCache(2, 32, 32, residual_window=64)
    c2.append(k, v)
    k2, _ = c2.get()
    e2 = ((k2 - k).pow(2).sum() / k.pow(2).sum()).sqrt().item()
    assert e4 < 0.1 < e2 or e4 < e2
    assert c4.nbytes() < 0.7 * c4.fp16_nbytes()


def test_kivi_per_channel_keys_beat_per_token_keys():
    k, _ = _kv(t=96)
    from forge.formats.int_quant import dequantize_int, quantize_int

    q, s, z = quantize_int(k, 4, 32, sym=False, dim=2)  # per channel over 32-token groups
    e_chan = (dequantize_int(q, s, z, 32, dim=2) - k).pow(2).sum().item()
    q, s, z = quantize_int(k, 4, 32, sym=False, dim=3)  # per token over 32-channel groups
    e_tok = (dequantize_int(q, s, z, 32, dim=3) - k).pow(2).sum().item()
    assert e_chan < e_tok


def test_kv_accuracy_report_orders_schemes():
    k, v = _kv(t=128)
    rows = {r["scheme"]: r for r in KV.kv_accuracy_report(k, v)}
    assert set(rows) == {"fp16", "fp8_e4m3", "fp8_e5m2", "int8", "kivi4", "kivi2"}
    e = {n: r["attn_out_rel_err"] for n, r in rows.items()}
    assert e["fp16"] < e["int8"] < e["fp8_e4m3"] < e["fp8_e5m2"]  # mantissa bits: 7+ > 3 > 2
    assert e["int8"] < e["kivi4"] < e["kivi2"] and e["fp8_e4m3"] < e["kivi2"]
    assert rows["kivi4"]["quantized_token_fraction"] > 0.5
    assert rows["fp8_e4m3"]["compression_vs_fp16"] == pytest.approx(2.0, rel=0.1)
    assert rows["kivi4"]["compression_vs_fp16"] > rows["fp8_e4m3"]["compression_vs_fp16"]
    assert rows["kivi4"]["gated"] and not rows["fp8_e4m3"]["gated"]
    assert all(r["provenance"] == "measured" for r in rows.values())
    cfg = KV.KVQuantConfig(scheme="kivi", bits=4)
    assert KV.bytes_per_token_per_layer(cfg, 2, 32, 4096) < KV.bytes_per_token_per_layer(KV.KVQuantConfig(scheme="fp8_e4m3"), 2, 32)
    # reference attention vs torch sdpa
    q = torch.randn(1, 4, 5, 32)
    out = KV.reference_attention(q, k, v)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=False,
                                                            attn_mask=torch.ones(5, 128, dtype=torch.bool).tril(128 - 5))
    assert torch.allclose(out, sdpa, atol=1e-5)


def test_model_kv_report():
    m = make_lm()
    rep = KV.model_kv_report(m, rand_tokens(4, 64), max_layers=2)
    assert rep["pre_rope_proxy"] and {r["scheme"] for r in rep["rows"]} >= {"fp8_e4m3", "kivi4"}


# ------------------------------------------------------------------------------------------------ low rank
def test_breakeven_rank():
    assert LR.breakeven_rank(64, 64) == 31  # r < n/2
    assert LR.breakeven_rank(100, 50) == 33  # r < 33.33
    r = LR.breakeven_rank(300, 200)
    assert r * (300 + 200) < 300 * 200 <= (r + 1) * (300 + 200)


def test_whitened_svd_loss_equals_discarded_energy_and_beats_plain():
    torch.manual_seed(0)
    n, m, T = 32, 24, 400
    mix = torch.randn(n, n)
    X = (torch.randn(T, n) * torch.logspace(0, -1.5, n)) @ mix  # anisotropic activations
    W = torch.randn(m, n)
    cov = X.t() @ X
    r = 8
    A, B, S = LR.whitened_svd(W, cov, r, damp=0.0)
    err = ((W - A @ B) @ X.t()).pow(2).sum().item()
    assert err == pytest.approx(S[r:].pow(2).sum().item(), rel=2e-3)
    A0, B0, _ = LR.whitened_svd(W, None, r)
    err_plain = ((W - A0 @ B0) @ X.t()).pow(2).sum().item()
    assert err < err_plain
    # full rank reproduces W
    Af, Bf, _ = LR.whitened_svd(W, cov, None, damp=0.0)
    assert torch.allclose(Af @ Bf, W, atol=1e-3)


def test_rank_allocation_respects_budget_and_breakeven():
    torch.manual_seed(1)
    shapes = {"a": (64, 64), "b": (128, 64), "c": (64, 128)}
    spectra = {"a": torch.logspace(0, -3, 64), "b": torch.logspace(0, -1, 64), "c": torch.ones(64)}
    ranks, info = LR.allocate_ranks(spectra, shapes, 0.5, min_rank=2)
    kept = sum(r * sum(shapes[n]) for n, r in ranks.items())
    assert kept <= 0.5 * sum(m * n for m, n in shapes.values()) + 1e-6 and info["ratio"] <= 0.5 + 1e-9
    assert ranks["a"] < ranks["c"]  # fast-decaying spectrum needs fewer ranks than a flat one
    uni, uinfo = LR.allocate_ranks(spectra, shapes, 0.5, mode="uniform", min_rank=2)
    assert info["mean_rel_energy_lost"] <= uinfo["mean_rel_energy_lost"] + 1e-9
    big, binfo = LR.allocate_ranks(spectra, shapes, 1.0, min_rank=2)
    assert all(big[n] <= LR.breakeven_rank(*shapes[n]) for n in big) and set(binfo["breakeven_capped"]) == set(shapes)
    ok, _ = LR.allocate_ranks(spectra, shapes, 1.0, min_rank=2, allow_above_breakeven=True)
    assert max(ok.values()) > max(big.values())


def test_lowrank_model_compression_healing_and_roundtrip(tmp_path):
    m = make_lm()
    teacher = copy.deepcopy(m)
    ref = logits_of(m, TOK)
    info = LR.lowrank_compress_model(m, LR.LowRankConfig(target_ratio=0.8, n_seqs=16, seq_len=32, heal_steps=0), "synthetic", teacher=teacher)
    assert info["bytes_after"] < info["bytes_before"] and info["ratio"] <= 0.8 + 1e-6
    assert all(isinstance(blk.mlp.down_proj, LR.LowRankLinear) for blk in m.layers)
    assert type(m.lm_head).__name__ == "DenseLinear"
    toks = rand_tokens(16, 32, seed=2)
    def kl(a, b):
        return torch.nn.functional.kl_div(torch.log_softmax(a, -1).reshape(-1, 128), torch.softmax(b, -1).reshape(-1, 128), reduction="batchmean").item()

    kl0 = kl(logits_of(m, TOK), ref)
    h = LR.lowrank_heal(m, toks, steps=40, lr=3e-4, teacher=teacher, batch_size=4)
    assert h["loss_last"] < h["loss_first"]
    assert kl(logits_of(m, TOK), ref) < kl0  # healing generalises to held-out tokens
    assert all(p.requires_grad for p in m.parameters())
    out = save_quantized(m, tmp_path / "lr")
    m2, _ = load_quantized(out)
    assert torch.equal(logits_of(m2, TOK), logits_of(m, TOK))


def test_lowrank_whitening_beats_plain_on_model():
    toks = rand_tokens(16, 32, seed=2)
    ref = logits_of(make_lm(outliers=True), TOK)
    errs = {}
    for whiten in (True, False):
        m = make_lm(outliers=True)
        LR.lowrank_compress_model(m, LR.LowRankConfig(target_ratio=0.6, whiten=whiten, n_seqs=16, seq_len=32), toks)
        errs[whiten] = rel_err(logits_of(m, TOK), ref)
    assert errs[True] < errs[False]


# ------------------------------------------------------------------------------------------------ delta
def _finetune(base, scale=0.02, seed=1):
    ft = copy.deepcopy(base)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in ft.named_parameters():
            if p.ndim == 2 and n.startswith("layers."):
                p.add_(torch.randn(p.shape, generator=g) * scale)
            elif n.startswith(("embed", "norm")):
                p.add_(torch.randn(p.shape, generator=g) * scale * 0.5)
    return ft


def test_bitdelta_sizes_apply_merge_and_calibration(tmp_path):
    base = make_lm()
    ft = _finetune(base)
    d = D.compute_delta(base.state_dict(), ft.state_dict(), D.DeltaConfig(method="bitdelta"))
    e = d.entries["layers.0.self_attn.q_proj"]
    delta_true = ft.layers[0].self_attn.q_proj.weight - base.layers[0].self_attn.q_proj.weight
    assert e.kind == "bitdelta" and e.alpha.item() == pytest.approx(delta_true.abs().mean().item(), rel=1e-5)
    rec = D.reconstruct_matrix(e)
    assert (rec - delta_true).pow(2).sum() < delta_true.pow(2).sum()
    sz = d.size_report(sum(t.numel() * 2 for t in ft.state_dict().values()))
    assert sz["matrix_bytes"] < 0.07 * sum(ft.state_dict()[k + ".weight"].numel() * 2 for k, en in d.entries.items() if en.kind == "bitdelta")
    assert sz["compression_vs_dense"] > 2.5
    ref = logits_of(ft, TOK)
    applied = D.apply_delta(base, d)
    e_applied = rel_err(logits_of(applied, TOK), ref)
    base_err = rel_err(logits_of(base, TOK), ref)
    assert e_applied < base_err and isinstance(applied.layers[0].self_attn.q_proj, D.BitDeltaLinear)
    # calibrating the scales by logit distillation does not make things worse and lowers the training loss
    rep = D.calibrate_bitdelta(base, ft, d, rand_tokens(16, 24, seed=7), steps=40)
    assert rep["loss_after"] < rep["loss_before"]
    merged = D.merge_delta(D.apply_delta(base, d))
    assert isinstance(merged.layers[0].self_attn.q_proj, nn.Linear) and not isinstance(merged.layers[0].self_attn.q_proj, D.BitDeltaLinear)
    assert rel_err(logits_of(merged, TOK), ref) <= base_err
    assert torch.allclose(logits_of(merged, TOK), logits_of(D.apply_delta(base, d), TOK), atol=1e-4)
    d.save(tmp_path / "d")
    d2 = D.Delta.load(tmp_path / "d")
    assert d2.nbytes() == d.nbytes() and torch.equal(d2.entries["layers.1.mlp.down_proj"].bits, d.entries["layers.1.mlp.down_proj"].bits)
    # the base is never modified
    assert torch.equal(base.layers[0].self_attn.q_proj.weight, make_lm().layers[0].self_attn.q_proj.weight)


def test_lowrank_delta_exact_for_low_rank_finetune():
    base = make_lm()
    ft = copy.deepcopy(base)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in ft.named_parameters():
            if p.ndim == 2 and n.startswith("layers."):
                p.add_(torch.randn(p.shape[0], 4, generator=g) @ torch.randn(4, p.shape[1], generator=g) * 0.02)
    d = D.compute_delta(base.state_dict(), ft.state_dict(), D.DeltaConfig(method="lowrank", rank=4))
    applied = D.apply_delta(base, d)
    assert rel_err(logits_of(applied, TOK), logits_of(ft, TOK)) < 1e-8
    assert d.size_report(10**6)["delta_bytes"] < 0.5 * sum(t.numel() * 4 for k, t in ft.state_dict().items() if k.startswith("layers."))
    with pytest.raises(ConfigError):
        D.compute_delta(base.state_dict(), {k: v for k, v in list(ft.state_dict().items())[1:]})


def test_multi_tenant_linear_matches_dense_per_tenant():
    torch.manual_seed(0)
    base = nn.Linear(32, 16, bias=False)
    mt = D.MultiTenantLinear(base)
    deltas = [torch.randn(16, 32) * 0.05 for _ in range(3)]
    from forge.formats.packing import pack_bits

    for dl in deltas:
        mt.add_tenant(pack_bits((dl > 0).to(torch.uint8), 1), dl.abs().mean())
    x = torch.randn(6, 32)
    ids = torch.tensor([0, 1, 2, -1, 1, 0])
    y = mt(x, ids)
    for i, t in enumerate(ids.tolist()):
        w = base.weight.detach().clone()
        if t >= 0:
            w = w + deltas[t].abs().mean() * (deltas[t] > 0).float().mul(2).sub(1)
        assert torch.allclose(y[i], x[i] @ w.t(), atol=1e-5)


# ------------------------------------------------------------------------------------------------ lossless
def test_lossless_roundtrip_and_ratio(tmp_path):
    m = make_lm(dtype=torch.bfloat16)
    sd = {k: v for k, v in m.state_dict().items()}
    rep = LL.lossless_report(sd)
    assert rep["verified"] and rep["ratio"] < 0.85, rep["ratio"]
    assert rep["mean_exponent_entropy_bits"] is not None and rep["mean_exponent_entropy_bits"] < 5.0
    assert "bfloat16" in rep["by_dtype"]
    LL.lossless_report(sd, {"write_container": True}, out_dir=tmp_path / "flc")
    back = LL.lossless_decode_dir(tmp_path / "flc")
    assert set(back) == set(sd)
    for k in sd:
        assert LL._roundtrip_equal(back[k], sd[k].contiguous()), k


def test_lossless_quantized_model_reports_honest_ratio():
    m = make_lm(dtype=torch.float16)
    quantize_model(m, "rtn", 4, 32)
    rep = LL.lossless_report(m)
    assert rep["verified"]
    u8 = rep["by_dtype"].get("uint8")
    assert u8 is not None and u8["ratio"] > 0.8  # packed 4-bit payloads are near incompressible


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16, torch.float32, torch.uint8, torch.int8, torch.float8_e4m3fn])
def test_fallback_codec_roundtrip(dt):
    t = (torch.randn(257, 13) * 3).to(dt) if dt.is_floating_point else torch.randint(0, 100, (257, 13)).to(dt)
    assert LL._roundtrip_equal(LL._fb_decode(LL._fb_encode(t)), t)


# ---------------------------------------------------------------------------------------------- device-handling regressions (GPU report 2)
def test_random_orthogonal_hadamard_honours_device_meta():
    from forge.compress.rotation import random_orthogonal_hadamard

    q = random_orthogonal_hadamard(16, 0, device="meta", dtype=torch.float64)
    assert q.device.type == "meta" and q.dtype == torch.float64


def test_apply_r2_builds_rotation_on_weight_device(monkeypatch):
    """apply_r2 used to build q2 on the default device (CPU) while the weights were on CUDA -> bmm device mismatch."""
    import forge.compress.rotation as rot

    seen = []
    orig = rot.random_orthogonal_hadamard

    def spy(n, seed=0, device=None, dtype=torch.float32):
        seen.append(device)
        return orig(n, seed, device=device, dtype=dtype)

    monkeypatch.setattr(rot, "random_orthogonal_hadamard", spy)
    from forge.models.tiny import build_tiny_model

    m = build_tiny_model(0)
    rot.fuse_norms(m, False)
    rot.apply_r2(m, 0)
    assert seen and all(torch.device(d) == next(m.parameters()).device for d in seen)


def test_int_scale_uses_true_division_and_matches_reference():
    """scale = amax / qmax must be a true IEEE division (CUDA rewrites tensor / python-scalar as a reciprocal multiply)."""
    from forge.formats.int_quant import _div_const, compute_qparams

    x = torch.randn(64, 4, 32)
    for bits in (4, 8):
        s, _ = compute_qparams(x, bits, True)
        ref = x.abs().amax(-1, keepdim=True) / float((1 << (bits - 1)) - 1)
        assert torch.equal(s, ref)
    a = torch.rand(1000) * 5
    assert torch.equal(_div_const(a, 7.0), a / 7.0)
