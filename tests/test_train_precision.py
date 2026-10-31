"""NVFP4 / FP8 emulation numerics, precision manager (exemptions, healing, gating), TR-2 diagnostics."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from test_train_fixtures import make_model

from forge.core.errors import ConfigError
from forge.train.config import PrecisionConfig
from forge.train.diagnostics import (
    low_norm_channel_experiment,
    predicted_zero_fraction_2d,
    relative_loss_gap,
    scale_stats,
    sr_bias_test,
    update_weight_ratio,
    zero_grad_tile_stats,
)
from forge.train.fp4sim import (
    E2M1_GRID,
    FP4Rng,
    e4m3_round,
    fake_quant_fp8,
    fake_quant_nvfp4,
    hadamard_matrix,
    random_hadamard,
    round_e2m1_rtn,
    round_e2m1_sr,
)
from forge.train.precision import PrecisionManager, QuantLinear, is_exempt, resolve_compute_dtype


# ----------------------------------------------------------------------------------------------- numerics
def test_e2m1_rtn_ties_and_saturation():
    z = torch.tensor([0.0, 0.2, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 6.0, 9.0])
    q = round_e2m1_rtn(z)
    assert q.tolist() == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0]  # ties-to-even code
    assert torch.equal(round_e2m1_rtn(-z), -q)
    assert set(round_e2m1_rtn(torch.randn(1000) * 3).abs().tolist()) <= set(E2M1_GRID)


def test_e2m1_sr_unbiased_and_on_grid():
    z = torch.linspace(0.0, 6.0, 25).repeat(4000, 1)
    q = round_e2m1_sr(z, FP4Rng(0))
    assert set(q.abs().unique().tolist()) <= set(E2M1_GRID)
    bias = (q.mean(0) - z[0]).abs().max()
    assert bias < 0.06  # Monte-Carlo level (4000 draws, spacing <= 2)
    assert round_e2m1_sr(torch.tensor([9.0]), FP4Rng(0)).item() == 6.0


def test_e4m3_round():
    x = torch.tensor([0.0, 1e-5, 2 ** -9, 1.0, 1.06, 448.0, 1000.0])
    r = e4m3_round(x)
    assert r[0] == 0 and r[1] == 0  # underflow flushes
    assert r[2] == pytest.approx(2 ** -9) and r[3] == 1.0 and r[-1] == 448.0 and r[-2] == 448.0


def test_nvfp4_quant_error_and_structure():
    torch.manual_seed(0)
    x = torch.randn(256, 256)
    q1, info = fake_quant_nvfp4(x, (1, 16), "rtn", return_info=True)
    q2 = fake_quant_nvfp4(x, (16, 16), "rtn")
    rel1 = float(((q1 - x) ** 2).sum() / (x ** 2).sum())
    rel2 = float(((q2 - x) ** 2).sum() / (x ** 2).sum())
    assert 0.007 < rel1 < 0.012  # paper: relative MSE ~0.009 on Gaussian data
    assert rel1 < rel2 < 0.016  # 2D tiles cost a little accuracy (paper: ~0.012)
    assert info.stored_scales.shape == (256, 16)
    # every block holds <= 8 distinct magnitudes after dividing out its scale
    blk = q1[0, :16]
    s = (info.stored_scales[0, 0] / info.tensor_scale).item()
    assert set(round(abs(v) / s, 4) for v in blk.tolist()) <= set(round(g, 4) for g in E2M1_GRID)
    # SR is noisier (about 2x MSE) but unbiased on average
    qs = fake_quant_nvfp4(x, (1, 16), "sr", FP4Rng(1))
    rels = float(((qs - x) ** 2).sum() / (x ** 2).sum())
    assert 1.5 < rels / rel1 < 2.6
    # shapes and padding
    y = torch.randn(3, 5, 20)
    assert fake_quant_nvfp4(y, (1, 16)).shape == y.shape
    assert fake_quant_nvfp4(torch.zeros(4, 16)).abs().sum() == 0


def test_fp8_fake_quant_and_hadamard():
    x = torch.randn(64, 64) * 10
    q = fake_quant_fp8(x, "e4m3")
    assert float((q - x).abs().max() / x.abs().max()) < 0.07
    assert fake_quant_fp8(x, "e5m2").shape == x.shape
    with pytest.raises(ValueError):
        fake_quant_fp8(x, "e9m9")
    h = hadamard_matrix(16)
    assert torch.allclose(h @ h.T, torch.eye(16), atol=1e-6)
    y = torch.randn(4, 48)
    t = random_hadamard(y, 16, seed=3)
    assert torch.allclose(random_hadamard(t, 16, seed=3, inverse=True), y, atol=1e-5)
    a, b = torch.randn(5, 32), torch.randn(7, 32)
    assert torch.allclose(random_hadamard(a, 16, 1) @ random_hadamard(b, 16, 1).T, a @ b.T, atol=1e-4)  # inner products preserved
    with pytest.raises(ValueError):
        random_hadamard(torch.randn(3, 20), 16)


def test_rht_flattens_outlier_blocks():
    """RHT spreads a spike over the block: the crest factor (max/rms) drops (it helps SR variance, not RTN error -- paper ch. 6)."""
    torch.manual_seed(0)
    x = torch.randn(512, 64)
    x[:, 3] *= 40  # a persistent outlier channel

    def crest(t):
        b = t.reshape(-1, 16)
        return float((b.abs().amax(-1) / b.pow(2).mean(-1).sqrt()).mean())

    blk = x[:, :16].contiguous()  # the blocks that contain the outlier column
    assert crest(random_hadamard(blk, 16, 0)) < 0.6 * crest(blk)


# ----------------------------------------------------------------------------------------------- QuantLinear
def _make_ql(mode, fp4=True, **kw):
    from forge.train.precision import _FP4Settings

    lin = nn.Linear(32, 48, bias=True)
    torch.manual_seed(0)
    nn.init.normal_(lin.weight)
    lin._forge_orig_cls = type(lin)
    lin.__class__ = QuantLinear
    lin._forge_q = True
    lin.qmode = mode
    lin._mm_dtype = torch.float32
    lin._fp8_grad_fmt = "e5m2"
    lin._fp4 = _FP4Settings(rng=FP4Rng(0), **kw) if fp4 else None
    return lin


def test_fp4_linear_forward_matches_manual_and_backward_is_close_to_exact():
    lin = _make_ql("fp4", rht=True)
    x = torch.randn(64, 32, requires_grad=True)
    y = lin(x)
    manual = fake_quant_nvfp4(x, (1, 16)) @ fake_quant_nvfp4(lin.weight, (16, 16)).T + lin.bias
    assert torch.allclose(y, manual, atol=1e-4)
    y.sum().backward()
    # reference gradients through an unquantised linear
    ref = nn.Linear(32, 48)
    ref.load_state_dict({"weight": lin.weight.detach().clone(), "bias": lin.bias.detach().clone()})
    xr = x.detach().clone().requires_grad_(True)
    ref(xr).sum().backward()
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))  # noqa: E731
    assert cos(x.grad, xr.grad) > 0.9 and cos(lin.weight.grad, ref.weight.grad) > 0.9
    assert torch.allclose(lin.bias.grad, ref.bias.grad)
    assert lin.weight.grad.shape == lin.weight.shape and x.grad.shape == x.shape


def test_fp4_wgrad_sr_is_unbiased_in_expectation():
    torch.manual_seed(0)
    x = torch.randn(128, 32)
    R = torch.randn(128, 48)
    w = torch.randn(48, 32, generator=torch.Generator().manual_seed(1))
    outs = []
    for seed in range(80):
        lin = _make_ql("fp4", rht=False, sr=True)
        lin.weight.data.copy_(w)
        lin._fp4.rng = FP4Rng(seed)
        (lin(x) * R).sum().backward()
        outs.append(lin.weight.grad.clone())
    mean = torch.stack(outs).mean(0)
    # expectation of the SR wgrad = R^T @ RTN-quantised activations (token-dim blocks); SR adds zero-mean noise only
    expected = R.t() @ fake_quant_nvfp4(x.t().contiguous(), (1, 16)).t()
    err_mean = float((mean - expected).norm() / expected.norm())
    err_single = float((outs[0] - expected).norm() / expected.norm())
    assert err_single > 0.05 and err_mean < 0.35 * err_single


def test_fp4_handles_ragged_token_counts_and_3d_inputs():
    lin = _make_ql("fp4", rht=True)
    x = torch.randn(3, 7, 32, requires_grad=True)  # 21 tokens: not a multiple of 16
    y = lin(x)
    assert y.shape == (3, 7, 48)
    y.pow(2).mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(lin.weight.grad).all()


def test_fp8_linear_close_to_exact():
    lin = _make_ql("fp8", fp4=False)
    x = torch.randn(16, 32, requires_grad=True)
    y = lin(x)
    exact = nn.functional.linear(x, lin.weight, lin.bias)
    assert float((y - exact).detach().norm() / exact.detach().norm()) < 0.08
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    lin.qmode = "off"
    assert torch.allclose(lin(x), exact, atol=1e-6)


# ----------------------------------------------------------------------------------------------- manager
def test_exemptions_and_conversion():
    cfg = PrecisionConfig(rung="R1", exempt_first_layers=1, exempt_last_layers=1)
    assert is_exempt("lm_head", cfg, 4) and is_exempt("embed_tokens", cfg, 4)
    assert is_exempt("layers.0.mlp.up_proj", cfg, 4) and is_exempt("layers.3.mlp.up_proj", cfg, 4)
    assert not is_exempt("layers.1.mlp.up_proj", cfg, 4)
    model = make_model(0, layers=4)
    pm = PrecisionManager(cfg, device="cpu", total_steps=10)
    summ = pm.prepare(model)
    assert summ["backend"] == "emulated" and summ["n_layers"] == 4
    conv = [n for n, m in model.named_modules() if isinstance(m, QuantLinear)]
    assert conv and all(".1." in n or ".2." in n for n in conv) and not any("lm_head" in n for n in conv)
    ids = torch.randint(0, 100, (2, 16))
    out = model(ids).logits
    assert torch.isfinite(out).all()
    n = pm.restore(model)
    assert n == len(conv) and not any(isinstance(m, QuantLinear) for m in model.modules())


def test_r2_is_gated_and_runs():
    with pytest.raises(ConfigError, match="gated"):
        PrecisionManager(PrecisionConfig(rung="R2"), device="cpu")
    model = make_model(0, layers=4)
    pm = PrecisionManager(PrecisionConfig(rung="R2", allow_unvalidated_fp4=True, exempt_first_layers=1, exempt_last_layers=1), device="cpu", total_steps=10)
    pm.prepare(model)
    assert pm.backend == "emulated" and pm.quantised
    ids = torch.randint(0, 100, (2, 16))
    with pm.autocast():
        loss = torch.nn.functional.cross_entropy(model(ids).logits.view(-1, 288), ids.view(-1))
    loss.backward()
    d = pm.diagnostics(model)
    for k in ("fp4/zero_grad_tile_frac", "fp4/scale_flush_frac", "fp4/scale_occupancy", "fp4/sr_bias", "fp4/sr_bias_z", "fp4/saturation_frac"):
        assert k in d and math.isfinite(d[k]), k
    # high precision context disables quantisation
    with torch.no_grad():
        q = model(ids).logits
        with pm.high_precision():
            hp = model(ids).logits
        again = model(ids).logits
    assert not torch.allclose(q, hp) and torch.allclose(q, again)


def test_healing_phase_switch():
    model = make_model(0, layers=4)
    pm = PrecisionManager(PrecisionConfig(rung="R2", allow_unvalidated_fp4=True, healing_steps=3, healing_rung="R0"), device="cpu", total_steps=10)
    pm.prepare(model)
    assert pm.healing_start == 7
    assert not pm.set_step(0) and pm._current_mode() == "fp4"
    assert pm.set_step(7) and pm._current_mode() == "off"
    assert all(m.qmode == "off" for m in model.modules() if isinstance(m, QuantLinear))
    pm2 = PrecisionManager(PrecisionConfig(rung="R2", allow_unvalidated_fp4=True, healing_fraction=0.2, healing_rung="R1"), device="cpu", total_steps=10)
    assert pm2.healing_start == 8
    pm2.prepare(make_model(0, layers=4))
    pm2.set_step(8)
    assert pm2._current_mode() == "fp8"


def test_compute_dtype_resolution():
    assert resolve_compute_dtype("auto", "cpu") is None
    assert resolve_compute_dtype("bfloat16", "cpu") == torch.bfloat16
    assert resolve_compute_dtype("float32", "cuda") is None


def test_fp8_backend_fallback_is_announced(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="forge.train.precision")
    pm = PrecisionManager(PrecisionConfig(rung="R1", backend="torchao"), device="cpu", total_steps=5)
    pm.prepare(make_model(0, layers=4))
    assert pm.backend == "emulated"
    assert any("EMULATED" in r.message for r in caplog.records)


# ----------------------------------------------------------------------------------------------- TR-2 diagnostics
def test_zero_grad_tile_fraction():
    g = torch.randn(64, 64)
    g[:16, :16] = 0
    g[16:32, 16:32] = 0
    st = zero_grad_tile_stats(g)
    assert st["zero_tile_frac"] == pytest.approx(2 / 16)
    assert st["zero_elem_frac"] == pytest.approx(2 * 256 / 4096)
    assert zero_grad_tile_stats(torch.zeros(30, 30))["zero_tile_frac"] == 1.0  # ragged edge tiles
    assert zero_grad_tile_stats(torch.randn(5, 4, 32))["zero_tile_frac"] == 0.0


def test_scale_flush_and_occupancy():
    torch.manual_seed(0)
    x = torch.randn(64, 256)
    ok = scale_stats(x)
    assert ok["flush_frac"] == 0 and ok["subnormal_frac"] == 0 and ok["occupancy"] < 0.4
    x2 = x.clone()
    x2[0, 0] = 1e9  # one huge value: tensor scale makes all other blocks' E4M3 scales underflow
    bad = scale_stats(x2)
    assert bad["flush_frac"] > 0.9 and bad["occupancy"] <= 1 / 18 + 1e-9
    mid = x.clone()
    mid[:, :16] *= 1e3  # wide dynamic range across blocks -> broad exponent occupancy
    assert scale_stats(mid)["occupancy"] > ok["occupancy"]


def test_sr_bias_test_detects_rtn_bias_scale():
    torch.manual_seed(0)
    g = torch.randn(64, 64) * 0.01
    r = sr_bias_test(g, n_draws=48)
    assert r["z"] < 5.0  # SR bias is at the Monte-Carlo noise level
    assert r["rtn_rel_err"] > 0.05 and r["rel_bias"] < r["rtn_rel_err"]


def test_relative_loss_gap_and_update_ratio():
    assert relative_loss_gap(2.04, 2.0) == pytest.approx(0.02)
    assert relative_loss_gap([1.0, 2.2], [1.0, 2.0]) == pytest.approx([0.0, 0.1])
    with pytest.raises(ValueError):
        relative_loss_gap([1.0], [1.0, 2.0])
    a = [torch.ones(4)]
    b = [torch.ones(4) * 1.01]
    assert update_weight_ratio(a, b) == pytest.approx(0.01, rel=1e-3)


@pytest.mark.parametrize("r,f", [(1.0, 0.1), (0.3, 0.1), (0.1, 0.1), (0.03, 0.1), (0.3, 0.5)])
def test_p5_low_norm_channel_zero_gradient_failure_reproduced(r, f):
    """Paper section 15.4 / Table 15.2: 2D (16x16) scaling zeroes low-norm channels; 1D does not; total MSE hides it."""
    res = low_norm_channel_experiment(512, 512, f, r, seed=0)
    pred = predicted_zero_fraction_2d(r)
    assert res["1d"]["zero_low"] == pytest.approx(0.068, abs=0.02)  # 1x16 stays at ~6.8% whatever the channel norm
    tol = 0.03 if f == 0.1 else 0.07  # prediction ignores the fall of the tile max when many rows are low-norm
    assert res["2d"]["zero_low"] == pytest.approx(pred, abs=tol)
    if r <= 0.3:
        assert res["2d"]["zero_low"] > 3 * res["1d"]["zero_low"] or r == 1.0
    if r <= 0.03:
        assert res["2d"]["zero_low"] > 0.99 and res["2d"]["rel_mse_low"] > 0.99  # channel erased completely
        # ... yet the whole-matrix MSE is almost unchanged (the failure is invisible to loss/MSE-type diagnostics)
        assert abs(res["2d"]["rel_mse_all"] - res["1d"]["rel_mse_all"]) < 0.01
    # the 2D zero fraction of the *whole matrix* is the cheap detector: it grows by f*pred
    assert predicted_zero_fraction_2d(1.0) == pytest.approx(0.10, abs=0.01)


def test_numerics_agree_with_forge_formats():
    """Independent re-implementation vs forge.formats.fp4 (the oracle used by PTQ / kernels): same quantiser."""
    F = pytest.importorskip("forge.formats.fp4")
    torch.manual_seed(0)
    x = torch.randn(64, 96) * torch.logspace(-2, 2, 96)
    assert torch.allclose(fake_quant_nvfp4(x, (1, 16)), F.fake_quant_nvfp4(x, block_size=16), rtol=1e-4, atol=1e-4)
    assert torch.allclose(fake_quant_nvfp4(x, (16, 16)), F.fake_quant_nvfp4(x, block_size=16, scale_2d=True), rtol=1e-4, atol=1e-4)
    z = torch.randn(2000) * 3
    assert torch.equal(round_e2m1_rtn(z), F.round_e2m1(z))


def test_lora_adapters_are_never_quantised():
    from forge.train.config import LoraConfig
    from forge.train.lora import LoraBackend

    model = make_model(0, layers=4)
    LoraBackend(LoraConfig(r=4, target_modules=["q_proj", "v_proj"]), prefer_builtin=True).apply(model)
    pm = PrecisionManager(PrecisionConfig(rung="R1", exempt_first_layers=0, exempt_last_layers=0), device="cpu", total_steps=5)
    pm.prepare(model)
    for n, m in model.named_modules():
        if "lora_" in n:
            assert not isinstance(m, QuantLinear), n
    assert any(isinstance(m, QuantLinear) for n, m in model.named_modules() if n.endswith("k_proj"))
