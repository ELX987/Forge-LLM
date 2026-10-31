"""Adversarial-review tests for forge.formats: independent re-implementations of the NVFP4 / MXFP4 / E2M1 definitions."""
from __future__ import annotations

import math

import pytest
import torch

from forge.formats.fp4 import (
    E2M1_GRID,
    encode_e2m1,
    quantize_mxfp4,
    quantize_nvfp4,
    stochastic_round_fp4,
)
from forge.formats.int_quant import dequantize_int, quantize_int

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _rtn_even_ref(x: float) -> int:
    """Brute-force nearest E2M1 magnitude code, ties to the even code, saturating at 6 (sign handled by the caller)."""
    a = min(abs(x), 6.0)
    best = None
    for i, g in enumerate(GRID):
        d = abs(a - g)
        if best is None or d < best[0] - 1e-12 or (abs(d - best[0]) <= 1e-12 and i % 2 == 0 and best[1] % 2 == 1):
            best = (d, i)
    return best[1]


def test_e2m1_rtn_matches_bruteforce_including_every_tie():
    xs = torch.cat([torch.linspace(-7, 7, 1401), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -5.0, 6.0, -6.0])])
    codes = encode_e2m1(xs)
    for x, c in zip(xs.tolist(), codes.tolist()):
        assert (c & 7) == _rtn_even_ref(x), x
        assert bool(c & 8) == (x < 0 and _rtn_even_ref(x) > 0)  # canonical: no negative zero code


def _e4m3_round(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def _nvfp4_ref(x: torch.Tensor) -> torch.Tensor:
    """Paper definition: s_enc = 6*448/amax(x); stored_b = e4m3(amax_b/6 * s_enc); x_hat = RTN(x/(stored_b*s_dec)) * stored_b * s_dec."""
    xf = x.float()
    amax = xf.abs().max()
    s_enc = 6.0 * 448.0 / amax
    s_dec = 1.0 / s_enc
    out = torch.zeros_like(xf)
    for r in range(xf.shape[0]):
        for b in range(xf.shape[1] // 16):
            blk = xf[r, b * 16 : (b + 1) * 16]
            stored = _e4m3_round(blk.abs().max() / 6.0 * s_enc)
            if stored == 0:
                continue
            sc = stored * s_dec
            for j, v in enumerate((blk / sc).tolist()):
                code = _rtn_even_ref(v)
                out[r, b * 16 + j] = math.copysign(GRID[code], v) * sc
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nvfp4_matches_independent_paper_reference(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(6, 48, generator=g) * torch.exp(torch.randn(6, 1, generator=g))
    x[0, :16] = 0.0  # an all-zero block
    x[1, 3] = 50.0  # outlier inside a block
    t = quantize_nvfp4(x)
    assert torch.allclose(t.dequantize(), _nvfp4_ref(x), rtol=1e-6, atol=0)
    assert t.block_scales_e4m3.dtype == torch.float8_e4m3fn and t.payload_u8.shape == (6, 24)
    assert float(t.tensor_scale) == pytest.approx(float(x.abs().max()) / (6 * 448), rel=1e-6)


def test_mxfp4_floor_mode_follows_ocp_reference_and_ceil_never_saturates():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 64, generator=g) * 5
    for mode in ("floor", "ceil", "near"):
        t = quantize_mxfp4(x, scale_mode=mode)
        e = t.scales_e8m0_u8.float() - 127
        amax = x.reshape(4, 2, 32).abs().amax(-1)
        if mode == "floor":
            assert torch.equal(e, torch.floor(torch.log2(amax)) - 2)  # OCP: 2**(floor(log2(amax)) - emax_elem)
        if mode == "ceil":
            assert bool((amax / 2.0**e <= 6.0 + 1e-6).all())
        assert t.dequantize().shape == x.shape


def test_sr_is_unbiased_on_every_grid_interval():
    for v in (0.1, 0.7, 1.2, 1.9, 2.4, 3.7, 5.2, -0.3, -4.4):
        m = stochastic_round_fp4(torch.full((200_000,), v), seed=7).mean().item()
        assert m == pytest.approx(v, abs=0.02), v
    assert set(stochastic_round_fp4(torch.full((1000,), 2.3), seed=1).unique().tolist()) <= {2.0, 3.0}


@pytest.mark.parametrize("scale", [1e-30, 1e-6, 1.0, 1e6, 1e30])
def test_fp4_scaling_is_range_invariant(scale):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 64, generator=g, dtype=torch.float64) * scale
    xf = x.float()
    for q in (quantize_nvfp4(xf).dequantize(), quantize_mxfp4(xf).dequantize()):
        assert torch.isfinite(q).all()
        assert float((q.double() - xf.double()).norm() / xf.double().norm()) < 0.2


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dim", [-1, 0])
def test_int_quant_error_is_bounded_by_half_a_step(bits, sym, dim):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 128, generator=g)
    q, s, z = quantize_int(w, bits, 32, sym, dim)
    assert int(q.max()) <= 2**bits - 1 and int(q.min()) >= 0
    d = dequantize_int(q, s, z, 32, dim)
    step = s.movedim(dim, -1).repeat_interleave(32, dim=-1).movedim(-1, dim)
    # within the representable range the error is at most half a quantisation step
    assert bool(((d - w).abs() <= step * 0.5 + 1e-6).float().mean() > 0.99)


def test_e2m1_grid_constant():
    assert E2M1_GRID.tolist() == GRID
