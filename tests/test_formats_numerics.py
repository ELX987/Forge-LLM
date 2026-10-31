"""Conformance tests for forge.formats: reproduce numbers of the NVFP4 report (tables 3.2, 4.5, 4.6) and basic invariants."""
from __future__ import annotations

import math

import pytest
import torch

from forge.formats import (
    E2M1_GRID,
    bits_per_element,
    counter_uniform,
    dequantize_int,
    dequantize_mxfp4,
    dequantize_nvfp4,
    fake_quant_fp8,
    fake_quant_int,
    fake_quant_mxfp4,
    fake_quant_nvfp4,
    from_fp8,
    pack_fp4,
    quantize_int,
    quantize_mxfp4,
    quantize_nvfp4,
    round_e2m1,
    stochastic_round_fp4,
    to_fp8,
    unpack_fp4,
)
from forge.formats import fp8 as F8
from forge.formats.fp4 import quantize_fp4_ideal
from forge.formats.packing import pack_bits, unpack_bits


def rel_mse(x, y):
    return ((x - y).pow(2).mean() / x.pow(2).mean()).item()


@pytest.fixture(scope="module")
def gauss():
    g = torch.Generator().manual_seed(1234)
    return torch.randn(1 << 18, generator=g)


# ------------------------------------------------------------------------------------------------ E2M1
def test_e2m1_grid_values():
    assert E2M1_GRID.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_e2m1_rtn_ties_to_even_code():
    # paper App. B rtn_fp4: ties 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5.0->4
    xs = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert round_e2m1(xs).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    assert round_e2m1(-xs).tolist() == [-0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0]
    assert round_e2m1(torch.tensor([7.0, -100.0])).tolist() == [6.0, -6.0]


def test_e2m1_rtn_is_nearest_everywhere():
    x = torch.linspace(-7, 7, 20001)
    r = round_e2m1(x)
    grid = torch.cat([-E2M1_GRID.flip(0)[:-1], E2M1_GRID])
    xc = x.clamp(-6, 6)
    best = grid[(xc.unsqueeze(1) - grid).abs().argmin(1)]
    assert torch.allclose((r - xc).abs(), (best - xc).abs(), atol=1e-6)


def test_fp4_pack_roundtrip_all_codes():
    codes = torch.arange(16, dtype=torch.uint8).repeat(5, 2)
    p = pack_fp4(codes)
    assert p.shape == (5, 16) and p.dtype == torch.uint8
    assert torch.equal(unpack_fp4(p), codes)
    assert int(p[0, 0]) == (1 << 4) | 0  # even index in the low nibble (code 0), odd index in the high nibble (code 1)
    odd = torch.arange(7, dtype=torch.uint8)
    assert torch.equal(unpack_fp4(pack_fp4(odd), 7), odd)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_pack_bits_roundtrip(bits):
    c = torch.randint(0, 1 << bits, (3, 37), dtype=torch.uint8)
    assert torch.equal(unpack_bits(pack_bits(c, bits), bits, 37), c)


# ------------------------------------------------------------------------------------------------ stochastic rounding
def test_counter_rng_chunk_invariance_and_uniformity():
    a = counter_uniform((1000,), seed=7, offset=0)
    b = torch.cat([counter_uniform((300,), 7, 0), counter_uniform((700,), 7, 300)])
    assert torch.equal(a, b)
    assert not torch.equal(a, counter_uniform((1000,), seed=8))
    big = counter_uniform((1 << 18,), seed=3)
    assert 0.0 <= big.min() and big.max() < 1.0
    assert abs(big.mean().item() - 0.5) < 0.005
    hist = torch.histc(big, bins=16, min=0, max=1)
    assert (hist / big.numel()).sub(1 / 16).abs().max() < 0.004


def test_sr_is_unbiased_and_matches_grid():
    x = torch.full((1 << 17,), 2.3)
    y = stochastic_round_fp4(x, seed=11)
    assert set(y.unique().tolist()) <= {2.0, 3.0}
    assert abs(y.mean().item() - 2.3) < 0.01
    g = E2M1_GRID.repeat(100)
    assert torch.equal(stochastic_round_fp4(g, seed=5), g)
    xn = torch.full((1 << 16,), -0.3)
    assert abs(stochastic_round_fp4(xn, seed=2).mean().item() + 0.3) < 0.01
    assert torch.equal(stochastic_round_fp4(x[:100], seed=11), stochastic_round_fp4(x[:100], seed=11))


# ------------------------------------------------------------------------------------------------ NVFP4 / MXFP4
def test_nvfp4_scale_arithmetic():
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    x[3, 5] = 100.0  # global max
    t = quantize_nvfp4(x)
    amax = x.abs().max()
    assert torch.isclose(t.tensor_scale, amax / (448.0 * 6.0), rtol=1e-6)
    assert t.block_scales_e4m3.float().max().item() == 448.0
    assert t.block_scales_e4m3.float()[3, 5 // 16].item() == 448.0
    assert t.block_scales_e4m3.dtype == torch.float8_e4m3fn
    assert t.payload_u8.shape == (64, 32)
    assert t.block_scales_e4m3.shape == (64, 4)
    assert abs(dequantize_nvfp4(t)[3, 5].item() - 100.0) < 1e-3


def test_nvfp4_bits_per_element():
    assert bits_per_element("nvfp4") == 4.5
    assert bits_per_element("mxfp4") == 4.25
    assert abs(bits_per_element("int4", 128, 16) - 4.125) < 1e-9
    t = quantize_nvfp4(torch.randn(128, 256))
    assert abs(t.bits_per_element - 4.5) < 0.01  # + 32 bit tensor scale
    m = quantize_mxfp4(torch.randn(128, 256))
    assert abs(m.bits_per_element - 4.25) < 1e-9


def test_nvfp4_fake_matches_packed_and_pads():
    torch.manual_seed(1)
    for shape in [(8, 48), (5, 37), (3, 4, 32), (1, 16)]:
        x = torch.randn(*shape)
        a = dequantize_nvfp4(quantize_nvfp4(x))
        b = fake_quant_nvfp4(x)
        assert a.shape == x.shape
        assert torch.equal(a, b)


def test_nvfp4_zero_and_tiny_blocks():
    x = torch.zeros(4, 32)
    assert torch.equal(fake_quant_nvfp4(x), x)
    x[0, :16] = 1e-3
    x[1, 0] = 5.0
    assert torch.isfinite(fake_quant_nvfp4(x)).all()


def test_nvfp4_vs_mxfp4_mse_matches_paper_table(gauss):
    """Table 4.5 (Gaussian): FP4 G=16 ideal 0.00886, NVFP4 0.00904, MXFP4 ceil 0.01329 / floor 0.01320, NVFP4 G=32 0.01032."""
    x = gauss.reshape(-1, 1024)
    nv = rel_mse(x, fake_quant_nvfp4(x))
    mx_ceil = rel_mse(x, fake_quant_mxfp4(x, scale_mode="ceil"))
    mx_floor = rel_mse(x, fake_quant_mxfp4(x, scale_mode="floor"))
    ideal = rel_mse(x, quantize_fp4_ideal(x, 16))
    nv32 = rel_mse(x, fake_quant_nvfp4(x, block_size=32))
    assert nv == pytest.approx(0.00904, rel=0.04)
    assert ideal == pytest.approx(0.00886, rel=0.04)
    assert mx_ceil == pytest.approx(0.01329, rel=0.04)
    assert mx_floor == pytest.approx(0.01320, rel=0.04)
    assert nv32 == pytest.approx(0.01032, rel=0.04)
    assert nv < nv32 < mx_ceil
    assert 1.0 - nv / mx_ceil == pytest.approx(0.32, abs=0.03)  # paper: 32 % reduction (1.67 dB)


def test_nvfp4_four_over_six_improves_mse(gauss):
    """Table 4.5: NVFP4 + Four Over Six (MSE) 0.00756 vs 0.00904."""
    x = gauss.reshape(-1, 1024)
    v = rel_mse(x, fake_quant_nvfp4(x, four_over_six=True))
    assert v == pytest.approx(0.00756, rel=0.05)
    assert v < rel_mse(x, fake_quant_nvfp4(x))


def test_nvfp4_sr_mse_and_unbiased(gauss):
    """Table 4.5: NVFP4 with SR 0.01892 (about 2x RTN); SR must be unbiased in the mean."""
    x = gauss.reshape(-1, 1024)
    y = fake_quant_nvfp4(x, rounding="stochastic", seed=3)
    assert rel_mse(x, y) == pytest.approx(0.01892, rel=0.06)
    xs = x[:16]
    acc = torch.zeros_like(xs)
    n = 400
    gmax = gauss.abs().max()
    for i in range(n):
        acc += fake_quant_nvfp4(xs, rounding="stochastic", seed=100 + i, tensor_amax=gmax)
    assert (acc / n - xs).abs().mean().item() < 0.01


def test_nvfp4_block_max_error_table_4_6(gauss):
    """Table 4.6: relative squared error of the block-maximum element, NVFP4 6.9e-4 (Gaussian)."""
    x = gauss.reshape(-1, 16)
    y = fake_quant_nvfp4(x.reshape(-1, 1024)).reshape(-1, 16)
    idx = x.abs().argmax(1, keepdim=True)
    xm, ym = x.gather(1, idx), y.gather(1, idx)
    rel = ((xm - ym) / xm).pow(2).mean().item()
    assert rel == pytest.approx(6.9e-4, rel=0.2)


def test_mxfp4_scales_are_powers_of_two_and_modes():
    torch.manual_seed(0)
    x = torch.randn(16, 128) * 3
    for mode in ("ceil", "floor", "near"):
        t = quantize_mxfp4(x, scale_mode=mode)
        assert t.scales_e8m0_u8.dtype == torch.uint8
        assert torch.equal(dequantize_mxfp4(t), fake_quant_mxfp4(x, scale_mode=mode))
    t = quantize_mxfp4(x, scale_mode="ceil")
    s = torch.pow(2.0, t.scales_e8m0_u8.float() - 127).unsqueeze(-1)
    amax = x.reshape(16, -1, 32).abs().amax(-1, keepdim=True)
    assert (amax / s <= 6.0 + 1e-6).all()  # ceil never saturates
    tf = quantize_mxfp4(x, scale_mode="floor")
    sf = torch.pow(2.0, tf.scales_e8m0_u8.float() - 127).unsqueeze(-1)
    assert (amax / sf <= 8.0 + 1e-6).all() and (amax / sf > 6.0).any()  # floor (OCP rule) can saturate


def test_nvfp4_scale_2d_tiles():
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    t = quantize_nvfp4(x, scale_2d=True)
    sc = t.block_scales_e4m3.float().reshape(2, 16, 4)
    assert torch.equal(sc[:, :1].expand(2, 16, 4), sc)  # all 16 rows of a tile share one scale
    assert torch.equal(dequantize_nvfp4(t), fake_quant_nvfp4(x, scale_2d=True))


# ------------------------------------------------------------------------------------------------ FP8
def test_fp8_emulation_matches_torch_cast():
    x = torch.cat([torch.randn(100000) * s for s in (1e-4, 1e-2, 1.0, 30.0, 400.0)] + [torch.linspace(-500, 500, 100001)])
    for fmt in ("e4m3", "e5m2"):
        assert torch.equal(F8.round_fp8(x, fmt), F8.emulate_fp8(x, fmt)), fmt


def test_fp8_scaling_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(16, 64) * 10
    x8, s = to_fp8(x, "e4m3")
    assert x8.dtype == torch.float8_e4m3fn and s.numel() == 1
    assert torch.isclose(s, x.abs().max() / 448.0)
    assert rel_mse(x, from_fp8(x8, s)) < 3e-3
    x8r, sr = to_fp8(x, "e4m3", dim=-1)
    assert sr.shape == (16, 1)
    assert rel_mse(x, from_fp8(x8r, sr)) <= rel_mse(x, from_fp8(x8, s)) * 1.05
    x8b, sb = to_fp8(x, "e5m2", block=16)
    assert sb.shape == (16, 4)
    assert rel_mse(x, from_fp8(x8b, sb, block=16)) < 1e-2
    assert torch.equal(fake_quant_fp8(x, "e4m3", dim=-1), from_fp8(x8r, sr))
    y8, _ = to_fp8(torch.tensor([1e9, -1e9]), "e4m3", scale=1.0)
    assert torch.isfinite(y8.float()).all() and y8.float().abs().max() == 448.0


# ------------------------------------------------------------------------------------------------ INT
@pytest.mark.parametrize("bits,sym", [(4, True), (4, False), (8, True), (3, False), (2, True)])
def test_int_roundtrip_properties(bits, sym):
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    q, s, z = quantize_int(w, bits, 64, sym)
    assert q.dtype == torch.uint8 and q.shape == w.shape
    assert s.shape == (32, 4) and z.shape == (32, 4)
    assert int(q.max()) <= (1 << bits) - 1
    d = dequantize_int(q, s, z, 64)
    step = s.repeat_interleave(64, dim=-1)
    assert ((d - w).abs() <= step * 0.5001 + 1e-6).all()
    if sym:
        assert torch.equal(d, dequantize_int(q, s, None, 64, bits=bits))
    assert torch.equal(fake_quant_int(w, bits, 64, sym), d)


def test_int_asym_represents_zero_and_dim():
    w = torch.randn(64, 32)
    w[:, 0] = 0.0
    q, s, z = quantize_int(w, 4, 32, False, dim=0)
    assert s.shape == (2, 32)
    d = dequantize_int(q, s, z, 32, dim=0)
    assert torch.equal(d[:, 0], torch.zeros(64))


def test_int_per_row_and_clip_search_not_worse():
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    w[:, 7] *= 20
    _, s, _ = quantize_int(w, 4, 0, True)
    assert s.shape == (16, 1)
    e0 = rel_mse(w, fake_quant_int(w, 4, 128, True))
    e1 = rel_mse(w, fake_quant_int(w, 4, 128, True, clip_search=True))
    assert e1 <= e0 + 1e-9


def test_int_stochastic_rounding_unbiased():
    w = torch.full((1, 1 << 16), 0.37)
    w[0, 0] = 1.0  # defines the scale: amax=1 -> step 1/7
    d = fake_quant_int(w, 4, 0, True, rounding="stochastic", seed=1)
    assert abs(d[0, 1:].mean().item() - 0.37) < 0.003


def test_six_db_per_bit_theory():
    """High-rate rule: each extra bit buys ~6.02 dB SQNR for a uniform quantiser on uniform data."""
    torch.manual_seed(0)
    x = torch.rand(1 << 18) * 2 - 1
    sq = {}
    for b in (4, 5, 6, 7, 8, 9, 10):
        d = fake_quant_int(x.reshape(1, -1), b, 0, sym=False)
        sq[b] = 10 * math.log10(x.pow(2).mean().item() / (x - d.reshape(-1)).pow(2).mean().item())
    # exact theory uses 2**b levels; a (2**b - 1)-level grid adds 20*log10((2**(b+1)-1)/(2**b-1)) - 6.02 dB at low b
    for b in range(6, 11):
        assert sq[b] - sq[b - 1] == pytest.approx(6.02, abs=0.15)
    assert (sq[10] - sq[4]) / 6 == pytest.approx(6.02, abs=0.1)
