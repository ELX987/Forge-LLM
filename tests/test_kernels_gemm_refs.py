"""CPU tests of the GEMM oracles (Plan 6.7 edge-case list as parametrised tests of refs/quantisers)."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from forge.kernels.layouts import (
    FRAGMENT_W4,
    PORTABLE_W4,
    PackedFP4,
    pack_w4,
    pack_w8,
    quantize_w4_rtn,
    quantize_w8_rtn,
)
from forge.kernels.ref import (
    compare,
    dequantize_fp4_ref,
    dequantize_fp8_ref,
    e2m1_decode,
    e2m1_encode,
    fp4_block_gemm_ref,
    fp4_w4a4_gemm_ref,
    fp8_gemm_ref,
    int8_gemm_ref,
    quantize_fp4_ref,
    quantize_fp8_ref,
    quantize_int8_ref,
    quantize_mxfp4_ref,
    quantize_nvfp4_ref,
    rmsnorm_act_quant_fp8_ref,
    w4a16_gemm_ref,
    w4a16_gemm_ref_parts,
    w8a16_gemm_ref,
)
from forge.kernels.ref.gemm_fp4 import (
    e4m3_bits_to_float,
    e8m0_bits_to_float,
    float_to_e4m3_bits,
)
from forge.kernels.ref.gemm_fp8 import expand_scale
from forge.kernels.ref.gemm_int8 import int8_matmul_exact
from forge.kernels.spec import load_spec


def rnd(*shape, seed=0, scale=1.0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale


# ===================================================================================== W4A16 / W8A16
@pytest.mark.parametrize("m", [0, 1, 3, 16, 33])
@pytest.mark.parametrize("n,k,gs", [(7, 100, 32), (64, 65, 16), (130, 257, 64), (5, 24, -1)])
@pytest.mark.parametrize("layout", [PORTABLE_W4, FRAGMENT_W4], ids=["portable", "fragment"])
def test_w4a16_ref_equals_dequant_then_matmul(m, n, k, gs, layout):
    """K not a multiple of the group, odd N, M=0 included: oracle == fp64 dequant-then-matmul."""
    w = rnd(n, k, seed=n + k)
    q, s, z = quantize_w4_rtn(w, gs, sym=False)
    p = pack_w4(q, s, z, gs, layout)
    x = rnd(m, k, seed=3)
    bias = rnd(n, seed=4)
    y = w4a16_gemm_ref(x, p, bias)
    assert y.shape == (m, n) and y.dtype == torch.float32
    gi = torch.arange(k) // (k if gs <= 0 else gs)
    wd = (q.double() - z.double()[:, gi]) * s.double()[:, gi]
    expect = x.double() @ wd.t() + bias.double()
    assert torch.allclose(y.double(), expect, rtol=1e-5, atol=1e-4)
    assert torch.allclose(w4a16_gemm_ref_parts(x, q, s, z, gs, bias), y, atol=1e-5)


def test_w4a16_ref_symmetric_zero_point_is_8_and_batch_dims_and_dtypes():
    w = rnd(32, 64)
    q, s, z = quantize_w4_rtn(w, 32, sym=True)
    assert z is None
    p = pack_w4(q, s, None, 32)
    x = rnd(2, 5, 64).bfloat16()
    y = w4a16_gemm_ref(x, p, out_dtype=None)
    assert y.shape == (2, 5, 32) and y.dtype == torch.bfloat16
    wd = p.dequantize()
    assert torch.allclose(y.float(), (x.float() @ wd.t()), rtol=2e-2, atol=2e-2)
    assert torch.equal(wd, (q.float() - 8) * s.float().repeat_interleave(32, dim=1))
    with pytest.raises(ValueError):
        w4a16_gemm_ref(rnd(2, 63), p)


def test_w4a16_ref_g_idx_act_order():
    n, k = 16, 64
    q = torch.randint(0, 16, (n, k), dtype=torch.uint8)
    s = rnd(n, 4).abs() + 0.1
    g_idx = torch.randint(0, 4, (k,), dtype=torch.int32)
    p = pack_w4(q, s, None, 16, g_idx=g_idx)
    wd = (q.float() - 8) * s[:, g_idx.long()]
    x = rnd(3, k)
    assert torch.allclose(w4a16_gemm_ref(x, p), x @ wd.t(), atol=1e-4)


@pytest.mark.parametrize("gs", [-1, 32])
def test_w8a16_ref(gs):
    w = rnd(21, 100)
    q, s = quantize_w8_rtn(w, gs)
    p = pack_w8(q, s, gs)
    x = rnd(4, 100)
    y = w8a16_gemm_ref(x, p, bias=torch.ones(21))
    assert torch.allclose(y, x @ p.dequantize().t() + 1, atol=1e-4)
    assert (y - (x @ w.t() + 1)).abs().max() < 0.05 * y.abs().max()
    assert w8a16_gemm_ref(torch.zeros(0, 100), p).shape == (0, 21)


# ===================================================================================== compare()
def test_compare_tolerance_rules_and_wrong_kernel_rejected():
    spec = load_spec("w4a16_gemm")
    ref = rnd(8, 64)
    assert compare(ref + 1e-5, ref, spec.tolerance).ok
    assert not compare(ref * 1.1, ref, spec.tolerance).ok  # deliberately wrong kernel
    bad = ref.clone()
    bad[3, 3] += 0.5
    assert not compare(bad, ref, spec.tolerance).ok
    # bf16 output rounding is allowed (dtype-aware tolerance)
    assert compare(ref.bfloat16(), ref, spec.tolerance, out_dtype=torch.bfloat16).ok
    nan = ref.clone()
    nan[0, 0] = float("nan")
    assert not compare(nan, ref).ok and compare(nan, nan).ok  # NaN must match positionally
    inf = ref.clone()
    inf[1, 1] = float("inf")
    assert compare(inf, inf).ok and not compare(inf, ref).ok
    assert not compare(ref[:4], ref).ok  # shape mismatch
    ints = torch.arange(10, dtype=torch.int32)
    assert compare(ints, ints.clone(), exact=True).ok and not compare(ints + 1, ints, exact=True).ok


# ===================================================================================== FP8
@pytest.mark.parametrize("gran", ["per_tensor", "per_token", "block"])
@pytest.mark.parametrize("k", [64, 100, 300])
def test_fp8_quant_dequant_error_bound(gran, k):
    x = rnd(7, k, scale=3.0)
    q, s = quantize_fp8_ref(x, gran, block=128)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
    d = dequantize_fp8_ref(q, s, block=128)
    # E4M3: 3 mantissa bits -> relative error <= 2**-4 of the element, plus subnormal floor of the scale
    bound = x.abs() * 2 ** -4 + s.max() * 2 ** -9 * (1 if gran != "block" else 1)
    assert ((d - x).abs() <= bound + 1e-6).all()
    assert q.float().abs().max() <= 448.0


def test_fp8_shapes_and_all_zero_rows_and_saturation():
    x = rnd(4, 130)
    x[1] = 0
    q, s = quantize_fp8_ref(x, "per_token")
    assert s.shape == (4,) and s[1].item() == 1.0 and (q[1].float() == 0).all()
    _, sb = quantize_fp8_ref(x, "block", block=128)
    assert sb.shape == (4, 2)
    _, s2 = quantize_fp8_ref(rnd(130, 70), "block2d", block=128)
    assert s2.shape == (2, 1)
    # static scale smaller than needed: values saturate at +-448, never NaN (torch casts would give NaN)
    big = torch.tensor([[1000.0, -1000.0, 1.0]])
    q, _ = quantize_fp8_ref(big, "per_tensor", scale=torch.tensor([1.0]))
    assert q.float().tolist() == [[448.0, -448.0, 1.0]]
    # the largest element maps onto the format maximum
    q, s = quantize_fp8_ref(torch.tensor([[2.0, -4.0]]), "per_token")
    assert q.float().abs().max().item() == 448.0
    assert expand_scale(torch.tensor([2.0]), 3, 5).shape == (3, 5)
    with pytest.raises(ValueError):
        expand_scale(torch.ones(3), 4, 5)


def test_e4m3_fn_conventions():
    bits = float_to_e4m3_bits(torch.tensor([0.0, 1.0, 448.0, 449.0, 1e9, -1e9, float("inf"), float("nan"), 2 ** -9, 2 ** -11]))
    vals = e4m3_bits_to_float(bits)
    assert vals[:3].tolist() == [0.0, 1.0, 448.0]
    assert vals[3].item() == 448.0 and vals[4].item() == 448.0 and vals[5].item() == -448.0  # saturating
    assert vals[6].item() == 448.0  # FN format has no infinity; +inf saturates here
    assert math.isnan(vals[7].item()) and bits[7].item() == 0x7F
    assert vals[8].item() == 2 ** -9 and vals[9].item() == 0.0  # smallest subnormal / underflow flush


@pytest.mark.parametrize("gx,gw", [("per_tensor", "per_tensor"), ("per_token", "per_channel"), ("block", "block2d")])
@pytest.mark.parametrize("m,n,k", [(1, 16, 64), (5, 40, 200), (130, 130, 260), (0, 8, 32)])
def test_fp8_gemm_ref_matches_manual_dequant(gx, gw, m, n, k):
    x, w = rnd(m, k, seed=1), rnd(n, k, seed=2)
    xq, sx = quantize_fp8_ref(x, gx)
    wq, sw = quantize_fp8_ref(w, gw)
    y = fp8_gemm_ref(xq, sx, wq, sw, bias=torch.arange(n).float())
    xd = dequantize_fp8_ref(xq, sx)
    wd = dequantize_fp8_ref(wq, sw)
    assert torch.allclose(y, xd @ wd.t() + torch.arange(n).float(), atol=1e-4, rtol=1e-5)
    if m:
        assert (y - (x @ w.t() + torch.arange(n).float())).abs().max() <= 0.12 * (x @ w.t()).abs().max() + 1e-3


def test_fused_rmsnorm_quant_oracle():
    x = rnd(6, 96, scale=4.0)
    w = rnd(96).abs() + 0.5
    q, s, y = rmsnorm_act_quant_fp8_ref(x, w, 1e-6)
    expect = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(y, expect, atol=1e-6)
    qq, ss = quantize_fp8_ref(expect, "per_token")
    assert torch.equal(q.view(torch.uint8), qq.view(torch.uint8)) and torch.equal(s, ss)
    q2, s2, _ = rmsnorm_act_quant_fp8_ref(x, None, 1e-6, granularity="block", block=32)
    assert s2.shape == (6, 3)


# ===================================================================================== INT8
def test_int8_exact_and_rounding_and_clamp():
    x = rnd(9, 70, scale=2)
    q, s = quantize_int8_ref(x)
    assert q.dtype == torch.int8 and q.min() >= -127 and (q.abs().amax(dim=1) == 127).all()
    # round-half-to-even on exact halves
    q2, _ = quantize_int8_ref(torch.tensor([[0.5, 1.5, 2.5, -0.5, -1.5, 127.0]]), scale=torch.tensor([1.0]))
    assert q2.tolist() == [[0, 2, 2, 0, -2, 127]]
    q3, _ = quantize_int8_ref(torch.tensor([[300.0, -300.0]]), scale=torch.tensor([1.0]))
    assert q3.tolist() == [[127, -127]]
    wq = torch.randint(-127, 128, (11, 70), dtype=torch.int8)
    acc = int8_matmul_exact(q, wq)
    assert acc.dtype == torch.int32 and torch.equal(acc, (q.to(torch.int64) @ wq.to(torch.int64).t()).to(torch.int32))
    sw = rnd(11).abs() + 0.1
    y = int8_gemm_ref(q, s, wq, sw, bias=torch.ones(11))
    assert torch.allclose(y, acc.float() * s[:, None] * sw[None, :] + 1, rtol=1e-6)
    # large K stays exact
    big = int8_matmul_exact(torch.full((1, 40000), 127, dtype=torch.int8), torch.full((1, 40000), 127, dtype=torch.int8))
    assert big.item() == 127 * 127 * 40000


# ===================================================================================== FP4
def test_e2m1_codec_table_and_rne_ties():
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    codes = torch.arange(16, dtype=torch.uint8)
    vals = e2m1_decode(codes)
    assert vals[:8].tolist() == grid and vals[8:].tolist() == [-v for v in grid]
    assert torch.equal(e2m1_encode(vals) & 7, codes & 7)  # every grid value encodes to its own magnitude code
    # exact ties -> even code (RNE): 0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5->4.0
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert e2m1_decode(e2m1_encode(ties)).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    just_above = torch.tensor([0.2500001, 0.7500001, 5.0000005, 6.0, 100.0, -100.0])
    assert e2m1_decode(e2m1_encode(just_above)).tolist() == [0.5, 1.0, 6.0, 6.0, 6.0, -6.0]  # saturating at +-6
    # nearest-value property on a dense sweep
    sweep = torch.linspace(-7, 7, 4001)
    d = e2m1_decode(e2m1_encode(sweep))
    allv = e2m1_decode(codes)
    best = allv[(sweep[:, None] - allv[None, :]).abs().argmin(dim=1)]
    assert ((d - sweep).abs() <= (best - sweep).abs() + 1e-6).all()


def test_nvfp4_matches_formula_on_known_block():
    x = torch.zeros(1, 32)
    x[0, :16] = torch.linspace(-6, 6, 16)  # block amax 6
    x[0, 16:] = torch.linspace(-3, 3, 16)  # block amax 3
    p = quantize_nvfp4_ref(x)
    ts = 6.0 / (448 * 6)
    assert math.isclose(p.tensor_scale.item(), ts, rel_tol=1e-6)
    stored = e4m3_bits_to_float(p.scales)  # (amax_b / 6) / ts
    assert torch.allclose(stored, torch.tensor([[448.0, 224.0]]))
    d = dequantize_fp4_ref(p)
    assert d.shape == x.shape and (d - x).abs().max() <= 1.0 + 1e-5  # half of the widest grid step (4..6) x block scale 1.0
    assert p.payload.shape == (1, 16) and p.scales.shape == (1, 2) and p.payload.dtype == torch.uint8


def test_nvfp4_bits_per_element_is_4_5_and_mxfp4_4_25():
    x = rnd(256, 512)
    n = quantize_nvfp4_ref(x)
    m = quantize_mxfp4_ref(x)
    per = lambda p: 8 * (p.payload.numel() + p.scales.numel()) / x.numel()  # noqa: E731  (tensor scale excluded)
    assert per(n) == 4.5 and per(m) == 4.25
    assert n.bits_per_element == pytest.approx(4.5, abs=0.01)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("k", [16, 20, 32, 33, 100])
def test_fp4_roundtrip_error_and_k_padding(fmt, k):
    x = rnd(9, k, seed=k, scale=2.0)
    p = quantize_fp4_ref(x, fmt)
    assert p.k == k and p.k_pad % p.block == 0 and p.payload.shape == (9, p.k_pad // 2)
    d = dequantize_fp4_ref(p)
    assert d.shape == x.shape
    # per-block bound: <= 1 grid step of the block (max step is 2 between 4 and 6) times the block scale
    assert (d - x).abs().max() <= x.abs().max() * 0.5 + 1e-5
    rel = (d - x).norm() / x.norm()
    assert rel < 0.25
    for layout in ("logical", "swizzled"):
        assert torch.equal(dequantize_fp4_ref(p.with_scale_layout(layout)), d)


def test_nvfp4_zero_block_and_e4m3_underflow_flush_blocks():
    x = torch.zeros(2, 48)
    x[0, 16:32] = 100.0 * torch.linspace(0.1, 1, 16)  # normal block
    x[1, 0:16] = 1e-7  # underflows E4M3 relative to the tensor scale
    x[1, 16:32] = 50.0
    p = quantize_nvfp4_ref(x)
    d = dequantize_fp4_ref(p)
    assert torch.isfinite(d).all()  # flushed blocks must not turn into 0/0
    assert (d[0, :16] == 0).all() and (d[0, 32:] == 0).all()  # all-zero blocks stay exactly zero
    assert (d[1, :16] == 0).all()  # underflowing block flushed (scale rounds to 0 in E4M3)
    assert e4m3_bits_to_float(p.scales)[1, 0].item() == 0.0
    assert (d[1, 16:32] - 50.0).abs().max() < 4.0


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fp4_nan_inf_poison_their_block_only(fmt, bad):
    blk = 16 if fmt == "nvfp4" else 32
    x = rnd(2, 2 * blk)
    x[1, blk + 3] = bad
    d = dequantize_fp4_ref(quantize_fp4_ref(x, fmt))
    assert torch.isnan(d[1, blk:]).all()  # whole offending block is NaN (FP4 has no NaN code; the scale carries it)
    assert torch.isfinite(d[1, :blk]).all() and torch.isfinite(d[0]).all()


def test_mxfp4_scale_exponent_rule_and_saturation():
    def scale_code(amax):
        x = torch.zeros(1, 32)
        x[0, 0] = amax
        return quantize_mxfp4_ref(x).scales[0, 0].item()

    assert scale_code(6.0) == 127  # floor(log2 6)=2, minus emax(2) -> 2**0
    assert scale_code(7.9) == 127
    assert scale_code(8.0) == 128 and scale_code(1.0) == 125 and scale_code(0.0) == 127  # zero block -> scale 1
    d = dequantize_fp4_ref(quantize_mxfp4_ref(torch.tensor([[7.9] + [0.0] * 31])))
    assert d[0, 0].item() == 6.0  # saturates at the E2M1 maximum
    assert e8m0_bits_to_float(torch.tensor([127, 128, 0], dtype=torch.uint8)).tolist() == [1.0, 2.0, 2.0 ** -127]
    assert math.isnan(e8m0_bits_to_float(torch.tensor([255], dtype=torch.uint8)).item())


def test_fp4_gemm_ref_weight_only_and_w4a4():
    w, x = rnd(24, 96, seed=1), rnd(5, 96, seed=2)
    for fmt in ("nvfp4", "mxfp4"):
        wp = quantize_fp4_ref(w, fmt)
        y = fp4_block_gemm_ref(x, wp, bias=torch.ones(24))
        assert torch.allclose(y, x @ dequantize_fp4_ref(wp).t() + 1, atol=1e-4)
        xp = quantize_fp4_ref(x, fmt)
        y44 = fp4_block_gemm_ref(xp, wp)
        assert torch.allclose(y44, dequantize_fp4_ref(xp) @ dequantize_fp4_ref(wp).t(), atol=1e-4)
        assert torch.allclose(fp4_w4a4_gemm_ref(x, wp), y44, atol=1e-5)
        assert (y - x @ w.t()).norm() / (x @ w.t()).norm() < 0.3
    with pytest.raises(ValueError):
        quantize_fp4_ref(x, "int4")


def test_from_nvfp4_tensor_adapter():
    class Fake:  # duck-typed forge.formats.fp4.NVFP4Tensor
        pass

    p = quantize_nvfp4_ref(rnd(8, 64))
    f = Fake()
    f.payload_u8, f.block_scales_e4m3 = p.payload, p.scales.view(torch.float8_e4m3fn)
    f.tensor_scale, f.shape = p.tensor_scale, (8, 64)
    q = PackedFP4.from_nvfp4_tensor(f)
    assert q.fmt == "nvfp4" and torch.equal(q.scales, p.scales) and torch.equal(dequantize_fp4_ref(q), dequantize_fp4_ref(p))


def test_int8_ref_scale_is_correctly_rounded_division():
    import torch
    from forge.kernels.ref import quantize_int8_ref

    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, 64, generator=g) * torch.rand(512, 1, generator=g) * 10
    _, s = quantize_int8_ref(x, "per_token")
    amax = x.abs().amax(dim=-1)
    assert torch.equal(s, (amax.double() / 127.0).float())


def test_int8_quant_agreement_flags_only_untied_mismatches():
    import torch
    from forge.kernels.ref import int8_quant_agreement, quantize_int8_ref

    x = torch.randn(8, 128)
    q, s = quantize_int8_ref(x, "per_token")
    st = int8_quant_agreement(x, q, s, q, s)
    assert st["code_mismatches"] == 0 and st["scale_ulp_max"] == 0
    q2 = q.clone()
    q2[0, 0] += 1  # a random (non-tie) flip must be reported as untied
    st = int8_quant_agreement(x, q2, s, q, s)
    assert st["untied_mismatches"] == 1 and st["max_code_diff"] == 1
    s2 = torch.nextafter(s, torch.full_like(s, 1e9))
    assert int8_quant_agreement(x, q, s2, q, s)["scale_ulp_max"] == 1.0
    # an exact tie: x/s == 2.5 -> oracle gives 2 (half-even); a kernel giving 3 is excused
    xt = torch.tensor([[2.5, 127.0]])
    qt, st_ = quantize_int8_ref(xt, "per_token")
    assert qt[0, 0] == 2
    qk = qt.clone(); qk[0, 0] = 3
    assert int8_quant_agreement(xt, qk, st_, qt, st_)["untied_mismatches"] == 0
