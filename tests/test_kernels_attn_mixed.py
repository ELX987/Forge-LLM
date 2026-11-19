"""CPU tests for mixed-precision block-scaled attention (``ref.attn_mixed`` oracle + ``ops_attn.mixed_attention``).

Covers: the format numerics (E2M1 / ternary / E4M3 / E8M0 codecs, block quantise/dequantise round-trips), the two-pass
oracle against a plain dense softmax for every KV format, sliding window / softcap / sink / GQA / ragged lengths, and the
dispatcher's reference path.  The Triton kernel itself is exercised in the Triton CPU interpreter by
``tests/interp_attn_mixed_cases.py``; GPU validation lives in ``tests_gpu/test_kernels_attn_mixed.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from forge.formats.fp4 import decode_e2m1
from forge.kernels.ops_attn import mixed_attention
from forge.kernels.ref.attn_kv import (
    build_paged_pools,
    decode_e2m1_codes,
    decode_ternary_codes,
    dequantize_block_pool,
    e8m0_exponent,
    is_block_quantised,
    kv_bytes_per_element,
    kv_kind_id,
    logical_head_dim,
    pack_int4,
    pack_ternary_codes,
    unpack_int4,
)
from forge.kernels.ref.attn_mixed import MIXED_KV_DTYPES, mixed_attention_ref

BLOCK_FORMATS = ("nvfp4", "mxfp4", "ternary")
ALL_FORMATS = MIXED_KV_DTYPES


def _make(b, hq, hkv, d, lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    lmax = max(lens) if lens else 1
    q = torch.randn(b, hq, d, generator=g)
    k = torch.randn(b, hkv, lmax, d, generator=g)
    v = torch.randn(b, hkv, lmax, d, generator=g)
    return q, k, v


def _dense_ref(q, k, v, n, scale, window=None):
    """Straightforward per-head softmax over the *full-precision* k/v (no quantisation) for one sequence."""
    hq, d = q.shape
    hkv = k.shape[0]
    g = hq // hkv
    out = torch.zeros(hq, d)
    lo = 0 if not window else max(0, n - window)
    for h in range(hq):
        kk, vv = k[h // g, lo:n].float(), v[h // g, lo:n].float()
        out[h] = torch.softmax((kk @ q[h].float()) * scale, dim=0) @ vv
    return out


# -------------------------------------------------------------------------------------------------- format numerics
def test_e2m1_codec_matches_reference():
    codes = torch.arange(16, dtype=torch.uint8)
    torch.testing.assert_close(decode_e2m1_codes(codes), decode_e2m1(codes), atol=0, rtol=0)


def test_ternary_codec_table():
    codes = torch.arange(4, dtype=torch.uint8)
    assert decode_ternary_codes(codes).tolist() == [0.0, 1.0, 0.0, -1.0]


def test_pack_ternary_roundtrip():
    t = torch.tensor([[-1, 0, 1, 1, 0, -1, 1, 0]], dtype=torch.int16)
    packed = pack_ternary_codes(t)
    from forge.formats.packing import unpack_bits

    back = decode_ternary_codes(unpack_bits(packed, 2))
    torch.testing.assert_close(back, t.float())


def test_int4_roundtrip_still_works():
    q = torch.randint(-8, 8, (4, 32), dtype=torch.int8)
    torch.testing.assert_close(unpack_int4(pack_int4(q)), q)


def test_e8m0_exponent_avoids_log2_boundary():
    # matches ceil(log2(x)) exactly (with all-zero blocks -> 0) but via frexp, so it never rounds a maximum just above a
    # power of two *down* onto it (a log2 approximation can)
    amax = torch.tensor([2.0000005, 1.9999995, 4.0, 3.9999998, 0.0, 1.0, 2.0, 0.5, 6.0])
    assert e8m0_exponent(amax).tolist() == [2, 1, 2, 2, 0, 0, 1, -1, 3]
    nz = amax > 0
    assert torch.equal(e8m0_exponent(amax)[nz].float(), torch.ceil(torch.log2(amax[nz])))
    assert (torch.exp2(e8m0_exponent(amax).float()) >= amax.clamp_min(1e-30)).all()


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
def test_block_pool_roundtrip_close(fmt):
    # one full page so the dequantised pool rows line up 1:1 with the source k/v
    q, k, v = _make(1, 2, 2, 64, [16], seed=1)
    pools = build_paged_pools(k, v, [16], 16, fmt, shuffle=False, extra_pages=0)
    kd = dequantize_block_pool(pools["k_pool"], pools["k_block_scale"], fmt, "k", pools.get("tensor_scale"))
    vd = dequantize_block_pool(pools["v_pool"], pools["v_block_scale"], fmt, "v", pools.get("tensor_scale"))
    pg = pools["block_tables"][0, 0].long()
    rel_k = float((kd[pg] - k[0]).abs().max() / k[0].abs().max())
    rel_v = float((vd[pg] - v[0]).abs().max() / v[0].abs().max())
    tol = {"nvfp4": 0.25, "mxfp4": 0.25, "ternary": 0.7}[fmt]
    assert rel_k < tol and rel_v < tol, (fmt, rel_k, rel_v)


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
def test_block_scale_shapes(fmt):
    q, k, v = _make(1, 2, 2, 64, [16], seed=2)
    p = build_paged_pools(k, v, [16], 16, fmt, shuffle=False, extra_pages=0)
    bs = 32 if fmt in ("mxfp4", "ternary") else 16
    assert p["k_block_scale"].shape[-1] == 64 // bs
    assert p["v_block_scale"].shape[-1] == 64 // bs


def test_bytes_per_element_and_kind():
    assert kv_bytes_per_element("nvfp4") == pytest.approx(4.5 / 8)
    assert kv_bytes_per_element("mxfp4") == pytest.approx(4.25 / 8)
    assert kv_bytes_per_element("ternary") == pytest.approx(3.0 / 8)
    assert kv_kind_id("nvfp4") == 4 and kv_kind_id("mxfp4") == 5 and kv_kind_id("ternary") == 6
    assert is_block_quantised("nvfp4") and is_block_quantised("ternary") and not is_block_quantised("int8")
    assert logical_head_dim(torch.zeros(1, 1, 16, 16), "ternary") == 64


# -------------------------------------------------------------------------------------------------- oracle correctness
@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_oracle_matches_dense_fp_for_float_formats(fmt):
    # the exact float formats must match a plain softmax; the quantised formats are only *close* (their error is the
    # quantisation error, bounded loosely here - the point is that the two-pass maths is not the source of error)
    q, k, v = _make(1, 4, 2, 32, [40], seed=3)
    pools = build_paged_pools(k, v, [40], 16, fmt, shuffle=False, extra_pages=0)
    out = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
                              k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
                              tensor_scale=pools.get("tensor_scale"))
    ref = _dense_ref(q[0], k[0], v[0], 40, 1 / math.sqrt(32))
    rel = float((out[0] - ref).abs().max() / ref.abs().max())
    tol = {"bf16": 1e-2, "fp16": 1e-2, "fp32": 1e-5, "fp8_e4m3": 0.2, "int8": 0.1, "int4": 0.6,
           "nvfp4": 0.4, "mxfp4": 0.4, "ternary": 1.6}[fmt]
    assert rel < tol, (fmt, rel)


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_oracle_runs_and_is_finite(fmt):
    q, k, v = _make(3, 4, 2, 32, [1, 17, 40], seed=4)
    pools = build_paged_pools(k, v, [1, 17, 40], 16, fmt, shuffle=False)
    out = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
                              k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
                              tensor_scale=pools.get("tensor_scale"))
    assert torch.isfinite(out.float()).all()
    assert out.shape == q.shape


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
def test_oracle_quantisation_error_ordering(fmt):
    # a 1.58-bit cache is lossier than FP4, which is lossier than int8 - the expected precision ordering
    q, k, v = _make(1, 4, 2, 64, [64], seed=5)
    ref = _dense_ref(q[0], k[0], v[0], 64, 1 / math.sqrt(64))
    errs = {}
    for f in ("int8", fmt):
        pools = build_paged_pools(k, v, [64], 16, f, shuffle=False)
        o = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                kv_dtype=f, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
                                k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
                                tensor_scale=pools.get("tensor_scale"))
        errs[f] = float((o[0] - ref).abs().mean())
    assert errs["int8"] < errs[fmt]


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
@pytest.mark.parametrize("window", [1, 5, 16])
def test_oracle_sliding_window(fmt, window):
    q, k, v = _make(1, 4, 2, 32, [40], seed=6)
    pools = build_paged_pools(k, v, [40], 16, fmt, shuffle=False)
    kw = dict(kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
              k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
              tensor_scale=pools.get("tensor_scale"))
    # compare the windowed oracle against the windowed *float* oracle (both attend the same last `window` tokens)
    p_float = build_paged_pools(k, v, [40], 16, "fp32", shuffle=False)
    win = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              window=window, **kw)
    ref = mixed_attention_ref(q, p_float["k_pool"], p_float["v_pool"], p_float["block_tables"], p_float["seq_lens"],
                              kv_dtype="fp32", window=window)
    rel = float((win[0] - ref[0]).abs().max() / ref[0].abs().max())
    assert rel < 1.6, (fmt, window, rel)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_oracle_softcap(fmt):
    q, k, v = _make(1, 4, 2, 32, [40], seed=7)
    pools = build_paged_pools(k, v, [40], 16, fmt, shuffle=False)
    kw = dict(kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
              k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
              tensor_scale=pools.get("tensor_scale"))
    big = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              softcap=1e6, **kw)
    base = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"], **kw)
    torch.testing.assert_close(big, base, atol=1e-5, rtol=1e-5)  # a huge softcap is the identity


def test_oracle_sink_is_noop_at_minus_inf():
    q, k, v = _make(1, 4, 2, 32, [40], seed=8)
    pools = build_paged_pools(k, v, [40], 16, "nvfp4", shuffle=False)
    kw = dict(kv_dtype="nvfp4", k_block_scale=pools["k_block_scale"], v_block_scale=pools["v_block_scale"],
              tensor_scale=pools["tensor_scale"], return_lse=True)
    base, lse0 = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"], **kw)
    sink, lse1 = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                     sink=torch.full((4,), float("-inf")), **kw)
    torch.testing.assert_close(sink, base, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lse1, lse0, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("bad_sink", [1000.0, float("inf"), float("nan")])
def test_oracle_sink_extreme_values_stay_finite_or_sink_only(bad_sink):
    # a huge / +inf sink must not overflow to inf from exp2(sink - m); a NaN sink must be cancelled, not poison the row.
    q, k, v = _make(1, 4, 2, 32, [40], seed=15)
    pools = build_paged_pools(k, v, [40], 16, "nvfp4", shuffle=False)
    kw = dict(kv_dtype="nvfp4", k_block_scale=pools["k_block_scale"], v_block_scale=pools["v_block_scale"],
              tensor_scale=pools["tensor_scale"], return_lse=True)
    base, lse0 = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"], **kw)
    out, lse = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                   sink=torch.full((4,), bad_sink), **kw)
    assert not torch.isnan(lse).any()  # never NaN; +inf is allowed for a +inf sink (the row is sink-only)
    assert torch.isfinite(out.float()).all()
    if bad_sink == float("nan"):
        # a NaN sink has no contribution: identical to no sink
        torch.testing.assert_close(out, base, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse, lse0, atol=1e-5, rtol=1e-5)
    elif bad_sink == float("inf"):
        # the sink dominates every row: output collapses to zero (sink carries no value)
        torch.testing.assert_close(out.float(), torch.zeros_like(out.float()), atol=1e-6, rtol=0)


def test_oracle_quantized_p_path():
    # p_dtype rounds P to fp16/bf16 before PV; the LSE stays exact, so the result is close to the fp32 path
    q, k, v = _make(1, 4, 2, 32, [40], seed=16)
    pools = build_paged_pools(k, v, [40], 16, "nvfp4", shuffle=False)
    kw = dict(kv_dtype="nvfp4", k_block_scale=pools["k_block_scale"], v_block_scale=pools["v_block_scale"],
              tensor_scale=pools["tensor_scale"])
    fp32 = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"], **kw)
    fp16p = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                p_dtype=torch.float16, **kw)
    rel = float((fp16p[0] - fp32[0]).abs().max() / fp32[0].abs().max())
    assert rel < 1e-2, rel


def test_oracle_sink_lowers_output_magnitude():
    q, k, v = _make(1, 4, 2, 32, [40], seed=9)
    pools = build_paged_pools(k, v, [40], 16, "nvfp4", shuffle=False)
    kw = dict(kv_dtype="nvfp4", k_block_scale=pools["k_block_scale"], v_block_scale=pools["v_block_scale"],
              tensor_scale=pools["tensor_scale"])
    base = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"], **kw)
    with_sink = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                    sink=torch.full((4,), 5.0), **kw)
    assert float(with_sink.abs().sum()) < float(base.abs().sum())


@pytest.mark.parametrize("gsize", [1, 2, 4, 8])
def test_oracle_gqa_groups(gsize):
    hkv = 2
    q, k, v = _make(2, hkv * gsize, hkv, 32, [40, 19], seed=10)
    pools = build_paged_pools(k, v, [40, 19], 16, "mxfp4", shuffle=False)
    out = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              kv_dtype="mxfp4", k_block_scale=pools["k_block_scale"],
                              v_block_scale=pools["v_block_scale"], tensor_scale=pools["tensor_scale"])
    assert out.shape == q.shape and torch.isfinite(out.float()).all()


def test_oracle_empty_sequence_is_zero():
    q, k, v = _make(2, 4, 2, 32, [0, 10], seed=11)
    pools = build_paged_pools(k, v, [0, 10], 16, "ternary", shuffle=False)
    out = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              kv_dtype="ternary", k_block_scale=pools["k_block_scale"],
                              v_block_scale=pools["v_block_scale"])
    assert torch.count_nonzero(out[0]) == 0


def test_oracle_rejects_bad_head_dim():
    # 48 is not a multiple of 32 -> ternary can not pack it
    k = torch.randn(1, 2, 16, 48)
    v = torch.randn(1, 2, 16, 48)
    with pytest.raises(ValueError):
        build_paged_pools(k, v, [16], 16, "ternary")


# -------------------------------------------------------------------------------------------------- dispatcher
@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_dispatch_returns_reference_on_cpu(fmt):
    q, k, v = _make(2, 4, 2, 32, [40, 19], seed=12)
    pools = build_paged_pools(k, v, [40, 19], 16, fmt, shuffle=False)
    out = mixed_attention(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                          kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
                          k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
                          tensor_scale=pools.get("tensor_scale"))
    ref = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
                              k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
                              tensor_scale=pools.get("tensor_scale"))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_dispatch_float_delegates_to_paged_kernel_path():
    # bf16 goes through the existing K4 path; on CPU both resolve to the same reference maths
    q, k, v = _make(1, 4, 2, 32, [30], seed=13)
    pools = build_paged_pools(k, v, [30], 16, "bf16", shuffle=False)
    out = mixed_attention(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"], kv_dtype="bf16")
    ref = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              kv_dtype="bf16")
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_dispatch_unknown_format_raises():
    q = torch.randn(1, 2, 32)
    with pytest.raises(ValueError):
        mixed_attention(q, torch.zeros(1, 1, 16, 32), torch.zeros(1, 1, 16, 32), torch.zeros(1, 1, dtype=torch.int32),
                        torch.tensor([0]), kv_dtype="bf8")


def test_dispatch_with_softcap_and_sink_uses_oracle():
    q, k, v = _make(1, 4, 2, 32, [40], seed=14)
    pools = build_paged_pools(k, v, [40], 16, "nvfp4", shuffle=False)
    kw = dict(kv_dtype="nvfp4", k_block_scale=pools["k_block_scale"], v_block_scale=pools["v_block_scale"],
              tensor_scale=pools["tensor_scale"])
    out = mixed_attention(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                          softcap=10.0, sink=torch.zeros(4), **kw)
    ref = mixed_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                              softcap=10.0, sink=torch.zeros(4), **kw)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
