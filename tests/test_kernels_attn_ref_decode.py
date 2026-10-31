"""CPU tests of the decode-attention oracles (Plan K4 / §6.7 edge cases)."""
from __future__ import annotations

import math

import pytest
import torch

from forge.kernels.ref.attn_decode import (
    dense_attention_ref,
    merge_partials_ref,
    mla_decode_attention_ref,
    paged_decode_attention_ref,
    split_kv_decode_ref,
)
from forge.kernels.ref.attn_kv import (
    build_paged_pools,
    decode_e4m3_bits,
    dequantize_pool,
    pack_int4,
    unpack_int4,
)

torch.manual_seed(0)


def _dense_ref(q, k, v, n, scale, window=None):
    """Independent straightforward implementation: q [Hq,D], k,v [Hkv,L,D]."""
    hq, d = q.shape
    hkv = k.shape[0]
    g = hq // hkv
    out = torch.zeros(hq, d)
    lo = 0 if not window else max(0, n - window)
    for h in range(hq):
        kk, vv = k[h // g, lo:n].float(), v[h // g, lo:n].float()
        p = torch.softmax((kk @ q[h].float()) * scale, dim=0)
        out[h] = p @ vv
    return out


def _make(b, hq, hkv, d, lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    lmax = max(lens)
    q = torch.randn(b, hq, d, generator=g)
    k = torch.randn(b, hkv, lmax, d, generator=g)
    v = torch.randn(b, hkv, lmax, d, generator=g)
    return q, k, v


@pytest.mark.parametrize("gsize", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("page", [16, 32])
def test_gqa_groups_and_ragged_lengths(gsize, page):
    hkv, d = 2, 32
    hq = hkv * gsize
    lens = [1, page - 1, page, page + 1, 3 * page + 5, 0]
    q, k, v = _make(len(lens), hq, hkv, d, [max(lens)] * len(lens))
    pools = build_paged_pools(k, v, lens, page, "fp32")
    out = paged_decode_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                     kv_dtype="fp32")
    scale = 1 / math.sqrt(d)
    for bi, n in enumerate(lens):
        if n == 0:
            assert torch.count_nonzero(out[bi]) == 0
        else:
            torch.testing.assert_close(out[bi], _dense_ref(q[bi], k[bi], v[bi], n, scale), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("window", [1, 7, 16, 100])
def test_sliding_window(window):
    lens = [5, 33, 70]
    q, k, v = _make(3, 8, 2, 16, lens)
    pools = build_paged_pools(k, v, lens, 16, "fp32")
    out = paged_decode_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                     kv_dtype="fp32", window=window)
    for bi, n in enumerate(lens):
        torch.testing.assert_close(out[bi], _dense_ref(q[bi], k[bi], v[bi], n, 1 / math.sqrt(16), window), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kvd,tol", [("fp8_e4m3", 0.12), ("int8", 0.02), ("int4", 0.35)])
@pytest.mark.parametrize("asym", [False, True])
def test_quantised_kv_close_to_fp(kvd, tol, asym):
    if asym and kvd == "fp8_e4m3":
        pytest.skip("fp8 has no zero point")
    lens = [20, 47]
    q, k, v = _make(2, 4, 2, 32, lens)
    pools = build_paged_pools(k, v, lens, 16, kvd, asymmetric=asym)
    out = paged_decode_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                     kv_dtype=kvd, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
                                     k_zero=pools["k_zero"], v_zero=pools["v_zero"])
    for bi, n in enumerate(lens):
        ref = _dense_ref(q[bi], k[bi], v[bi], n, 1 / math.sqrt(32))
        err = (out[bi] - ref).abs().max() / ref.abs().max()
        assert err < tol, (kvd, float(err))


def test_quantisation_error_ordering():
    lens = [64]
    q, k, v = _make(1, 4, 2, 32, lens)
    errs = {}
    ref = _dense_ref(q[0], k[0], v[0], 64, 1 / math.sqrt(32))
    for kvd in ("int8", "fp8_e4m3", "int4"):
        p = build_paged_pools(k, v, lens, 16, kvd)
        o = paged_decode_attention_ref(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], kv_dtype=kvd,
                                       k_scale=p["k_scale"], v_scale=p["v_scale"])
        errs[kvd] = float((o[0] - ref).abs().mean())
    assert errs["int8"] < errs["fp8_e4m3"] < errs["int4"]


def test_residual_window_matches_full_precision():
    lens = [40, 21]
    res = [7, 0]
    q, k, v = _make(2, 4, 2, 16, lens)
    pools = build_paged_pools(k, v, lens, 16, "fp32")
    kr = torch.zeros(2, 2, 8, 16)
    vr = torch.zeros(2, 2, 8, 16)
    kr[0, :, :7] = k[0, :, 33:40]
    vr[0, :, :7] = v[0, :, 33:40]
    out_res = paged_decode_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                         kv_dtype="fp32", k_res=kr, v_res=vr, res_lens=torch.tensor(res))
    out_full = paged_decode_attention_ref(q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"],
                                          kv_dtype="fp32")
    torch.testing.assert_close(out_res, out_full, atol=1e-5, rtol=1e-5)


def test_topk_selection_all_tokens_equals_dense_and_subset_differs():
    lens = [30]
    q, k, v = _make(1, 4, 2, 16, lens)
    pools = build_paged_pools(k, v, lens, 16, "fp32")
    args = (q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"])
    dense = paged_decode_attention_ref(*args, kv_dtype="fp32")
    sel_all = torch.arange(30, dtype=torch.int32).view(1, 1, 30)
    torch.testing.assert_close(paged_decode_attention_ref(*args, kv_dtype="fp32", topk_idx=sel_all), dense, atol=1e-5, rtol=1e-5)
    sel_sub = torch.tensor([[[0, 3, 7, -1]]], dtype=torch.int32)
    sub = paged_decode_attention_ref(*args, kv_dtype="fp32", topk_idx=sel_sub)
    expect = torch.zeros(4, 16)
    for h in range(4):
        kk, vv = k[0, h // 2, [0, 3, 7]], v[0, h // 2, [0, 3, 7]]
        expect[h] = torch.softmax(kk @ q[0, h] / 4.0, 0) @ vv
    torch.testing.assert_close(sub[0], expect, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("splits", [1, 2, 3, 7])
def test_split_kv_lse_merge_is_exact(splits):
    lens = [100, 9, 33]
    q, k, v = _make(3, 8, 2, 16, lens)
    pools = build_paged_pools(k, v, lens, 16, "fp32")
    args = (q, pools["k_pool"], pools["v_pool"], pools["block_tables"], pools["seq_lens"])
    full = paged_decode_attention_ref(*args, kv_dtype="fp32")
    spl = split_kv_decode_ref(*args, num_splits=splits, kv_dtype="fp32")
    torch.testing.assert_close(spl, full, atol=1e-5, rtol=1e-5)


def test_merge_handles_empty_splits_and_extreme_logits():
    o = torch.randn(2, 3, 4, 8)
    lse = torch.tensor([[[-math.inf, 1000.0, -math.inf]] * 2] * 2).reshape(2, 2, 3)
    lse = lse.unsqueeze(1).expand(2, 3, 2, 3)[:, :, 0]  # [2,3,3]
    o = torch.randn(2, 3, 3, 8)
    out, tot = merge_partials_ref(o, lse)
    torch.testing.assert_close(out, o[:, :, 1])
    assert torch.isfinite(tot).all()
    out0, tot0 = merge_partials_ref(torch.randn(1, 2, 8), torch.full((1, 2), -math.inf))
    assert torch.count_nonzero(out0) == 0 and torch.isinf(tot0).all()


def test_int4_roundtrip_and_fp8_bit_decode():
    q = torch.randint(-8, 8, (5, 32), dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)
    allb = torch.arange(256, dtype=torch.uint8)
    torch_vals = allb.view(torch.float8_e4m3fn).float()
    mine = decode_e4m3_bits(allb)
    nan = torch.isnan(torch_vals)
    assert torch.equal(torch.isnan(mine), nan)
    assert torch.equal(mine[~nan], torch_vals[~nan])


def test_dequantize_pool_shapes_and_scale_forms():
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    p = build_paged_pools(k, v, [16], 16, "int8", shuffle=False, extra_pages=0)
    full = dequantize_pool(p["k_pool"], p["k_scale"], None, "int8", "k")
    assert full.shape == (1, 2, 16, 8)
    # scalar / [P,H,1] / [H,D] forms are accepted
    s1 = dequantize_pool(p["k_pool"], p["k_scale"].amax().item(), None, "int8", "k")
    s2 = dequantize_pool(p["k_pool"], p["k_scale"][:, :, :1], None, "int8", "k")
    s3 = dequantize_pool(p["k_pool"], p["k_scale"][0], None, "int8", "k")
    assert s1.shape == s2.shape == s3.shape == full.shape
    torch.testing.assert_close(s3, full)


def test_mla_decode_ref_matches_explicit_and_sparse():
    b, h, dc, dr, ps = 2, 4, 32, 8, 16
    lens = [40, 17]
    g = torch.Generator().manual_seed(3)
    c = torch.randn(b, max(lens), dc, generator=g)
    r = torch.randn(b, max(lens), dr, generator=g)
    qn = torch.randn(b, h, dc, generator=g)
    qr = torch.randn(b, h, dr, generator=g)
    total = 3 * b
    c_pool = torch.zeros(total, ps, dc)
    r_pool = torch.zeros(total, ps, dr)
    bt = torch.zeros(b, 3, dtype=torch.int32)
    nxt = 5
    for bi in range(b):
        for pi in range((lens[bi] + ps - 1) // ps):
            bt[bi, pi] = nxt % total
            lo, hi = pi * ps, min(lens[bi], (pi + 1) * ps)
            c_pool[nxt % total, : hi - lo] = c[bi, lo:hi]
            r_pool[nxt % total, : hi - lo] = r[bi, lo:hi]
            nxt += 1
    scale = 0.2
    w_uv = torch.randn(h, dc, 6, generator=g)
    out = mla_decode_attention_ref(qn, qr, c_pool, r_pool, bt, torch.tensor(lens), softmax_scale=scale, w_uv=w_uv)
    for bi, n in enumerate(lens):
        s = (qn[bi] @ c[bi, :n].T + qr[bi] @ r[bi, :n].T) * scale
        o = torch.softmax(s, -1) @ c[bi, :n]
        exp = torch.einsum("hc,hcv->hv", o, w_uv)
        torch.testing.assert_close(out[bi], exp, atol=1e-4, rtol=1e-4)
    sel = torch.tensor([[0, 5, 9, -1], [1, 2, -1, -1]], dtype=torch.int32)
    sp = mla_decode_attention_ref(qn, qr, c_pool, r_pool, bt, torch.tensor(lens), softmax_scale=scale, topk_idx=sel)
    for bi in range(b):
        idx = [int(i) for i in sel[bi] if i >= 0]
        s = (qn[bi] @ c[bi, idx].T + qr[bi] @ r[bi, idx].T) * scale
        torch.testing.assert_close(sp[bi], torch.softmax(s, -1) @ c[bi, idx], atol=1e-4, rtol=1e-4)
    # window
    win = mla_decode_attention_ref(qn, qr, c_pool, r_pool, bt, torch.tensor(lens), softmax_scale=scale, window=5)
    n = lens[0]
    s = (qn[0] @ c[0, n - 5 : n].T + qr[0] @ r[0, n - 5 : n].T) * scale
    torch.testing.assert_close(win[0], torch.softmax(s, -1) @ c[0, n - 5 : n], atol=1e-4, rtol=1e-4)


def test_dense_attention_ref_fully_masked_row():
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 4, 8)
    o, l = dense_attention_ref(q, k, k, torch.zeros(1, 4, dtype=torch.bool), 0.3)
    assert torch.count_nonzero(o) == 0 and torch.isinf(l).all()
