"""Adversarial-review cases for the attention / MoE / util Triton kernels, executed in the Triton CPU interpreter.

Run by ``test_kernels_attn_review_interp.py`` in a subprocess with ``TRITON_INTERPRET=1`` (must precede the first triton import).
These cases target edge geometry that the original suites do not cover: non-power-of-two head dims and GQA groups, page
sizes 1 / 8 / 64, empty pools with only a residual window, windows longer than the sequence, batch-varying tree masks, wide
trees (several node blocks), W4 experts with asymmetric zeros, dropped MoE assignments, large SR offsets, bf16/fp16 inputs.
"""
from __future__ import annotations

import os

os.environ["TRITON_INTERPRET"] = "1"

import importlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton")

from forge.kernels.ref.attn_decode import paged_decode_attention_ref  # noqa: E402
from forge.kernels.ref.attn_kv import build_paged_pools  # noqa: E402
from forge.kernels.ref.attn_spec import build_tree_mask, rejection_verify_ref, tree_attention_ref  # noqa: E402
from forge.kernels.ref.moe_ref import (  # noqa: E402
    apply_expert_map_ref,
    moe_grouped_gemm_ref,
    moe_permute_ref,
    moe_route_ref,
    moe_unpermute_combine_ref,
    pack_w4_experts,
)
from forge.kernels.ref.util_hadamard import hadamard_ref, random_signs  # noqa: E402
from forge.kernels.ref.util_quant import quantize_rowwise_ref, rmsnorm_quant_ref, stochastic_round_ref  # noqa: E402
from forge.kernels.ref.util_rope import make_cos_sin_cache, rope_and_kv_write_ref  # noqa: E402


@pytest.fixture(scope="module")
def tk():
    names = ["attn_decode", "attn_spec", "moe_kernels", "util_hadamard", "util_quant", "util_rope"]
    return SimpleNamespace(**{n: importlib.import_module("forge.kernels.triton." + n) for n in names})


def _setup(b, hq, hkv, d, lens, ps, kvd, seed=0, asym=False):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hq, d, generator=g)
    k = torch.randn(b, hkv, max(lens), d, generator=g)
    v = torch.randn(b, hkv, max(lens), d, generator=g)
    return q, k, v, build_paged_pools(k, v, lens, ps, kvd, asymmetric=asym, generator=g)


def _kw(p):
    return dict(kv_dtype=p["kv_dtype"], k_scale=p["k_scale"], v_scale=p["v_scale"], k_zero=p["k_zero"], v_zero=p["v_zero"])


@pytest.mark.parametrize("d,hq,hkv,ps", [(24, 6, 2, 1), (80, 15, 5, 8), (48, 3, 1, 64), (16, 2, 2, 5)])
@pytest.mark.parametrize("kvd", ["fp32", "int8", "int4"])
def test_decode_odd_geometry(tk, d, hq, hkv, ps, kvd):
    lens = [70, 9, 1]
    q, k, v, p = _setup(3, hq, hkv, d, lens, ps, kvd, seed=d)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = paged_decode_attention_ref(*args, **_kw(p))
    for s in (1, 5):
        out = tk.attn_decode.paged_decode_attention_triton(*args, **_kw(p), num_splits=s)
        torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("window", [None, 3, 1000])
def test_decode_residual_only_and_long_window(tk, window):
    # the whole sequence lives in the residual window (empty pool part) for seq 1; window larger than the sequence
    lens = [40, 6]
    res = torch.tensor([10, 6], dtype=torch.int32)
    q, k, v, p = _setup(2, 4, 2, 32, lens, 16, "int8", seed=8)
    kr = torch.zeros(2, 2, 16, 32)
    vr = torch.zeros(2, 2, 16, 32)
    kr[0, :, :10] = k[0, :, 30:40]
    vr[0, :, :10] = v[0, :, 30:40]
    kr[1, :, :6] = k[1, :, :6]
    vr[1, :, :6] = v[1, :, :6]
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    kw = dict(_kw(p), k_res=kr, v_res=vr, res_lens=res, window=window)
    ref, lref = paged_decode_attention_ref(*args, **kw, return_lse=True)
    for s in (1, 3):
        out, lse = tk.attn_decode.paged_decode_attention_triton(*args, **kw, return_lse=True, num_splits=s)
        torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(lse, lref, atol=3e-5, rtol=3e-5)


def test_decode_fp8_pool_uint8_view_and_scalar_scale(tk):
    lens = [33, 20]
    q, k, v, p = _setup(2, 4, 2, 32, lens, 16, "fp8_e4m3", seed=2)
    kp8, vp8 = p["k_pool"], p["v_pool"]
    args = (q, kp8, vp8, p["block_tables"], p["seq_lens"])
    ref = paged_decode_attention_ref(*args, **_kw(p))
    out = tk.attn_decode.paged_decode_attention_triton(q, kp8.view(torch.uint8), vp8.view(torch.uint8), p["block_tables"],
                                                       p["seq_lens"], **_kw(p))
    torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)


def test_tree_batch_varying_mask_wide_tree_int4(tk):
    g = torch.Generator().manual_seed(21)
    hq, hkv, d, ps, b, n = 8, 2, 32, 8, 3, 21  # 21 nodes x group 4 -> several node blocks
    ctx = [30, 1, 17]
    kc = torch.randn(b, hkv, max(ctx), d, generator=g)
    vc = torch.randn(b, hkv, max(ctx), d, generator=g)
    p = build_paged_pools(kc, vc, ctx, ps, "int4", generator=g)
    q = torch.randn(b, n, hq, d, generator=g)
    kt = torch.randn(b, n, hkv, d, generator=g)
    vt = torch.randn(b, n, hkv, d, generator=g)
    masks = []
    for bi in range(b):
        par = torch.tensor([-1] + [int(torch.randint(-1, i, (1,), generator=g)) for i in range(1, n)], dtype=torch.int32)
        masks.append(build_tree_mask(par))
    mask = torch.stack(masks)
    cl = torch.tensor(ctx, dtype=torch.int32)
    for window in (None, 9):
        ref = tree_attention_ref(q, kt, vt, p["k_pool"], p["v_pool"], p["block_tables"], cl, mask, kv_dtype="int4",
                                 k_scale=p["k_scale"], v_scale=p["v_scale"], window=window)
        out = tk.attn_spec.tree_attention_triton(q, kt, vt, p["k_pool"], p["v_pool"], p["block_tables"], cl, mask,
                                                 kv_dtype="int4", k_scale=p["k_scale"], v_scale=p["v_scale"],
                                                 window=window)
        torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("g_,v", [(1, 5), (5, 130), (2, 1)])
def test_rejection_verify_shapes_sparse_support_and_bf16(tk, g_, v):
    torch.manual_seed(g_ * 31 + v)
    b = 40
    p = torch.rand(b, g_ + 1, v)
    p[torch.rand(b, g_ + 1, v) < 0.5] = 0.0  # sparse support: zero-probability tokens
    p[..., 0] += 0.1
    p = p / p.sum(-1, keepdim=True)
    q = torch.rand(b, g_, v)
    q[torch.rand(b, g_, v) < 0.5] = 0.0
    q[..., -1] += 0.1
    q = q / q.sum(-1, keepdim=True)
    draft = torch.multinomial(q.reshape(-1, v), 1).reshape(b, g_)
    for dt in (torch.float32, torch.bfloat16):
        pp, qq = p.to(dt), q.to(dt)
        rt, rn = rejection_verify_ref(draft, qq, pp, 77, 3)
        tt, tn = tk.attn_spec.rejection_verify_triton(draft, qq, pp, 77, 3, block_v=16)
        assert torch.equal(rn, tn) and torch.equal(rt, tt)
        # an emitted token must have positive target probability
        for bi in range(b):
            n = int(rn[bi])
            tok = int(rt[bi, n])
            assert 0 <= tok < v and float(pp[bi, n, tok]) > 0.0


def test_grouped_gemm_w4_zero_points_dropped_rows_and_scatter(tk):
    torch.manual_seed(5)
    e, k, n, t, topk = 6, 64, 20, 25, 3
    logits = torch.randn(t, e)
    w, ids = moe_route_ref(logits, topk)
    emap = torch.tensor([0, -1, 1, 2, -1, 3])  # prune two experts -> dropped assignments
    ids2, w2 = apply_expert_map_ref(ids, w, emap, renormalize=True)
    pr = moe_permute_ref(ids2, 4)
    x = torch.randn(t, k)
    wt = torch.randn(4, n, k) * 0.3
    for sym in (True, False):
        qw, sc, zr = pack_w4_experts(wt, 32, sym=sym)
        ref = moe_grouped_gemm_ref(x, qw, pr.expert_offsets, gather_idx=pr.token_idx, w_scales=sc, w_zeros=zr,
                                   group_size=32)
        out = tk.moe_kernels.moe_grouped_gemm_triton(x, qw, pr.expert_offsets, gather_idx=pr.token_idx, w_scales=sc,
                                                     w_zeros=zr, group_size=32, total_rows=t * topk)
        torch.testing.assert_close(out, ref, atol=2e-4, rtol=2e-4)
    # scatter + accumulate + alpha
    xs = x[pr.token_idx.long()]
    alpha = torch.rand(4)
    base = torch.randn(t * topk, n)
    o1 = base.clone()
    o2 = base.clone()
    ref = moe_grouped_gemm_ref(xs, wt, pr.expert_offsets, out_idx=pr.slot_idx, out=o1, accumulate=True, alpha=alpha)
    out = tk.moe_kernels.moe_grouped_gemm_triton(xs, wt, pr.expert_offsets, out_idx=pr.slot_idx, out=o2, accumulate=True,
                                                 alpha=alpha, total_rows=t * topk)
    torch.testing.assert_close(out, ref, atol=2e-4, rtol=2e-4)
    # combine with dropped assignments
    y = torch.randn(t * topk, n)
    ref = moe_unpermute_combine_ref(y, pr.inv_perm, w2)
    outc = tk.moe_kernels.moe_combine_triton(y, pr.inv_perm, w2)
    torch.testing.assert_close(outc, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("m_rows", [1, 15, 16, 17, 33])
def test_grouped_gemm_single_expert_tile_boundaries(tk, m_rows):
    torch.manual_seed(m_rows)
    e, k, n = 3, 24, 40
    offs = torch.tensor([0, 0, m_rows, m_rows], dtype=torch.int32)  # only the middle expert is non-empty
    x = torch.randn(m_rows, k)
    w = torch.randn(e, n, k)
    ref = moe_grouped_gemm_ref(x, w, offs)
    for bm in (16, 32):
        out = tk.moe_kernels.moe_grouped_gemm_triton(x, w, offs, total_rows=m_rows, block_m=bm)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("block,d,dt", [(16, 48, torch.float32), (128, 256, torch.float32), (32, 64, torch.float16)])
def test_hadamard_signs_3d_and_dtypes(tk, block, d, dt):
    torch.manual_seed(block)
    x = torch.randn(2, 3, d).to(dt)
    s = random_signs(d, 4)
    ref = hadamard_ref(x, block, s)
    out = tk.util_hadamard.hadamard_triton(x, block, s)
    tol = 2e-5 if dt == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)


def test_hadamard_rejects_wrong_signs_length(tk):
    x = torch.randn(4, 64)
    with pytest.raises(ValueError):
        tk.util_hadamard.hadamard_triton(x, 16, random_signs(16, 0))


def test_sr_large_offset_scalar_scale_and_reproducible(tk):
    torch.manual_seed(0)
    x = torch.randn(7, 33)
    for fmt in ("int8", "int4", "fp4"):
        scale = torch.tensor(0.37)
        ref, _ = stochastic_round_ref(x, fmt, group_size=None, scale=scale.expand(7, 1), seed=99, offset=(1 << 33) + 5)
        out = tk.util_quant.stochastic_round_triton(x, scale.expand(7, 1).contiguous(), 33, fmt, 99, (1 << 33) + 5)
        assert torch.equal(ref, out), fmt


@pytest.mark.parametrize("k,qd,gs", [(100, "int8", None), (96, "int8", 32), (7, "int8", None)])
def test_quant_and_rmsnorm_non_pow2(tk, k, qd, gs):
    torch.manual_seed(k)
    x = torch.randn(5, k) * 3
    x[2] = 0
    q, s = tk.util_quant.quantize_rowwise_triton(x, qd, gs)
    qr, sr = quantize_rowwise_ref(x, qd, gs)
    assert torch.equal(q, qr)
    torch.testing.assert_close(s.reshape(sr.shape), sr)
    w = torch.rand(k) + 0.5
    r = torch.randn(5, k)
    q2, s2, res = tk.util_quant.rmsnorm_quant_triton(x, w, 1e-6, "int8", r)
    q2r, s2r, resr = rmsnorm_quant_ref(x, w, 1e-6, "int8", r)
    assert (q2.int() - q2r.int()).abs().max() <= 1  # fp32 summation order may flip a half-way case
    torch.testing.assert_close(s2, s2r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(res, resr)


@pytest.mark.parametrize("interleaved,rd", [(False, 16), (True, 16), (False, 8), (True, 8)])
@pytest.mark.parametrize("kvd", ["fp32", "int8"])
def test_rope_write_partial_rotary_negative_slots(tk, interleaved, rd, kvd):
    torch.manual_seed(1)
    t, hq, hkv, d, ps, pages = 6, 4, 2, 16, 4, 5
    q = torch.randn(t, hq, d)
    k = torch.randn(t, hkv, d)
    v = torch.randn(t, hkv, d)
    pos = torch.randint(0, 30, (t,))
    cache = make_cos_sin_cache(32, rd)
    slots = torch.tensor([3, -1, 7, 8, 19, -1])
    dt = torch.int8 if kvd == "int8" else torch.float32
    kp1 = torch.zeros(pages, hkv, ps, d, dtype=dt)
    vp1 = torch.zeros_like(kp1)
    kp2, vp2 = kp1.clone(), vp1.clone()
    vs1 = torch.zeros(pages, hkv, ps)
    vs2 = vs1.clone()
    ks = torch.rand(hkv, d) + 0.5
    kw = dict(kv_dtype=kvd, k_scale=ks, rotary_dim=rd, interleaved=interleaved)
    qr = rope_and_kv_write_ref(q, k, v, pos, cache, slots, kp1, vp1, v_scale_pool=vs1, **kw)
    qt = tk.util_rope.rope_and_kv_write_triton(q, k, v, pos, cache, slots, kp2, vp2, v_scale_pool=vs2, **kw)
    torch.testing.assert_close(qt, qr, atol=1e-5, rtol=1e-5)
    if kvd == "int8":
        assert (kp1.int() - kp2.int()).abs().max() <= 1 and (vp1.int() - vp2.int()).abs().max() <= 1
    else:
        torch.testing.assert_close(kp2, kp1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(vp2, vp1, atol=1e-6, rtol=1e-6)


def test_tree_attention_with_residual_window_matches_reference(tk):
    g = torch.Generator().manual_seed(33)
    hq, hkv, d, ps, b, n = 4, 2, 32, 8, 2, 6
    ctx = [29, 12]
    res = [7, 12]  # second sequence lives entirely in the residual window (empty pool part)
    kc = torch.randn(b, hkv, max(ctx), d, generator=g)
    vc = torch.randn(b, hkv, max(ctx), d, generator=g)
    pool_lens = [c - r for c, r in zip(ctx, res)]
    p = build_paged_pools(kc[:, :, : max(pool_lens)], vc[:, :, : max(pool_lens)], pool_lens, ps, "int8", generator=g)
    kr = torch.zeros(b, hkv, 16, d)
    vr = torch.zeros(b, hkv, 16, d)
    for bi in range(b):
        kr[bi, :, : res[bi]] = kc[bi, :, pool_lens[bi]: ctx[bi]]
        vr[bi, :, : res[bi]] = vc[bi, :, pool_lens[bi]: ctx[bi]]
    parents = torch.tensor([-1, 0, 0, 1, 2, -1], dtype=torch.int32)
    mask = build_tree_mask(parents)
    q = torch.randn(b, n, hq, d, generator=g)
    kt = torch.randn(b, n, hkv, d, generator=g)
    vt = torch.randn(b, n, hkv, d, generator=g)
    cl = torch.tensor(ctx, dtype=torch.int32)
    rl = torch.tensor(res, dtype=torch.int32)
    for window in (None, 10):
        kw = dict(kv_dtype="int8", k_scale=p["k_scale"], v_scale=p["v_scale"], window=window, k_res=kr, v_res=vr,
                  res_lens=rl)
        ref = tree_attention_ref(q, kt, vt, p["k_pool"], p["v_pool"], p["block_tables"], cl, mask, **kw)
        out = tk.attn_spec.tree_attention_triton(q, kt, vt, p["k_pool"], p["v_pool"], p["block_tables"], cl, mask,
                                                 num_splits=2, **kw)
        torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)


def test_rejection_verify_device_offset_tensor_equals_int_offset(tk):
    torch.manual_seed(4)
    b, g_, v = 16, 3, 21
    p = torch.rand(b, g_ + 1, v) + 0.05
    p = p / p.sum(-1, keepdim=True)
    q = torch.rand(b, g_, v) + 0.05
    q = q / q.sum(-1, keepdim=True)
    draft = torch.multinomial(q.reshape(-1, v), 1).reshape(b, g_)
    for off in (0, 7, (1 << 33) + 3):
        a = tk.attn_spec.rejection_verify_triton(draft, q, p, 9, off, block_v=16)
        c = tk.attn_spec.rejection_verify_triton(draft, q, p, 9, torch.tensor(off, dtype=torch.int64), block_v=16)
        assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])
        r = rejection_verify_ref(draft, q, p, 9, off)
        assert torch.equal(r[0], c[0]) and torch.equal(r[1], c[1])
    # bumping the device counter in place changes the stream without re-launching with a new python constant
    cnt = torch.zeros((), dtype=torch.int64)
    first = tk.attn_spec.rejection_verify_triton(draft, q, p, 9, cnt, block_v=16)
    cnt += 1000
    second = tk.attn_spec.rejection_verify_triton(draft, q, p, 9, cnt, block_v=16)
    ref2 = rejection_verify_ref(draft, q, p, 9, 1000)
    assert torch.equal(second[0], ref2[0]) and not torch.equal(first[0], second[0])


def test_ops_stochastic_round_scalar_scale_and_kv_compact_guard():
    from forge.kernels import ops_util

    x = torch.randn(5, 16)
    for fmt in ("int8", "fp4"):
        a, _ = ops_util.stochastic_round(x, fmt, scale=torch.tensor(0.4), seed=3, offset=11)
        b_, _ = stochastic_round_ref(x, fmt, scale=torch.full((5, 1), 0.4), seed=3, offset=11)
        assert torch.equal(a, b_)
    kp = torch.zeros(2, 1, 4, 8, dtype=torch.int8)
    with pytest.raises(ValueError):
        ops_util.kv_compact(kp, kp.clone(), torch.zeros(1, 2, dtype=torch.int32), torch.tensor([1]),
                            torch.zeros(1, 2, dtype=torch.int32), torch.tensor([1]))


def test_kv_compact_long_overlapping_paths_odd_head_dim(tk):
    torch.manual_seed(3)
    ps, h, d, b, a = 4, 3, 24, 3, 40
    ctx = torch.tensor([5, 0, 11], dtype=torch.int32)
    bt = torch.arange(b * 20, dtype=torch.int32).view(b, 20)
    kp = torch.randn(b * 20, h, ps, d)
    vp = torch.randn(b * 20, h, ps, d)
    acc = torch.full((b, a), -1, dtype=torch.int32)
    alen = torch.tensor([37, 1, 20], dtype=torch.int32)
    for bi in range(b):
        n = int(alen[bi])
        acc[bi, :n] = torch.arange(n, dtype=torch.int32) + torch.randint(0, 2, (n,)).cumsum(0).int()
    from forge.kernels.ref.attn_spec import kv_compact_ref

    k1, v1, k2, v2 = kp.clone(), vp.clone(), kp.clone(), vp.clone()
    kv_compact_ref(k1, v1, bt, ctx, acc, alen)
    tk.attn_spec.kv_compact_triton(k2, v2, bt, ctx, acc, alen)
    assert torch.equal(k1, k2) and torch.equal(v1, v2)
