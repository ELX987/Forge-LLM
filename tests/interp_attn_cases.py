"""Triton attention kernels executed in the Triton **CPU interpreter** (no GPU involved) and compared with the oracles.

Not collected directly (no ``test_`` prefix): ``test_kernels_attn_triton_interp.py`` runs this file in a *subprocess* with
``TRITON_INTERPRET=1`` because the interpreter must be enabled before ``triton`` is first imported in the process.

This validates indexing / masking / online-softmax / split-merge logic of the kernels; it does NOT validate GPU compilation,
performance, bf16/fp8 hardware paths or race behaviour - those are covered by ``tests_gpu/`` on a real device.
Skipped automatically when ``triton`` is not importable.
"""
from __future__ import annotations

import os

os.environ["TRITON_INTERPRET"] = "1"  # must precede the first `import triton`

import importlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton")

from forge.kernels.ref.attn_decode import mla_decode_attention_ref, paged_decode_attention_ref  # noqa: E402
from forge.kernels.ref.attn_kv import build_paged_pools  # noqa: E402
from forge.kernels.ref.attn_prefill import prefill_attention_ref  # noqa: E402
from forge.kernels.ref.attn_sparse import indexer_scores_ref, topk_select_ref  # noqa: E402
from forge.kernels.ref.attn_spec import (  # noqa: E402
    build_tree_mask,
    kv_compact_ref,
    rejection_verify_ref,
    tree_attention_ref,
)


@pytest.fixture(scope="module")
def tk():
    names = ["attn_common", "attn_decode", "attn_mla", "attn_prefill", "attn_sparse", "attn_spec"]
    return SimpleNamespace(**{n: importlib.import_module("forge.kernels.triton." + n) for n in names})


def _setup(b, hq, hkv, d, lens, ps, kvd, seed=0, asym=False):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hq, d, generator=g)
    k = torch.randn(b, hkv, max(lens), d, generator=g)
    v = torch.randn(b, hkv, max(lens), d, generator=g)
    return q, k, v, build_paged_pools(k, v, lens, ps, kvd, asymmetric=asym, generator=g)


def _kw(p):
    return dict(kv_dtype=p["kv_dtype"], k_scale=p["k_scale"], v_scale=p["v_scale"], k_zero=p["k_zero"], v_zero=p["v_zero"])


@pytest.mark.parametrize("gsize", [1, 2, 4, 8, 16])
def test_decode_gqa_group_sizes_ragged(tk, gsize):
    lens = [1, 17, 33, 0]
    q, k, v, p = _setup(4, 2 * gsize, 2, 32, lens, 16, "fp32")
    ref = paged_decode_attention_ref(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], kv_dtype="fp32")
    out = tk.attn_decode.paged_decode_attention_triton(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"],
                                                       kv_dtype="fp32")
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("kvd,asym", [("int8", False), ("int8", True), ("int4", False), ("int4", True), ("fp8_e4m3", False)])
@pytest.mark.parametrize("splits", [1, 3])
def test_decode_quantised_kv(tk, kvd, asym, splits):
    lens = [40, 19]
    q, k, v, p = _setup(2, 4, 2, 32, lens, 16, kvd, asym=asym)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = paged_decode_attention_ref(*args, **_kw(p))
    out = tk.attn_decode.paged_decode_attention_triton(*args, num_splits=splits, **_kw(p))
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("window", [1, 5, 16, 33])
def test_decode_sliding_window_and_splits(tk, window):
    lens = [50, 20, 3]
    q, k, v, p = _setup(3, 4, 2, 16, lens, 16, "fp32", seed=2)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = paged_decode_attention_ref(*args, kv_dtype="fp32", window=window)
    for s in (1, 4):
        out = tk.attn_decode.paged_decode_attention_triton(*args, kv_dtype="fp32", window=window, num_splits=s)
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


def test_decode_return_lse_and_residual_window(tk):
    lens = [45, 30]
    res = torch.tensor([9, 0], dtype=torch.int32)
    q, k, v, p = _setup(2, 4, 2, 32, lens, 16, "int8", seed=4)
    kr = torch.zeros(2, 2, 16, 32)
    vr = torch.zeros(2, 2, 16, 32)
    kr[0, :, :9] = k[0, :, 36:45]
    vr[0, :, :9] = v[0, :, 36:45]
    # the pool holds only the first seq_len - res tokens (positions >= 36 of seq 0 live in the residual window)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    kw = dict(_kw(p), k_res=kr, v_res=vr, res_lens=res)
    ref, lse_ref = paged_decode_attention_ref(*args, **kw, return_lse=True)
    out, lse = tk.attn_decode.paged_decode_attention_triton(*args, **kw, return_lse=True, num_splits=2)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse, lse_ref, atol=2e-5, rtol=2e-5)
    # window that cuts into the residual part
    kw["window"] = 6
    torch.testing.assert_close(tk.attn_decode.paged_decode_attention_triton(*args, **kw),
                               paged_decode_attention_ref(*args, **kw), atol=2e-5, rtol=2e-5)


def test_decode_fp16_queries_interpreter(tk):
    lens = [33]
    q, k, v, p = _setup(1, 4, 2, 32, lens, 16, "fp32")
    q16 = q.half()
    args = (q16, p["k_pool"].half(), p["v_pool"].half(), p["block_tables"], p["seq_lens"])
    ref = paged_decode_attention_ref(*args, kv_dtype="fp16")
    out = tk.attn_decode.paged_decode_attention_triton(*args, kv_dtype="fp16")
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("per_head", [False, True])
def test_decode_sparse_gather(tk, per_head):
    lens = [60, 25]
    q, k, v, p = _setup(2, 4, 2, 32, lens, 16, "int8", seed=5)
    idx = torch.full((2, 2 if per_head else 1, 10), -1, dtype=torch.int32)
    torch.manual_seed(1)
    for bi, n in enumerate(lens):
        for h in range(idx.shape[1]):
            idx[bi, h, :8] = torch.randperm(n)[:8].int()
    idx[0, 0, 8] = 500  # out-of-range selection must be ignored
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = paged_decode_attention_ref(*args, **_kw(p), topk_idx=idx)
    for s in (1, 2):
        out = tk.attn_decode.paged_decode_attention_triton(*args, **_kw(p), topk_idx=idx, num_splits=s)
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


def test_mla_decode(tk):
    b, h, dc, dr, ps = 2, 6, 32, 16, 16
    lens = [37, 18]
    g = torch.Generator().manual_seed(7)
    c = torch.randn(b, max(lens), dc, generator=g)
    r = torch.randn(b, max(lens), dr, generator=g)
    qn = torch.randn(b, h, dc, generator=g)
    qr = torch.randn(b, h, dr, generator=g)
    total = 12
    c_pool = torch.zeros(total, ps, dc)
    r_pool = torch.zeros(total, ps, dr)
    bt = torch.zeros(b, 3, dtype=torch.int32)
    nxt = 3
    for bi in range(b):
        for pi in range((lens[bi] + ps - 1) // ps):
            pg = nxt * 2 % total
            nxt += 1
            bt[bi, pi] = pg
            lo, hi = pi * ps, min(lens[bi], (pi + 1) * ps)
            c_pool[pg, : hi - lo] = c[bi, lo:hi]
            r_pool[pg, : hi - lo] = r[bi, lo:hi]
    sl = torch.tensor(lens, dtype=torch.int32)
    w_uv = torch.randn(h, dc, 8, generator=g)
    args = (qn, qr, c_pool, r_pool, bt, sl)
    ref = mla_decode_attention_ref(*args, softmax_scale=0.17, w_uv=w_uv)
    out = tk.attn_mla.mla_decode_attention_triton(*args, softmax_scale=0.17, w_uv=w_uv, num_splits=2)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
    sel = torch.tensor([[0, 7, 30, -1], [2, 3, -1, -1]], dtype=torch.int32)
    refs = mla_decode_attention_ref(*args, softmax_scale=0.17, topk_idx=sel)
    outs = tk.attn_mla.mla_decode_attention_triton(*args, softmax_scale=0.17, topk_idx=sel, num_splits=2)
    torch.testing.assert_close(outs, refs, atol=2e-5, rtol=2e-5)
    refw = mla_decode_attention_ref(*args, softmax_scale=0.17, window=5)
    outw = tk.attn_mla.mla_decode_attention_triton(*args, softmax_scale=0.17, window=5)
    torch.testing.assert_close(outw, refw, atol=2e-5, rtol=2e-5)
    # fp8 latent cache with per-token scales (portable integer decode)
    amax = c_pool.abs().amax(-1).clamp_min(1e-6)
    cs = amax / 448.0
    c8 = (c_pool / cs.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref8 = mla_decode_attention_ref(qn, qr, c8, r_pool, bt, sl, softmax_scale=0.17, c_scale=cs, kv_dtype="fp8_e4m3")
    out8 = tk.attn_mla.mla_decode_attention_triton(qn, qr, c8, r_pool, bt, sl, softmax_scale=0.17, c_scale=cs,
                                                   kv_dtype="fp8_e4m3")
    torch.testing.assert_close(out8, ref8, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("hq,hkv", [(4, 2), (2, 2), (8, 1)])
def test_prefill_varlen_chunked_prefix_and_window(tk, hq, hkv):
    g = torch.Generator().manual_seed(3)
    d = 32
    q_lens, k_lens = [20, 5, 1], [20, 37, 9]  # 2nd/3rd: cached prefix (k_len > q_len)
    cuq = torch.tensor([0, 20, 25, 26], dtype=torch.int32)
    cuk = torch.tensor([0, 20, 57, 66], dtype=torch.int32)
    q = torch.randn(26, hq, d, generator=g)
    k = torch.randn(66, hkv, d, generator=g)
    v = torch.randn(66, hkv, d, generator=g)
    for causal, window in [(True, None), (True, 7), (False, None)]:
        ref = prefill_attention_ref(q, k, v, causal=causal, window=window, cu_seqlens_q=cuq, cu_seqlens_k=cuk)
        out = tk.attn_prefill.prefill_attention_varlen_triton(q, k, v, cuq, cuk, max_seqlen_q=20, causal=causal,
                                                              window=window)
        torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)


def test_indexer_scan_and_topk_kernels(tk):
    g = torch.Generator().manual_seed(11)
    b, hi, di, ps = 3, 4, 32, 16
    lens = [50, 16, 0]
    ik = torch.randn(b, 1, 64, di, generator=g)
    p = build_paged_pools(ik, ik, lens, ps, "fp32", generator=g)
    ik_pool = p["k_pool"][:, 0].contiguous()
    iq = torch.randn(b, hi, di, generator=g)
    iw = torch.rand(b, hi, generator=g)
    args = (iq, iw, ik_pool, None, p["block_tables"], p["seq_lens"])
    ref = indexer_scores_ref(*args)
    out = tk.attn_sparse.indexer_scores_triton(*args, block_t=32)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
    # fp8 index keys with per-token scale: portable decode path
    scale = ik_pool.abs().amax(-1).clamp_min(1e-6) / 448.0
    k8 = (ik_pool / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    args8 = (iq, iw, k8, scale, p["block_tables"], p["seq_lens"])
    torch.testing.assert_close(tk.attn_sparse.indexer_scores_triton(*args8), indexer_scores_ref(*args8), atol=2e-5,
                               rtol=2e-5)


def test_topk_kernel_exact_with_ties_invalid_and_short_rows(tk):
    torch.manual_seed(0)
    s = torch.randn(4, 300)
    s[0, ::7] = 1.5  # many ties at the cutoff region
    s[1, :150] = 0.25
    s[2, 10] = float("nan")
    s[2, 20] = float("-inf")
    s[3, 5] = -0.0
    s[3, 6] = 0.0
    lens = torch.tensor([300, 300, 280, 7], dtype=torch.int32)
    for k in (1, 16, 64):
        ref = topk_select_ref(s, k, lens)
        out = tk.attn_sparse.topk_select_triton(s, k, lens, block=128)
        assert torch.equal(out, ref), (k, out[3], ref[3])
    ref_all = topk_select_ref(s, 40)
    assert torch.equal(tk.attn_sparse.topk_select_triton(s, 40, block=64), ref_all)


def test_tree_attention_with_residual(tk):
    g = torch.Generator().manual_seed(9)
    hq, hkv, d, ps = 4, 2, 32, 16
    parents = torch.tensor([-1, 0, 0, 1, 2, 2, 4], dtype=torch.int32)
    n = parents.numel()
    ctx = [37, 20]
    b = 2
    kc = torch.randn(b, hkv, max(ctx), d, generator=g)
    vc = torch.randn(b, hkv, max(ctx), d, generator=g)
    p = build_paged_pools(kc, vc, ctx, ps, "int8", generator=g)
    q = torch.randn(b, n, hq, d, generator=g)
    kt = torch.randn(b, n, hkv, d, generator=g)
    vt = torch.randn(b, n, hkv, d, generator=g)
    mask = build_tree_mask(parents)
    cl = torch.tensor(ctx, dtype=torch.int32)
    for window in (None, 12):
        ref = tree_attention_ref(q, kt, vt, p["k_pool"], p["v_pool"], p["block_tables"], cl, mask, kv_dtype="int8",
                                 k_scale=p["k_scale"], v_scale=p["v_scale"], window=window)
        out = tk.attn_spec.tree_attention_triton(q, kt, vt, p["k_pool"], p["v_pool"], p["block_tables"], cl, mask,
                                                 kv_dtype="int8", k_scale=p["k_scale"], v_scale=p["v_scale"],
                                                 window=window, num_splits=2)
        torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)


def test_rejection_verify_kernel_matches_reference_bitwise(tk):
    torch.manual_seed(3)
    b, g, v = 64, 3, 37
    p = torch.rand(b, g + 1, v) + 0.05
    p = p / p.sum(-1, keepdim=True)
    q = torch.rand(b, g, v) + 0.05
    q = q / q.sum(-1, keepdim=True)
    draft = torch.multinomial(q.reshape(-1, v), 1).reshape(b, g)
    for seed, off in [(5, 0), (123456789012, 7)]:
        rt, rn = rejection_verify_ref(draft, q, p, seed, off)
        tt, tn = tk.attn_spec.rejection_verify_triton(draft, q, p, seed, off, block_v=16)
        assert torch.equal(rn, tn)
        assert torch.equal(rt, tt)
    # draft == target -> always accepts
    toks, nacc = tk.attn_spec.rejection_verify_triton(draft, p[:, :g], p, 1)
    assert (nacc == g).all()


def test_kv_compact_kernel_matches_reference(tk):
    torch.manual_seed(1)
    ps, h, d, b = 4, 2, 16, 2
    ctx = torch.tensor([5, 8], dtype=torch.int32)
    bt = torch.tensor([[0, 1, 2, 3, 8], [4, 5, 6, 7, 9]], dtype=torch.int32)
    kp = torch.randn(12, h, ps, d)
    vp = torch.randn(12, h, ps, d)
    acc = torch.tensor([[0, 2, 3, 5, -1, -1], [1, 2, -1, -1, -1, -1]], dtype=torch.int32)
    alen = torch.tensor([4, 2], dtype=torch.int32)
    k1, v1, k2, v2 = kp.clone(), vp.clone(), kp.clone(), vp.clone()
    kv_compact_ref(k1, v1, bt, ctx, acc, alen)
    tk.attn_spec.kv_compact_triton(k2, v2, bt, ctx, acc, alen)
    assert torch.equal(k1, k2) and torch.equal(v1, v2)
