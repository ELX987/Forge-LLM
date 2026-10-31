"""Triton kernels for learned sparse attention (Plan K5, KR-5): lightning-indexer scan and top-k selection.

**Status: implemented, not yet validated on NVIDIA hardware** (both run in the Triton CPU interpreter in the unit tests).
Min Triton 2.1 (``tl.cumsum``); sm_70+ (fp16 index queries) / sm_80+ (bf16).  The *gather attention* of K5 is the
``topk_idx`` mode of :mod:`forge.kernels.triton.attn_decode` / :mod:`forge.kernels.triton.attn_mla`.

Indexer scan
------------
``score[b, t] = sum_j w[b, j] * relu(iq[b, j, :] . ik[t, :]) * ik_scale[t]`` over a paged index-key cache
``ik_pool [P, page_size, dI]`` (FP8-E4M3 raw bytes / bf16 / fp16) that uses the same page table as the main KV cache (a separate,
narrower page - Plan K5).  Bandwidth-bound: each key tile is read exactly once and used for all ``HI`` indexer heads
through one ``tl.dot``.  Positions ``>= seq_len`` are left untouched (the dispatcher pre-fills ``-inf``).

Top-k selection
---------------
Exact, deterministic, one program per row: a 32-step bisection over the monotone int32 image of the fp32 scores finds the
k-th largest key ``thr`` (``count(key >= thr) >= k`` maximal), then a single compaction pass emits every position with
``key > thr`` plus the first ``k - count(key > thr)`` positions with ``key == thr`` - i.e. ties go to the lower position and
the output is sorted ascending, identical to :func:`forge.kernels.ref.attn_sparse.topk_select_ref`.  Cost: ``32`` passes over L2-resident scores per row;
fine for ``T <= ~1M`` and small batch, ``torch.topk`` stays the default for large batch (see ``ops_util.topk_select``).
"""
from __future__ import annotations

import torch

from forge.kernels.ref.attn_kv import normalize_kv_dtype
from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import cdiv, fp8_native, next_pow2

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from forge.kernels.triton.attn_common import decode_e4m3

    @triton.jit
    def _indexer_scan_kernel(
        iq_ptr, iw_ptr, ik_ptr, iks_ptr, bt_ptr, seqlen_ptr, out_ptr,
        siq_b, siq_h, siw_b, sik_pg, sik_t, siks_pg, siks_t, sbt_b, so_b,
        HI: tl.constexpr, DI: tl.constexpr, BLOCK_HI: tl.constexpr, BLOCK_DI: tl.constexpr, BLOCK_T: tl.constexpr,
        PAGE_SIZE: tl.constexpr, FP8: tl.constexpr, FP8_NATIVE: tl.constexpr, HAS_SCALE: tl.constexpr,
    ):
        b = tl.program_id(0)
        pid_t = tl.program_id(1)
        seq_len = tl.load(seqlen_ptr + b)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        tv = offs_t < seq_len
        offs_h = tl.arange(0, BLOCK_HI)
        hm = offs_h < HI
        offs_d = tl.arange(0, BLOCK_DI)
        dm = offs_d < DI
        iq = tl.load(iq_ptr + b * siq_b + offs_h[:, None] * siq_h + offs_d[None, :],
                     mask=hm[:, None] & dm[None, :], other=0.0)
        iw = tl.load(iw_ptr + b * siw_b + offs_h, mask=hm, other=0.0).to(tl.float32)
        page = tl.load(bt_ptr + b * sbt_b + offs_t // PAGE_SIZE, mask=tv, other=0)
        slot = offs_t % PAGE_SIZE
        row_off = page.to(tl.int64) * sik_pg + slot.to(tl.int64) * sik_t
        raw = tl.load(ik_ptr + row_off[:, None] + offs_d[None, :], mask=tv[:, None] & dm[None, :], other=0)
        if FP8:
            if FP8_NATIVE:
                key = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            else:
                key = decode_e4m3(raw)
        else:
            key = raw.to(tl.float32)
        s = tl.dot(key.to(iq.dtype), tl.trans(iq))  # [BLOCK_T, BLOCK_HI]
        if HAS_SCALE:
            sc = tl.load(iks_ptr + page.to(tl.int64) * siks_pg + slot.to(tl.int64) * siks_t, mask=tv, other=1.0)
            s = s * sc[:, None]
        s = tl.maximum(s, 0.0) * iw[None, :]
        score = tl.sum(s, axis=1)
        tl.store(out_ptr + b * so_b + offs_t, score, mask=tv)

    @triton.jit
    def _topk_kernel(
        scores_ptr, lens_ptr, out_ptr, s_b, o_b, T,
        K: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """One program per row: exact top-K by bisection on the int32 image of the scores + ordered compaction."""
        b = tl.program_id(0)
        n = tl.minimum(tl.load(lens_ptr + b), T)
        row = scores_ptr + b * s_b
        # ---- bisection for thr = max{t : count(valid & key >= t) >= K}  (lo is a feasible lower bound) -------
        lo = tl.full([1], -2147483648, tl.int64)
        hi = tl.full([1], 2147483647, tl.int64)
        for _ in range(32):
            mid = lo + (hi - lo + 1) // 2
            cnt = tl.zeros([1], tl.int32)
            for st in range(0, n, BLOCK):
                offs = st + tl.arange(0, BLOCK)
                m = offs < n
                x = tl.load(row + offs, mask=m, other=0.0)
                x = tl.where(x != x, float("-inf"), tl.where(x == 0.0, 0.0, x))
                bits = x.to(tl.int32, bitcast=True)
                key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
                ge = (key.to(tl.int64) >= mid) & m
                cnt += tl.sum(ge.to(tl.int32), axis=0)
            ok = cnt >= K
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid - 1)
        thr = lo
        # ---- count keys strictly greater than thr -----------------------------------------------------------
        gt_cnt = tl.zeros([1], tl.int32)
        for st in range(0, n, BLOCK):
            offs = st + tl.arange(0, BLOCK)
            m = offs < n
            x = tl.load(row + offs, mask=m, other=0.0)
            x = tl.where(x != x, float("-inf"), tl.where(x == 0.0, 0.0, x))
            bits = x.to(tl.int32, bitcast=True)
            key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
            gt = (key.to(tl.int64) > thr) & m
            gt_cnt += tl.sum(gt.to(tl.int32), axis=0)
        need_eq = K - gt_cnt  # >= 1 whenever n >= K, otherwise irrelevant
        # ---- ordered compaction ---------------------------------------------------------------------------
        run_sel = tl.zeros([1], tl.int32)
        run_eq = tl.zeros([1], tl.int32)
        for st in range(0, n, BLOCK):
            offs = st + tl.arange(0, BLOCK)
            m = offs < n
            x = tl.load(row + offs, mask=m, other=0.0)
            x = tl.where(x != x, float("-inf"), tl.where(x == 0.0, 0.0, x))
            bits = x.to(tl.int32, bitcast=True)
            key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
            k64 = key.to(tl.int64)
            gt = (k64 > thr) & m
            eq = (k64 == thr) & m
            eq_i = eq.to(tl.int32)
            eq_rank = run_eq + tl.cumsum(eq_i, axis=0)  # 1-based rank among ties
            take = gt | (eq & (eq_rank <= need_eq))
            take_i = take.to(tl.int32)
            outpos = run_sel + tl.cumsum(take_i, axis=0) - 1
            tl.store(out_ptr + b * o_b + outpos, offs, mask=take & (outpos < K))
            run_sel += tl.sum(take_i, axis=0)
            run_eq += tl.sum(eq_i, axis=0)


def indexer_scores_triton(iq, iw, ik_pool, ik_scale, block_tables, seq_lens, *, block_t: int = 64):
    """``iq [B, HI, dI]``, ``iw [B, HI]``, ``ik_pool [P, ps, dI]`` -> ``scores [B, max_pages*ps]`` float32 (``-inf`` beyond lens)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    b, hi, di = iq.shape
    ps = ik_pool.shape[1]
    dev = iq.device
    iq = iq.contiguous()
    iw = iw.contiguous()
    bt = block_tables.to(torch.int32).contiguous()
    sl = seq_lens.to(torch.int32).contiguous()
    fp8 = ik_pool.dtype in (torch.uint8, getattr(torch, "float8_e4m3fn", None))
    view = ik_pool.view(torch.uint8) if (fp8 and ik_pool.dtype != torch.uint8) else ik_pool
    tmax = bt.shape[1] * ps
    out = torch.full((b, tmax), float("-inf"), dtype=torch.float32, device=dev)
    has_scale = ik_scale is not None
    iks = ik_scale.to(torch.float32) if has_scale else sl
    _indexer_scan_kernel[(b, cdiv(tmax, block_t))](
        iq, iw, view, iks, bt, sl, out,
        iq.stride(0), iq.stride(1), iw.stride(0), view.stride(0), view.stride(1),
        iks.stride(0) if has_scale else 0, iks.stride(1) if has_scale else 0, bt.stride(0), out.stride(0),
        HI=hi, DI=di, BLOCK_HI=max(16, next_pow2(hi)), BLOCK_DI=max(16, next_pow2(di)), BLOCK_T=block_t, PAGE_SIZE=ps,
        FP8=bool(fp8), FP8_NATIVE=bool(fp8 and fp8_native(dev)), HAS_SCALE=has_scale, num_warps=4, num_stages=2)
    return out


def topk_select_triton(scores, k: int, valid_lens=None, *, block: int = 1024):
    """Exact deterministic top-k (see module docstring). ``scores [B, T]`` float32 -> ``idx [B, k]`` int32 (``-1`` padded)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    b, t = scores.shape
    s = scores.to(torch.float32).contiguous()
    lens = (valid_lens.to(torch.int32).contiguous() if valid_lens is not None
            else torch.full((b,), t, dtype=torch.int32, device=scores.device))
    out = torch.full((b, k), -1, dtype=torch.int32, device=scores.device)
    _topk_kernel[(b,)](s, lens, out, s.stride(0), out.stride(0), t, K=k, BLOCK=block, BLOCK_K=max(16, next_pow2(k)),
                       num_warps=8)
    return out
