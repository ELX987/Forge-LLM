"""Triton MoE kernels (Plan K7, KR-9): fused routing, grouped GEMM over variable per-expert M, weighted combine.

**Status: implemented, not yet validated on NVIDIA hardware** (all run in the Triton CPU interpreter in the unit tests).
Min Triton 2.1; sm_70+ (fp16) / sm_80+ (bf16).

Routing (:func:`moe_route_triton`) - one program per token: softmax (or sigmoid + selection bias), ``top_k`` rounds of
arg-max with lowest-index tie-break (identical to the stable-sort oracle), optional renormalisation, all in registers.

Grouped GEMM (:func:`moe_grouped_gemm_triton`) - ``y[r] = x[gather[r]] @ W[e(r)]^T`` for token slabs sorted by expert.
The tile schedule (``block_expert / block_row_start / block_rows``) is built **on device without a host sync** by
``ref.moe_ref.build_block_schedule_ref``; its length ``ceil(M / BLOCK_M) + E`` is a static upper bound, surplus tiles have
``block_rows = 0`` and every access is masked, so empty experts (M = 0), tiny experts (M = 1) and huge experts share one
launch.  Optional fusions: row gather of the activations (no permuted copy), output scatter + accumulate, per-expert output
scale (LoRA scaling), and a W4A16 weight path (``uint8`` packed nibbles + group scales/zeros: the *hook* for 4-bit experts).
Combine (:func:`moe_combine_triton`) - ``out[t] = sum_s w[t, s] * y[inv_perm[t * k + s]]`` in fp32.
"""
from __future__ import annotations

import torch

from forge.kernels.ref.moe_ref import build_block_schedule_ref, max_blocks_bound
from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import cdiv, next_pow2

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _route_kernel(
        logits_ptr, bias_ptr, w_ptr, ids_ptr, s_l, routed_scaling,
        E: tl.constexpr, BLOCK_E: tl.constexpr, TOP_K: tl.constexpr, BLOCK_K: tl.constexpr,
        SCORING: tl.constexpr, NORM: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        t = tl.program_id(0)
        offs = tl.arange(0, BLOCK_E)
        em = offs < E
        x = tl.load(logits_ptr + t * s_l + offs, mask=em, other=float("-inf")).to(tl.float32)
        if SCORING == 0:
            mx = tl.max(x, axis=0)
            p = tl.exp(x - mx)
            p = tl.where(em, p, 0.0)
            p = p / tl.sum(p, axis=0)
            sel = tl.where(em, p, float("-inf"))
        else:
            p = 1.0 / (1.0 + tl.exp(-x))
            p = tl.where(em, p, 0.0)
            sel = p
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offs, mask=em, other=0.0).to(tl.float32)
                sel = p + bias
            sel = tl.where(em, sel, float("-inf"))
        koffs = tl.arange(0, BLOCK_K)
        wvec = tl.zeros([BLOCK_K], tl.float32)
        ivec = tl.zeros([BLOCK_K], tl.int32)
        for i in tl.static_range(TOP_K):
            mx = tl.max(sel, axis=0)
            idx = tl.min(tl.where(sel == mx, offs, BLOCK_E), axis=0)
            w = tl.sum(tl.where(offs == idx, p, 0.0), axis=0)
            wvec = tl.where(koffs == i, w, wvec)
            ivec = tl.where(koffs == i, idx, ivec)
            sel = tl.where(offs == idx, float("-inf"), sel)
        km = koffs < TOP_K
        if NORM:
            tot = tl.sum(tl.where(km, wvec, 0.0), axis=0)
            wvec = wvec / tl.maximum(tot, 1e-20)
        wvec = wvec * routed_scaling
        tl.store(w_ptr + t * TOP_K + koffs, wvec, mask=km)
        tl.store(ids_ptr + t * TOP_K + koffs, ivec, mask=km)

    @triton.jit
    def _grouped_gemm_kernel(
        x_ptr, w_ptr, y_ptr, ws_ptr, wz_ptr, gather_ptr, outidx_ptr, alpha_ptr, be_ptr, brs_ptr, bnr_ptr,
        N, K,
        sx_m, sx_k, sw_e, sw_n, sw_k, sy_m, sy_n, sws_e, sws_n, sws_g, swz_e, swz_n, swz_g,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        W4: tl.constexpr, HAS_ZERO: tl.constexpr, GROUP_SIZE: tl.constexpr,
        GATHER: tl.constexpr, OUT_SCATTER: tl.constexpr, ACCUM: tl.constexpr, HAS_ALPHA: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(be_ptr + pid_b)
        row0 = tl.load(brs_ptr + pid_b)
        nrows = tl.load(bnr_ptr + pid_b)
        e64 = tl.maximum(e, 0).to(tl.int64)
        offs_m = tl.arange(0, BLOCK_M)
        rmask = offs_m < nrows
        rows = row0 + offs_m
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        if GATHER:
            src = tl.load(gather_ptr + rows, mask=rmask, other=0)
        else:
            src = rows
        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            kmask = offs_k < K
            a = tl.load(x_ptr + src[:, None].to(tl.int64) * sx_m + offs_k[None, :] * sx_k,
                        mask=rmask[:, None] & kmask[None, :], other=0.0)
            wmask = kmask[:, None] & nmask[None, :]
            if W4:
                byte = tl.load(w_ptr + e64 * sw_e + offs_n[None, :] * sw_n + (offs_k // 2)[:, None] * sw_k,
                               mask=wmask, other=0).to(tl.int32)
                nib = (byte >> ((offs_k % 2) * 4)[:, None]) & 15
                g = offs_k // GROUP_SIZE
                s = tl.load(ws_ptr + e64 * sws_e + offs_n[None, :] * sws_n + g[:, None] * sws_g,
                            mask=wmask, other=0.0).to(tl.float32)
                if HAS_ZERO:
                    z = tl.load(wz_ptr + e64 * swz_e + offs_n[None, :] * swz_n + g[:, None] * swz_g,
                                mask=wmask, other=0).to(tl.float32)
                else:
                    z = 8.0
                wt = ((nib.to(tl.float32) - z) * s).to(a.dtype)
            else:
                wt = tl.load(w_ptr + e64 * sw_e + offs_n[None, :] * sw_n + offs_k[:, None] * sw_k,
                             mask=wmask, other=0.0).to(a.dtype)
            acc = tl.dot(a, wt, acc)
        if HAS_ALPHA:
            acc = acc * tl.load(alpha_ptr + e64).to(tl.float32)
        if OUT_SCATTER:
            orow = tl.load(outidx_ptr + rows, mask=rmask, other=0)
        else:
            orow = rows
        y_ptrs = y_ptr + orow[:, None].to(tl.int64) * sy_m + offs_n[None, :] * sy_n
        omask = rmask[:, None] & nmask[None, :]
        if ACCUM:
            prev = tl.load(y_ptrs, mask=omask, other=0.0).to(tl.float32)
            acc = acc + prev
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=omask)

    @triton.jit
    def _combine_kernel(
        y_ptr, inv_ptr, w_ptr, out_ptr, H, sy_m, so_t,
        TOP_K: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        t = tl.program_id(0)
        hb = tl.program_id(1)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        hm = offs < H
        acc = tl.zeros([BLOCK_H], tl.float32)
        for s in tl.static_range(TOP_K):
            inv = tl.load(inv_ptr + t * TOP_K + s)
            w = tl.load(w_ptr + t * TOP_K + s).to(tl.float32)
            lim = tl.where(inv >= 0, H, 0)  # dropped assignments (inv < 0) contribute nothing
            row = tl.maximum(inv, 0).to(tl.int64)
            y = tl.load(y_ptr + row * sy_m + offs, mask=offs < lim, other=0.0).to(tl.float32)
            acc += y * w
        tl.store(out_ptr + t.to(tl.int64) * so_t + offs, acc.to(out_ptr.dtype.element_ty), mask=hm)


# ------------------------------------------------------------------------------------------------------------
def moe_route_triton(logits, top_k: int, *, scoring: str = "softmax", norm_topk_prob: bool = True,
                     routed_scaling: float = 1.0, score_bias=None):
    """Fused routing -> ``(weights [T, k] float32, ids [T, k] int32)``; semantics of :func:`ref.moe_ref.moe_route_ref`."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    t, e = logits.shape
    logits = logits.contiguous()
    w = torch.empty(t, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(t, top_k, dtype=torch.int32, device=logits.device)
    if t == 0:
        return w, ids
    bias = score_bias.contiguous() if score_bias is not None else logits
    _route_kernel[(t,)](
        logits, bias, w, ids, logits.stride(0), float(routed_scaling),
        E=e, BLOCK_E=max(16, next_pow2(e)), TOP_K=top_k, BLOCK_K=max(2, next_pow2(top_k)),
        SCORING=0 if scoring == "softmax" else 1, NORM=bool(norm_topk_prob), HAS_BIAS=score_bias is not None,
        num_warps=4)
    return w, ids


def pick_block_m(avg_rows: float) -> int:
    """Row-tile size from the average rows per expert (small-M experts use the skinny 16-row tile)."""
    if avg_rows <= 16:
        return 16
    if avg_rows <= 40:
        return 32
    if avg_rows <= 96:
        return 64
    return 128


def moe_grouped_gemm_triton(x, w, offsets, *, gather_idx=None, out_idx=None, out=None, accumulate: bool = False,
                            alpha=None, w_scales=None, w_zeros=None, group_size: int | None = None,
                            total_rows: int | None = None, num_rows_out: int | None = None, block_m: int | None = None,
                            block_n: int | None = None, block_k: int | None = None, schedule=None):
    """Grouped GEMM launcher (see module docstring).  ``x [Mx, K]``; ``w [E, N, K]`` (bf16/fp16) or packed W4
    (``uint8 [E, N, K/2]`` + ``w_scales [E, N, K/G]`` (+ ``w_zeros uint8``) + ``group_size``).  ``total_rows`` is the number
    of sorted rows ``M`` (static upper bound of ``offsets[-1]``; default ``gather_idx.numel()`` or ``x.shape[0]``)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    w4 = w.dtype == torch.uint8
    e = w.shape[0]
    n = w.shape[1]
    k = w.shape[2] * (2 if w4 else 1)
    m_rows = total_rows if total_rows is not None else (gather_idx.numel() if gather_idx is not None else x.shape[0])
    bm = block_m or pick_block_m(m_rows / max(1, e))
    bn = block_n or min(128, max(16, next_pow2(n)))
    bk = block_k or min(64, max(16, next_pow2(k)))
    if w4 and group_size is None:
        raise ValueError("W4 weights need group_size")
    mb = max_blocks_bound(m_rows, e, bm)
    be, brs, bnr = schedule if schedule is not None else build_block_schedule_ref(offsets.to(torch.int32), bm, mb)
    if out is None:
        rows_out = num_rows_out if num_rows_out is not None else m_rows
        out = torch.zeros(rows_out, n, dtype=x.dtype, device=x.device)
    dummy = offsets
    gi = gather_idx.to(torch.int32).contiguous() if gather_idx is not None else dummy
    oi = out_idx.to(torch.int32).contiguous() if out_idx is not None else dummy
    al = alpha.to(torch.float32).contiguous() if alpha is not None else dummy
    if w4:
        ws = w_scales
        wz = w_zeros if w_zeros is not None else w_scales
        s_ws = (ws.stride(0), ws.stride(1), ws.stride(2))
        s_wz = (wz.stride(0), wz.stride(1), wz.stride(2))
        s_w = (w.stride(0), w.stride(1), w.stride(2))
    else:
        ws = wz = w
        s_ws = s_wz = (0, 0, 0)
        s_w = (w.stride(0), w.stride(1), w.stride(2))
    if m_rows == 0:
        return out
    grid = (be.numel(), cdiv(n, bn))
    _grouped_gemm_kernel[grid](
        x, w, out, ws, wz, gi, oi, al, be, brs, bnr, n, k,
        x.stride(0), x.stride(1), s_w[0], s_w[1], s_w[2], out.stride(0), out.stride(1),
        s_ws[0], s_ws[1], s_ws[2], s_wz[0], s_wz[1], s_wz[2],
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, W4=w4, HAS_ZERO=bool(w4 and w_zeros is not None),
        GROUP_SIZE=int(group_size) if w4 else 1, GATHER=gather_idx is not None, OUT_SCATTER=out_idx is not None,
        ACCUM=bool(accumulate), HAS_ALPHA=alpha is not None, num_warps=4 if bm <= 32 else 8, num_stages=2)
    return out


def moe_combine_triton(y_sorted, inv_perm, topk_weights):
    """``out[t] = sum_s w[t, s] * y_sorted[inv_perm[t * k + s]]`` (``inv_perm < 0`` contributes 0)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    t, k = topk_weights.shape
    h = y_sorted.shape[1]
    y = y_sorted.contiguous()
    out = torch.empty(t, h, dtype=y.dtype, device=y.device)
    if t == 0:
        return out
    inv = inv_perm.to(torch.int32).contiguous()
    w = topk_weights.to(torch.float32).contiguous()
    bh = min(1024, max(16, next_pow2(h)))
    _combine_kernel[(t, cdiv(h, bh))](y, inv, w, out, h, y.stride(0), out.stride(0), TOP_K=k, BLOCK_H=bh, num_warps=4)
    return out
