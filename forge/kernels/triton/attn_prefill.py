"""Triton causal GQA flash-attention for prefill / chunked prefill (Plan K4 prefill side).

**Status: implemented, not yet validated on NVIDIA hardware** (runs in the Triton CPU interpreter in the unit tests).
Min Triton 2.1; sm_70+ for fp16, sm_80+ for bf16.

Variable-length, packed layout: ``q [Tq, Hq, D]``, ``k, v [Tk, Hkv, D]`` and ``cu_seqlens_q / cu_seqlens_k [B+1]`` (int32).
Batched ``[B, S, H, D]`` inputs are the special case ``cu = arange(B) * S`` (done by the dispatcher through a free reshape).
Causal masking is bottom-right aligned (query ``i`` of a sequence with ``q_len`` queries and ``k_len`` keys sits at
position ``k_len - q_len + i``), which makes prefix-cache hits and chunked prefill (``k_len > q_len``) work unchanged.
Optional sliding window.  Grid ``(cdiv(max_seqlen_q, BLOCK_M), B, Hq)``; each program streams key blocks with online softmax
and skips blocks beyond the causal frontier / before the window.
"""
from __future__ import annotations

import math

import torch

from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import cdiv, next_pow2, smem_limit_bytes

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from forge.kernels.triton.attn_common import online_softmax_update

    @triton.jit
    def _prefill_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, cuq_ptr, cuk_ptr,
        sm_scale,
        sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, so_t, so_h,
        G: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr, WINDOW: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        b = tl.program_id(1)
        h = tl.program_id(2)
        hkv = h // G
        q_start = tl.load(cuq_ptr + b)
        q_len = tl.load(cuq_ptr + b + 1) - q_start
        k_start = tl.load(cuk_ptr + b)
        k_len = tl.load(cuk_ptr + b + 1) - k_start
        shift = k_len - q_len
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_valid = offs_m < q_len
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q = tl.load(q_ptr + (q_start + offs_m)[:, None].to(tl.int64) * sq_t + h * sq_h + offs_d[None, :],
                    mask=m_valid[:, None] & d_mask[None, :], other=0.0)
        hi = k_len
        lo = k_len * 0
        if CAUSAL:
            hi = tl.minimum(k_len, (pid_m + 1) * BLOCK_M + shift)
        if WINDOW > 0:
            lo = tl.maximum(pid_m * BLOCK_M + shift - WINDOW + 1, 0)
            lo = (lo // BLOCK_N) * BLOCK_N
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        qk_scale = sm_scale * 1.4426950408889634
        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = offs_n < k_len
            k = tl.load(k_ptr + (k_start + offs_n)[:, None].to(tl.int64) * sk_t + hkv * sk_h + offs_d[None, :],
                        mask=n_valid[:, None] & d_mask[None, :], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            valid = n_valid[None, :] & m_valid[:, None]
            if CAUSAL:
                valid = valid & (offs_n[None, :] <= (offs_m[:, None] + shift))
            if WINDOW > 0:
                valid = valid & (offs_n[None, :] > (offs_m[:, None] + shift - WINDOW))
            v = tl.load(v_ptr + (k_start + offs_n)[:, None].to(tl.int64) * sv_t + hkv * sv_h + offs_d[None, :],
                        mask=n_valid[:, None] & d_mask[None, :], other=0.0)
            acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, v, valid)
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        o = acc / l_safe[:, None]
        tl.store(o_ptr + (q_start + offs_m)[:, None].to(tl.int64) * so_t + h * so_h + offs_d[None, :],
                 o.to(o_ptr.dtype.element_ty), mask=m_valid[:, None] & d_mask[None, :])


def prefill_attention_varlen_triton(q, k, v, cu_seqlens_q, cu_seqlens_k, *, max_seqlen_q: int,
                                    softmax_scale: float | None = None, causal: bool = True,
                                    window: int | None = None):
    """``q [Tq, Hq, D]``, ``k, v [Tk, Hkv, D]`` (last dim contiguous) -> ``[Tq, Hq, D]`` in ``q.dtype``."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    tq, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("Hq must be a multiple of Hkv")
    q = q if q.stride(-1) == 1 else q.contiguous()
    k = k if k.stride(-1) == 1 else k.contiguous()
    v = v if v.stride(-1) == 1 else v.contiguous()
    cuq = cu_seqlens_q.to(torch.int32).contiguous()
    cuk = cu_seqlens_k.to(torch.int32).contiguous()
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    b = cuq.numel() - 1
    out = torch.empty_like(q)
    block_d = max(16, next_pow2(d))
    block_m, block_n = (64, 64) if block_d <= 128 else (32, 32)
    if block_d > 128 and smem_limit_bytes(q.device) < 80 * 1024:  # sm_75 class (64 KiB)
        block_n = 16
    num_warps = 4 if block_d <= 128 else 8
    _prefill_kernel[(cdiv(max_seqlen_q, block_m), b, hq)](
        q, k, v, out, cuq, cuk, float(scale),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        G=hq // hkv, D=d, BLOCK_D=block_d, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=bool(causal),
        WINDOW=int(window) if window else 0, num_warps=num_warps, num_stages=2)
    return out
