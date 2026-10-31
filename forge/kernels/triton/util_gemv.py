"""Triton TEAL-style activation-sparsity gather-GEMV (Plan K8; P2 F5).

**Status: implemented, not yet validated on NVIDIA hardware** (runs in the Triton CPU interpreter in the unit tests).  Min Triton 2.1.

``y[b, :] = sum_{k : |x[b, k]| > thr} x[b, k] * W_t[k, :] (+ bias)`` with the weight stored transposed ``W_t [K, N]`` so a
surviving input channel is one contiguous row of ``N`` weights: loads of inactive rows are masked off and never touch HBM
(sector granularity), which is where the reported 1.3-1.8x decode speed-ups at 40-50 % sparsity come from.  The threshold
decision is per token (``b``), so batch rows have different active sets; grid ``(cdiv(N, BLOCK_N), B)``.
Accumulation is fp32; weights/activations fp16/bf16 (fp32 supported for tests).
"""
from __future__ import annotations

import torch

from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import cdiv, next_pow2

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _teal_gemv_kernel(
        x_ptr, w_ptr, b_ptr, y_ptr, thr, K, N,
        sx_b, sw_k, sy_b,
        BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        bidx = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = offs_n < N
        acc = tl.zeros([BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            km = offs_k < K
            x = tl.load(x_ptr + bidx.to(tl.int64) * sx_b + offs_k, mask=km, other=0.0).to(tl.float32)
            active = (tl.abs(x) > thr) & km
            w = tl.load(w_ptr + offs_k[:, None].to(tl.int64) * sw_k + offs_n[None, :],
                        mask=active[:, None] & nm[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * x[:, None], axis=0)
        if HAS_BIAS:
            acc += tl.load(b_ptr + offs_n, mask=nm, other=0.0).to(tl.float32)
        tl.store(y_ptr + bidx.to(tl.int64) * sy_b + offs_n, acc.to(y_ptr.dtype.element_ty), mask=nm)


def teal_gemv_triton(x: torch.Tensor, w_t: torch.Tensor, threshold: float, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x [B, K]``, ``w_t [K, N]`` -> ``[B, N]`` in ``x.dtype``."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    b, k = x.shape
    n = w_t.shape[1]
    xc = x.contiguous()
    wc = w_t if w_t.stride(1) == 1 else w_t.contiguous()
    y = torch.empty(b, n, dtype=x.dtype, device=x.device)
    block_n = min(128, max(16, next_pow2(n)))
    _teal_gemv_kernel[(cdiv(n, block_n), b)](
        xc, wc, bias if bias is not None else xc, y, float(threshold), k, n, xc.stride(0), wc.stride(0), y.stride(0),
        BLOCK_K=32, BLOCK_N=block_n, HAS_BIAS=bias is not None, num_warps=4, num_stages=2)
    return y
