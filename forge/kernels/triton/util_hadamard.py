"""Triton block-Walsh-Hadamard transform (Plan K8).

**Status: implemented, not yet validated on NVIDIA hardware** (runs in the Triton CPU interpreter in the unit tests).
Min Triton 2.1; fp16/bf16 inputs on sm_70+/sm_80+.

The transform of each contiguous ``BLOCK`` (16/32/64/128) elements of the last dimension is computed as
``y = (x * signs) @ H_BLOCK / sqrt(BLOCK)`` where the +-1 Sylvester matrix is generated *in registers* from the parity of
``popcount(i & j)`` - no weight tensor is read, the op stays memory-bound (one read + one write of ``x``) and the +-1
products are exact in bf16/fp16, so the result equals the butterfly oracle up to fp32 summation order.
Larger blocks (e.g. head_dim 256) are composed by the dispatcher (``H_{2n} = H_2 (x) H_n``) or use the torch butterfly.
"""
from __future__ import annotations

import math

import torch

from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import cdiv

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _hadamard_kernel(
        x_ptr, signs_ptr, y_ptr, R, scale,
        B: tl.constexpr, LOG2B: tl.constexpr, NB: tl.constexpr, BLOCK_R: tl.constexpr, HAS_SIGNS: tl.constexpr,
    ):
        """``x`` viewed as ``[R, B]`` rows; row ``r`` is block ``r % NB`` of its original row (for the sign vector)."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rm = rows < R
        cols = tl.arange(0, B)
        offs = rows[:, None].to(tl.int64) * B + cols[None, :]
        x = tl.load(x_ptr + offs, mask=rm[:, None], other=0.0)
        if HAS_SIGNS:
            sg = tl.load(signs_ptr + (rows % NB)[:, None] * B + cols[None, :], mask=rm[:, None], other=1.0)
            x = (x.to(tl.float32) * sg).to(x.dtype)
        i = tl.arange(0, B)[:, None]
        j = tl.arange(0, B)[None, :]
        v = i & j
        par = v * 0
        for bit in tl.static_range(LOG2B):
            par = par ^ ((v >> bit) & 1)
        h = (1 - 2 * par).to(x.dtype)
        y = tl.dot(x, h) * scale
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=rm[:, None])


def hadamard_triton(x: torch.Tensor, block: int, signs: torch.Tensor | None = None, normalize: bool = True) -> torch.Tensor:
    """Block Hadamard along the last dim for ``block`` in {16, 32, 64, 128} (fp16/bf16 ``x``)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    if block not in (16, 32, 64, 128):
        raise ValueError("Triton Hadamard supports blocks 16/32/64/128")
    d = x.shape[-1]
    if d % block:
        raise ValueError("last dim must be a multiple of block")
    if signs is not None and signs.numel() != d:
        # the kernel indexes ``signs[(row % (d // block)) * block + col]``: a shorter vector would be read out of bounds
        raise ValueError(f"signs must have one entry per element of the last dim ({d}), got {signs.numel()}")
    xc = x.contiguous()
    rows = xc.numel() // block
    y = torch.empty_like(xc)
    sg = signs.to(torch.float32).contiguous() if signs is not None else xc
    block_r = 32 if block <= 64 else 16
    _hadamard_kernel[(cdiv(rows, block_r),)](
        xc, sg, y, rows, (1.0 / math.sqrt(block)) if normalize else 1.0,
        B=block, LOG2B=int(math.log2(block)), NB=d // block, BLOCK_R=block_r, HAS_SIGNS=signs is not None,
        num_warps=4, num_stages=2)
    return y.view(x.shape)
