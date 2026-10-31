"""Oracle for the W4A16 / W8A16 weight-only GEMMs: dequantise to fp32, matmul in fp32 (Plan C.2 ``w4a16_gemm_fp32``)."""
from __future__ import annotations

import torch

from forge.kernels.layouts import PackedW4, PackedW8


def _linear_fp32(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    lead = x.shape[:-1]
    y = x.reshape(-1, x.shape[-1]).float() @ w.float().t()
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*lead, w.shape[0])


def w4a16_gemm_ref(
    x: torch.Tensor, packed: PackedW4, bias: torch.Tensor | None = None, *, out_dtype: torch.dtype | None = torch.float32
) -> torch.Tensor:
    """``Y[..., N] = X[..., K] @ dequant(W)^T + bias`` in fp32 (any layout; unpacks first).

    ``out_dtype=None`` returns ``x.dtype``.
    """
    if x.shape[-1] != packed.k:
        raise ValueError(f"x has K={x.shape[-1]} but the packed weight has K={packed.k}")
    y = _linear_fp32(x, packed.dequantize(torch.float32), bias)
    return y.to(out_dtype or x.dtype)


def w4a16_gemm_ref_parts(
    x: torch.Tensor,
    q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same oracle on logical parts ``q [N,K] uint8, scales [N,G], zeros [N,G]|None`` (fp32 output)."""
    from forge.kernels.layouts import dequantize_parts

    return _linear_fp32(x, dequantize_parts(q, scales, zeros, group_size, g_idx), bias)


def w8a16_gemm_ref(
    x: torch.Tensor, packed: PackedW8, bias: torch.Tensor | None = None, *, out_dtype: torch.dtype | None = torch.float32
) -> torch.Tensor:
    """``Y = X @ dequant(Wq)^T + bias`` for symmetric int8 weights, fp32 math."""
    if x.shape[-1] != packed.k:
        raise ValueError(f"x has K={x.shape[-1]} but the packed weight has K={packed.k}")
    y = _linear_fp32(x, packed.dequantize(torch.float32), bias)
    return y.to(out_dtype or x.dtype)
