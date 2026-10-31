"""Triton W8A16 GEMM launcher for :class:`forge.kernels.layouts.PackedW8` (symmetric int8, portable_row)."""
from __future__ import annotations

from typing import Any

import torch

from forge.kernels.layouts import PackedW8

from .gemm_common import HAS_TRITON
from .gemm_wq import wq_gemm


def w8a16_gemm_triton(
    x: torch.Tensor,
    packed: PackedW8,
    bias: torch.Tensor | None = None,
    *,
    config: dict[str, Any] | None = None,
    deterministic: bool = True,
    sm: int | None = None,
    num_sms: int = 0,
) -> torch.Tensor:
    """``Y = X @ dequant(Wq)^T + bias`` with ``x [..., K]`` (returns ``x.dtype``)."""
    lead = x.shape[:-1]
    y = wq_gemm(
        x.reshape(-1, x.shape[-1]), packed.qweight, packed.scales, None, None, bias,
        bits=8, n=packed.n, k=packed.k, group_size=packed.group_size, config=config, deterministic=deterministic,
        sm=sm, num_sms=num_sms,
    )
    return y.reshape(*lead, packed.n)


__all__ = ["w8a16_gemm_triton", "HAS_TRITON"]
