"""Triton W4A16 GEMM launcher for :class:`forge.kernels.layouts.PackedW4` in the ``portable_row`` layout.

See :mod:`forge.kernels.triton.gemm_wq` for the kernel, schedule and validation status.
"""
from __future__ import annotations

from typing import Any

import torch

from forge.core.errors import ConfigError
from forge.kernels.layouts import KIND_PORTABLE_ROW, PackedW4

from .gemm_common import HAS_TRITON
from .gemm_wq import wq_gemm


def w4a16_gemm_triton(
    x: torch.Tensor,
    packed: PackedW4,
    bias: torch.Tensor | None = None,
    *,
    config: dict[str, Any] | None = None,
    deterministic: bool = True,
    sm: int | None = None,
    num_sms: int = 0,
) -> torch.Tensor:
    """``Y = X @ dequant(W)^T + bias`` with ``x [..., K]`` (returns ``x.dtype``).  ``packed`` must be ``portable_row``
    (use :func:`forge.kernels.layouts.repack` first otherwise)."""
    if packed.layout.kind != KIND_PORTABLE_ROW:
        raise ConfigError(f"Triton W4A16 kernel needs the portable_row layout, got {packed.layout_id}; repack() first")
    lead = x.shape[:-1]
    y = wq_gemm(
        x.reshape(-1, x.shape[-1]), packed.qweight, packed.scales, packed.zeros, packed.g_idx, bias,
        bits=4, n=packed.n, k=packed.k, group_size=packed.group_size, config=config, deterministic=deterministic,
        sm=sm, num_sms=num_sms,
    )
    return y.reshape(*lead, packed.n)


__all__ = ["w4a16_gemm_triton", "HAS_TRITON"]
