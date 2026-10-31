"""Counter-based random numbers for stochastic rounding (device independent, shape independent).

``counter_uniform(shape, seed, offset)`` returns float32 uniforms in ``[0, 1)`` where element ``i`` (flat index) depends *only*
on ``(seed, offset + i)``.  Consequences that matter for stochastic-rounding (SR) numerics:

* the result is identical on CPU and CUDA (only exact 64-bit integer ops are used),
* generating a tensor in one call or in chunks (with ``offset`` advanced) gives identical values,
* SR is reproducible from ``(seed, step, tensor id)`` alone, which is what a deterministic training stage needs
  (``offset = step * stride + tensor_offset``).

The mixer is the well-known "lowbias32" integer hash applied twice (key schedule + counter).  It is *not* cryptographic.
"""
from __future__ import annotations

import torch

_M32 = 0xFFFFFFFF


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """``(x * c) mod 2**32`` for ``0 <= x < 2**32`` without relying on int64 overflow."""
    lo = (x & 0xFFFF) * c
    hi = ((x >> 16) * c) & 0xFFFF
    return (lo + (hi << 16)) & _M32


def _mix32(x: torch.Tensor) -> torch.Tensor:
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    x = x ^ (x >> 16)
    return x


def counter_bits(numel: int, seed: int, offset: int = 0, device: torch.device | str | None = None) -> torch.Tensor:
    """``numel`` pseudo-random 32-bit values (as int64 in ``[0, 2**32)``)."""
    idx = torch.arange(numel, dtype=torch.int64, device=device) + int(offset)
    seed = int(seed) & ((1 << 64) - 1)
    k_lo = _mix32(torch.tensor((seed & _M32) ^ 0x9E3779B9, dtype=torch.int64, device=device))
    k_hi = _mix32(torch.tensor(((seed >> 32) & _M32) ^ 0x85EBCA6B, dtype=torch.int64, device=device))
    lo = idx & _M32
    hi = (idx >> 32) & _M32
    h = _mix32(lo ^ k_lo)
    h = _mix32(h ^ _mix32(hi ^ k_hi))
    return h


def counter_uniform(
    shape: tuple[int, ...] | torch.Size | int,
    seed: int = 0,
    offset: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Uniform float32 in ``[0, 1)`` with 24 random mantissa bits; element ``i`` depends only on ``(seed, offset+i)``."""
    if isinstance(shape, int):
        shape = (shape,)
    n = 1
    for s in shape:
        n *= int(s)
    h = counter_bits(n, seed, offset, device)
    u = (h >> 8).to(torch.float32) * (1.0 / (1 << 24))
    return u.reshape(tuple(shape))
