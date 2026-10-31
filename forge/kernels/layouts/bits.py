"""Tiny bit-packing helpers shared by the packers (pure torch, CPU-safe, exact)."""
from __future__ import annotations

import torch

_SHIFTS_CACHE: dict[str, torch.Tensor] = {}


def _shifts(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _SHIFTS_CACHE.get(key)
    if t is None:
        t = torch.arange(8, dtype=torch.int64, device=device) * 4
        _SHIFTS_CACHE[key] = t
    return t


def to_int32_wrap(x: torch.Tensor) -> torch.Tensor:
    """Reinterpret non-negative int64 values < 2**32 as two's-complement int32 (exact)."""
    x = x & 0xFFFFFFFF
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def pack_nibbles_int32(nib: torch.Tensor) -> torch.Tensor:
    """``nib[..., 8]`` values 0..15 -> ``int32[...]`` with nibble ``j`` at bits ``4j``."""
    if nib.shape[-1] != 8:
        raise ValueError("last dim must be 8")
    w = (nib.to(torch.int64) & 0xF) << _shifts(nib.device)
    return to_int32_wrap(w.sum(dim=-1))


def unpack_nibbles_int32(words: torch.Tensor) -> torch.Tensor:
    """``int32[...]`` -> ``uint8[..., 8]`` (inverse of :func:`pack_nibbles_int32`)."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    return ((w.unsqueeze(-1) >> _shifts(words.device)) & 0xF).to(torch.uint8)


def roundup(x: int, m: int) -> int:
    return -(-int(x) // int(m)) * int(m)


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))
