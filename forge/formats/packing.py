"""Bit packing helpers: ``bits`` in {1, 2, 4, 8} codes per byte, little-endian inside a byte.

Element ``j`` of a row lives in byte ``j // (8 // bits)`` at bit offset ``bits * (j % (8 // bits))`` -- i.e. for 4-bit codes the
*even* element is the low nibble and the *odd* element the high nibble (the layout used by ``pack_fp4`` and the portable
int4 format of ``forge.compress``).  All functions pack along the **last** dimension, padding with zeros if needed.
"""
from __future__ import annotations

import torch

from forge.core.errors import ConfigError

_VALID = (1, 2, 4, 8)


def packed_len(n: int, bits: int) -> int:
    per = 8 // bits
    return (n + per - 1) // per


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned ``codes`` (values ``< 2**bits``) along the last dim into ``uint8``."""
    if bits not in _VALID:
        raise ConfigError(f"pack_bits supports bits in {_VALID}, got {bits}")
    c = codes.to(torch.uint8)
    if bits == 8:
        return c.contiguous()
    per = 8 // bits
    n = c.shape[-1]
    pad = (-n) % per
    if pad:
        c = torch.nn.functional.pad(c, (0, pad))
    c = c.reshape(*c.shape[:-1], -1, per).to(torch.int32)
    out = torch.zeros(c.shape[:-1], dtype=torch.int32, device=c.device)
    for i in range(per):
        out |= (c[..., i] & ((1 << bits) - 1)) << (bits * i)
    return out.to(torch.uint8)


def unpack_bits(packed: torch.Tensor, bits: int, n: int | None = None) -> torch.Tensor:
    """Inverse of :func:`pack_bits`; returns ``uint8`` codes, truncated to ``n`` elements along the last dim if given."""
    if bits not in _VALID:
        raise ConfigError(f"unpack_bits supports bits in {_VALID}, got {bits}")
    if bits == 8:
        out = packed.to(torch.uint8)
    else:
        per = 8 // bits
        p = packed.to(torch.int32)
        parts = [(p >> (bits * i)) & ((1 << bits) - 1) for i in range(per)]
        out = torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], -1).to(torch.uint8)
    if n is not None:
        out = out[..., :n]
    return out.contiguous()


def pack_uint4(codes: torch.Tensor) -> torch.Tensor:
    """Two 4-bit codes per byte (even index = low nibble)."""
    return pack_bits(codes, 4)


def unpack_uint4(packed: torch.Tensor, n: int | None = None) -> torch.Tensor:
    return unpack_bits(packed, 4, n)
