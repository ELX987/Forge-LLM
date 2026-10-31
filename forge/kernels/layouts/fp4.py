"""Block-scaled FP4 containers and the padded + swizzled 128x4 scale layout (Plan 6.4 K2, paper P5 section 11.2).

* ``payload``  ``uint8 [rows, K_pad/2]``: two E2M1 codes per byte, **low nibble = even k**.
* ``scales``   raw ``uint8`` bit patterns: E4M3 (NVFP4, block 16) or E8M0 (MXFP4, block 32, value ``2**(c-127)``).
  Either *logical* ``[rows, K_pad/B]`` or *swizzled*: padded to ``[ceil(rows/128)*128, ceil(C/4)*4]`` and rearranged
  into 128x4 atoms (512 contiguous bytes each), flattened to 1-D.
* ``tensor_scale`` fp32 scalar (NVFP4; 1.0 for MXFP4).

The swizzle is the cuBLAS block-scale-factor layout: inside one 128x4 atom, element ``(r, c)`` lives at byte
``(r % 32) * 16 + (r // 32) * 4 + c``; atoms are ordered row-block major, then column-block.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from forge.core.errors import ConfigError

from .bits import ceil_div, roundup
from .layout import FP4_BLOCK, Layout, fp4_layout, get_layout

ATOM_ROWS = 128
ATOM_COLS = 4


def padded_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """Scale tensor shape after padding rows to a multiple of 128 and columns to a multiple of 4."""
    return roundup(rows, ATOM_ROWS), roundup(cols, ATOM_COLS)


def swizzled_scale_index(r: torch.Tensor, c: torch.Tensor, n_col_blocks: int) -> torch.Tensor:
    """Flat byte offset of logical scale ``(r, c)`` in the swizzled layout (also used inside the Triton kernels)."""
    rb, ri = r // ATOM_ROWS, r % ATOM_ROWS
    cb, ci = c // ATOM_COLS, c % ATOM_COLS
    return (rb * n_col_blocks + cb) * 512 + (ri % 32) * 16 + (ri // 32) * 4 + ci


def _swizzle_perm(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """``flat[idx[i, j]]`` positions: index of every padded logical element in the swizzled buffer."""
    rp, cp = padded_scale_shape(rows, cols)
    r = torch.arange(rp, device=device).view(-1, 1).expand(rp, cp)
    c = torch.arange(cp, device=device).view(1, -1).expand(rp, cp)
    return swizzled_scale_index(r, c, cp // ATOM_COLS)


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Logical ``[rows, cols]`` (any 1-byte dtype or float) -> padded + swizzled flat tensor (pad value 0)."""
    if scales.dim() != 2:
        raise ConfigError("scales must be 2-D [rows, cols]")
    rows, cols = scales.shape
    rp, cp = padded_scale_shape(rows, cols)
    padded = torch.zeros((rp, cp), dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    idx = _swizzle_perm(rows, cols, scales.device)
    out = torch.empty(rp * cp, dtype=scales.dtype, device=scales.device)
    out[idx.reshape(-1)] = padded.reshape(-1)
    return out


def unswizzle_scales(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Exact inverse of :func:`swizzle_scales` (crops the padding)."""
    rp, cp = padded_scale_shape(rows, cols)
    if flat.numel() != rp * cp:
        raise ConfigError(f"swizzled scale buffer has {flat.numel()} elements, expected {rp * cp}")
    idx = _swizzle_perm(rows, cols, flat.device)
    padded = flat.reshape(-1)[idx.reshape(-1)].view(rp, cp)
    return padded[:rows, :cols].contiguous()


# Architecture-contract names (docs/ARCHITECTURE.md 5.5): pack_nvfp4_scales / unpack (128x4 swizzle)
pack_nvfp4_scales = swizzle_scales
unpack_nvfp4_scales = unswizzle_scales
pack_mxfp4_scales = swizzle_scales
unpack_mxfp4_scales = unswizzle_scales


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """``uint8 [..., K]`` codes 0..15 (K even) -> ``uint8 [..., K/2]``, low nibble = even index."""
    if codes.shape[-1] % 2:
        raise ConfigError("last dim must be even to pack two nibbles per byte")
    c = codes.to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def unpack_e2m1(payload: torch.Tensor) -> torch.Tensor:
    """``uint8 [..., K/2]`` -> ``uint8 [..., K]`` codes."""
    lo = payload & 0xF
    hi = payload >> 4
    return torch.stack((lo, hi), dim=-1).reshape(*payload.shape[:-1], payload.shape[-1] * 2)


@dataclass
class PackedFP4:
    """A block-scaled FP4 matrix (weights or quantised activations).  ``fmt`` is ``'nvfp4'`` or ``'mxfp4'``."""

    fmt: str
    payload: torch.Tensor  # uint8 [rows, k_pad/2]
    scales: torch.Tensor  # uint8 bit patterns; logical [rows, k_pad/B] or swizzled flat
    tensor_scale: torch.Tensor  # fp32 0-d
    rows: int
    k: int  # logical K (before padding to a multiple of the block)
    scale_layout: str = "logical"  # 'logical' | 'swizzled'
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fmt not in FP4_BLOCK:
            raise ConfigError(f"unknown fp4 format {self.fmt!r}")
        if self.scale_layout not in ("logical", "swizzled"):
            raise ConfigError("scale_layout must be 'logical' or 'swizzled'")

    @property
    def block(self) -> int:
        return FP4_BLOCK[self.fmt]

    @property
    def k_pad(self) -> int:
        return roundup(self.k, self.block)

    @property
    def n_blocks(self) -> int:
        return self.k_pad // self.block

    @property
    def layout(self) -> Layout:
        return fp4_layout(self.fmt, self.scale_layout)

    @property
    def layout_id(self) -> str:
        return self.layout.id

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.k)

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.payload, self.scales, self.tensor_scale))

    @property
    def bits_per_element(self) -> float:
        return 8.0 * self.nbytes / max(1, self.rows * self.k)

    def logical_scales(self) -> torch.Tensor:
        """uint8 scale bits as ``[rows, n_blocks]`` regardless of storage layout."""
        if self.scale_layout == "logical":
            return self.scales
        return unswizzle_scales(self.scales, self.rows, self.n_blocks)

    def with_scale_layout(self, scale_layout: str) -> PackedFP4:
        """Re-layout the scales (pure permutation, bit exact)."""
        if scale_layout == self.scale_layout:
            return self
        if scale_layout == "swizzled":
            sc = swizzle_scales(self.logical_scales())
        else:
            sc = self.logical_scales().contiguous()
        return PackedFP4(self.fmt, self.payload, sc, self.tensor_scale, self.rows, self.k, scale_layout, dict(self.meta))

    def to(self, device: Any) -> PackedFP4:
        return PackedFP4(self.fmt, self.payload.to(device), self.scales.to(device), self.tensor_scale.to(device),
                         self.rows, self.k, self.scale_layout, dict(self.meta))

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"payload": self.payload, "scales": self.scales, "tensor_scale": self.tensor_scale}

    @staticmethod
    def tensor_roles() -> dict[str, str]:
        return {"payload": "payload", "scales": "scale", "tensor_scale": "tensor_scale"}

    def state(self) -> dict[str, Any]:
        return {"fmt": self.fmt, "rows": self.rows, "k": self.k, "scale_layout": self.scale_layout, "layout_id": self.layout_id}

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor], state: dict[str, Any]) -> PackedFP4:
        return cls(state["fmt"], tensors["payload"], tensors["scales"], tensors["tensor_scale"],
                   int(state["rows"]), int(state["k"]), state.get("scale_layout", "logical"))

    @classmethod
    def from_nvfp4_tensor(cls, t: Any) -> PackedFP4:
        """Adapt ``forge.formats.fp4.NVFP4Tensor{payload_u8, block_scales_e4m3, tensor_scale, shape}`` (duck-typed)."""
        payload = t.payload_u8
        scales = t.block_scales_e4m3
        if scales.dtype != torch.uint8:
            scales = scales.contiguous().view(torch.uint8)
        rows, k = (int(s) for s in t.shape[-2:]) if len(t.shape) >= 2 else (1, int(t.shape[0]))
        ts = torch.as_tensor(t.tensor_scale, dtype=torch.float32).reshape(())
        return cls("nvfp4", payload.reshape(rows, -1).contiguous(), scales.reshape(rows, -1).contiguous(), ts, rows, k)
