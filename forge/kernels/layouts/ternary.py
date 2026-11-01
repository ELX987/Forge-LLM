"""Ternary (1.58-bit) weight packer: layout ``ternary2_row``.

``qweight uint8 [N, K/4]`` holds 2-bit codes (00 = 0, 01 = +1, 11 = -1, 10 reserved -> 0), element ``j`` at bits ``2*(j%4)`` of byte
``j//4`` -- byte-identical to tern158's int32 words (16 codes each, little-endian) -- and ``scales fp32 [N, G]`` (``[1, 1]`` when
``group_size == 0``, i.e. one scale per tensor; ``-1`` = one per output row).  ``K`` must be a multiple of 16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from forge.core.errors import ConfigError
from forge.formats import ternary as _tn

from .layout import KIND_PORTABLE, Layout, _make_id, get_layout, register_layout


def ternary_layout() -> Layout:
    sig = {
        "kind": KIND_PORTABLE, "bits": 2, "atom": "", "tile_k": 16, "tile_n": 1, "interleave": [0, 1, 2, 3],
        "scales": {"orientation": "row_major_NxG", "dtype": "fp32", "group": [-1, 0, "k"]},
        "swizzle": {"kind": "none"}, "codes": {"00": 0, "01": 1, "11": -1, "10": "reserved->0"},
    }
    lay = Layout(id=_make_id("ternary2.row.v1", sig), kind=KIND_PORTABLE, bits=2, atom="", tile_k=16, tile_n=1,
                 interleave=(0, 1, 2, 3), scales=sig["scales"], swizzle=sig["swizzle"], executor="triton", chip_family="any")
    return register_layout(lay)


TERNARY2_ROW = ternary_layout()


@dataclass
class PackedTernary:
    qweight: torch.Tensor  # uint8 [N, K/4]
    scales: torch.Tensor  # fp32 [N, G] or [1, 1]
    layout: Layout
    n: int
    k: int
    group_size: int  # >0 group, -1 per row, 0 per tensor
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def layout_id(self) -> str:
        return self.layout.id

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.k)

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.scales))

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.nbytes / max(1, self.n * self.k)

    def to(self, device: Any = None, dtype: torch.dtype | None = None) -> PackedTernary:
        return PackedTernary(self.qweight.to(device or self.qweight.device), self.scales.to(device=device or self.scales.device, dtype=dtype or self.scales.dtype),
                             self.layout, self.n, self.k, self.group_size, dict(self.meta))

    def unpack(self) -> tuple[torch.Tensor, torch.Tensor]:
        return unpack_ternary_weight(self)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        t, s = unpack_ternary_weight(self)
        return _tn.dequantize_ternary(t, s, self.group_size).to(dtype)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"qweight": self.qweight, "scales": self.scales}

    @staticmethod
    def tensor_roles() -> dict[str, str]:
        return {"qweight": "payload", "scales": "scale"}

    def state(self) -> dict[str, Any]:
        return {"layout_id": self.layout.id, "layout": self.layout.to_dict(), "n": self.n, "k": self.k, "group_size": self.group_size,
                "scale_dtype": str(self.scales.dtype).replace("torch.", "")}

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor], state: dict[str, Any]) -> PackedTernary:
        lay = Layout.from_dict(state["layout"]) if "layout" in state else get_layout(state["layout_id"])
        return cls(tensors["qweight"], tensors["scales"], lay, int(state["n"]), int(state["k"]), int(state["group_size"]))


def pack_ternary_weight(t: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> PackedTernary:
    """``t int8 [N, K]`` in {-1,0,1}, ``scales fp32 [N, G]`` (``[1, 1]`` for group_size 0) -> :class:`PackedTernary`."""
    if t.dim() != 2 or t.dtype != torch.int8:
        raise ConfigError("t must be int8 [N, K]")
    n, k = t.shape
    g = 1 if group_size == 0 else k // _tn.effective_group(group_size, k)
    want = (1, 1) if group_size == 0 else (n, g)
    if tuple(scales.shape) != want:
        raise ConfigError(f"scales must be {want}, got {tuple(scales.shape)}")
    return PackedTernary(_tn.pack_ternary(t), scales.float().contiguous(), TERNARY2_ROW, n, k, int(group_size))


def unpack_ternary_weight(p: PackedTernary) -> tuple[torch.Tensor, torch.Tensor]:
    return _tn.unpack_ternary(p.qweight, p.k), p.scales


def quantize_pack_ternary(w: torch.Tensor, group_size: int = 128, rule: str = "lloyd", iters: int = 1) -> PackedTernary:
    t, s = _tn.quantize_ternary(w, group_size, rule, iters)
    return pack_ternary_weight(t, s, group_size)
