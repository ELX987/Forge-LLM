"""W8 weight packer (portable row layout): ``qweight int8 [N, K]`` + ``scales [G, N]`` (column-contiguous).

Symmetric signed int8 (``-128..127``), per-channel (``group_size = -1``) or grouped along K.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from forge.core.errors import ConfigError

from .layout import PORTABLE_W8, Layout, get_layout
from .w4 import effective_group, num_groups


@dataclass
class PackedW8:
    qweight: torch.Tensor  # int8 [N, K]
    scales: torch.Tensor  # [G, N]
    layout: Layout
    n: int
    k: int
    group_size: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def layout_id(self) -> str:
        return self.layout.id

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.k)

    @property
    def num_groups(self) -> int:
        return int(self.scales.shape[0])

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.scales))

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.nbytes / max(1, self.n * self.k)

    def to(self, device: Any = None, dtype: torch.dtype | None = None) -> PackedW8:
        return PackedW8(
            self.qweight.to(device or self.qweight.device),
            self.scales.to(device=device or self.scales.device, dtype=dtype or self.scales.dtype),
            self.layout, self.n, self.k, self.group_size, dict(self.meta),
        )

    def unpack(self) -> tuple[torch.Tensor, torch.Tensor]:
        return unpack_w8(self)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        q, s = unpack_w8(self)
        return dequantize_w8_parts(q, s, self.group_size).to(dtype)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"qweight": self.qweight, "scales": self.scales}

    @staticmethod
    def tensor_roles() -> dict[str, str]:
        return {"qweight": "payload", "scales": "scale"}

    def state(self) -> dict[str, Any]:
        return {"layout_id": self.layout.id, "layout": self.layout.to_dict(), "n": self.n, "k": self.k,
                "group_size": self.group_size, "scale_dtype": str(self.scales.dtype).replace("torch.", "")}

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor], state: dict[str, Any]) -> PackedW8:
        lay = Layout.from_dict(state["layout"]) if "layout" in state else get_layout(state["layout_id"])
        return cls(tensors["qweight"], tensors["scales"], lay, int(state["n"]), int(state["k"]), int(state["group_size"]))


def pack_w8(q: torch.Tensor, scales: torch.Tensor, group_size: int = -1, layout: Layout | str | None = None) -> PackedW8:
    """``q int8 [N, K]``, ``scales [N, G]`` -> :class:`PackedW8`."""
    lay = PORTABLE_W8 if layout is None else (get_layout(layout) if isinstance(layout, str) else layout)
    if lay.bits != 8:
        raise ConfigError(f"layout {lay.id} is not an 8-bit layout")
    if q.dim() != 2 or q.dtype != torch.int8:
        raise ConfigError("q must be int8 [N, K]")
    n, k = q.shape
    g = num_groups(k, group_size)
    if tuple(scales.shape) != (n, g):
        raise ConfigError(f"scales must be [N={n}, G={g}], got {tuple(scales.shape)}")
    return PackedW8(q.contiguous(), scales.t().contiguous(), lay, n, k, int(group_size))


def unpack_w8(p: PackedW8) -> tuple[torch.Tensor, torch.Tensor]:
    return p.qweight, p.scales.t().contiguous()


def dequantize_w8_parts(q: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    n, k = q.shape
    gi = torch.arange(k, device=q.device) // effective_group(group_size, k)
    return q.float() * scales.float()[:, gi]
