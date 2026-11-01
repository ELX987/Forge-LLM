"""W4 weight packer / unpacker / repacker (Plan 6.3, 6.4 K1, C-041).

Logical input of the packer (the contract of ``docs/ARCHITECTURE.md``):

* ``q``      ``uint8 [N, K]`` values ``0..15`` (unsigned code)
* ``scales`` ``[N, G]`` (fp16 / bf16 / fp32), ``G = ceil(K / group_size)`` (``group_size = -1`` -> one group = per channel)
* ``zeros``  ``uint8 [N, G]`` values ``0..15`` or ``None`` (``None`` means the symmetric zero-point 8)
* ``g_idx``  optional ``int32 [K]`` (external GPTQ desc_act checkpoints: group id of every input column; Forge's own act-order GPTQ uses static groups and stores none)

Dequantised weight: ``W[n, k] = (q[n, k] - zero[n, g(k)]) * scale[n, g(k)]`` with ``g(k) = k // group_size``
(or ``g_idx[k]``).

Two physical layouts are implemented (both exactly invertible, both property-tested):

``portable_row``  (consumed by the Triton kernels)
    ``qweight int32 [N, ceil(K/8)]``, nibble ``j`` of word ``w`` holds ``q[n, 8*w + j]``;
    ``scales [G, N]`` column-contiguous; ``zeros int32 [G, ceil(N/8)]`` (8 channels per word).

``mma_fragment`` (Marlin-style; for a native kernel)
    ``qweight int32 [K_pad/16, N_pad*2]``: the matrix is cut in 16(K) x 64(N) tiles; inside a tile the 1024
    nibbles are stored lane-major in ``m16n8k16`` B-fragment order (lane ``l`` owns ``k = 2*(l%4) + {0,1} + 8h``,
    ``n = 8*nb + l//4``; 32 nibbles = one 16-byte vector per lane) and bit-interleaved inside each int32
    (``MARLIN_INTERLEAVE``) so that ``q & 0x000f000f`` yields a ready fp16x2/bf16x2 pair.  ``K_pad`` and ``N_pad``
    are multiples of 16 and 64; ``scales``/``zeros`` are padded to ``N_pad`` (scale 0).

Padding never changes numerics: padded weights multiply zero-padded activations / zero scales and are cropped
by :func:`unpack_w4`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from forge.core.errors import ConfigError

from .bits import pack_nibbles_int32, roundup, unpack_nibbles_int32
from .layout import KIND_FRAGMENT, KIND_PORTABLE, PORTABLE_W4, Layout, get_layout

_PERM_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

TILE_K = 16
TILE_N = 64


def effective_group(group_size: int, k: int) -> int:
    """Group size in elements (``-1``/``0`` -> whole row)."""
    return int(k) if group_size is None or group_size <= 0 else int(group_size)


def num_groups(k: int, group_size: int) -> int:
    return -(-int(k) // effective_group(group_size, k))


def fragment_tile_perm(interleave: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Index tables for one 16x64 tile.

    Returns ``(idx, inv)`` where ``idx[pos]`` is the flat ``k*64 + n`` tile coordinate stored at nibble position
    ``pos`` (0..1023, word ``pos//8``, bit ``4*(pos%8)``) and ``inv`` is the inverse permutation.
    """
    key = tuple(interleave)
    hit = _PERM_CACHE.get(key)
    if hit is not None:
        return hit
    idx = torch.empty(TILE_K * TILE_N, dtype=torch.int64)
    for word in range(128):
        for slot in range(8):
            natural = word * 8 + interleave[slot]  # natural (pre-interleave) position inside the tile
            lane, e = divmod(natural, 32)
            nb, r = divmod(e, 4)
            k = (lane % 4) * 2 + (r % 2) + 8 * (r // 2)
            n = nb * 8 + lane // 4
            idx[word * 8 + slot] = k * TILE_N + n
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), dtype=torch.int64)
    if len(set(idx.tolist())) != idx.numel():  # self-check: must be a permutation
        raise AssertionError("fragment permutation is not bijective")
    _PERM_CACHE[key] = (idx, inv)
    return idx, inv


@dataclass
class PackedW4:
    """A packed 4-bit weight + scales (+ zeros, g_idx) in some :class:`Layout`."""

    qweight: torch.Tensor  # int32
    scales: torch.Tensor  # [G, n_pad]
    zeros: torch.Tensor | None  # int32 [G, ceil(n_pad/8)] or None (symmetric, zero-point 8)
    layout: Layout
    n: int
    k: int
    group_size: int  # as given (-1 = per channel)
    g_idx: torch.Tensor | None = None  # int32 [K]
    meta: dict[str, Any] = field(default_factory=dict)

    # -- properties ------------------------------------------------------------------------------
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
        t = [self.qweight, self.scales] + [x for x in (self.zeros, self.g_idx) if x is not None]
        return sum(x.numel() * x.element_size() for x in t)

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.nbytes / max(1, self.n * self.k)

    # -- conversions -----------------------------------------------------------------------------
    def to(self, device: Any = None, dtype: torch.dtype | None = None) -> PackedW4:
        """Move to a device (``dtype`` applies to the scales only)."""
        def mv(t, dt=None):
            return None if t is None else t.to(device=device or t.device, dtype=dt or t.dtype)

        return PackedW4(
            mv(self.qweight), mv(self.scales, dtype), mv(self.zeros), self.layout, self.n, self.k, self.group_size,
            mv(self.g_idx), dict(self.meta),
        )

    def unpack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return unpack_w4(self)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dense ``[N, K]`` weight (fp32 math, then cast)."""
        q, s, z = unpack_w4(self)
        return dequantize_parts(q, s, z, self.group_size, self.g_idx).to(dtype)

    def tensors(self) -> dict[str, torch.Tensor]:
        """Named tensors for artifact storage (see :meth:`tensor_roles`)."""
        out = {"qweight": self.qweight, "scales": self.scales}
        if self.zeros is not None:
            out["zeros"] = self.zeros
        if self.g_idx is not None:
            out["g_idx"] = self.g_idx
        return out

    @staticmethod
    def tensor_roles() -> dict[str, str]:
        return {"qweight": "payload", "scales": "scale", "zeros": "zero", "g_idx": "perm"}

    def state(self) -> dict[str, Any]:
        """JSON-serialisable metadata (pair with :meth:`tensors`; restore with :meth:`from_tensors`)."""
        return {
            "layout_id": self.layout.id,
            "layout": self.layout.to_dict(),
            "n": self.n,
            "k": self.k,
            "group_size": self.group_size,
            "scale_dtype": str(self.scales.dtype).replace("torch.", ""),
        }

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor], state: dict[str, Any]) -> PackedW4:
        lay = Layout.from_dict(state["layout"]) if "layout" in state else get_layout(state["layout_id"])
        return cls(
            tensors["qweight"], tensors["scales"], tensors.get("zeros"), lay, int(state["n"]), int(state["k"]),
            int(state["group_size"]), tensors.get("g_idx"),
        )


def _check_inputs(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor | None, group_size: int, g_idx: torch.Tensor | None):
    if q.dim() != 2:
        raise ConfigError(f"q must be [N, K], got {tuple(q.shape)}")
    n, k = q.shape
    if n == 0 or k == 0:
        raise ConfigError("empty weight")
    if group_size is not None and group_size > 0 and group_size % 8 != 0:
        raise ConfigError(f"group_size must be -1 or a multiple of 8, got {group_size}")
    if g_idx is None:
        g = num_groups(k, group_size)
    else:
        if g_idx.shape != (k,):
            raise ConfigError("g_idx must have shape [K]")
        g = int(scales.shape[1])
        if int(g_idx.max()) >= g or int(g_idx.min()) < 0:
            raise ConfigError("g_idx out of range for scales")
    if tuple(scales.shape) != (n, g):
        raise ConfigError(f"scales must be [N={n}, G={g}], got {tuple(scales.shape)}")
    if zeros is not None and tuple(zeros.shape) != (n, g):
        raise ConfigError(f"zeros must be [N={n}, G={g}], got {tuple(zeros.shape)}")
    if q.dtype != torch.uint8:
        raise ConfigError("q must be uint8 (values 0..15)")
    if int(q.max()) > 15:
        raise ConfigError("q contains values > 15")
    if zeros is not None and (int(zeros.max()) > 15 or int(zeros.min()) < 0):
        raise ConfigError("zeros must be in 0..15")  # an out-of-range zero would silently wrap in the uint8 cast / nibble pack


def pack_w4(
    q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
    layout: Layout | str | None = None,
    g_idx: torch.Tensor | None = None,
) -> PackedW4:
    """Pack logical ``(q, scales, zeros)`` into ``layout`` (default ``portable_row``).  See module docstring."""
    lay = PORTABLE_W4 if layout is None else (get_layout(layout) if isinstance(layout, str) else layout)
    if lay.bits != 4:
        raise ConfigError(f"layout {lay.id} is not a 4-bit layout")
    _check_inputs(q, scales, zeros, group_size, g_idx)
    n, k = q.shape
    g = scales.shape[1]
    gs = effective_group(group_size, k)
    if lay.kind == KIND_FRAGMENT and group_size > 0 and gs % TILE_K != 0:
        raise ConfigError(f"fragment layout needs group_size multiple of 16 (got {group_size})")
    if lay.kind == KIND_PORTABLE:
        k_pad, n_pad = roundup(k, 8), n
    elif lay.kind == KIND_FRAGMENT:
        k_pad, n_pad = roundup(k, TILE_K), roundup(n, TILE_N)
    else:
        raise ConfigError(f"cannot pack W4 into layout kind {lay.kind}")

    dev = q.device
    zp = zeros.to(torch.uint8) if zeros is not None else torch.full((n, g), 8, dtype=torch.uint8, device=dev)
    qp = torch.zeros((n_pad, k_pad), dtype=torch.uint8, device=dev)
    qp[:n, :k] = q
    if k_pad > k:  # pad columns carry the zero-point of the column's group so they dequantise to exactly 0
        if g_idx is None:
            gk = torch.clamp(torch.arange(k, k_pad, device=dev) // gs, max=g - 1)
        else:
            gk = g_idx[-1].long().repeat(k_pad - k).to(dev)
        qp[:n, k:] = zp[:, gk]

    if lay.kind == KIND_PORTABLE:
        qweight = pack_nibbles_int32(qp.view(n_pad, k_pad // 8, 8))
    else:
        idx, _ = fragment_tile_perm(lay.interleave)
        kt, nt = k_pad // TILE_K, n_pad // TILE_N
        tiles = qp.t().contiguous().view(kt, TILE_K, nt, TILE_N).permute(0, 2, 1, 3).reshape(kt, nt, TILE_K * TILE_N)
        seq = tiles[..., idx.to(dev)]
        qweight = pack_nibbles_int32(seq.view(kt, nt, 128, 8)).reshape(kt, nt * 128)

    sc = torch.zeros((g, n_pad), dtype=scales.dtype, device=dev)
    sc[:, :n] = scales.t()
    zq = None
    if zeros is not None:
        zt = torch.zeros((g, roundup(n_pad, 8)), dtype=torch.uint8, device=dev)
        zt[:, :n] = zp.t()
        zq = pack_nibbles_int32(zt.view(g, -1, 8))
    gi = None if g_idx is None else g_idx.to(torch.int32).contiguous()
    return PackedW4(qweight.contiguous(), sc.contiguous(), zq, lay, n, k, int(group_size), gi,
                    {"n_pad": n_pad, "k_pad": k_pad})


def unpack_w4(p: PackedW4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Exact inverse of :func:`pack_w4`: ``(q uint8 [N,K], scales [N,G], zeros uint8 [N,G] | None)``."""
    lay, n, k = p.layout, p.n, p.k
    if lay.kind == KIND_PORTABLE:
        nib = unpack_nibbles_int32(p.qweight)  # [N, K8, 8]
        q = nib.reshape(nib.shape[0], -1)[:n, :k]
    elif lay.kind == KIND_FRAGMENT:
        k_pad, n_pad = roundup(k, TILE_K), roundup(n, TILE_N)
        kt, nt = k_pad // TILE_K, n_pad // TILE_N
        _, inv = fragment_tile_perm(lay.interleave)
        seq = unpack_nibbles_int32(p.qweight.view(kt, nt, 128)).reshape(kt, nt, TILE_K * TILE_N)
        tiles = seq[..., inv.to(seq.device)]
        full = tiles.view(kt, nt, TILE_K, TILE_N).permute(0, 2, 1, 3).reshape(k_pad, n_pad)
        q = full.t()[:n, :k]
    else:
        raise ConfigError(f"cannot unpack layout kind {lay.kind}")
    q = q.contiguous()
    scales = p.scales[:, :n].t().contiguous()
    zeros = None
    if p.zeros is not None:
        zeros = unpack_nibbles_int32(p.zeros).reshape(p.zeros.shape[0], -1)[:, :n].t().contiguous()
    return q, scales, zeros


def repack(p: PackedW4, new_layout: Layout | str) -> PackedW4:
    """Re-pack into another layout.  Pure permutation: the logical ``(q, scales, zeros)`` are untouched
    (no re-quantisation), verified bit-exactly by the property tests."""
    lay = get_layout(new_layout) if isinstance(new_layout, str) else new_layout
    if lay.id == p.layout.id:
        return p
    q, s, z = unpack_w4(p)
    return pack_w4(q, s, z, p.group_size, lay, p.g_idx)


def dequantize_parts(
    q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor | None, group_size: int, g_idx: torch.Tensor | None = None
) -> torch.Tensor:
    """fp32 ``[N, K]`` dequantisation of logical parts (shared by the oracle and :meth:`PackedW4.dequantize`)."""
    n, k = q.shape
    if g_idx is None:
        gi = torch.arange(k, device=q.device) // effective_group(group_size, k)
    else:
        gi = g_idx.long().to(q.device)
    s = scales.float()[:, gi]
    z = 8.0 if zeros is None else zeros.float()[:, gi]
    return (q.float() - z) * s


def sort_columns_by_group(p: PackedW4) -> tuple[PackedW4, torch.Tensor] | None:
    """Make an act-order (``g_idx``) weight regular: stable-sort the K columns by group id.

    GPTQ ``act_order`` stores a permuted column order, so the group of column ``k`` is ``g_idx[k]`` (non-monotonic).
    Sorting the columns by group makes every group a contiguous run, i.e. the plain ``k // group_size`` rule holds
    again and kernels can use a vector scale per K tile.  Returns ``(sorted_packed, perm)`` with
    ``perm[j]`` = original column stored at sorted position ``j`` (int32 ``[K]``), so that

        ``x @ W^T == x[:, perm] @ W_sorted^T``      (exact in real arithmetic; fp32 sums are re-ordered)

    or ``None`` when ``p`` has no ``g_idx`` or the groups are irregular (not exactly ``group_size`` columns each, last
    group the remainder) -- callers then use the per-column ``g_idx`` kernel.  Pure permutation, no re-quantisation.
    """
    if p.g_idx is None:
        return None
    gs, g = effective_group(p.group_size, p.k), int(p.scales.shape[0])
    gi = p.g_idx.long().cpu()
    if gi.numel() != p.k or int(gi.min()) < 0 or int(gi.max()) >= g:
        return None
    want = torch.full((g,), gs, dtype=torch.int64)
    want[-1] = p.k - gs * (g - 1)
    if int(want[-1]) <= 0 or not torch.equal(torch.bincount(gi, minlength=g), want):
        return None
    perm = torch.argsort(gi, stable=True)
    q, s, z = unpack_w4(p)
    dev = q.device
    sorted_p = pack_w4(q[:, perm.to(dev)], s, z, p.group_size, p.layout, None)
    return sorted_p, perm.to(torch.int32).to(dev)
