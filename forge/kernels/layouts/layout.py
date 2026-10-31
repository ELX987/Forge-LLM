"""Layout descriptors and the Plan 6.3 layout deriver.

A :class:`Layout` describes how a quantised weight (and its scales) is arranged in memory.  It is part of the
*artifact* (``layout_id`` is written to the manifest), not of the kernel (Plan 6.1).

Layout ids are deterministic: ``<base>-<6 hex of blake2b(canonical storage description)>``.  Only fields that
change the *stored bytes* enter the id (kind, bits, storage tile, nibble interleave, fragment, scale
orientation, swizzle).  Pure schedule choices (tile defaults, stages, split-K) are kept in ``schedule`` and do
NOT change the id, so two chips that need the same bytes share artifacts instead of forcing a re-pack.
(Deviation from the Plan 6.3 pseudo-code which hashes the tile too; recorded in README_GEMM.md.)

Executors
---------
``triton``  - the Triton kernels in :mod:`forge.kernels.triton` consume ``portable_row`` only.
``native``  - a hand-written/CUTLASS-class kernel consuming the fragment-ordered, bit-interleaved layout
              (Marlin style).  No such kernel ships in Forge yet; the layout, packer and repacker are
              implemented and CPU-tested so artifacts can already be produced for it.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from forge.core.errors import ConfigError, NotFound

from .chip import ChipInfo, chip_info

# Marlin-style bit interleave inside one int32: slot j (bits 4j..4j+3) stores natural element INTERLEAVE[j].
# Then (q & 0x000f000f) yields natural elements (0, 1) as a packed 16-bit pair, (q >> 4) & 0x000f000f -> (2, 3), ...
MARLIN_INTERLEAVE: tuple[int, ...] = (0, 2, 4, 6, 1, 3, 5, 7)
IDENTITY_INTERLEAVE: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

KIND_PORTABLE = "portable_row"
KIND_PORTABLE_ROW = KIND_PORTABLE
KIND_FRAGMENT = "mma_fragment"
KIND_FP4_SCALES = "fp4_scales"


@dataclass(frozen=True)
class Layout:
    """Storage layout descriptor (see module docstring)."""

    id: str
    kind: str
    bits: int
    atom: str = ""  # MMA atom the fragment order was derived for (informational for portable)
    tile_k: int = 1  # storage tile along K
    tile_n: int = 1  # storage tile along N
    vec_bytes: int = 16
    interleave: tuple[int, ...] = IDENTITY_INTERLEAVE
    scales: dict = field(default_factory=dict, hash=False, compare=False)
    swizzle: dict = field(default_factory=dict, hash=False, compare=False)
    partition: dict = field(default_factory=dict, hash=False, compare=False)
    schedule: dict = field(default_factory=dict, hash=False, compare=False)  # tile/stage defaults (NOT in id)
    executor: str = "triton"
    chip_family: str = "generic"

    @property
    def perm(self) -> str:
        """Human-readable permutation name (the 'perm' of Plan 6.3)."""
        if self.kind == KIND_FRAGMENT:
            return f"mma_frag_k{self.tile_k}n{self.tile_n}_il{''.join(map(str, self.interleave))}"
        return "identity"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["interleave"] = list(self.interleave)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Layout:
        d = dict(d)
        d["interleave"] = tuple(d.get("interleave", IDENTITY_INTERLEAVE))
        return cls(**d)

    def storage_signature(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "bits": self.bits,
            "atom": self.atom if self.kind == KIND_FRAGMENT else "",
            "tile_k": self.tile_k,
            "tile_n": self.tile_n,
            "interleave": list(self.interleave),
            "scales": self.scales,
            "swizzle": self.swizzle,
        }


def _make_id(base: str, signature: dict[str, Any]) -> str:
    blob = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return f"{base}-{hashlib.blake2b(blob, digest_size=3).hexdigest()}"


# --------------------------------------------------------------------------------------------------
# registry (so artifacts that only carry a ``layout_id`` can be unpacked)
# --------------------------------------------------------------------------------------------------
_LOCK = threading.RLock()
_REGISTRY: dict[str, Layout] = {}


def register_layout(layout: Layout) -> Layout:
    with _LOCK:
        _REGISTRY[layout.id] = layout
    return layout


def get_layout(layout_id: str) -> Layout:
    """Look up a layout by id (builtin layouts are registered at import)."""
    with _LOCK:
        lay = _REGISTRY.get(layout_id)
    if lay is None:
        raise NotFound(f"unknown layout id {layout_id!r}", details={"known": sorted(_REGISTRY)})
    return lay


def known_layouts() -> list[Layout]:
    with _LOCK:
        return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def portable_layout(bits: int) -> Layout:
    """The portable row layout consumed by the Triton kernels.

    * 4-bit: ``qweight int32 [N, ceil(K/8)]`` sequential nibbles (element ``k`` at word ``k//8``, bits ``4*(k%8)``),
      scales ``[G, N]`` column-contiguous, zeros ``int32 [G, ceil(N/8)]`` packed along N.
    * 8-bit: ``qweight int8 [N, K]``, scales ``[G, N]``.
    """
    if bits not in (4, 8):
        raise ConfigError(f"portable layout supports 4 or 8 bits, got {bits}")
    sig = {
        "kind": KIND_PORTABLE,
        "bits": bits,
        "atom": "",
        "tile_k": 8 if bits == 4 else 1,
        "tile_n": 1,
        "interleave": list(IDENTITY_INTERLEAVE),
        "scales": {"orientation": "col_contig_GxN", "pad_n_to": 1},
        "swizzle": {"kind": "none"},
    }
    lay = Layout(
        id=_make_id(f"w{bits}.portable_row.v1", sig),
        kind=KIND_PORTABLE,
        bits=bits,
        atom="",
        tile_k=sig["tile_k"],
        tile_n=1,
        scales=sig["scales"],
        swizzle=sig["swizzle"],
        executor="triton",
        chip_family="any",
    )
    return register_layout(lay)


def fragment_layout(atom: str = "m16n8k16", interleave: tuple[int, ...] = MARLIN_INTERLEAVE, vec_bytes: int = 16) -> Layout:
    """Marlin-style fragment-ordered, bit-interleaved 4-bit layout for a 16(K) x 64(N) storage tile.

    One lane of a ``m16n8k16`` MMA owns, per 8-wide N block, the B elements
    ``(k, n) = (2*(lane%4) + {0,1} + 8*h, lane//4)``, ``h in {0,1}`` (4 elements).  Eight N blocks give 32
    nibbles = 128 bits = one 16-byte vector per lane (``elems_per_v = vec_bytes*8/4``).
    """
    if vec_bytes != 16:
        raise ConfigError("fragment layout is derived for 16-byte vector loads")
    sig = {
        "kind": KIND_FRAGMENT,
        "bits": 4,
        "atom": "mma_b_k2x2",  # atom-independent fragment ordering shared by m16n8k8 (x2) and m16n8k16
        "tile_k": 16,
        "tile_n": 64,
        "interleave": list(interleave),
        "scales": {"orientation": "col_contig_GxN", "pad_n_to": 64},
        "swizzle": {"kind": "xor", "vec_bytes": vec_bytes, "rule": "vec_col ^= row % 8"},
    }
    lay = Layout(
        id=_make_id("w4.mma_fragment_k16n64.v1", sig),
        kind=KIND_FRAGMENT,
        bits=4,
        atom=atom,
        tile_k=16,
        tile_n=64,
        vec_bytes=vec_bytes,
        interleave=tuple(interleave),
        scales=sig["scales"],
        swizzle=sig["swizzle"],
        executor="native",
    )
    return register_layout(lay)


FP4_BLOCK = {"nvfp4": 16, "mxfp4": 32}


def fp4_layout(fmt: str = "nvfp4", scale_layout: str = "swizzled") -> Layout:
    """Layout of a block-scaled FP4 tensor: payload ``uint8 [rows, K/2]`` (low nibble = even k) plus block scales
    either ``logical`` ``[rows, K/B]`` or ``swizzled`` (padded to ``[128m, 4n]`` and rearranged into 128x4 atoms,
    P5 section 11.2).  The tensor scale (NVFP4) is a separate fp32 scalar."""
    if fmt not in FP4_BLOCK:
        raise ConfigError(f"unknown fp4 format {fmt!r} (expected nvfp4 or mxfp4)")
    if scale_layout not in ("logical", "swizzled"):
        raise ConfigError("scale_layout must be 'logical' or 'swizzled'")
    sig = {
        "kind": KIND_FP4_SCALES,
        "bits": 4,
        "atom": fmt,
        "tile_k": FP4_BLOCK[fmt],
        "tile_n": 1,
        "interleave": [],
        "scales": {
            "format": "e4m3" if fmt == "nvfp4" else "e8m0",
            "block": FP4_BLOCK[fmt],
            "layout": scale_layout,
            "atom": [128, 4] if scale_layout == "swizzled" else None,
        },
        "swizzle": {"kind": "none"},
    }
    lay = Layout(
        id=_make_id(f"fp4.{fmt}.{'swz128x4' if scale_layout == 'swizzled' else 'logical'}.v1", sig),
        kind=KIND_FP4_SCALES,
        bits=4,
        atom=fmt,
        tile_k=FP4_BLOCK[fmt],
        tile_n=1,
        interleave=(),
        scales=sig["scales"],
        swizzle=sig["swizzle"],
        executor="triton",
    )
    return register_layout(lay)


# register builtins
PORTABLE_W4 = portable_layout(4)
PORTABLE_W8 = portable_layout(8)
FRAGMENT_W4 = fragment_layout()
for _f in ("nvfp4", "mxfp4"):
    for _s in ("logical", "swizzled"):
        fp4_layout(_f, _s)


# --------------------------------------------------------------------------------------------------
# deriver (Plan 6.3)
# --------------------------------------------------------------------------------------------------
def mma_atom_for(chip: ChipInfo, compute_dtype: str = "bf16") -> str:
    """Matrix-unit instruction used for 16-bit operands (the weights are dequantised into this dtype)."""
    if chip.sm is None:
        return "none"
    if chip.sm >= 80:
        return "m16n8k16"
    if chip.sm >= 75:
        return "m16n8k8"
    return "m8n8k4"  # Volta HMMA: fragment order differs -> no fragment layout is derived


def candidate_tiles(spec: Any, m_hint: int = 16) -> list[tuple[int, int, int]]:
    """``(tile_m, tile_n, tile_k)`` candidates from the spec's search space (M capped by the shape hint)."""
    tm = [t for t in spec.search.tile_m if t <= max(16, _next_pow2(m_hint))] or [min(spec.search.tile_m)]
    return [(m, n, k) for m in tm for n in spec.search.tile_n for k in spec.search.tile_k]


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


def tile_smem_bytes(tile: tuple[int, int, int], stages: int, bits: float, group: int, act_bytes: int = 2, scale_bytes: int = 2) -> int:
    """Shared-memory bytes of the multi-stage ring: ``stages * (A + B + scales)`` (Plan 6.3 step 3)."""
    tm, tn, tk = tile
    g = tk if group <= 0 else group
    a = tm * tk * act_bytes
    b = int(tk * tn * bits / 8)
    s = max(1, tk // g) * tn * scale_bytes
    return stages * (a + b + s)


def tile_acc_regs_per_thread(tile: tuple[int, int, int], warps: int) -> int:
    tm, tn, _ = tile
    return max(1, tm * tn // (32 * warps))


def choose_tile(spec: Any, chip: ChipInfo, *, m_hint: int = 16, group: int = 128, bits: float | None = None) -> dict[str, int]:
    """Pick tile / stages / warps maximising reuse under smem and register capacity (Plan 6.3 step 3)."""
    bits = float(bits if bits is not None else spec.quant.weight_bits)
    budget = int((chip.smem_per_sm_kib or 48) * 1024 * 0.9)
    best: tuple[float, dict[str, int]] | None = None
    for tile in candidate_tiles(spec, m_hint):
        for warps in spec.search.warps:
            if tile_acc_regs_per_thread(tile, warps) > 128:  # keep accumulators within ~half the register budget
                continue
            for stages in spec.search.stages:
                if not chip.has_cp_async and stages > 2:
                    continue  # no async-copy ring on pre-Ampere parts
                if tile_smem_bytes(tile, stages, bits, group) > budget:
                    continue
                tm, tn, tk = tile
                reuse = tm * tn / (tm + tn)  # FLOPs per loaded element (roofline-ish)
                score = reuse * 1000 + tn * tk * 0.01 + stages  # prefer big N tile, then deeper pipeline
                cand = {"tile_m": tm, "tile_n": tn, "tile_k": tk, "stages": stages, "warps": warps}
                if best is None or score > best[0]:
                    best = (score, cand)
    if best is None:  # extremely small smem: smallest everything
        return {"tile_m": 16, "tile_n": 64, "tile_k": 64, "stages": 2, "warps": 4}
    return best[1]


def choose_split_k(m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, num_sms: int, max_split: int = 8) -> int:
    """Smallest split-K that fills the SMs for skinny decode shapes (Plan 6.2 'wave quantisation').

    Never splits so far that a split gets fewer than two K tiles.  ``num_sms <= 0`` (unknown) -> 1.
    """
    if num_sms <= 0:
        return 1
    tiles = -(-m // tile_m) * -(-n // tile_n)
    k_tiles = -(-k // tile_k)
    split = 1
    while tiles * split < num_sms and split < max_split and k_tiles // (split * 2) >= 2:
        split *= 2
    return split


def xor_swizzle(banks: int, bank_width: int, vec_bytes: int) -> dict[str, Any]:
    """Shared-memory XOR swizzle eliminating bank conflicts for 16-byte vector accesses down a column."""
    row_bytes = banks * bank_width  # bytes spanned by all banks (128 on NVIDIA)
    vecs_per_row = max(1, row_bytes // vec_bytes)
    return {
        "kind": "xor",
        "vec_bytes": vec_bytes,
        "vecs_per_row": vecs_per_row,
        "rule": f"vec_col ^= row % {vecs_per_row}",
    }


def derive_layout(
    spec: Any,
    chip: Any = None,
    *,
    executor: str = "triton",
    shape: tuple[int, int, int] | None = None,
    group_size: int = 128,
    bits: int | None = None,
    fmt: str = "nvfp4",
) -> Layout:
    """Derive the storage layout + schedule defaults of ``spec`` for ``chip`` (Plan 6.3).

    ``executor="triton"`` (default) returns the portable row layout (the only one the shipped kernels read) but
    still attaches chip-derived schedule defaults and partitioning; ``executor="native"`` returns the
    fragment-ordered bit-interleaved layout when the chip's matrix unit supports it.
    ``shape=(M, N, K)`` additionally resolves ``split_k='auto'`` into a number.
    """
    c = chip_info(chip)
    if spec.kernel == "fp4_block_gemm":
        base4 = fp4_layout(fmt, "swizzled" if c.has_fp4_block_scaled else "logical")
        return Layout(**{**base4.__dict__, "chip_family": c.family, "schedule": choose_tile(spec, c, bits=4, group=FP4_BLOCK[fmt])})
    wbits = int(bits if bits is not None else spec.quant.weight_bits)
    if executor not in ("triton", "native"):
        raise ConfigError(f"unknown executor {executor!r} (expected 'triton' or 'native')")
    m_hint = int(shape[0]) if shape else 16
    sched = choose_tile(spec, c, m_hint=m_hint, group=group_size, bits=wbits)
    atom = mma_atom_for(c, spec.compute_dtype)

    use_fragment = executor == "native" and wbits == 4 and c.sm is not None and atom in ("m16n8k16", "m16n8k8")
    if wbits not in (4, 8):
        raise ConfigError(f"derive_layout supports 4/8-bit weights, spec has {wbits}")
    base = fragment_layout(atom) if use_fragment else portable_layout(wbits)

    sched = dict(sched)
    sched["split_k"] = "auto"
    if shape:
        sched["split_k"] = choose_split_k(shape[0], shape[1], shape[2], sched["tile_m"], sched["tile_n"], sched["tile_k"], c.num_sms)
    swz = xor_swizzle(c.smem_banks, c.bank_width_bytes, c.vec_bytes) if use_fragment else base.swizzle
    partition = {
        "policy": "striped",
        "units": c.num_sms,
        "split_k": sched["split_k"],
        "deterministic_reduce": True,
        "smem_budget_bytes": int((c.smem_per_sm_kib or 0) * 1024 * 0.9),
    }
    lay = Layout(
        id=base.id,
        kind=base.kind,
        bits=base.bits,
        atom=atom if use_fragment else base.atom,
        tile_k=base.tile_k,
        tile_n=base.tile_n,
        vec_bytes=min(c.vec_bytes, 16),
        interleave=base.interleave,
        scales=dict(base.scales),
        swizzle=swz,
        partition=partition,
        schedule=sched,
        executor=executor,
        chip_family=c.family,
    )
    return lay
