"""forge.kernels.layouts: layout deriver (Plan 6.3), W4/W8 packers, FP4 scale swizzle.

Public API (see docs/ARCHITECTURE.md 5.5)::

    derive_layout(spec, chip) -> Layout
    pack_w4(q, scales, zeros, group_size, layout) -> PackedW4 ; unpack_w4(packed) ; repack(packed, new_layout)
    pack_w8 / unpack_w8 ; swizzle_scales / unswizzle_scales ; pack_nvfp4_scales / unpack_nvfp4_scales
"""
from __future__ import annotations

from .bits import ceil_div, pack_nibbles_int32, roundup, unpack_nibbles_int32
from .chip import ChipInfo, chip_from_sm, chip_info
from .fp4 import (
    PackedFP4,
    pack_e2m1,
    pack_mxfp4_scales,
    pack_nvfp4_scales,
    padded_scale_shape,
    swizzle_scales,
    swizzled_scale_index,
    unpack_e2m1,
    unpack_mxfp4_scales,
    unpack_nvfp4_scales,
    unswizzle_scales,
)
from .layout import (
    FP4_BLOCK,
    FRAGMENT_W4,
    KIND_FP4_SCALES,
    KIND_FRAGMENT,
    KIND_PORTABLE,
    KIND_PORTABLE_ROW,
    IDENTITY_INTERLEAVE,
    MARLIN_INTERLEAVE,
    PORTABLE_W4,
    PORTABLE_W8,
    Layout,
    choose_split_k,
    choose_tile,
    derive_layout,
    fp4_layout,
    fragment_layout,
    get_layout,
    known_layouts,
    portable_layout,
    register_layout,
    tile_smem_bytes,
)
from .rtn import quantize_w4_rtn, quantize_w8_rtn
from .w4 import (
    PackedW4,
    dequantize_parts,
    effective_group,
    fragment_tile_perm,
    num_groups,
    pack_w4,
    repack,
    sort_columns_by_group,
    unpack_w4,
)
from .w8 import PackedW8, dequantize_w8_parts, pack_w8, unpack_w8

__all__ = [n for n in dir() if not n.startswith("_")]
