"""forge.kernels.ref: pure-torch oracles of every kernel (runnable on CPU).

GEMM family (``gemm_*.py``): ``w4a16_gemm_ref``, ``w8a16_gemm_ref``, ``fp8_gemm_ref``, ``fp4_block_gemm_ref``,
``int8_gemm_ref`` + quantisers.  Other families (attention / MoE / util) live in sibling modules written by other
owners; names not defined here are resolved lazily from those sibling modules, so
``from forge.kernels.ref import paged_decode_attention_ref`` keeps working whoever owns the file.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from .gemm_common import CompareResult, compare
from .gemm_fp4 import (
    dequantize_fp4_ref,
    e2m1_decode,
    e2m1_encode,
    fp4_block_gemm_ref,
    fp4_w4a4_gemm_ref,
    quantize_fp4_ref,
    quantize_mxfp4_ref,
    quantize_nvfp4_ref,
)
from .gemm_fp8 import (
    dequantize_fp8_ref,
    fp8_gemm_ref,
    quantize_fp8_ref,
    rmsnorm_act_quant_fp8_ref,
)
from .gemm_int8 import int8_gemm_ref, int8_quant_agreement, quantize_int8_ref
from .gemm_w4a16 import w4a16_gemm_ref, w4a16_gemm_ref_parts, w8a16_gemm_ref

_SKIP = {"gemm_common", "gemm_fp4", "gemm_fp8", "gemm_int8", "gemm_w4a16", "gemm_w8a16"}


def __getattr__(name: str) -> Any:  # PEP 562: resolve oracles defined by other kernel owners
    if name.startswith("__"):
        raise AttributeError(name)
    for mod in pkgutil.iter_modules(__path__):
        if mod.name in _SKIP:
            continue
        try:
            m = importlib.import_module(f"{__name__}.{mod.name}")
        except Exception:
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompareResult", "compare",
    "dequantize_fp4_ref", "e2m1_decode", "e2m1_encode", "fp4_block_gemm_ref", "fp4_w4a4_gemm_ref",
    "quantize_fp4_ref", "quantize_mxfp4_ref", "quantize_nvfp4_ref",
    "dequantize_fp8_ref", "fp8_gemm_ref", "quantize_fp8_ref", "rmsnorm_act_quant_fp8_ref",
    "int8_gemm_ref", "int8_quant_agreement", "quantize_int8_ref", "w4a16_gemm_ref", "w4a16_gemm_ref_parts", "w8a16_gemm_ref",
]
