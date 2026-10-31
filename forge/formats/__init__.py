"""forge.formats -- fake-quant numerics shared by PTQ, QAT and kernel oracles (CPU/GPU, plain torch).

Submodules: :mod:`int_quant`, :mod:`fp8`, :mod:`fp4` (E2M1, NVFP4, MXFP4, SR), :mod:`packing`, :mod:`rng`, :mod:`spec`.
Imports are cheap (torch only).
"""
from __future__ import annotations

from forge.formats.fp4 import (
    E2M1_GRID,
    MXFP4Tensor,
    NVFP4Tensor,
    dequantize_mxfp4,
    dequantize_nvfp4,
    fake_quant_mxfp4,
    fake_quant_nvfp4,
    pack_fp4,
    quantize_mxfp4,
    quantize_nvfp4,
    round_e2m1,
    stochastic_round_fp4,
    unpack_fp4,
)
from forge.formats.fp8 import fake_quant_fp8, from_fp8, to_fp8
from forge.formats.int_quant import dequantize_int, fake_quant_int, quantize_int
from forge.formats.rng import counter_uniform
from forge.formats.spec import bits_per_element, fake_quantize, format_info

__all__ = [
    "E2M1_GRID",
    "MXFP4Tensor",
    "NVFP4Tensor",
    "bits_per_element",
    "counter_uniform",
    "dequantize_int",
    "dequantize_mxfp4",
    "dequantize_nvfp4",
    "fake_quant_fp8",
    "fake_quant_int",
    "fake_quant_mxfp4",
    "fake_quant_nvfp4",
    "fake_quantize",
    "format_info",
    "from_fp8",
    "pack_fp4",
    "quantize_int",
    "quantize_mxfp4",
    "quantize_nvfp4",
    "round_e2m1",
    "stochastic_round_fp4",
    "to_fp8",
    "unpack_fp4",
]
