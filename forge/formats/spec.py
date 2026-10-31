"""Format registry: name -> fake-quant callable and storage cost (bits per element), for planners, QAT and PTQ.

Names: ``intN`` (e.g. ``int4``; needs ``group_size``), ``fp8_e4m3``, ``fp8_e5m2``, ``nvfp4``, ``mxfp4``, ``bf16``/``fp16``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from forge.core.errors import ConfigError
from forge.formats import fp4 as _fp4
from forge.formats import fp8 as _fp8
from forge.formats import int_quant as _iq


@dataclass(frozen=True)
class FormatInfo:
    name: str
    kind: str  # int | fp8 | fp4_block | float
    element_bits: float
    block: int | None
    scale_bits: float
    description: str

    @property
    def bits_per_element(self) -> float:
        return self.element_bits + (self.scale_bits / self.block if self.block else 0.0)


def format_info(name: str, group_size: int | None = None, scale_bits: float = 16.0, zero_bits: float = 0.0) -> FormatInfo:
    n = name.lower()
    if n in ("bf16", "fp16", "float16", "bfloat16"):
        return FormatInfo(n, "float", 16, None, 0, "16-bit float")
    if n in ("fp8_e4m3", "fp8_e5m2", "e4m3", "e5m2"):
        return FormatInfo(n, "fp8", 8, None, 0, "FP8 with a per-tensor/channel scale (scale overhead negligible)")
    if n == "nvfp4":
        return FormatInfo(n, "fp4_block", 4, 16, 8, "E2M1 elements, 16-block E4M3 scale + FP32 tensor scale (4.5 b/elem)")
    if n == "mxfp4":
        return FormatInfo(n, "fp4_block", 4, 32, 8, "E2M1 elements, 32-block E8M0 scale (4.25 b/elem)")
    m = re.fullmatch(r"(?:int|w)(\d+)", n)
    if m:
        bits = int(m.group(1))
        blk = group_size if group_size and group_size > 0 else None
        return FormatInfo(n, "int", bits, blk, scale_bits + zero_bits, f"{bits}-bit integer, group {blk or 'per-row'}")
    raise ConfigError(f"unknown format {name!r}")


def bits_per_element(name: str, group_size: int | None = None, scale_bits: float = 16.0, zero_bits: float = 0.0) -> float:
    """Effective storage bits per weight, e.g. ``nvfp4`` 4.5, ``mxfp4`` 4.25, ``int4`` g128 with fp16 scale 4.125 (Plan App. D)."""
    return format_info(name, group_size, scale_bits, zero_bits).bits_per_element


def fake_quantize(x: torch.Tensor, fmt: str, **kw: Any) -> torch.Tensor:
    """Quantise-dequantise ``x`` in format ``fmt``. Keyword args are forwarded to the format's ``fake_quant_*`` function."""
    n = fmt.lower()
    if n in ("bf16", "bfloat16"):
        return x.to(torch.bfloat16).to(x.dtype)
    if n in ("fp16", "float16"):
        return x.to(torch.float16).to(x.dtype)
    if n in ("fp8_e4m3", "e4m3"):
        return _fp8.fake_quant_fp8(x, "e4m3", **kw)
    if n in ("fp8_e5m2", "e5m2"):
        return _fp8.fake_quant_fp8(x, "e5m2", **kw)
    if n == "nvfp4":
        return _fp4.fake_quant_nvfp4(x, **kw)
    if n == "mxfp4":
        return _fp4.fake_quant_mxfp4(x, **kw)
    m = re.fullmatch(r"(?:int|w)(\d+)", n)
    if m:
        kw.setdefault("group_size", 128)
        return _iq.fake_quant_int(x, int(m.group(1)), **kw)
    raise ConfigError(f"unknown format {fmt!r}")
