"""Re-export of the W8A16 oracle (lives next to the W4A16 one because both share ``_linear_fp32``)."""
from __future__ import annotations

from .gemm_w4a16 import w8a16_gemm_ref

__all__ = ["w8a16_gemm_ref"]
