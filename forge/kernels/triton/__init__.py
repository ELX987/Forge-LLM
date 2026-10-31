"""forge.kernels.triton: Triton kernels (GEMM family here; attention/moe/util by their owner).

Importing this package never imports ``triton``; import the ``gemm_*`` modules (or use ``forge.kernels.ops_gemm``).
"""
from __future__ import annotations
