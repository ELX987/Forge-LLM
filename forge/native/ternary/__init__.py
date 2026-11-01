"""Optional native CUDA tier for ternary kernels (ADR 0007).  Never required: ``available()`` is False unless ``forge_ternary_C`` is built.

The extension is the tern158 v0.1 CUDA library (mma.sync / wmma / dp4a / bf16+cuBLAS GEMM backends with start-up self-validation, quantisers,
fused norm+quant, loss and optimiser kernels) renamed from ``tern158_C``.  On its first hardware run (RTX PRO 6000 Blackwell, sm_120, CUDA 13.4, as tern158)
23/23 selftests passed and the mma path reached 346-411 TOPS; none of that has been reproduced through Forge.  It is NOT wired into the Forge dispatcher
yet: ``forge.kernels.ops_gemm.ternary_linear`` uses Triton or the torch oracle.
"""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("forge.native.ternary")

_MOD: Any = None
_ERR: str | None = None
_TRIED = False
ROOT = Path(__file__).resolve().parent


def load(force: bool = False) -> Any:
    """The compiled ``forge_ternary_C`` module, or None (see :func:`error`)."""
    global _MOD, _ERR, _TRIED
    if _TRIED and not force:
        return _MOD
    _TRIED = True
    if os.environ.get("FORGE_FORCE_REF") == "1":
        _MOD, _ERR = None, "FORGE_FORCE_REF=1"
        return None
    try:
        _MOD, _ERR = importlib.import_module("forge_ternary_C"), None
    except Exception as e:  # ImportError, or a broken build
        _MOD, _ERR = None, f"{type(e).__name__}: {e}"
    return _MOD


def available() -> bool:
    return load() is not None


def error() -> str | None:
    load()
    return _ERR


def build_jit(verbose: bool = False) -> Any:
    """JIT-build the extension with ``torch.utils.cpp_extension.load`` (needs nvcc and a CUDA torch); returns the module."""
    import subprocess
    import sys

    from torch.utils.cpp_extension import load as _load

    out = subprocess.run([sys.executable, str(ROOT / "tools" / "detect_arch.py"), "--gencode"], capture_output=True, text=True).stdout.split()
    gencode = [out[i + j] for i in range(0, len(out) - 1, 2) for j in (0, 1)]
    srcs = sorted(str(p) for p in (ROOT / "kernel" / "src").glob("*.cu")) + [str(ROOT / "kernel" / "torch_ext" / "bindings.cpp")]
    global _MOD, _ERR, _TRIED
    _MOD = _load(name="forge_ternary_C", sources=srcs, extra_include_paths=[str(ROOT / "kernel" / "include")], extra_ldflags=["-lcublas"],
                 extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo", *gencode], verbose=verbose)
    _ERR, _TRIED = None, True
    return _MOD


__all__ = ["ROOT", "available", "build_jit", "error", "load"]
