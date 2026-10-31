"""Shared dispatch helpers for ``ops_attn`` / ``ops_moe`` / ``ops_util``.

A Triton kernel is used only when **all** of these hold: ``FORGE_FORCE_REF`` is not set, ``triton`` imports, the
tensors live on a CUDA device whose compute capability is high enough for the kernel (or ``TRITON_INTERPRET=1`` is set
and the tensors are on CPU - this is how the CPU unit tests exercise the kernels), and the shape is supported.
Otherwise the pure-torch reference path runs.  ``FORGE_KERNEL_DEBUG=1`` logs which path ran.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

log = logging.getLogger("forge.kernels")

_TRUE = ("1", "true", "yes", "on")


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in _TRUE


def force_ref() -> bool:
    """``FORGE_FORCE_REF=1`` forces every dispatcher onto its reference implementation."""
    return env_flag("FORGE_FORCE_REF")


def debug_enabled() -> bool:
    return env_flag("FORGE_KERNEL_DEBUG")


def interpret_mode() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


@lru_cache(maxsize=1)
def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except Exception:
        return None


def device_sm(device) -> int | None:
    """Compute capability as ``major*10+minor`` (e.g. 90) or ``None`` for non-CUDA devices."""
    try:
        import torch

        d = torch.device(device)
        if d.type != "cuda" or not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(d)
        return major * 10 + minor
    except Exception:
        return None


@lru_cache(maxsize=16)
def num_sms(index: int = 0) -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(index).multi_processor_count)
    except Exception:
        pass
    return 8


def use_triton(*tensors, min_sm: int = 70, op: str = "") -> bool:
    """Decide whether the Triton path may run for ``tensors`` (all must be on the same, capable device)."""
    if force_ref() or not triton_available():
        return False
    ts = [t for t in tensors if t is not None and hasattr(t, "device")]
    if not ts:
        return False
    dev = ts[0].device
    if any(t.device != dev for t in ts):
        return False
    if dev.type == "cpu":
        return interpret_mode()
    sm = device_sm(dev)
    return sm is not None and sm >= min_sm


def note_path(op: str, path: str) -> None:
    if debug_enabled():
        log.warning("forge.kernels: %s -> %s", op, path)


def next_pow2(n: int) -> int:
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def fp8_native(device) -> bool:
    """True when the GPU converts E4M3 natively (sm_89+) - used to pick the fast FP8 decode / store paths."""
    sm = device_sm(device)
    return sm is not None and sm >= 89 and not interpret_mode()


def bf16_ok(dtype, device) -> bool:
    """bf16 tensor-core dots need sm_80+ (fp16 works on sm_70+)."""
    import torch

    if dtype != torch.bfloat16:
        return True
    sm = device_sm(device)
    return sm is None or sm >= 80


def smem_limit_bytes(device=None) -> int:
    """Opt-in shared memory per block of the device (bytes); 99 KiB (sm_86/89/120 class) when unknown (CPU / old torch)."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device if device is not None else 0)
            for name in ("shared_memory_per_block_optin", "shared_memory_per_block"):
                v = getattr(props, name, None)
                if v:
                    return int(v)
    except Exception:
        pass
    return 99 * 1024
