"""Shared plumbing of the Triton GEMM kernels: guarded import, default launch configs, small helpers.

Minimum versions: ``triton >= 3.0`` (``tl.float8e4nv``, ``tl.dot`` fp8/int8), ``torch >= 2.1``.  The kernels use only
``tl.load/store`` with masks, ``tl.dot``, ``tl.sum/max`` and elementwise math; the one exotic API
(``tl.dot_scaled``) is isolated in :mod:`forge.kernels.triton.gemm_fp4` and version guarded.

Nothing here needs a GPU to *import*; ``HAS_TRITON`` is False on machines without Triton.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - environment dependent
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = True
    TRITON_VERSION = tuple(int(p) for p in str(triton.__version__).split("+")[0].split(".")[:3] if p.isdigit())
except Exception:  # ImportError, or a broken install
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False
    TRITON_VERSION = (0, 0, 0)


def require_triton() -> None:
    if not HAS_TRITON:
        from forge.core.errors import HardwareUnsupported

        raise HardwareUnsupported("Triton is not installed; use the reference/fallback path (FORGE_FORCE_REF=1)")


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def has_dot_scaled() -> bool:
    """True if the installed Triton exposes ``tl.dot_scaled`` (the block-scaled MMA entry point)."""
    return bool(HAS_TRITON and hasattr(tl, "dot_scaled"))


_FIT_MEMO: dict[tuple, dict[str, int]] = {}


def is_out_of_resources(e: BaseException) -> bool:
    """True for Triton's ``OutOfResources`` (shared memory / registers over the device limit) in any release."""
    return type(e).__name__ == "OutOfResources" or "out of resource" in str(e).lower()


def launch_fitting(key: tuple, cfg: dict[str, Any], launch: Any, *, min_tile_k: int = 16) -> Any:
    """Run ``launch(cfg)``; on ``OutOfResources`` retry with fewer pipeline stages, then a smaller ``tile_k``.

    Default configs are tuned for datacenter parts (164-227 KiB of shared memory); consumer parts (sm_86/89/120/121)
    only have ~99 KiB per block, where e.g. the 3-stage asymmetric W4 decode tile needs ~148 KiB.  Instead of failing
    (and silently dropping to the slow reference path) the launcher shrinks the schedule; the working schedule is
    memoised per ``key`` so later calls pay nothing.  Numerics do not depend on ``stages`` / ``tile_k`` except for
    the fp32 summation order.
    """
    memo = _FIT_MEMO.get(key)
    c = dict(cfg)
    if memo:
        c.update(memo)
    for _ in range(12):
        try:
            out = launch(c)
        except Exception as e:  # noqa: BLE001
            if not is_out_of_resources(e):
                raise
            if int(c.get("stages", 1)) > 1:
                c["stages"] = int(c["stages"]) - 1
            elif int(c["tile_k"]) // 2 >= min_tile_k:
                c["tile_k"] = int(c["tile_k"]) // 2
            else:
                raise
            continue
        if any(c.get(f) != cfg.get(f) for f in ("stages", "tile_k")):
            _FIT_MEMO[key] = {"stages": int(c["stages"]), "tile_k": int(c["tile_k"])}
        return out
    raise RuntimeError("could not fit the kernel schedule into the device's shared memory")


def default_wq_config(m: int, n: int, k: int, *, sm: int | None = 80, num_sms: int = 0, bits: int = 4) -> dict[str, Any]:
    """Heuristic launch config for the weight-only GEMMs, bucketed by M (skinny decode vs prefill).

    Conservative on purpose (the autotuner is the place to be clever): BLOCK_M >= 16 for ``tl.dot``,
    2 pipeline stages before Ampere (no ``cp.async``), split-K only for skinny M when the tile grid cannot fill the SMs.
    """
    from forge.kernels.layouts.layout import choose_split_k

    # sm<80: no cp.async; sm86/89/120/121 have <=~99 KiB shared memory per block -> 2 stages (launch_fitting shrinks further)
    stages = 2 if (sm is not None and (sm < 80 or sm in (86, 89, 120, 121))) else 3
    if m <= 16:
        cfg = dict(tile_m=16, tile_n=64, tile_k=128 if bits == 4 else 64, warps=4, stages=stages)
    elif m <= 32:
        cfg = dict(tile_m=32, tile_n=64, tile_k=64, warps=4, stages=stages)
    elif m <= 64:
        cfg = dict(tile_m=64, tile_n=64, tile_k=64, warps=4, stages=stages)
    else:
        cfg = dict(tile_m=128, tile_n=64, tile_k=64, warps=8, stages=stages)
    cfg["split_k"] = choose_split_k(m, n, k, cfg["tile_m"], cfg["tile_n"], cfg["tile_k"], num_sms) if m <= 64 else 1
    return cfg


def default_mm_config(m: int, n: int, k: int, *, sm: int | None = 89) -> dict[str, Any]:
    """Heuristic config for the FP8 / INT8 / FP4 GEMMs (no split-K: scales are applied per tile/epilogue)."""
    if m <= 16:
        return dict(tile_m=16, tile_n=128, tile_k=128, warps=4, stages=4, split_k=1)
    if m <= 64:
        return dict(tile_m=64, tile_n=128, tile_k=128, warps=4, stages=4, split_k=1)
    return dict(tile_m=128, tile_n=128, tile_k=128, warps=8, stages=3, split_k=1)
