"""Triton W4A16 GEMM launcher for :class:`forge.kernels.layouts.PackedW4` in the ``portable_row`` layout.

Three kernel families sit behind :func:`w4a16_gemm_triton` (default routing by M in :mod:`forge.kernels.w4_tuning`):

* the **nibble-split** kernels of :mod:`forge.kernels.triton.gemm_w4_fast` (``impl='dot'`` tensor cores, ``impl='gemv'``
  fp32 FMA for M <= 8) -- the default; coalesced 16-byte weight loads, hoisted scale / zero loads, act-order handled
  by a one-time column sort;
* the **dequantise + cuBLAS** path of :mod:`forge.kernels.triton.gemm_w4_dequant` (``impl='dequant_matmul'``) -- the default for
  large M (>= 32): a memory-bound Triton kernel expands the weight to bf16/fp16 in row chunks (<= 1 GiB temporary) and cuBLAS
  does the GEMM; act-order ``g_idx`` is dequantised directly (no column sort, no activation permutation);
* the **legacy** per-element-gather kernel of :mod:`forge.kernels.triton.gemm_wq` -- kept as the fallback
  (``config={'impl': 'legacy'}``, or ``FORGE_W4_IMPL=legacy`` to force it everywhere) and for irregular act-order groups.

If the nibble-split kernel fails to compile / launch (and the caller did not pin ``impl`` explicitly), the launcher logs
once and falls back to the legacy kernel for that process instead of dropping to the slow torch reference.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch

from forge.core.errors import ConfigError
from forge.kernels.layouts import KIND_PORTABLE_ROW, MISSING_SORT, PackedW4, cached_group_sort

from .gemm_common import HAS_TRITON, is_compile_error
from .gemm_wq import wq_gemm

log = logging.getLogger("forge.kernels.triton.gemm_w4a16")

_FAST_BROKEN: set[tuple] = set()  # (sm, dtype, impl, n, k, group_size, rows bucket) whose nibble-split kernel failed to *compile* -> legacy
_LOGGED: set[tuple] = set()


def forced_impl() -> str | None:
    """``FORGE_W4_IMPL`` = ``legacy`` | ``dot`` | ``gemv`` | ``dequant_matmul`` (kill switch / A-B switch); ``None`` = automatic."""
    v = os.environ.get("FORGE_W4_IMPL", "").strip().lower()
    return v if v in ("legacy", "dot", "gemv", "dequant_matmul") else None


def _capturing(t: torch.Tensor) -> bool:
    """True while a CUDA graph is being captured on ``t``'s device (no host sync; False on CPU tensors)."""
    try:
        return bool(t.is_cuda and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def group_sorted(packed: PackedW4, *, compute: bool = True) -> tuple[PackedW4, torch.Tensor] | None | Any:
    """``(sorted weight, perm)`` of an act-order weight from the cache in ``packed.meta`` (stamped with the identity / version
    of the source tensors); ``None`` if the groups are irregular, ``MISSING_SORT`` if the cache is cold and ``compute`` is false.

    A cold cache is filled here (one device sync) -- :func:`prepare_w4` / ``KernelW4A16Linear.prepare`` do that at load time so
    the launch path is sync-free, and the launcher passes ``compute=False`` inside CUDA graph capture."""
    return cached_group_sort(packed, compute=compute)


def prepare_w4(packed: PackedW4) -> PackedW4:
    """Pre-compute everything the launch path would otherwise derive lazily (the act-order column sort + permutation).  Call once
    after packing / loading / moving a ``portable_row`` weight, OUTSIDE CUDA graph capture; it synchronises the device.
    Returns ``packed``."""
    if packed.g_idx is not None and packed.layout.kind == KIND_PORTABLE_ROW:
        cached_group_sort(packed, compute=True)
    return packed


def w4a16_gemm_triton(
    x: torch.Tensor,
    packed: PackedW4,
    bias: torch.Tensor | None = None,
    *,
    config: dict[str, Any] | None = None,
    deterministic: bool = True,
    sm: int | None = None,
    num_sms: int = 0,
    allow_fallback: bool | None = None,
) -> torch.Tensor:
    """``Y = X @ dequant(W)^T + bias`` with ``x [..., K]`` (returns ``x.dtype``).  ``packed`` must be ``portable_row``
    (use :func:`forge.kernels.layouts.repack` first otherwise).

    ``allow_fallback``: fall back to the legacy kernel when the nibble-split kernel raises.  ``None`` = only when
    ``config`` does not pin an ``impl`` (so explicit candidates, e.g. in the autotuner, fail loudly)."""
    if packed.layout.kind != KIND_PORTABLE_ROW:
        raise ConfigError(f"Triton W4A16 kernel needs the portable_row layout, got {packed.layout_id}; repack() first")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    forced = forced_impl()
    impl = forced or (config or {}).get("impl")
    if allow_fallback is None:
        allow_fallback = impl is None
    legacy_cfg = None if (config and config.get("impl") not in (None, "legacy")) else config

    def _legacy() -> torch.Tensor:
        return wq_gemm(
            x2, packed.qweight, packed.scales, packed.zeros, packed.g_idx, bias,
            bits=4, n=packed.n, k=packed.k, group_size=packed.group_size, config=legacy_cfg, deterministic=deterministic,
            sm=sm, num_sms=num_sms,
        )

    if impl == "legacy" or not HAS_TRITON:
        return _legacy().reshape(*lead, packed.n)

    cfg = config
    if forced:
        cfg = {**(config or {}), "impl": forced}
    rows = int(x2.shape[0])
    from .gemm_w4_fast import _resolve, w4_fast_gemm

    gsz = packed.k if packed.group_size <= 0 else int(packed.group_size)
    eff = _resolve(rows, packed.n, packed.k, cfg, sm, num_sms, gsz)[0]["impl"]  # the impl the config policy will launch
    if eff == "legacy":
        return _legacy().reshape(*lead, packed.n)
    fkey = (sm, x.dtype, eff, packed.n, packed.k, packed.group_size, 1 if rows <= 1 else 1 << (rows - 1).bit_length())
    if fkey in _FAST_BROKEN and allow_fallback:
        return _legacy().reshape(*lead, packed.n)

    try:
        if eff == "dequant_matmul":  # act-order g_idx is dequantised directly: no column sort, no x permutation, no sync
            from .gemm_w4_dequant import w4_dequant_matmul

            y = w4_dequant_matmul(
                x2, packed.qweight, packed.scales, packed.zeros, bias, packed.g_idx, n=packed.n, k=packed.k,
                group_size=packed.group_size, config=cfg,
            )
        else:
            w, perm = packed, None
            if packed.g_idx is not None:
                srt = group_sorted(packed, compute=not _capturing(x2))
                if srt is None or srt is MISSING_SORT:  # irregular groups, or a cold cache under graph capture: per-column g_idx kernel
                    if srt is MISSING_SORT and ("capture", fkey) not in _LOGGED and len(_LOGGED) < 64:
                        _LOGGED.add(("capture", fkey))
                        log.warning("act-order W4 weight not prepared (call prepare_w4 / KernelW4A16Linear.prepare before CUDA graph "
                                    "capture): using the legacy g_idx kernel for %s", fkey)
                    return _legacy().reshape(*lead, packed.n)
                w, perm = srt
            y = w4_fast_gemm(
                x2, w.qweight, w.scales, w.zeros, bias, perm, n=packed.n, k=packed.k, group_size=packed.group_size,
                config=cfg, deterministic=deterministic, sm=sm, num_sms=num_sms,
            )
    except Exception as e:  # noqa: BLE001 - compile errors surface as many exception types
        if not allow_fallback:
            raise
        if is_compile_error(e):
            _FAST_BROKEN.add(fkey)  # deterministic for this (device, dtype, shape): do not re-compile on every call
            log.warning("W4A16 nibble-split kernel failed to compile (%s: %s); using the legacy kernel for %s", type(e).__name__, e, fkey)
        else:  # transient (OOM, ...): fall back for this call only and retry the fast kernel next time
            if (type(e).__name__, fkey) not in _LOGGED and len(_LOGGED) < 64:
                _LOGGED.add((type(e).__name__, fkey))
                log.warning("W4A16 nibble-split kernel failed (%s: %s); using the legacy kernel for this call only (%s)", type(e).__name__, e, fkey)
        return _legacy().reshape(*lead, packed.n)
    return y.reshape(*lead, packed.n)


__all__ = ["w4a16_gemm_triton", "HAS_TRITON", "forced_impl", "group_sorted", "prepare_w4"]
