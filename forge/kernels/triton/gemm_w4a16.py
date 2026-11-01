"""Triton W4A16 GEMM launcher for :class:`forge.kernels.layouts.PackedW4` in the ``portable_row`` layout.

Two kernel families sit behind :func:`w4a16_gemm_triton`:

* the **nibble-split** kernels of :mod:`forge.kernels.triton.gemm_w4_fast` (``impl='dot'`` tensor cores, ``impl='gemv'``
  fp32 FMA for M <= 8) -- the default; coalesced 16-byte weight loads, hoisted scale / zero loads, act-order handled
  by a one-time column sort;
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
from forge.kernels.layouts import KIND_PORTABLE_ROW, PackedW4, sort_columns_by_group

from .gemm_common import HAS_TRITON, is_compile_error
from .gemm_wq import wq_gemm

log = logging.getLogger("forge.kernels.triton.gemm_w4a16")

_FAST_BROKEN: set[tuple] = set()  # (sm, dtype, impl, n, k, group_size, rows bucket) whose nibble-split kernel failed to *compile* -> legacy
_LOGGED: set[tuple] = set()
_NO_SORT = object()


def forced_impl() -> str | None:
    """``FORGE_W4_IMPL`` = ``legacy`` | ``dot`` | ``gemv`` (kill switch / A-B switch); ``None`` = automatic."""
    v = os.environ.get("FORGE_W4_IMPL", "").strip().lower()
    return v if v in ("legacy", "dot", "gemv") else None


def _stamp(*ts: torch.Tensor | None) -> tuple:
    return tuple((t.data_ptr(), t._version, tuple(t.shape), str(t.device)) for t in ts if t is not None)


def group_sorted(packed: PackedW4) -> tuple[PackedW4, torch.Tensor] | None:
    """``(sorted weight, perm)`` of an act-order weight, computed once and cached in ``packed.meta`` (stamped with the
    identity / version of the source tensors); ``None`` if the groups are irregular."""
    stamp = _stamp(packed.qweight, packed.scales, packed.zeros, packed.g_idx)
    hit = packed.meta.get("_gsorted")
    if hit is None or hit[0] != stamp:
        res = sort_columns_by_group(packed)
        hit = packed.meta["_gsorted"] = (stamp, _NO_SORT if res is None else res)
    return None if hit[1] is _NO_SORT else hit[1]


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

    w, perm = packed, None
    if packed.g_idx is not None:
        srt = group_sorted(packed)
        if srt is None:  # irregular act-order groups: per-column g_idx kernel
            return _legacy().reshape(*lead, packed.n)
        w, perm = srt
    cfg = config
    if forced:
        cfg = {**(config or {}), "impl": forced}
    rows = int(x2.shape[0])
    fkey = (sm, x.dtype, cfg.get("impl") if cfg else None, packed.n, packed.k, packed.group_size, 1 if rows <= 1 else 1 << (rows - 1).bit_length())
    if fkey in _FAST_BROKEN and allow_fallback:
        return _legacy().reshape(*lead, packed.n)
    from .gemm_w4_fast import w4_fast_gemm

    try:
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


__all__ = ["w4a16_gemm_triton", "HAS_TRITON", "forced_impl", "group_sorted"]
