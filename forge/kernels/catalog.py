"""Kernel catalog for the UI / API: ``list_kernels()`` -> one row per kernel spec with availability on this host.

Rows are built from ``kernels/specs/*.yaml`` (every family's specs, whoever wrote them) plus entries registered with
:func:`register_catalog_entry` or returned by an optional ``catalog_entries()`` function in ``ops_attn`` / ``ops_moe``
/ ``ops_util``.  Availability separates the *native* path (Triton on this GPU) from the always-present reference
fallback, and carries the honest validation status ("implemented, not validated on hardware" until a GPU test run
says otherwise).
"""
from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

from forge.core import registry
from forge.kernels.layouts.chip import ChipInfo, chip_info
from forge.kernels.spec import KernelSpec, list_specs, load_spec

log = logging.getLogger("forge.kernels.catalog")

_LOCK = threading.Lock()
_EXTRA: dict[str, dict[str, Any]] = {}

#: kernels whose native path needs more than "triton + sm >= min_sm" (name -> (ChipInfo attr, human reason))
_NEEDS_FEATURE = {
    "fp8_gemm": ("has_fp8", "native FP8 tensor cores (sm_89+)"),
    "ternary_w158a8_gemm": ("has_int8_fast", "fast INT8 tensor cores (sm_75+, not throttled: B300/GB300 sm_103 excluded)"),
}

#: honest status of the shipped GEMM kernels (updated by whoever runs tests_gpu on real hardware)
_GEMM_VALIDATION = {
    "triton": "implemented; logic checked in the Triton CPU interpreter",
    "gpu": "not validated on NVIDIA hardware (run scripts/run_gpu_tests.sh on the GPU host)",
}


def register_catalog_entry(entry: dict[str, Any]) -> None:
    """Add/replace a catalog row (``name`` required).  For kernel families whose specs carry extra information."""
    if "name" not in entry:
        raise ValueError("catalog entry needs a 'name'")
    with _LOCK:
        _EXTRA[entry["name"]] = dict(entry)


def _triton_state() -> tuple[bool, str | None]:
    try:
        from forge.kernels.triton import gemm_common

        return bool(gemm_common.HAS_TRITON), ".".join(map(str, gemm_common.TRITON_VERSION)) if gemm_common.HAS_TRITON else None
    except Exception:
        return False, None


def _spec_row(spec: KernelSpec, chip: ChipInfo, has_triton: bool, triton_version: str | None) -> dict[str, Any]:
    reasons: list[str] = []
    native = True
    if not has_triton:
        native = False
        reasons.append("triton is not installed")
    if not chip.is_cuda:
        native = False
        reasons.append("no CUDA device")
    elif chip.sm is not None and chip.sm < spec.min_sm:
        native = False
        reasons.append(f"needs sm_{spec.min_sm}+, device is sm_{chip.sm}")
    feat = _NEEDS_FEATURE.get(spec.kernel)
    if feat and chip.is_cuda and not getattr(chip, feat[0], False):
        native = False
        reasons.append(f"device lacks {feat[1]}; falls back to dequantise + 16-bit matmul")
    if spec.kernel == "fp4_block_gemm" and chip.is_cuda and not chip.has_fp4_block_scaled:
        reasons.append("no block-scaled FP4 MMA: only the dequantise-to-16-bit kernel applies (weight-only)")
    row: dict[str, Any] = {
        "name": spec.kernel,
        "family": spec.family,
        "op": spec.op,
        "description": spec.description,
        "spec": spec.model_dump(mode="json"),
        "oracle": spec.oracle,
        "available": native,  # native Triton path usable on this host
        "reference_available": True,  # the pure-torch oracle/fallback always runs
        "reasons": reasons,
        "min_sm": spec.min_sm,
        "experimental": spec.experimental,
        "triton_version": triton_version,
        "dtypes": spec.dtypes,
        "tolerance": spec.tolerance.model_dump(),
        "kpi": spec.kpi.model_dump(),
        "notes": spec.notes,
    }
    if spec.family == "gemm":
        row["validation"] = dict(_GEMM_VALIDATION)
    else:
        row["validation"] = {"triton": "see docs/KERNELS.md", "gpu": "not validated on NVIDIA hardware unless stated"}
    return row


def _family_row(name: str, meta: dict[str, Any], modname: str, chip: ChipInfo, has_triton: bool,
                triton_version: str | None) -> dict[str, Any]:
    """Row for a kernel of the attention / MoE / utility families, built from the ``KERNELS`` dict of its ops module."""
    min_sm = int(meta.get("min_sm", 70))
    reasons: list[str] = []
    native = True
    if not has_triton:
        native = False
        reasons.append("triton is not installed")
    if not chip.is_cuda:
        native = False
        reasons.append("no CUDA device")
    elif chip.sm is not None and chip.sm < min_sm:
        native = False
        reasons.append(f"needs sm_{min_sm}+, device is sm_{chip.sm}")
    return {
        "name": name, "family": meta.get("family", "?"), "op": name, "description": meta.get("description", ""),
        "oracle": meta.get("ref"), "triton_module": meta.get("triton"), "ops_module": modname,
        "available": native, "reference_available": True, "reasons": reasons, "min_sm": min_sm,
        "experimental": bool(meta.get("experimental", False)), "triton_version": triton_version,
        "validation": {"triton": "compiled offline / checked in the Triton CPU interpreter (see docs/KERNELS.md)",
                       "gpu": "not validated on NVIDIA hardware (run scripts/run_gpu_tests.sh on the GPU host)"},
    }


def _tuned_counts(chip: ChipInfo) -> dict[str, int]:
    try:
        from forge.kernels.autotune import AutotuneCache, default_cache_path

        counts: dict[str, int] = {}
        for e in AutotuneCache(default_cache_path(chip.fingerprint)).entries():
            op = e.get("fields", {}).get("op", "?")
            counts[op] = counts.get(op, 0) + 1
        return counts
    except Exception:
        return {}


def list_kernels(chip: Any = None) -> list[dict[str, Any]]:
    """Catalog rows ``{name, family, spec, available, reasons, min_sm, ...}`` for the UI (sorted by family, name)."""
    c = chip_info(chip)
    has_triton, tv = _triton_state()
    rows: dict[str, dict[str, Any]] = {}
    for name in list_specs():
        try:
            spec = load_spec(name)
        except Exception as e:  # a broken spec of another owner must not hide the others
            rows[name] = {"name": name, "family": "?", "available": False, "reference_available": False,
                          "reasons": [f"invalid spec: {e}"], "min_sm": None}
            continue
        rows[spec.kernel] = _spec_row(spec, c, has_triton, tv)
    for modname in ("forge.kernels.ops_attn", "forge.kernels.ops_moe", "forge.kernels.ops_util", "forge.kernels.ops_gemm"):
        try:
            fn = getattr(importlib.import_module(modname), "catalog_entries", None)
            if fn:
                for e in fn():
                    rows.setdefault(e["name"], {}).update(e)
        except Exception as ex:
            log.debug("catalog_entries from %s unavailable: %s", modname, ex)
    for modname in ("forge.kernels.ops_attn", "forge.kernels.ops_moe", "forge.kernels.ops_util"):
        try:
            kerns = getattr(importlib.import_module(modname), "KERNELS", None) or {}
        except Exception as ex:
            log.debug("KERNELS from %s unavailable: %s", modname, ex)
            continue
        for kname, meta in kerns.items():
            rows.setdefault(kname, _family_row(kname, meta, modname, c, has_triton, tv))
    with _LOCK:
        for name, e in _EXTRA.items():
            rows.setdefault(name, {}).update(e)
    tuned = _tuned_counts(c)
    out = []
    for name, row in rows.items():
        row["autotune_entries"] = tuned.get(name, 0)
        row["chip"] = c.fingerprint
        try:
            registry.register_obj("kernel", name, row, meta={"family": row.get("family"), "min_sm": row.get("min_sm")})
        except Exception:
            pass
        out.append(row)
    return sorted(out, key=lambda r: (str(r.get("family")), r["name"]))


def get_kernel(name: str, chip: Any = None) -> dict[str, Any]:
    from forge.core.errors import NotFound

    for r in list_kernels(chip):
        if r["name"] == name:
            return r
    raise NotFound(f"no kernel named {name!r}")


__all__ = ["get_kernel", "list_kernels", "register_catalog_entry"]
