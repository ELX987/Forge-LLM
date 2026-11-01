"""Single import point for every kernel op: ``from forge.kernels.ops import w4a16_linear, paged_decode_attention, ...``.

Everything public in :mod:`~forge.kernels.ops_gemm`, :mod:`~forge.kernels.ops_attn`, :mod:`~forge.kernels.ops_moe` and
:mod:`~forge.kernels.ops_util` is re-exported here (names are resolved lazily on first attribute access, so importing
this module does not import the op modules, torch or triton).  ``available_ops()`` reports which documented names resolved.
"""
from __future__ import annotations

import importlib
from typing import Any

_MODULES = ("forge.kernels.ops_gemm", "forge.kernels.ops_attn", "forge.kernels.ops_moe", "forge.kernels.ops_util")

#: ops promised by docs/ARCHITECTURE.md 5.5 (checked by available_ops and the tests)
_DOCUMENTED = (
    "w4a16_linear", "w8a16_linear", "fp8_linear", "quantize_act_fp8", "int8_linear", "quantize_act_int8", "fp4_linear",
    "quantize_act_fp4", "make_packed_linear", "last_path", "ternary_linear", "quantize_act_ternary",
    "paged_decode_attention", "prefill_attention", "sparse_topk_attention", "tree_attention", "mla_decode_attention",
    "moe_route", "moe_permute", "moe_unpermute", "grouped_gemm",
    "hadamard_transform", "quantize_fp8_rowwise", "stochastic_round", "rmsnorm_quant", "rope_and_kv_write",
    "rejection_verify", "kv_compact", "topk_select",
)
_GEMM_OPS = _DOCUMENTED[:12]
_INDEX: dict[str, str] | None = None
_NOT_EXPORTED = {"bench_cases", "catalog_entries", "KERNELS"}


def _index() -> dict[str, str]:
    """name -> module for every public callable defined in (or listed in ``__all__`` of) the op modules."""
    global _INDEX
    if _INDEX is None:
        idx: dict[str, str] = {}
        for modname in _MODULES:
            try:
                mod = importlib.import_module(modname)
            except Exception:  # a broken family must not hide the others
                continue
            names = getattr(mod, "__all__", None)
            if names is None:
                names = [n for n in dir(mod) if not n.startswith("_")
                         and getattr(getattr(mod, n), "__module__", None) == mod.__name__]
            for n in names:
                if n not in _NOT_EXPORTED and not n.startswith("_"):
                    idx.setdefault(n, modname)
        _INDEX = idx
    return _INDEX


def __getattr__(name: str) -> Any:  # PEP 562 lazy re-export
    if name.startswith("__"):
        raise AttributeError(name)
    modname = _index().get(name)
    if modname is None:
        raise AttributeError(f"module 'forge.kernels.ops' has no attribute {name!r}")
    obj = getattr(importlib.import_module(modname), name)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_index()))


def available_ops() -> dict[str, bool]:
    """``{op name: importable on this host}`` for every documented op."""
    idx = _index()
    return {n: n in idx for n in _DOCUMENTED}


def all_ops() -> list[str]:
    """Every public op name re-exported by this module."""
    return sorted(_index())
