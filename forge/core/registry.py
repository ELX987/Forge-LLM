"""Tiny typed plug-in registry: ``@register("stage", "ptq")`` and ``get("stage", "ptq")``.

Kinds in use: ``stage`` (recipe stages), ``job_type``, ``kernel``, ``gate``, ``suite``, ``chip``.
Entries can also be registered lazily by import path (``register_lazy``) so listing is cheap.
"""
from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Callable
from typing import Any

from forge.core.errors import NotFound

log = logging.getLogger("forge.core.registry")

_LOCK = threading.RLock()
_ENTRIES: dict[str, dict[str, dict[str, Any]]] = {}  # kind -> name -> {"obj":..., "lazy":str|None, "meta":dict}


def register(kind: str, name: str, *, meta: dict | None = None) -> Callable[[Any], Any]:
    """Decorator registering ``obj`` under ``(kind, name)``. Re-registering a name replaces it (module reload safe)."""

    def deco(obj: Any) -> Any:
        register_obj(kind, name, obj, meta=meta)
        return obj

    return deco


def register_obj(kind: str, name: str, obj: Any, *, meta: dict | None = None) -> None:
    with _LOCK:
        bucket = _ENTRIES.setdefault(kind, {})
        if name in bucket and bucket[name].get("obj") is not obj:
            log.debug("registry: replacing %s/%s", kind, name)
        bucket[name] = {"obj": obj, "lazy": None, "meta": dict(meta or {})}


def register_lazy(kind: str, name: str, target: str, *, meta: dict | None = None) -> None:
    """Register ``"package.module:attr"`` imported on first :func:`get`."""
    if ":" not in target:
        raise ValueError("lazy target must look like 'package.module:attr'")
    with _LOCK:
        _ENTRIES.setdefault(kind, {})[name] = {"obj": None, "lazy": target, "meta": dict(meta or {})}


def unregister(kind: str, name: str) -> None:
    with _LOCK:
        _ENTRIES.get(kind, {}).pop(name, None)


def has(kind: str, name: str) -> bool:
    with _LOCK:
        return name in _ENTRIES.get(kind, {})


def get(kind: str, name: str) -> Any:
    """Return the registered object (resolving lazy targets). Raises :class:`NotFound`."""
    with _LOCK:
        entry = _ENTRIES.get(kind, {}).get(name)
        if entry is None:
            raise NotFound(
                f"no {kind} named {name!r} (available: {', '.join(sorted(_ENTRIES.get(kind, {}))) or 'none'})",
                details={"kind": kind, "name": name},
            )
        if entry["obj"] is None and entry["lazy"]:
            mod_name, attr = entry["lazy"].split(":", 1)
            obj: Any = importlib.import_module(mod_name)
            for part in attr.split("."):
                obj = getattr(obj, part)
            entry["obj"] = obj
        return entry["obj"]


def meta(kind: str, name: str) -> dict:
    """Metadata dict given at registration (copy). Raises :class:`NotFound`."""
    with _LOCK:
        entry = _ENTRIES.get(kind, {}).get(name)
        if entry is None:
            raise NotFound(f"no {kind} named {name!r}", details={"kind": kind, "name": name})
        return dict(entry["meta"])


def names(kind: str) -> list[str]:
    with _LOCK:
        return sorted(_ENTRIES.get(kind, {}))


def kinds() -> list[str]:
    with _LOCK:
        return sorted(k for k, v in _ENTRIES.items() if v)


def describe(kind: str) -> list[dict[str, Any]]:
    """``[{name, meta, doc, lazy}]`` for every entry of a kind (does not import lazy targets)."""
    out = []
    with _LOCK:
        for name in sorted(_ENTRIES.get(kind, {})):
            e = _ENTRIES[kind][name]
            obj = e["obj"]
            doc = (getattr(obj, "__doc__", None) or "").strip().split("\n\n")[0] if obj is not None else ""
            out.append({"name": name, "meta": dict(e["meta"]), "doc": doc, "lazy": e["lazy"]})
    return out
