"""Shared helpers for API routers: lazy module access, flexible calls, dict merging, service access."""
from __future__ import annotations

import importlib
import inspect
from typing import Any

from fastapi import Request

from forge.core.errors import ForgeError


class NotImplementedYet(ForgeError):
    """A feature depends on a module that is not installed/available (HTTP 501)."""

    code = "module_unavailable"
    http_status = 501


def lazy_module(name: str) -> Any | None:
    """Import ``name`` or return None (never raises)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def require_module(name: str, feature: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise NotImplementedYet(
            f"{feature} needs module '{name}', which is not available on this host ({type(e).__name__}: {e})",
            details={"module": name, "feature": feature},
        ) from e


def get_attr_any(mod: Any, *names: str) -> Any | None:
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    return None


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def call_flexible(fn: Any, aliases: dict[tuple[str, ...], Any], **extra: Any) -> Any:
    """Call ``fn`` mapping our canonical values onto its parameter names.

    ``aliases`` maps tuples of accepted parameter names to a value, e.g. ``{("model_cfg", "model"): cfg}``.
    Parameters we cannot fill are left to their defaults; a required one we cannot fill raises
    :class:`NotImplementedYet` (signature mismatch with the sibling module)."""
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    lookup: dict[str, Any] = {}
    for names, val in aliases.items():
        for n in names:
            lookup[n] = val
    lookup.update(extra)
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in lookup and lookup[name] is not None:
            kwargs[name] = lookup[name]
        elif p.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise NotImplementedYet(
            f"cannot call {getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', fn)}: no value for required parameter(s) {missing}",
            details={"missing": missing, "signature": str(sig)},
        )
    if has_var_kw:
        for k, v in extra.items():
            kwargs.setdefault(k, v)
    return fn(**kwargs)


def services(request: Request) -> Any:
    return request.app.state.services


def store_or_501() -> Any:
    """The artifact store, or an HTTP 501 when forge.store is unavailable."""
    from forge.jobs.repo import open_store

    try:
        return open_store()
    except Exception as e:
        raise NotImplementedYet(
            f"the artifact store is not available ({type(e).__name__}: {e})", details={"module": "forge.store"}
        ) from e
