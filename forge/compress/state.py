"""Process-local model registry for the recipe engine, plus persistence of stage outputs to the artifact store.

Recipe stages exchange *digests* (strings) through the engine namespace (digests are hashable, JSON-cacheable and look like store hashes).
The live ``nn.Module`` behind a digest lives in :data:`_LIVE` for the duration of a run:

* with an artifact store each model-producing stage writes its model directory to the store and the **store manifest hash is the digest**
  (so a cached stage output is only reused if the artifact still exists, and a later process can ``materialize`` it);
* without a store the digest is a pure function of ``(parent digest, stage, params)`` (deterministic) and caching is not used.

``begin_run`` / ``end_run`` scope the registry so models do not leak between jobs in one worker.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.core.context import RunContext
from forge.core.errors import ConfigError, NotFound
from forge.core.hashing import hash_json, is_digest

log = logging.getLogger("forge.compress.state")

_LOCK = threading.RLock()


@dataclass
class LiveModel:
    model: nn.Module
    tokenizer: Any = None
    meta: dict[str, Any] = field(default_factory=dict)  # licence, distill_allowed, name, params_b, artifact (store hash), kv config, reports...


_LIVE: dict[str, LiveModel] = {}
_RUN_KEYS: dict[str, list[str]] = {}
_OPTIONS: dict[str, Any] = {"persist_intermediate": True, "copy_below_bytes": 2 << 30, "run": None}


def begin_run(run_id: str, *, persist_intermediate: bool = True, copy_below_gb: float = 2.0) -> None:
    with _LOCK:
        _OPTIONS.update({"persist_intermediate": persist_intermediate, "copy_below_bytes": int(copy_below_gb * (1 << 30)), "run": run_id})
        _RUN_KEYS.setdefault(run_id, [])


def end_run(run_id: str) -> None:
    with _LOCK:
        for k in _RUN_KEYS.pop(run_id, []):
            _LIVE.pop(k, None)
        if _OPTIONS.get("run") == run_id:
            _OPTIONS["run"] = None


def options() -> dict[str, Any]:
    return _OPTIONS


def put_live(digest: str, model: nn.Module, tokenizer: Any = None, **meta: Any) -> str:
    with _LOCK:
        _LIVE[digest] = LiveModel(model, tokenizer, dict(meta))
        run = _OPTIONS.get("run")
        if run is not None:
            _RUN_KEYS.setdefault(run, []).append(digest)
    return digest


def drop_live(digest: str) -> None:
    with _LOCK:
        _LIVE.pop(digest, None)


def get_live(digest: str) -> LiveModel | None:
    with _LOCK:
        return _LIVE.get(digest)


def stage_digest(parent: str, stage: str, params: dict[str, Any]) -> str:
    """Deterministic digest of a stage output (used when no store is available)."""
    return hash_json({"parent": parent, "stage": stage, "params": params})


def resolve_model(digest: str, ctx: RunContext | None = None, *, dtype: str | None = None, device: str | None = None) -> LiveModel:
    """Live model for ``digest``; otherwise materialise it from the artifact store (or load a local directory) and cache it."""
    lm = get_live(digest)
    if lm is not None:
        return lm
    store = getattr(ctx, "store", None) if ctx is not None else None
    if store is not None and is_digest(digest) and store.has(digest):
        dest = Path(tempfile.mkdtemp(prefix="forge_resolve_"))
        try:
            store.materialize_artifact(digest, dest)
            model, tok = load_model_dir(dest, dtype=dtype, device=device)
        finally:
            shutil.rmtree(dest, ignore_errors=True)
        return LiveModel(model, tok, {"artifact": digest}) if put_live(digest, model, tok, artifact=digest) else None  # type: ignore[return-value]
    if isinstance(digest, str) and Path(digest).is_dir():
        model, tok = load_model_dir(Path(digest), dtype=dtype, device=device)
        put_live(digest, model, tok)
        return get_live(digest)  # type: ignore[return-value]
    if isinstance(digest, str) and not is_digest(digest) and "/" in digest and not digest.startswith("sha256:"):
        # a repo id / models-dir id used as a *reference* (e.g. the base model of a delta stage): load it like the job's own model
        from forge.models.loader import load_model

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, tok = load_model(digest, dtype=dtype or ("bfloat16" if dev.startswith("cuda") else "float32"), device=dev)
        put_live(digest, model, tok, source=digest)
        return get_live(digest)  # type: ignore[return-value]
    raise NotFound(
        f"model {str(digest)[:48]!r} is not available in this process (it was modified in place, evicted, or the stage cache refers to an "
        "artifact that was never stored). Re-run with use_cache=false or enable the artifact store.",
        details={"digest": digest},
    )


def load_model_dir(path: Path, dtype: str | None = None, device: str | None = None) -> tuple[nn.Module, Any]:
    """Load a model directory written by ``save_quantized`` / ``loader.save_model`` (compressed or dense).

    ``dtype=None``: the dtype recorded in the artifact (``forge_compress.json`` / ``config.json``), else float32 on CPU / bfloat16 on CUDA.
    ``device=None``: CUDA when available."""
    import json

    import torch

    from forge.compress.serialize import load_quantized

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None and not (path / "forge_compress.json").exists():
        try:
            dtype = json.loads((path / "config.json").read_text()).get("torch_dtype")
        except Exception:
            dtype = None
        dtype = dtype or ("bfloat16" if device.startswith("cuda") else "float32")

    if (path / "forge_compress.json").exists():
        model, meta = load_quantized(path, dtype=dtype, device=device)
        tok = None
        try:
            from forge.models.loader import load_tokenizer

            tok = load_tokenizer(path)
        except Exception:
            pass
        return model, tok
    from forge.models.loader import load_model

    return load_model(path, dtype=dtype, device=device)


def persist_model(
    ctx: RunContext,
    model: nn.Module,
    tokenizer: Any,
    *,
    name: str,
    kind: str,
    parents: list[str],
    recipe_hash: str | None,
    extra: dict[str, Any],
    report: dict[str, Any] | None = None,
    licence: str | None = None,
    force: bool = False,
) -> str | None:
    """Write ``model`` to ``<run_dir>/stages/<name>`` and ingest it into the artifact store. Returns the manifest hash (or None)."""
    store = getattr(ctx, "store", None)
    if store is None or not (force or _OPTIONS.get("persist_intermediate", True)):
        return None
    from forge.compress.serialize import save_quantized

    out = Path(ctx.run_dir) / "stages" / name
    try:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        save_quantized(model, out, report=report, tokenizer=tokenizer, extra=extra)
        kw: dict[str, Any] = {}
        if licence:
            kw["licence"] = licence
        h = store.put_tensor_dir(out, kind=kind, recipe_hash=recipe_hash, parents=[p for p in parents if is_digest(p) and store.has(p)], extra=extra, **kw)
        for p in parents:
            if is_digest(p) and p != h and store.has(p):
                try:
                    store.add_edge(h, p, recipe_hash)
                except Exception as e:  # lineage is best effort
                    log.debug("add_edge failed: %s", e)
        return h
    except Exception as e:
        log.warning("could not persist stage output %s to the store: %s", name, e)
        return None


def require_live(digest: str | None, ctx: RunContext, what: str = "model") -> LiveModel:
    if digest is None:
        raise ConfigError(f"no {what} in the recipe namespace (is the recipe input 'model' set?)")
    return resolve_model(digest, ctx)
