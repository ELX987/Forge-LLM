"""/checkpoints : discover checkpoints written by training jobs and export them as models."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query

from forge.api.common import lazy_module
from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import CheckpointExportRequest
from forge.core.errors import ConfigError, NotFound
from forge.core.paths import runs_dir

router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])
_SUBDIRS = ("checkpoints", "ckpt", "checkpoint")


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _check_run_id(run_id: str) -> str:
    """``run_id`` becomes a path component under runs/: refuse separators, ``..`` and absolute paths."""
    if not _RUN_ID_RE.match(run_id or "") or ".." in run_id:
        raise ConfigError(f"invalid run id {run_id!r}")
    return run_id


def scan_run(run_id: str) -> list[dict[str, Any]]:
    base = runs_dir() / _check_run_id(run_id)
    out: list[dict[str, Any]] = []
    for sub in _SUBDIRS:
        d = base / sub
        if not d.is_dir():
            continue
        for item in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime):
            if item.name.startswith("."):
                continue
            out.append(
                {
                    "id": f"{run_id}:{item.name}",
                    "run": run_id,
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size_bytes": _size(item),
                    "modified": item.stat().st_mtime,
                    "safetensors": item.is_dir() and any(item.glob("*.safetensors")),
                }
            )
    return out


def store_rows(run: str | None) -> list[dict[str, Any]]:
    """Delta-chained checkpoints in the artifact store (forge.store.checkpoints), if that module is present."""
    mod = lazy_module("forge.store.checkpoints")
    if mod is None or not hasattr(mod, "list_checkpoints"):
        return []
    try:
        from forge.jobs.repo import open_store

        rows = mod.list_checkpoints(open_store(), run, with_sizes=True)
    except Exception:
        return []
    out = []
    for r in rows:
        out.append(
            {
                "id": f"{r['run_id']}:{r['step']}", "run": r["run_id"], "name": str(r["step"]), "step": r["step"],
                "source": "store", "path": None, "is_dir": False, "size_bytes": r.get("stored_bytes"),
                "logical_bytes": r.get("logical_bytes"), "modified": r.get("created_at"), "tags": r.get("tags", []),
                "metric": r.get("metric"), "is_keyframe": r.get("is_keyframe"), "manifest_hash": r.get("manifest_hash"),
                "safetensors": False,
            }
        )
    return out


@router.get("", summary="Checkpoints: store-backed (delta chained) and plain directories under runs/<run>/checkpoints")
def list_checkpoints(run: str | None = Query(None, description="Run (job) id; default all runs"), svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    if run:
        _check_run_id(run)
    out: list[dict[str, Any]] = store_rows(run)
    if run:
        return out + [dict(c, source="fs") for c in scan_run(run)]
    root = runs_dir()
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.is_dir():
            out += [dict(c, source="fs") for c in scan_run(d.name)]
    return out


@router.post("/{ckpt_id}/export", status_code=201, summary="Export to a safetensors model dir and register it (job `checkpoint_export`)")
def export(ckpt_id: str, body: CheckpointExportRequest | None = None, svc: Services = Depends(get_services)) -> dict[str, Any]:
    run, _, name = ckpt_id.partition(":")
    _check_run_id(run)
    if not name:
        name = "latest"
    if not (runs_dir() / run).is_dir() and not store_rows(run):
        raise NotFound(f"run {run!r} not found")
    body = body or CheckpointExportRequest()
    params = {"run_id": run, "checkpoint": name, "register_model": body.register_model}
    if body.name:
        params["name"] = body.name
    if body.dtype:
        params["dtype"] = body.dtype
    return submit(svc, "checkpoint_export", params, name=f"export {ckpt_id}")
