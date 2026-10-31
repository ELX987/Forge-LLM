"""/models : registry of local / Hugging Face models."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Response

from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import ModelCreate, ModelDownloadRequest, ModelPatch, RegisterLocal
from forge.core.errors import ConfigError
from forge.core.paths import models_dir
from forge.core.util import new_id, now
from forge.jobs.modelinfo import inspect_model_dir, model_local_path, register_local_model

router = APIRouter(prefix="/models", tags=["models"])

_PERMISSIVE = ("apache", "mit", "bsd", "cc0", "unlicense", "isc")
_RESTRICTED = ("llama", "gemma", "openrail", "cc-by-nc", "research", "non-commercial", "proprietary")


def licence_hint(licence: str | None) -> dict[str, Any]:
    """Heuristic only: which way the licence name leans for distillation. Always read the actual licence."""
    if not licence:
        return {"distill": "unknown", "note": "no licence found - check the model card before distilling"}
    l = licence.lower()
    if any(k in l for k in _RESTRICTED):
        return {"distill": "check_terms", "note": "licence has use restrictions (name matches a restricted family) - check whether training on outputs is allowed"}
    if any(k in l for k in _PERMISSIVE):
        return {"distill": "likely_ok", "note": "permissive licence name - still verify the model card"}
    return {"distill": "unknown", "note": "unrecognised licence - check manually"}


def _public(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["local_path"] = model_local_path(row)
    row["downloaded"] = bool(row["local_path"] and Path(row["local_path"]).exists())
    if row.get("distill_allowed") is not None:
        row["distill_allowed"] = bool(row["distill_allowed"])
    return row


@router.get("", summary="List models")
def list_models(
    response: Response, source: str | None = None, q: str | None = Query(None, description="Substring of name / path"),
    limit: int = Query(500, ge=1, le=5000), svc: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    rows = svc.models.list({"source": source}, limit=limit)
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in r["name"].lower() or ql in (r["path_or_repo"] or "").lower()]
    response.headers["X-Total-Count"] = str(len(rows))
    return [_public(r) for r in rows]


@router.post("", status_code=201, summary="Add a model entry (HF repo id, local path or artifact hash)")
def create(body: ModelCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    if body.source == "local":
        return _public(register_local_model(
            body.path_or_repo, name=body.name, licence=body.licence, distill_allowed=body.distill_allowed, notes=body.notes, models=svc.models
        ))
    if body.source not in ("hf", "artifact"):
        raise ConfigError("source must be one of: hf, local, artifact")
    if body.source == "hf" and "/" not in body.path_or_repo:
        raise ConfigError("a Hugging Face repo id looks like `org/name`")
    existing = svc.models.find_by_repo(body.path_or_repo, body.revision) if body.source == "hf" else None
    if existing:
        return _public(existing)
    t = now()
    row = svc.models.insert(
        id=new_id("mdl"), name=body.name, source=body.source, path_or_repo=body.path_or_repo, revision=body.revision,
        licence=body.licence, distill_allowed=body.distill_allowed, status="remote" if body.source == "hf" else "ready",
        info={}, tags=body.tags, notes=body.notes, created_at=t, updated_at=t,
    )
    return _public(row)


@router.post("/register-local", status_code=201, summary="Inspect a local model directory and register it")
def register_local(body: RegisterLocal, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return _public(register_local_model(
        body.path, name=body.name, licence=body.licence, distill_allowed=body.distill_allowed, notes=body.notes, models=svc.models
    ))


@router.get("/{model_id}")
def get(model_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return _public(svc.models.require(model_id, "model"))


@router.patch("/{model_id}")
def patch(model_id: str, body: ModelPatch, svc: Services = Depends(get_services)) -> dict[str, Any]:
    svc.models.require(model_id, "model")
    fields = body.model_dump(exclude_unset=True)
    return _public(svc.models.update(model_id, **fields))


@router.delete("/{model_id}", status_code=204, summary="Remove from the registry (optionally delete downloaded files)")
def delete(model_id: str, delete_files: bool = False, svc: Services = Depends(get_services)) -> Response:
    row = svc.models.require(model_id, "model")
    if delete_files:
        p = model_local_path(row)
        if p:
            pp = Path(p).resolve()
            root = models_dir().resolve()
            if root in pp.parents:  # never delete outside the managed models dir
                shutil.rmtree(pp, ignore_errors=True)
            else:
                raise ConfigError(f"refusing to delete files outside the models directory ({root})")
    svc.models.delete(model_id)
    return Response(status_code=204)


@router.post("/{model_id}/download", status_code=201, summary="Download a Hugging Face model (starts a model_download job)")
def download(model_id: str, body: ModelDownloadRequest | None = None, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = svc.models.require(model_id, "model")
    if row["source"] != "hf":
        raise ConfigError(f"model {model_id} has source '{row['source']}'; only 'hf' models can be downloaded")
    body = body or ModelDownloadRequest()
    params: dict[str, Any] = {"repo_id": row["path_or_repo"], "model_id": model_id, "local_name": row["name"], "prefer_safetensors": body.prefer_safetensors}
    if row.get("revision") or body.revision:
        params["revision"] = body.revision or row["revision"]
    if body.allow_patterns:
        params["allow_patterns"] = body.allow_patterns
    if row.get("licence"):
        params["licence"] = row["licence"]
    job = submit(svc, "model_download", params, name=f"download {row['path_or_repo']}", gpus=body.gpus)
    svc.models.update(model_id, status="downloading")
    return job


@router.get("/{model_id}/inspect", summary="config.json summary: params, KV bytes/token, dtype, licence hint")
def inspect(
    model_id: str,
    remote: bool = Query(False, description="For not-yet-downloaded HF models: fetch only config.json from the Hub (network)."),
    svc: Services = Depends(get_services),
) -> dict[str, Any]:
    row = svc.models.require(model_id, "model")
    local = model_local_path(row)
    if local and Path(local).exists():
        info = inspect_model_dir(local)
        if info.get("params_b") and row.get("params_b") != info["params_b"]:
            svc.models.update(model_id, params_b=info["params_b"], info={**(row.get("info") or {}), **info})
    elif remote and row["source"] == "hf":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ConfigError("huggingface_hub is not installed") from e
        from forge.core.settings import get_settings

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            hf_hub_download(row["path_or_repo"], "config.json", revision=row.get("revision"), local_dir=td, token=get_settings().hf_token)
            info = inspect_model_dir_config_only(Path(td))
    else:
        raise ConfigError("model is not downloaded; download it first or pass ?remote=true to read config.json from the Hub")
    info["licence_hint"] = licence_hint(row.get("licence") or info.get("licence"))
    info["model_id"] = model_id
    return info


def inspect_model_dir_config_only(path: Path) -> dict[str, Any]:
    """Like inspect_model_dir but tolerant of a directory that only holds config.json."""
    return inspect_model_dir(path)
