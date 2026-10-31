"""/configs : saved configs CRUD, templates, validation."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Response

from forge.api.deps import Services, get_services
from forge.api.schemas import ConfigCreate, ConfigUpdate, ValidateRequest
from forge.core.errors import NotFound
from forge.core.schemas import JobSpec
from forge.core.util import new_id, now
from forge.jobs.manager import InvalidParams
from forge.jobs.registry import get_job_type, list_job_types
from forge.jobs.templates import list_templates

router = APIRouter(prefix="/configs", tags=["configs"])


def _validated(svc: Services, job_type: str, params: dict[str, Any]) -> dict[str, Any]:
    _jt, clean = svc.jobs.validate(JobSpec(type=job_type, params=params))
    return clean


@router.get("/templates", summary="Read-only templates shipped in configs/<type>/")
def templates(type: str | None = Query(None, description="Job type filter")) -> list[dict[str, Any]]:
    if type:
        return list_templates(get_job_type(type))
    out: list[dict[str, Any]] = []
    for s in list_job_types():
        out += list_templates(s)
    return out


@router.post("/validate", summary="Validate params against a job type's config model")
def validate(body: ValidateRequest, svc: Services = Depends(get_services)) -> dict[str, Any]:
    try:
        return {"valid": True, "errors": [], "params": _validated(svc, body.job_type, body.params)}
    except InvalidParams as e:
        return {"valid": False, "errors": e.details.get("errors", []), "params": None}


@router.get("", summary="List saved configs")
def list_configs(response: Response, job_type: str | None = None, svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    rows = svc.configs.list({"job_type": job_type}, order="updated_at DESC", limit=1000)
    response.headers["X-Total-Count"] = str(len(rows))
    return rows


@router.post("", status_code=201, summary="Save a config (validated)")
def create(body: ConfigCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    clean = _validated(svc, body.job_type, body.params)
    t = now()
    return svc.configs.insert(
        id=new_id("cfg"), job_type=body.job_type, name=body.name, description=body.description, params=clean, created_at=t, updated_at=t
    )


@router.get("/{config_id}")
def get(config_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return svc.configs.require(config_id, "config")


@router.put("/{config_id}", summary="Update a saved config (validated)")
def update(config_id: str, body: ConfigUpdate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = svc.configs.require(config_id, "config")
    fields: dict[str, Any] = {}
    if body.name is not None:
        fields["name"] = body.name
    if body.description is not None:
        fields["description"] = body.description
    if body.params is not None:
        fields["params"] = _validated(svc, row["job_type"], body.params)
    return svc.configs.update(config_id, **fields)


@router.delete("/{config_id}", status_code=204)
def delete(config_id: str, svc: Services = Depends(get_services)) -> Response:
    if not svc.configs.delete(config_id):
        raise NotFound(f"config {config_id!r} not found")
    return Response(status_code=204)
