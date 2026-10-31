"""/schemas/job-types : schema-driven form definitions for every job type."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from forge.jobs.registry import get_job_type, list_job_types
from forge.jobs.templates import job_type_payload, list_templates

router = APIRouter(prefix="/schemas", tags=["schemas"])


@router.get("/job-types", summary="Config schema, defaults and templates for every job type")
def all_job_types(include_unavailable: bool = Query(True, description="Include types whose module is missing (available=false).")) -> list[dict[str, Any]]:
    out = [job_type_payload(s) for s in list_job_types()]
    return out if include_unavailable else [p for p in out if p["available"]]


@router.get("/job-types/{job_type}", summary="One job type")
def one_job_type(job_type: str, with_templates: bool = Query(False, description="Include template parameters, not just names.")) -> dict[str, Any]:
    spec = get_job_type(job_type)
    p = job_type_payload(spec)
    if with_templates:
        p["template_items"] = list_templates(spec)
    return p
