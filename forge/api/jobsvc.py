"""Helpers shared by routers that create jobs."""
from __future__ import annotations

from typing import Any

from forge.api.common import deep_merge
from forge.api.deps import Services
from forge.core.errors import NotFound
from forge.core.schemas import JobSpec
from forge.jobs.registry import get_job_type
from forge.jobs.templates import list_templates


def submit(
    svc: Services,
    type_: str,
    params: dict[str, Any] | None = None,
    *,
    name: str = "",
    gpus: list[int] | None = None,
    tags: list[str] | None = None,
    template: str | None = None,
    config_id: str | None = None,
) -> dict[str, Any]:
    """Create a job (merging template/saved config under ``params``) and return its public JSON."""
    base: dict[str, Any] = {}
    if template:
        found = next((t for t in list_templates(get_job_type(type_)) if t["name"] == template), None)
        if found is None:
            raise NotFound(f"template {template!r} not found for job type {type_!r}")
        base = deep_merge(base, found["params"])
    if config_id:
        cfg = svc.configs.require(config_id, "config")
        if cfg["job_type"] != type_:
            raise NotFound(f"config {config_id!r} is for job type {cfg['job_type']!r}, not {type_!r}")
        base = deep_merge(base, cfg["params"])
    merged = deep_merge(base, params or {})
    spec = JobSpec(type=type_, name=name or "", params=merged, gpus=gpus, tags=tags or [])
    row = svc.jobs.submit(spec)
    return svc.jobs_repo.public(row)
