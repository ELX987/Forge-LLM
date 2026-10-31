"""/storage : store statistics, GC, tiers."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from forge.api.common import store_or_501
from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import StorageGcRequest
from forge.jobs.repo import to_jsonable

router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("/stats", summary="Logical vs unique bytes, dedup ratio, per-kind / per-tier breakdown")
def stats() -> Any:
    return to_jsonable(store_or_501().stats())


@router.post("/gc", summary="Garbage-collect the store. dry_run=true returns the report; false starts a `gc` job")
def gc(body: StorageGcRequest | None = None, svc: Services = Depends(get_services)) -> Any:
    body = body or StorageGcRequest()
    if body.dry_run:
        return {"dry_run": True, "report": to_jsonable(store_or_501().gc(dry_run=True))}
    store_or_501()  # fail early with 501 when the store is missing
    return {"dry_run": False, "job": submit(svc, "gc", {"dry_run": False}, name="store gc")}


@router.get("/tiers", summary="Per-tier object/byte counts and directories")
def tiers() -> Any:
    st = store_or_501()
    if hasattr(st, "tiers"):
        return to_jsonable(st.tiers())
    s = to_jsonable(st.stats())
    return s.get("per_tier", s) if isinstance(s, dict) else s
