"""/artifacts : content-addressed store browsing, lineage, pin/unpin, tiering."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

from forge.api.common import store_or_501
from forge.api.schemas import PinRequest, TierRequest
from forge.core.errors import ConfigError, NotFound
from forge.jobs.repo import to_jsonable

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("", summary="List artifacts in the store")
def list_artifacts(
    response: Response,
    kind: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    st = store_or_501()
    rows = to_jsonable(st.list_artifacts(kind=kind, limit=limit, offset=offset))
    try:
        response.headers["X-Total-Count"] = str(st.count_artifacts(kind))
    except Exception:
        pass
    return rows


@router.get("/{h}", summary="Artifact info (+ manifest when it is a manifest)")
def get_artifact(h: str, manifest: bool = True) -> dict[str, Any]:
    st = store_or_501()
    if not st.has(h):
        raise NotFound(f"artifact {h!r} not found")
    out: dict[str, Any] = {"hash": h}
    if hasattr(st, "info"):
        out["info"] = to_jsonable(st.info(h))
    try:
        out["size"] = st.size(h)
        out["stored_size"] = st.stored_size(h)
    except Exception:
        pass
    try:
        out["tier"] = st.tier_of(h)
        out["pinned"] = st.is_pinned(h) if hasattr(st, "is_pinned") else None
    except Exception:
        pass
    if manifest:
        try:
            out["manifest"] = to_jsonable(st.get_manifest(h))
        except Exception:
            out["manifest"] = None  # plain blob / chunk list, not an artifact manifest
    return out


@router.get("/{h}/lineage", summary="Lineage graph {nodes, edges}")
def lineage(h: str, depth: int | None = Query(None, ge=1, le=50), direction: str = Query("both", pattern="^(both|up|down)$")) -> Any:
    st = store_or_501()
    if not st.has(h):
        raise NotFound(f"artifact {h!r} not found")
    try:
        return to_jsonable(st.lineage(h, depth, direction))
    except TypeError:
        return to_jsonable(st.lineage(h, depth))


@router.post("/{h}/pin")
def pin(h: str, body: PinRequest | None = None) -> dict[str, Any]:
    st = store_or_501()
    if not st.has(h):
        raise NotFound(f"artifact {h!r} not found")
    st.pin(h, (body or PinRequest()).reason)
    return {"hash": h, "pinned": True}


@router.post("/{h}/unpin")
def unpin(h: str) -> dict[str, Any]:
    st = store_or_501()
    if not st.has(h):
        raise NotFound(f"artifact {h!r} not found")
    st.unpin(h)
    return {"hash": h, "pinned": False}


@router.post("/{h}/tier", summary="Move an artifact between hot / warm / cold tiers")
def tier(h: str, body: TierRequest) -> dict[str, Any]:
    st = store_or_501()
    if not st.has(h):
        raise NotFound(f"artifact {h!r} not found")
    to = body.to.lower()
    if to not in ("hot", "warm", "cold"):
        raise ConfigError("tier must be hot, warm or cold")
    moved = st.promote(h, "hot") if to == "hot" else st.demote(h, to)
    return {"hash": h, "tier": to, "moved": to_jsonable(moved)}
