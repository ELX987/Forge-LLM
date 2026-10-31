"""/events : global SSE stream for the dashboard (job changes + GPU telemetry)."""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from forge.api.deps import Services, get_services
from forge.api.sse import sse_response, sse_stream

router = APIRouter(tags=["events"])


@router.get("/events", summary="SSE: `hello`, `job` (any job row change), `gpus` (new telemetry sample)")
async def events(
    request: Request,
    interval: float = Query(1.0, ge=0.1, le=30),
    since: float | None = Query(None, description="Only report job changes after this epoch time (default: now)."),
    max_events: int | None = Query(None, ge=1, description="Close after N events (testing)."),
    svc: Services = Depends(get_services),
):
    state: dict[str, Any] = {"last": since if since is not None else time.time() - 0.001, "gpu_ts": None}

    def step() -> tuple[list[tuple[str, Any]], bool]:
        out: list[tuple[str, Any]] = []
        rows = svc.jobs_repo.db.query(
            "SELECT * FROM jobs WHERE updated_at > ? ORDER BY updated_at ASC LIMIT 200", (state["last"],)
        )
        for r in rows:
            row = svc.jobs_repo._decode(dict(r))
            out.append(("job", svc.jobs_repo.public(row)))  # type: ignore[arg-type]
            state["last"] = max(state["last"], r["updated_at"])
        latest = svc.monitor.latest()
        if latest.get("ts") != state["gpu_ts"]:
            state["gpu_ts"] = latest.get("ts")
            out.append(("gpus", {"gpus": latest.get("gpus", []), "host": latest.get("host", {}), "ts": latest.get("ts"), "busy": sorted(svc.jobs.busy_gpus())}))
        return out, False

    hello = ("hello", {"jobs": svc.jobs_repo.counts(), "server_time": time.time()})
    return sse_response(sse_stream(request, step, interval=interval, max_events=max_events, hello=hello))
