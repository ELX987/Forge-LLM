"""/jobs : submit, list, inspect, cancel, delete, clone, logs, metrics, SSE stream."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response

from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import JobCreate
from forge.api.sse import sse_response, sse_stream
from forge.jobs.repo import TERMINAL
from forge.telemetry.logs import log_path, read_log, tail_lines

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", summary="List jobs (newest first)")
def list_jobs(
    response: Response,
    status: str | None = Query(None, description="queued | running | succeeded | failed | cancelled"),
    type: str | None = Query(None, description="Job type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    svc: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    rows = svc.jobs_repo.list_jobs(status=status, type_=type, limit=limit, offset=offset)
    response.headers["X-Total-Count"] = str(svc.jobs_repo.count({k: v for k, v in (("status", status), ("type", type)) if v}))
    return [svc.jobs_repo.public(r) for r in rows]


@router.post("", status_code=201, summary="Submit a job")
def create_job(body: JobCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return submit(
        svc, body.type, body.params, name=body.name, gpus=body.gpus, tags=body.tags, template=body.template, config_id=body.config_id
    )


@router.get("/{job_id}", summary="Job record (+ latest metric values)")
def get_job(job_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = svc.jobs_repo.require(job_id, "job")
    d = svc.jobs_repo.public(row)
    d["latest_metrics"] = svc.metrics.latest(job_id)
    return d


@router.post("/{job_id}/cancel", summary="Cancel a queued or running job (SIGTERM, then SIGKILL after the grace period)")
def cancel(job_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return svc.jobs_repo.public(svc.jobs.cancel(job_id))


@router.post("/{job_id}/clone", status_code=201, summary="Re-submit a job with the same spec")
def clone(job_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return svc.jobs_repo.public(svc.jobs.clone(job_id))


@router.delete("/{job_id}", status_code=204, summary="Delete a finished job (record, metrics and run directory)")
def delete(job_id: str, purge_files: bool = True, svc: Services = Depends(get_services)) -> Response:
    svc.jobs.delete(job_id, purge_files=purge_files)
    return Response(status_code=204)


@router.get("/{job_id}/logs", summary="Log text: last `tail` lines, or everything after byte `offset`")
def logs(
    job_id: str,
    tail: int = Query(200, ge=0, le=100000),
    offset: int | None = Query(None, ge=0),
    max_bytes: int = Query(1 << 20, ge=1, le=16 << 20),
    svc: Services = Depends(get_services),
) -> dict[str, Any]:
    row = svc.jobs_repo.require(job_id, "job")
    path = log_path(row["run_dir"] or "")
    if offset is not None:
        text, new = read_log(path, offset, max_bytes)
        return {"lines": text.splitlines(), "offset": new, "status": row["status"]}
    lines, end = tail_lines(path, tail)
    return {"lines": lines, "offset": end, "status": row["status"]}


@router.get("/{job_id}/metrics", summary="Metric series (downsampled to max_points)")
def metrics(
    job_id: str,
    keys: str | None = Query(None, description="Comma-separated metric keys (default: all)"),
    since_step: int | None = Query(None),
    max_points: int = Query(2000, ge=3, le=100000),
    rollup: str = Query("auto", description="auto | raw | lttb | minmax | mean | 10 | 100"),
    svc: Services = Depends(get_services),
) -> dict[str, Any]:
    svc.jobs_repo.require(job_id, "job")
    klist = [k.strip() for k in keys.split(",") if k.strip()] if keys else None
    svc.metrics.flush()
    seq = svc.metrics.max_seq(job_id)  # read BEFORE the query: a stream resumed from `seq` may repeat a row, never skip one
    res = svc.metrics.query(job_id, klist, since_step, max_points, rollup)
    res["seq"] = seq
    return res


@router.get("/{job_id}/stream", summary="SSE: events `status`, `progress`, `metric`, `log`, `end`")
async def stream(
    job_id: str,
    request: Request,
    since_seq: int | None = Query(None, ge=0, description="Metric sequence number to resume from (default: replay the last `replay` rows)."),
    replay: int = Query(5000, ge=0, le=200000, description="Metric rows to replay on connect."),
    log_tail: int = Query(100, ge=0, le=5000, description="Log lines to send on connect."),
    log_offset: int | None = Query(None, ge=0, description="Resume the log at this byte offset instead of sending the tail."),
    max_events: int | None = Query(None, ge=1, description="Close after N events (testing)."),
    svc: Services = Depends(get_services),
):
    row0 = svc.jobs_repo.require(job_id, "job")
    path = log_path(row0["run_dir"] or "")
    svc.metrics.flush()
    if since_seq is None and replay == 0:
        seq0 = svc.metrics.max_seq(job_id)
    elif since_seq is None:
        rows = svc.metrics.db.query(
            "SELECT id FROM metrics WHERE run_id=? ORDER BY id DESC LIMIT 1 OFFSET ?", (job_id, max(0, replay - 1))
        )
        seq0 = (rows[0]["id"] - 1) if rows else 0
    else:
        seq0 = since_seq
    state: dict[str, Any] = {"seq": seq0, "sig": None, "prog": None, "first_log": True, "drained_after_end": 0}
    if log_offset is not None:
        state["off"] = log_offset
        state["first_log"] = False
    else:
        lines, end = tail_lines(path, log_tail) if log_tail else ([], path.stat().st_size if Path(path).exists() else 0)
        state["off"] = end
        state["initial_lines"] = lines

    def step() -> tuple[list[tuple[str, Any]], bool]:
        events: list[tuple[str, Any]] = []
        row = svc.jobs_repo.get(job_id)
        if row is None:
            return [("end", {"status": "deleted"})], True
        pub = svc.jobs_repo.public(row)
        sig = (row["status"], row.get("error"), row.get("pid"), repr(row.get("outputs")))
        if sig != state["sig"]:
            state["sig"] = sig
            events.append(("status", pub))
        prog = (row.get("progress"), row.get("progress_msg"))
        if prog != state["prog"]:
            state["prog"] = prog
            events.append(("progress", {"progress": prog[0], "message": prog[1], "status": row["status"]}))
        svc.metrics.flush()
        mrows, state["seq"] = svc.metrics.since_seq(job_id, state["seq"], 5000)
        if mrows:
            pts: dict[str, list[list[Any]]] = {}
            for r in mrows:
                pts.setdefault(r["key"], []).append([r["step"], r["value"], r["ts"]])
            events.append(("metric", {"points": pts, "seq": state["seq"]}))
        lines: list[str] = []
        if state.get("initial_lines"):
            lines.extend(state.pop("initial_lines"))
        text, state["off"] = read_log(path, state["off"], 1 << 20)
        if text:
            lines.extend(text.splitlines())
        if lines:
            events.append(("log", {"lines": lines, "offset": state["off"]}))
        terminal = row["status"] in TERMINAL
        if terminal and not mrows and not text:
            # one extra drain pass after the terminal status so late log/metric flushes are not lost
            state["drained_after_end"] += 1
            if state["drained_after_end"] >= 2:
                events.append(("end", {"status": row["status"], "error": row.get("error"), "exit_code": row.get("exit_code")}))
                return events, True
        else:
            state["drained_after_end"] = 0
        return events, False

    return sse_response(sse_stream(request, step, interval=svc.stream_poll_s, max_events=max_events))
