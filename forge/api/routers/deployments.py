"""/deployments : start `serve` jobs, discover the port, proxy chat (SSE) and stats to the engine."""
from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from forge.api.common import deep_merge
from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import DeploymentCreate
from forge.core.errors import ConfigError, ForgeError
from forge.core.settings import get_settings
from forge.core.util import new_id, now
from forge.jobs.modelinfo import model_local_path
from forge.jobs.registry import get_job_type
from forge.jobs.templates import list_templates

router = APIRouter(prefix="/deployments", tags=["deployments"])

_MODEL_FIELDS = ("model", "model_path", "artifact", "model_id")


class BadGateway(ForgeError):
    code = "engine_unreachable"
    http_status = 502


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(svc: Services, preferred: int | None = None) -> int:
    used = {r["port"] for r in svc.deployments.list(limit=1000) if r.get("port") and r["status"] in ("created", "starting", "running")}
    if preferred and preferred not in used and _port_free(preferred):
        return preferred
    base = get_settings().serve_base_port
    for p in range(base, base + 200):
        if p not in used and _port_free(p):
            return p
    raise ConfigError(f"no free port in {base}..{base + 199}")


def _client(svc: Services, timeout: float | None = None) -> httpx.AsyncClient:
    transport = svc.extras.get("engine_transport")
    return httpx.AsyncClient(transport=transport, timeout=timeout or 30.0, base_url="http://engine")


def _base(row: dict[str, Any]) -> str:
    if not row.get("port"):
        raise ConfigError("deployment has no port yet (the serve job has not started)")
    return f"http://127.0.0.1:{row['port']}"


def sync_status(svc: Services, row: dict[str, Any]) -> dict[str, Any]:
    """Derive the deployment status from its serve job (no network)."""
    if not row.get("job_id"):
        return row
    job = svc.jobs_repo.get(row["job_id"])
    if job is None:
        new = "stopped"
    else:
        new = {"queued": "starting", "running": "starting", "succeeded": "stopped", "cancelled": "stopped", "failed": "failed"}[job["status"]]
        port = (job.get("outputs") or {}).get("port")
        if isinstance(port, int) and port != row.get("port"):
            row = svc.deployments.update(row["id"], port=port)
        outs = job.get("outputs") or {}
        if job["status"] == "running" and (outs.get("ready") is True or outs.get("url")):  # forge.serve.run sets `url` once listening
            new = "running"
    if new != row["status"]:
        row = svc.deployments.update(row["id"], status=new)
    return row


def _public(svc: Services, row: dict[str, Any]) -> dict[str, Any]:
    row = sync_status(svc, row)
    out = dict(row)
    job = svc.jobs_repo.get(row["job_id"]) if row.get("job_id") else None
    out["job_status"] = job["status"] if job else None
    out["job_progress"] = job["progress"] if job else None
    out["url"] = f"http://127.0.0.1:{row['port']}" if row.get("port") else None
    sc = dict(row.get("serve_config") or {})
    out["api_key"] = "set" if sc.get("api_key") else None
    if sc.get("api_key"):
        sc["api_key"] = "***"  # never echo the engine key back to the browser
    out["serve_config"] = sc
    return out


async def _probe(svc: Services, row: dict[str, Any]) -> dict[str, Any]:
    """sync_status + a short /health probe for deployments whose job runs but is not yet marked running."""
    row = sync_status(svc, row)
    if row["status"] == "starting" and row.get("port"):
        job = svc.jobs_repo.get(row["job_id"]) if row.get("job_id") else None
        if job and job["status"] == "running":
            async with _client(svc, 0.5) as c:
                try:
                    r = await c.get(_base(row) + "/health")
                    if r.status_code == 200:
                        row = svc.deployments.update(row["id"], status="running")
                except httpx.HTTPError:
                    pass
    return row


@router.get("", summary="List deployments")
async def list_deployments(response: Response, svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    rows = [_public(svc, await _probe(svc, r)) for r in svc.deployments.list(limit=500)]
    response.headers["X-Total-Count"] = str(len(rows))
    return rows


def _preflight_servable(model_value: str) -> None:
    """Fail the request (instead of a crashed serve job) when a local model's architecture is not servable by the paged runner."""
    import json

    cfg_path = Path(str(model_value)).expanduser() / "config.json"
    try:
        if not cfg_path.is_file():
            return
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        from forge.models import ModelConfig
        from forge.serve.runner import check_servable

        cfg = ModelConfig.from_hf_config(raw)
    except Exception:  # noqa: BLE001 - unreadable / unparseable configs are reported by the serve job itself
        return
    check_servable(cfg)


@router.post("", status_code=201, summary="Start serving a model (creates a `serve` job)")
def create(body: DeploymentCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    jt = get_job_type("serve")
    cls = jt.config_model()  # 501 when forge.serve is missing
    ref = body.artifact_ref
    row = svc.models.get(ref)
    model_value = ref
    if row:
        local = model_local_path(row)
        model_value = local or row["path_or_repo"]
    elif Path(ref).expanduser().is_dir():
        model_value = str(Path(ref).expanduser().resolve())
    _preflight_servable(model_value)
    params: dict[str, Any] = {}
    if body.template:
        t = next((t for t in list_templates(jt) if t["name"] == body.template), None)
        if t is None:
            raise ConfigError(f"serve template {body.template!r} not found")
        params = deep_merge(params, t["params"])
    params = deep_merge(params, body.serve_config)
    fields = getattr(cls, "model_fields", {})
    field = next((f for f in _MODEL_FIELDS if f in fields), None)
    if field and not params.get(field):
        params[field] = model_value
    port = pick_port(svc, params.get("port") if isinstance(params.get("port"), int) else None)
    if "port" in fields:
        params["port"] = port
    if "host" in fields and "host" not in body.serve_config:
        params["host"] = "127.0.0.1"
    job = submit(svc, "serve", params, name=f"serve {body.name or ref}", gpus=body.gpus)
    t = now()
    dep = svc.deployments.insert(
        id=new_id("dep"), name=body.name or Path(str(model_value)).name or ref, artifact_ref=ref,
        serve_config=job["spec"]["params"], job_id=job["id"], port=port, status="starting", created_at=t, updated_at=t,
    )
    return _public(svc, dep)


@router.get("/{dep_id}")
async def get(dep_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return _public(svc, await _probe(svc, svc.deployments.require(dep_id, "deployment")))


@router.post("/{dep_id}/stop", summary="Stop the serve job but keep the record")
def stop(dep_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = svc.deployments.require(dep_id, "deployment")
    if row.get("job_id"):
        svc.jobs.cancel(row["job_id"])
    return _public(svc, svc.deployments.update(dep_id, status="stopped"))


@router.delete("/{dep_id}", status_code=204, summary="Stop and remove")
def delete(dep_id: str, svc: Services = Depends(get_services)) -> Response:
    row = svc.deployments.require(dep_id, "deployment")
    if row.get("job_id"):
        svc.jobs.cancel(row["job_id"])
    svc.deployments.delete(dep_id)
    return Response(status_code=204)


def _headers(row: dict[str, Any]) -> dict[str, str]:
    key = (row.get("serve_config") or {}).get("api_key")
    return {"Authorization": f"Bearer {key}"} if key else {}


@router.get("/{dep_id}/stats", summary="Engine /stats (running/waiting seqs, KV usage, tok/s, TTFT/ITL, spec acceptance...)")
async def stats(dep_id: str, svc: Services = Depends(get_services)) -> Any:
    row = sync_status(svc, svc.deployments.require(dep_id, "deployment"))
    async with _client(svc, 10.0) as c:
        try:
            r = await c.get(_base(row) + "/stats", headers=_headers(row))
        except httpx.HTTPError as e:
            raise BadGateway(f"engine at {_base(row)} is not reachable: {type(e).__name__}: {e}", details={"status": row["status"]}) from e
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text}
    return JSONResponse(data, status_code=r.status_code)


@router.get("/{dep_id}/health")
async def health(dep_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = sync_status(svc, svc.deployments.require(dep_id, "deployment"))
    async with _client(svc, 3.0) as c:
        try:
            r = await c.get(_base(row) + "/health")
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
    if ok and row["status"] != "running":
        svc.deployments.update(dep_id, status="running")
    return {"ok": ok, "status": "running" if ok else row["status"]}


@router.post("/{dep_id}/chat", summary="Proxy an OpenAI chat-completions request to the engine (SSE when stream=true)")
async def chat(dep_id: str, request: Request, svc: Services = Depends(get_services)):
    row = sync_status(svc, svc.deployments.require(dep_id, "deployment"))
    try:
        body = await request.json()
    except ValueError as e:
        raise ConfigError("request body must be JSON") from e
    if not isinstance(body, dict) or "messages" not in body:
        raise ConfigError("body must be an OpenAI chat request with `messages`")
    sc = row.get("serve_config") or {}
    body.setdefault("model", sc.get("served_model_name") or Path(str(sc.get("model") or row["artifact_ref"])).name)
    stream = bool(body.get("stream", True))
    body["stream"] = stream
    url = _base(row) + "/v1/chat/completions"
    headers = _headers(row)
    if not stream:
        async with _client(svc, svc.http_timeout_s) as c:
            try:
                r = await c.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                raise BadGateway(f"engine not reachable: {type(e).__name__}: {e}") from e
        try:
            return JSONResponse(r.json(), status_code=r.status_code)
        except ValueError:
            return JSONResponse({"error": r.text}, status_code=r.status_code)

    client = _client(svc, svc.http_timeout_s)
    req = client.build_request("POST", url, json=body, headers=headers)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        raise BadGateway(f"engine not reachable: {type(e).__name__}: {e}") from e

    async def relay():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
                if await request.is_disconnected():
                    break
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
