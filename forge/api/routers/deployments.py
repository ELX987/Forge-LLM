"""/deployments : start `serve` jobs, discover the port, proxy chat (SSE) and stats to the engine."""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from forge.api.common import deep_merge
from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import DeploymentCreate
from forge.core.errors import ConfigError, ForgeError
from forge.core.jsonschema import validate_params
from forge.core.schemas import JobSpec
from forge.core.settings import get_settings
from forge.core.util import new_id, now
from forge.jobs.manager import InvalidParams
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


_EXPECTED_CACHE: dict[tuple[str, float], dict[str, Any] | None] = {}


def _expected_identity(model: Any) -> dict[str, Any] | None:
    """What the deployment was asked to serve, read from the model directory's own metadata (cheap, cached by mtime)."""
    if not model:
        return None
    p = Path(str(model)).expanduser()
    try:
        key = (str(p), (p / "config.json").stat().st_mtime)
    except OSError:
        return None
    if key not in _EXPECTED_CACHE:
        from forge.core.artifactmeta import read_artifact_summary

        s = read_artifact_summary(p)
        _EXPECTED_CACHE[key] = {"path": str(p), "quantized": bool(s and s["is_quantized"]), "label": s.get("label") if s else None,
                                "base_model": s.get("base_model") if s else None, "kv": s.get("kv")} if s else None
        if len(_EXPECTED_CACHE) > 256:
            _EXPECTED_CACHE.pop(next(iter(_EXPECTED_CACHE)))
    return _EXPECTED_CACHE[key]


def _identity_check(expected: dict[str, Any] | None, loaded: dict[str, Any] | None) -> str | None:
    """A human-readable warning when the engine reports something other than what was requested (``None`` = consistent or unknown)."""
    if not expected or not loaded:
        return None
    if expected.get("quantized") and not loaded.get("quantized"):
        return f"requested a quantised model ({expected.get('label')}) but the engine loaded dense weights ({loaded.get('weight_gb')} GB)"
    if expected.get("quantized") and expected.get("label") and loaded.get("label") and expected["label"].split()[0] != str(loaded["label"]).split()[0]:
        return f"requested {expected['label']} but the engine loaded {loaded['label']}"
    lp, ep = loaded.get("path"), expected.get("path")
    if lp and ep and Path(str(lp)).resolve() != Path(str(ep)).resolve():
        return f"requested {ep} but the engine loaded {lp}"
    return None


def _status_warnings(loaded: Any, memory: Any, expected: dict[str, Any] | None, kv_dtype: Any = "auto") -> list[str]:
    """Warnings to show next to a deployment: KV scheme the engine could not honour, memory-plan warnings (pool vs weights, caps)."""
    out: list[str] = []
    if isinstance(loaded, dict):
        out += [str(w) for w in loaded.get("warnings") or []]
    if isinstance(memory, dict):
        out += [str(w) for w in memory.get("warnings") or []]
    if not loaded and expected:  # not loaded yet: what the artifact's recorded KV scheme will mean for this config
        from forge.serve.memory import artifact_kv_choice

        out += artifact_kv_choice(str(kv_dtype), expected.get("kv"))[1]
    return out


def _public(svc: Services, row: dict[str, Any]) -> dict[str, Any]:
    row = sync_status(svc, row)
    out = dict(row)
    job = svc.jobs_repo.get(row["job_id"]) if row.get("job_id") else None
    out["job_status"] = job["status"] if job else None
    out["job_progress"] = job["progress"] if job else None
    outs = (job or {}).get("outputs") or {}
    out["expected_model"] = _expected_identity((row.get("serve_config") or {}).get("model"))
    out["loaded_model"] = outs.get("model_identity")  # what the engine really built (name, quantisation, weight GB)
    out["memory"] = outs.get("memory")  # weights / KV pool / free, as planned by the engine
    out["identity_warning"] = _identity_check(out["expected_model"], out["loaded_model"])
    out["warnings"] = _status_warnings(out["loaded_model"], out["memory"], out["expected_model"], (row.get("serve_config") or {}).get("kv_dtype", "auto"))
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

    root = Path(str(model_value)).expanduser()
    cfg_path = root / "config.json"
    if root.is_dir() and not cfg_path.is_file():
        from forge.core.errors import NotFound

        hint = root / "artifact"
        raise NotFound(f"{str(root)!r} has no config.json, so it is not a model directory"
                       + (f"; the artifact is in {str(hint)!r} - select that path" if (hint / "config.json").is_file() else ""),
                       details={"path": str(root), "suggestion": str(hint) if (hint / "config.json").is_file() else None})
    try:
        if not cfg_path.is_file():
            return
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        from forge.models import ModelConfig
        from forge.serve.runner import check_servable

        fb = (raw.get("forge") or {}).get("model_config")
        cfg = ModelConfig.model_validate(fb) if fb else ModelConfig.from_hf_config(raw)
    except Exception:  # noqa: BLE001 - unreadable / unparseable configs are reported by the serve job itself
        return
    check_servable(cfg)


def resolve_serve_model(svc: Services, ref: str) -> tuple[str, Any]:
    """``(value for ServeConfig.model, ResolvedModel | None)`` for a picker value / typed reference: a registered model id or name (base models and
    quantised artifacts), a model or artifact directory, an artifact-store digest or a Hugging Face repo id. Unresolvable references raise
    :class:`NotFound` here, so the user sees the problem in the form instead of a serve job that quietly loads something else."""
    from forge.core.modelref import resolve_model_ref

    row = svc.models.get(ref)
    if row and row.get("source") == "hf":
        from forge.jobs.modelinfo import model_local_path

        lp = model_local_path(row)
        if not (lp and Path(lp).is_dir()):
            return row["path_or_repo"], None  # not downloaded yet: the engine downloads the repo on start (previous behaviour)
    r = resolve_model_ref(ref, repo=svc.models)
    if r.path is not None:
        return str(Path(r.path).expanduser().resolve()), r
    return str(r.repo_id), r


class DeploymentEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_ref: str = Field(description="Same value as when creating the deployment (model id / name / path / digest).")
    serve_config: dict[str, Any] = Field(default_factory=dict)
    total_gpu_gb: float | None = Field(None, gt=0, description="GPU memory to plan for; default: the GPU of this server when it has one.")


def _draft_reservation(svc: Services, sc: Any, kv_name: str) -> tuple[int, dict[str, Any] | None]:
    """Bytes the draft model (weights + private KV pool) reserves next to the target, for the plan; ``(0, None)`` without a draft model."""
    spec = sc.speculation
    if spec.mode not in ("draft", "auto") or not spec.draft_model:
        return 0, None
    from forge.models.loader import read_config
    from forge.serve.memory import draft_pool_blocks
    from forge.serve.runner import ModelDims, kv_bytes_per_block

    dval, _dr = resolve_serve_model(svc, spec.draft_model)
    dp = Path(dval)
    if not (dp.is_dir() and (dp / "config.json").is_file()):
        raise ConfigError(f"draft model {spec.draft_model!r} is not available locally ({dval}); download it first to estimate its memory")
    dcfg, _ = read_config(dp)
    files = sum(f.stat().st_size for f in dp.glob("*.safetensors"))
    dsumm = None
    try:
        from forge.core.artifactmeta import read_artifact_summary

        dsumm = read_artifact_summary(dp)
    except Exception:  # noqa: BLE001
        pass
    dw = files if (dsumm and dsumm["is_quantized"] and files) else int(dcfg.param_count() * (4 if sc.dtype == "float32" else 2))
    dd = ModelDims.from_config(dcfg)
    blocks = draft_pool_blocks(max_num_seqs=sc.max_num_seqs, max_model_len=sc.max_model_len, block_size=sc.block_size, kv_pool=sc.kv_pool,
                               kv_pool_cap_tokens=sc.kv_pool_cap_tokens)
    kvb = (blocks + 1) * kv_bytes_per_block(dd.num_layers, dd.num_kv_heads, dd.head_dim, sc.block_size, kv_name)
    return dw + kvb, {"model": dval, "weight_gb": round(dw / 1e9, 3), "kv_pool_gb": round(kvb / 1e9, 3), "blocks": blocks}


@router.post("/estimate", summary="Memory estimate (weights / KV pool / free) for a model + serve config, before starting")
def estimate(body: DeploymentEstimate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    from forge.core.artifactmeta import read_artifact_summary
    from forge.models.loader import read_config
    from forge.serve.config import ServeConfig
    from forge.serve.memory import artifact_kv_choice, estimate_memory

    value, r = resolve_serve_model(svc, body.artifact_ref)
    p = Path(value)
    if not (p.is_dir() and (p / "config.json").is_file()):
        raise ConfigError(f"{body.artifact_ref!r} is not downloaded yet ({value}); download it first to estimate its memory")
    cfg_in = {k: v for k, v in body.serve_config.items() if k != "model"}
    clean, errs = validate_params(ServeConfig, cfg_in)
    if errs:
        raise InvalidParams("invalid serve_config: " + "; ".join(f"{e['loc'] or '<root>'}: {e['msg']}" for e in errs[:5]),
                            details={"errors": errs})
    sc = ServeConfig(**cfg_in)
    mcfg, _raw = read_config(p)
    summ = read_artifact_summary(p)
    files = sum(f.stat().st_size for f in p.glob("*.safetensors"))
    quantized = bool(summ and summ["is_quantized"])
    wbytes = files if (quantized and files) else int(mcfg.param_count() * (4 if sc.dtype == "float32" else 2))
    kv_name = {"auto": "bf16", "fp16": "fp16", "bf16": "bf16"}.get(sc.kv_dtype, sc.kv_dtype)
    chosen_kv, kv_warnings = artifact_kv_choice(sc.kv_dtype, (summ or {}).get("kv"))
    if chosen_kv:
        kv_name = chosen_kv
    total = int(body.total_gpu_gb * 1e9) if body.total_gpu_gb else None
    if total is None:
        try:
            from forge.serve.memory import device_memory

            import torch

            if torch.cuda.is_available():
                total = device_memory(torch.device("cuda"))[0]
        except Exception:  # noqa: BLE001
            total = None
    extra, draft_note = _draft_reservation(svc, sc, kv_name)
    est = estimate_memory(mcfg, weight_bytes=wbytes, packed_weight_bytes=files if quantized else 0, kv_name=kv_name, block_size=sc.block_size,
                          gpu_mem_util=sc.gpu_mem_util, activation_reserve_gb=sc.activation_reserve_gb, max_num_seqs=sc.max_num_seqs,
                          max_model_len=sc.max_model_len, kv_pool=sc.kv_pool, total_gpu_bytes=total, num_gpu_blocks=sc.num_gpu_blocks,
                          kv_pool_cap_tokens=sc.kv_pool_cap_tokens, extra_bytes=extra)
    est["model"] = {"path": value, "quantized": quantized, "label": (summ or {}).get("label"), "base_model": (summ or {}).get("base_model"),
                    "weight_gb": round(wbytes / 1e9, 3), "kind": getattr(r, "kind", None)}
    est["kv_warnings"] = kv_warnings
    if draft_note:
        est["draft"] = draft_note
    est["gpu_total_gb"] = None if total is None else round(total / 1e9, 2)
    return est


@router.post("", status_code=201, summary="Start serving a model (creates a `serve` job)")
def create(body: DeploymentCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    jt = get_job_type("serve")
    cls = jt.config_model()  # 501 when forge.serve is missing
    ref = body.artifact_ref.strip()
    params: dict[str, Any] = {}
    if body.template:
        t = next((t for t in list_templates(jt) if t["name"] == body.template), None)
        if t is None:
            raise ConfigError(f"serve template {body.template!r} not found")
        params = deep_merge(params, t["params"])
    params = deep_merge(params, body.serve_config)
    fields = getattr(cls, "model_fields", {})
    field = next((f for f in _MODEL_FIELDS if f in fields), None)
    # config problems (422) are reported before model problems, so the form can show field errors first
    svc.jobs.validate(JobSpec(type="serve", name="", params={**params, **({field: ref} if field else {})}))
    model_value, resolved = resolve_serve_model(svc, ref)
    if field and params.get(field):
        other, _r2 = resolve_serve_model(svc, str(params[field]))  # an explicit serve_config.model must be the very same model, never silently win
        if os.path.normcase(os.path.normpath(other)) != os.path.normcase(os.path.normpath(model_value)):
            raise ConfigError(f"serve_config.{field} ({params[field]!r}) and the selected model ({ref!r}) are different models; send only one "
                              f"(the deployment would serve {other} while listing {model_value})", details={"selected": model_value, "serve_config": other})
    quant_info = getattr(resolved, "artifact", None) or {}
    if quant_info.get("is_quantized") and params.get("quant"):
        raise ConfigError(f"{ref!r} is already quantised ({quant_info.get('label')}); leave `quant` empty")
    _preflight_servable(model_value)
    if field:
        params[field] = model_value  # always the resolved location: the job spec shows exactly what is loaded
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
