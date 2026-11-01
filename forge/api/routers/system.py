"""/system/* : info, health, GPUs (+SSE)."""
from __future__ import annotations

import platform
import sys
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from forge import __version__
from forge.api.common import lazy_module
from forge.api.deps import Services, get_services
from forge.api.sse import sse_response, sse_stream
from forge.core.settings import get_settings

router = APIRouter(prefix="/system", tags=["system"])
public_router = APIRouter(prefix="/system", tags=["system"])  # no auth: liveness probe


@public_router.get("/health", summary="Liveness probe")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "time": time.time()}


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for p in ("torch", "triton", "transformers", "huggingface_hub", "safetensors", "numpy", "pydantic", "fastapi", "uvicorn", "pyarrow", "nvidia-ml-py", "httpx"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _capabilities() -> dict[str, Any] | None:
    hw = lazy_module("forge.hw.capabilities")
    if hw is None or not hasattr(hw, "current_capabilities"):
        return None
    try:
        caps = hw.current_capabilities()
        from forge.jobs.repo import to_jsonable

        return to_jsonable(caps)
    except Exception as e:  # driver problems must not break /system/info
        return {"error": f"{type(e).__name__}: {e}"}


def _store_summary() -> dict[str, Any] | None:
    mod = lazy_module("forge.store")
    if mod is None or not hasattr(mod, "get_store"):
        return None
    try:
        st = mod.get_store().stats()
        from forge.jobs.repo import to_jsonable

        d = to_jsonable(st)
        return {k: d[k] for k in list(d)[:12]} if isinstance(d, dict) else d
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@router.get("/info", summary="Versions, home, store summary, capabilities, job counts")
def info(svc: Services = Depends(get_services)) -> dict[str, Any]:
    st = get_settings()
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "home": str(st.home),
        "settings": {**st.public_dict(), "api_token": "set" if svc.token else None},
        "auth_required": bool(svc.token),
        "packages": _package_versions(),
        "capabilities": _capabilities(),
        "store": _store_summary(),
        "jobs": svc.jobs_repo.counts(),
        "gpus": len(svc.monitor.gpus()),
        "max_concurrent_jobs": svc.jobs.max_concurrent,
        "gpu_kernels_note": "GPU kernels are implemented but untested on this host until you run `forge probe` / the GPU test suite.",
        "server_time": time.time(),
        "uptime_s": time.time() - svc.started_at,
    }


_CUDA_TTL_S = 60.0
_cuda_cache: dict[bool, tuple[float, dict[str, Any]]] = {}  # probe flag -> (monotonic ts, report)
_cuda_lock = threading.Lock()


def _cuda_report(probe: bool, refresh: bool) -> dict[str, Any]:
    """Blocking (subprocess probe): run in a worker thread.  Never raises; the probe runs in a child so this process gets no CUDA context."""
    with _cuda_lock:  # one probe at a time; concurrent callers reuse its result
        hit = _cuda_cache.get(probe)
        if hit and not refresh and time.monotonic() - hit[0] < _CUDA_TTL_S:
            return hit[1]
        try:
            from forge.hw import cuda_compat

            rep = cuda_compat.check(probe=probe)
        except Exception as e:  # a broken driver stack must not 500 the dashboard
            rep = {"ok": False, "worst": "fail", "stack": {}, "findings": [], "recommend": {}, "error": f"{type(e).__name__}: {e}"}
        _cuda_cache[probe] = (time.monotonic(), rep)
        return rep


@router.get("/cuda", summary="CUDA stack check: driver / torch / GPU / Triton compatibility (cached ~60 s)")
async def cuda(
    refresh: bool = Query(False, description="Bypass the cache."),
    probe: bool = Query(False, description="Run the per-device allocation test (in a subprocess). Skipped while Forge jobs hold GPUs unless force=1."),
    force: bool = Query(False, description="Run the allocation probe even though Forge jobs are running on GPUs."),
    svc: Services = Depends(get_services),
) -> dict[str, Any]:
    skipped: str | None = None
    if probe and not force:
        try:
            busy = sorted(svc.jobs.busy_gpus())
        except Exception:  # job bookkeeping must not break the check
            busy = []
        if busy:
            probe = False
            skipped = f"allocation probe skipped: Forge jobs are running on GPU(s) {busy}; pass force=1 to probe anyway"
    rep = await run_in_threadpool(_cuda_report, probe, refresh)
    return {**rep, "probe_skipped": skipped} if skipped else rep


@router.get("/gpus", summary="Live GPU telemetry")
def gpus(history: int = Query(0, ge=0, le=900, description="Also return the last N samples."), svc: Services = Depends(get_services)) -> dict[str, Any]:
    latest = svc.monitor.latest()
    out = {"gpus": latest.get("gpus", []), "host": latest.get("host", {}), "ts": latest.get("ts"), "busy": sorted(svc.jobs.busy_gpus())}
    if history:
        out["history"] = svc.monitor.history(history)
    return out


@router.get("/gpus/stream", summary="SSE: GPU telemetry samples")
async def gpus_stream(
    request: Request,
    interval: float = Query(1.0, ge=0.1, le=30),
    max_events: int | None = Query(None, ge=1, description="Close after N events (testing)."),
    svc: Services = Depends(get_services),
):
    def step() -> tuple[list[tuple[str, Any]], bool]:
        latest = svc.monitor.latest()
        return [("gpus", {"gpus": latest.get("gpus", []), "host": latest.get("host", {}), "ts": latest.get("ts"), "busy": sorted(svc.jobs.busy_gpus())})], False

    return sse_response(sse_stream(request, step, interval=interval, max_events=max_events))
