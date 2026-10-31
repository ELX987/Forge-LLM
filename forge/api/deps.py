"""Service container + FastAPI dependencies (auth, service access)."""
from __future__ import annotations

import hmac
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request

from forge.jobs.manager import JobManager
from forge.jobs.repo import ConfigsRepo, DatasetsRepo, DeploymentsRepo, JobsRepo, ModelsRepo, RecipesRepo
from forge.telemetry.metrics import MetricsStore
from forge.telemetry.system import SystemMonitor


@dataclass
class Services:
    jobs: JobManager
    jobs_repo: JobsRepo
    models: ModelsRepo
    datasets: DatasetsRepo
    configs: ConfigsRepo
    recipes: RecipesRepo
    deployments: DeploymentsRepo
    metrics: MetricsStore
    monitor: SystemMonitor
    token: str | None = None
    stream_poll_s: float = 0.5
    max_upload_bytes: int = 64 * 2**30
    started_at: float = field(default_factory=time.time)
    http_timeout_s: float = 600.0
    extras: dict[str, Any] = field(default_factory=dict)


def get_services(request: Request) -> Services:
    return request.app.state.services


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("x-forge-token")
        or request.query_params.get("token")
        or request.query_params.get("access_token")
    )


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def check_origin(request: Request) -> None:
    """CSRF / DNS-rebinding guard for state-changing requests.

    Browsers always attach ``Origin`` to cross-site POST/PUT/PATCH/DELETE. A "simple" cross-site POST (no body, or
    text/plain) needs no CORS preflight, so without this check any web page open in the user's browser could cancel
    jobs or start downloads on a token-less localhost server. Same-origin requests (the bundled UI) and origins
    listed in ``FORGE_CORS_ORIGINS`` pass; non-browser clients (no ``Origin``) are unaffected."""
    if request.method in _SAFE_METHODS:
        return
    origin = request.headers.get("origin")
    if not origin:
        return
    allowed = getattr(request.app.state, "allowed_origins", None) or []
    if "*" in allowed or origin in allowed:
        return
    from urllib.parse import urlsplit

    try:
        netloc = urlsplit(origin).netloc.lower()
    except ValueError:
        netloc = ""
    host = (request.headers.get("host") or "").lower()
    if netloc and netloc == host:
        return
    raise HTTPException(status_code=403, detail=f"cross-origin request from {origin!r} refused (set FORGE_CORS_ORIGINS to allow it)")


def require_auth(request: Request) -> None:
    """Optional bearer-token auth: enforced only when the app was created with / the env has ``FORGE_API_TOKEN``."""
    check_origin(request)
    token = request.app.state.services.token
    if not token:
        return
    supplied = _extract_token(request)
    if not supplied or not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API token (send `Authorization: Bearer <token>`)",
            headers={"WWW-Authenticate": "Bearer"},
        )
