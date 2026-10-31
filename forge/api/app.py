"""FastAPI application factory: ``create_app()``.

Run with ``forge serve-ui`` or ``uvicorn --factory forge.api.app:create_app``. Everything under ``/api/v1``
requires ``Authorization: Bearer $FORGE_API_TOKEN`` when a token is configured (``/api/v1/system/health`` and
the static UI are always public). OpenAPI docs live at ``/docs``.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from forge import __version__
from forge.api.deps import Services, require_auth
from forge.api.static import mount_static
from forge.core.errors import ForgeError
from forge.core.settings import get_settings

log = logging.getLogger("forge.api")

API_PREFIX = "/api/v1"

TAGS = [
    {"name": "system", "description": "Versions, health, GPU telemetry"},
    {"name": "jobs", "description": "Submit, monitor and cancel jobs (train / finetune / compress / eval / serve ...)"},
    {"name": "schemas", "description": "JSON schemas + defaults + templates of every job type (the UI renders forms from these)"},
    {"name": "configs", "description": "Saved configs, templates and validation"},
    {"name": "models", "description": "Model registry (local / Hugging Face)"},
    {"name": "datasets", "description": "Datasets: upload, preview, ingest"},
    {"name": "recipes", "description": "Compression / training recipes"},
    {"name": "deployments", "description": "Inference deployments + chat proxy"},
    {"name": "storage", "description": "Content-addressed store"},
]


def build_services(*, token: str | None = None, max_concurrent: int | None = None, gpu_provider: Any = None,
                   grace_s: float | None = None, stream_poll_s: float = 0.5, sample_interval: float | None = None) -> Services:
    from forge.core.db import get_db
    from forge.jobs.manager import JobManager
    from forge.jobs.repo import ConfigsRepo, DatasetsRepo, DeploymentsRepo, JobsRepo, ModelsRepo, RecipesRepo
    from forge.telemetry.metrics import get_metrics_store
    from forge.telemetry.system import SystemMonitor

    st = get_settings()
    db = get_db()
    metrics = get_metrics_store()
    jobs_repo = JobsRepo(db)
    monitor = SystemMonitor(sample_interval or st.telemetry_interval_s, metrics_store=metrics)
    jm = JobManager(repo=jobs_repo, metrics=metrics, max_concurrent=max_concurrent, grace_s=grace_s,
                    gpu_provider=gpu_provider or monitor.gpu_indices)
    monitor._running_jobs = jm.running_jobs
    max_mb = os.environ.get("FORGE_MAX_UPLOAD_MB", "").strip()
    return Services(
        jobs=jm, jobs_repo=jobs_repo, models=ModelsRepo(db), datasets=DatasetsRepo(db), configs=ConfigsRepo(db),
        recipes=RecipesRepo(db), deployments=DeploymentsRepo(db), metrics=metrics, monitor=monitor,
        token=token if token is not None else st.api_token, stream_poll_s=stream_poll_s,
        max_upload_bytes=int(float(max_mb) * 2**20) if max_mb else 64 * 2**30,
    )


def create_app(
    *,
    token: str | None = None,
    start_background: bool = True,
    max_concurrent: int | None = None,
    gpu_provider: Any = None,
    grace_s: float | None = None,
    stream_poll_s: float = 0.5,
    sample_interval: float | None = None,
    cors_origins: list[str] | None = None,
    serve_static: bool = True,
) -> FastAPI:
    """Build the app. ``token=None`` reads ``FORGE_API_TOKEN``; pass ``token=""`` to disable auth explicitly.
    ``start_background=False`` skips the job-manager and GPU-sampler threads (used by some tests)."""
    from forge.api.routers import (
        artifacts, checkpoints, configs, datasets, deployments, evals, events, hardware, jobs, kernels,
        models, planner, recipes, schemas_r, storage, system,
    )

    st = get_settings()
    services = build_services(
        token=token, max_concurrent=max_concurrent, gpu_provider=gpu_provider, grace_s=grace_s,
        stream_poll_s=stream_poll_s, sample_interval=sample_interval,
    )
    if services.token == "":
        services.token = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_background:
            services.monitor.start()
            services.jobs.start()
        try:
            yield
        finally:
            services.jobs.stop()
            services.monitor.stop()
            try:
                services.metrics.flush()
            except Exception:  # pragma: no cover
                pass

    app = FastAPI(
        title="Forge LLM API",
        version=__version__,
        description="Train, fine-tune, compress, evaluate and serve LLMs. See /docs for the endpoints; "
        "the web UI is served at `/`. GPU kernels are implemented but unvalidated until tested on NVIDIA hardware.",
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.services = services

    origins = cors_origins if cors_origins is not None else st.cors_origins
    app.state.allowed_origins = list(origins or [])
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
            expose_headers=["X-Total-Count"],
        )

    # -- errors ------------------------------------------------------------------------------
    @app.exception_handler(ForgeError)
    async def forge_error_handler(_req: Request, exc: ForgeError) -> JSONResponse:
        body = exc.to_dict()
        if "errors" in exc.details:
            body["errors"] = exc.details["errors"]  # field-level validation errors, top level for convenience
        if exc.http_status >= 500 and exc.http_status != 501:
            log.exception("unhandled Forge error", exc_info=exc)
        return JSONResponse(status_code=exc.http_status, content=body)

    # -- routers -----------------------------------------------------------------------------
    app.include_router(system.public_router, prefix=API_PREFIX)
    protected = [
        system.router, hardware.router, models.router, datasets.router, schemas_r.router, configs.router, jobs.router,
        recipes.router, planner.router, artifacts.router, storage.router, evals.router, kernels.router,
        deployments.router, checkpoints.router, events.router,
    ]
    for r in protected:
        app.include_router(r, prefix=API_PREFIX, dependencies=[Depends(require_auth)])
    if serve_static:
        mount_static(app)
    return app


def app_factory() -> FastAPI:  # for `uvicorn --factory forge.api.app:app_factory`
    return create_app()
