"""Runtime settings read from the environment (no file parsing, no network)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from forge.core import paths


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_opt(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


@dataclass
class Settings:
    """Environment-derived settings. Re-created on each :func:`get_settings` call (cheap)."""

    home: Path
    db_path: Path
    db_url: str
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_token: str | None = None
    hf_token: str | None = None
    max_concurrent_jobs: int = 1
    log_level: str = "INFO"
    job_kill_grace_s: float = 20.0
    telemetry_interval_s: float = 1.5
    cors_origins: list[str] = field(default_factory=list)
    serve_base_port: int = 8800
    force_ref_kernels: bool = False
    kernel_debug: bool = False

    def public_dict(self) -> dict[str, Any]:
        """Dict safe to show in the UI (tokens masked)."""
        d = asdict(self)
        d["home"] = str(self.home)
        d["db_path"] = str(self.db_path)
        d["api_token"] = "set" if self.api_token else None
        d["hf_token"] = "set" if self.hf_token else None
        return d


def get_settings() -> Settings:
    home = paths.forge_home()
    dbp = paths.db_path()
    cors = [c.strip() for c in os.environ.get("FORGE_CORS_ORIGINS", "").split(",") if c.strip()]
    return Settings(
        home=home,
        db_path=dbp,
        db_url=f"sqlite:///{dbp.as_posix()}",
        api_host=os.environ.get("FORGE_API_HOST", "127.0.0.1"),
        api_port=_env_int("FORGE_API_PORT", 8765),
        api_token=_env_opt("FORGE_API_TOKEN"),
        hf_token=_env_opt("HF_TOKEN") or _env_opt("HUGGING_FACE_HUB_TOKEN"),
        max_concurrent_jobs=max(1, _env_int("FORGE_MAX_CONCURRENT_JOBS", 1)),
        log_level=os.environ.get("FORGE_LOG_LEVEL", "INFO").upper(),
        job_kill_grace_s=_env_float("FORGE_JOB_KILL_GRACE_S", 20.0),
        telemetry_interval_s=_env_float("FORGE_TELEMETRY_INTERVAL_S", 1.5),
        cors_origins=cors,
        serve_base_port=_env_int("FORGE_SERVE_BASE_PORT", 8800),
        force_ref_kernels=os.environ.get("FORGE_FORCE_REF", "") in ("1", "true", "True"),
        kernel_debug=os.environ.get("FORGE_KERNEL_DEBUG", "") in ("1", "true", "True"),
    )
