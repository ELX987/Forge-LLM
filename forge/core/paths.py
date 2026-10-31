"""Filesystem layout under ``FORGE_HOME`` (default ``~/.forge``). All directories are created on demand.

The home is re-read from the environment on every call so tests (and users) can change ``FORGE_HOME``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")


def forge_home() -> Path:
    """Root directory of all Forge state (env ``FORGE_HOME``, default ``~/.forge``)."""
    env = os.environ.get("FORGE_HOME", "").strip()
    p = Path(env).expanduser() if env else Path.home() / ".forge"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(name: str) -> Path:
    """``<home>/<name>`` (created). ``name`` may contain ``/`` but never ``..`` or absolute parts."""
    if not _NAME_RE.match(name) or ".." in name.split("/"):
        raise ValueError(f"invalid data dir name {name!r}")
    p = forge_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir() -> Path:
    return data_dir("runs")


def run_dir(run_id: str) -> Path:
    return data_dir(f"runs/{run_id}")


def store_dir() -> Path:
    return data_dir("store")


def models_dir() -> Path:
    return data_dir("models")


def datasets_dir() -> Path:
    return data_dir("datasets")


def cache_dir() -> Path:
    return data_dir("cache")


def autotune_dir() -> Path:
    return data_dir("autotune")


def configs_dir() -> Path:
    return data_dir("configs")


def db_path() -> Path:
    return forge_home() / "forge.db"


def repo_root() -> Path | None:
    """Repository root when running from a source checkout (contains ``configs/`` and ``forge/``), else None."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "forge").is_dir():
            return parent
    return None
