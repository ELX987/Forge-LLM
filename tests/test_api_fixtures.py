"""Shared fixtures for the jobs/api/telemetry tests (imported by the test modules; no tests here)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated FORGE_HOME (fresh sqlite db, metrics store, runs dir) for one test."""
    from forge.core import db as fdb
    from forge.telemetry.metrics import reset_metrics_stores

    reset_metrics_stores()
    fdb.close_all()
    h = tmp_path / "forge_home"
    monkeypatch.setenv("FORGE_HOME", str(h))
    monkeypatch.setenv("FORGE_API_TOKEN", "")
    monkeypatch.delenv("FORGE_CONFIGS_DIR", raising=False)
    yield h
    reset_metrics_stores()
    fdb.close_all()


def make_client(**kw):
    from fastapi.testclient import TestClient

    from forge.api.app import create_app

    opts = dict(token="", stream_poll_s=0.1, grace_s=2.0, sample_interval=0.5)
    opts.update(kw)
    app = create_app(**opts)
    return TestClient(app)


@pytest.fixture
def client(home):
    c = make_client()
    with c:
        yield c
        c.app.state.services.jobs.stop(kill_running=True)


def wait_status(client, job_id: str, want=("succeeded", "failed", "cancelled"), timeout: float = 40.0) -> dict:
    end = time.time() + timeout
    while True:
        j = client.get(f"/api/v1/jobs/{job_id}").json()
        if j["status"] in want or time.time() > end:
            return j
        time.sleep(0.05)
