"""GET /system/cuda: cached, non-blocking CUDA stack report that never raises."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed in this interpreter (pip install -e \".[serve,dev]\")")

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, make_client  # noqa: E402,F401

API = "/api/v1"


@pytest.fixture
def fake_check(monkeypatch):
    from forge.api.routers import system
    from forge.hw import cuda_compat

    system._cuda_cache.clear()
    calls: list[dict] = []

    def check(*, probe=True, probe_timeout=120.0, stack=None):
        calls.append({"probe": probe})
        f = {"level": "fail", "code": "driver_too_old", "message": "driver 580 (CUDA 13.0) < torch CUDA 13.4 <host>", "fix": "upgrade the driver"}
        return {"ok": False, "worst": "fail", "stack": {"driver_cuda": "13.0"}, "findings": [f], "recommend": {"torch_index": "cu130"}}

    monkeypatch.setattr(cuda_compat, "check", check)
    yield calls
    system._cuda_cache.clear()


def test_cuda_report_shape_and_cache(client, fake_check):
    r = client.get(f"{API}/system/cuda")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False and body["worst"] == "fail" and body["findings"][0]["code"] == "driver_too_old"
    assert body["recommend"]["torch_index"] == "cu130" and body["stack"]["driver_cuda"] == "13.0"
    client.get(f"{API}/system/cuda")
    assert len(fake_check) == 1  # second call served from the 60 s cache
    client.get(f"{API}/system/cuda", params={"refresh": 1})
    assert len(fake_check) == 2  # refresh bypasses it


def test_cuda_probe_flag_and_separate_cache(client, fake_check):
    client.get(f"{API}/system/cuda", params={"probe": 0})
    client.get(f"{API}/system/cuda", params={"probe": 1})
    client.get(f"{API}/system/cuda", params={"probe": 0})
    client.get(f"{API}/system/cuda", params={"probe": 1})
    assert [c["probe"] for c in fake_check] == [False, True]


def test_cuda_probe_defaults_off(client, fake_check):
    client.get(f"{API}/system/cuda")
    assert [c["probe"] for c in fake_check] == [False]


def test_cuda_cache_expires(client, fake_check, monkeypatch):
    from forge.api.routers import system

    client.get(f"{API}/system/cuda", params={"probe": 1})
    ts, rep = system._cuda_cache[True]
    system._cuda_cache[True] = (ts - system._CUDA_TTL_S - 1, rep)
    client.get(f"{API}/system/cuda", params={"probe": 1})
    assert len(fake_check) == 2


def test_cuda_probe_skipped_while_jobs_hold_gpus(client, fake_check, monkeypatch):
    jobs = client.app.state.services.jobs
    monkeypatch.setattr(jobs, "busy_gpus", lambda: {0, 1})
    body = client.get(f"{API}/system/cuda", params={"probe": 1}).json()
    assert [c["probe"] for c in fake_check] == [False]  # static report only
    assert "probe_skipped" in body and "[0, 1]" in body["probe_skipped"] and "force=1" in body["probe_skipped"]
    body = client.get(f"{API}/system/cuda", params={"probe": 0}).json()
    assert "probe_skipped" not in body and len(fake_check) == 1  # the static report is cached and carries no note
    body = client.get(f"{API}/system/cuda", params={"probe": 1, "force": 1}).json()
    assert [c["probe"] for c in fake_check] == [False, True] and "probe_skipped" not in body
    monkeypatch.setattr(jobs, "busy_gpus", lambda: set())
    client.get(f"{API}/system/cuda", params={"probe": 1, "refresh": 1})
    assert [c["probe"] for c in fake_check] == [False, True, True]


def test_cuda_never_raises(client, monkeypatch):
    from forge.api.routers import system
    from forge.hw import cuda_compat

    system._cuda_cache.clear()

    def boom(**kw):
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr(cuda_compat, "check", boom)
    r = client.get(f"{API}/system/cuda")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and "RuntimeError: nvml exploded" in body["error"] and body["findings"] == []
    system._cuda_cache.clear()


def test_cuda_requires_token_when_auth_enabled(home, fake_check):
    c = make_client(token="s3cret")
    with c:
        assert c.get(f"{API}/system/cuda").status_code in (401, 403)
        assert c.get(f"{API}/system/cuda", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        c.app.state.services.jobs.stop(kill_running=True)
