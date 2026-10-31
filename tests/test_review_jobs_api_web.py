"""Regression tests from the adversarial review of forge/jobs, forge/api, forge/telemetry, web/static (CPU only)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, make_client  # noqa: E402,F401

API = "/api/v1"


# --------------------------------------------------------------------------------------------
# path traversal in model_download / checkpoint_export names (rmtree of an arbitrary directory)
# --------------------------------------------------------------------------------------------
def test_safe_dirname_is_a_single_component():
    from forge.core.errors import ConfigError
    from forge.jobs.builtin import safe_dirname

    for bad in ("..", ".", "", "   ", "..."):
        with pytest.raises(ConfigError):
            safe_dirname(bad)
    for evil in ("../../x", "..\\..\\x", "/etc/passwd", "C:\\Windows\\x", "a/b/c", "org/name"):
        n = safe_dirname(evil)
        assert "/" not in n and "\\" not in n and ":" not in n and n not in (".", ".."), (evil, n)
    assert safe_dirname("org/name") == "org--name"
    assert safe_dirname("llama-3.1-8b") == "llama-3.1-8b"


def test_checkpoint_export_never_deletes_outside_models_dir(home, tmp_path):
    from forge.core.errors import ForgeError
    from forge.core.context import NullContext
    from forge.jobs.builtin import CheckpointExportConfig, run_checkpoint_export

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("data")
    for name in (str(victim), "../victim", str(victim).replace("\\", "/")):
        cfg = CheckpointExportConfig(run_id="job_nothere", name=name)
        with pytest.raises(ForgeError):  # no such checkpoint -> NotFound, but only AFTER a safe dest was chosen
            run_checkpoint_export(cfg, NullContext())
        assert (victim / "precious.txt").read_text() == "data", name


def test_checkpoint_export_rejects_bad_run_id():
    from pydantic import ValidationError

    from forge.jobs.builtin import CheckpointExportConfig

    for bad in ("../x", "a/b", "..", "/abs", "a\\b", ""):
        with pytest.raises(ValidationError):
            CheckpointExportConfig(run_id=bad)


def test_checkpoints_api_rejects_traversal_run(client):
    r = client.get(f"{API}/checkpoints", params={"run": "../.."})
    assert r.status_code in (400, 422), r.text
    r = client.post(f"{API}/checkpoints/x..y:latest/export", json={})
    assert r.status_code in (400, 404, 422), r.text


# --------------------------------------------------------------------------------------------
# CSRF / cross-origin guard
# --------------------------------------------------------------------------------------------
def test_cross_origin_post_is_refused_but_same_origin_and_cli_pass(client):
    jid = client.post(f"{API}/jobs", json={"type": "selftest", "params": {"steps": 1, "step_delay_s": 0}}).json()["id"]
    # a web page on another origin can fire a "simple" POST without preflight
    r = client.post(f"{API}/jobs/{jid}/clone", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403, r.text
    # same origin (the bundled UI) and non-browser clients (no Origin header) are fine
    ok = client.post(f"{API}/jobs/{jid}/clone", headers={"Origin": "http://testserver"})
    assert ok.status_code == 201, ok.text
    assert client.post(f"{API}/jobs/{jid}/clone").status_code == 201
    # reads are never blocked
    assert client.get(f"{API}/jobs", headers={"Origin": "http://evil.example"}).status_code == 200


def test_cors_allowlist_permits_listed_origin(home):
    c = make_client(cors_origins=["http://good.example"])
    with c:
        jid = c.post(f"{API}/jobs", json={"type": "selftest", "params": {"steps": 1, "step_delay_s": 0}}).json()["id"]
        assert c.post(f"{API}/jobs/{jid}/clone", headers={"Origin": "http://good.example"}).status_code == 201
        assert c.post(f"{API}/jobs/{jid}/clone", headers={"Origin": "http://bad.example"}).status_code == 403
        c.app.state.services.jobs.stop(kill_running=True)


# --------------------------------------------------------------------------------------------
# auth: every /api/v1 route except health demands the token
# --------------------------------------------------------------------------------------------
def test_every_api_route_requires_token(home):
    c = make_client(token="s3cret")
    with c:
        app = c.app
        checked = 0
        for path, ops in app.openapi()["paths"].items():
            methods = {m.upper() for m in ops}
            if not path.startswith(API) or path == f"{API}/system/health":
                continue
            url = path
            for seg in ("{job_id}", "{dataset_id}", "{model_id}", "{config_id}", "{recipe_id}", "{dep_id}", "{h}", "{ckpt_id}",
                        "{key}", "{descriptor_id}", "{gate_id}", "{eval_id}", "{claim_id}", "{type}", "{id}"):
                url = url.replace(seg, "x")
            if "{" in url:
                url = "/".join("x" if p.startswith("{") else p for p in url.split("/"))
            for m in methods - {"HEAD", "OPTIONS"}:
                r = c.request(m, url)
                assert r.status_code == 401, (m, path, r.status_code)
                checked += 1
        assert checked > 60
        good = c.get(f"{API}/system/info", headers={"Authorization": "Bearer s3cret"})
        assert good.status_code == 200
        assert c.get(f"{API}/system/health").status_code == 200
        c.app.state.services.jobs.stop(kill_running=True)


# --------------------------------------------------------------------------------------------
# SSE / metrics contract used by the front end
# --------------------------------------------------------------------------------------------
def _idle_client(home):
    """App without the job-manager thread: jobs stay queued, so the only metrics are the ones the test writes."""
    return make_client(start_background=False)


def _events(body: str) -> list[tuple[str, str]]:
    out = []
    for blk in body.split("\n\n"):
        ev = next((l[7:] for l in blk.splitlines() if l.startswith("event: ")), None)
        data = next((l[6:] for l in blk.splitlines() if l.startswith("data: ")), "")
        if ev:
            out.append((ev, data))
    return out


def test_metrics_endpoint_returns_resume_cursor_and_stream_replay_zero_skips_history(home):
    from forge.telemetry.metrics import get_metrics_store

    with _idle_client(home) as client:
        m = get_metrics_store()
        jid = client.post(f"{API}/jobs", json={"type": "selftest", "params": {"steps": 1, "step_delay_s": 0}}).json()["id"]
        for i in range(10):
            m.write(jid, i, {"loss": 1.0 / (i + 1)})
        m.flush()
        d = client.get(f"{API}/jobs/{jid}/metrics").json()
        assert isinstance(d["seq"], int) and d["seq"] >= 10
        # replay=0 means "nothing from the past" (was: the newest row leaked through)
        with client.stream("GET", f"{API}/jobs/{jid}/stream", params={"replay": 0, "log_tail": 0, "max_events": 2}) as r:
            evs = _events("".join(r.iter_text()))
        assert [e for e, _ in evs] == ["status", "progress"], evs


def test_stream_resumes_from_since_seq_without_duplicates(home):
    from forge.telemetry.metrics import get_metrics_store

    with _idle_client(home) as client:
        m = get_metrics_store()
        jid = client.post(f"{API}/jobs", json={"type": "selftest", "params": {"steps": 1, "step_delay_s": 0}}).json()["id"]
        m.write(jid, 100, {"a": 1.0})
        m.flush()
        seq = client.get(f"{API}/jobs/{jid}/metrics").json()["seq"]
        m.write(jid, 101, {"a": 2.0})
        m.flush()
        with client.stream("GET", f"{API}/jobs/{jid}/stream", params={"since_seq": seq, "log_tail": 0, "max_events": 3}) as r:
            evs = _events("".join(r.iter_text()))
        metric = [d for e, d in evs if e == "metric"]
        assert len(metric) == 1 and "[101," in metric[0] and "[100," not in metric[0]


# --------------------------------------------------------------------------------------------
# manager: fast-worker race + device ordering
# --------------------------------------------------------------------------------------------
class _FastPopen:
    """Pretends to be a worker that already finished (and recorded 'succeeded') before the manager looks again."""

    created: list = []

    def __init__(self, cmd, **kw):
        from forge.core.db import get_db
        from forge.core.util import now

        self.env = kw.get("env", {})
        self.pid = 4242424
        self.returncode = 0
        jid = cmd[-1]
        get_db().execute("UPDATE jobs SET status='succeeded', finished_at=?, exit_code=0 WHERE id=?", (now(), jid))
        _FastPopen.created.append(self)

    def poll(self):
        return 0


def test_spawn_does_not_resurrect_job_finished_before_manager_update(home, monkeypatch):
    from forge.core.schemas import JobSpec
    from forge.jobs import manager as mm

    _FastPopen.created.clear()
    monkeypatch.setattr(mm.subprocess, "Popen", _FastPopen)
    m = mm.JobManager(poll_s=0.05, gpu_provider=lambda: [0, 1])
    row = m.submit(JobSpec(type="selftest", params={"steps": 1}))
    m.tick()
    after = m.repo.require(row["id"], "job")
    assert after["status"] == "succeeded", after  # used to flip back to running, then be reaped as failed
    m.tick()
    assert m.repo.require(row["id"], "job")["status"] == "succeeded"
    env = _FastPopen.created[0].env
    assert env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"  # NVML/UI indices == CUDA indices on mixed-GPU hosts


# --------------------------------------------------------------------------------------------
# telemetry: one shared sampler (NVML init is expensive and was repeated for every sample)
# --------------------------------------------------------------------------------------------
def test_sample_gpus_reuses_one_sampler(monkeypatch):
    import forge.hw.telemetry as ht
    from forge.telemetry import system as ts

    made = []

    class Fake:
        def __init__(self, *a, **k):
            made.append(1)

        def sample(self):
            return [{"index": 0, "name": "fake", "util": 1.0}]

    monkeypatch.setattr(ht, "GpuSampler", Fake)
    monkeypatch.setattr(ts, "_SAMPLER", None)
    for _ in range(5):
        assert ts.sample_gpus()[0]["name"] == "fake"
    assert len(made) == 1


# --------------------------------------------------------------------------------------------
# front end: static checks that guard the contracts / XSS-safety
# --------------------------------------------------------------------------------------------
STATIC = Path(__file__).resolve().parents[1] / "forge" / "web" / "static"


def test_frontend_never_uses_html_injection_sinks():
    import re

    bad = re.compile(r"\.innerHTML\s*=|\.outerHTML\s*=|insertAdjacentHTML|document\.write\(|\beval\(|new Function\(")
    hits = []
    for f in list((STATIC / "js").rglob("*.js")) + [STATIC / "index.html"]:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if bad.search(line):
                hits.append(f"{f.name}:{i}")
    assert not hits, hits


def test_job_monitor_understands_server_stream_event_shapes():
    src = (STATIC / "js" / "pages" / "jobdetail.js").read_text(encoding="utf-8")
    assert "d.points" in src and "Array.isArray(d.lines)" in src  # server: metric {points,seq}, log {lines,offset}
    api = (STATIC / "js" / "api.js").read_text(encoding="utf-8")
    assert "typeof query === 'function'" in api  # resume cursors are re-evaluated on reconnect


def test_static_server_blocks_traversal_and_sets_security_headers(client):
    secret = Path(__file__).resolve()  # a real file outside forge/web/static
    for p in ("/..%2f..%2f..%2ftests%2ftest_review_jobs_api_web.py", "/%2e%2e/%2e%2e/%2e%2e/tests/test_review_jobs_api_web.py",
              "/css/..%2f..%2f..%2f..%2f..%2ftests%2ftest_review_jobs_api_web.py", "/..%5c..%5c..%5ctests%5ctest_review_jobs_api_web.py"):
        r = client.get(p)
        assert secret.name not in r.text and "test_static_server_blocks" not in r.text, p
    r = client.get("/")
    assert r.status_code == 200 and r.headers.get("x-frame-options") == "DENY" and r.headers.get("x-content-type-options") == "nosniff"
    assert client.get("/js/nope.js").status_code == 404
    assert client.get("/api/v1/nope").status_code in (404, 405)
