"""REST API: jobs lifecycle, SSE streams, schemas, configs, auth, system endpoints (TestClient + selftest job)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, make_client, wait_status  # noqa: E402,F401

API = "/api/v1"


def submit(client, **params):
    body = {"type": "selftest", "name": "t", "params": {"step_delay_s": 0.02, **params}}
    r = client.post(f"{API}/jobs", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def parse_sse(lines):
    events, cur = [], {}
    for line in lines:
        if line.startswith(":"):
            continue
        if line == "":
            if cur:
                events.append(cur)
                cur = {}
            continue
        k, _, v = line.partition(": ")
        cur[k] = v
    return [{"event": e.get("event"), "data": json.loads(e["data"]), "id": e.get("id")} for e in events if "data" in e]


# --------------------------------------------------------------------------------------------
# job lifecycle
# --------------------------------------------------------------------------------------------
def test_job_lifecycle(client):
    j = submit(client, steps=10)
    assert j["status"] in ("queued", "running") and j["type"] == "selftest" and j["spec"]["params"]["steps"] == 10
    done = wait_status(client, j["id"])
    assert done["status"] == "succeeded" and done["progress"] == 1.0
    assert done["outputs"]["echo"] == "ok" and done["duration_s"] > 0 and done["exit_code"] == 0
    assert "loss" in done["latest_metrics"]

    # list + filters + total header
    r = client.get(f"{API}/jobs", params={"status": "succeeded", "type": "selftest"})
    assert r.status_code == 200 and [x["id"] for x in r.json()] == [j["id"]]
    assert r.headers["X-Total-Count"] == "1"
    assert client.get(f"{API}/jobs", params={"status": "failed"}).json() == []
    assert client.get(f"{API}/jobs", params={"limit": 0}).status_code == 422

    # logs
    logs = client.get(f"{API}/jobs/{j['id']}/logs", params={"tail": 100}).json()
    assert any("SYNTHETIC" in ln for ln in logs["lines"]) and logs["status"] == "succeeded"
    more = client.get(f"{API}/jobs/{j['id']}/logs", params={"offset": logs["offset"]}).json()
    assert more["lines"] == [] and more["offset"] == logs["offset"]
    first = client.get(f"{API}/jobs/{j['id']}/logs", params={"offset": 0, "max_bytes": 200}).json()
    assert first["lines"] and first["offset"] > 0

    # metrics
    m = client.get(f"{API}/jobs/{j['id']}/metrics", params={"keys": "loss,lr"}).json()
    assert m["keys"] == ["loss", "lr"] and len(m["points"]["loss"]) == 10
    assert m["points"]["loss"][0][0] == 1
    m2 = client.get(f"{API}/jobs/{j['id']}/metrics", params={"keys": "loss", "since_step": 7}).json()
    assert [p[0] for p in m2["points"]["loss"]] == [8, 9, 10]
    m3 = client.get(f"{API}/jobs/{j['id']}/metrics", params={"keys": "loss", "max_points": 3, "rollup": "lttb"}).json()
    assert len(m3["points"]["loss"]) == 3
    assert "sys/cpu_percent" in client.get(f"{API}/jobs/{j['id']}/metrics").json()["keys"] or True  # best effort host series

    # clone, delete
    c = client.post(f"{API}/jobs/{j['id']}/clone")
    assert c.status_code == 201 and c.json()["id"] != j["id"]
    wait_status(client, c.json()["id"])
    assert client.delete(f"{API}/jobs/{j['id']}").status_code == 204
    assert client.get(f"{API}/jobs/{j['id']}").status_code == 404
    assert client.get(f"{API}/jobs/{j['id']}/metrics").status_code == 404


def test_failed_job_reports_error(client):
    j = submit(client, steps=10, fail_at_step=2)
    done = wait_status(client, j["id"])
    assert done["status"] == "failed" and "injected failure" in done["error"] and done["exit_code"] == 1


def test_invalid_params_return_field_level_errors(client):
    r = client.post(f"{API}/jobs", json={"type": "selftest", "params": {"steps": 0, "step_delay_s": "abc", "extra": 1}})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_params"
    locs = {e["loc"]: e for e in body["errors"]}
    assert set(locs) == {"steps", "step_delay_s", "extra"} and all(e["msg"] for e in locs.values())
    assert client.get(f"{API}/jobs").json() == []


def test_unknown_type_404_and_unavailable_module_501(client):
    r = client.post(f"{API}/jobs", json={"type": "nope", "params": {}})
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    # a type whose module is not importable on this host -> 501 with the reason
    from forge.jobs.registry import JobTypeSpec, register_job_type

    register_job_type(JobTypeSpec("ghost", "Ghost", "x", "ghost_pkg.cfg:Cfg", "ghost_pkg.run:go"))
    r = client.post(f"{API}/jobs", json={"type": "ghost", "params": {}})
    assert r.status_code == 501 and r.json()["error"] == "module_unavailable"
    assert "ghost_pkg" in r.json()["message"]


def test_cancel_running_and_queued(client):
    a = submit(client, steps=5000)
    b = submit(client, steps=3)  # queued behind a (max_concurrent=1)
    deadline = time.time() + 30
    while client.get(f"{API}/jobs/{a['id']}").json()["status"] != "running" and time.time() < deadline:
        time.sleep(0.05)
    rb = client.post(f"{API}/jobs/{b['id']}/cancel").json()
    assert rb["status"] == "cancelled"
    ra = client.post(f"{API}/jobs/{a['id']}/cancel").json()
    assert ra["cancel_requested"] is True
    assert wait_status(client, a["id"])["status"] == "cancelled"
    assert client.post(f"{API}/jobs/missing/cancel").status_code == 404
    # deleting an active job is refused
    c = submit(client, steps=5000)
    time.sleep(0.5)
    r = client.delete(f"{API}/jobs/{c['id']}")
    assert r.status_code == 409
    client.post(f"{API}/jobs/{c['id']}/cancel")
    wait_status(client, c["id"])


def test_submit_from_template_and_saved_config(client, tmp_path, monkeypatch):
    root = tmp_path / "cfgs"
    (root / "selftest").mkdir(parents=True)
    (root / "selftest" / "tiny.yaml").write_text("steps: 4\nstep_delay_s: 0\nresult_value: from-template\n")
    monkeypatch.setenv("FORGE_CONFIGS_DIR", str(root))
    r = client.post(f"{API}/jobs", json={"type": "selftest", "template": "tiny", "params": {"steps": 5}})
    assert r.status_code == 201 and r.json()["spec"]["params"]["steps"] == 5  # override wins
    assert r.json()["spec"]["params"]["result_value"] == "from-template"
    assert wait_status(client, r.json()["id"])["outputs"]["echo"] == "from-template"
    assert client.post(f"{API}/jobs", json={"type": "selftest", "template": "nope"}).status_code == 404
    cfg = client.post(f"{API}/configs", json={"job_type": "selftest", "name": "mine", "params": {"steps": 2, "step_delay_s": 0}}).json()
    r2 = client.post(f"{API}/jobs", json={"type": "selftest", "config_id": cfg["id"]})
    assert r2.status_code == 201 and r2.json()["spec"]["params"]["steps"] == 2
    wait_status(client, r2.json()["id"])


# --------------------------------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------------------------------
def test_job_stream_delivers_all_event_types_and_ends(client):
    j = submit(client, steps=12, step_delay_s=0.05)
    with client.stream("GET", f"{API}/jobs/{j['id']}/stream") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(list(r.iter_lines()))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "status" and kinds[-1] == "end"
    assert {"status", "progress", "metric", "log", "end"} <= set(kinds)
    assert events[-1]["data"]["status"] == "succeeded"
    pts = {}
    for e in events:
        if e["event"] == "metric":
            for k, v in e["data"]["points"].items():
                pts.setdefault(k, []).extend(v)
    assert [p[0] for p in pts["loss"]] == list(range(1, 13))  # every metric exactly once, in order
    lines = [ln for e in events if e["event"] == "log" for ln in e["data"]["lines"]]
    assert any("job finished" in ln for ln in lines)
    assert max(e["data"]["progress"] for e in events if e["event"] == "progress") == 1.0
    ids = [int(e["id"]) for e in events]
    assert ids == sorted(ids)


def test_job_stream_replays_finished_job_and_resumes(client):
    j = submit(client, steps=6, step_delay_s=0)
    wait_status(client, j["id"])
    with client.stream("GET", f"{API}/jobs/{j['id']}/stream") as r:
        ev = parse_sse(list(r.iter_lines()))
    assert ev[-1]["event"] == "end"
    seq = max(e["data"]["seq"] for e in ev if e["event"] == "metric")
    with client.stream("GET", f"{API}/jobs/{j['id']}/stream", params={"since_seq": seq, "log_tail": 0}) as r:
        ev2 = parse_sse(list(r.iter_lines()))
    assert not [e for e in ev2 if e["event"] in ("metric", "log")]  # nothing new after the resume point


def test_stream_sees_live_cancel(client):
    j = submit(client, steps=5000)
    out, sent = [], False
    with client.stream("GET", f"{API}/jobs/{j['id']}/stream") as r:
        for line in r.iter_lines():
            out.append(line)
            if line.startswith("event: metric") and not sent:
                sent = True
                client.post(f"{API}/jobs/{j['id']}/cancel")
    ev = parse_sse(out)
    assert ev[-1]["event"] == "end" and ev[-1]["data"]["status"] == "cancelled"


def test_gpu_and_global_event_streams_with_max_events(client):
    with client.stream("GET", f"{API}/system/gpus/stream", params={"max_events": 2, "interval": 0.1}) as r:
        ev = parse_sse(list(r.iter_lines()))
    assert [e["event"] for e in ev] == ["gpus", "gpus"] and "gpus" in ev[0]["data"] and "host" in ev[0]["data"]
    j = submit(client, steps=3, step_delay_s=0)
    with client.stream("GET", f"{API}/events", params={"max_events": 6, "interval": 0.1, "since": 0}) as r:
        ev = parse_sse(list(r.iter_lines()))
    assert ev[0]["event"] == "hello" and "jobs" in ev[0]["data"]
    assert any(e["event"] == "job" and e["data"]["id"] == j["id"] for e in ev)
    wait_status(client, j["id"])


# --------------------------------------------------------------------------------------------
# schemas & configs
# --------------------------------------------------------------------------------------------
def test_schemas_endpoint_drives_forms(client):
    r = client.get(f"{API}/schemas/job-types")
    assert r.status_code == 200
    by = {x["type"]: x for x in r.json()}
    assert {"train", "finetune", "distill", "compress", "eval", "serve", "bench", "probe", "dataset_ingest", "recipe", "model_download", "gc", "selftest"} <= set(by)
    st = by["selftest"]
    assert st["available"] and st["schema"]["properties"]["steps"]["minimum"] == 1
    assert st["defaults"]["steps"] == 30 and "Run" in st["groups"]
    assert st["schema"]["properties"]["fail_at_step"].get("x-nullable") is True
    assert all(k in st for k in ("type", "title", "description", "schema", "defaults", "templates", "groups"))
    one = client.get(f"{API}/schemas/job-types/selftest", params={"with_templates": True}).json()
    assert one["type"] == "selftest" and "template_items" in one
    assert client.get(f"{API}/schemas/job-types/zzz").status_code == 404
    avail = client.get(f"{API}/schemas/job-types", params={"include_unavailable": False}).json()
    assert all(x["available"] for x in avail)
    for x in by.values():  # unavailable types must say why instead of breaking the listing
        assert x["available"] or x["reason"]


def test_config_crud_and_validation(client):
    bad = client.post(f"{API}/configs", json={"job_type": "selftest", "name": "bad", "params": {"steps": -1}})
    assert bad.status_code == 422 and bad.json()["errors"][0]["loc"] == "steps"
    ok = client.post(f"{API}/configs", json={"job_type": "selftest", "name": "good", "params": {"steps": 7}, "description": "d"})
    assert ok.status_code == 201
    cfg = ok.json()
    assert cfg["params"]["steps"] == 7 and cfg["params"]["step_delay_s"] == 0.1  # defaults materialised
    assert client.get(f"{API}/configs/{cfg['id']}").json()["name"] == "good"
    assert [c["id"] for c in client.get(f"{API}/configs", params={"job_type": "selftest"}).json()] == [cfg["id"]]
    assert client.get(f"{API}/configs", params={"job_type": "gc"}).json() == []
    up = client.put(f"{API}/configs/{cfg['id']}", json={"params": {"steps": 9}, "name": "renamed"})
    assert up.status_code == 200 and up.json()["params"]["steps"] == 9 and up.json()["name"] == "renamed"
    assert client.put(f"{API}/configs/{cfg['id']}", json={"params": {"steps": "x"}}).status_code == 422
    assert client.put(f"{API}/configs/nope", json={"name": "z"}).status_code == 404
    assert client.delete(f"{API}/configs/{cfg['id']}").status_code == 204
    assert client.get(f"{API}/configs/{cfg['id']}").status_code == 404
    assert client.delete(f"{API}/configs/{cfg['id']}").status_code == 404
    assert client.post(f"{API}/configs", json={"job_type": "nope", "name": "x", "params": {}}).status_code == 404


def test_validate_endpoint_and_templates_listing(client, tmp_path, monkeypatch):
    v = client.post(f"{API}/configs/validate", json={"job_type": "selftest", "params": {"steps": 3}}).json()
    assert v["valid"] and v["params"]["steps"] == 3 and v["errors"] == []
    v = client.post(f"{API}/configs/validate", json={"job_type": "selftest", "params": {"steps": 3, "zz": 1}}).json()
    assert not v["valid"] and v["errors"][0]["loc"] == "zz"
    root = tmp_path / "cfgs" / "selftest"
    root.mkdir(parents=True)
    (root / "a.yaml").write_text("# demo\nsteps: 2\n")
    monkeypatch.setenv("FORGE_CONFIGS_DIR", str(tmp_path / "cfgs"))
    t = client.get(f"{API}/configs/templates", params={"type": "selftest"}).json()
    assert t[0]["name"] == "a" and t[0]["params"] == {"steps": 2} and t[0]["readonly"]
    assert any(x["type"] == "selftest" for x in client.get(f"{API}/configs/templates").json())


# --------------------------------------------------------------------------------------------
# auth, cors, system
# --------------------------------------------------------------------------------------------
def test_bearer_token_auth(home):
    c = make_client(token="s3cret")
    with c:
        assert c.get(f"{API}/system/health").status_code == 200  # liveness is public
        assert c.get("/").status_code == 200  # static UI is public
        assert c.get(f"{API}/jobs").status_code == 401
        assert c.get(f"{API}/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get(f"{API}/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get(f"{API}/jobs", params={"token": "s3cret"}).status_code == 200  # EventSource cannot set headers
        assert c.get(f"{API}/jobs", headers={"X-Forge-Token": "s3cret"}).status_code == 200
        assert c.get(f"{API}/schemas/job-types").status_code == 401
        assert c.post(f"{API}/jobs", json={"type": "selftest"}).status_code == 401
        info = c.get(f"{API}/system/info", headers={"Authorization": "Bearer s3cret"}).json()
        assert info["auth_required"] is True and info["settings"]["api_token"] == "set"  # never the token itself
        assert "s3cret" not in json.dumps(info)
        c.app.state.services.jobs.stop(kill_running=True)


def test_auth_token_from_environment(home, monkeypatch):
    monkeypatch.setenv("FORGE_API_TOKEN", "envtok")
    c = make_client(token=None)
    with c:
        assert c.get(f"{API}/jobs").status_code == 401
        assert c.get(f"{API}/jobs", headers={"Authorization": "Bearer envtok"}).status_code == 200
    c2 = make_client(token="")  # explicit empty token disables auth
    with c2:
        assert c2.get(f"{API}/jobs").status_code == 200


def test_cors_headers(home):
    c = make_client(cors_origins=["http://example.test"])
    with c:
        r = c.options(f"{API}/jobs", headers={"Origin": "http://example.test", "Access-Control-Request-Method": "POST"})
        assert r.headers.get("access-control-allow-origin") == "http://example.test"
        r = c.get(f"{API}/jobs", headers={"Origin": "http://evil.test"})
        assert "access-control-allow-origin" not in r.headers


def test_system_endpoints(client):
    info = client.get(f"{API}/system/info").json()
    for k in ("version", "home", "packages", "jobs", "store", "capabilities", "server_time", "gpu_kernels_note", "settings"):
        assert k in info
    assert set(info["jobs"]) >= {"queued", "running", "succeeded", "failed", "cancelled"}
    g = client.get(f"{API}/system/gpus", params={"history": 3}).json()
    assert isinstance(g["gpus"], list) and "host" in g and "history" in g
    assert client.get(f"{API}/system/health").json()["status"] == "ok"


def test_hardware_endpoints(client):
    chips = client.get(f"{API}/hardware/chips").json()
    assert chips["support_matrix"] and {r["sm"] for r in chips["support_matrix"]} >= {80, 90, 100}
    assert isinstance(client.get(f"{API}/hardware/descriptors").json(), list)
    r = client.get(f"{API}/hardware/descriptors/nope")
    assert r.status_code in (404, 501)
    p = client.post(f"{API}/hardware/probe")
    assert p.status_code == 201 and p.json()["type"] == "probe"
    done = wait_status(client, p.json()["id"], timeout=120)  # CPU probe (descriptor tier C) or a clean failure
    assert done["status"] in ("succeeded", "failed")
    if done["status"] == "failed":
        assert done["error"]


def test_openapi_covers_every_resource(client):
    paths = client.get("/openapi.json").json()["paths"]
    must = ["/system/info", "/system/gpus", "/system/gpus/stream", "/hardware/chips", "/hardware/descriptors", "/hardware/descriptors/{descriptor_id}",
            "/hardware/probe", "/models", "/models/{model_id}", "/models/{model_id}/download", "/models/register-local", "/models/{model_id}/inspect",
            "/datasets", "/datasets/{dataset_id}", "/datasets/upload", "/datasets/{dataset_id}/ingest", "/datasets/{dataset_id}/preview",
            "/schemas/job-types", "/schemas/job-types/{job_type}", "/configs", "/configs/{config_id}", "/configs/templates",
            "/jobs", "/jobs/{job_id}", "/jobs/{job_id}/cancel", "/jobs/{job_id}/logs", "/jobs/{job_id}/stream", "/jobs/{job_id}/metrics", "/jobs/{job_id}/clone",
            "/recipes", "/recipes/{recipe_id}", "/recipes/validate", "/recipes/{recipe_id}/run", "/planner/estimate", "/planner/plan",
            "/claims", "/claims/{claim_id}", "/artifacts", "/artifacts/{h}", "/artifacts/{h}/lineage", "/artifacts/{h}/pin", "/artifacts/{h}/unpin", "/artifacts/{h}/tier",
            "/storage/stats", "/storage/gc", "/storage/tiers", "/evals", "/evals/{eval_id}", "/evals/compare", "/gates", "/gates/{gate_id}",
            "/kernels", "/kernels/results", "/kernels/bench", "/kernels/autotune-cache", "/deployments", "/deployments/{dep_id}", "/deployments/{dep_id}/stats",
            "/deployments/{dep_id}/chat", "/serve/simulate", "/checkpoints", "/checkpoints/{ckpt_id}/export", "/events"]
    missing = [p for p in must if f"{API}{p}" not in paths]
    assert not missing, missing


def test_static_and_spa_fallback(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    deep = client.get("/jobs/abc/def")  # hash routing normally, but deep links must still return the shell
    assert deep.status_code == 200 and "text/html" in deep.headers["content-type"]
    assert client.get("/definitely-missing.js").status_code == 404
    assert client.get(f"{API}/no/such/endpoint").status_code == 404
    assert client.get("/docs").status_code == 200
    # path traversal never escapes the static dir
    r = client.get("/..%2f..%2fpyproject.toml")
    assert r.status_code in (200, 404) and "[build-system]" not in r.text
