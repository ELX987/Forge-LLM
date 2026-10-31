"""REST API: models, datasets (upload), recipes, deployments (proxy), storage, planner delegation, evals, checkpoints."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, make_client, wait_status  # noqa: E402,F401

API = "/api/v1"
HF_CFG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
          "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100, "max_position_embeddings": 256, "torch_dtype": "bfloat16"}


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "tiny-llama"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(HF_CFG))
    (d / "model.safetensors").write_bytes(b"\0" * 2048)
    (d / "tokenizer.json").write_text("{}")
    (d / "README.md").write_text("---\nlicense: apache-2.0\n---\n# tiny\n")
    return d


# --------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------
def test_register_local_model_and_inspect(client, model_dir):
    r = client.post(f"{API}/models/register-local", json={"path": str(model_dir), "name": "tiny"})
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["name"] == "tiny" and m["source"] == "local" and m["downloaded"] and m["licence"] == "apache-2.0"
    assert m["params_b"] and m["size_bytes"] == 2048
    # re-registering the same path is idempotent
    assert client.post(f"{API}/models/register-local", json={"path": str(model_dir)}).json()["id"] == m["id"]
    info = client.get(f"{API}/models/{m['id']}/inspect").json()
    assert info["model_type"] == "llama" and info["layers"] == 2 and info["kv_heads"] == 2 and info["has_tokenizer"]
    kv16 = info["kv_bytes_per_token"]["16"]
    assert kv16 == 2 * 2 * 2 * 16 * 2  # 2 (K,V) * layers * kv_heads * head_dim * 2 bytes
    assert info["kv_bytes_per_token"]["8"] == kv16 / 2 or info["kv_bytes_per_token"]["8"] < kv16
    assert info["licence_hint"]["distill"] == "likely_ok"
    # patch / list / get
    p = client.patch(f"{API}/models/{m['id']}", json={"notes": "n", "distill_allowed": True, "tags": ["a"]}).json()
    assert p["notes"] == "n" and p["distill_allowed"] is True and p["tags"] == ["a"]
    assert [x["id"] for x in client.get(f"{API}/models", params={"q": "tiny"}).json()] == [m["id"]]
    assert client.get(f"{API}/models", params={"q": "zzz"}).json() == []
    # delete (never deletes files outside the managed models dir)
    assert client.delete(f"{API}/models/{m['id']}", params={"delete_files": True}).status_code == 400
    assert model_dir.exists()
    assert client.delete(f"{API}/models/{m['id']}").status_code == 204
    assert client.get(f"{API}/models/{m['id']}").status_code == 404


def test_register_local_errors(client, tmp_path):
    assert client.post(f"{API}/models/register-local", json={"path": str(tmp_path / "missing")}).status_code == 404
    empty = tmp_path / "empty"
    empty.mkdir()
    r = client.post(f"{API}/models/register-local", json={"path": str(empty)})
    assert r.status_code == 400 and "config.json" in r.json()["message"]
    (empty / "config.json").write_text("{not json")
    assert client.post(f"{API}/models/register-local", json={"path": str(empty)}).status_code == 400


def test_hf_model_entry_and_download_job(client):
    assert client.post(f"{API}/models", json={"name": "x", "source": "hf", "path_or_repo": "noslash"}).status_code == 400
    r = client.post(f"{API}/models", json={"name": "llama", "source": "hf", "path_or_repo": "org/llama-8b", "licence": "llama3"})
    assert r.status_code == 201 and r.json()["status"] == "remote" and not r.json()["downloaded"]
    mid = r.json()["id"]
    assert client.post(f"{API}/models", json={"name": "dup", "source": "hf", "path_or_repo": "org/llama-8b"}).json()["id"] == mid
    insp = client.get(f"{API}/models/{mid}/inspect")
    assert insp.status_code == 400 and "not downloaded" in insp.json()["message"]
    d = client.post(f"{API}/models/{mid}/download", json={"allow_patterns": ["*.json"]})
    assert d.status_code == 201
    job = d.json()
    assert job["type"] == "model_download" and job["spec"]["params"]["repo_id"] == "org/llama-8b"
    assert job["spec"]["params"]["model_id"] == mid and job["spec"]["params"]["licence"] == "llama3"
    assert client.get(f"{API}/models/{mid}").json()["status"] == "downloading"
    client.post(f"{API}/jobs/{job['id']}/cancel")  # no network in tests: stop it
    wait_status(client, job["id"], timeout=30)
    # only hf models can be downloaded
    local = client.post(f"{API}/models", json={"name": "a", "source": "artifact", "path_or_repo": "blake3:abc"}).json()
    assert client.post(f"{API}/models/{local['id']}/download").status_code == 400


def test_licence_hint_heuristics():
    from forge.api.routers.models import licence_hint

    assert licence_hint("apache-2.0")["distill"] == "likely_ok"
    assert licence_hint("llama3.1")["distill"] == "check_terms"
    assert licence_hint(None)["distill"] == "unknown" and licence_hint("weird")["distill"] == "unknown"


# --------------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------------
def test_register_dataset_in_place_and_preview(client, tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text("\n".join(json.dumps({"text": f"doc {i}", "i": i}) for i in range(20)) + "\n")
    r = client.post(f"{API}/datasets", json={"name": "d1", "path": str(f)})
    assert r.status_code == 201 and r.json()["format"] == "jsonl" and r.json()["size_bytes"] == f.stat().st_size and r.json()["num_files"] == 1
    did = r.json()["id"]
    pv = client.get(f"{API}/datasets/{did}/preview", params={"n": 3}).json()
    assert [x["text"] for x in pv["records"]] == ["doc 0", "doc 1", "doc 2"]
    d = client.get(f"{API}/datasets/{did}").json()
    assert d["files"] == [{"name": "data.jsonl", "size": f.stat().st_size}]
    # uploads into a registered-in-place dataset are refused
    r = client.put(f"{API}/datasets/{did}/content", params={"filename": "x.txt"}, content=b"abc")
    assert r.status_code == 400
    assert client.delete(f"{API}/datasets/{did}").status_code == 204
    assert f.exists()  # registered-in-place data is never deleted
    assert client.post(f"{API}/datasets", json={"name": "bad", "path": str(tmp_path / "nope")}).status_code == 404


def test_multipart_upload_whole_and_chunked(client):
    r = client.post(f"{API}/datasets/upload", files={"file": ("notes.txt", b"para one\n\npara two\n\npara three", "text/plain")}, data={"name": "notes"})
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["name"] == "notes" and ds["status"] == "ready" and ds["size_bytes"] == 30 and ds["format"] == "text"
    pv = client.get(f"{API}/datasets/{ds['id']}/preview", params={"n": 2}).json()
    assert pv["records"] == ["para one", "para two"]

    # chunked: first chunk creates the dataset, later chunks append at the right offset
    payload = b"".join(json.dumps({"k": i}).encode() + b"\n" for i in range(100))
    c1, c2, c3 = payload[:300], payload[300:700], payload[700:]
    r1 = client.post(f"{API}/datasets/upload", files={"file": ("big.jsonl", c1)}, data={"name": "big", "offset": "0", "final": "false"})
    assert r1.status_code == 201 and r1.json()["status"] == "uploading"
    did = r1.json()["id"]
    bad = client.post(f"{API}/datasets/upload", files={"file": ("big.jsonl", c2)}, data={"dataset_id": did, "filename": "big.jsonl", "offset": "999", "final": "false"})
    assert bad.status_code == 409 and bad.json()["details"]["expected_offset"] == 300
    r2 = client.post(f"{API}/datasets/upload", files={"file": ("big.jsonl", c2)}, data={"dataset_id": did, "filename": "big.jsonl", "offset": "300", "final": "false"})
    assert r2.status_code == 201 and r2.json()["file_size"] == 700
    r3 = client.post(f"{API}/datasets/upload", files={"file": ("big.jsonl", c3)}, data={"dataset_id": did, "filename": "big.jsonl", "offset": "700", "final": "true"})
    assert r3.json()["status"] == "ready" and r3.json()["size_bytes"] == len(payload)
    pv = client.get(f"{API}/datasets/{did}/preview", params={"n": 100}).json()
    assert [x["k"] for x in pv["records"]] == list(range(100))  # reassembled byte-exactly
    got = Path(client.get(f"{API}/datasets/{did}").json()["path"]) / "big.jsonl"
    assert got.read_bytes() == payload
    # deleting a managed dataset removes its files
    assert client.delete(f"{API}/datasets/{did}").status_code == 204
    assert not got.exists()


def test_upload_limits_and_filename_safety(home):
    c = make_client()
    with c:
        c.app.state.services.max_upload_bytes = 1000
        r = c.post(f"{API}/datasets/upload", files={"file": ("big.bin", b"x" * 5000)}, data={"name": "toobig"})
        assert r.status_code == 413 and r.json()["error"] == "payload_too_large"
        assert c.get(f"{API}/datasets").json() == []  # nothing left behind by the failed upload
        ok = c.post(f"{API}/datasets/upload", files={"file": ("../../evil.txt", b"hello")}, data={"name": "safe"})
        assert ok.status_code == 201
        d = Path(c.get(f"{API}/datasets/{ok.json()['id']}").json()["path"])
        assert [p.name for p in d.iterdir()] == ["evil.txt"]  # path components stripped
        bad = c.post(f"{API}/datasets/upload", files={"file": (".hidden", b"x")}, data={"name": "h"})
        assert bad.status_code == 400
        c.app.state.services.jobs.stop(kill_running=True)


def test_raw_put_upload(client):
    ds = client.post(f"{API}/datasets", json={"name": "raw"}).json()
    assert ds["status"] == "empty"
    r = client.put(f"{API}/datasets/{ds['id']}/content", params={"filename": "a.csv", "offset": 0, "final": False}, content=b"x,y\n1,2\n")
    assert r.status_code == 200 and r.json()["size_bytes"] == 8 and r.json()["status"] == "uploading"
    r = client.put(f"{API}/datasets/{ds['id']}/content", params={"filename": "a.csv", "offset": 8}, content=b"3,4\n")
    assert r.json()["status"] == "ready" and r.json()["format"] == "csv"
    pv = client.get(f"{API}/datasets/{ds['id']}/preview").json()
    assert pv["records"] == [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]
    assert client.put(f"{API}/datasets/{ds['id']}/content", params={"filename": "a.csv", "offset": 3}, content=b"z").status_code == 409


def test_preview_parquet_and_json(client, tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    t = pa.table({"text": [f"row {i}" for i in range(10)], "n": list(range(10))})
    f = tmp_path / "t.parquet"
    pq.write_table(t, str(f))
    did = client.post(f"{API}/datasets", json={"name": "pq", "path": str(f)}).json()["id"]
    pv = client.get(f"{API}/datasets/{did}/preview", params={"n": 2}).json()
    assert pv["num_rows"] == 10 and pv["columns"] == ["text", "n"] and pv["records"][1] == {"text": "row 1", "n": 1}
    j = tmp_path / "x.json"
    j.write_text(json.dumps([{"a": 1}, {"a": 2}, {"a": 3}]))
    jid = client.post(f"{API}/datasets", json={"name": "js", "path": str(j)}).json()["id"]
    assert client.get(f"{API}/datasets/{jid}/preview", params={"n": 2}).json()["records"] == [{"a": 1}, {"a": 2}]
    long = tmp_path / "long.jsonl"
    long.write_text(json.dumps({"t": "z" * 5000}) + "\n")
    lid = client.post(f"{API}/datasets", json={"name": "long", "path": str(long)}).json()["id"]
    assert "chars)" in client.get(f"{API}/datasets/{lid}/preview").json()["records"][0]["t"]  # long values truncated


def test_ingest_endpoint_fills_dataset_path(client, tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("hello world")
    did = client.post(f"{API}/datasets", json={"name": "ing", "path": str(f)}).json()["id"]
    r = client.post(f"{API}/datasets/{did}/ingest", json={"params": {}})
    if r.status_code == 501:  # forge.train.data not importable on this host
        assert r.json()["error"] == "module_unavailable"
        return
    # the job may fail validation (required fields we cannot guess) -> field-level error; or be created
    assert r.status_code in (201, 422)
    if r.status_code == 201:
        assert r.json()["type"] == "dataset_ingest"
        assert client.get(f"{API}/datasets/{did}").json()["meta"]["ingest_job_id"] == r.json()["id"]
        client.post(f"{API}/jobs/{r.json()['id']}/cancel")


# --------------------------------------------------------------------------------------------
# recipes
# --------------------------------------------------------------------------------------------
RECIPE = "recipe: demo\ninputs: {x: 1}\nstages:\n  - jobs_api_test_stage: {k: 2}\n"


def test_recipe_validate_crud_and_run(client):
    from forge.core import registry

    v = client.post(f"{API}/recipes/validate", json={"yaml": RECIPE}).json()
    assert not v["valid"] and "not registered" in v["problems"][0] and v["recipe"]["name"] == "demo" and v["hash"]
    registry.register_obj("stage", "jobs_api_test_stage", lambda inputs, params, ctx: {})
    try:
        v = client.post(f"{API}/recipes/validate", json={"yaml": RECIPE}).json()
        assert v["valid"] and v["problems"] == [] and "jobs_api_test_stage" in v["yaml"]
        v2 = client.post(f"{API}/recipes/validate", json={"json": {"name": "j", "stages": [{"name": "jobs_api_test_stage", "params": {}}]}}).json()
        assert v2["valid"]
    finally:
        registry.unregister("stage", "jobs_api_test_stage")
    bad = client.post(f"{API}/recipes/validate", json={"yaml": "recipe: [unclosed"}).json()
    assert not bad["valid"] and bad["errors"][0]["msg"]
    assert client.post(f"{API}/recipes/validate", json={}).status_code == 400

    c = client.post(f"{API}/recipes", json={"yaml": RECIPE, "description": "d"})
    assert c.status_code == 201 and c.json()["name"] == "demo" and c.json()["recipe_hash"]
    rid = c.json()["id"]
    assert client.post(f"{API}/recipes", json={"yaml": "stages: ["}).status_code == 400
    g = client.get(f"{API}/recipes/{rid}").json()
    assert g["parsed"]["stages"][0]["name"] == "jobs_api_test_stage" and g["problems"]  # unknown stage reported, still stored
    u = client.put(f"{API}/recipes/{rid}", json={"name": "renamed", "yaml": RECIPE.replace("k: 2", "k: 3")}).json()
    assert u["name"] == "renamed" and "k: 3" in u["yaml"]
    assert [r["id"] for r in client.get(f"{API}/recipes").json()] == [rid]
    run = client.post(f"{API}/recipes/{rid}/run", json={"force": True})
    assert run.status_code == 201 and run.json()["type"] == "recipe" and run.json()["spec"]["params"]["recipe_id"] == rid
    done = wait_status(client, run.json()["id"])
    assert done["status"] == "failed" and "not registered" in done["error"]  # the worker process has no such stage
    assert client.delete(f"{API}/recipes/{rid}").status_code == 204
    assert client.get(f"{API}/recipes/{rid}").status_code == 404
    assert isinstance(client.get(f"{API}/recipes/stages").json(), list)
    assert isinstance(client.get(f"{API}/recipes/templates").json(), list)


# --------------------------------------------------------------------------------------------
# deployments: serve job creation + chat/stats proxy against a fake engine
# --------------------------------------------------------------------------------------------
from fastapi import FastAPI, Request  # noqa: E402  (module level: FastAPI resolves string annotations from globals)
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402


def _fake_engine():
    app = FastAPI()
    seen: dict = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/stats")
    def stats():
        return {"running": 2, "waiting": 1, "kv_usage": 0.25, "tokens_per_s": 123.0}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        seen["body"] = body
        seen["auth"] = req.headers.get("authorization")
        if not body.get("stream"):
            return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

        async def gen():
            for tok in ("Hel", "lo", "!"):
                yield f'data: {json.dumps({"choices": [{"delta": {"content": tok}}]})}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.state.seen = seen
    return app


def test_deployment_chat_and_stats_proxy(client):
    import httpx

    engine = _fake_engine()
    svc = client.app.state.services
    svc.extras["engine_transport"] = httpx.ASGITransport(app=engine)
    svc.deployments.insert(id="dep_t", name="fake", artifact_ref="m", serve_config={"model": "m", "api_key": "k3y"}, port=8999,
                                 status="running", created_at=1.0, updated_at=1.0)
    st = client.get(f"{API}/deployments/dep_t/stats")
    assert st.status_code == 200 and st.json()["running"] == 2
    got = client.get(f"{API}/deployments/dep_t").json()
    assert got["url"] == "http://127.0.0.1:8999" and got["api_key"] == "set"  # the key itself is never returned
    assert "k3y" not in json.dumps(got)
    # streaming passthrough
    with client.stream("POST", f"{API}/deployments/dep_t/chat", json={"messages": [{"role": "user", "content": "hey"}], "temperature": 0.3}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert body.count("data: ") == 4 and body.rstrip().endswith("data: [DONE]")
    assert engine.state.seen["body"]["stream"] is True and engine.state.seen["body"]["temperature"] == 0.3
    assert engine.state.seen["body"]["model"] == "m" and engine.state.seen["auth"] == "Bearer k3y"
    # non-streaming
    j = client.post(f"{API}/deployments/dep_t/chat", json={"messages": [{"role": "user", "content": "hey"}], "stream": False}).json()
    assert j["choices"][0]["message"]["content"] == "hi"
    assert client.post(f"{API}/deployments/dep_t/chat", json={"prompt": "no messages"}).status_code == 400
    assert client.get(f"{API}/deployments/dep_t/health").json()["ok"] is True
    # unreachable engine -> 502, not a crash
    svc.extras["engine_transport"] = httpx.MockTransport(lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    r = client.get(f"{API}/deployments/dep_t/stats")
    assert r.status_code == 502 and r.json()["error"] == "engine_unreachable"
    assert client.delete(f"{API}/deployments/dep_t").status_code == 204
    assert client.get(f"{API}/deployments/dep_t").status_code == 404


def test_deployment_create_starts_serve_job(client, model_dir):
    reg = client.post(f"{API}/models/register-local", json={"path": str(model_dir)}).json()
    r = client.post(f"{API}/deployments", json={"name": "d", "artifact_ref": reg["id"], "serve_config": {"max_model_len": 512}})
    if r.status_code == 501:
        pytest.skip("forge.serve.config not importable")
    assert r.status_code == 201, r.text
    dep = r.json()
    assert dep["status"] == "starting" and 1024 <= dep["port"] <= 65535 and dep["job_id"]
    job = client.get(f"{API}/jobs/{dep['job_id']}").json()
    p = job["spec"]["params"]
    assert job["type"] == "serve" and p["model"] == str(model_dir.resolve()) and p["port"] == dep["port"] and p["max_model_len"] == 512
    # a second deployment gets a different port
    r2 = client.post(f"{API}/deployments", json={"artifact_ref": str(model_dir)}).json()
    assert r2["port"] != dep["port"]
    for d in (dep, r2):
        assert client.post(f"{API}/deployments/{d['id']}/stop").status_code == 200
    for d in (dep, r2):
        wait_status(client, d["job_id"], timeout=60)
    assert client.get(f"{API}/deployments/{dep['id']}").json()["status"] in ("stopped", "failed")
    assert client.post(f"{API}/deployments", json={"artifact_ref": "x", "serve_config": {"max_model_len": -1}}).status_code == 422


# --------------------------------------------------------------------------------------------
# storage / artifacts / checkpoints
# --------------------------------------------------------------------------------------------
def test_storage_and_artifacts(client):
    r = client.get(f"{API}/storage/stats")
    if r.status_code == 501:
        assert client.get(f"{API}/artifacts").status_code == 501
        return
    assert r.status_code == 200 and isinstance(r.json(), dict)
    from forge.jobs.repo import open_store

    st = open_store()
    h = st.put_bytes(b"hello artifact" * 100, kind="blob")
    arts = client.get(f"{API}/artifacts").json()
    assert any(a.get("hash") == h or a.get("id") == h for a in arts)
    a = client.get(f"{API}/artifacts/{h}").json()
    assert a["hash"] == h and a["size"] == 1400
    assert client.get(f"{API}/artifacts/blake3:" + "0" * 64).status_code == 404
    assert client.post(f"{API}/artifacts/{h}/pin", json={"reason": "t"}).json()["pinned"] is True
    assert client.post(f"{API}/artifacts/{h}/unpin").json()["pinned"] is False
    assert isinstance(client.get(f"{API}/artifacts/{h}/lineage").json(), dict)
    assert client.post(f"{API}/artifacts/{h}/tier", json={"to": "frozen"}).status_code == 400
    gc = client.post(f"{API}/storage/gc", json={"dry_run": True}).json()
    assert gc["dry_run"] is True and "report" in gc
    assert client.get(f"{API}/storage/tiers").status_code == 200
    real = client.post(f"{API}/storage/gc", json={"dry_run": False}).json()
    assert real["job"]["type"] == "gc"
    wait_status(client, real["job"]["id"])


def test_checkpoints_listing_and_export(client, home):
    ck = home / "runs" / "job_run1" / "checkpoints" / "step-50"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"\0" * 128)
    (ck / "config.json").write_text(json.dumps(HF_CFG))
    rows = client.get(f"{API}/checkpoints", params={"run": "job_run1"}).json()
    assert [(r["name"], r["source"], r["safetensors"]) for r in rows] == [("step-50", "fs", True)]
    assert rows[0]["id"] == "job_run1:step-50" and rows[0]["size_bytes"] > 128
    assert any(r["run"] == "job_run1" for r in client.get(f"{API}/checkpoints").json())
    ex = client.post(f"{API}/checkpoints/job_run1:step-50/export", json={"name": "exported"})
    assert ex.status_code == 201 and ex.json()["type"] == "checkpoint_export"
    done = wait_status(client, ex.json()["id"])
    assert done["status"] == "succeeded", done
    models = client.get(f"{API}/models").json()
    assert [m["name"] for m in models] == ["exported"] and models[0]["downloaded"]
    assert client.post(f"{API}/checkpoints/nope:1/export").status_code == 404


# --------------------------------------------------------------------------------------------
# planner / serve simulate / claims: delegation + graceful 501
# --------------------------------------------------------------------------------------------
def test_planner_delegates_with_flexible_signature(client, monkeypatch):
    import forge.planner as planner

    calls = {}

    def estimate(model_cfg, chip, quant_scheme="bf16", workload=None, kv_dtype="bf16"):
        calls.update(model_cfg=model_cfg, chip=chip, quant=quant_scheme, workload=workload, kv=kv_dtype)
        return {"fits": True, "bound": "memory", "params": model_cfg.param_count()}

    monkeypatch.setattr(planner, "estimate", estimate, raising=False)
    r = client.post(f"{API}/planner/estimate", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g", "quant": "w4a16", "kv_dtype": "fp8_e4m3",
                                                     "workload": {"context_lengths": [4096]}})
    if r.status_code == 501 and "forge.models" in r.json()["message"]:
        pytest.skip("forge.models unavailable")
    assert r.status_code == 200, r.text
    assert r.json()["fits"] is True and calls["quant"] == "w4a16" and calls["kv"] == "fp8_e4m3"
    assert calls["workload"].context_lengths == [4096]  # validated into forge.core.schemas.Workload
    assert getattr(calls["chip"], "sm", None) == 90  # catalogue key -> Capabilities
    # invalid input is reported as 400/404, not 500
    assert client.post(f"{API}/planner/estimate", json={"chip": "h100-sxm5-80g"}).status_code == 400
    assert client.post(f"{API}/planner/estimate", json={"model": "nope", "chip": "h100-sxm5-80g"}).status_code == 404
    assert client.post(f"{API}/planner/estimate", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g", "workload": {"bogus": 1}}).status_code == 400

    def plan(model, chip, workload, budget):
        return [{"recipe": "w4a16-gptq", "budget": budget}]

    monkeypatch.setattr(planner, "plan", plan, raising=False)  # package attribute wins over the lazy forge.planner.search.plan
    p = client.post(f"{API}/planner/plan", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g", "budget": {"gpu_hours": 4}})
    assert p.status_code == 200 and p.json()[0]["budget"] == {"gpu_hours": 4}


def test_planner_missing_function_is_501(client, monkeypatch):
    import forge.planner as planner

    monkeypatch.setattr(planner, "estimate", None, raising=False)  # simulate a planner without estimate()
    r = client.post(f"{API}/planner/estimate", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g"})
    assert r.status_code == 501 and r.json()["error"] == "module_unavailable"


def test_serve_simulate_uses_capacity_planner(client):
    r = client.post(f"{API}/serve/simulate", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g", "weight_bits": 4, "kv_dtype": "fp8_e4m3",
                                                    "context_dist": {"mean_ctx": 2048}})
    if r.status_code == 501:
        pytest.skip("capacity planner unavailable")
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), dict) and r.json()
    assert client.post(f"{API}/serve/simulate", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g", "context_dist": {"nope": 1}}).status_code == 400


def test_claims_delegation(client, monkeypatch):
    import types

    mod = types.ModuleType("forge.planner.claims")
    store = {"C1": {"id": "C1", "statement": "FP8 KV is lossless", "source_paper": "P3", "status": "untested"}}
    mod.list_claims = lambda status=None, q=None: list(store.values())
    mod.get_claim = lambda cid: store.get(cid)

    def update_claim(claim_id, status=None, notes=None, linked_experiments=None):
        if claim_id not in store:
            raise KeyError(claim_id)
        if status:
            store[claim_id]["status"] = status
        return store[claim_id]

    mod.update_claim = update_claim
    monkeypatch.setitem(sys.modules, "forge.planner.claims", mod)
    assert client.get(f"{API}/claims").json()[0]["id"] == "C1"
    assert client.get(f"{API}/claims", params={"status": "reproduced"}).json() == []
    assert client.get(f"{API}/claims", params={"q": "lossless"}).json()[0]["id"] == "C1"
    assert client.get(f"{API}/claims/C1").json()["statement"].startswith("FP8")
    assert client.get(f"{API}/claims/ZZ").status_code == 404
    up = client.put(f"{API}/claims", json={"id": "C1", "status": "partial", "notes": "n"})
    assert up.status_code == 200 and up.json()["status"] == "partial"
    assert client.put(f"{API}/claims", json={"id": "ZZ", "status": "partial"}).status_code == 404


# --------------------------------------------------------------------------------------------
# evals / gates / kernels
# --------------------------------------------------------------------------------------------
def test_pareto_and_distinguishable_helpers():
    from forge.api.routers.evals import _distinguishable, pareto_front

    pts = [(1.0, 0.5), (2.0, 0.9), (3.0, 0.8), (1.5, 0.5)]  # minimise x, maximise y
    assert pareto_front(pts, x_min=True, y_min=False) == [True, True, False, False]
    a = {"ci_low": 0.1, "ci_high": 0.2}
    assert _distinguishable(a, {"ci_low": 0.3, "ci_high": 0.4}) is True
    assert _distinguishable(a, {"ci_low": 0.15, "ci_high": 0.4}) is False
    assert _distinguishable(a, {"ci_low": None, "ci_high": None}) is None


def test_evals_endpoints_with_stored_reports(client):
    mod = pytest.importorskip("forge.eval.db")
    from forge.core.schemas import EvalReport, MetricValue, Provenance

    def rep(i, art, val, lo, hi):
        return EvalReport(id=i, artifact_hash=art, parent_hash="blake3:" + "0" * 64, suite_versions={}, noise_floor={},
                          metrics={"suite_score": MetricValue(value=val, ci_low=lo, ci_high=hi, n=100, provenance=Provenance.measured)})

    mod.save_report(rep("ev_a", "blake3:" + "a" * 64, 0.80, 0.78, 0.82), bytes=100.0, candidate_label="A")
    mod.save_report(rep("ev_b", "blake3:" + "b" * 64, 0.60, 0.58, 0.62), bytes=50.0, candidate_label="B")
    lst = client.get(f"{API}/evals").json()
    assert {e["id"] for e in lst} == {"ev_a", "ev_b"}
    one = client.get(f"{API}/evals/ev_a").json()
    assert one["id"] == "ev_a" and one["_metrics"]["suite_score"]["value"] == 0.8
    assert client.get(f"{API}/evals/nope").status_code == 404
    cmp_ = client.post(f"{API}/evals/compare", json={"ids": ["ev_a", "ev_b"], "parent": "ev_a"})
    assert cmp_.status_code == 200, cmp_.text
    assert cmp_.json()["rows"] and "pareto" in cmp_.json()
    assert client.post(f"{API}/evals/compare", json={"ids": ["ev_a", "zzz"]}).status_code == 404


def test_evals_compare_fallback_implementation(client, monkeypatch):
    """The API's own comparison (used when forge.eval is missing) on synthetic rows."""
    from forge.api.routers import evals as ev

    def row(**metrics):
        return {"report": {"metrics": metrics}, "gate": "pass"}

    rows = {
        "fa": row(acc={"value": 0.9, "ci_low": 0.88, "ci_high": 0.92}, size={"value": 10.0}),
        "fb": row(acc={"value": 0.7, "ci_low": 0.68, "ci_high": 0.72}, size={"value": 5.0}),
        "fc": row(acc={"value": 0.69, "ci_low": 0.5, "ci_high": 0.9}, size={"value": 8.0}),
    }
    monkeypatch.setattr(ev, "_eval_mod", lambda: None)
    monkeypatch.setattr(ev, "_find_eval", lambda db, i: rows.get(i))
    out = client.post(f"{API}/evals/compare", json={"ids": ["fa", "fb", "fc"], "parent": "fa", "pareto": {"x": "size", "y": "acc", "x_min": True, "y_min": False}}).json()
    acc = next(r for r in out["rows"] if r["metric"] == "acc")
    assert acc["cells"]["fb"]["delta"] == pytest.approx(-0.2) and acc["cells"]["fb"]["distinguishable"] is True
    assert acc["cells"]["fc"]["distinguishable"] is False  # wide CI overlaps the parent's
    front = {p["id"]: p["on_front"] for p in out["pareto"]["points"]}
    assert front == {"fa": True, "fb": True, "fc": False}
    assert out["gates"]["fa"] == "pass"


def test_gates_from_config_files(client, tmp_path, monkeypatch):
    g = tmp_path / "cfgs" / "gates"
    g.mkdir(parents=True)
    (g / "g-test.yaml").write_text("id: g-test\ndescription: demo gate\nchecks: []\n")
    monkeypatch.setenv("FORGE_CONFIGS_DIR", str(tmp_path / "cfgs"))
    lst = client.get(f"{API}/gates").json()
    assert any(x["id"] == "g-test" and x["description"] == "demo gate" for x in lst)
    one = client.get(f"{API}/gates/g-test").json()
    assert one["definition"]["id"] == "g-test" and "demo gate" in one["yaml"]
    assert client.get(f"{API}/gates/missing").status_code == 404


def test_kernels_endpoints(client):
    k = client.get(f"{API}/kernels").json()
    assert set(k) >= {"kernels", "available", "gpu_note"} and "not been validated" in k["gpu_note"]
    assert client.get(f"{API}/kernels/results", params={"kernel": "w4a16"}).json() == []
    assert client.get(f"{API}/kernels/autotune-cache").json() == []
    from forge.core.paths import autotune_dir

    (autotune_dir() / "cache.json").write_text(json.dumps({"k": 1}))
    c = client.get(f"{API}/kernels/autotune-cache").json()
    assert c[0]["file"] == "cache.json" and c[0]["content"] == {"k": 1}
    b = client.post(f"{API}/kernels/bench", json={"params": {}})
    assert b.status_code in (201, 422, 501)  # BenchConfig may not exist / may require fields
    if b.status_code == 201:
        client.post(f"{API}/jobs/{b.json()['id']}/cancel")


def test_kernel_results_table(client):
    pytest.importorskip("forge.kernels.bench")
    db = client.app.state.services.metrics.db
    db.ensure_migrated()
    for kernel, chip, t, gbs in (("w4a16", "h100", 1.0, 1200.0), ("fp8", "h100", 2.0, 900.0), ("w4a16", "a100", 3.0, 700.0)):
        db.execute("INSERT INTO kernel_bench(kernel, chip, created_at, achieved_gbs, shape_json) VALUES(?,?,?,?,?)", (kernel, chip, t, gbs, "[1, 2]"))
    rows = client.get(f"{API}/kernels/results", params={"kernel": "w4a16"}).json()
    assert [r["chip"] for r in rows] == ["a100", "h100"]  # newest first
    assert rows[1]["shape"] == [1, 2] and rows[1]["achieved_gbs"] == 1200.0  # *_json columns decoded
    assert len(client.get(f"{API}/kernels/results", params={"chip": "h100"}).json()) == 2
    assert len(client.get(f"{API}/kernels/results", params={"kernel": "fp8", "chip": "a100"}).json()) == 0


def test_real_planner_and_claims_integration(client):
    """Against the real forge.planner (skipped when the sibling module is not importable)."""
    pytest.importorskip("forge.planner.claims")
    claims = client.get(f"{API}/claims")
    assert claims.status_code == 200, claims.text
    rows = claims.json()
    assert len(rows) >= 30 and {"id", "statement", "status"} <= set(rows[0])
    cid = rows[0]["id"]
    assert client.get(f"{API}/claims/{cid}").json()["id"] == cid
    up = client.put(f"{API}/claims", json={"id": cid, "status": "partial", "notes": "from api test"})
    assert up.status_code == 200 and up.json()["status"] == "partial"
    assert client.get(f"{API}/claims", params={"status": "partial"}).json()[0]["id"] == cid
    assert client.put(f"{API}/claims", json={"id": "nope-nope", "status": "partial"}).status_code == 404
    assert "summary" in client.get(f"{API}/claims/summary").json()

    est = client.post(f"{API}/planner/estimate", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g", "quant": "w4a16-g128", "kv_dtype": "fp8"})
    assert est.status_code == 200, est.text
    assert isinstance(est.json(), dict) and est.json()
    pl = client.post(f"{API}/planner/plan", json={"hf_config": HF_CFG, "chip": "h100-sxm5-80g", "budget": {}})
    assert pl.status_code == 200, pl.text


def test_source_preview_local_and_errors(client, home):
    tmp_path = home / "datasets"
    tmp_path.mkdir(parents=True, exist_ok=True)
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps({"body": f"doc {i}"}) for i in range(5)) + "\n")
    r = client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(f), "text_field": "body"}, "n": 2})
    assert r.status_code == 200, r.text
    assert r.json()["local"] and r.json()["samples"] == ["doc 0", "doc 1"] and r.json()["columns"] == ["body"]
    assert client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(f), "bogus": 1}}).status_code == 400
    assert client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(tmp_path / "nope")}}).status_code == 404
