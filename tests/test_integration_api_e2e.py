"""Integration: hit every REST endpoint of ARCHITECTURE section 7 with a real (tiny) model, run real jobs end to end
(selftest, probe, bench, dataset ingest, serve deployment) and assert there is no 5xx anywhere and the front-end
contract holds."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, wait_status  # noqa: E402,F401

API = "/api/v1"
WL = {"context_lengths": [4096], "batch_sizes": [1, 16]}
JSONL = b'{"text":"hello world foo bar"}\n' * 50


@pytest.fixture
def tiny_model_id(client, home):
    from forge.models import build_tiny_model, save_model

    md = home.parent / "tinymodel"
    save_model(build_tiny_model(0), md)
    r = client.post(f"{API}/models/register-local", json={"path": str(md), "name": "tiny"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def ok(r):
    assert r.status_code < 400, f"{r.request.method} {r.request.url}: {r.status_code} {r.text[:400]}"
    return r


def test_read_endpoints_no_5xx(client):
    for p in ["/system/health", "/system/info", "/system/gpus", "/hardware/chips", "/hardware/descriptors", "/models", "/datasets",
              "/schemas/job-types", "/configs", "/configs/templates?type=train", "/jobs", "/recipes", "/recipes/stages",
              "/recipes/templates", "/claims", "/claims/summary", "/artifacts", "/storage/stats", "/storage/tiers", "/evals", "/gates",
              "/kernels", "/kernels/results", "/kernels/autotune-cache", "/deployments", "/checkpoints",
              "/system/gpus/stream?max_events=1", "/events?max_events=1"]:
        ok(client.get(API + p))
    chips = client.get(f"{API}/hardware/chips").json()
    ok(client.get(f"{API}/hardware/chips/{chips['chips'][0]['key']}"))
    gates = client.get(f"{API}/gates").json()
    assert gates
    gid = gates[0]["id"] if isinstance(gates[0], dict) else gates[0]
    ok(client.get(f"{API}/gates/{gid}"))
    kernels = client.get(f"{API}/kernels").json()
    names = {k["name"] for k in (kernels["kernels"] if isinstance(kernels, dict) else kernels)}
    assert {"w4a16_gemm", "paged_decode_attention", "moe_route", "hadamard"} <= names


def test_schemas_for_every_job_type(client):
    items = ok(client.get(f"{API}/schemas/job-types")).json()
    assert {i["type"] for i in items} >= {"train", "finetune", "distill", "compress", "eval", "serve", "bench", "probe",
                                           "dataset_ingest", "recipe", "model_download", "gc", "selftest"}
    for i in items:
        if i["type"] in {"crasher", "selftest_gpu", "ghost"}:  # fault-injection types other tests register in-process
            continue
        assert i["available"], (i["type"], i["reason"])
        assert i["schema"]["properties"]
        ok(client.get(f"{API}/schemas/job-types/{i['type']}"))


def test_jobs_selftest_probe_bench(client):
    jid = ok(client.post(f"{API}/jobs", json={"type": "selftest", "name": "st", "params": {}})).json()["id"]
    j = wait_status(client, jid)
    assert j["status"] == "succeeded", j
    ok(client.get(f"{API}/jobs/{jid}/logs?tail=5"))
    assert ok(client.get(f"{API}/jobs/{jid}/metrics")).json()["points"]
    ok(client.get(f"{API}/jobs/{jid}/stream?max_events=3"))
    ok(client.post(f"{API}/jobs/{jid}/clone"))
    ok(client.get(f"{API}/jobs?status=succeeded&type=selftest&limit=5"))
    p = ok(client.post(f"{API}/hardware/probe", json={})).json()
    pj = wait_status(client, p["id"], timeout=90)
    assert pj["status"] == "succeeded", pj
    assert ok(client.get(f"{API}/hardware/descriptors")).json()
    b = ok(client.post(f"{API}/kernels/bench", json={"params": {"allow_cpu": True, "kernels": ["rmsnorm_quant"], "dtypes": ["bf16"],
                                                                  "shapes": [{"rows": 4, "dim": 64}], "repeats": 3, "warmup": 1}})).json()
    bj = wait_status(client, b["id"], timeout=90)
    assert bj["status"] == "succeeded", bj
    ok(client.delete(f"{API}/jobs/{jid}"))


def test_planner_claims_storage(client, tiny_model_id):
    ok(client.get(f"{API}/models/{tiny_model_id}/inspect"))
    ok(client.patch(f"{API}/models/{tiny_model_id}", json={"name": "tiny2"}))
    body = {"model": tiny_model_id, "chip": "H100", "quant": "w4a16", "kv_dtype": "fp8_e4m3", "workload": WL}
    assert ok(client.post(f"{API}/planner/estimate", json=body)).json()["fit"]["fits"] is True
    ok(client.post(f"{API}/planner/plan", json={"model": tiny_model_id, "chip": "H100", "workload": WL, "budget": {}}))
    ok(client.post(f"{API}/serve/simulate", json={"model": tiny_model_id, "chip": "H100", "weight_bits": 4, "kv_dtype": "fp8_e4m3",
                                                   "context_dist": {"mean_ctx": 4096, "p99_ctx": 8192}}))
    cl = ok(client.get(f"{API}/claims")).json()
    assert len(cl) >= 30
    c0 = cl[0]
    ok(client.get(f"{API}/claims/{c0['id']}"))
    # the UI sends the whole claim object back
    ok(client.put(f"{API}/claims", json={**c0, "notes": "n1", "status": "partial"}))
    ok(client.put(f"{API}/claims/{c0['id']}", json={**c0, "notes": "n2"}))
    assert client.get(f"{API}/claims/{c0['id']}").json()["notes"] == "n2"

    from forge.store import get_store

    h = get_store().put_bytes(b"hello" * 1000, kind="blob")
    ok(client.get(f"{API}/artifacts"))
    ok(client.get(f"{API}/artifacts/{h}"))
    ok(client.get(f"{API}/artifacts/{h}/lineage"))
    ok(client.post(f"{API}/artifacts/{h}/pin", json={"reason": "t"}))
    ok(client.post(f"{API}/artifacts/{h}/unpin"))
    ok(client.post(f"{API}/storage/gc", json={"dry_run": True}))
    ok(client.get(f"{API}/storage/stats"))
    assert client.post(f"{API}/evals/compare", json={"ids": ["nope"]}).status_code < 500


def test_configs_recipes_datasets(client):
    c = ok(client.post(f"{API}/configs", json={"job_type": "selftest", "name": "c1", "params": {}})).json()
    ok(client.get(f"{API}/configs/{c['id']}"))
    ok(client.put(f"{API}/configs/{c['id']}", json={"name": "c1b"}))
    ok(client.post(f"{API}/configs/validate", json={"job_type": "train", "params": {}}))
    ok(client.delete(f"{API}/configs/{c['id']}"))
    ry = (Path(__file__).parent.parent / "configs/recipes/w4a16-gptq-g128.yaml").read_text(encoding="utf-8")
    ok(client.post(f"{API}/recipes/validate", json={"yaml": ry}))
    r = ok(client.post(f"{API}/recipes", json={"name": "r1", "yaml": ry})).json()
    ok(client.get(f"{API}/recipes/{r['id']}"))
    ok(client.put(f"{API}/recipes/{r['id']}", json={"name": "r1b", "yaml": ry}))
    ok(client.delete(f"{API}/recipes/{r['id']}"))

    ds = ok(client.post(f"{API}/datasets/upload", files={"file": ("a.jsonl", JSONL)}, data={"name": "up"})).json()["id"]
    ok(client.get(f"{API}/datasets/{ds}"))
    ok(client.get(f"{API}/datasets/{ds}/preview"))
    ok(client.put(f"{API}/datasets/{ds}/content?filename=b.jsonl", content=b'{"text":"abc def"}\n' * 10))
    ij = ok(client.post(f"{API}/datasets/{ds}/ingest", json={})).json()
    j = wait_status(client, ij["id"], timeout=90)
    assert j["status"] == "succeeded", j
    assert ok(client.get(f"{API}/datasets/{ds}")).json()["status"] != "ingest_failed"
    ok(client.delete(f"{API}/datasets/{ds}"))


def test_deployment_checkpoint_flow(client, tiny_model_id):
    import torch

    from forge.store import get_store
    from forge.store.checkpoints import CheckpointWriter

    w = CheckpointWriter(get_store(), run_id="runx")
    w.save(1, {"model": {"a": torch.zeros(3)}}, {})
    w.close()
    ck = ok(client.get(f"{API}/checkpoints?run=runx")).json()
    assert ck and ck[0]["run"] == "runx"

    did = ok(client.post(f"{API}/deployments", json={"name": "d1", "artifact_ref": tiny_model_id, "serve_config": {}})).json()["id"]
    end = time.time() + 90
    stats = None
    while time.time() < end:
        s = client.get(f"{API}/deployments/{did}/stats")
        if s.status_code == 200:
            stats = s.json()
            break
        time.sleep(0.5)
    assert stats is not None and "running" in stats, "serve job never became reachable"
    ok(client.get(f"{API}/deployments/{did}/health"))
    ok(client.get(f"{API}/deployments"))
    ok(client.post(f"{API}/deployments/{did}/stop"))
    ok(client.delete(f"{API}/deployments/{did}"))
