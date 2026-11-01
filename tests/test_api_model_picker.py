"""Compress outputs as selectable models: automatic registration by the compress job, backfill/import, the picker listing (/models?kind=quantized),
model resolution for deployments (id / name / path / digest, conflicts and unknown references), and the memory estimate used by the deployment form."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, wait_status  # noqa: E402,F401
from test_compress_recipes import _job  # noqa: E402

from forge.core.context import NullContext  # noqa: E402

API = "/api/v1"
DIGEST = "blake3:" + "ab" * 32


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Path:
    from forge.compress.run import run_compress

    out = tmp_path_factory.mktemp("picker") / "artifact"
    run_compress(_job(save_dir=str(out)), NullContext())
    return out


def _place(home: Path, artifact: Path, job_id: str = "job_old") -> Path:
    d = home / "runs" / job_id / "artifact"
    shutil.copytree(artifact, d)
    return d


def test_compress_job_registers_its_output_as_a_model(home):
    """The job entry (WorkerContext) registers the artifact dir in the models table; scripts/tests with a NullContext do not."""
    from forge.compress.run import run_compress
    from forge.core.db import get_db
    from forge.jobs.context import WorkerContext
    from forge.jobs.repo import ModelsRepo

    ctx = WorkerContext("job_reg", home / "runs" / "job_reg", {}, db=get_db())
    res = run_compress(_job(), ctx)
    assert res["model_id"] and ctx._outputs["model_id"] == res["model_id"]
    row = ModelsRepo().get(res["model_id"])
    assert row["source"] == "artifact" and row["path_or_repo"] == str((home / "runs" / "job_reg" / "artifact").resolve())
    info = row["info"]
    assert info["kind"] == "quantized" and info["quantization"]["label"].startswith("W4A16") and info["base_model"] == "tiny:3"
    assert info["quantization"]["layout_id"].startswith("w4.portable_row.v1") and row["size_bytes"] > 0
    assert "quantized" in row["tags"]
    run_compress(_job(), NullContext())  # a NullContext run registers nothing
    assert len(ModelsRepo().list()) == 1


def test_picker_listing_import_and_badges(client, home, artifact):
    _place(home, artifact)
    # first listing backfills automatically (jobs that finished before models were registered)
    rows = client.get(f"{API}/models", params={"kind": "quantized"}).json()
    assert len(rows) == 1
    m = rows[0]
    assert m["kind"] == "quantized" and m["quantization"].startswith("W4A16 g") and m["servable"] and m["downloaded"]
    assert m["weight_gb"] is not None and m["base_model"] == "tiny:3" and m["source"] == "artifact"
    assert m["quant_summary"]["layer_kinds"] == {"w4a16": 14}
    # base models are listed separately
    assert client.get(f"{API}/models", params={"kind": "base"}).json() == []
    # explicit import is idempotent: nothing new, one refreshed
    r = client.post(f"{API}/models/import-artifacts", json={}).json()
    assert r["imported"] == [] and r["updated"] == 1 and r["scanned"] == 1
    # another directory (e.g. a copied artifact outside FORGE_HOME) via paths; not-an-artifact dirs are reported, not fatal
    other = home.parent / "elsewhere" / "q"
    shutil.copytree(artifact, other)
    r = client.post(f"{API}/models/import-artifacts", json={"paths": [str(other), str(home.parent)]}).json()
    assert [x["path_or_repo"] for x in r["imported"]] == [str(other.resolve())]
    assert any("forge_compress.json" in s["reason"] for s in r["skipped"])
    assert len(client.get(f"{API}/models", params={"kind": "quantized"}).json()) == 2
    # the inspect endpoint carries the packed layout description
    ins = client.get(f"{API}/models/{m['id']}/inspect").json()
    assert ins["quantization"]["label"] == m["quantization"] and ins["forge_supported"]


def test_import_without_any_artifact_is_a_noop(client):
    r = client.post(f"{API}/models/import-artifacts", json={}).json()
    assert r == {"imported": [], "updated": 0, "skipped": [], "scanned": 0}


def test_base_model_link_and_unique_names(client, home, artifact):
    base = home.parent / "base-model"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,'
                                      '"intermediate_size":128,"vocab_size":100}')
    (base / "model.safetensors").write_bytes(b"\0" * 64)
    bid = client.post(f"{API}/models", json={"source": "local", "path_or_repo": str(base), "name": "tiny:3"}).json()["id"]
    _place(home, artifact, "job_a")
    _place(home, artifact, "job_b")
    rows = client.post(f"{API}/models/import-artifacts", json={}).json()["imported"]
    assert len(rows) == 2 and rows[0]["name"] != rows[1]["name"]  # same recipe name, different runs: names stay distinct
    assert all(r["info"]["base_model_id"] == bid for r in rows)


def test_deployment_resolves_artifacts_by_id_name_path_and_digest(client, home, artifact):
    from forge.jobs.modelinfo import register_artifact_dir

    d = _place(home, artifact)
    row = register_artifact_dir(d, artifact_hash=DIGEST, name="qwen-w4a16")
    want = str(d.resolve())
    started = []
    for ref in (row["id"], "qwen-w4a16", str(d), DIGEST):
        r = client.post(f"{API}/deployments", json={"name": "d", "artifact_ref": ref, "serve_config": {"max_model_len": 256}})
        if r.status_code == 501:
            pytest.skip("forge.serve.config not importable")
        assert r.status_code == 201, (ref, r.text)
        dep = r.json()
        started.append(dep)
        p = client.get(f"{API}/jobs/{dep['job_id']}").json()["spec"]["params"]
        assert p["model"] == want, ref  # the job spec always names the exact directory that will be loaded
        assert dep["expected_model"]["quantized"] and dep["expected_model"]["label"].startswith("W4A16")
    for dep in started:
        client.post(f"{API}/deployments/{dep['id']}/stop")
    for dep in started:
        wait_status(client, dep["job_id"], timeout=60)


def test_deployment_rejects_conflicts_unknown_models_and_requantising(client, home, artifact):
    from forge.jobs.modelinfo import register_artifact_dir

    d = _place(home, artifact)
    row = register_artifact_dir(d, name="qwen-w4a16")
    other = home.parent / "other-model"
    other.mkdir()
    (other / "config.json").write_text("{}")
    # serve_config.model naming a different model than the picker selection is an error, never a silent override
    r = client.post(f"{API}/deployments", json={"artifact_ref": row["id"], "serve_config": {"model": str(other)}})
    assert r.status_code == 400 and "different models" in r.json()["message"]
    # an unknown reference fails at the API with a clear 404 instead of starting a job that loads something else
    r = client.post(f"{API}/deployments", json={"artifact_ref": "no-such-model"})
    assert r.status_code == 404
    r = client.post(f"{API}/deployments", json={"artifact_ref": "blake3:" + "cd" * 32})
    assert r.status_code == 404
    r = client.post(f"{API}/deployments", json={"artifact_ref": str(home.parent / "nowhere")})
    assert r.status_code == 404
    # quantising an already-packed artifact again is refused
    r = client.post(f"{API}/deployments", json={"artifact_ref": row["id"], "serve_config": {"quant": "fp8"}})
    assert r.status_code == 400 and "already quantised" in r.json()["message"]
    # form errors still win over model errors (422)
    assert client.post(f"{API}/deployments", json={"artifact_ref": "no-such-model", "serve_config": {"max_model_len": -1}}).status_code == 422


def test_memory_estimate_endpoint(client, home, artifact):
    from forge.jobs.modelinfo import register_artifact_dir

    row = register_artifact_dir(_place(home, artifact), name="qwen-w4a16")
    r = client.post(f"{API}/deployments/estimate", json={"artifact_ref": row["id"], "total_gpu_gb": 96,
                                                         "serve_config": {"gpu_mem_util": 0.85, "max_model_len": 512, "max_num_seqs": 8}})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["fits"] and e["model"]["quantized"] and e["model"]["label"].startswith("W4A16") and e["gpu_total_gb"] == 96
    assert e["limited_by"] == "workload" and e["kv_dtype"] == "fp8_e4m3"  # the recipe's KV scheme is applied with kv_dtype=auto
    assert e["weights_gb"] == e["model"]["weight_gb"] and e["free_gb"] > 80
    # knobs move the plan: a smaller card + a fill policy
    f = client.post(f"{API}/deployments/estimate", json={"artifact_ref": row["id"], "total_gpu_gb": 24,
                                                         "serve_config": {"kv_pool": "fill", "gpu_mem_util": 0.5}}).json()
    assert f["limited_by"] in ("memory", "workload") and f["kv_pool_gb"] <= 12
    assert client.post(f"{API}/deployments/estimate", json={"artifact_ref": "nope", "total_gpu_gb": 24}).status_code == 404


def test_identity_check_flags_a_wrong_load():
    from forge.api.routers.deployments import _identity_check

    exp = {"quantized": True, "label": "W4A16 g128", "path": "/w/forge-home/runs/j/artifact"}
    ok = {"quantized": True, "label": "W4A16 g128", "path": "/w/forge-home/runs/j/artifact", "weight_gb": 10.08}
    assert _identity_check(exp, ok) is None and _identity_check(None, ok) is None and _identity_check(exp, None) is None
    assert "dense weights" in _identity_check(exp, {"quantized": False, "weight_gb": 29.5, "path": exp["path"]})
    assert "loaded W8A16" in _identity_check(exp, {**ok, "label": "W8A16"})
    assert "engine loaded" in _identity_check(exp, {**ok, "path": "/models/Qwen3-14B"})


def test_store_digest_resolves_and_registers_without_the_run_directory(client, home, artifact):
    """A registered artifact whose run dir is gone is rebuilt from the artifact store (resolve_model_ref) and can be registered as a model."""
    from forge.core.modelref import resolve_model_ref
    from forge.jobs.repo import open_store

    st = open_store()
    h = st.put_tensor_dir(artifact, kind="quantized", extra={"name": "stored"})
    r = resolve_model_ref(h)
    assert r.kind == "artifact" and r.is_quantized and r.artifact["label"].startswith("W4A16") and (r.path / "config.json").is_file()
    assert resolve_model_ref(h).path == r.path  # second resolve reuses the materialised directory
    reg = client.post(f"{API}/artifacts/{h}/register-model")
    assert reg.status_code == 201, reg.text
    m = reg.json()
    assert m["kind"] == "quantized" and m["hash"] == h and m["servable"]
    assert resolve_model_ref(h).row["id"] == m["id"]  # now found in the registry by digest
    assert client.post(f"{API}/artifacts/{'blake3:' + 'ee' * 32}/register-model").status_code == 404
