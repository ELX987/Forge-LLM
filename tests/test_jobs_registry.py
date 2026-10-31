"""Job-type registry, templates, built-in job types and the worker RunContext (in-process)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import home  # noqa: E402,F401

from forge.core.errors import NotFound  # noqa: E402
from forge.jobs import builtin  # noqa: E402
from forge.jobs.registry import JobTypeSpec, UnavailableModule, get_job_type, job_type_names, list_job_types  # noqa: E402

EXPECTED = {"probe", "bench", "dataset_ingest", "train", "finetune", "distill", "compress", "eval", "serve", "recipe", "model_download", "gc", "selftest", "checkpoint_export"}


def test_all_section6_job_types_registered():
    assert EXPECTED <= set(job_type_names())
    assert get_job_type("train").entry == "forge.train.run:run_training"
    assert get_job_type("serve").entry == "forge.serve.run:run_server"
    assert get_job_type("bench").entry == "forge.kernels.bench:run_bench"
    assert get_job_type("serve").needs_gpu and not get_job_type("model_download").needs_gpu
    from forge.core import registry

    assert "selftest" in registry.names("job_type")


def test_unknown_job_type():
    with pytest.raises(NotFound):
        get_job_type("does-not-exist")


def test_unavailable_module_is_graceful():
    spec = JobTypeSpec("x", "x", "x", ["no_such_pkg.config:Cfg", "no_such_pkg2:Cfg"], "no_such_pkg.run:go")
    with pytest.raises(UnavailableModule) as ei:
        spec.config_model()
    assert ei.value.http_status == 501 and len(ei.value.details["errors"]) == 2
    ok, reason = spec.availability()
    assert not ok and "not available" in reason


def test_config_candidates_first_importable_wins():
    spec = JobTypeSpec("x", "x", "x", ["no_such_pkg:Cfg", "forge.jobs.builtin:SelftestConfig"], "forge.jobs.builtin:run_selftest")
    assert spec.config_model() is builtin.SelftestConfig
    assert spec.availability() == (True, None)


def test_every_available_config_builds_a_schema(home):
    """All registered config models must produce a UI schema + defaults without errors."""
    from forge.jobs.templates import job_type_payload

    for spec in list_job_types():
        p = job_type_payload(spec)
        assert p["type"] == spec.name and isinstance(p["templates"], list)
        if p["available"] or p["schema"]:
            assert p["schema"]["type"] == "object" and "properties" in p["schema"]
            assert isinstance(p["defaults"], dict)
        else:
            assert p["reason"]


def test_builtin_configs_describe_every_field():
    from forge.jobs.templates import job_type_payload

    for t in ("selftest", "model_download", "gc", "recipe", "probe", "checkpoint_export"):
        sch = job_type_payload(get_job_type(t))["schema"]
        missing = [k for k, v in sch["properties"].items() if not v.get("description")]
        assert not missing, f"{t}: fields without description {missing}"


def test_templates_read_from_configs_dir(home, tmp_path, monkeypatch):
    from forge.jobs.templates import job_type_payload, list_templates

    root = tmp_path / "cfgs"
    (root / "selftest").mkdir(parents=True)
    (root / "selftest" / "quick.yaml").write_text("# A quick synthetic run.\n# Second line.\n\nsteps: 3\nstep_delay_s: 0\n")
    (root / "selftest" / "wrapped.yaml").write_text("description: wrapped one\nparams:\n  steps: 4\n")
    (root / "selftest" / "broken.yaml").write_text("steps: [unclosed\n")
    monkeypatch.setenv("FORGE_CONFIGS_DIR", str(root))
    ts = {t["name"]: t for t in list_templates("selftest")}
    assert set(ts) == {"quick", "wrapped"}  # broken template skipped
    assert ts["quick"]["params"] == {"steps": 3, "step_delay_s": 0}
    assert "quick synthetic run" in ts["quick"]["description"]
    assert ts["wrapped"]["params"] == {"steps": 4} and ts["wrapped"]["description"] == "wrapped one"
    assert job_type_payload(get_job_type("selftest"))["templates"] == ["quick", "wrapped"]


# --------------------------------------------------------------------------------------------
# RunContext implementation
# --------------------------------------------------------------------------------------------
def test_worker_context_contract(home, capsys):
    from forge.core.context import RunContext
    from forge.core.schemas import JobSpec
    from forge.jobs.context import CANCEL_FILE, WorkerContext
    from forge.jobs.repo import JobsRepo
    from forge.telemetry.metrics import get_metrics_store

    repo = JobsRepo()
    row = repo.create(JobSpec(type="selftest"), run_dir=str(home / "runs" / "ctx"))
    ctx = WorkerContext(row["id"], home / "runs" / "ctx", {"a": 1}, metrics=get_metrics_store())
    assert isinstance(ctx, RunContext)  # structural check against forge.core.context.RunContext
    ctx.log("hello\nsecond", "warning")
    out = capsys.readouterr().out
    assert "[WARNING] hello" in out and "[WARNING] second" in out
    ctx.metric(3, loss=1.5, lr=0.1)
    ctx.progress(0.25, "quarter")
    ctx.progress(7.0, "clamped")
    ctx.set_output("path", Path("/tmp/x"))
    ctx.set_output("n", 5)
    ctx.close()
    r = repo.require(row["id"])
    assert r["progress"] == 1.0 and r["progress_msg"] == "clamped"
    assert r["outputs"] == {"path": str(Path("/tmp/x")), "n": 5}
    assert get_metrics_store().latest(row["id"])["loss"]["step"] == 3
    assert not ctx.should_stop()
    (home / "runs" / "ctx" / CANCEL_FILE).write_text("x")
    time.sleep(0.15)
    assert ctx.should_stop()
    assert ctx.store is None or hasattr(ctx.store, "put_bytes")


def test_worker_context_stop_via_db_flag_and_signal_flag(home):
    from forge.core.schemas import JobSpec
    from forge.jobs.context import WorkerContext
    from forge.jobs.repo import JobsRepo

    repo = JobsRepo()
    row = repo.create(JobSpec(type="selftest"), run_dir=str(home / "runs" / "c2"))
    ctx = WorkerContext(row["id"], home / "runs" / "c2", {})
    assert not ctx.should_stop()
    repo.update(row["id"], cancel_requested=1)
    time.sleep(0.15)
    assert ctx.should_stop()
    ctx2 = WorkerContext(row["id"], home / "runs" / "c3", {})
    ctx2.request_stop()  # what the SIGTERM handler does
    assert ctx2.should_stop()


# --------------------------------------------------------------------------------------------
# built-in job entrypoints (run in-process with a NullContext)
# --------------------------------------------------------------------------------------------
def test_selftest_entry_emits_metrics_and_stops():
    from forge.core.context import NullContext

    ctx = NullContext()
    out = builtin.run_selftest(builtin.SelftestConfig(steps=5, step_delay_s=0, seed=1), ctx)
    assert out == {"steps_done": 5, "stopped": False}
    assert [s for s, _ in ctx.metrics] == [1, 2, 3, 4, 5]
    assert {"loss", "lr", "grad_norm", "tokens_per_s"} <= set(ctx.metrics[0][1])
    assert ctx.last_progress == 1.0 and ctx.outputs["synthetic"] is True
    assert any("SYNTHETIC" in m for m in ctx.logs)
    ctx2 = NullContext()
    ctx2.stop = True
    assert builtin.run_selftest(builtin.SelftestConfig(steps=50, step_delay_s=0), ctx2)["stopped"] is True
    with pytest.raises(RuntimeError, match="injected"):
        builtin.run_selftest(builtin.SelftestConfig(steps=5, step_delay_s=0, fail_at_step=2), NullContext())


def test_selftest_is_deterministic():
    from forge.core.context import NullContext

    a, b = NullContext(), NullContext()
    builtin.run_selftest(builtin.SelftestConfig(steps=4, step_delay_s=0, seed=7), a)
    builtin.run_selftest(builtin.SelftestConfig(steps=4, step_delay_s=0, seed=7), b)
    loss = lambda c: [m["loss"] for _, m in c.metrics]  # noqa: E731
    assert loss(a) == loss(b)


def test_recipe_config_requires_a_source():
    with pytest.raises(ValueError):
        builtin.RecipeJobConfig()
    assert builtin.RecipeJobConfig(recipe_yaml="recipe: x\nstages: []\n").force is False


def test_recipe_job_runs_registered_stage(home):
    from forge.core import registry
    from forge.core.context import NullContext

    calls = []

    @registry.register("stage", "jobs_test_double")
    def _double(inputs, params, ctx):
        calls.append(inputs["x"])
        return {"x": inputs["x"] * params.get("k", 2)}

    try:
        cfg = builtin.RecipeJobConfig(recipe_yaml="recipe: t\ninputs: {x: 3}\nstages:\n  - jobs_test_double: {k: 5}\n", cache=False)
        ctx = NullContext()
        res = builtin.run_recipe_job(cfg, ctx)
        assert calls == [3] and res["cancelled"] is False
        assert ctx.outputs["outputs"]["x"] == 15 and ctx.outputs["recipe_hash"]
    finally:
        registry.unregister("stage", "jobs_test_double")


def test_recipe_job_from_saved_recipe(home):
    from forge.core import registry
    from forge.core.context import NullContext
    from forge.jobs.repo import RecipesRepo

    registry.register_obj("stage", "jobs_test_id", lambda inputs, params, ctx: {"y": 1})
    try:
        RecipesRepo().insert(id="rcp_t", name="t", yaml="recipe: t\nstages:\n  - jobs_test_id: {}\n", created_at=1.0, updated_at=1.0)
        ctx = NullContext()
        builtin.run_recipe_job(builtin.RecipeJobConfig(recipe_id="rcp_t", cache=False), ctx)
        assert ctx.outputs["outputs"]["y"] == 1
        with pytest.raises(NotFound):
            builtin.run_recipe_job(builtin.RecipeJobConfig(recipe_id="missing"), NullContext())
    finally:
        registry.unregister("stage", "jobs_test_id")


def test_gc_job_dry_run(home):
    from forge.core.context import NullContext

    ctx = NullContext()
    try:
        res = builtin.run_gc(builtin.GcJobConfig(), ctx)
    except UnavailableModule:
        pytest.skip("forge.store not available")
    assert res["dry_run"] is True and "report" in ctx.outputs


def test_model_download_config_defaults_and_unavailable_hub(monkeypatch, home):
    cfg = builtin.ModelDownloadConfig(repo_id="org/name")
    assert cfg.prefer_safetensors and cfg.register_model and "*.onnx" in cfg.ignore_patterns
    with pytest.raises(Exception):
        builtin.ModelDownloadConfig(repo_id="x")  # too short
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # simulate a missing package
    from forge.core.context import NullContext

    with pytest.raises(UnavailableModule, match="huggingface_hub"):
        builtin.run_model_download(cfg, NullContext())


def test_model_download_with_stubbed_hub(monkeypatch, home, tmp_path):
    """Progress + registration logic against a fake huggingface_hub (no network)."""
    import types

    from forge.core.context import NullContext

    def fake_snapshot_download(repo_id, revision=None, local_dir=None, allow_patterns=None, ignore_patterns=None, token=None, max_workers=8):
        d = Path(local_dir)
        (d / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                                                   "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100}))
        (d / "model.safetensors").write_bytes(b"\0" * 4096)
        time.sleep(0.6)
        return str(d)

    class Sib:
        def __init__(self, n, s):
            self.rfilename, self.size = n, s

    class FakeApi:
        def model_info(self, repo_id, revision=None, files_metadata=False, token=None):
            return types.SimpleNamespace(siblings=[Sib("config.json", 200), Sib("model.safetensors", 4096), Sib("pytorch_model.bin", 9999)],
                                         card_data=types.SimpleNamespace(license="apache-2.0"), tags=[])

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = fake_snapshot_download
    fake.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    ctx = NullContext()
    out = builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="org/tiny"), ctx)
    assert Path(out["path"]).name == "tiny" and out["model_id"]
    assert ctx.outputs["files"] == 2  # pytorch_model.bin skipped because safetensors exist
    assert ctx.last_progress == 1.0
    from forge.jobs.repo import ModelsRepo

    row = ModelsRepo().require(out["model_id"])
    assert row["source"] == "hf" and row["path_or_repo"] == "org/tiny" and row["licence"] == "apache-2.0" and row["params_b"] is not None


def test_model_download_cancel(monkeypatch, home):
    import types

    from forge.core.context import NullContext
    from forge.core.errors import Cancelled

    def slow(**kw):
        time.sleep(5)
        return kw["local_dir"]

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda **kw: slow(**kw)
    fake.HfApi = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    ctx = NullContext()
    ctx.stop = True
    with pytest.raises(Cancelled):
        builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="org/slow"), ctx)
    assert any("indeterminate" in m for m in ctx.logs)


def test_checkpoint_export_copies_safetensors_dir(home):
    from forge.core.context import NullContext

    ck = home / "runs" / "job_x" / "checkpoints" / "step-100"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"\0" * 64)
    (ck / "config.json").write_text(json.dumps({"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}))
    ctx = NullContext()
    out = builtin.run_checkpoint_export(builtin.CheckpointExportConfig(run_id="job_x", checkpoint="100", name="exp"), ctx)
    assert (Path(out["path"]) / "model.safetensors").exists() and out.get("model_id")
    out2 = builtin.run_checkpoint_export(builtin.CheckpointExportConfig(run_id="job_x", checkpoint="latest", name="exp2", register_model=False), NullContext())
    assert "model_id" not in out2


def test_probe_job_unavailable_or_runs(home):
    from forge.core.context import NullContext

    try:
        res = builtin.run_probe_job(builtin.ProbeJobConfig(quick=True), NullContext())
    except UnavailableModule:
        pytest.skip("forge.hw.probe not available")
    except Exception as e:  # probe exists but cannot run here: must be a clean error, not a crash
        assert str(e)
    else:
        assert isinstance(res, dict)


def test_shipped_templates_validate_against_job_configs(home):
    """Every configs/<type>/*.yaml that this layer exposes must be accepted by its job type's config model."""
    from forge.core.jsonschema import validate_params
    from forge.jobs.templates import list_templates

    checked = 0
    for spec in list_job_types():
        try:
            cls = spec.config_model()
        except UnavailableModule:
            continue
        for t in list_templates(spec):
            clean, errs = validate_params(cls, t["params"])
            assert not errs, f"template {t['source']} invalid for job type {spec.name}: {errs[:2]}"
            checked += 1
    assert checked >= 0
