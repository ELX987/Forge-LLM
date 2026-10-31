"""Job entry points end-to-end: run_training export/registration, launcher process, torchrun command, job-registry wiring."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_train_fixtures import ingest_tiny, small_train_cfg

from forge.core.context import NullContext
from forge.core.errors import ConfigError, NotFound
from forge.train import dist as D
from forge.train.checkpoint import CheckpointManager
from forge.train.config import TrainConfig
from forge.train.data import IngestConfig, ingest_dataset
from forge.train.run import run_training

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ds_dir(tmp_path_factory):
    return ingest_tiny(tmp_path_factory.mktemp("runds"))


def _ctx(tmp_path, with_store=True):
    from forge.store import ArtifactStore

    return NullContext(run_dir=tmp_path / "run", store=ArtifactStore(tmp_path / "store") if with_store else None)


def _exported_state(model_dir):
    from forge.models.loader import load_model

    m, _ = load_model(model_dir, dtype="float32", device="cpu", with_tokenizer=False)
    return m.state_dict()


@pytest.mark.parametrize("rung", ["R0", "R1"])
def test_run_training_exports_loadable_model_and_registers_lineage(ds_dir, tmp_path, rung):
    cfg = small_train_cfg(ds_dir, budget={"max_steps": 5}, save_final_model=True, register_artifact=True, final_dtype="float32",
                          precision={"rung": rung, "exempt_first_layers": 0, "exempt_last_layers": 0, "loss_gap_every": 0, "diagnostics_every": 0})
    ctx = _ctx(tmp_path)
    res = run_training(cfg, ctx)
    assert res["status"] == "completed"
    md = ctx.outputs["model_dir"]
    # the exported weights are exactly the trained weights (final checkpoint), with plain linears (no QuantLinear residue)
    exp = _exported_state(md)
    ck = CheckpointManager(tmp_path / "run" / "checkpoints", cfg.checkpoint).load("latest")[0]["model"]
    assert set(exp) == set(ck) and all(torch.equal(exp[k], ck[k].float()) for k in exp)
    assert (Path(md) / "forge_lineage.json").exists()
    h = ctx.outputs["model_hash"]
    man = ctx.store.get_manifest(h)
    lin = man.extra["lineage"]
    assert lin["precision_rung"] == rung and lin["data_fingerprint"] and lin["tokens_seen"] == res["tokens_seen"]
    assert ctx.outputs["status"] == "completed" and ctx.outputs["steps"] == 5


def test_run_training_validates_inputs(ds_dir, tmp_path):
    with pytest.raises(ConfigError, match="data.dataset"):
        run_training(small_train_cfg(ds_dir, data={"dataset": []}), NullContext(run_dir=tmp_path / "a"))
    with pytest.raises(ConfigError, match="vocab"):
        run_training(small_train_cfg(ds_dir, model={"source": "tiny", "tokenizer": "byte", "arch_overrides": {"vocab_size": 100}}), NullContext(run_dir=tmp_path / "b"))
    with pytest.raises(NotFound):
        run_training(small_train_cfg(ds_dir, data={"dataset": ["no-such-dataset"]}), NullContext(run_dir=tmp_path / "c"))
    # accepts a plain dict (what the job worker may pass)
    cfg = small_train_cfg(ds_dir, budget={"max_steps": 1}).model_dump(mode="json")
    assert run_training(cfg, NullContext(run_dir=tmp_path / "d"))["steps"] == 1


def test_val_dataset_and_ingest_job_compose(tmp_path):
    """dataset_ingest -> train with a separately ingested validation set."""
    from test_train_fixtures import write_corpus

    src = write_corpus(tmp_path, 8)
    for name in ("tr", "va"):
        ingest_dataset(IngestConfig(name=name, sources=[{"path": str(src)}], tokenizer="byte", output_dir=str(tmp_path / name)), NullContext(run_dir=tmp_path / ("i" + name)))
    cfg = small_train_cfg(tmp_path / "tr", data={"dataset": [str(tmp_path / "tr")], "val_dataset": [str(tmp_path / "va")]}, budget={"max_steps": 4},
                          eval={"every_steps": 2, "num_batches": 2})
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_training(cfg, ctx)
    assert res["status"] == "completed" and ctx.metric_series("eval_loss")


def test_launch_module_single_process_roundtrip(ds_dir, tmp_path):
    """python -m forge.train.launch (what torchrun executes per rank) works standalone and emits events/result files."""
    cfg = small_train_cfg(ds_dir, budget={"max_steps": 3})
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(cfg.model_dump_json(), encoding="utf-8")
    run_dir = tmp_path / "launch_run"
    env = {**__import__("os").environ, "FORGE_HOME": str(tmp_path / "home"), "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "-m", "forge.train.launch", "--kind", "train", "--config", str(cfgp), "--run-dir", str(run_dir)],
                       capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=240)
    assert p.returncode == 0, p.stderr[-2000:]
    res = json.loads((run_dir / "result.json").read_text())
    assert res["status"] == "completed" and res["steps"] == 3
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(e["t"] == "metric" and "loss" in e["values"] for e in events) and any(e["t"] == "output" and e["key"] == "status" for e in events)
    # pre-existing STOP file => cooperative cancel before the first step
    (run_dir / "STOP").write_text("x")
    p2 = subprocess.run([sys.executable, "-m", "forge.train.launch", "--kind", "train", "--config", str(cfgp), "--run-dir", str(run_dir)],
                        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=240)
    assert p2.returncode == 0 and json.loads((run_dir / "result.json").read_text())["status"] == "stopped"


def test_torchrun_command_and_auto_launch_decision(ds_dir, tmp_path, monkeypatch):
    cmd = D.build_torchrun_command("finetune", "/x/cfg.json", "/x/run", 4, 1)
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"] and "--nproc_per_node=4" in cmd
    assert cmd[cmd.index("-m", 3) + 1] == "forge.train.launch" and cmd[cmd.index("--kind") + 1] == "finetune"
    with pytest.raises(ConfigError):
        D.build_torchrun_command("bogus", "c", "r", 2)
    calls = []
    monkeypatch.setattr(D, "launch_torchrun", lambda kind, cfg, ctx, nproc=None: calls.append(kind) or {"status": "launched"})
    cfg = small_train_cfg(ds_dir, parallel={"strategy": "ddp", "nproc_per_node": 2})
    assert run_training(cfg, NullContext(run_dir=tmp_path / "r")) == {"status": "launched"} and calls == ["train"]
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert D.under_torchrun()  # already inside torchrun => no re-launch (would try to init the process group instead)


def test_job_registry_entrypoints_resolve():
    """The job system finds our config models and entry points by the names in ARCHITECTURE.md."""
    reg = pytest.importorskip("forge.jobs.registry")
    from forge.train.config import DistillConfig, FinetuneConfig

    expect = {"train": TrainConfig, "finetune": FinetuneConfig, "distill": DistillConfig, "dataset_ingest": IngestConfig}
    for jt, cls in expect.items():
        spec = reg.get_job_type(jt)
        assert spec.config_model() is cls
        assert callable(spec.entrypoint())


def test_package_lazy_exports():
    import forge.train as T

    assert T.TrainConfig is TrainConfig and callable(T.run_training) and callable(T.ingest_dataset)
    with pytest.raises(AttributeError):
        _ = T.nonexistent
