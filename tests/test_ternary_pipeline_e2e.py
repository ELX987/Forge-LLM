"""The whole ternary pipeline through the real CLI and job manager on CPU: ingest -> `forge ternary` (QAT, health metrics, export) -> `compress` recipe
(ternarize + pack) -> a registered artifact labelled 'Ternary 1.58b' that loads.  This is the CPU analogue of scripts/ternary_smoke.sh (a few seconds of training)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _forge(args, home):
    env = {**os.environ, "FORGE_HOME": str(home), "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "-m", "forge.cli", *args], capture_output=True, text=True, timeout=600, env=env, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    return r.stdout + r.stderr


@pytest.mark.slow
def test_cli_pipeline_ingest_ternary_train_ternarize(tmp_path):
    home = tmp_path / "home"
    _forge(["job", "submit", "dataset_ingest", "-c", "configs/dataset/dataset-jsonl.yaml", "--gpus", "cpu", "--follow"], home)
    out = _forge(["ternary", "-c", "configs/ternary/tiny-smoke.yaml", "--gpus", "cpu", "--follow"], home)
    assert "succeeded" in out and "EXPERIMENTAL" in out
    run_dirs = sorted((home / "runs").glob("job_*"))
    metrics = [json.loads(ln) for rd in run_dirs if (rd / "metrics.jsonl").exists() for ln in (rd / "metrics.jsonl").read_text().splitlines()]
    keys = {k for m in metrics for k in m}
    assert {"ternary/zero_frac", "ternary/flip_rate", "ternary/loss_gap"} <= keys
    zf = [m["ternary/zero_frac"] for m in metrics if "ternary/zero_frac" in m]
    assert all(0.2 < z < 0.6 for z in zf)  # healthy range for the tiny model (TARGET band is 0.40-0.52 at scale)
    model_dir = next((home / "models").glob("ternary-tiny-smoke-*"))
    recipe = tmp_path / "r.yaml"
    recipe.write_text("recipe: ternary-smoke\nstages:\n  - ternarize: {group_size: 32}\n  - pack: {}\n")
    art = tmp_path / "art"
    out = _forge(["compress", "--set", f"model={model_dir}", "--set", f"recipe_path={recipe}", "--set", "use_store=false", "--set", f"save_dir={art}",
                  "--set", "dtype=float32", "--set", "device=cpu", "--gpus", "cpu", "--follow"], home)
    assert "Ternary 1.58b" in out and "succeeded" in out
    from forge.core.artifactmeta import read_artifact_summary
    from forge.models.loader import load_model

    s = read_artifact_summary(art)
    assert s["is_quantized"] and s["label"] == "Ternary 1.58b" and s["layout_id"].startswith("ternary2.row.v1-")
    m, _ = load_model(art, dtype="float32", device="cpu", with_tokenizer=False)
    assert {x.kind for x in m.modules() if getattr(x, "kind", "") == "t158"} == {"t158"}
