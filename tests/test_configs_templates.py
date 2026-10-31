"""Every shipped config template must load through its pydantic model and contain no unknown fields.

Templates live in ``configs/<dir>/*.yaml``. The model for each directory is resolved through
``forge.jobs.registry`` (falling back to direct imports). A directory whose model cannot be imported yet is skipped
with a reason (never silently passed), so CI on an incomplete checkout stays green but the skip is visible with -rs.
"""
from __future__ import annotations

import importlib
import json
import re
import types
import typing
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

# configs/<dir> -> (job type for the registry, direct-import fallbacks "module:Class")
JOB_DIRS: dict[str, tuple[str, list[str]]] = {
    "train": ("train", ["forge.train.config:TrainConfig"]),
    "finetune": ("finetune", ["forge.train.config:FinetuneConfig"]),
    "distill": ("distill", ["forge.train.config:DistillConfig"]),
    "dataset": ("dataset_ingest", ["forge.train.data:IngestConfig"]),
    "compress": ("compress", ["forge.compress.run:CompressJobConfig", "forge.compress.config:CompressJobConfig"]),
    "eval": ("eval", ["forge.eval.run:EvalConfig", "forge.eval.config:EvalConfig"]),
    "serve": ("serve", ["forge.serve.config:ServeConfig"]),
    "bench": ("bench", ["forge.kernels.bench:BenchConfig", "forge.kernels.config:BenchConfig"]),
}


def _templates(dirname: str) -> list[Path]:
    d = CONFIGS / dirname
    return sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))) if d.is_dir() else []


def _import(target: str):
    mod, _, attr = target.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _resolve_model(dirname: str):
    job_type, fallbacks = JOB_DIRS[dirname]
    errors: list[str] = []
    try:
        from forge.jobs.registry import get_job_type

        return get_job_type(job_type).config_model()
    except Exception as e:  # registry missing or module unavailable: fall back to direct imports
        errors.append(f"registry: {type(e).__name__}: {e}")
    for target in fallbacks:
        try:
            return _import(target)
        except Exception as e:
            errors.append(f"{target}: {type(e).__name__}: {e}")
    pytest.skip(f"config model for configs/{dirname} not importable yet ({'; '.join(errors)[:300]})")


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_classes(annotation) -> list[type]:
    """BaseModel subclasses mentioned in an annotation (through Optional/Union/list/dict)."""
    from pydantic import BaseModel

    out: list[type] = []
    origin = typing.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            out.append(annotation)
        return out
    if origin in (typing.Union, types.UnionType):
        for a in typing.get_args(annotation):
            out += _model_classes(a)
    return out


def _unknown_fields(model, data, path: str = "") -> list[str]:
    """Walk ``data`` against ``model`` and report keys that are not declared fields (even if the model allows extras)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return problems
    fields = model.model_fields
    for key, value in data.items():
        if key not in fields:
            problems.append(f"{path}{key}")
            continue
        if isinstance(value, dict):
            nested = _model_classes(fields[key].annotation)
            if len(nested) == 1:
                problems += _unknown_fields(nested[0], value, f"{path}{key}.")
    return problems


ALL_JOB_TEMPLATES = [(d, p) for d in JOB_DIRS for p in _templates(d)]


@pytest.mark.parametrize("dirname,path", ALL_JOB_TEMPLATES, ids=[f"{d}/{p.stem}" for d, p in ALL_JOB_TEMPLATES])
def test_job_template_loads_into_model(dirname: str, path: Path) -> None:
    model = _resolve_model(dirname)
    data = _load_yaml(path)
    assert isinstance(data, dict), f"{path.name}: top level must be a mapping"
    unknown = _unknown_fields(model, data)
    assert not unknown, f"{path.name}: unknown fields {unknown}"
    cfg = model.model_validate(data)  # extra="forbid" models also reject unknown nested keys here
    assert cfg is not None


@pytest.mark.parametrize("dirname,path", ALL_JOB_TEMPLATES, ids=[f"{d}/{p.stem}" for d, p in ALL_JOB_TEMPLATES])
def test_template_has_header_comment(dirname: str, path: Path) -> None:
    first = path.read_text(encoding="utf-8").lstrip().splitlines()[0]
    assert first.startswith("#"), f"{path.name}: first line must be a comment explaining the template"


def test_expected_templates_exist() -> None:
    expected = {
        "train": {"tiny-smoke", "llama-8b-full-bf16", "qwen-fp8", "nvfp4-pilot"},
        "finetune": {"llama-8b-lora"},
        "distill": {"offline-logit-7b-to-1.5b", "on-policy-7b-to-1.5b"},
        "serve": {"serve-8b-w4a16", "serve-spec-ngram"},
        "dataset": {"dataset-jsonl"},
        "compress": {"w4a16-gptq-g128", "fp8-w8a8", "nvfp4-ptq", "prune-distill-8b-to-4b", "lowrank", "delta"},
        "eval": {"eval-gw4"},
        "bench": {"bench-gemm-sweep"},
    }
    for d, names in expected.items():
        have = {p.stem for p in _templates(d)}
        assert names <= have, f"configs/{d} is missing templates: {sorted(names - have)}"


def test_nvfp4_pilot_is_marked_experimental() -> None:
    text = (CONFIGS / "train" / "nvfp4-pilot.yaml").read_text(encoding="utf-8")
    assert "EXPERIMENTAL" in text and "allow_unvalidated_fp4: true" in text


def test_no_secrets_in_templates() -> None:
    pat = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|api_key:\s*['\"]?[A-Za-z0-9]{16,})")
    for p in CONFIGS.rglob("*"):
        if p.is_file() and p.suffix in {".yaml", ".yml", ".jsonl", ".md"}:
            assert not pat.search(p.read_text(encoding="utf-8")), f"possible secret in {p}"


# ---- recipes (Plan Appendix C.1) ---------------------------------------------------------------------------------
RECIPES = _templates("recipes")


@pytest.mark.parametrize("path", RECIPES, ids=[p.stem for p in RECIPES])
def test_recipe_templates_load(path: Path) -> None:
    from forge.core.recipe import load_recipe

    recipe = load_recipe(path)
    assert recipe.stages, f"{path.name}: no stages"
    assert recipe.hash().startswith(("blake3:", "b2:"))
    assert recipe.name


def test_recipe_stage_names_are_known() -> None:
    """Stage names in recipes must be registered stages (once forge.compress.stages exists)."""
    try:
        importlib.import_module("forge.compress.stages")
    except Exception as e:
        pytest.skip(f"forge.compress.stages not importable yet: {e}")
    from forge.core import registry
    from forge.core.recipe import load_recipe

    known = set(registry.names("stage"))
    for p in RECIPES:
        for s in load_recipe(p).stages:
            assert s.name in known, f"{p.name}: unknown stage {s.name!r} (registered: {sorted(known)})"


# ---- chips (Appendix B) ----------------------------------------------------------------------------------------
CHIPS = _templates("chips")


@pytest.mark.parametrize("path", CHIPS, ids=[p.stem for p in CHIPS])
def test_chip_examples_validate(path: Path) -> None:
    from forge.core.schemas import ChipDescriptor

    chip = ChipDescriptor.model_validate(_load_yaml(path))
    assert chip.id and chip.sm is not None
    assert chip.tier == "C", "datasheet examples must not claim a validated tier"
    assert chip.supports("bf16") and chip.memory.hbm_gb > 0


def test_b200_fp4_is_not_claimed_as_measured() -> None:
    from forge.core.schemas import ChipDescriptor

    chip = ChipDescriptor.model_validate(_load_yaml(CONFIGS / "chips" / "b200-blackwell-hypothesis.yaml"))
    mu = chip.matrix_unit_for("nvfp4")
    assert mu is not None and mu.peak_tflops.measured is None
    assert chip.probe_provenance["matrix_units"].value == "untested"


# ---- fleets (Appendix C.4) ---------------------------------------------------------------------------------------
FLEETS = _templates("fleets")


@pytest.mark.parametrize("path", FLEETS, ids=[p.stem for p in FLEETS])
def test_fleet_manifest_shape(path: Path) -> None:
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    for key in ("fleet", "base", "roles", "shared_prefix", "cascade", "speculation", "eviction_policy", "scheduler"):
        assert key in doc, f"{path.name}: missing {key}"
    assert doc["roles"] and all(isinstance(v, dict) and "prompt" in v for v in doc["roles"].values())
    assert any(("adapter" in v or "delta" in v) for v in doc["roles"].values())


# ---- gates (owned by eval_planner; only checked for syntax here) -------------------------------------------------
def test_gate_yaml_parses() -> None:
    for p in _templates("gates"):
        assert isinstance(_load_yaml(p), dict), f"{p} must be a YAML mapping"


# ---- bundled sample data ---------------------------------------------------------------------------------------
def test_sample_corpus_jsonl() -> None:
    lines = (CONFIGS / "dataset" / "sample-corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 20
    for ln in lines:
        assert isinstance(json.loads(ln)["text"], str)


def test_sample_sft_jsonl() -> None:
    lines = (CONFIGS / "dataset" / "sample-sft.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 10
    for ln in lines:
        msgs = json.loads(ln)["messages"]
        assert msgs[-1]["role"] == "assistant" and any(m["role"] == "user" for m in msgs)


def test_shell_scripts_are_lf_and_strict() -> None:
    for p in (ROOT / "scripts").glob("*.sh"):
        raw = p.read_bytes()
        assert b"\r" not in raw, f"{p.name} has CRLF line endings"
        text = raw.decode("utf-8")
        assert text.startswith("#!/usr/bin/env bash"), p.name
        if p.name != "_common.sh":
            assert "set -euo pipefail" in text, f"{p.name} must use strict mode"


# ---- recipe stage parameters validate against the stage parameter models ------------------------------------------
def _stage_params_problems(recipe) -> list[str]:
    try:
        from forge.compress.stages import _params
    except Exception as e:
        pytest.skip(f"forge.compress.stages not importable: {e}")
    problems = []
    for s in recipe.stages:
        try:
            _params(s.name, s.params)
        except Exception as e:
            problems.append(f"{s.name}: {e}")
    return problems


@pytest.mark.parametrize("path", RECIPES, ids=[p.stem for p in RECIPES])
def test_recipe_stage_params_valid(path: Path) -> None:
    from forge.core.recipe import load_recipe

    assert not _stage_params_problems(load_recipe(path)), path.name


def test_compress_templates_reference_existing_recipes() -> None:
    for p in _templates("compress"):
        doc = _load_yaml(p)
        rp = doc.get("recipe_path")
        if rp:
            assert (ROOT / rp).is_file(), f"{p.name}: recipe_path {rp} does not exist"
        else:
            from forge.core.recipe import load_recipe

            assert not _stage_params_problems(load_recipe(doc["recipe"])), p.name
