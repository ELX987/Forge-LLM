"""Config models: schema rendering, descriptions, defaults, validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge.core.jsonschema import defaults_for, schema_for, validate_params
from forge.train.config import DistillConfig, FinetuneConfig, TrainConfig
from forge.train.data import IngestConfig

CONFIGS = [TrainConfig, FinetuneConfig, DistillConfig, IngestConfig]


def _walk(schema, path=""):
    props = schema.get("properties", {})
    for name, s in props.items():
        yield f"{path}{name}", s
        if s.get("type") == "object" and "properties" in s:
            yield from _walk(s, f"{path}{name}.")
        if s.get("type") == "array" and isinstance(s.get("items"), dict) and "properties" in s["items"]:
            yield from _walk(s["items"], f"{path}{name}[].")


@pytest.mark.parametrize("cls", CONFIGS)
def test_every_field_has_description(cls):
    sch = schema_for(cls)
    missing = [p for p, s in _walk(sch) if not s.get("description")]
    assert not missing, missing


@pytest.mark.parametrize("cls", CONFIGS)
def test_defaults_roundtrip(cls):
    d = defaults_for(cls)
    obj = cls.model_validate(d)
    assert obj.model_dump(mode="json") == d


@pytest.mark.parametrize("cls", [TrainConfig, FinetuneConfig, DistillConfig])
def test_groups_and_advanced_hints(cls):
    sch = schema_for(cls)
    assert len(sch["x-groups"]) >= 8
    for key in ("optim", "schedule", "precision", "parallel", "checkpoint", "eval", "stability", "budget"):
        assert "x-group" in sch["properties"][key], key
    adv = [p for p, s in _walk(sch) if s.get("x-advanced")]
    assert len(adv) > 20


def test_finetune_defaults_differ_from_pretrain():
    ft, tr = FinetuneConfig(), TrainConfig()
    assert ft.optim.lr != tr.optim.lr and ft.optim.weight_decay == 0.0
    assert ft.method == "lora"


def test_validation_errors():
    clean, errs = validate_params(TrainConfig, {"optim": {"lr": -1}})
    assert clean is None and any("optim.lr" in e["loc"] for e in errs)
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"unknown_field": 1})
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"budget": {"max_steps": None}})
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"optim": {"betas": [0.9, 1.5]}})


def test_param_coverage():
    """Everything the task asks for is a config knob."""
    t = TrainConfig()
    assert set(t.optim.model_fields) >= {"name", "lr", "weight_decay", "betas", "grad_clip"}
    assert {"adamw", "adamw8bit", "lion", "sgd"} == set(__import__("typing").get_args(t.optim.model_fields["name"].annotation))
    assert set(__import__("typing").get_args(t.schedule.model_fields["name"].annotation)) >= {"cosine", "linear", "wsd"}
    assert set(t.precision.model_fields) >= {"rung", "exempt_patterns", "healing_steps", "fp4_stochastic_rounding", "fp4_rht_wgrad"}
    assert set(t.parallel.model_fields) >= {"strategy", "compile", "fsdp_sharding"}
    assert set(t.stability.model_fields) >= {"spike_zscore", "max_restarts", "on_nonfinite"}
