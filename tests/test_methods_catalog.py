"""forge.methods: descriptors must point at things that exist, and 'planned' entries must not pretend to be runnable."""
from __future__ import annotations

from pathlib import Path

import pytest

from forge.core.errors import ConfigError, NotFound
from forge.jobs.registry import get_job_type
from forge.methods import MethodSpec, get_method, list_methods, register_method

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_catalog_is_populated_and_sorted():
    ms = list_methods()
    assert len(ms) >= 20 and ms == sorted(ms, key=lambda m: (m.family, m.id))
    assert {m.family for m in ms} >= {"pretrain", "low_precision_training", "peft", "distillation", "ptq", "compression", "ternary"}


@pytest.mark.parametrize("m", [m for m in list_methods() if m.status != "planned"], ids=lambda m: m.id)
def test_runnable_methods_reference_real_job_types_and_templates(m):
    assert m.job_type, m.id
    get_job_type(m.job_type)
    if m.template:
        assert (CONFIGS / m.template).is_file(), m.template


@pytest.mark.parametrize("m", [m for m in list_methods() if m.status == "planned"], ids=lambda m: m.id)
def test_planned_methods_have_no_job_or_template(m):
    assert m.job_type is None and m.template is None and not m.gpu_validated


def test_nothing_claims_gpu_validation_yet():
    assert not any(m.gpu_validated for m in list_methods())


def test_ternary_train_is_experimental_and_in_ternary_family():
    m = get_method("ternary-train")
    assert m.status == "experimental" and m.family == "ternary" and m.diagnostics_prefix == "ternary" and m.job_type == "ternary_train"
    assert {x.id for x in list_methods(family="ternary", status="experimental")} == {"ternary-train"}


def test_register_and_lookup_errors():
    with pytest.raises(NotFound):
        get_method("nope")
    with pytest.raises(ConfigError):
        register_method(get_method("lora"))
    register_method(MethodSpec(id="x-test", title="t", family="peft", status="planned", summary="s"))
    assert get_method("x-test").title == "t"
