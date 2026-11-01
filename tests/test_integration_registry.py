"""Integration: every job type resolves to an existing ConfigModel + entrypoint, defaults validate, and every
shipped template validates against its job type's model."""
from __future__ import annotations

import json

import pytest

from forge.core.jsonschema import defaults_for, schema_for
from forge.jobs.registry import list_job_types
from forge.jobs.templates import list_templates

SPECS = list_job_types()


def test_all_expected_types_registered():
    names = {s.name for s in SPECS}
    assert {"probe", "bench", "dataset_ingest", "train", "finetune", "distill", "compress", "eval", "serve", "recipe",
            "model_download", "gc", "selftest", "ternary_train"} <= names


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_type_resolves(spec):
    ok, reason = spec.availability()
    assert ok, reason
    assert callable(spec.entrypoint())
    cls = spec.config_model()
    sch = schema_for(cls)
    json.dumps(sch)  # JSON serialisable
    assert sch.get("properties"), spec.name
    d = defaults_for(cls)
    json.dumps(d)
    from pydantic import ValidationError

    try:
        cls.model_validate(d)
    except ValidationError as e:  # the only acceptable defaults failure is "the user must supply this required field"
        assert all(x["type"] in ("missing", "value_error") for x in e.errors()), e


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_templates_validate(spec):
    cls = spec.config_model()
    for t in list_templates(spec):
        params = t["params"]
        try:
            cls.model_validate(params)
        except Exception as e:  # pragma: no cover - message only
            pytest.fail(f"{t['source']} does not validate as {spec.name}: {e}")
