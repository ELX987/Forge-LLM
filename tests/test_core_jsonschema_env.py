from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge.core import env
from forge.core.jsonschema import defaults_for, schema_for, ui, validate_params
from forge.core.schemas import ChipDescriptor, Recipe


class Mode(str, Enum):
    fast = "fast"
    slow = "slow"


class Inner(BaseModel):
    """Inner block."""

    a: int = Field(default=1, ge=0, le=10, description="inner a", json_schema_extra=ui(unit="ms"))
    mode: Mode = Field(default=Mode.fast, description="mode")


class Leaf(BaseModel):
    x: float = Field(default=0.5, description="x")
    child: Inner | None = Field(default=None, description="optional recursive-free child")


class Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="required name", json_schema_extra=ui(group="Basics"))
    lr: float = Field(default=1e-3, gt=0, description="learning rate", json_schema_extra=ui(group="Optimiser", advanced=True))
    inner: Inner = Field(default_factory=Inner, description="inner config")
    inners: list[Inner] = Field(default_factory=list, description="many")
    leaf: Leaf | None = Field(default=None, description="optional leaf")
    prec: Literal["bf16", "fp8"] = Field(default="bf16", description="precision")
    opt_int: int | None = Field(default=None, description="nullable int")
    tags: dict[str, float] = Field(default_factory=dict, description="mapping")


def _no_refs(node) -> bool:
    if isinstance(node, dict):
        return "$ref" not in node and "$defs" not in node and all(_no_refs(v) for v in node.values())
    if isinstance(node, list):
        return all(_no_refs(v) for v in node)
    return True


def test_schema_is_fully_inlined_and_json_serialisable():
    s = schema_for(Cfg)
    assert _no_refs(s)
    json.dumps(s)
    props = s["properties"]
    assert props["inner"]["properties"]["a"]["maximum"] == 10
    assert props["inner"]["properties"]["a"]["x-unit"] == "ms"
    assert props["inner"]["description"] == "inner config" or props["inner"].get("title") == "Inner"
    assert props["inner"]["properties"]["mode"]["enum"] == ["fast", "slow"]
    assert props["inners"]["items"]["properties"]["a"]["type"] == "integer"
    assert props["prec"]["enum"] == ["bf16", "fp8"]
    assert s["required"] == ["name"]


def test_hints_groups_and_nullable():
    s = schema_for(Cfg)
    assert s["properties"]["lr"]["x-advanced"] is True and s["properties"]["lr"]["x-group"] == "Optimiser"
    assert s["x-groups"] == ["Basics", "Optimiser"]
    p = s["properties"]["opt_int"]
    assert p["type"] == "integer" and p["x-nullable"] is True and "anyOf" not in p
    leaf = s["properties"]["leaf"]
    assert leaf["x-nullable"] is True and leaf["type"] == "object"
    assert leaf["properties"]["child"]["properties"]["a"]["default"] == 1  # nested ref inlined too
    assert leaf["description"] == "optional leaf"  # sibling description wins


def test_defaults_for():
    d = defaults_for(Cfg)  # 'name' is required -> model can't be built; defaults still returned
    assert "name" not in d
    assert d["lr"] == 1e-3 and d["inner"] == {"a": 1, "mode": "fast"} and d["inners"] == [] and d["leaf"] is None
    assert d["prec"] == "bf16"
    json.dumps(d)
    assert defaults_for(Inner) == {"a": 1, "mode": "fast"}


def test_validate_params():
    clean, errs = validate_params(Cfg, {"name": "n", "inner": {"a": 3}})
    assert errs == [] and clean["inner"]["a"] == 3 and clean["inner"]["mode"] == "fast"
    clean, errs = validate_params(Cfg, {"inner": {"a": 99}, "zzz": 1})
    assert clean is None
    locs = {e["loc"] for e in errs}
    assert {"name", "inner.a", "zzz"} <= locs


def test_core_schemas_render():
    for cls in (Recipe, ChipDescriptor):
        s = schema_for(cls)
        assert _no_refs(s)
        json.dumps(s)
    s = schema_for(Recipe)
    assert s["properties"]["stages"]["items"]["properties"]["name"]["type"] == "string"
    assert defaults_for(Recipe).get("seeds") == [0]


def test_recursive_models_do_not_blow_up():
    class Node(BaseModel):
        value: int = Field(default=0, description="v")
        children: list[Node] = Field(default_factory=list, description="kids")

    Node.model_rebuild()
    s = schema_for(Node)
    json.dumps(s)
    assert "properties" in s


def test_seed_everything_reproducible():
    import numpy as np
    import torch

    env.seed_everything(123)
    a = (torch.rand(4), np.random.rand(3))
    env.seed_everything(123)
    b = (torch.rand(4), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and (a[1] == b[1]).all()
    env.seed_everything(124)
    assert not torch.equal(a[0], torch.rand(4))


def test_capture_environment_shape():
    e = env.capture_environment()
    json.dumps(e)
    assert e["python"] and e["torch"]["installed"] in (True, False)
    assert "packages" in e and "git_sha" in e and "gpu_fingerprint" in e
    if e["torch"].get("installed") and not e["torch"].get("cuda_available"):
        assert e["gpu_fingerprint"] is None
