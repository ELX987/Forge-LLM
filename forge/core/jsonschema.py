"""JSON-Schema helpers used by the API and the front end. The UI renders forms from these outputs only.

``schema_for`` returns the pydantic JSON schema with every ``$ref`` inlined (no ``$defs``) so the browser
needs no reference resolver, plus UI hints:

* ``x-group``     (str)   field group / section title
* ``x-advanced``  (bool)  hide behind the "advanced" toggle
* ``x-unit``      (str)   unit label shown next to the input
* ``x-nullable``  (bool)  ``anyOf: [X, null]`` collapsed to ``X``

Hints are declared with ``Field(json_schema_extra=ui(group="Optimiser", advanced=True, unit="ms"))``.
"""
from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ValidationError

_MAX_DEPTH = 24


def ui(group: str | None = None, advanced: bool | None = None, unit: str | None = None, **extra: Any) -> dict[str, Any]:
    """Build a ``json_schema_extra`` dict with Forge UI hints."""
    d: dict[str, Any] = {}
    if group is not None:
        d["x-group"] = group
    if advanced is not None:
        d["x-advanced"] = bool(advanced)
    if unit is not None:
        d["x-unit"] = unit
    for k, v in extra.items():
        d[k if k.startswith("x-") else f"x-{k.replace('_', '-')}"] = v
    return d


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    prefix = "#/$defs/"
    if ref.startswith(prefix):
        return defs.get(ref[len(prefix):])
    return None


def _inline(node: Any, defs: dict[str, Any], stack: tuple[str, ...], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return {"type": "object", "description": "(recursion limit)"}
    if isinstance(node, list):
        return [_inline(n, defs, stack, depth + 1) for n in node]
    if not isinstance(node, dict):
        return node

    node = dict(node)
    # --- $ref (with siblings) -------------------------------------------------------------
    if "$ref" in node:
        ref = node.pop("$ref")
        target = _resolve_ref(ref, defs)
        if target is None or ref in stack:
            # unknown/recursive reference: leave a permissive object
            out = {"type": "object", "description": f"(recursive reference {ref})"}
            out.update(_inline(node, defs, stack, depth + 1))
            return out
        merged = _inline(copy.deepcopy(target), defs, stack + (ref,), depth + 1)
        siblings = _inline(node, defs, stack, depth + 1)
        merged.update(siblings)
        return merged
    # --- allOf with a single element (pydantic wraps refs with defaults this way) ----------
    if "allOf" in node and isinstance(node["allOf"], list) and len(node["allOf"]) == 1:
        inner = _inline(node.pop("allOf")[0], defs, stack, depth + 1)
        rest = _inline(node, defs, stack, depth + 1)
        inner.update(rest)
        return inner
    out = {k: _inline(v, defs, stack, depth + 1) for k, v in node.items() if k != "$defs"}
    # --- Optional[X] -> X + x-nullable ---------------------------------------------------------
    any_of = out.get("anyOf")
    if isinstance(any_of, list):
        non_null = [a for a in any_of if not (isinstance(a, dict) and a.get("type") == "null")]
        if len(non_null) == 1 and len(non_null) != len(any_of):
            merged = dict(non_null[0])
            for k, v in out.items():
                if k != "anyOf":
                    merged[k] = v
            merged["x-nullable"] = True
            return merged
    return out


def _collect_groups(schema: dict[str, Any]) -> list[str]:
    groups: list[str] = []

    def walk(s: Any) -> None:
        if isinstance(s, dict):
            g = s.get("x-group")
            if isinstance(g, str) and g not in groups:
                groups.append(g)
            for v in s.get("properties", {}).values():
                walk(v)
            for key in ("items", "additionalProperties"):
                if key in s:
                    walk(s[key])
            for key in ("anyOf", "oneOf"):
                for v in s.get(key, []) or []:
                    walk(v)

    walk(schema)
    return groups


def schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Self-contained JSON schema for ``model_cls`` (all refs inlined) with ``x-groups`` summary."""
    raw = model_cls.model_json_schema(mode="validation")
    defs = raw.get("$defs", {})
    body = {k: v for k, v in raw.items() if k != "$defs"}
    out = _inline(body, defs, (), 0)
    out["x-groups"] = _collect_groups(out)
    return out


def _field_default(field: Any) -> tuple[bool, Any]:
    if field.is_required():
        return False, None
    return True, field.get_default(call_default_factory=True)


def defaults_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """JSON-ready default values. Required fields without defaults are omitted (the UI shows them empty)."""
    try:
        return model_cls().model_dump(mode="json")
    except ValidationError:
        pass
    out: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        ok, val = _field_default(field)
        if ok:
            if isinstance(val, BaseModel):
                val = val.model_dump(mode="json")
            out[field.alias or name] = _jsonable(val)
    return out


def _jsonable(v: Any) -> Any:
    import enum

    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json")
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def validate_params(model_cls: type[BaseModel], params: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Validate ``params`` against ``model_cls``. Returns ``(clean_json_dict | None, errors)`` where each
    error is ``{"loc": "a.b.0", "msg": str, "type": str}`` (empty list when valid)."""
    try:
        obj = model_cls.model_validate(params)
    except ValidationError as e:
        errs = [
            {"loc": ".".join(str(p) for p in err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
        return None, errs
    return obj.model_dump(mode="json"), []
