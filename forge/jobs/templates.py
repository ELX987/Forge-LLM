"""Config templates (``configs/<type>/*.yaml``, read-only seeds) and the schema payloads served to the UI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from forge.jobs.registry import JobTypeSpec, UnavailableModule, list_job_types, templates_root

log = logging.getLogger("forge.jobs.templates")


def _header_comment(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            lines.append(s.lstrip("#").strip())
        elif s == "":
            if lines:
                break
        else:
            break
    return " ".join(x for x in lines if x)[:500]


def _unwrap(doc: Any) -> dict[str, Any]:
    """Templates are usually plain params; also accept ``{params: {...}}`` / ``{type:..., params:...}``."""
    if isinstance(doc, dict) and isinstance(doc.get("params"), dict) and set(doc) <= {"type", "name", "description", "params", "gpus", "tags"}:
        return doc["params"]
    return doc if isinstance(doc, dict) else {}


def list_templates(job_type: str | JobTypeSpec, *, include_params: bool = True) -> list[dict[str, Any]]:
    from forge.jobs.registry import get_job_type

    spec = get_job_type(job_type) if isinstance(job_type, str) else job_type
    root = templates_root()
    if root is None:
        return []
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for d in spec.templates_dirs():
        folder = root / d
        if not folder.is_dir():
            continue
        for f in sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")) + list(folder.glob("*.json"))):
            if f in seen:
                continue
            seen.add(f)
            try:
                text = f.read_text(encoding="utf-8")
                doc = yaml.safe_load(text)
            except Exception as e:
                log.warning("bad template %s: %s", f, e)
                continue
            item: dict[str, Any] = {
                "type": spec.name,
                "name": f.stem,
                "source": f"{d}/{f.name}",
                "description": (doc.get("description") if isinstance(doc, dict) and isinstance(doc.get("description"), str) else "") or _header_comment(text),
                "readonly": True,
            }
            if include_params:
                item["params"] = {"recipe_yaml": text} if spec.template_wrap == "recipe_yaml" else _unwrap(doc)
            out.append(item)
    return out


def job_type_payload(spec: JobTypeSpec, *, with_schema: bool = True) -> dict[str, Any]:
    """``{type, title, description, schema, defaults, templates:[names], groups:[...], available, reason, ...}``."""
    from forge.core.jsonschema import defaults_for, schema_for

    payload: dict[str, Any] = {
        "type": spec.name,
        "title": spec.title,
        "description": spec.description,
        "group": spec.group,
        "needs_gpu": spec.needs_gpu,
        "light": spec.light,
        "experimental": spec.experimental,
        "available": True,
        "reason": None,
        "schema": None,
        "defaults": {},
        "groups": [],
        "templates": [],
    }
    try:
        cls = spec.config_model()
        if with_schema:
            sch = schema_for(cls)
            payload["schema"] = sch
            payload["groups"] = sch.get("x-groups", [])
            payload["defaults"] = defaults_for(cls)
        try:
            spec.entrypoint()
        except UnavailableModule as e:
            payload["available"] = False
            payload["reason"] = e.message
    except UnavailableModule as e:
        payload["available"] = False
        payload["reason"] = e.message
    except Exception as e:  # a buggy config model must not break the whole listing
        payload["available"] = False
        payload["reason"] = f"config model failed to build a schema: {type(e).__name__}: {e}"
    payload["templates"] = [t["name"] for t in list_templates(spec, include_params=False)]
    return payload


def all_job_type_payloads() -> list[dict[str, Any]]:
    return [job_type_payload(s) for s in list_job_types()]


def find_repo_dir(*names: str) -> Path | None:
    """First existing ``configs/<name>`` directory."""
    root = templates_root()
    if root is None:
        return None
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None
