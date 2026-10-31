"""/evals (read-only view of the table written by forge.eval) and /gates."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Query, Response

from forge.api.common import lazy_module
from forge.api.deps import Services, get_services
from forge.api.schemas import EvalCompareRequest
from forge.core.errors import ConfigError, NotFound
from forge.jobs.templates import find_repo_dir

router = APIRouter(tags=["evals"])


# --------------------------------------------------------------------------------------------
# defensive reading of the `evals` table (owner: forge.eval; schema unknown to this module)
# --------------------------------------------------------------------------------------------
def _columns(db: Any, table: str) -> list[str]:
    try:
        db.ensure_migrated()  # a sibling module may have registered its table after this handle was opened
        return [r["name"] for r in db.query(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, str) and (k.endswith("_json") or (v[:1] in "{[" and v[-1:] in "}]")):
            try:
                out[k[:-5] if k.endswith("_json") else k] = json.loads(v)
                continue
            except ValueError:
                pass
        out[k] = v
    return out


def _eval_mod() -> Any | None:
    mod = lazy_module("forge.eval.db")
    return mod if mod is not None and hasattr(mod, "list_reports") and hasattr(mod, "compare") else None


def read_evals(db: Any, *, limit: int = 100, offset: int = 0, where: str = "", params: tuple = ()) -> list[dict[str, Any]]:
    cols = _columns(db, "evals")
    if not cols:
        return []
    order = "created_at DESC" if "created_at" in cols else "rowid DESC"
    try:
        rows = db.query(f"SELECT * FROM evals {where} ORDER BY {order} LIMIT ? OFFSET ?", (*params, int(limit), int(offset)))
    except sqlite3.Error:
        return []
    return [_decode_row(r) for r in rows]


def _find_eval(db: Any, eval_id: str) -> dict[str, Any] | None:
    cols = _columns(db, "evals")
    for key in ("id", "eval_id", "report_id"):
        if key in cols:
            rows = read_evals(db, limit=1, where=f"WHERE {key}=?", params=(eval_id,))
            if rows:
                return rows[0]
    return None


def _metrics_of(ev: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalise to ``{name: {value, ci_low, ci_high, n, provenance, within_noise}}``."""
    src: Any = None
    for key in ("metrics", "report", "report_json", "data"):
        v = ev.get(key)
        if isinstance(v, dict):
            src = v.get("metrics", v) if key != "metrics" else v
            break
    out: dict[str, dict[str, Any]] = {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = {"value": v.get("value"), "ci_low": v.get("ci_low"), "ci_high": v.get("ci_high"), "n": v.get("n"), "provenance": v.get("provenance"), "within_noise": v.get("within_noise")}
        elif isinstance(v, (int, float)):
            out[k] = {"value": float(v), "ci_low": None, "ci_high": None, "n": None, "provenance": None, "within_noise": None}
    return out


def pareto_front(points: list[tuple[float, float]], x_min: bool = True, y_min: bool = False) -> list[bool]:
    """Boolean mask of non-dominated points (x minimised if ``x_min`` else maximised; same for y)."""
    sx = 1.0 if x_min else -1.0
    sy = 1.0 if y_min else -1.0
    mask = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if sx * xj <= sx * xi and sy * yj <= sy * yi and (sx * xj < sx * xi or sy * yj < sy * yi):
                dominated = True
                break
        mask.append(not dominated)
    return mask


def _distinguishable(a: dict[str, Any], b: dict[str, Any]) -> bool | None:
    """True when the confidence intervals of a and b do not overlap; None without CIs."""
    try:
        if None in (a["ci_low"], a["ci_high"], b["ci_low"], b["ci_high"]):
            return None
        return a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]
    except KeyError:
        return None


@router.get("/evals", summary="Stored evaluation reports (newest first)")
def list_evals(
    response: Response,
    artifact: str | None = Query(None, description="Filter by candidate artifact hash"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    svc: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    db = svc.metrics.db
    mod = _eval_mod()
    cols = _columns(db, "evals")
    if mod is not None and cols:
        response.headers["X-Total-Count"] = str(db.scalar("SELECT COUNT(*) FROM evals", (), 0))
        return mod.list_reports(artifact_hash=artifact, limit=limit, offset=offset)
    where, params = "", ()
    if artifact and "artifact_hash" in cols:
        where, params = "WHERE artifact_hash=?", (artifact,)
    rows = read_evals(db, limit=limit, offset=offset, where=where, params=params)
    response.headers["X-Total-Count"] = str(db.scalar("SELECT COUNT(*) FROM evals", (), 0) if cols else 0)
    return rows


@router.get("/evals/{eval_id}")
def get_eval(eval_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    mod = _eval_mod()
    if mod is not None and hasattr(mod, "get_row") and _columns(svc.metrics.db, "evals"):
        ev = mod.get_row(eval_id)  # raises NotFound -> 404
        ev["_metrics"] = _metrics_of(ev)
        return ev
    ev = _find_eval(svc.metrics.db, eval_id)
    if ev is None:
        raise NotFound(f"eval {eval_id!r} not found")
    ev["_metrics"] = _metrics_of(ev)
    return ev


@router.post("/evals/compare", summary="Side-by-side table with deltas, CI-based distinguishability and a Pareto front")
def compare(body: EvalCompareRequest, svc: Services = Depends(get_services)) -> dict[str, Any]:
    db = svc.metrics.db
    mod = _eval_mod()
    if mod is not None and _columns(db, "evals"):
        kw: dict[str, Any] = {"parent": body.parent}
        if body.metrics:
            kw["metrics"] = body.metrics
        if body.quality_metric:
            kw["quality_metric"] = body.quality_metric
        return mod.compare(body.ids, **kw)
    evs = []
    for i in body.ids:
        ev = _find_eval(db, i)
        if ev is None:
            raise NotFound(f"eval {i!r} not found")
        evs.append((i, ev, _metrics_of(ev)))
    parent_id = body.parent or body.ids[0]
    parent = next(((i, e, m) for i, e, m in evs if i == parent_id), None)
    if parent is None:
        pe = _find_eval(db, parent_id)
        if pe is None:
            raise NotFound(f"parent eval {parent_id!r} not found")
        parent = (parent_id, pe, _metrics_of(pe))
    names: list[str] = []
    for _i, _e, m in evs + [parent]:
        for k in m:
            if k not in names:
                names.append(k)
    rows = []
    for name in names:
        pm = parent[2].get(name)
        cells = {}
        for i, _e, m in evs:
            cm = m.get(name)
            if cm is None:
                cells[i] = None
                continue
            cell = dict(cm)
            if pm and pm["value"] is not None and cm["value"] is not None:
                cell["delta"] = cm["value"] - pm["value"]
                cell["delta_rel"] = (cm["value"] - pm["value"]) / abs(pm["value"]) if pm["value"] else None
                cell["distinguishable"] = _distinguishable(cm, pm) if i != parent_id else None
            cells[i] = cell
        rows.append({"metric": name, "parent": pm, "cells": cells})
    out: dict[str, Any] = {"parent": parent_id, "ids": body.ids, "rows": rows, "gates": {i: e.get("gate") or e.get("verdict") for i, e, _m in evs}}
    xk, yk = body.pareto.get("x"), body.pareto.get("y")
    if not (xk and yk) and len(names) >= 2:
        xk, yk = names[0], names[1]
    if xk and yk:
        pts, who = [], []
        for i, _e, m in evs:
            if xk in m and yk in m and m[xk]["value"] is not None and m[yk]["value"] is not None:
                pts.append((float(m[xk]["value"]), float(m[yk]["value"])))
                who.append(i)
        mask = pareto_front(pts, bool(body.pareto.get("x_min", True)), bool(body.pareto.get("y_min", False)))
        out["pareto"] = {"x": xk, "y": yk, "points": [{"id": i, "x": p[0], "y": p[1], "on_front": f} for i, p, f in zip(who, pts, mask, strict=True)]}
    return out


# --------------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------------
def _gate_files() -> dict[str, Any]:
    d = find_repo_dir("gates")
    out = {}
    if d:
        for f in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
            out[f.stem] = f
    return out


@router.get("/gates", summary="Gate definitions (configs/gates/*.yaml + registered gate classes)")
def list_gates() -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for gid, f in _gate_files().items():
        try:
            doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            doc = {}
        out[gid] = {"id": gid, "source": f"gates/{f.name}", "description": (doc.get("description") if isinstance(doc, dict) else "") or "", "registered": False}
    lazy_module("forge.eval.gates")
    from forge.core.registry import describe

    for e in describe("gate"):
        d = out.setdefault(e["name"], {"id": e["name"], "source": "registry", "description": ""})
        d["registered"] = True
        d["description"] = d["description"] or e.get("doc", "")
    return sorted(out.values(), key=lambda x: x["id"])


@router.get("/gates/{gate_id}")
def get_gate(gate_id: str) -> dict[str, Any]:
    files = _gate_files()
    lazy_module("forge.eval.gates")
    from forge.core import registry

    if gate_id not in files and not registry.has("gate", gate_id):
        raise NotFound(f"gate {gate_id!r} not found")
    out: dict[str, Any] = {"id": gate_id, "registered": registry.has("gate", gate_id)}
    if gate_id in files:
        text = files[gate_id].read_text(encoding="utf-8")
        out["yaml"] = text
        try:
            out["definition"] = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError(f"gate file is not valid YAML: {e}") from e
    return out
