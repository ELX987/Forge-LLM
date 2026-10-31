"""Eval reports in the Forge database: table ``evals`` + list / get / compare (with CIs, noise flags and Pareto frontier).

``compare`` never ranks values that are within noise: every difference carries ``distinguishable`` (CI excludes 0 **and**
|diff| exceeds the noise floor when one is known) and ``within_noise`` (its negation), and the Pareto frontier treats
quality differences below the noise tolerance as ties (cheaper wins ties; equal-bytes ties stay on the frontier together).
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

from forge.core.db import get_db, register_migration
from forge.core.errors import NotFound
from forge.core.schemas import EvalReport, MetricValue
from forge.core.util import now
from forge.eval import stats

register_migration(
    "eval",
    1,
    """
CREATE TABLE IF NOT EXISTS evals(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    artifact_hash TEXT NOT NULL,
    parent_hash TEXT,
    candidate_label TEXT NOT NULL DEFAULT '',
    parent_label TEXT NOT NULL DEFAULT '',
    gate_id TEXT,
    verdict TEXT,
    bytes REAL,
    bytes_provenance TEXT NOT NULL DEFAULT 'measured',
    params_b REAL,
    quality_metric TEXT NOT NULL DEFAULT 'suite_score',
    run_id TEXT,
    attachments_json TEXT NOT NULL DEFAULT '{}',
    config_json TEXT NOT NULL DEFAULT '{}',
    tags_json TEXT NOT NULL DEFAULT '[]',
    report_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evals_artifact ON evals(artifact_hash);
CREATE INDEX IF NOT EXISTS idx_evals_parent ON evals(parent_hash);
CREATE INDEX IF NOT EXISTS idx_evals_created ON evals(created_at);
""",
)


def save_report(
    report: EvalReport,
    *,
    config: dict[str, Any] | None = None,
    name: str = "",
    candidate_label: str = "",
    parent_label: str = "",
    bytes: float | None = None,  # noqa: A002 - column name
    bytes_provenance: str = "measured",
    params_b: float | None = None,
    quality_metric: str = "suite_score",
    run_id: str | None = None,
    attachments: dict[str, str] | None = None,
    tags: Sequence[str] = (),
) -> str:
    """Insert or replace ``report`` (by id).  Returns the report id."""
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO evals(id,name,created_at,artifact_hash,parent_hash,candidate_label,parent_label,gate_id,verdict,bytes,"
        "bytes_provenance,params_b,quality_metric,run_id,attachments_json,config_json,tags_json,report_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            report.id, name, report.created_at, report.artifact_hash, report.parent_hash, candidate_label, parent_label,
            report.gate.gate_id if report.gate else None, report.gate.verdict if report.gate else None, bytes, bytes_provenance, params_b,
            quality_metric, run_id, json.dumps(attachments or {}), json.dumps(config or {}, default=str), json.dumps(list(tags)),
            report.model_dump_json(),
        ),
    )
    return report.id


def _row_meta(r: dict[str, Any]) -> dict[str, Any]:
    r = dict(r)
    for k in ("attachments_json", "config_json", "tags_json"):
        if k in r:
            r[k[:-5]] = json.loads(r.pop(k) or "null")
    return r


def get_row(eval_id: str) -> dict[str, Any]:
    """Full row: metadata + parsed ``report``."""
    r = get_db().query_one("SELECT * FROM evals WHERE id=?", (eval_id,))
    if r is None:
        raise NotFound(f"no eval report {eval_id!r}")
    rep = json.loads(r.pop("report_json"))
    out = _row_meta(r)
    out["report"] = rep
    return out


def get_report(eval_id: str) -> EvalReport:
    return EvalReport(**get_row(eval_id)["report"])


def list_reports(
    *,
    artifact_hash: str | None = None,
    parent_hash: str | None = None,
    gate_id: str | None = None,
    verdict: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Summary rows (no metrics payload), newest first."""
    where, params = [], []
    for col, val in (("artifact_hash", artifact_hash), ("parent_hash", parent_hash), ("gate_id", gate_id), ("verdict", verdict)):
        if val:
            where.append(f"{col}=?")
            params.append(val)
    if q:
        where.append("(name LIKE ? OR candidate_label LIKE ? OR id LIKE ?)")
        params += [f"%{q}%"] * 3
    sql = ("SELECT id,name,created_at,artifact_hash,parent_hash,candidate_label,parent_label,gate_id,verdict,bytes,bytes_provenance,params_b,"
           "quality_metric,run_id,tags_json,attachments_json FROM evals" + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY created_at DESC LIMIT ? OFFSET ?")
    return [_row_meta(r) for r in get_db().query(sql, (*params, int(limit), int(offset)))]


def delete_report(eval_id: str) -> bool:
    cur = get_db().execute("DELETE FROM evals WHERE id=?", (eval_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------------
def _se(mv: dict[str, Any]) -> float | None:
    lo, hi = mv.get("ci_low"), mv.get("ci_high")
    if lo is None or hi is None:
        return None
    return max((hi - lo) / (2 * 1.959964), 0.0)


def _metric_cell(name: str, mv: dict[str, Any], base: dict[str, Any] | None, noise: dict[str, float], own: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "value": mv["value"], "ci_low": mv.get("ci_low"), "ci_high": mv.get("ci_high"), "n": mv.get("n"),
        "provenance": mv.get("provenance", "measured"),
    }
    nf = noise.get(name)
    if base is not None and name in base:
        b = base[name]
        diff = mv["value"] - b["value"]
        s1, s2 = _se(mv), _se(b)
        if s1 is not None and s2 is not None:
            se = math.hypot(s1, s2)
            lo, hi = diff - 1.959964 * se, diff + 1.959964 * se
        else:
            lo = hi = None
        d = stats.distinguishable(diff, noise_floor=nf, ci_low=lo, ci_high=hi)
        cell.update(diff_vs_baseline=diff, diff_ci_low=lo, diff_ci_high=hi, distinguishable=d.distinguishable, within_noise=not d.distinguishable,
                    diff_reason=d.reason, diff_ci_method="independent combination of the two CIs (conservative for paired data)")
    else:
        for key in (f"{name}_diff_pts", f"{name}_diff"):
            if key in own:
                dm = own[key]
                d = stats.distinguishable(dm["value"], noise_floor=nf, ci_low=dm.get("ci_low"), ci_high=dm.get("ci_high"))
                cell.update(diff_vs_own_parent=dm["value"], diff_ci_low=dm.get("ci_low"), diff_ci_high=dm.get("ci_high"),
                            distinguishable=d.distinguishable, within_noise=not d.distinguishable, diff_reason=d.reason,
                            diff_ci_method="paired bootstrap against the report's own parent")
                break
        else:
            if mv.get("within_noise") is not None:
                cell["within_noise"] = mv["within_noise"]
                cell["distinguishable"] = not mv["within_noise"]
    return cell


def pareto_frontier(points: Sequence[dict[str, Any]], *, tol_key: str = "quality_tol") -> list[dict[str, Any]]:
    """Noise-aware Pareto frontier over (bytes: lower is better, quality: higher is better).

    ``points``: dicts with ``id``, ``bytes``, ``quality`` and optional ``quality_tol`` (noise tolerance) and provenance info.
    A dominates B iff bytes_A <= bytes_B and either
      * quality_A > quality_B + tol  (distinguishably better), with at least one strict improvement, or
      * bytes_A < bytes_B and |quality_A - quality_B| <= tol  (cheaper at indistinguishable quality), or
      * bytes_A < bytes_B and quality_A >= quality_B.
    Equal-bytes points within tolerance of each other do not dominate each other (both stay)."""
    out = []
    for p in points:
        q = dict(p)
        q["dominated_by"] = []
        q["indistinguishable_from"] = []
        out.append(q)
    for i, b in enumerate(out):
        for j, a in enumerate(out):
            if i == j:
                continue
            tol = max(float(a.get(tol_key) or 0.0), float(b.get(tol_key) or 0.0))
            if abs(a["quality"] - b["quality"]) <= tol and a["bytes"] == b["bytes"]:
                b["indistinguishable_from"].append(a["id"])
            if a["bytes"] > b["bytes"]:
                continue
            better_q = a["quality"] > b["quality"] + tol
            cheaper = a["bytes"] < b["bytes"]
            within = abs(a["quality"] - b["quality"]) <= tol
            if (better_q and (a["bytes"] <= b["bytes"])) or (cheaper and (within or a["quality"] >= b["quality"])):
                b["dominated_by"].append(a["id"])
    for p in out:
        p["on_frontier"] = not p["dominated_by"]
    return out


def compare(
    ids: Sequence[str],
    *,
    parent: str | None = None,
    metrics: Sequence[str] | None = None,
    quality_metric: str = "suite_score",
) -> dict[str, Any]:
    """Comparison table for several eval reports.

    ``parent`` (an eval id from ``ids`` or an artifact hash of one of them) becomes the baseline; otherwise every row is
    compared with its own embedded parent (``<metric>_diff`` paired estimates).  Returns
    ``{"rows": [...], "metrics": [...], "baseline": id|None, "pareto": [...], "excluded": [...]}``."""
    rows_raw = [get_row(i) for i in ids]
    base_row = None
    if parent:
        for r in rows_raw:
            if r["id"] == parent or r["artifact_hash"] == parent:
                base_row = r
                break
        if base_row is None:
            try:
                base_row = get_row(parent)
            except NotFound:
                base_row = None
    base_metrics = base_row["report"]["metrics"] if base_row else None
    all_names: list[str] = []
    for r in rows_raw:
        for n in r["report"]["metrics"]:
            if n not in all_names:
                all_names.append(n)
    if metrics:
        all_names = list(metrics)
    rows = []
    for r in rows_raw:
        rep = r["report"]
        met = rep["metrics"]
        noise = rep.get("noise_floor") or {}
        is_base = base_row is not None and r["id"] == base_row["id"]
        cells = {n: _metric_cell(n, met[n], None if is_base else base_metrics, noise, met) for n in all_names if n in met}
        rows.append({
            "id": r["id"], "name": r["name"], "candidate_label": r["candidate_label"], "artifact_hash": r["artifact_hash"],
            "parent_hash": r["parent_hash"], "bytes": r["bytes"], "bytes_provenance": r["bytes_provenance"], "params_b": r["params_b"],
            "gate": rep.get("gate"), "metrics": cells, "is_baseline": is_base, "noise_floor": noise, "created_at": r["created_at"],
        })
    pts, excluded = [], []
    for row, r in zip(rows, rows_raw):
        q = row["metrics"].get(quality_metric)
        if row["bytes"] is None or q is None:
            excluded.append({"id": row["id"], "reason": "missing bytes" if row["bytes"] is None else f"missing quality metric {quality_metric!r}"})
            continue
        pts.append({"id": row["id"], "label": row["candidate_label"] or row["name"] or row["id"], "bytes": float(row["bytes"]), "quality": float(q["value"]),
                    "quality_tol": float(row["noise_floor"].get(quality_metric, 0.0) or 0.0),
                    "provenance": {"quality": q["provenance"], "bytes": row["bytes_provenance"]}, "artifact_hash": row["artifact_hash"],
                    "eval_id": row["id"], "quality_ci": [q["ci_low"], q["ci_high"]]})
    return {"rows": rows, "metrics": all_names, "baseline": base_row["id"] if base_row else None,
            "pareto": pareto_frontier(pts), "excluded": excluded, "quality_metric": quality_metric, "generated_at": now()}
