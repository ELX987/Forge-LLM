"""Claims ledger (Plan 5.5): what the papers assert versus what Forge verified.

Table ``claims`` in the Forge database; CRUD; :func:`seed_claims` (>= 30 real claims from P1-P7, see ``claims_seed``);
:func:`claims_needing_attention` (the ledger's job 1: prioritise Phase-1 experiments by how many rules depend on a claim);
:func:`evaluate_claim` (turn a measurement into reproduced / partial / contradicted and link the experiment).

Planner defaults should only come from ``reproduced`` claims (risk R1): :func:`rule_evidence` summarises, per rule, how much of
its supporting evidence has actually been reproduced.
"""
from __future__ import annotations

import json
import math
from typing import Any, Iterable, Sequence

from forge.core.db import Database, get_db, register_migration
from forge.core.errors import ConfigError, NotFound
from forge.core.schemas import Claim
from forge.core.util import now

register_migration(
    "planner",
    1,
    """
CREATE TABLE IF NOT EXISTS claims(
    id TEXT PRIMARY KEY,
    source_paper TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    statement TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'untested',
    prior TEXT NOT NULL DEFAULT 'untrusted',
    linked_experiments_json TEXT NOT NULL DEFAULT '[]',
    depends_rules_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    expectation_json TEXT,
    seeded INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status_changed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_paper ON claims(source_paper);
""",
)

STATUSES = ("untested", "reproduced", "contradicted", "partial")
PRIORS = ("trusted", "untrusted")


def _db(db: Database | None) -> Database:
    return db or get_db()


def _row_to_claim(r: dict[str, Any]) -> Claim:
    return Claim(
        id=r["id"], source_paper=r["source_paper"], location=r["location"], statement=r["statement"], status=r["status"], prior=r["prior"],
        linked_experiments=json.loads(r["linked_experiments_json"]), depends_rules=json.loads(r["depends_rules_json"]), notes=r["notes"],
    )


# --------------------------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------------------------
def upsert_claim(claim: Claim | dict[str, Any], *, expectation: dict[str, Any] | None = None, db: Database | None = None) -> Claim:
    """Insert or replace a claim (all fields)."""
    c = claim if isinstance(claim, Claim) else Claim(**claim)
    d = _db(db)
    t = now()
    old = d.query_one("SELECT created_at, status, expectation_json FROM claims WHERE id=?", (c.id,))
    exp_json = json.dumps(expectation) if expectation is not None else (old["expectation_json"] if old else None)
    d.execute(
        "INSERT OR REPLACE INTO claims(id,source_paper,location,statement,status,prior,linked_experiments_json,depends_rules_json,notes,expectation_json,seeded,"
        "created_at,updated_at,status_changed_at) VALUES(?,?,?,?,?,?,?,?,?,?,COALESCE((SELECT seeded FROM claims WHERE id=?),0),?,?,?)",
        (c.id, c.source_paper, c.location, c.statement, c.status, c.prior, json.dumps(c.linked_experiments), json.dumps(c.depends_rules), c.notes, exp_json, c.id,
         old["created_at"] if old else t, t, t if (not old or old["status"] != c.status) else None),
    )
    return get_claim(c.id, db=d)


def get_claim(claim_id: str, *, db: Database | None = None) -> Claim:
    r = _db(db).query_one("SELECT * FROM claims WHERE id=?", (claim_id,))
    if r is None:
        raise NotFound(f"no claim {claim_id!r}")
    return _row_to_claim(r)


def get_claim_row(claim_id: str, *, db: Database | None = None) -> dict[str, Any]:
    """Claim plus ledger metadata (expectation, timestamps, dependents)."""
    r = _db(db).query_one("SELECT * FROM claims WHERE id=?", (claim_id,))
    if r is None:
        raise NotFound(f"no claim {claim_id!r}")
    out = _row_to_claim(r).model_dump()
    out["expectation"] = json.loads(r["expectation_json"]) if r["expectation_json"] else None
    out.update(created_at=r["created_at"], updated_at=r["updated_at"], status_changed_at=r["status_changed_at"], seeded=bool(r["seeded"]))
    return out


def list_claims(*, status: str | None = None, paper: str | None = None, prior: str | None = None, rule: str | None = None, q: str | None = None,
                limit: int = 500, offset: int = 0, db: Database | None = None) -> list[Claim]:
    where, params = [], []
    for col, val in (("status", status), ("source_paper", paper), ("prior", prior)):
        if val:
            where.append(f"{col}=?")
            params.append(val)
    if q:
        where.append("(statement LIKE ? OR id LIKE ? OR notes LIKE ?)")
        params += [f"%{q}%"] * 3
    sql = "SELECT * FROM claims" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY source_paper, id LIMIT ? OFFSET ?"
    rows = _db(db).query(sql, (*params, int(limit), int(offset)))
    out = [_row_to_claim(r) for r in rows]
    if rule:
        out = [c for c in out if rule in c.depends_rules]
    return out


def update_claim(claim_id: str, *, db: Database | None = None, **fields: Any) -> Claim:
    """Update ``status``, ``prior``, ``notes``, ``statement``, ``location``, ``linked_experiments`` or ``depends_rules``."""
    allowed = {"status", "prior", "notes", "statement", "location", "linked_experiments", "depends_rules"}
    bad = set(fields) - allowed
    if bad:
        raise ConfigError(f"cannot update claim fields {sorted(bad)}")
    cur = get_claim(claim_id, db=db)
    if "status" in fields and fields["status"] not in STATUSES:
        raise ConfigError(f"status must be one of {STATUSES}")
    if "prior" in fields and fields["prior"] not in PRIORS:
        raise ConfigError(f"prior must be one of {PRIORS}")
    data = cur.model_dump()
    data.update(fields)
    new = Claim(**data)
    d = _db(db)
    t = now()
    d.execute(
        "UPDATE claims SET source_paper=?, location=?, statement=?, status=?, prior=?, linked_experiments_json=?, depends_rules_json=?, notes=?, updated_at=?,"
        "status_changed_at=CASE WHEN status<>? THEN ? ELSE status_changed_at END WHERE id=?",
        (new.source_paper, new.location, new.statement, new.status, new.prior, json.dumps(new.linked_experiments), json.dumps(new.depends_rules), new.notes, t, new.status, t, claim_id),
    )
    return get_claim(claim_id, db=d)


def delete_claim(claim_id: str, *, db: Database | None = None) -> bool:
    return _db(db).execute("DELETE FROM claims WHERE id=?", (claim_id,)).rowcount > 0


def link_experiment(claim_id: str, experiment_id: str, *, db: Database | None = None) -> Claim:
    """Attach an eval report / job id to the claim (idempotent)."""
    c = get_claim(claim_id, db=db)
    if experiment_id in c.linked_experiments:
        return c
    return update_claim(claim_id, db=db, linked_experiments=[*c.linked_experiments, experiment_id])


def set_status(claim_id: str, status: str, *, notes: str | None = None, experiment: str | None = None, db: Database | None = None) -> Claim:
    fields: dict[str, Any] = {"status": status}
    c = get_claim(claim_id, db=db)
    if notes:
        fields["notes"] = (c.notes + "\n" if c.notes else "") + notes
    if experiment and experiment not in c.linked_experiments:
        fields["linked_experiments"] = [*c.linked_experiments, experiment]
    return update_claim(claim_id, db=db, **fields)


# --------------------------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------------------------
def seed_claims(*, overwrite: bool = False, db: Database | None = None) -> list[str]:
    """Insert the seed claims that are missing (never touches existing rows unless ``overwrite``: then statement/location/prior
    are refreshed but ``status``, ``linked_experiments`` and ``notes`` the user wrote are kept).  ``depends_rules`` is
    (re)derived from the rulebase.  Returns the ids inserted or refreshed."""
    from forge.planner.claims_seed import SEED_CLAIMS
    from forge.planner.rulebase import rule_ids_for_claim

    d = _db(db)
    touched = []
    for row in SEED_CLAIMS:
        deps = rule_ids_for_claim(row["id"])
        existing = d.query_one("SELECT * FROM claims WHERE id=?", (row["id"],))
        if existing is None:
            claim = Claim(id=row["id"], source_paper=row["source_paper"], location=row["location"], statement=row["statement"], status="untested", prior=row["prior"],
                          linked_experiments=[], depends_rules=deps, notes=row["notes"])
            upsert_claim(claim, expectation=row.get("expectation"), db=d)
            d.execute("UPDATE claims SET seeded=1 WHERE id=?", (row["id"],))
            touched.append(row["id"])
        elif overwrite:
            d.execute(
                "UPDATE claims SET source_paper=?, location=?, statement=?, prior=?, depends_rules_json=?, expectation_json=?, seeded=1, updated_at=? WHERE id=?",
                (row["source_paper"], row["location"], row["statement"], row["prior"], json.dumps(deps), json.dumps(row.get("expectation")) if row.get("expectation") else None, now(), row["id"]),
            )
            touched.append(row["id"])
        elif json.loads(existing["depends_rules_json"]) != deps:
            d.execute("UPDATE claims SET depends_rules_json=? WHERE id=?", (json.dumps(deps), row["id"]))
    return touched


def ensure_seeded(*, db: Database | None = None) -> int:
    """Seed on first use (UI/API convenience).  Returns the number of claims in the ledger."""
    d = _db(db)
    if (d.scalar("SELECT COUNT(*) FROM claims", default=0) or 0) == 0:
        seed_claims(db=d)
    return int(d.scalar("SELECT COUNT(*) FROM claims", default=0) or 0)


# --------------------------------------------------------------------------------------------
# attention ranking
# --------------------------------------------------------------------------------------------
_STATUS_WEIGHT = {"untested": 1.0, "partial": 0.6, "contradicted": 2.5, "reproduced": 0.0}
_PRIOR_WEIGHT = {"untrusted": 1.5, "trusted": 1.0}


def claims_needing_attention(limit: int = 10, *, include_unreferenced: bool = False, db: Database | None = None) -> list[dict[str, Any]]:
    """Claims ranked by how urgently they need an experiment: ``dependents x status weight x prior weight``.

    * contradicted claims outrank everything with the same fan-out (the rules relying on them must be revisited);
    * untrusted (vendor / single-source) claims count 1.5x;
    * reproduced claims are excluded; claims no rule depends on are excluded unless ``include_unreferenced``.
    Each row has ``score``, ``n_rules`` and a ``reason`` string for the dashboard."""
    d = _db(db)
    rows = d.query("SELECT * FROM claims")
    out = []
    for r in rows:
        c = _row_to_claim(r)
        w = _STATUS_WEIGHT[c.status]
        n = len(c.depends_rules)
        if w == 0.0 or (n == 0 and not include_unreferenced):
            continue
        score = max(n, 0.5 if include_unreferenced else 0) * w * _PRIOR_WEIGHT[c.prior]
        reason = f"{n} rule{'s' if n != 1 else ''} depend on it; status {c.status}; prior {c.prior}"
        if c.status == "contradicted":
            reason += " - rules relying on it must be revisited"
        out.append({**c.model_dump(), "score": round(score, 4), "n_rules": n, "reason": reason, "expectation": json.loads(r["expectation_json"]) if r["expectation_json"] else None})
    out.sort(key=lambda x: (-x["score"], x["id"]))
    return out[:limit]


def ledger_summary(*, db: Database | None = None) -> dict[str, Any]:
    d = _db(db)
    by_status = {r["status"]: r["n"] for r in d.query("SELECT status, COUNT(*) AS n FROM claims GROUP BY status")}
    by_prior = {r["prior"]: r["n"] for r in d.query("SELECT prior, COUNT(*) AS n FROM claims GROUP BY prior")}
    return {"total": sum(by_status.values()), "by_status": {s: by_status.get(s, 0) for s in STATUSES}, "by_prior": by_prior}


def rule_evidence(rule_id: str, *, db: Database | None = None) -> dict[str, Any]:
    """How much of a rule's evidence is reproduced?  ``{claims, reproduced, contradicted, untested, partial, ready}``;
    ``ready`` is True only if every claim is reproduced (Plan G1: every planner default points at a ledger entry)."""
    from forge.planner.rulebase import FLOW_BY_ID, RULES_BY_ID

    rule = RULES_BY_ID.get(rule_id) or FLOW_BY_ID.get(rule_id)
    if rule is None:
        raise NotFound(f"no rule {rule_id!r}")
    ids = rule["claims"]
    d = _db(db)
    counts = {s: 0 for s in STATUSES}
    missing = []
    for cid in ids:
        r = d.query_one("SELECT status FROM claims WHERE id=?", (cid,))
        if r is None:
            missing.append(cid)
        else:
            counts[r["status"]] += 1
    return {"rule": rule_id, "claims": list(ids), **counts, "missing": missing, "ready": bool(ids) and counts["reproduced"] == len(ids)}


# --------------------------------------------------------------------------------------------
# turning measurements into statuses
# --------------------------------------------------------------------------------------------
def evaluate_claim(
    claim_id: str,
    observed: float,
    *,
    experiment: str | None = None,
    expected: float | None = None,
    tolerance: float | None = None,
    direction: str | None = None,
    apply: bool = True,
    db: Database | None = None,
) -> dict[str, Any]:
    """Compare a measurement with the claim's expectation (stored or passed) and set ``reproduced`` / ``partial`` / ``contradicted``.

    * ``direction='approx'``: reproduced if ``|observed - expected| <= tol``; partial within ``2*tol``; else contradicted.
    * ``'at_least'``: reproduced if ``observed >= expected - tol``; partial if ``observed >= expected - 3*tol`` (tol=0: partial when
      within 10% of the target); else contradicted.   ``'at_most'`` is the mirror image.
    The verdict is only as good as the experiment: it links ``experiment`` (an eval report / job id) to the claim."""
    row = get_claim_row(claim_id, db=db)
    exp = dict(row["expectation"] or {})
    if expected is not None:
        exp["expected"] = expected
    if tolerance is not None:
        exp["tolerance"] = tolerance
    if direction is not None:
        exp["direction"] = direction
    if "expected" not in exp:
        raise ConfigError(f"claim {claim_id} has no machine-checkable expectation; pass expected=")
    e, tol, dr = float(exp["expected"]), float(exp.get("tolerance", 0.0)), exp.get("direction", "approx")
    if not math.isfinite(observed):
        raise ConfigError("observed value must be finite")
    wide = 3 * tol if tol > 0 else 0.1 * abs(e)
    if dr == "approx":
        w2 = 2 * tol if tol > 0 else 0.1 * abs(e)
        status = "reproduced" if abs(observed - e) <= tol else "partial" if abs(observed - e) <= w2 else "contradicted"
    elif dr == "at_least":
        status = "reproduced" if observed >= e - tol else "partial" if observed >= e - wide else "contradicted"
    elif dr == "at_most":
        status = "reproduced" if observed <= e + tol else "partial" if observed <= e + wide else "contradicted"
    else:
        raise ConfigError(f"unknown direction {dr!r}")
    msg = f"[{experiment or 'manual'}] observed {observed:g} vs expected {e:g} ({dr}, tol {tol:g}) -> {status}"
    result = {"claim": claim_id, "status": status, "observed": observed, "expected": e, "tolerance": tol, "direction": dr, "message": msg}
    if apply:
        set_status(claim_id, status, notes=msg, experiment=experiment, db=db)
    return result


def update_from_eval_report(report: Any, mapping: dict[str, str], *, db: Database | None = None) -> list[dict[str, Any]]:
    """Evaluate claims from an :class:`EvalReport`: ``mapping`` is ``{claim_id: metric_name}``; the metric's point value is the observation."""
    out = []
    for cid, metric in mapping.items():
        mv = report.metrics.get(metric)
        if mv is None:
            continue
        out.append(evaluate_claim(cid, mv.value, experiment=report.id, db=db))
    return out
