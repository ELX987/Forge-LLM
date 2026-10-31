"""Evaluation gates (Plan 10.1 + Appendix C.3): YAML-driven non-inferiority tests against the parent.

A gate is a set of **metric rules** (``min`` / ``max`` thresholds, ``within_noise``, non-inferiority margins), **flags**
(boolean evidence such as ``licence_gate_passed``) and statistical settings.  ``Gate.evaluate(Measurements)`` returns a
:class:`forge.core.schemas.GateVerdict` with verdict ``pass | fail | inconclusive`` and human-readable reasons.

Decision rule per threshold rule (``ci_policy: conservative``, the default)
-------------------------------------------------------------------------
``min`` metric (higher is better):  pass if the CI lower bound >= threshold; fail if the point estimate < threshold **and**
the CI upper bound < threshold; otherwise inconclusive.  ``max`` is symmetric.  Without a CI the point estimate decides.
``ci_policy: point`` decides on point estimates only.  The overall verdict is *fail* if any rule fails, *inconclusive* if none
fails but any is inconclusive/missing, else *pass*.

Non-inferiority rules (``noninferior_pts``) test ``candidate - parent >= -margin`` with the harness-supplied paired bootstrap
distribution; the one-sided p-values of all such rules in a gate are Benjamini-Hochberg adjusted (``fdr``) before comparing to
``alpha``.

Thresholds in the shipped presets are **TARGET** values recalibrated at Gate G1 (Plan 10.1), not validated constants.
"""
from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from forge.core.errors import ConfigError, NotFound
from forge.core.schemas import GateVerdict, MetricValue

log = logging.getLogger("forge.eval.gates")


# --------------------------------------------------------------------------------------------
# spec models
# --------------------------------------------------------------------------------------------
class MetricRule(BaseModel):
    """One rule of a gate over one named metric."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = Field(default=None, description="Metric must be >= this (higher is better).")
    max: float | None = Field(default=None, description="Metric must be <= this (lower is better).")
    applies_if: str | None = Field(default=None, description="Safe boolean expression over the run context (e.g. 'params_b >= 30'); if false, else_min/else_max apply.")
    else_min: float | None = Field(default=None, description="Replacement minimum when applies_if is false.")
    else_max: float | None = Field(default=None, description="Replacement maximum when applies_if is false.")
    at_length: Literal["deployment"] | None = Field(default=None, description="'deployment': the metric must have been measured at the deployment context length.")
    within_noise: bool | None = Field(default=None, description="True: the metric (a delta) must be indistinguishable from run-to-run noise.")
    noninferior: bool = Field(default=False, description="Test non-inferiority of this (paired difference) metric using the gate's margin.")
    noninferior_pts: float | None = Field(default=None, description="Non-inferiority margin in points for this rule (overrides the gate default).")
    required: bool = Field(default=True, description="If True a missing metric makes the verdict inconclusive.")
    description: str = Field(default="", description="Free text.")

    @field_validator("min", "max", "else_min", "else_max")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("thresholds must be finite")
        return v


class NoninferiorityCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    margin_pts: float | None = Field(default=None, ge=0, description="Default non-inferiority margin in points (0.01 of a 0-1 score = 1 point).")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="One-sided significance level.")
    fdr: Literal["benjamini_hochberg", "none"] = Field(default="benjamini_hochberg", description="Multiple-comparison control across the gate's non-inferiority tests.")


class NoiseFloorCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="from_parent_repeats", description="How the noise floor is estimated.")
    n: int = Field(default=5, ge=0, description="Number of parent repeats (different seeds / batch composition); 0 disables.")


class GateSpec(BaseModel):
    """Declarative gate definition (the YAML schema of Appendix C.3, extended)."""

    model_config = ConfigDict(extra="forbid")

    gate: str = Field(description="Gate id, e.g. 'G-W4-v2'.")
    status: str = Field(default="TARGET", description="TARGET until recalibrated at G1; then CALIBRATED.")
    title: str = Field(default="", description="Short title.")
    description: str = Field(default="", description="What the gate protects; applies-to classes.")
    applies_to: list[str] = Field(default_factory=list, description="Change classes this gate guards.")
    parent: Literal["required", "optional", "none"] = Field(default="required", description="Whether a parent comparison is needed.")
    ci_policy: Literal["conservative", "point"] = Field(default="conservative", description="conservative: use CI bounds; point: point estimates.")
    noninferiority: NoninferiorityCfg = Field(default_factory=NoninferiorityCfg, description="Non-inferiority settings.")
    metrics: dict[str, MetricRule] = Field(default_factory=dict, description="Metric rules.")
    flags: dict[str, str] = Field(default_factory=dict, description="Boolean evidence that must be true: name -> description.")
    noise_floor: NoiseFloorCfg = Field(default_factory=NoiseFloorCfg, description="Noise floor estimation.")
    suites: list[str] = Field(default_factory=list, description="Suites that produce the metrics (hint for UIs / the harness).")
    source: str = Field(default="", description="Plan section / paper provenance of the thresholds.")

    @field_validator("noise_floor", mode="before")
    @classmethod
    def _parse_noise(cls, v: Any) -> Any:
        if isinstance(v, str):
            m = re.fullmatch(r"\s*(\w+)\s*(?:\(\s*n\s*=\s*(\d+)\s*\))?\s*", v)
            if not m:
                raise ValueError(f"cannot parse noise_floor {v!r}")
            return {"source": m.group(1), "n": int(m.group(2) or 5)}
        return v

    @field_validator("metrics", mode="before")
    @classmethod
    def _parse_metrics(cls, v: Any) -> Any:
        return v or {}

    @field_validator("noninferiority", mode="before")
    @classmethod
    def _parse_ni(cls, v: Any) -> Any:
        return v or {}


# --------------------------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------------------------
@dataclass
class Measurements:
    """Everything a gate sees."""

    metrics: dict[str, MetricValue]
    boot: dict[str, np.ndarray] = field(default_factory=dict)
    noise_floor: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool | None] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # params_b, deployment_length, has_parent, ...


# --------------------------------------------------------------------------------------------
# safe expression evaluation
# --------------------------------------------------------------------------------------------
class _Unknown(Exception):
    pass


def eval_condition(expr: str, ctx: dict[str, Any]) -> bool:
    """Evaluate a restricted boolean expression (comparisons, and/or/not, names, numbers, strings).  Raises ``_Unknown``
    if a name is missing from ``ctx``."""
    tree = ast.parse(expr.strip(), mode="eval")

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in ("True", "False"):
                return n.id == "True"
            if n.id not in ctx or ctx[n.id] is None:
                raise _Unknown(n.id)
            return ctx[n.id]
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                ok = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                      ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}.get(type(op))
                if ok is None:
                    raise ConfigError(f"unsupported operator in condition {expr!r}")
                if not ok(left, right):
                    return False
                left = right
            return True
        raise ConfigError(f"unsupported syntax in condition {expr!r}")

    return bool(ev(tree))


# --------------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------------
class Gate:
    """Executable gate built from a :class:`GateSpec`."""

    def __init__(self, spec: GateSpec) -> None:
        self.spec = spec

    @property
    def id(self) -> str:
        return self.spec.gate

    # -- threshold resolution --------------------------------------------------------------
    def _thresholds(self, rule: MetricRule, ctx: dict[str, Any]) -> tuple[float | None, float | None, str]:
        note = ""
        if rule.applies_if:
            try:
                applies = eval_condition(rule.applies_if, ctx)
            except _Unknown as e:
                # unknown context: use the stricter of the two thresholds and say so
                lo = max([v for v in (rule.min, rule.else_min) if v is not None], default=None)
                hi = min([v for v in (rule.max, rule.else_max) if v is not None], default=None)
                return lo, hi, f"context value '{e}' unknown for '{rule.applies_if}'; stricter threshold used"
            if not applies:
                return rule.else_min, rule.else_max, f"'{rule.applies_if}' is false -> else-threshold"
        return rule.min, rule.max, note

    def required_metrics(self) -> list[str]:
        return [k for k, r in self.spec.metrics.items() if r.required]

    # -- evaluation ------------------------------------------------------------------------
    def evaluate(self, m: Measurements) -> GateVerdict:
        spec = self.spec
        ctx = dict(m.context)
        reasons: list[str] = []
        details: dict[str, Any] = {}
        status: dict[str, str] = {}
        ni_tests: list[tuple[str, float, float, float, float]] = []  # name, diff, margin, p, p_inf

        has_parent = ctx.get("has_parent", True)
        if spec.parent == "required" and not has_parent:
            reasons.append("gate requires a parent model for the paired comparison but none was evaluated")
            return GateVerdict(gate_id=spec.gate, verdict="inconclusive", reasons=reasons, details={"status": "no_parent"})

        for name, rule in spec.metrics.items():
            mv = m.metrics.get(name)
            d: dict[str, Any] = {}
            details[name] = d
            if mv is None:
                if rule.required:
                    status[name] = "inconclusive"
                    d["status"] = "missing"
                    reasons.append(f"{name}: metric not measured (run the suite that produces it)")
                else:
                    d["status"] = "skipped"
                continue
            d.update(value=mv.value, ci=[mv.ci_low, mv.ci_high], n=mv.n, provenance=mv.provenance.value)
            lo_thr, hi_thr, note = self._thresholds(rule, ctx)
            if note:
                d["note"] = note
            st = "pass"
            msgs: list[str] = []

            if rule.at_length == "deployment" and m.flags.get("longctx_length_matches_deployment") is False:
                st = "inconclusive"
                msgs.append(f"{name}: not measured at the deployment context length ({ctx.get('deployment_length')})")

            def judge_min(thr: float) -> tuple[str, str]:
                v, cl, ch = mv.value, mv.ci_low, mv.ci_high
                if spec.ci_policy == "point" or cl is None:
                    return ("pass", "") if v >= thr else ("fail", f"{name}={v:.4g} < min {thr:g}")
                if cl >= thr:
                    return "pass", ""
                if v < thr and ch is not None and ch < thr:
                    return "fail", f"{name}={v:.4g} (CI {cl:.4g}..{ch:.4g}) is below min {thr:g}"
                return "inconclusive", f"{name}={v:.4g} (CI {cl:.4g}..{ch if ch is not None else float('nan'):.4g}) cannot be separated from min {thr:g}"

            def judge_max(thr: float) -> tuple[str, str]:
                v, cl, ch = mv.value, mv.ci_low, mv.ci_high
                if spec.ci_policy == "point" or ch is None:
                    return ("pass", "") if v <= thr else ("fail", f"{name}={v:.4g} > max {thr:g}")
                if ch <= thr:
                    return "pass", ""
                if v > thr and cl is not None and cl > thr:
                    return "fail", f"{name}={v:.4g} (CI {cl:.4g}..{ch:.4g}) is above max {thr:g}"
                return "inconclusive", f"{name}={v:.4g} (CI {cl if cl is not None else float('nan'):.4g}..{ch:.4g}) cannot be separated from max {thr:g}"

            for thr, fn in ((lo_thr, judge_min), (hi_thr, judge_max)):
                if thr is None:
                    continue
                s, msg = fn(thr)
                d.setdefault("thresholds", {})["min" if fn is judge_min else "max"] = thr
                if msg:
                    msgs.append(msg)
                st = _worse(st, s)

            if rule.within_noise:
                wn = mv.within_noise
                nf = m.noise_floor.get(name)
                if wn is None and nf is not None:
                    wn = abs(mv.value) <= nf
                    d["noise_floor"] = nf
                if wn is None and mv.ci_low is not None and mv.ci_high is not None:
                    wn = bool(mv.ci_low <= 0.0 <= mv.ci_high)
                    d["note"] = (d.get("note", "") + " within-noise judged from CI containing 0").strip()
                if wn is True:
                    pass
                elif wn is False:
                    st = _worse(st, "fail")
                    msgs.append(f"{name}={mv.value:.4g} is distinguishable from noise (not within noise)")
                else:
                    st = _worse(st, "inconclusive")
                    msgs.append(f"{name}: no noise floor / CI to decide 'within noise'")

            if rule.noninferior or rule.noninferior_pts is not None:
                margin = rule.noninferior_pts if rule.noninferior_pts is not None else spec.noninferiority.margin_pts
                if margin is None:
                    raise ConfigError(f"gate {spec.gate}: rule {name} is noninferior but no margin_pts is configured")
                p, p_inf = self._ni_p(name, mv, m, margin)
                ni_tests.append((name, mv.value, margin, p, p_inf))
                d["noninferiority"] = {"margin_pts": margin, "p_value": p, "p_inferior": p_inf}
            status[name] = st
            d["status"] = st
            reasons.extend(msgs)

        # non-inferiority: FDR across all NI tests of this gate
        if ni_tests:
            ps = [t[3] for t in ni_tests]
            if spec.noninferiority.fdr == "benjamini_hochberg":
                from forge.eval.stats import benjamini_hochberg

                _, adj = benjamini_hochberg(ps, spec.noninferiority.alpha)
            else:
                adj = ps
            alpha = spec.noninferiority.alpha
            for (name, diff, margin, p, p_inf), ap in zip(ni_tests, adj):
                d = details[name]
                d["noninferiority"]["adjusted_p"] = ap
                if ap < alpha:
                    s = "pass"
                elif p_inf < alpha:
                    s = "fail"
                    reasons.append(f"{name}: candidate is inferior by more than the {margin:g}-pt margin (diff {diff:+.4g})")
                else:
                    s = "inconclusive"
                    reasons.append(f"{name}: non-inferiority at margin {margin:g} pts not established (adjusted p={ap:.3g}, diff {diff:+.4g})")
                status[name] = _worse(status.get(name, "pass"), s)
                d["status"] = status[name]

        for fname, desc in spec.flags.items():
            val = m.flags.get(fname)
            details[f"flag:{fname}"] = {"value": val, "description": desc}
            if val is True:
                status[f"flag:{fname}"] = "pass"
            elif val is False:
                status[f"flag:{fname}"] = "fail"
                reasons.append(f"flag {fname} is false: {desc}")
            else:
                status[f"flag:{fname}"] = "inconclusive"
                reasons.append(f"flag {fname} not provided: {desc}")

        verdicts = list(status.values())
        if not verdicts:
            verdict = "inconclusive"
            reasons.append("gate has no evaluable rules")
        elif "fail" in verdicts:
            verdict = "fail"
        elif "inconclusive" in verdicts:
            verdict = "inconclusive"
        else:
            verdict = "pass"
        if verdict == "pass":
            reasons.append(f"all {len(verdicts)} rules satisfied ({spec.status} thresholds)")
        details["_status"] = status
        details["_gate_status"] = spec.status
        return GateVerdict(gate_id=spec.gate, verdict=verdict, reasons=reasons, details=details)

    def _ni_p(self, name: str, mv: MetricValue, m: Measurements, margin: float) -> tuple[float, float]:
        boot = m.boot.get(name)
        if boot is not None and len(boot) > 1:
            b = np.asarray(boot, dtype=np.float64)
            b = b[np.isfinite(b)]
            return float((np.sum(b <= -margin) + 1) / (b.size + 1)), float((np.sum(b >= -margin) + 1) / (b.size + 1))
        if mv.ci_low is not None and mv.ci_high is not None:
            from forge.eval.stats import noninferiority_from_summary

            se = max((mv.ci_high - mv.ci_low) / (2 * 1.959964), 1e-12)
            r = noninferiority_from_summary(mv.value, se, margin, alpha=self.spec.noninferiority.alpha)
            return r.p_value, r.p_inferior
        # point estimate only: neither non-inferiority nor inferiority can be established
        return 1.0, 1.0


def _worse(a: str, b: str) -> str:
    order = {"pass": 0, "inconclusive": 1, "fail": 2}
    return a if order[a] >= order[b] else b


# --------------------------------------------------------------------------------------------
# loading / registry
# --------------------------------------------------------------------------------------------
def gate_dirs() -> list[Path]:
    from forge.core.paths import forge_home

    dirs = [forge_home() / "configs" / "gates"]
    repo = Path(__file__).resolve().parents[2] / "configs" / "gates"
    dirs.append(repo)
    return dirs


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def gate_from_dict(d: dict[str, Any]) -> Gate:
    try:
        return Gate(GateSpec(**d))
    except Exception as e:
        raise ConfigError(f"invalid gate definition: {e}") from e


def load_gate_file(path: str | Path) -> Gate:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: gate file must contain a mapping")
    g = gate_from_dict(data)
    return g


def _all_files() -> list[Path]:
    seen: dict[str, Path] = {}
    for d in gate_dirs():
        if d.is_dir():
            for f in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
                seen.setdefault(f.stem.lower(), f)  # earlier dirs (user) win
    return list(seen.values())


def list_gates() -> list[dict[str, Any]]:
    """For the UI/API: id, title, status, description, file for every available gate."""
    out = []
    ids: set[str] = set()
    for f in _all_files():
        try:
            g = load_gate_file(f)
        except Exception as e:
            log.warning("skipping gate file %s: %s", f, e)
            continue
        ids.add(_norm(g.id))
        out.append({"id": g.id, "file": str(f), "stem": f.stem, "title": g.spec.title, "status": g.spec.status,
                    "description": g.spec.description, "applies_to": g.spec.applies_to, "builtin": False})
    try:
        from forge.eval.gate_presets import PRESETS

        for key, data in PRESETS.items():
            if _norm(data["gate"]) not in ids:
                out.append({"id": data["gate"], "file": None, "stem": key, "title": data.get("title", ""), "status": data.get("status", "TARGET"),
                            "description": data.get("description", ""), "applies_to": data.get("applies_to", []), "builtin": True})
    except ImportError:
        pass
    return sorted(out, key=lambda r: r["id"])


def load_gate(name_or_path: str | Path) -> Gate:
    """Load by file path, by id (``G-W4-v2``), by file stem (``g-w4``) or by id without the version (``G-W4``)."""
    p = Path(str(name_or_path))
    if p.suffix in (".yaml", ".yml") and p.exists():
        return load_gate_file(p)
    key = _norm(str(name_or_path))
    cands: list[tuple[int, Gate]] = []
    for f in _all_files():
        try:
            g = load_gate_file(f)
        except Exception:
            continue
        ids = {_norm(g.id), _norm(f.stem), _norm(re.sub(r"-v\d+$", "", g.id, flags=re.I))}
        if key in ids:
            cands.append((0 if _norm(g.id) == key or _norm(f.stem) == key else 1, g))
    if cands:
        return sorted(cands, key=lambda t: t[0])[0][1]
    try:
        from forge.eval.gate_presets import PRESETS

        for k, data in PRESETS.items():
            if key in {_norm(data["gate"]), _norm(k), _norm(re.sub(r"-v\d+$", "", data["gate"], flags=re.I))}:
                return gate_from_dict(data)
    except ImportError:
        pass
    raise NotFound(f"no gate named {str(name_or_path)!r}; available: {', '.join(g['id'] for g in list_gates())}")


def register_gates() -> list[str]:
    """Register all gates under kind ``gate`` in :mod:`forge.core.registry` (idempotent)."""
    from forge.core.registry import register_obj

    names = []
    for row in list_gates():
        try:
            g = load_gate(row["file"] or row["stem"])
        except Exception:
            continue
        register_obj("gate", g.id, g, meta={"status": g.spec.status, "title": g.spec.title})
        names.append(g.id)
    return names


def regenerate_presets(out_path: str | Path | None = None) -> Path:
    """Developer helper: rewrite ``forge/eval/gate_presets.py`` (embedded fallback copy) from ``configs/gates/*.yaml``.

    Run after editing a preset: ``python -c "from forge.eval.gates import regenerate_presets; regenerate_presets()"``;
    ``tests/test_eval_gates.py`` fails if the two drift apart."""
    import pprint

    src = Path(__file__).resolve().parents[2] / "configs" / "gates"
    out = Path(out_path) if out_path else Path(__file__).resolve().parent / "gate_presets.py"
    lines = [
        '"""Embedded copy of configs/gates/*.yaml so gates work when the package is installed without the repo\'s configs/ folder.',
        "",
        "AUTO-GENERATED by forge.eval.gates.regenerate_presets() from configs/gates/*.yaml; tests/test_eval_gates.py checks it stays in sync.",
        "Edit the YAML files, then regenerate.  Thresholds are TARGET values (Plan 10.1).",
        '"""',
        "from __future__ import annotations",
        "",
        "PRESETS: dict[str, dict] = {",
    ]
    for f in sorted(src.glob("*.yaml")):
        d = yaml.safe_load(f.read_text(encoding="utf-8"))
        d["description"] = " ".join(str(d.get("description", "")).split())
        lines.append(f"    {f.stem!r}: " + pprint.pformat(d, width=120, sort_dicts=False).replace("\n", "\n    ") + ",")
    lines.append("}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
