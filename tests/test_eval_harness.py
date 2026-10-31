from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from forge.core.context import NullContext
from forge.core.errors import NotFound
from forge.core.jsonschema import schema_for
from forge.core.schemas import EvalReport, MetricValue
from forge.eval import db as evdb
from forge.eval.config import EvalConfig, ModelSource, SuiteSelection
from forge.eval.noise import BatchPerturbedProvider
from forge.eval.run import run_eval
from forge.eval.testing import BigramProvider, SolverProvider

FAST_SUITES = [
    {"name": "distribution", "params": {"max_seq_len": 48}},
    {"name": "mcq"},
    {"name": "longctx", "params": {"lengths": [600], "n_per_task": 1, "max_new_tokens": 64}},
    {"name": "longgen", "params": {"steps": [4], "n_per_length": 4, "max_new_tokens": 300, "divergence_prompts": 2}},
    {"name": "safety"},
    {"name": "agentic", "params": {"n_per_kind": 2}},
]


def cfg(**kw):
    base = dict(name="t", candidate=ModelSource(kind="path", ref="cand", label="cand-w4"), parent=ModelSource(kind="path", ref="par", label="parent-bf16"),
                suites=FAST_SUITES, n_bootstrap=200, noise_repeats=0, deployment_length=600, context={"params_b": 8})
    base.update(kw)
    return EvalConfig(**base)


def test_config_schema_and_coercion():
    c = EvalConfig(suites=["distribution", {"name": "mcq", "params": {"scoring": "sum"}}])
    assert [s.name for s in c.suites] == ["distribution", "mcq"] and c.suites[1].params == {"scoring": "sum"}
    sch = schema_for(EvalConfig)
    props = sch["properties"]
    for k in ("candidate", "parent", "suites", "gate", "seeds", "n_bootstrap", "noise_repeats", "deployment_length"):
        assert k in props and props[k].get("description"), k
    with pytest.raises(Exception):
        EvalConfig(bogus=1)
    with pytest.raises(Exception):
        EvalConfig(seeds=[])


def test_identical_models_pass_g_w4_and_report_is_persisted():
    par = SolverProvider(1.0, name="parent")
    cand = SolverProvider(1.0, name="cand")
    ctx = NullContext()
    rep = run_eval(cfg(gate="g-w4", noise_repeats=2), ctx, providers={"candidate": cand, "parent": par})
    assert isinstance(rep, EvalReport)
    assert rep.gate.verdict == "pass", (rep.gate.reasons, rep.env["suite_errors"])
    assert rep.env["suite_errors"] == {} and rep.suite_versions["longctx"] == "1"
    m = rep.metrics
    assert m["kl_p99"].value == 0 and m["top1_flip"].value == 0 and m["suite_score"].value == m["suite_score_parent"].value
    assert m["suite_score_ratio"].value == pytest.approx(1.0) and m["longctx_ratio"].value == pytest.approx(1.0)
    assert m["longgen_reasoning"].value == pytest.approx(1.0) and m["safety_delta"].value == 0
    assert rep.noise_floor and "suite_score" in rep.noise_floor and "suite_score_diff_pts" in rep.noise_floor
    assert rep.env["noise_method"] and rep.env["has_parent"] is True
    # ctx plumbing
    assert ctx.outputs["report_id"] == rep.id and ctx.outputs["verdict"] == "pass" and ctx.outputs["gate"] == "pass"
    assert ctx.last_progress == 1.0 and any(k.startswith("distribution/") for _, d in ctx.metrics for k in d)
    # persisted + generations attached (EV-7)
    got = evdb.get_report(rep.id)
    assert got.gate.verdict == "pass" and got.metrics["kl_p99"].value == 0
    row = evdb.get_row(rep.id)
    assert row["candidate_label"] == "cand-w4" and row["bytes"] == par.nbytes() and row["params_b"] == 8 and row["verdict"] == "pass"
    gen_path = Path(row["attachments"]["generations_path"])
    assert gen_path.exists() and int(row["attachments"]["generations_count"]) > 10


def test_degraded_candidate_fails_gate_with_reasons():
    par = SolverProvider(1.0, name="parent")
    cand = SolverProvider(0.4, name="cand")
    import torch

    g = torch.Generator().manual_seed(1234)  # hermetic: never depend on the global RNG state left by other tests / devices
    cand.table = par.table + 3.0 * torch.randn(par.table.shape, generator=g)  # also distort the logits
    rep = run_eval(cfg(gate="g-w4"), providers={"candidate": cand, "parent": par}, persist=False)
    assert rep.gate.verdict == "fail"
    assert any("longgen" in r or "longctx" in r or "kl_p99" in r or "top1_flip" in r for r in rep.gate.reasons)
    assert rep.metrics["longctx_ratio"].value < 0.8 and rep.metrics["kl_p99"].value > 0.05
    assert rep.metrics["safety_delta"].within_noise is False


def test_no_parent_gives_absolute_scores_and_inconclusive_gate():
    cand = SolverProvider(1.0)
    c = cfg(parent=None, gate="g-w4", suites=[{"name": "mcq"}, {"name": "longgen", "params": {"steps": [3], "n_per_length": 3}}])
    rep = run_eval(c, providers={"candidate": cand}, persist=False)
    assert "mcq_acc" in rep.metrics and "mcq_acc_parent" not in rep.metrics and rep.parent_hash is None
    assert rep.gate.verdict == "inconclusive" and "requires a parent" in rep.gate.reasons[0]


def test_suite_failure_is_recorded_not_fatal_and_all_failing_raises():
    cand = SolverProvider(1.0)
    c = cfg(parent=None, suites=[{"name": "mcq"}, {"name": "mcq", "params": {"data_path": "/nonexistent/file.jsonl"}}])
    rep = run_eval(c, providers={"candidate": cand}, persist=False)
    assert "mcq" in rep.env["suite_errors"] and "mcq_acc" in rep.metrics
    from forge.core.errors import ForgeError

    with pytest.raises(ForgeError):
        run_eval(cfg(parent=None, suites=[{"name": "mcq", "params": {"data_path": "/nonexistent/file.jsonl"}}]), providers={"candidate": cand}, persist=False)


def test_extra_metrics_flags_and_gate_overrides():
    par = SolverProvider(1.0)
    cand = SolverProvider(1.0)
    c = cfg(gate="g-struct", suites=[{"name": "mcq"}], extra_flags={"size_target_met": True, "beats_fresh_student_arm": True, "licence_gate_passed": True},
            gate_overrides={"metrics": {"suite_score_ratio": {"min": 0.5}}})
    rep = run_eval(c, providers={"candidate": cand, "parent": par}, persist=False)
    assert rep.gate.verdict == "pass", rep.gate.reasons
    rep2 = run_eval(c.model_copy(update={"extra_flags": {"size_target_met": True, "beats_fresh_student_arm": False, "licence_gate_passed": True}}),
                    providers={"candidate": cand, "parent": par}, persist=False)
    assert rep2.gate.verdict == "fail"
    fp4 = cfg(gate="g-fp4train", suites=[{"name": "mcq"}], extra_metrics={"loss_gap_rel": 0.012}, extra_flags={"tr2_diagnostics_green": True, "healing_decision_documented": True})
    rep3 = run_eval(fp4, providers={"candidate": cand, "parent": par}, persist=False)
    assert rep3.metrics["loss_gap_rel"].value == 0.012 and rep3.gate.details["loss_gap_rel"]["status"] == "pass"


def test_lossless_gate_with_context():
    par = SolverProvider(1.0)
    c = cfg(gate="g-lossless", suites=[{"name": "distribution"}], context={"storage_transform": True})
    rep = run_eval(c, providers={"candidate": SolverProvider(1.0), "parent": par}, persist=False)
    assert rep.gate.verdict == "pass" and rep.metrics["max_abs_logit_diff"].value == 0
    noisy = par.perturbed(1e-3)
    rep = run_eval(c, providers={"candidate": noisy, "parent": par}, persist=False)
    assert rep.gate.verdict == "fail"


def test_noise_floor_uses_perturbations_and_flags_within_noise():
    par = SolverProvider(1.0)
    cand = SolverProvider(0.97, name="cand")
    c = cfg(suites=[{"name": "mcq"}, {"name": "longgen", "params": {"steps": [4], "n_per_length": 6}}], noise_repeats=4, seeds=[0, 1, 2])
    rep = run_eval(c, providers={"candidate": cand, "parent": par}, persist=False)
    assert rep.noise_floor["suite_score"] >= 0
    detail = rep.env["noise_detail"]["suite_score"]
    assert detail["n"] == 4 and detail["type"] == "score_halfwidth"
    d = rep.metrics["mcq_acc_diff_pts"]
    assert d.within_noise is not None


def test_batch_perturbed_provider_preserves_logits():
    import torch

    base = BigramProvider(seed=3)
    pert = BatchPerturbedProvider(base, seed=5)
    ids = torch.randint(0, 257, (5, 9))
    mask = torch.ones(5, 9, dtype=torch.bool)
    assert torch.allclose(pert.logits(ids, mask), base.logits(ids, mask))
    assert pert.nbytes() == base.nbytes() and len(pert.generate([1, 2], 4)) == 4


def test_load_source_errors_without_model_loader(tmp_path):
    c = EvalConfig(candidate=ModelSource(kind="path", ref=str(tmp_path / "nothing")), suites=[SuiteSelection(name="mcq")])
    with pytest.raises(Exception):
        run_eval(c, persist=False)


def test_contamination_only_run_needs_no_model():
    c = EvalConfig(candidate=ModelSource(kind="none"), suites=[SuiteSelection(name="contamination")])
    rep = run_eval(c, persist=False)
    assert rep.env["suite_errors"] == {} and rep.metrics == {}


# ------------------------------------------------------------------------------- db / compare
def _report(name, cand, par, gate=None, **kw):
    c = cfg(name=name, candidate=ModelSource(kind="path", ref=name, label=name), suites=[{"name": "mcq"}, {"name": "longgen", "params": {"steps": [4], "n_per_length": 6}}],
            gate=gate, **kw)
    return run_eval(c, providers={"candidate": cand, "parent": par})


def test_list_get_compare_and_pareto():
    par = SolverProvider(1.0, name="bf16")
    big = SolverProvider(1.0, name="w8")
    big._nbytes = 800
    small = SolverProvider(0.6, name="w4")
    small._nbytes = 400
    tiny = SolverProvider(0.0, name="w2")
    tiny._nbytes = 300
    wasteful = SolverProvider(0.6, name="w4-bad-pack")
    wasteful._nbytes = 900
    r_big, r_small, r_tiny, r_waste = (_report(n, p, par) for n, p in (("w8", big), ("w4", small), ("w2", tiny), ("w4b", wasteful)))
    ids = [r_big.id, r_small.id, r_tiny.id, r_waste.id]
    rows = evdb.list_reports(limit=100)
    assert {r["id"] for r in rows} >= set(ids)
    assert evdb.list_reports(artifact_hash=r_small.artifact_hash)[0]["id"] == r_small.id
    assert evdb.list_reports(q="w2")[0]["id"] == r_tiny.id
    with pytest.raises(NotFound):
        evdb.get_report("eval_missing")

    out = evdb.compare(ids, quality_metric="longgen_acc")
    assert out["baseline"] is None and out["quality_metric"] == "longgen_acc" and not out["excluded"]
    byid = {r["id"]: r for r in out["rows"]}
    cell = byid[r_small.id]["metrics"]["longgen_acc"]
    assert cell["value"] == r_small.metrics["longgen_acc"].value
    assert "diff_vs_own_parent" not in cell or cell["diff_ci_method"].startswith("paired")
    pareto = {p["id"]: p for p in out["pareto"]}
    assert pareto[r_big.id]["on_frontier"] and pareto[r_tiny.id]["on_frontier"]
    assert not pareto[r_waste.id]["on_frontier"] and r_small.id in pareto[r_waste.id]["dominated_by"]
    assert pareto[r_small.id]["on_frontier"]
    assert pareto[r_big.id]["provenance"]["bytes"] == "measured" and pareto[r_big.id]["provenance"]["quality"] == "measured"
    # explicit baseline: others are compared against it with CIs and distinguishability flags
    out2 = evdb.compare(ids, parent=r_big.id, metrics=["longgen_acc", "mcq_acc"])
    assert out2["baseline"] == r_big.id and out2["metrics"] == ["longgen_acc", "mcq_acc"]
    c2 = {r["id"]: r for r in out2["rows"]}[r_tiny.id]["metrics"]["longgen_acc"]
    assert c2["diff_vs_baseline"] == pytest.approx(r_tiny.metrics["longgen_acc"].value - r_big.metrics["longgen_acc"].value)
    assert c2["distinguishable"] is True and c2["within_noise"] is False
    assert "conservative" in c2["diff_ci_method"]
    assert {r["id"]: r for r in out2["rows"]}[r_big.id]["is_baseline"] is True
    assert evdb.delete_report(r_waste.id) and not evdb.delete_report(r_waste.id)


def test_pareto_frontier_noise_awareness():
    pts = [
        {"id": "a", "bytes": 100, "quality": 0.90, "quality_tol": 0.0},
        {"id": "b", "bytes": 50, "quality": 0.895, "quality_tol": 0.02},   # cheaper, indistinguishable from a -> dominates a
        {"id": "c", "bytes": 50, "quality": 0.80, "quality_tol": 0.02},    # same bytes as b, distinguishably worse -> dominated by b
        {"id": "d", "bytes": 20, "quality": 0.60, "quality_tol": 0.02},    # cheapest -> frontier
        {"id": "e", "bytes": 20, "quality": 0.61, "quality_tol": 0.02},    # same bytes, within noise of d -> both stay
    ]
    out = {p["id"]: p for p in evdb.pareto_frontier(pts)}
    assert out["a"]["dominated_by"] == ["b"] and not out["a"]["on_frontier"]
    assert not out["c"]["on_frontier"] and "b" in out["c"]["dominated_by"]
    assert out["b"]["on_frontier"] and out["d"]["on_frontier"] and out["e"]["on_frontier"]
    assert out["d"]["indistinguishable_from"] == ["e"]


def test_compare_excludes_rows_missing_bytes_or_quality():
    par = SolverProvider(1.0)
    cand = BigramProvider(seed=5)
    cand._nbytes = None  # type: ignore[attr-defined]
    rep = run_eval(cfg(suites=[{"name": "distribution"}]), providers={"candidate": cand, "parent": par})
    out = evdb.compare([rep.id])
    assert out["pareto"] == [] and out["excluded"][0]["id"] == rep.id
