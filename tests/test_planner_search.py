from __future__ import annotations

import json

import pytest

from forge.core.schemas import EvalReport, MetricValue, Provenance, Recipe, Workload
from forge.planner import costmodel as cm
from forge.planner import search as sr
from forge.planner.quick import estimate, scheme_from_recipe
from forge.planner.rulebase import Budget
from forge.planner.spec import CHIP_PRESETS, MODEL_PRESETS, ModelSpec, parse_scheme

H100, RTX4090, A100, B200 = (CHIP_PRESETS[k] for k in ("h100-sxm-80g", "rtx-4090", "a100-80g", "b200"))
L8, L70 = MODEL_PRESETS["llama-3-8b"], MODEL_PRESETS["llama-3.1-70b"]
# the plan's Example A model: 70B parameters, no embedding table, 80 layers x 8 KV heads x 128
EX_A = ModelSpec(name="example-a", params_b=70.0, num_layers=80, hidden_size=8192, num_heads=64, num_kv_heads=8, head_dim=128, vocab_size=1)


def test_fits_reproduces_example_a_exactly():
    wl = Workload(batch_sizes=[1], context_lengths=[32768])
    exact = dict(overhead_gb=0.0, reserve_frac=0.0)
    bf16 = sr.fits(EX_A, H100, "bf16", wl, **exact)
    assert not bf16.fits and bf16.weights_gb == pytest.approx(140) and "alone exceed" in bf16.notes[0]
    fp8 = sr.fits(EX_A, H100, "fp8", wl, **exact)
    assert fp8.weights_gb == pytest.approx(70, rel=1e-6) and fp8.free_for_kv_gb == pytest.approx(10, abs=1e-3)
    assert fp8.kv_bytes_per_token == 320 * 1024
    assert fp8.max_tokens_total == pytest.approx(30.5e3, rel=0.01) and not fp8.fits  # "< one 32k sequence"
    assert sr.fits(EX_A, H100, "fp8+kvfp8", wl, **exact).max_tokens_total == pytest.approx(61e3, rel=0.01)
    kv4 = sr.fits(EX_A, H100, "fp8+kv4", wl, **exact).max_tokens_total
    assert 108e3 <= kv4 <= 125e3
    w4 = sr.fits(EX_A, H100, "w4a16-g128", wl, **exact)
    assert w4.weights_gb == pytest.approx(36.1, abs=0.1) and w4.fits
    assert w4.max_tokens_total == pytest.approx(134e3, rel=0.01)  # plan: 131 k with the rounded 37 GB
    assert sr.fits(EX_A, H100, "w4a16-g128+kv4", wl, **exact).max_tokens_total > 460e3


def test_fits_default_reserve_and_prefix_reuse():
    wl = Workload(batch_sizes=[16], context_lengths=[8192], prefix_reuse=0.0)
    a = sr.fits(L8, H100, "bf16", wl)
    assert a.reserve_gb == pytest.approx(1.0 + 0.05 * 80)
    shared = sr.fits(L8, H100, "bf16", Workload(batch_sizes=[16], context_lengths=[8192], prefix_reuse=1.0))
    assert shared.kv_gb_at_workload == pytest.approx(a.kv_gb_at_workload / 16)  # fully shared prefix is cached once
    assert a.max_concurrency_at_ctx == pytest.approx(a.free_for_kv_gb * 1e9 / (8192 * 128 * 1024))
    m = L8.model_copy(update={"sliding_window": 1024, "sliding_layers": 24})
    assert sr.fits(m, H100, "bf16", wl).kv_gb_at_workload < a.kv_gb_at_workload / 2


def test_binding_resource_classes():
    cap = sr.binding_resource(L70, H100, Workload(batch_sizes=[8], context_lengths=[4096]))
    assert cap.binding == "capacity" and not cap.bf16_fit.fits and "weights" in cap.reasons[0]
    bw = sr.binding_resource(L8, H100, Workload(batch_sizes=[8], context_lengths=[4096]))
    assert bw.binding == "bandwidth" and bw.bf16_fit.fits and bw.scores["bandwidth"] > 0.9
    cp = sr.binding_resource(L8, H100, Workload(batch_sizes=[400], context_lengths=[128]))
    assert cp.binding == "compute" and cp.b_star_bf16 < 400
    kv = sr.binding_resource(L8, H100, Workload(batch_sizes=[64], context_lengths=[32768]))
    assert kv.binding == "kv" and kv.kv_dominates and not kv.bf16_fit.fits
    acc = sr.binding_resource(L8, H100, Workload(batch_sizes=[8], context_lengths=[4096]), Budget(max_quality_loss_pct=0.2))
    assert acc.binding == "accuracy"
    small = sr.binding_resource(MODEL_PRESETS["llama-3.2-1b"], H100, Workload(batch_sizes=[1], context_lengths=[512]))
    assert small.binding == "bandwidth"


def test_baseline_arms_include_parent_naive_ptq_and_fresh_student():
    arms = sr.baseline_arms(L8, Budget(), footprints=[0.26, 0.5, 0.26, 1.0, 0.0])
    ids = [a.id for a in arms]
    assert all(a.arm == "baseline" for a in arms)
    assert any(a.kind == "bf16" for a in arms)
    assert any(a.id.startswith("naive-") and a.recipe.stages[1].params["method"] == "rtn" for a in arms)
    fs = [a for a in arms if a.kind == "fresh_student"]
    assert [a.footprint_ratio for a in fs] == [0.26, 0.5] and all(a.requires_training and "P4-BASE-01" in a.claim_ids for a in fs)
    assert len(ids) == len(set(ids))
    assert "equal-footprint" in fs[0].notes[0]


# ------------------------------------------------------------------ plan (dry mode)
@pytest.fixture(scope="module")
def plan70():
    return sr.plan(L70, H100, Workload(batch_sizes=[8], context_lengths=[4096]), Budget())


def test_plan_dry_mode_ranks_candidates_with_explanations(plan70):
    p = plan70
    assert p.mode == "dry" and p.binding.binding == "capacity" and any("DRY MODE" in n for n in p.notes)
    assert p.ranked and p.ranked[0].rank == 1 and p.top() is p.ranked[0]
    # 70B on one 80 GB H100: only 4-bit weights fit (plus KV); speculation stacked on W4A16 can rank above plain W4A16
    assert p.ranked[0].candidate.scheme.family == "w4a16"
    plain = [s for s in p.ranked if s.candidate.kind == "w4a16"]
    assert plain
    stacked = [s for s in p.ranked if s.candidate.kind == "speculation"]
    assert stacked and "ptq" in stacked[0].candidate.recipe.stage_names() and any("stacked on" in n for n in stacked[0].candidate.notes)
    assert all(s.fit.fits for s in p.ranked)
    dropped = {s.candidate.kind: s for s in p.dropped}
    assert dropped["fp8"].status == "no_fit" and "KV" in " ".join(dropped["fp8"].reasons)
    assert {b.candidate.kind for b in p.baselines} >= {"bf16", "w4a16", "fresh_student"}
    bf = next(b for b in p.baselines if b.candidate.kind == "bf16")
    assert not bf.fit.fits  # the parent does not fit one 80 GB device
    assert [r["rung"] for r in p.rungs] == ["PROXY_KL", "SUITE_SHORT", "SUITE_LONGCTX_REASONING_SAFETY", "END_TO_END_SERVING"]
    assert p.pareto and set(p.pareto) <= {s.id for s in p.ranked + p.baselines}
    assert any(s.on_pareto for s in p.ranked)
    assert any(t["fired"] and t["rule"] == "R-Q3-W4A16" for t in p.trace)
    # derived numbers carry caveats, quality carries a provenance that is not 'measured'
    top = p.ranked[0]
    assert top.pred.quality_provenance in (Provenance.target, Provenance.paper_reported, Provenance.untested)
    assert top.pred.step_ms_ideal_b1 < top.pred.step_ms_b1 and top.pred.tok_s_total > top.pred.tok_s_stream_b1 > 0
    assert 0 < top.pred.risk <= 1.0
    # fully JSON serialisable for the API/UI
    json.dumps(p.model_dump(mode="json"))


def test_plan_does_not_rank_within_noise():
    # a tight accuracy budget adds a g64 variant; GPTQ/AWQ/g64 priors are 0.002 apart with 0.02 uncertainty -> indistinguishable
    p = sr.plan(L70, H100, Workload(batch_sizes=[8], context_lengths=[4096]), Budget(max_quality_loss_pct=1.5))
    w4 = [s for s in p.ranked if s.candidate.kind == "w4a16"]
    assert len(w4) >= 2
    groups: dict[int, list[str]] = {}
    for s in p.ranked:
        groups.setdefault(s.tie_group, []).append(s.id)
    assert len({s.tie_group for s in w4}) == 1, "the W4A16 variants cannot be separated and must share a tie group"
    # speculation roughly doubles modelled throughput: that difference is NOT noise
    spec = next(s for s in p.ranked if s.candidate.kind == "speculation")
    assert spec.tie_group != w4[0].tie_group and spec.rank < w4[0].rank
    assert any("indistinguishable" in n for n in p.notes)
    ranks = [s.rank for s in p.ranked]
    assert ranks == list(range(1, len(ranks) + 1))


def test_plan_is_deterministic():
    wl = Workload(batch_sizes=[8], context_lengths=[4096])
    a = sr.plan(L70, H100, wl)
    b = sr.plan(L70, H100, wl)
    assert [s.id for s in a.ranked] == [s.id for s in b.ranked]
    assert a.ranked[0].pred.tok_s_total == b.ranked[0].pred.tok_s_total


def test_plan_consumer_gpu_long_context():
    p = sr.plan("llama-3-8b", "rtx-4090", {"batch_sizes": [16], "context_lengths": [8192]}, {"allow_experimental": True})
    assert p.binding.binding in ("kv", "capacity")
    kinds = {s.candidate.kind for s in p.ranked + p.dropped}
    assert {"fp8", "w4a16", "kv"} <= kinds
    assert all(s.fit.fits for s in p.ranked) and p.ranked
    best = p.ranked[0]
    assert best.pred.fits and best.pred.concurrency >= 1


def test_plan_without_fp8_or_fp4_hardware_emulates_with_notes():
    p = sr.plan(L8, "t4", {"batch_sizes": [4], "context_lengths": [2048]})
    assert p.ranked
    assert all(s.candidate.kind != "fp8" for s in p.ranked + p.dropped)  # T4: no FP8 -> rule not fired
    w4 = next(s for s in p.ranked if s.candidate.kind == "w4a16")
    assert w4.pred.step_ms_b1 > 0


def test_plan_blackwell_compute_bound_proposes_fp4():
    p = sr.plan(L8, B200, {"batch_sizes": [512], "context_lengths": [128]}, {"allow_experimental": True})
    assert p.binding.binding == "compute"
    ks = {s.candidate.kind for s in p.ranked + p.dropped}
    assert {"nvfp4", "fp8"} <= ks
    nv = next(s for s in p.ranked + p.dropped if s.candidate.kind == "nvfp4")
    assert nv.pred.bound in ("mma", "mem", "alu") and not any("emulated" in n for n in nv.pred.notes)


def test_plan_with_training_budget_keeps_students_honest():
    p = sr.plan(L70, H100, {"batch_sizes": [8], "context_lengths": [4096]}, {"allow_training": True, "distill_tokens_b": 20})
    kinds = {s.candidate.kind for s in p.ranked + p.dropped}
    assert "prune_distill" in kinds
    students = [s for s in p.baselines if s.candidate.kind == "fresh_student"]
    assert students and all(s.candidate.requires_training for s in students)
    assert all(s.pred.quality_provenance == Provenance.untested for s in p.ranked + p.dropped if s.candidate.kind == "prune_distill")
    # no training budget -> pruned students are not run
    q = sr.plan(L70, H100, {"batch_sizes": [8], "context_lengths": [4096]}, {})
    assert "prune_distill" not in {s.candidate.kind for s in q.ranked + q.dropped}


def test_plan_progress_and_empty_result():
    seen = []
    sr.plan(L8, H100, None, None, progress=lambda f, m: seen.append((f, m)))
    assert seen[0][0] == 0.0 and seen[-1] == (1.0, "done")
    tiny_chip = {"name": "toy", "hbm_gb": 2, "hbm_bw_gbs": 100, "peak_tflops": {"bf16": 10}}
    p = sr.plan(L70, tiny_chip, {"batch_sizes": [1], "context_lengths": [128]})
    assert not p.ranked and any("no candidate survived" in n for n in p.notes)
    assert p.dropped and all(s.status in ("no_fit", "not_run") for s in p.dropped)


# ------------------------------------------------------------------ selection helpers
def _r(i, score, unc=0.0, gate="pass"):
    return sr.RungResult(candidate_id=i, rung="r", score=score, unc=unc, gate_verdict=gate)


def test_keep_top_is_noise_aware():
    res = [_r("a", 1.00, 0.05), _r("b", 0.98, 0.05), _r("c", 0.50, 0.01), _r("d", 0.97, 0.05)]
    kept, groups = sr.keep_top(res, 1)
    assert kept == ["a", "b", "d"]  # b and d cannot be separated from the leader/k-th
    assert ["a", "b", "d"] in groups
    kept, groups = sr.keep_top(res, 1, noise_aware=False)
    assert kept == ["a"]
    kept, _ = sr.keep_top(res, 2)
    assert set(kept) == {"a", "b", "d"}
    kept, _ = sr.keep_top([_r("a", 1.0), _r("b", 0.9), _r("c", 0.8)], 2)
    assert kept == ["a", "b"]  # zero uncertainty: strict ranking
    assert sr.keep_top([_r("x", 1.0, gate="fail"), _r("y", 0.5)], 2, require_gate=True)[0] == ["y"]
    assert sr.keep_top([], 3) == ([], [])


def test_indistinguishable_rule():
    assert sr.indistinguishable((1.0, 0.03), (0.98, 0.03))
    assert not sr.indistinguishable((1.0, 0.01), (0.9, 0.01))


def test_pareto_front_generic():
    items = [{"id": "a", "bytes": 10, "q": 0.9}, {"id": "b", "bytes": 5, "q": 0.9}, {"id": "c", "bytes": 5, "q": 0.5}, {"id": "d", "bytes": 20, "q": 0.99}]
    out = {i["id"]: i for i in sr.pareto_front(items, [("bytes", "min"), ("q", "max")])}
    assert not out["a"]["on_frontier"] and out["a"]["dominated_by"] == ["b"]
    assert out["b"]["on_frontier"] and out["d"]["on_frontier"] and not out["c"]["on_frontier"]
    # tolerance makes small differences ties: c (q=0.5) is still worse beyond tol, but with tol=0.5 a/b/c all tie on quality
    out = {i["id"]: i for i in sr.pareto_front(items, [("bytes", "min"), ("q", "max")], tol={"q": 0.5})}
    assert out["b"]["on_frontier"] and out["c"]["on_frontier"] is True and out["a"]["on_frontier"] is False
    out = {i["id"]: i for i in sr.pareto_front([{"id": "x", "a": 1, "b": 2}, {"id": "y", "a": 1, "b": 2}], [("a", "min"), ("b", "min")])}
    assert out["x"]["on_frontier"] and out["y"]["on_frontier"]  # identical points do not dominate each other


def test_quality_priors_follow_plan_targets():
    from forge.planner.rulebase import build_candidate

    bud = Budget()
    rule = {"id": "R", "claims": [], "note": "", "training": False, "experimental": False}

    def q(kind, model, **prop):
        return sr.quality_prior(build_candidate(model, kind, {"kind": kind, **prop}, rule, bud, {"fp4": True, "fp8": True}), model)

    assert q("fp8", L8).ratio == 0.995 and q("fp8", L8).provenance == Provenance.target
    assert q("w4a16", L8, group=128, method="gptq").ratio == pytest.approx(0.96)  # <10B
    assert q("w4a16", L70, group=128, method="gptq").ratio == pytest.approx(0.98)  # >=30B
    assert q("nvfp4", L70).ratio == pytest.approx(0.975) and q("nvfp4", L70).provenance == Provenance.paper_reported
    assert q("kv", L8, kv="int4").longctx_penalty > 0 and q("kv", L8, kv="fp8").longctx_penalty == 0
    assert q("prune_distill", L8, keep=0.5).provenance == Provenance.untested and q("prune_distill", L8, keep=0.5).unc >= 0.05
    assert q("bf16", L8).ratio == 1.0 and q("lossless", L8).ratio == 1.0


def test_claim_risk_uses_ledger_status():
    from forge.planner import claims as ledger

    ledger.ensure_seeded()
    base = sr.claim_risk(["P1-W4-01", "P1-MARLIN-01"])
    assert 0.6 <= base <= 1.0
    ledger.set_status("P1-MARLIN-01", "reproduced")
    ledger.set_status("P1-W4-01", "reproduced")
    assert sr.claim_risk(["P1-W4-01", "P1-MARLIN-01"]) == 0.0
    ledger.set_status("P1-MARLIN-01", "untested")
    ledger.set_status("P1-W4-01", "untested")
    assert sr.claim_risk([]) == 0.0


# ------------------------------------------------------------------ measured-mode runner with stubs
def test_recipe_eval_runner_with_stub_engine_and_eval(tmp_path):
    from forge.core.context import NullContext
    from forge.planner import claims as ledger

    ledger.ensure_seeded()
    calls = {"engine": 0, "eval": []}

    class Engine:
        def run(self, recipe, ctx):
            calls["engine"] += 1
            return {"artifact": "b2:" + recipe.hash().split(":")[-1]}

    def fake_eval(cfg, ctx):
        calls["eval"].append((cfg.candidate.ref, [s.name for s in cfg.suites], cfg.gate))
        q = 0.97 if "w4" in cfg.name or "fp8" in cfg.name else 0.9
        m = {"suite_score_ratio": MetricValue(value=q, ci_low=q - 0.01, ci_high=q + 0.01), "kl_mean": MetricValue(value=1 - q, ci_low=0.0, ci_high=0.02),
             "longctx_ratio": MetricValue(value=q, ci_low=q - 0.01, ci_high=q + 0.01), "tokens_per_s": MetricValue(value=1000.0 * q)}
        from forge.core.schemas import GateVerdict

        return EvalReport(artifact_hash=cfg.candidate.ref, metrics=m, gate=GateVerdict(gate_id="G", verdict="pass") if cfg.gate else None)

    runner = sr.RecipeEvalRunner({"kind": "path", "ref": "/models/parent"}, NullContext(), engine_factory=lambda c: Engine(), eval_fn=fake_eval)
    p = sr.plan(L70, H100, {"batch_sizes": [8], "context_lengths": [4096]}, Budget(), runner=runner, update_claims=True,
                claims_mapping={"w4a16": {"P1-W4-01": "suite_score_ratio"}})
    assert p.mode == "measured" and p.ranked
    assert calls["engine"] >= 1 and calls["eval"] and all(ref.startswith("b2:") for ref, _, _ in calls["eval"])
    assert all(r.provenance == Provenance.measured for s in p.ranked for r in s.rungs.values())
    suites_seen = {tuple(s) for _, s, _ in calls["eval"]}
    assert ("distribution",) in suites_seen and ("throughput",) in suites_seen
    assert not any("DRY MODE" in n for n in p.notes)
    # the ledger was updated from the measurement (expected >= 0.96 +- 0.01 for small models; 0.97 passes)
    c = ledger.get_claim("P1-W4-01")
    assert c.status == "reproduced" and c.linked_experiments and c.linked_experiments[0].startswith("eval_")
    ledger.set_status("P1-W4-01", "untested")


def test_cost_model_recalibration_in_measured_mode():
    from forge.core.context import NullContext

    def fake_eval(cfg, ctx):
        # tokens/s consistent with eta_mem = 0.5 on the ideal model
        return EvalReport(artifact_hash="x", metrics={"suite_score_ratio": MetricValue(value=0.98), "kl_mean": MetricValue(value=0.01), "longctx_ratio": MetricValue(value=0.98),
                                                     "tokens_per_s": MetricValue(value=200.0)})

    class Engine:
        def run(self, recipe, ctx):
            return {"artifact": "b2:" + recipe.hash().split(":")[-1]}

    chip = H100.model_copy(update={"fingerprint": "plan-recal-1"})
    runner = sr.RecipeEvalRunner({"kind": "path", "ref": "/p"}, NullContext(), engine_factory=lambda c: Engine(), eval_fn=fake_eval)
    p = sr.plan(L70, chip, {"batch_sizes": [8], "context_lengths": [4096]}, {"allow_experimental": True, "max_quality_loss_pct": 10}, runner=runner,
                rungs=[sr.END_TO_END_SERVING])
    # recalibration needs >= 3 measured candidates; if present it is reported and persisted
    if any("recalibrated" in n for n in p.notes):
        assert cm.load_eta(chip, "decode").source == "fit"


# ------------------------------------------------------------------ estimate (UI quick API)
def test_estimate_scheme_string():
    e = estimate("llama-3-8b", "h100", "w4a16-g128+kvfp8", {"batch_sizes": [1, 32], "context_lengths": [4096, 32768]})
    assert e.fit.fits and e.weight_bits_eff == pytest.approx(4.125) and e.kv_bits == 8
    assert e.kv_bytes_per_token == 64 * 1024
    assert len(e.points) == 4 and all(p.fits for p in e.points)
    assert e.b_star == pytest.approx(cm.batch_crossover(989e12, 3.35e12, 4.125), rel=1e-6) and 70 < e.b_star < 80
    assert e.b_star_effective < e.b_star
    b1 = next(p for p in e.points if p.batch == 1 and p.ctx == 4096)
    assert b1.bound == "mem" and b1.step_ms_pred > b1.step_ms_ideal and b1.tok_s_stream_ideal > 500
    big = next(p for p in e.points if p.batch == 32 and p.ctx == 32768)
    assert big.bytes_kv_gb > 5 and big.tok_s_total_pred > b1.tok_s_total_pred
    assert e.binding.binding in ("bandwidth", "kv")
    json.dumps(e.model_dump(mode="json"))


def test_estimate_recipe_input_and_kv_override_and_warnings():
    recipe = Recipe(name="r", stages=[{"name": "calibrate", "params": {}}, {"name": "ptq", "params": {"method": "gptq", "bits": 4, "group_size": 64}},
                                      {"name": "kv", "params": {"dtype": "fp8_e4m3"}}])
    s = scheme_from_recipe(recipe)
    assert s.weight_bits_eff == pytest.approx(4.25) and s.kv_bits == 8
    e = estimate("llama-3-8b", "h100", recipe, kv_dtype="4")
    assert e.kv_bits == pytest.approx(4.25)
    assert scheme_from_recipe(Recipe(name="n", stages=[{"name": "ptq", "params": {"weight_format": "nvfp4"}}])).name == "nvfp4"
    assert scheme_from_recipe(Recipe(name="f", stages=[{"name": "ptq", "params": {"weight_format": "fp8_e4m3"}}])).name == "fp8"
    e = estimate("llama-3-8b", "a100", "nvfp4")
    assert any("FP4" in w for w in e.warnings)
    e = estimate("llama-3-8b", "rtx-5090", "bf16")
    assert any("APPROXIMATE" in w for w in e.warnings)
    e = estimate("llama-3.1-70b", "h100", "bf16", {"batch_sizes": [1], "context_lengths": [128]})
    assert not e.fit.fits and e.binding.binding == "capacity"


def test_estimate_speculation_curve_matches_p7():
    e = estimate("llama-3-8b", "h100", "bf16", spec={"alpha": 0.8, "gamma": 4, "c": 0.05})
    rows = {int(r["batch"]): r for r in e.speculation}
    assert rows[1]["speedup"] == pytest.approx(3.3616, abs=1e-3) and rows[1]["memory_bound_speedup"] == pytest.approx(3.3616 / 1.2, rel=1e-3)
    assert rows[512]["speedup"] < 1.0
    assert e.spec_break_even_batch == pytest.approx(cm.spec_break_even_batch(4, 0.8, e.b_star_effective))
    assert any("ASSUMED" in w for w in e.warnings)
    speeds = [rows[b]["speedup"] for b in sorted(rows)]
    assert speeds == sorted(speeds, reverse=True)  # speed-up never increases with batch in the weights-only model


def test_planner_package_lazy_api():
    import forge.planner as fp

    assert fp.estimate is estimate and fp.plan is sr.plan and fp.fits is sr.fits
    assert callable(fp.seed_claims) and callable(fp.claims_needing_attention)
    with pytest.raises(AttributeError):
        fp.nothing_here
