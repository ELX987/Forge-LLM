from __future__ import annotations

import numpy as np
import pytest

from forge.core.errors import ConfigError, NotFound
from forge.core.schemas import MetricValue
from forge.eval import gates
from forge.eval.gates import Gate, GateSpec, Measurements, eval_condition, load_gate

PRESETS = ["g-lossless", "g-fp8", "g-w4", "g-struct", "g-long", "g-spec", "g-fp4train"]


def mv(value, lo=None, hi=None, **kw):
    lo = value if lo is None else lo
    hi = value if hi is None else hi
    return MetricValue(value=value, ci_low=lo, ci_high=hi, n=100, **kw)


def w4_measurements(**over):
    m = {
        "kl_p99": mv(0.02, 0.015, 0.03), "top1_flip": mv(0.02, 0.015, 0.025), "suite_score_ratio": mv(0.99, 0.985, 0.995),
        "suite_score_diff_pts": mv(-0.4, -1.0, 0.2), "longctx_ratio": mv(0.98, 0.96, 1.0), "longgen_reasoning": mv(0.99, 0.97, 1.0),
        "safety_delta": mv(0.0, -1.0, 1.0, within_noise=True),
    }
    m.update(over)
    rng = np.random.default_rng(0)
    boot = {"suite_score_diff_pts": rng.normal(-0.4, 0.3, 2000)}
    return Measurements(metrics=m, boot=boot, context={"params_b": 8, "has_parent": True, "deployment_length": 4096})


def test_all_presets_load_and_are_labelled_target():
    found = {g["stem"] for g in gates.list_gates()}
    assert set(PRESETS) <= found
    for name in PRESETS:
        g = load_gate(name)
        assert isinstance(g, Gate) and g.spec.status == "TARGET" and g.spec.title and g.spec.source
        assert g.spec.noise_floor.n >= 3


def test_w4_matches_appendix_c3():
    g = load_gate("G-W4-v2")
    s = g.spec
    assert s.noninferiority.margin_pts == 2.0 and s.noninferiority.alpha == 0.05 and s.noninferiority.fdr == "benjamini_hochberg"
    assert s.metrics["kl_p99"].max == 0.05 and s.metrics["top1_flip"].max == 0.04
    r = s.metrics["suite_score_ratio"]
    assert (r.min, r.applies_if, r.else_min) == (0.98, "params_b >= 30", 0.96)
    assert s.metrics["longctx_ratio"].min == 0.95 and s.metrics["longctx_ratio"].at_length == "deployment"
    assert s.metrics["longgen_reasoning"].min == 0.95 and s.metrics["safety_delta"].within_noise is True
    assert s.noise_floor.n == 5 and s.parent == "required"


def test_other_presets_match_plan_table():
    fp8 = load_gate("g-fp8").spec
    assert fp8.metrics["kl_p99"].max == 0.01 and fp8.metrics["top1_flip"].max == 0.01
    assert fp8.noninferiority.margin_pts == 0.5 and fp8.metrics["longctx_ratio"].min == 0.99
    assert {"size_target_met", "beats_fresh_student_arm", "licence_gate_passed"} == set(load_gate("g-struct").spec.flags)
    assert {"exactness_test_passed", "production_batch_measured"} == set(load_gate("g-spec").spec.flags)
    assert {"tr2_diagnostics_green", "healing_decision_documented"} <= set(load_gate("g-fp4train").spec.flags)
    long_ = load_gate("g-long").spec
    assert "longctx_multihop_diff_pts" in long_.metrics and "multiturn_reuse_noninferior" in long_.flags


def test_lookup_variants_and_errors():
    assert load_gate("g-w4").id == load_gate("G-W4").id == load_gate("G-W4-v2").id == "G-W4-v2"
    with pytest.raises(NotFound):
        load_gate("g-nope")
    with pytest.raises(ConfigError):
        gates.gate_from_dict({"gate": "x", "bogus": 1})


def test_user_dir_overrides_repo(tmp_path, monkeypatch):
    from forge.core.paths import forge_home

    d = forge_home() / "configs" / "gates"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "g-w4.yaml"
    f.write_text("gate: G-W4-custom\nstatus: CALIBRATED\nmetrics: {kl_p99: {max: 0.5}}\n")
    try:
        g = load_gate("g-w4")
        assert g.id == "G-W4-custom" and g.spec.status == "CALIBRATED"
        assert any(x["id"] == "G-W4-custom" for x in gates.list_gates())
    finally:
        f.unlink()
    assert load_gate("g-w4").id == "G-W4-v2"
    p = tmp_path / "mine.yaml"
    p.write_text("gate: G-MINE\nmetrics: {x: {min: 1}}\n")
    assert load_gate(p).id == "G-MINE"


def test_register_gates_populates_registry():
    from forge.core import registry

    names = gates.register_gates()
    assert "G-W4-v2" in names and "G-W4-v2" in registry.names("gate")


def test_eval_condition_is_safe():
    assert eval_condition("params_b >= 30", {"params_b": 70}) is True
    assert eval_condition("params_b >= 30 and not tiny", {"params_b": 70, "tiny": False}) is True
    assert eval_condition("10 < params_b <= 30", {"params_b": 8}) is False
    assert eval_condition("storage_transform", {"storage_transform": True}) is True
    with pytest.raises(gates._Unknown):
        eval_condition("params_b >= 30", {})
    for evil in ("__import__('os').system('echo hi')", "params_b.__class__", "[1][0]", "lambda: 1", "params_b + 1 > 2"):
        with pytest.raises(Exception):
            eval_condition(evil, {"params_b": 1})


def test_w4_pass():
    v = load_gate("g-w4").evaluate(w4_measurements())
    assert v.verdict == "pass", v.reasons
    assert v.gate_id == "G-W4-v2" and v.details["_gate_status"] == "TARGET"
    assert v.details["suite_score_diff_pts"]["noninferiority"]["adjusted_p"] < 0.05


def test_w4_fail_when_confidently_over_threshold():
    v = load_gate("g-w4").evaluate(w4_measurements(kl_p99=mv(0.09, 0.07, 0.12)))
    assert v.verdict == "fail" and any("kl_p99" in r and "above max" in r for r in v.reasons)


def test_w4_inconclusive_when_ci_straddles():
    v = load_gate("g-w4").evaluate(w4_measurements(kl_p99=mv(0.04, 0.02, 0.08)))
    assert v.verdict == "inconclusive" and any("cannot be separated" in r for r in v.reasons)
    # point policy decides on the estimate
    spec = load_gate("g-w4").spec.model_copy(update={"ci_policy": "point"})
    assert Gate(spec).evaluate(w4_measurements(kl_p99=mv(0.04, 0.02, 0.08))).verdict == "pass"


def test_size_dependent_threshold():
    big = w4_measurements(suite_score_ratio=mv(0.97, 0.965, 0.975))
    small = w4_measurements(suite_score_ratio=mv(0.97, 0.965, 0.975))
    big.context["params_b"] = 70
    small.context["params_b"] = 7
    g = load_gate("g-w4")
    assert g.evaluate(big).verdict == "fail"
    assert g.evaluate(small).verdict == "pass"
    unknown = w4_measurements(suite_score_ratio=mv(0.97, 0.965, 0.975))
    unknown.context.pop("params_b")
    v = g.evaluate(unknown)
    assert v.verdict == "fail" and "stricter threshold" in v.details["suite_score_ratio"]["note"]


def test_missing_metric_and_no_parent_are_inconclusive():
    m = w4_measurements()
    del m.metrics["longgen_reasoning"]
    v = load_gate("g-w4").evaluate(m)
    assert v.verdict == "inconclusive" and any("longgen_reasoning" in r and "not measured" in r for r in v.reasons)
    m2 = w4_measurements()
    m2.context["has_parent"] = False
    v = load_gate("g-w4").evaluate(m2)
    assert v.verdict == "inconclusive" and "requires a parent" in v.reasons[0]


def test_longctx_at_deployment_length_flag():
    m = w4_measurements()
    m.flags["longctx_length_matches_deployment"] = False
    v = load_gate("g-w4").evaluate(m)
    assert v.verdict == "inconclusive" and any("deployment context length" in r for r in v.reasons)


def test_within_noise_rule():
    g = load_gate("g-w4")
    v = g.evaluate(w4_measurements(safety_delta=mv(3.0, 1.0, 5.0, within_noise=False)))
    assert v.verdict == "fail" and any("distinguishable from noise" in r for r in v.reasons)
    m = w4_measurements(safety_delta=mv(0.5, -1.0, 2.0))  # within_noise unset -> decided from the noise floor
    m.noise_floor["safety_delta"] = 1.0
    assert g.evaluate(m).verdict == "pass"
    m = w4_measurements(safety_delta=mv(0.5, 0.5, 0.5))
    assert g.evaluate(m).verdict == "fail"  # CI excludes 0 -> distinguishable from noise
    m = w4_measurements(safety_delta=MetricValue(value=0.5))
    assert g.evaluate(m).verdict == "inconclusive"  # nothing to decide "within noise" with
    m = w4_measurements(safety_delta=mv(0.5, -1.0, 2.0))
    assert g.evaluate(m).verdict == "pass"  # CI contains 0 -> within noise (noted)


def test_noninferiority_with_bh_and_inferior_case():
    rng = np.random.default_rng(1)
    g = load_gate("g-long")
    ms = {
        "longctx_multihop_diff_pts": mv(-0.5, -2, 1), "longctx_aggregation_diff_pts": mv(-6.0, -9, -3),
        "longctx_ratio": mv(0.99, 0.97, 1.0),
    }
    boot = {"longctx_multihop_diff_pts": rng.normal(-0.5, 0.5, 4000), "longctx_aggregation_diff_pts": rng.normal(-6.0, 1.0, 4000)}
    m = Measurements(ms, boot, flags={"multiturn_reuse_noninferior": True, "multi_benchmark_evidence": True}, context={"has_parent": True, "deployment_length": None})
    v = g.evaluate(m)
    assert v.verdict == "fail"  # aggregation is inferior by more than the 3-pt margin
    assert v.details["longctx_multihop_diff_pts"]["status"] == "pass"
    assert v.details["longctx_aggregation_diff_pts"]["status"] == "fail"
    # BH adjusted p >= raw p
    nid = v.details["longctx_multihop_diff_pts"]["noninferiority"]
    assert nid["adjusted_p"] >= nid["p_value"]


def test_noninferiority_inconclusive_when_noisy():
    rng = np.random.default_rng(2)
    g = load_gate("g-fp8")
    ms = {"kl_p99": mv(0.005), "top1_flip": mv(0.005), "suite_score_diff_pts": mv(-0.2, -2.0, 1.6), "longctx_ratio": mv(1.0, 0.995, 1.0)}
    m = Measurements(ms, {"suite_score_diff_pts": rng.normal(-0.2, 1.0, 4000)}, context={"has_parent": True})
    v = g.evaluate(m)
    assert v.verdict == "inconclusive" and any("non-inferiority" in r for r in v.reasons)


def test_ni_without_bootstrap_falls_back_to_ci_or_point():
    g = load_gate("g-fp8")
    base = {"kl_p99": mv(0.005), "top1_flip": mv(0.005), "longctx_ratio": mv(1.0)}
    ok = Measurements({**base, "suite_score_diff_pts": mv(0.0, -0.1, 0.1)}, context={"has_parent": True})
    assert g.evaluate(ok).verdict == "pass"
    pt = Measurements({**base, "suite_score_diff_pts": MetricValue(value=0.0)}, context={"has_parent": True})
    assert g.evaluate(pt).verdict == "inconclusive"  # a bare point estimate can never establish non-inferiority


def test_flags():
    g = load_gate("g-struct")
    ms = {"suite_score_ratio": mv(0.95, 0.93, 0.97)}
    flags = {"size_target_met": True, "beats_fresh_student_arm": True, "licence_gate_passed": True}
    assert g.evaluate(Measurements(ms, flags=dict(flags), context={"has_parent": True})).verdict == "pass"
    bad = dict(flags, licence_gate_passed=False)
    v = g.evaluate(Measurements(ms, flags=bad, context={"has_parent": True}))
    assert v.verdict == "fail" and any("licence_gate_passed" in r for r in v.reasons)
    miss = {k: v for k, v in flags.items() if k != "size_target_met"}
    v = g.evaluate(Measurements(ms, flags=miss, context={"has_parent": True}))
    assert v.verdict == "inconclusive"


def test_lossless_strict_vs_serving():
    g = load_gate("g-lossless")
    ident = {"max_abs_logit_diff": mv(0.0), "kl_p99": mv(0.0), "top1_flip": mv(0.0)}
    assert g.evaluate(Measurements(dict(ident), context={"has_parent": True, "storage_transform": True})).verdict == "pass"
    noisy = {"max_abs_logit_diff": mv(1e-3), "kl_p99": mv(1e-6), "top1_flip": mv(0.0)}
    assert g.evaluate(Measurements(dict(noisy), context={"has_parent": True, "storage_transform": True})).verdict == "fail"
    assert g.evaluate(Measurements(dict(noisy), context={"has_parent": True, "storage_transform": False})).verdict == "pass"
    # unknown context -> stricter
    assert g.evaluate(Measurements(dict(noisy), context={"has_parent": True})).verdict == "fail"


def test_gate_with_no_rules_is_inconclusive_and_optional_parent():
    g = Gate(GateSpec(gate="G-EMPTY"))
    assert g.evaluate(Measurements({}, context={"has_parent": True})).verdict == "inconclusive"
    spec = GateSpec(gate="G-OPT", parent="optional", metrics={"x": {"min": 1.0}})
    assert Gate(spec).evaluate(Measurements({"x": mv(2.0)}, context={"has_parent": False})).verdict == "pass"


def test_noise_floor_string_parsing():
    s = GateSpec(gate="G", noise_floor="from_parent_repeats(n=7)")
    assert s.noise_floor.n == 7 and s.noise_floor.source == "from_parent_repeats"
    assert GateSpec(gate="G", noise_floor="from_parent_repeats").noise_floor.n == 5
    with pytest.raises(Exception):
        GateSpec(gate="G", noise_floor="???")


def test_gate_presets_module_is_in_sync_with_yaml():
    from forge.eval.gate_presets import PRESETS

    import yaml
    from pathlib import Path

    d = Path(gates.__file__).resolve().parents[2] / "configs" / "gates"
    for name in PRESETS:
        data = yaml.safe_load((d / f"{name}.yaml").read_text(encoding="utf-8"))
        a, b = GateSpec(**data).model_dump(), GateSpec(**PRESETS[name]).model_dump()
        a["description"], b["description"] = " ".join(a["description"].split()), " ".join(b["description"].split())
        assert a == b, name
    assert set(PRESETS) == {f.stem for f in d.glob("*.yaml")}
