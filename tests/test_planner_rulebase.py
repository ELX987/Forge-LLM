from __future__ import annotations

import pytest

from forge.core.schemas import Recipe, Workload
from forge.planner import claims as ledger
from forge.planner import rulebase as rb
from forge.planner.claims_seed import SEED_CLAIMS
from forge.planner.search import binding_resource
from forge.planner.spec import CHIP_PRESETS, MODEL_PRESETS

H100, A100, B200, RTX4090 = (CHIP_PRESETS[k] for k in ("h100-sxm-80g", "a100-80g", "b200", "rtx-4090"))
L8, L70 = MODEL_PRESETS["llama-3-8b"], MODEL_PRESETS["llama-3.1-70b"]


def kinds(cands):
    return [c.kind for c in cands]


# ------------------------------------------------------------------ the rulebase is data and is consistent with the ledger
def test_flow_and_table_cover_the_plan():
    assert [n["id"] for n in rb.FLOW] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert len([r for r in rb.RULES if r["table"].startswith("Table 5.1")]) == 11
    assert len([r for r in rb.RULES if r["table"] == "Fig 5.3"]) == 6
    assert len({r["id"] for r in rb.RULES}) == len(rb.RULES)
    for r in rb.RULES:
        assert r["claims"], f"{r['id']} points at no ledger entry (G1: every default points at a ledger entry)"
        assert r["proposals"] and r["title"] and r["paper_refs"]


def test_every_rule_claim_exists_in_the_seed_ledger_and_vice_versa():
    seed_ids = {c["id"] for c in SEED_CLAIMS}
    assert rb.all_claim_ids() <= seed_ids, rb.all_claim_ids() - seed_ids
    unreferenced = seed_ids - rb.all_claim_ids()
    # claims no rule uses are allowed, but the vast majority must matter to some rule
    assert len(unreferenced) <= 0.2 * len(seed_ids), sorted(unreferenced)
    assert rb.rule_ids_for_claim("P1-W4-01") == ["R-Q3-W4A16"]
    assert "R-Q4-FP4" in rb.rule_ids_for_claim("P5-OVH-01")


def test_describe_rulebase_is_json_serialisable():
    import json

    d = rb.describe_rulebase()
    json.dumps(d)
    assert d["flow"][0]["id"] == "Q1" and "validate the deployed artifact" in d["footer"]


# ------------------------------------------------------------------ engine behaviour
def test_h100_70b_capacity_proposes_fp8_and_w4a16():
    wl = Workload(batch_sizes=[8], context_lengths=[4096])
    bind = binding_resource(L70, H100, wl)
    assert bind.binding == "capacity"
    cands, trace = rb.propose_with_trace(L70, bind.binding, H100, rb.Budget(), wl, bf16_fits=bind.bf16_fit.fits, kv_dominates=bind.kv_dominates)
    ks = kinds(cands)
    assert "fp8" in ks and "w4a16" in ks and "bf16" not in ks
    assert all(isinstance(c.recipe, Recipe) for c in cands)
    fired = {t["rule"] for t in trace if t["fired"]}
    assert {"R-Q2-FP8", "R-Q3-W4A16"} <= fired
    skipped = {t["rule"]: t["reason"] for t in trace if not t["fired"]}
    assert "needs training" in skipped["T-PRUNE-DISTILL"]
    assert "experimental" in skipped["T-LONG-TRAIN"] or "needs training" in skipped["T-LONG-TRAIN"]
    # recipes are executable-looking: calibrate -> ptq -> pack -> eval -> gate
    w4 = next(c for c in cands if c.kind == "w4a16")
    assert w4.recipe.stage_names() == ["calibrate", "ptq", "pack", "eval", "gate"]
    ptq = w4.recipe.stages[1].params
    assert ptq["bits"] == 4 and ptq["group_size"] == 128 and ptq["method"] in ("gptq", "awq")
    assert w4.gate == "g-w4" and w4.scheme.weight_bits_eff == pytest.approx(4.125)
    assert "P1-MARLIN-01" in w4.claim_ids and w4.rule_ids == ["R-Q3-W4A16"]
    assert w4.recipe.hash() == Recipe(**w4.recipe.model_dump()).hash()


def test_no_fp8_hardware_falls_through_to_w4a16_only():
    wl = Workload(batch_sizes=[4], context_lengths=[2048])
    cands = rb.propose(L70, "capacity", A100, rb.Budget(), wl, bf16_fits=False)
    ks = kinds(cands)
    assert "fp8" not in ks and "w4a16" in ks
    skipped = [t for t in rb.propose_with_trace(L70, "capacity", A100, rb.Budget(), wl, bf16_fits=False)[1] if t["rule"] == "R-Q2-FP8"][0]
    assert "FP8" in skipped["reason"].upper() or "fp8" in skipped["reason"]


def test_compute_bound_blackwell_gets_fp4_and_hopper_gets_w4a8_only():
    wl = Workload(batch_sizes=[512], context_lengths=[1024])
    b = kinds(rb.propose(L8, "compute", B200, rb.Budget(), wl))
    assert "nvfp4" in b and "mxfp4" in b and "bf16" in b and "fp8" in b
    h = kinds(rb.propose(L8, "compute", H100, rb.Budget(), wl))
    assert "nvfp4" not in h and "w4a8" in h


def test_kv_binding_adds_kv_candidates_with_int4_gated():
    wl = Workload(batch_sizes=[64], context_lengths=[32768])
    cands = rb.propose(L8, "kv", H100, rb.Budget(), wl)
    kvs = [c for c in cands if c.kind == "kv"]
    assert [c.scheme.kv_bits for c in kvs] == [8.0]  # 4-bit KV is experimental -> only with allow_experimental
    cands = rb.propose(L8, "kv", H100, rb.Budget(allow_experimental=True), wl)
    kvs = [c for c in cands if c.kind == "kv"]
    assert sorted(c.scheme.kv_bits for c in kvs) == [4.25, 8.0] and any(c.experimental for c in kvs)
    int4 = next(c for c in kvs if c.scheme.kv_bits == 4.25)
    assert any("full deployment length" in n for n in int4.notes)
    assert "serving_lossless" in kinds(cands)  # paging + prefix caching + chunked prefill (T-LONG-NOTRAIN)


def test_training_budget_unlocks_prune_distill_with_licence_note():
    wl = Workload(batch_sizes=[8], context_lengths=[4096])
    no = kinds(rb.propose(L70, "capacity", H100, rb.Budget(), wl))
    assert "prune_distill" not in no
    cands = rb.propose(L70, "capacity", H100, rb.Budget(allow_training=True, distill_tokens_b=30), wl)
    pd = [c for c in cands if c.kind == "prune_distill"]
    assert [c.footprint_ratio for c in pd] == [0.75, 0.5]
    c = pd[0]
    assert c.requires_training and c.recipe.stage_names()[:3] == ["calibrate", "prune", "distill"]
    assert c.recipe.stages[2].params["tokens_b"] == 30 and "fresh-student" in c.recipe.baselines
    assert any("licence" in n.lower() for n in c.notes)
    blocked = MODEL_PRESETS["llama-3.1-70b"].model_copy(update={"distill_allowed": False})
    c2 = [x for x in rb.propose(blocked, "capacity", H100, rb.Budget(allow_training=True), wl) if x.kind == "prune_distill"][0]
    assert any("does not permit distillation" in n for n in c2.notes)


def test_moe_rules_only_for_moe_and_speculation_depends_on_concurrency():
    moe = MODEL_PRESETS["qwen3-30b-a3b"]
    wl = Workload(batch_sizes=[4], context_lengths=[4096])
    assert "expert_prune" in kinds(rb.propose(moe, "capacity", RTX4090, rb.Budget(), wl))
    assert "expert_prune" not in kinds(rb.propose(L8, "capacity", RTX4090, rb.Budget(), wl))
    low = rb.propose(L8, "bandwidth", H100, rb.Budget(context_copying=True), Workload(batch_sizes=[1, 4]))
    sp = [c for c in low if c.kind == "speculation"]
    assert [c.serving["speculation"]["mode"] for c in sp] == ["ngram", "auto"]
    high = rb.propose(L8, "bandwidth", H100, rb.Budget(), Workload(batch_sizes=[64]))
    sph = [c for c in high if c.kind == "speculation"]
    assert len(sph) == 1 and sph[0].serving["speculation"]["allow_zero"] is True and "P7-BATCH-01" in sph[0].claim_ids


def test_sub35_and_training_precision_and_budget_flags():
    wl = Workload(batch_sizes=[1])
    c = rb.propose(L70, "capacity", H100, rb.Budget(max_bits_per_weight=3.0), wl)
    assert "sub35" not in kinds(c)  # Tier 3 / gated
    c = rb.propose(L70, "capacity", H100, rb.Budget(max_bits_per_weight=3.0, allow_experimental=True, allow_training=True), wl)
    assert {"sub35", "sub35_qat"} <= set(kinds(c)) and all(x.experimental for x in c if x.kind.startswith("sub35"))
    t = rb.propose(L8, "compute", B200, rb.Budget(purpose="training", allow_training=True, allow_experimental=True), wl)
    assert {"train_fp8", "train_nvfp4"} <= set(kinds(t))
    nv = next(x for x in t if x.kind == "train_nvfp4")
    assert nv.experimental and nv.gate == "g-fp4train" and nv.recipe.stages[0].params["bf16_exempt_last_layers"] == 0.15
    t2 = rb.propose(L8, "compute", H100, rb.Budget(purpose="training", allow_training=True), wl)
    assert "train_nvfp4" not in kinds(t2) and "train_fp8" in kinds(t2)
    ls = kinds(rb.propose(L8, "bandwidth", H100, rb.Budget(optimise_storage=True), wl))
    assert "lossless" in ls
    d = kinds(rb.propose(L8, "bandwidth", H100, rb.Budget(many_finetunes=True, allow_experimental=True), wl))
    assert d.count("delta") == 2


def test_tight_accuracy_adds_g64_and_candidates_are_deduplicated():
    wl = Workload(batch_sizes=[1])
    loose = [c for c in rb.propose(L70, "capacity", H100, rb.Budget(max_quality_loss_pct=4), wl) if c.kind == "w4a16"]
    tight = [c for c in rb.propose(L70, "capacity", H100, rb.Budget(max_quality_loss_pct=1.5), wl) if c.kind == "w4a16"]
    assert len(tight) == len(loose) + 1 and any(c.scheme.weight_group == 64 for c in tight)
    all_ids = [c.id for c in rb.propose(L70, "capacity", H100, rb.Budget(allow_training=True, allow_experimental=True), wl)]
    assert len(all_ids) == len(set(all_ids))
    capped = rb.propose(L70, "capacity", H100, rb.Budget(allow_training=True, allow_experimental=True, max_candidates=3), wl)
    assert len(capped) == 3


def test_candidate_ids_are_stable_functions_of_content():
    wl = Workload(batch_sizes=[8])
    a = [c.id for c in rb.propose(L70, "capacity", H100, rb.Budget(), wl)]
    b = [c.id for c in rb.propose(L70, "capacity", H100, rb.Budget(), wl)]
    assert a == b
    c = [x.id for x in rb.propose(L70, "capacity", H100, rb.Budget(distill_tokens_b=5), wl)]
    assert a == c  # distill tokens only affect training recipes


def test_conditions_are_safe_and_missing_names_are_false():
    assert rb._truthy("a and b", {"a": True, "b": 1}) is True
    assert rb._truthy("missing_flag", {}) is False
    assert rb._truthy("x >= 3", {"x": None}) is False
    assert rb._truthy("binding in ('kv', 'capacity')", {"binding": "kv"}) is True
    assert rb._truthy("__import__('os')", {}) is False
    assert rb._truthy(None, {}) is True


# ------------------------------------------------------------------ claims ledger
def test_seed_claims_are_real_and_complete():
    assert len(SEED_CLAIMS) >= 30
    ids = [c["id"] for c in SEED_CLAIMS]
    assert len(ids) == len(set(ids))
    papers = {c["source_paper"] for c in SEED_CLAIMS}
    assert papers == {"P1", "P2", "P3", "P4", "P5", "P6", "P7"}
    for c in SEED_CLAIMS:
        assert c["statement"].endswith(".") and len(c["statement"]) > 40 and c["location"]
        assert c["prior"] in ("trusted", "untrusted") and c["id"].startswith(c["source_paper"] + "-")
    assert any(c["prior"] == "trusted" for c in SEED_CLAIMS) and sum(c["prior"] == "untrusted" for c in SEED_CLAIMS) >= 15
    byid = {c["id"]: c for c in SEED_CLAIMS}
    # vendor / single-source claims start untrusted (Plan 5.5)
    for cid in ("P7-EAGLE-01", "P5-OVH-01", "P1-FP4-02", "P3-SPARSE-01", "P1-TRAIN-01", "P7-SSD-01"):
        assert byid[cid]["prior"] == "untrusted", cid
    # the plan's own ledger example
    assert "Marlin" in byid["P1-MARLIN-01"]["statement"] and "16" in byid["P1-MARLIN-01"]["statement"]
    for c in SEED_CLAIMS:
        e = c["expectation"]
        if e:
            assert e["direction"] in ("approx", "at_least", "at_most") and {"metric", "expected", "tolerance"} <= set(e)


def test_seed_and_crud(tmp_path):
    from forge.core.db import Database

    db = Database(tmp_path / "claims.db")
    db.migrate()
    added = ledger.seed_claims(db=db)
    assert len(added) == len(SEED_CLAIMS)
    assert ledger.seed_claims(db=db) == []  # idempotent
    allc = ledger.list_claims(db=db)
    assert len(allc) == len(SEED_CLAIMS) and all(c.status == "untested" for c in allc)
    c = ledger.get_claim("P1-W4-01", db=db)
    assert c.depends_rules == ["R-Q3-W4A16"] and c.prior == "trusted"
    assert ledger.list_claims(paper="P7", db=db) and all(x.source_paper == "P7" for x in ledger.list_claims(paper="P7", db=db))
    assert {x.id for x in ledger.list_claims(rule="R-Q4-FP4", db=db)} >= {"P5-OVH-01", "P1-FP4-01"}
    assert ledger.list_claims(q="Marlin", db=db)[0].id == "P1-MARLIN-01"
    u = ledger.update_claim("P1-W4-01", status="partial", notes="reproduced at B=1 only", db=db)
    assert u.status == "partial" and u.notes == "reproduced at B=1 only"
    ledger.link_experiment("P1-W4-01", "eval_abc", db=db)
    ledger.link_experiment("P1-W4-01", "eval_abc", db=db)
    assert ledger.get_claim("P1-W4-01", db=db).linked_experiments == ["eval_abc"]
    # re-seeding keeps user status/notes and does not duplicate
    ledger.seed_claims(db=db)
    assert ledger.get_claim("P1-W4-01", db=db).status == "partial"
    ledger.seed_claims(overwrite=True, db=db)
    assert ledger.get_claim("P1-W4-01", db=db).status == "partial" and ledger.get_claim("P1-W4-01", db=db).notes == "reproduced at B=1 only"
    with pytest.raises(Exception):
        ledger.update_claim("P1-W4-01", status="bogus", db=db)
    with pytest.raises(Exception):
        ledger.update_claim("P1-W4-01", depends="x", db=db)
    with pytest.raises(Exception):
        ledger.get_claim("nope", db=db)
    new = ledger.upsert_claim({"id": "X-1", "source_paper": "P9", "statement": "custom claim", "prior": "trusted"}, db=db)
    assert new.status == "untested" and ledger.delete_claim("X-1", db=db) and not ledger.delete_claim("X-1", db=db)
    s = ledger.ledger_summary(db=db)
    assert s["total"] == len(SEED_CLAIMS) and s["by_status"]["partial"] == 1


def test_claims_needing_attention_ranking(tmp_path):
    from forge.core.db import Database

    db = Database(tmp_path / "att.db")
    db.migrate()
    ledger.seed_claims(db=db)
    top = ledger.claims_needing_attention(5, db=db)
    assert len(top) == 5 and top == sorted(top, key=lambda r: (-r["score"], r["id"]))
    assert all(t["n_rules"] >= 1 and t["status"] != "reproduced" for t in top)
    # an untrusted claim with several dependents beats a trusted one with the same fan-out
    by = {t["id"]: t for t in ledger.claims_needing_attention(100, db=db)}
    assert by["P1-FP4-01"]["score"] == pytest.approx(len(by["P1-FP4-01"]["depends_rules"]) * 1.5)
    assert by["P1-W4-01"]["score"] == pytest.approx(1.0)
    # reproduced claims drop out, contradicted ones jump the queue
    ledger.set_status("P1-FP4-01", "reproduced", db=db)
    ledger.set_status("P1-W4-01", "contradicted", db=db)
    after = {t["id"]: t for t in ledger.claims_needing_attention(100, db=db)}
    assert "P1-FP4-01" not in after
    assert after["P1-W4-01"]["score"] == pytest.approx(2.5) and "must be revisited" in after["P1-W4-01"]["reason"]
    assert ledger.claims_needing_attention(1, db=db)[0]["score"] >= 2.5
    unref = {t["id"] for t in ledger.claims_needing_attention(100, include_unreferenced=True, db=db)} - set(after)
    assert unref  # claims no rule depends on only show with include_unreferenced


def test_evaluate_claim_statuses(tmp_path):
    from forge.core.db import Database

    db = Database(tmp_path / "ev.db")
    db.migrate()
    ledger.seed_claims(db=db)
    r = ledger.evaluate_claim("P1-CROSS-01", 73.8, experiment="exp1", db=db)  # expected 74 +- 1
    assert r["status"] == "reproduced" and ledger.get_claim("P1-CROSS-01", db=db).linked_experiments == ["exp1"]
    assert ledger.evaluate_claim("P1-CROSS-01", 75.5, db=db)["status"] == "partial"
    assert ledger.evaluate_claim("P1-CROSS-01", 120, db=db)["status"] == "contradicted"
    assert ledger.get_claim("P1-CROSS-01", db=db).status == "contradicted" and "contradicted" in ledger.get_claim("P1-CROSS-01", db=db).notes
    # at_least: Marlin claim expects >= 3.4 +- 0.5
    assert ledger.evaluate_claim("P1-MARLIN-01", 3.1, apply=False, db=db)["status"] == "reproduced"
    assert ledger.evaluate_claim("P1-MARLIN-01", 2.2, apply=False, db=db)["status"] == "partial"
    assert ledger.evaluate_claim("P1-MARLIN-01", 1.2, apply=False, db=db)["status"] == "contradicted"
    assert ledger.get_claim("P1-MARLIN-01", db=db).status == "untested"  # apply=False leaves the ledger alone
    # at_most: EAGLE etc. use approx; 2:4 decode speed-up at_most 1.2 +- 0.15
    assert ledger.evaluate_claim("P2-SP-02", 1.5, apply=False, db=db)["status"] == "partial"
    with pytest.raises(Exception):
        ledger.evaluate_claim("P1-EVAL-01", 1.0, db=db)  # no expectation
    assert ledger.evaluate_claim("P1-EVAL-01", 1.0, expected=1.0, tolerance=0.1, direction="approx", apply=False, db=db)["status"] == "reproduced"
    with pytest.raises(Exception):
        ledger.evaluate_claim("P1-CROSS-01", float("nan"), db=db)


def test_rule_evidence_and_eval_report_hook(tmp_path):
    from forge.core.db import Database
    from forge.core.schemas import EvalReport, MetricValue

    db = Database(tmp_path / "re.db")
    db.migrate()
    ledger.seed_claims(db=db)
    ev = ledger.rule_evidence("R-Q3-W4A16", db=db)
    assert ev["untested"] == 4 and not ev["ready"] and ev["reproduced"] == 0
    for cid in rb.RULES_BY_ID["R-Q3-W4A16"]["claims"]:
        ledger.set_status(cid, "reproduced", db=db)
    assert ledger.rule_evidence("R-Q3-W4A16", db=db)["ready"] is True
    rep = EvalReport(artifact_hash="a", metrics={"suite_score_ratio": MetricValue(value=0.975)})
    out = ledger.update_from_eval_report(rep, {"P1-W4-01": "suite_score_ratio", "P1-W8-01": "missing_metric"}, db=db)
    assert len(out) == 1 and out[0]["status"] == "reproduced" and rep.id in ledger.get_claim("P1-W4-01", db=db).linked_experiments
    with pytest.raises(Exception):
        ledger.rule_evidence("R-NOPE", db=db)


def test_ensure_seeded_uses_default_db():
    n = ledger.ensure_seeded()
    assert n >= len(SEED_CLAIMS)
    assert ledger.get_claim("P7-EXACT-01").prior == "trusted"
