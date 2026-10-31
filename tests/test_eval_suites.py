from __future__ import annotations

import random
import time

import numpy as np
import pytest
import torch

from forge.eval import suites
from forge.eval.metrics import CallableProvider
from forge.eval.suites import agentic, contamination, distribution, longctx, longgen, mcq, safety, throughput
from forge.eval.suites.base import ByteTokenizer, SuiteEnv, read_jsonl, resolve_data_path, score_continuations
from forge.eval.testing import BigramProvider, SolverProvider

TOK = ByteTokenizer()


def env(cand, parent=None, **kw):
    return SuiteEnv(parent=parent, candidate=cand, tokenizer=TOK, n_boot=200, seed=kw.pop("seed", 0), **kw)


# ----------------------------------------------------------------------------- registry / data
def test_registry_and_schemas():
    names = suites.list_suites()
    for n in suites.BUILTIN:
        assert n in names
    desc = suites.describe_suites()
    assert {d["name"] for d in desc} >= set(suites.BUILTIN)
    d = next(x for x in desc if x["name"] == "longctx")
    assert "lengths" in d["schema"]["properties"] and d["defaults"]["n_per_task"] == 8
    with pytest.raises(Exception):
        suites.get_suite("nope")
    with pytest.raises(Exception):
        suites.get_suite("mcq").make_config({"scoring": "bad"})
    with pytest.raises(Exception):
        suites.get_suite("mcq").make_config({"unknown_field": 1})


def test_bundled_data_is_flagged_seed_and_parses():
    for f in ("heldout_seed.jsonl", "mcq_smoke.jsonl", "safety_refusal_seed.jsonl", "safety_stereotype_pairs_seed.jsonl"):
        p = resolve_data_path(None, f)
        assert "SEED EXAMPLES, NOT A BENCHMARK" in p.read_text(encoding="utf-8").splitlines()[0]
        rows = read_jsonl(p)
        assert len(rows) >= 12
    assert len(read_jsonl(resolve_data_path("bundled:mcq_smoke.jsonl"))) == 24


# ----------------------------------------------------------------------------- distribution
def test_distribution_parent_vs_noisy_candidate():
    parent = BigramProvider(seed=0)
    cand = parent.perturbed(0.6)
    cfg = distribution.DistributionConfig(max_seq_len=64, batch_size=3)
    r = distribution.DistributionSuite().run(cfg, env(cand, parent))
    m = r.metrics
    for k in ("kl_mean", "kl_p99", "top1_flip", "entropy_shift", "ppl_candidate", "ppl_parent", "ppl_ratio", "ece_candidate", "ece_parent", "ece_delta"):
        assert k in m, k
    assert m["kl_p99"].value >= m["kl_mean"].value > 0
    assert m["kl_mean"].ci_low <= m["kl_mean"].value <= m["kl_mean"].ci_high
    assert 0 < m["top1_flip"].value < 1 and m["top1_flip"].n == r.details["n_tokens"]
    assert r.flags["output_identical"] is False
    assert "top1_flip" in r.boot and r.details["n_sequences"] == 16


def test_distribution_identical_models_and_no_parent():
    p = BigramProvider(seed=1)
    r = distribution.DistributionSuite().run(distribution.DistributionConfig(), env(p, p))
    assert r.metrics["kl_mean"].value == 0 and r.metrics["top1_flip"].value == 0 and r.flags["output_identical"] is True
    assert r.metrics["ppl_ratio"].value == pytest.approx(1.0)
    r2 = distribution.DistributionSuite().run(distribution.DistributionConfig(), env(p))
    assert "kl_mean" not in r2.metrics and "ppl_candidate" in r2.metrics and r2.notes


def test_distribution_custom_token_file(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"token_ids": [1,2,3,4,5,6]}\n{"token_ids": [9]}\n{"text": "hello world"}\n')
    cfg = distribution.DistributionConfig(data_path=str(path), max_seq_len=4)
    seqs = distribution.load_sequences(cfg, env(BigramProvider()))
    assert seqs == [[1, 2, 3, 4], list(b"hell")]
    txt = tmp_path / "t.txt"
    txt.write_text("first doc here\n\nsecond doc here")
    assert len(distribution.load_sequences(distribution.DistributionConfig(data_path=str(txt)), env(BigramProvider()))) == 2


# ----------------------------------------------------------------------------- mcq
def test_mcq_matches_manual_scoring_and_pairs():
    parent = BigramProvider(seed=2)
    cand = parent.perturbed(0.5)
    cfg = mcq.MCQConfig(scoring="sum")
    items = mcq.load_items(cfg)
    assert len(items) == 24
    scores = mcq.choice_scores(parent, TOK, cfg, items)
    # manual check of the first item/choice
    it = items[0]
    ctx = TOK.encode(mcq.build_prompt(cfg, it))
    lp, _ = score_continuations(parent, [ctx], [TOK.encode(" " + it["choices"][0])])
    assert scores[0][0] == pytest.approx(lp[0])
    r = mcq.MCQSuite().run(cfg, env(cand, parent))
    for k in ("mcq_acc", "mcq_acc_parent", "mcq_acc_diff_pts", "mcq_acc_ratio", "mcq_ece", "mcq_ece_parent"):
        assert k in r.metrics, k
    par_acc = np.mean([np.argmax(s) == i["answer"] for s, i in zip(scores, items)])
    assert r.metrics["mcq_acc_parent"].value == pytest.approx(par_acc)
    assert r.contributes_to_suite_score and r.items["score"][1].shape == (24,)
    assert "mcq_acc_diff_pts" in r.boot and set(r.details["by_subject"]) >= {"arithmetic", "code"}
    assert r.generations and r.details["chance"] == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize("scoring", ["sum", "mean_token", "byte_norm", "letter"])
def test_mcq_all_scoring_modes_run(scoring):
    r = mcq.MCQSuite().run(mcq.MCQConfig(scoring=scoring, max_items=6), env(BigramProvider(seed=3)))
    assert 0 <= r.metrics["mcq_acc"].value <= 1 and "mcq_acc_parent" not in r.metrics


def test_mcq_shuffle_changes_order_but_keeps_answers_consistent(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"question":"Q1","choices":["a","b","c"],"answer":"B"}\n{"question":"Q2","choices":["x","y"],"answer":0}\n')
    cfg = mcq.MCQConfig(data_path=str(p), shuffle_choices=True)
    items = mcq.load_items(cfg)
    assert items[0]["answer"] == 1 and items[1]["answer"] == 0
    with pytest.raises(Exception):
        mcq.parse_answer("Z", 3)
    with pytest.raises(Exception):
        mcq.parse_answer(7, 3)
    r = mcq.MCQSuite().run(cfg, env(BigramProvider(), seed=5))
    assert r.details["n_items"] == 2


# ----------------------------------------------------------------------------- longctx
@pytest.mark.parametrize("task", longctx.TASKS)
def test_longctx_generators_length_control_and_answers_present(task):
    for L in (700, 2000):
        items = longctx.generate_items(task, L, 3, seed=11, tok=TOK, max_new=32)
        for it in items:
            assert it.actual_tokens <= L, (task, L, it.actual_tokens)
            assert it.actual_tokens >= 0.8 * L, (task, L, it.actual_tokens)
            assert it.answers
            if task != "aggregation":
                for a in it.answers:
                    assert a in it.prompt
    # reproducible
    a = longctx.generate_items(task, 700, 2, seed=11, tok=TOK)
    b = longctx.generate_items(task, 700, 2, seed=11, tok=TOK)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert longctx.generate_items(task, 700, 1, seed=12, tok=TOK)[0].prompt != a[0].prompt


def test_longctx_score_output():
    assert longctx.score_output("the number is 1234567.", ["1234567"]) == 1.0
    assert longctx.score_output("number 12345678", ["1234567"]) == 0.0  # digit boundary
    assert longctx.score_output("a b", ["A", "c"]) == 0.5
    assert longctx.score_output("", []) == 0.0


def test_longctx_aggregation_truth_is_correct():
    it = longctx.gen_aggregation(TOK, 800, 3, 32)
    words = it.prompt.split("Count how often each appears.\n")[1].split("\n\n")[0].split()
    cnt = {w: words.count(w) for w in set(words)}
    assert set(it.answers) == set(sorted(cnt, key=lambda w: -cnt[w])[:3])


def test_longctx_suite_solver_perfect_vs_degraded():
    good, weak = SolverProvider(1.0), SolverProvider(0.0, name="weak")
    cfg = longctx.LongCtxConfig(lengths=[600, 1200], n_per_task=2, max_new_tokens=64)
    r = longctx.LongCtxSuite().run(cfg, env(good))
    assert r.metrics["longctx_score"].value == pytest.approx(1.0), {k: v.value for k, v in r.metrics.items() if k.startswith("longctx_") and v.value < 1}
    r = longctx.LongCtxSuite().run(cfg, env(weak, good, deployment_length=1200))
    assert r.metrics["longctx_score"].value < 0.2 and r.metrics["longctx_score_parent"].value == pytest.approx(1.0)
    assert r.metrics["longctx_ratio"].value < 0.2 and r.details["ratio_length"] == 1200
    assert r.flags["longctx_length_matches_deployment"] is True
    assert "longctx_multihop_diff_pts" in r.metrics and "longctx_aggregation_diff_pts" in r.boot
    assert {"longctx_len600", "longctx_len1200"} <= set(r.metrics)
    r = longctx.LongCtxSuite().run(cfg, env(good, good, deployment_length=5000))
    assert r.flags["longctx_length_matches_deployment"] is False and r.notes
    assert r.metrics["longctx_ratio"].value == pytest.approx(1.0)


# ----------------------------------------------------------------------------- longgen
def test_longgen_chain_scoring():
    rng = random.Random(0)
    ch = longgen.make_chain(rng, 5)
    good = "\n".join(f"step {i}: {v}" for i, v in enumerate(ch.values, 1)) + f"\nAnswer: {ch.values[-1]}"
    s = longgen.score_chain(good, ch)
    assert s["final"] == 1.0 and s["step_acc"] == 1.0 and s["first_error"] == 1.0
    bad = good.replace(f"step 3: {ch.values[2]}", "step 3: 999999")
    s = longgen.score_chain(bad, ch)
    assert s["step_acc"] == pytest.approx(4 / 5) and s["first_error"] == pytest.approx(3 / 6)
    assert longgen.score_chain("nothing", ch)["final"] == 0.0


def test_longgen_suite_ratio_alias_and_divergence():
    parent, cand = SolverProvider(1.0), SolverProvider(0.5, name="half")
    cfg = longgen.LongGenConfig(steps=[4, 8], n_per_length=10, max_new_tokens=400, divergence_prompts=6)
    r = longgen.LongGenSuite().run(cfg, env(cand, parent))
    assert r.metrics["longgen_acc_parent"].value == pytest.approx(1.0)
    assert 0.2 < r.metrics["longgen_acc"].value < 0.8
    assert r.metrics["longgen_reasoning"].value == pytest.approx(r.metrics["longgen_acc_ratio"].value)
    assert "longgen_reasoning" in r.boot and "longgen_first_divergence" in r.metrics
    assert 0.0 <= r.metrics["longgen_identical_rate"].value <= 1.0
    zero = longgen.LongGenSuite().run(longgen.LongGenConfig(steps=[3], n_per_length=3), env(BigramProvider(), BigramProvider(seed=9)))
    assert any("undefined" in n for n in zero.notes)


# ----------------------------------------------------------------------------- safety
def test_safety_refusal_pairs_and_delta():
    good = SolverProvider(1.0)
    r = safety.SafetySuite().run(safety.SafetyConfig(), env(good, good))
    assert r.metrics["refusal_rate_harmful"].value == 1.0 and r.metrics["over_refusal_rate"].value == 0.0
    assert r.metrics["safety_delta"].value == 0.0 and r.metrics["safety_delta"].within_noise is True
    assert "stereotype_pref_rate" in r.metrics
    weak = SolverProvider(0.0, name="unsafe")
    r = safety.SafetySuite().run(safety.SafetyConfig(), env(weak, good))
    assert r.metrics["refusal_rate_harmful"].value == 0.0
    assert r.metrics["refusal_rate_harmful_diff_pts"].value == pytest.approx(-100.0)
    assert r.metrics["safety_delta"].value >= 100.0 and r.metrics["safety_delta"].within_noise is False
    assert r.details["worst_metric"] == "refusal_rate_harmful"
    assert any("not a benchmark" in n for n in r.notes)


def test_safety_stereotype_pairs_and_custom_judge(tmp_path):
    prov = BigramProvider(seed=4)
    r = safety.SafetySuite().run(safety.SafetyConfig(run_refusal=False), env(prov, prov))
    assert r.metrics["stereotype_pref_rate"].value == r.metrics["stereotype_pref_rate_parent"].value
    assert r.metrics["stereotype_pref_rate_diff_pts"].value == 0
    suite = safety.SafetySuite()
    suite.judge = lambda text, row: True
    r = suite.run(safety.SafetyConfig(run_pairs=False, max_items=20), env(prov))
    assert r.metrics["refusal_rate_harmful"].value == 1.0 and r.metrics["over_refusal_rate"].value == 1.0


# ----------------------------------------------------------------------------- contamination
def test_ngram_hashes_and_overlap(tmp_path):
    ids = contamination.word_ids("the quick brown fox jumps over the lazy dog again and again")
    h = contamination.ngram_hashes(ids, 4)
    assert h.size == len(ids) - 3
    assert contamination.ngram_hashes(ids[:2], 4).size == 0
    ref = "the quick brown fox jumps over the lazy dog every single morning without fail " * 2
    idx = contamination.NgramIndex(5)
    idx.add_text(ref)
    rep = contamination.scan_overlap(
        ["The quick brown fox jumps over the lazy dog every single morning", "completely different words appear in this unrelated sentence here today", "short"],
        idx, ids=["a", "b", "c"],
    )
    assert rep.too_short == 1 and rep.items == 2
    assert rep.flagged_ids == ["a"] and rep.contamination_rate == 0.5 and rep.per_item[0] == 1.0 and rep.per_item[1] == 0.0


def test_contamination_suite_end_to_end(tmp_path):
    ev = tmp_path / "ev.jsonl"
    ev.write_text('{"id":"e1","question":"what is the capital city of the small country located between the two great mountain ranges","choices":["a","b"],"answer":0}\n'
                  '{"id":"e2","question":"an entirely novel question about something that never appears in any training text whatsoever ok","choices":["c","d"],"answer":1}\n')
    ref = tmp_path / "train"
    ref.mkdir()
    (ref / "a.txt").write_text("blah blah\n\nwhat is the capital city of the small country located between the two great mountain ranges a b\n\nmore text")
    cfg = contamination.ContaminationConfig(eval_paths=[str(ev)], reference_paths=[str(ref)], n=8)
    r = contamination.ContaminationSuite().run(cfg, env(None))
    assert r.flags["contamination_scanned"] and r.metrics["contamination_rate"].value == pytest.approx(0.5)
    assert r.details["flagged_ids"] == ["ev.jsonl:e1"]
    r = contamination.ContaminationSuite().run(contamination.ContaminationConfig(), env(None))
    assert r.flags["contamination_scanned"] is False and r.notes


# ----------------------------------------------------------------------------- throughput
class FakeClient:
    name = "fake"

    def __init__(self, delay=0.002):
        self.delay = delay

    def stream(self, prompt_ids, max_tokens, *, temperature=0.0, seed=None):
        time.sleep(self.delay * 3)  # "prefill"
        for i in range(max_tokens):
            time.sleep(self.delay)
            yield i


def test_throughput_with_fake_clients_and_speedup():
    cfg = throughput.ThroughputConfig(concurrency=[1, 3], gen_len=6, requests_per_level=3, prompt_len=4, warmup_requests=1)
    r = throughput.ThroughputSuite().run(cfg, env(None, candidate_serve=FakeClient(0.002), parent_serve=FakeClient(0.004)))
    for k in ("ttft_p50_ms_c1", "itl_p50_ms_c3", "tokens_per_s_c3", "tokens_per_s_parent_c1", "tokens_per_s", "itl_p99_ms", "throughput_speedup", "throughput_speedup_c1"):
        assert k in r.metrics, k
    assert r.metrics["throughput_speedup_c1"].value > 1.2  # candidate sleeps half as long
    assert r.metrics["tokens_per_s_c3"].value > r.metrics["tokens_per_s_c1"].value * 1.3  # concurrency helps
    assert r.metrics["ttft_p50_ms_c1"].value >= 6.0


def test_throughput_falls_back_to_provider_client_and_labels_it():
    p = BigramProvider(vocab=64)
    cfg = throughput.ThroughputConfig(concurrency=[1], gen_len=3, requests_per_level=2, prompt_len=5, prompt_vocab=60, warmup_requests=0)
    r = throughput.ThroughputSuite().run(cfg, env(p))
    assert r.metrics["tokens_per_s"].value > 0 and any("NOT serving-engine numbers" in n for n in r.notes)
    r = throughput.ThroughputSuite().run(cfg, env(None))
    assert not r.metrics and r.notes


def test_http_serve_client_parses_sse(monkeypatch):
    import io
    import json
    import urllib.request

    from forge.eval.serve_client import HTTPServeClient

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    chunks = [json.dumps({"choices": [{"text": t, "finish_reason": None}]}) for t in ("a", "b", "c")]
    body = ("".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data.decode())
        seen["auth"] = req.get_header("Authorization")
        return Resp(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = HTTPServeClient("http://x:1/", model="m", api_key="k")
    assert list(c.stream([1, 2, 3], 5, seed=3)) == ["a", "b", "c"]
    assert seen["url"] == "http://x:1/v1/completions" and seen["body"]["prompt"] == [1, 2, 3] and seen["body"]["stream"] is True
    assert seen["auth"] == "Bearer k" and seen["body"]["seed"] == 3


# ----------------------------------------------------------------------------- agentic
@pytest.mark.parametrize("kind", agentic.KINDS)
def test_agentic_environment_solved_by_oracle(kind):
    for seed in range(40):
        e = agentic.SimEnvironment(seed)
        t = agentic.make_task(e, kind, random.Random(seed), seed)
        for tool, args in agentic.oracle_actions(e, t):
            obs = e.execute(tool, args)
            assert not obs.startswith("ERROR"), (kind, seed, tool, args, obs)
        assert e.submitted == t.answer


def test_agentic_tools_and_parser():
    e = agentic.SimEnvironment(1)
    assert e.lookup("nobody", "x").startswith("ERROR: unknown entity")
    p = e.people[0]
    assert e.lookup(p, "salary").startswith("ERROR: entity")
    assert e.calc("2 * (3 + 4)") == "14" and e.calc("__import__('os')").startswith("ERROR")
    assert e.calc("10 / 4") == "2.5"
    assert e.execute("nope", ()).startswith("ERROR: unknown tool")
    assert e.execute("lookup", ("a",)).startswith("ERROR: wrong arguments")
    assert agentic.parse_action('blah\nAction: lookup("A b", "city")\nmore') == ("lookup", ("A b", "city"))
    assert agentic.parse_action("no action here") is None
    assert agentic.parse_action("Action: lookup(garbage") is None
    assert agentic.fmt_action("calc", ("1 + 2",)) == 'Action: calc("1 + 2")'
    assert "Action: lookup" in agentic.demo_trace()


def test_agentic_suite_solver_vs_weak():
    good, weak = SolverProvider(1.0), SolverProvider(0.0, name="weak")
    cfg = agentic.AgenticConfig(n_per_kind=4)
    r = agentic.AgenticSuite().run(cfg, env(good, good))
    assert r.metrics["agentic_success"].value == 1.0, r.details
    assert r.metrics["agentic_step_match"].value == 1.0 and r.metrics["agentic_valid_call_rate"].value == 1.0
    r = agentic.AgenticSuite().run(cfg, env(weak, good))
    assert r.metrics["agentic_success"].value < 0.3 and r.metrics["agentic_success_diff_pts"].value < -50
    assert set(r.details["by_kind"]) == set(agentic.KINDS)
    # a model that never produces a well-formed action fails cleanly
    r = agentic.AgenticSuite().run(agentic.AgenticConfig(n_per_kind=1, max_turns=2), env(BigramProvider(seed=7)))
    assert r.metrics["agentic_success"].value == 0.0 and r.metrics["agentic_valid_call_rate"].value <= 1.0
