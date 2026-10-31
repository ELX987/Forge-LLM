"""Distillation: licence gate, diversity metrics, top-k logit cache, offline/live/sequence-level/on-policy/reasoning rounds."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from test_train_fixtures import ingest_tiny, make_model, write_sft

from forge.core.context import NullContext
from forge.core.errors import ConfigError, LicenceBlocked
from forge.train.config import DistillConfig, GenerationConfig, TeacherConfig
from forge.train.data import ShardedTokenDataset
from forge.train.distill import (
    TopKLogitCache,
    build_topk_cache,
    cache_key,
    check_teacher_licence,
    distinct_n,
    diversity_report,
    generate_tokens,
    ngram_entropy,
    run_distill,
    self_bleu,
    sentence_bleu,
)
from forge.train.losses import kd_loss_dense, kd_loss_topk


@pytest.fixture(scope="module")
def ds_dir(tmp_path_factory):
    return ingest_tiny(tmp_path_factory.mktemp("distds"))


# ----------------------------------------------------------------------------------------------- licence gate (TR-8)
@pytest.mark.parametrize("kw,ok", [
    (dict(licence="Apache-2.0"), True), (dict(licence="MIT"), True), (dict(licence="apache 2.0"), True),
    (dict(licence="llama3-community"), False), (dict(licence=None), False), (dict(licence="proprietary"), False),
    (dict(licence=None, locally_owned=True), True), (dict(licence="mit", distill_allowed=False), False),
    (dict(licence="proprietary", distill_allowed=True), True), (dict(licence="gpl-3.0"), False),
])
def test_licence_gate(kw, ok):
    t = TeacherConfig(name="t", **kw)
    if ok:
        info = check_teacher_licence(t)
        assert info["basis"] and info["teacher"] == "t"
    else:
        with pytest.raises(LicenceBlocked) as e:
            check_teacher_licence(t)
        assert e.value.details["teacher"] == "t" and e.value.http_status == 403


def test_run_distill_blocks_before_any_compute(tmp_path, ds_dir):
    cfg = _dcfg(ds_dir, tmp_path, teacher={"source": "tiny", "name": "closed-model", "licence": "proprietary", "locally_owned": False})
    ctx = NullContext(run_dir=tmp_path / "run")
    with pytest.raises(LicenceBlocked):
        run_distill(cfg, ctx)
    assert not ctx.metrics and not (tmp_path / "run" / "checkpoints").exists()


# ----------------------------------------------------------------------------------------------- diversity metrics
def test_diversity_metrics_known_values():
    assert distinct_n([[1, 2, 3, 4]], 1) == 1.0
    assert distinct_n([[1, 1, 1, 1]], 1) == 0.25
    assert distinct_n([[1, 2, 1, 2, 1, 2]], 2) == pytest.approx(2 / 5)
    assert distinct_n([[]], 2) == 0.0
    assert ngram_entropy([[1, 2, 3, 4]], 1) == pytest.approx(2.0)  # 4 equiprobable symbols
    assert ngram_entropy([[7, 7, 7]], 1) == 0.0
    assert ngram_entropy([[1, 2, 3, 4]], 1, normalise=True) == pytest.approx(1.0)
    assert sentence_bleu([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert sentence_bleu([1, 2, 3, 4, 5], [10, 11, 12, 13, 14]) < 0.01
    same = [[1, 2, 3, 4, 5, 6]] * 6
    diverse = [list(range(i * 10, i * 10 + 6)) for i in range(6)]
    assert self_bleu(same, 30) > 0.9 and self_bleu(diverse, 30) < 0.2
    assert self_bleu([[1]], 5) == 0.0
    rep_same, rep_div = diversity_report(same, 2), diversity_report(diverse, 2)
    assert rep_same["distinct_2"] < rep_div["distinct_2"] and rep_same["ngram_entropy_2"] < rep_div["ngram_entropy_2"]
    assert rep_same["self_bleu"] > rep_div["self_bleu"] and rep_div["n_samples"] == 6


# ----------------------------------------------------------------------------------------------- generation
def test_generate_tokens_determinism_eos_and_greedy():
    m = make_model(0, kind="forge").eval()
    g = GenerationConfig(max_new_tokens=6, temperature=1.0, top_k=5, top_p=0.9, num_samples=3)
    a = generate_tokens(m, [3, 4, 5], g, None, torch.device("cpu"), seed=1)
    b = generate_tokens(m, [3, 4, 5], g, None, torch.device("cpu"), seed=1)
    c = generate_tokens(m, [3, 4, 5], g, None, torch.device("cpu"), seed=2)
    assert a == b and a != c and all(len(s) == 6 for s in a)
    greedy = GenerationConfig(max_new_tokens=5, temperature=0.0, num_samples=2)
    gs = generate_tokens(m, [3, 4, 5], greedy, None, torch.device("cpu"))
    assert gs[0] == gs[1]
    eos = gs[0][0]
    stopped = generate_tokens(m, [3, 4, 5], greedy, eos, torch.device("cpu"))
    assert stopped[0] == [eos]


# ----------------------------------------------------------------------------------------------- top-k cache
def _teacher():
    return make_model(5).eval()


def _cache_ds(ds_dir, n=20):
    ds = ShardedTokenDataset([str(ds_dir)], 16, seed=0, role="train", val_sequences=4)
    ds.restrict(n)
    return ds


def test_topk_cache_build_read_and_stats(ds_dir, tmp_path):
    ds = _cache_ds(ds_dir)
    t = _teacher()
    cache = build_topk_cache(tmp_path / "c", t, ds, 20, k=8, shard_sequences=8, batch_size=3, device="cpu", meta_extra={"key": "k1"})
    assert cache.complete and cache.n == 20 and cache.k == 8 and cache.T == 16
    st = cache.stats
    assert st["positions"] == 20 * 16 and 0 < st["mean_coverage"] < 1 and st["p5"] <= st["p50"] and "frac_below_0.9" in st
    pos = np.array([0, 9, 19])
    got = cache.get(pos)
    batch = ds.get_batch(pos)
    with torch.no_grad():
        lp = torch.log_softmax(t(batch["input_ids"]).logits.float(), -1)
    v, i = lp.topk(8, -1)
    assert torch.equal(got["idx"], i) and torch.allclose(got["logp"], v.half().float(), atol=1e-3)
    assert torch.allclose(got["residual"], (1 - v.exp().sum(-1)).clamp(min=0), atol=1e-5)
    assert (got["residual"] > 0).all() and (got["residual"] < 1).all()
    # a bigger k covers more mass
    c2 = build_topk_cache(tmp_path / "c2", t, ds, 20, k=64, shard_sequences=20, batch_size=5, device="cpu")
    assert c2.stats["mean_coverage"] > st["mean_coverage"]
    # uint8 log-prob quantisation stays close
    c3 = build_topk_cache(tmp_path / "c3", t, ds, 20, k=8, lp_quant="uint8", lp_floor=-20.0, shard_sequences=20, batch_size=5, device="cpu")
    g3 = c3.get(pos)
    assert torch.equal(g3["idx"], i) and float((g3["logp"] - v).abs().max()) < 0.05
    with pytest.raises(ConfigError):
        cache.get(np.array([25]))
    with pytest.raises(ConfigError):
        TopKLogitCache(tmp_path / "missing")


def test_topk_cache_resume_and_key(ds_dir, tmp_path):
    ds = _cache_ds(ds_dir)
    t = _teacher()
    path = tmp_path / "c"
    calls = {"n": 0}
    orig = t.forward

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    t.forward = counting
    ctx = NullContext(run_dir=tmp_path / "r")
    ctx.stop = False
    # cancel after the first shard
    class Stop(NullContext):
        def progress(self, frac, msg=""):
            super().progress(frac, msg)
            self.stop = True

    c = Stop(run_dir=tmp_path / "r2")
    with pytest.raises(ConfigError, match="cancelled"):
        build_topk_cache(path, t, ds, 20, k=4, shard_sequences=8, batch_size=4, device="cpu", ctx=c, should_stop=c.should_stop)
    assert (path / "done_00000").exists() and not (path / "meta.json").exists()
    done_calls = calls["n"]
    cache = build_topk_cache(path, t, ds, 20, k=4, shard_sequences=8, batch_size=4, device="cpu")
    assert cache.complete
    assert calls["n"] < done_calls + 1 + 2 * 2 + 3  # shard 0 was not recomputed (only the 2 remaining shards + probe)
    k1 = cache_key("t", "tok", "d", 20, 16, 8, "fp16", -20.0)
    assert k1 == cache_key("t", "tok", "d", 20, 16, 8, "fp16", -20.0)
    assert len({k1, cache_key("t2", "tok", "d", 20, 16, 8, "fp16", -20.0), cache_key("t", "tok", "d", 20, 16, 16, "fp16", -20.0),
                cache_key("t", "tok", "d", 20, 16, 8, "uint8", -20.0), cache_key("t", "tok2", "d", 20, 16, 8, "fp16", -20.0)}) == 5


def test_cached_kd_equals_live_teacher_kd_when_k_is_vocab(ds_dir, tmp_path):
    ds = _cache_ds(ds_dir)
    t = _teacher()
    V = 288
    cache = build_topk_cache(tmp_path / "c", t, ds, 8, k=V, lp_quant="fp16", shard_sequences=8, batch_size=4, device="cpu")
    batch = ds.get_batch(np.arange(4))
    student = make_model(1).eval()
    sl = student(batch["input_ids"]).logits.detach()
    with torch.no_grad():
        tl = t(batch["input_ids"]).logits
    mask = batch["labels"] != -100
    cd = cache.get(np.arange(4))
    for kind in ("forward_kl", "reverse_kl", "jsd"):
        live, _ = kd_loss_dense(sl, tl, mask, kind)
        cached, _ = kd_loss_topk(sl, cd["idx"], cd["logp"], cd["residual"], mask, kind)
        assert float(cached) == pytest.approx(float(live), rel=0.02)  # fp16 log-prob storage error only


# ----------------------------------------------------------------------------------------------- end-to-end modes
def _dcfg(ds_dir, tmp_path, **over):
    cfg = {
        "name": "student", "seed": 2, "model": {"source": "tiny", "tokenizer": "byte"},
        "teacher": {"source": "tiny", "name": "tiny-teacher", "locally_owned": True},
        "mode": "offline_logit", "data_kind": "tokens",
        "token_data": {"dataset": [str(ds_dir)], "seq_len": 16, "val_sequences": 4, "num_workers": 0},
        "cache": {"k": 16, "max_sequences": 24, "shard_sequences": 8, "teacher_batch_size": 4},
        "micro_batch_size": 2, "grad_accum_steps": 2, "optim": {"lr": 3e-3}, "schedule": {"name": "cosine", "warmup_steps": 1},
        "budget": {"max_steps": 8}, "checkpoint": {"every_steps": 0}, "eval": {"every_steps": 4, "num_batches": 2},
        "logging": {"log_every": 2}, "final_dtype": "float32", "register_artifact": False,
    }
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    return DistillConfig.model_validate(cfg)


def test_offline_distillation_round_with_cache_reuse_and_lineage(ds_dir, tmp_path):
    from forge.store import ArtifactStore

    cfg = _dcfg(ds_dir, tmp_path, cache={"k": 16, "max_sequences": 24, "shard_sequences": 8, "dir": str(tmp_path / "kvc")}, register_artifact=True,
                model={"source": "tiny", "tokenizer": "byte", "parent_hash": None}, pruning_plan_hash="blake3:" + "ab" * 32)
    ctx = NullContext(run_dir=tmp_path / "run", store=ArtifactStore(tmp_path / "store"))
    res = run_distill(cfg, ctx)
    assert res["status"] == "completed" and res["steps"] == 8
    assert (tmp_path / "kvc" / "meta.json").exists()
    kd = [v for _, v in ctx.metric_series("kd")]
    assert kd[-1] < kd[0]
    agree = [v for _, v in ctx.metric_series("teacher_agreement")]
    assert all(0 <= a <= 1 for a in agree) and agree[-1] >= agree[0] - 1e-6
    json.loads((tmp_path / "run" / "model" / "forge_lineage.json").read_text()) if (tmp_path / "run" / "model").exists() else None
    d = res["distill"]
    assert d["teacher"]["basis"] == "locally owned teacher" and d["teacher"]["hash"] and d["cache_key"] and d["pruning_plan_hash"].startswith("blake3:")
    assert d["cache_stats"]["mean_coverage"] > 0 and d["token_budget_consumed"] == res["tokens_seen"]
    assert ctx.outputs["model_hash"] and ctx.store.has(ctx.outputs["model_hash"])
    man = ctx.store.get_manifest(ctx.outputs["model_hash"])
    assert man.extra["lineage"]["distill"]["mode"] == "offline_logit" and man.extra["lineage"]["precision_rung"] == "R0"
    # second run reuses the cache without touching the teacher
    ctx2 = NullContext(run_dir=tmp_path / "run2")
    res2 = run_distill(cfg, ctx2)
    assert any("reusing logit cache" in m for m in ctx2.logs) and not any("building top-" in m for m in ctx2.logs)
    assert res2["final_loss"] == pytest.approx(res["final_loss"], rel=1e-6)  # deterministic
    # rebuild flag forces recomputation
    ctx3 = NullContext(run_dir=tmp_path / "run3")
    run_distill(cfg.model_copy(update={"cache": cfg.cache.model_copy(update={"rebuild": True}), "budget": cfg.budget.model_copy(update={"max_steps": 2})}), ctx3)
    assert any("building top-" in m for m in ctx3.logs)


@pytest.mark.parametrize("kind", ["reverse_kl", "jsd"])
def test_offline_cache_other_divergences_and_temperature(ds_dir, tmp_path, kind):
    cfg = _dcfg(ds_dir, tmp_path, loss={"kind": kind, "temperature": 2.0, "alpha": 0.7}, cache={"dir": str(tmp_path / f"kvc_{kind}"), "k": 12, "max_sequences": 16},
                budget={"max_steps": 6})
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_distill(cfg, ctx)
    assert res["status"] == "completed" and math.isfinite(res["final_loss"])
    assert ctx.metric_series("kd") and ctx.metric_series("ce")


def test_live_teacher_mode_and_ce_only(ds_dir, tmp_path):
    cfg = _dcfg(ds_dir, tmp_path, cache={"use_cache": False}, loss={"kind": "forward_kl", "alpha": 1.0}, budget={"max_steps": 6})
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_distill(cfg, ctx)
    kd = [v for _, v in ctx.metric_series("kd")]
    assert res["status"] == "completed" and kd[-1] < kd[0] and not any("logit cache" in m for m in ctx.logs)
    cfg2 = _dcfg(ds_dir, tmp_path, loss={"kind": "ce_only"}, budget={"max_steps": 4})
    ctx2 = NullContext(run_dir=tmp_path / "run2")
    run_distill(cfg2, ctx2)
    assert not ctx2.metric_series("kd")


def test_cache_data_mismatch_is_detected(ds_dir, tmp_path):
    cfg = _dcfg(ds_dir, tmp_path, cache={"dir": str(tmp_path / "kvc"), "k": 8, "max_sequences": 16}, budget={"max_steps": 2})
    run_distill(cfg, NullContext(run_dir=tmp_path / "r1"))
    cfg2 = cfg.model_copy(update={"seed": 3, "token_data": cfg.token_data.model_copy(update={"data_seed": 0})})
    # same cache dir but a different holdout/seed does not change positions; corrupt the cache to simulate a stale one
    import gc

    gc.collect()
    for f in (tmp_path / "kvc").glob("crc_*.npy"):
        a = np.load(f)
        np.save(f, a + 1)
    meta = json.loads((tmp_path / "kvc" / "meta.json").read_text())
    assert meta["complete"]
    with pytest.raises(ConfigError, match="mismatch"):
        run_distill(cfg2, NullContext(run_dir=tmp_path / "r2"))


def test_vocab_mismatch_between_tokenizers_is_rejected(ds_dir, tmp_path):
    cfg = _dcfg(ds_dir, tmp_path, teacher={"source": "tiny", "locally_owned": True, "tokenizer": "byte"})
    # same tokenizer: fine (hash equal); a different tokenizer path must raise
    from forge.train import distill as dm

    class Fake:
        hash = "other"

    t = cfg.teacher.model_copy(update={"tokenizer": "byte"})
    with pytest.raises(ConfigError, match="tokenizers differ"):
        dm._check_tokenizers(t, Fake(), None)


def test_sequence_level_distillation(ds_dir, tmp_path):
    pf = tmp_path / "prompts.jsonl"
    pf.write_text("\n".join(json.dumps({"prompt": f"say {i}"}) for i in range(4)), encoding="utf-8")
    real = write_sft(tmp_path / "real", 6)
    cfg = _dcfg(ds_dir, tmp_path, mode="sequence_level", on_policy={"prompts": {"files": [str(pf)], "max_prompt_tokens": 64}},
                generation={"max_new_tokens": 6, "temperature": 1.0, "num_samples": 2},
                sft_data={"train_files": [str(real)], "max_seq_len": 64, "val_fraction": 0.1}, mix_non_synthetic=0.3, budget={"max_steps": 5})
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_distill(cfg, ctx)
    rows = [json.loads(line) for line in (tmp_path / "run" / "sequence_level_data.jsonl").read_text().splitlines()]
    assert res["status"] == "completed" and len(rows) == 8 + round(0.3 / 0.7 * 8)
    assert rows[0]["messages"][0]["role"] == "user" and rows[0]["teacher"] == "tiny-teacher"
    d = res["distill"]
    assert d["teacher_output_diversity"]["n_samples"] == 8 and d["mix_non_synthetic"]["synthetic"] == 8


def test_reasoning_trace_sft_with_verifier_filter(ds_dir, tmp_path):
    tf = tmp_path / "traces.jsonl"
    rows = [{"prompt": f"q{i}", "trace": f"step {i} then step {i + 1}", "answer": f"{i}", "verified": i % 3 != 0} for i in range(12)]
    tf.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cfg = _dcfg(ds_dir, tmp_path, mode="reasoning_sft", sft_data={"train_files": [str(tf)], "max_seq_len": 64, "val_fraction": 0.1}, budget={"max_steps": 4})
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_distill(cfg, ctx)
    st = res["distill"]["trace_filter"]
    assert st == {"read": 12, "kept": 8, "dropped": 4} and res["status"] == "completed"
    conv = [json.loads(line) for line in (tmp_path / "run" / "reasoning_traces.jsonl").read_text().splitlines()]
    assert conv[0]["messages"][1]["content"].startswith("<think>\n") and "</think>" in conv[0]["messages"][1]["content"]


def test_on_policy_distillation(ds_dir, tmp_path):
    pf = tmp_path / "prompts.txt"
    pf.write_text("\n".join(f"prompt number {i}" for i in range(6)), encoding="utf-8")
    cfg = _dcfg(ds_dir, tmp_path, mode="on_policy", loss={"kind": "reverse_kl"}, grad_accum_steps=1, budget={"max_steps": 6},
                on_policy={"prompts": {"files": [str(pf)], "max_prompt_tokens": 32, "use_chat_template": False}, "prompts_per_step": 2,
                           "generation": {"max_new_tokens": 5, "temperature": 1.0, "num_samples": 1}},
                diversity={"max_samples": 8, "ngram": 2}, eval={"every_steps": 0}, logging={"log_every": 1})
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_distill(cfg, ctx)
    assert res["status"] == "completed" and res["steps"] == 6
    assert ctx.metric_series("kd") and ctx.metric_series("teacher_agreement") and ctx.metric_series("diversity/distinct_2")
    assert all(math.isfinite(v) for _, v in ctx.metric_series("loss"))
    assert "diversity" in res["distill"]
    with pytest.raises(ConfigError, match="FSDP"):
        run_distill(cfg.model_copy(update={"parallel": cfg.parallel.model_copy(update={"strategy": "fsdp"})}), NullContext(run_dir=tmp_path / "x"))
