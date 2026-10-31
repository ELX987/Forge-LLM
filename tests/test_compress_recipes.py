"""End-to-end recipe tests: stage registration, Appendix C.1 on a tiny random ForgeLM (CPU), caching, store registration, licence gate."""
from __future__ import annotations

import pytest
import torch
import yaml

from forge.compress.config import STAGE_PARAM_MODELS, CompressJobConfig, stage_schemas
from forge.compress.run import run_compress
from forge.core import registry
from forge.core.context import NullContext
from forge.core.errors import ConfigError, GateFailed, LicenceBlocked, NotFound, StageFailed
from forge.core.recipe import dump_recipe, load_recipe
from tests.compress_helpers import logits_of, rand_tokens, rel_err

# Plan Appendix C.1 verbatim (only the model/calibration names are symbolic placeholders, as in the Plan)
APPENDIX_C1 = """
recipe: w4a16-fp8kv-70b
inputs: {model: sha256:base70b_bf16, calib: sha256:mix_v3, chip: chipA-r3}
stages:
- calibrate: {set: mix_v3, n_seqs: 512, seq_len: 2048, stats: [hessian, channel_scale]}
- ptq: {method: gptq, bits: 4, group: 128, act_order: true, damp: 0.01, sym: false}
- pack: {layout: auto, target: chipA-r3} # emits layout_id
- kv: {dtype: fp8_e4m3, residual_window: 128}
- gate: {suite: G-W4-v2, parent: sha256:base70b_bf16, seeds: 3, stats: bootstrap_2000}
baselines: [fp8_w8a8, fresh_student_8b]
budget: {gpu_hours: 40, max_wall_hours: 12}
policy: {licence_check: true}
"""

TEXTS = ["Calibration sample number %d. The quick brown fox jumps over the lazy dog while counting to %d." % (i, i * 7) for i in range(40)]


def _job(**kw):
    base = dict(model="tiny:3", dtype="float32", device="cpu", recipe=APPENDIX_C1, use_store=False, calibration_texts={"mix_v3": TEXTS},
                stage_overrides={"calibrate": {"n_seqs": 16, "seq_len": 48}, "kv": {"report_seqs": 4, "report_len": 48}, "gate": {"seeds": 1}},
                fail_on_gate=False)
    base.update(kw)
    return CompressJobConfig(**base)


def test_all_stages_registered_with_param_models():
    import forge.compress.run  # noqa: F401

    assert set(registry.names("stage")) >= {"calibrate", "ptq", "pack", "kv", "rotate", "prune", "lowrank", "delta", "lossless", "eval", "gate", "distill", "finetune"}
    assert set(STAGE_PARAM_MODELS) == {"calibrate", "ptq", "pack", "kv", "rotate", "prune", "lowrank", "delta", "lossless", "eval", "gate", "distill", "finetune"}
    sch = stage_schemas()
    assert sch["ptq"]["properties"]["method"]["enum"] and "description" in sch["ptq"]["properties"]["bits"]
    for stage, s in sch.items():  # every field of every stage model is documented (the UI renders forms from this)
        for fname, f in s["properties"].items():
            assert f.get("description"), (stage, fname)


def test_job_config_validation():
    with pytest.raises(ValueError):
        CompressJobConfig(model="tiny", recipe="a", recipe_path="b")
    with pytest.raises(ValueError):
        CompressJobConfig(model="tiny")
    cfg = CompressJobConfig(model="tiny", recipe_path="x.yaml")
    assert cfg.fail_on_gate


def test_appendix_c1_recipe_end_to_end_on_tiny_model():
    ctx = NullContext()
    res = run_compress(_job(), ctx)
    st = res["stages"]
    assert [k.split("_", 1)[1] for k in st] == ["calibrate", "ptq", "pack", "kv", "gate"]
    assert res["compression"] > 1.5 and res["bytes_output"] < res["bytes_input"]
    # ptq: the C.1 parameters were honoured (group alias, act_order, asymmetric, damp)
    rep = st["01_ptq"]["ptq_report"]
    assert rep["config"]["method"] == "gptq" and rep["config"]["act_order"] is True and rep["config"]["sym"] is False and rep["config"]["group_size"] == 128
    assert rep["layer_kinds"] == ["w4a16"]
    # pack emitted a layout id; kv and gate recorded
    assert res["layout_id"]
    assert res["kv"]["scheme"] == "fp8_e4m3" and res["kv"]["residual_window"] == 128
    assert res["gate"]["verdict"] in ("pass", "fail", "inconclusive") and res["gate"]["reasons"]
    assert (ctx.outputs["model_dir"] and ctx.outputs["recipe_hash"].startswith(("blake3:", "b2:")))
    # the saved directory loads through the loader (packed linears restored) and reproduces the in-memory model
    from forge.models.loader import load_model

    m2, _ = load_model(res["model_dir"], dtype="float32", device="cpu", with_tokenizer=False)
    assert any(type(m).__name__ == "W4A16Linear" for m in m2.modules())
    from forge.models.tiny import build_tiny_model

    t = rand_tokens(4, 24, vocab=256)
    ref = logits_of(build_tiny_model(3), t)
    assert rel_err(logits_of(m2, t), ref) < 0.5  # quantised but alive


def test_recipe_is_deterministic():
    a = run_compress(_job(), NullContext())
    b = run_compress(_job(), NullContext())
    assert a["recipe_hash"] == b["recipe_hash"]
    from forge.compress.serialize import load_quantized

    ma, _ = load_quantized(a["model_dir"], dtype="float32")
    mb, _ = load_quantized(b["model_dir"], dtype="float32")
    t = rand_tokens(2, 16, vocab=256)
    assert torch.equal(logits_of(ma, t), logits_of(mb, t))  # bit-reproducible stage chain


def test_gate_failure_registers_then_raises_and_policy_off():
    # absurd 2-bit RTN fails any sane KL gate; the job still produces the artifact first
    recipe = """
recipe: w2-bad
stages:
- ptq: {method: rtn, bits: 2, group: 32}
- gate: {suite: g-w4, seeds: 1, stats: bootstrap_1000}
"""
    ctx = NullContext()
    with pytest.raises(GateFailed) as ei:
        run_compress(CompressJobConfig(model="tiny", dtype="float32", device="cpu", recipe=recipe, use_store=False), ctx)
    assert ei.value.verdict["verdict"] == "fail"
    assert ctx.outputs.get("model_dir")  # artifact written before raising
    res = run_compress(CompressJobConfig(model="tiny", dtype="float32", device="cpu", recipe=recipe, use_store=False, fail_on_gate=False), NullContext())
    assert res["gate_failed"] is True


def test_prune_distill_quantize_pipeline_with_licence_gate(tmp_path):
    recipe = """
recipe: prune-distill-q8
stages:
- calibrate: {set: synthetic, n_seqs: 16, seq_len: 32}
- prune: {importance: activation, mlp_keep: 0.5, head_mode: group, head_keep: 0.5, depth_remove: 1, protect_ends: false, n_seqs: 8, seq_len: 32}
- distill: {mode: builtin, steps: 30, lr: 0.0003, n_seqs: 16, seq_len: 32}
- ptq: {method: rtn, bits: 8, group: 32}
- eval: {builtin: true, n_seqs: 8, seq_len: 32}
"""
    ctx = NullContext()
    res = run_compress(CompressJobConfig(model="tiny", dtype="float32", device="cpu", recipe=recipe, use_store=False, fail_on_gate=False, seed=1), ctx)
    st = res["stages"]
    pr = st["01_prune"]["prune_report"]
    assert pr["needs_recovery"] and pr["intermediate_size"] == 64 and pr["num_heads"] == 2 and pr["layers_removed"] and pr["ratio"] < 0.7
    d = st["02_distill"]["distill_report"]
    assert d["kl_after"] < d["kl_before"]  # recovery helped
    assert res["compression"] > 2.0  # embeddings + head (kept fp32) dominate a tiny model
    m = st["04_eval"]["eval_report"]["metrics"]
    assert m["ppl"]["value"] > 0 and "kl_mean" in m
    # licence gate: a model loaded from a directory is not "locally owned" -> needs a permissive licence or an attestation
    from forge.models.loader import save_model
    from forge.models.tiny import build_tiny_model

    d = save_model(build_tiny_model(0), tmp_path / "in")
    base = dict(model=str(d), dtype="float32", device="cpu", recipe=recipe, use_store=False, fail_on_gate=False)
    with pytest.raises(LicenceBlocked):
        run_compress(CompressJobConfig(**base), NullContext())  # unknown licence, no attestation
    with pytest.raises(LicenceBlocked):
        run_compress(CompressJobConfig(**base, licence="llama-community", distill_allowed=False), NullContext())
    ok = run_compress(CompressJobConfig(**base, licence="apache-2.0"), NullContext())
    assert ok["stages"]["02_distill"]["distill_report"]["steps"] == 30
    ok2 = run_compress(CompressJobConfig(**base, licence="custom", distill_allowed=True), NullContext())
    assert ok2["artifact"] is None and ok2["model_dir"]
    # recipe policy can switch the check off explicitly
    off = recipe + "policy: {licence_check: false}\n"
    assert run_compress(CompressJobConfig(**{**base, "recipe": off}), NullContext())["stages"]


def test_rotate_prune_lowrank_delta_lossless_stages():
    recipe = """
recipe: misc
stages:
- rotate: {mode: hadamard, r2: true}
- lowrank: {target_ratio: 0.9, n_seqs: 8, seq_len: 32}
- lossless: {}
"""
    res = run_compress(CompressJobConfig(model="tiny", dtype="float32", device="cpu", recipe=recipe, use_store=False), NullContext())
    st = res["stages"]
    assert st["00_rotate"]["rotate_report"]["orthogonality_error"] < 1e-9
    assert st["01_lowrank"]["lowrank_report"]["ratio"] <= 0.9 + 1e-6
    assert st["02_lossless"]["lossless_report"]["verified"]
    # ptq refuses nothing but rotate/prune refuse already-quantised models
    bad = """
recipe: bad-order
stages:
- ptq: {method: rtn, bits: 8}
- rotate: {}
"""
    with pytest.raises((ConfigError, StageFailed)):
        run_compress(CompressJobConfig(model="tiny", dtype="float32", device="cpu", recipe=bad, use_store=False), NullContext())


def test_unknown_stage_and_bad_params_rejected_before_running():
    with pytest.raises(NotFound):
        run_compress(CompressJobConfig(model="tiny", device="cpu", recipe="recipe: x\nstages:\n- frobnicate: {}", use_store=False), NullContext())
    with pytest.raises((ConfigError, StageFailed)):
        run_compress(CompressJobConfig(model="tiny", device="cpu", recipe="recipe: x\nstages:\n- ptq: {bits: 99}", use_store=False), NullContext())


def test_recipe_roundtrip_yaml():
    r = load_recipe(APPENDIX_C1)
    assert [s.name for s in r.stages] == ["calibrate", "ptq", "pack", "kv", "gate"]
    assert load_recipe(dump_recipe(r)).hash() == r.hash()
    assert yaml.safe_load(dump_recipe(r))["recipe"] == "w4a16-fp8kv-70b"


def test_store_registration_lineage_and_stage_cache(forge_home):
    """With an artifact store: every model stage is persisted (digest = manifest hash), the final artifact is registered with calibration hash,
    licence and lineage, and a second identical run is served from the stage cache."""
    from forge.store import get_store

    store = get_store()
    recipe = """
recipe: store-demo
stages:
- calibrate: {set: synthetic, n_seqs: 8, seq_len: 32}
- ptq: {method: rtn, bits: 8, group: 32}
"""
    cfg = CompressJobConfig(model="tiny:5", dtype="float32", device="cpu", recipe=recipe, licence="apache-2.0", chip="chipA", fail_on_gate=False)
    ctx = NullContext(store=store)
    res = run_compress(cfg, ctx)
    h = res["artifact"]
    assert h and store.has(h)
    man = store.get_manifest(h)
    assert man.licence == "apache-2.0" and man.recipe_hash == res["recipe_hash"] and man.calib_hash
    assert "chipA" in man.hardware_families
    assert any(t.name.endswith("qweight") and t.role in ("payload", "other") for t in man.tensors)
    lin = store.lineage(h)
    assert len(lin["nodes"]) >= 2 and lin["edges"]
    assert res["cache_hits"] == 0
    ctx2 = NullContext(store=store)
    res2 = run_compress(cfg, ctx2)
    assert res2["cache_hits"] >= 1  # unchanged recipe + input -> cached stages
    assert res2["artifact"] == h or store.has(res2["artifact"])
    # stored artifact can be re-run as an input model (digest input)
    cfg3 = CompressJobConfig(model=h, dtype="float32", device="cpu", recipe="recipe: again\nstages:\n- lossless: {}", fail_on_gate=False)
    res3 = run_compress(cfg3, NullContext(store=store))
    assert res3["stages"]["00_lossless"]["lossless_report"]["verified"]


def test_distill_delegates_to_forge_train(tmp_path):
    """prune -> distill(mode=delegate): the student/teacher directories are exported, forge.train.run.run_distill runs on a real (tiny) dataset,
    and its output model directory becomes the stage output."""
    from test_train_fixtures import ingest_tiny

    ds = ingest_tiny(tmp_path / "ds")
    recipe = {
        "recipe": "prune-distill-delegate",
        "stages": [
            {"prune": {"mlp_keep": 0.5, "importance": "magnitude"}},
            {"distill": {"mode": "delegate", "config": {
                "name": "student", "seed": 2, "model": {"tokenizer": "byte"}, "teacher": {"tokenizer": "byte"},
                "mode": "offline_logit", "data_kind": "tokens",
                "token_data": {"dataset": [str(ds)], "seq_len": 16, "val_sequences": 4, "num_workers": 0},
                "cache": {"k": 16, "max_sequences": 24, "shard_sequences": 8, "teacher_batch_size": 4, "dir": str(tmp_path / "kvc")},
                "micro_batch_size": 2, "grad_accum_steps": 2, "optim": {"lr": 3e-3}, "schedule": {"name": "cosine", "warmup_steps": 1},
                "budget": {"max_steps": 6}, "checkpoint": {"every_steps": 0}, "eval": {"every_steps": 3, "num_batches": 2},
                "logging": {"log_every": 2}, "final_dtype": "float32", "register_artifact": False}}},
        ],
    }
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_compress(CompressJobConfig(model="tiny", tiny_overrides={"vocab_size": 512}, dtype="float32", device="cpu", recipe=recipe, use_store=False), ctx)
    rep = res["stages"]["01_distill"]["distill_report"]
    assert rep["result"]["status"] == "completed" and rep["result"]["steps"] == 6 and rep["model_dir"]
    from forge.models.loader import load_model

    m, _ = load_model(res["model_dir"], dtype="float32", device="cpu", with_tokenizer=False)
    assert m.config.intermediate_size == 64  # the pruned architecture survived the round trip through the trainer


@pytest.mark.parametrize("name", ["fp8-w8a8", "lowrank", "nvfp4-ptq", "w4a16-gptq-g128"])
def test_shipped_recipe_templates_run_on_tiny_model(name):
    """The recipes under configs/recipes (written for 8B models) execute end to end on the CPU smoke model with shrunken calibration."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "recipes" / f"{name}.yaml"
    cfg = CompressJobConfig(
        model="tiny", dtype="float32", device="cpu", recipe_path=str(path), use_store=False, fail_on_gate=False,
        stage_overrides={"calibrate": {"n_seqs": 8, "seq_len": 32}, "gate": {"suites": ["distribution"], "seeds": 1}, "kv": {"report_seqs": 2, "report_len": 32}},
    )
    res = run_compress(cfg, NullContext())
    assert res["compression"] > 1.0 or name == "lowrank"
    assert res["gate"] and res["gate"]["verdict"] in ("pass", "fail", "inconclusive")


def test_delta_stage_with_local_base(tmp_path):
    import copy as _copy

    from forge.models.loader import load_model, save_model
    from forge.models.tiny import build_tiny_model

    base = build_tiny_model(0)
    ft = _copy.deepcopy(base)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for n, p in ft.named_parameters():
            if p.ndim == 2 and n.startswith("layers."):
                p.add_(torch.randn(p.shape, generator=g) * 0.02)
    bdir, fdir = save_model(base, tmp_path / "base"), save_model(ft, tmp_path / "ft")
    recipe = {"recipe": "delta-demo", "stages": [{"delta": {"method": "bitdelta", "base": str(bdir), "calibrate_steps": 20, "n_seqs": 8, "seq_len": 24}}]}
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_compress(CompressJobConfig(model=str(fdir), dtype="float32", device="cpu", recipe=recipe, use_store=False, licence="mit"), ctx)
    rep = res["stages"]["00_delta"]["delta_report"]
    assert rep["method"] == "bitdelta" and rep["compression_vs_dense"] > 2.0
    assert rep["calibration"]["loss_after"] <= rep["calibration"]["loss_before"]
    from forge.compress.delta import Delta, apply_delta

    d = Delta.load(tmp_path / "run" / "delta")
    base2, _ = load_model(bdir, dtype="float32", device="cpu", with_tokenizer=False)
    ft2, _ = load_model(fdir, dtype="float32", device="cpu", with_tokenizer=False)
    applied = apply_delta(base2, d)
    t = rand_tokens(2, 16, vocab=256)
    assert rel_err(logits_of(applied, t), logits_of(ft2, t)) < rel_err(logits_of(base2, t), logits_of(ft2, t))


def test_ptq_config_combination_validation():
    from forge.compress.ptq import PTQConfig

    with pytest.raises(ValueError):
        PTQConfig(method="awq", format="nvfp4")
    with pytest.raises(ValueError):
        PTQConfig(method="smoothquant", format="fp8")
    with pytest.raises(ValueError):
        PTQConfig(mixed_precision=True, format="mxfp4")
    assert PTQConfig(method="gptq", format="nvfp4").format == "nvfp4"


def test_delta_stage_resolves_base_on_the_finetuned_models_device(tmp_path, monkeypatch):
    """The base model must be loaded on the fine-tune's device/dtype, not resolve_model's 'cuda if available' default."""
    from forge.compress import state, stages
    from forge.models.loader import save_model
    from forge.models.tiny import build_tiny_model

    bdir = save_model(build_tiny_model(0), tmp_path / "base")
    seen = {}
    orig = state.resolve_model

    def spy(digest, ctx=None, *, dtype=None, device=None):
        seen.update(dtype=dtype, device=device)
        return orig(digest, ctx, dtype=dtype, device=device)

    monkeypatch.setattr(state, "resolve_model", spy)
    ft = build_tiny_model(0)
    state.put_live("ft-test-digest", ft, None)
    try:
        state.require_live(str(bdir), None, "base", **stages._placement({"model": "ft-test-digest"}))
    finally:
        pass
    assert seen["device"] == "cpu" and seen["dtype"] == "float32"
