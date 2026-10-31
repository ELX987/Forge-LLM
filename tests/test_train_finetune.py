"""Fine-tuning: NF4/int8 base quantisation, LoRA backends, full/LoRA/QLoRA/DPO end-to-end on a tiny model."""
from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from test_train_fixtures import make_model, write_sft

from forge.core.context import NullContext
from forge.core.errors import ConfigError
from forge.train.config import FinetuneConfig, LoraConfig, QLoraConfig
from forge.train.lora import (
    NF4_CODE,
    LoraBackend,
    LoRALinear,
    QuantizedLinear,
    adapter_disabled,
    merge_adapter_state,
    quantize_base_linears,
)
from forge.train.run import run_finetune


# ----------------------------------------------------------------------------------------------- quantised base
def test_nf4_codebook_and_roundtrip():
    assert len(NF4_CODE) == 16 and NF4_CODE[0] == -1.0 and NF4_CODE[-1] == 1.0 and 0.0 in NF4_CODE
    torch.manual_seed(0)
    lin = nn.Linear(100, 24)  # 100 not divisible by the block size: padding path
    q = QuantizedLinear.from_linear(lin, bits=4, block_size=32)
    w = q.dequantize(torch.float32)
    assert w.shape == lin.weight.shape
    rel = float((w - lin.weight).detach().norm() / lin.weight.detach().norm())
    assert 0.05 < rel < 0.14  # NF4 relative error on Gaussian weights ~0.1
    assert q.qweight.dtype == torch.uint8 and q.qweight.shape[1] == 64  # two codes per byte (128 padded / 2)
    x = torch.randn(5, 100)
    assert float((q(x) - lin(x)).norm() / lin(x).norm()) < 0.2
    q8 = QuantizedLinear.from_linear(lin, bits=8, block_size=32)
    assert float((q8.dequantize() - lin.weight).norm() / lin.weight.norm()) < 0.01
    assert not any(p.requires_grad for p in q.parameters()) and len(list(q.parameters())) == 0  # frozen buffers only


def test_quantize_base_linears_excludes_head():
    m = make_model(0)
    n = quantize_base_linears(m, QLoraConfig(bits=4, block_size=16), torch.float32)
    assert n > 0 and isinstance(m.layers[0].self_attn.q_proj, QuantizedLinear) and isinstance(m.lm_head, nn.Linear)
    out = m(torch.randint(0, 100, (2, 8))).logits
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------------------------------- LoRA
def _lora_model(kind="local", builtin=True, **lkw):
    model = make_model(0, kind=kind)
    be = LoraBackend(LoraConfig(r=4, alpha=8, **lkw), prefer_builtin=builtin)
    names = be.apply(model)
    return model, be, names


def test_builtin_lora_semantics():
    ids = torch.randint(0, 100, (2, 8))
    base = make_model(0)
    ref = base(ids).logits.detach()
    model, be, names = _lora_model(builtin=True)
    assert be.impl == "builtin" and len(names) == 3 * 7
    assert torch.allclose(model(ids).logits, ref, atol=1e-6)  # B = 0 => identity at init
    train = [n for n, p in model.named_parameters() if p.requires_grad]
    assert train and all(".lora_A." in n or ".lora_B." in n for n in train)
    assert not model.lm_head.weight.requires_grad
    # train one step: only adapters change; base frozen
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(ids).logits.pow(2).mean().backward()
    opt.step()
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all(".lora_" in k for k in changed)
    out_on = model(ids).logits.detach()
    with adapter_disabled(model):
        out_off = model(ids).logits.detach()
    assert torch.allclose(out_off, ref, atol=1e-6) and not torch.allclose(out_on, ref, atol=1e-4)
    assert torch.allclose(model(ids).logits, out_on)  # re-enabled afterwards
    # merge is exact
    sd = be.state_dict(model)
    assert all(".lora_" in k for k in sd)
    merged = be.merge(model)
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert torch.allclose(merged(ids).logits, out_on, atol=1e-5)


def test_lora_target_matching_and_modules_to_save():
    m = make_model(0)
    be = LoraBackend(LoraConfig(r=2, target_modules=["q_proj", "v_proj"], modules_to_save=["lm_head"]), prefer_builtin=True)
    names = be.apply(m)
    assert sorted({n.split(".")[-1] for n in names}) == ["q_proj", "v_proj"]
    assert m.lm_head.weight.requires_grad
    assert "lm_head.weight" in be.state_dict(m)
    with pytest.raises(ConfigError):
        LoraBackend(LoraConfig(target_modules=["nonexistent"]), prefer_builtin=True).apply(make_model(0))
    m2 = make_model(0)
    n_all = LoraBackend(LoraConfig(target_modules=["all-linear"], exclude_modules=["lm_head"]), prefer_builtin=True).apply(m2)
    assert len(n_all) == 21


def test_forge_models_lora_backend_if_available(tmp_path):
    pytest.importorskip("forge.models.lora")
    ids = torch.randint(0, 100, (2, 8))
    model, be, names = _lora_model(kind="forge", builtin=False)
    if be.impl != "forge":
        pytest.skip("forge.models.lora not usable")
    assert names
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.2)
    for _ in range(2):
        model(ids).logits.pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    out = model(ids).logits.detach()
    d = be.save_adapter(model, tmp_path / "ad", base_model="tiny")
    assert (d / "adapter_config.json").exists()
    sd = be.state_dict(model)
    merged = be.merge(model)
    assert torch.allclose(merged(ids).logits, out, atol=1e-4)
    assert all(".lora_" in k for k in sd)


def test_merge_adapter_state_into_plain_model():
    model, be, _ = _lora_model(builtin=True)
    ids = torch.randint(0, 100, (2, 8))
    for p in model.parameters():
        if p.requires_grad:
            p.data.add_(0.05 * torch.randn_like(p))
    ref = model(ids).logits.detach()
    sd = be.state_dict(model)
    fresh = make_model(0)
    n = merge_adapter_state(fresh, sd, be.scalings)
    assert n == 21 and torch.allclose(fresh(ids).logits, ref, atol=1e-4)


def test_lora_on_quantized_base_trains_and_merges_into_fresh():
    ids = torch.randint(0, 100, (2, 8))
    model = make_model(0)
    quantize_base_linears(model, QLoraConfig(bits=4, block_size=16), torch.float32)
    be = LoraBackend(LoraConfig(r=4, alpha=8), prefer_builtin=True)
    be.apply(model)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.2)
    first = None
    for _ in range(5):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(ids).logits.view(-1, 288), ids.view(-1))
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first
    sd = be.state_dict(model)
    assert sd and all(".lora_" in k for k in sd)


# ----------------------------------------------------------------------------------------------- end-to-end
def _ft_cfg(tmp_path, **over):
    sft = write_sft(tmp_path)
    cfg = {
        "name": "ft", "seed": 1, "model": {"source": "tiny", "tokenizer": "byte"}, "method": "lora",
        "lora": {"r": 4, "alpha": 8, "use_forge_lora": True},
        "micro_batch_size": 2, "grad_accum_steps": 2,
        "data": {"train_files": [str(sft)], "val_fraction": 0.2, "max_seq_len": 48},
        "optim": {"lr": 3e-3}, "schedule": {"name": "cosine", "warmup_steps": 1, "min_lr_ratio": 0.1},
        "budget": {"max_steps": 8}, "checkpoint": {"every_steps": 0}, "eval": {"every_steps": 4, "num_batches": 2},
        "logging": {"log_every": 2}, "final_dtype": "float32", "merged_dtype": "float32",
    }
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    return FinetuneConfig.model_validate(cfg)


def _store_ctx(tmp_path):
    from forge.store import ArtifactStore

    return NullContext(run_dir=tmp_path / "run", store=ArtifactStore(tmp_path / "store"))


def _load_logits(model_dir, ids):
    from forge.models.loader import load_model

    m, _ = load_model(model_dir, dtype="float32", device="cpu", with_tokenizer=False)
    with torch.no_grad():
        return m(ids).logits


def test_lora_finetune_end_to_end_adapter_and_merged_export(tmp_path):
    cfg = _ft_cfg(tmp_path)
    ctx = _store_ctx(tmp_path)
    res = run_finetune(cfg, ctx)
    assert res["status"] == "completed" and res["method"] == "lora" and res["steps"] == 8
    losses = [v for _, v in ctx.metric_series("loss")]
    assert losses[-1] < losses[0]
    assert any(m.startswith("LoRA (forge)") for m in ctx.logs)  # forge.models.lora is used when importable
    assert ctx.metric_series("eval_loss")
    _ad, md = ctx.outputs["adapter_dir"], ctx.outputs["model_dir"]
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").exists() and (tmp_path / "run" / "adapter").exists()
    assert ctx.outputs["adapter_hash"] and ctx.outputs["model_hash"] and ctx.store.has(ctx.outputs["model_hash"])
    lineage = json.loads((tmp_path / "run" / "adapter" / "forge_lineage.json").read_text())
    assert lineage["method"] == "lora" and lineage["config_hash"] and lineage["tokens_seen"] > 0
    # merged export is a loadable Forge model that differs from the (identically seeded) base
    ids = torch.randint(0, 200, (2, 12))
    merged = _load_logits(md, ids)
    from forge.models.tiny import build_tiny_model

    base = build_tiny_model(1, vocab_size=272)
    assert torch.isfinite(merged).all() and not torch.allclose(merged, base(ids).logits.detach(), atol=1e-3)
    assert (tmp_path / "run" / "checkpoints").exists()


def test_lora_checkpoint_stores_only_adapter_and_resume_is_exact(tmp_path):
    from forge.train.checkpoint import CheckpointManager

    kw = dict(schedule={"total_steps": 8})
    c_full = _ft_cfg(tmp_path / "a", **kw, budget={"max_steps": 8}, save_merged=False, save_adapter=False)
    ctx_a = NullContext(run_dir=tmp_path / "ra")
    run_finetune(c_full, ctx_a)
    ck = str(tmp_path / "ck")
    c1 = _ft_cfg(tmp_path / "b", **kw, budget={"max_steps": 4}, checkpoint={"dir": ck, "every_steps": 0}, save_merged=False, save_adapter=False)
    run_finetune(c1, NullContext(run_dir=tmp_path / "rb1"))
    state, _ = CheckpointManager(ck, c1.checkpoint).load("latest")
    assert state["model"] and all(".lora_" in k for k in state["model"])  # adapter only, not the frozen base
    c2 = _ft_cfg(tmp_path / "b", **kw, budget={"max_steps": 8}, checkpoint={"dir": ck, "resume": "auto"}, save_merged=False, save_adapter=False)
    ctx_b = NullContext(run_dir=tmp_path / "rb2")
    run_finetune(c2, ctx_b)
    fa = CheckpointManager(tmp_path / "ra" / "checkpoints", c_full.checkpoint).load("latest")[0]["model"]
    fb = CheckpointManager(ck, c2.checkpoint).load("latest")[0]["model"]
    assert fa.keys() == fb.keys() and all(torch.equal(fa[k], fb[k]) for k in fa)


def test_qlora_finetune_end_to_end(tmp_path):
    cfg = _ft_cfg(tmp_path, method="qlora", qlora={"bits": 4, "block_size": 16})
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_finetune(cfg, ctx)
    assert res["method"] == "qlora" and res["status"] == "completed"
    assert any("QLoRA: quantised" in m for m in ctx.logs)
    losses = [v for _, v in ctx.metric_series("loss")]
    assert losses[-1] < losses[0]
    assert (tmp_path / "run" / "adapter" / "adapter_model.safetensors").exists()
    md = ctx.outputs["model_dir"]  # merged into a fresh full-precision base
    out = _load_logits(md, torch.randint(0, 200, (1, 10)))
    assert torch.isfinite(out).all()


def test_full_finetune_with_freeze_patterns(tmp_path):
    cfg = _ft_cfg(tmp_path, method="full", freeze_patterns=["embed_tokens"], optim={"lr": 1e-3}, budget={"max_steps": 4})
    ctx = _store_ctx(tmp_path)
    res = run_finetune(cfg, ctx)
    assert res["method"] == "full" and res["status"] == "completed"
    from forge.models.tiny import build_tiny_model

    base = build_tiny_model(1, vocab_size=272)
    from forge.models.loader import load_model

    m, _ = load_model(ctx.outputs["model_dir"], dtype="float32", device="cpu", with_tokenizer=False)
    assert torch.equal(m.embed_tokens.weight, base.embed_tokens.weight)  # frozen
    assert not torch.equal(m.layers[0].self_attn.q_proj.weight, base.layers[0].self_attn.q_proj.weight)
    assert ctx.outputs["model_hash"]


def _pref_file(tmp_path, n=16):
    p = tmp_path / "pref.jsonl"
    p.write_text("\n".join(json.dumps({"prompt": f"q{i % 3}", "chosen": "good answer", "rejected": "bad"}) for i in range(n)), encoding="utf-8")
    return p


@pytest.mark.parametrize("method", ["lora", "full"])
def test_dpo_finetune(tmp_path, method):
    pref = _pref_file(tmp_path)
    cfg = _ft_cfg(tmp_path, method=method, preference={"enabled": True, "beta": 0.2, "sft_weight": 0.1},
                  data={"train_files": [str(pref)], "val_fraction": 0.2, "max_seq_len": 48}, optim={"lr": 2e-3 if method == "full" else 1e-2},
                  budget={"max_steps": 8}, save_merged=False)
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_finetune(cfg, ctx)
    assert res["status"] == "completed"
    losses = [v for _, v in ctx.metric_series("loss")]
    assert losses[0] == pytest.approx(math.log(2) + 0.1 * 0.0, abs=2.0)  # policy == reference at step 0
    assert losses[-1] < losses[0]
    assert ctx.metric_series("reward_acc") and ctx.metric_series("reward_margin")
    assert ctx.metric_series("reward_acc")[-1][1] >= 0.5


def test_preference_config_errors(tmp_path):
    pref = _pref_file(tmp_path)
    cfg = _ft_cfg(tmp_path, data={"train_files": [str(pref)], "max_seq_len": 48})
    with pytest.raises(ConfigError, match="preference.enabled"):
        run_finetune(cfg, NullContext(run_dir=tmp_path / "r1"))
    cfg2 = _ft_cfg(tmp_path, preference={"enabled": True})
    with pytest.raises(ConfigError, match="chosen/rejected"):
        run_finetune(cfg2, NullContext(run_dir=tmp_path / "r2"))
    with pytest.raises(ValidationError):
        FinetuneConfig.model_validate({"method": "qlora", "preference": {"enabled": True, "reference": "frozen_copy"}, "budget": {"max_steps": 1}})
