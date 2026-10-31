"""Logit equivalence against Hugging Face reference implementations on tiny random configs (skipped without transformers)."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from forge.models import ForgeLM, ModelConfig  # noqa: E402
from forge.models.loader import load_hf_tensors, materialize  # noqa: E402

T = 40


def _build_pair(hf_cfg):
    torch.manual_seed(0)
    hf = transformers.AutoModelForCausalLM.from_config(hf_cfg).float().eval()
    # randomise norms so the check is not trivially identity
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
    cfg = ModelConfig.from_hf_config(hf.config.to_dict())
    model = ForgeLM(cfg, dtype=torch.float32, device="meta", init_weights=False)
    materialize(model, "cpu", init=False)
    sd = hf.state_dict()
    rep = load_hf_tensors(model, iter(sd.items()))
    assert not rep.missing, rep.missing
    model.eval()
    return hf, model, cfg


def _compare(hf, model, vocab, atol=2e-4):
    ids = torch.randint(0, vocab, (2, T))
    with torch.no_grad():
        ref = hf(ids).logits
        out = model(ids).logits
    err = (ref - out).abs().max().item()
    assert err < atol, f"max abs logit diff {err}"
    return ids


def test_llama_gqa_rope_llama3():
    c = transformers.LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rope_theta=10000.0,
        rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 32},
    )
    hf, model, cfg = _build_pair(c)
    assert cfg.rope_scaling and cfg.rope_scaling["rope_type"] == "llama3"
    _compare(hf, model, 128)


@pytest.mark.parametrize("scaling", [{"rope_type": "linear", "factor": 4.0}, {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}])
def test_llama_rope_variants(scaling):
    c = transformers.LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, rope_theta=500000.0, rope_scaling=scaling, tie_word_embeddings=True,
    )
    hf, model, cfg = _build_pair(c)
    assert cfg.tie_word_embeddings
    _compare(hf, model, 128)


def test_qwen2_bias():
    c = transformers.Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=True,
    )
    hf, model, cfg = _build_pair(c)
    assert cfg.attention_bias and not cfg.o_proj_bias
    _compare(hf, model, 128)


def test_qwen3_qknorm_headdim():
    c = transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=24, max_position_embeddings=256,
    )
    hf, model, cfg = _build_pair(c)
    assert cfg.qk_norm and cfg.head_dim == 24
    _compare(hf, model, 128)


def test_mistral_sliding_window():
    c = transformers.MistralConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, sliding_window=8,
    )
    hf, model, cfg = _build_pair(c)
    assert cfg.sliding_window == 8
    _compare(hf, model, 128)


def test_gemma():
    c = transformers.GemmaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=256,
    )
    hf, model, cfg = _build_pair(c)
    assert cfg.norm_unit_offset and cfg.scale_embeddings
    _compare(hf, model, 128)


@pytest.mark.parametrize("kind", ["qwen3_moe", "mixtral", "qwen2_moe"])
def test_moe_models(kind):
    common = dict(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    if kind == "qwen3_moe":
        c = transformers.Qwen3MoeConfig(**common, intermediate_size=96, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, head_dim=16)
    elif kind == "mixtral":
        c = transformers.MixtralConfig(**common, intermediate_size=32, num_local_experts=4, num_experts_per_tok=2)
    else:
        c = transformers.Qwen2MoeConfig(**common, intermediate_size=96, moe_intermediate_size=32, shared_expert_intermediate_size=48, num_experts=4, num_experts_per_tok=2, norm_topk_prob=False)
    hf, model, cfg = _build_pair(c)
    assert cfg.moe is not None and cfg.moe.num_experts == 4
    _compare(hf, model, 128, atol=5e-4)


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe", "mixtral"])
def test_load_real_hf_checkpoint_dir(tmp_path, kind):
    """Round trip through ``save_pretrained`` on disk (whatever layout this transformers version writes) -> forge loader."""
    from forge.models import load_model

    common = dict(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    if kind == "llama":
        c = transformers.LlamaConfig(**common, intermediate_size=96)
    elif kind == "qwen3_moe":
        c = transformers.Qwen3MoeConfig(**common, intermediate_size=96, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2, head_dim=16)
    else:
        c = transformers.MixtralConfig(**common, intermediate_size=32, num_local_experts=4, num_experts_per_tok=2)
    torch.manual_seed(0)
    hf = transformers.AutoModelForCausalLM.from_config(c).float().eval()
    hf.save_pretrained(str(tmp_path / "hf"))
    model, _ = load_model(tmp_path / "hf", dtype="float32", device="cpu", with_tokenizer=False)
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        assert (hf(ids).logits - model(ids).logits).abs().max().item() < 5e-4
    hf2, _ = load_model(tmp_path / "hf", dtype="float32", device="cpu", backend="hf", with_tokenizer=False)
    assert type(hf2).__name__.endswith("ForCausalLM")
