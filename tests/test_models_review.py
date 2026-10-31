"""Adversarial-review regression tests for forge.models (each failed before the corresponding fix)."""
from __future__ import annotations

import json

import pytest
import torch

from forge.core.errors import ConfigError
from forge.models.config import ModelConfig
from forge.models.loader import checkpoint_files, load_model, save_model
from forge.models.lora import add_lora, merge_lora
from forge.models.modeling import ForgeLM
from forge.models.tiny import build_tiny_model, make_tiny_config


def _gemma_hf(**over):
    d = dict(model_type="gemma", vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6, max_position_embeddings=128, hidden_activation="gelu_pytorch_tanh")
    d.update(over)
    return d


def test_gemma_defaults_to_tied_embeddings_when_config_is_silent():
    # HF GemmaConfig defaults tie_word_embeddings=True and the released checkpoints ship no lm_head.weight
    assert ModelConfig.from_hf_config(_gemma_hf()).tie_word_embeddings is True
    assert ModelConfig.from_hf_config(_gemma_hf(tie_word_embeddings=False)).tie_word_embeddings is False
    llama = dict(model_type="llama", vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4)
    assert ModelConfig.from_hf_config(llama).tie_word_embeddings is False


def test_gemma_checkpoint_without_lm_head_loads_strictly(tmp_path):
    cfg = ModelConfig.from_hf_config(_gemma_hf())
    m = ForgeLM(cfg)
    save_model(m, tmp_path)
    hf = json.loads((tmp_path / "config.json").read_text())
    hf.pop("forge")
    hf.pop("tie_word_embeddings")  # as in the original Gemma-1 config.json
    (tmp_path / "config.json").write_text(json.dumps(hf))
    m2, _ = load_model(tmp_path, dtype="float32", device="cpu", with_tokenizer=False)
    assert m2.config.tie_word_embeddings and torch.equal(m2.embed_tokens.weight, m.embed_tokens.weight)


def test_index_json_cannot_point_outside_the_model_directory(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (tmp_path / "evil.safetensors").write_bytes(b"x")
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../evil.safetensors"}}))
    with pytest.raises(ConfigError):
        checkpoint_files(d)


def test_lora_merge_preserves_logits_in_a_full_model():
    torch.manual_seed(0)
    m = build_tiny_model(0, vocab_size=64).eval()
    ids = torch.randint(0, 64, (2, 9))
    add_lora(m, r=4, alpha=8, target_modules="all-linear")
    for mod in (x for x in m.modules() if hasattr(x, "lora_B")):
        torch.nn.init.normal_(mod.lora_B.weight, std=0.05)
    before = m(ids).logits
    merge_lora(m)
    assert torch.allclose(m(ids).logits, before, atol=1e-4)


def test_config_roundtrip_through_hf_dict_is_lossless_for_moe_variants():
    for kw in ({"moe": True}, {"arch": "mixtral", "moe": {"num_experts": 4, "top_k": 2, "moe_intermediate_size": 128}}):
        c = make_tiny_config(**kw)
        c2 = ModelConfig.from_hf_config(c.to_hf_config())
        assert c2.moe is not None and c2.moe.num_experts == c.moe.num_experts and c2.moe.top_k == c.moe.top_k
        assert c2.param_count() == c.param_count()


def test_paged_decode_kernel_receives_head_major_pools_and_matches_the_reference_path():
    """The ops_attn kernel contract is head-major pools [P, Hkv, page, D]; SimplePagedPool is token-major [P, page, Hkv, D]."""
    from forge.models import AttnMeta, PagedKVView, SimplePagedPool

    m = build_tiny_model(0).eval()
    ids = torch.randint(0, 256, (2, 10))
    full = m(ids).logits
    cfg = m.config
    bs, nb = 4, 3
    pool = SimplePagedPool(cfg, num_blocks=2 * nb + 1, block_size=bs, dtype=torch.float32)
    tables = torch.arange(2 * nb).reshape(2, nb).int()
    view = PagedKVView(tables, torch.zeros(2, dtype=torch.long), pool)
    m(ids[:, :8], kv_cache=view)
    out = m(ids[:, 8:9], kv_cache=view, attn_meta=AttnMeta(use_paged_kernel=True)).logits  # real ops_attn (CPU -> oracle)
    assert all(layer.self_attn._paged_ok for layer in m.layers), "kernel path was rejected (layout mismatch)"
    assert torch.allclose(out[:, 0], full[:, 8], atol=1e-4)


def test_paged_decode_kernel_is_skipped_for_quantised_pools(monkeypatch):
    """Per-token KV scales of SimplePagedPool do not match the kernel's per-channel-K scale contract -> reference path only."""
    import sys
    import types

    from forge.models import AttnMeta, PagedKVView, SimplePagedPool

    called = {"n": 0}
    fake = types.ModuleType("forge.kernels.ops_attn")
    fake.paged_decode_attention = lambda *a, **k: called.__setitem__("n", called["n"] + 1)
    monkeypatch.setitem(sys.modules, "forge.kernels.ops_attn", fake)
    import forge.kernels as fk

    monkeypatch.setattr(fk, "ops_attn", fake, raising=False)
    m = build_tiny_model(0).eval()
    ids = torch.randint(0, 256, (1, 9))
    pool = SimplePagedPool(m.config, num_blocks=4, block_size=4, dtype=torch.float32, kv_dtype="int8")
    view = PagedKVView(torch.arange(3).reshape(1, 3).int(), torch.zeros(1, dtype=torch.long), pool)
    m(ids[:, :8], kv_cache=view)
    out = m(ids[:, 8:9], kv_cache=view, attn_meta=AttnMeta(use_paged_kernel=True)).logits
    assert called["n"] == 0 and torch.isfinite(out).all()
