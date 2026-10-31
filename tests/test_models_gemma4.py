"""Gemma 4 text decoder: config parsing, per-layer geometry, KV sharing, PLE, soft-capping, HF-name IO and (when the installed
transformers ships Gemma4) logit equivalence against ``Gemma4ForCausalLM`` on tiny random configs.  CPU only, hermetic."""
from __future__ import annotations

import pytest
import torch

from forge.core.errors import ConfigError
from forge.models import ForgeLM, ModelConfig, load_model
from forge.models.layers import compute_inv_freq
from forge.models.loader import (
    forge_state_dict_to_hf,
    hf_to_forge_name,
    load_hf_tensors,
    materialize,
    save_model,
)

T = 24
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"]


def _text_cfg(**over):
    """Tiny HF-style ``text_config`` dict (E4B-like: PLE, 3 KV-shared layers, softcap, global_head_dim)."""
    d = dict(
        model_type="gemma4_text", vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, global_head_dim=32, layer_types=list(LAYER_TYPES), sliding_window=8,
        final_logit_softcapping=30.0, hidden_size_per_layer_input=8, vocab_size_per_layer_input=128, num_kv_shared_layers=3,
        max_position_embeddings=256, hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6, tie_word_embeddings=True,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        },
    )
    d.update(over)
    return d


def _model(cfg: ModelConfig, seed: int = 0) -> ForgeLM:
    torch.manual_seed(seed)
    m = ForgeLM(cfg, dtype=torch.float32)
    with torch.no_grad():  # make every norm / layer_scalar non-trivial so a wrong formula cannot hide behind identity
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    return m.eval()


# ------------------------------------------------------------------------------------------------ config
def test_from_hf_config_nested_text_config_and_layer_geometry():
    top = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "vision_config": {}, "audio_config": {}, "text_config": _text_cfg()}
    cfg = ModelConfig.from_hf_config(top)
    assert cfg.arch == "gemma4"
    assert cfg.norm_unit_offset is False and cfg.attn_scale == 1.0 and cfg.qk_norm and cfg.v_norm and cfg.sandwich_norm and cfg.layer_scalar
    assert cfg.rope_theta == 1e6 and cfg.rope_local_theta == 1e4
    assert cfg.rope_scaling == {"rope_type": "proportional", "partial_rotary_factor": 0.25}
    assert [cfg.layer_head_dim(i) for i in range(6)] == [16, 16, 32, 16, 16, 32]
    assert [cfg.layer_window(i) for i in range(6)] == [8, 8, None, 8, 8, None]
    assert [cfg.kv_source_layer(i) for i in range(6)] == [None, None, None, 1, 1, 2]
    assert cfg.kv_layers == [0, 1, 2]
    # plain gemma4_text and the architectures-only spelling parse identically
    assert ModelConfig.from_hf_config(_text_cfg()) == cfg
    assert ModelConfig.from_hf_config({**_text_cfg(), "model_type": None, "architectures": ["Gemma4ForCausalLM"]}) == cfg
    # 12B-style wrapper alias
    with pytest.raises(ConfigError, match="unverified"):
        ModelConfig.from_hf_config({"model_type": "gemma4_unified", "text_config": _text_cfg(model_type="gemma4_unified_text")})


def test_from_hf_config_defaults_and_rejections():
    d = _text_cfg()
    del d["layer_types"]
    d["num_hidden_layers"], d["num_kv_shared_layers"] = 12, 0
    cfg = ModelConfig.from_hf_config(d)  # default 5:1 pattern, last layer global
    assert [i for i in range(12) if cfg.layer_is_full(i)] == [5, 11]
    with pytest.raises(ConfigError, match="MoE"):
        ModelConfig.from_hf_config(_text_cfg(enable_moe_block=True))
    with pytest.raises(ConfigError, match="rope_type"):
        ModelConfig.from_hf_config(_text_cfg(rope_parameters={"full_attention": {"rope_type": "yarn", "rope_theta": 1e6, "factor": 2.0}}))


def test_to_hf_config_roundtrip_and_per_layer_config_spelling():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    hf = cfg.to_hf_config()
    assert hf["model_type"] == "gemma4_text" and hf["architectures"] == ["Gemma4ForCausalLM"]
    assert ModelConfig.from_hf_config(hf) == cfg
    # transformers writes `per_layer_config` instead of `global_head_dim` back from to_dict(); both spellings must give the same config
    d = _text_cfg()
    gh = d.pop("global_head_dim")
    d["per_layer_config"] = {"2": {"head_dim": gh}, "5": {"head_dim": gh}}
    assert ModelConfig.from_hf_config(d) == cfg
    # exact config survives the forge JSON block
    assert ModelConfig.model_validate(cfg.model_dump(mode="json")) == cfg


def test_k_eq_v_geometry():
    cfg = ModelConfig.from_hf_config(
        _text_cfg(hidden_size_per_layer_input=0, num_kv_shared_layers=0, attention_k_eq_v=True, num_global_key_value_heads=1, use_double_wide_mlp=False)
    )
    assert cfg.layer_kv_heads(2) == 1 and cfg.layer_kv_heads(0) == 2
    assert not cfg.layer_has_v_proj(2) and cfg.layer_has_v_proj(0)
    m = ForgeLM(cfg, dtype=torch.float32)
    assert m.layers[2].self_attn.v_proj is None and m.layers[0].self_attn.v_proj is not None
    assert m.layers[2].self_attn.k_proj.weight.shape == (32, 64)  # 1 kv head x global_head_dim 32
    assert cfg.param_count() == m.num_parameters() + cfg.num_layers  # + layer_scalar buffers


def test_param_count_and_kv_bytes_match_instantiated_model():
    for over in ({}, {"hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0}, {"use_double_wide_mlp": True}):
        cfg = ModelConfig.from_hf_config(_text_cfg(**over))
        m = ForgeLM(cfg, dtype=torch.float32)
        # layer_scalar is a buffer, everything else is a parameter
        assert cfg.param_count() == m.num_parameters() + cfg.num_layers, over
    cfg = ModelConfig.from_hf_config(_text_cfg())
    # layers 0..2 own K/V: kv_heads 2 x (16, 16, 32) wide, K and V, bf16
    assert cfg.kv_bytes_per_token(16) == 2 * 2 * (2 * 16 + 2 * 16 + 2 * 32)
    c = cfg.flops_per_token(64, "decode")
    assert c > 0
    assert cfg.param_count(include_embeddings=False) < cfg.param_count()


def test_double_wide_mlp_only_on_shared_layers():
    cfg = ModelConfig.from_hf_config(_text_cfg(use_double_wide_mlp=True))
    m = ForgeLM(cfg, dtype=torch.float32)
    assert [m.layers[i].mlp.intermediate_size for i in range(6)] == [96, 96, 96, 192, 192, 192]


# ------------------------------------------------------------------------------------------------ rope
def test_proportional_rope_inv_freq():
    inv, att = compute_inv_freq(32, 1e6, {"rope_type": "proportional", "partial_rotary_factor": 0.25}, 256)
    assert att == 1.0 and inv.shape == (16,)
    rot = 1.0 / (1e6 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 32))  # 4 rotated angles, exponent over the FULL head_dim
    assert torch.allclose(inv[:4], rot) and torch.count_nonzero(inv[4:]) == 0


# ------------------------------------------------------------------------------------------------ forward behaviour
def test_logit_softcap_and_loss_path():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    m = _model(cfg)
    ids = torch.randint(0, 128, (2, T))
    with torch.no_grad():
        out = m(ids, return_hidden=True)
        raw = m.lm_head(out.hidden_states)
        assert out.logits.abs().max() <= 30.0
        assert torch.allclose(out.logits, 30.0 * torch.tanh(raw / 30.0), atol=1e-5)
        labels = ids.clone()
        loss = m(ids, labels=labels).loss
        ref = torch.nn.functional.cross_entropy((30.0 * torch.tanh(raw / 30.0))[:, :-1].reshape(-1, 128), labels[:, 1:].reshape(-1))
        assert torch.allclose(loss, ref, atol=1e-5)
        m.loss_chunk_size = 7
        assert torch.allclose(m(ids, labels=labels).loss, ref, atol=1e-5)


def test_kv_cache_prefill_decode_matches_full_forward():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    m = _model(cfg)
    ids = torch.randint(0, 128, (2, T))
    with torch.no_grad():
        full = m(ids).logits
        cache = m.new_kv_cache(2, T)
        assert cache.k[3].numel() == 0 and cache.k[2].shape[-2:] == (2, 32) and cache.k[0].shape[-2:] == (2, 16)  # shared layers own no cache
        a = m(ids[:, :10], kv_cache=cache).logits
        b = m(ids[:, 10:17], kv_cache=cache).logits
        steps = [m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(17, T)]
    got = torch.cat([a, b, *steps], dim=1)
    assert (got - full).abs().max() < 2e-4
    # sliding window really matters in this setup (window 8 < T): shrinking it changes the output
    cfg2 = cfg.model_copy(update={"sliding_window": 3})
    m2 = ForgeLM(cfg2, dtype=torch.float32)
    m2.load_state_dict(m.state_dict())
    assert (m2.eval()(ids).logits - full).abs().max() > 1e-3


def test_kv_cache_matches_with_quantised_dtype_shapes():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    m = _model(cfg)
    cache = m.new_kv_cache(1, 16, kv_dtype="int8")
    with torch.no_grad():
        m(torch.randint(0, 128, (1, 10)), kv_cache=cache)
        out = m(torch.randint(0, 128, (1, 1)), kv_cache=cache).logits
    assert torch.isfinite(out).all()


def test_generate_runs():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    m = _model(cfg)
    seq = m.generate(torch.randint(0, 128, (1, 5)), max_new_tokens=4)
    assert seq.shape == (1, 9)


def test_gradient_checkpointing_with_kv_sharing_trains():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    m = _model(cfg).train()
    m.gradient_checkpointing_enable()
    ids = torch.randint(0, 128, (2, 12))
    loss = m(ids, labels=ids).loss
    loss.backward()
    assert m.layers[0].self_attn.q_proj.weight.grad is not None and m.embed_tokens_per_layer.weight.grad is not None


# ------------------------------------------------------------------------------------------------ names / IO
def _multimodal_state_dict(model: ForgeLM) -> dict[str, torch.Tensor]:
    """What a published ``Gemma4ForConditionalGeneration`` safetensors looks like for this model (+ junk the loader must ignore)."""
    cfg = model.config
    sd = {"model.language_model." + k[len("model.") :]: v for k, v in forge_state_dict_to_hf(model).items()}
    sd["model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"] = torch.zeros(4, 4)
    sd["model.audio_tower.layers.0.norm.weight"] = torch.zeros(4)
    sd["model.embed_vision.embedding_projection.weight"] = torch.zeros(4, 4)
    sd["model.embed_audio.embedding_projection.weight"] = torch.zeros(4, 4)
    for i in range(cfg.num_layers):
        if cfg.is_kv_shared_layer(i):  # shipped by the real checkpoints, dropped by HF
            hd = cfg.layer_head_dim(i)
            pre = f"model.language_model.layers.{i}.self_attn."
            sd[pre + "k_proj.weight"] = torch.randn(cfg.layer_kv_dim(i), cfg.hidden_size)
            sd[pre + "v_proj.weight"] = torch.randn(cfg.layer_kv_dim(i), cfg.hidden_size)
            sd[pre + "k_norm.weight"] = torch.randn(hd)
    return sd


def test_name_mapping_language_model_prefix_and_ignored_tensors():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    f = lambda n: hf_to_forge_name(n, cfg)  # noqa: E731
    assert f("model.language_model.embed_tokens.weight") == "embed_tokens.weight"
    assert f("model.language_model.embed_tokens_per_layer.weight") == "embed_tokens_per_layer.weight"
    assert f("model.language_model.per_layer_model_projection.weight") == "per_layer_model_projection.weight"
    assert f("model.language_model.per_layer_projection_norm.weight") == "per_layer_projection_norm.weight"
    assert f("model.language_model.norm.weight") == "norm.weight"
    assert f("model.language_model.layers.2.layer_scalar") == "layers.2.layer_scalar"
    assert f("model.language_model.layers.0.self_attn.k_proj.weight") == "layers.0.self_attn.k_proj.weight"
    assert f("model.language_model.layers.4.self_attn.k_proj.weight") is None  # KV-shared layer
    assert f("model.language_model.layers.4.self_attn.q_proj.weight") == "layers.4.self_attn.q_proj.weight"
    for junk in ("model.vision_tower.x.weight", "model.audio_tower.layers.0.w", "model.embed_vision.embedding_projection.weight", "model.embed_audio.embedding_projection.weight"):
        assert f(junk) is None
    # plain text-only layout keeps working
    assert f("model.layers.0.input_layernorm.weight") == "layers.0.input_layernorm.weight"


def test_load_multimodal_checkpoint_names_strictly():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    src = _model(cfg, seed=1)
    dst = ForgeLM(cfg, dtype=torch.float32, device="meta", init_weights=False)
    materialize(dst, "cpu", init=False)
    rep = load_hf_tensors(dst, iter(_multimodal_state_dict(src).items()))
    assert not rep.missing and not rep.unexpected
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        assert torch.equal(src(ids).logits, dst.eval()(ids).logits)


def test_save_load_roundtrip(tmp_path):
    cfg = ModelConfig.from_hf_config(_text_cfg())
    m = _model(cfg, seed=2)
    save_model(m, tmp_path)
    m2, _ = load_model(tmp_path, dtype="float32", device="cpu", with_tokenizer=False)
    assert m2.config == cfg
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        assert torch.equal(m(ids).logits, m2(ids).logits)
    from safetensors import safe_open

    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as sf:
        names = set(sf.keys())
    assert "model.layers.0.layer_scalar" in names and "model.embed_tokens_per_layer.weight" in names
    assert "model.layers.4.self_attn.k_proj.weight" not in names and "lm_head.weight" not in names


# ------------------------------------------------------------------------------------------------ HF equivalence
transformers = pytest.importorskip("transformers")
Gemma4TextConfig = getattr(transformers, "Gemma4TextConfig", None)
needs_g4 = pytest.mark.skipif(Gemma4TextConfig is None, reason="installed transformers has no Gemma4")

_VARIANTS = {
    "e4b_like": {},
    "e2b_double_wide": {"use_double_wide_mlp": True, "num_key_value_heads": 1},
    "no_ple_no_share": {"hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0},
    "k_eq_v_31b_like": {"hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "attention_k_eq_v": True, "num_global_key_value_heads": 1},
    "long_window": {"sliding_window": 64},
    "no_softcap": {"final_logit_softcapping": None},
}


def _hf_pair(over):
    d = _text_cfg(**over)
    d.pop("model_type")
    hf_cfg = Gemma4TextConfig(**d)
    torch.manual_seed(0)
    hf = transformers.Gemma4ForCausalLM(hf_cfg).float().eval()
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
        for n, b in hf.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    cfg = ModelConfig.from_hf_config(hf.config.to_dict())
    model = ForgeLM(cfg, dtype=torch.float32, device="meta", init_weights=False)
    materialize(model, "cpu", init=False)
    rep = load_hf_tensors(model, iter(hf.state_dict().items()))
    assert not rep.missing, rep.missing
    return hf, model.eval(), cfg


@needs_g4
@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_logits_match_transformers_gemma4(variant):
    hf, model, cfg = _hf_pair(_VARIANTS[variant])
    ids = torch.randint(0, 128, (2, T))
    with torch.no_grad():
        ref = hf(ids).logits
        out = model(ids).logits
        err = (ref - out).abs().max().item()
        assert err < 3e-4, f"{variant}: max abs logit diff {err}"
        # incremental decode through the Forge cache agrees with HF's full forward too
        cache = model.new_kv_cache(2, T)
        inc = torch.cat([model(ids[:, :9], kv_cache=cache).logits] + [model(ids[:, i : i + 1], kv_cache=cache).logits for i in range(9, T)], dim=1)
        assert (inc - ref).abs().max().item() < 3e-4


@needs_g4
def test_loss_matches_transformers_gemma4():
    hf, model, _ = _hf_pair({})
    ids = torch.randint(0, 128, (2, T))
    with torch.no_grad():
        assert abs(hf(ids, labels=ids).loss.item() - model(ids, labels=ids).loss.item()) < 2e-4


@needs_g4
def test_saved_directory_loads_in_transformers(tmp_path):
    cfg = ModelConfig.from_hf_config(_text_cfg())
    m = _model(cfg, seed=3)
    save_model(m, tmp_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32).eval()
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        assert (hf(ids).logits - m(ids).logits).abs().max().item() < 3e-4
