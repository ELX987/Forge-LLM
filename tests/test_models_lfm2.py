"""Liquid LFM2 / LFM2-MoE (hybrid gated short-conv + attention): config + accounting, the ShortConv recurrence (full sequence vs
chunked vs token-by-token), conv-state cache handling (dense, paged, ragged padding), HF name IO, logit equivalence against
transformers' Lfm2 / Lfm2Moe (skipped when not installed), the paged serving engine, planner / capacity and compression guards.
CPU only, hermetic (no downloads)."""
from __future__ import annotations

import pytest
import torch

from forge.core.errors import ConfigError
from forge.models import ForgeLM, ModelConfig, load_model
from forge.models.attention import AttnMeta
from forge.models.kvcache import DenseKVCache, PagedKVView, SimplePagedPool
from forge.models.loader import (
    forge_state_dict_to_hf,
    forge_to_hf_name,
    hf_to_forge_name,
    load_hf_tensors,
    materialize,
    save_model,
)
from forge.models.moe import route_sigmoid

LT5 = ["conv", "full_attention", "conv", "full_attention", "conv"]


def _hf_dict(**over):
    """Tiny HF-style ``Lfm2Config`` dict (legacy block_* keys included, as on the hub)."""
    d = dict(
        model_type="lfm2", vocab_size=128, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
        block_ff_dim=96, block_auto_adjust_ff_dim=False, norm_eps=1e-5, conv_L_cache=3, conv_bias=False, tie_embedding=True,
        layer_types=list(LT5), max_position_embeddings=256, rope_theta=1000000.0,
    )
    d.update(over)
    return d


def _moe_hf_dict(**over):
    d = _hf_dict(model_type="lfm2_moe", num_experts=4, num_experts_per_tok=2, moe_intermediate_size=24, num_dense_layers=2,
                 use_expert_bias=True, routed_scaling_factor=1.5, norm_topk_prob=True, intermediate_size=64)
    d.pop("block_ff_dim"), d.pop("block_auto_adjust_ff_dim")
    d.update(over)
    return d


def _model(cfg: ModelConfig, seed: int = 0) -> ForgeLM:
    torch.manual_seed(seed)
    m = ForgeLM(cfg, dtype=torch.float32)
    with torch.no_grad():  # non-trivial norms / conv taps / expert bias so a wrong formula cannot hide behind identity
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
        for n, b in m.named_buffers():
            if n.endswith("expert_bias"):
                b.copy_(torch.randn_like(b) * 0.3)
    return m.eval()


def _cfg(**over) -> ModelConfig:
    return ModelConfig.from_hf_config(_hf_dict(**over))


# ------------------------------------------------------------------------------------------------ config + accounting
CFG_350M = dict(
    model_type="lfm2", vocab_size=65536, hidden_size=1024, num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=8,
    block_ff_dim=6656, block_auto_adjust_ff_dim=True, block_ffn_dim_multiplier=1.0, block_multiple_of=256, norm_eps=1e-5,
    conv_L_cache=3, conv_bias=False, tie_embedding=True, full_attn_idxs=[2, 5, 8, 10, 12, 14], max_position_embeddings=128000,
    rope_theta=1000000.0,
)


def test_350m_analytic_numbers():
    c = ModelConfig.from_hf_config(CFG_350M)
    assert c.arch == "lfm2" and c.tie_word_embeddings and c.qk_norm and c.head_dim == 64
    assert c.intermediate_size == 4608  # 2/3 of block_ff_dim 6656 rounded up to a multiple of 256
    assert c.param_count() == 354_483_968
    assert c.attn_layer_ids == [2, 5, 8, 10, 12, 14] and len(c.conv_layer_ids) == 10 and c.is_hybrid
    assert c.kv_layers == c.attn_layer_ids
    assert c.kv_bytes_per_token() == 2 * 6 * 512 * 2  # only the 6 attention layers cache K/V (12 KiB, not 32 KiB)
    assert c.recurrent_state_bytes_per_seq() == 10 * 1024 * 2 * 2 == 40_960
    assert c.recurrent_state_bytes_per_seq(4) == 81_920
    # same plan via layer_types spelled the HF-new way
    lt = ["full_attention" if i in (2, 5, 8, 10, 12, 14) else "conv" for i in range(16)]
    d2 = {k: v for k, v in CFG_350M.items() if k != "full_attn_idxs"}
    assert ModelConfig.from_hf_config({**d2, "layer_types": lt}) == c


def test_mlp_width_variants():
    d = dict(CFG_350M, hidden_size=1536, num_attention_heads=24, block_ff_dim=10240)
    assert ModelConfig.from_hf_config(d).intermediate_size == 6912  # 700M
    d = dict(CFG_350M, hidden_size=2048, num_attention_heads=32, block_ff_dim=12288)
    assert ModelConfig.from_hf_config(d).intermediate_size == 8192  # 1.2B
    d = dict(CFG_350M, hidden_size=2048, num_attention_heads=32, block_ff_dim=10752, block_auto_adjust_ff_dim=False, num_hidden_layers=30)
    d.pop("full_attn_idxs")
    d["layer_types"] = ["full_attention" if i in (2, 5, 9, 13, 17, 21, 24, 27) else "conv" for i in range(30)]
    c = ModelConfig.from_hf_config(d)  # 2.6B: no auto adjust
    assert c.intermediate_size == 10752 and 2.5e9 < c.param_count() < 2.7e9
    d = dict(CFG_350M, block_ffn_dim_multiplier=None)  # round-up only happens with a multiplier
    assert ModelConfig.from_hf_config(d).intermediate_size == int(2 * 6656 / 3)


def test_vl_text_config_unwrap_and_defaults():
    inner = {k: v for k, v in CFG_350M.items() if k not in ("tie_embedding",)}
    c = ModelConfig.from_hf_config({"model_type": "lfm2_vl", "architectures": ["Lfm2VlForConditionalGeneration"], "text_config": inner})
    assert c.arch == "lfm2" and c.tie_word_embeddings is True  # text_config.tie_word_embeddings absent => tied
    # architectures-only spelling
    assert ModelConfig.from_hf_config({**CFG_350M, "model_type": None, "architectures": ["Lfm2ForCausalLM"]}) == ModelConfig.from_hf_config(CFG_350M)


def test_hf_config_round_trip_and_validation():
    for c in (_cfg(), ModelConfig.from_hf_config(_moe_hf_dict()), ModelConfig.from_hf_config(CFG_350M)):
        back = ModelConfig.from_hf_config(c.to_hf_config())
        assert back == c
        assert c.to_hf_config()["layer_types"] == ["conv" if c.is_conv_layer(i) else "full_attention" for i in range(c.num_layers)]
    with pytest.raises(ValueError):
        ModelConfig(arch="llama", layer_types=["conv"] * 32)  # conv only exists for lfm2*
    with pytest.raises(ValueError):
        ModelConfig(arch="lfm2", num_layers=3, layer_types=["conv", "full_attention"])
    with pytest.raises(ConfigError):
        ModelConfig.from_hf_config(_hf_dict(layer_types=["conv"]))


def test_non_hybrid_accounting_unchanged():
    c = ModelConfig(arch="llama", hidden_size=64, intermediate_size=128, num_layers=4, num_heads=4, num_kv_heads=2, vocab_size=100)
    assert not c.is_hybrid and c.recurrent_state_bytes_per_seq() == 0 and c.kv_layers == [0, 1, 2, 3]
    assert c.kv_bytes_per_token() == 2 * 4 * 2 * 16 * 2


@pytest.mark.parametrize("hf", [lambda: _hf_dict(), lambda: _hf_dict(conv_bias=True), lambda: _moe_hf_dict()], ids=["dense", "bias", "moe"])
def test_param_count_matches_instantiated_model(hf):
    cfg = ModelConfig.from_hf_config(hf())
    m = ForgeLM(cfg, dtype=torch.float32)
    assert m.num_parameters() == cfg.param_count()  # expert_bias is a buffer, not a parameter


def test_flops_skip_conv_layers_for_attention_term():
    c = _cfg()
    f_short = c.flops_per_token(0, "decode")
    f_long = c.flops_per_token(1000, "decode")
    assert f_long - f_short == pytest.approx(4.0 * len(c.attn_layer_ids) * c.q_dim * 1000)


# ------------------------------------------------------------------------------------------------ ShortConv recurrence
def test_full_sequence_vs_token_by_token_decode():
    cfg = _cfg()
    m = _model(cfg)
    ids = torch.randint(0, 128, (3, 13))
    with torch.no_grad():
        full = m(ids).logits
        cache = m.new_kv_cache(3, 16)
        steps = torch.cat([m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(13)], dim=1)
    assert (full - steps).abs().max() < 1e-4
    assert cache.get_conv(0).shape == (3, 32, 2) and cache.get_conv(1 - 1).abs().sum() > 0


def test_chunked_prefill_then_decode_matches_full():
    cfg = _cfg()
    m = _model(cfg)
    ids = torch.randint(0, 128, (2, 17))
    with torch.no_grad():
        full = m(ids).logits
        cache = m.new_kv_cache(2, 24)
        parts = [m(ids[:, a:b], kv_cache=cache).logits for a, b in ((0, 1), (1, 6), (6, 13), (13, 14), (14, 17))]  # chunks shorter than K-1 too
    assert (full - torch.cat(parts, 1)).abs().max() < 1e-4


def test_right_padded_ragged_batch_matches_individual_runs():
    cfg = _cfg()
    m = _model(cfg)
    la, lb = 9, 4
    a, b = torch.randint(0, 128, (1, la)), torch.randint(0, 128, (1, lb))
    ids = torch.zeros(2, la, dtype=torch.long)
    ids[0], ids[1, :lb] = a[0], b[0]
    am = torch.zeros(2, la, dtype=torch.long)
    am[0], am[1, :lb] = 1, 1
    with torch.no_grad():
        cache = m.new_kv_cache(2, 16)
        out = m(ids, kv_cache=cache, attn_meta=AttnMeta(attention_mask=am)).logits
        ra, rb = m(a).logits, m(b).logits
        assert (out[0] - ra[0]).abs().max() < 1e-4 and (out[1, :lb] - rb[0]).abs().max() < 1e-4
        # decode one more token per row from the padded-prefill state
        nxt = torch.randint(0, 128, (2, 1))
        step = m(nxt, kv_cache=cache).logits
        ra2 = m(torch.cat([a, nxt[:1]], 1)).logits[:, -1]
        rb2 = m(torch.cat([b, nxt[1:]], 1)).logits[:, -1]
    assert (step[0, 0] - ra2[0]).abs().max() < 1e-4 and (step[1, 0] - rb2[0]).abs().max() < 1e-4


def test_cache_less_attention_mask_zeroes_pads():
    cfg = _cfg()
    m = _model(cfg)
    ids = torch.randint(0, 128, (1, 6))
    am = torch.tensor([[1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        padded = m(ids, attn_meta=AttnMeta(attention_mask=am)).logits
        plain = m(ids[:, :4]).logits
    assert (padded[:, :4] - plain).abs().max() < 1e-4


def test_conv_kernel_variants_and_bias():
    for kernel, bias in ((1, False), (2, False), (4, True)):
        cfg = _cfg(conv_L_cache=kernel, conv_bias=bias)
        m = _model(cfg)
        if bias:
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n.endswith("conv.conv.bias") or n.endswith("in_proj.bias"):
                        p.normal_(0, 0.2)
        ids = torch.randint(0, 128, (2, 9))
        with torch.no_grad():
            full = m(ids).logits
            cache = m.new_kv_cache(2, 12)
            steps = torch.cat([m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(9)], 1)
        assert (full - steps).abs().max() < 1e-4, (kernel, bias)


def test_generate_runs_and_gradients_flow_through_conv():
    cfg = _cfg()
    m = _model(cfg)
    out = m.generate(torch.randint(0, 128, (2, 5)), max_new_tokens=6)
    assert out.shape == (2, 11)
    m.train()
    ids = torch.randint(0, 128, (2, 8))
    m(ids, labels=ids).loss.backward()
    g = [p.grad for n, p in m.named_parameters() if ".conv.conv.weight" in n]
    assert g and all(x is not None and x.abs().sum() > 0 for x in g)
    m.gradient_checkpointing_enable()
    m(ids, labels=ids).loss.backward()


# ------------------------------------------------------------------------------------------------ caches
def test_dense_cache_layout_reset_and_truncate():
    cfg = _cfg()
    cache = DenseKVCache(2, 8, cfg, dtype=torch.float32)
    assert [cache.k[i].shape[1] for i in range(5)] == [0, 8, 0, 8, 0]  # conv layers hold no KV
    assert sorted(cache.conv) == [0, 2, 4] and cache.conv[0].shape == (2, 32, 2)
    assert cache.nbytes() > cache.bytes_per_token() * 2 * 8 - 1  # conv state counted in nbytes, not in bytes_per_token
    cache.conv[0].fill_(1.0)
    cache.commit(3)
    with pytest.raises(ConfigError, match="rolled back"):
        cache.truncate([1, 1])
    cache.truncate([5, 5])  # not shrinking: fine
    cache.reset([1])
    assert cache.conv[0][1].abs().sum() == 0 and cache.conv[0][0].abs().sum() > 0 and cache.seq_lens.tolist() == [3, 0]


def test_paged_view_with_state_slots_matches_dense():
    cfg = _cfg()
    m = _model(cfg)
    ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        ref = m(ids).logits
        pool = SimplePagedPool(cfg, 16, 4, torch.float32, num_state_slots=5)
        tables = torch.tensor([[0, 1, 2], [3, 4, 5]])
        slots = torch.tensor([4, 1])
        outs = []
        lens = torch.zeros(2, dtype=torch.long)
        for a, b in ((0, 6), (6, 9), (9, 10)):
            view = PagedKVView(tables, lens, pool, state_slots=slots)
            outs.append(m(ids[:, a:b], kv_cache=view).logits)
            lens = view.seq_lens
    assert (torch.cat(outs, 1) - ref).abs().max() < 1e-4
    with pytest.raises(ConfigError):
        SimplePagedPool(cfg, 4, 4, torch.float32)  # hybrid needs state slots
    with pytest.raises(ConfigError, match="state_slots"):
        PagedKVView(tables, torch.zeros(2, dtype=torch.long), pool).get_conv(0)


# ------------------------------------------------------------------------------------------------ names + IO
def test_hf_name_mapping_round_trip():
    cfg = _cfg()
    pairs = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.embedding_norm.weight": "norm.weight",
        "model.layers.1.operator_norm.weight": "layers.1.input_layernorm.weight",
        "model.layers.1.ffn_norm.weight": "layers.1.post_attention_layernorm.weight",
        "model.layers.1.feed_forward.w1.weight": "layers.1.mlp.gate_proj.weight",
        "model.layers.1.feed_forward.w3.weight": "layers.1.mlp.up_proj.weight",
        "model.layers.1.feed_forward.w2.weight": "layers.1.mlp.down_proj.weight",
        "model.layers.1.self_attn.out_proj.weight": "layers.1.self_attn.o_proj.weight",
        "model.layers.1.self_attn.q_layernorm.weight": "layers.1.self_attn.q_norm.weight",
        "model.layers.1.self_attn.k_layernorm.weight": "layers.1.self_attn.k_norm.weight",
        "model.layers.1.self_attn.q_proj.weight": "layers.1.self_attn.q_proj.weight",
        "model.layers.0.conv.in_proj.weight": "layers.0.conv.in_proj.weight",
        "model.layers.0.conv.conv.weight": "layers.0.conv.conv.weight",
        "model.layers.0.conv.out_proj.weight": "layers.0.conv.out_proj.weight",
        # VL checkpoints: language_model prefix stripped, towers dropped
        "model.language_model.embedding_norm.weight": "norm.weight",
        "model.language_model.layers.0.conv.in_proj.weight": "layers.0.conv.in_proj.weight",
        "model.vision_tower.encoder.layers.0.weight": None,
        "model.multi_modal_projector.linear_1.weight": None,
    }
    for hf, fg in pairs.items():
        assert hf_to_forge_name(hf, cfg) == fg, hf
        if fg is not None and "language_model" not in hf:
            assert forge_to_hf_name(fg, cfg) == hf, fg
    mcfg = ModelConfig.from_hf_config(_moe_hf_dict())
    assert hf_to_forge_name("model.layers.3.feed_forward.gate.weight", mcfg) == "layers.3.mlp.gate.weight"
    assert hf_to_forge_name("model.layers.3.feed_forward.expert_bias", mcfg) == "layers.3.mlp.expert_bias"
    assert hf_to_forge_name("model.layers.3.feed_forward.experts.2.w1.weight", mcfg) == "layers.3.mlp.experts.2.gate_proj.weight"
    assert forge_to_hf_name("layers.3.mlp.experts.2.down_proj.weight", mcfg) == "model.layers.3.feed_forward.experts.2.w2.weight"
    assert forge_to_hf_name("layers.3.mlp.expert_bias", mcfg) == "model.layers.3.feed_forward.expert_bias"


@pytest.mark.parametrize("hf", [lambda: _hf_dict(), lambda: _moe_hf_dict()], ids=["dense", "moe"])
def test_state_dict_round_trip_and_save_load(tmp_path, hf):
    cfg = ModelConfig.from_hf_config(hf())
    m = _model(cfg)
    sd = forge_state_dict_to_hf(m)
    assert "lm_head.weight" not in sd and "model.embedding_norm.weight" in sd
    assert not any(k.startswith("model.layers.0.self_attn") for k in sd)  # layer 0 is a conv layer
    assert any(k == "model.layers.1.self_attn.out_proj.weight" for k in sd)
    m2 = ForgeLM(cfg, dtype=torch.float32, device="meta", init_weights=False)
    materialize(m2, "cpu", init=False)
    rep = load_hf_tensors(m2, iter(sd.items()))
    assert not rep.missing and not rep.unexpected
    ids = torch.randint(0, 128, (2, 7))
    with torch.no_grad():
        ref = m(ids).logits
        assert (m2.eval()(ids).logits - ref).abs().max() < 1e-6
    save_model(m, tmp_path / "ck")
    m3, _ = load_model(tmp_path / "ck", dtype="float32", device="cpu", with_tokenizer=False)
    with torch.no_grad():
        assert (m3(ids).logits - ref).abs().max() < 1e-6
    assert m3.config == cfg


def test_expert_bias_is_fp32_and_only_biases_selection():
    cfg = ModelConfig.from_hf_config(_moe_hf_dict())
    m = _model(cfg)
    moe_layer = next(l for l in m.layers if l.is_moe)
    assert moe_layer.mlp.expert_bias.dtype == torch.float32
    m = m.to(torch.bfloat16)
    assert moe_layer.mlp.expert_bias.dtype == torch.float32  # never rounded with the weights
    logits = torch.tensor([[2.0, 1.9, -1.0, 0.0]])
    bias = torch.tensor([0.0, 0.0, 5.0, 0.0])  # expert 2 is selected thanks to the bias, but weighted by its own (tiny) score
    w, ids, probs = route_sigmoid(logits, 2, True, bias, 1.5)
    assert set(ids[0].tolist()) == {2, 0}
    p = torch.sigmoid(logits)
    exp = torch.stack([p[0, ids[0, 0]], p[0, ids[0, 1]]])
    assert torch.allclose(w[0], 1.5 * exp / (exp.sum() + 1e-6), atol=1e-6)
    assert torch.allclose(probs.sum(-1), torch.ones(1))
    w2, ids2, _ = route_sigmoid(logits, 2, False, None, 1.0)
    assert ids2[0].tolist() == [0, 1] and torch.allclose(w2[0], p[0, :2])


def test_moe_decode_matches_full_and_dense_layers_first():
    cfg = ModelConfig.from_hf_config(_moe_hf_dict())
    assert [cfg.is_moe_layer(i) for i in range(5)] == [False, False, True, True, True]
    m = _model(cfg)
    ids = torch.randint(0, 128, (2, 9))
    with torch.no_grad():
        full = m(ids).logits
        cache = m.new_kv_cache(2, 12)
        steps = torch.cat([m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(9)], 1)
    assert (full - steps).abs().max() < 1e-4


# ------------------------------------------------------------------------------------------------ transformers cross-check
try:
    import transformers
except ImportError:  # pragma: no cover
    transformers = None
needs_hf = pytest.mark.skipif(transformers is None, reason="transformers not installed")


def _hf_pair(model_type: str):
    if not hasattr(transformers, "Lfm2Config" if model_type == "lfm2" else "Lfm2MoeConfig"):
        pytest.skip("installed transformers has no LFM2")
    torch.manual_seed(0)
    if model_type == "lfm2":
        c = transformers.Lfm2Config(
            vocab_size=128, hidden_size=32, intermediate_size=96, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, conv_L_cache=3, layer_types=list(LT5), block_auto_adjust_ff_dim=True, block_multiple_of=16,
            block_ffn_dim_multiplier=1.0, tie_word_embeddings=True,
        )
        hf = transformers.Lfm2ForCausalLM(c)
    else:
        c = transformers.Lfm2MoeConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, moe_intermediate_size=24, num_hidden_layers=5, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=256, conv_L_cache=3, layer_types=list(LT5), num_experts=4,
            num_experts_per_tok=2, num_dense_layers=2, tie_word_embeddings=True,
        )
        hf = transformers.Lfm2MoeForCausalLM(c)
    hf = hf.float().eval()
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "norm" in n or "layernorm" in n:
                p.add_(torch.randn_like(p) * 0.1)
        for n, b in hf.named_buffers():
            if n.endswith("expert_bias"):
                b.copy_(torch.randn_like(b) * 0.3)
    cfg = ModelConfig.from_hf_config(hf.config.to_dict())
    model = ForgeLM(cfg, dtype=torch.float32, device="meta", init_weights=False)
    materialize(model, "cpu", init=False)
    rep = load_hf_tensors(model, iter(hf.state_dict().items()))
    assert not rep.missing, rep.missing
    return hf, model.eval(), cfg


@needs_hf
@pytest.mark.parametrize("kind", ["lfm2", "lfm2_moe"])
def test_matches_transformers_logits_and_cached_decode(kind):
    hf, model, cfg = _hf_pair(kind)
    assert cfg.layer_types == LT5 and cfg.is_hybrid
    ids = torch.randint(0, 128, (2, 21))
    with torch.no_grad():
        ref = hf(ids).logits
        out = model(ids).logits
        assert (ref - out).abs().max() < 2e-4
        cache = model.new_kv_cache(2, 24)
        steps = torch.cat([model(ids[:, i : i + 1], kv_cache=cache).logits for i in range(21)], 1)
    assert (ref - steps).abs().max() < 2e-4


@needs_hf
def test_intermediate_width_matches_transformers_formula():
    if not hasattr(transformers, "Lfm2Config"):
        pytest.skip("no LFM2 in transformers")
    for ff, mult, mo in ((6656, 1.0, 256), (10240, 1.0, 256), (96, 1.0, 16), (100, 1.5, 16)):
        c = transformers.Lfm2Config(
            vocab_size=64, hidden_size=32, intermediate_size=ff, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            layer_types=["conv", "full_attention"], block_auto_adjust_ff_dim=True, block_multiple_of=mo, block_ffn_dim_multiplier=mult,
        )
        w = transformers.Lfm2ForCausalLM(c).model.layers[0].feed_forward.w1.weight.shape[0]
        assert ModelConfig.from_hf_config(c.to_dict()).intermediate_size == w, (ff, mult, mo)


# ------------------------------------------------------------------------------------------------ serving engine
def _engine(model, **kw):
    from forge.serve._testing import ByteTokenizer
    from forge.serve.config import ServeConfig
    from forge.serve.engine import LLMEngine

    base = dict(model="t", device="cpu", dtype="float32", block_size=4, num_gpu_blocks=64, max_model_len=64, max_num_seqs=4)
    base.update(kw)
    return LLMEngine(ServeConfig(**base), model=model, tokenizer=ByteTokenizer())


def _ref_greedy(model, prompt, n):
    ids, out = list(prompt), []
    with torch.no_grad():
        for _ in range(n):
            t = int(model(torch.tensor([ids])).logits[0, -1].argmax())
            ids.append(t)
            out.append(t)
    return out


@pytest.mark.parametrize("hf", [lambda: _hf_dict(), lambda: _moe_hf_dict()], ids=["dense", "moe"])
def test_engine_matches_forgelm_greedy(hf):
    from forge.serve.sampling import SamplingParams

    cfg = ModelConfig.from_hf_config(hf())
    m = _model(cfg)
    e = _engine(m, max_num_batched_tokens=16)  # small budget: prompts are prefilled in chunks (conv state carried across chunks)
    assert e.pool.num_layers == 2 and e.hybrid and e.stats()  # KV pool only for the 2 attention layers
    prompts = [list(range(5, 31)), list(range(40, 49)), [7, 8, 9]]
    sp = SamplingParams(temperature=0, max_tokens=10, ignore_eos=True)
    outs = e.generate(prompts, [sp] * 3)
    for p, o in zip(prompts, outs):
        assert o.token_ids == _ref_greedy(m, p, 10)
    assert not e.runner._conv_state  # states of finished requests are dropped


def test_engine_hybrid_policy_errors_and_prefix_cache_off():
    m = _model(_cfg())
    e = _engine(m, prefix_caching=True)
    assert e.manager.enable_prefix_caching is False  # cached prefix blocks carry no conv state
    with pytest.raises(ConfigError, match="swap"):
        _engine(m, preemption_mode="swap", host_kv_gb=0.01)
    with pytest.raises(ConfigError, match="speculative"):
        _engine(m, speculation={"mode": "ngram"})
    with pytest.raises(ConfigError, match="speculative"):
        e.attach_eagle_head(torch.nn.Linear(1, 1))
    # a pure-attention model keeps prefix caching
    from forge.models.tiny import build_tiny_model

    assert _engine(build_tiny_model(1).eval(), prefix_caching=True).manager.enable_prefix_caching is True


def test_runner_requires_state_key_and_resets_on_fresh_sequence():
    from forge.serve.runner import SeqInput

    m = _model(_cfg())
    e = _engine(m)
    p = list(range(3, 15))
    e.manager.allocate("a", p)
    with pytest.raises(ConfigError, match="state_key"):
        e.runner.forward([SeqInput(p, 0, e.manager.block_table("a"))])
    out1 = e.runner.forward([SeqInput(p, 0, e.manager.block_table("a"), state_key="a")]).logits[0][-1]
    out2 = e.runner.forward([SeqInput(p, 0, e.manager.block_table("a"), state_key="a")]).logits[0][-1]  # start_pos 0 => fresh state
    ref = m(torch.tensor([p])).logits[0, -1]
    assert torch.allclose(out1, ref, atol=1e-4) and torch.allclose(out2, ref, atol=1e-4)


# ------------------------------------------------------------------------------------------------ planner / capacity / compression
def test_capacity_and_planner_use_attention_layers_only():
    from forge.planner.spec import ModelSpec as PlanSpec
    from forge.serve.capacity import ModelSpec as CapSpec

    c = ModelConfig.from_hf_config(CFG_350M)
    cap = CapSpec.from_any(c)
    assert cap.kv_layers == 6 and cap.state_bytes_per_seq == 40_960
    assert cap.kv_bytes_per_token("bf16") == c.kv_bytes_per_token(16) == 12_288
    for src in (CFG_350M, c):
        ps = PlanSpec.from_config(src)
        assert ps.attention == "hybrid" and ps.full_attn_layers == 6 and ps.kv_layers == 6
        assert ps.params_b == pytest.approx(0.354483968, rel=1e-9)
        assert ps.kv_elems_per_token() == 2.0 * 6 * 8 * 64


def test_capacity_plan_counts_conv_state():
    from forge.serve.capacity import capacity_plan

    c = ModelConfig.from_hf_config(CFG_350M)
    r = capacity_plan(c, "H100", weight_bits=16)
    assert any("conv state" in n for n in r.notes)


def test_model_summary_reports_state():
    from forge.models.accounting import model_summary

    s = model_summary(ModelConfig.from_hf_config(CFG_350M))
    assert s["conv_layers"] == 10 and s["recurrent_state_bytes_per_seq"] == 40_960 and s["kv_cache_layers"] == 6


def test_compression_guards_and_lora_targets():
    from forge.compress.prune import PruneConfig, prune_model
    from forge.compress.rotation import fuse_norms
    from forge.models.lora import add_lora

    m = _model(_cfg())
    with pytest.raises(ConfigError, match="hybrid"):
        prune_model(m, PruneConfig(mlp_keep=0.5), tokens=torch.randint(0, 128, (2, 8)))
    with pytest.raises(ConfigError, match="hybrid"):
        fuse_norms(m)
    names = add_lora(m, 4, 8.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"])
    assert any(".conv.in_proj" in n for n in names) and any(".conv.out_proj" in n for n in names)
    assert not any("conv.conv" in n for n in names)  # the depthwise filter is not a Linear
    m2 = _model(_cfg())
    all_names = add_lora(m2, 4, 8.0, target_modules="all-linear")
    assert any(".conv.in_proj" in n for n in all_names) and any(".self_attn.q_proj" in n for n in all_names)
    ids = torch.randint(0, 128, (1, 6))
    m2.train()
    m2(ids, labels=ids).loss.backward()
    assert any(p.grad is not None for n, p in m2.named_parameters() if "conv.in_proj.lora_A" in n)


def test_ptq_style_linear_quantisation_skips_depthwise_conv():
    from forge.models.linear import quantize_linears

    m = _model(_cfg())
    try:
        quantize_linears(m, "fp8")
    except ConfigError:
        pytest.skip("fp8 linear kind unavailable on this host")
    assert isinstance(m.layers[0].conv.conv, torch.nn.Conv1d)  # untouched
    with torch.no_grad():
        assert torch.isfinite(m(torch.randint(0, 128, (1, 5))).logits).all()
