"""Checkpoint IO: name mapping, save/load round trip, sharding, dtype cast, meta/materialise, quantisation manifest, tokenizer."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from forge.core.errors import ConfigError
from forge.models import ForgeLM, build_tiny_model, load_model, make_tiny_config, save_model
from forge.models.loader import (
    expand_hf_tensor,
    forge_state_dict_to_hf,
    forge_to_hf_name,
    hf_to_forge_name,
    iter_checkpoint,
    load_hf_tensors,
    load_tokenizer,
    materialize,
)


def _same(a: ForgeLM, b: ForgeLM):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_name_mapping_roundtrip():
    for ov in [{}, {"moe": True}, {"arch": "mixtral", "moe": {"num_experts": 4, "top_k": 2, "moe_intermediate_size": 32}}]:
        m = build_tiny_model(**ov)
        cfg = m.config
        for k in m.state_dict():
            hf = forge_to_hf_name(k, cfg)
            assert hf_to_forge_name(hf, cfg) == k, (k, hf)
    cfg = make_tiny_config(arch="mixtral", moe={"num_experts": 4, "top_k": 2, "moe_intermediate_size": 32})
    assert forge_to_hf_name("layers.1.mlp.experts.3.gate_proj.weight", cfg) == "model.layers.1.block_sparse_moe.experts.3.w1.weight"
    assert forge_to_hf_name("layers.1.mlp.gate.weight", cfg) == "model.layers.1.block_sparse_moe.gate.weight"
    assert hf_to_forge_name("model.layers.0.self_attn.rotary_emb.inv_freq", cfg) is None
    assert hf_to_forge_name("lm_head.weight", cfg) == "lm_head.weight"
    assert hf_to_forge_name("model.embed_tokens.weight", cfg) == "embed_tokens.weight"


def test_fused_expert_split():
    cfg = make_tiny_config(moe=True)
    e, h, i = 4, 64, 32
    fused = torch.randn(e, 2 * i, h)
    parts = dict(expand_hf_tensor("model.layers.0.mlp.experts.gate_up_proj", fused, cfg))
    assert torch.equal(parts["layers.0.mlp.experts.2.gate_proj.weight"], fused[2, :i])
    assert torch.equal(parts["layers.0.mlp.experts.2.up_proj.weight"], fused[2, i:])
    down = torch.randn(e, h, i)
    p2 = dict(expand_hf_tensor("model.layers.1.mlp.experts.down_proj", down, cfg))
    assert torch.equal(p2["layers.1.mlp.experts.3.down_proj.weight"], down[3])


@pytest.mark.parametrize("ov", [{}, {"tie_word_embeddings": True}, {"moe": True}, {"arch": "qwen2", "attention_bias": True}])
def test_save_load_roundtrip(tmp_path, ov):
    m = build_tiny_model(**ov)
    save_model(m, tmp_path / "ck", as_dtype="float32")
    assert (tmp_path / "ck" / "config.json").exists() and (tmp_path / "ck" / "model.safetensors").exists()
    cfgj = json.loads((tmp_path / "ck" / "config.json").read_text())
    assert cfgj["architectures"][0].endswith("ForCausalLM") and cfgj["forge"]["model_config"]["arch"] == m.config.arch
    m2, tok = load_model(tmp_path / "ck", dtype="float32", device="cpu")
    assert tok is None
    _same(m, m2)
    ids = torch.randint(0, 256, (2, 7))
    m.eval()
    assert torch.equal(m(ids).logits, m2(ids).logits)
    if ov.get("tie_word_embeddings"):
        assert m2.lm_head.weight is m2.embed_tokens.weight
        assert "lm_head.weight" not in load_file(str(tmp_path / "ck" / "model.safetensors"))


def test_loads_plain_hf_layout_without_forge_block(tmp_path):
    m = build_tiny_model(arch="qwen3", qk_norm=True)
    save_model(m, tmp_path / "ck", as_dtype="float32")
    cj = json.loads((tmp_path / "ck" / "config.json").read_text())
    cj.pop("forge")
    (tmp_path / "ck" / "config.json").write_text(json.dumps(cj))
    m2, _ = load_model(tmp_path / "ck", dtype="float32", device="cpu")
    _same(m, m2)


def test_sharded_save_and_dtype_cast(tmp_path):
    m = build_tiny_model()
    save_model(m, tmp_path / "ck", as_dtype="bfloat16", shard_gb=60_000 / (1 << 30))  # ~60 kB shards
    files = sorted((tmp_path / "ck").glob("model-*.safetensors"))
    assert len(files) > 2
    idx = json.loads((tmp_path / "ck" / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"].values()) == {f.name for f in files}
    assert idx["metadata"]["total_size"] > 0
    assert json.loads((tmp_path / "ck" / "config.json").read_text())["torch_dtype"] == "bfloat16"
    m2, _ = load_model(tmp_path / "ck", dtype="bfloat16", device="cpu")
    assert m2.dtype == torch.bfloat16
    ref = {k: v.to(torch.bfloat16) for k, v in m.state_dict().items()}
    for k, v in m2.state_dict().items():
        assert torch.equal(v, ref[k]), k
    # streaming iterator yields every tensor exactly once
    names = [k for k, _ in iter_checkpoint(tmp_path / "ck")]
    assert len(names) == len(set(names)) == len(forge_state_dict_to_hf(m))


def test_meta_load_and_materialise(tmp_path):
    m = build_tiny_model()
    save_model(m, tmp_path / "ck", as_dtype="float32")
    meta, _ = load_model(tmp_path / "ck", dtype="float32", device="meta")
    assert all(p.is_meta for p in meta.parameters())
    materialize(meta, "cpu", init=False)
    rep = load_hf_tensors(meta, iter_checkpoint(tmp_path / "ck"))
    assert not rep.missing and not rep.unexpected and rep.loaded == len(m.state_dict())
    _same(m, meta)


def test_missing_and_mismatch_errors(tmp_path):
    m = build_tiny_model()
    save_model(m, tmp_path / "ck", as_dtype="float32")
    sd = load_file(str(tmp_path / "ck" / "model.safetensors"))
    sd.pop("model.norm.weight")
    save_file(sd, str(tmp_path / "ck" / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ConfigError, match="missing"):
        load_model(tmp_path / "ck", dtype="float32", device="cpu")
    m3, _ = load_model(tmp_path / "ck", dtype="float32", device="cpu", strict=False)
    assert isinstance(m3, ForgeLM)
    sd["model.norm.weight"] = torch.ones(3)
    save_file(sd, str(tmp_path / "ck" / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ConfigError, match="shape mismatch"):
        load_model(tmp_path / "ck", dtype="float32", device="cpu")
    with pytest.raises(Exception):
        load_model(tmp_path / "does-not-exist-xyz", device="cpu", backend="forge")


def test_quantised_model_roundtrip_via_manifest(tmp_path):
    from forge.models.linear import Fp8Linear, linear_report

    m = build_tiny_model()
    m.eval()
    ids = torch.randint(0, 256, (1, 8))
    from forge.models.linear import quantize_linears

    names = quantize_linears(m, "fp8")
    assert names and all("lm_head" not in n for n in names)
    assert isinstance(m.layers[0].self_attn.q_proj, Fp8Linear) and type(m.lm_head).__name__ == "DenseLinear"
    ref = m(ids).logits
    save_model(m, tmp_path / "q")
    cj = json.loads((tmp_path / "q" / "config.json").read_text())
    assert cj["forge"]["quantization"]["layers.0.self_attn.q_proj"]["kind"] == "fp8"
    m2, _ = load_model(tmp_path / "q", dtype="float32", device="cpu")
    assert linear_report(m2)["layers.0.self_attn.q_proj"] == "fp8"
    assert torch.allclose(m2(ids).logits, ref, atol=1e-5)
    # post-load quant argument
    m3, _ = load_model(tmp_path / "q", dtype="float32", device="cpu")
    clean = build_tiny_model()
    save_model(clean, tmp_path / "c", as_dtype="float32")
    m4, _ = load_model(tmp_path / "c", dtype="float32", device="cpu", quant="fp8")
    assert linear_report(m4)["layers.1.mlp.down_proj"] == "fp8"
    with pytest.raises(ConfigError):
        load_model(tmp_path / "c", dtype="float32", device="cpu", quant="no-such-kind")


def test_save_refuses_unmerged_lora(tmp_path):
    from forge.models import add_lora

    m = build_tiny_model()
    add_lora(m, r=2)
    with pytest.raises(ConfigError, match="LoRA"):
        save_model(m, tmp_path / "x")


def test_tokenizer_wrapper(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "</s>": 1, "hello": 2, "world": 3}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    d = tmp_path / "tok"
    d.mkdir()
    tk.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "</s>"}))
    t = load_tokenizer(d)
    assert t is not None
    ids = t("hello world")["input_ids"] if not hasattr(t, "encode") else t.encode("hello world")
    ids = ids["input_ids"] if isinstance(ids, dict) else ids
    assert list(ids)[:2] == [2, 3]
    assert load_tokenizer(tmp_path / "nothing-here") is None
