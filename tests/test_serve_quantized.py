"""Quantised artifacts end to end on CPU: compress (tiny model, W4A16 recipe) -> artifact dir -> model resolution -> loader (packed, never a silent
dense stand-in) -> serving engine (identity + explicit memory plan), plus the pure memory-budget arithmetic for a 96 GB card."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from compress_helpers import logits_of, rand_tokens  # noqa: E402
from test_compress_recipes import _job  # noqa: E402

from forge.core.artifactmeta import read_artifact_summary  # noqa: E402
from forge.core.context import NullContext  # noqa: E402
from forge.core.errors import ConfigError, NotFound  # noqa: E402
from forge.models.loader import (  # noqa: E402
    convert_packed_for_kernels,
    describe_loaded_model,
    load_model,
    model_weight_bytes,
)
from forge.serve._testing import ByteTokenizer  # noqa: E402
from forge.serve.config import ServeConfig  # noqa: E402
from forge.serve.engine import LLMEngine  # noqa: E402
from forge.serve.memory import GB, estimate_memory, plan_kv_memory  # noqa: E402
from forge.serve.sampling import SamplingParams  # noqa: E402


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Path:
    """A real compress output: the Appendix C.1 W4A16-GPTQ recipe (with fp8 KV) on the tiny model, saved to a tmp dir."""
    from forge.compress.run import run_compress

    out = tmp_path_factory.mktemp("w4a16") / "artifact"
    res = run_compress(_job(save_dir=str(out)), NullContext())
    assert Path(res["model_dir"]) == out and res["layout_id"]
    return out


@pytest.fixture
def art(artifact, tmp_path) -> Path:
    """A private copy so a test may damage it."""
    d = tmp_path / "artifact"
    shutil.copytree(artifact, d)
    return d


# ------------------------------------------------------------------------------------------------ metadata
def test_artifact_summary_describes_the_packed_layout(artifact):
    s = read_artifact_summary(artifact)
    assert s["is_quantized"] and s["is_compress_artifact"]
    assert s["scheme"] == "w4a16" and s["bits"] == 4 and s["label"].startswith("W4A16 g")
    assert s["n_quantized_layers"] == sum(s["layer_kinds"].values()) > 0
    assert s["layout_id"] and s["layout_id"].startswith("w4.portable_row.v1")  # the pack stage's layout id travels with the artifact
    assert s["kv"]["scheme"] == "fp8_e4m3"
    assert s["recipe"] and s["base_model"] == "tiny:3"


def test_summary_of_non_artifacts():
    assert read_artifact_summary(Path("definitely/not/here")) is None


# ------------------------------------------------------------------------------------------------ loader
def test_loader_builds_packed_linears_and_matches_the_dequantised_reference(artifact):
    m, _ = load_model(artifact, dtype="float32", device="cpu", with_tokenizer=False)
    info = m.forge_load_info
    assert info["quantized"] and info["label"].startswith("W4A16") and info["layer_kinds"] == {"w4a16": 14}
    # weights are the packed bytes, far below the dense equivalent (4-bit codes + group scales)
    wb = model_weight_bytes(m)
    assert 0 < wb["packed"] < wb["total"] and info["weight_bytes"] == wb["total"]
    dense_equiv = sum(mod.in_features * mod.out_features * 4 for mod in m.modules() if hasattr(mod, "dequantize") and hasattr(mod, "in_features"))
    assert wb["packed"] < dense_equiv / 4
    # reference: the same model with every packed linear replaced by a dense linear holding its dequantised weight
    ref = load_model(artifact, dtype="float32", device="cpu", with_tokenizer=False)[0]
    for name, mod in list(ref.named_modules()):
        if hasattr(mod, "dequantize") and hasattr(mod, "in_features"):
            lin = torch.nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            lin.weight.data = mod.dequantize(torch.float32)
            if mod.bias is not None:
                lin.bias.data = mod.bias.detach().float()
            parent = ref.get_submodule(name.rpartition(".")[0]) if "." in name else ref
            setattr(parent, name.rpartition(".")[2], lin)
    t = rand_tokens(4, 24, vocab=256)
    assert (logits_of(m, t) - logits_of(ref, t)).abs().max() < 1e-4


def test_loader_never_serves_a_dense_stand_in(art):
    # 1) the config lost its quantisation manifest but the metadata says the weights are packed -> error, not a dense model
    cfg = json.loads((art / "config.json").read_text())
    cfg["forge"].pop("quantization")
    (art / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ConfigError):
        load_model(art, dtype="float32", device="cpu", with_tokenizer=False)


def test_loader_rejects_quant_override_on_a_packed_artifact(artifact):
    with pytest.raises(ConfigError, match="already quantised"):
        load_model(artifact, dtype="float32", device="cpu", with_tokenizer=False, quant="w4a16")


def test_loader_rejects_manifest_naming_unknown_modules(art):
    cfg = json.loads((art / "config.json").read_text())
    ent = next(iter(cfg["forge"]["quantization"].values()))
    cfg["forge"]["quantization"]["layers.99.self_attn.q_proj"] = ent
    (art / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ConfigError, match="architecture does not have"):
        load_model(art, dtype="float32", device="cpu", with_tokenizer=False)


def test_hf_gptq_checkpoints_get_a_clear_error(tmp_path):
    d = tmp_path / "hf-gptq"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                                               "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100,
                                               "quantization_config": {"quant_method": "gptq", "bits": 4}}))
    with pytest.raises(ConfigError, match="gptq"):
        load_model(d, dtype="float32", device="cpu", with_tokenizer=False)


def test_kernel_layout_conversion_keeps_the_numbers_and_drops_the_portable_copy(artifact):
    m, _ = load_model(artifact, dtype="float32", device="cpu", with_tokenizer=False)
    t = rand_tokens(2, 16, vocab=256)
    before = logits_of(m, t)
    assert convert_packed_for_kernels(m)["converted"] == 0  # CPU modules stay portable by default
    out = convert_packed_for_kernels(m, require_cuda=False)  # exercises the kernel module (torch-oracle path on CPU)
    assert out["converted"] == 14 and not out["errors"] and out["avoided_duplicate_bytes"] > 0
    assert {type(x).__name__ for x in m.modules() if getattr(x, "kind", "") == "w4a16"} == {"KernelW4A16Linear"}
    assert (logits_of(m, t) - before).abs().max() < 1e-4
    assert describe_loaded_model(m)["layer_kinds"] == {"w4a16": 14}


# ------------------------------------------------------------------------------------------------ engine
def _cfg(path, **kw):
    base = dict(model=str(path), device="cpu", dtype="float32", block_size=4, max_model_len=128, max_num_seqs=4)
    base.update(kw)
    return ServeConfig(**base)


def test_engine_serves_the_packed_artifact_and_reports_what_it_loaded(artifact):
    eng = LLMEngine(_cfg(artifact), tokenizer=ByteTokenizer())
    ident = eng.identity
    assert ident["quantized"] and ident["label"].startswith("W4A16") and ident["layer_kinds"] == {"w4a16": 14}
    assert ident["path"] == str(artifact) and 0 < ident["packed_bytes"] < ident["weight_bytes"]
    assert ident["base_model"] == "tiny:3" and ident["layout_id"].startswith("w4.portable_row")
    assert eng.kv_name == "fp8_e4m3"  # kv_dtype=auto follows the recipe's KV scheme
    st = eng.stats()
    assert st["model"]["label"] == ident["label"] and st["memory"]["weights_gb"] == ident["weight_gb"]
    # the engine's output equals the packed model's own greedy decoding
    m, _ = load_model(artifact, dtype="float32", device="cpu", with_tokenizer=False)
    p = list(range(5, 30))
    ids, want = list(p), []
    with torch.no_grad():
        for _ in range(8):
            want.append(int(m(torch.tensor([ids])).logits[0, -1].argmax()))
            ids.append(want[-1])
    got = LLMEngine(_cfg(artifact, kv_dtype="fp32"), tokenizer=ByteTokenizer()).generate(
        [p], [SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)])[0].token_ids
    assert got == want


def test_engine_fails_instead_of_loading_something_else(tmp_path):
    with pytest.raises(NotFound):
        LLMEngine(_cfg(tmp_path / "missing-artifact"), tokenizer=ByteTokenizer())
    with pytest.raises(NotFound):
        LLMEngine(_cfg("blake3:" + "0" * 64), tokenizer=ByteTokenizer())


# ------------------------------------------------------------------------------------------------ memory arithmetic
# Qwen3-14B: 40 layers x 8 KV heads x 128 dim; bf16 KV = 163,840 B/token -> 2.62 MB per 16-token block
PER_BLOCK_BF16 = 2 * 40 * 16 * 8 * 128 * 2
PER_BLOCK_FP8 = PER_BLOCK_BF16 // 2


def _plan(**kw):
    base = dict(device="cuda", bytes_per_block=PER_BLOCK_BF16, block_size=16, weight_bytes=int(10.08 * GB), packed_weight_bytes=int(8.5 * GB),
                gpu_mem_util=0.85, activation_reserve_gb=1.0, max_num_seqs=64, max_model_len=4096, total_bytes=int(102.6 * GB),
                free_bytes=int(102 * GB), allocated_bytes=int(10.08 * GB))
    base.update(kw)
    return plan_kv_memory(**base)


def test_fit_workload_sizes_the_pool_to_the_workload_not_the_card():
    p = _plan()
    assert p.limited_by == "workload" and p.num_blocks == 64 * 256
    assert p.kv_pool_bytes == p.num_blocks * PER_BLOCK_BF16 and abs(p.kv_pool_bytes / GB - 42.99) < 0.1
    assert p.weight_bytes == int(10.08 * GB)
    # weights + pool + reserve is well below the card: the rest is reported as free
    assert p.headroom_bytes > 40 * GB and p.planned_total_bytes < 0.85 * p.total_bytes
    assert "weights 10.08 GB" in p.summary() and "KV pool" in p.summary() and "free" in p.summary()


def test_fill_policy_uses_the_whole_budget_and_that_is_where_87gb_came_from():
    p = _plan(kv_pool="fill", gpu_mem_util=0.90)
    assert p.limited_by == "memory"
    # weights + KV + reserve reach gpu_mem_util x total however small the weights are: ~92 GB on a 102.6 GB (96 GiB) card
    assert abs(p.planned_total_bytes / GB - 0.90 * 102.6) < 0.01 * 102.6
    # the same card with bf16 weights (29.5 GB) occupies the same total: the number says nothing about what was loaded
    q = _plan(kv_pool="fill", gpu_mem_util=0.90, weight_bytes=int(29.5 * GB), allocated_bytes=int(29.5 * GB))
    assert abs(q.planned_total_bytes - p.planned_total_bytes) < 0.01 * GB and q.num_blocks < p.num_blocks


def test_budget_respects_free_memory_of_other_processes():
    p = _plan(kv_pool="fill", free_bytes=int(30 * GB))
    assert p.kv_pool_bytes <= 30 * GB - 1.0 * GB


def test_explicit_blocks_and_cpu_cap_and_small_pool_warning():
    assert _plan(num_gpu_blocks=100).limited_by == "explicit" and _plan(num_gpu_blocks=100).num_blocks == 100
    c = _plan(device="cpu", total_bytes=None, free_bytes=None, allocated_bytes=None, max_num_seqs=1024, max_model_len=8192)
    assert c.limited_by == "cpu_cap" and c.num_blocks == 2048 and c.headroom_bytes is None
    w = _plan(num_gpu_blocks=10)
    assert w.warnings and "max_model_len" in w.warnings[0]


def test_model_that_does_not_fit_fails_loudly():
    with pytest.raises(ConfigError, match="does not fit the memory budget"):
        _plan(weight_bytes=int(95 * GB), allocated_bytes=int(95 * GB))
    with pytest.raises(ConfigError, match="not enough GPU memory for a KV cache"):
        _plan(weight_bytes=int(85.5 * GB), allocated_bytes=int(85.5 * GB), bytes_per_block=PER_BLOCK_BF16 * 100)  # ~0.7 GB left, 262 MB blocks
    with pytest.raises(ConfigError):
        _plan(kv_pool="whatever")


def test_estimate_memory_for_qwen3_14b_w4a16_on_96gb():
    from forge.models.config import ModelConfig

    cfg = ModelConfig(arch="qwen3", vocab_size=151936, hidden_size=5120, intermediate_size=17408, num_layers=40, num_heads=40, num_kv_heads=8,
                      head_dim=128, max_position_embeddings=40960, qk_norm=True)
    e = estimate_memory(cfg, weight_bytes=int(10.08 * GB), packed_weight_bytes=int(8.5 * GB), kv_name="fp8_e4m3", total_gpu_bytes=int(102.6 * GB),
                        max_num_seqs=32, max_model_len=8192)
    assert e["fits"] and e["limited_by"] == "workload" and e["kv_bytes_per_token"] == 40 * 8 * 128 * 2  # fp8: 1 byte per element
    assert abs(e["kv_pool_gb"] - 32 * 8192 * 81920 / 1e9) < 0.05 and e["weights_gb"] == 10.08
    assert estimate_memory(cfg, weight_bytes=int(200 * GB), total_gpu_bytes=int(102.6 * GB))["fits"] is False
    cpu = estimate_memory(cfg, weight_bytes=int(10 * GB))
    assert cpu["fits"] and cpu["free_gb"] is None
