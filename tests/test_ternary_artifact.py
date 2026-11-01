"""Ternary artifact path on CPU (P1 exit criteria): quantise a tiny ForgeLM -> save -> load -> identical logits; the manifest, label and
layout round-trip; .t158.safetensors import/export is bit-exact."""
from __future__ import annotations

import json

import pytest
import torch

from forge.core.artifactmeta import read_artifact_summary
from forge.formats import ternary as tn
from forge.kernels.layouts.ternary import quantize_pack_ternary
from forge.models.linear import PackedLinear, quantize_linears
from forge.models.loader import describe_loaded_model, load_model, model_weight_bytes, save_model
from forge.models.tiny import build_tiny_model
from forge.ternary.export import T158File, infer_shapes, read_t158, split_rows, write_t158


@pytest.fixture(scope="module")
def tern_dir(tmp_path_factory):
    model = build_tiny_model(seed=1)
    quantize_linears(model, "t158", group_size=16)
    d = tmp_path_factory.mktemp("t158") / "art"
    save_model(model, d)
    return model, d


def test_save_load_logits_identical(tern_dir):
    model, d = tern_dir
    kinds = {m.kind for m in model.modules() if isinstance(m, PackedLinear)}
    assert kinds == {"t158"}
    loaded, _ = load_model(d, dtype=torch.float32, device="cpu")
    ids = torch.randint(0, model.config.vocab_size, (2, 9))
    with torch.no_grad():
        a, b = model.eval()(ids), loaded(ids)
    a, b = (getattr(o, "logits", o) for o in (a, b))
    assert torch.equal(a, b)
    assert {m.kind for m in loaded.modules() if isinstance(m, PackedLinear)} == {"t158"}


def test_summary_label_and_bytes(tern_dir):
    model, d = tern_dir
    s = read_artifact_summary(d)
    assert s["is_quantized"] and s["label"] == "Ternary 1.58b" and set(s["layer_kinds"]) == {"t158"}
    cfg = json.loads((d / "config.json").read_text())
    assert any(e["kind"] == "t158" and e["spec"]["group_size"] == 16 for e in cfg["forge"]["quantization"].values())
    loaded, _ = load_model(d, dtype=torch.float32, device="cpu")
    info = describe_loaded_model(loaded)
    assert info["quantized"] and model_weight_bytes(loaded)["packed"] > 0
    lin = next(m for m in loaded.modules() if isinstance(m, PackedLinear))
    assert lin.packed_nbytes() < lin.in_features * lin.out_features * 2 / 3  # g16 fp32 scales cost 4 bit/weight: still far under bf16


def test_t158_file_roundtrip_bit_exact(tmp_path):
    g = torch.Generator().manual_seed(7)
    layers = {"l0.qkv": quantize_pack_ternary(torch.randn(96, 64, generator=g), 32), "l0.o": quantize_pack_ternary(torch.randn(64, 64, generator=g), -1),
              "l1.up": quantize_pack_ternary(torch.randn(128, 64, generator=g), 0)}
    f = T158File(layers=layers, norms={"l0.qkv": torch.ones(64)}, dense={"embed.weight": torch.randn(10, 64).bfloat16()}, meta={"config": "x"})
    path = tmp_path / "m.t158.safetensors"
    assert write_t158(path, f) > 0
    shapes = {k: (p.n, p.k) for k, p in layers.items()}
    back = read_t158(path, shapes)
    for k, p in layers.items():
        q = back.layers[k]
        assert torch.equal(q.qweight, p.qweight) and torch.equal(q.scales, p.scales) and q.group_size == p.group_size, k
    assert torch.equal(back.norms["l0.qkv"], torch.ones(64)) and set(back.dense) == {"embed.weight"}
    with pytest.raises(Exception):
        read_t158(path, {**shapes, "l0.o": (32, 64)})


def test_split_rows_matches_fused():
    p = quantize_pack_ternary(torch.randn(48, 64), 32)
    parts = split_rows(p, [16, 8, 24])
    assert torch.equal(torch.cat([q.dequantize() for q in parts]), p.dequantize())
    with pytest.raises(Exception):
        split_rows(quantize_pack_ternary(torch.randn(8, 64), 0), [4, 4])


def test_infer_shapes_llama_like():
    cfg = {"model": {"hidden": 512, "layers": 2, "heads": 8, "kv_heads": 4, "head_dim": 0, "ffn": 1536, "act": "swiglu"}}
    sh = infer_shapes(cfg)
    assert sh["l0.qkv"] == ((8 + 8) * 64, 512) and sh["l1.up"] == (3072, 512) and sh["l0.down"] == (512, 1536) and sh["l0.o"] == (512, 512)


def test_engine_serves_ternary_artifact_and_matches_greedy(tern_dir):
    from forge.serve._testing import ByteTokenizer
    from forge.serve.config import ServeConfig
    from forge.serve.engine import LLMEngine
    from forge.serve.sampling import SamplingParams

    _, d = tern_dir
    cfg = ServeConfig(model=str(d), device="cpu", dtype="float32", block_size=4, max_model_len=128, max_num_seqs=4, kv_dtype="fp32")
    eng = LLMEngine(cfg, tokenizer=ByteTokenizer())
    ident = eng.identity
    assert ident["quantized"] and ident["label"] == "Ternary 1.58b" and 0 < ident["packed_bytes"] < ident["weight_bytes"]
    m, _ = load_model(d, dtype="float32", device="cpu", with_tokenizer=False)
    p = list(range(5, 30))
    ids, want = list(p), []
    with torch.no_grad():
        for _ in range(8):
            want.append(int(m(torch.tensor([ids])).logits[0, -1].argmax()))
            ids.append(want[-1])
    got = eng.generate([p], [SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)])[0].token_ids
    assert got == want
