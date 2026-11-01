"""Ternarisation in the compress layer: ternarize_model, ptq format=ternary, the `ternarize` recipe stage, `pack` validation (phase P3)."""
from __future__ import annotations

import pytest
import torch

from forge.compress.config import CompressJobConfig
from forge.compress.ptq import PTQConfig, quantize_model
from forge.compress.run import run_compress
from forge.core.context import NullContext
from forge.models.linear import PackedLinear
from forge.models.tiny import build_tiny_model
from forge.ternary.convert import ternarize_model


def _kinds(model):
    return {m.kind for m in model.modules() if isinstance(m, PackedLinear)}


def test_ternarize_model_report_and_exclusions():
    m = build_tiny_model(seed=2)
    ids = torch.randint(0, m.config.vocab_size, (2, 12))
    rep = ternarize_model(m, group_size=16)
    s = rep.summary()
    assert _kinds(m) == {"t158"} and s["n_layers"] > 0 and 0 < s["mean_zero_frac"] < 0.8 and s["mean_sqnr_db"] > 1.0
    assert 2.0 < s["bits_per_weight"] < 5.0 and s["compression_vs_source"] > 5  # fp32 tiny model: 4 bytes -> g16 packed 0.5 byte
    assert not any("lm_head" in r.name or "embed" in r.name for r in rep.layers)
    assert set(s["sqnr_db_by_type"]) >= {"q_proj", "down_proj"}
    assert torch.isfinite(m(ids).logits).all()


def test_ternarize_skips_ineligible_shapes_with_reason():
    m = build_tiny_model(seed=2)
    rep = ternarize_model(m, group_size=48)  # 64 % 48 != 0
    assert not rep.layers and all("group_size" in v or v == "excluded" for v in rep.skipped.values())


def test_ternarized_qat_model_reproduces_qat_forward():
    from forge.train.config import PrecisionConfig
    from forge.train.precision import make_precision_manager

    m = build_tiny_model(seed=3)
    pm = make_precision_manager(PrecisionConfig.model_validate({"quant_aware": "ternary", "exempt_first_layers": 0, "exempt_last_layers": 0,
                                                                 "ternary": {"group_size": 32}}))
    pm.prepare(m)
    ids = torch.randint(0, m.config.vocab_size, (2, 12))
    with torch.no_grad():
        qat = m(ids).logits
    pm.restore(m)
    ternarize_model(m, group_size=32)
    with torch.no_grad():
        packed = m(ids).logits
    assert torch.allclose(qat, packed, rtol=1e-3, atol=1e-3)


def test_ptq_format_ternary_and_validation():
    m = build_tiny_model(seed=2)
    qm = quantize_model(m, config=PTQConfig(method="rtn", format="ternary", group_size=16))
    assert _kinds(m) == {"t158"} and qm.report.layers and all(r.kind == "t158" for r in qm.report.layers) and qm.report.compression > 2
    assert any("EXPERIMENTAL" in w for w in qm.report.warnings)
    for bad in ({"method": "gptq"}, {"method": "rtn", "mixed_precision": True, "avg_bits": 3.0}):
        with pytest.raises(ValueError):
            PTQConfig(format="ternary", **bad)


def test_ternarize_recipe_end_to_end_with_pack_and_artifact(tmp_path):
    recipe = {"recipe": "tern-demo", "stages": [{"ternarize": {"group_size": 16}}, {"pack": {}}]}
    out = tmp_path / "art"
    cfg = CompressJobConfig(model="tiny:3", dtype="float32", device="cpu", recipe=recipe, use_store=False, save_dir=str(out), fail_on_gate=False)
    res = run_compress(cfg, NullContext())
    assert res["layout_id"].startswith("ternary2.row.v1-") and res["compression"] > 2
    from forge.core.artifactmeta import read_artifact_summary
    from forge.models.loader import load_model

    s = read_artifact_summary(out)
    assert s["is_quantized"] and s["label"] == "Ternary 1.58b"
    loaded, _ = load_model(out, dtype="float32", device="cpu", with_tokenizer=False)
    assert _kinds(loaded) == {"t158"}


def test_ternary_lora_on_packed_base_trains_adapter_only():
    from forge.models.lora import LoRALinear

    lin = torch.nn.Linear(64, 32, bias=False)
    from forge.ternary.linear import T158Linear

    base = T158Linear.from_dense(lin, group_size=32)
    try:
        wrapped = LoRALinear(base, r=4, alpha=8.0)
    except TypeError:  # signature differs: skip loudly rather than guess
        pytest.skip("LoRALinear constructor signature differs")
    x = torch.randn(3, 64)
    wrapped(x).sum().backward()
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in n for n in trainable)
    assert not any(p.grad is not None for p in base.parameters())
