"""Accounting helpers, lazy exports, fallbacks when sibling modules are absent."""
from __future__ import annotations

import importlib

import torch

import forge.models as fm
from forge.models import build_tiny_model, count_parameters, model_summary, quantize_linears


def test_lazy_exports_resolve():
    for name in fm.__all__:
        assert getattr(fm, name) is not None, name
    assert "ForgeLM" in dir(fm)
    try:
        fm.does_not_exist
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")


def test_model_summary_plan_numbers():
    from forge.models import ModelConfig

    c = ModelConfig(arch="llama", vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_layers=32, num_heads=32, num_kv_heads=8)
    s = model_summary(c)
    assert s["kv_bytes_per_token"]["16"] == 131072 and s["kv_bytes_per_token"]["8"] == 65536
    assert abs(s["params_b"] - 8.03) < 0.01 and s["provenance"] == "derived"
    assert s["weight_bytes"]["16"] == 2 * s["params"] and s["flops_per_token"]["train"] > s["flops_per_token"]["prefill"]


def test_count_parameters_packed_bytes():
    m = build_tiny_model()
    base = count_parameters(m)
    assert base["total"] == m.config.param_count() and base["trainable"] == base["total"]
    quantize_linears(m, "fp8")
    c = count_parameters(m)
    assert c["bytes_fp8"] > 0 and c["total"] < base["total"]


def test_all_modules_import_without_optional_deps():
    for mod in ("config", "linear", "layers", "kvcache", "attention", "moe", "modeling", "lora", "loader", "tiny", "accounting"):
        importlib.import_module(f"forge.models.{mod}")


def test_fake_quant_works_without_formats_module(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "forge.formats.fp4", None)  # force ImportError path
    from forge.models.linear import fake_quant_nvfp4

    w = torch.randn(8, 32)
    assert fake_quant_nvfp4(w).shape == w.shape


def test_fake_quant_prefers_formats_when_available(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.ModuleType("forge.formats.fp4")
    fake.quantize_nvfp4 = lambda x, **k: calls.append("q") or x
    fake.dequantize_nvfp4 = lambda t, **k: t * 0.5
    monkeypatch.setitem(sys.modules, "forge.formats.fp4", fake)
    import forge.formats as ff

    monkeypatch.setattr(ff, "fp4", fake, raising=False)
    from forge.models.linear import fake_quant_nvfp4

    w = torch.ones(4, 16)
    assert torch.allclose(fake_quant_nvfp4(w), w * 0.5) and calls == ["q"]
