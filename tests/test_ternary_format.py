"""Ternary format, layout, oracle and linear kinds (phase P1). Golden vectors were generated from tern158's ``ref_ops`` (sm_120 selftest-validated
semantics): ``tests/data/ternary_golden.safetensors``."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from forge.formats import ternary as tn
from forge.formats.packing import pack_bits
from forge.kernels.layouts.ternary import TERNARY2_ROW, PackedTernary, pack_ternary_weight, quantize_pack_ternary
from forge.kernels.ref.gemm_ternary import ternary_bwd_data_ref, ternary_gemm_ref, ternary_linear_ref
from forge.models.linear import get_linear_cls, linear_kinds, quantization_manifest, apply_quantization_manifest, quantize_linears

GOLD = load_file(str(Path(__file__).parent / "data" / "ternary_golden.safetensors"))
CASES = [(r, g) for g in (0, 32, 64) for r in ("absmean", "twn", "lloyd")]


def _iters(rule: str) -> int:
    return 2 if rule == "lloyd" else 1


def test_pack_roundtrip_and_reserved_code():
    t = torch.randint(-1, 2, (6, 80), dtype=torch.int8)
    p = tn.pack_ternary(t)
    assert p.dtype == torch.uint8 and p.shape == (6, 20)
    assert torch.equal(tn.unpack_ternary(p, 80), t)
    assert torch.equal(p, pack_bits((t.to(torch.int16) & 3).to(torch.uint8), 2))
    assert tn.unpack_ternary(torch.tensor([[0b10]], dtype=torch.uint8), 1).item() == 0  # reserved code 10 decodes to 0


def test_k_must_be_multiple_of_16():
    with pytest.raises(Exception):
        tn.pack_ternary(torch.zeros(2, 20, dtype=torch.int8))


@pytest.mark.parametrize("rule,group", CASES)
def test_quantizer_matches_tern158_golden(rule, group):
    w = GOLD["w"]
    t, s = tn.quantize_ternary(w, group, rule, _iters(rule))
    key = f"{rule}_g{group}"
    assert torch.equal(t, GOLD[key + "_t"])
    assert torch.allclose(s.reshape(-1), GOLD[key + "_scales"], rtol=1e-6, atol=0)
    # byte-identical to tern158's int32 words
    words = tn.pack_words_int32(tn.pack_ternary(t))
    assert torch.equal(words, GOLD[key + "_words"])
    assert torch.equal(tn.words_to_packed(GOLD[key + "_words"], 8, 64), tn.pack_ternary(t))


@pytest.mark.parametrize("rule,group", CASES)
def test_gemm_oracle_matches_tern158_golden(rule, group):
    xq, sx = tn.act_quant_int8(GOLD["x"])
    assert torch.equal(xq, GOLD["xq"]) and torch.allclose(sx, GOLD["sx"], rtol=0, atol=0)
    key = f"{rule}_g{group}"
    qw = tn.words_to_packed(GOLD[key + "_words"], 8, 64)
    scales = GOLD[key + "_scales"].reshape(1, 1) if group == 0 else GOLD[key + "_scales"].reshape(8, -1)
    y = ternary_gemm_ref(xq, sx, qw, scales, group, 64)
    assert torch.allclose(y, GOLD[key + "_y"], rtol=1e-6, atol=1e-7)


def test_lloyd_beats_absmean_and_dead_segment():
    w = torch.randn(32, 256) * 0.02
    e = {r: float((tn.dequantize_ternary(*tn.quantize_ternary(w, 128, r, 1), 128) - w).pow(2).mean()) for r in ("absmean", "lloyd")}
    assert e["lloyd"] <= e["absmean"]
    t, s = tn.quantize_ternary(torch.zeros(2, 32), 32, "lloyd")
    assert not t.any() and not s.any()


def test_scale_modes_shapes_and_bits():
    w = torch.randn(4, 64)
    assert tn.quantize_ternary(w, 0)[1].shape == (1, 1)
    assert tn.quantize_ternary(w, -1)[1].shape == (4, 1)
    assert tn.quantize_ternary(w, 16)[1].shape == (4, 4)
    assert tn.bits_per_weight(4096, 128) == pytest.approx(2.25) and tn.bits_per_weight(4096, 128, packed=False) == pytest.approx(1.585 + 0.25, abs=1e-3)


def test_helpers():
    assert tn.predicted_flip_rate(3e-4, 0.1) == pytest.approx(0.59 * (2 * 3e-4 * 0.1) ** 0.5)
    assert tn.warmup_lambda(0, 100) == 0 and tn.warmup_lambda(50, 100) == 0.5 and tn.warmup_lambda(500, 100) == 1 and tn.warmup_lambda(3, 0) == 1


def test_packed_layout_roundtrip_and_state():
    w = torch.randn(16, 128)
    p = quantize_pack_ternary(w, 64)
    assert p.layout_id == TERNARY2_ROW.id and p.layout_id.startswith("ternary2.row.v1-")
    q = PackedTernary.from_tensors(p.tensors(), p.state())
    assert torch.equal(q.qweight, p.qweight) and torch.equal(q.dequantize(), p.dequantize())
    assert p.bits_per_weight == pytest.approx(2.0 + 32 / 64)
    t, s = tn.quantize_ternary(w, 64)
    assert torch.equal(pack_ternary_weight(t, s, 64).qweight, p.qweight)


def test_bwd_data_oracle():
    w = torch.randn(8, 64)
    p = quantize_pack_ternary(w, 32)
    dy = torch.randn(3, 8)
    assert torch.allclose(ternary_bwd_data_ref(dy, p.qweight, p.scales, 32, 64), dy @ p.dequantize(), rtol=1e-5, atol=1e-6)


def test_t158_linear_matches_oracle_and_dequant():
    lin = torch.nn.Linear(64, 24, bias=True)
    cls = get_linear_cls("t158")
    m = cls.from_dense(lin, group_size=32)
    x = torch.randn(5, 64)
    q, s = tn.act_quant_int8(x)
    ref = (tn.act_dequant(q, s) @ m.dequantize().t()) + lin.bias
    assert torch.allclose(m(x), ref, rtol=1e-4, atol=1e-4)
    assert m.packed_nbytes() > 0 and m.quant_spec()["group_size"] == 32
    # quality: quantised layer approximates the dense one reasonably for gaussian weights
    assert float((m(x) - lin(x)).norm() / lin(x).norm()) < 0.9


def test_fq_t158_ste_and_conversion():
    lin = torch.nn.Linear(64, 16, bias=False)
    fq = get_linear_cls("fq_t158").from_dense(lin, group_size=32)
    x = torch.randn(4, 64, requires_grad=True)
    fq(x).sum().backward()
    assert fq.weight.grad is not None and fq.weight.grad.abs().sum() > 0 and x.grad is not None
    packed = get_linear_cls("t158").from_dense(fq, group_size=32)
    xe = torch.randn(4, 64)
    assert torch.allclose(packed(xe), fq.eval()(xe), rtol=1e-4, atol=1e-4)
    fq.set_lam(0.0)
    assert torch.allclose(fq(xe), torch.nn.functional.linear(xe, fq.weight), atol=1e-6)
    st = fq.stats()
    assert abs(sum(st.values()) - 1) < 1e-6


def test_registered_and_manifest_roundtrip():
    assert "t158" in linear_kinds() or get_linear_cls("t158")
    model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False), torch.nn.Linear(32, 16, bias=False))
    quantize_linears(model, "t158", group_size=16)
    man = quantization_manifest(model)
    assert {e["kind"] for e in man.values()} == {"t158"} and man["0"]["spec"]["group_size"] == 16
    fresh = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False), torch.nn.Linear(32, 16, bias=False))
    apply_quantization_manifest(fresh, man)
    fresh.load_state_dict(model.state_dict())
    x = torch.randn(3, 64)
    assert torch.equal(fresh(x), model(x))


def test_format_registry_and_label():
    from forge.core.artifactmeta import quant_label
    from forge.formats.spec import fake_quantize, format_info

    assert format_info("ternary", 128).bits_per_element == pytest.approx(1.585 + 0.25, abs=1e-3)
    w = torch.randn(4, 64)
    assert torch.equal(fake_quantize(w, "ternary", group_size=32), tn.fake_quant_ternary(w, 32))
    assert quant_label({"t158": 3}, {}) == "Ternary 1.58b"
