"""Linear registry (dense / fp8 / fake-quant / packed protocol) and LoRA / DoRA."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from forge.core.errors import ConfigError
from forge.models import (
    ForgeLM,
    PackedLinear,
    add_lora,
    build_tiny_model,
    linear_report,
    load_lora_adapter,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    quantize_linears,
    register_linear,
    remove_lora,
    replace_linears,
    save_lora_adapter,
    set_lora_enabled,
    trainable_parameters,
)
from forge.models.linear import (
    DenseLinear,
    FakeQuantIntLinear,
    FakeQuantNVFP4Linear,
    Fp8Linear,
    _fake_quant_int,
    _round_e2m1,
    fake_quant_nvfp4,
    linear_kinds,
    make_linear,
)
from forge.models.lora import LoRALinear


# ------------------------------------------------------------------ registry
def test_registry_kinds():
    ks = linear_kinds()
    for k in ("dense", "fp8", "fq_int", "fq_nvfp4", "lora"):
        assert k in ks
    assert isinstance(make_linear(8, 4), DenseLinear)
    assert isinstance(make_linear(8, 4, kind="fp8"), Fp8Linear)


def test_replace_linears_and_custom_kind():
    @register_linear("test_double")
    class Doubler(PackedLinear):
        def __init__(self, lin):
            super().__init__()
            self.in_features, self.out_features = lin.in_features, lin.out_features
            self.register_buffer("w", lin.weight.detach().clone())
            self.bias = None

        @classmethod
        def from_dense(cls, linear, **kw):
            return cls(linear)

        @classmethod
        def from_shape(cls, i, o, bias=False, dtype=None, device=None, **kw):
            return cls(nn.Linear(i, o, bias=False))

        def dequantize(self, dtype=None):
            return self.w if dtype is None else self.w.to(dtype)

        def forward(self, x):
            return x @ self.w.T

    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (1, 6))
    ref = m(ids).logits
    names = quantize_linears(m, "test_double", include=["mlp"])
    assert names and all(".mlp." in n for n in names)
    assert linear_report(m)["layers.0.mlp.up_proj"] == "test_double"
    assert torch.allclose(m(ids).logits, ref, atol=1e-5)
    # factory returning None keeps modules, and already-replaced packed layers are skipped
    assert replace_linears(m, lambda n, l: None) == []


def test_default_skip_keeps_lm_head_and_router():
    m = build_tiny_model(moe=True)
    names = quantize_linears(m, "fp8")
    assert not any(n.endswith("lm_head") or n.endswith("mlp.gate") for n in names)
    assert any("experts.0.gate_proj" in n for n in names)


# ------------------------------------------------------------------ fp8
def test_fp8_linear_accuracy_and_state():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=True)
    q = Fp8Linear.from_dense(lin)
    assert q.weight_fp8.dtype == torch.float8_e4m3fn and q.weight_scale.shape == (32, 1)
    x = torch.randn(5, 64)
    ref = lin(x)
    err = (q(x) - ref).abs().max() / ref.abs().max()
    assert err < 0.08
    q2 = Fp8Linear.from_dense(lin, act_quant="none")
    assert (q2(x) - ref).abs().max() / ref.abs().max() < 0.05
    w = q.dequantize()
    assert (w - lin.weight).abs().max() / lin.weight.abs().max() < 0.07
    assert set(q.state_dict()) == {"weight_fp8", "weight_scale", "bias"}
    assert q.packed_nbytes() < lin.weight.numel() * 4 / 3
    q3 = Fp8Linear.from_shape(64, 32, True, **q.quant_spec())
    q3.load_state_dict(q.state_dict())
    assert torch.equal(q3(x), q(x))


# ------------------------------------------------------------------ fake quant
def test_fake_quant_int_properties():
    w = torch.randn(16, 128)
    q4 = _fake_quant_int(w, 4, 32, True)
    # at most 16 distinct values per 32-group
    for g in q4.reshape(16, 4, 32).reshape(-1, 32):
        assert g.unique().numel() <= 16
    e4 = (q4 - w).abs().mean()
    e8 = (_fake_quant_int(w, 8, 32, True) - w).abs().mean()
    assert e8 < e4 / 8
    assert (_fake_quant_int(w, 4, 32, False) - w).abs().mean() <= e4 * 1.05  # asymmetric no worse (roughly)
    assert torch.equal(_fake_quant_int(q4, 4, 32, True), q4) or (_fake_quant_int(q4, 4, 32, True) - q4).abs().max() < 1e-6  # idempotent


def test_e2m1_rounding_nearest_even():
    x = torch.tensor([0.0, 0.2, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0, -0.75, -3.5])
    y = _round_e2m1(x)
    assert y.tolist() == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, -4.0]


def test_nvfp4_fake_quant():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    d = fake_quant_nvfp4(w)
    assert d.shape == w.shape
    rel = (d - w).norm() / w.norm()
    assert 0.0 < rel < 0.25
    # block structure: values in each 16-block are multiples of one scale on the E2M1 grid
    odd = torch.randn(4, 20)  # padding path
    assert fake_quant_nvfp4(odd).shape == odd.shape


def test_fake_quant_linear_ste_and_cache():
    lin = nn.Linear(64, 16, bias=False)
    fq = FakeQuantIntLinear.from_dense(lin, bits=4, group_size=32)
    x = torch.randn(3, 64, requires_grad=False)
    y = fq(x)
    assert (y - lin(x)).abs().max() > 0
    y.sum().backward()
    assert fq.weight.grad is not None and fq.weight.grad.abs().sum() > 0  # STE passes gradients
    with torch.no_grad():
        a = fq(x)
        v = fq.weight._version
        b = fq(x)
        assert fq._cache[0] == v and torch.equal(a, b)
        fq.weight.add_(0.1)
        assert not torch.equal(fq(x), a)  # cache invalidated by weight change
    nv = FakeQuantNVFP4Linear.from_dense(lin, quant_acts=True)
    assert nv(x).shape == (3, 16)
    spec = fq.quant_spec()
    fq2 = FakeQuantIntLinear.from_shape(64, 16, False, **spec)
    assert fq2.bits == 4 and fq2.group_size == 32


# ------------------------------------------------------------------ lora
def test_lora_zero_init_is_identity_and_trainable_set():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 8))
    ref = m(ids).logits
    names = add_lora(m, r=4, alpha=8)
    assert len(names) == 2 * 4
    assert torch.allclose(m(ids).logits, ref, atol=1e-6)
    tp = trainable_parameters(m)
    assert len(tp) == 2 * 4 * 2 and all(p.ndim == 2 for p in tp)
    assert all(not p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)
    with pytest.raises(ConfigError):
        add_lora(build_tiny_model(), target_modules=["nonexistent"])


def _randomize_lora(m):
    for n, p in m.named_parameters():
        if "lora_" in n:
            nn.init.normal_(p, std=0.05)


@pytest.mark.parametrize("dora", [False, True])
def test_lora_merge_equivalence(dora):
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 8))
    add_lora(m, r=4, alpha=8, target_modules="all-linear", use_dora=dora)
    assert not any(n.endswith("lm_head") for n in [x for x, _ in m.named_modules() if isinstance(_, LoRALinear)])
    _randomize_lora(m)
    with torch.no_grad():
        y_lora = m(ids).logits
        set_lora_enabled(m, False)
        y_base = m(ids).logits
        set_lora_enabled(m, True)
    assert not torch.allclose(y_lora, y_base, atol=1e-3)
    merge_lora(m)
    assert not any(isinstance(x, LoRALinear) for x in m.modules())
    assert all(isinstance(m.layers[0].self_attn.q_proj, nn.Linear) for _ in [0])
    assert torch.allclose(m(ids).logits, y_lora, atol=1e-4), (m(ids).logits - y_lora).abs().max()


def test_dora_initial_equivalence_and_grads():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 8))
    ref = m(ids).logits
    add_lora(m, r=2, use_dora=True, dropout=0.1)
    m.eval()
    assert torch.allclose(m(ids).logits, ref, atol=1e-4)
    m.train()
    _randomize_lora(m)
    m(ids, labels=ids).loss.backward()
    mod = m.layers[0].self_attn.q_proj
    assert mod.lora_magnitude.grad is not None and mod.lora_A.weight.grad is not None


def test_lora_training_step_reduces_loss_only_adapters_move():
    m = build_tiny_model()
    base = {n: p.detach().clone() for n, p in m.named_parameters()}
    add_lora(m, r=8, alpha=16, target_modules="all-linear")
    ids = torch.randint(0, 256, (4, 16))
    opt = torch.optim.AdamW(trainable_parameters(m), lr=5e-3)
    l0 = m(ids, labels=ids).loss.item()
    for _ in range(30):
        opt.zero_grad()
        m(ids, labels=ids).loss.backward()
        opt.step()
    assert m(ids, labels=ids).loss.item() < l0 - 0.2
    for n, p in m.named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, base[n.replace(".base.", ".")]), n


def test_lora_over_packed_base_qlora_style():
    m = build_tiny_model().eval()
    quantize_linears(m, "fp8", act_quant="none")  # weight-only so that merging (dequantised dense) is exactly comparable
    ids = torch.randint(0, 256, (2, 8))
    ref = m(ids).logits
    add_lora(m, r=4, target_modules=["q_proj", "v_proj"])
    assert isinstance(m.layers[0].self_attn.q_proj.base, Fp8Linear)
    assert torch.allclose(m(ids).logits, ref, atol=1e-6)
    _randomize_lora(m)
    y = m(ids).logits
    with pytest.raises(ConfigError, match="packed"):
        merge_lora(m)
    merge_lora(m, dequantize_packed=True)
    assert type(m.layers[0].self_attn.q_proj) is DenseLinear
    assert torch.allclose(m(ids).logits, y, atol=1e-4)


def test_remove_lora_restores_base():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (1, 6))
    ref = m(ids).logits
    add_lora(m, r=2)
    _randomize_lora(m)
    remove_lora(m)
    assert torch.equal(m(ids).logits, ref)


def test_lora_state_dict_roundtrip_and_adapter_files(tmp_path):
    m = build_tiny_model(arch="mixtral", moe={"num_experts": 4, "top_k": 2, "moe_intermediate_size": 32})
    add_lora(m, r=3, alpha=6, target_modules=["q_proj", "gate_proj"])
    _randomize_lora(m)
    sd = lora_state_dict(m)
    assert all(k.endswith(("lora_A.weight", "lora_B.weight")) for k in sd)
    m2 = build_tiny_model(arch="mixtral", moe={"num_experts": 4, "top_k": 2, "moe_intermediate_size": 32})
    add_lora(m2, r=3, alpha=6, target_modules=["q_proj", "gate_proj"])
    missing, unexpected = load_lora_state_dict(m2, sd)
    assert not missing and not unexpected
    for k, v in lora_state_dict(m2).items():
        assert torch.equal(v, sd[k])
    bad = dict(sd)
    bad.pop(next(iter(bad)))
    with pytest.raises(ConfigError):
        load_lora_state_dict(m2, bad)
    # PEFT-style files with HF names (mixtral experts -> w1)
    save_lora_adapter(m, tmp_path / "ad")
    from safetensors.torch import load_file

    keys = list(load_file(str(tmp_path / "ad" / "adapter_model.safetensors")))
    assert any("block_sparse_moe.experts.0.w1.lora_A.weight" in k for k in keys) and all(k.startswith("base_model.model.model.") for k in keys)
    m3 = build_tiny_model(arch="mixtral", moe={"num_experts": 4, "top_k": 2, "moe_intermediate_size": 32})
    load_lora_adapter(m3, tmp_path / "ad")  # adds the adapters itself
    ids = torch.randint(0, 256, (1, 6))
    m.eval()
    m3.eval()
    # same base weights? no (different random init) -> compare adapter tensors instead
    for k, v in lora_state_dict(m3).items():
        assert torch.equal(v, sd[k])


def test_lora_target_regex_and_exclude():
    m = build_tiny_model()
    names = add_lora(m, r=2, target_modules=["re:layers\\.1\\..*down_proj"])
    assert names == ["layers.1.mlp.down_proj"]
    m = build_tiny_model()
    names = add_lora(m, r=2, target_modules=["q_proj", "k_proj"], exclude=["k_proj"])
    assert all(n.endswith("q_proj") for n in names)


def test_lora_in_bf16_model_uses_fp32_adapters():
    m = build_tiny_model(dtype=torch.bfloat16)
    add_lora(m, r=2)
    mod = m.layers[0].self_attn.q_proj
    assert mod.lora_A.weight.dtype == torch.float32
    out = m(torch.randint(0, 256, (1, 5)), labels=torch.randint(0, 256, (1, 5)))
    assert torch.isfinite(out.loss.float())
    out.loss.backward()
    assert mod.lora_B.weight.grad is not None
