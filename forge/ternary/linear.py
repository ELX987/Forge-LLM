"""Ternary linear layers: ``t158`` (packed, inference) and ``fq_t158`` (latent-weight QAT with a straight-through estimator).

* ``t158``: buffers ``qweight uint8 [N, K/4]`` (2-bit codes, layout ``ternary2_row``) and ``scales fp32 [N, G]``; forward quantises
  activations per token to int8 and runs the ternary GEMM (oracle path here; the Triton / native kernels dispatch from
  ``forge.kernels.ops_gemm`` in phase P4).  Not a QAT module: all tensors are buffers.
* ``fq_t158``: an ``nn.Linear`` with a latent fp ``weight`` Parameter; forward uses ``Q(W)`` (scale recomputed every step, never learned,
  detached) and int8 per-token activations, with the optional HF-style warm-up blend ``W_eff = W + lam * (Q(W) - W)``.
  ``T158Linear.from_dense(fq_layer)`` converts a trained QAT layer to the packed form.
"""
from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.core.errors import ConfigError
from forge.formats import ternary as tn
from forge.kernels.layouts.ternary import TERNARY2_ROW, PackedTernary, pack_ternary_weight
from forge.kernels.ref.gemm_ternary import ternary_linear_ref
from forge.models.linear import PackedLinear, _FakeQuantLinear, register_linear


def _check_dims(in_features: int, group_size: int) -> None:
    if in_features % tn.CODES_PER_WORD:
        raise ConfigError(f"ternary layers need in_features % {tn.CODES_PER_WORD} == 0, got {in_features}")
    if group_size > 0 and in_features % group_size:
        raise ConfigError(f"group_size {group_size} must divide in_features {in_features}")


def _num_groups(in_features: int, group_size: int) -> int:
    return 1 if group_size == 0 else in_features // tn.effective_group(group_size, in_features)


@register_linear("t158")
class T158Linear(PackedLinear):
    """Packed ternary weights (2 bits each) + fp32 group scales; per-token int8 activations."""

    kind: ClassVar[str] = "t158"

    def __init__(self, in_features: int, out_features: int, bias: bool = False, *, group_size: int = 128, dtype: torch.dtype | None = None, device: Any = None) -> None:
        super().__init__()
        _check_dims(in_features, group_size)
        self.in_features, self.out_features, self.group_size = in_features, out_features, int(group_size)
        g = _num_groups(in_features, group_size)
        shape = (1, 1) if group_size == 0 else (out_features, g)
        self.register_buffer("qweight", torch.zeros(out_features, in_features // 4, dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.zeros(*shape, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype or torch.float32, device=device), requires_grad=False) if bias else None

    @classmethod
    def from_packed(cls, p: PackedTernary, bias: torch.Tensor | None = None, dtype: torch.dtype | None = None) -> T158Linear:
        m = cls(p.k, p.n, bias is not None, group_size=p.group_size, dtype=dtype or (bias.dtype if bias is not None else None), device=p.qweight.device)
        m.qweight.copy_(p.qweight)
        m.scales.copy_(p.scales)
        if bias is not None:
            m.bias.data.copy_(bias.detach().to(m.bias.dtype))
        return m

    @classmethod
    def from_dense(cls, linear: nn.Linear, group_size: int = 128, rule: str = "lloyd", iters: int = 1, **_: Any) -> T158Linear:
        w = linear.weight.detach()
        _check_dims(w.shape[1], group_size)
        t, s = tn.quantize_ternary(w, group_size, rule, iters)
        return cls.from_packed(pack_ternary_weight(t, s, group_size), linear.bias, dtype=w.dtype if linear.bias is None else linear.bias.dtype)

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, **kw: Any) -> T158Linear:
        kw = {k: v for k, v in kw.items() if k in ("group_size",)}
        return cls(in_features, out_features, bias, dtype=dtype, device=device, **kw)

    def packed(self) -> PackedTernary:
        return PackedTernary(self.qweight, self.scales, TERNARY2_ROW, self.out_features, self.in_features, self.group_size)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.packed().dequantize(torch.float32)
        return w.to(dtype) if dtype is not None else w

    def quant_spec(self) -> dict[str, Any]:
        return {"group_size": self.group_size, "layout": TERNARY2_ROW.id}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from forge.kernels.ops_gemm import ternary_linear

        return ternary_linear(x, self.qweight, self.scales, self.group_size, self.in_features, None if self.bias is None else self.bias.to(x.dtype))


@register_linear("fq_t158")
class FakeQuantTernaryLinear(_FakeQuantLinear):
    """Latent-weight ternary QAT: weights -> ``Q(W)`` ({-1,0,1} * scale), activations -> per-token int8, STE backward (scale detached)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, *, dtype=None, device=None, group_size: int = 128,
                 rule: str = "lloyd", iters: int = 1, quant_acts: bool = True):
        super().__init__(in_features, out_features, bias, dtype=dtype, device=device, quant_acts=quant_acts)
        _check_dims(in_features, group_size)
        if rule not in tn.RULES:
            raise ConfigError(f"unknown ternary rule {rule!r}; expected one of {tn.RULES}")
        self.group_size, self.rule, self.iters = int(group_size), rule, int(iters)
        self.lam = 1.0
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, 'enabled', True):  # high-precision shadow/eval forward
            return F.linear(x, self.weight, self.bias)
        return super().forward(x)

    def set_lam(self, lam: float) -> None:
        """Warm-up blend in [0, 1]: 0 = full precision, 1 = fully quantised."""
        self.lam = float(min(1.0, max(0.0, lam)))
        self._cache = None

    def _wq(self, w: torch.Tensor) -> torch.Tensor:
        q = tn.fake_quant_ternary(w, self.group_size, self.rule, self.iters)
        return q if self.lam >= 1.0 else w + self.lam * (q - w)

    def _aq(self, x: torch.Tensor) -> torch.Tensor:
        q = tn.fake_quant_act_int8(x)
        return q if self.lam >= 1.0 else x + self.lam * (q - x)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = tn.fake_quant_ternary(self.weight.detach(), self.group_size, self.rule, self.iters)
        return w.to(dtype) if dtype is not None else w

    def quant_spec(self) -> dict[str, Any]:
        return {"group_size": self.group_size, "rule": self.rule, "iters": self.iters, "quant_acts": self.quant_acts}

    def stats(self) -> dict[str, float]:
        """Ternary health stats of the current latent weights: fractions of -1/0/+1."""
        t, _ = tn.quantize_ternary(self.weight.detach(), self.group_size, self.rule, self.iters)
        return tn.code_stats(t)
