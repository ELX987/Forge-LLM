"""Mixture-of-experts block: router, top-k routing, routed experts, optional shared expert, load-balancing aux loss.

Routed experts are separate :class:`~forge.models.layers.MLP` modules (``mlp.experts.<j>.{gate,up,down}_proj``) so that
quantisation / LoRA / checkpoint names work per expert.  The default compute path is a sort-by-expert loop in pure torch
(differentiable, runs anywhere).  For inference on CUDA, when every expert projection is a plain dense linear, the
``forge.kernels.ops_moe.grouped_gemm`` path stacks the expert weights (cached) and runs three grouped GEMMs; it is lazily
imported and falls back to the loop on any failure (**written, not validated on GPU hardware**; assumed signature
``grouped_gemm(x_sorted[M,K], expert_weights[E,N,K], offsets[E+1]) -> [M,N]`` with rows sorted by expert).
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.models.config import ModelConfig
from forge.models.layers import MLP, get_activation
from forge.models.linear import DenseLinear, make_linear

log = logging.getLogger("forge.models.moe")


def route_topk(logits: torch.Tensor, top_k: int, norm_topk_prob: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Softmax over experts, pick top-k. Returns ``(weights [N,k] fp32, ids [N,k], probs [N,E] fp32)``."""
    probs = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return w, ids, probs


def load_balancing_loss(probs: torch.Tensor, ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer auxiliary loss ``E * sum_e f_e * P_e`` (``f_e`` = fraction of routed slots to e, ``P_e`` = mean prob).

    Equals 1.0 for perfectly uniform routing; >1 when imbalanced.
    """
    n, k = ids.shape
    counts = torch.bincount(ids.reshape(-1), minlength=num_experts).to(probs.dtype)
    f = counts / float(n * k)
    p = probs.mean(dim=0)
    return num_experts * (f * p).sum()


class MoEBlock(nn.Module):
    """Sparse MoE FFN. ``forward`` returns ``(output [B,T,H], aux_loss scalar)``."""

    def __init__(self, cfg: ModelConfig, dtype=None, device=None):
        super().__init__()
        m = cfg.moe
        assert m is not None
        self.cfg_moe = m
        self.hidden = cfg.hidden_size
        self.num_experts, self.top_k, self.norm_topk = m.num_experts, m.top_k, m.norm_topk_prob
        self.gate = make_linear(cfg.hidden_size, m.num_experts, False, dtype=dtype, device=device)
        self.experts = nn.ModuleList(
            [MLP(cfg.hidden_size, m.moe_intermediate_size, cfg.hidden_act, dtype=dtype, device=device) for _ in range(m.num_experts)]
        )
        self.act = get_activation(cfg.hidden_act)
        if m.shared_size:
            self.shared_expert = MLP(cfg.hidden_size, m.shared_size, cfg.hidden_act, dtype=dtype, device=device)
            self.shared_expert_gate = make_linear(cfg.hidden_size, 1, False, dtype=dtype, device=device) if m.shared_expert_gate else None
        else:
            self.shared_expert = None
            self.shared_expert_gate = None
        self._stack_cache: tuple[tuple, dict[str, torch.Tensor]] | None = None
        self._grouped_ok = True
        self.last_expert_counts: torch.Tensor | None = None

    # ---------------------------------------------------------------- compute paths
    def _loop(self, x2: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n, k = ids.shape
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // k
        counts = torch.bincount(flat, minlength=self.num_experts)
        self.last_expert_counts = counts.detach()
        xs = x2[tok]
        ys = torch.empty_like(xs)
        offs = [0]
        for c in counts.tolist():
            offs.append(offs[-1] + c)
        outs = []
        for e in range(self.num_experts):
            s, t = offs[e], offs[e + 1]
            if t > s:
                outs.append(self.experts[e](xs[s:t]))
        ys = torch.cat(outs, dim=0) if outs else ys
        ys = ys * w.reshape(-1)[order].to(ys.dtype).unsqueeze(-1)
        return torch.zeros_like(x2).index_add(0, tok, ys)

    def _can_group(self, x2: torch.Tensor) -> bool:
        if not (self.cfg_moe.use_grouped_gemm and self._grouped_ok and x2.is_cuda and not torch.is_grad_enabled()):
            return False
        if os.environ.get("FORGE_FORCE_REF", "0") == "1":
            return False
        return all(type(l) is DenseLinear or type(l) is nn.Linear for e in self.experts for l in (e.gate_proj, e.up_proj, e.down_proj))

    def _stacked(self) -> dict[str, torch.Tensor]:
        key = tuple(l.weight._version for e in self.experts for l in (e.gate_proj, e.up_proj, e.down_proj)) + (
            self.experts[0].gate_proj.weight.data_ptr(),
        )
        if self._stack_cache is None or self._stack_cache[0] != key:
            st = {
                "gate": torch.stack([e.gate_proj.weight for e in self.experts]).contiguous(),
                "up": torch.stack([e.up_proj.weight for e in self.experts]).contiguous(),
                "down": torch.stack([e.down_proj.weight for e in self.experts]).contiguous(),
            }
            self._stack_cache = (key, st)
        return self._stack_cache[1]

    def _grouped(self, x2: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
        try:
            from forge.kernels import ops_moe

            n, k = ids.shape
            flat = ids.reshape(-1)
            order = torch.argsort(flat, stable=True)
            tok = order // k
            counts = torch.bincount(flat, minlength=self.num_experts)
            offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
            self.last_expert_counts = counts.detach()
            xs = x2[tok]
            st = self._stacked()
            g = ops_moe.grouped_gemm(xs, st["gate"], offsets)
            u = ops_moe.grouped_gemm(xs, st["up"], offsets)
            y = ops_moe.grouped_gemm(self.act(g) * u, st["down"], offsets)
            y = y * w.reshape(-1)[order].to(y.dtype).unsqueeze(-1)
            return torch.zeros_like(x2).index_add(0, tok, y)
        except Exception as e:
            if os.environ.get("FORGE_KERNEL_STRICT", "0") == "1":
                raise
            log.warning("ops_moe.grouped_gemm unavailable (%s); using loop path", e)
            self._grouped_ok = False
            return None

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, want_aux: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        logits = self.gate(x2)
        w, ids, probs = route_topk(logits, self.top_k, self.norm_topk)
        w = w.to(x2.dtype)
        out = self._grouped(x2, ids, w) if self._can_group(x2) else None
        if out is None:
            out = self._loop(x2, ids, w)
        if self.shared_expert is not None:
            sh = self.shared_expert(x2)
            if self.shared_expert_gate is not None:
                sh = sh * torch.sigmoid(self.shared_expert_gate(x2))
            out = out + sh
        aux = load_balancing_loss(probs, ids, self.num_experts) if (want_aux or self.training) else x.new_zeros((), dtype=torch.float32)
        return out.reshape(shp), aux
