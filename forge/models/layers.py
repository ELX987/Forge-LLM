"""Building blocks: RMSNorm, rotary embeddings (default / linear / llama3 / yarn), activations, SwiGLU MLP."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.core.errors import ConfigError
from forge.models.config import ModelConfig
from forge.models.linear import make_linear


# ------------------------------------------------------------------------------------------------ norm
class RMSNorm(nn.Module):
    """RMSNorm computed in fp32. ``unit_offset=True`` uses ``(1 + weight)`` (gemma)."""

    def __init__(self, dim: int, eps: float = 1e-6, unit_offset: bool = False, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device) if unit_offset else torch.ones(dim, dtype=dtype, device=device))
        self.eps = eps
        self.unit_offset = unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.unit_offset:
            return (xf * (1.0 + self.weight.float())).to(dt)
        return self.weight * xf.to(dt)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}, unit_offset={self.unit_offset}"


# ------------------------------------------------------------------------------------------------ rope
def compute_inv_freq(head_dim: int, theta: float, scaling: dict[str, Any] | None, max_position: int) -> tuple[torch.Tensor, float]:
    """Return ``(inv_freq[head_dim/2] float32, attention_factor)`` for the RoPE variant in ``scaling``.

    Supported ``rope_type``: ``None`` (plain), ``linear``, ``llama3`` (Llama-3.1 smooth interpolation), ``yarn``.
    """
    dim = head_dim
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
    inv_freq = 1.0 / pos_freqs
    att = 1.0
    if not scaling:
        return inv_freq, att
    rt = scaling.get("rope_type")
    factor = float(scaling.get("factor", 1.0))
    if rt == "linear":
        return inv_freq / factor, att
    if rt == "llama3":
        low_ff = float(scaling.get("low_freq_factor", 1.0))
        high_ff = float(scaling.get("high_freq_factor", 4.0))
        old_ctx = float(scaling.get("original_max_position_embeddings", 8192))
        low_wl = old_ctx / low_ff
        high_wl = old_ctx / high_ff
        wavelen = 2 * math.pi / inv_freq
        out = torch.where(wavelen > low_wl, inv_freq / factor, inv_freq)
        smooth = (old_ctx / wavelen - low_ff) / (high_ff - low_ff)
        smoothed = (1 - smooth) * out / factor + smooth * out
        medium = ~(wavelen < high_wl) & ~(wavelen > low_wl)
        return torch.where(medium, smoothed, out), att
    if rt == "yarn":
        orig = float(scaling.get("original_max_position_embeddings") or max_position)
        beta_fast = float(scaling.get("beta_fast", 32.0))
        beta_slow = float(scaling.get("beta_slow", 1.0))

        def get_mscale(scale: float, m: float = 1.0) -> float:
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

        af = scaling.get("attention_factor")
        if af is None:
            ms, msa = scaling.get("mscale"), scaling.get("mscale_all_dim")
            af = get_mscale(factor, ms) / get_mscale(factor, msa) if (ms and msa) else get_mscale(factor)
        att = float(af)

        def corr_dim(rot: float) -> float:
            return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(theta))

        low, high = corr_dim(beta_fast), corr_dim(beta_slow)
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low) / (high - low)).clamp(0, 1)
        extrap = 1.0 - ramp
        inv = (1.0 / (factor * pos_freqs)) * (1 - extrap) + inv_freq * extrap
        return inv, att
    raise ConfigError(f"unsupported rope_type {rt!r} (supported: linear, llama3, yarn)")


class RotaryEmbedding(nn.Module):
    """Produces ``cos, sin`` of shape ``[B, T, head_dim]`` (float32) for arbitrary integer positions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.head_dim = int(cfg.head_dim)  # type: ignore[arg-type]
        self.theta = float(cfg.rope_theta)
        self.scaling = cfg.rope_scaling
        self.max_pos = cfg.max_position_embeddings
        self._cache: dict[Any, tuple[torch.Tensor, float]] = {}
        self.inv_freq_cpu, self.attention_factor = compute_inv_freq(self.head_dim, self.theta, self.scaling, self.max_pos)

    def _inv(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        if key not in self._cache:
            self._cache[key] = (self.inv_freq_cpu.to(device), self.attention_factor)
        return self._cache[key][0]

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self._inv(positions.device)
        freqs = positions.float().unsqueeze(-1) * inv  # [B,T,D/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos() * self.attention_factor, emb.sin() * self.attention_factor


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q, k``: ``[B, T, H, D]``; ``cos, sin``: ``[B, T, D]``."""
    cos = cos.to(q.dtype).unsqueeze(2)
    sin = sin.to(q.dtype).unsqueeze(2)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ------------------------------------------------------------------------------------------------ mlp
def get_activation(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "gelu_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "relu":
        return F.relu
    raise ConfigError(f"unknown activation {name!r}")


class MLP(nn.Module):
    """Gated MLP ``down(act(gate(x)) * up(x))`` (SwiGLU for ``silu``)."""

    def __init__(self, hidden: int, inter: int, act: str = "silu", dtype=None, device=None):
        super().__init__()
        self.hidden_size, self.intermediate_size = hidden, inter
        self.gate_proj = make_linear(hidden, inter, False, dtype=dtype, device=device)
        self.up_proj = make_linear(hidden, inter, False, dtype=dtype, device=device)
        self.down_proj = make_linear(inter, hidden, False, dtype=dtype, device=device)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
