"""Liquid LFM2 gated short convolution (``ShortConv``): ``y = out_proj(C * conv1d(B * x))`` with ``B, C, x = in_proj(h).chunk(3)``.

The depthwise causal conv has ``conv_kernel`` taps, so a sequence only needs the last ``conv_kernel - 1`` values of ``u = B * x``
of every conv layer to continue: that ``[hidden, K-1]`` tensor is the layer's recurrent state.  The same code path serves training
and cache-less forward (zero left context), chunked prefill and token-by-token decode (the cache supplies / stores the state).

Cache protocol (:class:`~forge.models.kvcache.DenseKVCache` and :class:`~forge.models.kvcache.PagedKVView`)::

    prev = cache.get_conv(layer, dtype)          # [B, H, K-1] state before this forward (zeros for a fresh sequence)
    cache.set_conv(layer, new_state)             # state after the valid new tokens (rows with no new token pass `prev` back)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.models.config import ModelConfig
from forge.models.linear import make_linear


def conv_state_after(z: torch.Tensor, n_new: torch.Tensor | None, k1: int) -> torch.Tensor:
    """Last ``k1`` columns of ``z [B,H,k1+T]`` ending at each row's last VALID new token (right padding): ``z[:, :, n : n + k1]``."""
    b, h, _ = z.shape
    if n_new is None:
        return z[:, :, z.shape[-1] - k1 :]
    if isinstance(n_new, int):
        n_new = torch.full((b,), n_new, dtype=torch.long)
    idx = n_new.to(z.device, torch.long).view(b, 1, 1) + torch.arange(k1, device=z.device).view(1, 1, k1)
    return z.gather(2, idx.expand(b, h, k1))


class ShortConv(nn.Module):
    """Parameter names equal the HF ones (``in_proj``, ``conv.weight [H,1,K]``, ``out_proj``) so no renaming is needed inside ``conv.*``."""

    def __init__(self, cfg: ModelConfig, layer_idx: int, dtype=None, device=None):
        super().__init__()
        self.layer_idx = layer_idx
        h, k = cfg.hidden_size, cfg.conv_kernel
        self.hidden_size, self.kernel = h, k
        self.in_proj = make_linear(h, 3 * h, cfg.conv_bias, dtype=dtype, device=device)
        self.out_proj = make_linear(h, h, cfg.conv_bias, dtype=dtype, device=device)
        self.conv = nn.Conv1d(h, h, k, groups=h, bias=cfg.conv_bias, padding=0, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        b, t, h = x.shape
        k1 = self.kernel - 1
        if ctx.attention_mask is not None:  # padded positions contribute nothing (HF apply_mask_to_padding_states)
            x = x * ctx.attention_mask.to(x.dtype).unsqueeze(-1)
        bb, cc, xx = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)  # each [B,H,T]
        u = bb * xx
        cache = ctx.cache
        if cache is not None and k1 > 0:
            prev = cache.get_conv(self.layer_idx, u.dtype).to(u.device)
        else:
            prev = u.new_zeros(b, h, k1)
        z = torch.cat([prev, u], dim=-1) if k1 > 0 else u
        w = self.conv.weight
        y = F.conv1d(z.to(w.dtype), w, self.conv.bias, groups=h).to(x.dtype)  # [B,H,T], no activation
        if cache is not None and k1 > 0:
            cache.set_conv(self.layer_idx, conv_state_after(z, ctx.n_new, k1))
        return self.out_proj((cc * y).transpose(1, 2))
