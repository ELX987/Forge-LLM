"""Attention module with pluggable backends.

* training / prefill without cache: ``torch.nn.functional.scaled_dot_product_attention`` (flash/mem-efficient/math chosen by
  torch), or ``flash_attn`` when ``attn_impl='flash_attn'`` and the package + a CUDA half-precision tensor are present;
* decode / chunked prefill with a :class:`~forge.models.kvcache.DenseKVCache` or :class:`~forge.models.kvcache.PagedKVView`:
  the cache returns all keys/values, a boolean mask encodes causality, ragged lengths and the sliding window;
* single-token decode on a paged cache on CUDA: ``forge.kernels.ops_attn.paged_decode_attention`` (lazy import, assumed
  contract documented in :func:`Attention._paged_kernel`; falls back to the gather+SDPA path on any failure unless
  ``FORGE_KERNEL_STRICT=1``).  **Written but not validated on GPU hardware.**
* ``attn_impl='eager'``: explicit matmul/softmax reference (fp32 softmax); used as an oracle in tests.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.models.config import ModelConfig
from forge.models.kvcache import PagedKVView, is_quantised
from forge.models.layers import RMSNorm, apply_rotary
from forge.models.linear import make_linear

log = logging.getLogger("forge.models.attention")


@dataclass
class AttnMeta:
    """Optional per-call attention metadata (``attn_meta`` argument of ``ForgeLM.forward``).

    attention_mask: ``[B, T]`` bool/int, 1 = real token among the *new* tokens (right padding only when a cache is used).
    n_new: ``[B]`` valid new tokens per sequence (derived from ``attention_mask`` when absent).
    use_paged_kernel: force (True) / forbid (False) the ops_attn paged decode kernel; None = auto (CUDA only).
    """

    attention_mask: torch.Tensor | None = None
    n_new: torch.Tensor | None = None
    use_paged_kernel: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttnContext:
    """Resolved per-forward state shared by all layers."""

    positions: torch.Tensor  # [B,T]
    cache: Any | None = None
    attention_mask: torch.Tensor | None = None
    n_new: torch.Tensor | None = None
    kv_lens: torch.Tensor | None = None  # [B] total valid keys after this call (cache only)
    impl: str = "auto"
    use_paged_kernel: bool | None = None


_GQA_NATIVE = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5)


def sdpa(q, k, v, mask, is_causal, scale):
    """``q [B,T,H,D]``, ``k/v [B,S,Hkv,D]`` -> ``[B,T,H,D]`` (GQA aware)."""
    h, hkv = q.shape[2], k.shape[2]
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    kw: dict[str, Any] = {}
    if hkv != h:
        if _GQA_NATIVE and q.is_cuda:
            kw["enable_gqa"] = True
        else:
            r = h // hkv
            kh, vh = kh.repeat_interleave(r, dim=1), vh.repeat_interleave(r, dim=1)
    o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, is_causal=is_causal and mask is None, scale=scale, **kw)
    return o.transpose(1, 2)


def eager_attention(q, k, v, mask, scale):
    """Reference attention with explicit softmax in fp32. ``mask`` bool broadcastable to ``[B,1,T,S]`` (True = attend)."""
    h, hkv = q.shape[2], k.shape[2]
    qh, kh, vh = q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
    if hkv != h:
        r = h // hkv
        kh, vh = kh.repeat_interleave(r, dim=1), vh.repeat_interleave(r, dim=1)
    s = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vh).transpose(1, 2).to(q.dtype)


def build_mask(q_pos: torch.Tensor, s: int, kv_lens: torch.Tensor | None, window: int | None, key_valid: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean mask ``[B,1,T,S]``: key j visible to query at position p iff ``j<=p``, ``j<kv_len``, within window, key valid.

    Rows that would see no key at all (padding) are given key 0 to avoid NaN softmax rows.
    """
    j = torch.arange(s, device=q_pos.device).view(1, 1, s)
    p = q_pos.unsqueeze(-1)
    allowed = j <= p
    if kv_lens is not None:
        allowed = allowed & (j < kv_lens.view(-1, 1, 1))
    if window is not None:
        allowed = allowed & ((p - j) < window)
    if key_valid is not None:
        allowed = allowed & key_valid.unsqueeze(1)
    allowed = allowed | (~allowed.any(dim=-1, keepdim=True) & (j == 0))
    return allowed.unsqueeze(1)


class Attention(nn.Module):
    """Grouped-query attention with RoPE, optional qk-norm, biases and sliding window."""

    def __init__(self, cfg: ModelConfig, layer_idx: int, dtype=None, device=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.kv_heads, int(cfg.head_dim)  # type: ignore[arg-type]
        self.scale = self.head_dim**-0.5
        self.window = cfg.sliding_window
        h = cfg.hidden_size
        b = cfg.attention_bias
        self.q_proj = make_linear(h, cfg.q_dim, b, dtype=dtype, device=device)
        self.k_proj = make_linear(h, cfg.kv_dim, b, dtype=dtype, device=device)
        self.v_proj = make_linear(h, cfg.kv_dim, b, dtype=dtype, device=device)
        self.o_proj = make_linear(cfg.q_dim, h, cfg.o_proj_bias, dtype=dtype, device=device)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        else:
            self.q_norm = self.k_norm = None
        self._paged_ok = True

    # ------------------------------------------------------------------ backends
    def _flash(self, q, k, v) -> torch.Tensor | None:
        if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
            return None
        try:
            from flash_attn import flash_attn_func  # type: ignore

            ws = (self.window - 1, 0) if self.window else (-1, -1)
            return flash_attn_func(q, k, v, softmax_scale=self.scale, causal=True, window_size=ws)
        except Exception as e:
            log.debug("flash_attn unavailable (%s); using SDPA", e)
            return None

    def _no_cache(self, q, k, v, ctx: AttnContext) -> torch.Tensor:
        t = q.shape[1]
        am = ctx.attention_mask
        if ctx.impl == "flash_attn" and am is None:
            o = self._flash(q, k, v)
            if o is not None:
                return o
        if am is None and self.window is None and ctx.impl != "eager":
            return sdpa(q, k, v, None, True, self.scale)
        idx = torch.arange(t, device=q.device).unsqueeze(0).expand(q.shape[0], t)
        mask = build_mask(idx, t, None, self.window, am.bool() if am is not None else None)
        if ctx.impl == "eager":
            return eager_attention(q, k, v, mask, self.scale)
        return sdpa(q, k, v, mask, False, self.scale)

    def _paged_kernel(self, q, k, v, ctx: AttnContext, cache: PagedKVView) -> torch.Tensor | None:
        """Single-token decode through ``forge.kernels.ops_attn.paged_decode_attention`` (**not validated on GPU hardware**).

        Kernel contract (``forge/kernels/ops_attn.py``): ``paged_decode_attention(q[B,Hq,D], k_pool[P,Hkv,page,D], v_pool,
        block_tables[B,nb] int32, seq_lens[B] int32 (total keys incl. the new token), softmax_scale=, kv_dtype=, window=)``.
        :class:`~forge.models.kvcache.SimplePagedPool` stores token-major ``[P, page, Hkv, D]``, so the pool is handed over as a
        zero-copy ``transpose(1, 2)`` view (the Triton kernel is stride-aware).  Quantised pools keep *per-token* K/V scales whereas
        the kernel expects per-channel-K / per-token-V layouts, so they always take the gather+SDPA path.
        """
        use = ctx.use_paged_kernel
        if use is None:
            use = q.is_cuda
        if not use or not self._paged_ok or os.environ.get("FORGE_FORCE_REF", "0") == "1":
            return None
        if is_quantised(cache.kv_dtype):
            return None
        try:
            from forge.kernels import ops_attn

            cache.write(self.layer_idx, k, v, ctx.positions, ctx.n_new)
            end = cache.end_lens(ctx.n_new, 1)
            pool = cache.pool
            o = ops_attn.paged_decode_attention(
                q[:, 0],
                pool.k(self.layer_idx).transpose(1, 2),
                pool.v(self.layer_idx).transpose(1, 2),
                cache.block_tables.to(torch.int32),
                end.to(torch.int32),
                softmax_scale=self.scale,
                kv_dtype=cache.kv_dtype,
                window=self.window,
            )
            return o.reshape(q.shape[0], 1, self.num_heads, self.head_dim).to(q.dtype)
        except Exception as e:
            if os.environ.get("FORGE_KERNEL_STRICT", "0") == "1":
                raise
            log.warning("paged_decode_attention unavailable (%s); using gather+SDPA path", e)
            self._paged_ok = False
            return None

    def _cached(self, q, k, v, ctx: AttnContext) -> torch.Tensor:
        cache = ctx.cache
        if isinstance(cache, PagedKVView) and q.shape[1] == 1:
            o = self._paged_kernel(q, k, v, ctx, cache)
            if o is not None:
                return o
        kf, vf = cache.update(self.layer_idx, k, v, ctx.positions, ctx.n_new)
        mask = build_mask(ctx.positions, kf.shape[1], ctx.kv_lens, self.window)
        if ctx.impl == "eager":
            return eager_attention(q, kf, vf, mask, self.scale)
        return sdpa(q, kf, vf, mask, False, self.scale)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary(q, k, cos, sin)
        o = self._no_cache(q, k, v, ctx) if ctx.cache is None else self._cached(q, k, v, ctx)
        return self.o_proj(o.reshape(b, t, self.num_heads * self.head_dim))
