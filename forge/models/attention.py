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
from forge.models.layers import RMSNorm, apply_rotary, rotate_half
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
    # gemma4: ``{layer_type: (cos, sin)}`` (overrides the cos/sin passed to a layer), PLE tensor ``[B,T,L,P]`` and the per-forward
    # K/V hand-over between a layer and the KV-shared layers that read it ``{source_layer: (K, V)}``
    rope: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    per_layer_inputs: torch.Tensor | None = None
    shared_kv: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = field(default_factory=dict)


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
    """Grouped-query attention with RoPE, optional qk-norm, biases and sliding window.

    gemma4 additions (all driven by ``ModelConfig``): per-layer head_dim / kv heads / window, softmax scale ``attn_scale``, scale-less
    ``v_norm``, ``attention_k_eq_v`` (no v_proj) and KV sharing (a shared layer owns only q_proj / o_proj / q_norm and attends over the
    K/V produced by ``cfg.kv_source_layer``).
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int, dtype=None, device=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.layer_kv_heads(layer_idx), cfg.layer_head_dim(layer_idx)
        self.scale = cfg.attn_scale if cfg.attn_scale is not None else self.head_dim**-0.5
        self.window = cfg.layer_window(layer_idx)
        self.layer_type = cfg.layer_types[layer_idx] if cfg.layer_types is not None else None
        self.kv_source = cfg.kv_source_layer(layer_idx)  # None = this layer computes its own K/V
        self.is_kv_source = any(cfg.kv_source_layer(j) == layer_idx for j in range(layer_idx + 1, cfg.num_layers))
        h = cfg.hidden_size
        b = cfg.attention_bias
        q_dim, kv_dim = cfg.layer_q_dim(layer_idx), cfg.layer_kv_dim(layer_idx)
        fp32 = cfg.norm_fp32_scale
        self.q_proj = make_linear(h, q_dim, b, dtype=dtype, device=device)
        self.o_proj = make_linear(q_dim, h, cfg.o_proj_bias, dtype=dtype, device=device)
        owns_kv = self.kv_source is None
        self.k_proj = make_linear(h, kv_dim, b, dtype=dtype, device=device) if owns_kv else None
        self.v_proj = make_linear(h, kv_dim, b, dtype=dtype, device=device) if (owns_kv and cfg.layer_has_v_proj(layer_idx)) else None
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device, fp32_scale=fp32)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device, fp32_scale=fp32) if owns_kv else None
        else:
            self.q_norm = self.k_norm = None
        self.v_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, with_scale=False) if (cfg.v_norm and owns_kv) else None
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

    def _attend_cached(self, q, kf, vf, ctx: AttnContext) -> torch.Tensor:
        mask = build_mask(ctx.positions, kf.shape[1], ctx.kv_lens, self.window)
        if ctx.impl == "eager":
            return eager_attention(q, kf, vf, mask, self.scale)
        return sdpa(q, kf, vf, mask, False, self.scale)

    def _cached(self, q, k, v, ctx: AttnContext):
        """Returns ``(o, K_all, V_all)``; the K/V pair is None when the paged decode kernel consumed the cache directly."""
        cache = ctx.cache
        if isinstance(cache, PagedKVView) and q.shape[1] == 1:
            o = self._paged_kernel(q, k, v, ctx, cache)
            if o is not None:
                return o, None, None
        kf, vf = cache.update(self.layer_idx, k, v, ctx.positions, ctx.n_new)
        return self._attend_cached(q, kf, vf, ctx), kf, vf

    def _shared_kv(self, ctx: AttnContext, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V of the source layer for a KV-shared layer (stashed this forward, else read back from the cache)."""
        kv = ctx.shared_kv.get(self.kv_source)  # type: ignore[arg-type]
        if kv is not None and kv[0] is not None:
            return kv  # type: ignore[return-value]
        cache = ctx.cache
        if cache is None or ctx.kv_lens is None:
            raise RuntimeError(f"layer {self.layer_idx} reads the K/V of layer {self.kv_source}, which has not run in this forward")
        return cache.read(self.kv_source, int(ctx.kv_lens.max().item()), like.dtype, ctx.kv_lens)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.kv_source is not None:  # KV-shared layer: own q only, K/V from the source layer (already normed + rotated)
            cq, sq = cos.to(q.dtype).unsqueeze(2), sin.to(q.dtype).unsqueeze(2)
            q = q * cq + rotate_half(q) * sq
            kf, vf = self._shared_kv(ctx, q)
            if ctx.cache is None:
                o = self._no_cache(q, kf, vf, ctx)
            else:
                o = self._attend_cached(q, kf, vf, ctx)
            return self.o_proj(o.reshape(b, t, self.num_heads * self.head_dim))
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim) if self.v_proj is not None else k  # k_eq_v: V = pre-norm K
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.v_norm is not None:
            v = self.v_norm(v)
        q, k = apply_rotary(q, k, cos, sin)
        if ctx.cache is None:
            o = self._no_cache(q, k, v, ctx)
            kf, vf = k, v
        else:
            o, kf, vf = self._cached(q, k, v, ctx)
        if self.is_kv_source:
            ctx.shared_kv[self.layer_idx] = (kf, vf)
        return self.o_proj(o.reshape(b, t, self.num_heads * self.head_dim))
