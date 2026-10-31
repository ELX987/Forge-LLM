"""KV-cache quantisation (Plan CP-4): FP8 KV and KIVI-style asymmetric low-bit KV with a full-precision residual window.

Schemes (``KVQuantConfig.scheme``)
----------------------------------
``fp16``        reference (no quantisation).
``fp8_e4m3`` / ``fp8_e5m2``
                one scale per (token, head) over ``head_dim`` (dynamic, written with the token) -- the production default of the Plan;
                ``static_scale`` switches to one calibrated per-tensor scale per layer (no per-token scale bytes).
``int8``        per-token symmetric INT8 (same scale layout as FP8).
``kivi``        KIVI (Liu et al.): **keys quantised per channel** (statistics over a group of ``group_size`` tokens, because key outliers sit in a
                few channels), **values quantised per token** (groups of ``value_group`` channels), asymmetric ``bits`` in {2, 4}.  The newest
                ``residual_window`` tokens stay in full precision and are flushed to the quantised store ``group_size`` tokens at a time.
                4-bit KV is meant to sit *behind a long-context gate* (Plan CP-4); this module only provides numerics + accounting.

Everything is a reference implementation (plain torch); attention over the cache is :func:`reference_attention`.  Serving kernels
that read quantised KV live in ``forge.kernels``; this module defines the numerics they must match.
:func:`kv_accuracy_report` produces the accuracy-vs-bytes table; :func:`model_kv_report` fills it from a real model's k/v projections
(pre-RoPE proxy: keys/queries are taken before rotary embedding and say so in the report).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from forge.core.errors import ConfigError
from forge.formats.fp8 import FP8_MAX, from_fp8, round_fp8, to_fp8
from forge.formats.int_quant import dequantize_int, quantize_int
from forge.formats.packing import pack_bits, unpack_bits


class KVQuantConfig(BaseModel):
    """KV-cache quantisation settings (stage ``kv``)."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["fp16", "fp8_e4m3", "fp8_e5m2", "int8", "kivi"] = Field("fp8_e4m3", description="KV storage scheme (fp8_e4m3 is the default recommendation; kivi = asymmetric low-bit with residual window).")
    bits: int = Field(4, description="KIVI bits per element (2 or 4).")
    group_size: int = Field(32, ge=1, description="KIVI: tokens per key-quantisation group (keys are quantised per channel over this many tokens).")
    value_group: int = Field(32, ge=1, description="KIVI: channels per value-quantisation group (values are quantised per token).")
    residual_window: int = Field(128, ge=0, description="Most recent tokens kept in full precision (KIVI). For FP8/INT8 recorded for the serving engine only.")
    static_scale: bool = Field(False, description="FP8: use one calibrated per-tensor scale per layer instead of per-token scales.")
    report_seqs: int = Field(8, ge=1, description="Calibration sequences used for the accuracy-vs-bytes report.")
    report_len: int = Field(256, ge=16, description="Tokens per sequence in the report.")
    long_context_gated: bool = Field(True, description="Mark 4-bit/2-bit KV as requiring the long-context gate before serving (Plan CP-4).")


# ------------------------------------------------------------------------------------------------ simple per-token schemes
def quantize_kv_fp8(x: torch.Tensor, fmt: str = "e4m3", scale: torch.Tensor | float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [..., T, D]`` -> ``(x8, scale)``; ``scale=None``: one scale per (token, head) over D (shape ``[..., T, 1]``)."""
    return to_fp8(x, fmt, scale, dim=-1 if scale is None else None)


def dequantize_kv_fp8(x8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return from_fp8(x8, scale, dtype)


def quantize_kv_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = (x.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-12)
    return torch.round(x.float() / s).clamp(-128, 127).to(torch.int8), s


def dequantize_kv_int8(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (q.float() * s).to(dtype)


# ------------------------------------------------------------------------------------------------ KIVI cache
@dataclass
class _Chunk:
    k_codes: torch.Tensor  # packed uint8 [B,H,G,D*bits/8]
    k_scale: torch.Tensor  # [B,H,1,D]
    k_zero: torch.Tensor
    v_codes: torch.Tensor
    v_scale: torch.Tensor  # [B,H,G,D/vg]
    v_zero: torch.Tensor
    tokens: int


class KiviKVCache:
    """One layer's KIVI cache: quantised chunks of ``group_size`` tokens + a full-precision residual of ``residual_window..+group_size`` tokens.

    ``append(k, v)`` with ``k, v [B, H, T, D]`` (any T); ``get()`` returns the dequantised ``(K, V)`` over all stored tokens.
    """

    def __init__(self, bits: int = 4, group_size: int = 32, value_group: int = 32, residual_window: int = 128, scale_dtype: torch.dtype = torch.float16) -> None:
        if bits not in (2, 4, 8):
            raise ConfigError("KIVI bits must be 2, 4 or 8")
        self.bits, self.G, self.vg, self.R = bits, group_size, value_group, residual_window
        self.scale_dtype = scale_dtype
        self.chunks: list[_Chunk] = []
        self.res_k: torch.Tensor | None = None
        self.res_v: torch.Tensor | None = None
        self._d: int | None = None

    @property
    def seq_len(self) -> int:
        return sum(c.tokens for c in self.chunks) + (0 if self.res_k is None else self.res_k.shape[2])

    @property
    def residual_len(self) -> int:
        return 0 if self.res_k is None else self.res_k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape or k.ndim != 4:
            raise ConfigError("k, v must both be [B, H, T, D]")
        d = k.shape[-1]
        self._d = d
        vg = min(self.vg, d)
        if d % vg:
            raise ConfigError(f"head_dim {d} not divisible by value_group {vg}")
        self.res_k = k if self.res_k is None else torch.cat([self.res_k, k], dim=2)
        self.res_v = v if self.res_v is None else torch.cat([self.res_v, v], dim=2)
        while self.res_k.shape[2] >= self.R + self.G:
            self.chunks.append(self._quantize_chunk(self.res_k[:, :, : self.G], self.res_v[:, :, : self.G]))
            self.res_k, self.res_v = self.res_k[:, :, self.G :], self.res_v[:, :, self.G :]

    def _quantize_chunk(self, k: torch.Tensor, v: torch.Tensor) -> _Chunk:
        b, h, g, d = k.shape
        vg = min(self.vg, d)
        # keys: per channel, statistics over the G tokens (group dim = token axis)
        kq, ks, kz = quantize_int(k.float(), self.bits, g, sym=False, dim=2)
        # values: per token, groups of vg channels along D
        vq, vs, vz = quantize_int(v.float(), self.bits, vg, sym=False, dim=3)
        return _Chunk(
            pack_bits(kq, self.bits), ks.to(self.scale_dtype), kz, pack_bits(vq, self.bits), vs.to(self.scale_dtype), vz, g,
        )

    def _dequant_chunk(self, c: _Chunk) -> tuple[torch.Tensor, torch.Tensor]:
        d = self._d or 0
        vg = min(self.vg, d)
        kq = unpack_bits(c.k_codes, self.bits, d)
        vq = unpack_bits(c.v_codes, self.bits, d)
        k = dequantize_int(kq, c.k_scale.float(), c.k_zero, c.tokens, dim=2)
        v = dequantize_int(vq, c.v_scale.float(), c.v_zero, vg, dim=3)
        return k, v

    def get(self, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        ks, vs = [], []
        for c in self.chunks:
            k, v = self._dequant_chunk(c)
            ks.append(k)
            vs.append(v)
        if self.res_k is not None:
            ks.append(self.res_k.float())
            vs.append(self.res_v.float())  # type: ignore[union-attr]
        if not ks:
            raise ConfigError("empty cache")
        return torch.cat(ks, 2).to(dtype), torch.cat(vs, 2).to(dtype)

    def nbytes(self, residual_bytes_per_elem: int = 2) -> int:
        """Bytes actually needed: packed codes + scales + zero points of quantised chunks, plus the residual at 16 bit."""
        n = 0
        for c in self.chunks:
            n += sum(t.numel() * t.element_size() for t in (c.k_codes, c.k_scale, c.k_zero, c.v_codes, c.v_scale, c.v_zero))
        if self.res_k is not None:
            n += 2 * self.res_k.numel() * residual_bytes_per_elem
        return n

    def fp16_nbytes(self) -> int:
        if self._d is None:
            return 0
        b, h = (self.res_k.shape[0], self.res_k.shape[1]) if self.res_k is not None else (self.chunks[0].k_codes.shape[0], self.chunks[0].k_codes.shape[1])
        return 2 * b * h * self.seq_len * self._d * 2


# ------------------------------------------------------------------------------------------------ reference attention
def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """GQA softmax attention. ``q [B,Hq,Tq,D]``, ``k,v [B,Hkv,T,D]`` (queries are the *last* Tq positions); returns ``[B,Hq,Tq,D]``."""
    b, hq, tq, d = q.shape
    hkv, t = k.shape[1], k.shape[2]
    rep = hq // hkv
    kk = k.repeat_interleave(rep, dim=1)
    vv = v.repeat_interleave(rep, dim=1)
    att = (q.float() @ kk.float().transpose(-1, -2)) / math.sqrt(d)
    if causal:
        qpos = torch.arange(t - tq, t, device=q.device).unsqueeze(1)
        att = att.masked_fill(torch.arange(t, device=q.device).unsqueeze(0) > qpos, float("-inf"))
    return torch.softmax(att, dim=-1) @ vv.float()


# ------------------------------------------------------------------------------------------------ accounting + report
def bytes_per_token_per_layer(cfg: KVQuantConfig, kv_heads: int, head_dim: int, context: int = 4096) -> float:
    """Average bytes per token per layer (K+V) for a context of ``context`` tokens."""
    elems = 2 * kv_heads * head_dim
    if cfg.scheme == "fp16":
        return elems * 2.0
    if cfg.scheme.startswith("fp8") or cfg.scheme == "int8":
        scale = 0.0 if cfg.static_scale else 2.0 * 2 * kv_heads  # one fp16 scale per (token, head) for each of K and V
        return elems * 1.0 + scale
    # KIVI: residual tokens at 16 bit, the rest packed
    g, vg = cfg.group_size, min(cfg.value_group, head_dim)
    quant_elems = elems * cfg.bits / 8.0
    k_overhead = kv_heads * head_dim * (2 + 1) / g  # per-channel scale(fp16)+zero(u8) per group of G tokens
    v_overhead = kv_heads * (head_dim / vg) * (2 + 1)
    q_tok = quant_elems + k_overhead + v_overhead
    res = min(cfg.residual_window + g / 2, context)
    return (q_tok * max(context - res, 0) + elems * 2.0 * res) / max(context, 1)


_LAST_KIVI: dict[str, float] = {}


def _roundtrip(cfg: KVQuantConfig, k: torch.Tensor, v: torch.Tensor, static: tuple[torch.Tensor, torch.Tensor] | None = None):
    if cfg.scheme == "fp16":
        return k.half().float(), v.half().float(), 2.0 * k.numel() * 2
    if cfg.scheme in ("fp8_e4m3", "fp8_e5m2"):
        fmt = cfg.scheme.split("_")[1]
        if cfg.static_scale:
            ks, vs = static if static is not None else (k.abs().amax() / FP8_MAX[fmt], v.abs().amax() / FP8_MAX[fmt])
            kq, vq = round_fp8(k / ks, fmt) * ks, round_fp8(v / vs, fmt) * vs
            return kq, vq, (k.numel() + v.numel()) * 1.0
        k8, kscale = quantize_kv_fp8(k, fmt)
        v8, vscale = quantize_kv_fp8(v, fmt)
        return dequantize_kv_fp8(k8, kscale), dequantize_kv_fp8(v8, vscale), (k.numel() + v.numel()) * 1.0 + 2.0 * (kscale.numel() + vscale.numel())
    if cfg.scheme == "int8":
        kq, ks = quantize_kv_int8(k)
        vq, vs = quantize_kv_int8(v)
        return dequantize_kv_int8(kq, ks), dequantize_kv_int8(vq, vs), (k.numel() + v.numel()) * 1.0 + 2.0 * (ks.numel() + vs.numel())
    cache = KiviKVCache(cfg.bits, cfg.group_size, cfg.value_group, cfg.residual_window)
    cache.append(k, v)
    kd, vd = cache.get()
    _LAST_KIVI["quantized_fraction"] = 1.0 - cache.residual_len / max(cache.seq_len, 1)
    return kd, vd, float(cache.nbytes())


def kv_accuracy_report(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor | None = None,
    configs: Sequence[KVQuantConfig] | None = None,
) -> list[dict[str, Any]]:
    """Accuracy-vs-bytes rows for each config on given ``k, v [B,Hkv,T,D]`` (and queries ``q [B,Hq,Tq,D]``; default: the last 16 key
    vectors repeated per group, a proxy).  Columns: key/value relative error, attention-output relative error, cosine similarity,
    bytes per token per layer and compression vs FP16.  ``provenance`` is ``measured`` (on the supplied tensors)."""
    b, hkv, t, d = k.shape
    if configs is None:
        rw = max(min(128, t // 4), 0)  # residual window must be shorter than the context or nothing gets quantised
        configs = [
            KVQuantConfig(scheme="fp16"),
            KVQuantConfig(scheme="fp8_e4m3"),
            KVQuantConfig(scheme="fp8_e5m2"),
            KVQuantConfig(scheme="int8"),
            KVQuantConfig(scheme="kivi", bits=4, residual_window=rw),
            KVQuantConfig(scheme="kivi", bits=2, residual_window=rw),
        ]
    if q is None:
        q = k[:, :, -min(16, t) :].repeat_interleave(2, dim=1)
    ref = reference_attention(q, k, v)
    rows = []
    for c in configs:
        _LAST_KIVI.clear()
        kd, vd, nb = _roundtrip(c, k.float(), v.float())
        out = reference_attention(q, kd, vd)
        kerr = ((kd - k).pow(2).sum() / k.float().pow(2).sum().clamp(min=1e-30)).sqrt().item()
        verr = ((vd - v).pow(2).sum() / v.float().pow(2).sum().clamp(min=1e-30)).sqrt().item()
        oerr = ((out - ref).pow(2).sum() / ref.pow(2).sum().clamp(min=1e-30)).sqrt().item()
        cos = torch.nn.functional.cosine_similarity(out.reshape(-1, d), ref.reshape(-1, d), dim=-1).mean().item()
        measured_bpt = nb / (b * t * hkv) * hkv  # bytes per token (K+V) for one layer, measured on this tensor
        rows.append({
            "scheme": c.scheme if c.scheme != "kivi" else f"kivi{c.bits}", "key_rel_err": kerr, "value_rel_err": verr, "attn_out_rel_err": oerr,
            "attn_out_cosine": cos, "bytes_per_token_layer": measured_bpt / 1.0, "compression_vs_fp16": (2.0 * 2 * hkv * d) / (measured_bpt / 1.0),
            "analytic_bytes_per_token_layer": bytes_per_token_per_layer(c, hkv, d, t), "provenance": "measured",
            "gated": bool(c.long_context_gated and c.scheme == "kivi"),
            "quantized_token_fraction": _LAST_KIVI.get("quantized_fraction", 1.0), "residual_window": c.residual_window if c.scheme == "kivi" else None,
        })
    return rows


@torch.no_grad()
def model_kv_report(
    model: nn.Module, tokens: torch.Tensor, configs: Sequence[KVQuantConfig] | None = None, max_layers: int = 4, batch_size: int = 4
) -> dict[str, Any]:
    """Run the model on ``tokens`` and evaluate KV schemes on its k/v (and q) projection outputs, averaged over up to ``max_layers`` layers.

    The projections are taken **before RoPE** (documented proxy): per-channel key outlier structure is preserved, rotation is not applied.
    """
    from forge.compress.modelutil import find_blocks

    prefix, blocks = find_blocks(model)
    cfg = getattr(model, "config", None)
    hq = int(getattr(cfg, "num_heads", 0)) or None
    hkv = int(getattr(cfg, "kv_heads", getattr(cfg, "num_kv_heads", 0))) or None
    hd = int(getattr(cfg, "head_dim", 0)) or None
    if not (hq and hkv and hd):
        raise ConfigError("model_kv_report needs model.config with num_heads / num_kv_heads / head_dim")
    sel = list(range(len(blocks)))
    if len(sel) > max_layers:
        step = len(sel) / max_layers
        sel = [sel[int(i * step)] for i in range(max_layers)]
    cap: dict[int, dict[str, list[torch.Tensor]]] = {i: {"q": [], "k": [], "v": []} for i in sel}
    handles = []
    for i in sel:
        attn = blocks[i].self_attn
        for key, mod in (("q", attn.q_proj), ("k", attn.k_proj), ("v", attn.v_proj)):
            handles.append(mod.register_forward_hook(lambda _m, _i, out, i=i, key=key: cap[i][key].append(out.detach().float())))
    was = model.training
    model.eval()
    try:
        for b in tokens.split(batch_size):
            model(b.to(next(model.parameters()).device))
    finally:
        for h in handles:
            h.remove()
        model.train(was)
    agg: dict[str, dict[str, float]] = {}
    for i in sel:
        def heads(xs: list[torch.Tensor], h: int) -> torch.Tensor:
            x = torch.cat(xs, 0)
            return x.reshape(x.shape[0], x.shape[1], h, hd).permute(0, 2, 1, 3)

        rows = kv_accuracy_report(heads(cap[i]["k"], hkv), heads(cap[i]["v"], hkv), heads(cap[i]["q"], hq), configs)
        for r in rows:
            a = agg.setdefault(r["scheme"], {})
            for kx, vx in r.items():
                if isinstance(vx, float):
                    a[kx] = a.get(kx, 0.0) + vx / len(sel)
    return {"layers": sel, "pre_rope_proxy": True, "rows": [{"scheme": s, **vals, "provenance": "measured"} for s, vals in agg.items()]}
