"""Ternary (1.58-bit) numerics: weights in {-1, 0, +1} with per-group fp32 scales, per-token int8 activations.

This is the single definition shared by PTQ, QAT (``fq_t158``), the kernel oracle (``forge.kernels.ref.gemm_ternary``) and tests
(ADR 0007).  Semantics are taken from tern158 (``ref_ops.py``) so artifacts and golden vectors are interchangeable:

* Weight quantiser: ``W ~= alpha * T``; ``T = sign(w) * (|w| > thr)``.  Rules (per segment = group, row or whole tensor):
  ``absmean`` (alpha = mean|w|, thr = alpha/2), ``twn`` (thr = 0.75 mean|w|, alpha = mean of |w| above thr), ``lloyd``
  (absmean start, then ``iters`` steps of alpha = mean of |w| above alpha/2; thr = alpha/2).  An all-zero segment gets alpha = 0.
* ``group_size``: ``> 0`` group along K (must divide K), ``-1`` one scale per output row, ``0`` one scale per tensor.
  Scales are returned as fp32 ``[N, G]`` (``[1, 1]`` for per-tensor).
* Activations: per-token absmax int8, ``q = clamp(round(x * (127 / max)), -127, 127)``, ``s = max / 127``.
* Packing: 2 bits per weight, code ``00 -> 0``, ``01 -> +1``, ``11 -> -1`` (``10`` is reserved and decodes to 0), element ``j`` of a row
  at bits ``2 * (j % 4)`` of byte ``j // 4``.  This is byte-identical to tern158's little-endian 16-codes-per-int32 words and to
  ``pack_bits(t & 3, 2)``.  ``K`` must be a multiple of 16 (the kernel word size).
* Straight-through estimator for QAT lives in :func:`fake_quant_ternary`; the scale is detached (never learned).
"""
from __future__ import annotations

import math

import torch

from forge.core.errors import ConfigError

from .packing import pack_bits, unpack_bits

RULES = ("absmean", "twn", "lloyd")
CODES_PER_WORD = 16  # kernel word size; K must be a multiple of this
BITS_PACKED = 2.0
BITS_IDEAL = math.log2(3.0)


def _check_k(k: int) -> None:
    if k % CODES_PER_WORD:
        raise ConfigError(f"ternary packing needs K to be a multiple of {CODES_PER_WORD}, got K={k}")


def effective_group(group_size: int, k: int) -> int:
    """Segment length along K: ``k`` for per-row (-1), the group otherwise; 0 (per tensor) is handled by the caller."""
    if group_size == -1:
        return k
    if group_size <= 0 or k % group_size:
        raise ConfigError(f"group_size must be -1, 0 or a divisor of K={k}, got {group_size}")
    return group_size


def bits_per_weight(k: int, group_size: int, scale_bits: int = 32, packed: bool = True) -> float:
    """Storage bits per weight including scales (2.0 packed, +0.25 for G=128 fp32 scales: 32 bits per 128 weights)."""
    base = BITS_PACKED if packed else BITS_IDEAL
    if group_size == 0:
        return base
    return base + scale_bits / effective_group(group_size, k)


def _alpha_thr(a: torch.Tensor, rule: str, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``a``: |w| as fp32 ``[S, L]`` -> (alpha, thr) each ``[S, 1]``."""
    mean = a.mean(dim=1, keepdim=True)
    alpha, thr = mean, 0.5 * mean
    if rule == "twn":
        thr = 0.75 * mean
        m = a > thr
        cnt = m.sum(dim=1, keepdim=True)
        alpha = torch.where(cnt > 0, (a * m).sum(dim=1, keepdim=True) / cnt.clamp(min=1), mean)
    elif rule == "lloyd":
        for _ in range(iters):
            m = a > 0.5 * alpha
            cnt = m.sum(dim=1, keepdim=True)
            alpha = torch.where(cnt > 0, (a * m).sum(dim=1, keepdim=True) / cnt.clamp(min=1), alpha)
        thr = 0.5 * alpha
    elif rule != "absmean":
        raise ConfigError(f"unknown ternary rule {rule!r}; expected one of {RULES}")
    dead = ~(alpha > 0)
    alpha = torch.where(dead, torch.zeros_like(alpha), alpha)
    thr = torch.where(dead, torch.full_like(thr, float("inf")), thr)
    return alpha, thr


def quantize_ternary(w: torch.Tensor, group_size: int = 128, rule: str = "lloyd", iters: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """``w [N, K]`` -> (``t`` int8 ``[N, K]`` in {-1,0,1}, ``scales`` fp32 ``[N, G]`` or ``[1, 1]`` when ``group_size == 0``)."""
    if w.dim() != 2:
        raise ConfigError("quantize_ternary expects a 2-D weight [N, K]")
    n, k = w.shape
    wf = w.detach().float()
    if group_size == 0:
        seg = wf.reshape(1, -1)
    else:
        seg = wf.reshape(n * (k // effective_group(group_size, k)), effective_group(group_size, k))
    a = seg.abs()
    alpha, thr = _alpha_thr(a, rule, iters)
    t = torch.where(a > thr, torch.sign(seg), torch.zeros_like(seg)).to(torch.int8).reshape(n, k)
    scales = alpha.reshape(1, 1) if group_size == 0 else alpha.reshape(n, k // effective_group(group_size, k))
    return t, scales.contiguous()


def expand_scales(scales: torch.Tensor, n: int, k: int, group_size: int) -> torch.Tensor:
    """Broadcast ``scales`` to ``[N, K]`` (fp32)."""
    s = scales.float()
    if group_size == 0:
        return s.reshape(1, 1).expand(n, k)
    g = effective_group(group_size, k)
    return s.reshape(n, k // g, 1).expand(n, k // g, g).reshape(n, k)


def dequantize_ternary(t: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """``t int8 [N, K]`` + scales -> fp32 ``[N, K]``."""
    n, k = t.shape
    return t.float() * expand_scales(scales, n, k, group_size)


def fake_quant_ternary(w: torch.Tensor, group_size: int = 128, rule: str = "lloyd", iters: int = 1) -> torch.Tensor:
    """Quantise-dequantise in the dtype of ``w`` (no gradient; wrap in an STE for QAT)."""
    t, s = quantize_ternary(w, group_size, rule, iters)
    return dequantize_ternary(t, s, group_size).to(w.dtype)


def act_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (last-dim rows) absmax int8: returns (``q`` int8 ``[M, K]``, ``s`` fp32 ``[M]``) with ``x ~= q * s``."""
    k = x.shape[-1]
    xf = x.detach().reshape(-1, k).float()
    mx = xf.abs().amax(dim=1)
    inv = torch.where(mx > 0, 127.0 / mx, torch.zeros_like(mx))
    q = torch.round(xf * inv[:, None]).clamp(-127, 127).to(torch.int8)
    return q, mx / 127.0


def act_dequant(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (q.float() * s[:, None]).to(dtype)


def fake_quant_act_int8(x: torch.Tensor) -> torch.Tensor:
    q, s = act_quant_int8(x)
    return act_dequant(q, s, x.dtype).reshape(x.shape)


def pack_ternary(t: torch.Tensor) -> torch.Tensor:
    """``t int8 [N, K]`` in {-1,0,1} -> ``uint8 [N, K/4]`` (see module docstring for the bit layout)."""
    if t.dim() != 2:
        raise ConfigError("pack_ternary expects int8 [N, K]")
    _check_k(t.shape[1])
    return pack_bits((t.to(torch.int16) & 3).to(torch.uint8), 2)


def unpack_ternary(qweight: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Inverse of :func:`pack_ternary`; the reserved code ``10`` decodes to 0."""
    c = unpack_bits(qweight, 2, k)
    t = torch.where((c & 1) == 1, torch.where((c & 2) == 2, -1, 1), 0)
    return t.to(torch.int8)


def pack_words_int32(qweight: torch.Tensor) -> torch.Tensor:
    """View packed ``uint8 [N, K/4]`` as tern158 int32 words ``[N * K/16]`` (little-endian)."""
    flat = qweight.contiguous().reshape(-1)
    if flat.numel() % 4:
        raise ConfigError("packed ternary weights must hold a whole number of int32 words")
    return flat.view(torch.int32) if flat.dtype == torch.uint8 else flat


def words_to_packed(words: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """tern158 int32 codes ``[N*K/16]`` -> ``uint8 [N, K/4]`` (inverse of :func:`pack_words_int32`)."""
    _check_k(k)
    return words.contiguous().reshape(-1).view(torch.uint8).reshape(n, k // 4)


def predicted_flip_rate(lr: float, weight_decay: float) -> float:
    """Equilibrium per-step code flip rate of AdamW on ternary latents (tern158 REPORT_NOTES; TARGET until re-measured in Forge)."""
    return 0.59 * math.sqrt(max(0.0, 2.0 * lr * weight_decay))


def warmup_lambda(step: int, warmup_steps: int) -> float:
    """Quantisation blend ``lam = min(1, step / warmup)`` for ``W_eff = W + lam * (Q(W) - W)``; 1 when there is no warm-up."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, step / float(warmup_steps)))


def code_stats(t: torch.Tensor) -> dict[str, float]:
    """Fractions of -1 / 0 / +1 in a ternary tensor."""
    n = max(1, t.numel())
    return {"neg": float((t < 0).sum()) / n, "zero": float((t == 0).sum()) / n, "pos": float((t > 0).sum()) / n}
