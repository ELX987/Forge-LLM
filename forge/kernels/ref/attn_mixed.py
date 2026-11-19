"""Pure-torch oracle for **mixed-precision block-scaled attention** (``ops_attn.mixed_attention``).

One attention path for every KV precision Forge supports:

======================  ==========  ==============  =========================  ==============================
``kv_dtype``            bits/elem   element         key scale                  value scale
======================  ==========  ==============  =========================  ==============================
``bf16`` / ``fp16``     16          IEEE half       -                          -
``fp32``                32          IEEE single     -                          -
``fp8_e4m3``            8           E4M3FN          per channel (fp32)         per token (fp32)
``int8``                8           int8            per channel                per token
``int4``                4           int4 (packed)   per channel                per token
``nvfp4``               4.5         E2M1 + E4M3 bs  per channel block (16)     per token block (16)
``mxfp4``               4.25        E2M1 + E8M0 bs  per channel block (32)     per token block (32)
``ternary`` (1.58-bit)  3.0         {-1,0,+1} bs    per channel block (32)     per token block (32)
======================  ==========  ==============  =========================  ==============================

The format numerics (pools, packing, dequantisation) live in :mod:`forge.kernels.ref.attn_kv`; this module only uses
them, so the attention maths and the kernel numerics cannot drift apart.

Algorithm (what ``forge.kernels.triton.attn_mixed`` implements)
---------------------------------------------------------------
A flash-style **two-pass** kernel with the low-bit payload dequantised *once per tile*, fused into the matmuls
(the block-quantisation trick of FlashAttention-3, adapted to FP4 / ternary):

1. **Pass 1** streams the KV in tiles, computes ``q @ k^T`` from the decoded key tile, applies the key block scale,
   and runs the online-softmax recurrence in the log2 domain to obtain the per-row max ``m`` and normaliser ``l``.
   V is **not** read in this pass.
2. **Pass 2** streams the same tiles, recomputes the scores and accumulates ``O`` using the *final* LSE from pass 1,
   so every probability is exact (no per-tile renormalisation error).  Value block scales are folded into ``P``
   before ``P @ V`` (``sum_c p * v_c s_c = sum_c (p s_c) * v_c``), so the V payload is expanded once too.

The two passes read the same packed bytes; the point is that those bytes are 3-4.5 bits (ternary/FP4) instead of 16
(bf16).  A fused single-pass variant is future work (``docs/KERNELS.md``); the two-pass form is the contract this
oracle and the kernel share, so the kernel can be checked tile-for-tile against it.
"""
from __future__ import annotations

import math

import torch

from forge.kernels.ref.attn_kv import (
    dequantize_block_rows,
    expand_k_scale,
    expand_v_scale,
    expand_zero_like,
    is_block_quantised,
    logical_head_dim,
    normalize_kv_dtype,
    unpack_int4,
)

#: KV formats ``mixed_attention`` understands
MIXED_KV_DTYPES = ("bf16", "fp16", "fp32", "fp8_e4m3", "int8", "int4", "nvfp4", "mxfp4", "ternary")

NEG_INF = float("-inf")
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


# ----------------------------------------------------------------------------------------------------------------
# per-format row gathering (payload + folded dequantisation)
# ----------------------------------------------------------------------------------------------------------------
def _dequant_rows_simple(raw: torch.Tensor, kv: str, scale, zero) -> torch.Tensor:
    """fp8 / int8 / int4 rows ``[T, D']`` -> float32 ``[T, D]``.

    ``scale``/``zero`` are already per row: ``[T, D]`` for keys (per channel), ``[T]`` for values (per token).
    """
    if kv == "fp8_e4m3":
        x = (raw if raw.dtype == torch.float8_e4m3fn else raw.view(torch.float8_e4m3fn)).float()
    elif kv == "int8":
        x = raw.float()
    else:
        x = unpack_int4(raw).float()
    if scale is not None:
        x = x * (scale if scale.dim() == 2 else scale.unsqueeze(-1))
    if zero is not None:
        x = x + (zero if zero.dim() == 2 else zero.unsqueeze(-1))
    return x


def _gather_head(k_pool, v_pool, pages, h, n, kv, k_scale, v_scale, k_zero, v_zero,
                 k_block_scale, v_block_scale, tensor_scale):
    """Dequantised ``(k_rows [n, D], v_rows [n, D])`` float32 for one KV head of one sequence."""
    p, hkv, ps, dpk = k_pool.shape
    n_pages = len(pages)
    k_raw = k_pool[pages, h].reshape(n_pages * ps, dpk)[:n]
    v_raw = v_pool[pages, h].reshape(n_pages * ps, dpk)[:n]
    if kv in ("bf16", "fp16", "fp32"):
        return k_raw.float(), v_raw.float()
    d = logical_head_dim(k_pool, kv)
    if not is_block_quantised(kv):
        # fp8/int8/int4: keys per channel (per page), values per token (per page) -> expand over the page's tokens
        ks = expand_k_scale(k_scale, p, hkv, d, k_pool.device)[pages, h]          # [n_pages, D]
        vs = expand_v_scale(v_scale, p, hkv, ps, k_pool.device)[pages, h]         # [n_pages, ps]
        kz = expand_zero_like(k_zero, expand_k_scale(k_scale, p, hkv, d, k_pool.device))
        vz = expand_zero_like(v_zero, expand_v_scale(v_scale, p, hkv, ps, k_pool.device))
        kz = kz[pages, h] if kz is not None else None
        vz = vz[pages, h] if vz is not None else None
        ks = ks.repeat_interleave(ps, dim=0)[:n]                                   # [n, D]
        vs = vs.reshape(-1)[:n]                                                    # [n]
        kz = kz.repeat_interleave(ps, dim=0)[:n] if kz is not None else None
        vz = vz.reshape(-1)[:n] if vz is not None else None
        return _dequant_rows_simple(k_raw, kv, ks, kz), _dequant_rows_simple(v_raw, kv, vs, vz)
    # block-scaled: keys carry per-channel-block scales (broadcast over tokens), values per-token-block scales
    kb = k_block_scale[pages, h]  # [n_pages, D/bs]
    vb = v_block_scale[pages, h]  # [n_pages, ps, D/bs]
    kb = kb.unsqueeze(1).expand(n_pages, ps, kb.shape[-1]).reshape(n_pages * ps, -1)[:n]
    vb = vb.reshape(n_pages * ps, -1)[:n]
    k = dequantize_block_rows(k_raw, kb, kv, n_channels=d, tensor_scale=tensor_scale)
    v = dequantize_block_rows(v_raw, vb, kv, n_channels=d, tensor_scale=tensor_scale)
    return k, v


# ----------------------------------------------------------------------------------------------------------------
# two-pass flash maths (generic over dense float tiles - the kernel uses the same recurrence)
# ----------------------------------------------------------------------------------------------------------------
def _pass1(q: torch.Tensor, k: torch.Tensor, scale: float, mask: torch.Tensor | None,
           softcap: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax statistics of one query block over dense keys (log2 domain).

    ``q [M, D]``, ``k [N, D]`` float32, ``mask [M, N]`` bool (True = attend) or ``None``.  ``softcap`` applies the
    Gemma-2 ``c * tanh(S / c)`` squashing to the (already scaled) logits before the softmax.  Returns ``(m [M], l [M])``.
    """
    s = _logits(q, k, scale, mask, softcap)
    m = s.amax(dim=-1)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.exp2(s - m_safe.unsqueeze(-1))
    lsum = p.sum(dim=-1)
    return m, lsum


def _logits(q: torch.Tensor, k: torch.Tensor, scale: float, mask: torch.Tensor | None,
            softcap: float | None) -> torch.Tensor:
    """Scaled logits ``[M, N]`` in the log2 domain (with softcap and mask), shared by both passes."""
    s = (q @ k.transpose(-1, -2)) * scale  # natural units
    if softcap is not None:
        s = softcap * torch.tanh(s / softcap)
    s = s * LOG2E
    if mask is not None:
        s = s.masked_fill(~mask, NEG_INF)
    return s


def _pass2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lse: torch.Tensor, scale: float,
           mask: torch.Tensor | None, softcap: float | None = None, p_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Accumulate ``O = softmax(QK^T) V`` given the *final* log2-domain LSE (``lse [M]``).

    ``p_dtype`` rounds the probabilities to that float dtype before ``P @ V`` (the kernel's "quantized P" path, e.g.
    E4M3 for FP8 PV); the LSE itself is always the exact float32 one (FA4's ``track_unquantized_sum``).
    """
    p = torch.exp2(_logits(q, k, scale, mask, softcap) - lse.unsqueeze(-1))
    if p_dtype is not None:
        p = p.to(p_dtype).float()
    return p @ v


def _lse_log2(m: torch.Tensor, lsum: torch.Tensor, sink: torch.Tensor | None = None) -> torch.Tensor:
    """Log2-domain LSE from pass-1 statistics (``-inf`` for a row with no valid key).

    ``sink`` (FA4 learnable attention sink, natural-log units) adds one extra softmax column per row that contributes to
    the denominator and the LSE but has no value vector, so it does not change ``O``'s numerator.
    """
    if sink is not None:
        # Fold the sink as one extra softmax column with logit `sink` (natural units) and no value vector.  Treat it as a
        # real column of the row - bump the running max when the sink dominates and rescale `lsum` - so a large sink never
        # overflows `exp2(sink - m)` into inf.  A NaN sink is mapped to -inf (no contribution) rather than poisoning the row.
        sink_log2 = sink.to(torch.float32) * LOG2E  # [M]
        sink_log2 = torch.where(torch.isnan(sink_log2), torch.full_like(sink_log2, NEG_INF), sink_log2)
        m_new = torch.maximum(m, sink_log2)
        m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        alpha = torch.exp2(torch.where(torch.isinf(m), torch.zeros_like(m), m) - m_safe)  # -inf m -> 0
        beta = torch.exp2(torch.where(torch.isinf(sink_log2), torch.full_like(sink_log2, NEG_INF), sink_log2) - m_safe)
        lsum = lsum * alpha + beta
        m = m_new
    lsafe = torch.where(lsum == 0, torch.ones_like(lsum), lsum)
    return torch.where(lsum == 0, torch.full_like(lsum, NEG_INF), m + torch.log2(lsafe))


def _sliding_mask(n: int, qpos: int, window: int | None, device) -> torch.Tensor | None:
    if not window or window <= 0:
        return None
    kpos = torch.arange(n, device=device)
    return (kpos > (qpos - window)).unsqueeze(0)  # [1, n], broadcasts over the query rows


# ----------------------------------------------------------------------------------------------------------------
# public oracle
# ----------------------------------------------------------------------------------------------------------------
def mixed_attention_ref(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    kv_dtype: str = "bf16",
    k_scale=None,
    v_scale=None,
    k_zero=None,
    v_zero=None,
    k_block_scale: torch.Tensor | None = None,
    v_block_scale: torch.Tensor | None = None,
    tensor_scale: torch.Tensor | None = None,
    window: int | None = None,
    softcap: float | None = None,
    sink: torch.Tensor | None = None,
    p_dtype: torch.dtype | None = None,
    return_lse: bool = False,
):
    """Mixed-precision paged decode attention oracle for every KV format (the reference for ``ops_attn.mixed_attention``).

    Args:
        q: ``[B, Hq, D]`` decode queries; the decode query sits at position ``seq_len - 1``.
        k_pool, v_pool: head-major pools ``[P, Hkv, ps, D']`` (``D'`` = ``D`` for bf16/fp16/fp32/fp8/int8, ``D/2`` for
            int4/nvfp4/mxfp4, ``D/4`` for ternary).
        block_tables: ``[B, max_pages]`` int32 page ids; ``seq_lens``: ``[B]`` total cached tokens (incl. the current one).
        kv_dtype: one of :data:`MIXED_KV_DTYPES`.
        k_scale, v_scale, k_zero, v_zero: fp8/int8/int4 family scales/zeros (see :mod:`forge.kernels.ref.attn_kv`).
        k_block_scale, v_block_scale, tensor_scale: nvfp4/mxfp4/ternary block scales.
        window: sliding window (attend the last ``window`` tokens).
        softcap: Gemma-2 style ``softcap * tanh(S / softcap)`` on the logits (``None`` = off).
        sink: optional ``[Hq]`` per-head learnable sink logit (natural-log units) that enlarges the softmax denominator
            (and the LSE) without contributing a value - FA4's attention sink.
        p_dtype: if given, probabilities are rounded to this float dtype before ``P @ V`` (quantized-P path); the LSE
            stays exact float32 (FA4 ``track_unquantized_sum``).
    Returns ``out [B, Hq, D]`` in ``q.dtype`` (and ``lse [B, Hq]`` float32 if ``return_lse``).
    """
    kv = normalize_kv_dtype(kv_dtype)
    if kv not in MIXED_KV_DTYPES:
        raise ValueError(f"unsupported kv_dtype {kv_dtype!r}; expected one of {MIXED_KV_DTYPES}")
    b, hq, d = q.shape
    hkv = k_pool.shape[1]
    if hq % hkv:
        raise ValueError("Hq must be a multiple of Hkv")
    if logical_head_dim(k_pool, kv) != d:
        raise ValueError(f"pool head dim {logical_head_dim(k_pool, kv)} != q head dim {d}")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    dev = q.device
    g = hq // hkv
    out = torch.zeros(b, hq, d, dtype=torch.float32, device=dev)
    lse = torch.full((b, hq), NEG_INF, dtype=torch.float32, device=dev)

    for bi in range(b):
        n = int(seq_lens[bi])
        if n <= 0:
            continue
        n_pages = (n + k_pool.shape[2] - 1) // k_pool.shape[2]
        pages = block_tables[bi, :n_pages].long()
        mask = _sliding_mask(n, n - 1, window, dev)
        for h in range(hkv):
            k, v = _gather_head(k_pool, v_pool, pages, h, n, kv, k_scale, v_scale, k_zero, v_zero,
                                k_block_scale, v_block_scale, tensor_scale)
            qs = q[bi, h * g:(h + 1) * g].float()  # [g, D]
            mmask = mask.expand(qs.shape[0], -1) if mask is not None else None
            m, lsum = _pass1(qs, k, softmax_scale, mmask, softcap)
            sink_h = sink[h * g:(h + 1) * g].to(torch.float32) if sink is not None else None
            lse_h = _lse_log2(m, lsum, sink_h)
            o = _pass2(qs, k, v, lse_h, softmax_scale, mmask, softcap, p_dtype)
            out[bi, h * g:(h + 1) * g] = o
            lse[bi, h * g:(h + 1) * g] = lse_h * LN2  # report natural-log LSE (matches paged_decode_attention_ref)
    out = out.to(q.dtype)
    return (out, lse) if return_lse else out


__all__ = ["MIXED_KV_DTYPES", "mixed_attention_ref"]
