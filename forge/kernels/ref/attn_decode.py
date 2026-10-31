"""Pure-torch oracles for decode attention (Plan K4, KR-4).

* :func:`paged_decode_attention_ref` - paged, GQA-grouped decode attention over (optionally quantised) KV pools with
  a full-precision residual window, sliding window, ragged sequence lengths and an optional token-selection list
  (used by the sparse-attention path).
* :func:`merge_partials_ref` - the numerically stable log-sum-exp merge used by split-KV (flash-decoding).
* :func:`mla_decode_attention_ref` - absorbed multi-head-latent-attention decode (latent + rope parts).
* :func:`dense_attention_ref` - tiny helper: softmax attention of a few queries over explicit K/V with a mask.

All maths is done in float32 and cast back to the query dtype, so these are the numerical ground truth the Triton
kernels are compared against.  Conventions: the query of a decode step sits at absolute position ``seq_len - 1``
(the current token's K/V is already in the cache and counted in ``seq_lens``); query head ``h`` uses KV head
``h // (Hq // Hkv)`` (contiguous GQA grouping).
"""
from __future__ import annotations

import math

import torch

from forge.kernels.ref.attn_kv import (
    dequantize_rows,
    expand_k_scale,
    expand_v_scale,
    expand_zero_like,
    is_quantised,
    logical_head_dim,
    normalize_kv_dtype,
)

NEG_INF = float("-inf")


def dense_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None,
                        scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax attention of ``Nq`` queries over ``n`` keys for all heads.

    ``q [Nq, Hq, D]``, ``k, v [Hkv, n, D]``, ``mask [Nq, n]`` or ``[Nq, Hkv, n]`` bool (True = attend) or ``None``.
    Returns ``(out [Nq, Hq, D] float32, lse [Nq, Hq] float32)``; rows without any valid key give ``0`` / ``-inf``.
    """
    nq, hq, d = q.shape
    hkv, n, _ = k.shape
    g = hq // hkv
    qf = q.float().reshape(nq, hkv, g, d)
    s = torch.einsum("qhgd,hnd->qhgn", qf, k.float()) * scale
    if mask is not None:
        m = mask if mask.dim() == 3 else mask.unsqueeze(1).expand(nq, hkv, n)
        s = s.masked_fill(~m.unsqueeze(2), NEG_INF)
    mx = s.amax(dim=-1, keepdim=True)
    mx_safe = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
    p = torch.exp(s - mx_safe)
    l = p.sum(dim=-1, keepdim=True)
    o = torch.einsum("qhgn,hnd->qhgd", p, v.float()) / torch.where(l == 0, torch.ones_like(l), l)
    lse = torch.where(l.squeeze(-1) == 0, torch.full_like(l.squeeze(-1), NEG_INF),
                      mx_safe.squeeze(-1) + torch.log(l.squeeze(-1).clamp_min(1e-38)))
    return o.reshape(nq, hq, d), lse.reshape(nq, hq)


def merge_partials_ref(o_part: torch.Tensor, lse_part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge split-KV partial results.

    ``o_part [..., S, D]`` (each already normalised by its own softmax denominator) and ``lse_part [..., S]``
    (natural log of the partial denominators, ``-inf`` for empty splits).  Returns ``(out [..., D], lse [...])``.
    """
    o = o_part.float()
    lse = lse_part.float()
    m = lse.amax(dim=-1, keepdim=True)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w = torch.exp(lse - m_safe)  # exp(-inf) = 0 for empty splits
    z = w.sum(dim=-1, keepdim=True)
    out = (w.unsqueeze(-1) * o).sum(dim=-2) / torch.where(z == 0, torch.ones_like(z), z)
    lse_tot = torch.where(z.squeeze(-1) == 0, torch.full_like(z.squeeze(-1), NEG_INF),
                          m_safe.squeeze(-1) + torch.log(z.squeeze(-1).clamp_min(1e-38)))
    return out, lse_tot


def gather_paged_kv_ref(k_pool, v_pool, block_table_row, n_tokens: int, *, kv_dtype="bf16", k_scale=None, v_scale=None,
                        k_zero=None, v_zero=None):
    """Gather and dequantise the first ``n_tokens`` tokens of one sequence -> ``k, v [Hkv, n, D]`` float32."""
    kv = normalize_kv_dtype(kv_dtype)
    p, h, ps, _ = k_pool.shape
    d = logical_head_dim(k_pool, kv)
    n_pages = (n_tokens + ps - 1) // ps
    if n_tokens == 0:
        z = torch.zeros(h, 0, d, device=k_pool.device)
        return z, z.clone()
    pages = block_table_row[:n_pages].long()
    kr = k_pool[pages]  # [np, H, ps, D']
    vr = v_pool[pages]
    if is_quantised(kv):
        ks = expand_k_scale(k_scale, p, h, d, k_pool.device)[pages]
        kz = expand_zero_like(k_zero, expand_k_scale(k_scale, p, h, d, k_pool.device))
        kz = kz[pages] if kz is not None else None
        vs = expand_v_scale(v_scale, p, h, ps, k_pool.device)[pages]
        vz = expand_zero_like(v_zero, expand_v_scale(v_scale, p, h, ps, k_pool.device))
        vz = vz[pages] if vz is not None else None
        kf = dequantize_rows(kr, ks, kz, kv, scale_axis="channel")
        vf = dequantize_rows(vr, vs, vz, kv, scale_axis="token")
    else:
        kf, vf = kr.float(), vr.float()
    kf = kf.permute(1, 0, 2, 3).reshape(h, n_pages * ps, d)[:, :n_tokens]
    vf = vf.permute(1, 0, 2, 3).reshape(h, n_pages * ps, d)[:, :n_tokens]
    return kf, vf


def paged_decode_attention_ref(
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
    window: int | None = None,
    k_res: torch.Tensor | None = None,
    v_res: torch.Tensor | None = None,
    res_lens: torch.Tensor | None = None,
    topk_idx: torch.Tensor | None = None,
    return_lse: bool = False,
):
    """Paged decode attention oracle.

    Args:
        q: ``[B, Hq, D]`` decode queries.
        k_pool, v_pool: ``[P, Hkv, page_size, D]`` (int4: last dim ``D/2``, packed).
        block_tables: ``[B, max_pages]`` int32 physical page ids.
        seq_lens: ``[B]`` total tokens per sequence **including** the residual window tokens.
        kv_dtype: one of ``bf16 fp16 fp32 fp8_e4m3 int8 int4``.
        k_scale, v_scale, k_zero, v_zero: see :mod:`forge.kernels.ref.attn_kv`.
        window: sliding window - attend only to the last ``window`` tokens.
        k_res, v_res: ``[B, Hkv, R, D]`` full-precision residual window; the last ``res_lens[b]`` tokens of the
            sequence live there (token ``pool_len + j`` is ``k_res[b, :, j]``), the pool holds the first
            ``seq_len - res_len`` tokens.
        topk_idx: optional ``[B, Hsel, K]`` int32 selected token positions (``-1`` = unused); ``Hsel`` is ``1`` (shared
            by all heads) or ``Hkv``.  Only selected tokens (that exist) are attended (sparse attention).
    Returns ``out [B, Hq, D]`` in ``q.dtype`` (and ``lse [B, Hq]`` float32 if ``return_lse``).
    """
    b, hq, d = q.shape
    hkv = k_pool.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    dev = q.device
    out = torch.zeros(b, hq, d, dtype=torch.float32, device=dev)
    lse = torch.full((b, hq), NEG_INF, dtype=torch.float32, device=dev)
    for bi in range(b):
        n = int(seq_lens[bi])
        if n <= 0:
            continue
        r = int(res_lens[bi]) if res_lens is not None else 0
        pool_n = n - r
        kf, vf = gather_paged_kv_ref(k_pool, v_pool, block_tables[bi], pool_n, kv_dtype=kv_dtype, k_scale=k_scale,
                                     v_scale=v_scale, k_zero=k_zero, v_zero=v_zero)
        if r > 0:
            kf = torch.cat([kf, k_res[bi, :, :r].float()], dim=1)
            vf = torch.cat([vf, v_res[bi, :, :r].float()], dim=1)
        pos = torch.arange(n, device=dev)
        mask = torch.ones(1, n, dtype=torch.bool, device=dev)
        if window is not None and window > 0:
            mask = mask & (pos >= n - window).unsqueeze(0)
        if topk_idx is not None:
            idx = topk_idx[bi].long().to(dev)  # [Hsel, K]
            sel = torch.zeros(idx.shape[0], n, dtype=torch.bool, device=dev)
            ok = (idx >= 0) & (idx < n)
            for hs in range(idx.shape[0]):
                sel[hs, idx[hs][ok[hs]]] = True
            m = (mask & sel).unsqueeze(0)  # [1, Hsel, n]
            if sel.shape[0] == 1:
                m = m.expand(1, hkv, n)
            mask3 = m
        else:
            mask3 = mask.unsqueeze(0).expand(1, hkv, n)
        o, l = dense_attention_ref(q[bi : bi + 1], kf, vf, mask3, softmax_scale)
        out[bi] = o[0]
        lse[bi] = l[0]
    out = out.to(q.dtype)
    return (out, lse) if return_lse else out


def split_kv_decode_ref(q, k_pool, v_pool, block_tables, seq_lens, num_splits: int, **kw):
    """Compute decode attention as ``num_splits`` independent chunks and merge them with
    :func:`merge_partials_ref` - emulates the flash-decoding schedule on CPU (used to test the merge)."""
    b, hq, d = q.shape
    ps = k_pool.shape[2]
    hkv = k_pool.shape[1]
    scale = kw.pop("softmax_scale", None) or 1.0 / math.sqrt(d)
    kv_dtype = kw.pop("kv_dtype", "bf16")
    outs, lses = [], []
    for bi in range(b):
        n = int(seq_lens[bi])
        kf, vf = gather_paged_kv_ref(k_pool, v_pool, block_tables[bi], n, kv_dtype=kv_dtype,
                                     k_scale=kw.get("k_scale"), v_scale=kw.get("v_scale"),
                                     k_zero=kw.get("k_zero"), v_zero=kw.get("v_zero"))
        chunk = max(ps, ((n + num_splits - 1) // num_splits + ps - 1) // ps * ps)
        os_, ls_ = [], []
        for s in range(num_splits):
            lo, hi = s * chunk, min((s + 1) * chunk, n)
            if lo >= hi:
                os_.append(torch.zeros(hq, d, device=q.device)); ls_.append(torch.full((hq,), NEG_INF, device=q.device)); continue
            o, l = dense_attention_ref(q[bi : bi + 1], kf[:, lo:hi], vf[:, lo:hi], None, scale)
            os_.append(o[0]); ls_.append(l[0])
        outs.append(torch.stack(os_, dim=1)); lses.append(torch.stack(ls_, dim=1))
    o_part = torch.stack(outs)  # [B, Hq, S, D]
    l_part = torch.stack(lses)
    o, _ = merge_partials_ref(o_part, l_part)
    return o.to(q.dtype)


def mla_decode_attention_ref(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_pool: torch.Tensor,
    r_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    softmax_scale: float,
    w_uv: torch.Tensor | None = None,
    c_scale: torch.Tensor | None = None,
    kv_dtype: str = "bf16",
    window: int | None = None,
    topk_idx: torch.Tensor | None = None,
    return_lse: bool = False,
):
    """Absorbed MLA decode oracle (Plan K4 "MLA variant", P3 F2).

    Args:
        q_nope: ``[B, H, Dc]`` queries already multiplied by ``W_UK`` (absorbed), i.e. living in latent space.
        q_rope: ``[B, H, Dr]`` rotary part of the query.
        c_pool: ``[P, page_size, Dc]`` latent cache (shared by all heads); fp8 with per-token ``c_scale [P, ps]``
            supported (``kv_dtype="fp8_e4m3"``) or bf16/fp16.
        r_pool: ``[P, page_size, Dr]`` rotary key cache (kept in the native float dtype).
        w_uv: optional ``[H, Dc, Dv]`` value up-projection; if given the latent output is projected to ``[B, H, Dv]``.
        topk_idx: optional ``[B, K]`` selected token positions (DSA-style sparse attention), ``-1`` = unused.
    Scores are ``(q_nope . c + q_rope . r) * softmax_scale``; values are the latent ``c`` itself.
    """
    b, h, dc = q_nope.shape
    kv = normalize_kv_dtype(kv_dtype)
    ps = c_pool.shape[1]
    dev = q_nope.device
    out = torch.zeros(b, h, dc, dtype=torch.float32, device=dev)
    lse = torch.full((b, h), NEG_INF, device=dev)
    for bi in range(b):
        n = int(seq_lens[bi])
        if n <= 0:
            continue
        n_pages = (n + ps - 1) // ps
        pages = block_tables[bi, :n_pages].long()
        c = c_pool[pages]
        if kv == "fp8_e4m3":
            cf = (c if c.dtype == torch.float8_e4m3fn else c.view(torch.float8_e4m3fn)).float()
            if c_scale is not None:
                cf = cf * torch.as_tensor(c_scale, dtype=torch.float32)[pages].unsqueeze(-1)
        else:
            cf = c.float()
        cf = cf.reshape(n_pages * ps, dc)[:n]
        rf = r_pool[pages].float().reshape(n_pages * ps, -1)[:n]
        s = (q_nope[bi].float() @ cf.T + q_rope[bi].float() @ rf.T) * softmax_scale  # [H, n]
        pos = torch.arange(n, device=dev)
        mask = torch.ones(n, dtype=torch.bool, device=dev)
        if window is not None and window > 0:
            mask &= pos >= n - window
        if topk_idx is not None:
            idx = topk_idx[bi].long().to(dev)
            idx = idx[(idx >= 0) & (idx < n)]
            sel = torch.zeros(n, dtype=torch.bool, device=dev)
            sel[idx] = True
            mask &= sel
        s = s.masked_fill(~mask.unsqueeze(0), NEG_INF)
        mx = s.amax(dim=-1, keepdim=True)
        mxs = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        p = torch.exp(s - mxs)
        l = p.sum(-1, keepdim=True)
        out[bi] = (p @ cf) / torch.where(l == 0, torch.ones_like(l), l)
        lse[bi] = torch.where(l.squeeze(-1) == 0, torch.full_like(l.squeeze(-1), NEG_INF),
                              mxs.squeeze(-1) + torch.log(l.squeeze(-1).clamp_min(1e-38)))
    if w_uv is not None:
        out = torch.einsum("bhc,hcv->bhv", out, w_uv.float())
    out = out.to(q_nope.dtype)
    return (out, lse) if return_lse else out
