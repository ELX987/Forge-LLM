"""Pure-torch oracles for prefill / chunked-prefill attention and paged KV writes (Plan K4/K5 prefill side).

Causal convention (identical to flash-attn "bottom-right aligned"): a sequence has ``q_len`` queries and ``k_len``
keys with ``k_len >= q_len``; query ``i`` sits at absolute position ``k_len - q_len + i`` and attends to keys
``j <= k_len - q_len + i`` (and, with a sliding ``window``, ``j > k_len - q_len + i - window``).  A prefix-cache hit or
chunked prefill is therefore simply ``k_len > q_len`` (the keys contain the cached prefix + the new chunk).
"""
from __future__ import annotations

import math

import torch

from forge.kernels.ref.attn_decode import NEG_INF


def _attend_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool, window: int | None):
    """``q [Sq, Hq, D]``, ``k, v [Sk, Hkv, D]`` -> ``[Sq, Hq, D]`` float32."""
    sq, hq, d = q.shape
    sk, hkv, _ = k.shape
    if sq == 0:
        return torch.zeros(0, hq, d, dtype=torch.float32, device=q.device)
    g = hq // hkv
    s = torch.einsum("qhgd,khd->qhgk", q.float().reshape(sq, hkv, g, d), k.float()) * scale
    qpos = (sk - sq) + torch.arange(sq, device=q.device)
    kpos = torch.arange(sk, device=q.device)
    mask = torch.ones(sq, sk, dtype=torch.bool, device=q.device)
    if causal:
        mask &= kpos.unsqueeze(0) <= qpos.unsqueeze(1)
    if window is not None and window > 0:
        mask &= kpos.unsqueeze(0) > (qpos.unsqueeze(1) - window)
    s = s.masked_fill(~mask[:, None, None, :], NEG_INF)
    mx = s.amax(-1, keepdim=True)
    mxs = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
    p = torch.exp(s - mxs)
    l = p.sum(-1, keepdim=True)
    p = p / torch.where(l == 0, torch.ones_like(l), l)
    o = torch.einsum("qhgk,khd->qhgd", p, v.float())
    return o.reshape(sq, hq, d)


def prefill_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    window: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    layout: str = "bshd",
) -> torch.Tensor:
    """Flash-style attention oracle.

    * ``layout="bshd"``: ``q [B, Sq, Hq, D]``, ``k, v [B, Sk, Hkv, D]`` -> ``[B, Sq, Hq, D]``.
    * ``layout="bhsd"``: ``q [B, Hq, Sq, D]``, ``k, v [B, Hkv, Sk, D]`` -> ``[B, Hq, Sq, D]``.
    * varlen (``cu_seqlens_q`` given): ``q [Tq, Hq, D]``, ``k, v [Tk, Hkv, D]`` packed, ``cu_seqlens_* [B+1]`` int32.
    """
    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    if cu_seqlens_q is not None:
        cu_k = cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q
        out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        for i in range(len(cu_seqlens_q) - 1):
            qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            ks, ke = int(cu_k[i]), int(cu_k[i + 1])
            out[qs:qe] = _attend_one(q[qs:qe], k[ks:ke], v[ks:ke], scale, causal, window)
        return out.to(q.dtype)
    if layout == "bhsd":
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    elif layout != "bshd":
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")
    outs = [_attend_one(q[i], k[i], v[i], scale, causal, window) for i in range(q.shape[0])]
    out = torch.stack(outs).to(q.dtype)
    return out.transpose(1, 2) if layout == "bhsd" else out


def paged_kv_write_ref(k: torch.Tensor, v: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                       slot_mapping: torch.Tensor) -> None:
    """Write per-token ``k, v [T, Hkv, D]`` into pools ``[P, Hkv, ps, D]`` (same dtype) at ``slot_mapping [T]``
    (``slot = page * page_size + offset``; negative slots are skipped).  In place."""
    ps = k_pool.shape[2]
    slot = slot_mapping.long()
    ok = slot >= 0
    page = torch.div(slot[ok], ps, rounding_mode="floor")
    off = slot[ok] % ps
    k_pool[page, :, off, :] = k[ok].to(k_pool.dtype)
    v_pool[page, :, off, :] = v[ok].to(v_pool.dtype)


def gather_prefix_kv_ref(k_pool, v_pool, block_tables, ctx_lens, *, kv_dtype="bf16", k_scale=None, v_scale=None,
                         k_zero=None, v_zero=None):
    """Gather the first ``ctx_lens[b]`` tokens of every sequence from the pools into packed float32
    ``k, v [sum(ctx_lens), Hkv, D]`` plus ``cu_seqlens [B+1]`` (used to run chunked prefill over a paged prefix)."""
    from forge.kernels.ref.attn_decode import gather_paged_kv_ref

    ks, vs, cu = [], [], [0]
    for bi in range(block_tables.shape[0]):
        n = int(ctx_lens[bi])
        kf, vf = gather_paged_kv_ref(k_pool, v_pool, block_tables[bi], n, kv_dtype=kv_dtype, k_scale=k_scale,
                                     v_scale=v_scale, k_zero=k_zero, v_zero=v_zero)
        ks.append(kf.permute(1, 0, 2)); vs.append(vf.permute(1, 0, 2)); cu.append(cu[-1] + n)
    k_all = torch.cat(ks, 0) if ks else torch.zeros(0)
    v_all = torch.cat(vs, 0) if vs else torch.zeros(0)
    return k_all, v_all, torch.tensor(cu, dtype=torch.int32, device=k_pool.device)
