"""Oracles for speculative-decoding kernels (Plan K6, KR-6/KR-7; SpecDec paper P7 §6, App. A.1).

* tree attention with an arbitrary ancestor mask and per-node positions,
* exact rejection-sampling verification (accept ``min(1, p/q)``, resample from ``(p-q)+ / Z``, bonus token) driven by a
  counter-based RNG (reproducible and identical in the Triton kernel),
* greedy tree verification and KV accept/rollback compaction over ragged accept lengths.
"""
from __future__ import annotations

import math

import torch

from forge.kernels.ref.attn_decode import NEG_INF, dense_attention_ref, gather_paged_kv_ref, merge_partials_ref
from forge.kernels.ref.util_rng import philox_uniform


# ------------------------------------------------------------------------------------------------------------
# tree helpers
# ------------------------------------------------------------------------------------------------------------
def tree_depths(parents: torch.Tensor) -> torch.Tensor:
    """Depth of every node (root = 0) from a parent array (``-1`` = child of the committed context). Parents must
    appear before their children (topological order)."""
    n = parents.numel()
    depth = torch.zeros(n, dtype=torch.int32)
    par = parents.tolist()
    for i in range(n):
        depth[i] = 0 if par[i] < 0 else int(depth[par[i]]) + 1
    return depth


def build_tree_mask(parents: torch.Tensor) -> torch.Tensor:
    """``[n, n]`` bool, ``mask[i, j] = True`` iff node ``j`` is node ``i`` itself or one of its ancestors."""
    n = parents.numel()
    m = torch.zeros(n, n, dtype=torch.bool)
    par = parents.tolist()
    for i in range(n):
        m[i, i] = True
        p = par[i]
        while p >= 0:
            m[i, p] = True
            p = par[p]
    return m


def tree_position_ids(parents: torch.Tensor, base_len: int | torch.Tensor) -> torch.Tensor:
    """Absolute positions of tree nodes: ``base_len + depth`` (``base_len`` int or ``[B]`` tensor -> ``[B, n]``)."""
    d = tree_depths(parents).long()
    if isinstance(base_len, int):
        return d + base_len
    return d.unsqueeze(0) + base_len.long().unsqueeze(1)


def chain_parents(n: int) -> torch.Tensor:
    """Parent array of a linear chain ``0 <- 1 <- 2 ...``."""
    return torch.arange(n, dtype=torch.int32) - 1


# ------------------------------------------------------------------------------------------------------------
# tree attention
# ------------------------------------------------------------------------------------------------------------
def tree_attention_ref(
    q: torch.Tensor,
    k_tree: torch.Tensor,
    v_tree: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    ctx_lens: torch.Tensor,
    tree_mask: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    kv_dtype: str = "bf16",
    k_scale=None,
    v_scale=None,
    k_zero=None,
    v_zero=None,
    window: int | None = None,
    node_pos: torch.Tensor | None = None,
    k_res: torch.Tensor | None = None,
    v_res: torch.Tensor | None = None,
    res_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tree-verification attention oracle.

    ``q [B, N, Hq, D]`` are the queries of ``N`` draft-tree nodes; ``k_tree, v_tree [B, N, Hkv, D]`` their (already
    RoPE'd) keys/values; the committed context (``ctx_lens[b]`` tokens) lives in the paged pools.  Node ``i`` attends to
    **all** context tokens (restricted by ``window`` relative to its position) and to the tree nodes ``j`` with
    ``tree_mask[(b,) i, j]``.  ``node_pos [B, N]`` defaults to ``ctx_len + depth`` where depth = (#ancestors incl. self) - 1.
    With ``k_res, v_res [B, Hkv, R, D]`` + ``res_lens [B]`` the **last** ``res_lens[b]`` of the ``ctx_lens[b]`` context tokens live in
    the full-precision residual window and the pool holds only the first ``ctx_len - res_len`` (same convention as the decode kernel).
    """
    b, n, hq, d = q.shape
    hkv = k_tree.shape[2]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    out = torch.zeros(b, n, hq, d, dtype=torch.float32, device=q.device)
    for bi in range(b):
        c = int(ctx_lens[bi])
        r = int(res_lens[bi]) if (k_res is not None and res_lens is not None) else 0
        kf, vf = gather_paged_kv_ref(k_pool, v_pool, block_tables[bi], c - r, kv_dtype=kv_dtype, k_scale=k_scale,
                                     v_scale=v_scale, k_zero=k_zero, v_zero=v_zero)
        if r > 0:
            kf = torch.cat([kf, k_res[bi, :, :r].float()], dim=1)
            vf = torch.cat([vf, v_res[bi, :, :r].float()], dim=1)
        kt = k_tree[bi].float().permute(1, 0, 2)  # [Hkv, N, D]
        vt = v_tree[bi].float().permute(1, 0, 2)
        k_all = torch.cat([kf, kt], dim=1)
        v_all = torch.cat([vf, vt], dim=1)
        tm = tree_mask[bi] if tree_mask.dim() == 3 else tree_mask
        tm = tm.to(q.device)
        if node_pos is not None:
            npos = node_pos[bi].long()
        else:
            npos = c + tm.sum(-1).long() - 1
        ctx_pos = torch.arange(c, device=q.device)
        m_ctx = torch.ones(n, c, dtype=torch.bool, device=q.device)
        if window is not None and window > 0:
            m_ctx = ctx_pos.unsqueeze(0) > (npos.unsqueeze(1) - window)
        mask = torch.cat([m_ctx, tm.bool()], dim=1)  # [N, c+N]
        o, _ = dense_attention_ref(q[bi], k_all, v_all, mask, scale)
        out[bi] = o
    return out.to(q.dtype)


# ------------------------------------------------------------------------------------------------------------
# rejection sampling verification
# ------------------------------------------------------------------------------------------------------------
def rng_counter(batch_idx: torch.Tensor, pos: torch.Tensor, kind: int, gamma: int, offset: int = 0) -> torch.Tensor:
    """RNG offset used by :func:`rejection_verify_ref` and the Triton kernel:
    ``((offset + b) * (gamma + 1) + pos) * 2 + kind`` (``kind`` 0 = accept test, 1 = resample/bonus draw)."""
    return ((offset + batch_idx.long()) * (gamma + 1) + pos.long()) * 2 + kind


def _sample_inverse_cdf(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Sample index ~ weights/sum(weights) via inverse CDF with ``u in [0,1)``; first index with ``cumsum > u*Z``.
    Falls back to the last index with positive weight when rounding makes the CDF never exceed the target."""
    z = weights.sum(-1, keepdim=True)
    cdf = torch.cumsum(weights, dim=-1)
    target = u.unsqueeze(-1) * z
    hit = cdf > target
    v = weights.shape[-1]
    idx = torch.arange(v, device=weights.device).expand_as(weights)
    first = torch.where(hit, idx, torch.full_like(idx, v)).amin(-1)
    last_pos = torch.where(weights > 0, idx, torch.full_like(idx, -1)).amax(-1)
    return torch.where(first >= v, last_pos.clamp_min(0), first)


def rejection_verify_ref(draft_tokens: torch.Tensor, q_probs: torch.Tensor, p_probs: torch.Tensor, seed: int,
                         offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact speculative-sampling verification for a chain of ``gamma`` draft tokens.

    ``draft_tokens [B, G]``; ``q_probs [B, G, V]`` draft distributions; ``p_probs [B, G+1, V]`` target distributions
    (row ``i`` is the target law of the token at draft position ``i``; row ``G`` is the bonus position).  Returns
    ``(tokens [B, G+1] int32, n_accepted [B] int32)``: ``tokens[b, :n_acc+1]`` are the emitted tokens (accepted drafts
    followed by one resampled/bonus token), the rest ``-1``.  The emitted sequence is distributed exactly as
    autoregressive sampling from the target (Leviathan et al. 2023, Theorem 1).
    """
    b, g = draft_tokens.shape
    v = p_probs.shape[-1]
    dev = draft_tokens.device
    pf, qf = p_probs.float(), q_probs.float()
    x = draft_tokens.long()
    px = pf[:, :g].gather(-1, x.unsqueeze(-1)).squeeze(-1)
    qx = qf.gather(-1, x.unsqueeze(-1)).squeeze(-1)
    bi = torch.arange(b, device=dev).unsqueeze(1).expand(b, g)
    pi = torch.arange(g, device=dev).unsqueeze(0).expand(b, g)
    u = philox_uniform(seed, rng_counter(bi, pi, 0, g, offset))
    accept = (u * qx) < px  # u < min(1, p/q) without a division (exact for q == 0)
    n_acc = torch.cumprod(accept.to(torch.int64), dim=1).sum(1)  # [B]
    rows = torch.arange(b, device=dev)
    p_stop = pf[rows, n_acc]  # [B, V] target law at the stopping position
    q_stop = torch.cat([qf, torch.zeros(b, 1, v, device=dev)], dim=1)[rows, n_acc]
    resid = (p_stop - q_stop).clamp_min(0.0)
    resid = torch.where(n_acc.unsqueeze(-1) < g, resid, p_stop)  # bonus position samples from p directly
    zr = resid.sum(-1, keepdim=True)
    resid = torch.where(zr > 0, resid, p_stop)  # numerical safety: p == q exactly
    u2 = philox_uniform(seed, rng_counter(rows, n_acc, 1, g, offset))
    new_tok = _sample_inverse_cdf(resid, u2)
    out = torch.full((b, g + 1), -1, dtype=torch.int32, device=dev)
    pos = torch.arange(g + 1, device=dev).unsqueeze(0)
    pad = torch.cat([draft_tokens.to(torch.int32), torch.zeros(b, 1, dtype=torch.int32, device=dev)], dim=1)
    out = torch.where(pos < n_acc.unsqueeze(1), pad, out)
    out = torch.where(pos == n_acc.unsqueeze(1), new_tok.to(torch.int32).unsqueeze(1).expand(b, g + 1), out)
    return out, n_acc.to(torch.int32)


def greedy_verify_ref(draft_tokens: torch.Tensor, target_argmax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy (temperature 0) chain verification: accept while ``draft[i] == argmax p_i``; emit the target argmax at
    the first mismatch (or the bonus position).  ``target_argmax [B, G+1]``."""
    b, g = draft_tokens.shape
    match = (draft_tokens.long() == target_argmax[:, :g].long())
    n_acc = torch.cumprod(match.to(torch.int64), 1).sum(1)
    rows = torch.arange(b, device=draft_tokens.device)
    new_tok = target_argmax.long()[rows, n_acc]
    pos = torch.arange(g + 1, device=draft_tokens.device).unsqueeze(0)
    pad = torch.cat([draft_tokens.long(), torch.zeros(b, 1, dtype=torch.long, device=draft_tokens.device)], 1)
    out = torch.full((b, g + 1), -1, dtype=torch.long, device=draft_tokens.device)
    out = torch.where(pos < n_acc.unsqueeze(1), pad, out)
    out = torch.where(pos == n_acc.unsqueeze(1), new_tok.unsqueeze(1).expand(b, g + 1), out)
    return out.to(torch.int32), n_acc.to(torch.int32)


def tree_greedy_verify_ref(tokens: torch.Tensor, parents: torch.Tensor, target_argmax: torch.Tensor,
                           root_pred: torch.Tensor):
    """Greedy verification of a draft tree.

    ``tokens [B, N]`` draft tokens, ``parents [N]`` (``-1`` = child of the committed context; parents precede
    children), ``target_argmax [B, N]`` argmax of the target logits **at** each node (= prediction for its children),
    ``root_pred [B]`` argmax at the last committed token (= prediction for root nodes).

    Returns ``(accepted_idx [B, N] int32 (-1 padded, ordered root -> leaf), accept_len [B] int32, bonus [B] int32)``.
    The accepted path is the deepest path whose every node matches its parent's prediction (ties: lowest node index).
    """
    b, n = tokens.shape
    par = parents.tolist()
    depth = tree_depths(parents).tolist()
    acc_idx = torch.full((b, n), -1, dtype=torch.int32)
    acc_len = torch.zeros(b, dtype=torch.int32)
    bonus = torch.zeros(b, dtype=torch.int32)
    for bi in range(b):
        ok = [False] * n
        for i in range(n):
            pred = int(root_pred[bi]) if par[i] < 0 else int(target_argmax[bi, par[i]])
            ok[i] = (par[i] < 0 or ok[par[i]]) and int(tokens[bi, i]) == pred
        best, best_d = -1, -1
        for i in range(n):
            if ok[i] and depth[i] > best_d:
                best, best_d = i, depth[i]
        path = []
        c = best
        while c >= 0:
            path.append(c)
            c = par[c]
        path.reverse()
        for t, node in enumerate(path):
            acc_idx[bi, t] = node
        acc_len[bi] = len(path)
        bonus[bi] = int(target_argmax[bi, best]) if best >= 0 else int(root_pred[bi])
    return acc_idx, acc_len, bonus


def kv_compact_ref(k_pool: torch.Tensor, v_pool: torch.Tensor, block_tables: torch.Tensor, ctx_lens: torch.Tensor,
                   accepted_idx: torch.Tensor, accept_lens: torch.Tensor) -> None:
    """KV accept/rollback compaction, in place on ``[P, Hkv, ps, D]`` pools.

    Tree node ``j`` of sequence ``b`` was written at absolute position ``ctx_len + j``.  After verification the accepted
    path ``accepted_idx[b, :accept_len]`` must occupy positions ``ctx_len .. ctx_len + accept_len - 1`` in path order;
    this copies row ``ctx_len + accepted_idx[b, t]`` -> ``ctx_len + t`` for every ``t`` (all sources are read before any
    write, so overlapping moves are safe).  Rejected nodes simply stay beyond the new sequence length (rollback is the
    caller shrinking ``seq_len``).
    """
    ps = k_pool.shape[2]
    for bi in range(block_tables.shape[0]):
        c = int(ctx_lens[bi])
        a = int(accept_lens[bi])
        if a == 0:
            continue
        src = (c + accepted_idx[bi, :a].long())
        dst = c + torch.arange(a, device=src.device)
        sp = block_tables[bi][torch.div(src, ps, rounding_mode="floor")].long()
        so = src % ps
        dp = block_tables[bi][torch.div(dst, ps, rounding_mode="floor")].long()
        do = dst % ps
        kk = k_pool[sp, :, so, :].clone()
        vv = v_pool[sp, :, so, :].clone()
        k_pool[dp, :, do, :] = kk
        v_pool[dp, :, do, :] = vv


__all__ = [
    "tree_depths", "build_tree_mask", "tree_position_ids", "chain_parents", "tree_attention_ref", "rng_counter",
    "rejection_verify_ref", "greedy_verify_ref", "tree_greedy_verify_ref", "kv_compact_ref", "merge_partials_ref",
    "NEG_INF",
]
