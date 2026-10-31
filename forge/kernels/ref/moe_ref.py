"""Pure-torch oracles for the MoE kernels (Plan K7, KR-9).

Pipeline: ``moe_route_ref`` (softmax/top-k) -> ``apply_expert_map_ref`` (pruned/merged experts) -> ``moe_permute_ref``
(deterministic grouping of token slots by expert) -> ``moe_grouped_gemm_ref`` (variable per-expert M, including M = 0)
-> ``moe_unpermute_combine_ref`` (weighted combine).  Ordering is fully deterministic: ties in routing go to the lower
expert id, and tokens inside an expert slab are ordered by their flat assignment index ``token * top_k + slot``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# ------------------------------------------------------------------------------------------------------------
# routing
# ------------------------------------------------------------------------------------------------------------
def moe_route_ref(logits: torch.Tensor, top_k: int, *, scoring: str = "softmax", norm_topk_prob: bool = True,
                  routed_scaling: float = 1.0, score_bias: torch.Tensor | None = None):
    """Route tokens: returns ``(weights [T, k] float32, ids [T, k] int32)``.

    ``scoring="softmax"``: ``p = softmax(logits)``; pick top-k of ``p``; weights = ``p[ids]`` (renormalised to sum 1 if
    ``norm_topk_prob``).  ``scoring="sigmoid"`` (DeepSeek-V3 style): ``s = sigmoid(logits)``; selection by
    ``s + score_bias`` but weights from the unbiased ``s``.  Ties -> lower expert id.  ``routed_scaling`` multiplies the
    final weights.
    """
    lf = logits.float()
    if scoring == "softmax":
        s = torch.softmax(lf, dim=-1)
        sel = s
    elif scoring == "sigmoid":
        s = torch.sigmoid(lf)
        sel = s + (score_bias.float() if score_bias is not None else 0.0)
    else:
        raise ValueError(f"unknown scoring {scoring!r}")
    order = torch.sort(sel, dim=-1, descending=True, stable=True).indices[:, :top_k]
    w = s.gather(-1, order)
    if norm_topk_prob:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    return w * routed_scaling, order.to(torch.int32)


def apply_expert_map_ref(topk_ids: torch.Tensor, topk_weights: torch.Tensor, expert_map: torch.Tensor,
                         renormalize: bool = False):
    """Expert indirection for pruned / merged experts.

    ``expert_map [E_orig]`` int: original expert id -> kept expert slot, ``-1`` = pruned (the slot is dropped, its weight
    set to zero).  Merged experts simply map several originals to one slot.  Returns ``(ids, weights)`` with dropped
    slots marked ``-1`` / weight 0; with ``renormalize`` the surviving weights are rescaled to the original row sum.
    """
    em = expert_map.to(topk_ids.device).long()
    mapped = em[topk_ids.long()]
    keep = mapped >= 0
    w = torch.where(keep, topk_weights, torch.zeros_like(topk_weights))
    if renormalize:
        tot = topk_weights.sum(-1, keepdim=True)
        w = w * (tot / w.sum(-1, keepdim=True).clamp_min(1e-20))
        w = torch.where(keep.any(-1, keepdim=True), w, torch.zeros_like(w))
    return mapped.to(torch.int32), w


@dataclass
class PermuteResult:
    """Output of :func:`moe_permute_ref`.

    * ``token_idx [M]`` - source token (row of ``x``) for each sorted slot ``M = T * k`` (invalid slots point to token 0
      and sit at the tail; only the first ``num_valid`` rows are meaningful),
    * ``slot_idx [M]`` - the flat assignment index ``token * k + slot`` of every sorted row,
    * ``expert_offsets [E + 1]`` int32 prefix offsets (M = 0 experts have equal consecutive offsets),
    * ``inv_perm [T * k]`` - sorted-row index of every flat assignment (``-1`` for dropped assignments),
    * ``num_valid`` - 0-d tensor with the number of valid (non-dropped) assignments.
    """

    token_idx: torch.Tensor
    slot_idx: torch.Tensor
    expert_offsets: torch.Tensor
    inv_perm: torch.Tensor
    num_valid: torch.Tensor


def moe_permute_ref(topk_ids: torch.Tensor, num_experts: int) -> PermuteResult:
    """Deterministic grouping of ``T * k`` token slots by expert id (stable sort on the flat assignment index)."""
    t, k = topk_ids.shape
    flat = topk_ids.reshape(-1).long()
    valid = (flat >= 0) & (flat < num_experts)
    key = torch.where(valid, flat, torch.full_like(flat, num_experts))
    order = torch.sort(key, stable=True).indices  # sorted position -> flat assignment
    # scatter_add instead of bincount/boolean indexing: no host synchronisation (CUDA-graph friendly)
    counts = torch.zeros(num_experts + 1, dtype=torch.long, device=flat.device).scatter_add_(0, key, torch.ones_like(key))
    counts = counts[:num_experts]
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=flat.device)
    inv = torch.where(valid, inv, torch.full_like(inv, -1))
    token_idx = torch.div(order, k, rounding_mode="floor")
    sorted_valid = valid[order]
    token_idx = torch.where(sorted_valid, token_idx, torch.zeros_like(token_idx))
    return PermuteResult(token_idx.to(torch.int32), order.to(torch.int32), offsets, inv.to(torch.int32),
                         valid.sum())


def build_block_schedule_ref(offsets: torch.Tensor, block_m: int, max_blocks: int):
    """Tile schedule for the grouped GEMM that needs **no host synchronisation**.

    Every expert owns ``ceil(count / block_m)`` row-tiles (0 for empty experts).  Returns three int32 vectors of length
    ``max_blocks`` (>= ``ceil(M / block_m) + E`` always suffices): ``block_expert`` (``-1`` = unused tile),
    ``block_row_start`` (first sorted row) and ``block_rows`` (valid rows, 0 for unused tiles).
    """
    off = offsets.long()
    e = off.numel() - 1
    counts = off[1:] - off[:-1]
    nblk = (counts + block_m - 1) // block_m
    cum = torch.cumsum(nblk, 0)
    starts = cum - nblk
    total = cum[-1] if e > 0 else torch.zeros((), dtype=torch.long, device=off.device)
    b = torch.arange(max_blocks, device=off.device, dtype=torch.long)
    ex = torch.searchsorted(cum.contiguous(), b, right=True).clamp(max=max(e - 1, 0))
    valid = b < total
    row_start = off[ex] + (b - starts[ex]) * block_m
    rows = torch.clamp(off[ex + 1] - row_start, max=block_m)
    rows = torch.where(valid, rows, torch.zeros_like(rows))
    ex = torch.where(valid, ex, torch.full_like(ex, -1))
    row_start = torch.where(valid, row_start, torch.zeros_like(row_start))
    return ex.to(torch.int32), row_start.to(torch.int32), rows.to(torch.int32)


def max_blocks_bound(total_rows: int, num_experts: int, block_m: int) -> int:
    """Static upper bound of the number of row-tiles (``ceil(M/BM) + E``)."""
    return (total_rows + block_m - 1) // block_m + num_experts


# ------------------------------------------------------------------------------------------------------------
# W4A16 expert weights (portable "plain" format used by the grouped GEMM W4 hook)
# ------------------------------------------------------------------------------------------------------------
def pack_w4_experts(w: torch.Tensor, group_size: int = 128, sym: bool = True):
    """Quantise expert weights ``w [E, N, K]`` to 4 bit groupwise along ``K``.

    Returns ``(qweight uint8 [E, N, K/2], scales float [E, N, K/G], zeros uint8 [E, N, K/G] | None)``; two offset
    nibbles per byte (low nibble = even ``k``).  Dequant: ``(q - z) * s`` with ``z = 8`` for symmetric (``zeros is None``).
    """
    e, n, k = w.shape
    if k % group_size or k % 2:
        raise ValueError("K must be a multiple of group_size (and even)")
    g = w.float().reshape(e, n, k // group_size, group_size)
    if sym:
        s = g.abs().amax(-1).clamp_min(1e-8) / 7.0
        q = torch.round(g / s.unsqueeze(-1)).clamp(-8, 7) + 8
        z = None
    else:
        mn, mx = g.amin(-1), g.amax(-1)
        s = ((mx - mn) / 15.0).clamp_min(1e-8)
        z = torch.round(-mn / s).clamp(0, 15)
        q = torch.round(g / s.unsqueeze(-1) + z.unsqueeze(-1)).clamp(0, 15)
        z = z.to(torch.uint8)
    q = q.reshape(e, n, k).to(torch.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    return packed, s.to(w.dtype if w.dtype.is_floating_point else torch.float32), z


def dequant_w4_experts(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor | None, group_size: int):
    """Inverse of :func:`pack_w4_experts` -> float32 ``[E, N, K]``."""
    lo = (qweight & 0x0F).to(torch.float32)
    hi = (qweight >> 4).to(torch.float32)
    q = torch.stack([lo, hi], dim=-1).reshape(*qweight.shape[:-1], qweight.shape[-1] * 2)
    e, n, k = q.shape
    z = 8.0 if zeros is None else zeros.float().unsqueeze(-1)
    qg = q.reshape(e, n, k // group_size, group_size)
    return ((qg - z) * scales.float().unsqueeze(-1)).reshape(e, n, k)


# ------------------------------------------------------------------------------------------------------------
# grouped GEMM
# ------------------------------------------------------------------------------------------------------------
def moe_grouped_gemm_ref(x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor, *, gather_idx: torch.Tensor | None = None,
                         out_idx: torch.Tensor | None = None, out: torch.Tensor | None = None, accumulate: bool = False,
                         alpha: torch.Tensor | None = None, w_scales: torch.Tensor | None = None,
                         w_zeros: torch.Tensor | None = None, group_size: int | None = None) -> torch.Tensor:
    """Grouped GEMM: for sorted row ``r`` in expert ``e``'s slab ``[offsets[e], offsets[e+1])``:
    ``y[r] = x[gather_idx[r] or r] @ W[e]^T`` (``W[e]`` is ``[N, K]``).

    * ``gather_idx [M]`` - fused row gather from ``x`` (so the permuted activations are never materialised),
    * ``out_idx [M]`` - scatter: result row ``r`` goes to ``out[out_idx[r]]``; with ``accumulate`` it is added to the
      existing content (rows of one call must be unique), ``out`` supplies/receives the buffer,
    * ``alpha [E]`` - per-expert scaling of the result (LoRA scaling),
    * ``w`` may be packed W4A16 (``uint8 [E, N, K/2]`` + ``w_scales [E, N, K/G]`` (+ ``w_zeros``) + ``group_size``).
    Rows beyond ``offsets[-1]`` (dropped assignments) are left untouched / zero.
    """
    if w.dtype == torch.uint8:
        wf = dequant_w4_experts(w, w_scales, w_zeros, int(group_size))
    else:
        wf = w.float()
    e, n, _ = wf.shape
    m = gather_idx.numel() if gather_idx is not None else x.shape[0]
    rows_out = out.shape[0] if out is not None else m
    y = out if out is not None else torch.zeros(rows_out, n, dtype=x.dtype, device=x.device)
    for ei in range(e):
        lo, hi = int(offsets[ei]), int(offsets[ei + 1])
        if hi <= lo:
            continue
        src = gather_idx[lo:hi].long() if gather_idx is not None else torch.arange(lo, hi, device=x.device)
        r = x[src].float() @ wf[ei].T
        if alpha is not None:
            r = r * alpha[ei].float()
        dst = out_idx[lo:hi].long() if out_idx is not None else torch.arange(lo, hi, device=x.device)
        if accumulate:
            y[dst] = (y[dst].float() + r).to(y.dtype)
        else:
            y[dst] = r.to(y.dtype)
    return y


def moe_unpermute_combine_ref(y_sorted: torch.Tensor, inv_perm: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """``out[t] = sum_s w[t, s] * y_sorted[inv_perm[t * k + s]]`` (dropped assignments contribute 0). Float32 accumulate."""
    t, k = topk_weights.shape
    inv = inv_perm.long().reshape(t, k)
    ok = inv >= 0
    rows = y_sorted.float()[inv.clamp_min(0)]  # [T, k, H]
    w = torch.where(ok, topk_weights.float(), torch.zeros_like(topk_weights.float()))
    return (rows * w.unsqueeze(-1)).sum(1).to(y_sorted.dtype)


def moe_experts_forward_ref(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, w_gate_up: torch.Tensor,
                            w_down: torch.Tensor, *, activation: str = "silu") -> torch.Tensor:
    """Dense reference of a full MoE FFN: per token sum of ``w * down(act(gate) * up)`` over its selected experts.

    ``w_gate_up [E, 2I, H]`` (gate rows first, then up), ``w_down [E, H, I]``.  Dropped slots (``id < 0``) are skipped.
    """
    t, h = x.shape
    out = torch.zeros(t, h, dtype=torch.float32, device=x.device)
    act = {"silu": torch.nn.functional.silu, "gelu": torch.nn.functional.gelu}[activation]
    inter = w_down.shape[-1]
    for ti in range(t):
        for si in range(topk_ids.shape[1]):
            e = int(topk_ids[ti, si])
            if e < 0:
                continue
            gu = w_gate_up[e].float() @ x[ti].float()
            hdn = act(gu[:inter]) * gu[inter:]
            out[ti] += float(topk_weights[ti, si]) * (w_down[e].float() @ hdn)
    return out.to(x.dtype)
