"""Attention launchers / dispatchers (Plan K4-K6; KR-4..KR-7): paged decode (GQA-grouped, split-KV, quantised KV, residual
window, sliding window), MLA latent decode, prefill / chunked prefill, learned sparse attention (indexer scan, top-k, gather
attention, cross-layer sharing, selector-byte accounting), tree attention for speculative decoding.

Dispatch rule (all ops): Triton only when CUDA + Triton + capability + supported shape (and ``FORGE_FORCE_REF`` unset),
otherwise the pure-torch oracle from ``forge.kernels.ref`` (or ``torch`` SDPA for plain prefill).  **The Triton kernels are
implemented but not yet validated on NVIDIA hardware**; run ``tests_gpu/test_kernels_attn_*.py`` on the target GPU.

Conventions (see ``README_ATTN.md``): pools are head-major ``[num_pages, Hkv, page_size, D]``; ``seq_lens`` count every cached
token including the current one (and the residual window); the decode query is at position ``seq_len - 1``; GQA head ``h`` uses
KV head ``h // (Hq // Hkv)``.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from forge.kernels.ops_util import topk_select  # noqa: F401  (re-export)
from forge.kernels.ref import attn_decode as _dec
from forge.kernels.ref import attn_prefill as _pre
from forge.kernels.ref import attn_sparse as _sp
from forge.kernels.ref import attn_spec as _spec
from forge.kernels.ref.attn_kv import (  # noqa: F401  (public re-exports)
    build_paged_pools,
    normalize_kv_dtype,
    quantize_kv_page,
)
from forge.kernels.ref.attn_sparse import (  # noqa: F401
    SharedSelector,
    SparseByteReport,
    dense_attention_bytes,
    expand_block_indices_ref,
    index_key_bytes,
    indexer_crossover_tokens,
    pool_block_scores_ref,
    prefill_indexer_flop_ratio,
    sparse_decode_layer_bytes,
)
from forge.kernels.ref.attn_spec import (  # noqa: F401
    build_tree_mask,
    chain_parents,
    greedy_verify_ref,
    tree_depths,
    tree_greedy_verify_ref,
    tree_position_ids,
)
from forge.kernels.triton import util_dispatch as _d

KERNELS: dict[str, dict[str, Any]] = {
    "paged_decode_attention": {"family": "attention", "min_sm": 70, "triton": "forge.kernels.triton.attn_decode",
                               "ref": "attn_decode.paged_decode_attention_ref"},
    "mla_decode_attention": {"family": "attention", "min_sm": 80, "triton": "forge.kernels.triton.attn_mla",
                             "ref": "attn_decode.mla_decode_attention_ref"},
    "prefill_attention": {"family": "attention", "min_sm": 70, "triton": "forge.kernels.triton.attn_prefill",
                          "ref": "attn_prefill.prefill_attention_ref"},
    "lightning_indexer": {"family": "sparse", "min_sm": 70, "triton": "forge.kernels.triton.attn_sparse",
                          "ref": "attn_sparse.indexer_scores_ref"},
    "tree_attention": {"family": "spec", "min_sm": 70, "triton": "forge.kernels.triton.attn_spec",
                       "ref": "attn_spec.tree_attention_ref"},
}

_FLOATS = (torch.float16, torch.bfloat16)


def _tri(*tensors, min_sm: int = 70, op: str = "") -> bool:
    ok = _d.use_triton(*tensors, min_sm=min_sm, op=op)
    if ok:
        for t in tensors:
            if t is not None and t.dtype == torch.bfloat16 and not _d.bf16_ok(t.dtype, t.device):
                ok = False
    _d.note_path(op, "triton" if ok else "reference")
    return ok


def _q_ok(q: torch.Tensor) -> bool:
    """fp16/bf16 queries on GPU (fp32 only inside the Triton CPU interpreter, used by the unit tests)."""
    return q.dtype in _FLOATS or (q.dtype == torch.float32 and _d.interpret_mode() and q.device.type == "cpu")


# ------------------------------------------------------------------------------------------------------------
# paged decode
# ------------------------------------------------------------------------------------------------------------
def paged_decode_attention(
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
    window: int | None = None,
    k_zero=None,
    v_zero=None,
    k_res: torch.Tensor | None = None,
    v_res: torch.Tensor | None = None,
    res_lens: torch.Tensor | None = None,
    topk_idx: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    num_splits: int | None = None,
    return_lse: bool = False,
):
    """Paged GQA decode attention (Plan K4).

    Args:
        q: ``[B, Hq, D]`` queries of the current decode step.
        k_pool, v_pool: head-major pools ``[P, Hkv, page_size, D]`` (INT4: ``[.., D/2]`` packed bytes; FP8: ``float8_e4m3fn``
            or its ``uint8`` view; INT8: ``int8``).
        block_tables: ``[B, max_pages]`` int32 page ids; ``seq_lens``: ``[B]`` total cached tokens per sequence.
        softmax_scale: default ``1/sqrt(D)``.
        kv_dtype: ``bf16 | fp16 | fp32 | fp8_e4m3 | int8 | int4``.
        k_scale: per-page/per-channel key scales (``[P, Hkv, D]``, ``[P, Hkv, 1]``, ``[Hkv, D]`` or scalar);
            v_scale: per-token value scales (``[P, Hkv, page_size]``, ``[P, Hkv, 1]`` or scalar); optional additive zeros
            ``k_zero`` / ``v_zero`` of the same shapes.
        window: sliding-window size (attend to the last ``window`` tokens only).
        k_res, v_res, res_lens: full-precision residual window ``[B, Hkv, R, D]`` holding the **last** ``res_lens[b]`` tokens
            (pool holds the first ``seq_len - res_len``) - the KIVI recipe of Plan K4.
        topk_idx: ``[B, 1 | Hkv, K]`` selected token positions (``-1`` unused) -> sparse gather attention (Plan K5).
        max_seq_len: static upper bound of the sequence length (avoids host syncs; default ``max_pages * page_size``).
        num_splits: force the number of split-KV chunks (default: chosen from the SM count).
    Returns ``out [B, Hq, D]`` (+ ``lse [B, Hq]`` if ``return_lse``).
    """
    kv = normalize_kv_dtype(kv_dtype)
    can_tri = (_q_ok(q) and 16 <= q.shape[-1] <= 256 and (kv != "int4" or q.shape[-1] % 2 == 0)
               and not (topk_idx is not None and k_res is not None)
               and (k_res is None or k_res.shape[2] <= 256) and q.shape[1] % k_pool.shape[1] == 0)
    if can_tri and _tri(q, k_pool, v_pool, block_tables, seq_lens, op="paged_decode_attention"):
        from forge.kernels.triton.attn_decode import paged_decode_attention_triton

        return paged_decode_attention_triton(
            q, k_pool, v_pool, block_tables, seq_lens, softmax_scale=softmax_scale, kv_dtype=kv, k_scale=k_scale,
            v_scale=v_scale, k_zero=k_zero, v_zero=v_zero, window=window, k_res=k_res, v_res=v_res, res_lens=res_lens,
            topk_idx=topk_idx, max_seq_len=max_seq_len, num_splits=num_splits, return_lse=return_lse)
    return _dec.paged_decode_attention_ref(
        q, k_pool, v_pool, block_tables, seq_lens, softmax_scale=softmax_scale, kv_dtype=kv, k_scale=k_scale,
        v_scale=v_scale, k_zero=k_zero, v_zero=v_zero, window=window, k_res=k_res, v_res=v_res, res_lens=res_lens,
        topk_idx=topk_idx, return_lse=return_lse)


def mla_decode_attention(q_nope: torch.Tensor, q_rope: torch.Tensor, c_pool: torch.Tensor, r_pool: torch.Tensor,
                         block_tables: torch.Tensor, seq_lens: torch.Tensor, *, softmax_scale: float,
                         w_uv: torch.Tensor | None = None, c_scale: torch.Tensor | None = None, kv_dtype: str = "bf16",
                         window: int | None = None, topk_idx: torch.Tensor | None = None,
                         max_seq_len: int | None = None, num_splits: int | None = None):
    """Absorbed MLA decode (Plan K4): ``q_nope [B, H, Dc]`` (already multiplied by ``W_UK``), ``q_rope [B, H, Dr]``, latent cache
    ``c_pool [P, ps, Dc]`` (bf16/fp16 or FP8 with per-token ``c_scale [P, ps]``) and rotary-key cache ``r_pool [P, ps, Dr]``.
    Returns the latent output ``[B, H, Dc]``, or ``[B, H, Dv]`` if ``w_uv [H, Dc, Dv]`` is given.  ``topk_idx [B, K]`` makes
    it DSA-style sparse decoding."""
    kv = normalize_kv_dtype(kv_dtype)
    can_tri = (_q_ok(q_nope) and q_nope.shape[-1] <= 512 and q_rope.shape[-1] <= 128
               and kv in ("bf16", "fp16", "fp32", "fp8_e4m3"))
    if can_tri and _tri(q_nope, q_rope, c_pool, r_pool, block_tables, seq_lens, min_sm=80, op="mla_decode_attention"):
        from forge.kernels.triton.attn_mla import mla_decode_attention_triton

        return mla_decode_attention_triton(q_nope, q_rope, c_pool, r_pool, block_tables, seq_lens,
                                           softmax_scale=softmax_scale, w_uv=w_uv, c_scale=c_scale, kv_dtype=kv,
                                           window=window, topk_idx=topk_idx, max_seq_len=max_seq_len,
                                           num_splits=num_splits)
    return _dec.mla_decode_attention_ref(q_nope, q_rope, c_pool, r_pool, block_tables, seq_lens,
                                         softmax_scale=softmax_scale, w_uv=w_uv, c_scale=c_scale, kv_dtype=kv,
                                         window=window, topk_idx=topk_idx)


# ------------------------------------------------------------------------------------------------------------
# prefill
# ------------------------------------------------------------------------------------------------------------
def _sdpa_prefill(q, k, v, scale, causal, window):
    """Torch SDPA fallback (bshd tensors) for GPUs without Triton / unsupported shapes. Returns ``None`` if not applicable."""
    if window or (causal and q.shape[1] != k.shape[1]):
        return None
    hq, hkv = q.shape[2], k.shape[2]
    kk, vv = (k, v) if hq == hkv else (k.repeat_interleave(hq // hkv, dim=2), v.repeat_interleave(hq // hkv, dim=2))
    o = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), is_causal=causal, scale=scale)
    return o.transpose(1, 2)


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, softmax_scale: float | None = None,
                      causal: bool = True, window: int | None = None, cu_seqlens_q: torch.Tensor | None = None,
                      cu_seqlens_k: torch.Tensor | None = None, max_seqlen_q: int | None = None,
                      layout: str = "bshd") -> torch.Tensor:
    """Flash-style causal GQA attention for prefill / chunked prefill (bottom-right aligned causal mask: if ``Sk > Sq`` the
    queries are the **last** ``Sq`` positions, so cached-prefix keys work out of the box).

    * ``layout="bshd"`` (default): ``q [B, Sq, Hq, D]``, ``k, v [B, Sk, Hkv, D]``; ``layout="bhsd"`` for ``[B, H, S, D]``.
    * varlen: ``q [Tq, Hq, D]``, ``k, v [Tk, Hkv, D]`` + ``cu_seqlens_q/k [B+1]`` (``max_seqlen_q`` avoids a host sync).
    """
    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    can_tri = _q_ok(q) and 16 <= d <= 256 and q.shape[-2] % k.shape[-2] == 0
    if cu_seqlens_q is not None:
        cu_k = cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q
        if can_tri and _tri(q, k, v, op="prefill_attention"):
            from forge.kernels.triton.attn_prefill import prefill_attention_varlen_triton

            mx = max_seqlen_q if max_seqlen_q is not None else int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
            return prefill_attention_varlen_triton(q, k, v, cu_seqlens_q, cu_k, max_seqlen_q=mx, softmax_scale=scale,
                                                   causal=causal, window=window)
        return _pre.prefill_attention_ref(q, k, v, softmax_scale=scale, causal=causal, window=window,
                                          cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_k)
    if layout not in ("bshd", "bhsd"):
        raise ValueError("layout must be 'bshd' or 'bhsd'")
    qb, kb, vb = (q, k, v) if layout == "bshd" else (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    b, sq, hq, _ = qb.shape
    sk = kb.shape[1]
    out = None
    if can_tri and _tri(q, k, v, op="prefill_attention"):
        from forge.kernels.triton.attn_prefill import prefill_attention_varlen_triton

        dev = q.device
        cuq = torch.arange(b + 1, dtype=torch.int32, device=dev) * sq
        cuk = torch.arange(b + 1, dtype=torch.int32, device=dev) * sk
        o = prefill_attention_varlen_triton(qb.contiguous().view(b * sq, hq, d), kb.contiguous().view(b * sk, -1, d),
                                            vb.contiguous().view(b * sk, -1, d), cuq, cuk, max_seqlen_q=sq,
                                            softmax_scale=scale, causal=causal, window=window)
        out = o.view(b, sq, hq, d)
    elif q.device.type == "cuda" and not _d.force_ref():
        out = _sdpa_prefill(qb, kb, vb, scale, causal, window)
    if out is None:
        out = _pre.prefill_attention_ref(qb, kb, vb, softmax_scale=scale, causal=causal, window=window)
    return out if layout == "bshd" else out.transpose(1, 2)


def paged_kv_write(k: torch.Tensor, v: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                   slot_mapping: torch.Tensor) -> None:
    """Scatter per-token ``k, v [T, Hkv, D]`` into native-dtype pools at ``slot_mapping`` (``page * page_size + offset``,
    negative = skip).  A plain vectorised index-put (already bandwidth-bound); use ``ops_util.rope_and_kv_write`` for the fused
    RoPE + quantise + write."""
    _pre.paged_kv_write_ref(k, v, k_pool, v_pool, slot_mapping)


def chunked_prefill_attention(q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, k_pool: torch.Tensor,
                              v_pool: torch.Tensor, block_tables: torch.Tensor, ctx_lens: torch.Tensor,
                              cu_seqlens_q: torch.Tensor, *, softmax_scale: float | None = None, kv_dtype: str = "bf16",
                              k_scale=None, v_scale=None, k_zero=None, v_zero=None, window: int | None = None,
                              max_seqlen_q: int | None = None) -> torch.Tensor:
    """Chunked prefill / prefix-cache attention: ``q, k_new, v_new [Tq, H*, D]`` are the new chunk tokens of every sequence
    (``cu_seqlens_q``), whose first ``ctx_lens[b]`` tokens of context already sit in the (optionally quantised) paged pools.
    The prefix is gathered + dequantised to ``q.dtype`` and the packed varlen flash kernel runs over ``[prefix | chunk]``."""
    b = block_tables.shape[0]
    ks, vs, cu_k = [], [], [0]
    for bi in range(b):
        n = int(ctx_lens[bi])
        qs, qe = int(cu_seqlens_q[bi]), int(cu_seqlens_q[bi + 1])
        if n > 0:
            kf, vf = _dec.gather_paged_kv_ref(k_pool, v_pool, block_tables[bi], n, kv_dtype=kv_dtype, k_scale=k_scale,
                                              v_scale=v_scale, k_zero=k_zero, v_zero=v_zero)
            ks.append(kf.permute(1, 0, 2).to(q.dtype))
            vs.append(vf.permute(1, 0, 2).to(q.dtype))
        ks.append(k_new[qs:qe])
        vs.append(v_new[qs:qe])
        cu_k.append(cu_k[-1] + n + (qe - qs))
    k_all, v_all = torch.cat(ks, 0), torch.cat(vs, 0)
    cuk = torch.tensor(cu_k, dtype=torch.int32, device=q.device)
    return prefill_attention(q, k_all, v_all, softmax_scale=softmax_scale, causal=True, window=window,
                             cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cuk, max_seqlen_q=max_seqlen_q)


# ------------------------------------------------------------------------------------------------------------
# sparse attention (K5)
# ------------------------------------------------------------------------------------------------------------
def lightning_indexer_scores(iq: torch.Tensor, iw: torch.Tensor, ik_pool: torch.Tensor, ik_scale: torch.Tensor | None,
                             block_tables: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Low-precision lightning-indexer scan over a paged index-key cache (FP8 / bf16 / fp16 keys with optional per-token
    scale): ``score = sum_j w_j relu(iq_j . ik) * scale``. Returns ``[B, max_pages * page_size]`` float32 (``-inf`` beyond lens)."""
    fp8 = ik_pool.dtype in (torch.uint8, getattr(torch, "float8_e4m3fn", None))
    can_tri = (_q_ok(iq) and iq.shape[-1] <= 256 and (fp8 or ik_pool.dtype in _FLOATS))
    if can_tri and _tri(iq, iw, ik_pool, block_tables, seq_lens, op="lightning_indexer"):
        from forge.kernels.triton.attn_sparse import indexer_scores_triton

        return indexer_scores_triton(iq, iw, ik_pool, ik_scale, block_tables, seq_lens)
    return _sp.indexer_scores_ref(iq, iw, ik_pool, ik_scale, block_tables, seq_lens)


def select_sparse_indices(iq, iw, ik_pool, ik_scale, block_tables, seq_lens, top_k: int, *, block_size: int = 1,
                          pool_mode: str = "max", method: str = "auto") -> torch.Tensor:
    """Indexer scan + top-k. ``block_size == 1`` -> token-granular positions ``[B, 1, k]``; ``block_size > 1`` ->
    block-granular selection (``top_k`` is the **token budget**, ``top_k // block_size`` blocks chosen by ``pool_mode`` of the
    token scores) expanded to token positions ``[B, 1, (top_k // block_size) * block_size]``.  The selection is shared by all
    heads of the GQA group (arithmetic-intensity rule of Plan K5)."""
    scores = lightning_indexer_scores(iq, iw, ik_pool, ik_scale, block_tables, seq_lens)
    if block_size == 1:
        return topk_select(scores, top_k, seq_lens, method=method).unsqueeze(1)
    nblk = max(1, top_k // block_size)
    bscores = _sp.pool_block_scores_ref(scores, block_size, seq_lens, pool_mode)
    nvalid = (seq_lens.to(scores.device).long() + block_size - 1) // block_size
    blk = topk_select(bscores, nblk, nvalid, method=method).unsqueeze(1)
    return expand_block_indices_ref(blk, block_size, seq_lens)


def sparse_topk_attention(q, k_pool, v_pool, block_tables, seq_lens, topk_idx, **kw):
    """Token-granular gather attention over ``topk_idx [B, 1 | Hkv, K]`` (same keyword arguments as
    :func:`paged_decode_attention`)."""
    return paged_decode_attention(q, k_pool, v_pool, block_tables, seq_lens, topk_idx=topk_idx, **kw)


def sparse_decode_step(q, k_pool, v_pool, block_tables, seq_lens, iq, iw, ik_pool, ik_scale, *, top_k: int = 2048,
                       block_size: int = 1, selector: SharedSelector | None = None, layer: int = 0,
                       index_block_tables: torch.Tensor | None = None, **kw):
    """One sparse decode step = indexer scan + top-k + gather attention, with optional cross-layer indexer sharing
    (``selector.get(layer, ...)`` recomputes the selection only on anchor layers).  The index-key pool normally shares the
    page table of the KV cache; pass ``index_block_tables`` if it has its own."""
    ibt = index_block_tables if index_block_tables is not None else block_tables

    def compute():
        return select_sparse_indices(iq, iw, ik_pool, ik_scale, ibt, seq_lens, top_k, block_size=block_size)

    idx = selector.get(layer, compute) if selector is not None else compute()
    return sparse_topk_attention(q, k_pool, v_pool, block_tables, seq_lens, idx, **kw)


def selector_bytes(context_len: int, **kw) -> SparseByteReport:
    """Selector-bytes accounting (P6 Ch. 3): bytes read per layer per decoded token for dense vs indexer-sparse attention,
    incl. the index-scan share and cross-layer sharing (``index_share_layers``). See ``ref.attn_sparse``."""
    return sparse_decode_layer_bytes(context_len, **kw)


# ------------------------------------------------------------------------------------------------------------
# speculative decoding: tree attention
# ------------------------------------------------------------------------------------------------------------
def tree_attention(q: torch.Tensor, k_tree: torch.Tensor, v_tree: torch.Tensor, k_pool: torch.Tensor,
                   v_pool: torch.Tensor, block_tables: torch.Tensor, ctx_lens: torch.Tensor, tree_mask: torch.Tensor, *,
                   softmax_scale: float | None = None, kv_dtype: str = "bf16", k_scale=None, v_scale=None, k_zero=None,
                   v_zero=None, window: int | None = None, node_pos: torch.Tensor | None = None,
                   k_res: torch.Tensor | None = None, v_res: torch.Tensor | None = None,
                   res_lens: torch.Tensor | None = None, max_seq_len: int | None = None,
                   num_splits: int | None = None) -> torch.Tensor:
    """Draft-tree verification attention with a custom ancestor mask and per-node positions (Plan K6).

    ``q [B, N, Hq, D]``; ``k_tree, v_tree [B, N, Hkv, D]``; committed context in the paged pools (``ctx_lens`` tokens, last
    ``res_lens`` of them in the residual window); ``tree_mask [N, N]`` or ``[B, N, N]`` bool (``[i, j]`` = node ``j`` visible to
    ``i``; build with :func:`build_tree_mask`).  ``node_pos [B, N]`` (default ``ctx_len + depth``) drives the sliding window.
    Returns ``[B, N, Hq, D]``."""
    kv = normalize_kv_dtype(kv_dtype)
    can_tri = _q_ok(q) and 16 <= q.shape[-1] <= 256 and q.shape[1] <= 128 and (k_res is None or k_res.shape[2] <= 256)
    if can_tri and _tri(q, k_tree, v_tree, k_pool, v_pool, block_tables, ctx_lens, op="tree_attention"):
        from forge.kernels.triton.attn_spec import tree_attention_triton

        return tree_attention_triton(q, k_tree, v_tree, k_pool, v_pool, block_tables, ctx_lens, tree_mask,
                                     softmax_scale=softmax_scale, kv_dtype=kv, k_scale=k_scale, v_scale=v_scale,
                                     k_zero=k_zero, v_zero=v_zero, window=window, node_pos=node_pos, k_res=k_res,
                                     v_res=v_res, res_lens=res_lens, max_seq_len=max_seq_len, num_splits=num_splits)
    return _spec.tree_attention_ref(q, k_tree, v_tree, k_pool, v_pool, block_tables, ctx_lens, tree_mask,
                                    softmax_scale=softmax_scale, kv_dtype=kv, k_scale=k_scale, v_scale=v_scale,
                                    k_zero=k_zero, v_zero=v_zero, window=window, node_pos=node_pos, k_res=k_res,
                                    v_res=v_res, res_lens=res_lens)


# ------------------------------------------------------------------------------------------------------------
# benchmark cases
# ------------------------------------------------------------------------------------------------------------
def bench_cases() -> list[dict]:
    """Benchmark descriptors for ``forge.kernels.bench`` (``run``/``ref`` take the dict from ``make_inputs``)."""
    dt = lambda cfg: getattr(torch, cfg.get("dtype", "bfloat16"))  # noqa: E731
    dev = lambda cfg: torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))  # noqa: E731

    def decode_inputs(cfg):
        g = torch.Generator().manual_seed(0)
        b, hq, hkv, d, ctx, ps = cfg["batch"], cfg["hq"], cfg["hkv"], cfg["d"], cfg["ctx"], cfg.get("page", 16)
        kvd = cfg.get("kv_dtype", "bf16")
        k = torch.randn(b, hkv, ctx, d, generator=g)
        v = torch.randn(b, hkv, ctx, d, generator=g)
        p = build_paged_pools(k.to(dt(cfg)) if kvd == "bf16" else k, v.to(dt(cfg)) if kvd == "bf16" else v,
                              [ctx] * b, ps, kvd)
        to = lambda t: t.to(dev(cfg)) if t is not None else None  # noqa: E731
        return {"q": torch.randn(b, hq, d, generator=g).to(dev(cfg), dt(cfg)), "kv_dtype": kvd,
                **{n: to(p[n]) for n in ("k_pool", "v_pool", "block_tables", "seq_lens", "k_scale", "v_scale")}}

    def decode_run(i):
        return paged_decode_attention(i["q"], i["k_pool"], i["v_pool"], i["block_tables"], i["seq_lens"],
                                      kv_dtype=i["kv_dtype"], k_scale=i["k_scale"], v_scale=i["v_scale"])

    def decode_ref(i):
        return _dec.paged_decode_attention_ref(i["q"].cpu(), i["k_pool"].cpu(), i["v_pool"].cpu(), i["block_tables"].cpu(),
                                               i["seq_lens"].cpu(), kv_dtype=i["kv_dtype"],
                                               k_scale=None if i["k_scale"] is None else i["k_scale"].cpu(),
                                               v_scale=None if i["v_scale"] is None else i["v_scale"].cpu())

    kvb = {"bf16": 2.0, "fp16": 2.0, "fp8_e4m3": 1.0, "int8": 1.0, "int4": 0.5}

    def prefill_inputs(cfg):
        d = dev(cfg)
        return {"q": torch.randn(cfg["batch"], cfg["seq"], cfg["hq"], cfg["d"], device=d, dtype=dt(cfg)),
                "k": torch.randn(cfg["batch"], cfg["seq"], cfg["hkv"], cfg["d"], device=d, dtype=dt(cfg)),
                "v": torch.randn(cfg["batch"], cfg["seq"], cfg["hkv"], cfg["d"], device=d, dtype=dt(cfg))}

    def idx_inputs(cfg):
        d = dev(cfg)
        b, ctx, ps = cfg["batch"], cfg["ctx"], cfg.get("page", 64)
        n_pages = (ctx + ps - 1) // ps
        bt = torch.arange(b * n_pages, dtype=torch.int32, device=d).view(b, n_pages)
        ik = torch.randn(b * n_pages, ps, cfg.get("di", 128), device=d)
        return {"iq": torch.randn(b, cfg.get("hi", 64), cfg.get("di", 128), device=d, dtype=dt(cfg)),
                "iw": torch.rand(b, cfg.get("hi", 64), device=d), "ik": ik.to(dt(cfg)), "bt": bt,
                "sl": torch.full((b,), ctx, dtype=torch.int32, device=d)}

    return [
        {"name": "paged_decode_attention", "family": "attention", "make_inputs": decode_inputs, "run": decode_run,
         "ref": decode_ref,
         "shapes": [{"batch": b, "hq": 32, "hkv": 8, "d": 128, "ctx": c, "kv_dtype": kvd}
                    for b in (1, 16) for c in (4096, 32768) for kvd in ("bf16", "fp8_e4m3", "int4")],
         "bytes_moved": lambda cfg: int(2 * cfg["batch"] * cfg["hkv"] * cfg["ctx"] * cfg["d"] * kvb[cfg.get("kv_dtype", "bf16")]),
         "flops": lambda cfg: 4 * cfg["batch"] * cfg["hq"] * cfg["ctx"] * cfg["d"]},
        {"name": "prefill_attention", "family": "attention", "make_inputs": prefill_inputs,
         "run": lambda i: prefill_attention(i["q"], i["k"], i["v"]),
         "ref": lambda i: _pre.prefill_attention_ref(i["q"], i["k"], i["v"]),
         "shapes": [{"batch": 1, "seq": s, "hq": 32, "hkv": 8, "d": 128} for s in (512, 2048, 8192)],
         "bytes_moved": lambda cfg: 2 * 2 * cfg["batch"] * cfg["seq"] * (cfg["hq"] + 2 * cfg["hkv"]) * cfg["d"],
         "flops": lambda cfg: 2 * cfg["batch"] * cfg["hq"] * cfg["seq"] ** 2 * cfg["d"]},
        {"name": "lightning_indexer", "family": "sparse", "make_inputs": idx_inputs,
         "run": lambda i: lightning_indexer_scores(i["iq"], i["iw"], i["ik"], None, i["bt"], i["sl"]),
         "ref": lambda i: _sp.indexer_scores_ref(i["iq"], i["iw"], i["ik"], None, i["bt"], i["sl"]),
         "shapes": [{"batch": b, "ctx": c, "hi": 64, "di": 128} for b in (1, 8) for c in (8192, 131072)],
         "bytes_moved": lambda cfg: cfg["batch"] * cfg["ctx"] * cfg.get("di", 128) * 2,
         "flops": lambda cfg: 2 * cfg["batch"] * cfg["ctx"] * cfg.get("hi", 64) * cfg.get("di", 128)},
    ]
