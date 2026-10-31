"""Oracles + accounting for learned sparse attention (Plan K5, KR-5; SparseAttention paper P6 Ch. 3).

Contents
--------
* :func:`indexer_scores_ref` / :func:`indexer_scores_dense_ref` - DSA-style *lightning indexer*
  ``I[t, s] = sum_j w[t, j] * relu(q_idx[t, j] . k_idx[s])`` over a (possibly FP8) index-key cache.
* :func:`topk_select_ref` - deterministic top-k (ties -> lower position first, output sorted ascending, ``-1`` padded).
* :func:`pool_block_scores_ref`, :func:`expand_block_indices_ref` - block-granular selection.
* :func:`sparse_attention_ref` - gather attention over the selected positions (GQA/paged), thin wrapper over the decode oracle.
* **Selector byte accounting** (:func:`sparse_decode_layer_bytes` ...) reproducing the P6 numbers (index scan = 88 % of
  bytes at 128K; 7.7x -> 22.6x -> 57x byte reduction when one indexer pass is shared by 1 / 4 / 61 layers).
* :class:`SharedSelector` - cross-layer indexer sharing (compute the selection every ``share_every`` layers, reuse it).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from forge.kernels.ref.attn_decode import NEG_INF, paged_decode_attention_ref


# ------------------------------------------------------------------------------------------------------------
# indexer
# ------------------------------------------------------------------------------------------------------------
def _dequant_index_keys(ik: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if ik.dtype == torch.uint8:  # raw e4m3 bytes
        ik = ik.view(torch.float8_e4m3fn)
    x = ik.float()
    if scale is not None:
        x = x * scale.float().unsqueeze(-1)
    return x


def indexer_scores_ref(iq: torch.Tensor, iw: torch.Tensor, ik_pool: torch.Tensor, ik_scale: torch.Tensor | None,
                       block_tables: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Decode-time lightning-indexer scan.

    ``iq [B, HI, dI]``, ``iw [B, HI]`` head weights, ``ik_pool [P, ps, dI]`` (fp8/bf16/...), ``ik_scale [P, ps]`` or
    ``None`` (per-token dequant scale).  Returns ``scores [B, max_pages*ps]`` float32, ``-inf`` at positions
    ``>= seq_len``.
    """
    b = iq.shape[0]
    ps = ik_pool.shape[1]
    tmax = block_tables.shape[1] * ps
    out = torch.full((b, tmax), NEG_INF, dtype=torch.float32, device=iq.device)
    for bi in range(b):
        n = int(seq_lens[bi])
        if n <= 0:
            continue
        n_pages = (n + ps - 1) // ps
        pages = block_tables[bi, :n_pages].long()
        keys = _dequant_index_keys(ik_pool[pages], ik_scale[pages] if ik_scale is not None else None)
        keys = keys.reshape(n_pages * ps, -1)[:n]  # [n, dI]
        logits = torch.relu(torch.einsum("hd,nd->hn", iq[bi].float(), keys))
        out[bi, :n] = (iw[bi].float().unsqueeze(-1) * logits).sum(0)
    return out


def indexer_scores_dense_ref(iq: torch.Tensor, iw: torch.Tensor, ik: torch.Tensor, ik_scale: torch.Tensor | None = None,
                             *, causal: bool = True, q_offset: int | None = None) -> torch.Tensor:
    """Prefill-time indexer: ``iq [Sq, HI, dI]``, ``iw [Sq, HI]``, ``ik [T, dI]`` -> ``[Sq, T]`` (-inf where masked).

    Query ``i`` is at absolute position ``q_offset + i`` (default ``T - Sq``); causal masking hides later keys.
    """
    sq = iq.shape[0]
    t = ik.shape[0]
    keys = _dequant_index_keys(ik, ik_scale)
    logits = torch.relu(torch.einsum("qhd,td->qht", iq.float(), keys))
    s = (iw.float().unsqueeze(-1) * logits).sum(1)
    if causal:
        off = (t - sq) if q_offset is None else q_offset
        qpos = off + torch.arange(sq, device=iq.device)
        s = s.masked_fill(torch.arange(t, device=iq.device).unsqueeze(0) > qpos.unsqueeze(1), NEG_INF)
    return s


def pool_block_scores_ref(scores: torch.Tensor, block_size: int, valid_lens: torch.Tensor | None = None,
                          mode: str = "max") -> torch.Tensor:
    """Reduce token scores ``[B, T]`` to block scores ``[B, ceil(T/block)]`` (``max`` or ``mean``); invalid tail
    tokens (``>= valid_lens``) are ignored; empty blocks get ``-inf``."""
    b, t = scores.shape
    nb = (t + block_size - 1) // block_size
    pad = nb * block_size - t
    s = torch.nn.functional.pad(scores.float(), (0, pad), value=NEG_INF)
    if valid_lens is not None:
        pos = torch.arange(nb * block_size, device=scores.device).unsqueeze(0)
        s = s.masked_fill(pos >= valid_lens.to(scores.device).unsqueeze(1), NEG_INF)
    s = s.reshape(b, nb, block_size)
    if mode == "max":
        return s.amax(-1)
    if mode == "mean":
        ok = torch.isfinite(s)
        cnt = ok.sum(-1)
        tot = torch.where(ok, s, torch.zeros_like(s)).sum(-1)
        return torch.where(cnt > 0, tot / cnt.clamp_min(1), torch.full_like(tot, NEG_INF))
    raise ValueError("mode must be 'max' or 'mean'")


def topk_select_ref(scores: torch.Tensor, k: int, valid_lens: torch.Tensor | None = None) -> torch.Tensor:
    """Deterministic top-k per row.

    Selection rule: among the first ``valid_lens[b]`` positions (all ``T`` if ``None``) pick the ``k`` largest scores,
    ties broken towards the *lower position*; NaN counts as ``-inf``.  The result ``idx [B, k]`` int32 lists the chosen
    positions in **ascending position order**, padded with ``-1`` when fewer than ``k`` positions are valid.
    """
    b, t = scores.shape
    out = torch.full((b, k), -1, dtype=torch.int32, device=scores.device)
    s_all = torch.nan_to_num(scores.float(), nan=NEG_INF)
    for bi in range(b):
        n = t if valid_lens is None else min(int(valid_lens[bi]), t)
        if n <= 0:
            continue
        s = s_all[bi, :n]
        order = torch.sort(-s, stable=True).indices[: min(k, n)]
        sel = torch.sort(order).values
        out[bi, : sel.numel()] = sel.to(torch.int32)
    return out


def expand_block_indices_ref(block_idx: torch.Tensor, block_size: int, seq_lens: torch.Tensor) -> torch.Tensor:
    """Expand selected block ids ``[B, Hsel, Kb]`` (``-1`` unused) to token positions ``[B, Hsel, Kb*block_size]``;
    positions ``>= seq_len`` become ``-1``."""
    b, h, kb = block_idx.shape
    off = torch.arange(block_size, device=block_idx.device)
    tok = block_idx.long().unsqueeze(-1) * block_size + off  # [B, H, Kb, bs]
    bad = (block_idx.long().unsqueeze(-1) < 0) | (tok >= seq_lens.to(block_idx.device).long().view(b, 1, 1, 1))
    tok = torch.where(bad, torch.full_like(tok, -1), tok)
    return tok.reshape(b, h, kb * block_size).to(torch.int32)


def sparse_attention_ref(q, k_pool, v_pool, block_tables, seq_lens, topk_idx, **kw):
    """Gather attention over selected token positions (``topk_idx [B, Hsel, K]``); see
    :func:`forge.kernels.ref.attn_decode.paged_decode_attention_ref` for the remaining arguments."""
    return paged_decode_attention_ref(q, k_pool, v_pool, block_tables, seq_lens, topk_idx=topk_idx, **kw)


# ------------------------------------------------------------------------------------------------------------
# selector byte accounting (P6 Ch. 3, eqs. 3.3, 3.8, 3.9, 3.10)
# ------------------------------------------------------------------------------------------------------------
def index_key_bytes(d_idx: int = 128, bytes_per_elem: float = 1.0, scale_bytes: int = 4) -> int:
    """Bytes of one index key: ``d_idx`` elements (FP8 = 1 byte) + per-token scale (default 128 + 4 = 132, P6 Table 3.2)."""
    return int(round(d_idx * bytes_per_elem)) + scale_bytes


@dataclass(frozen=True)
class SparseByteReport:
    """Per-layer, per-decoded-token bytes read (bytes, not MB)."""

    context_len: int
    top_k: int
    dense_bytes: float
    index_bytes: float
    sparse_bytes: float
    total_bytes: float
    reduction: float
    index_share: float

    def as_dict(self) -> dict:
        return asdict(self)


def dense_attention_bytes(context_len: int, c_tok_elems: int = 576, bytes_per_elem: float = 2.0) -> float:
    """Bytes of a dense decode step over ``context_len`` tokens (MLA latent: ``c_KV + c_R = 576`` BF16 values/token)."""
    return float(c_tok_elems * bytes_per_elem * context_len)


def sparse_decode_layer_bytes(context_len: int, *, top_k: int = 2048, c_tok_elems: int = 576, bytes_per_elem: float = 2.0,
                              idx_key_bytes: float = 132.0, index_share_layers: int = 1,
                              index_kv_heads: int = 1) -> SparseByteReport:
    """Bytes read per layer per decoded token with a token-level indexer (P6 eq. 3.8).

    ``index_bytes = index_kv_heads * idx_key_bytes * T / index_share_layers`` (one indexer pass amortised over
    ``index_share_layers`` layers - IndexCache/HISA-style sharing) and
    ``sparse_bytes = c_tok * b * min(k, T)``.  ``reduction = dense / total``.
    """
    if index_share_layers < 1:
        raise ValueError("index_share_layers must be >= 1")
    dense = dense_attention_bytes(context_len, c_tok_elems, bytes_per_elem)
    idx = float(index_kv_heads) * idx_key_bytes * context_len / index_share_layers
    sp = float(c_tok_elems) * bytes_per_elem * min(top_k, context_len)
    total = idx + sp
    return SparseByteReport(context_len=context_len, top_k=top_k, dense_bytes=dense, index_bytes=idx, sparse_bytes=sp,
                            total_bytes=total, reduction=dense / total, index_share=idx / total)


def indexer_crossover_tokens(c_tok_elems: int = 576, bytes_per_elem: float = 2.0, top_k: int = 2048,
                             idx_key_bytes: float = 132.0, index_kv_heads: int = 1) -> float:
    """Context length beyond which the indexer reads more bytes than the sparse attention it feeds (P6 eq. 3.9)."""
    return c_tok_elems * bytes_per_elem * top_k / (index_kv_heads * idx_key_bytes)


def mla_prefill_flops_per_t2(h: int = 128, c_kv: int = 512, c_r: int = 64) -> float:
    """Dense absorbed-MLA causal prefill FLOPs divided by ``T^2``: ``2 H (c_KV + c_R + c_KV) / 2``."""
    return float(h * (c_kv + c_r + c_kv))


def prefill_indexer_flop_ratio(h_idx: int = 64, d_idx: int = 128, dense_flops_per_t2: float | None = None) -> float:
    """Indexer FLOPs (``H_I d_I T^2``) over dense attention FLOPs (P6 eq. 3.10; DSA example = 5.9 %)."""
    dense = mla_prefill_flops_per_t2() if dense_flops_per_t2 is None else dense_flops_per_t2
    return h_idx * d_idx / dense


# ------------------------------------------------------------------------------------------------------------
# cross-layer sharing
# ------------------------------------------------------------------------------------------------------------
class SharedSelector:
    """Cross-layer indexer sharing (Plan K5 / P6 §12.7): the selection computed at layer ``l`` with
    ``l % share_every == 0`` (an *anchor* layer) is reused by the following ``share_every - 1`` layers.

    Usage inside a model forward (per decode step)::

        sel = SharedSelector(share_every=4)
        for l, layer in enumerate(layers):
            idx = sel.get(l, lambda: topk_select(indexer_scores(...), k))
        ...
        sel.reset()      # at the start of the next step / micro-batch
    """

    def __init__(self, share_every: int = 1, anchor_layers: list[int] | None = None) -> None:
        if share_every < 1:
            raise ValueError("share_every must be >= 1")
        self.share_every = share_every
        self.anchor_layers = set(anchor_layers) if anchor_layers is not None else None
        self._cached: torch.Tensor | None = None
        self.computed_layers: list[int] = []
        self.reused_layers: list[int] = []

    def is_anchor(self, layer: int) -> bool:
        if self.anchor_layers is not None:
            return layer in self.anchor_layers
        return layer % self.share_every == 0

    def get(self, layer: int, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        if self._cached is None or self.is_anchor(layer):
            self._cached = compute()
            self.computed_layers.append(layer)
        else:
            self.reused_layers.append(layer)
        return self._cached

    def reset(self) -> None:
        self._cached = None
        self.computed_layers = []
        self.reused_layers = []

    @property
    def indexer_passes(self) -> int:
        return len(self.computed_layers)
