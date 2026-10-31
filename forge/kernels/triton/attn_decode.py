"""Triton paged decode attention (Plan K4 / KR-4) - GQA-grouped, split-KV, quantised KV, sliding window, ragged lengths.

**Status: implemented, not yet validated on NVIDIA hardware.**  The kernels run (slowly) in Triton's CPU interpreter in
``tests/test_kernels_attn_triton_interp.py``; GPU correctness/performance tests live in ``tests_gpu/``.
Minimum versions: Triton 2.1 (developed on 3.x), PyTorch 2.1.  Hardware: sm_70+ (fp16), sm_80+ (bf16); FP8 KV decode is
portable (integer decode) and uses the native converter on sm_89+.

Design
------
* Grid ``(B * Hkv, node_blocks, num_splits)``.  One program serves **all** ``g = Hq / Hkv`` query heads of a KV group (rows of
  the ``tl.dot`` M tile, padded to ``>= 16``) so every KV byte read feeds ``g`` heads - the arithmetic-intensity rule of K4.
  For tree verification several draft-tree nodes are packed into the same M tile (``rows = node * BLOCK_G + head``).
* Tokens are addressed per token (``pos -> page_table[pos // page_size], pos % page_size``) so ``BLOCK_N`` is independent of
  the page size and sequence lengths need not be page multiples.  Pages are head-major ``[P, Hkv, page_size, D]``.
* Split-KV: each split handles a contiguous token range and writes a normalised partial output + natural-log LSE; a second
  kernel merges splits with a numerically stable LSE combine (empty splits carry ``lse = -inf``).
* The full-precision *residual window* and the *draft tree* are attended by a small second kernel into extra partial slots
  and merged by the same LSE merge (so quantised pools never need the newest tokens).
* ``SPARSE`` mode takes a list of selected token positions (``topk_idx``) instead of a dense range - the token-granular
  gather attention of K5.
"""
from __future__ import annotations

import math
import os

import torch

from forge.kernels.ref.attn_kv import (
    expand_k_scale,
    expand_v_scale,
    expand_zero_like,
    kv_kind_id,
    logical_head_dim,
    normalize_kv_dtype,
)
from forge.kernels.triton.attn_common import HAVE_TRITON, merge_partials_triton
from forge.kernels.triton.util_dispatch import cdiv, fp8_native, next_pow2, num_sms

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from forge.kernels.triton.attn_common import (
        load_k_tile,
        load_v_tile,
        online_softmax_update,
    )

    @triton.jit
    def _decode_stage1_kernel(
        q_ptr, k_ptr, v_ptr, ks_ptr, kz_ptr, vs_ptr, vz_ptr, bt_ptr, seqlen_ptr, reslen_ptr, npos_ptr, idx_ptr,
        o_part_ptr, lse_part_ptr,
        sm_scale, NQ,
        sq_b, sq_n, sq_h,
        sk_pg, sk_h, sk_t, sk_d,
        sv_pg, sv_h, sv_t, sv_d,
        sks_pg, sks_h, sks_d,
        svs_pg, svs_h, svs_t,
        sbt_b, snp_b, si_b, si_h,
        so_b, so_n, so_h, so_s,
        sl_b, sl_n, sl_h,
        HKV: tl.constexpr, G: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr, PAGE_SIZE: tl.constexpr,
        NS: tl.constexpr, WINDOW: tl.constexpr,
        KV_KIND: tl.constexpr, HAS_ZERO: tl.constexpr, FP8_NATIVE: tl.constexpr,
        HAS_RES: tl.constexpr, HAS_NPOS: tl.constexpr,
        SPARSE: tl.constexpr, IDX_PER_HEAD: tl.constexpr, K_SEL: tl.constexpr, LIST_CHUNK: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_nb = tl.program_id(1)
        pid_s = tl.program_id(2)
        b = pid_bh // HKV
        hkv = pid_bh % HKV
        NPB: tl.constexpr = BLOCK_M // BLOCK_G
        rows = tl.arange(0, BLOCK_M)
        node = pid_nb * NPB + rows // BLOCK_G
        gi = rows % BLOCK_G
        row_valid = (gi < G) & (node < NQ)
        hq = hkv * G + gi
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q = tl.load(q_ptr + b * sq_b + node[:, None] * sq_n + hq[:, None] * sq_h + offs_d[None, :],
                    mask=row_valid[:, None] & d_mask[None, :], other=0.0)

        seq_len = tl.load(seqlen_ptr + b)
        if HAS_RES:
            res_len = tl.load(reslen_ptr + b)
        else:
            res_len = seq_len * 0
        pool_len = seq_len - res_len
        if HAS_NPOS:
            npos = tl.load(npos_ptr + b * snp_b + node, mask=row_valid, other=0)
        else:
            npos = seq_len - 1 + rows * 0

        lo = seq_len * 0
        if WINDOW > 0:
            min_np = tl.min(tl.where(row_valid, npos, 2147483647), axis=0)
            lo = tl.maximum(min_np - WINDOW + 1, 0)

        if SPARSE:
            loop_lo = pid_s * LIST_CHUNK
            loop_hi = tl.minimum(loop_lo + LIST_CHUNK, K_SEL)
        else:
            span = tl.maximum(pool_len - lo, 0)
            chunk = (((span + NS - 1) // NS + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
            loop_lo = lo + pid_s * chunk
            loop_hi = tl.minimum(loop_lo + chunk, pool_len)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        qk_scale = sm_scale * 1.4426950408889634

        for start in range(loop_lo, loop_hi, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            if SPARSE:
                jv = offs < loop_hi
                pos = tl.load(idx_ptr + b * si_b + hkv * IDX_PER_HEAD * si_h + offs, mask=jv, other=-1)
                tok_valid = jv & (pos >= 0) & (pos < pool_len)
                pos = tl.where(tok_valid, pos, 0)
            else:
                pos = offs
                tok_valid = offs < loop_hi
            page = tl.load(bt_ptr + b * sbt_b + pos // PAGE_SIZE, mask=tok_valid, other=0)
            slot = pos % PAGE_SIZE
            k = load_k_tile(k_ptr, ks_ptr, kz_ptr, page, slot, tok_valid, hkv, offs_d, d_mask,
                            sk_pg, sk_h, sk_t, sk_d, sks_pg, sks_h, sks_d, KV_KIND, HAS_ZERO, FP8_NATIVE)
            qk = tl.dot(q, tl.trans(k.to(q.dtype)))
            qk = qk * qk_scale
            valid = tok_valid[None, :] & row_valid[:, None]
            if WINDOW > 0:
                valid = valid & (pos[None, :] > (npos[:, None] - WINDOW))
            v = load_v_tile(v_ptr, vs_ptr, vz_ptr, page, slot, tok_valid, hkv, offs_d, d_mask,
                            sv_pg, sv_h, sv_t, sv_d, svs_pg, svs_h, svs_t, KV_KIND, HAS_ZERO, FP8_NATIVE)
            acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, v.to(q.dtype), valid)

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        o = acc / l_safe[:, None]
        lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.log2(l_safe)) * 0.6931471805599453)
        o_ptrs = o_part_ptr + b * so_b + node[:, None] * so_n + hq[:, None] * so_h + pid_s * so_s + offs_d[None, :]
        tl.store(o_ptrs, o, mask=row_valid[:, None] & d_mask[None, :])
        tl.store(lse_part_ptr + b * sl_b + node * sl_n + hq * sl_h + pid_s, lse, mask=row_valid)

    @triton.jit
    def _small_buffer_kernel(
        q_ptr, kb_ptr, vb_ptr, len_ptr, seqlen_ptr, reslen_ptr, npos_ptr, mask_ptr, o_part_ptr, lse_part_ptr,
        sm_scale, NQ, slot,
        sq_b, sq_n, sq_h,
        skb_b, skb_h, skb_l, svb_b, svb_h, svb_l,
        snp_b, smk_b, smk_n,
        so_b, so_n, so_h, so_s,
        sl_b, sl_n, sl_h,
        HKV: tl.constexpr, G: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
        MODE: tl.constexpr, NT: tl.constexpr, WINDOW: tl.constexpr, HAS_NPOS: tl.constexpr,
    ):
        """Attention of the query rows over a small dense buffer (``MODE`` 0: residual window with ``len_ptr`` valid
        tokens; ``MODE`` 1: draft-tree nodes, ``NT`` of them, visibility from ``mask_ptr[b, node, j]``)."""
        pid_bh = tl.program_id(0)
        pid_nb = tl.program_id(1)
        b = pid_bh // HKV
        hkv = pid_bh % HKV
        NPB: tl.constexpr = BLOCK_M // BLOCK_G
        rows = tl.arange(0, BLOCK_M)
        node = pid_nb * NPB + rows // BLOCK_G
        gi = rows % BLOCK_G
        row_valid = (gi < G) & (node < NQ)
        hq = hkv * G + gi
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q = tl.load(q_ptr + b * sq_b + node[:, None] * sq_n + hq[:, None] * sq_h + offs_d[None, :],
                    mask=row_valid[:, None] & d_mask[None, :], other=0.0)
        seq_len = tl.load(seqlen_ptr + b)
        if MODE == 0:
            buf_len = tl.load(len_ptr + b)
        else:
            buf_len = seq_len * 0 + NT
        if HAS_NPOS:
            npos = tl.load(npos_ptr + b * snp_b + node, mask=row_valid, other=0)
        else:
            npos = seq_len - 1 + rows * 0
        offs_l = tl.arange(0, BLOCK_L)
        lv = offs_l < buf_len
        k = tl.load(kb_ptr + b * skb_b + hkv * skb_h + offs_l[:, None] * skb_l + offs_d[None, :],
                    mask=lv[:, None] & d_mask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * (sm_scale * 1.4426950408889634)
        valid = lv[None, :] & row_valid[:, None]
        if MODE == 0:
            if WINDOW > 0:
                res_len = tl.load(reslen_ptr + b)
                kpos = (seq_len - res_len) + offs_l
                valid = valid & (kpos[None, :] > (npos[:, None] - WINDOW))
        else:
            tm = tl.load(mask_ptr + b * smk_b + node[:, None] * smk_n + offs_l[None, :],
                         mask=row_valid[:, None] & lv[None, :], other=0)
            valid = valid & (tm != 0)
        v = tl.load(vb_ptr + b * svb_b + hkv * svb_h + offs_l[:, None] * svb_l + offs_d[None, :],
                    mask=lv[:, None] & d_mask[None, :], other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, v.to(q.dtype), valid)
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        o = acc / l_safe[:, None]
        lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.log2(l_safe)) * 0.6931471805599453)
        o_ptrs = o_part_ptr + b * so_b + node[:, None] * so_n + hq[:, None] * so_h + slot * so_s + offs_d[None, :]
        tl.store(o_ptrs, o, mask=row_valid[:, None] & d_mask[None, :])
        tl.store(lse_part_ptr + b * sl_b + node * sl_n + hq * sl_h + slot, lse, mask=row_valid)


# ----------------------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------------------
MAX_SMALL_BUFFER = 256


def _pick_splits(base_programs: int, max_tiles: int, device, override: int | None) -> int:
    if override is not None:
        return max(1, int(override))
    env = os.environ.get("FORGE_DECODE_SPLITS")
    if env:
        return max(1, int(env))
    target = 2 * num_sms(device.index or 0) if device.type == "cuda" else 4
    want = max(1, cdiv(target, max(1, base_programs)))
    ns = max(1, min(64, want, max(1, max_tiles)))
    return 1 << (ns.bit_length() - 1)  # round down to a power of two (limits kernel specialisations)


def _as_pool_view(pool: torch.Tensor, kv: str) -> torch.Tensor:
    if kv == "fp8_e4m3" and pool.dtype != torch.uint8:
        return pool.view(torch.uint8)
    return pool


def run_paged_attention(
    q4: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_dtype: str = "bf16",
    k_scale=None, v_scale=None, k_zero=None, v_zero=None,
    window: int | None = None,
    k_res: torch.Tensor | None = None, v_res: torch.Tensor | None = None, res_lens: torch.Tensor | None = None,
    topk_idx: torch.Tensor | None = None,
    k_tree: torch.Tensor | None = None, v_tree: torch.Tensor | None = None, tree_mask: torch.Tensor | None = None,
    node_pos: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    num_splits: int | None = None,
    return_lse: bool = False,
):
    """General launcher.  ``q4 [B, NQ, Hq, D]`` (decode: ``NQ = 1``).  Returns ``out [B, NQ, Hq, D]`` in ``q4.dtype``.

    ``k_tree, v_tree [B, N, Hkv, D]`` + ``tree_mask [B or 1, N, N]`` (bool/uint8) add draft-tree self attention
    (``NQ`` must equal ``N``).  ``node_pos [B, NQ]`` int32 are the absolute query positions (default ``seq_len - 1``).
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    kv = normalize_kv_dtype(kv_dtype)
    kind = kv_kind_id(kv)
    q4 = q4 if q4.stride(-1) == 1 else q4.contiguous()
    b, nq, hq, d = q4.shape
    p_, hkv, ps, _ = k_pool.shape
    if logical_head_dim(k_pool, kv) != d:
        raise ValueError("pool head dim does not match q")
    if hq % hkv:
        raise ValueError("Hq must be a multiple of Hkv")
    g = hq // hkv
    dev = q4.device
    k_view, v_view = _as_pool_view(k_pool, kv), _as_pool_view(v_pool, kv)
    seq_lens = seq_lens.to(torch.int32).contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()

    if kind != 0:
        ks = expand_k_scale(k_scale, p_, hkv, d, dev)
        vs = expand_v_scale(v_scale, p_, hkv, ps, dev)
        kz = expand_zero_like(k_zero, ks)
        vz = expand_zero_like(v_zero, vs)
        has_zero = kz is not None
        ks_s = (ks.stride(0), ks.stride(1), ks.stride(2))
        vs_s = (vs.stride(0), vs.stride(1), vs.stride(2))
        kz = kz if kz is not None else ks
        vz = vz if vz is not None else vs
    else:
        ks = vs = kz = vz = q4
        has_zero = False
        ks_s = vs_s = (0, 0, 0)

    has_res = k_res is not None
    has_tree = k_tree is not None
    sparse = topk_idx is not None
    if has_res and sparse:
        raise ValueError("sparse (top-k) selection with a residual window is not supported")
    if has_res:
        if res_lens is None or v_res is None:
            raise ValueError("k_res requires v_res and res_lens")
        if k_res.shape[2] > MAX_SMALL_BUFFER:
            raise ValueError(f"residual window larger than {MAX_SMALL_BUFFER} tokens is not supported")
        k_res = k_res if k_res.stride(-1) == 1 else k_res.contiguous()
        v_res = v_res if v_res.stride(-1) == 1 else v_res.contiguous()
        res_lens = res_lens.to(torch.int32).contiguous()
    if has_tree:
        if v_tree is None or tree_mask is None or k_tree.shape[1] != nq:
            raise ValueError("tree attention needs v_tree, tree_mask and NQ == number of tree nodes")
        if nq > MAX_SMALL_BUFFER:
            raise ValueError("draft tree too large")
        k_tree = k_tree if k_tree.stride(-1) == 1 else k_tree.contiguous()
        v_tree = v_tree if v_tree.stride(-1) == 1 else v_tree.contiguous()
        tm = tree_mask.to(torch.uint8).contiguous()
        if tm.dim() == 2:
            tm = tm.unsqueeze(0)
    has_npos = node_pos is not None
    npos = node_pos.to(torch.int32).contiguous() if has_npos else seq_lens
    if sparse:
        idx = topk_idx.to(torch.int32).contiguous()
        idx_per_head = idx.shape[1] != 1
        if idx_per_head and idx.shape[1] != hkv:
            raise ValueError("topk_idx must have 1 or Hkv selection heads")
        k_sel = idx.shape[2]
    else:
        idx, idx_per_head, k_sel = seq_lens, False, 0

    block_g = next_pow2(g)
    nodes_wanted = next_pow2(nq)
    block_m = max(16, min(64, block_g * nodes_wanted))
    block_m = max(block_m, block_g)
    npb = block_m // block_g
    n_node_blocks = cdiv(nq, npb)
    block_d = max(16, next_pow2(d))
    block_n = 64 if d <= 128 else 32
    num_warps = 4 if block_d <= 128 else 8
    win = int(window) if window else 0

    max_len = int(max_seq_len) if max_seq_len is not None else block_tables.shape[1] * ps
    max_tiles = cdiv(k_sel, block_n) if sparse else cdiv(max_len, block_n)
    ns = _pick_splits(b * hkv * n_node_blocks, max_tiles, dev, num_splits)
    n_extra = int(has_res) + int(has_tree)
    nsp = ns + n_extra
    list_chunk = cdiv(cdiv(k_sel, ns), block_n) * block_n if sparse else 0

    o_part = torch.empty(b, nq, hq, nsp, d, dtype=torch.float32, device=dev)
    lse_part = torch.empty(b, nq, hq, nsp, dtype=torch.float32, device=dev)
    common_out = (o_part.stride(0), o_part.stride(1), o_part.stride(2), o_part.stride(3),
                  lse_part.stride(0), lse_part.stride(1), lse_part.stride(2))

    _decode_stage1_kernel[(b * hkv, n_node_blocks, ns)](
        q4, k_view, v_view, ks, kz, vs, vz, block_tables, seq_lens, res_lens if has_res else seq_lens, npos, idx,
        o_part, lse_part,
        float(softmax_scale), nq,
        q4.stride(0), q4.stride(1), q4.stride(2),
        k_view.stride(0), k_view.stride(1), k_view.stride(2), k_view.stride(3),
        v_view.stride(0), v_view.stride(1), v_view.stride(2), v_view.stride(3),
        ks_s[0], ks_s[1], ks_s[2],
        vs_s[0], vs_s[1], vs_s[2],
        block_tables.stride(0), npos.stride(0) if has_npos else 0, idx.stride(0) if sparse else 0,
        idx.stride(1) if sparse else 0,
        *common_out,
        HKV=hkv, G=g, BLOCK_G=block_g, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d, D=d, PAGE_SIZE=ps,
        NS=ns, WINDOW=win, KV_KIND=kind, HAS_ZERO=has_zero, FP8_NATIVE=bool(kind == 1 and fp8_native(dev)),
        HAS_RES=has_res, HAS_NPOS=has_npos, SPARSE=sparse, IDX_PER_HEAD=int(idx_per_head), K_SEL=k_sel,
        LIST_CHUNK=list_chunk, num_warps=num_warps, num_stages=2,
    )
    slot = ns
    if has_res:
        _small_buffer_kernel[(b * hkv, n_node_blocks)](
            q4, k_res, v_res, res_lens, seq_lens, res_lens, npos, seq_lens, o_part, lse_part,
            float(softmax_scale), nq, slot,
            q4.stride(0), q4.stride(1), q4.stride(2),
            k_res.stride(0), k_res.stride(1), k_res.stride(2), v_res.stride(0), v_res.stride(1), v_res.stride(2),
            npos.stride(0) if has_npos else 0, 0, 0,
            *common_out,
            HKV=hkv, G=g, BLOCK_G=block_g, BLOCK_M=block_m, BLOCK_L=max(16, next_pow2(k_res.shape[2])),
            BLOCK_D=block_d, D=d, MODE=0, NT=0, WINDOW=win, HAS_NPOS=has_npos, num_warps=4)
        slot += 1
    if has_tree:
        _small_buffer_kernel[(b * hkv, n_node_blocks)](
            q4, k_tree, v_tree, seq_lens, seq_lens, seq_lens, npos, tm, o_part, lse_part,
            float(softmax_scale), nq, slot,
            q4.stride(0), q4.stride(1), q4.stride(2),
            k_tree.stride(0), k_tree.stride(2), k_tree.stride(1), v_tree.stride(0), v_tree.stride(2), v_tree.stride(1),
            npos.stride(0) if has_npos else 0, tm.stride(0) if tm.shape[0] > 1 else 0, tm.stride(1),
            *common_out,
            HKV=hkv, G=g, BLOCK_G=block_g, BLOCK_M=block_m, BLOCK_L=max(16, next_pow2(nq)),
            BLOCK_D=block_d, D=d, MODE=1, NT=nq, WINDOW=0, HAS_NPOS=has_npos, num_warps=4)

    out = torch.empty(b, nq, hq, d, dtype=q4.dtype, device=dev)
    lse_out = torch.empty(b, nq, hq, dtype=torch.float32, device=dev) if return_lse else None
    merge_partials_triton(o_part.view(b * nq * hq, nsp, d), lse_part.view(b * nq * hq, nsp),
                          out.view(b * nq * hq, d), lse_out.view(-1) if lse_out is not None else None)
    return (out, lse_out) if return_lse else out


def paged_decode_attention_triton(q, k_pool, v_pool, block_tables, seq_lens, *, softmax_scale=None, kv_dtype="bf16",
                                  k_scale=None, v_scale=None, k_zero=None, v_zero=None, window=None, k_res=None,
                                  v_res=None, res_lens=None, topk_idx=None, max_seq_len=None, num_splits=None,
                                  return_lse=False):
    """Decode attention ``q [B, Hq, D]`` -> ``[B, Hq, D]`` (see :func:`run_paged_attention`)."""
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    r = run_paged_attention(q.unsqueeze(1), k_pool, v_pool, block_tables, seq_lens, softmax_scale=scale,
                            kv_dtype=kv_dtype, k_scale=k_scale, v_scale=v_scale, k_zero=k_zero, v_zero=v_zero,
                            window=window, k_res=k_res, v_res=v_res, res_lens=res_lens, topk_idx=topk_idx,
                            max_seq_len=max_seq_len, num_splits=num_splits, return_lse=return_lse)
    if return_lse:
        return r[0][:, 0], r[1][:, 0]
    return r[:, 0]
