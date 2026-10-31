"""Triton kernels for speculative decoding (Plan K6, KR-6/KR-7): tree attention, fused rejection-sampling verification and
KV accept/rollback compaction.

**Status: implemented, not yet validated on NVIDIA hardware** (all three run in the Triton CPU interpreter in the unit tests).
Min Triton 2.1 (``tl.cumsum``, ``tl.randint4x``).

* Tree attention = :func:`forge.kernels.triton.attn_decode.run_paged_attention` with ``NQ = #tree nodes`` query rows packed
  into the M tile, attending to the committed context (paged, any KV dtype, optional residual window) plus the draft tree
  through an arbitrary ancestor mask; per-node positions drive the sliding window.
* ``rejection_verify``: one program per sequence; no host round trip, CUDA-graph capturable (static shapes, device-side
  the seed is a launch constant; pass ``offset`` as a 0-d device tensor and bump it in place between graph replays).  Exactly
  the maths of :func:`forge.kernels.ref.attn_spec.rejection_verify_ref` (``u * q(x) < p(x)`` accept test, residual
  ``(p - q)+`` resample by inverse CDF, bonus token from ``p_gamma``) with the same Philox counters, so results are
  bit-identical except where the blockwise float32 CDF differs from ``torch.cumsum`` at a bucket boundary.
* ``kv_compact``: two ordered launches (gather accepted rows to a scratch buffer, then scatter) so overlapping source/destination
  ranges are safe on real hardware (a single read-then-write kernel races across warps).
"""
from __future__ import annotations

import math

import torch

from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import next_pow2

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from forge.kernels.triton.attn_common import philox_uniform

    @triton.jit
    def _rejection_verify_kernel(
        draft_ptr, q_ptr, p_ptr, out_ptr, nacc_ptr, seed, offset, off_ptr,
        sq_b, sq_g, sp_b, sp_g,
        G: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_T: tl.constexpr, OFF_PTR: tl.constexpr,
    ):
        b = tl.program_id(0)
        if OFF_PTR:
            # device-resident step counter: stays a *load* inside a captured CUDA graph, so every replay draws fresh randoms
            off = tl.load(off_ptr).to(tl.int64)
        else:
            off = offset
        b64 = b.to(tl.int64)  # large batch x vocab products overflow int32
        zero_i = b * 0
        # ---- sequential accept test ------------------------------------------------------------------------
        n_acc = zero_i
        alive = zero_i + 1
        for i in range(G):
            x = tl.load(draft_ptr + b * G + i).to(tl.int32)
            x = tl.minimum(tl.maximum(x, 0), V - 1)  # memory safety only: callers must pass ids in [0, V)
            px = tl.load(p_ptr + b64 * sp_b + i * sp_g + x).to(tl.float32)
            qx = tl.load(q_ptr + b64 * sq_b + i * sq_g + x).to(tl.float32)
            ctr = ((off + b) * (G + 1) + i) * 2
            u = tl.sum(philox_uniform(seed, tl.zeros([1], tl.uint32) + ctr.to(tl.uint32)), axis=0)
            acc_i = ((u * qx) < px).to(tl.int32)
            alive = alive * acc_i
            n_acc += alive
        # ---- distribution at the stopping position -----------------------------------------------------------
        use_res = n_acc < G
        q_lim = tl.where(use_res, V, 0)  # q row is only read when stopped on a rejection
        p_row = p_ptr + b64 * sp_b + n_acc * sp_g
        q_row = q_ptr + b64 * sq_b + n_acc * sq_g
        z_res = tl.zeros([1], tl.float32)
        z_p = tl.zeros([1], tl.float32)
        for v0 in range(0, V, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vm = offs < V
            pv = tl.load(p_row + offs, mask=vm, other=0.0).to(tl.float32)
            qv = tl.load(q_row + offs, mask=offs < q_lim, other=0.0).to(tl.float32)
            z_res += tl.sum(tl.maximum(pv - qv, 0.0), axis=0)
            z_p += tl.sum(pv, axis=0)
        # residual when stopped on a rejection (and non-degenerate), otherwise the target law itself
        take_res = use_res & (z_res > 0.0)
        z_sel = tl.where(take_res, z_res, z_p)
        ctr2 = ((off + b) * (G + 1) + n_acc) * 2 + 1
        u2 = tl.sum(philox_uniform(seed, tl.zeros([1], tl.uint32) + ctr2.to(tl.uint32)), axis=0)
        target = u2 * z_sel
        running = tl.zeros([1], tl.float32)
        chosen = tl.zeros([1], tl.int32) + V
        last_pos = tl.zeros([1], tl.int32) - 1
        for v0 in range(0, V, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vm = offs < V
            pv = tl.load(p_row + offs, mask=vm, other=0.0).to(tl.float32)
            qv = tl.load(q_row + offs, mask=offs < q_lim, other=0.0).to(tl.float32)
            w = tl.where(take_res, tl.maximum(pv - qv, 0.0), pv)
            w = tl.where(vm, w, 0.0)
            cdf = running + tl.cumsum(w, axis=0)
            hit = (cdf > target) & vm
            first = tl.min(tl.where(hit, offs, V), axis=0)
            chosen = tl.minimum(chosen, first)
            last_pos = tl.maximum(last_pos, tl.max(tl.where((w > 0.0) & vm, offs, -1), axis=0))
            running += tl.sum(w, axis=0)
        token = tl.where(chosen >= V, tl.maximum(last_pos, 0), chosen)
        # ---- emit ----------------------------------------------------------------------------------------------
        offs_t = tl.arange(0, BLOCK_T)
        dr = tl.load(draft_ptr + b * G + offs_t, mask=offs_t < G, other=0).to(tl.int32)
        res = tl.where(offs_t < n_acc, dr, tl.where(offs_t == n_acc, token, -1))
        tl.store(out_ptr + b * (G + 1) + offs_t, res, mask=offs_t <= G)
        tl.store(nacc_ptr + b, n_acc)

    @triton.jit
    def _kv_compact_kernel(
        kp_ptr, vp_ptr, bt_ptr, ctx_ptr, acc_ptr, alen_ptr, tmp_ptr,
        sk_pg, sk_h, sk_t, sv_pg, sv_h, sv_t, sbt_b, sacc_b,
        HKV: tl.constexpr, PAGE_SIZE: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
        PHASE: tl.constexpr,
    ):
        """Two-phase accepted-path move.  ``PHASE`` 0 gathers the source rows into ``tmp``, ``PHASE`` 1 scatters ``tmp`` to the
        destination rows.  Source and destination ranges overlap, and the threads of one program do **not** execute a load tile
        and the following store tile in lock-step (nothing orders thread A's store before thread B's load of the same row), so a
        single-kernel read-then-write is a data race on real hardware.  Two launches on one stream are strictly ordered."""
        pid = tl.program_id(0)
        b = pid // HKV
        h = pid % HKV
        ctx = tl.load(ctx_ptr + b)
        alen = tl.load(alen_ptr + b)
        t = tl.arange(0, BLOCK_T)
        tm = t < alen
        node = tl.load(acc_ptr + b * sacc_b + t, mask=tm, other=0)
        tm = tm & (node >= 0)
        src = ctx + node
        dst = ctx + t
        offs_d = tl.arange(0, BLOCK_D)
        dm = offs_d < D
        m2 = tm[:, None] & dm[None, :]
        tk_off = ((pid.to(tl.int64) * 2) * BLOCK_T + t[:, None].to(tl.int64)) * BLOCK_D + offs_d[None, :]
        tv_off = ((pid.to(tl.int64) * 2 + 1) * BLOCK_T + t[:, None].to(tl.int64)) * BLOCK_D + offs_d[None, :]
        if PHASE == 0:
            sp = tl.load(bt_ptr + b * sbt_b + src // PAGE_SIZE, mask=tm, other=0)
            so = src % PAGE_SIZE
            k_src = kp_ptr + sp[:, None].to(tl.int64) * sk_pg + h * sk_h + so[:, None].to(tl.int64) * sk_t + offs_d[None, :]
            v_src = vp_ptr + sp[:, None].to(tl.int64) * sv_pg + h * sv_h + so[:, None].to(tl.int64) * sv_t + offs_d[None, :]
            tl.store(tmp_ptr + tk_off, tl.load(k_src, mask=m2, other=0.0), mask=m2)
            tl.store(tmp_ptr + tv_off, tl.load(v_src, mask=m2, other=0.0), mask=m2)
        else:
            dp = tl.load(bt_ptr + b * sbt_b + dst // PAGE_SIZE, mask=tm, other=0)
            do = dst % PAGE_SIZE
            k_dst = kp_ptr + dp[:, None].to(tl.int64) * sk_pg + h * sk_h + do[:, None].to(tl.int64) * sk_t + offs_d[None, :]
            v_dst = vp_ptr + dp[:, None].to(tl.int64) * sv_pg + h * sv_h + do[:, None].to(tl.int64) * sv_t + offs_d[None, :]
            tl.store(k_dst, tl.load(tmp_ptr + tk_off, mask=m2, other=0.0), mask=m2)
            tl.store(v_dst, tl.load(tmp_ptr + tv_off, mask=m2, other=0.0), mask=m2)


def rejection_verify_triton(draft_tokens, q_probs, p_probs, seed: int, offset=0, *, block_v: int = 1024):
    """Fused verification. ``draft_tokens [B, G]``, ``q_probs [B, G, V]``, ``p_probs [B, G+1, V]`` (float32/bf16/fp16)
    -> ``(tokens [B, G+1] int32, n_accepted [B] int32)`` (see the reference for semantics).

    ``offset`` is the RNG stream position.  A python int is baked into the launch (fine for eager use).  Inside a CUDA graph pass a
    0-d **device** integer tensor instead and bump it in place between replays - a baked-in int would replay identical random
    numbers on every graph launch (correlated accept decisions, i.e. a biased sampler).  Callers must advance ``offset`` by at
    least ``B`` per call (counters are ``(offset + b) * (G + 1) + i``)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    b, g = draft_tokens.shape
    v = p_probs.shape[-1]
    if seed < 0:
        raise ValueError("seed must be non-negative (use util_rng.derive_seed)")
    draft = draft_tokens.to(torch.int32).contiguous()
    q = q_probs.contiguous()
    p = p_probs.contiguous()
    out = torch.empty(b, g + 1, dtype=torch.int32, device=draft.device)
    nacc = torch.empty(b, dtype=torch.int32, device=draft.device)
    off_is_ptr = isinstance(offset, torch.Tensor)
    if off_is_ptr:
        off_t = offset.to(device=draft.device, dtype=torch.int64).reshape(1)
        off_val = 0
    else:
        off_t, off_val = out, int(offset)
    _rejection_verify_kernel[(b,)](
        draft, q, p, out, nacc, int(seed), off_val, off_t,
        q.stride(0), q.stride(1), p.stride(0), p.stride(1),
        G=g, V=v, BLOCK_V=min(block_v, max(16, next_pow2(v))), BLOCK_T=max(2, next_pow2(g + 1)), OFF_PTR=off_is_ptr,
        num_warps=4)
    return out, nacc


def kv_compact_triton(k_pool, v_pool, block_tables, ctx_lens, accepted_idx, accept_lens):
    """In-place accepted-path compaction on ``[P, Hkv, ps, D]`` pools (native float dtypes). ``accepted_idx [B, A]`` with
    ``A <= 128``."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    b, a = accepted_idx.shape
    p_, hkv, ps, d = k_pool.shape
    if a > 128:
        raise ValueError("accepted path longer than 128 nodes: use the reference implementation")
    bt = block_tables.to(torch.int32).contiguous()
    ctx = ctx_lens.to(torch.int32).contiguous()
    acc = accepted_idx.to(torch.int32).contiguous()
    alen = accept_lens.to(torch.int32).contiguous()
    block_t, block_d = max(16, next_pow2(a)), max(16, next_pow2(d))
    tmp = torch.empty(b * hkv * 2 * block_t * block_d, dtype=k_pool.dtype, device=k_pool.device)
    for phase in (0, 1):  # gather, then scatter: strictly ordered launches (see the kernel docstring)
        _kv_compact_kernel[(b * hkv,)](
            k_pool, v_pool, bt, ctx, acc, alen, tmp,
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
            bt.stride(0), acc.stride(0),
            HKV=hkv, PAGE_SIZE=ps, D=d, BLOCK_D=block_d, BLOCK_T=block_t, PHASE=phase, num_warps=4)


def tree_attention_triton(q, k_tree, v_tree, k_pool, v_pool, block_tables, ctx_lens, tree_mask, *, softmax_scale=None,
                          kv_dtype="bf16", k_scale=None, v_scale=None, k_zero=None, v_zero=None, window=None,
                          node_pos=None, k_res=None, v_res=None, res_lens=None, max_seq_len=None, num_splits=None):
    """Tree-verification attention: ``q [B, N, Hq, D]``, ``k_tree, v_tree [B, N, Hkv, D]`` -> ``[B, N, Hq, D]``."""
    from forge.kernels.triton.attn_decode import run_paged_attention

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if node_pos is None:
        tm = tree_mask if tree_mask.dim() == 3 else tree_mask.unsqueeze(0)
        depth = tm.to(torch.int32).sum(-1) - 1  # [B or 1, N]
        node_pos = ctx_lens.to(torch.int32).view(-1, 1) + depth.expand(ctx_lens.shape[0], -1)
    return run_paged_attention(q, k_pool, v_pool, block_tables, ctx_lens, softmax_scale=scale, kv_dtype=kv_dtype,
                               k_scale=k_scale, v_scale=v_scale, k_zero=k_zero, v_zero=v_zero, window=window,
                               k_res=k_res, v_res=v_res, res_lens=res_lens, k_tree=k_tree, v_tree=v_tree,
                               tree_mask=tree_mask, node_pos=node_pos, max_seq_len=max_seq_len, num_splits=num_splits)
