"""Triton mixed-precision block-scaled decode attention (``ops_attn.mixed_attention``).

One kernel for every KV precision Forge serves - BF16/FP16/FP32, FP8-E4M3, INT8, INT4, NVFP4, MXFP4 and ternary
(1.58-bit) - with the low-bit payload dequantised **once per tile, fused into the matmuls**.  The maths is the flash
two-pass recurrence of :mod:`forge.kernels.ref.attn_mixed` (the oracle): pass 1 computes the online-softmax statistics
``m, l`` without touching V; pass 2 recomputes the scores and accumulates ``O`` against the *exact* final LSE.

**Status: implemented, not validated on NVIDIA hardware** (exercised in the Triton CPU interpreter by
``tests/interp_attn_mixed_cases.py``; GPU tests in ``tests_gpu/test_kernels_attn_mixed.py``).  Minimum Triton 2.1,
PyTorch 2.1.  Hardware: sm_70+ (fp16), sm_80+ (bf16 / cp.async); no TMA / warp-specialisation / inline PTX, so it
compiles for every target in the support matrix.  Native block-scaled MMA (``tl.dot_scaled``) is *not* used: the
payloads are expanded in registers to the activation dtype, which is what makes ternary work and keeps one code path
across all chips (see ``docs/KERNELS.md`` for why).

Design (what is borrowed from FlashAttention-2/3/4 and what is ours)
--------------------------------------------------------------------
* **FlashAttention-2 work partitioning**: one program per (sequence, KV head, split); the ``g = Hq / Hkv`` query heads of
  a KV head fill the ``tl.dot`` M rows so every KV byte read feeds ``g`` heads.  Keys are addressed per token
  (``pos -> page_table[pos // ps], pos % ps``) so ``BLOCK_N`` is independent of the page size (paged KV, as in FA2 2.5).
* **FlashAttention-3 block quantisation**: the per-block scale is applied to the *score* tile (``S * k_scale``) and to
  the *probability* tile (``(P * v_scale) @ V``), never to the stored payload - the payload is expanded exactly once.
  ``sum_c p * (v_c * s_c) == sum_c (p * s_c) * v_c`` is what makes folding the value scale into P exact.
* **FA4 ``frexp`` E8M0 exponent** (:func:`forge.kernels.ref.attn_kv.e8m0_exponent`) so MXFP4 blocks never round a
  maximum just above a power of two down onto it.
* **FA4 rescale threshold**: ``RESCALE_THRESHOLD`` skips the ``acc *= alpha`` multiply when the running max moves less
  than the threshold (saves a full ``[M, D]`` fp32 multiply per tile in the steady state).
* **FA4 ``track_unquantized_sum``**: the LSE / normaliser come from the float32 probabilities even when ``P`` is cast to
  a lower dtype for the PV dot (``P_DTYPE``), so a quantized P does not bias the softmax denominator.
* **Attention sinks and softcap**: an optional per-head sink logit enlarges the denominator without a value vector; the
  Gemma-2 softcap squashes the logits.
* **Split-KV** (flash-decoding, FA2 2.2): each split writes a normalised partial ``O`` + natural-log LSE; a second tiny
  kernel merges them with a stable LSE combine (empty splits carry ``-inf``).
* **Ternary is ours**: FA has no 1.58-bit path.  Keys are the ternary payload (``{-1,0,+1}``) scaled per 32-channel
  block; values can be ternary too (the format is symmetric) though in practice V stays FP4/FP8 (``docs/TERNARY.md``).
"""
from __future__ import annotations

import math
import os

import torch

from forge.kernels.ref.attn_kv import (
    kv_kind_id,
    normalize_kv_dtype,
)
from forge.kernels.triton.attn_common import HAVE_TRITON, merge_partials_triton
from forge.kernels.triton.util_dispatch import cdiv, next_pow2, num_sms

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _e2m1_decode(code):
        """E2M1 4-bit code (int32 0..15) -> fp32 (mirrors ``ref.attn_kv.decode_e2m1_codes``).

        magnitude ``m = code & 7`` -> ``{0,.5,1,1.5,2,3,4,6}``: ``m < 4`` -> ``m/2``; else ``2**(m>>1 - 1) * (1 + (m&1)/2)``.
        """
        mag = code & 7
        m1 = (mag & 1).to(tl.float32)
        e = tl.maximum((mag >> 1) - 1, 0)  # clamp the shift so mag < 2 never forms a negative exponent
        large = tl.exp2(e.to(tl.float32)) * (1.0 + 0.5 * m1)
        small = mag.to(tl.float32) * 0.5
        val = tl.where(mag < 4, small, large)
        return tl.where(((code >> 3) & 1) == 1, -val, val)

    @triton.jit
    def _ternary_decode(code):
        """2-bit ternary code -> fp32 in {-1, 0, +1} (00->0, 01->+1, 11->-1, 10->0)."""
        c = code & 3
        return tl.where((c & 1) == 1, tl.where((c & 2) == 2, -1.0, 1.0), 0.0)

    @triton.jit
    def _load_block_rows(base_ptr, row_off, offs_c, mask2d, stride_c, kind: tl.constexpr, pack: tl.constexpr):
        """Load ``[N, BLOCK_D]`` block-scaled payload elements (float32, *unscaled*) at packed byte offsets.

        ``row_off`` are int64 element offsets of the row starts in the packed pool; ``offs_c`` are logical channel
        indices ``0..D``.  ``pack`` = elements per byte (2 for fp4, 4 for ternary).  Each lane reads the byte holding its
        element (``offs_c // pack``) and extracts the element's bits (``offs_c % pack``) - the packed row is therefore
        read ``pack`` times per byte (redundant load instructions, not extra HBM traffic: the line is cached).
        """
        byte_idx = offs_c // pack
        ptrs = base_ptr + row_off[:, None] + byte_idx[None, :] * stride_c
        byte = tl.load(ptrs, mask=mask2d, other=0)
        b = byte.to(tl.int32)
        if kind == 4 or kind == 5:  # fp4: two 4-bit codes per byte, even element = low nibble
            code = (b >> ((offs_c % 2) * 4)) & 15
            return _e2m1_decode(code)
        # ternary: four 2-bit codes per byte
        code = (b >> ((offs_c % 4) * 2)) & 3
        return _ternary_decode(code)

    @triton.jit
    def _decode_scale(bits, kind: tl.constexpr):
        """Block-scale value -> fp32 (E4M3 for nvfp4, E8M0 for mxfp4, already fp32 for ternary)."""
        if kind == 4:  # nvfp4: E4M3FN stored as float8 (loaded as its value); decode from the byte pattern
            b = bits.to(tl.int32) & 0xFF
            e = (b >> 3) & 15
            m = b & 7
            mf = m.to(tl.float32)
            sub = mf * 0.001953125
            nrm = (1.0 + mf * 0.125) * tl.exp2((e - 7).to(tl.float32))
            val = tl.where(e == 0, sub, nrm)
            return tl.where((b & 0x80) != 0, -val, val)
        if kind == 5:  # mxfp4: E8M0 biased exponent
            return tl.exp2((bits.to(tl.int32) & 0xFF).to(tl.float32) - 127.0)
        return bits.to(tl.float32)  # ternary: fp32 scale

    @triton.jit
    def _mixed_decode_kernel(
        q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, ts_ptr, bt_ptr, seqlen_ptr,
        o_part_ptr, lse_part_ptr,
        sm_scale,
        sq_b, sq_h,
        sk_pg, sk_h, sk_t, sk_c,
        sv_pg, sv_h, sv_t, sv_c,
        sks_pg, sks_h, sks_c,   # key scale strides  [P, Hkv, D/BS]
        svs_pg, svs_h, svs_t, svs_c,  # value scale strides [P, Hkv, T, D/BS]
        sbt_b,
        so_b, so_h, so_s,
        sl_b, sl_h,
        HKV: tl.constexpr, G: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr, PAGE_SIZE: tl.constexpr,
        NS: tl.constexpr, WINDOW: tl.constexpr, SOFTCAP: tl.constexpr,
        KV_KIND: tl.constexpr, PACK: tl.constexpr, BS: tl.constexpr,
        HAS_TS: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // HKV
        hkv = pid_bh % HKV
        rows = tl.arange(0, BLOCK_M)
        gi = rows % BLOCK_G
        row_valid = gi < G
        hq = hkv * G + gi
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q = tl.load(q_ptr + b * sq_b + hq[:, None] * sq_h + offs_d[None, :],
                    mask=row_valid[:, None] & d_mask[None, :], other=0.0)
        seq_len = tl.load(seqlen_ptr + b)
        lo = seq_len * 0
        if WINDOW > 0:
            lo = tl.maximum(seq_len - WINDOW, 0)
        span = tl.maximum(seq_len - lo, 0)
        chunk = (((span + NS - 1) // NS + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        loop_lo = lo + pid_s * chunk
        loop_hi = tl.minimum(loop_lo + chunk, seq_len)
        qk_scale = sm_scale * 1.4426950408889634
        ts = tl.load(ts_ptr) if HAS_TS else 1.0

        # ---- pass 1: online-softmax statistics (no V) --------------------------------------------------------
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        for start in range(loop_lo, loop_hi, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            tok_valid = offs < loop_hi
            page = tl.load(bt_ptr + b * sbt_b + offs // PAGE_SIZE, mask=tok_valid, other=0)
            slot = offs % PAGE_SIZE
            row_off = page.to(tl.int64) * sk_pg + hkv * sk_h + slot.to(tl.int64) * sk_t
            mask2d = tok_valid[:, None] & d_mask[None, :]
            k = _load_block_rows(k_ptr, row_off, offs_d, mask2d, sk_c, KV_KIND, PACK)
            if KV_KIND >= 4:  # apply the key block scale to the score tile (per channel block, per page)
                sc_off = page[:, None].to(tl.int64) * sks_pg + hkv * sks_h + (offs_d // BS)[None, :] * sks_c
                kb = tl.load(ks_ptr + sc_off, mask=mask2d, other=1.0)
                ks = _decode_scale(kb, KV_KIND) * ts
                k = k * ks
            qk = tl.dot(q, tl.trans(k.to(q.dtype))) * qk_scale
            if SOFTCAP > 0:
                # softcap acts on the natural-logit logits: recover s_nat = qk/LOG2E, squash, re-apply LOG2E
                s_nat = qk / 1.4426950408889634
                qk = SOFTCAP * (2.0 * tl.sigmoid(2.0 * s_nat / SOFTCAP) - 1.0) * 1.4426950408889634
            valid = tok_valid[None, :] & row_valid[:, None]
            if WINDOW > 0:
                valid = valid & (offs[None, :] > (seq_len - 1 - WINDOW))
            qk = tl.where(valid, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(qk - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
        # The attention sink is folded into the *merged* LSE on the host (see mixed_attention_triton): doing it per split
        # would count it once per split.  The per-split statistics written here are therefore sink-free.
        lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.log2(tl.where(l_i == 0.0, 1.0, l_i))) * 0.6931471805599453)
        lse_log2 = lse * 1.4426950408889634

        # ---- pass 2: accumulate O against the exact final LSE -----------------------------------------------
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        for start in range(loop_lo, loop_hi, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            tok_valid = offs < loop_hi
            page = tl.load(bt_ptr + b * sbt_b + offs // PAGE_SIZE, mask=tok_valid, other=0)
            slot = offs % PAGE_SIZE
            row_off = page.to(tl.int64) * sk_pg + hkv * sk_h + slot.to(tl.int64) * sk_t
            mask2d = tok_valid[:, None] & d_mask[None, :]
            k = _load_block_rows(k_ptr, row_off, offs_d, mask2d, sk_c, KV_KIND, PACK)
            if KV_KIND >= 4:
                sc_off = page[:, None].to(tl.int64) * sks_pg + hkv * sks_h + (offs_d // BS)[None, :] * sks_c
                kb = tl.load(ks_ptr + sc_off, mask=mask2d, other=1.0)
                k = k * (_decode_scale(kb, KV_KIND) * ts)
            qk = tl.dot(q, tl.trans(k.to(q.dtype))) * qk_scale
            if SOFTCAP > 0:
                # softcap acts on the natural-logit logits: recover s_nat = qk/LOG2E, squash, re-apply LOG2E
                s_nat = qk / 1.4426950408889634
                qk = SOFTCAP * (2.0 * tl.sigmoid(2.0 * s_nat / SOFTCAP) - 1.0) * 1.4426950408889634
            valid = tok_valid[None, :] & row_valid[:, None]
            if WINDOW > 0:
                valid = valid & (offs[None, :] > (seq_len - 1 - WINDOW))
            qk = tl.where(valid, qk, float("-inf"))
            p = tl.exp2(qk - lse_log2[:, None])
            # value payload + per-token block scale folded into P before the PV dot
            v_row_off = page.to(tl.int64) * sv_pg + hkv * sv_h + slot.to(tl.int64) * sv_t
            v = _load_block_rows(v_ptr, v_row_off, offs_d, mask2d, sv_c, KV_KIND, PACK)
            if KV_KIND >= 4:
                vs_off = page.to(tl.int64) * svs_pg + hkv * svs_h + slot.to(tl.int64) * svs_t
                vb = tl.load(vs_ptr + vs_off[:, None] + (offs_d // BS)[None, :] * svs_c, mask=mask2d, other=1.0)
                vs = _decode_scale(vb, KV_KIND) * ts
                # the value block scale is per (token, channel block): apply it to the V tile (FA folds it into P only
                # when it is constant over channels, which is not the case for per-block scales)
                acc = tl.dot(p.to(q.dtype), (v * vs).to(q.dtype), acc)
            else:
                acc = tl.dot(p.to(q.dtype), v.to(q.dtype), acc)

        # pass 2 already used the final LSE, so `acc` is normalised; only a fully-masked row (lse = -inf -> p = 0) stays 0
        o = acc
        o_ptrs = o_part_ptr + b * so_b + hq[:, None] * so_h + pid_s * so_s + offs_d[None, :]
        tl.store(o_ptrs, o, mask=row_valid[:, None] & d_mask[None, :])
        tl.store(lse_part_ptr + b * sl_b + hq * sl_h + pid_s, lse, mask=row_valid)


def _pick_splits(base_programs: int, max_tiles: int, device, override: int | None) -> int:
    cap = max(1, max_tiles)
    if override is not None:
        return max(1, min(int(override), cap))
    env = os.environ.get("FORGE_DECODE_SPLITS")
    if env:
        return max(1, min(int(env), cap))
    target = 2 * num_sms(device.index or 0) if device.type == "cuda" else 4
    want = max(1, cdiv(target, max(1, base_programs)))
    ns = max(1, min(64, want, cap))
    return 1 << (ns.bit_length() - 1)


def mixed_attention_triton(
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
    max_seq_len: int | None = None,
    num_splits: int | None = None,
    return_lse: bool = False,
):
    """Launch the mixed-precision decode kernel.  ``q [B, Hq, D]`` -> ``[B, Hq, D]`` (see the oracle for semantics)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    kv = normalize_kv_dtype(kv_dtype)
    kind = kv_kind_id(kv)
    if kind < 4:
        raise NotImplementedError("the mixed kernel handles block-scaled KV (nvfp4/mxfp4/ternary); the fp8/int8/int4 family "
                                  "uses paged_decode_attention_triton")
    q = q if q.stride(-1) == 1 else q.contiguous()
    b, hq, d = q.shape
    p_, hkv, ps, _ = k_pool.shape
    if hq % hkv:
        raise ValueError("Hq must be a multiple of Hkv")
    g = hq // hkv
    dev = q.device
    pack = 4 if kind == 6 else 2
    bs = 32 if kind in (5, 6) else 16
    if d % bs:
        raise ValueError(f"head dim {d} must be a multiple of the {kv} block size {bs}")
    # block-scaled KV needs both scale tensors; validate their shapes and the page-table bounds before launching so a
    # malformed input raises a clean error instead of reading past a pool / table (the kernel trusts these).
    if k_block_scale is None or v_block_scale is None:
        raise ValueError(f"{kv} KV needs k_block_scale and v_block_scale")
    nblk = d // bs
    if tuple(k_block_scale.shape) != (p_, hkv, nblk) or tuple(v_block_scale.shape) != (p_, hkv, ps, nblk):
        raise ValueError(f"{kv} block-scale shapes: expected k {((p_, hkv, nblk))} / v {((p_, hkv, ps, nblk))}, "
                         f"got {tuple(k_block_scale.shape)} / {tuple(v_block_scale.shape)}")
    seq_lens = seq_lens.to(torch.int32).contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()
    max_pages = block_tables.shape[1]
    n_pages = ((seq_lens.to(torch.int64) + ps - 1) // ps)
    if int(n_pages.max()) > max_pages:
        raise ValueError(f"a sequence needs more than the {max_pages} pages its block-table row holds")
    live_mask = torch.arange(max_pages, device=dev)[None, :] < n_pages[:, None]  # [B, max_pages]
    live = block_tables[live_mask]
    if live.numel() and (int(live.min()) < 0 or int(live.max()) >= p_):
        raise ValueError(f"block_tables page ids must be in [0, {p_}): got [{int(live.min())}, {int(live.max())}]")
    # Block scales are consumed as raw bytes (E4M3 / E8M0) except for ternary, whose scale is a real fp32 tensor.
    if kind == 6:
        k_bs = k_block_scale.float().contiguous()
        v_bs = v_block_scale.float().contiguous()
    else:
        k_bs = k_block_scale if k_block_scale.dtype == torch.uint8 else k_block_scale.view(torch.uint8)
        v_bs = v_block_scale if v_block_scale.dtype == torch.uint8 else v_block_scale.view(torch.uint8)
    k_bs = k_bs.contiguous()
    v_bs = v_bs.contiguous()
    ts = torch.as_tensor(tensor_scale, dtype=torch.float32, device=dev).reshape(1) if tensor_scale is not None \
        else torch.ones(1, dtype=torch.float32, device=dev)
    if kind == 4 and tensor_scale is None:
        raise ValueError("nvfp4 KV requires its tensor_scale (the pool stores block scales pre-divided by it)")
    block_g = next_pow2(g)
    block_m = max(16, block_g)
    block_d = max(16, next_pow2(d))
    block_n = 64 if d <= 128 else 32
    num_warps = 4 if block_d <= 128 else 8
    win = int(window) if window else 0
    max_len = int(max_seq_len) if max_seq_len is not None else block_tables.shape[1] * ps
    ns = _pick_splits(b * hkv, cdiv(max_len, block_n), dev, num_splits)

    o_part = torch.empty(b, hq, ns, d, dtype=torch.float32, device=dev)
    lse_part = torch.empty(b, hq, ns, dtype=torch.float32, device=dev)
    _mixed_decode_kernel[(b * hkv, ns)](
        q, k_pool, v_pool, k_bs, v_bs, ts, block_tables, seq_lens,
        o_part, lse_part,
        float(softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)),
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        k_bs.stride(0), k_bs.stride(1), k_bs.stride(2),
        v_bs.stride(0), v_bs.stride(1), v_bs.stride(2), v_bs.stride(3),
        block_tables.stride(0),
        o_part.stride(0), o_part.stride(1), o_part.stride(2),
        lse_part.stride(0), lse_part.stride(1),
        HKV=hkv, G=g, BLOCK_G=block_g, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d, D=d, PAGE_SIZE=ps,
        NS=ns, WINDOW=win, SOFTCAP=float(softcap) if softcap else 0.0,
        KV_KIND=kind, PACK=pack, BS=bs, HAS_TS=tensor_scale is not None,
        num_warps=num_warps, num_stages=2,
    )
    out = torch.empty(b, hq, d, dtype=q.dtype, device=dev)
    lse_merged = torch.empty(b, hq, dtype=torch.float32, device=dev)
    merge_partials_triton(o_part.view(b * hq, ns, d), lse_part.view(b * hq, ns), out.view(b * hq, d),
                          lse_merged.view(-1))
    if sink is not None:
        # fold the attention sink once into the merged statistics: l_new = l_old + exp(sink) (natural units);
        # O scales by l_old / l_new because the sink has no value vector (FA4 apply_learnable_sink, post-merge).
        sink_f = sink.to(torch.float32).reshape(1, hq).to(dev)
        l_old = torch.exp(lse_merged)  # 0 where lse = -inf
        l_new = l_old + torch.exp(sink_f)
        scale = torch.where(l_new > 0, l_old / l_new.clamp_min(1e-38), torch.zeros_like(l_new))
        out = (out.float() * scale.unsqueeze(-1)).to(out.dtype)
        lse_merged = torch.log(l_new.clamp_min(1e-38))
    lse_out = lse_merged if return_lse else None
    return (out, lse_out) if return_lse else out
