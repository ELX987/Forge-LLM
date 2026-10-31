"""Triton fused RoPE + KV-quantise + KV-page write (Plan §6.7 fusion map: "One pass over K/V before they are cached").

**Status: implemented, not yet validated on NVIDIA hardware** (native-dtype and INT8 paths run in the Triton CPU interpreter in the
unit tests; FP8 stores need ``float8e4nv`` = sm_89+; INT4 is handled by the reference only).  Min Triton 2.1.

Grid ``(T, Hq + Hkv)``: programs ``h < Hq`` rotate the query head into ``q_out``; programs ``h >= Hq`` rotate one KV head of
``k``, quantise it with a *static* per-channel key scale, quantise ``v`` with a *dynamic* per-token scale (absmax over ``D``,
written to ``v_scale_pool``) and scatter both into the paged pools at ``slot_mapping`` (negative slot = skip).  Partial rotary
(``rotary_dim < D``) and both pairings (NeoX half-split / GPT-J interleaved) are supported.
"""
from __future__ import annotations

import torch

from forge.kernels.ref.attn_kv import normalize_kv_dtype
from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import next_pow2, require_fp8_store

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from forge.kernels.triton.util_quant import _round_half_away

    @triton.jit
    def _quant_store(ptrs, y, mask, KV_KIND: tl.constexpr):
        if KV_KIND == 0:
            tl.store(ptrs, y.to(ptrs.dtype.element_ty), mask=mask)
        elif KV_KIND == 1:
            yc = tl.minimum(tl.maximum(y, -448.0), 448.0)
            tl.store(ptrs, yc.to(ptrs.dtype.element_ty), mask=mask)
        else:
            qv = _round_half_away(tl.minimum(tl.maximum(y, -128.0), 127.0))
            tl.store(ptrs, qv.to(tl.int8), mask=mask)

    @triton.jit
    def _rope_kv_write_kernel(
        q_ptr, k_ptr, v_ptr, pos_ptr, cs_ptr, slot_ptr, qo_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr,
        sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, scs_p, sqo_t, sqo_h,
        skp_pg, skp_h, skp_t, svp_pg, svp_h, svp_t, sks_h, sks_d, svs_pg, svs_h, svs_t,
        HQ: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, RD: tl.constexpr, PAGE_SIZE: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, HAS_TAIL: tl.constexpr,
        INTERLEAVED: tl.constexpr, KV_KIND: tl.constexpr, QMAX: tl.constexpr,
    ):
        t = tl.program_id(0)
        h = tl.program_id(1)
        pos = tl.load(pos_ptr + t).to(tl.int64)
        i = tl.arange(0, BLOCK_R)
        im = i < HALF
        cos = tl.load(cs_ptr + pos * scs_p + i, mask=im, other=1.0).to(tl.float32)
        sin = tl.load(cs_ptr + pos * scs_p + HALF + i, mask=im, other=0.0).to(tl.float32)
        if INTERLEAVED:
            i1 = 2 * i
            i2 = 2 * i + 1
        else:
            i1 = i
            i2 = i + HALF
        tt = tl.arange(0, BLOCK_T)
        tm = tt < (D - RD)
        if h < HQ:
            base = q_ptr + t.to(tl.int64) * sq_t + h * sq_h
            qx1 = tl.load(base + i1, mask=im, other=0.0).to(tl.float32)
            qx2 = tl.load(base + i2, mask=im, other=0.0).to(tl.float32)
            ob = qo_ptr + t.to(tl.int64) * sqo_t + h * sqo_h
            tl.store(ob + i1, (qx1 * cos - qx2 * sin).to(qo_ptr.dtype.element_ty), mask=im)
            tl.store(ob + i2, (qx2 * cos + qx1 * sin).to(qo_ptr.dtype.element_ty), mask=im)
            if HAS_TAIL:
                qxt = tl.load(base + RD + tt, mask=tm, other=0.0)
                tl.store(ob + RD + tt, qxt, mask=tm)
        else:
            hk = h - HQ
            slot = tl.load(slot_ptr + t)
            ok = slot >= 0
            half_lim = tl.where(ok, HALF, 0)  # slot < 0 -> every store below is masked off
            tail_lim = tl.where(ok, D - RD, 0)
            d_lim = tl.where(ok, D, 0)
            slot_c = tl.maximum(slot, 0)
            page = (slot_c // PAGE_SIZE).to(tl.int64)
            off = (slot_c % PAGE_SIZE).to(tl.int64)
            kb = k_ptr + t.to(tl.int64) * sk_t + hk * sk_h
            kx1 = tl.load(kb + i1, mask=im, other=0.0).to(tl.float32)
            kx2 = tl.load(kb + i2, mask=im, other=0.0).to(tl.float32)
            o1 = kx1 * cos - kx2 * sin
            o2 = kx2 * cos + kx1 * sin
            if KV_KIND != 0:
                s1 = tl.load(ks_ptr + hk * sks_h + i1 * sks_d, mask=im, other=1.0)
                s2 = tl.load(ks_ptr + hk * sks_h + i2 * sks_d, mask=im, other=1.0)
                o1 = o1 / s1
                o2 = o2 / s2
            kdst = kp_ptr + page * skp_pg + hk * skp_h + off * skp_t
            _quant_store(kdst + i1, o1, i < half_lim, KV_KIND)
            _quant_store(kdst + i2, o2, i < half_lim, KV_KIND)
            if HAS_TAIL:
                kxt = tl.load(kb + RD + tt, mask=tm, other=0.0).to(tl.float32)
                if KV_KIND != 0:
                    st = tl.load(ks_ptr + hk * sks_h + (RD + tt) * sks_d, mask=tm, other=1.0)
                    kxt = kxt / st
                _quant_store(kdst + RD + tt, kxt, tt < tail_lim, KV_KIND)
            offs_d = tl.arange(0, BLOCK_D)
            dm = offs_d < D
            v = tl.load(v_ptr + t.to(tl.int64) * sv_t + hk * sv_h + offs_d, mask=dm, other=0.0).to(tl.float32)
            vdst = vp_ptr + page * svp_pg + hk * svp_h + off * svp_t
            if KV_KIND == 0:
                _quant_store(vdst + offs_d, v, offs_d < d_lim, KV_KIND)
            else:
                amax = tl.max(tl.abs(v), axis=0)
                vs0 = amax / QMAX
                vs = tl.where(vs0 > 0.0, vs0, 1.0)
                _quant_store(vdst + offs_d, v / vs, offs_d < d_lim, KV_KIND)
                tl.store(vs_ptr + page * svs_pg + hk * svs_h + off * svs_t, vs, mask=ok)


def rope_and_kv_write_triton(q, k, v, positions, cos_sin_cache, slot_mapping, k_pool, v_pool, *, kv_dtype: str = "bf16",
                             k_scale=None, v_scale_pool=None, rotary_dim: int | None = None, interleaved: bool = False):
    """Launch the fused kernel. Returns rotated ``q`` (new tensor). ``kv_dtype`` in bf16/fp16/fp32/fp8_e4m3/int8."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    kv = normalize_kv_dtype(kv_dtype)
    if kv == "int4":
        raise ValueError("INT4 fused write is reference-only")
    if kv == "fp8_e4m3":
        require_fp8_store(q.device, "rope_and_kv_write_triton")
    t, hq, d = q.shape
    hkv = k.shape[1]
    rd = d if rotary_dim is None else rotary_dim
    if rd % 2:
        raise ValueError("rotary_dim must be even")
    q = q if q.stride(-1) == 1 else q.contiguous()
    k = k if k.stride(-1) == 1 else k.contiguous()
    v = v if v.stride(-1) == 1 else v.contiguous()
    pos = positions.to(torch.int32).contiguous()
    slots = slot_mapping.to(torch.int32).contiguous()
    cs = cos_sin_cache.contiguous()
    q_out = torch.empty_like(q)
    kind = {"bf16": 0, "fp16": 0, "fp32": 0, "fp8_e4m3": 1, "int8": 2}[kv]
    if kind != 0:
        if v_scale_pool is None:
            raise ValueError("quantised KV write needs v_scale_pool")
        if k_scale is None:
            ksc = torch.ones(1, 1, dtype=torch.float32, device=q.device).expand(hkv, d)
        else:
            ksc = torch.as_tensor(k_scale, dtype=torch.float32, device=q.device)
            if ksc.numel() == 1:
                ksc = ksc.reshape(1, 1).expand(hkv, d)
            elif tuple(ksc.shape) != (hkv, d):
                raise ValueError("k_scale must be a scalar or [Hkv, D]")
        vsp = v_scale_pool
        kp_view, vp_view = k_pool, v_pool
        s_vs = (vsp.stride(0), vsp.stride(1), vsp.stride(2))
        s_ks = (ksc.stride(0), ksc.stride(1))
    else:
        ksc, vsp, kp_view, vp_view = cs, cs, k_pool, v_pool
        s_vs = (0, 0, 0)
        s_ks = (0, 0)
    half = rd // 2
    has_tail = rd < d
    _rope_kv_write_kernel[(t, hq + hkv)](
        q, k, v, pos, cs, slots, q_out, kp_view, vp_view, ksc, vsp,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), cs.stride(0),
        q_out.stride(0), q_out.stride(1),
        kp_view.stride(0), kp_view.stride(1), kp_view.stride(2), vp_view.stride(0), vp_view.stride(1), vp_view.stride(2),
        s_ks[0], s_ks[1], s_vs[0], s_vs[1], s_vs[2],
        HQ=hq, D=d, HALF=half, RD=rd, PAGE_SIZE=k_pool.shape[2], BLOCK_R=max(2, next_pow2(half)),
        BLOCK_D=max(16, next_pow2(d)), BLOCK_T=max(2, next_pow2(d - rd)) if has_tail else 2, HAS_TAIL=has_tail,
        INTERLEAVED=bool(interleaved), KV_KIND=kind, QMAX=448.0 if kind == 1 else 127.0, num_warps=4)
    return q_out
