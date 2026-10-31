"""RoPE + KV-quantise + KV page-write oracle (Plan §6.7 fusion map: "One pass over K/V before they are cached").

For ``T`` new tokens: rotate ``q`` and ``k`` with rotary embeddings, quantise ``k`` (static per-channel scale) and ``v``
(dynamic per-token scale), and write them into the paged pools at ``slot_mapping`` (``slot = page * page_size + offset``).

Conventions: ``cos_sin_cache [max_pos, rotary_dim]`` with ``cos = cache[:, :rd/2]`` and ``sin = cache[:, rd/2:]``
(vLLM layout, see :func:`make_cos_sin_cache`).  ``interleaved=False`` is GPT-NeoX/Llama style (pairs ``(i, i + rd/2)``),
``interleaved=True`` is GPT-J style (pairs ``(2i, 2i + 1)``).  Partial rotary (``rotary_dim < D``) leaves the tail dims
untouched.
"""
from __future__ import annotations

import torch

from forge.kernels.ref.attn_kv import E4M3_MAX, normalize_kv_dtype, pack_int4
from forge.kernels.ref.util_quant import round_half_away


def make_cos_sin_cache(max_pos: int, rotary_dim: int, theta: float = 10000.0, device=None) -> torch.Tensor:
    """``[max_pos, rotary_dim]`` float32 cache (cos | sin) for plain RoPE."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    f = torch.outer(t, inv)
    return torch.cat([f.cos(), f.sin()], dim=-1)


def apply_rope_ref(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, rotary_dim: int | None = None,
                   interleaved: bool = False) -> torch.Tensor:
    """Rotate ``x [T, H, D]`` (float32 result)."""
    d = x.shape[-1]
    rd = d if rotary_dim is None else rotary_dim
    half = rd // 2
    cs = cos_sin_cache.float()[positions.long()]  # [T, rd]
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:rd].unsqueeze(1)
    xf = x.float()
    if interleaved:
        x1, x2 = xf[..., 0:rd:2], xf[..., 1:rd:2]
        o1, o2 = x1 * cos - x2 * sin, x2 * cos + x1 * sin
        rot = torch.stack([o1, o2], dim=-1).reshape(*xf.shape[:-1], rd)
    else:
        x1, x2 = xf[..., :half], xf[..., half:rd]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rot, xf[..., rd:]], dim=-1) if rd < d else rot


def rope_and_kv_write_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    *,
    kv_dtype: str = "bf16",
    k_scale: torch.Tensor | float | None = None,
    v_scale_pool: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    """Returns the rotated queries (``q.dtype``); writes ``k``/``v`` into the pools in place.

    ``k_scale``: static key scale (scalar, ``[Hkv]`` is *not* accepted - use ``[Hkv, D]``) used as ``k_q = k_rot / scale``.
    ``v_scale_pool [P, Hkv, ps]``: receives the per-token value scales for quantised dtypes.
    """
    kv = normalize_kv_dtype(kv_dtype)
    ps = k_pool.shape[2]
    q_rot = apply_rope_ref(q, positions, cos_sin_cache, rotary_dim, interleaved).to(q.dtype)
    k_rot = apply_rope_ref(k, positions, cos_sin_cache, rotary_dim, interleaved)  # fp32 [T, Hkv, D]
    vf = v.float()
    slot = slot_mapping.long()
    ok = slot >= 0
    page = torch.div(slot[ok], ps, rounding_mode="floor")
    off = slot[ok] % ps
    kr, vr = k_rot[ok], vf[ok]
    if kv in ("bf16", "fp16", "fp32"):
        k_pool[page, :, off, :] = kr.to(k_pool.dtype)
        v_pool[page, :, off, :] = vr.to(v_pool.dtype)
        return q_rot
    if k_scale is None:
        ks = torch.ones(1, 1, device=k.device)
    else:
        ks = torch.as_tensor(k_scale, dtype=torch.float32, device=k.device)
        if ks.dim() == 0:
            ks = ks.reshape(1, 1)
    ks = ks.unsqueeze(0)  # [1, Hkv|1, D|1]
    qmax = {"fp8_e4m3": E4M3_MAX, "int8": 127.0, "int4": 7.0}[kv]
    amax = vr.abs().amax(-1)  # [n, Hkv]
    vs0 = amax / qmax
    vs = torch.where(vs0 > 0, vs0, torch.ones_like(vs0))
    ky = kr / ks
    vy = vr / vs.unsqueeze(-1)
    if kv == "fp8_e4m3":
        kq = ky.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        vq = vy.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    elif kv == "int8":
        kq = round_half_away(ky).clamp(-128, 127).to(torch.int8)
        vq = round_half_away(vy).clamp(-128, 127).to(torch.int8)
    else:
        kq = pack_int4(round_half_away(ky).clamp(-8, 7).to(torch.int8))
        vq = pack_int4(round_half_away(vy).clamp(-8, 7).to(torch.int8))
    k_pool[page, :, off, :] = kq
    v_pool[page, :, off, :] = vq
    if v_scale_pool is not None:
        v_scale_pool[page, :, off] = vs
    return q_rot


#: name used in docs/ARCHITECTURE.md section 5.5
rope_kv_write_ref = rope_and_kv_write_ref
