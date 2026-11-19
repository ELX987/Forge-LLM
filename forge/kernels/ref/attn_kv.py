"""KV-cache numerics shared by the attention oracles, the Triton kernels and the tests.

Page layout (Plan K4, head-major): ``pool[num_pages, Hkv, page_size, D]``.

Quantised KV storage (KIVI-style, Plan K4 / P1 F8):

* ``fp8_e4m3`` - ``torch.float8_e4m3fn`` (or its ``uint8`` bit view), scale optional.
* ``int8``     - ``torch.int8``, symmetric (+ optional additive zero point).
* ``int4``     - ``torch.uint8`` with two offset-binary nibbles per byte along ``D``
  (``byte = (q[2i] + 8) | ((q[2i+1] + 8) << 4)``, values in ``[-8, 7]``), pool last dim is ``D // 2``.

Block-quantised KV storage (``nvfp4`` / ``mxfp4`` - "block-scaled" family of the mixed-precision attention op):

* ``nvfp4`` - E2M1 elements (two codes per byte along ``D``, *even* element = low nibble, code ``s|ee|m`` like
  :func:`forge.formats.fp4.decode_e2m1`), **one E4M3 scale per 16 channels** (NVFP4 block size), optional fp32 tensor
  scale (default 1).  ``x = E2M1(code) * e4m3(scale) * tensor_scale``.  Keys get per-channel scales
  ``k_block_scale [P, Hkv, D/16]`` (one per channel block over all tokens of the page), values per-token-block scales
  ``v_block_scale [P, Hkv, ps, D/16]`` - the KIVI split (keys reduce over tokens, values keep one scale per token).
* ``mxfp4`` - E2M1 elements (same packing), **one E8M0 power-of-two scale per 32 channels** (OCP MX block size).
  ``x = E2M1(code) * 2**(scale_code - 127)``; same K/V scale split as NVFP4.
* Both are symmetric (no zero point).  The pool last dim is ``D // 2``.  Scales are *not* run through the float32
  :func:`expand_k_scale` / :func:`expand_v_scale` path (that would destroy the 1- or 0.25-byte scale encoding and
  cost 4 bytes per block), so their accepted shapes are the logical, head-major ones below.
* ``ternary`` - 1.58-bit keys/values in ``{-1, 0, +1}``: two bits per element (``00`` -> 0, ``01`` -> +1, ``11`` -> -1,
  ``10`` reserved -> 0 - the code of :mod:`forge.formats.ternary`, element ``j`` at bits ``2*(j%4)`` of byte ``j//4``),
  pool last dim ``D // 4``; **one fp32 scale per 32 channels** with the same K/V split.  ``x = t * scale``.  Only keys
  are expected to survive 1.58 bits in practice (see ``forge.kernels.ref.attn_mixed``); values are supported for
  completeness.

Scales for the ``fp8_e4m3`` / ``int8`` / ``int4`` family (all float32, accepted shapes are normalised by
:func:`expand_k_scale` / :func:`expand_v_scale`):

* keys   : per page, per KV head, **per channel** ``[P, Hkv, D]`` (also ``[P, Hkv, 1]``, ``[Hkv, D]``, scalar, ``None``)
* values : per page, per KV head, **per token**   ``[P, Hkv, page_size]`` (also ``[P, Hkv, 1]``, scalar, ``None``)

Dequantisation is ``x = q * scale (+ zero)``; ``zero`` has the same shape as the scale and is optional.
"""
from __future__ import annotations

import torch

KV_DTYPES = ("bf16", "fp16", "fp32", "fp8_e4m3", "int8", "int4", "nvfp4", "mxfp4", "ternary")
QUANT_KV_DTYPES = ("fp8_e4m3", "int8", "int4", "nvfp4", "mxfp4", "ternary")
#: KV formats whose scales live in a *block* layout (not per channel / per token float32) - see module docstring
BLOCK_KV_DTYPES = ("nvfp4", "mxfp4", "ternary")
E4M3_MAX = 448.0
E2M1_MAX = 6.0
#: elements per quantisation block of the block-scaled formats
KV_BLOCK_SIZE = {"nvfp4": 16, "mxfp4": 32, "ternary": 32}
#: packed elements per byte of the block-scaled formats
_KV_PACK = {"nvfp4": 2, "mxfp4": 2, "ternary": 4}

_ALIASES = {
    "bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16", "half": "fp16",
    "float32": "fp32", "fp32": "fp32", "float": "fp32",
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "e4m3": "fp8_e4m3", "fp8e4m3": "fp8_e4m3",
    "int8": "int8", "int4": "int4",
    "nvfp4": "nvfp4", "fp4": "nvfp4", "e2m1": "nvfp4",
    "mxfp4": "mxfp4", "ocp_fp4": "mxfp4",
    "ternary": "ternary", "t158": "ternary", "ternary158": "ternary", "ternary_158": "ternary", "1.58bit": "ternary",
    "1.58": "ternary", "int2": "ternary",
}


def normalize_kv_dtype(kv_dtype: str | torch.dtype | None) -> str:
    """Canonical KV dtype name (``None`` -> ``"bf16"``)."""
    if kv_dtype is None:
        return "bf16"
    if isinstance(kv_dtype, torch.dtype):
        m = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32",
             torch.int8: "int8", torch.uint8: "int4"}
        if hasattr(torch, "float8_e4m3fn"):
            m[torch.float8_e4m3fn] = "fp8_e4m3"
        if kv_dtype not in m:
            raise ValueError(f"unsupported KV torch dtype {kv_dtype}")
        return m[kv_dtype]
    key = str(kv_dtype).lower().replace("torch.", "")
    if key not in _ALIASES:
        raise ValueError(f"unknown kv_dtype {kv_dtype!r}; expected one of {KV_DTYPES}")
    return _ALIASES[key]


def is_quantised(kv_dtype: str) -> bool:
    return normalize_kv_dtype(kv_dtype) in QUANT_KV_DTYPES


def is_block_quantised(kv_dtype: str) -> bool:
    """True for the block-scaled formats (``nvfp4`` / ``mxfp4`` / ``ternary``) with their own scale layout."""
    return normalize_kv_dtype(kv_dtype) in BLOCK_KV_DTYPES


def kv_kind_id(kv_dtype: str) -> int:
    """Integer used as a Triton ``constexpr``: 0 native float, 1 fp8, 2 int8, 3 int4, 4 nvfp4, 5 mxfp4, 6 ternary."""
    return {"bf16": 0, "fp16": 0, "fp32": 0, "fp8_e4m3": 1, "int8": 2, "int4": 3,
            "nvfp4": 4, "mxfp4": 5, "ternary": 6}[normalize_kv_dtype(kv_dtype)]


def kv_bytes_per_element(kv_dtype: str) -> float:
    return {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "fp8_e4m3": 1.0, "int8": 1.0, "int4": 0.5,
            "nvfp4": 0.5 + 1.0 / 16, "mxfp4": 0.5 + 1.0 / 32, "ternary": 0.25 + 4.0 / 32}[normalize_kv_dtype(kv_dtype)]


def pool_dtype(kv_dtype: str, compute_dtype: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Storage dtype of a payload pool for ``kv_dtype`` (block scales are separate tensors, see :func:`block_scale_dtype`)."""
    k = normalize_kv_dtype(kv_dtype)
    if k == "fp8_e4m3":
        return torch.float8_e4m3fn
    if k == "int8":
        return torch.int8
    if k in ("int4", "nvfp4", "mxfp4", "ternary"):
        return torch.uint8
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[k]


def block_scale_dtype(kv_dtype: str) -> torch.dtype:
    """Storage dtype of a block-scale tensor: E4M3 for NVFP4, raw ``uint8`` codes for MXFP4 (E8M0) and ternary (fp32->stored fp32)."""
    k = normalize_kv_dtype(kv_dtype)
    if k == "nvfp4":
        return torch.float8_e4m3fn
    if k == "mxfp4":
        return torch.uint8
    if k == "ternary":
        return torch.float32
    raise ValueError(f"{kv_dtype!r} is not a block-quantised KV format")


# --------------------------------------------------------------------------------------------------
# int4 packing
# --------------------------------------------------------------------------------------------------
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed values in ``[-8, 7]`` along the last dim (even) into bytes (offset-binary nibbles)."""
    if q.shape[-1] % 2 != 0:
        raise ValueError("int4 packing needs an even last dimension")
    u = (q.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    lo = u[..., 0::2]
    hi = u[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4`; returns ``int8`` values in ``[-8, 7]`` with last dim doubled."""
    p = packed.to(torch.uint8) if packed.dtype != torch.uint8 else packed
    lo = (p & 0x0F).to(torch.int16) - 8
    hi = (p >> 4).to(torch.int16) - 8
    out = torch.stack([lo, hi], dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    return out.to(torch.int8)


# --------------------------------------------------------------------------------------------------
# fp8 e4m3 (FN convention: no inf, S.1111.111 = NaN, max 448) bit decode - mirrors the Triton routine
# --------------------------------------------------------------------------------------------------
def decode_e4m3_bits(u8: torch.Tensor) -> torch.Tensor:
    """Decode raw E4M3FN bytes to float32 using integer arithmetic only (the exact formula the portable
    Triton path uses on GPUs without native FP8 conversion)."""
    b = u8.to(torch.int32) & 0xFF
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    man = b & 0x7
    sub = man.to(torch.float32) * (2.0 ** -9)
    nrm = (1.0 + man.to(torch.float32) / 8.0) * torch.exp2((exp - 7).to(torch.float32))
    val = torch.where(exp == 0, sub, nrm)
    val = torch.where((exp == 15) & (man == 7), torch.full_like(val, float("nan")), val)
    return torch.where(sign == 1, -val, val)


def fp8_to_uint8(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.uint8)


def uint8_to_fp8(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.float8_e4m3fn)


# --------------------------------------------------------------------------------------------------
# scale normalisation
# --------------------------------------------------------------------------------------------------
def _scalar_or_none(t):
    return t is None


def expand_k_scale(k_scale, num_pages: int, hkv: int, d: int, device, *, default: float = 1.0) -> torch.Tensor:
    """Return ``[P, Hkv, D]`` float32 view (possibly stride-0 expanded) of any accepted key-scale shape."""
    if k_scale is None:
        return torch.full((1, 1, 1), default, dtype=torch.float32, device=device).expand(num_pages, hkv, d)
    s = torch.as_tensor(k_scale, dtype=torch.float32, device=device)
    if s.numel() == 1:
        return s.reshape(1, 1, 1).expand(num_pages, hkv, d)
    if s.dim() == 2 and s.shape == (hkv, d):
        return s.unsqueeze(0).expand(num_pages, hkv, d)
    if s.dim() == 2 and s.shape == (num_pages, hkv):
        return s.unsqueeze(-1).expand(num_pages, hkv, d)
    if s.dim() == 3 and s.shape[-1] == 1:
        return s.expand(num_pages, hkv, d)
    if s.dim() == 3 and s.shape == (num_pages, hkv, d):
        return s
    raise ValueError(f"unsupported key-scale shape {tuple(s.shape)} for pool P={num_pages} Hkv={hkv} D={d}")


def expand_v_scale(v_scale, num_pages: int, hkv: int, page_size: int, device, *, default: float = 1.0) -> torch.Tensor:
    """Return ``[P, Hkv, page_size]`` float32 view of any accepted value-scale shape."""
    if v_scale is None:
        return torch.full((1, 1, 1), default, dtype=torch.float32, device=device).expand(num_pages, hkv, page_size)
    s = torch.as_tensor(v_scale, dtype=torch.float32, device=device)
    if s.numel() == 1:
        return s.reshape(1, 1, 1).expand(num_pages, hkv, page_size)
    if s.dim() == 2 and s.shape == (num_pages, hkv):
        return s.unsqueeze(-1).expand(num_pages, hkv, page_size)
    if s.dim() == 3 and s.shape[-1] == 1:
        return s.expand(num_pages, hkv, page_size)
    if s.dim() == 3 and s.shape == (num_pages, hkv, page_size):
        return s
    raise ValueError(f"unsupported value-scale shape {tuple(s.shape)} for pool P={num_pages} Hkv={hkv} ps={page_size}")


def expand_zero_like(zero, scale_expanded: torch.Tensor) -> torch.Tensor | None:
    """Zero points share the scale's accepted shapes; ``None`` stays ``None``."""
    if zero is None:
        return None
    z = torch.as_tensor(zero, dtype=torch.float32, device=scale_expanded.device)
    if z.numel() == 1:
        return z.reshape(1, 1, 1).expand_as(scale_expanded)
    if z.dim() == 3 and z.shape[-1] == 1:
        return z.expand_as(scale_expanded)
    if z.dim() == 2 and scale_expanded.dim() == 3 and z.shape == (scale_expanded.shape[1], scale_expanded.shape[2]):
        return z.unsqueeze(0).expand_as(scale_expanded)
    if z.dim() == 2 and z.shape == scale_expanded.shape[:2]:
        return z.unsqueeze(-1).expand_as(scale_expanded)
    if z.shape == scale_expanded.shape:
        return z
    raise ValueError(f"zero-point shape {tuple(z.shape)} incompatible with scale {tuple(scale_expanded.shape)}")


# --------------------------------------------------------------------------------------------------
# quantise / dequantise
# --------------------------------------------------------------------------------------------------
def _qrange(kv_dtype: str) -> tuple[float, float]:
    return {"int8": (-128.0, 127.0), "int4": (-8.0, 7.0)}[kv_dtype]


def _quantise_with(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None, kv_dtype: str):
    xf = x.float()
    if zero is not None:
        xf = xf - zero
    y = xf / scale
    if kv_dtype == "fp8_e4m3":
        return y.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    lo, hi = _qrange(kv_dtype)
    q = torch.round(y).clamp(lo, hi)
    if kv_dtype == "int8":
        return q.to(torch.int8)
    return pack_int4(q.to(torch.int8))


def quantize_keys_per_channel(k: torch.Tensor, kv_dtype: str, *, asymmetric: bool = False, eps: float = 1e-8):
    """Quantise a block of keys ``[..., T, D]`` with one scale per channel (reduced over the token axis ``-2``).

    Returns ``(q, scale[..., D], zero[..., D] | None)``. ``q`` has the pool storage dtype (``int4`` -> packed ``D/2``).
    """
    kv = normalize_kv_dtype(kv_dtype)
    kf = k.float()
    return _quant_block(kf, kv, reduce_dim=-2, asymmetric=asymmetric, eps=eps, squeeze_dim=-2)


def quantize_values_per_token(v: torch.Tensor, kv_dtype: str, *, asymmetric: bool = False, eps: float = 1e-8):
    """Quantise values ``[..., T, D]`` with one scale per token (reduced over the channel axis ``-1``).

    Returns ``(q, scale[..., T], zero[..., T] | None)``.
    """
    kv = normalize_kv_dtype(kv_dtype)
    vf = v.float()
    return _quant_block(vf, kv, reduce_dim=-1, asymmetric=asymmetric, eps=eps, squeeze_dim=-1)


def _quant_block(xf: torch.Tensor, kv: str, *, reduce_dim: int, asymmetric: bool, eps: float, squeeze_dim: int):
    zero = None
    if kv == "fp8_e4m3":
        scale = xf.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(eps) / E4M3_MAX
    elif asymmetric:
        mx = xf.amax(dim=reduce_dim, keepdim=True)
        mn = xf.amin(dim=reduce_dim, keepdim=True)
        levels = 255.0 if kv == "int8" else 15.0
        scale = ((mx - mn) / levels).clamp_min(eps)
        zero = (mx + mn) / 2.0
    else:
        qmax = 127.0 if kv == "int8" else 7.0
        scale = xf.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(eps) / qmax
    q = _quantise_with(xf, scale, zero, kv)
    scale_o = scale.squeeze(squeeze_dim)
    zero_o = zero.squeeze(squeeze_dim) if zero is not None else None
    return q, scale_o, zero_o


def dequantize_rows(raw: torch.Tensor, scale: torch.Tensor | None, zero: torch.Tensor | None, kv_dtype: str,
                    *, scale_axis: str) -> torch.Tensor:
    """Dequantise ``raw[..., T, D']`` to float32 given ``scale`` broadcastable to the result.

    ``scale_axis`` is ``"channel"`` (scale ``[..., D]`` -> unsqueeze token axis) or ``"token"`` (scale ``[..., T]``).
    """
    kv = normalize_kv_dtype(kv_dtype)
    if kv in ("bf16", "fp16", "fp32"):
        return raw.float()
    if kv == "fp8_e4m3":
        x = (raw if raw.dtype == torch.float8_e4m3fn else raw.view(torch.float8_e4m3fn)).float()
    elif kv == "int8":
        x = raw.float()
    else:
        x = unpack_int4(raw).float()
    if scale is not None:
        s = scale.unsqueeze(-2) if scale_axis == "channel" else scale.unsqueeze(-1)
        x = x * s
    if zero is not None:
        z = zero.unsqueeze(-2) if scale_axis == "channel" else zero.unsqueeze(-1)
        x = x + z
    return x


def dequantize_pool(pool: torch.Tensor, scale, zero, kv_dtype: str, kind: str) -> torch.Tensor:
    """Dequantise a whole pool ``[P, Hkv, ps, D']`` to float32 ``[P, Hkv, ps, D]`` (``kind``: ``"k"`` or ``"v"``)."""
    kvn = normalize_kv_dtype(kv_dtype)
    p, h, ps, dd = pool.shape
    d = dd * 2 if kvn == "int4" else dd
    if kvn in ("bf16", "fp16", "fp32"):
        return pool.float()
    if kind == "k":
        sc = expand_k_scale(scale, p, h, d, pool.device)
        ze = expand_zero_like(zero, sc)
        return dequantize_rows(pool, sc, ze, kvn, scale_axis="channel")
    sc = expand_v_scale(scale, p, h, ps, pool.device)
    ze = expand_zero_like(zero, sc)
    return dequantize_rows(pool, sc, ze, kvn, scale_axis="token")


def logical_head_dim(pool: torch.Tensor, kv_dtype: str) -> int:
    k = normalize_kv_dtype(kv_dtype)
    if k == "int4":
        return pool.shape[-1] * 2
    if k in ("nvfp4", "mxfp4"):
        return pool.shape[-1] * 2
    if k == "ternary":
        return pool.shape[-1] * 4
    return pool.shape[-1]


# --------------------------------------------------------------------------------------------------
# page building helpers (used by tests, benchmarks and the residual-window flush)
# --------------------------------------------------------------------------------------------------
def quantize_kv_page(k_page: torch.Tensor, v_page: torch.Tensor, kv_dtype: str, *, asymmetric: bool = False):
    """Flush one full-precision page pair ``[Hkv, ps, D]`` into the quantised pool format.

    Returns ``dict(k=..., v=..., k_scale=[Hkv,D], v_scale=[Hkv,ps], k_zero, v_zero)``. This is the residual-window flush
    of the KIVI recipe: key scales are computed over the page's tokens (per channel), value scales per token.
    """
    kq, ks, kz = quantize_keys_per_channel(k_page, kv_dtype, asymmetric=asymmetric)
    vq, vs, vz = quantize_values_per_token(v_page, kv_dtype, asymmetric=asymmetric)
    return {"k": kq, "v": vq, "k_scale": ks, "v_scale": vs, "k_zero": kz, "v_zero": vz}


def build_paged_pools(k: torch.Tensor, v: torch.Tensor, seq_lens, page_size: int, kv_dtype: str = "bf16",
                      *, asymmetric: bool = False, shuffle: bool = True, generator: torch.Generator | None = None,
                      extra_pages: int = 2):
    """Build pools + block tables from dense ``k, v`` of shape ``[B, Hkv, Lmax, D]`` (testing/bench helper).

    Pages are assigned to a random permutation of physical page ids when ``shuffle`` (so tests catch indexing
    mistakes). Returns a dict with ``k_pool, v_pool, block_tables, seq_lens, k_scale, v_scale, k_zero, v_zero``.
    For the block-scaled formats (``nvfp4`` / ``mxfp4`` / ``ternary``) the scales come back as the block tensors
    ``k_block_scale [P, Hkv, D/BS]`` / ``v_block_scale [P, Hkv, ps, D/BS]`` (plus ``tensor_scale`` for NVFP4) and
    ``k_scale`` / ``v_scale`` are ``None``.
    """
    kvn = normalize_kv_dtype(kv_dtype)
    b, h, lmax, d = k.shape
    seq_lens_t = torch.as_tensor(seq_lens, dtype=torch.int32)
    max_pages = (lmax + page_size - 1) // page_size
    total = b * max_pages + extra_pages
    perm = torch.randperm(total, generator=generator) if shuffle else torch.arange(total)
    block_tables = torch.full((b, max_pages), 0, dtype=torch.int32)
    if kvn in BLOCK_KV_DTYPES:
        return _build_block_paged_pools(k, v, seq_lens_t, page_size, kvn, perm, block_tables, total)
    stored_d = d // 2 if kvn == "int4" else d
    pdt = pool_dtype(kvn, k.dtype)
    if kvn in ("bf16", "fp16", "fp32"):
        pdt = k.dtype if k.dtype in (torch.bfloat16, torch.float16, torch.float32) else pdt
    k_pool = torch.zeros(total, h, page_size, stored_d, dtype=pdt)
    v_pool = torch.zeros(total, h, page_size, stored_d, dtype=pdt)
    k_scale = v_scale = k_zero = v_zero = None
    if kvn != "bf16" and kvn in QUANT_KV_DTYPES:
        k_scale = torch.ones(total, h, d)
        v_scale = torch.ones(total, h, page_size)
        if asymmetric and kvn != "fp8_e4m3":
            k_zero = torch.zeros(total, h, d)
            v_zero = torch.zeros(total, h, page_size)
    nxt = 0
    for bi in range(b):
        n_pages = (int(seq_lens_t[bi]) + page_size - 1) // page_size
        for pi in range(n_pages):
            phys = int(perm[nxt])
            nxt += 1
            block_tables[bi, pi] = phys
            lo, hi = pi * page_size, min((pi + 1) * page_size, int(seq_lens_t[bi]))
            kp = torch.zeros(h, page_size, d, dtype=k.dtype)
            vp = torch.zeros(h, page_size, d, dtype=v.dtype)
            kp[:, : hi - lo] = k[bi, :, lo:hi]
            vp[:, : hi - lo] = v[bi, :, lo:hi]
            if kvn in QUANT_KV_DTYPES:
                # padding rows of a partial page are zero and do not influence per-channel absmax
                pg = quantize_kv_page(kp, vp, kvn, asymmetric=asymmetric)
                k_pool[phys] = pg["k"]
                v_pool[phys] = pg["v"]
                k_scale[phys] = pg["k_scale"]
                v_scale[phys] = pg["v_scale"]
                if k_zero is not None:
                    k_zero[phys] = pg["k_zero"]
                    v_zero[phys] = pg["v_zero"]
            else:
                k_pool[phys] = kp.to(pdt)
                v_pool[phys] = vp.to(pdt)
    return {"k_pool": k_pool, "v_pool": v_pool, "block_tables": block_tables, "seq_lens": seq_lens_t,
            "k_scale": k_scale, "v_scale": v_scale, "k_zero": k_zero, "v_zero": v_zero, "kv_dtype": kvn}


def _build_block_paged_pools(k, v, seq_lens_t, page_size, kvn, perm, block_tables, total):
    b, h, lmax, d = k.shape
    bs = KV_BLOCK_SIZE[kvn]
    if d % bs:
        raise ValueError(f"{kvn} KV needs a head dim that is a multiple of {bs}, got D={d}")
    nblk = d // bs
    k_pool = torch.zeros(total, h, page_size, d // _KV_PACK[kvn], dtype=torch.uint8)
    v_pool = torch.zeros(total, h, page_size, d // _KV_PACK[kvn], dtype=torch.uint8)
    k_bs = torch.zeros(total, h, nblk, dtype=block_scale_dtype(kvn))
    v_bs = torch.zeros(total, h, page_size, nblk, dtype=block_scale_dtype(kvn))
    tensor_scale = torch.ones((), dtype=torch.float32)
    nxt = 0
    for bi in range(b):
        n_pages = (int(seq_lens_t[bi]) + page_size - 1) // page_size
        for pi in range(n_pages):
            phys = int(perm[nxt])
            nxt += 1
            block_tables[bi, pi] = phys
            lo, hi = pi * page_size, min((pi + 1) * page_size, int(seq_lens_t[bi]))
            kp = torch.zeros(h, page_size, d, dtype=k.dtype)
            vp = torch.zeros(h, page_size, d, dtype=v.dtype)
            kp[:, : hi - lo] = k[bi, :, lo:hi]
            vp[:, : hi - lo] = v[bi, :, lo:hi]
            pg = quantize_kv_page_block(kp, vp, kvn)
            k_pool[phys] = pg["k"]
            v_pool[phys] = pg["v"]
            k_bs[phys] = pg["k_block_scale"]
            v_bs[phys] = pg["v_block_scale"]
            if kvn == "nvfp4":
                tensor_scale = pg["tensor_scale"]
    return {"k_pool": k_pool, "v_pool": v_pool, "block_tables": block_tables, "seq_lens": seq_lens_t,
            "k_scale": None, "v_scale": None, "k_zero": None, "v_zero": None,
            "k_block_scale": k_bs, "v_block_scale": v_bs, "tensor_scale": tensor_scale, "kv_dtype": kvn}


# --------------------------------------------------------------------------------------------------
# block-scaled formats: E2M1 / ternary element codecs, block quantise / dequantise
# --------------------------------------------------------------------------------------------------
def decode_e2m1_codes(c: torch.Tensor) -> torch.Tensor:
    """E2M1 4-bit code (0..15, bit 3 = sign) -> float32 (table lookup; mirrors the Triton ``_e2m1_decode``).

    Magnitude code ``m = c & 7`` -> ``{0, .5, 1, 1.5, 2, 3, 4, 6}``; bit 3 negates.  Code ``8`` decodes to ``0``
    (canonical, never ``-0``).
    """
    m = (c.to(torch.int32) & 7)
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=c.device)
    val = mag[m.long()]
    return torch.where(((c.to(torch.int32) >> 3) & 1) == 1, -val, val)


def decode_ternary_codes(c: torch.Tensor) -> torch.Tensor:
    """2-bit ternary code -> float32 in ``{-1, 0, +1}`` (``00``->0, ``01``->+1, ``11``->-1, ``10``->0)."""
    c = c.to(torch.int32) & 3
    return torch.where((c & 1) == 1, torch.where((c & 2) == 2, -1.0, 1.0), 0.0)


def decode_e4m3_scales(bits: torch.Tensor) -> torch.Tensor:
    """E4M3FN scale bytes (``uint8`` / ``float8_e4m3fn``) -> float32 (NaN code ``0x7F`` stays NaN, like the oracle)."""
    b = bits.view(torch.uint8) if bits.dtype != torch.uint8 else bits
    return b.view(torch.float8_e4m3fn).float() if b.numel() else b.float()


def decode_e8m0_scales(codes: torch.Tensor) -> torch.Tensor:
    """E8M0 scale code (``uint8``, ``scale = 2**(code - 127)``; code 0 -> smallest normal) -> float32."""
    return torch.exp2(codes.to(torch.float32) - 127.0)


def pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Two E2M1 codes per byte along the last dim (even element = low nibble)."""
    from forge.formats.packing import pack_uint4

    return pack_uint4(codes)


def pack_ternary_codes(t: torch.Tensor) -> torch.Tensor:
    """Signed values in ``{-1, 0, +1}`` ``[..., D]`` -> 2-bit codes packed four per byte along the last dim."""
    from forge.formats.packing import pack_bits

    return pack_bits((t.to(torch.int16) & 3).to(torch.uint8), 2)


def _block_scale_decode(scale: torch.Tensor, kvn: str, tensor_scale: torch.Tensor | None) -> torch.Tensor:
    if kvn == "nvfp4":
        s = decode_e4m3_scales(scale)
        if tensor_scale is not None:
            s = s * torch.as_tensor(tensor_scale, dtype=torch.float32, device=s.device)
        return s
    if kvn == "mxfp4":
        return decode_e8m0_scales(scale)
    return scale.float()  # ternary: fp32 scale


def e8m0_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 exponent for an ``amax`` (unbiased power-of-two exponent, ``scale = 2**exponent``).

    Uses ``frexp`` (not ``log2``) so a block maximum just above a power of two is not rounded down onto it - the FA4
    ``blockscaled_utils`` trick.  ``ceil(log2(amax))`` without a ``log2`` boundary error.  All-zero blocks get exponent
    0 (scale 1).  Returns an int32 tensor of the *unbiased* exponent clamped to the E8M0 range ``[-127, 127]``.
    """
    mantissa, exponent = torch.frexp(amax.float())
    e = exponent - 1 + (mantissa > 0.5).to(torch.int32)
    return torch.where(amax == 0, torch.zeros_like(e), e.clamp(-127, 127))


def quantize_block(x: torch.Tensor, kvn: str, *, tensor_scale: torch.Tensor | None = None,
                   block_scale: torch.Tensor | None = None, per_channel: bool = False):
    """Quantise ``x [..., T, D]`` to a block-scaled format.

    Returns ``(payload uint8 [..., T, D/PACK], block_scale, tensor_scale scalar | None)`` where ``block_scale`` is
    ``[..., T, D/BS]`` normally or ``[..., D/BS]`` when ``per_channel`` (one scale per channel block over the token axis -
    the KIVI "key" convention; values use per-token scales, the default).

    * ``nvfp4``: per-16-channel E4M3 block scale of ``amax/6`` divided by the tensor scale (default: the global amax of
      ``x`` over ``448*6``, the oracle convention of :mod:`forge.kernels.ref.gemm_fp4`).
    * ``mxfp4``: per-32-channel E8M0 power-of-two scale ``2**ceil(log2(amax/6))``.
    * ``ternary``: per-32-channel fp32 scale ``amax`` with elements ``sign(x)`` where ``|x| >= amax/2`` else 0.
    """
    bs = KV_BLOCK_SIZE[kvn]
    xf = x.float()
    d = xf.shape[-1]
    if d % bs:
        raise ValueError(f"{kvn} block size {bs} must divide D={d}")
    xb = xf.reshape(*xf.shape[:-1], -1, bs)  # [..., T, NB, bs]
    nblk = xb.shape[-2]
    if per_channel:  # reduce over tokens (dim -3) and within the block (dim -1) -> [..., NB, 1]
        amax = xb.abs().amax(dim=(-3, -1), keepdim=False).unsqueeze(-1)
        scale_shape = xf.shape[:-1] + (nblk,)  # [..., NB]
    else:  # reduce within the block (dim -1) -> [..., T, NB, 1]
        amax = xb.abs().amax(dim=-1, keepdim=True)
        scale_shape = xb.shape[:-1] + (1,)  # [..., T, NB, 1]
    if kvn == "nvfp4":
        g = (xf.abs().amax().clamp_min(1e-30) / (E2M1_MAX * E4M3_MAX)) if tensor_scale is None else \
            torch.as_tensor(tensor_scale, dtype=torch.float32, device=xf.device).clamp_min(1e-30)
        stored = (block_scale.to(torch.float8_e4m3fn).reshape(scale_shape) if block_scale is not None
                  else (amax / E2M1_MAX / g).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).expand_as(amax))
        sbc = _broadcast_scale(stored.float() * g, per_channel)  # [..., T, NB, 1]
        ok = sbc > 0
        scaled = torch.where(ok, xb / torch.where(ok, sbc, torch.ones_like(sbc)), torch.zeros_like(xb))
        return pack_e2m1_codes(_e2m1_encode(scaled).reshape(*xf.shape)), _scale_out(stored), g.reshape(())
    if kvn == "mxfp4":
        # E8M0 scale = 2**ceil(log2(amax / 6)) (never saturates: NVIDIA/OCP "ceil" advice); frexp avoids a log2 boundary
        e = e8m0_exponent(amax / E2M1_MAX)
        code = (e + 127).clamp(0, 254).to(torch.uint8)
        sdec = _broadcast_scale(torch.exp2(code.float() - 127.0), per_channel)
        scaled = torch.where(sdec > 0, xb / sdec, torch.zeros_like(xb))
        return pack_e2m1_codes(_e2m1_encode(scaled).reshape(*xf.shape)), _scale_out(code), None
    # ternary: absmean rule with the block maximum (alpha = amax/2 threshold, elements >= alpha keep their sign)
    s = (block_scale.to(torch.float32).reshape(scale_shape) if block_scale is not None
         else amax.clamp_min(1e-30).expand_as(amax))
    sbc = _broadcast_scale(s, per_channel)
    t = torch.where(xb.abs() >= 0.5 * sbc, torch.sign(xb), torch.zeros_like(xb))
    return pack_ternary_codes((t.to(torch.int16) & 3).reshape(*xf.shape)), _scale_out(s), None


def _broadcast_scale(scale: torch.Tensor, per_channel: bool) -> torch.Tensor:
    """Per-channel scale ``[..., NB, 1]`` -> ``[..., 1, NB, 1]`` (insert a token axis) so it broadcasts over
    ``[..., T, NB, bs]``; per-token scales are already ``[..., T, NB, 1]``."""
    return scale.unsqueeze(-3) if per_channel else scale


def _scale_out(scale: torch.Tensor) -> torch.Tensor:
    """Drop the trailing block-size axis: ``[..., T, NB, 1]`` -> ``[..., T, NB]`` / ``[..., NB, 1]`` -> ``[..., NB]``."""
    return scale.squeeze(-1).contiguous() if scale.dim() >= 2 and scale.shape[-1] == 1 else scale.contiguous()


def _e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Scaled values -> E2M1 codes (round-to-nearest-even, saturating at 6); same tie rules as ``formats.fp4``."""
    a = x.abs().clamp(max=E2M1_MAX)
    idx = ((a > 0.25).to(torch.int32) + (a >= 0.75).to(torch.int32) + (a > 1.25).to(torch.int32)
           + (a >= 1.75).to(torch.int32) + (a > 2.5).to(torch.int32) + (a >= 3.5).to(torch.int32) + (a > 5.0).to(torch.int32))
    sign = ((x < 0) & (idx > 0)).to(torch.int32)
    return (idx | (sign << 3)).to(torch.uint8)


def dequantize_block_pool(pool: torch.Tensor, block_scale: torch.Tensor, kv_dtype: str, kind: str,
                          tensor_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Whole block-scaled pool ``[P, Hkv, T, D']`` -> float32 ``[P, Hkv, T, D]`` (``kind``: ``"k"`` or ``"v"``)."""
    kvn = normalize_kv_dtype(kv_dtype)
    bs = KV_BLOCK_SIZE[kvn]
    if kvn == "ternary":
        from forge.formats.packing import unpack_bits

        d_pack = 4
        codes = unpack_bits(pool, 2).to(torch.int32)
        vals = decode_ternary_codes(codes)
    else:
        d_pack = 2
        from forge.formats.packing import unpack_uint4

        vals = decode_e2m1_codes(unpack_uint4(pool))
    vals = vals.reshape(*vals.shape[:-1], -1, bs)  # [..., T, NB, bs]
    s = _block_scale_decode(block_scale, kvn, tensor_scale)
    out = vals * _align_block_scale(s, vals, kind).unsqueeze(-1)
    return out.reshape(*pool.shape[:-1], pool.shape[-1] * d_pack)


def _align_block_scale(s: torch.Tensor, vals: torch.Tensor, kind: str) -> torch.Tensor:
    """Broadcast a block scale to the ``[..., T, NB]`` prefix of ``vals [..., T, NB, bs]``.

    Keys carry per-channel scales ``[..., NB]`` (broadcast over the token axis); values carry per-token-block scales
    ``[..., T, NB]`` (already the target shape).
    """
    target = tuple(vals.shape[:-1])  # (..., T, NB)
    if tuple(s.shape) == target:
        return s
    if kind == "k":
        return s.unsqueeze(-2).expand(target)
    return s.expand(target)


def dequantize_block_rows(raw: torch.Tensor, block_scale: torch.Tensor, kv_dtype: str, *, n_channels: int,
                          tensor_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantise block-scaled rows ``raw[..., T, D/PACK]`` + scales ``[..., T, D/BS]`` to float32 ``[..., T, D]``."""
    kvn = normalize_kv_dtype(kv_dtype)
    bs = KV_BLOCK_SIZE[kvn]
    from forge.formats.packing import unpack_bits, unpack_uint4

    if kvn == "ternary":
        vals = decode_ternary_codes(unpack_bits(raw, 2, n_channels))
    else:
        vals = decode_e2m1_codes(unpack_uint4(raw, n_channels))
    vb = vals.reshape(*vals.shape[:-1], -1, bs)
    s = _block_scale_decode(block_scale, kvn, tensor_scale)
    return (vb * s.unsqueeze(-1)).reshape(*vals.shape)


def quantize_kv_page_block(k_page: torch.Tensor, v_page: torch.Tensor, kv_dtype: str):
    """Flush one full-precision page pair ``[Hkv, ps, D]`` into a block-scaled format (residual-window flush).

    Keys get per-channel block scales ``[Hkv, D/BS]`` (reduced over the page's tokens), values per-token scales
    ``[Hkv, ps]`` (reduced over the channels of each block) - the KIVI convention.  For NVFP4 the tensor scale is
    shared by the K and V quantisation of the page.  Returns ``dict(k, v, k_block_scale, v_block_scale, tensor_scale)``.
    """
    kvn = normalize_kv_dtype(kv_dtype)
    if kvn not in BLOCK_KV_DTYPES:
        raise ValueError(f"{kv_dtype!r} is not block-quantised")
    if kvn == "nvfp4":
        amax = torch.cat([k_page.float().abs().reshape(-1), v_page.float().abs().reshape(-1)]).amax()
        ts = (amax / (E2M1_MAX * E4M3_MAX)).clamp_min(1e-30).reshape(())
        kq, ks, _ = quantize_block(k_page, kvn, tensor_scale=ts, per_channel=True)
        vq, vs, _ = quantize_block(v_page, kvn, tensor_scale=ts)
        return {"k": kq, "v": vq, "k_block_scale": ks, "v_block_scale": vs, "tensor_scale": ts}
    kq, ks, _ = quantize_block(k_page, kvn, per_channel=True)
    vq, vs, _ = quantize_block(v_page, kvn)
    return {"k": kq, "v": vq, "k_block_scale": ks, "v_block_scale": vs, "tensor_scale": None}
