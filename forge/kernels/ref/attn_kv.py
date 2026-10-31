"""KV-cache numerics shared by the attention oracles, the Triton kernels and the tests.

Page layout (Plan K4, head-major): ``pool[num_pages, Hkv, page_size, D]``.

Quantised KV storage (KIVI-style, Plan K4 / P1 F8):

* ``fp8_e4m3`` - ``torch.float8_e4m3fn`` (or its ``uint8`` bit view), scale optional.
* ``int8``     - ``torch.int8``, symmetric (+ optional additive zero point).
* ``int4``     - ``torch.uint8`` with two offset-binary nibbles per byte along ``D``
  (``byte = (q[2i] + 8) | ((q[2i+1] + 8) << 4)``, values in ``[-8, 7]``), pool last dim is ``D // 2``.

Scales (all float32, accepted shapes are normalised by :func:`expand_k_scale` / :func:`expand_v_scale`):

* keys   : per page, per KV head, **per channel** ``[P, Hkv, D]`` (also ``[P, Hkv, 1]``, ``[Hkv, D]``, scalar, ``None``)
* values : per page, per KV head, **per token**   ``[P, Hkv, page_size]`` (also ``[P, Hkv, 1]``, scalar, ``None``)

Dequantisation is ``x = q * scale (+ zero)``; ``zero`` has the same shape as the scale and is optional.
"""
from __future__ import annotations

import torch

KV_DTYPES = ("bf16", "fp16", "fp32", "fp8_e4m3", "int8", "int4")
QUANT_KV_DTYPES = ("fp8_e4m3", "int8", "int4")
E4M3_MAX = 448.0

_ALIASES = {
    "bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16", "half": "fp16",
    "float32": "fp32", "fp32": "fp32", "float": "fp32",
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "e4m3": "fp8_e4m3", "fp8e4m3": "fp8_e4m3",
    "int8": "int8", "int4": "int4",
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


def kv_kind_id(kv_dtype: str) -> int:
    """Integer used as a Triton ``constexpr``: 0 native float, 1 fp8, 2 int8, 3 int4."""
    return {"bf16": 0, "fp16": 0, "fp32": 0, "fp8_e4m3": 1, "int8": 2, "int4": 3}[normalize_kv_dtype(kv_dtype)]


def kv_bytes_per_element(kv_dtype: str) -> float:
    return {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "fp8_e4m3": 1.0, "int8": 1.0, "int4": 0.5}[normalize_kv_dtype(kv_dtype)]


def pool_dtype(kv_dtype: str, compute_dtype: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Storage dtype of a pool for ``kv_dtype``."""
    k = normalize_kv_dtype(kv_dtype)
    if k == "fp8_e4m3":
        return torch.float8_e4m3fn
    if k == "int8":
        return torch.int8
    if k == "int4":
        return torch.uint8
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[k]


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
    return pool.shape[-1] * (2 if normalize_kv_dtype(kv_dtype) == "int4" else 1)


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
    """
    kvn = normalize_kv_dtype(kv_dtype)
    b, h, lmax, d = k.shape
    seq_lens_t = torch.as_tensor(seq_lens, dtype=torch.int32)
    max_pages = (lmax + page_size - 1) // page_size
    total = b * max_pages + extra_pages
    perm = torch.randperm(total, generator=generator) if shuffle else torch.arange(total)
    block_tables = torch.full((b, max_pages), 0, dtype=torch.int32)
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
            phys = int(perm[nxt]); nxt += 1
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
