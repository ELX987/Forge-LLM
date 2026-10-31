"""Oracles for block-scaled FP4 (K2): E2M1 codec, NVFP4 / MXFP4 quantisers, dequantiser and GEMM (fp32 math).

Formats (Plan 6.4 K2, paper P5 sections 3.3 / 11.2)::

    NVFP4 : 16 x E2M1 + one E4M3 block scale (+ fp32 tensor scale).  tensor_scale = amax_tensor / (448 * 6)
            block scale (stored) = E4M3_rne( (amax_block / 6) / tensor_scale ),   value = e2m1 * scale * tensor_scale
    MXFP4 : 32 x E2M1 + one E8M0 (power-of-two) scale = 2**(floor(log2 amax_block) - 2)   (OCP MX v1.0)

E2M1 magnitudes (code 0..7): ``0, 0.5, 1, 1.5, 2, 3, 4, 6``; bit 3 = sign.  Rounding is round-to-nearest-even
(ties go to the neighbour with an even code), saturating at 6.

Conventions (documented edge cases, all CPU-tested):

* **All-zero block** -> scale 0 (NVFP4) / 2**0 (MXFP4), codes 0, dequantises to exactly 0.
* **E4M3 scale underflow**: a block whose scale rounds to 0 in E4M3 (block max < ~2.2e-6 x tensor max... P5 key
  finding 4) is *flushed*: codes 0 and dequantises to 0 (never 0/0 = NaN).
* **NaN / Inf** inside a block *poison* the block: scale = NaN (0x7F for E4M3, 0xFF for E8M0), codes 0, so every
  element of the block dequantises to NaN.  FP4 has no NaN/Inf encodings; the poison is carried by the scale.
  (The Triton fused quantiser assumes finite inputs and does not implement this convention.)
"""
from __future__ import annotations

import torch

from forge.kernels.layouts import PackedFP4, pack_e2m1, roundup, swizzle_scales, unpack_e2m1
from forge.kernels.layouts.layout import FP4_BLOCK

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
E2M1_MAX = 6.0
_E4M3_NAN = 0x7F
_E8M0_NAN = 0xFF

_LUT_CACHE: dict[str, torch.Tensor] = {}


def _lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _LUT_CACHE.get(key)
    if t is None:
        pos = torch.tensor(E2M1_GRID, dtype=torch.float32)
        t = torch.cat([pos, -pos]).to(device)  # code 8 = -0.0 (value -0.0 == 0.0)
        _LUT_CACHE[key] = t
    return t


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """``uint8`` codes 0..15 -> fp32 values."""
    return _lut(codes.device)[codes.long()]


def e2m1_encode(y: torch.Tensor) -> torch.Tensor:
    """fp32 values (already divided by the block scale) -> ``uint8`` codes, RNE, saturating at +-6.

    Thresholds sit at the midpoints ``0.25 0.75 1.25 1.75 2.5 3.5 5``; exact ties resolve to the even code
    (0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5->4.0).
    """
    a = y.abs()
    code = (
        (a > 0.25).to(torch.uint8)
        + (a >= 0.75).to(torch.uint8)
        + (a > 1.25).to(torch.uint8)
        + (a >= 1.75).to(torch.uint8)
        + (a > 2.5).to(torch.uint8)
        + (a >= 3.5).to(torch.uint8)
        + (a > 5.0).to(torch.uint8)
    )
    return code | ((y < 0).to(torch.uint8) << 3)


def e4m3_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    return bits.contiguous().view(torch.float8_e4m3fn).float()


def float_to_e4m3_bits(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> E4M3FN bit pattern (RNE, saturating at 448, NaN stays NaN)."""
    xc = torch.where(torch.isnan(x), x, x.clamp(-E4M3_MAX, E4M3_MAX))
    return xc.to(torch.float8_e4m3fn).view(torch.uint8)


def e8m0_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    f = torch.ldexp(torch.ones_like(bits, dtype=torch.float32), bits.to(torch.int32) - 127)
    return torch.where(bits == _E8M0_NAN, torch.full_like(f, float("nan")), f)


def _prep(x: torch.Tensor, block: int) -> tuple[torch.Tensor, int, int, int]:
    x2 = x.reshape(-1, x.shape[-1]).float()
    rows, k = x2.shape
    kp = roundup(k, block)
    if kp != k:
        x2 = torch.nn.functional.pad(x2, (0, kp - k))
    return x2, rows, k, kp


def quantize_nvfp4_ref(
    x: torch.Tensor, tensor_scale: float | torch.Tensor | None = None, *, scale_layout: str = "logical"
) -> PackedFP4:
    """RTN NVFP4 quantisation of ``x [..., K]`` (leading dims flattened to rows).  ``tensor_scale`` overrides the
    dynamic ``amax/(448*6)`` (e.g. a static calibrated or rank-synchronised global scale)."""
    xp, rows, k, kp = _prep(x, 16)
    fin = torch.isfinite(xp)
    xa = torch.where(fin, xp.abs(), torch.zeros_like(xp))
    if tensor_scale is None:
        ts = (xa.amax() if xa.numel() else torch.zeros((), device=xa.device)) / (E4M3_MAX * E2M1_MAX)
    else:
        ts = torch.as_tensor(tensor_scale, dtype=torch.float32, device=xp.device).reshape(())
    ts = torch.where(ts > 0, ts, torch.ones_like(ts))
    xb = xp.view(rows, kp // 16, 16)
    amax_b = xa.view(rows, kp // 16, 16).amax(dim=-1)
    sb_bits = float_to_e4m3_bits((amax_b / E2M1_MAX) / ts)
    bad = ~fin.view(rows, kp // 16, 16).all(dim=-1)
    sb_bits = torch.where(bad, torch.full_like(sb_bits, _E4M3_NAN), sb_bits)
    scale_dec = e4m3_bits_to_float(sb_bits) * ts  # NaN for poisoned blocks
    usable = (scale_dec > 0) & ~bad
    y = torch.where(usable.unsqueeze(-1), xb / torch.where(usable, scale_dec, torch.ones_like(scale_dec)).unsqueeze(-1), torch.zeros_like(xb))
    codes = e2m1_encode(y).reshape(rows, kp)
    p = PackedFP4("nvfp4", pack_e2m1(codes), sb_bits.contiguous(), ts.to(torch.float32).reshape(()).clone(), rows, k, "logical")
    return p.with_scale_layout(scale_layout)


def quantize_mxfp4_ref(x: torch.Tensor, *, scale_layout: str = "logical") -> PackedFP4:
    """RTN MXFP4 (OCP MX): block 32, E8M0 scale ``2**(floor(log2(amax)) - 2)``, elements saturate at 6."""
    xp, rows, k, kp = _prep(x, 32)
    fin = torch.isfinite(xp)
    xa = torch.where(fin, xp.abs(), torch.zeros_like(xp))
    amax_b = xa.view(rows, kp // 32, 32).amax(dim=-1)
    _, e = torch.frexp(amax_b)  # amax = m * 2**e, m in [0.5, 1)  =>  floor(log2 amax) = e - 1
    code = torch.where(amax_b > 0, (e - 1 - 2 + 127).clamp(0, 254), torch.full_like(e, 127)).to(torch.uint8)
    bad = ~fin.view(rows, kp // 32, 32).all(dim=-1)
    code = torch.where(bad, torch.full_like(code, _E8M0_NAN), code)
    scale = e8m0_bits_to_float(code)
    y = torch.where(bad.unsqueeze(-1), torch.zeros_like(xp.view(rows, -1, 32)), xp.view(rows, kp // 32, 32) / scale.unsqueeze(-1))
    codes = e2m1_encode(y).reshape(rows, kp)
    p = PackedFP4("mxfp4", pack_e2m1(codes), code.contiguous(), torch.ones((), dtype=torch.float32, device=xp.device), rows, k, "logical")
    return p.with_scale_layout(scale_layout)


def quantize_fp4_ref(x: torch.Tensor, fmt: str = "nvfp4", **kw) -> PackedFP4:
    if fmt == "nvfp4":
        return quantize_nvfp4_ref(x, **kw)
    if fmt == "mxfp4":
        return quantize_mxfp4_ref(x, **kw)
    raise ValueError(f"unknown fp4 format {fmt!r}")


def block_scale_values(p: PackedFP4) -> torch.Tensor:
    """fp32 per-block scale ``[rows, n_blocks]`` (without the tensor scale)."""
    bits = p.logical_scales()
    return e4m3_bits_to_float(bits) if p.fmt == "nvfp4" else e8m0_bits_to_float(bits)


def dequantize_fp4_ref(p: PackedFP4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dense ``[rows, K]`` tensor: ``e2m1 * block_scale * tensor_scale`` (fp32 math, then cast)."""
    codes = unpack_e2m1(p.payload)  # [rows, kp]
    vals = e2m1_decode(codes)
    sc = block_scale_values(p).repeat_interleave(p.block, dim=1)
    out = vals * sc * p.tensor_scale.float()
    return out[:, : p.k].to(dtype)


def fp4_block_gemm_ref(
    x: torch.Tensor | PackedFP4,
    w: PackedFP4,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """``Y = X @ dq(W)^T + bias`` in fp32.  ``x`` may be a float tensor (weight-only) or a :class:`PackedFP4`
    (W4A4: both operands dequantised exactly, as the block-scaled MMA computes them)."""
    xd = dequantize_fp4_ref(x) if isinstance(x, PackedFP4) else x.reshape(-1, x.shape[-1]).float()
    lead = (xd.shape[0],) if isinstance(x, PackedFP4) else x.shape[:-1]
    y = xd @ dequantize_fp4_ref(w).t()
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*lead, w.rows).to(out_dtype or torch.float32)


def fp4_w4a4_gemm_ref(x: torch.Tensor, w: PackedFP4, bias: torch.Tensor | None = None, *, act_tensor_scale=None) -> torch.Tensor:
    """Quantise activations with the weight's format (RTN) then multiply: the end-to-end W4A4 oracle."""
    xq = quantize_fp4_ref(x, w.fmt, **({"tensor_scale": act_tensor_scale} if (act_tensor_scale is not None and w.fmt == "nvfp4") else {}))
    return fp4_block_gemm_ref(xq, w, bias)


__all__ = [
    "E2M1_GRID", "e2m1_decode", "e2m1_encode", "quantize_nvfp4_ref", "quantize_mxfp4_ref", "quantize_fp4_ref",
    "dequantize_fp4_ref", "fp4_block_gemm_ref", "fp4_w4a4_gemm_ref", "block_scale_values",
    "e4m3_bits_to_float", "float_to_e4m3_bits", "e8m0_bits_to_float", "FP4_BLOCK", "swizzle_scales",
]
