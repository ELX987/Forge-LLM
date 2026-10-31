"""E2M1 / NVFP4 / MXFP4 numerics (Plan CP-5/CP-6; NVFP4 report ch. 3-5 and App. B).

Element grid (E2M1, 1-2-1, bias 1)::

    code (S|EM)  000  001  010  011  100  101  110  111
    magnitude     0   0.5   1   1.5   2    3    4    6

Rounding
--------
* **RTN-even**: round to nearest, ties to the neighbour whose 3-bit magnitude code is even (ties 0.25->0, 0.75->1, 1.25->1,
  1.75->2, 2.5->2, 3.5->4, 5.0->4), saturating at +-6.  Identical to the reference ``rtn_fp4`` of the NVFP4 report (App. B).
* **Stochastic** (SR): round up with probability ``(|x|-lo)/(hi-lo)``; unbiased on ``[-6, 6]``.  Uniforms come from the
  counter-based RNG (:mod:`forge.formats.rng`), so SR is reproducible from ``(seed, offset)``.

NVFP4 (block 16, E4M3 block scale, FP32 tensor scale)::

    s_enc = 6*448 / amax(X)            s_dec = 1 / s_enc = amax(X)/(6*448)           (stored as ``tensor_scale``)
    stored_b = e4m3( amax_b/6 * s_enc )                                              (stored as ``block_scales_e4m3``)
    q_i = RTN( x_i / (stored_b * s_dec) )          x^_i = q_i * stored_b * s_dec

MXFP4 (OCP, block 32, E8M0 power-of-two scale): ``scale_mode`` = ``ceil`` (never saturates; NVIDIA's advice), ``floor`` (OCP
reference conversion ``2**(floor(log2 amax) - 2)``) or ``near``.

Packed layout (portable "simple" layout; kernels re-pack with ``forge.kernels.layouts``): two E2M1 codes per byte, even
element in the low nibble, packed along the last dim; ``shape`` is the original (unpadded) shape; a last dim that is not a
multiple of the block size is zero-padded for quantisation.

These are *fake-quant reference numerics*: they run on CPU and CUDA with plain torch ops and do not use FP4 hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from forge.core.errors import ConfigError
from forge.formats.fp8 import round_e4m3
from forge.formats.packing import pack_uint4, unpack_uint4
from forge.formats.rng import counter_uniform

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
#: bits per element incl. amortised scales: NVFP4 4+8/16, MXFP4 4+8/32
NVFP4_BITS_PER_ELEMENT = 4.5
MXFP4_BITS_PER_ELEMENT = 4.25


# ------------------------------------------------------------------------------------------------ element rounding
def _rtn_index(a: torch.Tensor) -> torch.Tensor:
    """Magnitude code (0..7) of round-to-nearest-even for non-negative ``a`` (already clamped to <= 6)."""
    return (
        (a > 0.25).to(torch.uint8)
        + (a >= 0.75).to(torch.uint8)
        + (a > 1.25).to(torch.uint8)
        + (a >= 1.75).to(torch.uint8)
        + (a > 2.5).to(torch.uint8)
        + (a >= 3.5).to(torch.uint8)
        + (a > 5.0).to(torch.uint8)
    )


def _sr_index(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    grid = E2M1_GRID.to(a.device)
    idx = (torch.bucketize(a, grid, right=True) - 1).clamp(0, 6)
    lo, hi = grid[idx], grid[idx + 1]
    p = (a - lo) / (hi - lo)
    return (idx + (u < p).to(idx.dtype)).to(torch.uint8)


def encode_e2m1(x: torch.Tensor, rounding: str = "nearest", u: torch.Tensor | None = None) -> torch.Tensor:
    """Float values (already block-scaled) -> 4-bit codes ``uint8`` in ``[0, 15]`` (bit 3 = sign, canonical: no -0)."""
    xf = x.float()
    a = xf.abs().clamp(max=E2M1_MAX)
    if rounding == "nearest":
        idx = _rtn_index(a)
    elif rounding == "stochastic":
        if u is None:
            raise ConfigError("stochastic rounding needs uniforms `u` (use counter_uniform)")
        idx = _sr_index(a, u)
    else:
        raise ConfigError(f"unknown rounding {rounding!r} (nearest|stochastic)")
    sign = ((xf < 0) & (idx > 0)).to(torch.uint8)
    return idx | (sign << 3)


def decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """4-bit codes -> float32 values on the E2M1 grid."""
    grid = E2M1_GRID.to(codes.device)
    c = codes.to(torch.int64)
    v = grid[c & 7]
    return torch.where((c & 8) != 0, -v, v)


def round_e2m1(x: torch.Tensor, rounding: str = "nearest", u: torch.Tensor | None = None) -> torch.Tensor:
    """Round (scaled) values onto the signed E2M1 grid, saturating at +-6; float32 result."""
    return decode_e2m1(encode_e2m1(x, rounding, u))


def stochastic_round_fp4(
    x: torch.Tensor, seed: int = 0, offset: int = 0, u: torch.Tensor | None = None
) -> torch.Tensor:
    """Unbiased stochastic rounding of (already scaled, ``|x|<=6``) values onto the E2M1 grid.

    ``E[result] == x`` for ``|x| <= 6``.  Uniforms: ``counter_uniform(x.shape, seed, offset)`` unless ``u`` is given.
    """
    if u is None:
        u = counter_uniform(x.shape, seed, offset, device=x.device)
    return round_e2m1(x, "stochastic", u)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes two per byte along the last dim (even index = low nibble)."""
    return pack_uint4(codes)


def unpack_fp4(packed: torch.Tensor, n: int | None = None) -> torch.Tensor:
    return unpack_uint4(packed, n)


# ------------------------------------------------------------------------------------------------ scale formats
def e8m0_scale(ideal: torch.Tensor, mode: str = "ceil") -> torch.Tensor:
    """Power-of-two scale (float32) for ideal scales ``amax/6``; clamped to the E8M0 range 2**-127..2**127."""
    lg = torch.log2(ideal.clamp(min=2.0**-127))
    if mode == "ceil":
        e = torch.ceil(lg)
    elif mode == "near":
        e = torch.round(lg)
    elif mode == "floor":
        e = torch.floor(lg)
    else:
        raise ConfigError(f"unknown e8m0 scale mode {mode!r} (ceil|floor|near)")
    return torch.pow(2.0, e.clamp(-127, 127))


def _pad_last(x: torch.Tensor, block: int) -> tuple[torch.Tensor, int]:
    k = x.shape[-1]
    pad = (-k) % block
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, pad


# ------------------------------------------------------------------------------------------------ NVFP4
@dataclass
class NVFP4Tensor:
    """Packed NVFP4 tensor.

    ``payload_u8`` ``[..., ceil(K/16)*8]`` two codes/byte; ``block_scales_e4m3`` ``[..., ceil(K/16)]`` (float8_e4m3fn, the *stored*
    scale ``stored_b``); ``tensor_scale`` float32 scalar tensor ``s_dec`` (= amax / (6*448)); ``shape`` original shape.
    ``dequant = decode(code) * block_scale * tensor_scale``.
    """

    payload_u8: torch.Tensor
    block_scales_e4m3: torch.Tensor
    tensor_scale: torch.Tensor
    shape: tuple[int, ...]
    block_size: int = 16
    meta: dict = field(default_factory=dict)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize_nvfp4(self, dtype)

    @property
    def nbytes(self) -> int:
        return int(
            self.payload_u8.numel()
            + self.block_scales_e4m3.numel() * self.block_scales_e4m3.element_size()
            + self.tensor_scale.numel() * 4
        )

    @property
    def bits_per_element(self) -> float:
        n = 1
        for s in self.shape:
            n *= s
        return 8.0 * self.nbytes / max(n, 1)

    def to(self, device) -> NVFP4Tensor:
        return NVFP4Tensor(
            self.payload_u8.to(device), self.block_scales_e4m3.to(device), self.tensor_scale.to(device), self.shape, self.block_size, dict(self.meta)
        )

    def codes(self) -> torch.Tensor:
        """Unpacked 4-bit codes ``[..., padded K]``."""
        return unpack_fp4(self.payload_u8)


def _block_amax(xf: torch.Tensor, block: int, scale_2d: bool) -> torch.Tensor:
    """Per-block absolute maxima ``[..., nb, 1]`` for ``xf`` of shape ``[..., nb, block]``."""
    amax = xf.abs().amax(dim=-1, keepdim=True)
    if not scale_2d:
        return amax
    # 2-D (block x block) tile scaling for weights: share the max over `block` consecutive rows (dim -3 of [R, nb, 1])
    if amax.ndim < 3:
        raise ConfigError("scale_2d needs a tensor with at least 2 dims")
    lead = amax.shape[:-3]
    r, nb, _ = amax.shape[-3:]
    a = amax.reshape(-1, r, nb, 1)
    padr = (-r) % block
    if padr:
        a = torch.nn.functional.pad(a, (0, 0, 0, 0, 0, padr))
    t = a.reshape(a.shape[0], -1, block, nb, 1).amax(dim=2, keepdim=True)
    t = t.expand(-1, -1, block, -1, -1).reshape(a.shape[0], -1, nb, 1)[:, :r]
    return t.reshape(*lead, r, nb, 1)


def _nvfp4_core(
    x: torch.Tensor,
    block_size: int,
    rounding: str,
    seed: int,
    offset: int,
    tensor_amax: float | torch.Tensor | None,
    headroom: float,
    four_over_six: bool,
    scale_2d: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Returns ``(codes [..., Kp], stored_scales [..., nb] float32, s_dec scalar, pad)``."""
    xf, pad = _pad_last(x.detach().float(), block_size)
    xb = xf.reshape(*xf.shape[:-1], -1, block_size)
    if tensor_amax is None:
        gamax = xf.abs().amax()
    else:
        gamax = torch.as_tensor(tensor_amax, dtype=torch.float32, device=xf.device)
    m_t = 256.0 if four_over_six else E4M3_MAX  # 4/6 leaves headroom: 1.5 * 256 = 384 <= 448
    gamax = gamax.clamp(min=1e-30)
    s_enc = (E2M1_MAX * m_t) / gamax
    s_dec = 1.0 / s_enc
    amax_b = _block_amax(xb, block_size, scale_2d)
    u = counter_uniform(xb.shape, seed, offset, device=xb.device) if rounding == "stochastic" else None

    def attempt(m: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stored = round_e4m3((amax_b / (m * headroom)) * s_enc)
        ok = stored > 0
        denom = torch.where(ok, stored * s_dec, torch.ones_like(stored))
        scaled = xb / denom
        scaled = torch.where(ok, scaled, torch.zeros_like(scaled))
        codes = encode_e2m1(scaled, rounding, u)
        deq = decode_e2m1(codes) * stored * s_dec
        return codes, stored, deq

    codes, stored, deq = attempt(E2M1_MAX)
    if four_over_six:
        codes4, stored4, deq4 = attempt(4.0)
        e6 = (deq - xb).pow(2).sum(dim=-1, keepdim=True)
        e4 = (deq4 - xb).pow(2).sum(dim=-1, keepdim=True)
        use4 = e4 < e6
        codes = torch.where(use4, codes4, codes)
        stored = torch.where(use4, stored4, stored)
    codes = codes.reshape(*xf.shape)
    return codes, stored.squeeze(-1), s_dec.reshape(()), pad


def quantize_nvfp4(
    x: torch.Tensor,
    *,
    block_size: int = 16,
    rounding: str = "nearest",
    seed: int = 0,
    offset: int = 0,
    tensor_amax: float | torch.Tensor | None = None,
    headroom: float = 1.0,
    four_over_six: bool = False,
    scale_2d: bool = False,
) -> NVFP4Tensor:
    """Quantise to packed NVFP4 along the last dim (see module docstring).

    ``tensor_amax``: override the global amax (e.g. synchronised across ranks).  ``headroom``: >1 shrinks block scales target
    (``amax_b/(6*headroom)``).  ``four_over_six``: per block pick between mapping the block max to 6 or to 4 by lowest block
    MSE ("Four Over Six"; tensor scale then uses 256 instead of 448).  ``scale_2d``: scales from 16x16 tiles (weights).
    """
    codes, stored, s_dec, _ = _nvfp4_core(x, block_size, rounding, seed, offset, tensor_amax, headroom, four_over_six, scale_2d)
    return NVFP4Tensor(
        pack_fp4(codes),
        stored.to(torch.float8_e4m3fn),
        s_dec.to(torch.float32),
        tuple(x.shape),
        block_size,
    )


def dequantize_nvfp4(t: NVFP4Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    k = t.shape[-1]
    codes = unpack_fp4(t.payload_u8)
    vals = decode_e2m1(codes)
    kp = vals.shape[-1]
    vb = vals.reshape(*vals.shape[:-1], kp // t.block_size, t.block_size)
    out = vb * t.block_scales_e4m3.float().unsqueeze(-1) * t.tensor_scale.float()
    return out.reshape(*vals.shape)[..., :k].to(dtype)


def fake_quant_nvfp4(
    x: torch.Tensor,
    *,
    block_size: int = 16,
    rounding: str = "nearest",
    seed: int = 0,
    offset: int = 0,
    tensor_amax: float | torch.Tensor | None = None,
    headroom: float = 1.0,
    four_over_six: bool = False,
    scale_2d: bool = False,
    ste: bool = False,
) -> torch.Tensor:
    """Quantise->dequantise without packing (same numerics as ``dequantize_nvfp4(quantize_nvfp4(x))``)."""
    codes, stored, s_dec, _ = _nvfp4_core(x, block_size, rounding, seed, offset, tensor_amax, headroom, four_over_six, scale_2d)
    vals = decode_e2m1(codes)
    vb = vals.reshape(*vals.shape[:-1], -1, block_size)
    out = (vb * stored.unsqueeze(-1) * s_dec).reshape(*vals.shape)[..., : x.shape[-1]].to(x.dtype)
    return x + (out - x).detach() if ste else out


# ------------------------------------------------------------------------------------------------ MXFP4
@dataclass
class MXFP4Tensor:
    """Packed MXFP4: ``payload_u8`` (2 codes/byte), ``scales_e8m0_u8`` (biased exponent: scale = 2**(u8-127)), ``shape``."""

    payload_u8: torch.Tensor
    scales_e8m0_u8: torch.Tensor
    shape: tuple[int, ...]
    block_size: int = 32
    meta: dict = field(default_factory=dict)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize_mxfp4(self, dtype)

    @property
    def nbytes(self) -> int:
        return int(self.payload_u8.numel() + self.scales_e8m0_u8.numel())

    @property
    def bits_per_element(self) -> float:
        n = 1
        for s in self.shape:
            n *= s
        return 8.0 * self.nbytes / max(n, 1)

    def to(self, device) -> MXFP4Tensor:
        return MXFP4Tensor(self.payload_u8.to(device), self.scales_e8m0_u8.to(device), self.shape, self.block_size, dict(self.meta))


def _mxfp4_core(
    x: torch.Tensor, block_size: int, scale_mode: str, rounding: str, seed: int, offset: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    xf, pad = _pad_last(x.detach().float(), block_size)
    xb = xf.reshape(*xf.shape[:-1], -1, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    if scale_mode == "floor":  # OCP reference conversion: 2**(floor(log2 amax) - emax_elem), emax_elem = 2
        e = torch.floor(torch.log2(amax.clamp(min=2.0**-126))) - 2.0
        s = torch.pow(2.0, e.clamp(-127, 127))
    else:
        s = e8m0_scale(amax / E2M1_MAX, scale_mode)
    s = torch.where(amax == 0, torch.ones_like(s), s)
    u = counter_uniform(xb.shape, seed, offset, device=xb.device) if rounding == "stochastic" else None
    codes = encode_e2m1(xb / s, rounding, u)
    return codes.reshape(*xf.shape), s.squeeze(-1), pad


def quantize_mxfp4(
    x: torch.Tensor,
    *,
    block_size: int = 32,
    scale_mode: str = "ceil",
    rounding: str = "nearest",
    seed: int = 0,
    offset: int = 0,
) -> MXFP4Tensor:
    """OCP MXFP4 (E2M1 elements, E8M0 block scale, default block 32). ``scale_mode``: ceil | floor | near."""
    codes, s, _ = _mxfp4_core(x, block_size, scale_mode, rounding, seed, offset)
    biased = (torch.log2(s).round() + 127).clamp(0, 254).to(torch.uint8)
    return MXFP4Tensor(pack_fp4(codes), biased, tuple(x.shape), block_size)


def dequantize_mxfp4(t: MXFP4Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    k = t.shape[-1]
    vals = decode_e2m1(unpack_fp4(t.payload_u8))
    kp = vals.shape[-1]
    vb = vals.reshape(*vals.shape[:-1], kp // t.block_size, t.block_size)
    s = torch.pow(2.0, t.scales_e8m0_u8.float() - 127.0).unsqueeze(-1)
    return (vb * s).reshape(*vals.shape)[..., :k].to(dtype)


def fake_quant_mxfp4(
    x: torch.Tensor,
    *,
    block_size: int = 32,
    scale_mode: str = "ceil",
    rounding: str = "nearest",
    seed: int = 0,
    offset: int = 0,
    ste: bool = False,
) -> torch.Tensor:
    codes, s, _ = _mxfp4_core(x, block_size, scale_mode, rounding, seed, offset)
    vals = decode_e2m1(codes)
    out = (vals.reshape(*vals.shape[:-1], -1, block_size) * s.unsqueeze(-1)).reshape(*vals.shape)[..., : x.shape[-1]].to(x.dtype)
    return x + (out - x).detach() if ste else out


def quantize_fp4_ideal(x: torch.Tensor, block_size: int = 16, rounding: str = "nearest", seed: int = 0) -> torch.Tensor:
    """E2M1 with an unquantised FP32 absmax block scale (the idealised bound used in the paper's tables); returns dequantised."""
    xf, _ = _pad_last(x.detach().float(), block_size)
    xb = xf.reshape(*xf.shape[:-1], -1, block_size)
    s = xb.abs().amax(dim=-1, keepdim=True) / E2M1_MAX
    s = torch.where(s == 0, torch.ones_like(s), s)
    u = counter_uniform(xb.shape, seed, 0, device=xb.device) if rounding == "stochastic" else None
    q = round_e2m1(xb / s, rounding, u) * s
    return q.reshape(*xf.shape)[..., : x.shape[-1]].to(x.dtype)
