"""Emulated low-precision numerics for the precision ladder (pure torch, runs on CPU and GPU).

NVFP4 (Plan 7.4 R2, NVFP4 paper chapters 3, 5, 6):
  * element format E2M1, grid {0, .5, 1, 1.5, 2, 3, 4, 6}; round-to-nearest-even (ties to the even code) or stochastic rounding
  * two-level scaling: FP32 per-tensor scale ``s_enc = 6*448/amax`` and an E4M3 per-block scale
    ``stored = E4M3(block_amax/6 * s_enc)``; effective block scale ``stored / s_enc``
  * blocks of 1x16 (activations, gradients) or 16x16 tiles (weights, 2D scaling)
  * optional random Hadamard transform (RHT) along the contraction dimension of the weight-gradient GEMM

This is *fake quantisation*: values are quantised and de-quantised in float32 and the GEMM runs in a regular dtype.  It is NOT a
speed emulation and has not been validated against Blackwell hardware numerics.
"""
from __future__ import annotations

import math
from typing import Any

import torch

__all__ = [
    "E2M1_GRID", "E4M3_MAX", "E4M3_MIN_NORMAL", "E4M3_MIN_SUBNORMAL", "FP4Rng", "QuantizedInfo", "e4m3_round", "fake_quant_fp8",
    "fake_quant_nvfp4", "hadamard_matrix", "random_hadamard", "random_signs", "round_e2m1_rtn", "round_e2m1_sr",
]

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6
E4M3_MIN_SUBNORMAL = 2.0 ** -9


class FP4Rng:
    """Holds per-device torch generators for stochastic rounding (seeded, reproducible)."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._gens: dict[str, torch.Generator] = {}

    def gen(self, device: torch.device) -> torch.Generator:
        key = str(device)
        g = self._gens.get(key)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            self._gens[key] = g
        return g

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._gens.clear()

    def rand(self, shape: Any, device: torch.device) -> torch.Tensor:
        return torch.rand(shape, device=device, generator=self.gen(device))


_DEFAULT_RNG = FP4Rng(0)


def round_e2m1_rtn(z: torch.Tensor) -> torch.Tensor:
    """Round (already scaled) values to the E2M1 grid: nearest, ties to the even mantissa code, saturating at +-6."""
    a = z.abs().clamp(max=6.0)
    grid = torch.tensor(E2M1_GRID, device=z.device, dtype=a.dtype)
    mids = torch.tensor(_MIDS, device=z.device, dtype=a.dtype)
    idx = torch.bucketize(a, mids, right=False)  # ties -> lower code
    midv = mids[idx.clamp(max=6)]
    tie_up = (a == midv) & (idx < 7) & (((idx + 1) % 2) == 0)
    idx = idx + tie_up.long()
    return torch.sign(z) * grid[idx]


def round_e2m1_sr(z: torch.Tensor, rng: FP4Rng | None = None) -> torch.Tensor:
    """Stochastic rounding onto the E2M1 grid: unbiased for |z| <= 6 (saturates beyond)."""
    rng = rng or _DEFAULT_RNG
    a = z.abs().clamp(max=6.0)
    grid = torch.tensor(E2M1_GRID, device=z.device, dtype=a.dtype)
    idx = (torch.bucketize(a, grid, right=True) - 1).clamp(0, 6)
    lo, hi = grid[idx], grid[idx + 1]
    p = (a - lo) / (hi - lo)
    u = rng.rand(a.shape, z.device).to(a.dtype)
    out = torch.where(u < p, hi, lo)
    return torch.sign(z) * out


def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round non-negative float32 values to E4M3 (flush-to-zero below half the smallest subnormal, saturate at 448)."""
    x = x.clamp(min=0.0, max=E4M3_MAX).to(torch.float32)
    return x.to(torch.float8_e4m3fn).to(torch.float32)


class QuantizedInfo:
    """Side information of one :func:`fake_quant_nvfp4` call (all float32 tensors / python floats)."""

    __slots__ = ("block", "stored_scales", "block_amax", "tensor_scale", "tensor_amax", "saturated_frac", "zero_frac", "shape")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _as_blocks(x2: torch.Tensor, br: int, bc: int) -> tuple[torch.Tensor, int, int]:
    R, C = x2.shape
    pr, pc = (-R) % br, (-C) % bc
    if pr or pc:
        x2 = torch.nn.functional.pad(x2, (0, pc, 0, pr))
    Rp, Cp = x2.shape
    return x2.reshape(Rp // br, br, Cp // bc, bc), pr, pc


def fake_quant_nvfp4(
    x: torch.Tensor, block: tuple[int, int] = (1, 16), rounding: str = "rtn", rng: FP4Rng | None = None,
    tensor_amax: torch.Tensor | float | None = None, return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, QuantizedInfo]:
    """NVFP4 quantise-dequantise of ``x`` (any shape; leading dims are flattened, blocks run over the last two dims).

    ``block=(1,16)`` -> 1x16 blocks along the last dim; ``(16,16)`` -> 2D tiles.  ``rounding`` in {'rtn','sr'}.
    Returns the de-quantised tensor in float32 (same shape as ``x``).
    """
    if rounding not in ("rtn", "sr"):
        raise ValueError("rounding must be 'rtn' or 'sr'")
    shape = x.shape
    x2 = x.detach().reshape(-1, shape[-1]).to(torch.float32)
    R, C = x2.shape
    br, bc = block
    xb, pr, pc = _as_blocks(x2, br, bc)  # [R/br, br, C/bc, bc]
    amax_b = xb.abs().amax(dim=(1, 3), keepdim=True)  # [R/br,1,C/bc,1]
    if tensor_amax is None:
        t_amax = x2.abs().max()
    else:
        t_amax = torch.as_tensor(tensor_amax, dtype=torch.float32, device=x2.device)
    safe_t = torch.where(t_amax > 0, t_amax, torch.ones_like(t_amax))
    s_enc = 6.0 * E4M3_MAX / safe_t
    stored = e4m3_round(amax_b / 6.0 * s_enc)  # E4M3 block scale (in "encoded" units)
    eff = stored / s_enc  # effective decode scale per block
    nz = eff > 0
    z = torch.where(nz, xb / torch.where(nz, eff, torch.ones_like(eff)), torch.zeros_like(xb))
    q = round_e2m1_rtn(z) if rounding == "rtn" else round_e2m1_sr(z, rng)
    deq = (q * eff).reshape(xb.shape[0] * br, xb.shape[2] * bc)[:R, :C].reshape(shape)
    if not return_info:
        return deq
    info = QuantizedInfo(
        block=block, stored_scales=stored.reshape(xb.shape[0], xb.shape[2]), block_amax=amax_b.reshape(xb.shape[0], xb.shape[2]),
        tensor_scale=s_enc.detach(), tensor_amax=t_amax.detach(),
        saturated_frac=float((z.abs() > 6.0).float().mean()), zero_frac=float((q == 0).float().mean()), shape=tuple(shape))
    return deq, info


def fake_quant_fp8(x: torch.Tensor, fmt: str = "e4m3", amax: torch.Tensor | None = None) -> torch.Tensor:
    """Per-tensor dynamic-scale FP8 quantise-dequantise (E4M3 max 448, E5M2 max 57344)."""
    if fmt == "e4m3":
        fmax, dt = 448.0, torch.float8_e4m3fn
    elif fmt == "e5m2":
        fmax, dt = 57344.0, torch.float8_e5m2
    else:
        raise ValueError(fmt)
    xf = x.to(torch.float32)
    a = xf.detach().abs().max() if amax is None else torch.as_tensor(amax, dtype=torch.float32, device=x.device)
    a = torch.where(a > 0, a, torch.ones_like(a))
    scale = fmax / a
    q = (xf * scale).clamp(-fmax, fmax).to(dt).to(torch.float32) / scale
    return q


# --- random Hadamard transform ---------------------------------------------------------------------------------------
_H_CACHE: dict[tuple[int, str, str], torch.Tensor] = {}


def hadamard_matrix(n: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix ``H`` (``H @ H.T == I``); ``n`` must be a power of two."""
    if n < 1 or (n & (n - 1)):
        raise ValueError("hadamard size must be a power of two")
    key = (n, str(device), str(dtype))
    h = _H_CACHE.get(key)
    if h is None:
        m = torch.ones(1, 1, dtype=torch.float32)
        while m.shape[0] < n:
            m = torch.cat([torch.cat([m, m], 1), torch.cat([m, -m], 1)], 0)
        h = (m / math.sqrt(n)).to(device=device, dtype=dtype)
        _H_CACHE[key] = h
    return h


def random_signs(n: int, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(int(seed) * 1000003 + int(n))
    return (torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) * 2 - 1).to(device=device, dtype=torch.float32)


def random_hadamard(x: torch.Tensor, block: int = 16, seed: int = 0, inverse: bool = False) -> torch.Tensor:
    """Apply ``y = H_blk (D x)`` independently to consecutive ``block`` chunks along the last dim (``D`` random +-1).

    Orthonormal, so inner products over the last dim are preserved: ``<Tx, Ty> = <x, y>``.  Last dim must be divisible by ``block``.
    """
    n = x.shape[-1]
    if n % block:
        raise ValueError(f"last dim {n} not divisible by RHT block {block}")
    d = random_signs(n, seed, x.device)
    h = hadamard_matrix(block, x.device, torch.float32)
    xf = x.to(torch.float32)
    if not inverse:
        y = (xf * d).reshape(*x.shape[:-1], n // block, block) @ h
        return y.reshape(x.shape)
    y = xf.reshape(*x.shape[:-1], n // block, block) @ h.T
    return y.reshape(x.shape) * d
