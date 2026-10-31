"""Fast Walsh-Hadamard transform oracle (Plan K8; rotation-based PTQ and the NVFP4 Wgrad recipe).

``hadamard_ref(x, block)`` applies an orthonormal Sylvester-ordered Hadamard transform independently to every
contiguous ``block`` elements of the last dimension (block-diagonal ``H_block``), optionally preceded by a diagonal
``signs`` matrix (randomised Hadamard transform, RHT: ``y = H (signs * x)``).  With ``normalize=True`` the transform is
orthonormal and self-inverse up to the sign vector: ``x = signs * H y``.
"""
from __future__ import annotations

import math

import torch


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(n: int, dtype=torch.float32, device=None, normalize: bool = False) -> torch.Tensor:
    """Sylvester Hadamard matrix ``H_n`` (entries +-1; ``normalize`` divides by ``sqrt(n)``)."""
    if not is_pow2(n):
        raise ValueError("Hadamard size must be a power of two")
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if normalize:
        h = h / math.sqrt(n)
    return h.to(dtype)


def fwht_butterfly(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised FWHT over the last dim (power of two) via ``log2 n`` butterfly stages (float math)."""
    n = x.shape[-1]
    if not is_pow2(n):
        raise ValueError("last dim must be a power of two")
    lead = x.shape[:-1]
    y = x
    h = 1
    while h < n:
        y = y.reshape(*lead, n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2)
        h *= 2
    return y.reshape(*lead, n)


def hadamard_ref(x: torch.Tensor, block: int | None = None, signs: torch.Tensor | None = None,
                 normalize: bool = True) -> torch.Tensor:
    """Block-diagonal (randomised) Hadamard transform along the last dim; result has ``x.dtype``."""
    d = x.shape[-1]
    block = d if block is None else block
    if d % block:
        raise ValueError(f"last dim {d} is not a multiple of block {block}")
    xf = x.float()
    if signs is not None:
        xf = xf * signs.float()
    y = fwht_butterfly(xf.reshape(*x.shape[:-1], d // block, block))
    if normalize:
        y = y / math.sqrt(block)
    return y.reshape(x.shape).to(x.dtype)


def random_signs(n: int, seed: int = 0, device=None) -> torch.Tensor:
    """Deterministic +-1 float32 vector for the randomised Hadamard transform."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return s.to(device) if device is not None else s
