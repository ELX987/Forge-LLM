"""Counter-based RNG oracle: Philox-4x32-10 (Salmon et al., Random123), bit-identical to ``tl.randint4x``.

Plan K8 / §6.7: "counter-based RNG reproducibility across launches" - the random number for element ``i`` depends
only on ``(seed, i)``, never on launch geometry, so stochastic rounding and rejection sampling are reproducible
across launches, grid sizes, devices and CPU/GPU.

Convention (shared with the Triton kernels): counter words are ``(c0, c1, c2, c3) = (offset & 0xFFFFFFFF, 0, 0, 0)``
and the 64-bit ``seed`` is split into the key ``(k0, k1) = (seed_lo, seed_hi)``.  Offsets are therefore 32-bit
(2^32 distinct numbers per seed); callers fold step counters / tensor ids into ``seed`` (see :func:`derive_seed`).
Known-answer test: ``philox4x32(0, 0, 0, 0, 0, 0) == (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)``.
"""
from __future__ import annotations

import numpy as np
import torch

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
_MASK = np.uint64(0xFFFFFFFF)


def _mulhilo(a: int, b: np.ndarray):
    p = np.uint64(a) * b.astype(np.uint64)
    return (p >> np.uint64(32)).astype(np.uint64), (p & _MASK).astype(np.uint64)


def philox4x32(c0, c1, c2, c3, k0, k1, rounds: int = 10):
    """Philox-4x32 on numpy arrays (or ints) of uint32 values; returns four uint64 arrays holding uint32 values."""
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) & _MASK for c in (c0, c1, c2, c3))
    c0, c1, c2, c3 = np.broadcast_arrays(c0, c1, c2, c3)
    k0 = np.asarray(k0, dtype=np.uint64) & _MASK
    k1 = np.asarray(k1, dtype=np.uint64) & _MASK
    for _ in range(rounds):
        hi0, lo0 = _mulhilo(PHILOX_M0, c0)
        hi1, lo1 = _mulhilo(PHILOX_M1, c2)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _MASK, lo1, (hi0 ^ c3 ^ k1) & _MASK, lo0
        k0 = (k0 + np.uint64(PHILOX_W0)) & _MASK
        k1 = (k1 + np.uint64(PHILOX_W1)) & _MASK
    return c0, c1, c2, c3


def _split_seed(seed: int) -> tuple[int, int]:
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    return s & 0xFFFFFFFF, s >> 32


def philox_u32(seed: int, offsets: torch.Tensor) -> torch.Tensor:
    """First Philox output word for each offset -> ``int64`` tensor holding uint32 values (same shape as ``offsets``)."""
    off = offsets.detach().cpu().to(torch.int64).numpy().astype(np.uint64) & _MASK
    k0, k1 = _split_seed(seed)
    r0, _, _, _ = philox4x32(off, 0, 0, 0, k0, k1)
    return torch.from_numpy(r0.astype(np.int64)).reshape(offsets.shape).to(offsets.device)


def philox_uniform(seed: int, offsets: torch.Tensor) -> torch.Tensor:
    """Uniform floats in ``[0, 1)`` (24 random bits): ``(r0 >> 8) * 2**-24`` - identical formula in the kernels."""
    r = philox_u32(seed, offsets)
    return ((r >> 8).to(torch.float32)) * (1.0 / 16777216.0)


def derive_seed(base_seed: int, *ids: int) -> int:
    """Fold integers (step, layer, tensor id, ...) into a 64-bit seed with a splitmix64-style mixer (host side)."""
    x = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    for i in ids:
        x = (x + 0x9E3779B97F4A7C15 + (int(i) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x = x ^ (x >> 31)
    # keep it in signed int64 range so it can be passed to kernels as a plain int
    return x & 0x7FFFFFFFFFFFFFFF
