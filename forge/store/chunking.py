"""Deterministic content-defined chunking (gear / FastCDC family), vectorised with numpy.

Boundary rule
-------------
A 64-bit *window hash* of the last ``window`` (48) bytes is computed for every position::

    H[i] = sum_{j<W} G[b[i-j]] * A^(-j)   (mod 2^64)

with ``G`` a fixed pseudo-random 64-bit table and ``A`` an odd multiplier.  Because ``A`` is invertible mod 2^64
this is a *translation invariant* rolling hash that can be evaluated for a whole array with one ``cumsum``
(``C[i] = sum_k G[b[k]] A^k``, ``H[i] = (C[i]-C[i-W]) * A^-i``), i.e. O(n) numpy passes instead of a Python loop.
A position ``p`` is a cut candidate when its top ``bits`` bits are zero (probability ``2^-bits``).  For a chunk
starting at ``s`` the cut is placed after the first candidate ``p >= s+min_size-1``; if none exists before
``s+max_size`` the chunk is cut at ``max_size``.  The result depends only on the bytes, never on how the data
was buffered, so byte strings and streams chunk identically.

Defaults: min 1 MiB, ``bits=20`` (mean ~= min + 2^20 = 2 MiB), max 4 MiB (Plan: 1-4 MiB chunks).
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MiB = 1 << 20
WINDOW = 48
_SEG = 1 << 20  # positions evaluated per numpy pass
_A = 0x9E3779B97F4A7C15  # odd multiplier
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class ChunkerParams:
    """Chunk size policy. ``min_size`` must be >= 2*WINDOW and ``max_size`` > ``min_size``."""

    min_size: int = 1 * MiB
    bits: int = 20
    max_size: int = 4 * MiB

    def __post_init__(self) -> None:
        if self.min_size < 2 * WINDOW:
            raise ValueError(f"min_size must be >= {2 * WINDOW}")
        if self.max_size <= self.min_size:
            raise ValueError("max_size must exceed min_size")
        if not (1 <= self.bits <= 40):
            raise ValueError("bits must be in 1..40")

    @property
    def expected_size(self) -> int:
        return self.min_size + (1 << self.bits)


DEFAULT_PARAMS = ChunkerParams()


@lru_cache(maxsize=1)
def _gear() -> np.ndarray:
    vals = [
        int.from_bytes(hashlib.blake2b(b"forge-gear-v1" + bytes([i]), digest_size=8).digest(), "little")
        for i in range(256)
    ]
    return np.array(vals, dtype=np.uint64)


@lru_cache(maxsize=1)
def _powers() -> tuple[np.ndarray, np.ndarray]:
    n = _SEG + WINDOW
    ainv = pow(_A, -1, 1 << 64)

    def pw(base: int) -> np.ndarray:
        a = np.full(n, base, dtype=np.uint64)
        a[0] = 1
        return np.cumprod(a, dtype=np.uint64)

    return pw(_A), pw(ainv)


def window_hashes(data: np.ndarray) -> np.ndarray:
    """Window hash for every index ``i >= WINDOW-1`` of ``data`` (uint8, length <= _SEG+WINDOW).

    Returns an array of the same length; entries ``< WINDOW-1`` are meaningless."""
    m = len(data)
    p, pinv = _powers()
    g = _gear()[data]
    c = np.cumsum(g * p[:m], dtype=np.uint64)
    w = c.copy()
    if m > WINDOW:
        w[WINDOW:] -= c[: m - WINDOW]
    return w * pinv[:m]


def find_boundary(data: np.ndarray, start: int, avail_end: int, params: ChunkerParams) -> int | None:
    """End offset (exclusive) of the chunk starting at ``start`` or ``None`` when a cut cannot be decided.

    ``data`` is a uint8 array; only ``data[:avail_end]`` is valid.  A cut is *decided* when a candidate is found,
    or when ``max_size`` bytes are available.  Otherwise ``None`` (caller must supply more data or hit EOF)."""
    first_p = start + params.min_size - 1
    last_p = min(start + params.max_size, avail_end) - 1  # inclusive
    thr = np.uint64(1 << (64 - params.bits))
    p0 = first_p
    while p0 <= last_p:
        p1 = min(p0 + _SEG - 1, last_p)
        lo = p0 - WINDOW + 1
        h = window_hashes(data[lo : p1 + 1])
        hit = np.flatnonzero(h[WINDOW - 1 :] < thr)
        if hit.size:
            return p0 + int(hit[0]) + 1
        p0 = p1 + 1
    if avail_end - start >= params.max_size:
        return start + params.max_size
    return None


def chunk_offsets(data: bytes | bytearray | memoryview | np.ndarray, params: ChunkerParams = DEFAULT_PARAMS) -> list[tuple[int, int]]:
    """``[(offset, length), ...]`` covering ``data`` exactly. Deterministic."""
    arr = data if isinstance(data, np.ndarray) else np.frombuffer(data, dtype=np.uint8)
    n = len(arr)
    out: list[tuple[int, int]] = []
    s = 0
    while s < n:
        if n - s <= params.min_size:
            out.append((s, n - s))
            break
        e = find_boundary(arr, s, n, params)
        if e is None:  # fewer than max_size bytes remain and no candidate: tail chunk
            e = n
        out.append((s, e - s))
        s = e
    return out


def iter_chunks(
    read: Callable[[int], bytes], params: ChunkerParams = DEFAULT_PARAMS, read_size: int | None = None
) -> Iterator[bytes]:
    """Chunk a stream. ``read(n)`` must return up to ``n`` bytes and ``b''`` at EOF."""
    read_size = read_size or max(8 * MiB, 2 * params.max_size)
    buf = bytearray()
    eof = False
    while True:
        while not eof and len(buf) < params.max_size + 1:
            blk = read(read_size)
            if not blk:
                eof = True
            else:
                buf += blk
        if not buf:
            return
        s = 0
        n = len(buf)
        cuts: list[int] = []
        arr = np.frombuffer(buf, dtype=np.uint8)
        while s < n:
            remaining = n - s
            if remaining <= params.min_size:
                if eof:
                    cuts.append(n)
                    s = n
                break
            e = find_boundary(arr, s, n, params)
            if e is None:
                if eof:
                    cuts.append(n)
                    s = n
                break
            cuts.append(e)
            s = e
        del arr
        prev = 0
        for c in cuts:
            yield bytes(buf[prev:c])
            prev = c
        del buf[:prev]
        if eof and not buf:
            return
