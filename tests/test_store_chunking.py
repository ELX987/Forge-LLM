from __future__ import annotations

import io

import numpy as np
import pytest

from forge.store.chunking import MiB, ChunkerParams, chunk_offsets, iter_chunks

SMALL = ChunkerParams(min_size=2048, bits=12, max_size=16384)


def _data(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def test_chunks_cover_and_respect_bounds():
    d = _data(600_000)
    offs = chunk_offsets(d, SMALL)
    assert offs[0][0] == 0
    pos = 0
    for o, ln in offs:
        assert o == pos
        pos += ln
    assert pos == len(d)
    sizes = [ln for _, ln in offs]
    assert all(s <= SMALL.max_size for s in sizes)
    assert all(s >= SMALL.min_size for s in sizes[:-1])
    mean = np.mean(sizes[:-1])
    assert 0.6 * SMALL.expected_size < mean < 1.6 * SMALL.expected_size


def test_deterministic_and_stream_equals_bytes():
    d = _data(300_000, 3)
    a = chunk_offsets(d, SMALL)
    assert a == chunk_offsets(d, SMALL)
    bio = io.BytesIO(d)
    # irregular read sizes must not change the cut points
    got = list(iter_chunks(lambda n: bio.read(min(n, 7919)), SMALL, read_size=7919))
    assert [len(c) for c in got] == [ln for _, ln in a]
    assert b"".join(got) == d


def test_shift_resilience():
    d = _data(400_000, 5)
    d2 = b"INSERTED-PREFIX-BYTES" + d
    c1 = {d[o : o + n] for o, n in chunk_offsets(d, SMALL)}
    c2 = {d2[o : o + n] for o, n in chunk_offsets(d2, SMALL)}
    assert len(c1 & c2) >= len(c1) - 3  # only the chunks around the edit change


def test_local_edit_changes_few_chunks():
    d = bytearray(_data(500_000, 9))
    c1 = {bytes(d[o : o + n]) for o, n in chunk_offsets(bytes(d), SMALL)}
    d[250_000] ^= 0xFF
    c2 = {bytes(d[o : o + n]) for o, n in chunk_offsets(bytes(d), SMALL)}
    assert len(c1 - c2) <= 3


def test_edge_sizes():
    assert chunk_offsets(b"", SMALL) == []
    assert chunk_offsets(b"x", SMALL) == [(0, 1)]
    d = _data(SMALL.min_size)
    assert chunk_offsets(d, SMALL) == [(0, SMALL.min_size)]
    assert list(iter_chunks(io.BytesIO(b"").read, SMALL)) == []


def test_default_params_match_plan():
    p = ChunkerParams()
    assert (p.min_size, p.max_size) == (1 * MiB, 4 * MiB)
    d = _data(12 * MiB, 1)
    sizes = [ln for _, ln in chunk_offsets(d, p)]
    assert max(sizes) <= 4 * MiB and min(sizes[:-1]) >= 1 * MiB


def test_param_validation():
    with pytest.raises(ValueError):
        ChunkerParams(min_size=10)
    with pytest.raises(ValueError):
        ChunkerParams(min_size=4096, max_size=4096)
