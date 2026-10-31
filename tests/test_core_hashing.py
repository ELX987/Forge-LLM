from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from forge.core import hashing as H


def test_known_vector_b2():
    assert H.hash_bytes(b"", "b2") == "b2:" + hashlib.blake2b(b"", digest_size=32).hexdigest()
    assert H.hash_bytes(b"abc", "sha256") == "sha256:" + hashlib.sha256(b"abc").hexdigest()


def test_blake3_known_vector_if_available():
    pytest.importorskip("blake3")
    # official BLAKE3 test vector for the empty input
    assert H.hash_bytes(b"", "blake3") == "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"


def test_default_algo_prefix(monkeypatch):
    monkeypatch.setenv("FORGE_HASH_ALGO", "b2")
    assert H.hash_algo() == "b2"
    assert H.hash_bytes(b"x").startswith("b2:")
    monkeypatch.delenv("FORGE_HASH_ALGO")
    assert H.hash_bytes(b"x").split(":")[0] in ("blake3", "b2")


def test_hasher_incremental_equals_oneshot():
    h = H.Hasher()
    h.update(b"hello ").update(b"world")
    assert h.hexdigest() == H.hash_bytes(b"hello world")
    c = h.copy()
    c.update(b"!")
    assert c.hexdigest() == H.hash_bytes(b"hello world!")
    assert h.hexdigest() == H.hash_bytes(b"hello world")


def test_hash_file(tmp_path: Path):
    p = tmp_path / "f.bin"
    data = bytes(range(256)) * 10000  # > 1 MiB chunk boundary
    p.write_bytes(data)
    assert H.hash_file(p) == H.hash_bytes(data)


def test_canonical_json_is_order_and_whitespace_independent():
    a = {"b": 1, "a": [1, 2, {"y": 1.5, "x": None}]}
    b = {"a": [1, 2, {"x": None, "y": 1.5}], "b": 1}
    assert H.canonical_json(a) == H.canonical_json(b) == '{"a":[1,2,{"x":null,"y":1.5}],"b":1}'
    assert H.hash_json(a) == H.hash_json(b)


def test_canonical_json_golden():
    obj = {"z": [1, 2.0, "é"], "a": {"k": True}, "t": (1, 2), "s": {3, 1, 2}}
    assert H.canonical_json(obj) == '{"a":{"k":true},"s":[1,2,3],"t":[1,2],"z":[1,2.0,"\\u00e9"]}'


def test_float_vs_int_differ_and_repr_used():
    assert H.hash_json(1) != H.hash_json(1.0)
    assert H.canonical_json(0.1) == "0.1"
    assert H.canonical_json(1e-7) == repr(1e-7)


def test_special_types():
    import enum

    class E(enum.Enum):
        A = "a"

    assert H.canonical_json({E.A: Path("x/y")}) == '{"a":"x/y"}'
    assert H.canonical_json(b"ab").startswith('"bytes:')
    import numpy as np

    assert H.canonical_json(np.float32(1.5)) == "1.5"
    assert H.canonical_json(np.arange(3)) == H.canonical_json(np.arange(3))
    with pytest.raises(TypeError):
        H.canonical_json(object())


def test_digest_helpers():
    d = H.hash_bytes(b"x", "b2")
    assert H.is_digest(d)
    assert H.digest_algo(d) == "b2"
    assert len(H.digest_hex(d)) == 64
    assert not H.is_digest("b2:xyz")
    assert not H.is_digest(5)
    assert H.short_digest(d, 8) == d[:3 + 8]
    assert H.is_digest("sha256:" + "0" * 64)
