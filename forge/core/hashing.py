"""Content hashing. Digest strings look like ``blake3:<64 hex>`` (or ``b2:<64 hex>`` fallback).

``blake3`` is used when the package is installed; otherwise ``blake2b`` with a 32 byte digest and prefix ``b2:``.
Set ``FORGE_HASH_ALGO=b2`` to force the fallback. A store must record its algorithm and never mix.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

try:  # pragma: no cover - depends on environment
    import blake3 as _blake3
except Exception:  # pragma: no cover
    _blake3 = None

_CHUNK = 1 << 20
_DIGEST_RE = re.compile(r"^(blake3|b2|sha256):[0-9a-f]{64}$")


def hash_algo() -> str:
    """Prefix of the active algorithm: ``"blake3"`` or ``"b2"``."""
    want = os.environ.get("FORGE_HASH_ALGO", "").strip().lower()
    if want in ("b2", "blake2b"):
        return "b2"
    return "blake3" if _blake3 is not None else "b2"


def _new_hasher(algo: str):
    if algo == "blake3":
        if _blake3 is None:
            raise RuntimeError("blake3 requested but package not installed")
        return _blake3.blake3()
    if algo == "b2":
        return hashlib.blake2b(digest_size=32)
    if algo == "sha256":
        return hashlib.sha256()
    raise ValueError(f"unknown hash algo {algo!r}")


class Hasher:
    """Incremental hasher. ``hexdigest()`` returns the prefixed digest string."""

    def __init__(self, algo: str | None = None) -> None:
        self.algo = algo or hash_algo()
        self._h = _new_hasher(self.algo)

    def update(self, b: bytes | bytearray | memoryview) -> Hasher:
        self._h.update(b)
        return self

    def hexdigest(self) -> str:
        return f"{self.algo}:{self._h.hexdigest()}"

    def raw_hexdigest(self) -> str:
        return self._h.hexdigest()

    def copy(self) -> Hasher:
        other = Hasher.__new__(Hasher)
        other.algo = self.algo
        other._h = self._h.copy()
        return other


def hash_bytes(b: bytes | bytearray | memoryview, algo: str | None = None) -> str:
    return Hasher(algo).update(b).hexdigest()


def hash_file(path: str | os.PathLike, algo: str | None = None) -> str:
    h = Hasher(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _norm(obj: Any) -> Any:
    """Normalise to JSON-native structures with deterministic ordering."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return _norm(obj.value)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj
    if isinstance(obj, dict):
        return {str(k.value if isinstance(k, enum.Enum) else k): _norm(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_norm(v) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return "bytes:" + hash_bytes(bytes(obj))
    if isinstance(obj, Path):
        return obj.as_posix()
    if hasattr(obj, "model_dump"):  # pydantic v2
        return _norm(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _norm(dataclasses.asdict(obj))
    if hasattr(obj, "item") and hasattr(obj, "dtype") and getattr(obj, "ndim", 1) == 0:  # numpy / torch scalar
        return _norm(obj.item())
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):  # numpy array / torch tensor
        return {"__array__": str(obj.dtype), "shape": list(obj.shape), "data": _norm(obj.tolist())}
    raise TypeError(f"cannot canonicalise object of type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, no whitespace, floats via ``repr``, ASCII only."""
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=True)


def hash_json(obj: Any, algo: str | None = None) -> str:
    return hash_bytes(canonical_json(obj).encode("ascii"), algo)


def is_digest(s: Any) -> bool:
    """True when ``s`` looks like ``<algo>:<64 hex>``."""
    return isinstance(s, str) and _DIGEST_RE.match(s) is not None


def digest_algo(d: str) -> str:
    return d.split(":", 1)[0]


def digest_hex(d: str) -> str:
    return d.split(":", 1)[1]


def short_digest(d: str, n: int = 12) -> str:
    """``blake3:abcdef012345`` for display."""
    if ":" not in d:
        return d[:n]
    a, h = d.split(":", 1)
    return f"{a}:{h[:n]}"
