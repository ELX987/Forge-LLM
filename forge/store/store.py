"""Content-addressed, chunk-deduplicated, tiered artifact store with a sqlite lineage DAG (Plan KR-1 / CP-1).

Layout on disk (``root`` defaults to ``forge.core.paths.store_dir()``)::

    root/store.json                  algo, chunker parameters, key fingerprint
    root/store.key                   HMAC signing key (generated on first use unless FORGE_STORE_HMAC_KEY is set)
    root/{hot,warm,cold}/objects/ab/cdef...[.zst|.mf]     objects (tier dirs may live on different disks)
    root/{hot,warm,cold}/tmp/        staging for atomic writes
    <db>  store_objects / store_refs / store_manifests / store_edges / store_pins / store_ckpt   (see store/db.py)

Object kinds
------------
``obj``      bytes on disk named by their own content hash (small blobs and CDC chunks; optionally zstd'd)
``chunked``  a logical blob > ``inline_threshold`` (4 MiB) = ordered list of chunk hashes (rows in ``store_refs``)
``manifest`` a signed JSON :class:`~forge.core.schemas.ArtifactManifest`; its address is ``manifest.hash``

Addresses never change: moving an object between tiers only moves its file and updates the ``tier`` column.

Garbage collection roots are *pins* plus every ``rooted`` object (user-visible artifacts).  Internal objects
(chunks, per-tensor blobs, headers, delta payloads) are kept alive only through hard references
(``store_refs``: chunk membership, manifest members and delta-chain bases) so a delta chain's keyframe can never
be collected while a live delta references it.
"""
from __future__ import annotations

import bisect
import contextlib
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import shutil
import threading
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from forge.core import db as _coredb
from forge.core.errors import ConfigError, IntegrityError, NotFound, StoreError
from forge.core.hashing import Hasher, hash_algo, hash_bytes, is_digest
from forge.core.paths import store_dir
from forge.core.schemas import ArtifactManifest
from forge.core.util import now
from forge.store import db as _storedb  # noqa: F401  (registers migrations)
from forge.store.chunking import DEFAULT_PARAMS, ChunkerParams, MiB, chunk_offsets, iter_chunks

try:  # pragma: no cover - environment dependent
    import zstandard as _zstd
except Exception:  # pragma: no cover
    _zstd = None

log = logging.getLogger("forge.store")

TIERS = ("hot", "warm", "cold")
INTERNAL_KINDS = ("chunk", "tensor", "file", "header", "ckpt_obj")
GENESIS = "genesis"
_INTERNAL_SQL = "(" + ",".join(f"'{k}'" for k in INTERNAL_KINDS) + ")"


def _hex(h: str) -> str:
    return h.split(":", 1)[1] if ":" in h else h


class ChunkedReader(io.RawIOBase):
    """Seekable read-only view over the chunks of a logical blob."""

    def __init__(self, store: ArtifactStore, chunks: list[tuple[str, int]]) -> None:
        super().__init__()
        self._store = store
        self._chunks = chunks
        self._starts: list[int] = []
        acc = 0
        for _, sz in chunks:
            self._starts.append(acc)
            acc += sz
        self._size = acc
        self._pos = 0
        self._cache_i = -1
        self._cache: bytes = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def readinto(self, b: Any) -> int:
        mv = memoryview(b).cast("B")
        want = len(mv)
        got = 0
        while got < want and self._pos < self._size:
            i = bisect.bisect_right(self._starts, self._pos) - 1
            if i != self._cache_i:
                self._cache = self._store._read_obj(self._chunks[i][0])
                self._cache_i = i
            off = self._pos - self._starts[i]
            take = min(want - got, len(self._cache) - off)
            mv[got : got + take] = self._cache[off : off + take]
            got += take
            self._pos += take
        return got


class ArtifactStore:
    """See module docstring. All public methods are thread-safe (sqlite WAL + atomic file writes)."""

    def __init__(
        self,
        root: str | os.PathLike | None = None,
        *,
        chunk_params: ChunkerParams | None = None,
        inline_threshold: int | None = None,
        compress: str = "auto",
        tier_dirs: dict[str, str | os.PathLike] | None = None,
        verify_reads: bool = True,
        durable: bool = False,
        hmac_key: bytes | str | None = None,
    ) -> None:
        default_root = store_dir()
        self.root = Path(root).expanduser().resolve() if root is not None else default_root
        self.root.mkdir(parents=True, exist_ok=True)
        use_default_db = root is None or self.root == default_root.resolve()
        self.db = _coredb.get_db() if use_default_db else _coredb.get_db(self.root / "store.db")
        if compress not in ("auto", "zstd", "none"):
            raise ConfigError("compress must be one of auto|zstd|none")
        self.compress = compress if _zstd is not None else "none"
        self.verify_reads = verify_reads
        self.durable = durable
        self._tier_dirs = {t: Path(tier_dirs[t]) if tier_dirs and t in tier_dirs else self.root / t for t in TIERS}
        for t in TIERS:
            (self._tier_dirs[t] / "objects").mkdir(parents=True, exist_ok=True)
            (self._tier_dirs[t] / "tmp").mkdir(parents=True, exist_ok=True)
        self._explicit_key: bytes | None = None
        if hmac_key is not None:
            self._explicit_key = hmac_key.encode() if isinstance(hmac_key, str) else bytes(hmac_key)
        self._key_cache: bytes | None = None
        self._key_lock = threading.Lock()
        self._init_config(chunk_params, inline_threshold)

    # ------------------------------------------------------------------ config / bookkeeping
    def _init_config(self, chunk_params: ChunkerParams | None, inline_threshold: int | None) -> None:
        cfg_path = self.root / "store.json"
        active = hash_algo()
        cfg: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text("utf-8"))
            except Exception as e:  # pragma: no cover
                raise StoreError(f"corrupt {cfg_path}: {e}") from e
            if cfg.get("algo") and cfg["algo"] != active:
                raise StoreError(
                    f"store at {self.root} uses hash algo {cfg['algo']!r} but the active algo is {active!r}; "
                    f"set FORGE_HASH_ALGO={cfg['algo']} (a store never mixes algorithms)"
                )
        self.algo = cfg.get("algo", active)
        c = cfg.get("chunk", {})
        self.chunk_params = chunk_params or ChunkerParams(
            c.get("min_size", DEFAULT_PARAMS.min_size), c.get("bits", DEFAULT_PARAMS.bits),
            c.get("max_size", DEFAULT_PARAMS.max_size),
        )
        self.inline_threshold = int(inline_threshold if inline_threshold is not None else cfg.get("inline_threshold", 4 * MiB))
        if not cfg:
            cfg = {
                "version": 1, "algo": self.algo, "created_at": now(),
                "chunk": {"min_size": self.chunk_params.min_size, "bits": self.chunk_params.bits,
                          "max_size": self.chunk_params.max_size},
                "inline_threshold": self.inline_threshold,
            }
            tmp = cfg_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cfg, indent=2), "utf-8")
            os.replace(tmp, cfg_path)
        self._cfg = cfg

    def _hash(self, data: bytes | memoryview) -> str:
        return hash_bytes(data, self.algo)

    def _hasher(self) -> Hasher:
        return Hasher(self.algo)

    def _signing_key(self) -> bytes | None:
        if self._key_cache is not None:
            return self._key_cache
        with self._key_lock:
            if self._key_cache is not None:
                return self._key_cache
            env = os.environ.get("FORGE_STORE_HMAC_KEY", "").strip()
            if self._explicit_key:
                self._key_cache = self._explicit_key
            elif env:
                self._key_cache = env.encode()
            else:
                kp = self.root / "store.key"
                try:
                    if not kp.exists():
                        tmp = kp.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
                        tmp.write_text(secrets.token_hex(32), "utf-8")
                        try:
                            os.chmod(tmp, 0o600)
                        except OSError:
                            pass
                        try:
                            os.link(tmp, kp)  # atomic create-if-absent
                        except FileExistsError:
                            pass
                        except OSError:
                            if not kp.exists():
                                os.replace(tmp, kp)
                        finally:
                            try:
                                tmp.unlink()
                            except OSError:
                                pass
                    self._key_cache = kp.read_text("utf-8").strip().encode()
                except OSError as e:
                    log.warning("no signing key available (%s); manifests are hash-chained only", e)
                    return None
            return self._key_cache

    def _key_fingerprint(self) -> str | None:
        k = self._signing_key()
        return hashlib.sha256(b"fp|" + k).hexdigest()[:16] if k else None

    # ------------------------------------------------------------------ paths & raw IO
    def _tier_dir(self, tier: str) -> Path:
        if tier not in TIERS:
            raise ConfigError(f"unknown tier {tier!r}; choose one of {TIERS}")
        return self._tier_dirs[tier]

    def _path(self, h: str, tier: str, codec: str = "raw", otype: str = "obj") -> Path:
        x = _hex(h)
        suffix = ".mf" if otype == "manifest" else (".zst" if codec == "zstd" else "")
        return self._tier_dir(tier) / "objects" / x[:2] / (x[2:] + suffix)

    def _row(self, h: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM store_objects WHERE hash=?", (h,))

    def _require(self, h: str) -> dict[str, Any]:
        r = self._row(h)
        if r is None:
            raise NotFound(f"object {h} not in store", details={"hash": h})
        return r

    def _file_for(self, row: dict[str, Any]) -> Path | None:
        p = self._path(row["hash"], row["tier"], row["codec"], row["otype"])
        if p.exists():
            return p
        for t in TIERS:  # self-heal if the row's tier is stale
            q = self._path(row["hash"], t, row["codec"], row["otype"])
            if q.exists():
                if t != row["tier"]:
                    self.db.execute("UPDATE store_objects SET tier=? WHERE hash=?", (t, row["hash"]))
                return q
        return None

    def _atomic_write(self, dest: Path, data: bytes, tier: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tier_dir(tier) / "tmp" / (uuid.uuid4().hex + ".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                if self.durable:
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(tmp, dest)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def _encode_codec(self, data: bytes, compress: str | None) -> tuple[bytes, str]:
        mode = compress or self.compress
        if mode == "none" or _zstd is None or len(data) < 256:
            return data, "raw"
        cctx = _zstd.ZstdCompressor(level=1)
        if mode == "auto":
            sample = data[:65536]
            if len(cctx.compress(sample)) > 0.9 * len(sample):
                return data, "raw"
        z = cctx.compress(data)
        if mode == "auto" and len(z) > 0.95 * len(data):
            return data, "raw"
        return (z, "zstd") if len(z) < len(data) else (data, "raw")

    def _read_obj(self, h: str) -> bytes:
        row = self._require(h)
        if row["otype"] == "chunked":
            return b"".join(self._read_obj(c) for c, _ in self._chunk_list(h))
        p = self._file_for(row)
        if p is None:
            raise IntegrityError(f"object file for {h} is missing on disk", details={"hash": h})
        data = p.read_bytes()
        if row["codec"] == "zstd":
            if _zstd is None:
                raise StoreError("zstandard is required to read compressed objects")
            data = _zstd.ZstdDecompressor().decompress(data, max_output_size=max(row["size"], 1))
        return data

    def _chunk_list(self, h: str) -> list[tuple[str, int]]:
        rows = self.db.query("SELECT dst, size FROM store_refs WHERE src=? AND rtype='chunk' ORDER BY ord", (h,))
        return [(r["dst"], r["size"]) for r in rows]

    def _touch(self, h: str, kind: str, rooted: bool) -> bool:
        """Mark an existing object as used (keeps GC away) and upgrade visibility. False if the row is gone."""
        visible = 0 if kind in INTERNAL_KINDS else 1
        cur = self.db.execute(
            "UPDATE store_objects SET accessed_at=?, rooted=MAX(rooted, ?), "
            f"kind=CASE WHEN ?=1 AND kind IN {_INTERNAL_SQL} THEN ? ELSE kind END WHERE hash=?",
            (now(), 1 if rooted else 0, visible, kind, h),
        )
        return cur.rowcount > 0

    def _store_single(self, h: str, data: bytes, kind: str, rooted: bool, compress: str | None = None, tier: str = "hot") -> None:
        if self._touch(h, kind, rooted):
            row = self._row(h)
            if row is not None and row["otype"] == "obj" and self._file_for(row) is not None:
                return
        payload, codec = self._encode_codec(data, compress)
        dest = self._path(h, tier, codec)
        self._atomic_write(dest, payload, tier)
        t = now()
        self.db.execute(
            "INSERT INTO store_objects(hash, otype, kind, size, stored_size, codec, tier, rooted, nchunks, created_at, accessed_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(hash) DO UPDATE SET stored_size=excluded.stored_size, codec=excluded.codec, tier=excluded.tier, "
            "accessed_at=excluded.accessed_at, rooted=MAX(store_objects.rooted, excluded.rooted)",
            (h, "obj", kind, len(data), len(payload), codec, tier, 1 if rooted else 0, 0, t, t),
        )

    def _register_chunked(self, h: str, kind: str, rooted: bool, size: int, chunks: list[tuple[str, int]]) -> None:
        if self._touch(h, kind, rooted):
            row = self._row(h)
            if row is not None and row["otype"] == "chunked" and self.has(h):
                return
        t = now()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO store_objects(hash, otype, kind, size, stored_size, codec, tier, rooted, nchunks, created_at, accessed_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(hash) DO UPDATE SET accessed_at=excluded.accessed_at, "
                "rooted=MAX(store_objects.rooted, excluded.rooted)",
                (h, "chunked", kind, size, 0, "raw", "hot", 1 if rooted else 0, len(chunks), t, t),
            )
            self.db.execute("DELETE FROM store_refs WHERE src=?", (h,))
            self.db.executemany(
                "INSERT INTO store_refs(src, ord, dst, rtype, size) VALUES(?,?,?, 'chunk', ?)",
                [(h, i, c, s) for i, (c, s) in enumerate(chunks)],
            )

    # ------------------------------------------------------------------ blobs
    def put_bytes(self, data: bytes | bytearray | memoryview, *, kind: str = "blob", rooted: bool | None = None,
                  compress: str | None = None) -> str:
        """Store ``data``; returns its content hash. Blobs above ``inline_threshold`` (4 MiB) are chunked with
        content-defined 1-4 MiB chunks and deduplicated chunk-wise."""
        data = bytes(data) if not isinstance(data, bytes) else data
        rooted = (kind not in INTERNAL_KINDS) if rooted is None else rooted
        h = self._hash(data)
        if len(data) <= self.inline_threshold:
            self._store_single(h, data, kind, rooted, compress)
            return h
        if self._touch(h, kind, rooted) and self.has(h):
            return h
        chunks: list[tuple[str, int]] = []
        for off, ln in chunk_offsets(data, self.chunk_params):
            piece = data[off : off + ln]
            ch = self._hash(piece)
            self._store_single(ch, piece, "chunk", False, compress)
            chunks.append((ch, ln))
        self._register_chunked(h, kind, rooted, len(data), chunks)
        return h

    def put_stream(self, fp: BinaryIO, *, kind: str = "blob", rooted: bool | None = None, compress: str | None = None) -> str:
        """Store a binary stream without holding it in memory (chunk by chunk)."""
        rooted = (kind not in INTERNAL_KINDS) if rooted is None else rooted
        first = fp.read(self.inline_threshold + 1)
        if len(first) <= self.inline_threshold:
            h = self._hash(first)
            self._store_single(h, first, kind, rooted, compress)
            return h
        pending = [first]

        def read(n: int) -> bytes:
            if pending:
                return pending.pop()
            return fp.read(n)

        hasher = self._hasher()
        chunks: list[tuple[str, int]] = []
        total = 0
        for piece in iter_chunks(read, self.chunk_params):
            hasher.update(piece)
            ch = self._hash(piece)
            self._store_single(ch, piece, "chunk", False, compress)
            chunks.append((ch, len(piece)))
            total += len(piece)
        h = hasher.hexdigest()
        self._register_chunked(h, kind, rooted, total, chunks)
        return h

    def put_file(self, path: str | os.PathLike, *, kind: str = "blob", rooted: bool | None = None, compress: str | None = None) -> str:
        with open(path, "rb") as f:
            return self.put_stream(f, kind=kind, rooted=rooted, compress=compress)

    def open(self, h: str) -> BinaryIO:
        """Seekable binary reader (not hash-verified; use :meth:`get_bytes`/:meth:`verify` for that)."""
        row = self._require(h)
        self._note_access(row)
        if row["otype"] == "chunked":
            return io.BufferedReader(ChunkedReader(self, self._chunk_list(h)), 1 << 20)  # type: ignore[return-value]
        if row["otype"] == "manifest":
            raise StoreError(f"{h} is a manifest; use get_manifest()/materialize_artifact()")
        if row["codec"] == "raw":
            p = self._file_for(row)
            if p is None:
                raise IntegrityError(f"object file for {h} is missing on disk")
            return open(p, "rb")
        return io.BytesIO(self._read_obj(h))

    def _note_access(self, row: dict[str, Any]) -> None:
        if now() - row["accessed_at"] > 60:
            self.db.execute("UPDATE store_objects SET accessed_at=? WHERE hash=?", (now(), row["hash"]))

    def get_bytes(self, h: str, *, verify: bool | None = None) -> bytes:
        row = self._require(h)
        if row["otype"] == "manifest":
            raise StoreError(f"{h} is a manifest; use get_manifest()")
        self._note_access(row)
        data = self._read_obj(h)
        if self.verify_reads if verify is None else verify:
            if self._hash(data) != h:
                raise IntegrityError(f"content of {h} does not match its address", details={"hash": h})
        return data

    def materialize(self, h: str, dest: str | os.PathLike) -> Path:
        """Write the blob (or artifact, for a manifest hash) to ``dest``; verifies the hash while copying."""
        row = self._require(h)
        if row["otype"] == "manifest":
            return self.materialize_artifact(h, dest)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".{uuid.uuid4().hex[:6]}.part")
        hasher = self._hasher()
        try:
            with self.open(h) as src, open(tmp, "wb") as out:
                while True:
                    b = src.read(8 * MiB)
                    if not b:
                        break
                    hasher.update(b)
                    out.write(b)
            if self.verify_reads and hasher.hexdigest() != h:
                raise IntegrityError(f"materialize: content of {h} does not match its address")
            os.replace(tmp, dest)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        return dest

    def has(self, h: str) -> bool:
        if not isinstance(h, str):
            return False
        row = self._row(h)
        if row is None:
            return False
        if row["otype"] == "chunked":
            n = self.db.scalar(
                "SELECT COUNT(*) FROM store_refs r JOIN store_objects o ON o.hash=r.dst WHERE r.src=? AND r.rtype='chunk'", (h,), 0
            )
            return n == row["nchunks"]
        return self._file_for(row) is not None

    def size(self, h: str) -> int:
        """Logical size in bytes."""
        return int(self._require(h)["size"])

    def info(self, h: str) -> dict[str, Any]:
        r = self._require(h)
        r["pinned"] = self.is_pinned(h)
        return r

    def _closure(self, h: str, include_chain: bool = False) -> dict[str, dict[str, Any]]:
        """Rows reachable from ``h`` through hard references (including ``h``)."""
        seen: dict[str, dict[str, Any]] = {}
        stack = [h]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            row = self._row(cur)
            if row is None:
                continue
            seen[cur] = row
            q = "SELECT dst FROM store_refs WHERE src=?" + ("" if include_chain else " AND rtype!='chain'")
            stack.extend(r["dst"] for r in self.db.query(q, (cur,)))
        return seen

    def stored_size(self, h: str) -> int:
        """Bytes this object occupies on disk counting every shared chunk once (unique-on-disk)."""
        return int(sum(r["stored_size"] for r in self._closure(h).values()))

    # ------------------------------------------------------------------ manifests
    def _scan_digests(self, obj: Any, out: list[str]) -> None:
        if isinstance(obj, str):
            if is_digest(obj):
                out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._scan_digests(v, out)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._scan_digests(v, out)

    def put_manifest(self, m: ArtifactManifest, *, sign: bool = True, rooted: bool = True,
                     chain_parent: str | None = None, logical_size: int | None = None) -> str:
        """Store a manifest (address = ``m.hash``), sign it, register member references and lineage edges."""
        h = m.hash
        if self._touch(h, m.kind, rooted):
            r = self._row(h)
            if r is not None and r["otype"] == "manifest" and self._file_for(r) is not None:
                return h
        cands: list[str] = [t.hash for t in m.tensors]
        scanned: list[str] = []
        self._scan_digests(m.extra, scanned)
        for extra_field in (m.calib_hash, *m.eval_report_hashes):
            if extra_field:
                cands.append(extra_field)
        for c in scanned:  # manifests named in ``extra`` (delta bases...) are lineage, not members; chains use chain_parent
            r0 = self._row(c)
            if r0 is not None and r0["otype"] != "manifest":
                cands.append(c)
        members: list[str] = []
        seen: set[str] = set()
        for c in cands:
            if c != h and c not in seen and self._row(c) is not None:
                seen.add(c)
                members.append(c)
        if chain_parent is not None:
            if self._row(chain_parent) is None:
                raise NotFound(f"chain parent {chain_parent} not in store")
        payload = json.dumps(m.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
        dest = self._path(h, "hot", "raw", "manifest")
        self._atomic_write(dest, payload, "hot")
        member_rows = [self._row(c) for c in members]
        size = int(logical_size) if logical_size is not None else int(sum(r["size"] for r in member_rows if r))
        t = now()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO store_objects(hash, otype, kind, size, stored_size, codec, tier, rooted, nchunks, created_at, accessed_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(hash) DO UPDATE SET accessed_at=excluded.accessed_at, "
                "rooted=MAX(store_objects.rooted, excluded.rooted)",
                (h, "manifest", m.kind, size, len(payload), "raw", "hot", 1 if rooted else 0, 0, t, t),
            )
            self.db.execute("DELETE FROM store_refs WHERE src=?", (h,))
            refs = [(h, i, c, "member", (r["size"] if r else 0)) for i, (c, r) in enumerate(zip(members, member_rows))]
            if chain_parent is not None:
                refs.append((h, len(refs), chain_parent, "chain", 0))
            self.db.executemany("INSERT INTO store_refs(src, ord, dst, rtype, size) VALUES(?,?,?,?,?)", refs)
            if self.db.query_one("SELECT 1 AS x FROM store_manifests WHERE hash=?", (h,)) is None:
                self._sign(h, m.kind, sign)
            for p in m.parents:
                self.add_edge(h, p, recipe_hash=m.recipe_hash, relation="derived_from")
        return h

    def _sign(self, h: str, kind: str, sign: bool) -> None:
        prev = self.db.query_one("SELECT chain FROM store_manifests ORDER BY seq DESC LIMIT 1")
        prev_chain = prev["chain"] if prev else GENESIS
        chain = hash_bytes(f"{prev_chain}|{h}".encode(), self.algo)
        key = self._signing_key() if sign else None
        if key:
            sig = hmac.new(key, f"{h}|{chain}".encode(), hashlib.sha256).hexdigest()
            alg = "hmac-sha256+chain"
        else:
            sig, alg = "", "chain"
        self.db.execute(
            "INSERT INTO store_manifests(hash, kind, prev_chain, chain, sig, sig_alg, created_at) VALUES(?,?,?,?,?,?,?)",
            (h, kind, prev_chain, chain, sig, alg, now()),
        )
        fp = self._key_fingerprint()
        if fp and "key_fp" not in self._cfg:
            self._cfg["key_fp"] = fp
            try:
                tmp = (self.root / "store.json").with_suffix(".tmp")
                tmp.write_text(json.dumps(self._cfg, indent=2), "utf-8")
                os.replace(tmp, self.root / "store.json")
            except OSError:  # pragma: no cover
                pass

    def get_manifest(self, h: str, *, verify_signature: bool = False) -> ArtifactManifest:
        row = self._require(h)
        if row["otype"] != "manifest":
            raise StoreError(f"{h} is not a manifest")
        p = self._file_for(row)
        if p is None:
            raise IntegrityError(f"manifest file for {h} is missing")
        m = ArtifactManifest.model_validate(json.loads(p.read_bytes()))
        if m.hash != h:
            raise IntegrityError(f"manifest {h} content does not match its address (recomputed {m.hash})")
        if verify_signature:
            st = self.signature_status(h)
            if st == "invalid":
                raise IntegrityError(f"manifest {h} has an invalid signature")
        self._note_access(row)
        return m

    def signature_status(self, h: str) -> str:
        """``valid`` | ``invalid`` | ``chain_only`` | ``key_mismatch`` | ``unsigned`` for a manifest."""
        r = self.db.query_one("SELECT * FROM store_manifests WHERE hash=?", (h,))
        if r is None:
            return "unsigned"
        if hash_bytes(f"{r['prev_chain']}|{h}".encode(), self.algo) != r["chain"]:
            return "invalid"
        if r["sig_alg"] == "chain":
            return "chain_only"
        key = self._signing_key()
        fp = self._cfg.get("key_fp")
        if key is None or (fp and fp != self._key_fingerprint()):
            return "key_mismatch"
        want = hmac.new(key, f"{h}|{r['chain']}".encode(), hashlib.sha256).hexdigest()
        return "valid" if hmac.compare_digest(want, r["sig"]) else "invalid"

    def verify_chain(self) -> dict[str, Any]:
        """Walk the whole manifest hash-chain ledger; reports the first inconsistency (tamper evidence)."""
        prev = GENESIS
        n = 0
        for r in self.db.query("SELECT * FROM store_manifests ORDER BY seq"):
            n += 1
            if r["prev_chain"] != prev or hash_bytes(f"{prev}|{r['hash']}".encode(), self.algo) != r["chain"]:
                return {"ok": False, "entries": n, "broken_at": r["hash"], "seq": r["seq"]}
            prev = r["chain"]
        return {"ok": True, "entries": n, "head": prev}

    # ------------------------------------------------------------------ tensor directories
    def put_tensor_dir(self, dir: str | os.PathLike, *, kind: str, recipe_hash: str | None = None,
                       parents: Iterable[str] = (), extra: dict[str, Any] | None = None, **kw: Any) -> str:
        """Ingest a model directory (safetensors shards + config/tokenizer files) with per-tensor addressing.

        Extra keywords: ``licence``, ``calib_hash``, ``hardware_families``, ``eval_report_hashes``,
        ``lossless`` (False | True | method name; entropy-codes float tensors with :mod:`forge.store.lossless`)."""
        from forge.store.tensordir import put_tensor_dir

        with self.writing("tensordir"):  # ingesting a big model can outlast the GC grace period
            return put_tensor_dir(self, Path(dir), kind=kind, recipe_hash=recipe_hash, parents=tuple(parents), extra=extra, **kw)

    def materialize_artifact(self, manifest_hash: str, dest: str | os.PathLike) -> Path:
        """Reconstruct a loadable directory (bit-identical to what was ingested) at ``dest``."""
        from forge.store.tensordir import materialize_artifact

        return materialize_artifact(self, manifest_hash, Path(dest))

    def get_tensor(self, manifest_hash: str, name: str) -> tuple[bytes, str, list[int]]:
        """Raw bytes, dtype string and shape of one tensor of a tensor-dir artifact."""
        from forge.store.tensordir import get_tensor

        return get_tensor(self, manifest_hash, name)

    # ------------------------------------------------------------------ lineage / pins
    def add_edge(self, child: str, parent: str, recipe_hash: str | None = None, relation: str = "derived_from") -> None:
        if child == parent:
            raise ConfigError("an artifact cannot be its own parent")
        if self._reaches(parent, child):
            raise ConfigError(f"edge {child} <- {parent} would create a cycle in the lineage DAG")
        self.db.execute(
            "INSERT OR IGNORE INTO store_edges(child, parent, relation, recipe_hash, created_at) VALUES(?,?,?,?,?)",
            (child, parent, relation, recipe_hash, now()),
        )

    def _reaches(self, start: str, target: str) -> bool:
        """True when ``target`` is an ancestor of ``start`` (following child->parent edges)."""
        seen: set[str] = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c == target:
                return True
            if c in seen:
                continue
            seen.add(c)
            stack.extend(r["parent"] for r in self.db.query("SELECT parent FROM store_edges WHERE child=?", (c,)))
        return False

    def lineage(self, h: str, depth: int | None = None, direction: str = "both") -> dict[str, Any]:
        """Connected lineage DAG around ``h``: ``{"nodes": [...], "edges": [...]}`` (``direction``: up|down|both)."""
        if direction not in ("up", "down", "both"):
            raise ConfigError("direction must be up|down|both")
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        dist: dict[str, int] = {h: 0}
        frontier = [h]
        d = 0
        while frontier and (depth is None or d < depth):
            d += 1
            nxt: list[str] = []
            for n in frontier:
                found: list[dict[str, Any]] = []
                if direction in ("up", "both"):
                    found += [(r, r["parent"]) for r in self.db.query("SELECT * FROM store_edges WHERE child=?", (n,))]  # type: ignore[misc]
                if direction in ("down", "both"):
                    found += [(r, r["child"]) for r in self.db.query("SELECT * FROM store_edges WHERE parent=?", (n,))]  # type: ignore[misc]
                for r, other in found:  # type: ignore[misc]
                    edges[(r["child"], r["parent"], r["relation"])] = {
                        "child": r["child"], "parent": r["parent"], "relation": r["relation"], "recipe_hash": r["recipe_hash"],
                    }
                    if other not in dist:
                        dist[other] = d
                        nxt.append(other)
            frontier = nxt
        nodes = []
        for n, dd in dist.items():
            row = self._row(n)
            nodes.append({
                "hash": n, "present": row is not None, "kind": row["kind"] if row else "unknown",
                "size": row["size"] if row else None, "tier": row["tier"] if row else None,
                "pinned": self.is_pinned(n), "distance": dd,
                "created_at": row["created_at"] if row else None,
            })
        return {"root": h, "nodes": nodes, "edges": list(edges.values())}

    def pin(self, h: str, reason: str = "") -> None:
        self._require(h)
        self.db.execute(
            "INSERT INTO store_pins(hash, reason, created_at) VALUES(?,?,?) "
            "ON CONFLICT(hash) DO UPDATE SET reason=excluded.reason",
            (h, reason, now()),
        )

    def unpin(self, h: str) -> None:
        self.db.execute("DELETE FROM store_pins WHERE hash=?", (h,))

    def is_pinned(self, h: str) -> bool:
        return self.db.query_one("SELECT 1 AS x FROM store_pins WHERE hash=?", (h,)) is not None

    def pins(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT hash, reason, created_at FROM store_pins ORDER BY created_at")

    def release(self, h: str) -> None:
        """Drop an artifact's *rooted* status and pin: it becomes collectable once nothing references it."""
        self._require(h)
        self.unpin(h)
        self.db.execute("UPDATE store_objects SET rooted=0 WHERE hash=?", (h,))

    # ------------------------------------------------------------------ listing / stats
    def list_artifacts(self, kind: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        q = (
            "SELECT o.hash, o.kind, o.otype, o.size, o.stored_size, o.tier, o.created_at, o.accessed_at, o.nchunks, "
            "(SELECT COUNT(*) FROM store_pins p WHERE p.hash=o.hash) AS pinned FROM store_objects o WHERE o.rooted=1"
        )
        args: list[Any] = []
        if kind:
            q += " AND o.kind=?"
            args.append(kind)
        q += " ORDER BY o.created_at DESC, o.hash LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        rows = self.db.query(q, args)
        for r in rows:
            r["pinned"] = bool(r["pinned"])
        return rows

    def count_artifacts(self, kind: str | None = None) -> int:
        if kind:
            return int(self.db.scalar("SELECT COUNT(*) FROM store_objects WHERE rooted=1 AND kind=?", (kind,), 0))
        return int(self.db.scalar("SELECT COUNT(*) FROM store_objects WHERE rooted=1", (), 0))

    def stats(self) -> dict[str, Any]:
        """Logical bytes (what a naive one-copy-per-artifact layout would use) vs unique bytes on disk."""
        logical = int(self.db.scalar("SELECT COALESCE(SUM(size),0) FROM store_objects WHERE rooted=1", (), 0))
        unique = int(self.db.scalar("SELECT COALESCE(SUM(stored_size),0) FROM store_objects", (), 0))
        raw_unique = int(self.db.scalar("SELECT COALESCE(SUM(size),0) FROM store_objects WHERE otype='obj'", (), 0))
        per_kind: dict[str, Any] = {}
        for r in self.db.query("SELECT kind, rooted, COUNT(*) AS n, SUM(size) AS sz, SUM(stored_size) AS st FROM store_objects GROUP BY kind, rooted"):
            d = per_kind.setdefault(r["kind"], {"objects": 0, "rooted_objects": 0, "logical_bytes": 0, "stored_bytes": 0})
            d["objects"] += r["n"]
            d["stored_bytes"] += int(r["st"] or 0)
            if r["rooted"]:
                d["rooted_objects"] += r["n"]
                d["logical_bytes"] += int(r["sz"] or 0)
        per_tier: dict[str, Any] = {t: {"objects": 0, "bytes": 0} for t in TIERS}
        for r in self.db.query("SELECT tier, COUNT(*) AS n, SUM(stored_size) AS st FROM store_objects WHERE otype IN ('obj','manifest') GROUP BY tier"):
            per_tier[r["tier"]] = {"objects": r["n"], "bytes": int(r["st"] or 0)}
        otypes = {r["otype"]: r["n"] for r in self.db.query("SELECT otype, COUNT(*) AS n FROM store_objects GROUP BY otype")}
        return {
            "root": str(self.root), "algo": self.algo,
            "logical_bytes": logical, "naive_bytes": logical, "unique_bytes": unique,
            "uncompressed_unique_bytes": raw_unique,
            "dedup_ratio": (logical / unique) if unique else 1.0,
            "compaction_ratio": (unique / logical) if logical else 1.0,
            "saved_bytes": logical - unique,
            "objects": otypes, "artifacts": self.count_artifacts(),
            "pins": int(self.db.scalar("SELECT COUNT(*) FROM store_pins", (), 0)),
            "edges": int(self.db.scalar("SELECT COUNT(*) FROM store_edges", (), 0)),
            "per_kind": per_kind, "per_tier": per_tier,
            "chunker": {"min": self.chunk_params.min_size, "bits": self.chunk_params.bits, "max": self.chunk_params.max_size,
                        "inline_threshold": self.inline_threshold},
        }

    def tiers(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        st = self.stats()["per_tier"]
        for t in TIERS:
            d = self._tier_dir(t)
            try:
                free = shutil.disk_usage(d).free
            except OSError:
                free = None
            out[t] = {"path": str(d), "objects": st[t]["objects"], "bytes": st[t]["bytes"], "free_bytes": free}
        return out

    # ------------------------------------------------------------------ tiers
    def tier_of(self, h: str) -> str:
        return self._require(h)["tier"]

    def tier_summary(self, h: str) -> dict[str, int]:
        """Physical bytes of ``h``'s closure per tier."""
        out: dict[str, int] = {}
        for r in self._closure(h).values():
            if r["otype"] in ("obj", "manifest"):
                out[r["tier"]] = out.get(r["tier"], 0) + int(r["stored_size"])
        return out

    def _move(self, row: dict[str, Any], to_tier: str) -> None:
        if row["otype"] == "chunked":
            self.db.execute("UPDATE store_objects SET tier=? WHERE hash=?", (to_tier, row["hash"]))
            return
        if row["tier"] == to_tier:
            return
        src = self._file_for(row)
        dst = self._path(row["hash"], to_tier, row["codec"], row["otype"])
        if src is not None and src != dst:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                src.unlink()
            else:
                try:
                    os.replace(src, dst)
                except OSError:  # cross-device
                    tmp = self._tier_dir(to_tier) / "tmp" / (uuid.uuid4().hex + ".tmp")
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, dst)
                    src.unlink()
        self.db.execute("UPDATE store_objects SET tier=? WHERE hash=?", (to_tier, row["hash"]))

    def demote(self, h: str, to_tier: str) -> int:
        """Move ``h`` and everything it references (not delta-chain bases) to ``to_tier``. Addresses are unchanged.
        Returns the number of objects moved. Chunks shared with other artifacts follow the latest tier operation."""
        self._tier_dir(to_tier)
        n = 0
        for row in self._closure(h).values():
            if row["tier"] != to_tier:
                self._move(row, to_tier)
                n += 1
        return n

    def promote(self, h: str, to_tier: str = "hot") -> int:
        return self.demote(h, to_tier)

    def auto_demote(self, older_than_s: float, to_tier: str = "warm", kinds: Iterable[str] | None = None,
                    dry_run: bool = True) -> list[str]:
        """Demote rooted, unpinned artifacts not accessed for ``older_than_s`` seconds."""
        cutoff = now() - older_than_s
        q = "SELECT hash, kind FROM store_objects WHERE rooted=1 AND accessed_at<? AND tier!=? AND hash NOT IN (SELECT hash FROM store_pins)"
        rows = self.db.query(q, (cutoff, to_tier))
        ks = set(kinds) if kinds else None
        hs = [r["hash"] for r in rows if ks is None or r["kind"] in ks]
        if not dry_run:
            for x in hs:
                self.demote(x, to_tier)
        return hs

    # ------------------------------------------------------------------ verification
    def check(self, h: str, deep: bool = False) -> list[str]:
        """List of problems found for ``h`` (empty = healthy). ``deep`` re-hashes every referenced byte."""
        problems: list[str] = []
        row = self._row(h)
        if row is None:
            return [f"{h}: not in store"]
        if row["otype"] == "obj":
            p = self._file_for(row)
            if p is None:
                return [f"{h}: file missing"]
            try:
                data = self._read_obj(h)
            except Exception as e:
                return [f"{h}: unreadable ({e})"]
            if self._hash(data) != h:
                problems.append(f"{h}: content hash mismatch")
        elif row["otype"] == "chunked":
            hasher = self._hasher()
            for c, sz in self._chunk_list(h):
                cr = self._row(c)
                if cr is None:
                    problems.append(f"{h}: chunk {c} missing from index")
                    continue
                try:
                    data = self._read_obj(c)
                except Exception as e:
                    problems.append(f"{h}: chunk {c} unreadable ({e})")
                    continue
                if self._hash(data) != c:
                    problems.append(f"{h}: chunk {c} corrupt")
                if len(data) != sz:
                    problems.append(f"{h}: chunk {c} wrong size")
                hasher.update(data)
            if not problems and hasher.hexdigest() != h:
                problems.append(f"{h}: reassembled content hash mismatch")
        elif row["otype"] == "manifest":
            try:
                self.get_manifest(h)
            except Exception as e:
                problems.append(f"{h}: {e}")
            st = self.signature_status(h)
            if st == "invalid":
                problems.append(f"{h}: invalid signature / chain entry")
            for r in self.db.query("SELECT dst, rtype FROM store_refs WHERE src=?", (h,)):
                if self._row(r["dst"]) is None:
                    problems.append(f"{h}: referenced {r['rtype']} {r['dst']} is missing")
                elif deep and r["rtype"] == "member":
                    problems += self.check(r["dst"], deep=True)
        return problems

    def verify(self, h: str, deep: bool = False) -> bool:
        probs = self.check(h, deep=deep)
        for p in probs:
            log.warning("verify: %s", p)
        return not probs

    def verify_all(self, deep: bool = True, kinds: Iterable[str] | None = None) -> dict[str, Any]:
        """Verify every rooted artifact (+ the signature ledger). Used by the CLI / UI storage page."""
        ks = set(kinds) if kinds else None
        bad: dict[str, list[str]] = {}
        n = 0
        for r in self.db.query("SELECT hash, kind FROM store_objects WHERE rooted=1"):
            if ks and r["kind"] not in ks:
                continue
            n += 1
            probs = self.check(r["hash"], deep=deep)
            if probs:
                bad[r["hash"]] = probs
        chain = self.verify_chain()
        return {"checked": n, "ok": not bad and chain["ok"], "bad": bad, "chain": chain}

    # ------------------------------------------------------------------ gc
    def gc(self, dry_run: bool = True, *, grace_s: float = 300.0, strict_pins: bool = False) -> Any:
        """Mark-and-sweep. Roots = pins + rooted artifacts (``strict_pins=True``: pins only).

        Never collects pinned objects nor anything reachable (chunks, members, delta-chain bases) from a root.
        Objects touched within ``grace_s`` seconds are skipped (protects concurrent writers). Returns ``GCReport``."""
        from forge.store.collect import run_gc

        return run_gc(self, dry_run=dry_run, grace_s=grace_s, strict_pins=strict_pins)

    def delete(self, h: str, *, collect: bool = False) -> None:
        """Release ``h`` (and unpin); with ``collect`` immediately sweep unreferenced objects."""
        self.release(h)
        if collect:
            self.gc(dry_run=False, grace_s=0.0)

    # ------------------------------------------------------------------ in-flight writers
    INFLIGHT_PREFIX = "store.inflight."
    INFLIGHT_TTL_S = 6 * 3600.0

    @contextlib.contextmanager
    def writing(self, token: str) -> Iterator[None]:
        """Mark a long multi-object write (e.g. a checkpoint save) as in flight. While any fresh marker exists
        (``INFLIGHT_TTL_S``, so a crashed writer cannot block GC forever) a sweeping :meth:`gc` widens its grace period to the TTL, so
        payload objects written minutes before their manifest exists are never collected under the writer."""
        key = f"{self.INFLIGHT_PREFIX}{token}.{uuid.uuid4().hex[:8]}"
        self.db.kv_set(key, now())
        try:
            yield
        finally:
            try:
                self.db.kv_delete(key)
            except Exception:  # pragma: no cover
                pass

    def inflight_writers(self) -> int:
        """Number of fresh in-flight write markers."""
        cutoff = now() - self.INFLIGHT_TTL_S
        rows = self.db.query("SELECT updated_at FROM kv WHERE key LIKE ?", (self.INFLIGHT_PREFIX + "%",))
        return sum(1 for r in rows if r["updated_at"] >= cutoff)

    # ------------------------------------------------------------------ misc
    def iter_hashes(self, rooted_only: bool = False) -> Iterator[str]:
        q = "SELECT hash FROM store_objects" + (" WHERE rooted=1" if rooted_only else "")
        for r in self.db.query(q):
            yield r["hash"]

    def __repr__(self) -> str:  # pragma: no cover
        return f"ArtifactStore(root={str(self.root)!r}, algo={self.algo!r})"
