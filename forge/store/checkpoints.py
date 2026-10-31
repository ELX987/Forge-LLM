"""Checkpoint series on top of the artifact store: keyframes + XOR-delta chains, exact restore, retention.

Format
------
Every ``save`` produces one signed manifest (``kind="checkpoint"``).  The state is an arbitrary tree of dicts /
lists / tuples / scalars / tensors (e.g. ``{"model": sd, "optim": osd, "rng": ..., "step": 12}``); tensors are
flattened to ``a/b/c`` names, everything else becomes a JSON *skeleton* stored in ``manifest.extra["struct"]``.

Per tensor the manifest records how its bytes are stored (``extra["entries"][name]``):

``raw``   full bytes (content-addressed, so identical tensors / chunks dedup across checkpoints and runs)
``flc``   full bytes entropy-coded with :mod:`forge.store.lossless` (``keyframe_codec="lossless"``)
``same``  bit-identical to the previous checkpoint's tensor: nothing stored (frozen layers, LoRA base weights)
``xor``   ``zstd(byte_shuffle(bytes XOR previous_bytes))`` - consecutive training steps share most exponent/high
          mantissa bytes so the delta is mostly zeros

A *keyframe* checkpoint stores every tensor as ``raw``/``flc``; the next ``keyframe_every-1`` checkpoints are deltas
against their predecessor.  A delta whose compressed payload would not beat ``delta_max_ratio`` falls back to
``raw`` for that tensor (which also shortens the chain).  Restore walks ``base`` links back to the nearest full
copy of each tensor, applies the XORs forward and **verifies the content hash of every tensor** - restore is
bit-for-bit exact or raises :class:`~forge.core.errors.IntegrityError`.

Each delta manifest holds a hard ``chain`` reference to its predecessor in the store, so garbage collection can
never remove a keyframe (or intermediate delta) that a live checkpoint still needs.

The bytes of the previous checkpoint are cached in RAM by the writer (``cache_base=True``) to avoid re-reading the
chain; set ``cache_base=False`` for huge states (the base is then re-materialised from the store on each save).

``save_async`` snapshots the state synchronously (cheap memcpy) and encodes / writes on a background thread, so the
training loop is blocked only for the snapshot.  Call ``flush()`` before exiting.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from forge.core.errors import ConfigError, IntegrityError, NotFound, StoreError
from forge.core.schemas import ArtifactManifest, TensorEntry
from forge.core.util import now

log = logging.getLogger("forge.store.checkpoints")

try:  # pragma: no cover - environment dependent
    import zstandard as _zstd
except Exception:  # pragma: no cover
    _zstd = None


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------
class RetentionPolicy(BaseModel):
    """Which checkpoints of a run survive. Dropped checkpoints are released; GC reclaims what nothing else needs."""

    keep_last: int = Field(default=3, ge=1, le=1000, description="Always keep the N most recent checkpoints.")
    keep_keyframes: bool = Field(default=True, description="Keep every keyframe (full-copy) checkpoint (restore anchors).")
    max_keyframes: int | None = Field(default=None, ge=1, description="If set, keep only the newest N keyframes (unless tagged).")
    keep_best: int = Field(default=1, ge=0, le=100, description="Keep the K best checkpoints by validation metric.")
    best_mode: Literal["min", "max"] = Field(default="min", description="Is a lower (min, e.g. val loss) or higher (max, e.g. accuracy) metric better.")
    keep_tags: list[str] = Field(default_factory=lambda: ["pre_anneal", "final"], description="Tags that protect a checkpoint forever (pre-anneal LR decay snapshot, final).")
    keep_every_n_steps: int | None = Field(default=None, ge=1, description="Also keep checkpoints whose step is a multiple of N (milestones).")


class CheckpointConfig(BaseModel):
    """Checkpoint storage settings (embedded by the training configs)."""

    keyframe_every: int = Field(default=10, ge=1, le=10000, description="Every Nth save is a full keyframe; the saves in between are XOR deltas.")
    zstd_level: int = Field(default=3, ge=1, le=19, description="zstd level for delta payloads (3 is fast; higher saves a little more).")
    keyframe_codec: Literal["raw", "lossless"] = Field(default="raw", description="raw = chunk-deduplicated bytes; lossless = exponent-split entropy coding of float tensors (smaller, ~30-70 MB/s in numpy).")
    delta_max_ratio: float = Field(default=0.9, gt=0.0, le=1.0, description="A delta tensor is stored as a delta only if compressed size <= this fraction of the raw size.")
    use_dictionary: bool = Field(default=False, description="Train a zstd dictionary on the first delta and reuse it (helps many small tensors).")
    cache_base: bool = Field(default=True, description="Keep the previous checkpoint's bytes in RAM for fast delta encoding (costs one extra copy of the state).")
    retention: RetentionPolicy = Field(default_factory=RetentionPolicy, description="Which checkpoints to keep.")


# --------------------------------------------------------------------------------------------
# tree flatten / unflatten
# --------------------------------------------------------------------------------------------
class _TRec:
    __slots__ = ("u8", "dtype", "shape", "tk", "isz", "shared")

    def __init__(self, u8: np.ndarray, dtype: str, shape: tuple[int, ...], tk: str, isz: int, shared: bool) -> None:
        # ``shared``: u8 may alias memory the caller keeps mutating (CPU tensors) and must be copied before it is retained
        self.u8, self.dtype, self.shape, self.tk, self.isz, self.shared = u8, dtype, shape, tk, isz, shared


def _is_tensor(o: Any) -> bool:
    return hasattr(o, "detach") and hasattr(o, "cpu") and hasattr(o, "dtype") and hasattr(o, "shape")


def _tensor_record(t: Any) -> _TRec:
    import torch

    if _is_tensor(t):
        if t.layout != torch.strided:
            raise StoreError("sparse tensors cannot be checkpointed; densify them first")
        on_cpu = t.device.type == "cpu"
        x = t.detach().cpu().contiguous()
        flat = x.reshape(-1)
        u8 = flat.view(torch.uint8).numpy() if flat.numel() else np.zeros(0, dtype=np.uint8)
        return _TRec(u8, str(x.dtype).split(".", 1)[1], tuple(x.shape), "torch", x.element_size(), on_cpu)
    a = np.ascontiguousarray(t)
    return _TRec(a.reshape(-1).view(np.uint8), a.dtype.name, tuple(a.shape), "np", a.dtype.itemsize, True)


def _enc_scalar(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, tuple):
        return {"__tuple__": [_enc_scalar(x) for x in v]}
    raise StoreError(f"unsupported dict key {v!r} ({type(v).__name__})")


def _dec_scalar(v: Any) -> Any:
    if isinstance(v, dict) and "__tuple__" in v:
        return tuple(_dec_scalar(x) for x in v["__tuple__"])
    return v


def _flatten(obj: Any, path: str, out: dict[str, _TRec]) -> Any:
    if _is_tensor(obj) or isinstance(obj, np.ndarray):
        if path in out:
            raise StoreError(f"duplicate tensor path {path!r} in checkpoint state")
        out[path] = _tensor_record(obj)
        return {"__t__": path}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        # always an ordered pair list: the manifest JSON is key-sorted, so a plain dict would lose insertion order
        return {"__dict__": [[_enc_scalar(k), _flatten(v, f"{path}/{k}" if path else str(k), out)] for k, v in obj.items()]}
    if isinstance(obj, list):
        return [_flatten(v, f"{path}/{i}" if path else str(i), out) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return {"__tuple__": [_flatten(v, f"{path}/{i}" if path else str(i), out) for i, v in enumerate(obj)]}
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:  # numpy / torch scalar
        return obj.item()
    if type(obj).__name__ == "dtype" and "torch" in type(obj).__module__:
        return {"__tdtype__": str(obj).split(".", 1)[1]}
    if type(obj).__name__ == "Size":
        return {"__tuple__": [int(x) for x in obj]}
    raise StoreError(f"cannot checkpoint object of type {type(obj).__name__} at {path!r}")


def _unflatten(sk: Any, tensors: dict[str, Any]) -> Any:
    if isinstance(sk, list):
        return [_unflatten(v, tensors) for v in sk]
    if isinstance(sk, dict):
        if "__t__" in sk:
            return tensors[sk["__t__"]]
        if "__tuple__" in sk:
            return tuple(_unflatten(v, tensors) for v in sk["__tuple__"])
        if "__dict__" in sk:
            return {_dec_scalar(k): _unflatten(v, tensors) for k, v in sk["__dict__"]}
        if "__bytes__" in sk:
            return base64.b64decode(sk["__bytes__"])
        if "__tdtype__" in sk:
            import torch

            return getattr(torch, sk["__tdtype__"])
        return {k: _unflatten(v, tensors) for k, v in sk.items()}
    return sk


def _u8_to_obj(buf: bytearray, dtype: str, shape: list[int], tk: str, device: Any = None) -> Any:
    n = 1
    for s in shape:
        n *= int(s)
    if tk == "np":
        return np.frombuffer(bytes(buf), dtype=np.dtype(dtype)).reshape(shape).copy() if n else np.zeros(shape, dtype=np.dtype(dtype))
    import torch

    tdt = getattr(torch, dtype)
    if n == 0:
        t = torch.empty(tuple(shape), dtype=tdt)
    else:
        t = torch.frombuffer(buf, dtype=torch.uint8).view(tdt).reshape(tuple(shape))
    return t.to(device) if device is not None else t


def _shuffle(x: np.ndarray, isz: int) -> np.ndarray:
    if isz <= 1 or x.size % isz:
        return x
    return x.reshape(-1, isz).T.copy().reshape(-1)


def _unshuffle(x: np.ndarray, isz: int) -> np.ndarray:
    if isz <= 1 or x.size % isz:
        return x
    return x.reshape(isz, -1).T.copy().reshape(-1)


# --------------------------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------------------------
def _get_store(store: Any) -> Any:
    if store is not None:
        return store
    from forge.store import get_store

    return get_store()


class CheckpointReader:
    """Read side of a checkpoint series (``run_id``)."""

    def __init__(self, store: Any = None, run_id: str = "default") -> None:
        self.store = _get_store(store)
        self.run_id = run_id
        self._zdict_cache: dict[str, Any] = {}

    # -- index ------------------------------------------------------------------------------
    def rows(self, include_dropped: bool = False) -> list[dict[str, Any]]:
        q = "SELECT * FROM store_ckpt WHERE run_id=?" + ("" if include_dropped else " AND dropped=0") + " ORDER BY step"
        rows = self.store.db.query(q, (self.run_id,))
        for r in rows:
            r["tags"] = json.loads(r["tags_json"])
            r["is_keyframe"] = bool(r["is_keyframe"])
            r["dropped"] = bool(r["dropped"])
        return rows

    def steps(self, include_dropped: bool = False) -> list[int]:
        return [r["step"] for r in self.rows(include_dropped)]

    def best_mode(self) -> str:
        return self.store.db.kv_get(f"ckpt.{self.run_id}.best_mode", "min")

    def resolve(self, step: int | str = "latest") -> dict[str, Any]:
        """Index row for ``step`` (int), ``"latest"``, ``"best"``, ``"final"``, ``"pre_anneal"`` or ``"tag:<name>"``."""
        if isinstance(step, str) and step.lstrip("-").isdigit():
            step = int(step)
        if isinstance(step, int):
            r = self.store.db.query_one("SELECT * FROM store_ckpt WHERE run_id=? AND step=?", (self.run_id, step))
            if r is None or not self.store.has(r["manifest_hash"]):
                raise NotFound(f"no checkpoint at step {step} for run {self.run_id!r}")
            return self._fix(r)
        rows = self.rows()
        if not rows:
            raise NotFound(f"run {self.run_id!r} has no checkpoints")
        if step == "latest":
            return rows[-1]
        if step == "best":
            scored = [r for r in rows if r["metric"] is not None]
            if not scored:
                raise NotFound("no checkpoint has a metric recorded; cannot pick 'best'")
            key = (lambda r: r["metric"]) if self.best_mode() == "min" else (lambda r: -r["metric"])
            return min(scored, key=key)
        tag = step[4:] if step.startswith("tag:") else step
        tagged = [r for r in rows if tag in r["tags"]]
        if tagged:
            return tagged[-1]
        if tag == "final":
            return rows[-1]
        raise NotFound(f"no checkpoint tagged {tag!r} in run {self.run_id!r}")

    @staticmethod
    def _fix(r: dict[str, Any]) -> dict[str, Any]:
        r["tags"] = json.loads(r["tags_json"])
        r["is_keyframe"] = bool(r["is_keyframe"])
        r["dropped"] = bool(r["dropped"])
        return r

    def manifest(self, step: int | str = "latest") -> ArtifactManifest:
        return self.store.get_manifest(self.resolve(step)["manifest_hash"])

    def info(self, step: int | str = "latest") -> dict[str, Any]:
        r = self.resolve(step)
        m = self.store.get_manifest(r["manifest_hash"])
        return {**{k: r[k] for k in ("run_id", "step", "manifest_hash", "is_keyframe", "metric", "tags", "dropped", "logical_bytes", "created_at")},
                "meta": m.extra.get("meta", {}), "tensors": len(m.tensors), "chain_length": len(self._chain(r["manifest_hash"])),
                "stored_bytes": self.store.stored_size(r["manifest_hash"])}

    # -- restore ----------------------------------------------------------------------------
    def _chain(self, manifest_hash: str) -> list[ArtifactManifest]:
        """Manifests from the nearest keyframe to ``manifest_hash`` (inclusive, oldest first)."""
        chain: list[ArtifactManifest] = []
        cur: str | None = manifest_hash
        seen: set[str] = set()
        while cur is not None:
            if cur in seen:
                raise IntegrityError("cycle in checkpoint chain")
            seen.add(cur)
            m = self.store.get_manifest(cur)
            chain.append(m)
            if m.extra.get("is_keyframe"):
                break
            cur = m.extra.get("base")
            if cur is None:
                raise IntegrityError(f"checkpoint chain of {manifest_hash} has no keyframe")
        chain.reverse()
        return chain

    def _zdict(self, h: str) -> Any:
        if h not in self._zdict_cache:
            if _zstd is None:
                raise StoreError("zstandard is required to read dictionary-compressed checkpoints")
            self._zdict_cache[h] = _zstd.ZstdCompressionDict(self.store.get_bytes(h))
        return self._zdict_cache[h]

    def _tensor_bytes(self, chain: list[ArtifactManifest], name: str) -> bytearray:
        last = chain[-1]
        te = last.tensor(name)
        if te is None:
            raise NotFound(f"tensor {name!r} not in checkpoint")
        # find the newest manifest holding a full copy
        i0 = len(chain) - 1
        while i0 >= 0:
            e = chain[i0].extra["entries"].get(name)
            if e is None:
                raise IntegrityError(f"tensor {name!r} missing from chain member {i0}")
            if e["mode"] in ("raw", "flc"):
                break
            i0 -= 1
        if i0 < 0:
            raise IntegrityError(f"no full copy of tensor {name!r} in chain")
        e0 = chain[i0].extra["entries"][name]
        if e0["mode"] == "raw":
            cur = bytearray(self.store.get_bytes(e0["obj"]))
        else:
            from forge.store import lossless as L

            cur, _dt, _sh = L.decode_bytes(self.store.get_bytes(e0["obj"]))
        arr = np.frombuffer(cur, dtype=np.uint8)
        for j in range(i0 + 1, len(chain)):
            e = chain[j].extra["entries"][name]
            if e["mode"] == "same":
                continue
            if e["mode"] != "xor":
                raise IntegrityError(f"unexpected mode {e['mode']} in delta chain")
            payload = self.store.get_bytes(e["obj"])
            if _zstd is None:
                raise StoreError("zstandard is required to restore delta checkpoints")
            zd = chain[j].extra.get("zdict") if e.get("zd") else None
            dctx = _zstd.ZstdDecompressor(dict_data=self._zdict(zd)) if zd else _zstd.ZstdDecompressor()
            x = np.frombuffer(dctx.decompress(payload, max_output_size=e["nbytes"] + 16), dtype=np.uint8)
            x = _unshuffle(x, e["isz"])
            if x.size != arr.size:
                raise IntegrityError(f"delta for {name!r} has wrong length")
            np.bitwise_xor(arr, x, out=arr)
        if self.store._hash(memoryview(arr)) != te.hash:
            raise IntegrityError(f"restored tensor {name!r} does not match its recorded hash", details={"tensor": name})
        return cur

    def load_flat(self, step: int | str = "latest", *, device: Any = None, names: list[str] | None = None) -> dict[str, Any]:
        """``{tensor path: tensor}`` for a checkpoint (bit-exact, hash-verified)."""
        row = self.resolve(step)
        chain = self._chain(row["manifest_hash"])
        last = chain[-1]
        out: dict[str, Any] = {}
        for te in last.tensors:
            if names is not None and te.name not in names:
                continue
            buf = self._tensor_bytes(chain, te.name)
            out[te.name] = _u8_to_obj(buf, te.dtype, te.shape, last.extra["entries"][te.name].get("tk", "torch"), device)
        return out

    def load(self, step: int | str = "latest", *, device: Any = None) -> Any:
        """The full saved state tree with tensors restored bit-for-bit (CPU, or ``device``)."""
        row = self.resolve(step)
        chain = self._chain(row["manifest_hash"])
        last = chain[-1]
        tensors: dict[str, Any] = {}
        for te in last.tensors:
            buf = self._tensor_bytes(chain, te.name)
            tensors[te.name] = _u8_to_obj(buf, te.dtype, te.shape, last.extra["entries"][te.name].get("tk", "torch"), device)
        return _unflatten(last.extra["struct"], tensors)

    def load_meta(self, step: int | str = "latest") -> dict[str, Any]:
        return dict(self.manifest(step).extra.get("meta", {}))

    def raw_state(self, step: int | str = "latest") -> dict[str, tuple[np.ndarray, str, tuple[int, ...], str, int]]:
        """``{name: (u8 array, dtype, shape, tk, itemsize)}`` - used by the writer to resume deltas."""
        row = self.resolve(step)
        chain = self._chain(row["manifest_hash"])
        last = chain[-1]
        out = {}
        for te in last.tensors:
            ent = last.extra["entries"][te.name]
            out[te.name] = (np.frombuffer(self._tensor_bytes(chain, te.name), dtype=np.uint8), te.dtype, tuple(te.shape), ent.get("tk", "torch"), ent["isz"])
        return out

    def verify(self, step: int | str = "latest") -> bool:
        """Restore every tensor of the checkpoint and compare hashes (raises nothing; returns False on failure)."""
        try:
            row = self.resolve(step)
            chain = self._chain(row["manifest_hash"])
            for te in chain[-1].tensors:
                self._tensor_bytes(chain, te.name)
            return not self.store.check(row["manifest_hash"])
        except Exception as e:
            log.warning("checkpoint verify failed: %s", e)
            return False

    def verify_all(self) -> dict[str, Any]:
        bad = [s for s in self.steps() if not self.verify(s)]
        return {"run_id": self.run_id, "checked": len(self.steps()), "bad_steps": bad, "ok": not bad}

    def export_safetensors(self, step: int | str, dest: str | Path, *, key: str = "model", config: dict[str, Any] | None = None) -> Path:
        """Write ``tree[key]`` (a flat or nested dict of tensors) to ``dest/model.safetensors`` (+ config.json)."""
        from safetensors.torch import save_file

        tree = self.load(step)
        sub = tree[key] if key else tree
        flat: dict[str, Any] = {}

        def walk(o: Any, p: str) -> None:
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{p}.{k}" if p else str(k))
            elif _is_tensor(o):
                flat[p] = o.contiguous().clone()

        walk(sub, "")
        if not flat:
            raise StoreError(f"checkpoint has no tensors under {key!r}")
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        save_file(flat, str(d / "model.safetensors"), metadata={"format": "pt"})
        if config:
            (d / "config.json").write_text(json.dumps(config, indent=2), "utf-8")
        return d


def list_checkpoints(store: Any = None, run_id: str | None = None, *, include_dropped: bool = False,
                     with_sizes: bool = False, limit: int = 1000) -> list[dict[str, Any]]:
    """Index rows of checkpoints for the API/UI (all runs when ``run_id`` is None)."""
    st = _get_store(store)
    q = "SELECT * FROM store_ckpt" + (" WHERE run_id=?" if run_id else "")
    if not include_dropped:
        q += (" AND" if run_id else " WHERE") + " dropped=0"
    q += " ORDER BY created_at DESC, step DESC LIMIT ?"
    rows = st.db.query(q, ((run_id, limit) if run_id else (limit,)))
    for r in rows:
        CheckpointReader._fix(r)
        if with_sizes:
            try:
                r["stored_bytes"] = st.stored_size(r["manifest_hash"])
            except Exception:
                r["stored_bytes"] = None
    return rows


def list_runs(store: Any = None) -> list[dict[str, Any]]:
    st = _get_store(store)
    return st.db.query(
        "SELECT run_id, COUNT(*) AS checkpoints, MAX(step) AS last_step, SUM(logical_bytes) AS logical_bytes, MAX(created_at) AS updated_at "
        "FROM store_ckpt WHERE dropped=0 GROUP BY run_id ORDER BY updated_at DESC"
    )


# --------------------------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------------------------
class _Prev:
    __slots__ = ("manifest_hash", "step", "since_kf", "keyframe_hash", "state", "zdict_hash")

    def __init__(self, manifest_hash: str, step: int, since_kf: int, keyframe_hash: str, state: dict[str, Any] | None,
                 zdict_hash: str | None) -> None:
        self.manifest_hash, self.step, self.since_kf, self.keyframe_hash = manifest_hash, step, since_kf, keyframe_hash
        self.state, self.zdict_hash = state, zdict_hash


class CheckpointWriter:
    """Write side. ``save`` is synchronous; ``save_async`` returns a ``Future`` resolving to the manifest hash."""

    def __init__(self, store: Any = None, run_id: str = "default", *, keyframe_every: int | None = None,
                 config: CheckpointConfig | None = None, retention: RetentionPolicy | None = None,
                 best_mode: str | None = None) -> None:
        self.store = _get_store(store)
        self.run_id = run_id
        cfg = config or CheckpointConfig()
        if keyframe_every is not None:
            cfg = cfg.model_copy(update={"keyframe_every": int(keyframe_every)})
        if retention is not None:
            cfg = cfg.model_copy(update={"retention": retention})
        if best_mode is not None:
            cfg = cfg.model_copy(update={"retention": cfg.retention.model_copy(update={"best_mode": best_mode})})
        self.cfg = cfg
        self.reader = CheckpointReader(self.store, run_id)
        self._lock = threading.RLock()
        self._exec: ThreadPoolExecutor | None = None
        self._pending: list[Future] = []
        self._prev: _Prev | None = None
        self._zdict: bytes | None = None
        self._zdict_hash: str | None = None
        self.last_stats: dict[str, Any] = {}
        self.store.db.kv_set(f"ckpt.{run_id}.best_mode", cfg.retention.best_mode)

    # -- public ----------------------------------------------------------------------------
    def save(self, step: int, state: Any, meta: dict[str, Any] | None = None, *, keyframe_every: int | None = None,
             metric: float | None = None, tags: list[str] | tuple[str, ...] = (), force_keyframe: bool = False) -> str:
        """Store ``state`` (tree of tensors/scalars) as checkpoint ``step``. Returns the manifest hash."""
        flat: dict[str, _TRec] = {}
        sk = _flatten(state, "", flat)
        return self._save_flat(int(step), flat, sk, meta, metric, list(tags), force_keyframe, keyframe_every)

    def save_async(self, step: int, state: Any, meta: dict[str, Any] | None = None, *, keyframe_every: int | None = None,
                   metric: float | None = None, tags: list[str] | tuple[str, ...] = (), force_keyframe: bool = False) -> Future:
        """Snapshot now, encode+write on a background thread (one at a time, in submission order)."""
        flat: dict[str, _TRec] = {}
        sk = _flatten(state, "", flat)
        for rec in flat.values():
            if rec.shared:  # the live tensor may be mutated by the next optimiser step
                rec.u8 = rec.u8.copy()
                rec.shared = False
        meta = json.loads(json.dumps(meta)) if meta else meta
        with self._lock:
            if self._exec is None:
                self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ckpt-{self.run_id}")
            fut = self._exec.submit(self._save_flat, int(step), flat, sk, meta, metric, list(tags), force_keyframe, keyframe_every)
            self._pending.append(fut)
            if len(self._pending) > 4096:  # bound memory for very long runs that never flush; keep failures
                self._pending = [f for f in self._pending if not f.done() or f.exception() is not None]
        return fut

    def flush(self) -> list[str]:
        """Wait for async saves; re-raises the first failure. Returns the manifest hashes in order."""
        with self._lock:
            pend = list(self._pending)
            self._pending = []
        return [f.result() for f in pend]

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._exec is not None:
                self._exec.shutdown(wait=True)
                self._exec = None

    def __enter__(self) -> CheckpointWriter:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    def tag(self, step: int, tag: str) -> None:
        """Add a tag (e.g. ``pre_anneal``, ``final``) to an existing checkpoint; tagged checkpoints are pinned."""
        row = self.reader.resolve(step)
        tags = sorted(set(row["tags"]) | {tag})
        self.store.db.execute("UPDATE store_ckpt SET tags_json=? WHERE run_id=? AND step=?", (json.dumps(tags), self.run_id, row["step"]))
        if tag in self.cfg.retention.keep_tags:
            self.store.pin(row["manifest_hash"], f"ckpt:{self.run_id}:{tag}")

    def apply_retention(self) -> list[int]:
        """Release checkpoints outside the retention policy. Returns the dropped steps."""
        pol = self.cfg.retention
        rows = self.reader.rows()
        if not rows:
            return []
        keep: set[int] = {r["step"] for r in rows[-pol.keep_last:]}
        kfs = [r for r in rows if r["is_keyframe"]]
        if pol.keep_keyframes:
            sel = kfs[-pol.max_keyframes:] if pol.max_keyframes else kfs
            keep |= {r["step"] for r in sel}
        if pol.keep_best:
            scored = [r for r in rows if r["metric"] is not None]
            scored.sort(key=lambda r: r["metric"], reverse=(pol.best_mode == "max"))
            keep |= {r["step"] for r in scored[: pol.keep_best]}
        for r in rows:
            if set(r["tags"]) & set(pol.keep_tags):
                keep.add(r["step"])
            if pol.keep_every_n_steps and r["step"] % pol.keep_every_n_steps == 0:
                keep.add(r["step"])
        dropped = []
        for r in rows:
            if r["step"] in keep:
                continue
            self.store.release(r["manifest_hash"])
            self.store.db.execute("UPDATE store_ckpt SET dropped=1 WHERE run_id=? AND step=?", (self.run_id, r["step"]))
            dropped.append(r["step"])
        return dropped

    # -- internals -------------------------------------------------------------------------
    def _prev_for(self, step: int) -> _Prev | None:
        row = self.store.db.query_one(
            "SELECT * FROM store_ckpt WHERE run_id=? AND step<? AND dropped=0 ORDER BY step DESC LIMIT 1", (self.run_id, step)
        )
        if row is None:
            return None
        if self._prev is not None and self._prev.manifest_hash == row["manifest_hash"] and (self._prev.state is not None or not self.cfg.cache_base):
            return self._prev
        if not self.store.has(row["manifest_hash"]):
            return None
        m = self.store.get_manifest(row["manifest_hash"])
        since = int(m.extra.get("since_keyframe", 0))
        kf = row["manifest_hash"] if m.extra.get("is_keyframe") else (m.extra.get("keyframe") or row["manifest_hash"])
        state = None
        if self.cfg.cache_base:
            raw = self.reader.raw_state(row["step"])
            state = {n: {"u8": a, "dtype": dt, "shape": sh, "tk": tk, "isz": isz, "hash": m.tensor(n).hash} for n, (a, dt, sh, tk, isz) in raw.items()}
        p = _Prev(row["manifest_hash"], row["step"], since, kf, state, m.extra.get("zdict"))
        if p.zdict_hash and self._zdict is None:
            try:
                self._zdict = self.store.get_bytes(p.zdict_hash)
                self._zdict_hash = p.zdict_hash
            except Exception:
                pass
        return p

    def _compressor(self) -> Any:
        if _zstd is None:
            raise StoreError("the zstandard package is required for delta checkpoints (pip install zstandard)")
        if self._zdict is not None:
            return _zstd.ZstdCompressor(level=self.cfg.zstd_level, dict_data=_zstd.ZstdCompressionDict(self._zdict)), True
        return _zstd.ZstdCompressor(level=self.cfg.zstd_level), False

    def _base_u8(self, prev: _Prev, name: str) -> np.ndarray | None:
        if prev.state is not None:
            b = prev.state.get(name)
            return b["u8"] if b else None
        try:
            chain = self.reader._chain(prev.manifest_hash)
            if chain[-1].tensor(name) is None:
                return None
            return np.frombuffer(self.reader._tensor_bytes(chain, name), dtype=np.uint8)
        except Exception:
            return None

    def _save_flat(self, step: int, flat: dict[str, _TRec], sk: Any, meta: dict[str, Any] | None, metric: float | None,
                   tags: list[str], force_keyframe: bool, keyframe_every: int | None) -> str:
        from forge.store import lossless as L

        with self._lock, self.store.writing(f"ckpt.{self.run_id}"):  # in-flight marker: a concurrent gc() widens its grace period
            t0 = time.perf_counter()
            st = self.store
            kf_every = int(keyframe_every or self.cfg.keyframe_every)
            prev = self._prev_for(step)
            is_kf = force_keyframe or prev is None or prev.since_kf + 1 >= kf_every
            comp = None
            used_dict = False
            entries: dict[str, dict[str, Any]] = {}
            tensors: list[TensorEntry] = []
            new_state: dict[str, Any] = {}
            payload_bytes = 0
            logical = 0
            counts = {"raw": 0, "flc": 0, "same": 0, "xor": 0}
            dict_samples: list[bytes] = []
            prev_manifest: ArtifactManifest | None = None
            for name, rec in flat.items():
                u8 = rec.u8
                nbytes = int(u8.size)
                logical += nbytes
                th = st._hash(memoryview(u8))
                ent: dict[str, Any] = {"mode": "raw", "obj": None, "nbytes": nbytes, "isz": rec.isz}
                if rec.tk != "torch":
                    ent["tk"] = rec.tk
                base_u8: np.ndarray | None = None
                base_hash: str | None = None
                if not is_kf and prev is not None:
                    if prev.state is not None:
                        base = prev.state.get(name)
                        if base is not None and base["dtype"] == rec.dtype and tuple(base["shape"]) == rec.shape:
                            base_u8, base_hash = base["u8"], base["hash"]
                    else:
                        if prev_manifest is None:
                            prev_manifest = st.get_manifest(prev.manifest_hash)
                        pte = prev_manifest.tensor(name)
                        if pte is not None and pte.dtype == rec.dtype and tuple(pte.shape) == rec.shape:
                            base_u8, base_hash = self._base_u8(prev, name), pte.hash
                stored = False
                if base_u8 is not None and base_hash == th:
                    ent["mode"] = "same"
                    counts["same"] += 1
                    stored = True
                elif base_u8 is not None and base_u8.size == nbytes and nbytes > 0:
                    x = np.bitwise_xor(u8, base_u8)
                    sh = _shuffle(x, rec.isz)
                    if comp is None:
                        comp, used_dict = self._compressor()
                    z = comp.compress(sh.tobytes())
                    if len(z) <= self.cfg.delta_max_ratio * nbytes:
                        ent["mode"] = "xor"
                        ent["obj"] = st.put_bytes(z, kind="ckpt_obj", compress="none")
                        if used_dict:
                            ent["zd"] = True
                        payload_bytes += len(z)
                        counts["xor"] += 1
                        stored = True
                        if self.cfg.use_dictionary and self._zdict is None and nbytes <= (1 << 22) and len(dict_samples) < 256:
                            dict_samples.append(sh.tobytes()[:65536])
                if not stored:
                    ent.update(self._store_full(st, L, rec, u8, th, counts))
                    payload_bytes += ent.pop("_written", nbytes)
                entries[name] = ent
                tensors.append(TensorEntry(name=name, dtype=rec.dtype, shape=list(rec.shape), hash=th, role="other"))
                if self.cfg.cache_base:
                    new_state[name] = {"u8": u8.copy() if rec.shared else u8, "dtype": rec.dtype, "shape": rec.shape, "tk": rec.tk, "isz": rec.isz, "hash": th}
            zdict_hash = None
            if used_dict and self._zdict_hash:
                zdict_hash = self._zdict_hash
            if self.cfg.use_dictionary and self._zdict is None and not is_kf and len(dict_samples) >= 8:
                d = L.train_dictionary(dict_samples)
                if d:
                    self._zdict = d
                    self._zdict_hash = st.put_bytes(d, kind="ckpt_obj")
            kf_hash = None if is_kf or prev is None else prev.keyframe_hash
            extra = {
                "run_id": self.run_id, "step": step, "is_keyframe": bool(is_kf), "base": None if is_kf or prev is None else prev.manifest_hash,
                "keyframe": kf_hash, "since_keyframe": 0 if is_kf or prev is None else prev.since_kf + 1,
                "struct": sk, "entries": entries, "meta": meta or {}, "metric": metric, "zdict": zdict_hash,
                "keyframe_codec": self.cfg.keyframe_codec, "format": "forge-ckpt-v1",
            }
            m = ArtifactManifest(kind="checkpoint", tensors=tensors, parents=[prev.manifest_hash] if (prev and not is_kf) else [], extra=extra)
            mh = st.put_manifest(m, chain_parent=extra["base"], logical_size=logical)
            # a re-save at an existing step (restart from an earlier checkpoint) abandons the later timeline
            for r in st.db.query("SELECT step, manifest_hash FROM store_ckpt WHERE run_id=? AND step>=? AND manifest_hash!=?", (self.run_id, step, mh)):
                st.release(r["manifest_hash"])
            st.db.execute("UPDATE store_ckpt SET dropped=1 WHERE run_id=? AND step>? ", (self.run_id, step))
            all_tags = sorted(set(tags))
            st.db.execute(
                "INSERT INTO store_ckpt(run_id, step, manifest_hash, is_keyframe, base_hash, keyframe_hash, metric, tags_json, dropped, logical_bytes, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,0,?,?) ON CONFLICT(run_id, step) DO UPDATE SET manifest_hash=excluded.manifest_hash, "
                "is_keyframe=excluded.is_keyframe, base_hash=excluded.base_hash, keyframe_hash=excluded.keyframe_hash, metric=excluded.metric, "
                "tags_json=excluded.tags_json, dropped=0, logical_bytes=excluded.logical_bytes, created_at=excluded.created_at",
                (self.run_id, step, mh, 1 if is_kf else 0, extra["base"], kf_hash or (mh if is_kf else None), metric, json.dumps(all_tags), logical, now()),
            )
            for t in all_tags:
                if t in self.cfg.retention.keep_tags:
                    st.pin(mh, f"ckpt:{self.run_id}:{t}")
            since = extra["since_keyframe"]
            self._prev = _Prev(mh, step, since, mh if (is_kf or prev is None) else prev.keyframe_hash, new_state if self.cfg.cache_base else None, zdict_hash)
            dropped = self.apply_retention()
            self.last_stats = {
                "step": step, "manifest": mh, "is_keyframe": bool(is_kf), "logical_bytes": logical, "payload_bytes": payload_bytes,
                "ratio": payload_bytes / logical if logical else 1.0, "modes": counts, "dropped_steps": dropped,
                "seconds": time.perf_counter() - t0,
            }
            return mh

    def _store_full(self, st: Any, L: Any, rec: _TRec, u8: np.ndarray, th: str, counts: dict[str, int]) -> dict[str, Any]:
        canon = None
        if self.cfg.keyframe_codec == "lossless" and rec.tk == "torch" and u8.size:
            try:
                c = L.canonical_dtype(rec.dtype)
                canon = c if (L.is_float_format(c) and c == rec.dtype) else None
            except ConfigError:
                canon = None
        if canon is not None:
            blob = L.encode_bytes(u8, canon, rec.shape)
            if len(blob) < u8.size:
                counts["flc"] += 1
                return {"mode": "flc", "obj": st.put_bytes(blob, kind="ckpt_obj", compress="none"), "_written": len(blob)}
        counts["raw"] += 1
        obj = st.put_bytes(u8.tobytes(), kind="ckpt_obj")
        return {"mode": "raw", "obj": obj, "_written": int(u8.size)}
