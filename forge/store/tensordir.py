"""Safetensors-directory ingestion with per-tensor addressing, and exact reconstruction.

``put_tensor_dir`` walks a model directory (HF layout: ``*.safetensors`` shards, ``config.json``, tokenizer files...).
For each safetensors shard the 8+N byte header (verbatim, so reconstruction is byte-exact) is stored as its own
object and **every tensor is stored as its own content-addressed blob** (chunked/deduplicated when large), so two
checkpoints that share frozen layers - or that differ in only part of a big tensor - share storage.  All other
files are stored as plain blobs.  The result is an :class:`~forge.core.schemas.ArtifactManifest` whose ``extra``
holds the file layout::

    extra["files"] = [{"path", "type": "safetensors", "header": <hash>, "tensors": [names in offset order], "size"},
                      {"path", "type": "raw", "hash", "size"}, ...]
    extra["enc"]   = {tensor_name: {"hash": <hash of codec blob>, "method": ...}}     # only with lossless=...

``materialize_artifact`` rebuilds the directory; every tensor is hash-verified while it is written.
"""
from __future__ import annotations

import io
import json
import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge.core.errors import IntegrityError, NotFound, StoreError
from forge.core.schemas import ArtifactManifest, TensorEntry

if TYPE_CHECKING:  # pragma: no cover
    from forge.store.store import ArtifactStore

_ST2CANON = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2", "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U64": "uint64",
    "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
}
_SHORT = {
    "bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float64": "fp64", "float8_e4m3fn": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
}
_ST_ITEM = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
            "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}
_SKIP_DIRS = {".git", ".cache", "__pycache__", ".ipynb_checkpoints"}


def _entry_dtype(st_dtype: str) -> str:
    c = _ST2CANON.get(st_dtype)
    return _SHORT.get(c, c) if c else st_dtype.lower()


def _role_for(name: str) -> str:
    n = name.lower()
    if n.endswith((".scales", ".scale", "_scale", ".weight_scale", ".weight_scale_2", ".input_scale")):
        return "scale"
    if n.endswith((".qzeros", ".zeros", ".zero_point")):
        return "zero"
    if n.endswith((".g_idx", ".perm", ".permutation")):
        return "perm"
    if "codebook" in n:
        return "codebook"
    return "payload"


def parse_safetensors_header(path: str | os.PathLike) -> dict[str, Any]:
    """Parse a safetensors header without loading data. Tensors are returned sorted by data offset.

    ``{"header_len", "data_start", "file_size", "metadata", "tensors": [{name, dtype, shape, begin, end}], "contiguous"}``"""
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise StoreError(f"{p}: not a safetensors file (too short)")
        (hl,) = struct.unpack("<Q", head)
        if hl > size - 8 or hl > 100 * 1024 * 1024:
            raise StoreError(f"{p}: implausible safetensors header length {hl}")
        hdr = json.loads(f.read(hl))
    meta = hdr.pop("__metadata__", None)
    tensors = []
    for name, v in hdr.items():
        b, e = v["data_offsets"]
        tensors.append({"name": name, "dtype": v["dtype"], "shape": list(v["shape"]), "begin": int(b), "end": int(e)})
    tensors.sort(key=lambda t: (t["begin"], t["end"], t["name"]))
    data_start = 8 + hl
    pos = 0
    contiguous = True
    for t in tensors:
        n = 1
        for s in t["shape"]:
            n *= s
        isz = _ST_ITEM.get(t["dtype"])
        if t["begin"] != pos or t["end"] < t["begin"] or (isz is not None and t["end"] - t["begin"] != n * isz):
            contiguous = False
        pos = t["end"]
    if pos != size - data_start:
        contiguous = False
    return {"header_len": hl, "data_start": data_start, "file_size": size, "metadata": meta, "tensors": tensors,
            "contiguous": contiguous}


class _Region(io.RawIOBase):
    """Read-only window ``[start, start+length)`` of an open file."""

    def __init__(self, f: Any, start: int, length: int) -> None:
        super().__init__()
        self._f = f
        self._start = start
        self._left = length

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        if self._left <= 0:
            return 0
        mv = memoryview(b).cast("B")
        want = min(len(mv), self._left)
        self._f.seek(self._start)
        got = self._f.readinto(mv[:want]) or 0
        self._start += got
        self._left -= got
        return got


def _iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for fn in sorted(files):
            if fn.startswith(".") or fn.endswith((".part", ".tmp")):
                continue
            out.append(Path(dirpath) / fn)
    return out


def put_tensor_dir(
    store: ArtifactStore,
    root: Path,
    *,
    kind: str,
    recipe_hash: str | None = None,
    parents: tuple[str, ...] = (),
    extra: dict[str, Any] | None = None,
    licence: str | None = None,
    calib_hash: str | None = None,
    hardware_families: list[str] | None = None,
    eval_report_hashes: list[str] | None = None,
    lossless: bool | str = False,
    model_cfg: dict[str, Any] | None = None,
) -> str:
    from forge.store import lossless as L

    if not root.is_dir():
        raise NotFound(f"{root} is not a directory")
    method = "auto" if lossless is True else (lossless or None)
    files_meta: list[dict[str, Any]] = []
    tensors: list[TensorEntry] = []
    enc: dict[str, dict[str, Any]] = {}
    seen_names: set[str] = set()
    cfg_json: dict[str, Any] | None = None
    for fp in _iter_files(root):
        rel = fp.relative_to(root).as_posix()
        info = None
        if fp.suffix == ".safetensors":
            try:
                info = parse_safetensors_header(fp)
            except Exception:
                info = None
            if info is not None and not info["contiguous"]:
                info = None  # unusual layout: keep byte-exact as an opaque blob
        if info is None:
            h = store.put_file(fp, kind="file")
            files_meta.append({"path": rel, "type": "raw", "hash": h, "size": fp.stat().st_size})
            if rel == "config.json":
                try:
                    cfg_json = json.loads(fp.read_text("utf-8"))
                except Exception:
                    cfg_json = None
            continue
        with open(fp, "rb") as f:
            header_blob = f.read(info["data_start"])
            header_hash = store.put_bytes(header_blob, kind="header")
            names: list[str] = []
            for t in info["tensors"]:
                name = t["name"]
                if name in seen_names:
                    raise StoreError(f"duplicate tensor name {name!r} across shards of {root}")
                seen_names.add(name)
                length = t["end"] - t["begin"]
                start = info["data_start"] + t["begin"]
                canon = _ST2CANON.get(t["dtype"])
                if method and canon and L.is_float_format(canon) and length > 0:
                    f.seek(start)
                    raw = f.read(length)
                    bio = io.BytesIO()
                    st = L.encode_stream(raw, canon, bio, shape=t["shape"], method=method)
                    eh = store.put_bytes(bio.getvalue(), kind="tensor")
                    th = st["hash"]
                    enc[name] = {"hash": eh, "method": st["method"], "ratio": round(st["ratio"], 6)}
                else:
                    th = store.put_stream(io.BufferedReader(_Region(f, start, length)), kind="tensor")  # type: ignore[arg-type]
                tensors.append(TensorEntry(name=name, dtype=_entry_dtype(t["dtype"]), shape=t["shape"], hash=th,
                                           role=_role_for(name)))  # type: ignore[arg-type]
                names.append(name)
        files_meta.append({"path": rel, "type": "safetensors", "header": header_hash, "tensors": names,
                           "size": info["file_size"], "st_dtypes": {t["name"]: t["dtype"] for t in info["tensors"]}})
    ex: dict[str, Any] = {"files": files_meta}
    if enc:
        ex["enc"] = enc
    if extra:
        for k, v in extra.items():
            if k in ("files", "enc"):
                raise StoreError(f"extra key {k!r} is reserved")
            ex[k] = v
    m = ArtifactManifest(
        kind=kind, model_cfg=model_cfg if model_cfg is not None else (cfg_json or {}), tensors=tensors,
        recipe_hash=recipe_hash, parents=list(parents), calib_hash=calib_hash, licence=licence,
        hardware_families=hardware_families or [], eval_report_hashes=eval_report_hashes or [], extra=ex,
    )
    logical = sum(f["size"] for f in files_meta)
    return store.put_manifest(m, logical_size=logical)


def _safe_dest(dest: Path, rel: str) -> Path:
    """``dest / rel`` after rejecting absolute paths, drive letters and ``..`` components (manifests may come from untrusted sources)."""
    r = str(rel).replace("\\", "/")
    parts = [p for p in r.split("/") if p not in ("", ".")]
    if not parts or r.startswith("/") or ":" in parts[0] or any(p == ".." for p in parts):
        raise StoreError(f"unsafe path {rel!r} in artifact manifest (absolute or parent-relative paths are refused)")
    out = dest.joinpath(*parts)
    root = os.path.realpath(dest)
    real = os.path.realpath(out)
    if os.path.commonpath([root, real]) != root:
        raise StoreError(f"path {rel!r} in artifact manifest escapes the destination directory")
    return out


def _tensor_blob_hash(m: ArtifactManifest, name: str) -> tuple[str, TensorEntry, dict[str, Any] | None]:
    te = m.tensor(name)
    if te is None:
        raise NotFound(f"tensor {name!r} not in artifact")
    e = (m.extra.get("enc") or {}).get(name)
    return (e["hash"] if e else te.hash), te, e


def _iter_tensor_bytes(store: ArtifactStore, blob_hash: str, enc: dict[str, Any] | None):
    if enc is not None:
        from forge.store import lossless as L

        with store.open(blob_hash) as f:
            yield from L.decode_iter(f, verify=True)
    else:
        with store.open(blob_hash) as f:
            while True:
                b = f.read(8 << 20)
                if not b:
                    break
                yield b


def get_tensor(store: ArtifactStore, manifest_hash: str, name: str) -> tuple[bytes, str, list[int]]:
    m = store.get_manifest(manifest_hash)
    blob, te, enc = _tensor_blob_hash(m, name)
    data = b"".join(_iter_tensor_bytes(store, blob, enc))
    if store._hash(data) != te.hash:
        raise IntegrityError(f"tensor {name} of {manifest_hash} failed hash verification")
    return data, te.dtype, list(te.shape)


def materialize_artifact(store: ArtifactStore, manifest_hash: str, dest: Path) -> Path:
    m = store.get_manifest(manifest_hash)
    files = m.extra.get("files")
    dest.mkdir(parents=True, exist_ok=True)
    if files is None:
        # generic manifest without a file layout: dump tensors individually + manifest.json
        for te in m.tensors:
            blob, _te, enc = _tensor_blob_hash(m, te.name)
            safe = "".join(c if (c.isalnum() or c in "._-") else "__" for c in te.name).lstrip(".") or "tensor"
            with open(dest / f"{safe}.bin", "wb") as out:
                for b in _iter_tensor_bytes(store, blob, enc):
                    out.write(b)
        (dest / "manifest.json").write_text(json.dumps(m.model_dump(mode="json"), indent=2), "utf-8")
        return dest
    for fm in files:
        out_path = _safe_dest(dest, fm["path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".part")
        try:
            if fm["type"] == "raw":
                store.materialize(fm["hash"], tmp)
            else:
                with open(tmp, "wb") as out:
                    out.write(store.get_bytes(fm["header"]))
                    for name in fm["tensors"]:
                        blob, te, enc = _tensor_blob_hash(m, name)
                        hasher = store._hasher()
                        for b in _iter_tensor_bytes(store, blob, enc):
                            hasher.update(b)
                            out.write(b)
                        if hasher.hexdigest() != te.hash:
                            raise IntegrityError(f"tensor {name} of {manifest_hash} failed hash verification")
            os.replace(tmp, out_path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
    return dest
