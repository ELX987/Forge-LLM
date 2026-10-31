"""/datasets : register, upload (multipart or raw chunked), preview, ingest."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from forge.api.common import lazy_module
from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import DatasetCreate, IngestRequest
from forge.core.errors import ConfigError, ForgeError, NotFound
from forge.core.paths import datasets_dir, forge_home
from forge.core.util import new_id, now
from forge.jobs.registry import get_job_type

router = APIRouter(prefix="/datasets", tags=["datasets"])

HAVE_MULTIPART = bool(importlib.util.find_spec("python_multipart") or importlib.util.find_spec("multipart"))
FORMATS = {".jsonl": "jsonl", ".ndjson": "jsonl", ".json": "json", ".txt": "text", ".md": "text", ".parquet": "parquet", ".csv": "csv", ".tsv": "csv", ".npy": "tokens", ".bin": "tokens"}
_CHUNK = 1 << 20


class PayloadTooLarge(ForgeError):
    code = "payload_too_large"
    http_status = 413


class Conflict(ForgeError):
    code = "conflict"
    http_status = 409


def _detect_format(path: Path) -> str:
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix.lower() in FORMATS:
                return FORMATS[f.suffix.lower()]
        return "mixed"
    return FORMATS.get(path.suffix.lower(), "unknown")


def _scan(path: Path) -> tuple[int, int]:
    if path.is_file():
        return path.stat().st_size, 1
    total = n = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
            n += 1
    return total, n


def _refresh(svc: Services, row: dict[str, Any]) -> dict[str, Any]:
    """Recompute size, and pick up the manifest of a finished ingest job."""
    p = Path(row["path"]) if row["path"] else None
    fields: dict[str, Any] = {}
    if p and p.exists():
        size, n = _scan(p)
        if size != row["size_bytes"] or n != row["num_files"]:
            fields.update(size_bytes=size, num_files=n)
    meta = dict(row.get("meta") or {})
    jid = meta.get("ingest_job_id")
    if jid and not row.get("manifest_hash"):
        job = svc.jobs_repo.get(jid)
        if job and job["status"] == "succeeded":
            outs = job.get("outputs") or {}
            result = outs.get("result") if isinstance(outs.get("result"), dict) else {}
            merged = {**result, **outs}
            mh = next((merged[k] for k in ("manifest_hash", "manifest", "dataset_hash", "hash") if isinstance(merged.get(k), str)), None)
            tokens = next((merged[k] for k in ("total_tokens", "tokens") if isinstance(merged.get(k), int)), None)
            if mh or tokens:
                fields.update(manifest_hash=mh, tokens=tokens, status="tokenized")
        elif job and job["status"] == "failed":
            fields["status"] = "ingest_failed"
    return svc.datasets.update(row["id"], **fields) if fields else row


@router.get("", summary="List datasets")
def list_datasets(response: Response, svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    rows = [_refresh(svc, r) for r in svc.datasets.list(limit=1000)]
    response.headers["X-Total-Count"] = str(len(rows))
    return rows


@router.post("", status_code=201, summary="Register a local path, or create an empty dataset to upload into")
def create(body: DatasetCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    t = now()
    did = new_id("ds")
    if body.path:
        p = Path(body.path).expanduser()
        if not p.exists():
            raise NotFound(f"path does not exist: {p}")
        size, n = _scan(p)
        fmt = body.format or _detect_format(p)
        return svc.datasets.insert(
            id=did, name=body.name, kind="raw", format=fmt, path=str(p.resolve()), status="ready", size_bytes=size,
            num_files=n, licence=body.licence, meta={"registered_in_place": True}, created_at=t, updated_at=t,
        )
    d = datasets_dir() / did
    d.mkdir(parents=True, exist_ok=True)
    return svc.datasets.insert(
        id=did, name=body.name, kind="raw", format=body.format, path=str(d), status="empty", size_bytes=0, num_files=0,
        licence=body.licence, meta={"managed": True}, created_at=t, updated_at=t,
    )


def _safe_filename(name: str | None) -> str:
    n = Path((name or "").replace("\\", "/")).name.strip()
    if not n or n in (".", "..") or n.startswith("."):
        raise ConfigError("a valid filename is required")
    return n


def _managed_dir(row: dict[str, Any]) -> Path:
    d = Path(row["path"])
    if not (row.get("meta") or {}).get("managed"):
        raise ConfigError("this dataset was registered in place; uploads only go to managed datasets")
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _append_stream(svc: Services, row: dict[str, Any], filename: str, offset: int | None, chunks, final: bool) -> dict[str, Any]:
    d = _managed_dir(row)
    target = d / filename
    cur = target.stat().st_size if target.exists() else 0
    if offset is None:
        offset = 0
        if target.exists():
            target.unlink()  # whole-file upload replaces
            cur = 0
    if offset != cur:
        raise Conflict(f"offset mismatch: file currently has {cur} bytes (got offset={offset})", details={"expected_offset": cur})
    total_other = sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and f != target)
    written = 0
    with open(target, "ab") as f:
        async for chunk in chunks:
            written += len(chunk)
            if total_other + cur + written > svc.max_upload_bytes:
                f.close()
                if cur == 0:
                    target.unlink(missing_ok=True)
                raise PayloadTooLarge(
                    f"upload exceeds the limit of {svc.max_upload_bytes} bytes (FORGE_MAX_UPLOAD_MB)",
                    details={"limit_bytes": svc.max_upload_bytes},
                )
            f.write(chunk)
    size, n = _scan(d)
    fields: dict[str, Any] = {"size_bytes": size, "num_files": n, "status": "ready" if final else "uploading"}
    if not row.get("format"):
        fields["format"] = _detect_format(d)
    out = svc.datasets.update(row["id"], **fields)
    out["received_bytes"] = written
    out["file_size"] = cur + written
    return out


async def _aiter_upload(file: Any):
    while True:
        b = await file.read(_CHUNK)
        if not b:
            break
        yield b


if HAVE_MULTIPART:
    from fastapi import File, Form, UploadFile

    @router.post("/upload", status_code=201, summary="Multipart upload (whole file, or chunks with `offset` + `dataset_id`)")
    async def upload(
        file: UploadFile = File(..., description="File (or chunk) bytes"),
        name: str | None = Form(None, description="Dataset name (new datasets)"),
        dataset_id: str | None = Form(None, description="Existing managed dataset to add to / continue"),
        filename: str | None = Form(None, description="Target filename (default: the uploaded filename)"),
        offset: int | None = Form(None, description="Byte offset of this chunk in the target file; must equal its current size. Omit for a whole-file upload."),
        final: bool = Form(True, description="Set false on all but the last chunk."),
        licence: str | None = Form(None),
        svc: Services = Depends(get_services),
    ) -> dict[str, Any]:
        fname = _safe_filename(filename or file.filename)
        if dataset_id:
            row = svc.datasets.require(dataset_id, "dataset")
        else:
            row = create(DatasetCreate(name=name or Path(fname).stem, licence=licence), svc)
        try:
            return await _append_stream(svc, row, fname, offset, _aiter_upload(file), final)
        except ForgeError:
            if not dataset_id:  # do not leave an empty dataset behind a failed first upload
                shutil.rmtree(row["path"], ignore_errors=True)
                svc.datasets.delete(row["id"])
            raise
else:  # pragma: no cover - depends on the environment

    @router.post("/upload", summary="Multipart upload (python-multipart is not installed)")
    async def upload_unavailable() -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={"error": "module_unavailable", "message": "multipart upload needs `pip install python-multipart`; use PUT /datasets/{id}/content for raw uploads", "details": {}},
        )


@router.put("/{dataset_id}/content", summary="Raw (non-multipart) upload/append of bytes to a managed dataset")
async def put_content(
    dataset_id: str,
    request: Request,
    filename: str = Query(..., description="Target filename"),
    offset: int | None = Query(None, ge=0, description="Append offset (must equal the current size); omit to replace"),
    final: bool = Query(True),
    svc: Services = Depends(get_services),
) -> dict[str, Any]:
    row = svc.datasets.require(dataset_id, "dataset")
    return await _append_stream(svc, row, _safe_filename(filename), offset, request.stream(), final)


@router.get("/{dataset_id}")
def get(dataset_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = _refresh(svc, svc.datasets.require(dataset_id, "dataset"))
    p = Path(row["path"]) if row["path"] else None
    files: list[dict[str, Any]] = []
    if p and p.exists():
        for f in sorted(p.rglob("*") if p.is_dir() else [p])[:500]:
            if f.is_file():
                files.append({"name": str(f.relative_to(p) if p.is_dir() else f.name), "size": f.stat().st_size})
    row["files"] = files
    return row


@router.delete("/{dataset_id}", status_code=204)
def delete(dataset_id: str, delete_files: bool = True, svc: Services = Depends(get_services)) -> Response:
    row = svc.datasets.require(dataset_id, "dataset")
    if delete_files and (row.get("meta") or {}).get("managed") and row["path"]:
        p = Path(row["path"]).resolve()
        if datasets_dir().resolve() in p.parents:
            shutil.rmtree(p, ignore_errors=True)
    svc.datasets.delete(dataset_id)
    return Response(status_code=204)


def _check_preview_path(raw: str, svc: Services) -> None:
    """Existing local paths may only be previewed from under the datasets dir, FORGE_HOME or a registered dataset's path."""
    try:
        p = Path(raw).expanduser().resolve()
        exists = p.exists()
    except (OSError, ValueError):
        raise ConfigError(f"invalid source path: {raw!r}") from None
    if not exists:
        return  # nothing local to read: a Hub id, or a NotFound raised by the reader
    roots = [datasets_dir().resolve(), forge_home().resolve()]
    for r in svc.datasets.list(limit=10000):
        if r.get("path"):
            try:
                roots.append(Path(r["path"]).expanduser().resolve())
            except (OSError, ValueError):
                pass
    if not any(p == r or r in p.parents for r in roots):
        raise ConfigError("source-preview only reads local paths under the datasets directory, FORGE_HOME or a registered dataset; "
                          "register or upload the dataset first")


class SourcePreviewRequest(BaseModel):
    source: dict[str, Any]
    n: int = 5


@router.post("/source-preview", summary="Resolve an ingest source (local path or Hugging Face id) and return config/split/column info plus a few samples")
def source_preview(body: SourcePreviewRequest, svc: Services = Depends(get_services)) -> dict[str, Any]:
    from forge.train.data import SourceSpec, describe_source

    try:
        spec = SourceSpec.model_validate(body.source)
    except ValidationError as e:
        raise ConfigError("invalid source: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())) from e
    _check_preview_path(spec.path, svc)
    return describe_source(spec, n=body.n)  # plain def: FastAPI runs it in the threadpool (it may touch the network)


_PATH_FIELDS = ("input_path", "input_paths", "path", "paths", "source", "sources", "input", "inputs", "data_path", "data_files", "files", "dataset_path")


@router.post("/{dataset_id}/ingest", status_code=201, summary="Tokenise + shard this dataset (starts a dataset_ingest job)")
def ingest(dataset_id: str, body: IngestRequest, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = svc.datasets.require(dataset_id, "dataset")
    jt = get_job_type("dataset_ingest")
    cls = jt.config_model()  # 501 if forge.train.data is missing
    params = dict(body.params)
    props = getattr(cls, "model_fields", {}) or {}
    for key in _PATH_FIELDS:
        if key == "sources" and "sources" in props and "sources" not in params:  # IngestConfig.sources: list[SourceSpec]
            params["sources"] = [{"kind": "auto", "path": row["path"], "name": row["name"]}]
            break
        if key in props and key not in params:
            ann = str(props[key].annotation)
            params[key] = [row["path"]] if ("list" in ann.lower() or "List" in ann) else row["path"]
            break
    if "name" in props and "name" not in params:
        params["name"] = row["name"]
    job = submit(svc, "dataset_ingest", params, name=body.name or f"ingest {row['name']}", gpus=body.gpus)
    meta = dict(row.get("meta") or {})
    meta["ingest_job_id"] = job["id"]
    svc.datasets.update(dataset_id, meta=meta, status="ingesting")
    return job


def _trunc(v: Any, n: int = 600) -> Any:
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + f"... (+{len(v) - n} chars)"
    if isinstance(v, dict):
        return {k: _trunc(x, n) for k, x in v.items()}
    if isinstance(v, list):
        return [_trunc(x, n) for x in v[:50]]
    return v


@router.get("/{dataset_id}/preview", summary="First N records of a text/jsonl/csv/parquet dataset")
def preview(
    dataset_id: str, n: int = Query(5, ge=1, le=100), file: str | None = Query(None, description="File inside a directory dataset"),
    svc: Services = Depends(get_services),
) -> dict[str, Any]:
    row = _refresh(svc, svc.datasets.require(dataset_id, "dataset"))
    out: dict[str, Any] = {"id": dataset_id, "format": row["format"], "status": row["status"], "manifest_hash": row.get("manifest_hash"), "tokens": row.get("tokens")}
    if row.get("manifest_hash"):
        store = lazy_module("forge.store")
        try:
            if store is not None:
                m = store.get_store().get_manifest(row["manifest_hash"])
                out["manifest"] = json.loads(m.model_dump_json()) if hasattr(m, "model_dump_json") else str(m)
        except Exception as e:
            out["manifest_error"] = f"{type(e).__name__}: {e}"
    p = Path(row["path"]) if row["path"] else None
    if not p or not p.exists():
        out["records"] = []
        return out
    target = p
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file())
        if file:
            cand = (p / file).resolve()
            if p.resolve() not in cand.parents or not cand.is_file():
                raise NotFound(f"file {file!r} not in dataset")
            target = cand
        elif files:
            target = next((f for f in files if f.suffix.lower() in FORMATS), files[0])
        else:
            out["records"] = []
            return out
    out["file"] = target.name
    fmt = FORMATS.get(target.suffix.lower(), "text")
    records: list[Any] = []
    try:
        if fmt == "jsonl":
            with open(target, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        records.append(_trunc(json.loads(line)))
                    except ValueError:
                        records.append(_trunc(line.rstrip("\n")))
                    if len(records) >= n:
                        break
        elif fmt == "json":
            if target.stat().st_size > 50 * 2**20:
                raise ConfigError("json file too large to preview (>50 MiB); use jsonl")
            data = json.loads(target.read_text(encoding="utf-8"))
            records = [_trunc(x) for x in (data[:n] if isinstance(data, list) else [data])]
        elif fmt == "csv":
            with open(target, encoding="utf-8", errors="replace", newline="") as f:
                delim = "\t" if target.suffix.lower() == ".tsv" else ","
                for i, r in enumerate(csv.DictReader(f, delimiter=delim)):
                    records.append(_trunc(r))
                    if i + 1 >= n:
                        break
        elif fmt == "parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise ConfigError("pyarrow is not installed; cannot preview parquet") from e
            pf = pq.ParquetFile(str(target))
            out["num_rows"] = pf.metadata.num_rows
            out["columns"] = pf.schema_arrow.names
            for batch in pf.iter_batches(batch_size=n):
                records = [_trunc(x) for x in batch.to_pylist()[:n]]
                break
        elif fmt == "tokens":
            out["note"] = "binary token shard: preview not available (see manifest)"
        else:
            with open(target, encoding="utf-8", errors="replace") as f:
                buf = io.StringIO(f.read(200_000))
            paras = [x.strip() for x in buf.getvalue().split("\n\n") if x.strip()]
            records = [_trunc(x) for x in paras[:n]]
    except ForgeError:
        raise
    except Exception as e:
        raise ConfigError(f"could not preview {target.name}: {type(e).__name__}: {e}") from e
    out["records"] = records
    return out
