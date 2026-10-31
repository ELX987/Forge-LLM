"""Thin data-access layer over the jobs_api tables (dict rows, JSON columns decoded transparently)."""
from __future__ import annotations

import dataclasses
import enum
import json
import threading
from pathlib import Path
from typing import Any

from forge.core.errors import NotFound
from forge.core.schemas import JobRecord, JobSpec, JobStatus
from forge.core.util import new_id, now
from forge.jobs import db as _jdb  # noqa: F401  (registers migrations)

TERMINAL = {"succeeded", "failed", "cancelled"}


def to_jsonable(o: Any) -> Any:
    """Best-effort conversion of arbitrary objects (paths, numpy/torch scalars, dataclasses, pydantic) to JSON types."""
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, enum.Enum):
        return to_jsonable(o.value)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, bytes):
        return f"<{len(o)} bytes>"
    if hasattr(o, "model_dump"):
        try:
            return to_jsonable(o.model_dump(mode="json"))
        except Exception:
            pass
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return to_jsonable(dataclasses.asdict(o))
    if hasattr(o, "item") and callable(o.item):
        try:
            return to_jsonable(o.item())
        except Exception:
            pass
    if hasattr(o, "tolist"):
        try:
            return to_jsonable(o.tolist())
        except Exception:
            pass
    return str(o)


def dumps(o: Any) -> str:
    return json.dumps(to_jsonable(o), separators=(",", ":"))


class Table:
    """Generic CRUD for one table. JSON columns are named ``<stem>_json`` and exposed as ``<stem>``."""

    def __init__(self, name: str, json_stems: tuple[str, ...] = (), bool_cols: tuple[str, ...] = (), db: Any = None):
        self.name = name
        self.json_stems = json_stems
        self.bool_cols = bool_cols
        self._db = db
        self._cols: set[str] | None = None
        self._lock = threading.Lock()

    @property
    def db(self) -> Any:
        if self._db is not None:
            return self._db
        from forge.core.db import get_db

        return get_db()

    def columns(self) -> set[str]:
        rows = self.db.query(f"PRAGMA table_info({self.name})")
        return {r["name"] for r in rows}

    def _decode(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        for s in self.json_stems:
            raw = row.pop(f"{s}_json", None)
            try:
                row[s] = json.loads(raw) if raw not in (None, "") else None
            except (TypeError, ValueError):
                row[s] = None
        for b in self.bool_cols:
            if b in row and row[b] is not None:
                row[b] = bool(row[b])
        return row

    def _encode(self, fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in fields.items():
            if k in self.json_stems:
                out[f"{k}_json"] = dumps(v)
            elif k in self.bool_cols and v is not None:
                out[k] = int(bool(v))
            else:
                out[k] = to_jsonable(v) if isinstance(v, (dict, list, Path)) else v
        return out

    def insert(self, **fields: Any) -> dict[str, Any]:
        enc = self._encode(fields)
        cols = ",".join(enc)
        q = ",".join("?" for _ in enc)
        self.db.execute(f"INSERT INTO {self.name}({cols}) VALUES({q})", tuple(enc.values()))
        return self.get(fields["id"])  # type: ignore[return-value]

    def get(self, id_: str) -> dict[str, Any] | None:
        return self._decode(self.db.query_one(f"SELECT * FROM {self.name} WHERE id=?", (id_,)))

    def require(self, id_: str, what: str | None = None) -> dict[str, Any]:
        row = self.get(id_)
        if row is None:
            raise NotFound(f"{what or self.name.rstrip('s')} {id_!r} not found", details={"id": id_})
        return row

    def list(
        self,
        where: dict[str, Any] | None = None,
        *,
        order: str = "created_at DESC",
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conds, params = [], []
        for k, v in (where or {}).items():
            if v is None:
                continue
            conds.append(f"{k}=?")
            params.append(v)
        sql = f"SELECT * FROM {self.name}"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
        params += [int(limit), int(offset)]
        return [self._decode(r) for r in self.db.query(sql, params)]  # type: ignore[misc]

    def count(self, where: dict[str, Any] | None = None) -> int:
        conds, params = [], []
        for k, v in (where or {}).items():
            conds.append(f"{k}=?")
            params.append(v)
        sql = f"SELECT COUNT(*) FROM {self.name}" + (" WHERE " + " AND ".join(conds) if conds else "")
        return int(self.db.scalar(sql, params, 0))

    def update(self, id_: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            return self.require(id_)
        allowed = self.columns()
        enc = {k: v for k, v in self._encode(fields).items() if k in allowed}
        if "updated_at" in allowed and "updated_at" not in enc:
            enc["updated_at"] = now()
        sets = ",".join(f"{k}=?" for k in enc)
        cur = self.db.execute(f"UPDATE {self.name} SET {sets} WHERE id=?", (*enc.values(), id_))
        if cur.rowcount == 0:
            raise NotFound(f"{self.name.rstrip('s')} {id_!r} not found", details={"id": id_})
        return self.require(id_)

    def delete(self, id_: str) -> bool:
        return self.db.execute(f"DELETE FROM {self.name} WHERE id=?", (id_,)).rowcount > 0


# --------------------------------------------------------------------------------------------
# repositories
# --------------------------------------------------------------------------------------------
class JobsRepo(Table):
    def __init__(self, db: Any = None):
        super().__init__("jobs", ("spec", "outputs", "gpus"), ("cancel_requested",), db)

    def create(self, spec: JobSpec, *, run_dir: str | None = None, job_id: str | None = None) -> dict[str, Any]:
        jid = job_id or new_id("job")
        t = now()
        return self.insert(
            id=jid,
            type=spec.type,
            name=spec.name or spec.type,
            spec=spec.model_dump(mode="json"),
            status="queued",
            created_at=t,
            updated_at=t,
            progress=0.0,
            outputs={},
            run_dir=run_dir,
            gpus=spec.gpus,
        )

    def to_record(self, row: dict[str, Any]) -> JobRecord:
        spec = JobSpec.model_validate(row["spec"])
        return JobRecord(
            id=row["id"],
            spec=spec,
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            pid=row.get("pid"),
            exit_code=row.get("exit_code"),
            progress=min(1.0, max(0.0, float(row.get("progress") or 0.0))),
            progress_msg=row.get("progress_msg") or "",
            error=row.get("error"),
            outputs=row.get("outputs") or {},
            run_dir=row.get("run_dir"),
        )

    def public(self, row: dict[str, Any]) -> dict[str, Any]:
        """JobRecord JSON + a few convenience fields for the UI (type, name, duration_s, cancel_requested...)."""
        d = self.to_record(row).model_dump(mode="json")
        d["type"] = row["type"]
        d["name"] = row["name"]
        d["gpus"] = row.get("gpus")
        d["cancel_requested"] = bool(row.get("cancel_requested"))
        d["updated_at"] = row.get("updated_at")
        start = row.get("started_at")
        end = row.get("finished_at") or (now() if row["status"] == "running" else None)
        d["duration_s"] = (end - start) if (start and end) else None
        d["tags"] = (row.get("spec") or {}).get("tags", [])
        return d

    def list_jobs(
        self, *, status: str | None = None, type_: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        return self.list({"status": status, "type": type_}, limit=limit, offset=offset)

    def counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        out = {s.value: 0 for s in JobStatus}
        out.update({r["status"]: r["n"] for r in rows})
        return out

    def running(self) -> list[dict[str, Any]]:
        return self.list({"status": "running"}, order="started_at ASC", limit=1000)

    def queued(self) -> list[dict[str, Any]]:
        return self.list({"status": "queued"}, order="created_at ASC", limit=1000)


class ModelsRepo(Table):
    def __init__(self, db: Any = None):
        super().__init__("models", ("info", "tags"), (), db)

    def find_by_path(self, path: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM models WHERE path_or_repo=? AND source!='hf' LIMIT 1", (path,))
        return self._decode(row)

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        """Exact display-name match, falling back to case-insensitive. Among duplicates prefer a ``ready`` row whose files are
        on disk, then any ``ready`` row, then the newest row."""
        from pathlib import Path

        for sql in ("SELECT * FROM models WHERE name=? ORDER BY created_at DESC", "SELECT * FROM models WHERE LOWER(name)=LOWER(?) ORDER BY created_at DESC"):
            rows = [self._decode(r) for r in self.db.query(sql, (name,))]
            if not rows:
                continue

            def usable(r: dict[str, Any]) -> bool:
                info = r.get("info") or {}
                lp = info.get("local_path") or (r.get("path_or_repo") if r.get("source") == "local" else None)
                try:
                    return bool(lp) and Path(lp).is_dir()
                except OSError:
                    return False

            ready = [r for r in rows if r.get("status") == "ready"]
            return next((r for r in ready if usable(r)), None) or (ready[0] if ready else rows[0])
        return None

    def find_by_repo(self, repo: str, revision: str | None = None) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM models WHERE source='hf' AND path_or_repo=? AND COALESCE(revision,'')=? LIMIT 1",
            (repo, revision or ""),
        )
        return self._decode(row)


class DatasetsRepo(Table):
    def __init__(self, db: Any = None):
        super().__init__("datasets", ("meta",), (), db)


class ConfigsRepo(Table):
    def __init__(self, db: Any = None):
        super().__init__("configs", ("params",), (), db)


class RecipesRepo(Table):
    def __init__(self, db: Any = None):
        super().__init__("recipes", (), (), db)


class DeploymentsRepo(Table):
    def __init__(self, db: Any = None):
        super().__init__("deployments", ("serve_config",), (), db)


_STORE_LOCK = threading.Lock()
_STORE_CACHE: dict[str, Any] = {}


def open_store() -> Any:
    """The artifact store for the current FORGE_HOME: ``forge.store.get_store()`` if present, else
    ``forge.store.store.ArtifactStore()``. Raises ``ImportError`` when the store module is unavailable."""
    try:
        from forge.store import get_store

        return get_store()
    except ImportError:
        pass
    from forge.core.paths import store_dir
    from forge.store.store import ArtifactStore

    key = str(store_dir())
    with _STORE_LOCK:
        st = _STORE_CACHE.get(key)
        if st is None:
            st = _STORE_CACHE[key] = ArtifactStore()
        return st
