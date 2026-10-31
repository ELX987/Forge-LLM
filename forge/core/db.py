"""SQLite access (stdlib ``sqlite3``): WAL, per-thread connections, dict rows, transactions, migrations.

Modules register their tables with :func:`register_migration` at import time of their own ``db.py``;
:func:`get_db` applies anything pending (idempotent), so import order does not matter.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from forge.core.errors import ConfigError
from forge.core.hashing import hash_bytes
from forge.core.util import now

log = logging.getLogger("forge.core.db")

# --------------------------------------------------------------------------------------------
# migration registry
# --------------------------------------------------------------------------------------------
_MIG_LOCK = threading.RLock()
_MIGRATIONS: dict[tuple[str, int], str] = {}
_EPOCH = 0  # bumped whenever a new migration is registered


def register_migration(module: str, version: int, sql: str) -> None:
    """Register SQL (one or more ``;``-separated statements) as ``module`` migration ``version``.

    Idempotent for identical SQL; registering different SQL for an existing (module, version) is a
    :class:`ConfigError` (migrations are append-only: add a new version instead).
    """
    global _EPOCH
    if version < 1:
        raise ConfigError("migration version must be >= 1")
    with _MIG_LOCK:
        key = (module, int(version))
        old = _MIGRATIONS.get(key)
        if old is not None:
            if old.strip() != sql.strip():
                raise ConfigError(f"conflicting migration for {module} v{version}")
            return
        _MIGRATIONS[key] = sql
        _EPOCH += 1


def registered_migrations() -> list[tuple[str, int]]:
    with _MIG_LOCK:
        return sorted(_MIGRATIONS)


def _split_statements(sql: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


# --------------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------------
class Database:
    """Thread-safe sqlite wrapper. File databases use one connection per thread; ``:memory:`` shares one."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._memory = self.path == ":memory:"
        self._local = threading.local()
        self._lock = threading.RLock()
        self._all: list[sqlite3.Connection] = []
        self._shared: sqlite3.Connection | None = None
        self._migrated_epoch = -1
        if not self._memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    # -- connections -----------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None, check_same_thread=not self._memory
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        if not self._memory:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._all.append(conn)
        return conn

    def conn(self) -> sqlite3.Connection:
        if self._memory:
            with self._lock:
                if self._shared is None:
                    self._shared = self._connect()
                return self._shared
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._connect()
            self._local.conn = c
        return c

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                try:
                    c.close()
                except Exception:
                    pass
            self._all.clear()
            self._shared = None
        self._local = threading.local()

    # -- query API -------------------------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] | dict = ()) -> sqlite3.Cursor:
        with self._lock if self._memory else _NULL_LOCK:
            return self.conn().execute(sql, params)

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        with self._lock if self._memory else _NULL_LOCK:
            return self.conn().executemany(sql, seq)

    def query(self, sql: str, params: Sequence[Any] | dict = ()) -> list[dict[str, Any]]:
        with self._lock if self._memory else _NULL_LOCK:
            cur = self.conn().execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] | dict = ()) -> dict[str, Any] | None:
        with self._lock if self._memory else _NULL_LOCK:
            cur = self.conn().execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row is not None else None

    def scalar(self, sql: str, params: Sequence[Any] | dict = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if not row:
            return default
        return next(iter(row.values()))

    @contextmanager
    def transaction(self) -> Iterator[Database]:
        """Atomic block. Re-entrant: inner blocks become SAVEPOINTs. Rolls back on any exception."""
        depth = getattr(self._local, "depth", 0)
        if self._memory:
            self._lock.acquire()
        c = self.conn()
        try:
            if depth == 0:
                c.execute("BEGIN IMMEDIATE")
            else:
                c.execute(f"SAVEPOINT sp{depth}")
            self._local.depth = depth + 1
            try:
                yield self
            except BaseException:
                if depth == 0:
                    c.execute("ROLLBACK")
                else:
                    c.execute(f"ROLLBACK TO sp{depth}")
                    c.execute(f"RELEASE sp{depth}")
                raise
            else:
                if depth == 0:
                    c.execute("COMMIT")
                else:
                    c.execute(f"RELEASE sp{depth}")
            finally:
                self._local.depth = depth
        finally:
            if self._memory:
                self._lock.release()

    # -- small kv helpers (table created by the core migration) ------------------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value_json FROM kv WHERE key=?", (key,))
        return json.loads(row["value_json"]) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key, value_json, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (key, json.dumps(value), now()),
        )

    def kv_delete(self, key: str) -> None:
        self.execute("DELETE FROM kv WHERE key=?", (key,))

    # -- migrations ------------------------------------------------------------------------
    def migrate(self) -> list[tuple[str, int]]:
        """Apply registered migrations not yet applied to this database. Returns those applied now."""
        applied_now: list[tuple[str, int]] = []
        with _MIG_LOCK:
            epoch = _EPOCH
            pending = dict(_MIGRATIONS)
        self.execute(
            "CREATE TABLE IF NOT EXISTS _migrations("
            "module TEXT NOT NULL, version INTEGER NOT NULL, checksum TEXT NOT NULL, applied_at REAL NOT NULL, "
            "PRIMARY KEY(module, version))"
        )
        done = {(r["module"], r["version"]): r["checksum"] for r in self.query("SELECT * FROM _migrations")}
        for key in sorted(pending):
            sql = pending[key]
            checksum = hash_bytes(sql.strip().encode("utf-8"), "b2")
            if key in done:
                if done[key] != checksum:
                    log.warning("migration %s v%d changed since it was applied", *key)
                continue
            try:
                with self.transaction():
                    for stmt in _split_statements(sql):
                        self.execute(stmt)
                    self.execute(
                        "INSERT INTO _migrations(module, version, checksum, applied_at) VALUES(?,?,?,?)",
                        (key[0], key[1], checksum, now()),
                    )
            except sqlite3.OperationalError as e:
                # another process may have applied it concurrently
                row = self.query_one("SELECT 1 AS x FROM _migrations WHERE module=? AND version=?", key)
                if row is None:
                    raise ConfigError(f"migration {key[0]} v{key[1]} failed: {e}") from e
                continue
            applied_now.append(key)
            log.debug("applied migration %s v%d", *key)
        self._migrated_epoch = epoch
        return applied_now

    def ensure_migrated(self) -> None:
        if self._migrated_epoch != _EPOCH:
            self.migrate()

    def tables(self) -> list[str]:
        return [r["name"] for r in self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


_NULL_LOCK = _NullLock()

# --------------------------------------------------------------------------------------------
# singleton access
# --------------------------------------------------------------------------------------------
_DBS: dict[str, Database] = {}
_DBS_LOCK = threading.Lock()


def get_db(path: str | Path | None = None) -> Database:
    """Database for ``path`` (default: ``<FORGE_HOME>/forge.db``), migrated and cached per path."""
    if path is None:
        from forge.core.paths import db_path

        path = db_path()
    key = str(path)
    with _DBS_LOCK:
        db = _DBS.get(key)
        if db is None:
            db = Database(key)
            _DBS[key] = db
    db.ensure_migrated()
    return db


def migrate(db: Database | None = None) -> list[tuple[str, int]]:
    """Apply all registered migrations to ``db`` (default database). Idempotent."""
    return (db or get_db()).migrate()


def close_all() -> None:
    """Close and forget every cached database (tests, shutdown)."""
    with _DBS_LOCK:
        for db in _DBS.values():
            db.close()
        _DBS.clear()


# --------------------------------------------------------------------------------------------
# core's own tables
# --------------------------------------------------------------------------------------------
register_migration(
    "core",
    1,
    """
CREATE TABLE IF NOT EXISTS kv(
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_cache(
    cache_key TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    recipe_hash TEXT,
    outputs_json TEXT NOT NULL,
    output_hashes_json TEXT NOT NULL,
    seconds REAL NOT NULL DEFAULT 0,
    hits INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_hit_at REAL
);
CREATE INDEX IF NOT EXISTS idx_stage_cache_stage ON stage_cache(stage);
""",
)
