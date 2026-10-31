"""Metrics store: sqlite-backed time series per run with a batching writer and downsampling.

``MetricsStore.write(run_id, step, values)`` is cheap (buffered; a daemon thread flushes every
``flush_interval`` seconds or ``max_batch`` rows). ``query`` flushes this process' buffer first, so a
writer and reader in the same process always agree; readers in other processes see data after the
writer's next flush (<= ``flush_interval``).

Downsampling (``max_points``):

* up to ``20 * max_points`` raw rows -> LTTB (largest-triangle-three-buckets, shape preserving)
* more -> SQL bucket aggregation (mean per bucket, with min/max "bands") so huge runs stay cheap
* ``rollup="minmax"`` -> min/max bucket reduction, ``rollup=10|100`` -> use precomputed rollup tables
  built by :meth:`MetricsStore.compact`.
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

from forge.core.errors import ForgeError
from forge.core.util import now
from forge.telemetry import db as _tdb  # noqa: F401  (registers migrations)

log = logging.getLogger("forge.telemetry")

Point = tuple[float, "float | None", float]  # (step, value, ts)


# --------------------------------------------------------------------------------------------
# downsampling
# --------------------------------------------------------------------------------------------
def lttb(points: list[Point], threshold: int) -> list[Point]:
    """Largest-Triangle-Three-Buckets. Keeps first and last points; ``None`` values are treated as 0 in
    the area computation (so gaps stay visible as outliers) but the original point is returned."""
    n = len(points)
    if threshold >= n:
        return list(points)
    if threshold < 3:
        return [points[0], points[-1]][: max(1, threshold)]
    xs = [float(p[0]) for p in points]
    ys = [0.0 if (p[1] is None or not math.isfinite(p[1])) else float(p[1]) for p in points]
    sampled = [0]
    every = (n - 2) / (threshold - 2)
    a = 0
    for i in range(threshold - 2):
        rs = int(math.floor((i + 1) * every) + 1)
        re_ = min(int(math.floor((i + 2) * every) + 1), n)
        if rs >= re_:
            rs = max(0, re_ - 1)
        cnt = max(1, re_ - rs)
        avg_x = sum(xs[rs:re_]) / cnt
        avg_y = sum(ys[rs:re_]) / cnt
        lo = int(math.floor(i * every) + 1)
        hi = min(int(math.floor((i + 1) * every) + 1), n - 1)
        ax, ay = xs[a], ys[a]
        best, best_i = -1.0, lo
        for j in range(lo, max(hi, lo + 1)):
            area = abs((ax - avg_x) * (ys[j] - ay) - (ax - xs[j]) * (avg_y - ay))
            if area > best:
                best, best_i = area, j
        sampled.append(best_i)
        a = best_i
    sampled.append(n - 1)
    return [points[i] for i in sampled]


def minmax_bucket(points: list[Point], max_points: int) -> list[Point]:
    """Keep each bucket's min and max (original order) -> envelope-preserving, <= ~max_points points."""
    n = len(points)
    if n <= max_points or max_points < 4:
        return list(points)
    buckets = max(1, (max_points - 2) // 2)
    size = n / buckets
    out: list[Point] = [points[0]]
    for b in range(buckets):
        lo, hi = int(b * size), min(n, int((b + 1) * size))
        seg = [(i, points[i]) for i in range(lo, hi) if points[i][1] is not None]
        if not seg:
            continue
        mn = min(seg, key=lambda t: t[1][1])  # type: ignore[index,return-value]
        mx = max(seg, key=lambda t: t[1][1])  # type: ignore[index,return-value]
        for i in sorted({mn[0], mx[0]}):
            if points[i] is not out[-1]:
                out.append(points[i])
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def _num(x: float) -> int | float:
    return int(x) if float(x).is_integer() else float(x)


# --------------------------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------------------------
class MetricsStore:
    """Time-series store on top of ``forge.core.db``. Thread-safe."""

    def __init__(self, db: Any = None, *, flush_interval: float = 0.25, max_batch: int = 2000, autostart: bool = True):
        self._db = db
        self.flush_interval = float(flush_interval)
        self.max_batch = int(max_batch)
        self._buf: list[tuple[str, str, int, float | None, float]] = []
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._autostart = autostart

    # -- plumbing -------------------------------------------------------------------------
    @property
    def db(self) -> Any:
        if self._db is None:
            from forge.core.db import get_db

            self._db = get_db()
        return self._db

    def _ensure_thread(self) -> None:
        if not self._autostart or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        t = threading.Thread(target=self._loop, name="forge-metrics-writer", daemon=True)
        self._thread = t
        t.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.flush_interval)
            self._wake.clear()
            try:
                self.flush()
            except Exception as e:  # pragma: no cover - disk full etc.
                log.warning("metrics flush failed: %s", e)
        try:
            self.flush()
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        """Stop the writer thread and flush."""
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5)
        self._thread = None
        self.flush()

    # -- write ----------------------------------------------------------------------------
    def write(self, run_id: str, step: int, values: dict[str, Any], ts: float | None = None) -> None:
        """Buffer ``values`` (key -> number) for ``run_id`` at ``step``. Non-numeric values are skipped,
        non-finite numbers are stored as NULL."""
        t = float(ts) if ts is not None else now()
        rows = []
        for k, v in values.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            rows.append((run_id, str(k), int(step), f if math.isfinite(f) else None, t))
        if not rows:
            return
        with self._lock:
            self._buf.extend(rows)
            full = len(self._buf) >= self.max_batch
        self._ensure_thread()
        if full:
            self.flush()

    def flush(self) -> int:
        """Write buffered rows. Returns the number of rows written."""
        with self._flush_lock:
            with self._lock:
                rows, self._buf = self._buf, []
            if not rows:
                return 0
            try:
                with self.db.transaction():
                    self.db.executemany(
                        "INSERT INTO metrics(run_id, key, step, value, ts) VALUES(?,?,?,?,?)", rows
                    )
            except Exception:
                with self._lock:  # put back so nothing is lost on a transient lock error
                    self._buf = rows + self._buf
                raise
            return len(rows)

    # -- read -----------------------------------------------------------------------------
    def keys(self, run_id: str) -> list[str]:
        self.flush()
        rows = self.db.query("SELECT DISTINCT key FROM metrics WHERE run_id=? ORDER BY key", (run_id,))
        return [r["key"] for r in rows]

    def latest(self, run_id: str) -> dict[str, dict[str, Any]]:
        """``{key: {"step", "value", "ts"}}`` of the most recently written row per key."""
        self.flush()
        rows = self.db.query(
            "SELECT key, step, value, ts FROM metrics WHERE id IN "
            "(SELECT MAX(id) FROM metrics WHERE run_id=? GROUP BY key)",
            (run_id,),
        )
        return {r["key"]: {"step": r["step"], "value": r["value"], "ts": r["ts"]} for r in rows}

    def count(self, run_id: str, key: str | None = None) -> int:
        self.flush()
        if key is None:
            return int(self.db.scalar("SELECT COUNT(*) FROM metrics WHERE run_id=?", (run_id,), 0))
        return int(self.db.scalar("SELECT COUNT(*) FROM metrics WHERE run_id=? AND key=?", (run_id, key), 0))

    def runs(self) -> list[str]:
        self.flush()
        return [r["run_id"] for r in self.db.query("SELECT DISTINCT run_id FROM metrics")]

    def since_seq(self, run_id: str, seq: int = 0, limit: int = 5000) -> tuple[list[dict[str, Any]], int]:
        """Rows written after sequence number ``seq`` (monotonic row id) -> ``(rows, new_seq)``. For SSE."""
        rows = self.db.query(
            "SELECT id, key, step, value, ts FROM metrics WHERE run_id=? AND id>? ORDER BY id LIMIT ?",
            (run_id, int(seq), int(limit)),
        )
        return rows, (rows[-1]["id"] if rows else int(seq))

    def max_seq(self, run_id: str) -> int:
        return int(self.db.scalar("SELECT COALESCE(MAX(id),0) FROM metrics WHERE run_id=?", (run_id,), 0))

    def query(
        self,
        run_id: str,
        keys: list[str] | None = None,
        since_step: int | None = None,
        max_points: int = 2000,
        rollup: str | int = "auto",
    ) -> dict[str, Any]:
        """Series for ``run_id``: ``{"keys": [...], "points": {key: [[step, value, ts], ...]}, "downsampled": {key: how}}``
        (+ ``"bands": {key: [[step, min, max], ...]}`` for aggregated series).

        ``rollup``: ``auto`` | ``raw`` | ``lttb`` | ``minmax`` | ``mean`` | an int (10/100: rollup level).
        """
        self.flush()
        max_points = max(3, int(max_points))
        all_keys = list(keys) if keys else self.keys(run_id)
        out: dict[str, list[list[Any]]] = {}
        info: dict[str, str] = {}
        bands: dict[str, list[list[Any]]] = {}
        for k in all_keys:
            pts, how, band = self._query_key(run_id, k, since_step, max_points, rollup)
            out[k] = [[_num(p[0]), p[1], p[2]] for p in pts]
            if how != "raw":
                info[k] = how
            if band:
                bands[k] = band
        res: dict[str, Any] = {"keys": all_keys, "points": out, "downsampled": info}
        if bands:
            res["bands"] = bands
        return res

    def _query_key(
        self, run_id: str, key: str, since_step: int | None, max_points: int, rollup: str | int
    ) -> tuple[list[Point], str, list[list[Any]] | None]:
        db = self.db
        cond = "run_id=? AND key=?"
        params: list[Any] = [run_id, key]
        if since_step is not None:
            cond += " AND step>?"
            params.append(int(since_step))
        if isinstance(rollup, int) or (isinstance(rollup, str) and rollup.isdigit()):
            level = int(rollup)
            if level > 1:
                rc = "run_id=? AND key=? AND level=?"
                rp: list[Any] = [run_id, key, level]
                if since_step is not None:
                    rc += " AND step>?"
                    rp.append(float(since_step))
                rows = db.query(f"SELECT step, mean, vmin, vmax, ts FROM metrics_rollup WHERE {rc} ORDER BY step", rp)
                pts: list[Point] = [(r["step"], r["mean"], r["ts"]) for r in rows]
                bnd: list[list[Any]] | None = [[r["step"], r["vmin"], r["vmax"]] for r in rows]
                if len(pts) > max_points:
                    pts, bnd = lttb(pts, max_points), None
                return pts, f"rollup{level}", bnd
            rollup = "raw"
        n = int(db.scalar(f"SELECT COUNT(*) FROM metrics WHERE {cond}", params, 0))
        if n == 0:
            return [], "raw", None
        mode = str(rollup)
        if mode == "auto":
            mode = "raw" if n <= max_points else ("lttb" if n <= 20 * max_points else "mean")
        if mode == "mean":
            span = db.query_one(f"SELECT MIN(step) AS a, MAX(step) AS b FROM metrics WHERE {cond}", params) or {}
            lo, hi = span.get("a", 0), span.get("b", 0)
            width = max(1, int(math.ceil((hi - lo + 1) / max_points)))
            rows = db.query(
                f"SELECT (step - ?) / ? AS b, AVG(step) AS step, AVG(value) AS mean, MIN(value) AS vmin, "
                f"MAX(value) AS vmax, MAX(ts) AS ts FROM metrics WHERE {cond} GROUP BY b ORDER BY b",
                [lo, width, *params],
            )
            return (
                [(r["step"], r["mean"], r["ts"]) for r in rows],
                "mean",
                [[r["step"], r["vmin"], r["vmax"]] for r in rows],
            )
        rows = db.query(f"SELECT step, value, ts FROM metrics WHERE {cond} ORDER BY step, id", params)
        pts = [(r["step"], r["value"], r["ts"]) for r in rows]
        if mode == "raw" or len(pts) <= max_points:
            return pts, "raw", None
        if mode == "minmax":
            return minmax_bucket(pts, max_points), "minmax", None
        return lttb(pts, max_points), "lttb", None

    # -- maintenance ----------------------------------------------------------------------
    def compact(
        self, run_id: str, factors: tuple[int, ...] = (10, 100), *, drop_raw_older_than_steps: int | None = None
    ) -> dict[str, int]:
        """Build rollup tables (mean/min/max per ``factor`` steps) for ``run_id``. Optionally delete raw rows
        older than ``latest_step - drop_raw_older_than_steps`` (rollups keep the history)."""
        self.flush()
        made: dict[str, int] = {}
        with self.db.transaction():
            for f in factors:
                if f <= 1:
                    continue
                self.db.execute("DELETE FROM metrics_rollup WHERE run_id=? AND level=?", (run_id, f))
                cur = self.db.execute(
                    "INSERT INTO metrics_rollup(run_id, key, level, bucket, step, mean, vmin, vmax, n, ts) "
                    "SELECT run_id, key, ?, step / ?, AVG(step), AVG(value), MIN(value), MAX(value), COUNT(*), MAX(ts) "
                    "FROM metrics WHERE run_id=? GROUP BY key, step / ?",
                    (f, f, run_id, f),
                )
                made[str(f)] = cur.rowcount
            if drop_raw_older_than_steps is not None:
                mx = self.db.scalar("SELECT MAX(step) FROM metrics WHERE run_id=?", (run_id,), 0) or 0
                cur = self.db.execute(
                    "DELETE FROM metrics WHERE run_id=? AND step < ?", (run_id, mx - int(drop_raw_older_than_steps))
                )
                made["raw_deleted"] = cur.rowcount
        return made

    def delete_run(self, run_id: str) -> int:
        self.flush()
        with self.db.transaction():
            n = self.db.execute("DELETE FROM metrics WHERE run_id=?", (run_id,)).rowcount
            self.db.execute("DELETE FROM metrics_rollup WHERE run_id=?", (run_id,))
        return n

    def export_parquet(self, run_id: str, path: str | Path | None = None) -> Path:
        """Write all raw rows of a run to a parquet file (needs ``pyarrow``)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ForgeError("pyarrow is not installed; `pip install pyarrow` to export parquet") from e
        self.flush()
        rows = self.db.query("SELECT key, step, value, ts FROM metrics WHERE run_id=? ORDER BY id", (run_id,))
        if path is None:
            from forge.core.paths import run_dir

            path = run_dir(run_id) / "metrics.parquet"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "key": pa.array([r["key"] for r in rows], pa.string()),
                "step": pa.array([r["step"] for r in rows], pa.int64()),
                "value": pa.array([r["value"] for r in rows], pa.float64()),
                "ts": pa.array([r["ts"] for r in rows], pa.float64()),
            }
        )
        pq.write_table(table, str(path), compression="zstd")
        return path


# --------------------------------------------------------------------------------------------
# singleton per database file
# --------------------------------------------------------------------------------------------
_STORES: dict[str, MetricsStore] = {}
_STORES_LOCK = threading.Lock()


def get_metrics_store() -> MetricsStore:
    """MetricsStore bound to the current ``FORGE_HOME`` database (cached per db path)."""
    from forge.core.db import get_db
    from forge.core.paths import db_path

    key = str(db_path())
    db = get_db()
    with _STORES_LOCK:
        st = _STORES.get(key)
        if st is None or st._db is not db:
            st = MetricsStore(db)
            _STORES[key] = st
        return st


def reset_metrics_stores() -> None:
    """Close and forget cached stores (tests)."""
    with _STORES_LOCK:
        for s in _STORES.values():
            try:
                s.close()
            except Exception:
                pass
        _STORES.clear()
