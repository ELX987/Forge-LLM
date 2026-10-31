"""RunContext implementation used inside job worker processes.

* ``log``      -> one formatted line on stdout (the manager redirects stdout/stderr to ``runs/<id>/log.txt``)
* ``metric``   -> :class:`forge.telemetry.metrics.MetricsStore` (batched, flushed <= 0.25 s)
* ``progress`` -> ``jobs`` row (throttled to 5 Hz unless the message changes or the job completes)
* ``set_output`` -> ``jobs.outputs_json`` (merged immediately)
* ``should_stop`` -> True after SIGTERM (POSIX), a ``CANCEL`` file in the run dir, or ``cancel_requested`` in the row
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from forge.core.util import now
from forge.jobs.repo import JobsRepo, dumps, to_jsonable
from forge.telemetry.logs import format_line

CANCEL_FILE = "CANCEL"


class WorkerContext:
    def __init__(self, job_id: str, run_dir: Path, params: dict[str, Any], *, db: Any = None, metrics: Any = None):
        self.run_id = job_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.params = dict(params)
        self._repo = JobsRepo(db)
        self._db = db
        self._metrics = metrics
        self._store: Any = None
        self._store_tried = False
        self._lock = threading.Lock()
        self._outputs: dict[str, Any] = {}
        self._stop_flag = threading.Event()
        self._last_progress_write = 0.0
        self._last_progress: tuple[float, str] = (-1.0, "")
        self._last_cancel_check = 0.0
        self._cancel_cached = False
        self._auto_step = 0

    # -- helpers ---------------------------------------------------------------------------
    @property
    def db(self) -> Any:
        if self._db is None:
            from forge.core.db import get_db

            self._db = get_db()
        return self._db

    @property
    def metrics_store(self) -> Any:
        if self._metrics is None:
            from forge.telemetry.metrics import get_metrics_store

            self._metrics = get_metrics_store()
        return self._metrics

    # -- RunContext API ----------------------------------------------------------------------
    def log(self, msg: str, level: str = "info") -> None:
        for line in str(msg).splitlines() or [""]:
            print(format_line(line, level), flush=True)

    def metric(self, step: int, **values: float) -> None:
        self.metrics_store.write(self.run_id, int(step), values)

    def progress(self, frac: float, msg: str = "") -> None:
        frac = min(1.0, max(0.0, float(frac)))
        t = time.monotonic()
        prev_frac, prev_msg = self._last_progress
        important = frac >= 1.0 or msg != prev_msg
        if not important and t - self._last_progress_write < 0.2:
            return
        if frac == prev_frac and msg == prev_msg:
            return
        self._last_progress_write = t
        self._last_progress = (frac, msg)
        try:
            self.db.execute(
                "UPDATE jobs SET progress=?, progress_msg=?, updated_at=? WHERE id=?", (frac, msg or "", now(), self.run_id)
            )
        except Exception as e:  # pragma: no cover - transient lock; progress is best-effort
            print(format_line(f"progress write failed: {e}", "warning"), flush=True)

    def should_stop(self) -> bool:
        if self._stop_flag.is_set():
            return True
        t = time.monotonic()
        if t - self._last_cancel_check < 0.1:
            return self._cancel_cached
        self._last_cancel_check = t
        hit = (self.run_dir / CANCEL_FILE).exists()
        if not hit:
            try:
                row = self.db.query_one("SELECT cancel_requested FROM jobs WHERE id=?", (self.run_id,))
                hit = bool(row and row["cancel_requested"])
            except Exception:  # pragma: no cover
                pass
        self._cancel_cached = hit
        if hit:
            self._stop_flag.set()
        return hit

    def request_stop(self) -> None:
        """Called by the SIGTERM handler."""
        self._stop_flag.set()

    def set_output(self, key: str, value: Any) -> None:
        with self._lock:
            self._outputs[key] = to_jsonable(value)
            payload = dumps(self._outputs)
        # merge with whatever the row already holds (e.g. manager-written keys)
        try:
            row = self.db.query_one("SELECT outputs_json FROM jobs WHERE id=?", (self.run_id,))
            import json

            cur = json.loads(row["outputs_json"]) if row and row["outputs_json"] else {}
            cur.update(json.loads(payload))
            self.db.execute(
                "UPDATE jobs SET outputs_json=?, updated_at=? WHERE id=?", (dumps(cur), now(), self.run_id)
            )
        except Exception as e:  # pragma: no cover
            print(format_line(f"set_output failed: {e}", "warning"), flush=True)

    @property
    def store(self) -> Any:
        if not self._store_tried:
            self._store_tried = True
            try:
                from forge.jobs.repo import open_store

                self._store = open_store()
            except Exception:
                self._store = None
        return self._store

    # -- lifecycle ---------------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.metrics_store.flush()
        except Exception:  # pragma: no cover
            pass

    @property
    def pid(self) -> int:
        return os.getpid()
