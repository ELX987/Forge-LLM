"""RunContext protocol: how long-running work reports progress and checks for cancellation.

The job worker supplies a real implementation (``forge.jobs``); :class:`NullContext` is an in-memory
implementation for tests, scripts and notebooks.
"""
from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from forge.core.util import new_id

if TYPE_CHECKING:  # pragma: no cover
    from forge.store import ArtifactStore


@runtime_checkable
class RunContext(Protocol):
    run_id: str
    run_dir: Path
    params: dict

    def log(self, msg: str, level: str = "info") -> None: ...

    def metric(self, step: int, **values: float) -> None: ...

    def progress(self, frac: float, msg: str = "") -> None: ...

    def should_stop(self) -> bool: ...

    def set_output(self, key: str, value: Any) -> None: ...

    @property
    def store(self) -> ArtifactStore | None: ...


class NullContext:
    """In-memory :class:`RunContext`. Inspect ``.logs``, ``.log_records``, ``.metrics``, ``.outputs``;
    set ``.stop = True`` (or call :meth:`request_stop`) to emulate cancellation."""

    def __init__(
        self,
        run_id: str | None = None,
        run_dir: str | Path | None = None,
        params: dict | None = None,
        store: Any = None,
        *,
        echo: bool = False,
    ) -> None:
        self.run_id = run_id or new_id("run")
        if run_dir is None:
            run_dir = Path(tempfile.mkdtemp(prefix="forge_nullctx_"))
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.params: dict = dict(params or {})
        self._store = store
        self.echo = echo
        self.logs: list[str] = []  # message text only
        self.log_records: list[tuple[str, str]] = []  # (level, message)
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.progress_events: list[tuple[float, str]] = []
        self.outputs: dict[str, Any] = {}
        self.stop: bool = False
        self._lock = threading.Lock()
        self._logger = logging.getLogger("forge.run")

    # -- RunContext API ---------------------------------------------------------------------
    def log(self, msg: str, level: str = "info") -> None:
        with self._lock:
            self.logs.append(msg)
            self.log_records.append((level, msg))
        self._logger.log(getattr(logging, level.upper(), logging.INFO), msg)
        if self.echo:
            print(f"[{level}] {msg}")

    def metric(self, step: int, **values: float) -> None:
        with self._lock:
            self.metrics.append((int(step), {k: float(v) for k, v in values.items()}))

    def progress(self, frac: float, msg: str = "") -> None:
        frac = min(1.0, max(0.0, float(frac)))
        with self._lock:
            self.progress_events.append((frac, msg))

    def should_stop(self) -> bool:
        return bool(self.stop)

    def set_output(self, key: str, value: Any) -> None:
        with self._lock:
            self.outputs[key] = value

    @property
    def store(self) -> Any:
        return self._store

    @store.setter
    def store(self, value: Any) -> None:
        self._store = value

    # -- conveniences -----------------------------------------------------------------------
    def request_stop(self) -> None:
        self.stop = True

    @property
    def last_progress(self) -> float:
        return self.progress_events[-1][0] if self.progress_events else 0.0

    def metric_series(self, key: str) -> list[tuple[int, float]]:
        return [(s, v[key]) for s, v in self.metrics if key in v]
