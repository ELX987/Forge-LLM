"""Log capture helpers.

A job worker's stdout/stderr are redirected by the JobManager into ``runs/<id>/log.txt``; ``RunContext.log``
prints formatted lines (``2026-01-01T12:00:00Z [INFO] message``) to stdout, and :func:`setup_worker_logging`
routes the python ``logging`` module to the same stream. Readers use :func:`read_log` (byte offsets, so a
live tail never re-reads the file) and :func:`tail_lines`.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.util import utc_iso

LOG_NAME = "log.txt"
_LINE_RE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\dT[\d:.]+Z) \[(?P<level>[A-Za-z]+)\] (?P<msg>.*)$")


def log_path(run_dir: str | os.PathLike) -> Path:
    return Path(run_dir) / LOG_NAME


def format_line(msg: str, level: str = "info") -> str:
    return f"{utc_iso()} [{level.upper()}] {msg}"


def parse_line(line: str) -> dict[str, Any]:
    """``{"ts", "level", "msg"}``; lines without the standard prefix (raw stdout/stderr) get level ``RAW``."""
    m = _LINE_RE.match(line)
    if m:
        return {"ts": m.group("ts"), "level": m.group("level"), "msg": m.group("msg")}
    return {"ts": None, "level": "RAW", "msg": line}


def read_log(path: str | os.PathLike, offset: int = 0, max_bytes: int = 1 << 20) -> tuple[str, int]:
    """Read up to ``max_bytes`` from byte ``offset``. Returns ``(text, new_offset)``. Never splits a UTF-8
    sequence or a line: the returned text ends at the last newline unless the file ended without one and
    nothing more was available. Missing file -> ``("", offset)``."""
    p = Path(path)
    if not p.exists():
        return "", offset
    size = p.stat().st_size
    if offset > size:  # truncated/rotated
        offset = 0
    with open(p, "rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
    if not data:
        return "", offset
    if len(data) == max_bytes or not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        if cut >= 0:
            data = data[: cut + 1]
        elif len(data) < max_bytes and offset + len(data) >= size:
            pass  # partial last line with no newline yet: return what exists
        else:
            data = data  # one giant line: return as is
    return data.decode("utf-8", errors="replace"), offset + len(data)


def tail_lines(path: str | os.PathLike, n: int = 200, block: int = 1 << 16) -> tuple[list[str], int]:
    """Last ``n`` lines of a file -> ``(lines, end_offset)`` (efficient for big logs)."""
    p = Path(path)
    if not p.exists():
        return [], 0
    size = p.stat().st_size
    if n <= 0 or size == 0:
        return [], size
    data = b""
    pos = size
    with open(p, "rb") as f:
        while pos > 0 and data.count(b"\n") <= n:
            step = min(block, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-n:], size


@dataclass
class LogTailer:
    """Incremental reader: ``poll()`` returns new complete lines since the last call."""

    path: Path
    offset: int = 0

    def poll(self, max_bytes: int = 1 << 20) -> list[str]:
        text, self.offset = read_log(self.path, self.offset, max_bytes)
        return text.splitlines() if text else []


class _StdoutHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            print(format_line(msg, record.levelname), file=sys.stdout, flush=True)
        except Exception:  # pragma: no cover
            self.handleError(record)


_SETUP_LOCK = threading.Lock()


def setup_worker_logging(level: str = "INFO") -> None:
    """Route python logging of a worker process to stdout in the standard line format (idempotent)."""
    with _SETUP_LOCK:
        root = logging.getLogger()
        if any(isinstance(h, _StdoutHandler) for h in root.handlers):
            return
        for h in list(root.handlers):
            root.removeHandler(h)
        h = _StdoutHandler()
        h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(h)
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        logging.captureWarnings(True)
