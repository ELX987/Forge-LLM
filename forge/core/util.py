"""Small dependency-free helpers shared across Forge modules."""
from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now() -> float:
    """Current UNIX time in seconds (float). All Forge timestamps are epoch seconds."""
    return time.time()


def utc_iso(ts: float | None = None) -> str:
    """ISO-8601 UTC string for an epoch timestamp (default: now)."""
    dt = datetime.fromtimestamp(now() if ts is None else ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_id(prefix: str = "") -> str:
    """Short random id such as ``job_1a2b3c4d5e6f``."""
    body = uuid.uuid4().hex[:12]
    return f"{prefix}_{body}" if prefix else body


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(text: str, default: str = "item") -> str:
    s = _SLUG_RE.sub("-", text.strip()).strip("-.")
    return s[:80] or default


def human_bytes(n: float) -> str:
    """1536 -> '1.50 KiB'."""
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0:
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} PiB"


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (temp file in same dir + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def atomic_write_text(path: str | os.PathLike, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))
