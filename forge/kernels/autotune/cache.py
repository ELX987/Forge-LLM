"""JSON autotune cache (Plan 6.6 steps 5-6).  Default location ``<FORGE_HOME>/autotune/<chip fingerprint>.json``.

Cache key = hash of ``(chip fingerprint, driver/compiler versions, op, dtype config, shape bucket, layout_id)``.
Only configs that passed the correctness gate and were measured un-throttled are ever stored.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from forge.core.hashing import hash_json

CACHE_VERSION = 1
_ENV_CACHE: dict[str, str] | None = None


def environment_versions() -> dict[str, str]:
    """Driver / compiler versions that invalidate tuned configs when they change."""
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    env: dict[str, str] = {}
    try:
        import torch

        env["torch"] = str(torch.__version__)
        env["cuda"] = str(torch.version.cuda)
    except Exception:
        env["torch"] = "none"
    try:
        import triton

        env["triton"] = str(triton.__version__)
    except Exception:
        env["triton"] = "none"
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        drv = pynvml.nvmlSystemGetDriverVersion()
        env["driver"] = drv.decode() if isinstance(drv, bytes) else str(drv)
    except Exception:
        env["driver"] = "unknown"
    _ENV_CACHE = env
    return env


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120] or "cpu"


def default_cache_path(chip_fingerprint: str) -> Path:
    from forge.core.paths import autotune_dir

    return autotune_dir() / f"{_safe_name(chip_fingerprint)}.json"


def make_key(fields: dict[str, Any]) -> str:
    return hash_json(fields)


class AutotuneCache:
    """Thread-safe JSON file cache with atomic writes; re-reads the file when it changed on disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._mtime: float | None = None
        self._load()

    # -- persistence -------------------------------------------------------------------------------
    def _load(self) -> None:
        try:
            st = self.path.stat()
        except OSError:
            self._entries, self._mtime = {}, None
            return
        if self._mtime == st.st_mtime and self._entries:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") == CACHE_VERSION:
                self._entries = dict(data.get("entries", {}))
            else:
                self._entries = {}
        except (OSError, ValueError):
            self._entries = {}  # corrupt cache => start fresh (it is only a cache)
        self._mtime = st.st_mtime

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "entries": self._entries}, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime

    # -- API ---------------------------------------------------------------------------------------
    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            self._load()
            e = self._entries.get(key)
            return dict(e) if e else None

    def put(self, key: str, fields: dict[str, Any], config: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "key": key,
            "fields": fields,
            "config": config,
            "median_ms": stats["median"],
            "iqr_ms": stats["iqr"],
            "n": stats["n"],
            "tuned_at": time.time(),
        }
        with self._lock:
            self._load()
            self._entries[key] = entry
            self._save()
        return entry

    def delete(self, key: str) -> bool:
        with self._lock:
            self._load()
            hit = self._entries.pop(key, None) is not None
            if hit:
                self._save()
            return hit

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            self._load()
            return [dict(v) for v in self._entries.values()]

    def clear(self) -> None:
        with self._lock:
            self._entries = {}
            self._save()

    def __len__(self) -> int:
        with self._lock:
            self._load()
            return len(self._entries)

    def check_regression(self, key: str, new_median_ms: float, tol: float = 0.05) -> dict[str, Any]:
        """Regression CI (Plan 6.6 step 6): flag when the median got more than ``tol`` slower than the stored baseline."""
        e = self.get(key)
        if e is None:
            return {"regressed": False, "baseline_ms": None, "ratio": None, "reason": "no baseline"}
        base = float(e["median_ms"])
        ratio = new_median_ms / base if base > 0 else float("inf")
        return {"regressed": ratio > 1.0 + tol, "baseline_ms": base, "ratio": ratio, "tol": tol,
                "reason": f"median {new_median_ms:.4g} ms vs baseline {base:.4g} ms"}
