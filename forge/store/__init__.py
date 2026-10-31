"""forge.store: content-addressed, chunk-deduplicated artifact store (see docs/ARCHITECTURE.md section 5.1).

Cheap to import: torch-dependent modules (``checkpoints``) and the lossless codec load lazily.

    from forge.store import get_store, ArtifactStore
    from forge.store.checkpoints import CheckpointWriter, CheckpointReader
    from forge.store import lossless
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from forge.store.chunking import ChunkerParams
from forge.store.store import TIERS, ArtifactStore

__all__ = [
    "ArtifactStore", "ChunkerParams", "TIERS", "get_store", "reset_stores", "GCReport",
    "CheckpointWriter", "CheckpointReader", "RetentionPolicy", "CheckpointConfig",
]

_STORES: dict[str, ArtifactStore] = {}
_LOCK = threading.Lock()


def get_store(root: str | os.PathLike | None = None) -> ArtifactStore:
    """Process-wide store singleton per root (default ``forge.core.paths.store_dir()``, which follows ``FORGE_HOME``)."""
    from forge.core.paths import store_dir

    key = str(Path(root).expanduser().resolve()) if root is not None else str(store_dir().resolve())
    with _LOCK:
        st = _STORES.get(key)
        if st is None:
            st = ArtifactStore(root)
            _STORES[key] = st
    return st


def reset_stores() -> None:
    """Forget cached store instances (tests, FORGE_HOME changes)."""
    with _LOCK:
        _STORES.clear()


def __getattr__(name: str) -> Any:
    if name == "GCReport":
        from forge.store.collect import GCReport

        return GCReport
    if name in ("CheckpointWriter", "CheckpointReader", "RetentionPolicy", "CheckpointConfig"):
        from forge.store import checkpoints

        return getattr(checkpoints, name)
    raise AttributeError(name)
