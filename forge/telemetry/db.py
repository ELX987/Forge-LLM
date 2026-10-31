"""Telemetry tables (registered with ``forge.core.db`` at import time)."""
from __future__ import annotations

from forge.core.db import register_migration

register_migration(
    "telemetry",
    1,
    """
CREATE TABLE IF NOT EXISTS metrics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    step INTEGER NOT NULL,
    value REAL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key_step ON metrics(run_id, key, step);
CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id, id);
CREATE TABLE IF NOT EXISTS metrics_rollup(
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    level INTEGER NOT NULL,
    bucket INTEGER NOT NULL,
    step REAL NOT NULL,
    mean REAL,
    vmin REAL,
    vmax REAL,
    n INTEGER NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY(run_id, key, level, bucket)
);
""",
)
