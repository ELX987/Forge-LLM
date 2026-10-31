"""Tables owned by the jobs/api layer: jobs, models, datasets, configs, recipes, deployments.

(``evals``, ``kernel_bench`` and ``claims`` are written by other modules; the API only reads them, defensively.)
"""
from __future__ import annotations

from forge.core.db import register_migration

register_migration(
    "jobs_api",
    1,
    """
CREATE TABLE IF NOT EXISTS jobs(
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    pid INTEGER,
    exit_code INTEGER,
    progress REAL NOT NULL DEFAULT 0,
    progress_msg TEXT NOT NULL DEFAULT '',
    error TEXT,
    outputs_json TEXT NOT NULL DEFAULT '{}',
    run_dir TEXT,
    gpus_json TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type, created_at);
CREATE TABLE IF NOT EXISTS models(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'local',
    path_or_repo TEXT NOT NULL DEFAULT '',
    revision TEXT,
    hash TEXT,
    licence TEXT,
    distill_allowed INTEGER,
    params_b REAL,
    status TEXT NOT NULL DEFAULT 'ready',
    size_bytes INTEGER,
    info_json TEXT NOT NULL DEFAULT '{}',
    tags_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'raw',
    format TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    num_files INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER,
    manifest_hash TEXT,
    licence TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS configs(
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    params_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_configs_type ON configs(job_type);
CREATE TABLE IF NOT EXISTS recipes(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    yaml TEXT NOT NULL,
    recipe_hash TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deployments(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    artifact_ref TEXT NOT NULL,
    serve_config_json TEXT NOT NULL DEFAULT '{}',
    job_id TEXT,
    port INTEGER,
    status TEXT NOT NULL DEFAULT 'created',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
""",
)
