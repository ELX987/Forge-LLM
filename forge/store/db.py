"""SQLite tables of the artifact store (registered through :mod:`forge.core.db` migrations).

Tables (all prefixed ``store_``):

* ``store_objects``  one row per content-addressed object (``obj`` = bytes on disk, ``chunked`` = logical blob
  made of chunks, ``manifest`` = signed manifest JSON). ``rooted`` marks user-visible artifacts (GC roots).
* ``store_refs``     hard references (``chunk`` membership, manifest ``member`` objects, delta-chain ``chain``).
* ``store_manifests`` signature / hash-chain record per manifest.
* ``store_edges``    lineage DAG (``derived_from`` ...), *not* a GC reference.
* ``store_pins``     explicit GC roots.
* ``store_ckpt``     checkpoint series index (run, step, keyframe/delta, tags, metric).
"""
from __future__ import annotations

from forge.core.db import register_migration

register_migration(
    "store",
    1,
    """
CREATE TABLE IF NOT EXISTS store_objects(
    hash TEXT PRIMARY KEY,
    otype TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    stored_size INTEGER NOT NULL DEFAULT 0,
    codec TEXT NOT NULL DEFAULT 'raw',
    tier TEXT NOT NULL DEFAULT 'hot',
    rooted INTEGER NOT NULL DEFAULT 0,
    nchunks INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    accessed_at REAL NOT NULL,
    meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_store_objects_kind ON store_objects(kind, rooted);
CREATE INDEX IF NOT EXISTS idx_store_objects_tier ON store_objects(tier);
CREATE TABLE IF NOT EXISTS store_refs(
    src TEXT NOT NULL,
    ord INTEGER NOT NULL,
    dst TEXT NOT NULL,
    rtype TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(src, ord)
);
CREATE INDEX IF NOT EXISTS idx_store_refs_dst ON store_refs(dst);
CREATE TABLE IF NOT EXISTS store_manifests(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    prev_chain TEXT NOT NULL,
    chain TEXT NOT NULL,
    sig TEXT NOT NULL,
    sig_alg TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS store_edges(
    child TEXT NOT NULL,
    parent TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'derived_from',
    recipe_hash TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY(child, parent, relation)
);
CREATE INDEX IF NOT EXISTS idx_store_edges_parent ON store_edges(parent);
CREATE TABLE IF NOT EXISTS store_pins(
    hash TEXT PRIMARY KEY,
    reason TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS store_ckpt(
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    manifest_hash TEXT NOT NULL,
    is_keyframe INTEGER NOT NULL,
    base_hash TEXT,
    keyframe_hash TEXT,
    metric REAL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    dropped INTEGER NOT NULL DEFAULT 0,
    logical_bytes INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY(run_id, step)
);
CREATE INDEX IF NOT EXISTS idx_store_ckpt_manifest ON store_ckpt(manifest_hash);
""",
)
