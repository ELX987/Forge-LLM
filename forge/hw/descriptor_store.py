"""Persistence of chip descriptors: sqlite table ``hw_descriptors`` (queryable, in ``forge.db``) plus a YAML copy
per descriptor under ``<FORGE_HOME>/hw/descriptors/<id>.yaml`` (human readable / diffable / hand-editable
templates; ``configs/chips/*.yaml`` use the same format).

``save_descriptor(d, report=None)`` upserts by ``d.id``; the optional probe *report* (raw micro-benchmark results,
throttling record, feature confirmations - everything that does not fit the descriptor schema) is stored next to
it as JSON in the table and as ``<id>.report.json``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forge.core.db import Database, get_db, register_migration
from forge.core.errors import NotFound
from forge.core.schemas import ChipDescriptor
from forge.core.util import atomic_write_text, now

log = logging.getLogger("forge.hw.store")

register_migration(
    "hw",
    1,
    """
CREATE TABLE IF NOT EXISTS hw_descriptors(
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    name TEXT NOT NULL,
    family TEXT,
    sm INTEGER,
    tier TEXT,
    n_measured INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'probe',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    descriptor_json TEXT NOT NULL,
    report_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_hw_desc_fp ON hw_descriptors(fingerprint, created_at);
""",
)


def descriptors_dir() -> Path:
    """``<FORGE_HOME>/hw/descriptors`` (created)."""
    from forge.core.paths import data_dir

    return data_dir("hw/descriptors")


def _db(db: Database | None) -> Database:
    return db or get_db()


def descriptor_to_yaml(d: ChipDescriptor) -> str:
    """YAML text of a descriptor (falls back to JSON text when PyYAML is missing)."""
    data = d.model_dump(mode="json")
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    except Exception:
        return json.dumps(data, indent=2)


def descriptor_from_yaml(text: str) -> ChipDescriptor:
    """Parse YAML (or JSON) text into a validated :class:`ChipDescriptor`."""
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("descriptor file must contain a mapping")
    return ChipDescriptor.model_validate(data)


def load_descriptor_file(path: str | Path) -> ChipDescriptor:
    """Load a descriptor from a YAML/JSON file (e.g. ``configs/chips/*.yaml``)."""
    return descriptor_from_yaml(Path(path).read_text(encoding="utf-8"))


def save_descriptor(d: ChipDescriptor, *, report: dict[str, Any] | None = None, source: str = "probe",
                    db: Database | None = None, write_files: bool = True) -> str:
    """Insert or replace descriptor ``d.id`` (and its probe report). Returns the id."""
    from forge.core.schemas import Provenance

    database = _db(db)
    t = now()
    n_meas = sum(1 for v in d.probe_provenance.values() if v == Provenance.measured)
    existing = database.query_one("SELECT created_at FROM hw_descriptors WHERE id=?", (d.id,))
    created = existing["created_at"] if existing else t
    database.execute(
        "INSERT OR REPLACE INTO hw_descriptors(id, fingerprint, name, family, sm, tier, n_measured, source, "
        "created_at, updated_at, descriptor_json, report_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (d.id, d.fingerprint, d.name, d.family, d.sm, d.tier, n_meas, source, created, t,
         d.model_dump_json(), json.dumps(report, default=str) if report is not None else None),
    )
    if write_files:
        try:
            base = descriptors_dir()
            atomic_write_text(base / f"{_safe(d.id)}.yaml", descriptor_to_yaml(d))
            if report is not None:
                atomic_write_text(base / f"{_safe(d.id)}.report.json", json.dumps(report, indent=2, default=str))
        except OSError as e:  # DB write succeeded; files are a convenience
            log.warning("could not write descriptor files: %s", e)
    return d.id


def _safe(id_: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in id_)


def load_descriptor(id: str, *, db: Database | None = None) -> ChipDescriptor:
    """Descriptor by id. Raises :class:`NotFound`."""
    row = _db(db).query_one("SELECT descriptor_json FROM hw_descriptors WHERE id=?", (id,))
    if row is None:
        raise NotFound(f"no chip descriptor {id!r}", details={"id": id})
    return ChipDescriptor.model_validate_json(row["descriptor_json"])


def load_report(id: str, *, db: Database | None = None) -> dict[str, Any] | None:
    """Raw probe report stored with a descriptor (None if there is none). Raises :class:`NotFound`."""
    row = _db(db).query_one("SELECT report_json FROM hw_descriptors WHERE id=?", (id,))
    if row is None:
        raise NotFound(f"no chip descriptor {id!r}", details={"id": id})
    return json.loads(row["report_json"]) if row["report_json"] else None


def list_descriptors(*, limit: int = 200, offset: int = 0, fingerprint: str | None = None,
                     db: Database | None = None) -> list[dict[str, Any]]:
    """Summaries (newest first): id, name, family, sm, tier, fingerprint, n_measured, source, created_at."""
    sql = ("SELECT id, fingerprint, name, family, sm, tier, n_measured, source, created_at, updated_at "
           "FROM hw_descriptors")
    params: list[Any] = []
    if fingerprint:
        sql += " WHERE fingerprint=?"
        params.append(fingerprint)
    sql += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
    params += [int(limit), int(offset)]
    return _db(db).query(sql, params)


def delete_descriptor(id: str, *, db: Database | None = None) -> bool:
    """Remove a descriptor (and its files). Returns True if it existed."""
    database = _db(db)
    existed = database.query_one("SELECT 1 AS x FROM hw_descriptors WHERE id=?", (id,)) is not None
    database.execute("DELETE FROM hw_descriptors WHERE id=?", (id,))
    for suffix in (".yaml", ".report.json"):
        try:
            (descriptors_dir() / f"{_safe(id)}{suffix}").unlink()
        except OSError:
            pass
    return existed


def latest_for_fingerprint(fingerprint: str, *, db: Database | None = None,
                           measured_only: bool = False) -> ChipDescriptor | None:
    """Newest stored descriptor of a device model (optionally only ones with measured fields)."""
    sql = "SELECT descriptor_json FROM hw_descriptors WHERE fingerprint=?"
    if measured_only:
        sql += " AND n_measured>0"
    sql += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
    row = _db(db).query_one(sql, (fingerprint,))
    return ChipDescriptor.model_validate_json(row["descriptor_json"]) if row else None


def latest_report_for_fingerprint(fingerprint: str, *, db: Database | None = None) -> dict[str, Any] | None:
    """Newest probe report (with ``features``) of a device model, or None."""
    row = _db(db).query_one(
        "SELECT report_json FROM hw_descriptors WHERE fingerprint=? AND report_json IS NOT NULL "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1", (fingerprint,))
    return json.loads(row["report_json"]) if row else None


def next_revision(fingerprint: str, *, db: Database | None = None) -> int:
    """1 + number of stored descriptors with this fingerprint (for ``-r<N>`` ids)."""
    row = _db(db).query_one("SELECT COUNT(*) AS n FROM hw_descriptors WHERE fingerprint=?", (fingerprint,))
    return int(row["n"]) + 1 if row else 1
