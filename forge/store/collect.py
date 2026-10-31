"""Mark-and-sweep garbage collection for :class:`forge.store.store.ArtifactStore`.

Invariants (exercised by the chaos tests in ``tests/test_store_gc.py``):

1. *Pinned* objects and everything reachable from them are never collected.
2. Every *rooted* artifact (user-visible, not released) is a root unless ``strict_pins=True``.
3. Reachability follows **hard references**: chunk membership, manifest members and delta-chain bases
   (``rtype='chain'``).  A keyframe that any live delta (transitively) depends on is therefore always live; the
   report counts such objects as ``chain_protected``.
4. Anything touched within ``grace_s`` seconds is skipped (concurrent writers have not yet attached their data).
5. The sweep re-checks the age predicate inside a write transaction and unlinks files while holding the lock,
   and a writer that re-uses an existing object bumps ``accessed_at`` first (see ``ArtifactStore._touch``).
6. A final defensive assertion refuses to delete anything that is pinned or referenced by a live object.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from forge.core.errors import StoreError
from forge.core.util import now

if TYPE_CHECKING:  # pragma: no cover
    from forge.store.store import ArtifactStore

log = logging.getLogger("forge.store.gc")

_BATCH = 400


class GCReport(BaseModel):
    """Outcome (or dry-run plan) of a garbage collection."""

    dry_run: bool = Field(description="True when nothing was deleted.")
    strict_pins: bool = Field(default=False, description="Only pins were treated as roots.")
    grace_s: float = Field(default=300.0, description="Objects touched more recently than this are never collected.")
    objects_total: int = Field(default=0, description="Objects (chunks, blobs, manifests) in the index.")
    objects_live: int = Field(default=0, description="Objects reachable from a root.")
    roots: int = Field(default=0, description="Number of GC roots (pins + rooted artifacts).")
    pinned: int = Field(default=0, description="Number of pins.")
    objects_garbage: int = Field(default=0, description="Unreachable objects eligible for deletion.")
    bytes_garbage: int = Field(default=0, description="On-disk bytes of the garbage objects.")
    logical_garbage_bytes: int = Field(default=0, description="Logical (uncompressed) bytes of the garbage objects.")
    deleted_objects: int = Field(default=0, description="Objects actually deleted (0 on a dry run).")
    bytes_freed: int = Field(default=0, description="On-disk bytes actually freed.")
    chain_protected_objects: int = Field(default=0, description="Live objects kept alive only by a delta-chain reference.")
    chain_protected_bytes: int = Field(default=0, description="On-disk bytes of chain-protected objects.")
    skipped_young: int = Field(default=0, description="Garbage objects skipped because they are within the grace period.")
    orphan_files: int = Field(default=0, description="Files on disk without an index row (older than the grace period).")
    orphan_bytes: int = Field(default=0, description="Bytes of orphan files.")
    by_kind: dict[str, dict[str, int]] = Field(default_factory=dict, description="Garbage per kind: {kind: {objects, bytes}}.")
    garbage_sample: list[dict[str, Any]] = Field(default_factory=list, description="Up to 200 garbage objects.")
    errors: list[str] = Field(default_factory=list, description="Non-fatal problems encountered.")
    duration_s: float = Field(default=0.0, description="Wall time.")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _mark(adj: dict[str, list[tuple[str, str]]], roots: set[str], follow_chain: bool) -> set[str]:
    live: set[str] = set()
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n in live:
            continue
        live.add(n)
        for dst, rtype in adj.get(n, ()):
            if rtype == "chain" and not follow_chain:
                continue
            if dst not in live:
                stack.append(dst)
    return live


def run_gc(store: ArtifactStore, *, dry_run: bool = True, grace_s: float = 300.0, strict_pins: bool = False) -> GCReport:
    t0 = time.perf_counter()
    mark_time = now()
    db = store.db
    inflight = store.inflight_writers()
    if inflight:  # a checkpoint save (or other long writer) is mid-flight: its payload objects have no referrer yet
        grace_s = max(grace_s, float(store.INFLIGHT_TTL_S))
        log.info("gc: %d in-flight writer(s); grace period widened to %.0fs", inflight, grace_s)
    rep = GCReport(dry_run=dry_run, strict_pins=strict_pins, grace_s=grace_s)

    rows = {r["hash"]: r for r in db.query("SELECT hash, otype, kind, size, stored_size, codec, tier, rooted, created_at, accessed_at FROM store_objects")}
    adj: dict[str, list[tuple[str, str]]] = {}
    for r in db.query("SELECT src, dst, rtype FROM store_refs"):
        adj.setdefault(r["src"], []).append((r["dst"], r["rtype"]))
    pins = {r["hash"] for r in db.query("SELECT hash FROM store_pins")}
    roots = {p for p in pins if p in rows}
    if not strict_pins:
        roots |= {h for h, r in rows.items() if r["rooted"]}
    live = _mark(adj, roots, follow_chain=True)
    live_nochain = _mark(adj, roots, follow_chain=False)

    rep.objects_total = len(rows)
    rep.objects_live = len(live)
    rep.roots = len(roots)
    rep.pinned = len(pins)
    chain_only = live - live_nochain
    rep.chain_protected_objects = len(chain_only)
    rep.chain_protected_bytes = int(sum(rows[h]["stored_size"] for h in chain_only if h in rows))

    cutoff = mark_time - grace_s
    garbage: list[str] = []
    for h, r in rows.items():
        if h in live:
            continue
        if max(r["created_at"], r["accessed_at"]) > cutoff:
            rep.skipped_young += 1
            continue
        garbage.append(h)

    # defensive invariant: never delete pinned / chain-referenced / reachable objects
    gset = set(garbage)
    violated = (gset & pins) | (gset & live)
    if violated:
        raise StoreError(f"gc invariant violated for {len(violated)} objects; refusing to sweep", details={"sample": sorted(violated)[:5]})
    for src, lst in adj.items():
        if src in live:
            for dst, _rt in lst:
                if dst in gset:  # pragma: no cover - impossible by construction
                    raise StoreError(f"gc invariant violated: live {src} references garbage {dst}")

    for h in garbage:
        r = rows[h]
        rep.objects_garbage += 1
        rep.bytes_garbage += int(r["stored_size"])
        rep.logical_garbage_bytes += int(r["size"])
        d = rep.by_kind.setdefault(r["kind"], {"objects": 0, "bytes": 0})
        d["objects"] += 1
        d["bytes"] += int(r["stored_size"])
        if len(rep.garbage_sample) < 200:
            rep.garbage_sample.append({"hash": h, "kind": r["kind"], "otype": r["otype"], "stored_size": r["stored_size"], "size": r["size"]})

    known_files = _orphan_scan(store, rows, cutoff, rep, delete=not dry_run)

    if not dry_run and garbage:
        _sweep(store, rows, garbage, cutoff, rep)
    rep.duration_s = time.perf_counter() - t0
    _ = known_files
    return rep


def _orphan_scan(store: ArtifactStore, rows: dict[str, dict[str, Any]], cutoff: float, rep: GCReport, delete: bool) -> int:
    """Count (and optionally delete) object files that have no index row, plus stale temp files."""
    expected: set[str] = set()
    for h, r in rows.items():
        if r["otype"] == "chunked":
            continue
        for t in ("hot", "warm", "cold"):
            expected.add(str(store._path(h, t, r["codec"], r["otype"])))
    n = 0
    for tier in ("hot", "warm", "cold"):
        base = store._tier_dir(tier)
        for sub in ("objects", "tmp"):
            d = base / sub
            if not d.exists():
                continue
            for dirpath, _dirs, files in os.walk(d):
                for fn in files:
                    p = os.path.join(dirpath, fn)
                    if p in expected:
                        continue
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    if st.st_mtime > cutoff:
                        continue
                    rep.orphan_files += 1
                    rep.orphan_bytes += st.st_size
                    n += 1
                    if delete:
                        try:
                            os.unlink(p)
                        except OSError as e:  # pragma: no cover
                            rep.errors.append(f"orphan {p}: {e}")
    return n


def _sweep(store: ArtifactStore, rows: dict[str, dict[str, Any]], garbage: list[str], cutoff: float, rep: GCReport) -> None:
    db = store.db
    # composites first so chunk rows lose their referrers before the chunks themselves are removed
    order = sorted(garbage, key=lambda h: 0 if rows[h]["otype"] != "obj" else 1)
    for _pass in range(6):  # blocked items (e.g. a base whose delta sits in a later batch) are retried
        before = rep.deleted_objects
        order = _sweep_pass(store, rows, order, cutoff, rep)
        if not order or rep.deleted_objects == before:
            break
    log.info("gc: deleted %d objects, freed %d bytes", rep.deleted_objects, rep.bytes_freed)


def _sweep_pass(store: ArtifactStore, rows: dict[str, dict[str, Any]], order: list[str], cutoff: float, rep: GCReport) -> list[str]:
    db = store.db
    remaining: list[str] = []
    for i in range(0, len(order), _BATCH):
        batch = order[i : i + _BATCH]
        ph = ",".join("?" * len(batch))
        with db.transaction():
            ok = [
                r["hash"]
                for r in db.query(
                    f"SELECT hash FROM store_objects WHERE hash IN ({ph}) AND accessed_at<=? AND created_at<=?",
                    (*batch, cutoff, cutoff),
                )
            ]
            if not ok:
                continue
            ph2 = ",".join("?" * len(ok))
            # re-validate: nothing live may reference these (a concurrent writer could have attached them)
            still_ref = db.query(
                f"SELECT DISTINCT dst FROM store_refs WHERE dst IN ({ph2}) AND src NOT IN ({ph2})", (*ok, *ok)
            )
            blocked = {r["dst"] for r in still_ref}
            remaining.extend(h for h in ok if h in blocked)
            ok = [h for h in ok if h not in blocked]
            pinned = {r["hash"] for r in db.query(f"SELECT hash FROM store_pins WHERE hash IN ({ph2})", (*ok,))} if ok else set()
            ok = [h for h in ok if h not in pinned]
            if not ok:
                continue
            ph3 = ",".join("?" * len(ok))
            for h in ok:
                r = rows[h]
                if r["otype"] == "chunked":
                    continue
                for t in ("hot", "warm", "cold"):
                    p = store._path(h, t, r["codec"], r["otype"])
                    try:
                        p.unlink()
                        rep.bytes_freed += int(r["stored_size"])
                        break
                    except FileNotFoundError:
                        continue
                    except OSError as e:  # pragma: no cover
                        rep.errors.append(f"unlink {p}: {e}")
            db.execute(f"DELETE FROM store_refs WHERE src IN ({ph3})", (*ok,))
            db.execute(f"DELETE FROM store_objects WHERE hash IN ({ph3})", (*ok,))
            db.execute(f"DELETE FROM store_ckpt WHERE manifest_hash IN ({ph3})", (*ok,))
            rep.deleted_objects += len(ok)
    return remaining
