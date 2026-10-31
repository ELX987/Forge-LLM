# Runbook: store garbage-collection drill

Goal: prove you can reclaim space without losing anything you need (Plan C-121, C-123).
Do this on a scratch `FORGE_HOME` first (for example `FORGE_HOME=$(mktemp -d) forge serve-ui`), then on the real one.

## How GC works

The store is content-addressed and chunk-deduplicated. `gc` is mark-and-sweep: everything reachable from a **pin**
(and from the lineage of pinned artifacts) is kept; unreachable chunks and manifests are deleted. Delta-chain
keyframes that still have live deltas are never collected.

## Drill

1. **Stats before**: UI *Storage*, or `GET /api/v1/storage/stats`. Note logical bytes, unique bytes, dedup ratio.
2. **Pin what matters**: final models, best checkpoints, eval-report artifacts, calibration sets
   (`POST /api/v1/artifacts/{hash}/pin`, with a reason). Unpin scratch artifacts.
3. **Dry run**: `POST /api/v1/storage/gc {"dry_run": true}` (UI: *Storage -> GC (dry run)*). Read the report:
   number of artifacts and bytes that would be deleted, by kind. **Check that nothing you need is listed.**
4. **Snapshot** (production only): stop the server and copy `forge.db` and `store/` (see `restore-drill.md`).
5. **Run**: `POST /api/v1/storage/gc {"dry_run": false}`. Large runs can be a `gc` job; watch its log.
6. **Verify**: for each pinned artifact, `GET /api/v1/artifacts/{hash}`, then materialise one (checkpoint export or
   `ArtifactStore.materialize_artifact`) and load it. `ArtifactStore.verify(hash)` must be true.
7. **Stats after**: compare with step 1. Record reclaimed bytes and elapsed time.

## Pass criteria

* The dry-run report equals what the real run deleted (same counts).
* All pinned artifacts verify and materialise bit-exactly (hashes match).
* Lineage of pinned artifacts is still complete (`GET /artifacts/{hash}/lineage`).

## If something is missing

Stop. Do not run GC again. Restore `forge.db` and `store/` from the snapshot (`restore-drill.md`) and report the
artifact hash and the GC report.
