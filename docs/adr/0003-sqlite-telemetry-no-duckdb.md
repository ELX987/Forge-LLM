# ADR 0003: SQLite for state and telemetry, optional Parquet export, no DuckDB

- Status: accepted
- Date: 2026-10-30
- Related Plan items: C-015 ("Parquet metrics, embedded query"), DT-1, NFR-R1

## Context

The Plan describes a "telemetry lake" with Parquet files and an embedded analytical query engine. Forge needs
to store (a) control-plane state (jobs, configs, deployments, claims, eval reports, kernel benchmark rows) and
(b) high-rate time series (training metrics, GPU telemetry).

DuckDB is a good embedded OLAP engine, but it is another binary dependency to install on every rented box and
cannot share a transaction with the state tables.

## Decision

* **SQLite (stdlib `sqlite3`, WAL mode)** is the only required database. Per-thread connections, versioned
  migrations registered per module (`forge.core.db.register_migration`).
* Time series live in a `metrics` table keyed `(run_id, key, step)`. The API downsamples on read
  (`max_points`, rollups of 1/10/100 steps) so charts stay fast with 100k+ points.
* **Parquet is an optional export** (`MetricsStore.export_parquet`) when `pyarrow` is installed; it is not
  required for any feature.
* No DuckDB. If analytical queries outgrow SQLite, DuckDB can read the exported Parquet files without a code
  change to producers.

## Consequences

* One file (`$FORGE_HOME/forge.db`) to back up, restore and copy; see `docs/runbooks/restore-drill.md`.
* Single-writer limit of SQLite is acceptable: writers are the API process and one worker per job, with WAL and
  short transactions. Metric writes are batched by the worker context.
* Very long runs (tens of millions of points) should enable rollup retention or Parquet export.
