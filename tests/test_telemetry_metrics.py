"""MetricsStore: batching writer, queries, downsampling, rollups, parquet export."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import home  # noqa: E402,F401

from forge.telemetry.metrics import MetricsStore, lttb, minmax_bucket  # noqa: E402


@pytest.fixture
def store(home):
    from forge.core.db import get_db

    s = MetricsStore(get_db(), flush_interval=0.05)
    yield s
    s.close()


def test_write_flush_query_roundtrip(store):
    for step in range(10):
        store.write("r1", step, {"loss": 10.0 - step, "lr": 1e-3})
    store.write("r2", 0, {"loss": 99.0})
    q = store.query("r1")
    assert q["keys"] == ["loss", "lr"]
    assert [p[0] for p in q["points"]["loss"]] == list(range(10))
    assert q["points"]["loss"][3][1] == 7.0
    assert store.keys("r2") == ["loss"]
    assert store.latest("r1")["loss"]["step"] == 9
    assert store.latest("r1")["loss"]["value"] == 1.0
    assert store.count("r1") == 20 and store.count("r1", "loss") == 10
    assert set(store.runs()) == {"r1", "r2"}


def test_background_writer_flushes_without_explicit_flush(store):
    import time

    store.write("bg", 1, {"x": 1.0})
    deadline = time.time() + 5
    while time.time() < deadline:
        rows = store.db.query("SELECT * FROM metrics WHERE run_id='bg'")
        if rows:
            break
        time.sleep(0.05)
    assert rows, "writer thread should flush within flush_interval"


def test_non_finite_and_non_numeric(store):
    store.write("r", 0, {"nan": float("nan"), "inf": float("inf"), "ok": 1, "bad": "text", "b": True})
    q = store.query("r")
    assert q["points"]["nan"][0][1] is None and q["points"]["inf"][0][1] is None
    assert q["points"]["ok"][0][1] == 1.0
    assert q["points"]["b"][0][1] == 1.0
    assert "bad" not in q["keys"]


def test_keys_filter_and_since_step(store):
    for s in range(20):
        store.write("r", s, {"a": s, "b": -s})
    q = store.query("r", keys=["a"], since_step=14)
    assert q["keys"] == ["a"]
    assert [p[0] for p in q["points"]["a"]] == [15, 16, 17, 18, 19]
    assert store.query("r", keys=["missing"])["points"]["missing"] == []


def test_since_seq_is_incremental(store):
    store.write("r", 0, {"a": 1})
    store.flush()
    rows, seq = store.since_seq("r", 0)
    assert len(rows) == 1
    assert store.since_seq("r", seq) == ([], seq)
    store.write("r", 1, {"a": 2})
    store.flush()
    rows2, seq2 = store.since_seq("r", seq)
    assert [r["step"] for r in rows2] == [1] and seq2 > seq
    assert store.max_seq("r") == seq2


def test_lttb_properties():
    pts = [(i, math.sin(i / 50.0), float(i)) for i in range(5000)]
    out = lttb(pts, 200)
    assert len(out) == 200
    assert out[0] == pts[0] and out[-1] == pts[-1]
    xs = [p[0] for p in out]
    assert xs == sorted(xs)
    # a single spike must survive the reduction
    spiky = [(i, 0.0, 0.0) for i in range(3000)]
    spiky[1234] = (1234, 100.0, 0.0)
    assert any(p[1] == 100.0 for p in lttb(spiky, 100))
    assert lttb(pts[:10], 50) == pts[:10]


def test_minmax_bucket_keeps_envelope():
    pts = [(i, float((i * 37) % 101), 0.0) for i in range(4000)]
    out = minmax_bucket(pts, 100)
    assert len(out) <= 110
    assert max(p[1] for p in out) == max(p[1] for p in pts)
    assert min(p[1] for p in out) == min(p[1] for p in pts)


def test_query_downsamples_to_max_points(store):
    store.db.executemany(
        "INSERT INTO metrics(run_id,key,step,value,ts) VALUES(?,?,?,?,?)", [("big", "v", i, float(i % 50), 0.0) for i in range(1500)]
    )
    q = store.query("big", max_points=100)
    assert len(q["points"]["v"]) == 100 and q["downsampled"]["v"] == "lttb"
    q2 = store.query("big", max_points=100, rollup="minmax")
    assert q2["downsampled"]["v"] == "minmax" and len(q2["points"]["v"]) <= 105
    raw = store.query("big", max_points=100, rollup="raw")
    assert len(raw["points"]["v"]) == 1500


def test_huge_series_uses_sql_buckets_with_bands(store):
    store.db.executemany(
        "INSERT INTO metrics(run_id,key,step,value,ts) VALUES(?,?,?,?,?)", [("huge", "v", i, float(i), 0.0) for i in range(5000)]
    )
    q = store.query("huge", max_points=50)  # 5000 > 20 * 50
    assert q["downsampled"]["v"] == "mean"
    assert len(q["points"]["v"]) <= 51
    assert "v" in q["bands"] and q["bands"]["v"][0][1] <= q["bands"]["v"][0][2]
    # means are monotonic for a monotonic series
    ys = [p[1] for p in q["points"]["v"]]
    assert ys == sorted(ys)


def test_compact_rollups(store):
    for i in range(1000):
        store.write("c", i, {"loss": float(i)})
    made = store.compact("c", (10, 100))
    assert made["10"] == 100 and made["100"] == 10
    q = store.query("c", rollup=100, max_points=100)
    assert q["downsampled"]["loss"] == "rollup100"
    assert len(q["points"]["loss"]) == 10
    assert q["bands"]["loss"][0][1] == 0.0 and q["bands"]["loss"][0][2] == 99.0
    made2 = store.compact("c", (10,), drop_raw_older_than_steps=100)
    assert made2["raw_deleted"] > 0
    assert store.count("c", "loss") < 1000


def test_delete_run(store):
    store.write("d", 0, {"a": 1})
    assert store.delete_run("d") == 1
    assert store.keys("d") == []


def test_export_parquet(store, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    for i in range(5):
        store.write("p", i, {"loss": float(i)})
    out = store.export_parquet("p", tmp_path / "m.parquet")
    t = pq.read_table(out)
    assert t.num_rows == 5 and set(t.column_names) == {"key", "step", "value", "ts"}


def test_get_metrics_store_is_cached_per_home(home):
    from forge.telemetry.metrics import get_metrics_store

    a = get_metrics_store()
    assert get_metrics_store() is a
