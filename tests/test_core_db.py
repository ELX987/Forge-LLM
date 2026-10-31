from __future__ import annotations

import threading

import pytest

from forge.core import db as D
from forge.core.errors import ConfigError


def test_core_migration_and_idempotence(forge_home):
    db = D.get_db()
    assert {"kv", "stage_cache", "_migrations"} <= set(db.tables())
    assert D.migrate(db) == []  # nothing pending
    assert D.migrate(db) == []
    rows = db.query("SELECT module, version FROM _migrations ORDER BY module, version")
    assert {"module": "core", "version": 1} in rows
    assert D.get_db() is db  # cached per path


def test_register_new_migration_applied_lazily(forge_home):
    db = D.get_db()
    D.register_migration("t_mod_a", 1, "CREATE TABLE IF NOT EXISTS t_a(id INTEGER PRIMARY KEY, v TEXT);")
    D.register_migration("t_mod_a", 1, "CREATE TABLE IF NOT EXISTS t_a(id INTEGER PRIMARY KEY, v TEXT);")  # same: ok
    db = D.get_db()  # picks up the new migration
    assert "t_a" in db.tables()
    D.register_migration(
        "t_mod_a", 2,
        "ALTER TABLE t_a ADD COLUMN w INTEGER DEFAULT 5;\nCREATE INDEX IF NOT EXISTS ix_t_a ON t_a(v);",
    )
    db = D.get_db()
    db.execute("INSERT INTO t_a(v) VALUES('x')")
    assert db.query_one("SELECT w FROM t_a")["w"] == 5
    with pytest.raises(ConfigError):
        D.register_migration("t_mod_a", 1, "CREATE TABLE other(x);")
    with pytest.raises(ConfigError):
        D.register_migration("t_mod_a", 0, "SELECT 1;")


def test_migrations_apply_to_each_new_database(tmp_path):
    D.register_migration("t_mod_b", 1, "CREATE TABLE IF NOT EXISTS t_b(id INTEGER);")
    d1 = D.get_db(tmp_path / "one.db")
    d2 = D.get_db(tmp_path / "two.db")
    assert "t_b" in d1.tables() and "t_b" in d2.tables() and d1 is not d2


def test_failed_migration_rolls_back(tmp_path):
    db = D.Database(tmp_path / "x.db")
    D.register_migration("t_mod_bad", 1, "CREATE TABLE t_bad_ok(id INTEGER);\nTHIS IS NOT SQL;")
    try:
        with pytest.raises(Exception):
            db.migrate()
        assert "t_bad_ok" not in db.tables()
        assert db.query("SELECT * FROM _migrations WHERE module='t_mod_bad'") == []
    finally:
        D._MIGRATIONS.pop(("t_mod_bad", 1), None)
    db.close()


def test_query_helpers(forge_home):
    db = D.get_db()
    db.execute("CREATE TABLE IF NOT EXISTS q(id INTEGER PRIMARY KEY, name TEXT)")
    db.executemany("INSERT INTO q(name) VALUES(?)", [("a",), ("b",), ("c",)])
    assert [r["name"] for r in db.query("SELECT * FROM q ORDER BY id")] == ["a", "b", "c"]
    assert db.query_one("SELECT name FROM q WHERE name=?", ("b",)) == {"name": "b"}
    assert db.query_one("SELECT name FROM q WHERE name='zz'") is None
    assert db.scalar("SELECT COUNT(*) FROM q") == 3
    assert db.scalar("SELECT 1 FROM q WHERE 0", default=-1) == -1
    cur = db.execute("INSERT INTO q(name) VALUES('d')")
    assert cur.lastrowid == 4
    assert db.query("SELECT * FROM q WHERE name=:n", {"n": "d"})[0]["id"] == 4


def test_transactions_commit_rollback_nested(forge_home):
    db = D.get_db()
    db.execute("CREATE TABLE IF NOT EXISTS tx(v INTEGER)")
    with db.transaction():
        db.execute("INSERT INTO tx VALUES(1)")
    assert db.scalar("SELECT COUNT(*) FROM tx") == 1
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.execute("INSERT INTO tx VALUES(2)")
            raise RuntimeError("boom")
    assert db.scalar("SELECT COUNT(*) FROM tx") == 1
    with db.transaction():
        db.execute("INSERT INTO tx VALUES(3)")
        with pytest.raises(ValueError):
            with db.transaction():  # savepoint
                db.execute("INSERT INTO tx VALUES(4)")
                raise ValueError
        db.execute("INSERT INTO tx VALUES(5)")
    assert sorted(r["v"] for r in db.query("SELECT v FROM tx")) == [1, 3, 5]


def test_threads_use_separate_connections(forge_home):
    db = D.get_db()
    db.execute("CREATE TABLE IF NOT EXISTS th(v INTEGER)")
    errs: list[BaseException] = []

    def work(i):
        try:
            for j in range(25):
                db.execute("INSERT INTO th VALUES(?)", (i * 100 + j,))
            with db.transaction():
                db.execute("INSERT INTO th VALUES(-1)")
        except BaseException as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    assert db.scalar("SELECT COUNT(*) FROM th") == 6 * 26


def test_memory_database():
    db = D.Database(":memory:")
    db.migrate()
    assert "kv" in db.tables()
    results = []

    def work():
        db.execute("INSERT INTO kv VALUES('a','1',0) ON CONFLICT(key) DO NOTHING")
        results.append(db.scalar("SELECT COUNT(*) FROM kv"))

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert results == [1] and db.scalar("SELECT COUNT(*) FROM kv") == 1
    with db.transaction():
        db.kv_set("x", 1)
    db.close()


def test_kv(forge_home):
    db = D.get_db()
    assert db.kv_get("nope", "dflt") == "dflt"
    db.kv_set("cfg", {"a": [1, 2]})
    db.kv_set("cfg", {"a": [3]})
    assert db.kv_get("cfg") == {"a": [3]}
    db.kv_delete("cfg")
    assert db.kv_get("cfg") is None


def test_wal_enabled(forge_home):
    db = D.get_db()
    assert db.scalar("PRAGMA journal_mode") == "wal"
    assert db.scalar("PRAGMA foreign_keys") == 1


def test_split_statements_handles_triggers_and_strings():
    sql = "CREATE TABLE a(x TEXT DEFAULT ';');\nCREATE TRIGGER t AFTER INSERT ON a BEGIN\n UPDATE a SET x='y'; END;\n"
    parts = D._split_statements(sql)
    assert len(parts) == 2 and parts[1].startswith("CREATE TRIGGER")
