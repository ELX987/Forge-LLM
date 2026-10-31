from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from forge.core.errors import ConfigError, IntegrityError, NotFound
from forge.core.schemas import ArtifactManifest, TensorEntry
from forge.store import ArtifactStore, get_store
from forge.store.chunking import ChunkerParams

SMALL = ChunkerParams(min_size=2048, bits=12, max_size=16384)


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "store", chunk_params=SMALL, inline_threshold=8192)


def rnd(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


# ---------------------------------------------------------------- blobs
def test_small_and_chunked_roundtrip(store):
    small, big = rnd(1000), rnd(300_000, 1)
    hs, hb = store.put_bytes(small), store.put_bytes(big)
    assert store.has(hs) and store.has(hb)
    assert store.get_bytes(hs) == small and store.get_bytes(hb) == big
    assert store.size(hb) == len(big)
    assert store.info(hb)["otype"] == "chunked" and store.info(hs)["otype"] == "obj"
    assert store.put_bytes(big) == hb  # idempotent
    assert store.verify(hb, deep=True) and store.verify(hs)
    with store.open(hb) as f:
        f.seek(123_456)
        assert f.read(1000) == big[123_456:124_456]
        assert f.read(0) == b""
        f.seek(-10, 2)
        assert f.read() == big[-10:]
    assert store.has("blake3:" + "0" * 64) is False


def test_put_file_and_stream_equal_put_bytes(store, tmp_path):
    d = rnd(200_000, 2)
    p = tmp_path / "x.bin"
    p.write_bytes(d)
    assert store.put_file(p) == store.put_bytes(d)
    import io

    assert store.put_stream(io.BytesIO(d)) == store.put_bytes(d)
    out = store.materialize(store.put_bytes(d), tmp_path / "o" / "y.bin")
    assert out.read_bytes() == d
    assert store.put_bytes(b"") == store.put_bytes(b"")
    assert store.get_bytes(store.put_bytes(b"")) == b""


def test_dedup_of_near_identical_blobs(store):
    a = rnd(400_000, 3)
    b = a[:150_000] + b"EDIT" + a[150_004:]
    ha, hb = store.put_bytes(a), store.put_bytes(b)
    assert ha != hb
    s = store.stats()
    assert s["logical_bytes"] == 800_000
    assert s["unique_bytes"] < 0.6 * s["logical_bytes"]  # most chunks are shared
    assert s["dedup_ratio"] > 1.7
    assert store.stored_size(ha) < len(a) * 1.05
    assert store.get_bytes(hb) == b


def test_compression_of_compressible_data(store):
    h = store.put_bytes(b"abc" * 100_000)
    assert store.stored_size(h) < 20_000
    assert store.get_bytes(h) == b"abc" * 100_000
    assert store.stats()["compaction_ratio"] < 0.2


def test_corruption_detected(store):
    d = rnd(100_000, 4)
    h = store.put_bytes(d)
    chunk = store._chunk_list(h)[1][0]
    p = store._path(chunk, "hot", store._row(chunk)["codec"])
    raw = bytearray(p.read_bytes())
    raw[10] ^= 0xFF
    p.write_bytes(bytes(raw))
    assert not store.verify(h)
    with pytest.raises(IntegrityError):
        store.get_bytes(h)
    p.unlink()
    assert not store.verify(h)
    with pytest.raises(IntegrityError):
        store.get_bytes(h)


def test_not_found(store):
    with pytest.raises(NotFound):
        store.get_bytes("blake3:" + "1" * 64)
    with pytest.raises(NotFound):
        store.pin("blake3:" + "1" * 64, "x")


# ---------------------------------------------------------------- manifests
def _manifest(store, payload: bytes, **kw) -> tuple[str, ArtifactManifest]:
    th = store.put_bytes(payload, kind="tensor")
    m = ArtifactManifest(kind=kw.pop("kind", "model"), tensors=[TensorEntry(name="w", dtype="uint8", shape=[len(payload)], hash=th)], **kw)
    return store.put_manifest(m), m


def test_manifest_roundtrip_signature_and_chain(store):
    h1, m1 = _manifest(store, b"one", extra={"a": 1})
    h2, m2 = _manifest(store, b"two", parents=[h1])
    assert h1 == m1.hash
    got = store.get_manifest(h2, verify_signature=True)
    assert got.parents == [h1] and got.tensors[0].name == "w"
    assert store.signature_status(h1) == "valid"
    assert store.verify(h1) and store.verify(h2, deep=True)
    assert store.verify_chain()["ok"] and store.verify_chain()["entries"] == 2
    assert os.path.exists(store.root / "store.key")
    # content-addressed: same content -> same hash, no new signature
    assert store.put_manifest(m1) == h1
    assert store.verify_chain()["entries"] == 2


def test_manifest_tamper_detection(store):
    h1, _ = _manifest(store, b"alpha")
    h2, _ = _manifest(store, b"beta")
    # 1. edit the manifest file
    p = store._path(h1, "hot", "raw", "manifest")
    doc = json.loads(p.read_text())
    doc["extra"] = {"evil": True}
    p.write_text(json.dumps(doc))
    with pytest.raises(IntegrityError):
        store.get_manifest(h1)
    assert not store.verify(h1)
    # 2. forge the signature
    store.db.execute("UPDATE store_manifests SET sig=? WHERE hash=?", ("0" * 64, h2))
    assert store.signature_status(h2) == "invalid"
    assert not store.verify(h2)
    # 3. break the hash chain
    store.db.execute("UPDATE store_manifests SET chain=? WHERE hash=?", ("f" * 64, h1))
    assert not store.verify_chain()["ok"]


def test_chain_only_mode_when_key_unavailable(tmp_path, monkeypatch):
    s = ArtifactStore(tmp_path / "s2")
    monkeypatch.setattr(s, "_signing_key", lambda: None)
    h, _ = _manifest(s, b"x")
    assert s.signature_status(h) == "chain_only"
    assert s.verify(h)


def test_env_hmac_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_STORE_HMAC_KEY", "secret-from-settings")
    s = ArtifactStore(tmp_path / "s3")
    h, _ = _manifest(s, b"x")
    assert s.signature_status(h) == "valid"
    assert not (tmp_path / "s3" / "store.key").exists()
    monkeypatch.setenv("FORGE_STORE_HMAC_KEY", "another-key")
    s2 = ArtifactStore(tmp_path / "s3")
    assert s2.signature_status(h) in ("invalid", "key_mismatch")


# ---------------------------------------------------------------- tensor dirs
def _make_model_dir(path: Path, seed: int = 0, change: dict[str, torch.Tensor] | None = None) -> Path:
    g = torch.Generator().manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    t1 = {
        "embed.weight": (torch.randn(512, 64, generator=g) * 0.02).to(torch.bfloat16),
        "layers.0.q.weight": (torch.randn(128, 128, generator=g) * 0.02).to(torch.bfloat16),
        "layers.0.norm.weight": torch.ones(32, dtype=torch.float32),
    }
    t2 = {
        "layers.1.q.weight": (torch.randn(128, 128, generator=g) * 0.02).to(torch.bfloat16),
        "layers.1.q.scales": torch.rand(4, 32, generator=g).to(torch.float16),
        "head.weight": (torch.randn(512, 64, generator=g) * 0.02).to(torch.bfloat16),
        "empty": torch.zeros(0, 3, dtype=torch.float32),
    }
    for k, v in (change or {}).items():
        (t1 if k in t1 else t2)[k] = v
    save_file(t1, str(path / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(t2, str(path / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps({"hidden_size": 32, "model_type": "llama"}))
    (path / "tokenizer.json").write_text('{"tok": 1}')
    (path / "sub").mkdir(exist_ok=True)
    (path / "sub" / "notes.txt").write_text("hello")
    return path


def _same_tree(a: Path, b: Path) -> bool:
    fa = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    return fa == fb and all((a / f).read_bytes() == (b / f).read_bytes() for f in fa)


def test_tensor_dir_roundtrip_is_byte_exact(store, tmp_path):
    src = _make_model_dir(tmp_path / "m1")
    h = store.put_tensor_dir(src, kind="model", licence="apache-2.0")
    m = store.get_manifest(h)
    assert m.kind == "model" and m.licence == "apache-2.0" and m.model_cfg["hidden_size"] == 32
    names = {t.name: t for t in m.tensors}
    assert set(names) == {"embed.weight", "layers.0.q.weight", "layers.0.norm.weight", "layers.1.q.weight",
                          "layers.1.q.scales", "head.weight", "empty"}
    assert names["embed.weight"].dtype == "bf16" and names["embed.weight"].shape == [512, 64]
    assert names["layers.1.q.scales"].role == "scale" and names["layers.0.norm.weight"].dtype == "fp32"
    out = store.materialize_artifact(h, tmp_path / "restored")
    assert _same_tree(src, out)
    assert store.materialize(h, tmp_path / "restored2") == tmp_path / "restored2"
    data, dt, shape = store.get_tensor(h, "embed.weight")
    assert dt == "bf16" and shape == [512, 64] and len(data) == 512 * 64 * 2
    assert store.verify(h, deep=True)
    # same dir again -> identical manifest hash, nothing new stored
    before = store.stats()["unique_bytes"]
    assert store.put_tensor_dir(src, kind="model", licence="apache-2.0") == h
    assert store.stats()["unique_bytes"] == before


def test_dedup_demo_two_near_identical_checkpoints(store, tmp_path):
    """Two checkpoints that differ in a single tensor share everything else (per-tensor addressing)."""
    a = _make_model_dir(tmp_path / "ckpt_a")
    changed = (torch.randn(128, 128, generator=torch.Generator().manual_seed(99)) * 0.02).to(torch.bfloat16)
    b = _make_model_dir(tmp_path / "ckpt_b", change={"layers.1.q.weight": changed})
    ha = store.put_tensor_dir(a, kind="model")
    after_a = store.stats()["unique_bytes"]
    hb = store.put_tensor_dir(b, kind="model", parents=[ha], recipe_hash="blake3:" + "a" * 64)
    s = store.stats()
    added = s["unique_bytes"] - after_a
    assert ha != hb
    assert added < 0.25 * after_a, (added, after_a)  # only the changed tensor (+ manifest/headers) is new
    assert s["dedup_ratio"] > 1.6
    assert _same_tree(b, store.materialize_artifact(hb, tmp_path / "rb"))
    assert _same_tree(a, store.materialize_artifact(ha, tmp_path / "ra"))
    lin = store.lineage(hb)
    assert {n["hash"] for n in lin["nodes"]} == {ha, hb}
    assert lin["edges"][0]["recipe_hash"] == "blake3:" + "a" * 64


def test_tensor_dir_partial_tensor_change_dedups_inside_big_tensor(store, tmp_path):
    big = torch.randn(256, 512, generator=torch.Generator().manual_seed(1)).to(torch.float32)  # 512 KiB
    d1, d2 = tmp_path / "p1", tmp_path / "p2"
    for d in (d1, d2):
        d.mkdir()
    save_file({"big": big}, str(d1 / "model.safetensors"))
    big2 = big.clone()
    big2[100:104] += 1.0  # touch a few rows
    save_file({"big": big2}, str(d2 / "model.safetensors"))
    h1 = store.put_tensor_dir(d1, kind="model")
    u1 = store.stats()["unique_bytes"]
    store.put_tensor_dir(d2, kind="model")
    assert store.stats()["unique_bytes"] - u1 < 0.4 * u1  # chunk-level dedup inside one tensor
    assert _same_tree(d1, store.materialize_artifact(h1, tmp_path / "r1"))


def test_tensor_dir_lossless_option(store, tmp_path):
    g = torch.Generator().manual_seed(5)
    d = tmp_path / "ll"
    d.mkdir()
    save_file({"w": (torch.randn(200_000, generator=g) * 0.02).to(torch.bfloat16), "b": torch.arange(10, dtype=torch.int32)}, str(d / "model.safetensors"))
    (d / "config.json").write_text("{}")
    raw_h = store.put_tensor_dir(d, kind="model")
    ll_store = ArtifactStore(tmp_path / "store_ll", chunk_params=SMALL, inline_threshold=8192, compress="none")
    ll_h = ll_store.put_tensor_dir(d, kind="model", lossless=True)
    m = ll_store.get_manifest(ll_h)
    assert "w" in m.extra["enc"] and "b" not in m.extra["enc"]
    assert m.extra["enc"]["w"]["ratio"] < 0.7
    assert ll_store.stored_size(ll_h) < 0.8 * 400_000
    assert _same_tree(d, ll_store.materialize_artifact(ll_h, tmp_path / "rl"))
    assert ll_store.get_tensor(ll_h, "w")[0] == store.get_tensor(raw_h, "w")[0]
    assert ll_store.verify(ll_h, deep=True)


def test_non_contiguous_or_foreign_files_stay_opaque(store, tmp_path):
    d = tmp_path / "odd"
    d.mkdir()
    (d / "weird.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}garbage")  # not a valid layout
    (d / "a.bin").write_bytes(rnd(50_000))
    h = store.put_tensor_dir(d, kind="model")
    assert _same_tree(d, store.materialize_artifact(h, tmp_path / "o"))


# ---------------------------------------------------------------- tiers / pins / lineage / listing
def test_tiering_moves_files_but_not_addresses(store):
    big = rnd(100_000, 6)
    h = store.put_bytes(big)
    assert store.tier_of(h) == "hot"
    chunk = store._chunk_list(h)[0][0]
    hot_path = store._path(chunk, "hot", store._row(chunk)["codec"])
    assert hot_path.exists()
    moved = store.demote(h, "cold")
    assert moved >= 2 and store.tier_of(h) == "cold"
    assert not hot_path.exists()
    assert store._path(chunk, "cold", store._row(chunk)["codec"]).exists()
    assert store.get_bytes(h) == big  # same address still resolves
    assert store.stats()["per_tier"]["cold"]["objects"] > 0 and store.stats()["per_tier"]["hot"]["objects"] == 0
    assert set(store.tier_summary(h)) == {"cold"}
    store.promote(h)
    assert store.tier_of(h) == "hot" and store.get_bytes(h) == big
    with pytest.raises(ConfigError):
        store.demote(h, "frozen")


def test_tier_demote_manifest_and_stale_row_self_heal(store, tmp_path):
    src = _make_model_dir(tmp_path / "tm")
    h = store.put_tensor_dir(src, kind="model")
    store.demote(h, "warm")
    assert _same_tree(src, store.materialize_artifact(h, tmp_path / "w"))
    # corrupt the tier column: lookup must still find the file
    store.db.execute("UPDATE store_objects SET tier='hot'")
    assert _same_tree(src, store.materialize_artifact(h, tmp_path / "w2"))


def test_auto_demote(store):
    h = store.put_bytes(rnd(500))
    store.db.execute("UPDATE store_objects SET accessed_at=accessed_at-1000 WHERE hash=?", (h,))
    assert store.auto_demote(100, "warm", dry_run=True) == [h] and store.tier_of(h) == "hot"
    store.auto_demote(100, "warm", dry_run=False)
    assert store.tier_of(h) == "warm"


def test_separate_tier_directories(tmp_path):
    s = ArtifactStore(tmp_path / "s", tier_dirs={"cold": tmp_path / "bigdisk"}, chunk_params=SMALL, inline_threshold=8192)
    h = s.put_bytes(rnd(60_000, 8))
    s.demote(h, "cold")
    assert any((tmp_path / "bigdisk" / "objects").rglob("*"))
    assert s.get_bytes(h) == rnd(60_000, 8)
    assert s.tiers()["cold"]["path"] == str(tmp_path / "bigdisk")


def test_pins_listing_and_kinds(store):
    h1 = store.put_bytes(b"a" * 10, kind="dataset")
    h2 = store.put_bytes(b"b" * 10, kind="blob")
    store.put_bytes(b"c" * 10, kind="tensor")  # internal kinds are not listed
    assert {r["hash"] for r in store.list_artifacts()} == {h1, h2}
    assert [r["hash"] for r in store.list_artifacts(kind="dataset")] == [h1]
    store.pin(h1, "keep me")
    assert store.is_pinned(h1) and store.list_artifacts(kind="dataset")[0]["pinned"]
    assert store.pins()[0]["reason"] == "keep me"
    store.unpin(h1)
    assert not store.is_pinned(h1)
    assert store.count_artifacts() == 2 and store.count_artifacts("dataset") == 1
    store.release(h2)
    assert {r["hash"] for r in store.list_artifacts()} == {h1}
    # re-putting a released/internal object makes it a visible artifact again
    assert store.put_bytes(b"c" * 10, kind="blob") and store.count_artifacts() == 2


def test_lineage_dag_and_cycle_rejection(store):
    a, b, c, d = (store.put_bytes(x) for x in (b"a", b"b", b"c", b"d"))
    store.add_edge(b, a, relation="derived_from")
    store.add_edge(c, b, recipe_hash="blake3:" + "1" * 64)
    store.add_edge(d, b, relation="evaluated_by")
    lin = store.lineage(c)
    assert {n["hash"] for n in lin["nodes"]} == {a, b, c, d}
    assert {n["hash"] for n in store.lineage(c, direction="up")["nodes"]} == {a, b, c}
    assert {n["hash"] for n in store.lineage(a, direction="down", depth=1)["nodes"]} == {a, b}
    assert len(lin["edges"]) == 3
    with pytest.raises(ConfigError):
        store.add_edge(a, c)
    with pytest.raises(ConfigError):
        store.add_edge(a, a)
    unknown = "blake3:" + "9" * 64
    store.add_edge(a, unknown)
    node = next(n for n in store.lineage(a)["nodes"] if n["hash"] == unknown)
    assert node["present"] is False


def test_stats_structure(store):
    store.put_bytes(rnd(50_000), kind="dataset")
    s = store.stats()
    for k in ("logical_bytes", "unique_bytes", "dedup_ratio", "compaction_ratio", "per_kind", "per_tier", "saved_bytes", "objects", "pins", "edges"):
        assert k in s
    assert s["per_kind"]["dataset"]["logical_bytes"] == 50_000
    assert set(s["per_tier"]) == {"hot", "warm", "cold"}


def test_verify_all_and_cli(store, tmp_path, capsys):
    store.put_tensor_dir(_make_model_dir(tmp_path / "cli"), kind="model")
    assert store.verify_all()["ok"]
    from forge.store.cli import main

    assert main(["--root", str(store.root), "--json", "stats"]) == 0
    assert main(["--root", str(store.root), "verify"]) == 0
    assert main(["--root", str(store.root), "gc"]) == 0
    out = capsys.readouterr().out
    assert "dedup_ratio" in out and "objects_total" in out
    # break something -> non-zero exit
    victim = store.put_bytes(rnd(30_000, 11))
    ch = store._chunk_list(victim)[0][0] if store.info(victim)["otype"] == "chunked" else victim
    store._path(ch, "hot", store._row(ch)["codec"]).write_bytes(b"corrupt")
    assert main(["--root", str(store.root), "verify"]) == 1


def test_get_store_singleton_follows_forge_home(forge_home):
    from forge.store import reset_stores

    reset_stores()
    s1, s2 = get_store(), get_store()
    assert s1 is s2 and str(s1.root).startswith(str(forge_home))
    h = s1.put_bytes(b"hello")
    assert get_store().get_bytes(h) == b"hello"
    reset_stores()


def test_algo_mismatch_is_refused(tmp_path, monkeypatch):
    ArtifactStore(tmp_path / "alg")
    cfgp = tmp_path / "alg" / "store.json"
    cfg = json.loads(cfgp.read_text())
    cfg["algo"] = "sha256" if cfg["algo"] != "sha256" else "b2"
    cfgp.write_text(json.dumps(cfg))
    from forge.core.errors import StoreError

    with pytest.raises(StoreError):
        ArtifactStore(tmp_path / "alg")
