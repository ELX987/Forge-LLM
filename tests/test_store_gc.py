"""GC correctness + chaos tests: pinned / referenced / delta-chain objects must never be collected."""
from __future__ import annotations

import random
import threading
import time

import numpy as np
import pytest
import torch

from forge.core.errors import NotFound
from forge.store import ArtifactStore
from forge.store.checkpoints import CheckpointReader, CheckpointWriter, RetentionPolicy
from forge.store.chunking import ChunkerParams
from forge.store.lossless import tensor_to_array

SMALL = ChunkerParams(min_size=2048, bits=12, max_size=16384)


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "store", chunk_params=SMALL, inline_threshold=8192)


def rnd(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def test_released_artifact_is_collected_pinned_is_not(store):
    keep, drop, pinned = rnd(80_000, 1), rnd(80_000, 2), rnd(80_000, 3)
    hk, hd, hp = store.put_bytes(keep), store.put_bytes(drop), store.put_bytes(pinned)
    store.release(hp)
    store.pin(hp, "important")
    store.release(hd)
    rep = store.gc(dry_run=True, grace_s=0)
    assert rep.dry_run and rep.deleted_objects == 0 and rep.objects_garbage > 0 and rep.bytes_garbage > 0
    assert store.has(hd) and store.get_bytes(hd) == drop  # dry run changes nothing
    before = store.stats()["unique_bytes"]
    rep2 = store.gc(dry_run=False, grace_s=0)
    assert rep2.deleted_objects == rep.objects_garbage and rep2.bytes_freed == rep.bytes_garbage
    assert store.stats()["unique_bytes"] == before - rep2.bytes_freed
    assert not store.has(hd)
    assert store.get_bytes(hk) == keep and store.get_bytes(hp) == pinned
    assert store.verify(hk, deep=True) and store.verify(hp, deep=True)
    # nothing left to collect
    assert store.gc(dry_run=False, grace_s=0).deleted_objects == 0


def test_shared_chunks_survive_when_one_owner_is_released(store):
    a = rnd(200_000, 4)
    b = a[:100_000] + b"X" * 50 + a[100_050:]
    ha, hb = store.put_bytes(a), store.put_bytes(b)
    store.release(ha)
    store.gc(dry_run=False, grace_s=0)
    assert store.get_bytes(hb) == b and store.verify(hb, deep=True)
    assert not store.has(ha)
    store.release(hb)
    store.gc(dry_run=False, grace_s=0)
    assert store.stats()["unique_bytes"] == 0


def test_strict_pins_mode(store):
    h1, h2 = store.put_bytes(rnd(1000, 1)), store.put_bytes(rnd(1000, 2))
    store.pin(h1, "x")
    rep = store.gc(dry_run=False, grace_s=0, strict_pins=True)
    assert store.has(h1) and not store.has(h2) and rep.deleted_objects == 1


def test_grace_period_protects_young_objects(store):
    h = store.put_bytes(rnd(50_000, 5))
    store.release(h)
    rep = store.gc(dry_run=False, grace_s=3600)
    assert rep.deleted_objects == 0 and rep.skipped_young > 0 and store.has(h)
    time.sleep(0.01)
    assert store.gc(dry_run=False, grace_s=0).deleted_objects > 0


def test_reput_of_garbage_object_keeps_it_alive(store):
    d = rnd(60_000, 6)
    h = store.put_bytes(d)
    store.release(h)
    assert store.gc(dry_run=True, grace_s=0).objects_garbage > 0
    assert store.put_bytes(d) == h  # re-use of an unreferenced object makes it rooted again
    assert store.gc(dry_run=False, grace_s=0).deleted_objects == 0
    assert store.get_bytes(h) == d


def test_manifest_members_are_live_while_manifest_is_live(store):
    t1 = store.put_bytes(rnd(30_000, 7), kind="tensor")
    t2 = store.put_bytes(rnd(30_000, 8), kind="tensor")
    from forge.core.schemas import ArtifactManifest, TensorEntry

    m = ArtifactManifest(kind="model", tensors=[TensorEntry(name="a", dtype="uint8", shape=[30000], hash=t1)])
    mh = store.put_manifest(m)
    store.gc(dry_run=False, grace_s=0)
    assert store.has(t1) and not store.has(t2)
    assert store.verify(mh, deep=True)
    store.release(mh)
    store.gc(dry_run=False, grace_s=0)
    assert not store.has(t1) and not store.has(mh)
    # the signature ledger is append-only: collected manifests remain in the hash chain
    assert store.verify_chain()["ok"] and store.verify_chain()["entries"] == 1


def test_orphan_files_are_reported_and_removed(store):
    orphan = store._tier_dir("hot") / "objects" / "ab" / ("cd" * 31 + "00")
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"leftover from a crashed writer")
    stale = store._tier_dir("hot") / "tmp" / "dead.tmp"
    stale.write_bytes(b"partial")
    import os

    old = time.time() - 100
    os.utime(orphan, (old, old))
    os.utime(stale, (old, old))
    rep = store.gc(dry_run=True, grace_s=0)
    assert rep.orphan_files == 2 and orphan.exists()
    store.gc(dry_run=False, grace_s=0)
    assert not orphan.exists() and not stale.exists()


def test_concurrent_writers_with_grace_never_lose_data(store):
    """Writers race with GC sweeps; the grace period must keep half-attached chunks alive."""
    written: dict[str, bytes] = {}
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer(seed: int) -> None:
        try:
            i = 0
            while not stop.is_set() and i < 25:
                d = rnd(40_000 + 1000 * i, seed * 1000 + i)
                h = store.put_bytes(d)
                written[h] = d
                i += 1
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=writer, args=(s,)) for s in (1, 2, 3)]
    for t in ts:
        t.start()
    for _ in range(15):
        store.gc(dry_run=False, grace_s=30)
        time.sleep(0.005)
    for t in ts:
        t.join()
    stop.set()
    assert not errors
    for h, d in list(written.items()):
        assert store.get_bytes(h) == d


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tensor_to_array(a)[0].tobytes() == tensor_to_array(b)[0].tobytes()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_chaos_random_operations_preserve_invariants(tmp_path, seed):
    """Random saves / retention / pins / releases / tier moves / GC. After every GC:
    * every non-dropped checkpoint still restores bit-exactly,
    * every pinned object still verifies,
    * the delta-chain closure of every live checkpoint is intact."""
    rng = random.Random(seed)
    st = ArtifactStore(tmp_path / f"chaos{seed}", chunk_params=SMALL, inline_threshold=8192)
    pol = RetentionPolicy(keep_last=rng.choice([1, 2, 3]), keep_best=rng.choice([0, 1]), keep_keyframes=rng.random() < 0.5,
                          keep_tags=["pre_anneal"])
    w = CheckpointWriter(st, "chaos", keyframe_every=rng.choice([2, 3, 5]), retention=pol)
    g = torch.Generator().manual_seed(seed)
    state = {"w": torch.randn(3000, generator=g).to(torch.bfloat16), "b": torch.randn(500, generator=g),
             "frozen": torch.randn(1000, generator=g)}
    snaps: dict[int, dict] = {}
    pinned_blobs: dict[str, bytes] = {}
    step = 0
    for op_i in range(40):
        op = rng.choices(["save", "save_tag", "blob", "pin", "release_blob", "gc", "tier", "gc_dry"], [10, 2, 3, 2, 2, 5, 2, 1])[0]
        if op in ("save", "save_tag"):
            step += rng.choice([1, 1, 1, 5])
            state["w"] = (state["w"].float() + 0.01 * torch.randn(3000, generator=g)).to(torch.bfloat16)
            state["b"] = state["b"] + 0.01 * torch.randn(500, generator=g)
            tags = ["pre_anneal"] if op == "save_tag" else []
            w.save(step, state, metric=rng.random(), tags=tags)
            snaps[step] = {k: v.clone() for k, v in state.items()}
        elif op == "blob":
            d = rnd(rng.randint(100, 50_000), rng.randint(0, 10**6))
            pinned_blobs[st.put_bytes(d)] = d
        elif op == "pin" and pinned_blobs:
            h = rng.choice(list(pinned_blobs))
            st.pin(h, "chaos")
        elif op == "release_blob" and pinned_blobs:
            h = rng.choice(list(pinned_blobs))
            st.release(h)  # also unpins
            pinned_blobs.pop(h)
        elif op == "tier":
            rows = CheckpointReader(st, "chaos").rows()
            if rows:
                st.demote(rng.choice(rows)["manifest_hash"], rng.choice(["warm", "cold", "hot"]))
        elif op in ("gc", "gc_dry"):
            rep = st.gc(dry_run=(op == "gc_dry"), grace_s=0)
            rd = CheckpointReader(st, "chaos")
            for r in rd.rows():
                got = rd.load(r["step"])
                for k, v in snaps[r["step"]].items():
                    assert _same(got[k], v), (seed, op_i, r["step"], k)
                assert st.verify(r["manifest_hash"], deep=True), (seed, op_i, r["step"])
            for h, d in pinned_blobs.items():
                if h in {p["hash"] for p in st.pins()}:
                    assert st.get_bytes(h) == d
            assert rep.objects_live <= rep.objects_total
    # final sweep, then the whole store must verify
    st.gc(dry_run=False, grace_s=0)
    assert st.verify_all(deep=True)["ok"]
    rd = CheckpointReader(st, "chaos")
    for r in rd.rows():
        got = rd.load(r["step"])
        for k, v in snaps[r["step"]].items():
            assert _same(got[k], v)
    # released/dropped history really is reclaimable: no live data is the same as nothing referenced
    for r in rd.rows():
        st.release(r["manifest_hash"])
    for h in list(pinned_blobs):
        st.release(h)
    st.gc(dry_run=False, grace_s=0)
    assert st.stats()["unique_bytes"] == 0, st.stats()


def test_chain_integrity_guard_refuses_unsafe_sweep(store, monkeypatch):
    """The defensive invariant check must abort the sweep (not delete) if marking were ever wrong."""
    from forge.core.errors import StoreError
    from forge.store import collect

    h = store.put_bytes(rnd(50_000, 12))
    store.pin(h, "x")
    monkeypatch.setattr(collect, "_mark", lambda adj, roots, follow_chain: set())  # sabotage: nothing is live
    with pytest.raises(StoreError):
        store.gc(dry_run=False, grace_s=0)
    monkeypatch.undo()
    assert store.get_bytes(h) == rnd(50_000, 12)


def test_dropped_checkpoint_rows_disappear_with_their_manifests(store):
    pol = RetentionPolicy(keep_last=1, keep_best=0, keep_keyframes=False, keep_tags=[])
    w = CheckpointWriter(store, "rows", keyframe_every=1, retention=pol)  # every save is a keyframe: no chains
    for s in range(1, 5):
        w.save(s, {"x": torch.randn(2000)})
    assert len(CheckpointReader(store, "rows").rows(include_dropped=True)) == 4
    store.gc(dry_run=False, grace_s=0)
    rd = CheckpointReader(store, "rows")
    assert rd.steps(include_dropped=True) == [4]
    with pytest.raises(NotFound):
        rd.load(1)
