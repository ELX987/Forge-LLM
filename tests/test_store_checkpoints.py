from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from forge.core.errors import IntegrityError, NotFound
from forge.store import ArtifactStore
from forge.store.checkpoints import (
    CheckpointConfig,
    CheckpointReader,
    CheckpointWriter,
    RetentionPolicy,
    list_checkpoints,
    list_runs,
)
from forge.store.chunking import ChunkerParams
from forge.store.lossless import tensor_to_array

SMALL = ChunkerParams(min_size=2048, bits=12, max_size=16384)


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "store", chunk_params=SMALL, inline_threshold=8192)


def make_state(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "model": {
            "embed.weight": (torch.randn(128, 64, generator=g) * 0.02).to(torch.bfloat16),
            "layer.q.weight": (torch.randn(64, 64, generator=g) * 0.02).to(torch.bfloat16),
            "layer.norm.weight": torch.ones(64),
            "frozen.weight": (torch.randn(64, 64, generator=g) * 0.02).to(torch.bfloat16),
            "scalar": torch.tensor(3.5),
            "empty": torch.zeros(0, 4),
        },
        "optim": {
            "state": {0: {"exp_avg": torch.randn(128, 64, generator=g), "step": torch.tensor(0.0)},
                      1: {"exp_avg": torch.randn(64, 64, generator=g), "step": torch.tensor(0.0)}},
            "param_groups": [{"lr": 1e-3, "betas": (0.9, 0.95), "params": [0, 1], "name": None, "flag": True}],
        },
        "rng": {"py": (3, 1, 4), "np": np.arange(5, dtype=np.int64)},
        "step": 0,
        "mask": torch.tensor([True, False, True]),
        "ids": torch.arange(10, dtype=torch.int64),
    }


def advance(state: dict, step: int, lr: float = 1e-3, seed: int = 0) -> dict:
    """Simulate an optimiser step: small perturbation of trainable tensors, frozen ones untouched."""
    g = torch.Generator().manual_seed(1000 + step + seed)
    for k in ("embed.weight", "layer.q.weight"):
        w = state["model"][k]
        state["model"][k] = (w.float() + lr * torch.randn(w.shape, generator=g)).to(w.dtype)
    for i in (0, 1):
        st = state["optim"]["state"][i]
        st["exp_avg"] = st["exp_avg"] * 0.9 + 0.1 * torch.randn(st["exp_avg"].shape, generator=g)
        st["step"] = st["step"] + 1
    state["step"] = step
    return state


def clone(o):
    if isinstance(o, dict):
        return {k: clone(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clone(v) for v in o]
    if isinstance(o, tuple):
        return tuple(clone(v) for v in o)
    if isinstance(o, torch.Tensor):
        return o.clone()
    if isinstance(o, np.ndarray):
        return o.copy()
    return o


def assert_same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert list(a) == list(b), path
        for k in a:
            assert_same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}/{i}")
    elif isinstance(a, torch.Tensor):
        ua, da, sa = tensor_to_array(a)
        ub, db, sb = tensor_to_array(b)
        assert (da, sa) == (db, sb), path
        assert ua.tobytes() == ub.tobytes(), path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), path
    else:
        assert a == b, path


def test_series_restores_every_step_bit_exact(store):
    w = CheckpointWriter(store, "r1", keyframe_every=3)
    state = make_state()
    snaps = {}
    for step in range(1, 9):
        advance(state, step)
        w.save(step, state, meta={"loss": 1.0 / step}, metric=1.0 / step)
        snaps[step] = clone(state)
    rd = CheckpointReader(store, "r1")
    assert rd.steps(include_dropped=True) == list(range(1, 9))
    kfs = [r["step"] for r in rd.rows(include_dropped=True) if r["is_keyframe"]]
    assert kfs == [1, 4, 7]
    for step, snap in snaps.items():
        assert_same(rd.load(step), snap)
    assert_same(rd.load("latest"), snaps[8])
    assert rd.load_meta(5) == {"loss": 0.2}
    assert rd.verify(8) and rd.verify_all()["ok"]
    assert rd.info(8)["chain_length"] == 2
    # restore lands on CPU torch tensors with correct dtypes
    t = rd.load(3)["model"]["embed.weight"]
    assert t.dtype == torch.bfloat16 and t.shape == (128, 64)


def test_modes_and_compression(store):
    w = CheckpointWriter(store, "r2", keyframe_every=5)
    state = make_state()
    w.save(1, state)
    assert w.last_stats["is_keyframe"] and w.last_stats["modes"]["xor"] == 0
    kf_payload = w.last_stats["payload_bytes"]
    advance(state, 2, lr=1e-5)
    w.save(2, state)
    st = w.last_stats
    assert not st["is_keyframe"]
    assert st["modes"]["same"] >= 3  # frozen/norm/ids/mask tensors are bit-identical -> nothing stored
    assert st["modes"]["xor"] >= 3
    assert st["payload_bytes"] < 0.75 * kf_payload, (st["payload_bytes"], kf_payload)
    assert st["ratio"] < 0.75
    m = store.get_manifest(w.last_stats["manifest"])
    assert m.extra["entries"]["model/frozen.weight"]["mode"] == "same"


def test_delta_fallback_to_raw_when_incompressible(store):
    w = CheckpointWriter(store, "r3", keyframe_every=10)
    a = {"x": torch.randn(4096)}
    w.save(1, a)
    w.save(2, {"x": torch.randn(4096)})  # unrelated tensor -> delta no better than raw
    m = store.get_manifest(w.last_stats["manifest"])
    assert m.extra["entries"]["x"]["mode"] == "raw"
    assert_same(CheckpointReader(store, "r3").load(2)["x"], CheckpointReader(store, "r3").load(2)["x"])


def test_resume_with_new_writer_and_reader(store):
    state = make_state()
    w1 = CheckpointWriter(store, "r4", keyframe_every=4)
    snaps = {}
    for step in (10, 20, 30):
        advance(state, step)
        w1.save(step, state)
        snaps[step] = clone(state)
    # a fresh process: new writer continues the chain from the stored latest checkpoint
    w2 = CheckpointWriter(store, "r4", keyframe_every=4)
    for step in (40, 50, 60):
        advance(state, step)
        w2.save(step, state)
        snaps[step] = clone(state)
    rd = CheckpointReader(store, "r4")
    for step, snap in snaps.items():
        assert_same(rd.load(step), snap)
    assert [r["is_keyframe"] for r in rd.rows(include_dropped=True)] == [True, False, False, False, True, False]


@pytest.mark.parametrize("cache_base", [True, False])
def test_cache_base_off_gives_identical_results(store, cache_base):
    cfg = CheckpointConfig(keyframe_every=4, cache_base=cache_base)
    w = CheckpointWriter(store, f"r5-{cache_base}", config=cfg)
    state = make_state()
    snaps = {}
    for step in range(1, 6):
        advance(state, step)
        w.save(step, state)
        snaps[step] = clone(state)
    rd = CheckpointReader(store, f"r5-{cache_base}")
    for step, snap in snaps.items():
        assert_same(rd.load(step), snap)


def test_cache_does_not_alias_live_tensors(store):
    """The writer must not keep references into tensors the training loop mutates in place."""
    w = CheckpointWriter(store, "alias", keyframe_every=10)
    p = torch.randn(1000)
    w.save(1, {"p": p})
    snap1 = p.clone()
    p.add_(0.01)  # in-place optimiser update
    snap2 = p.clone()
    w.save(2, {"p": p})
    rd = CheckpointReader(store, "alias")
    assert_same(rd.load(1)["p"], snap1)
    assert_same(rd.load(2)["p"], snap2)


def test_async_save_snapshots_state(store):
    w = CheckpointWriter(store, "async", keyframe_every=3)
    p = torch.randn(5000)
    snaps = {}
    futs = []
    for step in range(1, 6):
        p.add_(0.001 * step)
        snaps[step] = p.clone()
        futs.append(w.save_async(step, {"p": p, "step": step}))
        p.mul_(1.0001)  # mutate immediately after submission - must not leak into the saved snapshot
        snaps[step + 100] = None
    hashes = w.flush()
    assert len(hashes) == 5 and all(f.done() for f in futs)
    rd = CheckpointReader(store, "async")
    for step in range(1, 6):
        assert_same(rd.load(step)["p"], snaps[step])
        assert rd.load(step)["step"] == step
    w.close()


def test_async_errors_surface_on_flush(store):
    from forge.core.errors import StoreError

    w = CheckpointWriter(store, "asyncerr")
    with pytest.raises(StoreError):
        w.save_async(1, {"bad": object()})  # flattening happens synchronously -> raises immediately
    w.save_async(2, {"x": torch.randn(10)})
    w.flush()
    assert CheckpointReader(store, "asyncerr").steps() == [2]


def test_latest_best_final_and_tags(store):
    pol = RetentionPolicy(keep_last=2, keep_best=1, best_mode="min", keep_keyframes=False)
    w = CheckpointWriter(store, "r6", keyframe_every=3, retention=pol)
    state = make_state()
    metrics = {1: 5.0, 2: 3.0, 3: 4.0, 4: 6.0, 5: 7.0, 6: 8.0, 7: 9.0}
    for step, mval in metrics.items():
        advance(state, step)
        w.save(step, state, metric=mval, tags=["pre_anneal"] if step == 4 else [])
    rd = CheckpointReader(store, "r6")
    assert rd.resolve("best")["step"] == 2
    assert rd.resolve("latest")["step"] == 7
    assert rd.resolve("pre_anneal")["step"] == 4
    assert rd.resolve("final")["step"] == 7
    assert rd.resolve("tag:pre_anneal")["step"] == 4
    # retention: last 2 + best + pre_anneal tag survive; the rest are dropped
    assert set(rd.steps()) == {2, 4, 6, 7}
    assert w.last_stats["dropped_steps"] is not None
    with pytest.raises(NotFound):
        rd.resolve("nonexistent-tag")
    with pytest.raises(NotFound):
        rd.resolve(99)
    w.tag(7, "final")
    assert 7 in [r["step"] for r in rd.rows() if "final" in r["tags"]]
    assert store.is_pinned(rd.resolve(7)["manifest_hash"])
    # best_mode max
    w2 = CheckpointWriter(store, "r6max", retention=RetentionPolicy(best_mode="max"))
    for s, m in ((1, 0.1), (2, 0.9), (3, 0.5)):
        w2.save(s, {"x": torch.randn(4)}, metric=m)
    assert CheckpointReader(store, "r6max").resolve("best")["step"] == 2


def test_retention_then_gc_keeps_chains_loadable(store):
    pol = RetentionPolicy(keep_last=2, keep_best=0, keep_keyframes=False, keep_tags=[])
    w = CheckpointWriter(store, "r7", keyframe_every=4, retention=pol)
    state = make_state()
    snaps = {}
    for step in range(1, 11):
        advance(state, step)
        w.save(step, state)
        snaps[step] = clone(state)
    rd = CheckpointReader(store, "r7")
    assert rd.steps() == [9, 10]
    before = store.stats()["unique_bytes"]
    rep = store.gc(dry_run=False, grace_s=0)
    assert rep.deleted_objects > 0
    assert store.stats()["unique_bytes"] < before
    # 9 and 10 are deltas on keyframe 9? (keyframes 1,5,9): 9 is a keyframe, 10 a delta; both must still restore exactly
    assert_same(rd.load(9), snaps[9])
    assert_same(rd.load(10), snaps[10])
    # steps 1..8 are gone (no live checkpoint depends on them)
    with pytest.raises(NotFound):
        rd.load(2)
    assert store.verify_all(deep=True)["ok"]


def test_gc_protects_chain_of_kept_delta(store):
    """Keeping only a late delta must keep its whole chain (keyframe + intermediate deltas) alive."""
    pol = RetentionPolicy(keep_last=1, keep_best=0, keep_keyframes=False, keep_tags=[])
    w = CheckpointWriter(store, "r8", keyframe_every=100, retention=pol)
    state = make_state()
    for step in range(1, 7):
        advance(state, step)
        w.save(step, state)
    final = clone(state)
    rd = CheckpointReader(store, "r8")
    assert rd.steps() == [6]
    rep = store.gc(dry_run=True, grace_s=0)
    assert rep.chain_protected_objects >= 5  # steps 1..5 are kept alive only through chain references
    store.gc(dry_run=False, grace_s=0)
    assert_same(rd.load(6), final)
    assert rd.verify(6)


def test_resave_older_step_abandons_later_timeline(store):
    w = CheckpointWriter(store, "r9", keyframe_every=10, retention=RetentionPolicy(keep_last=50))
    state = make_state()
    for step in range(1, 6):
        advance(state, step)
        w.save(step, state)
    rd = CheckpointReader(store, "r9")
    state3 = rd.load(3)
    # training restarted from step 3 with a different data order
    new4 = advance(clone(state3), 4, seed=7)
    w.save(4, new4)
    assert rd.steps() == [1, 2, 3, 4]
    assert_same(rd.load(4), new4)
    assert_same(rd.load(3), state3)


def test_keyframe_codec_lossless(store):
    cfg = CheckpointConfig(keyframe_every=3, keyframe_codec="lossless")
    w = CheckpointWriter(store, "r10", config=cfg)
    state = make_state()
    snaps = {}
    for step in range(1, 5):
        advance(state, step)
        w.save(step, state)
        snaps[step] = clone(state)
        if step == 1:
            assert w.last_stats["modes"]["flc"] >= 3
    rd = CheckpointReader(store, "r10")
    for step, snap in snaps.items():
        assert_same(rd.load(step), snap)


def test_zstd_dictionary_option_roundtrips(store):
    cfg = CheckpointConfig(keyframe_every=50, use_dictionary=True)
    w = CheckpointWriter(store, "r11", config=cfg)
    g = torch.Generator().manual_seed(3)
    state = {f"t{i}": torch.randn(300, generator=g) for i in range(40)}
    snaps = {}
    for step in range(1, 6):
        for k in state:
            state[k] = state[k] + 1e-3 * torch.randn(300, generator=g)
        w.save(step, state)
        snaps[step] = clone(state)
    rd = CheckpointReader(store, "r11")
    for step, snap in snaps.items():
        assert_same(rd.load(step), snap)
    # a resumed writer picks the dictionary up again
    w2 = CheckpointWriter(store, "r11", config=cfg)
    for k in state:
        state[k] = state[k] + 1e-3
    w2.save(6, state)
    assert_same(rd.load(6), state)


def test_corruption_is_detected_on_restore(store):
    w = CheckpointWriter(store, "r12", keyframe_every=5)
    state = make_state()
    w.save(1, state)
    advance(state, 2)
    w.save(2, state)
    m = store.get_manifest(w.last_stats["manifest"])
    victim = next(e["obj"] for e in m.extra["entries"].values() if e["mode"] == "xor")
    row = store._row(victim)
    p = store._path(victim, row["tier"], row["codec"])
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0x7F
    p.write_bytes(bytes(raw))
    rd = CheckpointReader(store, "r12")
    with pytest.raises(IntegrityError):
        rd.load(2)
    assert not rd.verify(2)
    assert rd.verify(1)


def test_listing_helpers_and_export(store, tmp_path):
    w = CheckpointWriter(store, "r13", keyframe_every=2)
    state = make_state()
    for step in (1, 2, 3):
        advance(state, step)
        w.save(step, state)
    rows = list_checkpoints(store, "r13", with_sizes=True)
    assert [r["step"] for r in rows] == [3, 2, 1] and all(r["stored_bytes"] > 0 for r in rows)
    assert list_runs(store)[0]["run_id"] == "r13"
    assert len(list_checkpoints(store)) == 3
    out = CheckpointReader(store, "r13").export_safetensors(3, tmp_path / "exp", key="model", config={"x": 1})
    from safetensors.torch import load_file

    loaded = load_file(str(out / "model.safetensors"))
    assert set(loaded) == set(state["model"])
    assert loaded["embed.weight"].view(torch.int16).equal(state["model"]["embed.weight"].view(torch.int16))
    assert (out / "config.json").exists()


def test_checkpoint_manifests_are_signed_and_linked(store):
    w = CheckpointWriter(store, "r14", keyframe_every=5)
    h1 = w.save(1, {"x": torch.randn(100)})
    h2 = w.save(2, {"x": torch.randn(100)})
    assert store.signature_status(h1) == "valid" and store.verify(h2)
    assert store.get_manifest(h2).parents == [h1]
    assert any(r["dst"] == h1 and r["rtype"] == "chain" for r in store.db.query("SELECT * FROM store_refs WHERE src=?", (h2,)))
    assert store.stats()["per_kind"]["checkpoint"]["rooted_objects"] >= 2


def test_timing_of_async_vs_sync_is_not_asserted_but_works(store):
    w = CheckpointWriter(store, "r15")
    t0 = time.perf_counter()
    f = w.save_async(1, {"x": torch.randn(200_000)})
    submit = time.perf_counter() - t0
    f.result()
    assert submit < 5
    assert CheckpointReader(store, "r15").verify(1)
