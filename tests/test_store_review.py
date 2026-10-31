"""Adversarial-review regression tests for forge.store (security, GC safety, chain restore, chunking, tamper detection)."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch

from forge.core.errors import IntegrityError, StoreError
from forge.core.schemas import ArtifactManifest
from forge.store import ArtifactStore
from forge.store import lossless as L
from forge.store.checkpoints import CheckpointReader, CheckpointWriter, RetentionPolicy
from forge.store.chunking import ChunkerParams, chunk_offsets, iter_chunks

SMALL = ChunkerParams(min_size=2048, bits=12, max_size=16384)


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "store", chunk_params=SMALL, inline_threshold=8192)


def rnd(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


# ------------------------------------------------------------------------------------------------ security
@pytest.mark.parametrize("bad", ["../../escaped.txt", "/abs_escaped.txt", "a/../../escaped.txt", "..\\..\\escaped.txt", "C:/escaped.txt", ""])
def test_materialize_artifact_refuses_paths_that_escape_the_destination(store, tmp_path, bad):
    h = store.put_bytes(b"payload" * 10, kind="file")
    m = ArtifactManifest(kind="model", tensors=[], extra={"files": [{"path": bad, "type": "raw", "hash": h, "size": 70}]})
    mh = store.put_manifest(m)
    dest = tmp_path / "out" / "x"
    with pytest.raises(StoreError):
        store.materialize_artifact(mh, dest)
    assert not (tmp_path / "escaped.txt").exists() and not (tmp_path / "out" / "escaped.txt").exists()


def test_materialize_artifact_normal_nested_paths_still_work(store, tmp_path):
    h = store.put_bytes(b"hello", kind="file")
    m = ArtifactManifest(kind="model", tensors=[], extra={"files": [{"path": "sub/dir/./a.txt", "type": "raw", "hash": h, "size": 5}]})
    out = store.materialize_artifact(store.put_manifest(m), tmp_path / "ok")
    assert (out / "sub" / "dir" / "a.txt").read_bytes() == b"hello"


# ------------------------------------------------------------------------------------------------ tamper detection
def test_corrupt_chunk_is_detected_on_read_and_verify(store):
    d = rnd(100_000, 3)
    h = store.put_bytes(d)
    ch = store._chunk_list(h)[1][0]
    p = store._file_for(store._row(ch))
    b = bytearray(p.read_bytes())
    b[10] ^= 1
    p.write_bytes(bytes(b))
    with pytest.raises(IntegrityError):
        store.get_bytes(h)
    assert not store.verify(h)


def test_manifest_content_must_match_its_address(store):
    h = store.put_manifest(ArtifactManifest(kind="model", tensors=[], extra={"a": 1}))
    p = store._file_for(store._row(h))
    j = json.loads(p.read_bytes())
    j["extra"]["a"] = 2
    p.write_bytes(json.dumps(j).encode())
    with pytest.raises(IntegrityError):
        store.get_manifest(h)


# ------------------------------------------------------------------------------------------------ gc safety
def test_gc_grace_is_widened_while_a_long_writer_is_in_flight(store):
    h = store.put_bytes(rnd(50_000, 5), kind="ckpt_obj")  # unrooted payload object with no referrer yet (like a checkpoint mid-save)
    with store.writing("test"):
        assert store.inflight_writers() == 1
        rep = store.gc(dry_run=False, grace_s=0)
        assert rep.deleted_objects == 0 and rep.skipped_young > 0
        assert store.has(h)
    assert store.inflight_writers() == 0
    assert store.gc(dry_run=False, grace_s=0).deleted_objects > 0
    assert not store.has(h)


def test_checkpoint_save_registers_inflight_marker_and_clears_it(store):
    w = CheckpointWriter(store, "r", keyframe_every=3)
    w.save(1, {"x": torch.randn(100)})
    assert store.inflight_writers() == 0


def test_delta_chain_restores_bit_exact_after_retention_and_gc(store):
    w = CheckpointWriter(store, "r", keyframe_every=4, retention=RetentionPolicy(keep_last=1, keep_keyframes=False, keep_best=0))
    g = torch.Generator().manual_seed(0)
    state = {"w": torch.randn(64, 64, generator=g).to(torch.bfloat16), "b": torch.randn(64, generator=g), "i": torch.arange(10), "m": torch.zeros(5, dtype=torch.bool)}
    hist = {}
    for step in range(1, 16):
        state["w"] = (state["w"].float() + 0.01 * torch.randn(64, 64, generator=g)).to(torch.bfloat16)
        state["b"] = state["b"] + 0.001 * torch.randn(64, generator=g)
        w.save(step, {**state, "s": step}, {"k": step})
        hist[step] = {k: v.clone() for k, v in state.items()}
    rep = store.gc(dry_run=False, grace_s=0)
    assert rep.chain_protected_objects > 0  # keyframe/intermediate deltas kept alive only through chain references
    r = CheckpointReader(store, "r")
    out = r.load("latest")
    assert out["s"] == 15
    for k in ("w", "b", "i", "m"):
        assert torch.equal(out[k], hist[15][k]), k
    assert r.verify("latest")


def test_checkpoint_odd_tensors_and_async_snapshot_isolation(store):
    w = CheckpointWriter(store, "r", keyframe_every=3)
    x = torch.randn(1000)
    f8 = torch.randn(8).to(torch.float8_e4m3fn)
    w.save(1, {"v": x[::2], "e": torch.empty(0), "z": torch.tensor(3.5), "f8": f8})
    x2 = x + 1
    w.save(2, {"v": x2[::2], "e": torch.empty(0), "z": torch.tensor(3.5), "f8": f8})
    o = CheckpointReader(store, "r").load(2)
    assert torch.equal(o["v"], x2[::2]) and o["e"].shape == (0,) and o["z"].item() == 3.5 and o["f8"].dtype == torch.float8_e4m3fn
    assert torch.equal(o["f8"].view(torch.uint8), f8.view(torch.uint8))
    # async: mutating the live tensor right after save_async must not change the stored snapshot
    w2 = CheckpointWriter(store, "async", keyframe_every=5)
    y = torch.zeros(5000)
    w2.save_async(1, {"y": y})
    y.add_(1.0)
    w2.save_async(2, {"y": y})
    w2.flush()
    r = CheckpointReader(store, "async")
    assert torch.equal(r.load(1)["y"], torch.zeros(5000)) and torch.equal(r.load(2)["y"], torch.ones(5000))


# ------------------------------------------------------------------------------------------------ chunking
@pytest.mark.parametrize("read_size", [1, 7, 1000, 5000, 16384, 16385, 100_000, 10**7])
def test_stream_chunking_equals_byte_chunking_for_any_read_size(read_size):
    d = rnd(300_000, 1)
    ref = [d[o : o + n] for o, n in chunk_offsets(d, SMALL)]
    pos = [0]

    def rd(n):
        b = d[pos[0] : pos[0] + n]
        pos[0] += len(b)
        return b

    assert list(iter_chunks(rd, SMALL, read_size=read_size)) == ref
    assert b"".join(ref) == d
    sizes = [len(c) for c in ref]
    assert max(sizes) <= SMALL.max_size and min(sizes[:-1]) >= SMALL.min_size


def test_put_stream_and_put_bytes_agree_and_dedup(store):
    a = rnd(200_000, 9)
    h1 = store.put_bytes(a)
    unique = store.stats()["unique_bytes"]
    assert store.put_stream(io.BytesIO(a)) == h1 and store.put_bytes(a, kind="other") == h1
    assert store.stats()["unique_bytes"] == unique
    assert store.get_bytes(h1) == a


# ------------------------------------------------------------------------------------------------ lossless
@pytest.mark.parametrize("dtype,nbits", [("bfloat16", 16), ("float16", 16), ("float8_e4m3fn", 8), ("float8_e5m2", 8)])
@pytest.mark.parametrize("method", ["auto", "exp_huffman", "exp_zstd", "bytegroup_zstd", "raw"])
def test_lossless_is_bit_exact_over_every_bit_pattern(dtype, nbits, method):
    raw = np.arange(2**nbits, dtype=np.uint16 if nbits == 16 else np.uint8).tobytes()  # includes NaN/Inf/denormals/-0
    out, d, _ = L.decode_bytes(L.encode_bytes(raw, dtype, method=method))
    assert bytes(out) == raw and d == dtype


def test_lossless_tiny_and_empty_inputs():
    for n in (0, 1, 2, 3, 100):
        r = np.zeros(n, dtype=np.uint16).tobytes()
        assert bytes(L.decode_bytes(L.encode_bytes(r, "bfloat16"))[0]) == r


# ------------------------------------------------------------------------------------------------ writer restart / codec matrix
@pytest.mark.parametrize("cfg_kw", [
    {"keyframe_every": 4},
    {"keyframe_every": 4, "cache_base": False},
    {"keyframe_every": 5, "use_dictionary": True},
    {"keyframe_every": 3, "keyframe_codec": "lossless"},
    {"keyframe_every": 100, "delta_max_ratio": 0.01},
])
def test_every_checkpoint_restores_bit_exact_across_writer_restart_and_gc(store, cfg_kw):
    from forge.store.checkpoints import CheckpointConfig

    cfg = CheckpointConfig(**cfg_kw)
    g = torch.Generator().manual_seed(0)
    state = {f"w{i}": torch.randn(40, 40, generator=g).bfloat16() for i in range(12)}
    state["f"] = torch.randn(300, generator=g)
    hist = {}
    w = CheckpointWriter(store, "r", config=cfg)
    for s in range(1, 13):
        if s == 6:
            w = CheckpointWriter(store, "r", config=cfg)  # simulated process restart: the delta base must be re-materialised from the store
        for k in list(state):
            if k.startswith("w"):
                state[k] = (state[k].float() + 0.01 * torch.randn(40, 40, generator=g)).bfloat16()
        state["f"] = state["f"] + 1e-3 * torch.randn(300, generator=g)
        w.save(s, {**state, "step": s}, metric=1.0 / s if s % 3 else 0.5)
        hist[s] = {k: v.clone() for k, v in state.items()}
    r = CheckpointReader(store, "r")
    store.gc(dry_run=False, grace_s=0)
    for s in r.steps():
        out = r.load(s)
        assert out["step"] == s
        for k, v in hist[s].items():
            assert torch.equal(out[k], v), (s, k)
    assert r.verify_all()["ok"]
