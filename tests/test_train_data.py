"""Ingestion, sharding, deterministic loading, dedup, contamination."""
from __future__ import annotations

import json

import numpy as np
import pytest
from test_train_fixtures import ingest_tiny, write_corpus

from forge.core.context import NullContext
from forge.core.errors import ConfigError
from forge.core.schemas import DatasetManifest
from forge.train.data import (
    ContaminationScanner,
    IngestConfig,
    MinHashDeduper,
    ShardedTokenDataset,
    StepPrefetcher,
    epoch_permutation,
    ingest_dataset,
    load_dataset_manifest,
)
from forge.train.tokenization import ByteTokenizer, TokenizerAdapter, load_tokenizer_any


def _ingest(tmp, sources, **kw):
    cfg = IngestConfig(name="d", sources=sources, tokenizer="byte", output_dir=str(tmp / "out"), **kw)
    ctx = NullContext(run_dir=tmp / "run")
    return ingest_dataset(cfg, ctx), ctx


def test_byte_tokenizer_roundtrip_and_specials():
    t = ByteTokenizer()
    ids = t.encode("héllo<|im_start|>x</s>")
    assert t.special_to_id["<|im_start|>"] in ids and ids[-1] == t.eos_token_id
    assert t.decode(ids) == "héllo<|im_start|>x</s>"
    a = TokenizerAdapter(t)
    assert a.vocab_size == t.vocab_size and a.eos_id == t.eos_token_id and a.hash.startswith(("blake3:", "b2:"))
    assert load_tokenizer_any("byte").is_byte
    with pytest.raises(ConfigError):
        load_tokenizer_any("/definitely/not/a/tokenizer")


def test_ingest_jsonl_manifest_and_shards(tmp_path):
    src = write_corpus(tmp_path)
    summary, ctx = _ingest(tmp_path, [{"path": str(src)}], shard_size_mib=0.004)
    out = tmp_path / "out"
    m = DatasetManifest.model_validate(json.loads((out / "manifest.json").read_text()))
    assert m.total_tokens == summary["total_tokens"] == sum(s.tokens for s in m.shards)
    assert len(m.shards) >= 2  # tiny shards force splitting
    eos = ByteTokenizer().eos_token_id
    for s in m.shards:
        ex = s.model_extra
        arr = np.fromfile(out / ex["path"], dtype=ex["dtype"])
        offs = np.load(out / ex["offsets"])
        assert len(arr) == s.tokens and offs[0] == 0 and offs[-1] == s.tokens and len(offs) == s.docs + 1
        assert ex["stats"]["tokens"] == s.tokens and "unigram_entropy_bits" in ex["stats"]
        assert all(arr[offs[i + 1] - 1] == eos for i in range(s.docs))  # every doc ends with EOS
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["dtype"] == "uint16" and info["eos_id"] == eos
    assert ctx.outputs["dataset_id"] == m.id and m.id.startswith("ds_")


def test_ingest_sources_txt_parquet_dir_and_bad_records(tmp_path):
    (tmp_path / "a.txt").write_text("para one\n\npara two\n\npara three", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"text":"alpha beta"}\nnot json\n{"nottext":1}\n{"text":"gamma"}\n', encoding="utf-8")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"body": ["from parquet one", "from parquet two"]}), str(tmp_path / "c.parquet"))
    d = tmp_path / "folder"
    d.mkdir()
    (d / "x.txt").write_text("in a folder", encoding="utf-8")
    summary, _ = _ingest(tmp_path, [
        {"path": str(tmp_path / "a.txt"), "kind": "txt", "txt_split": "blank_line", "name": "txt"},
        {"path": str(tmp_path / "b.jsonl")},
        {"path": str(tmp_path / "c.parquet"), "text_field": "body"},
        {"path": str(d), "kind": "dir"},
    ])
    rep = summary["report"]
    assert rep["kept"] == 3 + 2 + 2 + 1
    assert rep["bad_records"] == 2
    m, _, _ = load_dataset_manifest(tmp_path / "out")
    assert m.mixture["txt"]["docs"] == 3


def test_exact_dedup_and_min_length(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = ["same doc", "same doc", "  same doc  ", "other", "x"]
    p.write_text("\n".join(json.dumps({"text": r}) for r in rows), encoding="utf-8")
    summary, _ = _ingest(tmp_path, [{"path": str(p)}], min_doc_tokens=2)
    rep = summary["report"]
    assert rep["exact_dup"] == 2 and rep["too_short"] == 1 and rep["kept"] == 2


def test_minhash_near_dup(tmp_path):
    base = " ".join(f"word{i}" for i in range(80))
    near = base + " extra tail"
    far = " ".join(f"other{i}" for i in range(80))
    dd = MinHashDeduper(num_perm=64, ngram=3, threshold=0.7)
    assert not dd.is_duplicate(base)
    assert dd.is_duplicate(near)
    assert not dd.is_duplicate(far)
    assert MinHashDeduper.jaccard_estimate(dd.signature(base), dd.signature(near)) > 0.7
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"text": t}) for t in (base, near, far)), encoding="utf-8")
    summary, _ = _ingest(tmp_path, [{"path": str(p)}], dedup_minhash=True, minhash_ngram=3, minhash_threshold=0.7)
    assert summary["report"]["minhash_dup"] == 1 and summary["report"]["kept"] == 2


def test_contamination_scan_report_and_drop(tmp_path):
    secret = "the answer to the benchmark question is forty two exactly"
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps({"question": secret}), encoding="utf-8")
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"text": t}) for t in ("clean text about cats and dogs " * 3, "leak: " + secret + " ok")), encoding="utf-8")
    s1, _ = _ingest(tmp_path, [{"path": str(p)}], contamination={"eval_files": [str(ev)], "ngram": 8, "action": "report"})
    assert s1["report"]["contaminated"] == 1 and s1["report"]["kept"] == 2
    # post-hoc scan of the ingested dataset
    sc = ContaminationScanner.from_files([str(ev)], TokenizerAdapter(ByteTokenizer()), 8)
    assert sc.scan_dataset(tmp_path / "out")["contaminated_docs"] == 1
    sub = tmp_path / "second"
    sub.mkdir()
    s2, _ = _ingest(sub, [{"path": str(p)}], contamination={"eval_files": [str(ev)], "ngram": 8, "action": "drop"})
    assert s2["report"]["contaminated_dropped"] == 1 and s2["report"]["kept"] == 1


def test_val_split_and_holdout(tmp_path):
    src = write_corpus(tmp_path, n_repeat=10)
    cfg = IngestConfig(name="v", sources=[{"path": str(src)}], tokenizer="byte", output_dir=str(tmp_path / "v"), val_fraction=0.2, shard_size_mib=0.01)
    ingest_dataset(cfg, NullContext(run_dir=tmp_path / "r"))
    m, _, _ = load_dataset_manifest(tmp_path / "v")
    assert {s.model_extra["split"] for s in m.shards} == {"train", "val"}
    tr = ShardedTokenDataset(tmp_path / "v", 8, role="train")
    va = ShardedTokenDataset(tmp_path / "v", 8, role="val")
    assert len(tr) > 0 and len(va) > 0 and not va.holdout_mode


def test_sharded_dataset_determinism_and_resume(tmp_path):
    d = ingest_tiny(tmp_path)
    a = ShardedTokenDataset(d, 16, seed=5)
    b = ShardedTokenDataset(d, 16, seed=5)
    c = ShardedTokenDataset(d, 16, seed=6)
    for step in (0, 1, 7, 100):  # includes epoch wrap-around
        x, y = a.batch_at(step, 6), b.batch_at(step, 6)
        assert (x["input_ids"] == y["input_ids"]).all() and (x["labels"] == y["labels"]).all()
    assert not (a.batch_at(3, 6)["input_ids"] == c.batch_at(3, 6)["input_ids"]).all()
    # labels are the inputs shifted by one inside the same window
    w = a.window(int(a.ids[0]))
    bt = a.get_batch(np.array([0]))
    assert (bt["input_ids"][0].numpy() == w[:-1]).all() and (bt["labels"][0].numpy() == w[1:]).all()
    # rank slices partition the global batch
    full = a.batch_at(2, 8)["seq_ids"].tolist()
    parts = a.batch_at(2, 8, 0, 2)["seq_ids"].tolist() + a.batch_at(2, 8, 1, 2)["seq_ids"].tolist()
    assert full == parts
    # every sequence appears exactly once per epoch
    n = len(a)
    seen = []
    for s in range(n // 4):
        seen += a.batch_at(s, 4)["seq_ids"].tolist()
    assert len(set(seen)) == len(seen) == (n // 4) * 4
    with pytest.raises(ConfigError):
        a.batch_at(0, 5, 0, 2)


def test_holdout_is_disjoint_and_stable(tmp_path):
    d = ingest_tiny(tmp_path)
    tr = ShardedTokenDataset(d, 16, role="train", val_sequences=10)
    va = ShardedTokenDataset(d, 16, role="val", val_sequences=10)
    assert len(va) == 10 and set(tr.ids.tolist()).isdisjoint(va.ids.tolist())
    tr2 = ShardedTokenDataset(d, 16, seed=99, role="train", val_sequences=10)
    assert tr.ids.tolist() == tr2.ids.tolist()  # holdout independent of the data seed
    eb = va.eval_batches(4, 2)
    assert sum(b["input_ids"].shape[0] for b in eb) == 8
    tr.restrict(20)
    assert len(tr) == 20


def test_epoch_permutation_pure():
    assert (epoch_permutation(50, 1, 2, 3) == epoch_permutation(50, 1, 2, 3)).all()
    assert not (epoch_permutation(50, 1, 2, 3) == epoch_permutation(50, 1, 3, 3)).all()


def test_prefetcher_order():
    pf = StepPrefetcher(lambda s: s * 10, start=3, end=9, depth=3, threads=2)
    assert list(pf) == [(s, s * 10) for s in range(3, 9)]
    pf.close()


def test_manifest_stored_in_store(tmp_path):
    from forge.store import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    src = write_corpus(tmp_path)
    ctx = NullContext(run_dir=tmp_path / "r", store=store)
    cfg = IngestConfig(name="s", sources=[{"path": str(src)}], tokenizer="byte", output_dir=str(tmp_path / "o"))
    summary = ingest_dataset(cfg, ctx)
    assert summary["manifest_hash"] and store.has(summary["manifest_hash"])


def test_hf_source_via_datasets_module(tmp_path, monkeypatch):
    import sys
    import types

    calls = {}

    def load_dataset(path, name, split=None, streaming=True):
        calls.update(path=path, name=name, split=split, streaming=streaming)
        return iter([{"content": "doc one"}, {"content": "doc two"}, {"other": 1}])

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    summary, _ = _ingest(tmp_path, [{"kind": "hf", "path": "org/ds", "hf_config": "en", "split": "validation", "text_field": "content", "name": "hf"}])
    assert summary["report"]["kept"] == 2 and summary["report"]["bad_records"] == 1
    assert calls == {"path": "org/ds", "name": "en", "split": "validation", "streaming": True}


def test_hf_source_without_datasets_package_is_a_clear_error(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "datasets", None)  # makes `import datasets` raise ImportError
    with pytest.raises(ConfigError, match="datasets"):
        _ingest(tmp_path, [{"kind": "hf", "path": "org/ds"}])
