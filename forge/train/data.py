"""Dataset ingestion (``dataset_ingest`` job) and deterministic sharded token loading (Plan 3.4, 7.4, 8.5).

Ingestion pipeline::

    sources (jsonl/txt/parquet/HF/dir) -> unicode normalise -> exact dedup -> optional MinHash near-dup ->
    tokenise -> optional contamination scan (n-gram overlap vs eval sets) -> length filter ->
    train/val split (by content hash) -> pack documents into uint16/uint32 shards
    (shard_00000.bin + shard_00000.offsets.npy) -> per-shard stats -> DatasetManifest (manifest.json + store)

Shard format: raw little-endian token ids (``.bin``, read with ``np.memmap``) plus an int64 ``.offsets.npy`` sidecar of
document start offsets (length docs+1).  Documents are never split across shards unless a single document exceeds the shard
size (then it is truncated, and counted in the stats).

Loading: :class:`ShardedTokenDataset` exposes ``batch_at(step, global_batch_size, rank, world)``, a *pure function* of
``(manifest, seed, step)`` -- so resuming at step *s* reproduces the exact data order with no loader state.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import logging
import re
import threading
import unicodedata
import zlib
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np

from forge.core.errors import ConfigError, NotFound
from forge.core.hashing import hash_file, hash_json
from forge.core.schemas import DatasetManifest, ShardEntry
from forge.train.config import FF, F, _Cfg
from forge.train.tokenization import TokenizerAdapter, load_tokenizer_any

log = logging.getLogger("forge.train.data")

__all__ = [
    "ContaminationConfig", "ContaminationScanner", "IngestConfig", "MinHashDeduper", "ShardedTokenDataset", "SourceSpec",
    "StepPrefetcher", "epoch_permutation", "ingest_dataset", "iter_source_texts", "load_dataset_manifest",
    "resolve_dataset_path", "step_positions",
]


# ---------------------------------------------------------------------------------------------------------------
# deterministic sampling
# ---------------------------------------------------------------------------------------------------------------
def _seed_seq(seed: int, epoch: int, salt: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF, int(salt) & 0xFFFFFFFF])


def epoch_permutation(n: int, seed: int, epoch: int, salt: int = 0) -> np.ndarray:
    """Permutation of ``range(n)`` for ``epoch`` -- a pure function of ``(n, seed, epoch, salt)``."""
    rng = np.random.default_rng(_seed_seq(seed, epoch, salt))
    return rng.permutation(n)


class _PermCache:
    def __init__(self, n: int, seed: int, salt: int, shuffle: bool) -> None:
        self.n, self.seed, self.salt, self.shuffle = n, seed, salt, shuffle
        self._cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        self._lock = threading.Lock()

    def perm(self, epoch: int) -> np.ndarray:
        with self._lock:
            if epoch in self._cache:
                self._cache.move_to_end(epoch)
                return self._cache[epoch]
        p = epoch_permutation(self.n, self.seed, epoch, self.salt) if self.shuffle else np.arange(self.n)
        with self._lock:
            self._cache[epoch] = p
            while len(self._cache) > 3:
                self._cache.popitem(last=False)
        return p


def step_positions(cache: _PermCache, step: int, global_batch: int) -> np.ndarray:
    """Dataset positions consumed by optimiser step ``step`` (deterministic; spans epoch boundaries)."""
    n = cache.n
    if n <= 0:
        raise ConfigError("dataset is empty")
    start = int(step) * int(global_batch)
    pos = np.arange(start, start + int(global_batch), dtype=np.int64)
    ep = pos // n
    off = pos % n
    out = np.empty_like(pos)
    for e in np.unique(ep):
        m = ep == e
        out[m] = cache.perm(int(e))[off[m]]
    return out


class StepPrefetcher:
    """Computes ``make(step)`` ahead of time on worker threads; yields ``(step, value)`` in order."""

    def __init__(self, make: Any, start: int = 0, end: int | None = None, depth: int = 4, threads: int = 1) -> None:
        self.make, self.next_step, self.end = make, int(start), end
        self.depth = max(1, int(depth))
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(threads))) if threads and threads > 0 else None
        self.q: collections.deque[tuple[int, Future | Any]] = collections.deque()
        self._submit_step = int(start)

    def _fill(self) -> None:
        while len(self.q) < self.depth and (self.end is None or self._submit_step < self.end):
            s = self._submit_step
            self._submit_step += 1
            self.q.append((s, self.pool.submit(self.make, s) if self.pool else self.make(s)))

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        return self

    def __next__(self) -> tuple[int, Any]:
        self._fill()
        if not self.q:
            raise StopIteration
        s, v = self.q.popleft()
        if isinstance(v, Future):
            v = v.result()
        self._fill()
        return s, v

    def close(self) -> None:
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)
        self.q.clear()


# ---------------------------------------------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------------------------------------------
class SourceSpec(_Cfg):
    """One input source of an ingest job."""

    kind: Literal["auto", "jsonl", "txt", "parquet", "hf", "dir"] = F("auto", "Source type. auto = from the path (file extension / directory).", "Source")
    path: str = F("", "File or directory path, or a Hugging Face dataset id when kind=hf.", "Source")
    name: str = F("", "Label of this source in the dataset mixture report (default: file/dir name).", "Source")
    text_field: str = F("text", "Field (dotted path allowed) holding the document text in jsonl/parquet/HF records.", "Source")
    split: str = F("train", "Hugging Face split.", "Source", True)
    hf_config: str | None = F(None, "Hugging Face dataset configuration name.", "Source", True)
    streaming: bool = F(True, "Stream Hugging Face datasets instead of downloading them whole.", "Source", True)
    txt_split: Literal["file", "blank_line", "line"] = F("file", "How .txt files are split into documents: whole file, blank-line separated paragraphs, or one per line.", "Source", True)
    max_docs: int | None = F(None, "Cap documents taken from this source.", "Source", ge=1)
    repeat: int = F(1, "Include this source's documents this many times (up-weighting; applied after dedup so duplicates are only repeated on purpose).", "Source", True, ge=1, le=100)


class ContaminationConfig(_Cfg):
    """n-gram overlap scan against evaluation sets (hook for the `contamination` eval suite)."""

    eval_files: list[str] = F([], "Evaluation text files (.jsonl with `text`/`prompt`/`question` fields, or .txt) whose n-grams must not appear in the data.", "Contamination")
    ngram: int = F(13, "Token n-gram length.", "Contamination", ge=3, le=64)
    action: Literal["report", "drop"] = F("report", "report = only count; drop = remove contaminated documents.", "Contamination")
    min_overlap: float = F(0.0, "A document is contaminated if at least this fraction of its n-grams hit (0 = any hit).", "Contamination", True, ge=0.0, le=1.0)


class IngestConfig(_Cfg):
    """Tokenise, filter, dedup and pack text into shards (job type `dataset_ingest`)."""

    name: str = F("dataset", "Dataset name.", "General")
    sources: list[SourceSpec] = FF(list, "Input sources (at least one).", "Sources")
    tokenizer: str = F("byte", "Tokenizer path / repo id; 'byte' = built-in byte-level tokenizer (tests, byte models).", "Tokenizer")
    add_bos: bool = F(False, "Prepend BOS to every document.", "Tokenizer")
    add_eos: bool = F(True, "Append EOS to every document (document separator in packed data).", "Tokenizer")
    dtype: Literal["auto", "uint16", "uint32"] = F("auto", "Token storage dtype. auto = uint16 when vocab <= 65536.", "Packing")
    shard_size_mib: float = F(256.0, "Target shard size.", "Packing", gt=0.0, unit="MiB")
    min_doc_tokens: int = F(1, "Drop documents shorter than this (after tokenisation, excluding BOS/EOS).", "Filtering", ge=0)
    max_doc_tokens: int = F(0, "Truncate documents to this many tokens (0 = unlimited; documents larger than a shard are always truncated).", "Filtering", True, ge=0)
    normalise_unicode: bool = F(True, "Apply NFC normalisation and strip NULs before hashing/tokenising.", "Filtering", True)
    strip_whitespace: bool = F(True, "Strip leading/trailing whitespace.", "Filtering", True)
    dedup_exact: bool = F(True, "Drop exact duplicate documents (hash of normalised text).", "Dedup")
    dedup_minhash: bool = F(False, "Drop near-duplicates with MinHash-LSH (word n-gram shingles).", "Dedup")
    minhash_num_perm: int = F(64, "MinHash permutations.", "Dedup", True, ge=8, le=1024)
    minhash_ngram: int = F(5, "Word n-gram shingle size.", "Dedup", True, ge=1, le=16)
    minhash_threshold: float = F(0.8, "Estimated Jaccard similarity above which a document is a near-duplicate.", "Dedup", True, gt=0.0, lt=1.0)
    val_fraction: float = F(0.0, "Fraction of documents (selected by content hash, so stable) written to separate val shards.", "Packing", ge=0.0, le=0.5)
    max_total_tokens: int | None = F(None, "Stop after this many tokens.", "Filtering", True, ge=1)
    contamination: ContaminationConfig = FF(ContaminationConfig, "Contamination scan against evaluation sets.", "Contamination", True)
    output_dir: str | None = F(None, "Output directory (default: <datasets_dir>/<name>).", "Output", True)
    licence: str | None = F(None, "Licence of the data (recorded in the manifest).", "Output")
    filtering_version: str = F("forge-ingest-1", "Version label of the filtering pipeline (recorded in the manifest).", "Output", True)
    register_in_store: bool = F(True, "Store the manifest in the artifact store when the context provides one.", "Output", True)
    seed: int = F(0, "Seed of the MinHash permutations.", "Dedup", True)
    log_every_docs: int = F(10000, "Progress log interval.", "Output", True, ge=1)


# ---------------------------------------------------------------------------------------------------------------
# source readers
# ---------------------------------------------------------------------------------------------------------------
def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if path.suffix == ".zst":
        import zstandard

        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _dotted(rec: Any, path: str) -> Any:
    cur = rec
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _guess_kind(p: Path) -> str:
    if p.is_dir():
        return "dir"
    name = p.name.lower()
    for suf in (".gz", ".zst"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    if name.endswith((".jsonl", ".json", ".ndjson")):
        return "jsonl"
    if name.endswith(".parquet"):
        return "parquet"
    return "txt"


def _iter_jsonl(path: Path, field: str, stats: dict) -> Iterator[str]:
    with _open_text(path) as f:
        text = None
        first = f.read(1)
        if first == "[":  # a JSON array file
            text = first + f.read()
            try:
                for rec in json.loads(text):
                    v = _dotted(rec, field) if isinstance(rec, dict) else None
                    if isinstance(v, str):
                        yield v
                    else:
                        stats["bad_records"] += 1
            except json.JSONDecodeError:
                stats["bad_records"] += 1
            return
        buf = first
        for line in f:
            line = (buf + line).strip()
            buf = ""
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_records"] += 1
                continue
            v = _dotted(rec, field) if isinstance(rec, dict) else (rec if isinstance(rec, str) else None)
            if isinstance(v, str):
                yield v
            else:
                stats["bad_records"] += 1


def _iter_txt(path: Path, mode: str) -> Iterator[str]:
    with _open_text(path) as f:
        data = f.read()
    if mode == "file":
        yield data
    elif mode == "line":
        for ln in data.splitlines():
            if ln.strip():
                yield ln
    else:
        for para in re.split(r"\n\s*\n", data):
            if para.strip():
                yield para


def _iter_parquet(path: Path, field: str, stats: dict) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # noqa: BLE001
        raise ConfigError("reading parquet needs pyarrow (pip install pyarrow)") from e
    pf = pq.ParquetFile(str(path))
    top = field.split(".")[0]
    for batch in pf.iter_batches(columns=[top]):
        for v in batch.column(0).to_pylist():
            if "." in field:
                v = _dotted({top: v}, field)
            if isinstance(v, str):
                yield v
            else:
                stats["bad_records"] += 1


def _iter_hf(spec: SourceSpec, stats: dict) -> Iterator[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ConfigError("Hugging Face sources need the `datasets` package (pip install datasets)") from e
    ds = load_dataset(spec.path, spec.hf_config, split=spec.split, streaming=spec.streaming)
    for rec in ds:
        v = _dotted(rec, spec.text_field)
        if isinstance(v, str):
            yield v
        else:
            stats["bad_records"] += 1


def iter_source_texts(spec: SourceSpec, stats: dict | None = None) -> Iterator[str]:
    """Yield documents of one source (bad records are counted in ``stats['bad_records']``)."""
    stats = stats if stats is not None else {"bad_records": 0}
    stats.setdefault("bad_records", 0)
    kind = spec.kind
    if kind == "hf":
        yield from _iter_hf(spec, stats)
        return
    p = Path(spec.path)
    if not p.exists():
        raise NotFound(f"source path not found: {spec.path}")
    if kind == "auto":
        kind = _guess_kind(p)
    if kind == "dir":
        exts = (".jsonl", ".json", ".ndjson", ".parquet", ".txt", ".md", ".gz", ".zst")
        for f in sorted(x for x in p.rglob("*") if x.is_file() and x.name.lower().endswith(exts)):
            sub = spec.model_copy(update={"kind": "auto", "path": str(f)})
            yield from iter_source_texts(sub, stats)
        return
    if kind == "jsonl":
        yield from _iter_jsonl(p, spec.text_field, stats)
    elif kind == "parquet":
        yield from _iter_parquet(p, spec.text_field, stats)
    else:
        yield from _iter_txt(p, spec.txt_split)


# ---------------------------------------------------------------------------------------------------------------
# near-duplicate detection (MinHash + LSH)
# ---------------------------------------------------------------------------------------------------------------
_WORD = re.compile(r"\w+", re.UNICODE)
_PRIME = np.uint64(4294967311)  # prime > 2^32


def _optimal_bands(num_perm: int, threshold: float, fp_weight: float = 0.5) -> tuple[int, int]:
    """(bands, rows) minimising weighted false-positive / false-negative probability mass."""
    xs = np.linspace(0.001, 0.999, 200)
    best, best_err = (1, num_perm), 1e9
    for r in range(1, num_perm + 1):
        if num_perm % r:
            continue
        b = num_perm // r
        p = 1.0 - (1.0 - xs ** r) ** b
        fp = float(np.sum(p[xs < threshold]) * (xs[1] - xs[0]))
        fn = float(np.sum((1.0 - p)[xs >= threshold]) * (xs[1] - xs[0]))
        err = fp_weight * fp + (1.0 - fp_weight) * fn
        if err < best_err:
            best, best_err = (b, r), err
    return best


class MinHashDeduper:
    """MinHash-LSH near-duplicate filter over word n-gram shingles (pure numpy, no external dependency)."""

    def __init__(self, num_perm: int = 64, ngram: int = 5, threshold: float = 0.8, seed: int = 0) -> None:
        self.num_perm, self.ngram, self.threshold = int(num_perm), int(ngram), float(threshold)
        rng = np.random.default_rng(int(seed) + 12345)
        self.a = rng.integers(1, 2**31 - 1, size=self.num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 2**31 - 1, size=self.num_perm, dtype=np.uint64)
        self.bands, self.rows = _optimal_bands(self.num_perm, self.threshold)
        self.tables: list[dict[bytes, int]] = [dict() for _ in range(self.bands)]
        self.n_seen = 0
        self.n_dup = 0

    def signature(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        if len(words) < self.ngram:
            shingles = [" ".join(words) if words else text]
        else:
            shingles = [" ".join(words[i:i + self.ngram]) for i in range(len(words) - self.ngram + 1)]
        h = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in shingles), dtype=np.uint64, count=len(shingles))
        # (a*h + b) mod p, minimum over shingles, for each permutation
        vals = (self.a[:, None] * h[None, :] + self.b[:, None]) % _PRIME
        return vals.min(axis=1)

    def is_duplicate(self, text: str, add: bool = True) -> bool:
        sig = self.signature(text)
        keys = [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]
        dup = any(k in self.tables[i] for i, k in enumerate(keys))
        self.n_seen += 1
        if dup:
            self.n_dup += 1
        elif add:
            for i, k in enumerate(keys):
                self.tables[i][k] = self.n_seen
        return dup

    @staticmethod
    def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


# ---------------------------------------------------------------------------------------------------------------
# contamination
# ---------------------------------------------------------------------------------------------------------------
class ContaminationScanner:
    """Token n-gram overlap against a set of evaluation texts."""

    def __init__(self, eval_token_seqs: Iterable[list[int]], ngram: int = 13, min_overlap: float = 0.0) -> None:
        self.ngram, self.min_overlap = int(ngram), float(min_overlap)
        self.grams: set[tuple[int, ...]] = set()
        n = self.ngram
        for seq in eval_token_seqs:
            for i in range(len(seq) - n + 1):
                self.grams.add(tuple(seq[i:i + n]))

    def check(self, toks: list[int]) -> tuple[int, float]:
        """(number of hit n-grams, hit fraction)."""
        n = self.ngram
        total = len(toks) - n + 1
        if total <= 0 or not self.grams:
            return 0, 0.0
        hits = sum(1 for i in range(total) if tuple(toks[i:i + n]) in self.grams)
        return hits, hits / total

    def is_contaminated(self, toks: list[int]) -> bool:
        hits, frac = self.check(toks)
        return hits > 0 and frac >= self.min_overlap

    @classmethod
    def from_files(cls, files: list[str], tokenizer: TokenizerAdapter, ngram: int = 13, min_overlap: float = 0.0) -> ContaminationScanner:
        def seqs() -> Iterator[list[int]]:
            for fpath in files:
                p = Path(fpath)
                if not p.exists():
                    raise NotFound(f"contamination eval file not found: {fpath}")
                if p.suffix in (".jsonl", ".json"):
                    with open(p, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(rec, str):
                                yield tokenizer.encode(rec)
                                continue
                            parts = [str(rec[k]) for k in ("text", "prompt", "question", "context", "answer", "completion") if isinstance(rec, dict) and rec.get(k)]
                            if parts:
                                yield tokenizer.encode("\n".join(parts))
                else:
                    yield tokenizer.encode(p.read_text(encoding="utf-8", errors="replace"))

        return cls(seqs(), ngram, min_overlap)

    def scan_dataset(self, path: str | Path, max_docs: int | None = None) -> dict[str, Any]:
        """Post-hoc scan of an ingested dataset; returns ``{docs, contaminated_docs, hit_ngrams, examples:[(shard, doc)]}``."""
        ds = ShardedTokenDataset(path, seq_len=2)
        docs = bad = hits_total = 0
        examples: list[tuple[int, int]] = []
        for si, sh in enumerate(ds.shards):
            mm = ds._mm(si)
            offs = np.load(sh["offsets"]) if sh.get("offsets") else np.array([0, sh["tokens"]])
            for di in range(len(offs) - 1):
                toks = mm[offs[di]:offs[di + 1]].astype(np.int64).tolist()
                h, fr = self.check(toks)
                docs += 1
                if h > 0 and fr >= self.min_overlap:
                    bad += 1
                    hits_total += h
                    if len(examples) < 20:
                        examples.append((si, di))
                if max_docs and docs >= max_docs:
                    break
        return {"docs": docs, "contaminated_docs": bad, "hit_ngrams": hits_total, "examples": examples}


# ---------------------------------------------------------------------------------------------------------------
# shard writer + ingest job
# ---------------------------------------------------------------------------------------------------------------
class _ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, dtype: np.dtype, shard_tokens: int, vocab_size: int, split: str) -> None:
        self.out_dir, self.prefix, self.dtype = out_dir, prefix, np.dtype(dtype)
        self.shard_tokens, self.vocab_size, self.split = int(shard_tokens), int(vocab_size), split
        self.docs: list[np.ndarray] = []
        self.n_tok = 0
        self.entries: list[ShardEntry] = []
        self.truncated_docs = 0

    def add(self, toks: np.ndarray) -> None:
        if len(toks) > self.shard_tokens:
            toks = toks[: self.shard_tokens]
            self.truncated_docs += 1
        if self.n_tok + len(toks) > self.shard_tokens and self.docs:
            self.flush()
        self.docs.append(toks)
        self.n_tok += len(toks)

    def flush(self) -> None:
        if not self.docs:
            return
        idx = len(self.entries)
        base = f"{self.prefix}_{idx:05d}"
        lens = np.array([len(d) for d in self.docs], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        arr = np.concatenate(self.docs).astype(self.dtype)
        bin_path, off_path = self.out_dir / f"{base}.bin", self.out_dir / f"{base}.offsets.npy"
        arr.tofile(bin_path)
        np.save(off_path, offsets)
        counts = np.bincount(arr.astype(np.int64), minlength=max(self.vocab_size, 1))
        p = counts[counts > 0] / max(1, int(counts.sum()))
        stats = {
            "tokens": int(arr.size), "docs": int(len(lens)), "doc_len_mean": float(lens.mean()), "doc_len_min": int(lens.min()),
            "doc_len_max": int(lens.max()), "unique_tokens": int((counts > 0).sum()),
            "unigram_entropy_bits": float(-(p * np.log2(p)).sum()), "top_token_frac": float(p.max()),
        }
        (self.out_dir / f"{base}.stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        self.entries.append(ShardEntry(
            hash=hash_file(bin_path), tokens=int(arr.size), docs=int(len(lens)), path=bin_path.name, offsets=off_path.name,
            offsets_hash=hash_file(off_path), dtype=str(self.dtype), split=self.split, stats=stats))
        self.docs, self.n_tok = [], 0


def _norm_text(t: str, cfg: IngestConfig) -> str:
    if cfg.normalise_unicode:
        t = unicodedata.normalize("NFC", t).replace("\x00", "")
    if cfg.strip_whitespace:
        t = t.strip()
    return t


def resolve_dataset_path(ref: str | Path) -> Path:
    """Resolve a dataset reference (dir, manifest.json path, or dataset id/name under the datasets dir) to its directory."""
    p = Path(str(ref))
    if p.is_file() and p.name.endswith(".json"):
        return p.parent
    if p.is_dir() and (p / "manifest.json").exists():
        return p
    from forge.core.paths import datasets_dir

    root = datasets_dir()
    for cand in (root / str(ref), root / str(ref).replace(":", "_")):
        if (cand / "manifest.json").exists():
            return cand
    for sub in sorted(root.iterdir()) if root.exists() else []:
        mf = sub / "manifest.json"
        if mf.exists():
            try:
                d = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if d.get("id") == str(ref) or d.get("name") == str(ref):
                return sub
    raise NotFound(f"dataset not found: {ref}")


def load_dataset_manifest(path: str | Path) -> tuple[DatasetManifest, dict[str, Any], Path]:
    """(manifest, dataset_info extras, directory)."""
    d = resolve_dataset_path(path)
    m = DatasetManifest.model_validate(json.loads((d / "manifest.json").read_text(encoding="utf-8")))
    info_p = d / "dataset_info.json"
    info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else {}
    return m, info, d


def _ctx_log(ctx: Any, msg: str, level: str = "info") -> None:
    if ctx is not None:
        ctx.log(msg, level)
    else:
        log.log(getattr(logging, level.upper(), logging.INFO), msg)


def ingest_dataset(cfg: IngestConfig | dict, ctx: Any = None) -> dict[str, Any]:
    """Job entry point (``dataset_ingest``). Returns a summary dict (also stored via ``ctx.set_output``)."""
    if not isinstance(cfg, IngestConfig):
        cfg = IngestConfig.model_validate(cfg)
    if not cfg.sources:
        raise ConfigError("ingest: at least one source is required")
    from forge.core.paths import datasets_dir

    tok = load_tokenizer_any(cfg.tokenizer)
    vocab = tok.vocab_size
    dtype = np.uint16 if (cfg.dtype == "uint16" or (cfg.dtype == "auto" and vocab <= 65536)) else np.uint32
    if dtype == np.uint16 and vocab > 65536:
        raise ConfigError(f"vocab size {vocab} does not fit uint16")
    out_dir = Path(cfg.output_dir) if cfg.output_dir else datasets_dir() / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_tokens = max(1024, int(cfg.shard_size_mib * 1024 * 1024 / np.dtype(dtype).itemsize))

    contam = None
    if cfg.contamination.eval_files:
        contam = ContaminationScanner.from_files(cfg.contamination.eval_files, tok, cfg.contamination.ngram, cfg.contamination.min_overlap)
        _ctx_log(ctx, f"contamination scanner: {len(contam.grams)} {cfg.contamination.ngram}-grams from {len(cfg.contamination.eval_files)} file(s)")

    mh = MinHashDeduper(cfg.minhash_num_perm, cfg.minhash_ngram, cfg.minhash_threshold, cfg.seed) if cfg.dedup_minhash else None
    seen: set[bytes] = set()
    writers = {
        "train": _ShardWriter(out_dir, "shard", dtype, shard_tokens, vocab, "train"),
        "val": _ShardWriter(out_dir, "val_shard", dtype, shard_tokens, vocab, "val"),
    }
    rep: dict[str, Any] = {
        "docs_in": 0, "empty": 0, "exact_dup": 0, "minhash_dup": 0, "too_short": 0, "contaminated": 0, "contaminated_dropped": 0,
        "contaminated_hit_ngrams": 0, "kept": 0, "bad_records": 0, "truncated_docs": 0,
    }
    mixture: dict[str, Any] = {}
    total_tokens = 0
    done = False
    val_cut = int(cfg.val_fraction * 10000)

    for si, spec in enumerate(cfg.sources):
        label = spec.name or Path(spec.path).name or f"source{si}"
        mix = mixture.setdefault(label, {"docs": 0, "tokens": 0, "repeat": spec.repeat})
        stats = {"bad_records": 0}
        taken = 0
        for raw in iter_source_texts(spec, stats):
            if spec.max_docs and taken >= spec.max_docs:
                break
            rep["docs_in"] += 1
            text = _norm_text(raw, cfg)
            if not text:
                rep["empty"] += 1
                continue
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            if cfg.dedup_exact:
                if digest in seen:
                    rep["exact_dup"] += 1
                    continue
                seen.add(digest)
            if mh is not None and mh.is_duplicate(text):
                rep["minhash_dup"] += 1
                continue
            ids = tok.encode(text)
            if len(ids) < cfg.min_doc_tokens or not ids:
                rep["too_short"] += 1
                continue
            if cfg.max_doc_tokens and len(ids) > cfg.max_doc_tokens:
                ids = ids[: cfg.max_doc_tokens]
            if contam is not None:
                hits, frac = contam.check(ids)
                if hits > 0 and frac >= contam.min_overlap:
                    rep["contaminated"] += 1
                    rep["contaminated_hit_ngrams"] += hits
                    if cfg.contamination.action == "drop":
                        rep["contaminated_dropped"] += 1
                        continue
            if cfg.add_bos and tok.bos_id is not None:
                ids = [tok.bos_id] + ids
            if cfg.add_eos and tok.eos_id is not None:
                ids = ids + [tok.eos_id]
            arr = np.asarray(ids, dtype=np.int64)
            if arr.max() >= (1 << (8 * np.dtype(dtype).itemsize)):
                raise ConfigError("token id exceeds storage dtype range")
            split = "val" if (val_cut and int.from_bytes(digest[:4], "little") % 10000 < val_cut) else "train"
            for _ in range(spec.repeat if split == "train" else 1):
                writers[split].add(arr)
            taken += 1
            rep["kept"] += 1
            mix["docs"] += 1
            mix["tokens"] += len(ids)
            total_tokens += len(ids)
            if rep["docs_in"] % cfg.log_every_docs == 0:
                _ctx_log(ctx, f"ingest: {rep['docs_in']} docs read, {rep['kept']} kept, {total_tokens} tokens")
                if ctx is not None:
                    ctx.progress(min(0.95, si / max(1, len(cfg.sources))), f"{rep['kept']} docs")
                    if ctx.should_stop():
                        _ctx_log(ctx, "ingest cancelled", "warning")
                        done = True
                        break
            if cfg.max_total_tokens and total_tokens >= cfg.max_total_tokens:
                done = True
                break
        rep["bad_records"] += stats["bad_records"]
        if done:
            break

    for w in writers.values():
        w.flush()
        rep["truncated_docs"] += w.truncated_docs
    shards = writers["train"].entries + writers["val"].entries
    if not writers["train"].entries and not writers["val"].entries:
        raise ConfigError("ingest produced no tokens (all documents filtered?)", details=rep)

    manifest = DatasetManifest(
        name=cfg.name, tokenizer_hash=tok.hash, shards=shards, mixture=mixture, filtering_version=cfg.filtering_version,
        licence=cfg.licence, dedup_report=rep, total_tokens=sum(s.tokens for s in shards))
    manifest.id = "ds_" + manifest.content_hash().split(":")[-1][:12]
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    info = {
        "tokenizer": cfg.tokenizer, "tokenizer_hash": tok.hash, "vocab_size": vocab, "dtype": str(np.dtype(dtype)),
        "eos_id": tok.eos_id, "bos_id": tok.bos_id, "pad_id": tok.pad_id, "add_bos": cfg.add_bos, "add_eos": cfg.add_eos,
        "contamination": {"enabled": contam is not None, "ngram": cfg.contamination.ngram, "action": cfg.contamination.action, **{k: rep[k] for k in ("contaminated", "contaminated_dropped", "contaminated_hit_ngrams")}},
        "train_tokens": sum(s.tokens for s in writers["train"].entries), "val_tokens": sum(s.tokens for s in writers["val"].entries),
        "config": cfg.model_dump(mode="json"),
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    mhash = None
    store = getattr(ctx, "store", None) if ctx is not None else None
    if store is not None and cfg.register_in_store:
        try:
            mhash = store.put_bytes(manifest.model_dump_json().encode("utf-8"), kind="dataset_manifest")
        except Exception as e:  # noqa: BLE001
            _ctx_log(ctx, f"could not store dataset manifest: {e!r}", "warning")
    summary = {"dataset_id": manifest.id, "dataset_dir": str(out_dir), "manifest_hash": mhash, "total_tokens": manifest.total_tokens,
               "shards": len(shards), "report": rep, "dtype": info["dtype"], "vocab_size": vocab}
    if ctx is not None:
        for k, v in summary.items():
            ctx.set_output(k, v)
        ctx.progress(1.0, f"ingested {manifest.total_tokens} tokens")
    _ctx_log(ctx, f"ingest done: {manifest.total_tokens} tokens in {len(shards)} shard(s) -> {out_dir}")
    return summary


# ---------------------------------------------------------------------------------------------------------------
# sharded dataset
# ---------------------------------------------------------------------------------------------------------------
class ShardedTokenDataset:
    """Fixed-length sequences over packed token shards with a stateless deterministic batch order.

    ``paths``: dataset dir(s)/manifest(s) (concatenated).  Sequence ``i`` is the token window ``[i*L, i*L+L]`` of its
    shard (``L+1`` tokens: inputs are the first L, aligned labels the last L); windows never cross shard boundaries.

    ``role='train'|'val'``: if the data has val-split shards, ``val`` reads those; otherwise ``val_sequences`` sequences are
    held out by a fixed (seed-independent) permutation derived from the manifest hash and excluded from ``train``.
    """

    def __init__(
        self, paths: str | Path | list[str | Path], seq_len: int, *, seed: int = 0, shuffle: bool = True, role: str = "train",
        val_sequences: int = 0, max_sequences: int | None = None,
    ) -> None:
        plist = paths if isinstance(paths, (list, tuple)) else [paths]
        if not plist:
            raise ConfigError("no dataset paths given")
        self.seq_len, self.seed, self.shuffle, self.role = int(seq_len), int(seed), bool(shuffle), role
        self.shards: list[dict[str, Any]] = []
        self.manifests: list[DatasetManifest] = []
        self.infos: list[dict[str, Any]] = []
        has_val = False
        for p in plist:
            m, info, d = load_dataset_manifest(p)
            self.manifests.append(m)
            self.infos.append(info)
            for s in m.shards:
                ex = s.model_extra or {}
                split = ex.get("split", "train")
                has_val = has_val or split == "val"
                self.shards.append({
                    "bin": d / ex.get("path", f"{s.hash}.bin"), "offsets": (d / ex["offsets"]) if ex.get("offsets") else None,
                    "tokens": int(s.tokens), "docs": int(s.docs), "dtype": ex.get("dtype", info.get("dtype", "uint16")),
                    "split": split, "hash": s.hash,
                })
        self.fingerprint = hash_json([m.content_hash() for m in self.manifests])
        self._salt = int(self.fingerprint.split(":")[-1][:8], 16)
        want_split = "val" if (role == "val" and has_val) else "train"
        if role == "train" and has_val:
            want_split = "train"
        sel = [i for i, s in enumerate(self.shards) if s["split"] == want_split]
        if role == "val" and not has_val and val_sequences <= 0:
            sel = []
        self._sel = sel
        self._mms: dict[int, np.memmap] = {}
        self._lock = threading.Lock()
        counts = [max(0, (self.shards[i]["tokens"] - 1) // self.seq_len) for i in sel]
        self._cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        total = int(self._cum[-1])
        ids = np.arange(total, dtype=np.int64)
        if max_sequences is not None:
            ids = ids[: int(max_sequences)]
        self.holdout_mode = (not has_val) and val_sequences > 0
        if self.holdout_mode:
            hold = np.random.default_rng(np.random.SeedSequence([self._salt, 777])).permutation(len(ids))[: int(val_sequences)]
            mask = np.zeros(len(ids), dtype=bool)
            mask[hold] = True
            ids = np.sort(ids[mask]) if role == "val" else ids[~mask]
        elif role == "val" and not has_val:
            ids = ids[:0]
        self.ids = ids
        self._perms = _PermCache(len(ids), self.seed, self._salt, self.shuffle)

    # -- basics --------------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self.ids))

    @property
    def tokens_per_epoch(self) -> int:
        return len(self) * self.seq_len

    @property
    def vocab_size(self) -> int:
        return int(self.infos[0].get("vocab_size", 0)) if self.infos else 0

    @property
    def eos_id(self) -> int | None:
        return self.infos[0].get("eos_id") if self.infos else None

    def _mm(self, shard_idx: int) -> np.memmap:
        with self._lock:
            mm = self._mms.get(shard_idx)
            if mm is None:
                s = self.shards[shard_idx]
                mm = np.memmap(s["bin"], dtype=np.dtype(s["dtype"]), mode="r")
                self._mms[shard_idx] = mm
            return mm

    def window(self, base_id: int) -> np.ndarray:
        """``seq_len + 1`` token ids of the sequence with global (base) id ``base_id``."""
        k = int(np.searchsorted(self._cum, base_id, side="right") - 1)
        local = int(base_id - self._cum[k])
        mm = self._mm(self._sel[k])
        st = local * self.seq_len
        return np.asarray(mm[st:st + self.seq_len + 1], dtype=np.int64)

    def get_batch(self, positions: np.ndarray) -> dict[str, Any]:
        """Batch for dataset *positions* (indices into ``self.ids``)."""
        import torch

        base = self.ids[np.asarray(positions, dtype=np.int64)]
        arr = np.stack([self.window(int(b)) for b in base])
        t = torch.from_numpy(arr)
        return {"input_ids": t[:, :-1].contiguous(), "labels": t[:, 1:].contiguous(), "seq_ids": torch.from_numpy(base),
                "positions": torch.from_numpy(np.asarray(positions, dtype=np.int64).copy())}

    def batch_at(self, step: int, global_batch_size: int, rank: int = 0, world: int = 1) -> dict[str, Any]:
        """Rank's slice of the global batch of ``step`` -- a pure function of (manifest, seed, step)."""
        if global_batch_size % world:
            raise ConfigError("global batch size must be divisible by world size")
        pos = step_positions(self._perms, step, global_batch_size)
        per = global_batch_size // world
        return self.get_batch(pos[rank * per:(rank + 1) * per])

    def positions_at(self, step: int, global_batch_size: int) -> np.ndarray:
        return step_positions(self._perms, step, global_batch_size)

    def restrict(self, n: int) -> None:
        """Keep only the first ``n`` positions (e.g. the part of the data covered by a teacher-logit cache)."""
        self.ids = self.ids[: int(n)]
        self._perms = _PermCache(len(self.ids), self.seed, self._salt, self.shuffle)

    def eval_batches(self, micro_batch: int, n_batches: int) -> list[dict[str, Any]]:
        """A fixed, order-stable list of held-out micro-batches (first ``n_batches*micro_batch`` positions, no shuffle)."""
        total = min(len(self), micro_batch * n_batches)
        out = []
        for i in range(0, total, micro_batch):
            pos = np.arange(i, min(i + micro_batch, total))
            out.append(self.get_batch(pos))
        return out

    def describe(self) -> dict[str, Any]:
        return {"sequences": len(self), "seq_len": self.seq_len, "tokens_per_epoch": self.tokens_per_epoch, "shards": len(self._sel),
                "fingerprint": self.fingerprint, "role": self.role, "holdout": self.holdout_mode}
