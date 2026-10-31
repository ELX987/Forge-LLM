"""``contamination`` suite (EV-2 hygiene): n-gram overlap scan between evaluation items and training/reference text.

Method (GPT-3 / Llama style): hash every ``n``-gram (default 13 words, lower-cased, punctuation stripped) of the reference
corpus into a sorted 64-bit array, then report for each evaluation item the fraction of its n-grams found in the reference.
Items with overlap >= ``flag_threshold`` are flagged.  Hashing is vectorised (polynomial rolling hash on uint64 with natural
wrap-around), so multi-GB reference corpora are feasible; collisions are possible but astronomically unlikely at 64 bits
for the corpus sizes involved (flagged items should still be inspected manually).

Programmatic API (used by data ingestion for the contamination hook)::

    index = NgramIndex(n=13); index.add_texts(train_texts)
    report = scan_overlap(eval_texts, index)
"""
from __future__ import annotations

import re
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from pydantic import Field

from forge.core.registry import register
from forge.eval import stats
from forge.eval.suites.base import Suite, SuiteConfig, SuiteEnv, SuiteResult, read_jsonl, resolve_data_path

_WORD = re.compile(r"[a-z0-9']+")
_BASE = np.uint64(1099511628211)  # FNV prime as polynomial base


def word_ids(text: str) -> np.ndarray:
    """Normalised words -> stable uint64 ids (crc32)."""
    return np.fromiter((zlib.crc32(w.encode()) for w in _WORD.findall(text.lower())), dtype=np.uint64)


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Rolling hash of every length-``n`` window of ``ids`` (uint64, wrapping arithmetic)."""
    L = ids.size
    if L < n:
        return np.zeros(0, dtype=np.uint64)
    ids = ids.astype(np.uint64, copy=False)
    h = np.zeros(L - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * _BASE + ids[i : L - n + 1 + i] + np.uint64(i + 1)
    return h


class NgramIndex:
    """Sorted array of reference n-gram hashes."""

    def __init__(self, n: int = 13, mode: str = "word") -> None:
        self.n = n
        self.mode = mode
        self._parts: list[np.ndarray] = []
        self._sorted: np.ndarray | None = None
        self.docs = 0
        self.tokens = 0

    def add_ids(self, ids: np.ndarray) -> None:
        self.docs += 1
        self.tokens += int(ids.size)
        h = ngram_hashes(np.asarray(ids), self.n)
        if h.size:
            self._parts.append(np.unique(h))
            if len(self._parts) > 256:
                self._compact()
        self._sorted = None

    def add_text(self, text: str) -> None:
        self.add_ids(word_ids(text))

    def add_texts(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add_text(t)

    def _compact(self) -> None:
        self._parts = [np.unique(np.concatenate(self._parts))] if self._parts else []

    def finalize(self) -> np.ndarray:
        if self._sorted is None:
            self._sorted = np.unique(np.concatenate(self._parts)) if self._parts else np.zeros(0, dtype=np.uint64)
        return self._sorted

    def contains(self, hashes: np.ndarray) -> np.ndarray:
        ref = self.finalize()
        if ref.size == 0 or hashes.size == 0:
            return np.zeros(hashes.size, dtype=bool)
        pos = np.searchsorted(ref, hashes)
        pos = np.minimum(pos, ref.size - 1)
        return ref[pos] == hashes

    def __len__(self) -> int:
        return int(self.finalize().size)


@dataclass
class OverlapReport:
    n: int
    items: int
    too_short: int
    flagged: int
    contamination_rate: float
    mean_overlap: float
    per_item: list[float] = field(default_factory=list)
    flagged_ids: list[Any] = field(default_factory=list)
    reference_ngrams: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_item"] = d["per_item"][:5000]
        return d


def scan_overlap(
    eval_texts: Sequence[str],
    index: NgramIndex,
    *,
    flag_threshold: float = 0.2,
    ids: Sequence[Any] | None = None,
    min_words: int | None = None,
) -> OverlapReport:
    """Overlap of each evaluation text with the reference index.

    ``overlap`` = fraction of the item's n-grams present in the reference; items with fewer than ``n`` words are
    counted in ``too_short`` and excluded."""
    n = index.n
    per: list[float] = []
    flagged_ids: list[Any] = []
    short = 0
    for i, t in enumerate(eval_texts):
        w = word_ids(t)
        if w.size < max(n, min_words or n):
            short += 1
            continue
        h = ngram_hashes(w, n)
        frac = float(index.contains(h).mean())
        per.append(frac)
        if frac >= flag_threshold:
            flagged_ids.append(ids[i] if ids is not None else i)
    k = len(per)
    return OverlapReport(
        n=n, items=k, too_short=short, flagged=len(flagged_ids), contamination_rate=(len(flagged_ids) / k if k else float("nan")),
        mean_overlap=float(np.mean(per)) if per else float("nan"), per_item=per, flagged_ids=flagged_ids[:500], reference_ngrams=len(index),
    )


def item_text(row: dict[str, Any]) -> str:
    """Text of a jsonl eval item: text | prompt | question(+context+choices)."""
    if "text" in row:
        return str(row["text"])
    parts = [str(row.get("context", "")), str(row.get("question", row.get("prompt", "")))]
    if "choices" in row:
        parts += [str(c) for c in row["choices"]]
    return " ".join(p for p in parts if p)


def iter_reference_texts(paths: Sequence[str]) -> Iterator[str]:
    """Yield documents from .txt/.md (split on blank lines), .jsonl ({text}), and directories of those."""
    for p in paths:
        path = Path(p).expanduser()
        files = [f for f in sorted(path.rglob("*")) if f.is_file()] if path.is_dir() else [path]
        for f in files:
            suf = f.suffix.lower()
            if suf in (".txt", ".md"):
                for block in f.read_text(encoding="utf-8", errors="ignore").split("\n\n"):
                    if block.strip():
                        yield block
            elif suf == ".jsonl":
                for row in read_jsonl(f):
                    yield item_text(row)


class ContaminationConfig(SuiteConfig):
    """N-gram overlap scan of evaluation sets against reference (training) text."""

    eval_paths: list[str] = Field(default_factory=lambda: ["bundled:mcq_smoke.jsonl"], description="jsonl evaluation sets to check (text | prompt | question+choices).")
    reference_paths: list[str] = Field(default_factory=list, description="Files / directories (.txt, .md, .jsonl) holding the training or reference corpus. Required for a meaningful scan.")
    n: int = Field(default=13, ge=3, le=64, description="n-gram length in words (13 is the common convention).")
    flag_threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Item is flagged when this fraction of its n-grams occurs in the reference.")


@register("suite", "contamination", meta={"requirement": "EV-2"})
class ContaminationSuite(Suite):
    """Which evaluation items also occur in the reference corpus (word n-gram overlap)."""

    name = "contamination"
    version = "1"
    title = "Contamination scan (n-gram overlap)"
    Config = ContaminationConfig
    needs_candidate_model = False
    describes = "Hygiene check before trusting task-suite scores; no model required."

    def run(self, cfg: ContaminationConfig, env: SuiteEnv) -> SuiteResult:  # type: ignore[override]
        res = SuiteResult(self.name, self.version)
        if not cfg.reference_paths:
            res.notes.append("no reference_paths given: nothing to compare against, suite skipped")
            res.flags["contamination_scanned"] = False
            return res
        index = NgramIndex(cfg.n)
        index.add_texts(iter_reference_texts(cfg.reference_paths))
        texts: list[str] = []
        ids: list[Any] = []
        for p in cfg.eval_paths:
            path = resolve_data_path(p)
            for j, row in enumerate(read_jsonl(path)):  # type: ignore[arg-type]
                texts.append(item_text(row))
                ids.append(f"{Path(str(path)).name}:{row.get('id', j)}")
                if cfg.max_items and len(texts) >= cfg.max_items:
                    break
        rep = scan_overlap(texts, index, flag_threshold=cfg.flag_threshold, ids=ids)
        res.flags["contamination_scanned"] = True
        if rep.items:
            k = rep.flagged
            p, lo, hi = stats.binomial_ci(k, rep.items, env.alpha)
            res.add("contamination_rate", p, ci=(lo, hi), n=rep.items)
            res.add("contamination_mean_overlap", rep.mean_overlap, n=rep.items)
        res.details.update({k: v for k, v in rep.to_dict().items() if k != "per_item"})
        if rep.too_short:
            res.notes.append(f"{rep.too_short} items shorter than {cfg.n} words were not scanned")
        return res
