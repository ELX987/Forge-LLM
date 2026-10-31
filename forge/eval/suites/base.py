"""Suite plumbing: config base, runtime environment, results, tokenizers, scoring helpers, jsonl IO."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from forge.core.errors import ConfigError
from forge.core.schemas import MetricValue, Provenance

if TYPE_CHECKING:  # pragma: no cover
    from forge.core.context import RunContext

log = logging.getLogger("forge.eval")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------------------------
# config base
# --------------------------------------------------------------------------------------------
class SuiteConfig(BaseModel):
    """Base class of every suite configuration (rendered as a form in the UI)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    max_items: int | None = Field(default=None, ge=1, description="Cap on the number of items evaluated (None = all).")


# --------------------------------------------------------------------------------------------
# tokenizers
# --------------------------------------------------------------------------------------------
@runtime_checkable
class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class ByteTokenizer:
    """Reversible byte-level tokenizer (ids 0..255 are bytes, 256 is EOS).  One character ~ one token for ASCII,
    which gives the synthetic generators exact length control; also the default for tiny test models."""

    vocab_size = 257
    eos_token_id = 256
    bos_token_id = None
    name = "bytes"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", errors="replace")


class HFTokenizerAdapter:
    """Adapter around a Hugging Face (or Forge-loaded) tokenizer: ``encode`` never adds special tokens."""

    def __init__(self, tok: Any) -> None:
        self.tok = tok
        self.eos_token_id = getattr(tok, "eos_token_id", None)
        self.bos_token_id = getattr(tok, "bos_token_id", None)
        self.vocab_size = getattr(tok, "vocab_size", None) or (len(tok) if hasattr(tok, "__len__") else None)
        self.name = getattr(tok, "name_or_path", "hf")

    def encode(self, text: str) -> list[int]:
        try:
            return list(self.tok.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tok.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        try:
            return self.tok.decode(list(ids), skip_special_tokens=True)
        except TypeError:
            return self.tok.decode(list(ids))


def adapt_tokenizer(tok: Any) -> Any:
    if tok is None:
        return ByteTokenizer()
    if isinstance(tok, (ByteTokenizer, HFTokenizerAdapter)):
        return tok
    return HFTokenizerAdapter(tok) if hasattr(tok, "encode") and hasattr(tok, "decode") else tok


# --------------------------------------------------------------------------------------------
# environment / results
# --------------------------------------------------------------------------------------------
@dataclass
class SuiteEnv:
    """Runtime inputs handed to a suite."""

    parent: Any = None  # LogitsProvider | None
    candidate: Any = None  # LogitsProvider | None
    tokenizer: Any = None
    seed: int = 0
    n_boot: int = 2000
    alpha: float = 0.05
    ctx: "RunContext | None" = None
    parent_serve: Any = None  # ServeClient | None
    candidate_serve: Any = None
    deployment_length: int | None = None
    model_info: dict[str, Any] = field(default_factory=dict)  # params_b, licence, ...
    batch_size_override: int | None = None  # used by noise-floor repeats to vary batch composition
    sub_progress: Callable[[float, str], None] | None = None

    def log(self, msg: str, level: str = "info") -> None:
        if self.ctx is not None:
            self.ctx.log(msg, level)
        else:
            log.log(getattr(logging, level.upper(), logging.INFO), msg)

    def progress(self, frac: float, msg: str = "") -> None:
        if self.sub_progress:
            self.sub_progress(frac, msg)

    def should_stop(self) -> bool:
        return bool(self.ctx is not None and self.ctx.should_stop())

    @property
    def tok(self) -> Any:
        if self.tokenizer is None:
            self.tokenizer = ByteTokenizer()
        return self.tokenizer


@dataclass
class SuiteResult:
    """What a suite returns.

    * ``metrics``  - named :class:`MetricValue` (with CI / n / provenance=measured).
    * ``items``    - paired per-item arrays for further pooling, e.g. ``{"score": (parent_arr, cand_arr)}``.
    * ``boot``     - bootstrap distributions (paired diff in metric units) used by gate non-inferiority tests.
    * ``scalars``  - raw point values by metric name (what noise repeats collect).
    * ``flags``    - boolean evidence flags (e.g. ``output_identical``).
    * ``generations`` - records stored compressed (EV-7).
    """

    suite: str
    version: str
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    items: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    boot: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    generations: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    contributes_to_suite_score: bool = False

    def add(self, name: str, value: float, *, ci: tuple[float, float] | None = None, n: int | None = None,
            provenance: Provenance = Provenance.measured) -> MetricValue:
        v = float(value)
        mv = MetricValue(value=v if math.isfinite(v) else 0.0, ci_low=None if ci is None or not all(map(math.isfinite, ci)) else float(ci[0]),
                         ci_high=None if ci is None or not all(map(math.isfinite, ci)) else float(ci[1]), n=n, provenance=provenance)
        if not math.isfinite(v):
            self.notes.append(f"metric {name} is not finite ({value}); stored as 0.0 with provenance 'untested'")
            mv = MetricValue(value=0.0, n=n, provenance=Provenance.untested)
        self.metrics[name] = mv
        self.scalars[name] = v
        return mv


class Suite:
    """Base class of evaluation suites.  Subclasses set ``name``, ``version``, ``Config`` and implement ``run``."""

    name: ClassVar[str] = "suite"
    version: ClassVar[str] = "1"
    title: ClassVar[str] = ""
    Config: ClassVar[type[SuiteConfig]] = SuiteConfig
    needs_candidate_model: ClassVar[bool] = True
    needs_parent_model: ClassVar[bool] = False  # True = meaningless without the parent
    describes: ClassVar[str] = ""

    def run(self, cfg: SuiteConfig, env: SuiteEnv) -> SuiteResult:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def make_config(cls, params: dict[str, Any] | None = None) -> SuiteConfig:
        try:
            return cls.Config(**(params or {}))
        except Exception as e:  # pydantic ValidationError
            raise ConfigError(f"invalid config for suite {cls.name!r}: {e}") from e


# --------------------------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------------------------
def resolve_data_path(path: str | None, bundled: str | None = None) -> Path | None:
    """``path`` if given (absolute / cwd-relative, or ``bundled:<name>``), else the bundled seed file."""
    if path:
        if path.startswith("bundled:"):
            return DATA_DIR / path.split(":", 1)[1]
        return Path(path).expanduser()
    return DATA_DIR / bundled if bundled else None


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ConfigError(f"{path}:{ln}: invalid JSON ({e})") from e
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


# --------------------------------------------------------------------------------------------
# log-likelihood scoring of continuations
# --------------------------------------------------------------------------------------------
def score_continuations(
    provider: Any,
    contexts: Sequence[Sequence[int]],
    continuations: Sequence[Sequence[int]],
    *,
    batch_size: int = 8,
    max_len: int | None = None,
    pad_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of log-probabilities of each continuation given its context.

    Returns ``(logprob_sum[N], n_tokens[N])``.  If ``context+continuation`` exceeds ``max_len`` the context is
    left-truncated (the continuation is always kept)."""
    import torch

    seqs: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for c, k in zip(contexts, continuations):
        c, k = list(c), list(k)
        if not k:
            raise ValueError("empty continuation")
        if max_len is not None and len(c) + len(k) > max_len:
            keep = max(1, max_len - len(k))
            c = c[-keep:]
        s = c + k
        seqs.append(s)
        spans.append((len(c), len(s)))
    out_lp = np.zeros(len(seqs))
    out_n = np.zeros(len(seqs))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # length-sorted batches waste less padding
    for b in range(0, len(order), batch_size):
        idxs = order[b : b + batch_size]
        T = max(len(seqs[i]) for i in idxs)
        ids = torch.full((len(idxs), T), pad_id, dtype=torch.long)
        msk = torch.zeros((len(idxs), T), dtype=torch.bool)
        for j, i in enumerate(idxs):
            ids[j, : len(seqs[i])] = torch.tensor(seqs[i])
            msk[j, : len(seqs[i])] = True
        lg = provider.logits(ids, msk)
        lp = torch.log_softmax(lg.double(), dim=-1)
        for j, i in enumerate(idxs):
            s0, s1 = spans[i]
            # token at position p is predicted by logits at p-1
            tgt = ids[j, s0:s1]
            pred = lp[j, s0 - 1 : s1 - 1] if s0 >= 1 else None
            if pred is None:
                raise ValueError("context must contain at least one token (add a BOS/space)")
            out_lp[i] = float(pred.gather(-1, tgt.unsqueeze(-1)).sum().item())
            out_n[i] = s1 - s0
    return out_lp, out_n


def paired_metric(
    res: SuiteResult,
    name: str,
    parent_items: np.ndarray | None,
    cand_items: np.ndarray,
    env: SuiteEnv,
    *,
    as_points: bool = False,
    prefix: str | None = None,
) -> None:
    """Add ``name`` (candidate mean + bootstrap CI) and, with a parent, ``name_parent``, ``name_diff`` (+ pts) and the
    bootstrap distribution under ``res.boot[name_diff]``."""
    from forge.eval import stats

    cand_items = np.asarray(cand_items, dtype=np.float64)
    n = int(cand_items.size)
    if n == 0:
        return
    est, lo, hi = stats.bootstrap_ci(cand_items, env.n_boot, alpha=env.alpha, seed=env.seed)
    res.add(name, est, ci=(lo, hi), n=n)
    if parent_items is None:
        return
    p = np.asarray(parent_items, dtype=np.float64)
    pe, plo, phi = stats.bootstrap_ci(p, env.n_boot, alpha=env.alpha, seed=env.seed + 1)
    res.add(f"{name}_parent", pe, ci=(plo, phi), n=n)
    pb = stats.paired_bootstrap(p, cand_items, env.n_boot, alpha=env.alpha, seed=env.seed + 2)
    scale = 100.0 if as_points else 1.0
    key = f"{name}_diff_pts" if as_points else f"{name}_diff"
    res.add(key, pb.estimate * scale, ci=(pb.ci_low * scale, pb.ci_high * scale), n=n)
    res.metrics[key].within_noise = None
    res.boot[key] = stats.bootstrap_distribution(p, cand_items, stat="mean_diff", n=env.n_boot, seed=env.seed + 2) * scale
    ratio = stats.paired_bootstrap(p, cand_items, env.n_boot, stat="ratio", alpha=env.alpha, seed=env.seed + 3)
    res.add(f"{name}_ratio", ratio.estimate, ci=(ratio.ci_low, ratio.ci_high), n=n)
    res.boot[f"{name}_ratio"] = stats.bootstrap_distribution(p, cand_items, stat="ratio", n=env.n_boot, seed=env.seed + 3)
    res.details.setdefault("p_values", {})[name] = pb.p_value
