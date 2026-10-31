"""Distribution-level metrics (Plan EV-1, 10.1 "Metric definitions").

All functions take **logits tensors** ``[..., V]`` (any float dtype; computed in float32/float64 internally,
chunked over rows so a 150k vocabulary does not blow memory) and an optional boolean ``mask`` over the leading
dims.  Model objects participate through the tiny :class:`LogitsProvider` protocol, so these metrics work for
``ForgeLM``, Hugging Face models or serving engines alike.

Definitions
-----------
* KL(parent || candidate) per position, nats, averaged over unmasked positions; ``p99`` is the 99th percentile
  over positions (the "tail" the gates use).
* top-1 flip rate: fraction of positions where ``argmax`` differs.
* ECE: expected calibration error with equal-width confidence bins.
* entropy shift: mean(H(candidate) - H(parent)) in nats.
* perplexity: ``exp(mean NLL)`` over the supplied targets.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

_ROW_CHUNK = 2048


def _torch():
    import torch  # lazy

    return torch


# --------------------------------------------------------------------------------------------
# LogitsProvider protocol + adapters
# --------------------------------------------------------------------------------------------
@runtime_checkable
class LogitsProvider(Protocol):
    """Anything that maps token ids to next-token logits.

    ``logits(input_ids)`` takes a ``LongTensor [B, T]`` (right-padded; ``attention_mask`` optional ``[B, T]`` of
    0/1) and returns float logits ``[B, T, V]`` where position ``t`` predicts token ``t+1``.
    Optional extras (used when present): ``generate(prompt_ids, max_new_tokens, ...)`` for suites that need
    free generation, ``nbytes()`` for the size axis of the Pareto plot, ``name``.
    """

    name: str

    def logits(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor | None" = None) -> "torch.Tensor": ...


def _extract_logits(out: Any) -> "torch.Tensor":
    torch = _torch()
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"cannot extract logits from {type(out)!r}")


class ModuleProvider:
    """Wrap a ``torch.nn.Module`` (ForgeLM, HF ``AutoModelForCausalLM``, ...) as a :class:`LogitsProvider`.

    The module is called as ``model(input_ids)`` (``attention_mask=`` is passed only if ``pass_mask=True``,
    since the Forge model takes positions rather than a mask; with right padding and causal attention the logits
    at real positions are unaffected by padding, so the default is safe for every decoder-only model)."""

    def __init__(
        self,
        model: Any,
        *,
        name: str = "model",
        device: str | None = None,
        pass_mask: bool = False,
        max_batch: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.name = name
        self.pass_mask = pass_mask
        self.max_batch = max_batch
        self.eos_token_id = eos_token_id
        if device is None:
            try:
                device = str(next(model.parameters()).device)
            except Exception:
                device = "cpu"
        self.device = device
        try:
            model.eval()
        except Exception:
            pass

    def logits(self, input_ids, attention_mask=None):
        torch = _torch()
        ids = input_ids.to(self.device)
        with torch.no_grad():
            if self.pass_mask and attention_mask is not None:
                out = self.model(input_ids=ids, attention_mask=attention_mask.to(self.device))
            else:
                out = self.model(ids)
            lg = _extract_logits(out)
        return lg.float().cpu() if lg.device.type != "cpu" else lg.float()

    def nbytes(self) -> int:
        total = 0
        seen: set[int] = set()
        for t in list(self.model.parameters()) + list(self.model.buffers()):
            if id(t) in seen:
                continue
            seen.add(id(t))
            total += t.numel() * t.element_size()
        return int(total)


class CallableProvider:
    """Wrap ``fn(input_ids[B,T]) -> logits[B,T,V]`` (useful for tests, serving engines, remote scorers)."""

    def __init__(self, fn: Callable[..., Any], *, name: str = "callable", nbytes: int | None = None, eos_token_id: int | None = None) -> None:
        self.fn = fn
        self.name = name
        self._nbytes = nbytes
        self.eos_token_id = eos_token_id

    def logits(self, input_ids, attention_mask=None):
        lg = _extract_logits(self.fn(input_ids))
        return lg.float()

    def nbytes(self) -> int | None:
        return self._nbytes


def as_provider(obj: Any, *, name: str | None = None, **kw: Any) -> LogitsProvider:
    """Coerce a module / callable / provider into a :class:`LogitsProvider`."""
    if hasattr(obj, "logits") and callable(obj.logits):
        return obj
    torch = _torch()
    if isinstance(obj, torch.nn.Module):
        return ModuleProvider(obj, name=name or type(obj).__name__, **kw)
    if callable(obj):
        return CallableProvider(obj, name=name or getattr(obj, "__name__", "callable"), **kw)
    raise TypeError(f"cannot make a LogitsProvider from {type(obj)!r}")


def provider_nbytes(p: Any) -> int | None:
    fn = getattr(p, "nbytes", None)
    if callable(fn):
        try:
            v = fn()
            return None if v is None else int(v)
        except Exception:
            return None
    return None


def greedy_generate(
    provider: LogitsProvider,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
    stop_ids: Sequence[int] = (),
    temperature: float = 0.0,
    seed: int | None = None,
    max_context: int | None = None,
) -> list[int]:
    """Reference generation from any provider by re-running the full forward each step (O(T^2)).

    Providers with a faster ``generate`` method (KV-cached engines) are used instead by :func:`generate`."""
    torch = _torch()
    ids = list(int(t) for t in prompt_ids)
    out: list[int] = []
    gen = torch.Generator().manual_seed(int(seed or 0)) if temperature > 0 else None
    stops = set(int(s) for s in stop_ids)
    if eos_token_id is not None:
        stops.add(int(eos_token_id))
    for _ in range(max_new_tokens):
        ctx = ids if max_context is None else ids[-max_context:]
        lg = provider.logits(torch.tensor([ctx], dtype=torch.long))[0, -1].float()
        if temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(lg / temperature, -1), 1, generator=gen).item())
        else:
            nxt = int(torch.argmax(lg).item())
        ids.append(nxt)
        out.append(nxt)
        if nxt in stops:
            break
    return out


def generate(
    provider: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    stop_ids: Sequence[int] = (),
    temperature: float = 0.0,
    seed: int | None = None,
) -> list[int]:
    """Generate with ``provider.generate`` when available, else :func:`greedy_generate`."""
    fn = getattr(provider, "generate", None)
    eos = getattr(provider, "eos_token_id", None)
    if callable(fn):
        return [int(t) for t in fn(list(prompt_ids), max_new_tokens, stop_ids=tuple(stop_ids), temperature=temperature, seed=seed)]
    return greedy_generate(provider, prompt_ids, max_new_tokens, eos_token_id=eos, stop_ids=stop_ids, temperature=temperature, seed=seed)


# --------------------------------------------------------------------------------------------
# core per-position computations
# --------------------------------------------------------------------------------------------
def _flat(logits, mask):
    """-> (rows[N,V] float tensor view, keep bool[N] numpy or None)"""
    torch = _torch()
    lg = logits.reshape(-1, logits.shape[-1])
    if mask is None:
        return lg, None
    m = torch.as_tensor(mask).reshape(-1).bool()
    if m.numel() != lg.shape[0]:
        raise ValueError(f"mask has {m.numel()} entries but logits have {lg.shape[0]} positions")
    return lg, m


def _select(lg, m):
    return lg if m is None else lg[m]


def kl_per_position(parent_logits, cand_logits, mask=None, temperature: float = 1.0) -> "torch.Tensor":
    """KL(parent || candidate) in nats per unmasked position, float64 ``[N]``."""
    torch = _torch()
    p, m = _flat(parent_logits, mask)
    c, m2 = _flat(cand_logits, mask)
    if p.shape != c.shape:
        raise ValueError(f"logit shapes differ: {tuple(p.shape)} vs {tuple(c.shape)}")
    p, c = _select(p, m), _select(c, m)
    outs = []
    for i in range(0, p.shape[0], _ROW_CHUNK):
        lp = torch.log_softmax(p[i : i + _ROW_CHUNK].double() / temperature, dim=-1)
        lc = torch.log_softmax(c[i : i + _ROW_CHUNK].double() / temperature, dim=-1)
        pr = lp.exp()
        # p == 0 contributes 0 (also when log p = -inf); p > 0 with q == 0 gives +inf, as it should
        outs.append(torch.where(pr > 0, pr * (lp - lc), torch.zeros_like(pr)).sum(-1))
    if not outs:
        return torch.zeros(0, dtype=torch.float64)
    # guard tiny negative values from rounding
    return torch.cat(outs).clamp_min(0.0)


@dataclass
class KLStats:
    """Summary of per-position KL(parent||candidate) in nats."""

    mean: float
    p50: float
    p99: float
    max: float
    n: int
    per_position: Any = field(default=None, repr=False)  # numpy array if requested

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("per_position", None)
        return d


def kl_divergence(parent_logits, cand_logits, p99: bool = True, mask=None, temperature: float = 1.0, keep_per_position: bool = False) -> KLStats:
    """KL(parent || candidate) summary (mean, median, p99, max) over unmasked positions.

    ``p99=False`` skips the percentile computation (p99 is then reported as NaN)."""
    kl = kl_per_position(parent_logits, cand_logits, mask, temperature).numpy()
    if kl.size == 0:
        return KLStats(float("nan"), float("nan"), float("nan"), float("nan"), 0, kl if keep_per_position else None)
    return KLStats(
        mean=float(kl.mean()), p50=float(np.median(kl)), p99=float(np.quantile(kl, 0.99)) if p99 else float("nan"),
        max=float(kl.max()), n=int(kl.size), per_position=kl if keep_per_position else None,
    )


def top1_flips(parent_logits, cand_logits, mask=None) -> np.ndarray:
    """Boolean numpy array: True where argmax(parent) != argmax(candidate)."""
    p, m = _flat(parent_logits, mask)
    c, _ = _flat(cand_logits, mask)
    pa = _select(p, m).argmax(-1)
    ca = _select(c, m).argmax(-1)
    return (pa != ca).numpy()


@dataclass
class FlipStats:
    rate: float
    n: int
    flips: int
    ci_low: float
    ci_high: float
    per_position: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("per_position", None)
        return d


def top1_flip_rate(parent_logits, cand_logits, mask=None, alpha: float = 0.05, keep_per_position: bool = False) -> FlipStats:
    """Top-1 flip rate with a Wilson CI (treats positions as independent - use sequence-level bootstrap in suites
    for the honest, cluster-aware interval)."""
    from forge.eval.stats import binomial_ci

    fl = top1_flips(parent_logits, cand_logits, mask)
    n, k = int(fl.size), int(fl.sum())
    p, lo, hi = binomial_ci(k, n, alpha) if n else (float("nan"), 0.0, 1.0)
    return FlipStats(p, n, k, lo, hi, fl if keep_per_position else None)


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first differing token of two generations (``None`` if one is a prefix of the other and equal
    length, i.e. identical; if lengths differ and all shared tokens match, the shorter length is returned)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def entropy_per_position(logits, mask=None) -> "torch.Tensor":
    torch = _torch()
    lg, m = _flat(logits, mask)
    lg = _select(lg, m)
    outs = []
    for i in range(0, lg.shape[0], _ROW_CHUNK):
        lp = torch.log_softmax(lg[i : i + _ROW_CHUNK].double(), -1)
        pr = lp.exp()
        outs.append(-torch.where(pr > 0, pr * lp, torch.zeros_like(pr)).sum(-1))
    return torch.cat(outs) if outs else torch.zeros(0, dtype=torch.float64)


def entropy_shift(parent_logits, cand_logits, mask=None) -> float:
    """mean(H(candidate)) - mean(H(parent)) in nats (positive = candidate less confident)."""
    hp = entropy_per_position(parent_logits, mask)
    hc = entropy_per_position(cand_logits, mask)
    if hp.numel() == 0:
        return float("nan")
    return float((hc - hp).mean().item())


def nll_per_position(logits, targets, mask=None) -> "torch.Tensor":
    """Negative log-likelihood (nats) of ``targets`` (same leading shape as logits without V) per unmasked position."""
    torch = _torch()
    lg, m = _flat(logits, mask)
    t = torch.as_tensor(targets).reshape(-1).long()
    if t.numel() != lg.shape[0]:
        raise ValueError("targets must have one entry per logits position")
    lg, t = _select(lg, m), (t if m is None else t[m])
    outs = []
    for i in range(0, lg.shape[0], _ROW_CHUNK):
        lp = torch.log_softmax(lg[i : i + _ROW_CHUNK].double(), -1)
        outs.append(-lp.gather(-1, t[i : i + _ROW_CHUNK].unsqueeze(-1)).squeeze(-1))
    return torch.cat(outs) if outs else torch.zeros(0, dtype=torch.float64)


def perplexity(logits, targets, mask=None) -> float:
    """exp(mean NLL) over unmasked positions (``targets`` aligned to ``logits``: position t predicts ``targets[t]``)."""
    nll = nll_per_position(logits, targets, mask)
    if nll.numel() == 0:
        return float("nan")
    return float(math.exp(float(nll.mean().item())))


@dataclass
class ECEResult:
    ece: float
    mce: float
    n: int
    bins: list[dict[str, float]]
    accuracy: float
    mean_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ece_from_confidences(confidences: Sequence[float] | np.ndarray, correct: Sequence[bool] | np.ndarray, n_bins: int = 10) -> ECEResult:
    """Expected calibration error from per-example confidence and correctness (equal-width bins)."""
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    n = conf.size
    if n == 0 or n != corr.size:
        return ECEResult(float("nan"), float("nan"), 0, [], float("nan"), float("nan"))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    bins = []
    for b in range(n_bins):
        sel = idx == b
        cnt = int(sel.sum())
        if cnt == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": 0, "confidence": float("nan"), "accuracy": float("nan")})
            continue
        acc, cf = float(corr[sel].mean()), float(conf[sel].mean())
        gap = abs(acc - cf)
        ece += cnt / n * gap
        mce = max(mce, gap)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": cnt, "confidence": cf, "accuracy": acc})
    return ECEResult(float(ece), float(mce), int(n), bins, float(corr.mean()), float(conf.mean()))


def ece(logits_or_probs, labels, n_bins: int = 10, mask=None, from_probs: bool = False) -> ECEResult:
    """Multi-class ECE.  ``logits_or_probs`` is ``[..., C]`` (logits by default), ``labels`` integer ``[...]``.
    Confidence is the max probability, correct means argmax == label.  For token-level ECE pass LM logits and the
    next-token targets; for multiple-choice pass the per-question option scores ``[Q, C]``."""
    torch = _torch()
    x, m = _flat(torch.as_tensor(logits_or_probs), mask)
    y = torch.as_tensor(labels).reshape(-1).long()
    x, y = _select(x, m), (y if m is None else y[m])
    confs, corr = [], []
    for i in range(0, x.shape[0], _ROW_CHUNK):
        chunk = x[i : i + _ROW_CHUNK].double()
        pr = chunk if from_probs else torch.softmax(chunk, -1)
        c, a = pr.max(-1)
        confs.append(c)
        corr.append(a == y[i : i + _ROW_CHUNK])
    if not confs:
        return ece_from_confidences([], [], n_bins)
    return ece_from_confidences(torch.cat(confs).numpy(), torch.cat(corr).numpy(), n_bins)


# --------------------------------------------------------------------------------------------
# streaming distribution comparison for model objects / providers
# --------------------------------------------------------------------------------------------
@dataclass
class SequenceDistribution:
    """Per-sequence arrays gathered by :func:`compare_distributions` (so callers can cluster-bootstrap)."""

    kl: list[np.ndarray] = field(default_factory=list)  # per-position KL per sequence
    flips: list[np.ndarray] = field(default_factory=list)
    h_parent: list[np.ndarray] = field(default_factory=list)
    h_cand: list[np.ndarray] = field(default_factory=list)
    nll_parent: list[np.ndarray] = field(default_factory=list)
    nll_cand: list[np.ndarray] = field(default_factory=list)
    conf_parent: list[np.ndarray] = field(default_factory=list)
    conf_cand: list[np.ndarray] = field(default_factory=list)
    correct_parent: list[np.ndarray] = field(default_factory=list)
    correct_cand: list[np.ndarray] = field(default_factory=list)
    max_abs_logit_diff: float = 0.0
    identical_logits: bool = True

    @property
    def n_sequences(self) -> int:
        return len(self.kl)

    @property
    def n_tokens(self) -> int:
        return int(sum(len(x) for x in self.kl))

    def pooled(self, name: str) -> np.ndarray:
        arrs = getattr(self, name)
        return np.concatenate(arrs) if arrs else np.zeros(0)


def iter_batches(sequences: Sequence[Sequence[int]], batch_size: int, pad_id: int = 0) -> Iterator[tuple[Any, Any, list[int]]]:
    """Right-pad ``sequences`` into ``(input_ids[B,T], mask[B,T], lengths)`` batches."""
    torch = _torch()
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        T = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        msk = torch.zeros((len(chunk), T), dtype=torch.bool)
        for j, s in enumerate(chunk):
            ids[j, : len(s)] = torch.tensor(list(s), dtype=torch.long)
            msk[j, : len(s)] = True
        yield ids, msk, [len(s) for s in chunk]


def compare_distributions(
    parent: LogitsProvider,
    candidate: LogitsProvider,
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int = 4,
    pad_id: int = 0,
    n_bins: int = 10,
    progress: Callable[[float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SequenceDistribution:
    """Teacher-forced comparison of ``candidate`` against ``parent`` over token sequences.

    For each sequence of length ``T`` the logits at positions ``0..T-2`` (predicting tokens ``1..T-1``) are
    compared.  Only per-position scalars are retained, so memory stays small even for a 150k vocabulary."""
    torch = _torch()
    out = SequenceDistribution()
    total = len(sequences)
    done = 0
    for ids, msk, lens in iter_batches([s for s in sequences if len(s) >= 2], batch_size, pad_id):
        if should_stop and should_stop():
            break
        lp = parent.logits(ids, msk)
        lc = candidate.logits(ids, msk)
        if lp.shape != lc.shape:
            raise ValueError(f"parent/candidate logits differ in shape: {tuple(lp.shape)} vs {tuple(lc.shape)}")
        diff = float((lp - lc).abs().max().item())
        out.max_abs_logit_diff = max(out.max_abs_logit_diff, diff)
        if diff != 0.0:
            out.identical_logits = False
        for j, L in enumerate(lens):
            P = lp[j, : L - 1]
            C = lc[j, : L - 1]
            tgt = ids[j, 1:L]
            out.kl.append(kl_per_position(P, C).numpy())
            out.flips.append(top1_flips(P, C))
            out.h_parent.append(entropy_per_position(P).numpy())
            out.h_cand.append(entropy_per_position(C).numpy())
            out.nll_parent.append(nll_per_position(P, tgt).numpy())
            out.nll_cand.append(nll_per_position(C, tgt).numpy())
            for logits_, store_c, store_k in ((P, out.conf_parent, out.correct_parent), (C, out.conf_cand, out.correct_cand)):
                pr = torch.softmax(logits_.double(), -1)
                cf, am = pr.max(-1)
                store_c.append(cf.numpy())
                store_k.append((am == tgt).numpy())
        done += len(lens)
        if progress:
            progress(min(1.0, done / max(1, total)))
    return out


def summarize_distribution(d: SequenceDistribution, n_bins: int = 10) -> dict[str, float]:
    """Point summaries (no CIs) of a :class:`SequenceDistribution`."""
    if d.n_tokens == 0:
        return {}
    kl = d.pooled("kl")
    fl = d.pooled("flips")
    hp, hc = d.pooled("h_parent"), d.pooled("h_cand")
    npar, ncan = d.pooled("nll_parent"), d.pooled("nll_cand")
    e_par = ece_from_confidences(d.pooled("conf_parent"), d.pooled("correct_parent"), n_bins)
    e_can = ece_from_confidences(d.pooled("conf_cand"), d.pooled("correct_cand"), n_bins)
    return {
        "kl_mean": float(kl.mean()), "kl_p50": float(np.median(kl)), "kl_p99": float(np.quantile(kl, 0.99)), "kl_max": float(kl.max()),
        "top1_flip": float(fl.mean()), "entropy_shift": float((hc - hp).mean()),
        "ppl_parent": float(math.exp(npar.mean())), "ppl_candidate": float(math.exp(ncan.mean())),
        "ppl_ratio": float(math.exp(ncan.mean() - npar.mean())),
        "ece_parent": e_par.ece, "ece_candidate": e_can.ece, "ece_delta": e_can.ece - e_par.ece,
        "n_tokens": float(d.n_tokens), "n_sequences": float(d.n_sequences),
    }
