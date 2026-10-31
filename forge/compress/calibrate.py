"""Calibration sets and activation statistics (Plan CP-1/CP-3, CP-7).

* **Registry** of named calibration sets (:func:`register_calibration_set`): inline texts, a text/jsonl file, raw token tensors,
  a dataset manifest from the data pipeline, or a weighted *mixture* of other sets (domain mix; the MoE pruning rule in the Plan
  needs a mix that includes generative samples).  Built-in sets: ``synthetic`` (Zipf tokens with local repetition; needs only a
  vocabulary size), ``builtin-prose|code|math|chat`` (tiny hand-written texts) and the mixture ``builtin-mix``.  ``file:<path>`` resolves on
  the fly.  Calibration *sensitivity* (the Plan marks it SHOULD-document) is therefore explicit: every result carries the set's content hash.
* :class:`StatsCache` keyed ``(model fingerprint, set hash, layer)`` holding ``hessian`` / ``channel_scale`` / ``router`` statistics.
* :func:`collect_stats` hooks the linear layers of a model and accumulates statistics over a calibration set (non-sequential, whole
  model, one pass).  GPTQ itself collects its Hessians *sequentially* inside :mod:`forge.compress.ptq` and does not need this cache;
  the cache serves analysis, mixed-precision sensitivity, pruning and the UI.

Tokenisation: anything with ``encode(text)`` / ``__call__(text)`` returning ids works (HF tokenizers, ``forge.models.loader``
tokenizers); with ``tokenizer=None`` text sets fall back to raw UTF-8 bytes folded into the vocabulary (documented fallback for
tiny test models).
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError, NotFound
from forge.core.hashing import hash_bytes, hash_json
from forge.compress.modelutil import (
    ChannelStats,
    HessianAccumulator,
    iter_dense_linears,
    model_device,
    model_fingerprint,
)

log = logging.getLogger("forge.compress.calibrate")

_BUILTIN_TEXTS: dict[str, list[str]] = {
    "builtin-prose": [
        "The committee met on Tuesday to discuss the new harbour plan. After a long debate about costs, timelines and the effect on local "
        "fishermen, the members agreed to commission a second study before any construction begins. Nobody expected the vote to be close.",
        "In the early morning the fog lifted slowly from the valley, revealing terraced fields, a narrow river and a village whose roofs "
        "were still dark with dew. A farmer walked to the edge of his field, looked at the sky and decided that the rain would hold off.",
        "Economists often disagree about the causes of inflation, but most accept that expectations matter: if firms and households expect "
        "prices to rise, they behave in ways that make it so. Central banks therefore spend as much effort on communication as on interest rates.",
    ],
    "builtin-code": [
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n"
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n"
        "    def pop(self):\n        return self.items.pop()\n",
        "#include <stdio.h>\nint main(void) {\n    int total = 0;\n    for (int i = 1; i <= 10; ++i) { total += i * i; }\n"
        "    printf(\"%d\\n\", total);\n    return 0;\n}\n",
        "SELECT customer_id, SUM(amount) AS revenue FROM orders WHERE created_at >= '2024-01-01' GROUP BY customer_id "
        "HAVING SUM(amount) > 1000 ORDER BY revenue DESC LIMIT 20;",
    ],
    "builtin-math": [
        "Let f(x) = x^2 - 4x + 3. Then f(x) = (x - 1)(x - 3), so the roots are x = 1 and x = 3. The vertex is at x = 2 where f(2) = -1. "
        "Therefore the minimum value of f is -1 and it is attained once.",
        "Proof. Suppose sqrt(2) = p/q in lowest terms. Then p^2 = 2 q^2, so p is even, say p = 2r. Hence 4 r^2 = 2 q^2 and q^2 = 2 r^2, "
        "so q is even too, contradicting lowest terms. Hence sqrt(2) is irrational.",
        "The sum of the first n odd numbers is n^2. Check: 1 = 1, 1 + 3 = 4, 1 + 3 + 5 = 9, 1 + 3 + 5 + 7 = 16. By induction, adding the "
        "next odd number 2n + 1 to n^2 gives (n + 1)^2.",
    ],
    "builtin-chat": [
        "User: Can you explain what a hash table is?\nAssistant: A hash table stores key-value pairs. A hash function maps each key to a bucket, "
        "which makes lookups, inserts and deletes O(1) on average.\nUser: What happens on a collision?\nAssistant: Two keys land in the same "
        "bucket; the table resolves it by chaining or open addressing.",
        "User: Give me three tips for sleeping better.\nAssistant: Keep a regular schedule, avoid screens for an hour before bed, and keep the "
        "bedroom cool and dark. If problems persist, talk to a doctor.",
    ],
}


# ------------------------------------------------------------------------------------------------ tokenisation
def encode_text(tokenizer: Any, text: str, vocab_size: int | None = None) -> list[int]:
    """Encode with an HF-like tokenizer / callable; ``None`` -> UTF-8 bytes folded into ``vocab_size`` (fallback)."""
    if tokenizer is None:
        ids = list(text.encode("utf-8"))
        if vocab_size is not None and vocab_size < 256:
            ids = [i % vocab_size for i in ids]
        return ids
    if hasattr(tokenizer, "encode"):
        try:
            out = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            out = tokenizer.encode(text)
    else:
        out = tokenizer(text)
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        out = out["input_ids"] if isinstance(out, dict) else out.input_ids
    if hasattr(out, "ids"):
        out = out.ids
    return [int(i) for i in (out.tolist() if hasattr(out, "tolist") else out)]


def _eos_id(tokenizer: Any) -> int | None:
    for attr in ("eos_token_id", "eos_id"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            return v
    return None


# ------------------------------------------------------------------------------------------------ calibration sets
@dataclass
class CalibrationSet:
    """A named source of calibration token sequences.

    ``kind``: ``synthetic`` | ``texts`` | ``file`` | ``tokens`` | ``dataset`` | ``mixture``.
    """

    name: str
    kind: str
    texts: list[str] = field(default_factory=list)
    path: str | None = None
    tokens: torch.Tensor | None = None
    dataset_ref: Any = None
    components: dict[str, float] = field(default_factory=dict)  # mixture: set name -> weight
    description: str = ""

    # -- identity -------------------------------------------------------------------------------
    def content_hash(self) -> str:
        """Digest of the set's content (texts / file bytes / tokens / components) -- part of the stats-cache key."""
        if self.kind == "synthetic":
            return hash_json({"kind": "synthetic", "name": self.name})
        if self.kind == "texts":
            return hash_json({"kind": "texts", "texts": self.texts})
        if self.kind == "file":
            p = Path(self.path or "")
            return hash_json({"kind": "file", "bytes": hash_bytes(p.read_bytes()) if p.exists() else str(p)})
        if self.kind == "tokens":
            assert self.tokens is not None
            return hash_json({"kind": "tokens", "h": hash_bytes(self.tokens.cpu().numpy().tobytes()), "shape": list(self.tokens.shape)})
        if self.kind == "mixture":
            return hash_json({"kind": "mixture", "c": {k: [v, get_calibration_set(k).content_hash()] for k, v in sorted(self.components.items())}})
        return hash_json({"kind": self.kind, "ref": repr(self.dataset_ref)})

    # -- sampling -------------------------------------------------------------------------------
    def sample_tokens(
        self, n_seqs: int, seq_len: int, *, tokenizer: Any = None, vocab_size: int | None = None, seed: int = 0
    ) -> torch.Tensor:
        """``LongTensor [n_seqs, seq_len]``, deterministic given ``(set, n_seqs, seq_len, tokenizer, vocab_size, seed)``."""
        return self.sample_labeled(n_seqs, seq_len, tokenizer=tokenizer, vocab_size=vocab_size, seed=seed)[0]

    def sample_labeled(
        self, n_seqs: int, seq_len: int, *, tokenizer: Any = None, vocab_size: int | None = None, seed: int = 0
    ) -> tuple[torch.Tensor, list[str]]:
        """Like :meth:`sample_tokens` but also returns the domain label (component set name) of every sequence."""
        if n_seqs < 1 or seq_len < 2:
            raise ConfigError("need n_seqs >= 1 and seq_len >= 2")
        if self.kind == "mixture":
            return self._sample_mixture(n_seqs, seq_len, tokenizer, vocab_size, seed)
        if self.kind == "synthetic":
            if vocab_size is None:
                raise ConfigError("the synthetic calibration set needs vocab_size")
            return _synthetic_tokens(n_seqs, seq_len, vocab_size, seed), [self.name] * n_seqs
        if self.kind == "tokens":
            assert self.tokens is not None
            flat = self.tokens.reshape(-1).long()
            return _windows(flat, n_seqs, seq_len, seed), [self.name] * n_seqs
        if self.kind == "dataset":
            flat = _tokens_from_dataset(self.dataset_ref, n_seqs * seq_len)
            return _windows(flat, n_seqs, seq_len, seed), [self.name] * n_seqs
        texts = self.texts if self.kind == "texts" else _read_texts(self.path)
        if not texts:
            raise ConfigError(f"calibration set {self.name!r} is empty")
        eos = _eos_id(tokenizer)
        stream: list[int] = []
        for t in texts:
            stream.extend(encode_text(tokenizer, t, vocab_size))
            if eos is not None:
                stream.append(eos)
        flat = torch.tensor(stream, dtype=torch.long)
        if vocab_size is not None and int(flat.max()) >= vocab_size:
            raise ConfigError(f"tokenizer produced id {int(flat.max())} >= model vocab_size {vocab_size}")
        return _windows(flat, n_seqs, seq_len, seed), [self.name] * n_seqs

    def _sample_mixture(self, n_seqs, seq_len, tokenizer, vocab_size, seed) -> tuple[torch.Tensor, list[str]]:
        comps = {k: v for k, v in self.components.items() if v > 0}
        total = sum(comps.values())
        quotas = {k: n_seqs * v / total for k, v in comps.items()}
        counts = {k: int(math.floor(q)) for k, q in quotas.items()}
        rest = n_seqs - sum(counts.values())
        for k in sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)[:rest]:
            counts[k] += 1
        parts, labels = [], []
        for i, (k, c) in enumerate(sorted(counts.items())):
            if c == 0:
                continue
            t, lab = get_calibration_set(k).sample_labeled(c, seq_len, tokenizer=tokenizer, vocab_size=vocab_size, seed=seed + 1009 * (i + 1))
            parts.append(t)
            labels += lab
        return torch.cat(parts, 0), labels

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "description": self.description, "n_texts": len(self.texts),
                "path": self.path, "components": dict(self.components)}


def _windows(flat: torch.Tensor, n_seqs: int, seq_len: int, seed: int) -> torch.Tensor:
    if flat.numel() < seq_len:
        reps = -(-seq_len // max(flat.numel(), 1))
        flat = flat.repeat(reps)
    g = torch.Generator().manual_seed(int(seed))
    starts = torch.randint(0, flat.numel() - seq_len + 1, (n_seqs,), generator=g)
    return torch.stack([flat[s : s + seq_len] for s in starts.tolist()])


def _synthetic_tokens(n_seqs: int, seq_len: int, vocab: int, seed: int) -> torch.Tensor:
    """Zipf(1.1)-distributed ids over a seeded permutation of the vocabulary, with 30 % local copies (gives attention something to do)."""
    g = torch.Generator().manual_seed(int(seed) + 7919)
    ranks = torch.arange(1, vocab + 1, dtype=torch.float64)
    p = ranks.pow(-1.1)
    p /= p.sum()
    perm = torch.randperm(vocab, generator=g)
    base = perm[torch.multinomial(p.float(), n_seqs * seq_len, replacement=True, generator=g)].reshape(n_seqs, seq_len)
    copy = torch.rand(n_seqs, seq_len, generator=g) < 0.3
    back = torch.randint(1, 9, (n_seqs, seq_len), generator=g)
    idx = (torch.arange(seq_len).unsqueeze(0) - back).clamp(min=0)
    return torch.where(copy, base.gather(1, idx), base).long()


def _read_texts(path: str | None) -> list[str]:
    if not path:
        raise ConfigError("file calibration set has no path")
    p = Path(path)
    if not p.exists():
        raise NotFound(f"calibration file not found: {path}")
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix in (".jsonl", ".ndjson"):
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                out.append(obj if isinstance(obj, str) else str(obj.get("text") or obj.get("content") or obj.get("prompt") or ""))
        return [t for t in out if t]
    paras = [t.strip() for t in raw.split("\n\n")]
    return [t for t in paras if t] or ([raw] if raw.strip() else [])


def _tokens_from_dataset(ref: Any, n_tokens: int) -> torch.Tensor:
    """Read tokens from a dataset manifest of the data pipeline (``forge.train.data``, imported lazily)."""
    try:
        from forge.train import data as tdata  # type: ignore
    except Exception as e:  # pragma: no cover - depends on module availability
        raise ConfigError(f"dataset-backed calibration sets need forge.train.data ({e})") from e
    ds = tdata.ShardedTokenDataset(ref) if not hasattr(ref, "__getitem__") else ref
    out, n, i = [], 0, 0
    while n < n_tokens and i < len(ds):
        item = ds[i]
        t = item["input_ids"] if isinstance(item, dict) else item
        t = torch.as_tensor(t).reshape(-1)
        out.append(t)
        n += t.numel()
        i += 1
    if not out:
        raise ConfigError("dataset yielded no tokens")
    return torch.cat(out).long()


_REGISTRY: dict[str, CalibrationSet] = {}


def register_calibration_set(
    name: str,
    texts: Sequence[str] | None = None,
    *,
    path: str | Path | None = None,
    tokens: torch.Tensor | None = None,
    dataset_ref: Any = None,
    components: dict[str, float] | None = None,
    description: str = "",
    overwrite: bool = True,
) -> CalibrationSet:
    """Register a calibration set. Exactly one of ``texts`` / ``path`` / ``tokens`` / ``dataset_ref`` / ``components`` (mixture)."""
    given = [x is not None for x in (texts, path, tokens, dataset_ref, components)]
    if sum(given) != 1:
        raise ConfigError("give exactly one of texts, path, tokens, dataset_ref, components")
    if name in _REGISTRY and not overwrite:
        raise ConfigError(f"calibration set {name!r} already registered")
    if texts is not None:
        cs = CalibrationSet(name, "texts", texts=[str(t) for t in texts], description=description)
    elif path is not None:
        cs = CalibrationSet(name, "file", path=str(path), description=description)
    elif tokens is not None:
        cs = CalibrationSet(name, "tokens", tokens=torch.as_tensor(tokens).long(), description=description)
    elif dataset_ref is not None:
        cs = CalibrationSet(name, "dataset", dataset_ref=dataset_ref, description=description)
    else:
        for k in components or {}:
            get_calibration_set(k)  # must exist
        cs = CalibrationSet(name, "mixture", components=dict(components or {}), description=description)
    _REGISTRY[name] = cs
    return cs


def _register_builtins() -> None:
    _REGISTRY["synthetic"] = CalibrationSet("synthetic", "synthetic", description="Zipf-distributed random tokens with local copies; needs only vocab_size")
    for k, v in _BUILTIN_TEXTS.items():
        _REGISTRY[k] = CalibrationSet(k, "texts", texts=list(v), description=f"tiny built-in {k.split('-')[1]} sample (smoke tests; register a real set for production)")
    _REGISTRY["builtin-mix"] = CalibrationSet(
        "builtin-mix", "mixture", components={"builtin-prose": 0.4, "builtin-code": 0.25, "builtin-math": 0.2, "builtin-chat": 0.15},
        description="weighted mix of the built-in domain samples",
    )


_register_builtins()


def get_calibration_set(name_or_set: str | CalibrationSet | torch.Tensor) -> CalibrationSet:
    if isinstance(name_or_set, CalibrationSet):
        return name_or_set
    if isinstance(name_or_set, torch.Tensor):
        return CalibrationSet("tokens-inline", "tokens", tokens=name_or_set.long())
    name = str(name_or_set)
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name.startswith("file:"):
        return CalibrationSet(name, "file", path=name[5:])
    if name.startswith("synthetic"):
        return CalibrationSet(name, "synthetic", description="synthetic variant (name acts as a seed namespace)")
    if Path(name).is_file():
        return CalibrationSet(f"file:{name}", "file", path=name)
    raise NotFound(f"unknown calibration set {name!r}; registered: {', '.join(sorted(_REGISTRY))}", details={"name": name})


def list_calibration_sets() -> list[dict[str, Any]]:
    return [cs.info() for _, cs in sorted(_REGISTRY.items())]


def get_calibration_tokens(
    calib: str | CalibrationSet | torch.Tensor | None,
    model: nn.Module | None = None,
    tokenizer: Any = None,
    *,
    n_seqs: int = 128,
    seq_len: int = 512,
    seed: int = 0,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Resolve ``calib`` (name / set / token tensor) into ``(tokens [n, L], domain labels)`` for ``model``."""
    if isinstance(calib, torch.Tensor) and calib.ndim == 2:
        return calib.long(), ["tokens"] * calib.shape[0]
    cs = get_calibration_set("synthetic" if calib is None else calib)
    if vocab_size is None and model is not None:
        cfg = getattr(model, "config", None)
        vocab_size = getattr(cfg, "vocab_size", None)
        if vocab_size is None:
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    vocab_size = m.num_embeddings
                    break
    return cs.sample_labeled(n_seqs, seq_len, tokenizer=tokenizer, vocab_size=vocab_size, seed=seed)


# ------------------------------------------------------------------------------------------------ stats cache
@dataclass
class RouterStats:
    """MoE router statistics of one layer over the calibration set (REAP-style saliency inputs)."""

    layer: str
    num_experts: int
    top_k: int
    tokens: int
    counts: torch.Tensor  # [E] tokens routed to each expert
    weight_sum: torch.Tensor  # [E] sum of routing weights received
    prob_sum: torch.Tensor  # [E] sum of full-softmax probabilities
    out_norm_sum: torch.Tensor | None = None  # [E] sum of ||expert output|| over routed tokens (per-row, before gating)
    coact: torch.Tensor | None = None  # [E,E] tokens routed to both experts
    domain_counts: dict[str, torch.Tensor] = field(default_factory=dict)
    domain_tokens: dict[str, int] = field(default_factory=dict)

    @property
    def freq(self) -> torch.Tensor:
        """Fraction of tokens routed to each expert (sums to ``top_k``)."""
        return self.counts.float() / max(self.tokens, 1)

    @property
    def gate_mass(self) -> torch.Tensor:
        """Mean routing weight per token (over *all* tokens, 0 when not selected)."""
        return self.weight_sum.float() / max(self.tokens, 1)

    @property
    def mean_gate_when_routed(self) -> torch.Tensor:
        return self.weight_sum.float() / self.counts.float().clamp(min=1)

    @property
    def mean_out_norm(self) -> torch.Tensor | None:
        if self.out_norm_sum is None:
            return None
        return self.out_norm_sum.float() / self.counts.float().clamp(min=1)

    def domain_freq(self, domain: str) -> torch.Tensor:
        return self.domain_counts[domain].float() / max(self.domain_tokens.get(domain, 1), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer, "num_experts": self.num_experts, "top_k": self.top_k, "tokens": self.tokens,
            "freq": self.freq.tolist(), "gate_mass": self.gate_mass.tolist(),
            "mean_out_norm": None if self.out_norm_sum is None else self.mean_out_norm.tolist(),  # type: ignore[union-attr]
            "domains": {d: self.domain_freq(d).tolist() for d in self.domain_counts},
        }


class StatsCache:
    """Calibration statistics keyed ``(model fingerprint, calibration-set hash, layer)`` -> ``{stat name: value}``.

    In-memory with optional on-disk spill (``root``): small stats are ``torch.save``d per key so a later process (a cached recipe stage)
    can reuse them.  ``hessian`` entries are float32 ``[K, K]`` and can be large; ``spill_hessian=False`` keeps them out of ``root``.
    """

    def __init__(self, root: str | Path | None = None, spill_hessian: bool = False) -> None:
        self._mem: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.root = Path(root) if root else None
        self.spill_hessian = spill_hessian
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: tuple[str, str, str]) -> Path:
        assert self.root is not None
        return self.root / (hash_json(list(key)).split(":")[-1][:40] + ".pt")

    def _load_disk(self, key: tuple[str, str, str]) -> dict[str, Any]:
        """Read a spilled entry with the restricted unpickler (``weights_only=True`` + ``RouterStats`` allow-listed): a tampered cache
        file can never execute code; anything it cannot decode raises and is treated as a cache miss by the callers."""
        try:  # torch >= 2.4
            torch.serialization.add_safe_globals([RouterStats])
        except AttributeError:  # pragma: no cover - older torch
            pass
        return torch.load(self._path(key), weights_only=True)["stats"]

    def put(self, model_hash: str, set_hash: str, layer: str, stat: str, value: Any) -> None:
        key = (model_hash, set_hash, layer)
        self._mem.setdefault(key, {})[stat] = value
        if self.root and (stat != "hessian" or self.spill_hessian):
            p = self._path(key)
            disk = {s: v for s, v in self._mem[key].items() if s != "hessian" or self.spill_hessian}
            torch.save({"key": list(key), "stats": disk}, p)

    def get(self, model_hash: str, set_hash: str, layer: str, stat: str, default: Any = None) -> Any:
        key = (model_hash, set_hash, layer)
        if key not in self._mem and self.root and self._path(key).exists():
            try:
                self._mem[key] = self._load_disk(key)
            except Exception as e:  # corrupt spill -> treat as miss
                log.warning("stats cache read failed for %s: %s", layer, e)
        return self._mem.get(key, {}).get(stat, default)

    def has(self, model_hash: str, set_hash: str, layer: str, stat: str | None = None) -> bool:
        key = (model_hash, set_hash, layer)
        if key in self._mem:
            return stat is None or stat in self._mem[key]
        if self.root and self._path(key).exists():
            try:
                return stat is None or stat in self._load_disk(key)
            except Exception:  # corrupt / untrusted spill -> miss
                return False
        return False

    def layers(self, model_hash: str, set_hash: str) -> list[str]:
        return sorted(k[2] for k in self._mem if k[0] == model_hash and k[1] == set_hash)

    def keys(self) -> list[tuple[str, str, str]]:
        return sorted(self._mem)

    def __len__(self) -> int:
        return len(self._mem)

    def clear(self) -> None:
        self._mem.clear()


# ------------------------------------------------------------------------------------------------ router collection
class _RouterAccumulator:
    def __init__(self, layer: str, moe: nn.Module, device: torch.device, track_coact: bool) -> None:
        self.layer = layer
        e = int(getattr(moe, "num_experts", 0) or moe.gate.out_features)
        self.e = e
        self.top_k = int(getattr(moe, "top_k", 2))
        self.norm_topk = bool(getattr(moe, "norm_topk", True))
        self.tokens = 0
        self.counts = torch.zeros(e, dtype=torch.float64, device=device)
        self.weight_sum = torch.zeros(e, dtype=torch.float64, device=device)
        self.prob_sum = torch.zeros(e, dtype=torch.float64, device=device)
        self.out_norm = torch.zeros(e, dtype=torch.float64, device=device)
        self.has_out_norm = False
        self.coact = torch.zeros(e, e, dtype=torch.float64, device=device) if track_coact else None
        self.domain = "all"
        self.d_counts: dict[str, torch.Tensor] = {}
        self.d_tokens: dict[str, int] = {}

    def on_logits(self, logits: torch.Tensor) -> None:
        lg = logits.detach().reshape(-1, self.e).float()
        probs = torch.softmax(lg, dim=-1)
        w, ids = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk:
            w = w / w.sum(-1, keepdim=True)
        n = lg.shape[0]
        self.tokens += n
        onehot = torch.zeros(n, self.e, dtype=torch.float64, device=lg.device)
        onehot.scatter_(1, ids, 1.0)
        c = onehot.sum(0)
        self.counts += c
        self.weight_sum += torch.zeros(n, self.e, dtype=torch.float64, device=lg.device).scatter_(1, ids, w.double()).sum(0)
        self.prob_sum += probs.double().sum(0)
        if self.coact is not None:
            self.coact += onehot.t() @ onehot
        self.d_counts[self.domain] = self.d_counts.get(self.domain, torch.zeros_like(c)) + c
        self.d_tokens[self.domain] = self.d_tokens.get(self.domain, 0) + n

    def on_expert_out(self, e: int, out: torch.Tensor) -> None:
        self.has_out_norm = True
        self.out_norm[e] += out.detach().float().reshape(-1, out.shape[-1]).norm(dim=-1).sum().double()

    def finish(self) -> RouterStats:
        return RouterStats(
            self.layer, self.e, self.top_k, self.tokens, self.counts.float().cpu(), self.weight_sum.float().cpu(), self.prob_sum.float().cpu(),
            self.out_norm.float().cpu() if self.has_out_norm else None, None if self.coact is None else self.coact.float().cpu(),
            {d: t.float().cpu() for d, t in self.d_counts.items()} if len(self.d_counts) > 1 else {},
            dict(self.d_tokens) if len(self.d_counts) > 1 else {},
        )


def find_moe_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Modules that look like MoE blocks: have a router ``gate`` linear and an ``experts`` ModuleList."""
    out = []
    for name, m in model.named_modules():
        if hasattr(m, "gate") and isinstance(getattr(m, "experts", None), nn.ModuleList) and isinstance(m.gate, nn.Module):
            out.append((name, m))
    return out


def collect_stats(
    model: nn.Module,
    tokenizer: Any,
    calib: str | CalibrationSet | torch.Tensor | None = "synthetic",
    layers: Sequence[str] | None = None,
    stats: Sequence[str] = ("hessian", "channel_scale", "router"),
    *,
    n_seqs: int = 32,
    seq_len: int = 256,
    batch_size: int = 4,
    seed: int = 0,
    cache: StatsCache | None = None,
    exclude: Sequence[str] | None = None,
    max_hessian_dim: int = 16384,
    track_coactivation: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> StatsCache:
    """Run ``model`` over a calibration set and fill a :class:`StatsCache`.

    ``layers``: regex patterns selecting linear layers by qualified name (``None`` = every dense linear except the default excludes).
    Stats per layer: ``hessian`` (``H = 2/n XX^T`` float32 -- skipped above ``max_hessian_dim`` inputs), ``channel_scale`` (dict of
    ``mean_abs``, ``abs_max``, ``mean_sq``, ``n``), and per MoE block ``router`` (:class:`RouterStats`, keyed by the MoE module name;
    per-domain frequencies are recorded when the set is a mixture).  The cache key uses :func:`model_fingerprint` and the set hash.
    """
    cache = cache if cache is not None else StatsCache()
    want = set(stats)
    unknown = want - {"hessian", "channel_scale", "router"}
    if unknown:
        raise ConfigError(f"unknown stats {sorted(unknown)}")
    cs = get_calibration_set("synthetic" if calib is None else calib)
    mh, sh = model_fingerprint(model), cs.content_hash()
    tokens, domains = get_calibration_tokens(cs, model, tokenizer, n_seqs=n_seqs, seq_len=seq_len, seed=seed)
    dev = model_device(model)
    excl = tuple(exclude) if exclude is not None else None
    targets = {}
    for name, m in iter_dense_linears(model, **({"exclude": excl} if excl is not None else {})):
        if layers is not None and not any(__import__("re").search(p, name) for p in layers):
            continue
        targets[name] = m
    hess: dict[str, HessianAccumulator] = {}
    chan: dict[str, ChannelStats] = {}
    routers: dict[str, _RouterAccumulator] = {}
    handles = []

    for name, m in targets.items():
        k = m.in_features
        if "hessian" in want and k <= max_hessian_dim:
            hess[name] = HessianAccumulator(k, dev)
        if "channel_scale" in want:
            chan[name] = ChannelStats(k, dev)

        def hook(_m, inp, _out, name=name):
            x = inp[0]
            if name in hess:
                hess[name].add(x)
            if name in chan:
                chan[name].add(x)

        handles.append(m.register_forward_hook(hook))
    if "router" in want:
        for name, moe in find_moe_blocks(model):
            acc = _RouterAccumulator(name, moe, dev, track_coactivation)
            routers[name] = acc
            handles.append(moe.gate.register_forward_hook(lambda _m, _i, out, acc=acc: acc.on_logits(out)))
            for ei, ex in enumerate(moe.experts):
                handles.append(ex.register_forward_hook(lambda _m, _i, out, acc=acc, ei=ei: acc.on_expert_out(ei, out)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # group sequences by domain so router statistics can be attributed to domains
            order = sorted(range(tokens.shape[0]), key=lambda i: domains[i])
            i = 0
            while i < len(order):
                if should_stop and should_stop():
                    break
                dom = domains[order[i]]
                idx = [j for j in order[i : i + batch_size] if domains[j] == dom]
                for acc in routers.values():
                    acc.domain = dom
                model(tokens[idx].to(dev))
                i += len(idx)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    for name, a in hess.items():
        cache.put(mh, sh, name, "hessian", a.H.cpu())
    for name, c in chan.items():
        cache.put(mh, sh, name, "channel_scale", {"mean_abs": c.mean_abs.cpu(), "abs_max": c.abs_max.cpu(), "mean_sq": c.mean_sq.cpu(), "n": c.n})
    for name, acc in routers.items():
        cache.put(mh, sh, name, "router", acc.finish())
    cache.put(mh, sh, "__meta__", "info", {"n_seqs": int(tokens.shape[0]), "seq_len": int(tokens.shape[1]), "tokens": int(tokens.numel()),
                                           "set": cs.name, "seed": seed, "stats": sorted(want)})
    cache.model_hash, cache.set_hash = mh, sh  # type: ignore[attr-defined]
    return cache
