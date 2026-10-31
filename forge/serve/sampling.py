"""Sampling parameters, logit processing, token sampling, logprobs and incremental detokenisation / stop strings.

``sampling_probs`` returns the *exact* distribution a token is drawn from (after bias, penalties, temperature,
top-k, top-p and min-p). Speculative decoding verifies against this same distribution, which is what makes it
output-distribution preserving (Plan SV-3, paper P7 App. A.1).
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from forge.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass
class SamplingParams:
    """Per-request sampling parameters (OpenAI/vLLM compatible naming)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 256
    min_tokens: int = 0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    include_stop_in_output: bool = False
    seed: int | None = None
    logprobs: int | None = None  # None = off; k >= 0 = chosen-token logprob + top-k alternatives
    logit_bias: dict[int, float] = field(default_factory=dict)
    n: int = 1
    skip_special_tokens: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.temperature < 0:
            raise ConfigError("temperature must be >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ConfigError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ConfigError("top_k must be >= 0")
        if not (0.0 <= self.min_p <= 1.0):
            raise ConfigError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ConfigError("repetition_penalty must be > 0")
        if self.max_tokens < 1:
            raise ConfigError("max_tokens must be >= 1")
        if self.min_tokens < 0 or self.min_tokens > self.max_tokens:
            raise ConfigError("min_tokens must be in [0, max_tokens]")
        if self.n < 1:
            raise ConfigError("n must be >= 1")
        if self.logprobs is not None and self.logprobs < 0:
            raise ConfigError("logprobs must be >= 0")
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        self.stop = [s for s in (self.stop or []) if s]
        self.logit_bias = {int(k): float(v) for k, v in (self.logit_bias or {}).items()}

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    @property
    def needs_history(self) -> bool:
        """True when the distribution depends on previously generated tokens (penalties)."""
        return (self.repetition_penalty != 1.0) or (self.presence_penalty != 0.0) or (self.frequency_penalty != 0.0)

    @classmethod
    def from_defaults(cls, defaults: Any, **overrides: Any) -> SamplingParams:
        """Build from a ``SamplingDefaults`` (or dict) plus request overrides (None overrides are ignored)."""
        base = defaults.model_dump() if hasattr(defaults, "model_dump") else dict(defaults or {})
        base.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in base.items() if k in known})

    def clone(self, **kw: Any) -> SamplingParams:
        d = {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        d.update(kw)
        return SamplingParams(**d)


# --------------------------------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------------------------------
def make_generator(seed: int | None, device: Any = "cpu") -> torch.Generator:
    import torch

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    try:
        g = torch.Generator(device=dev)
    except Exception:  # pragma: no cover - exotic device
        g = torch.Generator()
    g.manual_seed(secrets.randbits(63) if seed is None else int(seed) & ((1 << 63) - 1))
    return g


def process_logits(logits: torch.Tensor, params: SamplingParams, prompt_ids: list[int] | None = None,
                   output_ids: list[int] | None = None) -> torch.Tensor:
    """Apply logit bias and repetition/presence/frequency penalties. Returns a new float32 [V] tensor."""
    import torch

    x = logits.detach().to(torch.float32).clone()
    v = x.shape[-1]
    if params.logit_bias:
        idx = [i for i in params.logit_bias if 0 <= i < v]
        if idx:
            it = torch.tensor(idx, device=x.device, dtype=torch.long)
            val = torch.tensor([params.logit_bias[i] for i in idx], device=x.device, dtype=x.dtype)
            x.index_add_(0, it, val)
    if params.needs_history:
        out = output_ids or []
        if params.repetition_penalty != 1.0:
            seen = list(prompt_ids or []) + list(out)
            if seen:
                it = torch.tensor(sorted({t for t in seen if 0 <= t < v}), device=x.device, dtype=torch.long)
                if it.numel():
                    sel = x[it]
                    x[it] = torch.where(sel > 0, sel / params.repetition_penalty, sel * params.repetition_penalty)
        if (params.presence_penalty != 0.0 or params.frequency_penalty != 0.0) and out:
            ot = torch.tensor([t for t in out if 0 <= t < v], device=x.device, dtype=torch.long)
            if ot.numel():
                counts = torch.bincount(ot, minlength=v).to(x.dtype)
                x = x - params.frequency_penalty * counts - params.presence_penalty * (counts > 0).to(x.dtype)
    return x


def filter_logits(x: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Temperature then top-k / top-p / min-p; returns masked logits (-inf for removed tokens). x is [V]."""
    import torch

    x = x / params.temperature
    v = x.shape[-1]
    if params.top_k and params.top_k < v:
        kth = torch.topk(x, params.top_k).values[-1]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    if params.top_p < 1.0:
        sorted_x, order = torch.sort(x, descending=True)
        sp = torch.softmax(sorted_x, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        remove = (cum - sp) >= params.top_p  # keep the first token that crosses top_p
        remove[0] = False
        sorted_x = sorted_x.masked_fill(remove, float("-inf"))
        x = torch.full_like(x, float("-inf")).scatter(0, order, sorted_x)
    if params.min_p > 0.0:
        p = torch.softmax(x, dim=-1)
        thresh = params.min_p * p.max()
        x = torch.where(p < thresh, torch.full_like(x, float("-inf")), x)
    return x


def sampling_probs(logits: torch.Tensor, params: SamplingParams, prompt_ids: list[int] | None = None,
                   output_ids: list[int] | None = None) -> torch.Tensor:
    """The exact distribution (float32 [V], sums to 1) from which the next token is sampled."""
    import torch

    x = process_logits(logits, params, prompt_ids, output_ids)
    if params.greedy:
        p = torch.zeros_like(x)
        p[torch.argmax(x)] = 1.0
        return p
    return torch.softmax(filter_logits(x, params), dim=-1)


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    import torch

    if float(probs.max()) >= 1.0 - 1e-12 and int((probs > 0).sum()) == 1:
        return int(torch.argmax(probs))
    return int(torch.multinomial(probs, 1, generator=generator).item())


def sample_token(logits: torch.Tensor, params: SamplingParams, prompt_ids: list[int] | None = None,
                 output_ids: list[int] | None = None, generator: torch.Generator | None = None) -> int:
    """Draw one token id from a single [V] logits row."""
    if params.greedy:
        return int(process_logits(logits, params, prompt_ids, output_ids).argmax().item())
    return sample_from_probs(sampling_probs(logits, params, prompt_ids, output_ids), generator)


def token_logprobs(raw_logits: torch.Tensor, token: int, k: int) -> dict[str, Any]:
    """Logprob of ``token`` and the top-``k`` alternatives under the *unprocessed* model distribution."""
    import torch

    lp = torch.log_softmax(raw_logits.detach().to(torch.float32), dim=-1)
    out: dict[str, Any] = {"token_id": int(token), "logprob": float(lp[token].item()), "top": []}
    if k > 0:
        vals, idx = torch.topk(lp, min(k, lp.shape[-1]))
        out["top"] = [(int(i), float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
    return out


# --------------------------------------------------------------------------------------------------
# incremental detokenisation + stop strings
# --------------------------------------------------------------------------------------------------
class IncrementalDetokenizer:
    """Streams text for a growing token list, handles partial UTF-8 and stop strings.

    ``append`` returns the text that may be released to the client now (text that could still become the
    start of a stop string is held back until it is disambiguated or ``flush`` is called).
    """

    def __init__(self, tokenizer: Any, prompt_ids: list[int], stop: list[str] | None = None,
                 include_stop: bool = False, skip_special_tokens: bool = True) -> None:
        self.tok = tokenizer
        self.ids: list[int] = list(prompt_ids)
        self.stop = [s for s in (stop or []) if s]
        self.include_stop = include_stop
        self.skip_special = skip_special_tokens
        self.prefix_offset = max(len(self.ids) - 6, 0)
        self.read_offset = len(self.ids)
        self.text = ""          # full output text so far (truncated at a stop string)
        self.released = 0       # chars already handed to the caller
        self.stopped = False
        self.stop_string: str | None = None

    def _decode(self, ids: list[int]) -> str:
        if self.tok is None:
            return ""
        try:
            return self.tok.decode(ids, skip_special_tokens=self.skip_special)
        except TypeError:
            return self.tok.decode(ids)

    def append(self, token_ids: list[int]) -> str:
        if self.stopped:
            return ""
        if self.tok is None:
            self.ids.extend(token_ids)
            return ""
        delta = ""
        for t in token_ids:
            self.ids.append(int(t))
            prefix = self._decode(self.ids[self.prefix_offset:self.read_offset])
            new = self._decode(self.ids[self.prefix_offset:])
            if len(new) > len(prefix) and not new.endswith("�"):
                delta += new[len(prefix):]
                self.prefix_offset = self.read_offset
                self.read_offset = len(self.ids)
        if delta:
            self.text += delta
            self._check_stop(len(self.text) - len(delta))
        return self._release()

    def _check_stop(self, search_from: int) -> None:
        best = None
        for s in self.stop:
            start = max(0, search_from - len(s) + 1)
            pos = self.text.find(s, start)
            if pos != -1 and (best is None or pos < best[0]):
                best = (pos, s)
        if best is not None:
            pos, s = best
            self.text = self.text[: pos + len(s)] if self.include_stop else self.text[:pos]
            self.stopped = True
            self.stop_string = s

    def _release(self) -> str:
        if self.stopped:
            limit = len(self.text)
        else:
            hold = 0
            for s in self.stop:
                for k in range(min(len(s) - 1, len(self.text)), 0, -1):
                    if self.text.endswith(s[:k]):
                        hold = max(hold, k)
                        break
            limit = len(self.text) - hold
        out = self.text[self.released: limit] if limit > self.released else ""
        self.released = max(self.released, limit)
        return out

    def flush(self) -> str:
        """Release held-back text (call when generation ends for a reason other than a stop string)."""
        if not self.stopped and self.tok is not None and self.read_offset < len(self.ids):
            # tokens whose bytes never formed a complete character: emit them as replacement characters (same as a full decode)
            prefix = self._decode(self.ids[self.prefix_offset:self.read_offset])
            new = self._decode(self.ids[self.prefix_offset:])
            if len(new) > len(prefix):
                self.text += new[len(prefix):]
            self.prefix_offset = self.read_offset = len(self.ids)
        out = self.text[self.released:]
        self.released = len(self.text)
        return out
