"""Speculative decoding (Plan SV-3 / SV-4, paper P7).

* exact **rejection sampling** verification (:func:`rejection_sample`, :func:`rejection_sample_batched`) - the emitted
  tokens are distributed exactly as autoregressive sampling from the target (P7 Alg. 1 / App. A.1);
* drafters: :class:`NgramDrafter` (prompt lookup, no model), :class:`DraftModelDrafter` (small ``ForgeLM`` with its own
  paged KV), :class:`EagleDrafter` + :class:`EagleHead` (feature-level head on target hidden states; training lives in
  ``forge.train`` - :func:`eagle_training_loss` is a minimal reference loss);
* :class:`SpecController` - load-aware choice of the draft length per Plan 7.5 pseudocode with the closed-form expected
  tokens of P7 (Prop. 4.1 / Cor. 4.2), per-position acceptance EMAs and 3 % hysteresis;
* :class:`StepCostModel` - roofline-style ``verify_time(tokens, kv_bytes)`` used by the controller and the UI simulator.

Tree verification (multi-candidate draft trees) is **not implemented**; chain drafts only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from forge.core.errors import ConfigError
from forge.serve.sampling import SamplingParams, sample_from_probs, sampling_probs

# --------------------------------------------------------------------------------------------------
# closed forms (P7 section 4)
# --------------------------------------------------------------------------------------------------


def expected_tokens(alphas: list[float] | tuple[float, ...], gamma: int) -> float:
    """P7 Prop. 4.1: ``E[N] = sum_{k=0..gamma} prod_{j<=k} alpha_j`` (empty product = 1).

    ``alphas[j]`` is the probability that draft position ``j`` is accepted given all previous ones were."""
    total, prod = 1.0, 1.0
    for j in range(gamma):
        a = alphas[j] if j < len(alphas) else (alphas[-1] if alphas else 0.0)
        prod *= a
        total += prod
    return total


def expected_tokens_iid(alpha: float, gamma: int) -> float:
    """P7 Cor. 4.2: ``(1 - alpha^(gamma+1)) / (1 - alpha)`` (``gamma + 1`` when alpha == 1)."""
    if gamma <= 0:
        return 1.0
    if alpha >= 1.0 - 1e-12:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup_iid(alpha: float, gamma: int, c: float) -> float:
    """P7 Prop. 4.3: ``S = E[N] / (1 + gamma * c)`` with ``c`` = draft step cost / target step cost."""
    return expected_tokens_iid(alpha, gamma) / (1.0 + gamma * c)


def optimal_gamma_iid(alpha: float, c: float, max_gamma: int = 16) -> int:
    best, best_s = 0, 1.0
    for g in range(1, max_gamma + 1):
        s = speedup_iid(alpha, g, c)
        if s > best_s:
            best, best_s = g, s
    return best


# --------------------------------------------------------------------------------------------------
# rejection sampling
# --------------------------------------------------------------------------------------------------
def rejection_sample_batched(p: Any, q: Any | None, draft: Any, u: Any | None = None,
                             generator: Any = None) -> tuple[Any, Any]:
    """Vectorised exact verification for ``N`` sequences with equal draft length ``G``.

    Args:
        p: ``[N, G+1, V]`` target distributions (position ``i`` conditioned on the prefix plus ``draft[:, :i]``).
        q: ``[N, G, V]`` draft distributions the draft tokens were sampled from, or ``None`` for deterministic
           (point-mass) drafts such as n-gram lookup.
        draft: ``[N, G]`` draft tokens.
        u: optional ``[N, G]`` uniforms in ``[0, 1)`` (drawn from ``generator`` otherwise).
    Returns ``(tokens [N, G+1] with -1 padding, n_accepted [N])``: the first ``n_accepted+1`` entries of each row are the
    emitted tokens (accepted drafts followed by the residual/bonus sample).
    """
    import torch

    n, g = draft.shape
    v = p.shape[-1]
    dev = p.device
    p = p.to(torch.float32)
    if g == 0:
        y = torch.multinomial(p[:, 0], 1, generator=generator).squeeze(-1)
        return y.unsqueeze(-1), torch.zeros(n, dtype=torch.long, device=dev)
    di = draft.long().unsqueeze(-1)
    px = p[:, :g].gather(2, di).squeeze(-1)
    if q is None:
        qx = torch.ones_like(px)
    else:
        q = q.to(torch.float32)
        qx = q.gather(2, di).squeeze(-1)
    accept_p = torch.where(qx > 0, (px / qx.clamp_min(1e-30)).clamp(max=1.0), torch.ones_like(px))
    if u is None:
        u = torch.rand((n, g), generator=generator, device=dev)
    acc = u < accept_p
    rej = ~acc
    any_rej = rej.any(dim=1)
    first = torch.where(any_rej, rej.to(torch.int8).argmax(dim=1), torch.full((n,), g, dtype=torch.long, device=dev))
    ar = torch.arange(n, device=dev)
    pn = p[ar, first]                                     # [N, V]
    if q is None:
        qn = torch.zeros_like(pn)
        has = first < g
        idx = draft.long()[ar, first.clamp(max=g - 1)]
        qn[ar[has], idx[has]] = 1.0
    else:
        qn = q[ar, first.clamp(max=g - 1)]
        qn = torch.where((first < g).unsqueeze(-1), qn, torch.zeros_like(qn))
    resid = (pn - qn).clamp_min(0.0)
    z = resid.sum(dim=-1, keepdim=True)
    dist = torch.where(z > 1e-12, resid / z.clamp_min(1e-30), pn)  # numerically p == q but rejected -> sample from p
    y = torch.multinomial(dist, 1, generator=generator).squeeze(-1)
    out = torch.full((n, g + 1), -1, dtype=torch.long, device=dev)
    pos = torch.arange(g + 1, device=dev).unsqueeze(0)
    keep = pos < first.unsqueeze(1)
    out[:, :g] = torch.where(keep[:, :g], draft.long(), out[:, :g])
    out[ar, first] = y
    _ = v
    return out, first


def rejection_sample(p: Any, q: Any | None, draft_tokens: list[int], generator: Any = None) -> tuple[list[int], int]:
    """Single-sequence exact verification. ``p [G+1, V]``, ``q [G, V]`` or None. Returns ``(emitted_tokens, n_accepted)``."""
    import torch

    g = len(draft_tokens)
    d = torch.tensor([draft_tokens], dtype=torch.long, device=p.device) if g else torch.zeros((1, 0), dtype=torch.long, device=p.device)
    out, n = rejection_sample_batched(p.unsqueeze(0), None if q is None else q.unsqueeze(0), d, None, generator)
    k = int(n[0])
    return [int(t) for t in out[0, : k + 1].tolist()], k


# --------------------------------------------------------------------------------------------------
# cost model & controller
# --------------------------------------------------------------------------------------------------
@dataclass
class StepCostModel:
    """Roofline step time in *bandwidth-normalised* units (bytes / bandwidth).

    ``verify_time(tokens, kv_bytes) = max((W + kv_bytes) / BW, 2 * params * tokens / (ridge * BW)) + overhead`` where
    ``ridge`` = peak FLOP/s divided by bandwidth (FLOP per byte). With ``bandwidth_bytes_s`` given the result is seconds.
    """

    weight_bytes: float
    params: float
    ridge_flop_per_byte: float = 250.0
    bandwidth_bytes_s: float = 1.0
    overhead: float = 0.0

    def verify_time(self, tokens: float, kv_bytes: float = 0.0) -> float:
        mem = (self.weight_bytes + kv_bytes) / self.bandwidth_bytes_s
        comp = 2.0 * self.params * tokens / (self.ridge_flop_per_byte * self.bandwidth_bytes_s)
        return max(mem, comp) + self.overhead

    def base_step_time(self, batch: int, kv_bytes: float = 0.0) -> float:
        return self.verify_time(batch, kv_bytes)


class AcceptanceEstimator:
    """Per-workload-class EMA of the conditional acceptance probability at each draft position."""

    def __init__(self, max_gamma: int, decay: float = 0.05, prior: float = 0.5) -> None:
        self.max_gamma = max_gamma
        self.decay = decay
        self.prior = prior
        self._a: dict[str, list[float]] = {}
        self._n: dict[str, list[int]] = {}

    def _ensure(self, wc: str) -> None:
        if wc not in self._a:
            self._a[wc] = [self.prior] * self.max_gamma
            self._n[wc] = [0] * self.max_gamma

    def observe(self, wc: str, proposed: int, accepted: int) -> None:
        """Position ``k`` is *observed* iff it was proposed and all earlier positions were accepted."""
        self._ensure(wc)
        a, n = self._a[wc], self._n[wc]
        for k in range(min(proposed, self.max_gamma)):
            if k > accepted:
                break
            x = 1.0 if accepted > k else 0.0
            rate = max(self.decay, 1.0 / (n[k] + 1))
            a[k] += rate * (x - a[k])
            n[k] += 1

    def per_position(self, wc: str = "default") -> list[float]:
        self._ensure(wc)
        return list(self._a[wc])

    def counts(self, wc: str = "default") -> list[int]:
        self._ensure(wc)
        return list(self._n[wc])


class SpecController:
    """Plan 7.5 ``choose_gamma``: pick the draft length maximising modelled speed-up at the current batch."""

    def __init__(self, max_gamma: int = 5, *, cost: StepCostModel | None = None, draft_cost_ratio: float = 0.05,
                 hysteresis: float = 0.03, ema_decay: float = 0.05, prior_acceptance: float = 0.5,
                 max_batch_for_spec: int = 64, fixed: bool = False) -> None:
        self.max_gamma = int(max_gamma)
        self.cost = cost
        self.c = float(draft_cost_ratio)
        self.hysteresis = float(hysteresis)
        self.max_batch = int(max_batch_for_spec)
        self.fixed = fixed
        self.est = AcceptanceEstimator(self.max_gamma, ema_decay, prior_acceptance)
        self._last: dict[str, int] = {}
        self.proposed_by_pos = [0] * self.max_gamma
        self.accepted_by_pos = [0] * self.max_gamma
        self.steps = 0
        self.tokens_emitted = 0

    # ---- statistics ------------------------------------------------------------------------------
    def observe(self, wc: str, proposed: int, accepted: int) -> None:
        self.est.observe(wc, proposed, accepted)
        for k in range(min(proposed, self.max_gamma)):
            self.proposed_by_pos[k] += 1
            if accepted > k:
                self.accepted_by_pos[k] += 1
        self.steps += 1
        self.tokens_emitted += accepted + 1

    def update_draft_cost(self, measured_ratio: float, ema: float = 0.1) -> None:
        """Feed a measured ``drafter time / target step time`` ratio."""
        if measured_ratio >= 0 and math.isfinite(measured_ratio):
            self.c += ema * (measured_ratio - self.c)

    def acceptance_by_position(self) -> list[float | None]:
        """Empirical conditional acceptance per draft position (accepted / proposed-and-reached)."""
        out: list[float | None] = []
        reached = list(self.proposed_by_pos)
        for k in range(self.max_gamma):
            out.append(self.accepted_by_pos[k] / reached[k] if reached[k] else None)
        return out

    # ---- modelling -------------------------------------------------------------------------------
    def modelled_gain(self, alphas: list[float], gamma: int, batch: int, kv_bytes: float = 0.0) -> float:
        """Modelled throughput ratio vs autoregressive decoding at this batch (1.0 = break-even)."""
        if gamma <= 0:
            return 1.0
        e_tok = expected_tokens(alphas, gamma)
        if self.cost is None:
            return e_tok / (1.0 + gamma * self.c)
        base = self.cost.base_step_time(batch, kv_bytes)
        t_step = self.cost.verify_time(batch * (gamma + 1), kv_bytes) + gamma * self.c * base
        return e_tok / (t_step / base)

    def gains(self, wc: str = "default", batch: int = 1, kv_bytes: float = 0.0, max_gamma: int | None = None) -> list[float]:
        a = self.est.per_position(wc)
        mg = min(self.max_gamma, max_gamma if max_gamma is not None else self.max_gamma)
        return [1.0] + [self.modelled_gain(a, g, batch, kv_bytes) for g in range(1, mg + 1)]

    def choose_gamma(self, wc: str = "default", batch: int = 1, kv_bytes: float = 0.0, max_gamma: int | None = None) -> int:
        """gamma = 0 is always allowed; a larger gamma must beat the incumbent by ``hysteresis`` (default 3 %)."""
        mg = self.max_gamma if max_gamma is None else min(max_gamma, self.max_gamma)
        if mg <= 0 or batch > self.max_batch:
            return 0
        if self.fixed:
            return mg
        a = self.est.per_position(wc)
        best, best_gain = 0, 1.0
        for g in range(1, mg + 1):
            gain = self.modelled_gain(a, g, batch, kv_bytes)
            if gain > best_gain * (1.0 + self.hysteresis):
                best, best_gain = g, gain
        self._last[wc] = best
        return best

    def stats(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens_per_step": (self.tokens_emitted / self.steps) if self.steps else None,
            "acceptance_by_position": self.acceptance_by_position(),
            "ema_acceptance_default": self.est.per_position("default"),
            "draft_cost_ratio": self.c,
            "proposed_by_position": list(self.proposed_by_pos),
            "accepted_by_position": list(self.accepted_by_pos),
            "overall_accept_rate": (sum(self.accepted_by_pos) / sum(self.proposed_by_pos)) if sum(self.proposed_by_pos) else None,
            "last_gamma": dict(self._last),
        }


# --------------------------------------------------------------------------------------------------
# drafters
# --------------------------------------------------------------------------------------------------
@dataclass
class DraftRequest:
    seq_id: str
    tokens: list[int]            # all committed tokens, last one is the pending (not yet in target KV) token
    gamma: int
    params: SamplingParams
    prompt_len: int
    generator: Any = None
    hidden: Any = None           # target hidden state of the last KV position (EAGLE)


@dataclass
class DraftProposal:
    tokens: list[int] = field(default_factory=list)
    probs: Any = None            # [len(tokens), V] draft distributions or None (point-mass drafts)


class Drafter(Protocol):  # pragma: no cover - protocol
    name: str
    needs_hidden: bool

    def propose(self, reqs: list[DraftRequest]) -> list[DraftProposal]: ...
    def release(self, seq_id: str) -> None: ...


class NgramDrafter:
    """Prompt-lookup drafting: find the most recent earlier occurrence of the longest suffix n-gram and propose the
    tokens that followed it. Costs no model compute; point-mass drafts (``probs=None``)."""

    name = "ngram"
    needs_hidden = False
    cost_ratio = 0.01

    def __init__(self, ngram_max: int = 3, ngram_min: int = 1, window: int = 4096) -> None:
        if ngram_min > ngram_max:
            raise ConfigError("ngram_min must be <= ngram_max")
        self.nmax, self.nmin, self.window = ngram_max, ngram_min, window

    def lookup(self, tokens: list[int], gamma: int) -> list[int]:
        hist = tokens[-self.window:] if len(tokens) > self.window else tokens
        n_hist = len(hist)
        for n in range(min(self.nmax, n_hist - 1), self.nmin - 1, -1):
            suffix = hist[n_hist - n:]
            first = suffix[0]
            best: list[int] = []
            for i in range(n_hist - n - 1, -1, -1):  # most recent occurrence first
                if hist[i] == first and hist[i:i + n] == suffix:
                    cont = hist[i + n: i + n + gamma]
                    if len(cont) > len(best):
                        best = cont
                        if len(best) >= gamma:
                            break
            if best:
                return list(best)
        return []

    def propose(self, reqs: list[DraftRequest]) -> list[DraftProposal]:
        return [DraftProposal(self.lookup(r.tokens, r.gamma) if r.gamma > 0 else [], None) for r in reqs]

    def release(self, seq_id: str) -> None:
        return None


class DraftModelDrafter:
    """Small causal LM as drafter. Owns a private paged KV pool + manager (no prefix caching) and re-uses the longest
    common prefix of its cached tokens with the sequence each call, so accepted drafts are not recomputed."""

    name = "draft"
    needs_hidden = False

    def __init__(self, model: Any, *, device: Any, compute_dtype: Any, kv_name: str, block_size: int, num_blocks: int,
                 max_model_len: int, model_cfg: Any | None = None, kernel_attention: str = "auto",
                 expect_vocab: int | None = None) -> None:
        from forge.serve.kv import KVBlockManager
        from forge.serve.runner import KVPool, ModelAdapter, ModelRunner

        ad = ModelAdapter(model, model_cfg)
        if expect_vocab is not None and ad.dims.vocab_size != expect_vocab:
            raise ConfigError(f"draft model vocab {ad.dims.vocab_size} != target vocab {expect_vocab}")
        d = ad.dims
        self.pool = KVPool(num_blocks, block_size, d.num_layers, d.num_kv_heads, d.head_dim, compute_dtype, kv_name, device)
        self.manager = KVBlockManager(num_blocks, block_size, enable_prefix_caching=False, bytes_per_block=self.pool.bytes_per_block)
        self.runner = ModelRunner(model, self.pool, model_cfg=model_cfg, max_model_len=max_model_len, kernel_attention=kernel_attention)
        self.max_model_len = max_model_len
        self.kv_tokens: dict[str, list[int]] = {}
        self.weight_bytes = _module_bytes(model)

    def release(self, seq_id: str) -> None:
        if self.manager.has_seq(seq_id):
            self.manager.free(seq_id)
        self.kv_tokens.pop(seq_id, None)

    def propose(self, reqs: list[DraftRequest]) -> list[DraftProposal]:
        import torch

        from forge.serve.runner import SeqInput

        results = [DraftProposal() for _ in reqs]
        active: list[int] = []
        for i, r in enumerate(reqs):
            if r.gamma <= 0 or len(r.tokens) + r.gamma >= self.max_model_len:
                continue
            sid = r.seq_id
            if not self.manager.has_seq(sid):
                self.manager.allocate(sid, [], total_slots=0)
                self.kv_tokens[sid] = []
            kvt = self.kv_tokens[sid]
            lcp = 0
            m = min(len(kvt), len(r.tokens) - 1)
            while lcp < m and kvt[lcp] == r.tokens[lcp]:
                lcp += 1
            if lcp < self.manager.num_slots(sid):
                self.manager.rollback(sid, lcp)
            del kvt[lcp:]
            if self.manager.reserve(sid, len(r.tokens)) is None:
                self.release(sid)
                continue
            active.append(i)
        if not active:
            return results

        drafts: dict[int, list[int]] = {i: [] for i in active}
        qs: dict[int, list[Any]] = {i: [] for i in active}
        gam = {i: reqs[i].gamma for i in active}
        feed_tokens: dict[int, list[int]] = {}
        feed_start: dict[int, int] = {}
        for i in active:
            r = reqs[i]
            lcp = len(self.kv_tokens[r.seq_id])
            feed_tokens[i] = list(r.tokens[lcp:])
            feed_start[i] = lcp
        with torch.no_grad():
            for _ in range(max(gam.values())):
                cur = [i for i in active if len(drafts[i]) < gam[i]]
                if not cur:
                    break
                inputs = [SeqInput(feed_tokens[i], feed_start[i], self.manager.block_table(reqs[i].seq_id), -1) for i in cur]
                out = self.runner.forward(inputs)
                for idx, i in enumerate(cur):
                    r = reqs[i]
                    row = out.logits[idx][-1]
                    probs = sampling_probs(row, r.params, r.tokens[:r.prompt_len], r.tokens[r.prompt_len:] + drafts[i])
                    tok = sample_from_probs(probs, r.generator)
                    drafts[i].append(tok)
                    qs[i].append(probs)
                    fed_end = feed_start[i] + len(feed_tokens[i])
                    self.kv_tokens[r.seq_id] = (r.tokens + drafts[i][:-1])[:fed_end]
                    feed_tokens[i] = [tok]
                    feed_start[i] = fed_end
                    if len(drafts[i]) < gam[i] and self.manager.reserve(r.seq_id, fed_end + 1) is None:
                        gam[i] = len(drafts[i])  # out of draft blocks: stop drafting this sequence
        for i in active:
            results[i] = DraftProposal(drafts[i], torch.stack(qs[i]) if qs[i] else None)
        return results


def _module_bytes(model: Any) -> int:
    try:
        return int(sum(p.numel() * p.element_size() for p in model.parameters()) +
                   sum(b.numel() * b.element_size() for b in model.buffers()))
    except Exception:  # pragma: no cover
        return 0


# --------------------------------------------------------------------------------------------------
# EAGLE-style feature drafter
# --------------------------------------------------------------------------------------------------
def _make_eagle_head_cls():
    import torch
    import torch.nn as nn

    class EagleHead(nn.Module):
        """``f_next = head(f_prev, emb(token))``: fuse previous target feature with the next token embedding and refine
        with residual MLP blocks. The (frozen) target ``lm_head`` turns the predicted feature into draft logits."""

        def __init__(self, hidden_size: int, num_blocks: int = 1, expansion: int = 2) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_blocks = num_blocks
            self.expansion = expansion
            self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_blocks)])
            self.up = nn.ModuleList([nn.Linear(hidden_size, expansion * hidden_size, bias=False) for _ in range(num_blocks)])
            self.down = nn.ModuleList([nn.Linear(expansion * hidden_size, hidden_size, bias=False) for _ in range(num_blocks)])

        def forward(self, feature: torch.Tensor, token_emb: torch.Tensor) -> torch.Tensor:
            x = self.fc(torch.cat([token_emb, feature], dim=-1))
            for norm, up, down in zip(self.norms, self.up, self.down):
                x = x + down(torch.nn.functional.silu(up(norm(x))))
            return x

        def save(self, path: str) -> None:
            torch.save({"config": {"hidden_size": self.hidden_size, "num_blocks": self.num_blocks, "expansion": self.expansion},
                        "state": self.state_dict()}, path)

        @classmethod
        def load(cls, path: str, map_location: Any = "cpu") -> EagleHead:
            blob = torch.load(path, map_location=map_location, weights_only=True)
            head = cls(**blob["config"])
            head.load_state_dict(blob["state"])
            return head

    return EagleHead


_EAGLE_CLS: Any = None


def __getattr__(name: str) -> Any:  # lazy: torch is imported only when EagleHead is first requested
    global _EAGLE_CLS
    if name == "EagleHead":
        if _EAGLE_CLS is None:
            _EAGLE_CLS = _make_eagle_head_cls()
        return _EAGLE_CLS
    raise AttributeError(name)


def eagle_training_loss(head: Any, hidden: Any, input_ids: Any, embed: Any, lm_head: Any, ce_weight: float = 0.1) -> Any:
    """Reference loss for training an :class:`EagleHead`.

    ``hidden [B, T, H]`` post-norm target features, ``input_ids [B, T]``. Predicts ``hidden[:, t+1]`` from
    ``(hidden[:, t], embed(input_ids[:, t+1]))`` (smooth-L1) plus cross-entropy of ``lm_head(pred)`` against
    ``input_ids[:, t+2]``."""
    import torch.nn.functional as F

    f_prev = hidden[:, :-1]
    emb = embed(input_ids[:, 1:])
    pred = head(f_prev, emb)
    reg = F.smooth_l1_loss(pred, hidden[:, 1:].detach())
    if input_ids.shape[1] < 3:
        return reg
    logits = lm_head(pred[:, :-1])
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), input_ids[:, 2:].reshape(-1))
    return reg + ce_weight * ce


class EagleDrafter:
    """Drafts ``gamma`` tokens by rolling the :class:`EagleHead` over predicted features (needs target hidden states)."""

    name = "eagle"
    needs_hidden = True
    cost_ratio = 0.03

    def __init__(self, head: Any, embed: Any, lm_head_fn: Any) -> None:
        self.head, self.embed, self.lm_head_fn = head, embed, lm_head_fn

    def release(self, seq_id: str) -> None:
        return None

    def propose(self, reqs: list[DraftRequest]) -> list[DraftProposal]:
        import torch

        results = [DraftProposal() for _ in reqs]
        idx = [i for i, r in enumerate(reqs) if r.gamma > 0 and r.hidden is not None]
        if not idx:
            return results
        dev = reqs[idx[0]].hidden.device
        with torch.no_grad():
            feat = torch.stack([reqs[i].hidden.reshape(-1) for i in idx]).to(dev)
            tok = torch.tensor([reqs[i].tokens[-1] for i in idx], dtype=torch.long, device=dev)
            drafts: dict[int, list[int]] = {i: [] for i in idx}
            qs: dict[int, list[Any]] = {i: [] for i in idx}
            for _ in range(max(reqs[i].gamma for i in idx)):
                emb = self.embed(tok)
                if isinstance(emb, tuple):
                    emb = emb[0]
                feat = self.head(feat.to(emb.dtype), emb)
                logits = self.lm_head_fn(feat).float()
                new_tok = []
                for j, i in enumerate(idx):
                    r = reqs[i]
                    if len(drafts[i]) >= r.gamma:
                        new_tok.append(int(tok[j]))
                        continue
                    probs = sampling_probs(logits[j], r.params, r.tokens[:r.prompt_len], r.tokens[r.prompt_len:] + drafts[i])
                    t = sample_from_probs(probs, r.generator)
                    drafts[i].append(t)
                    qs[i].append(probs)
                    new_tok.append(t)
                tok = torch.tensor(new_tok, dtype=torch.long, device=dev)
        for i in idx:
            results[i] = DraftProposal(drafts[i], torch.stack(qs[i]) if qs[i] else None)
        return results


class CompositeDrafter:
    """``auto`` mode: n-gram first; fall back to a model-based drafter when n-gram finds nothing."""

    name = "auto"

    def __init__(self, primary: Any, fallback: Any | None) -> None:
        self.primary, self.fallback = primary, fallback

    @property
    def needs_hidden(self) -> bool:
        return bool(self.fallback is not None and self.fallback.needs_hidden)

    def propose(self, reqs: list[DraftRequest]) -> list[DraftProposal]:
        out = self.primary.propose(reqs)
        if self.fallback is None:
            return out
        todo = [i for i, o in enumerate(out) if not o.tokens and reqs[i].gamma > 0]
        if todo:
            sub = self.fallback.propose([reqs[i] for i in todo])
            for i, s in zip(todo, sub):
                out[i] = s
        return out

    def release(self, seq_id: str) -> None:
        self.primary.release(seq_id)
        if self.fallback is not None:
            self.fallback.release(seq_id)
