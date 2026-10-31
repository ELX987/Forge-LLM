"""``distribution`` suite (EV-1): teacher-forced KL / top-1 flip rate / ECE / entropy shift / perplexity vs the parent."""
from __future__ import annotations

import numpy as np
from pydantic import Field

from forge.core.registry import register
from forge.core.schemas import Provenance
from forge.eval import stats
from forge.eval.metrics import compare_distributions, summarize_distribution
from forge.eval.suites.base import Suite, SuiteConfig, SuiteEnv, SuiteResult, read_jsonl, resolve_data_path


class DistributionConfig(SuiteConfig):
    """Distribution-level comparison on held-out text (teacher-forced)."""

    data_path: str | None = Field(
        default=None,
        description="jsonl with {'text': ...} or {'token_ids': [...]} per line, or a plain .txt (one document per blank-line "
        "separated block). Empty = the small bundled SEED set (smoke test only, not a benchmark).",
    )
    max_seq_len: int = Field(default=512, ge=2, le=1_000_000, description="Truncate every sequence to this many tokens.")
    batch_size: int = Field(default=4, ge=1, le=512, description="Sequences per forward pass.")
    min_tokens: int = Field(default=2, ge=2, description="Skip sequences shorter than this many tokens.")
    n_bins: int = Field(default=10, ge=2, le=100, description="Bins for the token-level ECE.")
    pad_id: int = Field(default=0, ge=0, description="Padding token id (never contributes to metrics).")


def load_sequences(cfg: DistributionConfig, env: SuiteEnv) -> list[list[int]]:
    path = resolve_data_path(cfg.data_path, "heldout_seed.jsonl")
    assert path is not None
    seqs: list[list[int]] = []
    if path.suffix == ".txt":
        blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
        rows = [{"text": b} for b in blocks]
    else:
        rows = read_jsonl(path)
    for r in rows:
        if "token_ids" in r:
            ids = [int(t) for t in r["token_ids"]]
        else:
            ids = env.tok.encode(str(r["text"]))
        ids = ids[: cfg.max_seq_len]
        if len(ids) >= cfg.min_tokens:
            seqs.append(ids)
        if cfg.max_items and len(seqs) >= cfg.max_items:
            break
    return seqs


@register("suite", "distribution", meta={"requirement": "EV-1"})
class DistributionSuite(Suite):
    """KL(parent||candidate) mean and p99, top-1 flip rate, ECE, entropy shift and perplexity ratio on held-out text."""

    name = "distribution"
    version = "1"
    title = "Distribution vs parent (KL / flip / ECE)"
    Config = DistributionConfig
    describes = "EV-1; Plan 10.1 metric definitions. Needs the parent for KL/flip/entropy shift."

    def run(self, cfg: DistributionConfig, env: SuiteEnv) -> SuiteResult:  # type: ignore[override]
        res = SuiteResult(self.name, self.version)
        seqs = load_sequences(cfg, env)
        if not seqs:
            res.notes.append("no sequences long enough; suite produced no metrics")
            return res
        cand = env.candidate
        if cand is None:
            raise ValueError("distribution suite needs a candidate model")
        parent = env.parent if env.parent is not None else cand
        has_parent = env.parent is not None
        bs = env.batch_size_override or cfg.batch_size
        d = compare_distributions(
            parent, cand, seqs, batch_size=bs, pad_id=cfg.pad_id, n_bins=cfg.n_bins,
            progress=lambda f: env.progress(f, "distribution"), should_stop=env.should_stop,
        )
        if d.n_tokens == 0:
            res.notes.append("stopped before any batch finished")
            return res
        pt = summarize_distribution(d, cfg.n_bins)
        n_seq, n_tok = d.n_sequences, d.n_tokens
        res.details.update({"n_sequences": n_seq, "n_tokens": n_tok, "max_abs_logit_diff": d.max_abs_logit_diff, "has_parent": has_parent})
        a, B, s = env.alpha, min(env.n_boot, 1000), env.seed

        # per-sequence cluster statistics (tokens within a sequence are correlated -> resample sequences)
        lens = np.array([len(x) for x in d.kl], dtype=np.float64)
        if has_parent:
            kl_sum = np.array([x.sum() for x in d.kl])
            e, lo, hi = stats.cluster_bootstrap_ratio(kl_sum, lens, B, alpha=a, seed=s)
            res.add("kl_mean", e, ci=(lo, hi), n=n_tok)
            e, lo, hi = stats.cluster_bootstrap_quantile(d.kl, 0.99, min(B, 200), alpha=a, seed=s)
            res.add("kl_p99", e, ci=(lo, hi), n=n_tok)
            fl_sum = np.array([x.sum() for x in d.flips], dtype=np.float64)
            e, lo, hi = stats.cluster_bootstrap_ratio(fl_sum, lens, B, alpha=a, seed=s + 1)
            res.add("top1_flip", e, ci=(lo, hi), n=n_tok)
            dh = np.array([(c - p).sum() for c, p in zip(d.h_cand, d.h_parent)])
            e, lo, hi = stats.cluster_bootstrap_ratio(dh, lens, B, alpha=a, seed=s + 2)
            res.add("entropy_shift", e, ci=(lo, hi), n=n_tok)
            res.flags["output_identical"] = bool(d.identical_logits)
            res.add("max_abs_logit_diff", d.max_abs_logit_diff, n=n_tok)
            res.boot["top1_flip"] = _boot_ratio(fl_sum, lens, B, s + 1)
            res.boot["kl_mean"] = _boot_ratio(kl_sum, lens, B, s)
        npc = np.array([x.sum() for x in d.nll_cand])
        e, lo, hi = stats.cluster_bootstrap_ratio(npc, lens, B, alpha=a, seed=s + 3)
        res.add("ppl_candidate", float(np.exp(e)), ci=(float(np.exp(lo)), float(np.exp(hi))), n=n_tok)
        ece_c = _ece_ci(d.conf_cand, d.correct_cand, cfg.n_bins, B, a, s + 4)
        res.add("ece_candidate", ece_c[0], ci=(ece_c[1], ece_c[2]), n=n_tok)
        if has_parent:
            npp = np.array([x.sum() for x in d.nll_parent])
            e, lo, hi = stats.cluster_bootstrap_ratio(npp, lens, B, alpha=a, seed=s + 5)
            res.add("ppl_parent", float(np.exp(e)), ci=(float(np.exp(lo)), float(np.exp(hi))), n=n_tok)
            diff = np.array([(c - p).sum() for c, p in zip(d.nll_cand, d.nll_parent)])
            e, lo, hi = stats.cluster_bootstrap_ratio(diff, lens, B, alpha=a, seed=s + 6)
            res.add("ppl_ratio", float(np.exp(e)), ci=(float(np.exp(lo)), float(np.exp(hi))), n=n_tok)
            ece_p = _ece_ci(d.conf_parent, d.correct_parent, cfg.n_bins, B, a, s + 7)
            res.add("ece_parent", ece_p[0], ci=(ece_p[1], ece_p[2]), n=n_tok)
            res.add("ece_delta", ece_c[0] - ece_p[0], n=n_tok)
        res.details["point_summary"] = pt
        if not has_parent:
            res.notes.append("no parent supplied: only candidate perplexity / ECE are reported")
        return res


def _boot_ratio(num: np.ndarray, den: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n, num.size))
    return num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-300)


def _ece_ci(conf: list, corr: list, n_bins: int, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    from forge.eval.metrics import ece_from_confidences

    est = ece_from_confidences(np.concatenate(conf), np.concatenate(corr), n_bins).ece
    rng = np.random.default_rng(seed)
    K = len(conf)
    vals = []
    for _ in range(min(n_boot, 200)):
        pick = rng.integers(0, K, K)
        vals.append(ece_from_confidences(np.concatenate([conf[j] for j in pick]), np.concatenate([corr[j] for j in pick]), n_bins).ece)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)
