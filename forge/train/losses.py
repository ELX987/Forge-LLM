"""Loss functions: causal-LM cross-entropy (+z-loss, chunked), distillation divergences (dense and top-k cached), DPO family.

Conventions: ``labels`` are aligned with ``logits`` (``labels[t]`` is the target of ``logits[t]``), ``-100`` is ignored, and
every function returns **sums** plus counts so callers can normalise over gradient-accumulation micro-batches exactly.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

__all__ = [
    "IGNORE_INDEX", "ce_loss_sum", "dpo_loss", "kd_loss_dense", "kd_token_divergence", "kd_loss_topk", "response_logprobs", "token_logprobs",
]

IGNORE_INDEX = -100


def _ce_chunk(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float, z_loss: float) -> tuple[torch.Tensor, torch.Tensor]:
    lf = logits.float()
    ce = Fn.cross_entropy(lf, labels, ignore_index=IGNORE_INDEX, reduction="sum", label_smoothing=label_smoothing)
    if z_loss > 0:
        lse = torch.logsumexp(lf, dim=-1)
        z = (lse.pow(2) * (labels != IGNORE_INDEX)).sum() * z_loss
    else:
        z = ce.new_zeros(())
    return ce, z


def ce_loss_sum(logits: torch.Tensor, labels: torch.Tensor, *, label_smoothing: float = 0.0, z_loss: float = 0.0,
                chunk_tokens: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Summed cross-entropy over non-ignored positions: ``(loss_sum_incl_z, n_tokens, {'ce_sum', 'z_sum'})``.

    ``chunk_tokens > 0`` evaluates the loss in token chunks under activation checkpointing (peak memory ~ one chunk of float32 logits).
    """
    V = logits.shape[-1]
    lg = logits.reshape(-1, V)
    lb = labels.reshape(-1)
    n = (lb != IGNORE_INDEX).sum()
    if chunk_tokens and lg.shape[0] > chunk_tokens:
        ce_tot = lg.new_zeros((), dtype=torch.float32)
        z_tot = lg.new_zeros((), dtype=torch.float32)
        for i in range(0, lg.shape[0], chunk_tokens):
            a, b = lg[i:i + chunk_tokens], lb[i:i + chunk_tokens]
            if torch.is_grad_enabled() and a.requires_grad:
                ce, z = checkpoint(_ce_chunk, a, b, label_smoothing, z_loss, use_reentrant=False)
            else:
                ce, z = _ce_chunk(a, b, label_smoothing, z_loss)
            ce_tot = ce_tot + ce
            z_tot = z_tot + z
    else:
        ce_tot, z_tot = _ce_chunk(lg, lb, label_smoothing, z_loss)
    return ce_tot + z_tot, n, {"ce_sum": ce_tot.detach(), "z_sum": z_tot.detach()}


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position log-probability of the label ``(logp [B,T], mask [B,T])``; ignored positions give 0."""
    mask = labels != IGNORE_INDEX
    safe = labels.clamp(min=0)
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return lp * mask, mask


def response_logprobs(logits: torch.Tensor, labels: torch.Tensor, average: bool = False) -> torch.Tensor:
    """Sequence log-probability of the supervised tokens ``[B]`` (sum, or mean when ``average``)."""
    lp, mask = token_logprobs(logits, labels)
    s = lp.sum(-1)
    return s / mask.sum(-1).clamp(min=1) if average else s


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1, loss_type: str = "sigmoid", label_smoothing: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-pair DPO-family loss ``[n_pairs]`` and reward diagnostics."""
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    if loss_type == "sigmoid":
        loss = -Fn.logsigmoid(beta * logits) * (1.0 - label_smoothing) - Fn.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "ipo":
        loss = (logits - 1.0 / (2.0 * beta)) ** 2
    elif loss_type == "hinge":
        loss = torch.relu(1.0 - beta * logits)
    else:
        raise ValueError(f"unknown DPO loss_type {loss_type!r}")
    cr = beta * (policy_chosen - ref_chosen).detach()
    rr = beta * (policy_rejected - ref_rejected).detach()
    return loss, {"reward_chosen": cr, "reward_rejected": rr, "reward_margin": cr - rr, "reward_acc": (cr > rr).float()}


# ---------------------------------------------------------------------------------------------------------------
# distillation
# ---------------------------------------------------------------------------------------------------------------
def _divergence(kind: str, lp_t: torch.Tensor, lp_s: torch.Tensor, beta: float) -> torch.Tensor:
    """Divergence over the last dim between log-distributions (teacher ``lp_t``, student ``lp_s``) -> shape [...]."""
    if kind == "forward_kl":
        return (lp_t.exp() * (lp_t - lp_s)).sum(-1)
    if kind == "reverse_kl":
        return (lp_s.exp() * (lp_s - lp_t)).sum(-1)
    if kind == "jsd":
        lb, l1b = math.log(beta), math.log(1.0 - beta)
        lm = torch.logsumexp(torch.stack([lp_t + lb, lp_s + l1b]), dim=0)
        kl_t = (lp_t.exp() * (lp_t - lm)).sum(-1)
        kl_s = (lp_s.exp() * (lp_s - lm)).sum(-1)
        return beta * kl_t + (1.0 - beta) * kl_s
    raise ValueError(f"unknown divergence {kind!r}")


def kd_token_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, kind: str = "forward_kl", temperature: float = 1.0,
                        jsd_beta: float = 0.5) -> torch.Tensor:
    """Per-position ``T^2 * D(teacher || student)`` over the full vocabulary, shape ``logits.shape[:-1]``."""
    T = float(temperature)
    lp_s = torch.log_softmax(student_logits.float() / T, dim=-1)
    lp_t = torch.log_softmax(teacher_logits.float() / T, dim=-1)
    return _divergence(kind, lp_t, lp_s, jsd_beta) * (T * T)


def kd_loss_dense(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor, kind: str = "forward_kl",
                  temperature: float = 1.0, jsd_beta: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum over masked positions of ``T^2 * D(teacher || student)`` with full-vocabulary distributions. Returns ``(sum, n_tokens)``."""
    d = kd_token_divergence(student_logits, teacher_logits, kind, temperature, jsd_beta)
    m = mask.to(d.dtype)
    return (d * m).sum(), m.sum()


def kd_loss_topk(student_logits: torch.Tensor, topk_idx: torch.Tensor, topk_logp: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor,
                 kind: str = "forward_kl", temperature: float = 1.0, jsd_beta: float = 0.5, tail_mode: str = "bin") -> tuple[torch.Tensor, torch.Tensor]:
    """Distillation against a cached top-k teacher distribution (Plan 8.5).

    ``topk_idx`` [B,T,k] (long), ``topk_logp`` [B,T,k] teacher log-probs at T=1, ``residual`` [B,T] = 1 - sum(top-k probs).
    ``tail_mode``:
      * ``bin``     -- the residual mass is one extra category; the student's own tail mass ``1 - sum_k q`` is its counterpart.  Valid
        for every divergence (a coarse-grained lower bound of the true divergence).
      * ``uniform`` -- forward KL only: the residual is spread uniformly over the V-k tail tokens (falls back to ``bin`` otherwise).
      * ``renorm``  -- ignore the tail, renormalise both distributions over the k cached tokens.
    ``temperature != 1`` re-tempers the cached log-probs (``p^(1/T)`` renormalised over the k bins) -- an approximation, since the
    tail is treated as a single bin; use T=1 with caches when exactness matters.
    """
    T = float(temperature)
    V = student_logits.shape[-1]
    k = topk_idx.shape[-1]
    ls_full = torch.log_softmax(student_logits.float() / T, dim=-1)
    ls_k = ls_full.gather(-1, topk_idx.long())
    lt_k = topk_logp.float()
    res = residual.float().clamp(min=0.0, max=1.0)
    eps = 1e-12
    if tail_mode == "uniform" and kind != "forward_kl":
        tail_mode = "bin"
    if tail_mode == "renorm":
        t = torch.log_softmax(lt_k / T, dim=-1)
        s = torch.log_softmax(ls_k, dim=-1)
        d = _divergence(kind, t, s, jsd_beta) * (T * T)
    else:
        lt_tail = torch.log(res.clamp(min=eps)).unsqueeze(-1)
        t_bins = torch.cat([lt_k, lt_tail], dim=-1)
        if T != 1.0:
            t_bins = torch.log_softmax(t_bins / T, dim=-1)
        # tail masses where the residual is ~0 should not create spurious terms
        # student tail log-mass computed directly (logsumexp over the non-cached tokens): "1 - sum(top-k)" cancels
        # catastrophically in fp32 once the student is confident, which corrupts the tail-bin term.
        floor = math.log(1e-30)  # only guards -inf (empty tail); the gradient of log(tail mass) is bounded, no need for a larger floor
        if k < V:
            s_tail_lp = ls_full.scatter(-1, topk_idx.long(), float("-inf")).logsumexp(-1).clamp(min=floor)
        else:
            s_tail_lp = ls_k.new_full(ls_k.shape[:-1], floor)
        s_bins = torch.cat([ls_k, s_tail_lp.unsqueeze(-1)], dim=-1)
        if tail_mode == "uniform":
            p_k = t_bins[..., :k].exp()
            r = t_bins[..., k].exp()
            n_tail = max(1, V - k)
            fwd_k = (p_k * (t_bins[..., :k] - ls_k)).sum(-1)
            sum_tail_ls = ls_full.sum(-1) - ls_k.sum(-1)
            fwd_tail = r * (torch.log(r.clamp(min=eps)) - math.log(n_tail)) - (r / n_tail) * sum_tail_ls
            d = (fwd_k + fwd_tail) * (T * T)
        else:
            d = _divergence(kind, t_bins, s_bins, jsd_beta) * (T * T)
    m = mask.to(d.dtype)
    return (d * m).sum(), m.sum()
