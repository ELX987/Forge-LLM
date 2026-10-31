"""Optimisers (adamw / adamw8bit-if-available / lion / sgd) and learning-rate schedules (constant / cosine / linear / wsd)."""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

from forge.train.config import OptimConfig, ScheduleConfig

log = logging.getLogger("forge.train.optim")

__all__ = ["Lion", "anneal_start_step", "build_optimizer", "lr_factor", "resolve_warmup", "set_lr"]


# ---------------------------------------------------------------------------------------------------------------
# schedules (pure functions)
# ---------------------------------------------------------------------------------------------------------------
def resolve_warmup(s: ScheduleConfig, total_steps: int) -> int:
    w = int(round(s.warmup_ratio * total_steps)) if s.warmup_ratio is not None else int(s.warmup_steps)
    return max(0, min(w, max(0, total_steps - 1)))


def anneal_start_step(s: ScheduleConfig, total_steps: int) -> int | None:
    """Step at which the LR starts decaying in a way worth a 'pre-anneal' checkpoint (wsd only; None otherwise)."""
    if s.name != "wsd":
        return None
    return max(resolve_warmup(s, total_steps), int(round(total_steps * (1.0 - s.decay_ratio))))


def lr_factor(step: int, s: ScheduleConfig, total_steps: int) -> float:
    """Multiplier of the peak LR at optimiser step ``step`` (0-based)."""
    total_steps = max(1, int(total_steps))
    warm = resolve_warmup(s, total_steps)
    if warm > 0 and step < warm:
        return s.warmup_start_ratio + (1.0 - s.warmup_start_ratio) * min(1.0, (step + 1) / warm)
    lo = float(s.min_lr_ratio)
    if s.name == "constant":
        return 1.0
    if s.name in ("cosine", "linear"):
        p = min(1.0, max(0.0, (step - warm) / max(1, total_steps - warm)))
        if s.name == "cosine":
            return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * p))
        return 1.0 - (1.0 - lo) * p
    if s.name == "wsd":
        d0 = anneal_start_step(s, total_steps)
        assert d0 is not None
        if step < d0:
            return 1.0
        q = min(1.0, max(0.0, (step - d0) / max(1, total_steps - d0)))
        if s.wsd_decay_style == "linear":
            return 1.0 - (1.0 - lo) * q
        if s.wsd_decay_style == "cosine":
            return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * q))
        if s.wsd_decay_style == "sqrt":
            return lo + (1.0 - lo) * (1.0 - math.sqrt(q))
        return max(lo, 1e-8) ** q
    raise ValueError(f"unknown schedule {s.name!r}")


def set_lr(optimizer: Any, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)


# ---------------------------------------------------------------------------------------------------------------
# optimisers
# ---------------------------------------------------------------------------------------------------------------
def _make_lion() -> Any:
    import torch

    class _Lion(torch.optim.Optimizer):
        """Lion (Chen et al. 2023): sign of an interpolated momentum, decoupled weight decay."""

        def __init__(self, params: Iterable[Any], lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0) -> None:
            super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure: Any = None) -> Any:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for g in self.param_groups:
                b1, b2 = g["betas"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    m = st["exp_avg"]
                    if g["weight_decay"]:
                        p.mul_(1.0 - g["lr"] * g["weight_decay"])
                    update = (m.mul(b1).add(p.grad, alpha=1.0 - b1)).sign_()
                    p.add_(update, alpha=-g["lr"])
                    m.mul_(b2).add_(p.grad, alpha=1.0 - b2)
            return loss

    return _Lion


_LION_CLS: Any = None


def Lion(*args: Any, **kwargs: Any) -> Any:  # noqa: N802 - class-like factory (torch imported lazily)
    global _LION_CLS
    if _LION_CLS is None:
        _LION_CLS = _make_lion()
    return _LION_CLS(*args, **kwargs)


def build_param_groups(named_params: Iterable[tuple[str, Any]], cfg: OptimConfig) -> list[dict[str, Any]]:
    """Groups by (decay?, lr multiplier). Only ``requires_grad`` parameters are included."""
    buckets: dict[tuple[bool, float], list[Any]] = {}
    for name, p in named_params:
        if not p.requires_grad:
            continue
        lname = name.lower()
        decay = p.ndim >= 2 and not any(pat in lname for pat in cfg.no_decay_patterns)
        mult = 1.0
        for pat, m in cfg.lr_multipliers.items():
            if pat in lname:
                mult = float(m)
                break
        buckets.setdefault((decay, mult), []).append(p)
    groups = []
    for (decay, mult), ps in sorted(buckets.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
        groups.append({"params": ps, "weight_decay": cfg.weight_decay if decay else 0.0, "lr_mult": mult})
    return groups


def build_optimizer(named_params: Iterable[tuple[str, Any]], cfg: OptimConfig, *, cuda: bool = False) -> Any:
    import torch

    groups = build_param_groups(list(named_params), cfg)
    if not groups:
        raise ValueError("no trainable parameters")
    betas = (float(cfg.betas[0]), float(cfg.betas[1]))
    if cfg.name == "adamw":
        kw: dict[str, Any] = dict(lr=cfg.lr, betas=betas, eps=cfg.eps)
        if cfg.fused and cuda:
            kw["fused"] = True
        try:
            return torch.optim.AdamW(groups, **kw)
        except (RuntimeError, TypeError):
            kw.pop("fused", None)
            return torch.optim.AdamW(groups, **kw)
    if cfg.name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(groups, lr=cfg.lr, betas=betas, eps=cfg.eps)
        except Exception as e:  # noqa: BLE001
            log.warning("adamw8bit requested but bitsandbytes is unavailable (%r); falling back to adamw", e)
            return torch.optim.AdamW(groups, lr=cfg.lr, betas=betas, eps=cfg.eps)
    if cfg.name == "lion":
        return Lion(groups, lr=cfg.lr, betas=betas)
    if cfg.name == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov and cfg.momentum > 0)
    raise ValueError(f"unknown optimiser {cfg.name!r}")
