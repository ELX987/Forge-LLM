"""Ternary quantisation-aware training as a precision plug-in (``precision.quant_aware = ternary``).

:class:`TernaryPrecisionManager` has the same interface as :class:`forge.train.precision.PrecisionManager` and converts eligible
``nn.Linear`` modules by swapping their class to :class:`forge.ternary.linear.FakeQuantTernaryLinear` (so parameter names, optimiser
state and checkpoints are unchanged; :meth:`restore` reverses it before export).  The latent weights are the ordinary fp ``weight``
Parameters trained by the normal optimiser; the scale is recomputed every forward and never learned.

Health diagnostics (metric prefix ``ternary``) are sampled on the trainer's ``diagnostics_every`` cadence:
``ternary/zero_frac`` (+min/max), ``ternary/flip_rate`` (+max; fraction of codes changed per step, averaged over the steps since the
previous sample), ``ternary/scale_mean``, ``ternary/lam``.  Thresholds default to the tern158 report values (TARGET until re-measured
in Forge): healthy zero fraction 0.40-0.52, warn 0.60, stop 0.80.

EXPERIMENTAL: CPU-validated only; the sharded ``zero_ternary`` trainer is :mod:`forge.ternary.zero` (docs/ZERO_TERNARY.md); bf16-stochastic-rounded optimiser state is a later phase.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError
from forge.formats import ternary as tn
from forge.train.config import PrecisionConfig
from forge.train.precision import PrecisionManager, _num_layers, is_exempt

from .linear import FakeQuantTernaryLinear

log = logging.getLogger("forge.ternary.qat")

_ATTRS = ("group_size", "rule", "iters", "quant_acts", "lam", "enabled", "_cache", "_forge_t158", "_forge_orig_cls",
          "_enabled", "_zero", "_zero_q", "_zero_s")  # the last four belong to forge.ternary.zero.ZeroTernaryLinear


class TernaryPrecisionManager(PrecisionManager):
    def __init__(self, cfg: PrecisionConfig, **kw: Any) -> None:
        if cfg.rung != "R0":
            raise ConfigError("precision.quant_aware=ternary requires precision.rung=R0 (ternary QAT replaces the FP8/FP4 emulation)")
        super().__init__(cfg, **kw)
        self.tc = cfg.ternary
        self.backend = "ternary-qat"
        self._step = 0
        self._prev: dict[str, torch.Tensor] = {}
        self._prev_step = 0
        self._events: list[tuple[str, str]] = []
        self._warned: set[str] = set()

    # -- interface -------------------------------------------------------------------------------------
    @property
    def metric_prefix(self) -> str:
        return "ternary"

    @property
    def healing_start(self) -> int | None:
        return None

    @property
    def wants_diagnostics(self) -> bool:
        return bool(self._quant_names)

    @property
    def wants_loss_gap(self) -> bool:
        return bool(self._quant_names)

    def _current_mode(self) -> str:
        return "ternary" if self._hp_depth == 0 else "off"

    def _ternary_modules(self) -> list[tuple[str, FakeQuantTernaryLinear]]:
        if self._model is None:
            return []
        return [(n, m) for n, m in self._model.named_modules() if getattr(m, "_forge_t158", False)]

    def prepare(self, model: nn.Module) -> dict[str, Any]:
        self._model = model
        tc, cfg = self.tc, self.cfg
        names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        n_layers = _num_layers(names)
        exempt, skipped, eligible = [], [], []
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear) or isinstance(m, FakeQuantTernaryLinear):
                continue
            if "lora_" in n or ".lora" in n or is_exempt(n, cfg, n_layers):
                exempt.append(n)
            elif m.in_features % tn.CODES_PER_WORD or (tc.group_size > 0 and m.in_features % tc.group_size) or m.out_features < 16:
                skipped.append(n)
            else:
                eligible.append(n)
        for n, m in model.named_modules():
            if n in eligible:
                m._forge_orig_cls = type(m)
                m.__class__ = FakeQuantTernaryLinear
                m._forge_t158 = True
                m.group_size, m.rule, m.iters, m.quant_acts = tc.group_size, tc.rule, tc.iters, tc.quant_acts
                m.lam, m.enabled, m._cache = 1.0, True, None
                self._quant_names.append(n)
        if not eligible:
            raise ConfigError("ternary QAT found no eligible linear layers (need in_features % 16 == 0 and group_size | in_features; "
                              "check precision.exempt_* settings)")
        self.summary = {"rung": cfg.rung, "quant_aware": "ternary", "n_linear": len(names), "n_layers": n_layers, "eligible": len(eligible),
                        "exempt": len(exempt), "skipped_shape": len(skipped), "backend": self.backend, "quantised": len(eligible),
                        "group_size": tc.group_size, "rule": tc.rule, "warmup_steps": tc.warmup_steps, "exempt_names": exempt[:50]}
        self._set_lam(tn.warmup_lambda(0, tc.warmup_steps))
        log.warning("ternary QAT is EXPERIMENTAL (validated on CPU only): %d layers quantised, %d exempt, %d skipped by shape",
                    len(eligible), len(exempt), len(skipped))
        return self.summary

    def _set_lam(self, lam: float) -> None:
        for _, m in self._ternary_modules():
            m.set_lam(lam)

    def set_step(self, step: int) -> bool:
        self._step = int(step)
        if self.tc.warmup_steps > 0 and self._hp_depth == 0:
            self._set_lam(tn.warmup_lambda(step, self.tc.warmup_steps))
        return False

    @contextlib.contextmanager
    def high_precision(self):  # type: ignore[no-untyped-def]
        mods = [m for _, m in self._ternary_modules()]
        self._hp_depth += 1
        for m in mods:
            m.enabled = False
        try:
            yield
        finally:
            self._hp_depth -= 1
            for m in mods:
                m.enabled = True

    def restore(self, model: nn.Module | None = None) -> int:
        n = 0
        for _, m in (model or self._model).named_modules():
            if getattr(m, "_forge_t158", False):
                m.__class__ = getattr(m, "_forge_orig_cls", nn.Linear)
                for a in _ATTRS:
                    m.__dict__.pop(a, None)
                n += 1
        self._quant_names = []
        return n

    # -- health ----------------------------------------------------------------------------------------
    def drain_events(self) -> list[tuple[str, str]]:
        ev, self._events = self._events, []
        return ev

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._events.append(("warning", msg))

    def diagnostics(self, model: nn.Module) -> dict[str, float]:
        mods = [(n, m) for n, m in self._ternary_modules() if m.weight.ndim == 2]
        if not mods:
            return {}
        k = min(self.cfg.diag_max_tensors, len(mods))
        sample = mods[:: max(1, len(mods) // k)][:k]
        zs, flips, scales = [], [], []
        elapsed = max(1, self._step - self._prev_step)
        new_prev: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, m in sample:
                t, s = m.codes()
                zs.append(float((t == 0).float().mean()))
                scales.append(float(s.mean()))
                prev = self._prev.get(n)
                if prev is not None and prev.shape == t.shape:
                    flips.append(float((prev != t.cpu()).float().mean()) / elapsed)
                new_prev[n] = t.cpu()
        self._prev, self._prev_step = new_prev, self._step
        p = self.metric_prefix
        out = {f"{p}/zero_frac": sum(zs) / len(zs), f"{p}/zero_frac_min": min(zs), f"{p}/zero_frac_max": max(zs),
               f"{p}/scale_mean": sum(scales) / len(scales), f"{p}/lam": float(sample[0][1].lam)}
        if flips:
            out[f"{p}/flip_rate"] = sum(flips) / len(flips)
            out[f"{p}/flip_rate_max"] = max(flips)
        tc = self.tc
        z = out[f"{p}/zero_frac"]
        if z > tc.collapse_stop:
            msg = f"ternary zero-density collapse: {z:.1%} of sampled weights are zero (> {tc.collapse_stop:.0%})"
            self._events.append(("error" if tc.stop_on_collapse else "warning", msg))
        elif z > tc.collapse_warn:
            self._warn_once("zero_warn", f"ternary zero density {z:.1%} exceeds {tc.collapse_warn:.0%} (healthy: 40-52%)")
        if flips and out[f"{p}/flip_rate"] > tc.flip_warn:
            fr = out[f"{p}/flip_rate"]
            self._warn_once("flip_warn", f"ternary flip rate {fr:.2%}/step exceeds {tc.flip_warn:.0%}: updates are noise-dominated (lower lr)")
        return out
