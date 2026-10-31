"""The shared training engine behind pre-training, fine-tuning and distillation (Plan 3.4, 7.4).

``Trainer`` runs an optimiser loop over a :class:`Task` (which supplies micro-batches and the loss):

* autocast (bf16 / fp16 + dynamic grad scaler / fp32), gradient accumulation with exact token-weighted normalisation,
  global-norm clipping, schedules (:mod:`forge.train.optim`), DDP / FSDP wrapping (:mod:`forge.train.dist`),
  activation checkpointing (non-invasive ``forward`` patch), optional ``torch.compile``
* precision ladder R0/R1/R2 + healing phase + TR-2 diagnostics (:mod:`forge.train.precision`)
* NaN/Inf and loss/grad-norm spike detection, automatic rollback to the last good checkpoint with data skip + LR cool-down
* stateless deterministic data order (``task.get_batches(step)``), exact resume of model / optimiser / scaler / RNG / counters
* held-out evaluation, early stopping, best / pre-anneal / final checkpoints, cooperative cancel, time / token / step budgets
* metrics: ``loss, lr, grad_norm, tokens_per_s, tokens_seen, step_time_ms, gpu_mem_gb, mfu, update_weight_ratio`` (+ task metrics,
  TR-2 ``fp4/*`` diagnostics, ``restarts``, ``skipped_steps``)
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from forge.core.errors import ConfigError, NotFound
from forge.train import dist as D
from forge.train.checkpoint import CheckpointManager
from forge.train.config import CommonTrainConfig
from forge.train.data import StepPrefetcher
from forge.train.losses import IGNORE_INDEX, ce_loss_sum
from forge.train.optim import anneal_start_step, build_optimizer, lr_factor, set_lr
from forge.train.precision import PrecisionManager
from forge.train.stability import SpikeDetector, TrainingDiverged

log = logging.getLogger("forge.train.trainer")

__all__ = [
    "LMTask", "StepLoss", "Task", "Trainer", "enable_activation_checkpointing", "estimate_flops_per_token", "forward_logits",
    "load_reference_losses", "peak_tflops_for", "split_batch",
]


# ---------------------------------------------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class StepLoss:
    """Result of one micro-batch: ``loss_sum`` is back-propagated after dividing by the step's total ``items``."""

    loss_sum: torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)  # additional *sums* (normalised by items like the loss)


def forward_logits(model: nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run ``model(input_ids)`` and return ``(logits, aux_loss_or_None)`` for HF-style / Forge outputs."""
    out = model(input_ids)
    if hasattr(out, "logits"):
        return out.logits, getattr(out, "aux_loss", None)
    if isinstance(out, dict):
        return out["logits"], out.get("aux_loss")
    if isinstance(out, (tuple, list)):
        return out[0], None
    return out, None


def split_batch(batch: dict[str, Any], micro: int) -> list[dict[str, Any]]:
    """Split a dict of tensors along dim 0 into chunks of ``micro`` rows (non-tensor entries are copied)."""
    n = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim >= 1)
    out = []
    for i in range(0, n, micro):
        out.append({k: (v[i:i + micro] if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n else v) for k, v in batch.items()})
    return out


class Task:
    """Interface between the trainer and a data/loss definition."""

    name = "task"
    seq_len_hint = 0

    def get_batches(self, step: int, rank: int, world: int) -> list[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def count(self, batch: dict[str, Any]) -> float:
        """Number of loss items (tokens) in a micro-batch -- normaliser across the accumulation window."""
        return float((batch["labels"] != IGNORE_INDEX).sum())

    def tokens_in_batch(self, batch: dict[str, Any]) -> int:
        return int(batch["input_ids"].numel())

    def loss(self, model: nn.Module, batch: dict[str, Any], trainer: Trainer) -> StepLoss:  # pragma: no cover - interface
        raise NotImplementedError

    def eval_batches(self) -> list[dict[str, Any]]:
        return []

    def steps_per_epoch(self, global_batch: int) -> float | None:
        return None

    def tokens_per_step(self, global_batch: int) -> float | None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass

    def extra_metrics(self) -> dict[str, float]:
        """Additional scalars merged into the next emitted metric record (consumed once)."""
        return {}

    def describe(self) -> dict[str, Any]:
        return {}


class LMTask(Task):
    """Causal-LM cross-entropy over a dataset exposing ``batch_at(step, global_batch, rank, world)`` and ``eval_batches``."""

    name = "lm"

    def __init__(self, dataset: Any, cfg: CommonTrainConfig, val_dataset: Any | None = None, seq_len: int = 0, num_workers: int = 0,
                 prefetch_steps: int = 2) -> None:
        self.ds, self.val_ds, self.cfg = dataset, val_dataset, cfg
        self.num_workers, self.prefetch_steps = num_workers, prefetch_steps
        self.seq_len_hint = seq_len
        self._eval: list[dict[str, Any]] | None = None

    def get_batches(self, step: int, rank: int, world: int) -> list[dict[str, Any]]:
        mb, acc = self.cfg.micro_batch_size, self.cfg.grad_accum_steps
        b = self.ds.batch_at(step, mb * acc * world, rank, world)
        return split_batch(b, mb)

    def loss(self, model: nn.Module, batch: dict[str, Any], trainer: Trainer) -> StepLoss:
        c = self.cfg
        logits, aux = forward_logits(model, batch["input_ids"])
        n_el = logits.shape[0] * logits.shape[1] * logits.shape[2]
        chunk = 4096 if n_el > 3e8 else 0
        total, n, parts = ce_loss_sum(logits, batch["labels"], label_smoothing=c.label_smoothing, z_loss=c.z_loss, chunk_tokens=chunk)
        m: dict[str, Any] = {"ce": parts["ce_sum"]}
        if c.z_loss > 0:
            m["z_loss"] = parts["z_sum"]
        if aux is not None and c.aux_loss_coef > 0:
            # forward() returns the *raw* Switch balance loss (~1.0 when balanced); the model's own router_aux_loss_coef
            # (e.g. 1e-3) must be applied here, otherwise the balance term would dominate the LM loss by ~1000x.
            tm = getattr(trainer, "raw", None) or model
            moe_cfg = getattr(getattr(tm, "config", None), "moe", None)
            model_coef = float(getattr(moe_cfg, "router_aux_loss_coef", 1.0)) if moe_cfg is not None else 1.0
            total = total + c.aux_loss_coef * model_coef * aux.float() * n
            m["aux"] = aux.detach().float() * n
        return StepLoss(total, m)

    def eval_batches(self) -> list[dict[str, Any]]:
        if self._eval is None:
            src = self.val_ds if self.val_ds is not None else None
            self._eval = src.eval_batches(self.cfg.micro_batch_size, self.cfg.eval.num_batches) if src is not None and len(src) else []
        return self._eval

    def steps_per_epoch(self, global_batch: int) -> float | None:
        return len(self.ds) / global_batch if len(self.ds) else None

    def tokens_per_step(self, global_batch: int) -> float | None:
        return float(global_batch * self.seq_len_hint) if self.seq_len_hint else None

    def describe(self) -> dict[str, Any]:
        d = dict(self.ds.describe()) if hasattr(self.ds, "describe") else {}
        if self.val_ds is not None and hasattr(self.val_ds, "describe"):
            d["val"] = self.val_ds.describe()
        return d


# ---------------------------------------------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------------------------------------------
_PEAK_BF16_TFLOPS = [  # dense, vendor datasheet, approximate -- override with logging.peak_tflops_per_gpu for exact figures
    ("b200", 2250.0), ("gb200", 2500.0), ("h200", 989.0), ("h100", 989.0), ("a100", 312.0), ("l40s", 362.0), ("l40", 181.0),
    ("l4", 121.0), ("a10g", 70.0), ("a10", 125.0), ("a40", 149.7), ("a30", 165.0), ("rtx 6000 ada", 364.0), ("rtx pro 6000", 504.0),
    ("5090", 209.5), ("4090", 165.2), ("4080", 97.5), ("3090", 71.0), ("3080", 59.5), ("v100", 125.0), ("t4", 65.0),
]


def peak_tflops_for(device: torch.device, override: float | None = None) -> float | None:
    """Dense low-precision (bf16, else fp16) peak TFLOP/s of the device, or None when unknown / CPU.

    Source order: ``override`` -> ``forge.hw.chips`` datasheet DB (matched by name, SM level, SM count, memory) -> the small built-in
    table above.  Datasheet figures for GeForce parts may assume fp16 accumulation, so treat MFU there as indicative.
    """
    if override:
        return float(override)
    if device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        name = props.name
    except Exception:  # noqa: BLE001
        return None
    try:
        from forge.hw.chips import find_chip

        m = find_chip(name, sm=props.major * 10 + props.minor, num_sms=props.multi_processor_count, mem_gb=props.total_memory / 2**30)
        tf = (m.spec or {}).get("tflops", {})
        v = tf.get("bf16") if props.major >= 8 else tf.get("fp16")
        if v and m.kind != "unknown":
            return float(v)
    except Exception:  # noqa: BLE001
        pass
    lname = name.lower()
    for key, v in _PEAK_BF16_TFLOPS:
        if key in lname:
            return v
    return None


def estimate_flops_per_token(model: nn.Module, seq_len: int, all_trainable: bool = True) -> float:
    """PaLM-style training FLOPs/token: ``6N`` (``4N`` when weights are frozen) + ``12 L h T`` attention term."""
    n = 0
    has_head = False
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or (hasattr(m, "in_features") and hasattr(m, "out_features") and not list(m.children())):
            n += int(m.in_features) * int(m.out_features)
            has_head = has_head or "lm_head" in name
    cfg = getattr(model, "config", None)
    layers = int(getattr(cfg, "num_layers", getattr(cfg, "num_hidden_layers", 0)) or 0)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if not has_head and vocab and hidden:
        n += vocab * hidden
    mult = 6.0 if all_trainable else 4.0
    return mult * n + 12.0 * layers * hidden * seq_len


def enable_activation_checkpointing(model: nn.Module) -> int:
    """Recompute each transformer block in backward (non-reentrant) by patching ``forward`` -- state-dict keys are unchanged."""
    from torch.utils.checkpoint import checkpoint

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            return -1
        except Exception:  # noqa: BLE001
            pass
    layers = None
    for path in ("layers", "model.layers", "transformer.h", "h", "blocks"):
        cur: Any = model
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
        except AttributeError:
            continue
        if isinstance(cur, nn.ModuleList):
            layers = cur
            break
    if layers is None:
        log.warning("activation_checkpointing requested but no block list (layers/h/blocks) found; skipped")
        return 0
    for blk in layers:
        if getattr(blk, "_forge_ckpt", False):
            continue
        orig = blk.forward

        def make(o: Any, b: nn.Module) -> Any:
            def fwd(*a: Any, **k: Any) -> Any:
                if torch.is_grad_enabled() and b.training:
                    return checkpoint(o, *a, use_reentrant=False, **k)
                return o(*a, **k)

            return fwd

        blk.forward = make(orig, blk)
        blk._forge_ckpt = True
    return len(layers)


def load_reference_losses(path: str | Path) -> dict[int, float]:
    """Reference run losses from JSONL (``{"step":..,"loss":..}``) or two-column CSV."""
    out: dict[int, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("step", "#")):
            continue
        try:
            if line.startswith("{"):
                d = json.loads(line)
                out[int(d["step"])] = float(d["loss"])
            else:
                a, b = line.replace(";", ",").split(",")[:2]
                out[int(float(a))] = float(b)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return out


class _SilentCtx:
    """Context for non-main ranks."""

    def __init__(self, base: Any) -> None:
        self.run_id = getattr(base, "run_id", "")
        self.run_dir = Path(getattr(base, "run_dir", "."))
        self.params: dict = {}
        self.store = None

    def log(self, msg: str, level: str = "info") -> None:
        if level in ("warning", "error"):
            log.warning(msg)

    def metric(self, step: int, **values: float) -> None:
        pass

    def progress(self, frac: float, msg: str = "") -> None:
        pass

    def should_stop(self) -> bool:
        return False

    def set_output(self, key: str, value: Any) -> None:
        pass


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _rng_state() -> dict[str, Any]:
    st: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, model: nn.Module, cfg: CommonTrainConfig, task: Task, ctx: Any, *, dinfo: D.DistInfo | None = None,
                 run_dir: str | Path | None = None, state_filter: Any = None, extra_state: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self.task = task
        self.dinfo = dinfo or D.init_distributed(cfg.parallel)
        self.ctx = ctx if self.dinfo.is_main else _SilentCtx(ctx)
        self.base_ctx = ctx
        self.device = self.dinfo.device
        self.run_dir = Path(run_dir or getattr(ctx, "run_dir", "."))
        self.state_filter = state_filter  # callable(name, param)->bool selecting params saved in checkpoints (None = all)
        self.extra_state = extra_state or {}
        self.raw = model
        self.step = 0
        self.tokens_seen = 0
        self.data_skip = 0
        self.restarts = 0
        self.skipped_steps = 0
        self.best_eval: float | None = None
        self.bad_evals = 0
        self.rollback_events: list[dict[str, Any]] = []
        self.lr_scale_from: tuple[int, float] | None = None  # (rollback step, scale)
        self.last_loss: float | None = None
        self.status = "running"
        self._streak_nonfinite = 0
        self._streak_spike = 0
        self._anneal_saved: set[int] = set()
        self.ref_losses: dict[int, float] = {}
        self._t_start = time.time()
        self._prefetch: StepPrefetcher | None = None
        self._compile_checked = False

    # -- setup ----------------------------------------------------------------------------------------
    def _resolve_total_steps(self) -> int:
        c, b = self.cfg, self.cfg.budget
        gb = c.micro_batch_size * c.grad_accum_steps * self.dinfo.world
        cands: list[float] = []
        if b.max_steps:
            cands.append(float(b.max_steps))
        if b.max_tokens:
            tps = self.task.tokens_per_step(gb)
            if tps:
                cands.append(math.ceil(b.max_tokens / tps))
        if b.max_epochs:
            spe = self.task.steps_per_epoch(gb)
            if spe:
                cands.append(math.ceil(b.max_epochs * spe))
        self.budget_steps = int(min(cands)) if cands else 0
        total = self.budget_steps
        if c.schedule.total_steps:
            total = int(c.schedule.total_steps)
        if total <= 0:
            total = 10_000
            self.ctx.log("no step horizon known (only a time budget?): set schedule.total_steps; assuming 10000 for the LR schedule", "warning")
        return total

    def setup(self) -> None:
        c = self.cfg
        from forge.core.env import seed_everything

        seed_everything(c.seed + self.dinfo.rank, deterministic=c.deterministic)
        self.total_steps = self._resolve_total_steps()
        self.global_batch = c.micro_batch_size * c.grad_accum_steps * self.dinfo.world
        self.raw.to(self.device)
        self.precision = PrecisionManager(c.precision, device=self.device, total_steps=self.total_steps, seed=c.seed, rank=self.dinfo.rank)
        summ = self.precision.prepare(self.raw)
        self.ctx.log(f"precision: {json.dumps({k: v for k, v in summ.items() if k != 'exempt_names'})}")
        if c.activation_checkpointing:
            n = enable_activation_checkpointing(self.raw)
            self.ctx.log(f"activation checkpointing enabled ({'HF' if n < 0 else n} blocks)")
        self.model = D.wrap_model(self.raw, c.parallel, self.dinfo, self.precision.autocast_dtype)
        self.fmodel = self.model
        if c.parallel.compile:
            try:
                self.fmodel = torch.compile(self.model, mode=None if c.parallel.compile_mode == "default" else c.parallel.compile_mode)
            except Exception as e:  # noqa: BLE001
                self.ctx.log(f"torch.compile failed ({e!r}); running eager", "warning")
        named = [(n, p) for n, p in self.model.named_parameters()]
        self.trainable = [(n, p) for n, p in named if p.requires_grad]
        if not self.trainable:
            raise ConfigError("no trainable parameters")
        self.opt = build_optimizer(named, c.optim, cuda=self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.precision.needs_scaler else None
        self.detector = SpikeDetector(c.stability.spike_window, c.stability.spike_zscore, c.stability.spike_ratio, c.stability.grad_spike_factor)
        ck_dir = Path(c.checkpoint.dir) if c.checkpoint.dir else self.run_dir / "checkpoints"
        self.ckpt = CheckpointManager(ck_dir, c.checkpoint, store=getattr(self.base_ctx, "store", None), run_id=getattr(self.base_ctx, "run_id", ""))
        self.eval_set = self.task.eval_batches()
        # update/weight ratio sampling set
        budget = c.logging.update_ratio_max_numel
        self._ratio_params: list[torch.Tensor] = []
        used = 0
        if budget:
            for _, p in self.trainable:
                if used + p.numel() > budget and self._ratio_params:
                    continue
                if p.numel() <= budget:
                    self._ratio_params.append(p)
                    used += p.numel()
        all_train = all(p.requires_grad for p in self.raw.parameters())
        sl = getattr(self.task, "seq_len_hint", 0) or 0
        self.flops_per_token = estimate_flops_per_token(self.raw, sl, all_train)
        self.peak_tflops = peak_tflops_for(self.device, c.logging.peak_tflops_per_gpu)
        if c.precision.reference_loss_file:
            try:
                self.ref_losses = load_reference_losses(c.precision.reference_loss_file)
            except Exception as e:  # noqa: BLE001
                self.ctx.log(f"could not read reference_loss_file: {e!r}", "warning")
        mf = c.logging.metrics_file or str(self.run_dir / "metrics.jsonl")
        self._metrics_path = Path(mf) if self.dinfo.is_main else None
        if self._metrics_path:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        n_tr = sum(p.numel() for _, p in self.trainable)
        n_all = sum(p.numel() for p in self.raw.parameters())
        self.ctx.log(f"trainer: {n_tr/1e6:.2f}M trainable / {n_all/1e6:.2f}M params, device={self.device}, world={self.dinfo.world}, "
                     f"global_batch={self.global_batch} seq, total_steps={self.total_steps}, autocast={self.precision.autocast_dtype}")
        self._resume()

    # -- state ----------------------------------------------------------------------------------------
    def _trainer_state(self) -> dict[str, Any]:
        return {
            "step": self.step, "tokens_seen": self.tokens_seen, "data_skip": self.data_skip, "restarts": self.restarts,
            "skipped_steps": self.skipped_steps, "best_eval": self.best_eval, "bad_evals": self.bad_evals,
            "rollback_events": self.rollback_events, "lr_scale_from": self.lr_scale_from, "detector": self.detector.state_dict(),
            "anneal_saved": sorted(self._anneal_saved), "total_steps": self.total_steps, "fp4_rng": {k: g.get_state() for k, g in self.precision.rng._gens.items()},
        }

    def _names(self) -> set[str] | None:
        if self.state_filter is None:
            return None
        return {n for n, p in self.raw.named_parameters() if self.state_filter(n, p)} | {n for n, _ in self.raw.named_buffers() if self.state_filter(n, None)}

    def _collect_state(self, include_optim: bool = True) -> dict[str, Any]:
        model_sd = D.get_model_state(self.model, self._names())
        st: dict[str, Any] = {"model": model_sd, "trainer": self._trainer_state(), "extra": {**self.task.state_dict()}}
        if include_optim and self.cfg.checkpoint.save_optimizer:
            st["optim"] = D.get_optim_state(self.model, self.opt)
            if self.scaler is not None:
                st["scaler"] = self.scaler.state_dict()
        st["rng"] = _rng_state()
        return st

    def save_checkpoint(self, *, good: bool = True, best: bool = False, pre_anneal: bool = False, final: bool = False, meta: dict[str, Any] | None = None) -> Path | None:
        st = self._collect_state()  # collective under FSDP
        if not self.dinfo.is_main:
            return None
        m = {"loss": self.last_loss, "tokens_seen": self.tokens_seen, **(meta or {})}
        p = self.ckpt.save(self.step, st, good=good, best=best, pre_anneal=pre_anneal, final=final, meta=m)
        self.ctx.log(f"checkpoint saved: step {self.step} -> {p.name}" + (" [best]" if best else "") + (" [pre-anneal]" if pre_anneal else "") + (" [final]" if final else ""))
        return p

    def _load_state(self, state: dict[str, Any], *, restore_rng: bool, strict: bool = True) -> None:
        res = D.load_model_state(self.model, state["model"], strict=False)
        missing = list(getattr(res, "missing_keys", []))
        if strict and self.state_filter is None and missing:
            raise ConfigError("checkpoint is missing model keys", details={"missing": missing[:10]})
        if "optim" in state:
            D.load_optim_state(self.model, self.opt, state["optim"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        t = state.get("trainer", {})
        self.step = int(t.get("step", 0))
        self.tokens_seen = int(t.get("tokens_seen", 0))
        self.best_eval = t.get("best_eval")
        self.bad_evals = int(t.get("bad_evals", 0))
        self.restarts = max(self.restarts, int(t.get("restarts", 0)))
        self.skipped_steps = max(self.skipped_steps, int(t.get("skipped_steps", 0)))
        self.rollback_events = list(t.get("rollback_events", self.rollback_events))
        self.lr_scale_from = t.get("lr_scale_from")
        self._anneal_saved = set(t.get("anneal_saved", []))
        if restore_rng:
            self.data_skip = int(t.get("data_skip", 0))
            self.detector.load_state_dict(t.get("detector", {}))
            if "rng" in state:
                _set_rng_state(state["rng"])
            for k, s in t.get("fp4_rng", {}).items():
                try:
                    self.precision.rng.gen(torch.device(k)).set_state(s)
                except Exception:  # noqa: BLE001
                    pass
        self.task.load_state_dict(state.get("extra", {}))

    def _resume(self) -> None:
        spec = self.cfg.checkpoint.resume
        if spec in ("none", "", None):
            if self.cfg.checkpoint.save_initial:
                self.save_checkpoint(good=True, meta={"initial": True})
            return
        try:
            state, meta = self.ckpt.load("latest" if spec == "auto" else spec)
        except NotFound:
            if spec == "auto":
                self.ctx.log("resume=auto: no checkpoint found, starting fresh")
                if self.cfg.checkpoint.save_initial:
                    self.save_checkpoint(good=True, meta={"initial": True})
                return
            raise
        self._load_state(state, restore_rng=True)
        self.ctx.log(f"resumed from checkpoint at step {self.step} (tokens_seen={self.tokens_seen}, data_skip={self.data_skip})")

    # -- one optimiser step -------------------------------------------------------------------------------
    def _lr(self, step: int) -> float:
        f = lr_factor(step, self.cfg.schedule, self.total_steps)
        if self.lr_scale_from is not None:
            s0, scale = self.lr_scale_from
            cool = self.cfg.stability.rollback_cooldown_steps
            if cool > 0 and step < s0 + cool:
                f *= scale + (1.0 - scale) * (step - s0) / cool
            elif cool > 0:
                self.lr_scale_from = None
            elif cool == 0:
                self.lr_scale_from = None
        return self.cfg.optim.lr * f

    def _discard_scaled_step(self, *, backoff: bool = False) -> None:
        """Close a GradScaler iteration whose update is skipped.

        ``unscale_`` has already run, so the scaler must see ``update()`` before the next ``unscale_`` (otherwise torch raises
        "unscale_() has already been called on this optimizer since the last update()").  ``backoff`` forces the scale down
        (used when the loss itself was non-finite, in which case no inf-check may have been recorded).
        """
        if self.scaler is None:
            return
        try:
            if backoff:
                self.scaler.update(self.scaler.get_scale() * self.scaler.get_backoff_factor())
            else:
                self.scaler.update()
        except (AssertionError, RuntimeError):  # no inf-check recorded (no gradients) -> just reset the state
            self.scaler.update(self.scaler.get_scale())

    def _clip(self) -> float:
        mx = self.cfg.optim.grad_clip
        params = [p for _, p in self.trainable if p.grad is not None]
        if not params:
            return 0.0
        if hasattr(self.model, "clip_grad_norm_") and D._is_fsdp(self.model):
            return float(self.model.clip_grad_norm_(mx if mx > 0 else float("inf")))
        return float(torch.nn.utils.clip_grad_norm_(params, mx if mx > 0 else float("inf")))

    def _train_step(self, step: int, batches: list[dict[str, Any]], want_diag: bool, want_gap: bool, want_ratio: bool) -> dict[str, Any]:
        c = self.cfg
        lr = self._lr(step)
        set_lr(self.opt, lr)
        self.opt.zero_grad(set_to_none=c.optim.zero_grad_set_to_none)
        items = [self.task.count(b) for b in batches]
        total_items = max(1.0, float(sum(items)))
        if self.dinfo.initialized and self.model is not self.raw:
            # DDP/FSDP average gradients over ranks, so normalise by the *mean per-rank* item count: the result is the exact
            # global token-weighted mean even when ranks hold different numbers of supervised tokens (SFT masks, padding).
            import torch.distributed as dist

            t_items = torch.tensor([total_items], device=self.device, dtype=torch.float64)
            dist.all_reduce(t_items)
            total_items = max(1.0, float(t_items.item()) / self.dinfo.world)
        tokens = sum(self.task.tokens_in_batch(b) for b in batches)
        loss_acc = torch.zeros((), device=self.device)
        msum: dict[str, torch.Tensor] = {}
        first_mb_loss: torch.Tensor | None = None
        n = len(batches)
        self.model.train()
        for i, b in enumerate(batches):
            b = _to_device(b, self.device)
            sync_ctx = self.model.no_sync() if (i < n - 1 and hasattr(self.model, "no_sync")) else contextlib.nullcontext()
            with sync_ctx:
                with self.precision.autocast():
                    out = self.task.loss(self.fmodel, b, self)
                loss = out.loss_sum.float() / total_items
                (self.scaler.scale(loss) if self.scaler is not None else loss).backward()
            ld = loss.detach()
            if i == 0 and want_gap:
                first_mb_loss = out.loss_sum.detach().float() / max(1.0, items[0])
            loss_acc = loss_acc + ld
            for k, v in out.metrics.items():
                v = torch.as_tensor(v, device=self.device, dtype=torch.float32).detach() / total_items
                msum[k] = msum.get(k, 0) + v
        if self.dinfo.initialized:
            import torch.distributed as dist

            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
        loss_val = float(loss_acc)
        res: dict[str, Any] = {"loss": loss_val, "lr": lr, "tokens": tokens * self.dinfo.world, "metrics": {k: float(v) for k, v in msum.items()},
                               "anomaly": None, "skipped": False}
        # --- gradients ---------------------------------------------------------------------------------
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        gnorm = self._clip()
        res["grad_norm"] = gnorm
        finite = math.isfinite(loss_val) and math.isfinite(gnorm)
        if want_diag and finite:
            try:
                res["diag"] = self.precision.diagnostics(self.raw)
            except Exception as e:  # noqa: BLE001
                self.ctx.log(f"diagnostics failed: {e!r}", "warning")
        if want_gap and finite and first_mb_loss is not None:
            with torch.no_grad(), self.precision.high_precision(), self.precision.autocast():
                self.model.eval()
                ref = self.task.loss(self.fmodel, _to_device(batches[0], self.device), self)
                self.model.train()
            ref_val = float(ref.loss_sum.float() / max(1.0, items[0]))
            res["loss_ref"] = ref_val
            if ref_val > 0 and math.isfinite(ref_val):
                res["loss_gap"] = (float(first_mb_loss) - ref_val) / ref_val
        # --- anomaly handling ---------------------------------------------------------------------------
        st = c.stability
        if not finite and st.detect_nonfinite:
            if self.scaler is not None and math.isfinite(loss_val):
                # dynamic loss scaling: scaler.step skips the update and lowers the scale
                self.scaler.step(self.opt)
                self.scaler.update()
                res["skipped"] = True
                res["scaler_skip"] = True
                if self.scaler.get_scale() < st.min_loss_scale:
                    res["anomaly"] = "nonfinite"
                return res
            self._discard_scaled_step(backoff=True)
            res["anomaly"] = "nonfinite"
            res["skipped"] = True
            return res
        if st.detect_spikes and finite:
            sp = self.detector.check(loss_val, gnorm)
            res["spike"] = sp
            if (sp["loss_spike"] or sp["grad_spike"]) and st.on_spike != "ignore":
                res["anomaly"] = "spike"
                if st.on_spike in ("skip", "rollback"):
                    self._discard_scaled_step()
                    res["skipped"] = True
                    return res
                if st.on_spike == "warn":
                    self.ctx.log(f"step {step + 1}: spike detected (loss={loss_val:.4f}, grad_norm={gnorm:.3f}, z={sp['z']:.1f})", "warning")
        snap = [p.detach().clone() for p in self._ratio_params] if want_ratio and self._ratio_params else None
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        if snap is not None:
            from forge.train.diagnostics import update_weight_ratio

            res["update_weight_ratio"] = update_weight_ratio(snap, self._ratio_params)
        if res["anomaly"] is None:
            self.detector.record(loss_val, gnorm)
        return res

    # -- evaluation ----------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if not self.eval_set:
            return {}
        self.model.eval()
        tot = torch.zeros((), device=self.device)
        cnt = torch.zeros((), device=self.device)
        msum: dict[str, torch.Tensor] = {}
        ctxs = contextlib.ExitStack()
        if self.cfg.eval.in_high_precision:
            ctxs.enter_context(self.precision.high_precision())
        with ctxs:
            for b in self.eval_set:
                b = _to_device(b, self.device)
                with self.precision.autocast():
                    out = self.task.loss(self.fmodel, b, self)
                tot = tot + out.loss_sum.float()
                cnt = cnt + self.task.count(b)
                for k, v in out.metrics.items():
                    msum[k] = msum.get(k, 0) + torch.as_tensor(v, device=self.device, dtype=torch.float32)
        if self.dinfo.initialized:
            import torch.distributed as dist

            for t in [tot, cnt, *msum.values()]:
                dist.all_reduce(t)
        self.model.train()
        c = max(1.0, float(cnt))
        out_d = {"eval_loss": float(tot) / c}
        out_d["eval_ppl"] = math.exp(min(20.0, out_d["eval_loss"]))
        for k, v in msum.items():
            out_d[f"eval_{k}"] = float(v) / c
        return out_d

    # -- rollback --------------------------------------------------------------------------------------
    def _rollback(self, reason: str, cur_step: int) -> bool:
        st = self.cfg.stability
        self.restarts += 1
        if self.restarts > st.max_restarts:
            raise TrainingDiverged(f"training diverged ({reason}); exceeded max_restarts={st.max_restarts}",
                                   details={"step": cur_step, "reason": reason, "events": self.rollback_events})
        target = self.ckpt.last_good()
        if target is None:
            raise TrainingDiverged(f"{reason} at step {cur_step + 1} and no checkpoint to roll back to "
                                   "(enable checkpoint.save_initial or a smaller checkpoint.every_steps)", details={"reason": reason})
        state, _ = self.ckpt.load(target)
        old_skip = self.data_skip
        restarts, skipped, events = self.restarts, self.skipped_steps, list(self.rollback_events)
        self._load_state(state, restore_rng=False)
        self.restarts, self.skipped_steps = restarts, skipped
        self.data_skip = cur_step + old_skip + st.skip_batches_on_rollback - self.step
        ev = {"at_step": cur_step + 1, "to_step": self.step, "reason": reason, "data_skip": self.data_skip}
        self.rollback_events = events + [ev]
        self.lr_scale_from = (self.step, st.rollback_lr_scale) if st.rollback_cooldown_steps > 0 else None
        self.detector.reset()
        self._streak_nonfinite = self._streak_spike = 0
        self.opt.zero_grad(set_to_none=True)
        self.ctx.log(f"ROLLBACK ({reason}) at step {cur_step + 1}: restored step {self.step}, skipping data to offset {self.data_skip}, "
                     f"lr x{st.rollback_lr_scale} cool-down {st.rollback_cooldown_steps} steps (restart {self.restarts}/{st.max_restarts})", "warning")
        self._restart_prefetch()
        return True

    def _restart_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.close()
        w = max(0, getattr(self, "_workers", 0))
        make = lambda s: self.task.get_batches(s + self.data_skip, self.dinfo.rank, self.dinfo.world)  # noqa: E731
        self._prefetch = StepPrefetcher(make, start=self.step, end=None, depth=max(1, getattr(self, "_depth", 2)), threads=w)

    # -- main loop --------------------------------------------------------------------------------------
    def _budget_reached(self) -> str | None:
        b = self.cfg.budget
        if self.budget_steps and self.step >= self.budget_steps:
            return "max_steps" if b.max_steps and self.budget_steps == b.max_steps else "token/epoch budget"
        if b.max_tokens and self.tokens_seen >= b.max_tokens:
            return "max_tokens"
        if b.max_minutes and (time.time() - self._t_start) >= b.max_minutes * 60:
            return "max_minutes"
        return None

    def _progress_frac(self) -> float:
        b = self.cfg.budget
        fr = [0.0]
        if b.max_steps:
            fr.append(self.step / b.max_steps)
        elif self.budget_steps:
            fr.append(self.step / self.budget_steps)
        if b.max_tokens:
            fr.append(self.tokens_seen / b.max_tokens)
        if b.max_minutes:
            fr.append((time.time() - self._t_start) / (b.max_minutes * 60))
        return min(1.0, max(fr))

    def _stop_requested(self) -> bool:
        flag = False
        if self.dinfo.is_main:
            flag = bool(self.base_ctx.should_stop()) or (self.run_dir / "STOP").exists()
        return D.all_reduce_flag(flag, self.dinfo)

    def _emit(self, step_done: int, values: dict[str, float]) -> None:
        self.ctx.metric(step_done, **values)
        if self._metrics_path is not None:
            with open(self._metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step_done, **values}) + "\n")

    def fit(self) -> dict[str, Any]:
        self.setup()
        c = self.cfg
        self._workers = getattr(self.task, "num_workers", 0)
        self._depth = getattr(self.task, "prefetch_steps", 2)
        self._restart_prefetch()
        anneal = anneal_start_step(c.schedule, self.total_steps)
        heal = self.precision.healing_start
        pre_points = {p for p in (anneal, heal) if p}
        win_t0 = time.time()
        win_tokens = 0
        win_steps = 0
        status = "completed"
        stop_every = max(1, c.logging.stop_check_every)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            while True:
                reason = self._budget_reached()
                if reason:
                    self.ctx.log(f"budget reached: {reason}")
                    break
                if self.step % stop_every == 0 and self._stop_requested():
                    status = "stopped"
                    self.ctx.log("stop requested: saving checkpoint and exiting", "warning")
                    break
                if self.precision.set_step(self.step):
                    self.ctx.log(f"healing phase starts at step {self.step}: mode -> {self.precision._current_mode()}")
                if self.step in pre_points and self.step not in self._anneal_saved and c.checkpoint.pre_anneal and self.step > 0:
                    self._anneal_saved.add(self.step)
                    self.save_checkpoint(pre_anneal=True, meta={"tag": "pre_anneal"})
                step_idx = self.step
                done = step_idx + 1
                is_log = done % c.logging.log_every == 0
                prec = c.precision
                want_diag = prec.rung != "R0" and prec.diagnostics_every > 0 and done % prec.diagnostics_every == 0 and self.precision.phase == "main"
                want_gap = prec.rung != "R0" and prec.loss_gap_every > 0 and done % prec.loss_gap_every == 0 and self.precision.phase == "main" and self.precision.quantised
                want_ratio = is_log and c.logging.update_ratio_max_numel > 0
                try:
                    _, batches = next(self._prefetch)  # type: ignore[arg-type]
                except StopIteration:  # pragma: no cover
                    self.ctx.log("data exhausted", "warning")
                    break
                try:
                    res = self._train_step(step_idx, batches, want_diag, want_gap, want_ratio)
                except Exception as e:  # noqa: BLE001
                    if self.fmodel is self.model or self._compile_checked:
                        raise
                    # torch.compile compiles lazily: a toolchain problem surfaces at the first step -> fall back to eager
                    self.ctx.log(f"torch.compile failed at the first step ({e!r}); continuing in eager mode", "warning")
                    self.fmodel = self.model
                    self.opt.zero_grad(set_to_none=True)
                    res = self._train_step(step_idx, batches, want_diag, want_gap, want_ratio)
                self._compile_checked = True
                an = res["anomaly"]
                if an == "nonfinite":
                    self._streak_nonfinite += 1
                    self.skipped_steps += 1
                    self.ctx.log(f"step {done}: non-finite loss/grad (loss={res['loss']}, grad_norm={res['grad_norm']}); update skipped "
                                 f"({self._streak_nonfinite}/{c.stability.nonfinite_patience})", "warning")
                    if self._streak_nonfinite >= c.stability.nonfinite_patience:
                        if c.stability.on_nonfinite == "abort":
                            raise TrainingDiverged(f"non-finite values at step {done}", details={"step": done})
                        if c.stability.on_nonfinite == "rollback":
                            self._rollback("nonfinite", step_idx)
                            continue
                    self.step += 1
                    continue
                self._streak_nonfinite = 0
                if an == "spike":
                    self._streak_spike += 1
                    sp = res.get("spike", {})
                    self.ctx.log(f"step {done}: loss/grad spike (loss={res['loss']:.4f} z={sp.get('z', 0):.1f} grad_norm={res['grad_norm']:.3f}) "
                                 f"{self._streak_spike}/{c.stability.spike_patience} action={c.stability.on_spike}", "warning")
                    if c.stability.on_spike == "rollback" and self._streak_spike >= c.stability.spike_patience and self.ckpt.last_good() is not None:
                        self._rollback("loss_spike", step_idx)
                        continue
                    if res["skipped"]:
                        self.skipped_steps += 1
                        self.step += 1
                        continue
                else:
                    self._streak_spike = 0
                if res.get("scaler_skip"):
                    self.skipped_steps += 1
                # ---- successful step ------------------------------------------------------------------
                self.step += 1
                self.last_loss = res["loss"]
                self.tokens_seen += res["tokens"]
                win_tokens += res["tokens"]
                win_steps += 1
                if is_log:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    dt = max(1e-9, time.time() - win_t0)
                    tps = win_tokens / dt
                    vals: dict[str, float] = {
                        "loss": res["loss"], "lr": res["lr"], "grad_norm": res["grad_norm"], "tokens_per_s": tps,
                        "tokens_seen": float(self.tokens_seen), "step_time_ms": 1000.0 * dt / max(1, win_steps),
                        "gpu_mem_gb": (torch.cuda.max_memory_allocated(self.device) / 1e9) if (self.device.type == "cuda" and c.logging.track_gpu_memory) else 0.0,
                        "restarts": float(self.restarts), "skipped_steps": float(self.skipped_steps),
                    }
                    if self.peak_tflops and self.flops_per_token > 0:
                        vals["mfu"] = tps / max(1, self.dinfo.world) * self.flops_per_token / (self.peak_tflops * 1e12)
                    if "update_weight_ratio" in res:
                        vals["update_weight_ratio"] = res["update_weight_ratio"]
                    if self.scaler is not None:
                        vals["loss_scale"] = float(self.scaler.get_scale())
                    for k, v in res["metrics"].items():
                        vals[k] = v
                    vals.update(self.task.extra_metrics())
                    ref = self.ref_losses.get(done)
                    if ref:
                        vals[f"{self.precision.metric_prefix}/loss_gap_ref"] = (res["loss"] - ref) / ref
                    self._emit(done, vals)
                    win_t0, win_tokens, win_steps = time.time(), 0, 0
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                extra: dict[str, float] = {}
                if "diag" in res:
                    extra.update(res["diag"])
                if "loss_gap" in res:
                    extra[f"{self.precision.metric_prefix}/loss_gap"] = res["loss_gap"]
                    extra[f"{self.precision.metric_prefix}/loss_ref"] = res["loss_ref"]
                if extra:
                    self._emit(done, extra)
                self.ctx.progress(self._progress_frac(), f"step {done}/{self.budget_steps or self.total_steps} loss {res['loss']:.4f}")
                # ---- evaluation / early stopping -------------------------------------------------------
                ev = c.eval
                if ev.every_steps and done % ev.every_steps == 0 and self.eval_set:
                    ed = self.evaluate()
                    self._emit(done, ed)
                    self.ctx.log(f"step {done}: eval_loss={ed['eval_loss']:.4f} ppl={ed['eval_ppl']:.2f}")
                    improved = self.best_eval is None or ed["eval_loss"] < self.best_eval - ev.min_delta
                    if improved:
                        self.best_eval = ed["eval_loss"]
                        self.bad_evals = 0
                        if c.checkpoint.keep_best:
                            self.save_checkpoint(best=True, meta={"eval_loss": ed["eval_loss"]})
                    else:
                        self.bad_evals += 1
                        if ev.early_stopping and self.bad_evals >= ev.patience:
                            status = "early_stopped"
                            self.ctx.log(f"early stopping: no improvement for {self.bad_evals} evaluations")
                            break
                if c.checkpoint.every_steps and done % c.checkpoint.every_steps == 0:
                    self.save_checkpoint()
        finally:
            if self._prefetch is not None:
                self._prefetch.close()
        # final evaluation + checkpoint
        final_eval: dict[str, float] = {}
        if self.eval_set and status != "stopped":
            try:
                final_eval = self.evaluate()
                self._emit(self.step, final_eval)
                if self.best_eval is None or final_eval["eval_loss"] < self.best_eval:
                    self.best_eval = final_eval["eval_loss"]
            except Exception as e:  # noqa: BLE001
                self.ctx.log(f"final evaluation failed: {e!r}", "warning")
        self.save_checkpoint(final=True, meta={"status": status})
        if status == "early_stopped" and c.eval.restore_best and self.ckpt.best() is not None:
            state, _ = self.ckpt.load("best")
            D.load_model_state(self.model, state["model"], strict=False)
            self.ctx.log(f"restored best checkpoint (step {self.ckpt.best()})")
        self.status = status
        result = {
            "status": status, "steps": self.step, "tokens_seen": self.tokens_seen, "final_loss": self.last_loss, "best_eval_loss": self.best_eval,
            "final_eval": final_eval, "restarts": self.restarts, "skipped_steps": self.skipped_steps, "rollbacks": self.rollback_events,
            "checkpoint_dir": str(self.ckpt.root), "elapsed_s": time.time() - self._t_start, "precision": self.precision.summary,
            "total_steps": self.total_steps,
        }
        self.ctx.progress(1.0 if status != "stopped" else self._progress_frac(), f"{status}: {self.step} steps")
        return result
