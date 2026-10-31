"""Supervised fine-tuning: full, LoRA and QLoRA, with an optional DPO-family preference objective (Plan TR-*, UI "Fine-tune").

* data: :mod:`forge.train.sft` (chat templates, assistant-only loss masks, packing, preference pairs)
* LoRA: ``forge.models.lora`` when importable (preferred), else the built-in implementation (:mod:`forge.train.lora`);
  QLoRA: built-in NF4/int8 block-quantised frozen base + built-in LoRA
* outputs: adapter directory (PEFT-style) and a merged full model; both registered in the store with lineage
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError
from forge.train import dist as D
from forge.train.common import build_model, check_vocab, coerce_cfg, load_tokenizer_for, register_outputs
from forge.train.config import FinetuneConfig
from forge.train.lora import LoraBackend, adapter_disabled, quantize_base_linears
from forge.train.losses import dpo_loss, response_logprobs
from forge.train.sft import SFTDataset, build_sft_datasets
from forge.train.trainer import LMTask, StepLoss, Task, Trainer, forward_logits

log = logging.getLogger("forge.train.finetune")

__all__ = ["DPOTask", "SFTTask", "run_finetune"]


class SFTTask(LMTask):
    """Cross-entropy on assistant tokens; micro-batches are padded individually (less padding than padding per global batch)."""

    name = "sft"

    def get_batches(self, step: int, rank: int, world: int) -> list[dict[str, Any]]:
        mb, acc = self.cfg.micro_batch_size, self.cfg.grad_accum_steps
        per = mb * acc
        pos = self.ds.positions_at(step, per * world)[rank * per:(rank + 1) * per]
        return [self.ds.get_batch(pos[i:i + mb]) for i in range(0, per, mb)]

    def tokens_per_step(self, global_batch: int) -> float | None:
        n = len(self.ds)
        return float(self.ds.tokens_per_epoch) / n * global_batch if n else None


class DPOTask(Task):
    """DPO-family preference loss over ``prompt/chosen/rejected`` pairs. Items = pairs."""

    name = "dpo"

    def __init__(self, ds: SFTDataset, cfg: FinetuneConfig, val_ds: SFTDataset | None, ref_model: nn.Module | None, use_adapter_ref: bool) -> None:
        self.ds, self.val_ds, self.cfg = ds, val_ds, cfg
        self.pref = cfg.preference
        self.ref_model, self.use_adapter_ref = ref_model, use_adapter_ref
        self.seq_len_hint = cfg.data.max_seq_len
        self.num_workers, self.prefetch_steps = 0, 2
        self._eval: list[dict[str, Any]] | None = None

    def get_batches(self, step: int, rank: int, world: int) -> list[dict[str, Any]]:
        mb, acc = self.cfg.micro_batch_size, self.cfg.grad_accum_steps
        per = mb * acc
        pos = self.ds.positions_at(step, per * world)[rank * per:(rank + 1) * per]
        return [self.ds.get_batch(pos[i:i + mb]) for i in range(0, per, mb)]

    def count(self, batch: dict[str, Any]) -> float:
        return float(batch["n_pairs"])

    def loss(self, model: nn.Module, batch: dict[str, Any], trainer: Trainer) -> StepLoss:
        p = self.pref
        n = int(batch["n_pairs"])
        ids, labels = batch["input_ids"], batch["labels"]
        logits, _ = forward_logits(model, ids)
        lp = response_logprobs(logits, labels, average=p.length_normalise)
        with torch.no_grad():
            if self.use_adapter_ref:
                with adapter_disabled(trainer.raw):
                    rl, _ = forward_logits(model, ids)
            else:
                assert self.ref_model is not None
                rl, _ = forward_logits(self.ref_model, ids)
            rlp = response_logprobs(rl, labels, average=p.length_normalise)
        per_pair, diag = dpo_loss(lp[:n], lp[n:], rlp[:n], rlp[n:], p.beta, p.loss_type, p.label_smoothing)
        total = per_pair.sum()
        if p.sft_weight > 0:
            ntok = (labels[:n] != -100).sum(-1).clamp(min=1)
            nll = -lp[:n] if p.length_normalise else -lp[:n] / ntok
            total = total + p.sft_weight * nll.sum()
        m = {"dpo": per_pair.detach().sum(), "reward_acc": diag["reward_acc"].sum(), "reward_margin": diag["reward_margin"].sum(),
             "reward_chosen": diag["reward_chosen"].sum(), "reward_rejected": diag["reward_rejected"].sum()}
        return StepLoss(total, m)

    def eval_batches(self) -> list[dict[str, Any]]:
        if self._eval is None:
            self._eval = self.val_ds.eval_batches(self.cfg.micro_batch_size, self.cfg.eval.num_batches) if self.val_ds is not None and len(self.val_ds) else []
        return self._eval

    def steps_per_epoch(self, global_batch: int) -> float | None:
        return len(self.ds) / global_batch

    def tokens_per_step(self, global_batch: int) -> float | None:
        return float(self.ds.tokens_per_epoch) / len(self.ds) * global_batch

    def describe(self) -> dict[str, Any]:
        return self.ds.describe()


def _run_dir(ctx: Any) -> Path:
    return Path(ctx.run_dir)


def run_finetune(cfg: FinetuneConfig | dict, ctx: Any) -> dict[str, Any]:
    """Fine-tuning job entry (job type ``finetune``)."""
    from forge.train.run import _finish, _maybe_launch

    cfg = coerce_cfg(cfg, FinetuneConfig)
    launched = _maybe_launch("finetune", cfg, ctx)
    if launched is not None:
        return launched
    dinfo = D.init_distributed(cfg.parallel)
    try:
        tok = load_tokenizer_for(cfg.model, ctx)
        train_ds, val_ds = build_sft_datasets(cfg.data, tok, cfg.seed + cfg.data.data_seed)
        ctx.log(f"sft data: {train_ds.describe()}")
        is_pref = train_ds.kind == "preference"
        if cfg.preference.enabled and not is_pref:
            raise ConfigError("preference.enabled but the data has no prompt/chosen/rejected records")
        if is_pref and not cfg.preference.enabled:
            raise ConfigError("data contains preference records: set preference.enabled=true (or provide SFT data)")
        method = cfg.method
        model, _ = build_model(cfg.model, device=dinfo.device, seed=cfg.seed, tok=tok, ctx=ctx, frozen_base=method != "full")
        check_vocab(model, tok)
        backend: LoraBackend | None = None
        ref_model: nn.Module | None = None
        use_adapter_ref = False
        if method == "full":
            for n, p in model.named_parameters():
                if any(pat in n for pat in cfg.freeze_patterns):
                    p.requires_grad = False
            if is_pref:
                ref_model = copy.deepcopy(model).eval().requires_grad_(False)
        else:
            if method == "qlora":
                cd = torch.bfloat16 if dinfo.device.type == "cuda" else torch.float32
                nq = quantize_base_linears(model, cfg.qlora, cd)
                ctx.log(f"QLoRA: quantised {nq} linear layers to {cfg.qlora.bits}-bit blocks of {cfg.qlora.block_size}")
            backend = LoraBackend(cfg.lora, prefer_builtin=(method == "qlora" or is_pref))
            names = backend.apply(model)
            ctx.log(f"LoRA ({backend.impl}): adapted {len(names)} modules, r={cfg.lora.r}, alpha={cfg.lora.alpha}")
            use_adapter_ref = is_pref
        ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if ntrain == 0:
            raise ConfigError("no trainable parameters after applying the fine-tuning method")
        if is_pref:
            task: Task = DPOTask(train_ds, cfg, val_ds, ref_model, use_adapter_ref)
        else:
            task = SFTTask(train_ds, cfg, val_ds, cfg.data.max_seq_len)
        state_filter = (lambda n, p: p is not None and p.requires_grad) if method != "full" else None
        trainer = Trainer(model, cfg, task, ctx, dinfo=dinfo, state_filter=state_filter)
        result = trainer.fit()
        result["method"] = method
        result["trainable_params"] = ntrain
        raw = D.unwrap(trainer.model)
        parents = [cfg.model.parent_hash]
        if dinfo.is_main:
            trainer.precision.restore(raw)
            if backend is not None:
                _export_lora(cfg, ctx, trainer, raw, backend, tok, result, train_ds, parents)
            else:
                _finish(cfg, ctx, trainer, tok, result, train_ds.fingerprint, parents, "model", {"method": "full"})
        for k in ("status", "steps", "tokens_seen", "final_loss", "best_eval_loss", "checkpoint_dir"):
            ctx.set_output(k, result.get(k))
        return result
    finally:
        D.shutdown_distributed()


def _export_lora(cfg: FinetuneConfig, ctx: Any, trainer: Trainer, raw: nn.Module, backend: LoraBackend, tok: Any, result: dict[str, Any],
                 train_ds: Any, parents: list[str | None]) -> None:
    from forge.core.paths import models_dir
    from forge.train.common import base_lineage, export_model

    run_id = str(getattr(ctx, "run_id", "run"))[-6:]
    lin = base_lineage(cfg, ctx, result, train_ds.fingerprint, parents[0])
    lin.update({"method": cfg.method, "lora": cfg.lora.model_dump(mode="json"), "lora_impl": backend.impl})
    adapter_sd = backend.state_dict(raw)
    adapter_dir = None
    if cfg.save_adapter:
        adapter_dir = backend.save_adapter(raw, Path(ctx.run_dir) / "adapter", base_model=cfg.model.path)
        from forge.train.common import write_lineage

        write_lineage(adapter_dir / "forge_lineage.json", lin)
        ctx.log(f"saved adapter to {adapter_dir}")
        if cfg.register_artifact:
            register_outputs(ctx, "lora_adapter", adapter_dir, parents=[p for p in parents if p], extra={"lineage": lin}, out_key="adapter")
        else:
            ctx.set_output("adapter_dir", str(adapter_dir))
    if not (cfg.save_merged and cfg.save_final_model):
        return
    if D._is_fsdp(trainer.model):
        ctx.log("FSDP: merged export skipped (merge offline from the adapter)", "warning")
        return
    if cfg.method == "qlora":
        # merge into a fresh high-precision copy of the base (the training base is block-quantised)
        fresh_src = cfg.model.model_copy(update={"dtype": cfg.merged_dtype if cfg.merged_dtype != "auto" else "float32"})
        merged, _ = build_model(fresh_src, device=torch.device("cpu"), seed=cfg.seed, tok=tok, ctx=ctx, frozen_base=False)
        b2 = LoraBackend(cfg.lora, prefer_builtin=True)
        b2.apply(merged)
        b2.load_state_dict(merged, adapter_sd)
        merged = b2.merge(merged)
    else:
        merged = backend.merge(raw)
    out = models_dir() / f"{cfg.effective_output_name}-merged-{run_id}"
    with torch.no_grad():
        export_model(merged, out, tok, cfg.merged_dtype, lin)
    ctx.log(f"exported merged model to {out}")
    if cfg.register_artifact:
        register_outputs(ctx, "model", out, parents=[p for p in parents if p], extra={"lineage": lin}, out_key="model")
    else:
        ctx.set_output("model_dir", str(out))
