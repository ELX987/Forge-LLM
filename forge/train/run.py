"""Job entry points (``train`` / ``finetune`` / ``distill``): ``run_training``, ``run_finetune``, ``run_distill``.

Each takes ``(cfg, ctx)`` where ``cfg`` is the pydantic config (or a plain dict) and ``ctx`` a :class:`forge.core.context.RunContext`.
They emit metrics through ``ctx.metric`` every ``logging.log_every`` steps (keys: ``loss, lr, grad_norm, tokens_per_s,
tokens_seen, step_time_ms, gpu_mem_gb, mfu, update_weight_ratio`` plus ``fp4/*`` TR-2 diagnostics on the R2 rung), report
``ctx.progress``, honour ``ctx.should_stop()`` and return a result dict (also exposed through ``ctx.set_output``).
"""
from __future__ import annotations

import logging
from typing import Any

from forge.core.errors import ConfigError
from forge.train import dist as D
from forge.train.common import (
    base_lineage,
    build_model,
    check_vocab,
    coerce_cfg,
    export_model,
    load_tokenizer_for,
    register_outputs,
)
from forge.train.config import TrainConfig
from forge.train.data import ShardedTokenDataset
from forge.train.trainer import LMTask, Trainer

log = logging.getLogger("forge.train.run")

__all__ = ["run_distill", "run_finetune", "run_training"]


def _maybe_launch(kind: str, cfg: Any, ctx: Any) -> dict[str, Any] | None:
    """Re-run under torchrun when a multi-process strategy is requested but we are a single process."""
    p = cfg.parallel
    if p.strategy != "none" and p.nproc_per_node > 1 and not D.under_torchrun():
        ctx.log(f"parallel.strategy={p.strategy} with nproc_per_node={p.nproc_per_node}: launching torchrun")
        return D.launch_torchrun(kind, cfg, ctx)
    return None


def build_token_datasets(cfg: Any, data: Any, seed: int) -> tuple[ShardedTokenDataset, ShardedTokenDataset | None]:
    if not data.dataset:
        raise ConfigError("data.dataset is empty: ingest a dataset first (job type dataset_ingest)")
    s = seed + data.data_seed
    if data.val_dataset:
        train = ShardedTokenDataset(data.dataset, data.seq_len, seed=s, shuffle=data.shuffle, role="train", val_sequences=0, max_sequences=data.max_sequences)
        val = ShardedTokenDataset(data.val_dataset, data.seq_len, seed=s, shuffle=False, role="train", val_sequences=0)
    else:
        train = ShardedTokenDataset(data.dataset, data.seq_len, seed=s, shuffle=data.shuffle, role="train", val_sequences=data.val_sequences, max_sequences=data.max_sequences)
        val = ShardedTokenDataset(data.dataset, data.seq_len, seed=s, shuffle=False, role="val", val_sequences=data.val_sequences, max_sequences=data.max_sequences) if data.val_sequences else None
    if len(train) == 0:
        raise ConfigError("training dataset has no sequences at this seq_len (too few tokens?)")
    return train, val


def _check_dataset_tokenizers(datasets: list[ShardedTokenDataset | None], tok: Any) -> None:
    """Reject token shards produced by a different vocabulary before the first model step."""
    for ds in datasets:
        if ds is None:
            continue
        for info in ds.infos:
            expected = info.get("tokenizer_hash")
            if expected and expected != tok.hash:
                recorded = info.get("tokenizer") or "unknown tokenizer"
                raise ConfigError(
                    f"dataset tokenizer does not match model tokenizer: dataset was encoded with {recorded!r} "
                    f"(hash {expected}), but model.tokenizer is {tok.name!r} (hash {tok.hash}); "
                    "re-ingest the dataset with the same tokenizer used by this model"
                )


def run_training(cfg: TrainConfig | dict, ctx: Any) -> dict[str, Any]:
    """Continued / small-scale pre-training (job type ``train``)."""
    cfg = coerce_cfg(cfg, TrainConfig)
    launched = _maybe_launch("train", cfg, ctx)
    if launched is not None:
        return launched
    dinfo = D.init_distributed(cfg.parallel)
    try:
        tok = load_tokenizer_for(cfg.model, ctx)
        train_ds, val_ds = build_token_datasets(cfg, cfg.data, cfg.seed)
        _check_dataset_tokenizers([train_ds, val_ds], tok)
        ctx.log(f"data: {train_ds.describe()}")
        model, model_dir = build_model(cfg.model, device=dinfo.device, seed=cfg.seed, tok=tok, ctx=ctx)
        check_vocab(model, tok)
        task = LMTask(train_ds, cfg, val_ds, cfg.data.seq_len, num_workers=cfg.data.num_workers, prefetch_steps=cfg.data.prefetch_steps)
        trainer = Trainer(model, cfg, task, ctx, dinfo=dinfo)
        result = trainer.fit()
        _finish(cfg, ctx, trainer, tok, result, train_ds.fingerprint, [cfg.model.parent_hash], "model")
        return result
    finally:
        D.shutdown_distributed()


def _finish(cfg: Any, ctx: Any, trainer: Trainer, tok: Any, result: dict[str, Any], data_fp: str | None, parents: list[str | None],
            kind: str, lineage_extra: dict[str, Any] | None = None, export: bool = True, model: Any = None) -> None:
    """Export + register the final model (main rank), set outputs."""
    for k in ("status", "steps", "tokens_seen", "final_loss", "best_eval_loss", "checkpoint_dir"):
        ctx.set_output(k, result.get(k))
    if not trainer.dinfo.is_main:
        return
    ctx.set_output("result", {k: v for k, v in result.items() if k not in ("precision",)})
    if not (export and cfg.save_final_model):
        return
    import torch

    from forge.core.paths import models_dir

    raw = model if model is not None else D.unwrap(trainer.model)
    if D._is_fsdp(trainer.model):
        ctx.log("FSDP model: skipping direct export (use the final checkpoint with the export API)", "warning")
        return
    trainer.precision.restore(raw)
    lin = base_lineage(cfg, ctx, result, data_fp, parents[0] if parents else None)
    lin.update(lineage_extra or {})
    out = models_dir() / f"{cfg.effective_output_name}-{str(getattr(ctx, 'run_id', 'run'))[-6:]}"
    with torch.no_grad():
        export_model(raw, out, tok, cfg.final_dtype, lin)
    ctx.log(f"exported model to {out}")
    if cfg.register_artifact:
        register_outputs(ctx, "model", out, parents=[p for p in parents if p], extra={"lineage": lin}, out_key="model")
    else:
        ctx.set_output("model_dir", str(out))


def run_finetune(cfg: Any, ctx: Any) -> dict[str, Any]:
    from forge.train.finetune import run_finetune as _impl

    return _impl(cfg, ctx)


def run_distill(cfg: Any, ctx: Any) -> dict[str, Any]:
    from forge.train.distill import run_distill as _impl

    return _impl(cfg, ctx)
