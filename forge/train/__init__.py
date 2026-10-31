"""forge.train: data ingestion/loading, trainer (precision ladder R0/R1/R2), fine-tuning (full/LoRA/QLoRA/DPO), distillation, TR-2 diagnostics.

Public entry points (imported lazily so ``import forge.train`` stays cheap and torch-free):

* configs:  ``forge.train.config`` -> ``TrainConfig``, ``FinetuneConfig``, ``DistillConfig``; ``forge.train.data.IngestConfig``
* jobs:     ``forge.train.data.ingest_dataset``, ``forge.train.run.run_training / run_finetune / run_distill``
* data:     ``forge.train.data.ShardedTokenDataset``, ``forge.train.sft.SFTDataset``
* numerics: ``forge.train.fp4sim``, ``forge.train.precision``, ``forge.train.diagnostics``
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "TrainConfig": "forge.train.config", "FinetuneConfig": "forge.train.config", "DistillConfig": "forge.train.config",
    "IngestConfig": "forge.train.data", "ingest_dataset": "forge.train.data", "ShardedTokenDataset": "forge.train.data",
    "run_training": "forge.train.run", "run_finetune": "forge.train.run", "run_distill": "forge.train.run",
    "SFTDataset": "forge.train.sft", "Trainer": "forge.train.trainer",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'forge.train' has no attribute {name!r}")
    return getattr(importlib.import_module(mod), name)
