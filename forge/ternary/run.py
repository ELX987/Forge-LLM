"""Job entrypoints for ternary work (phase P2: training; convert / finetune / export arrive in P3)."""
from __future__ import annotations

from typing import Any

from forge.ternary.config import TernaryTrainConfig
from forge.train.common import coerce_cfg

__all__ = ["TernaryTrainConfig", "run_ternary_train"]


def run_ternary_train(cfg: TernaryTrainConfig | dict, ctx: Any) -> dict[str, Any]:
    """Ternary QAT training: the standard trainer with the ternary precision plug-in (see :mod:`forge.ternary.qat`)."""
    from forge.train.run import run_training

    cfg = coerce_cfg(cfg, TernaryTrainConfig)
    ctx.log("ternary_train: EXPERIMENTAL 1.58-bit QAT (validated on CPU only; see docs/TERNARY.md)", "warning")
    return run_training(cfg, ctx)
