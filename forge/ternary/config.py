"""Job config for ternary (1.58-bit) training: :class:`TrainConfig` with the ternary QAT mode preselected (EXPERIMENTAL)."""
from __future__ import annotations

from pydantic import model_validator

from forge.train.config import FF, PrecisionConfig, TernaryQATConfig, TrainConfig

__all__ = ["TernaryTrainConfig"]


def _ternary_precision() -> PrecisionConfig:
    # BitNet convention: every block is quantised (embeddings, norms, lm_head stay exempt through exempt_patterns); health sampled every 10 steps.
    return PrecisionConfig(quant_aware="ternary", exempt_first_layers=0, exempt_last_layers=0, diagnostics_every=10, loss_gap_every=50,
                           ternary=TernaryQATConfig())


class TernaryTrainConfig(TrainConfig):
    """Train a model with ternary {-1,0,+1} weights and int8 activations (latent-weight QAT, EXPERIMENTAL; CPU-validated only).

    Identical to the `train` job except that `precision.quant_aware` must be `ternary`.  Tune the quantiser under `precision.ternary`
    (group size, rule, warm-up, collapse thresholds).  Health metrics are emitted under `ternary/*`.
    """

    precision: PrecisionConfig = FF(_ternary_precision, "Precision settings; quant_aware is fixed to ternary for this job.", "Precision")

    @model_validator(mode="after")
    def _must_be_ternary(self) -> "TernaryTrainConfig":
        if self.precision.quant_aware != "ternary":
            raise ValueError("ternary_train requires precision.quant_aware = 'ternary' (use the `train` job for other precisions)")
        return self
