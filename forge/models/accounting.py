"""Parameter / FLOP / memory accounting for configs and instantiated models (used by the planner and the UI)."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from forge.models.config import ModelConfig
from forge.models.linear import PackedLinear


def kv_bytes_per_token(cfg: ModelConfig, kv_bits: float = 16, group_size: int | None = None, scale_bits: int = 16) -> int:
    """K+V cache bytes per token (see :meth:`ModelConfig.kv_bytes_per_token`)."""
    return cfg.kv_bytes_per_token(kv_bits, group_size=group_size, scale_bits=scale_bits)


def model_summary(cfg: ModelConfig, seq_len: int = 2048) -> dict[str, Any]:
    """JSON-able summary for ``/models/{id}/inspect`` and the capacity planner (all values derived analytically)."""
    return {
        "arch": cfg.arch,
        "params": cfg.param_count(),
        "active_params": cfg.active_param_count(),
        "params_b": cfg.param_count() / 1e9,
        "embedding_params": cfg.vocab_size * cfg.hidden_size * (1 if cfg.tie_word_embeddings else 2),
        "kv_bytes_per_token": {str(b): cfg.kv_bytes_per_token(b) for b in (16, 8, 4)},
        "weight_bytes": {str(b): cfg.weight_bytes(b) for b in (16, 8, 4)},
        "flops_per_token": {
            "train": cfg.flops_per_token(seq_len, "train"),
            "prefill": cfg.flops_per_token(seq_len, "prefill"),
            "decode": cfg.flops_per_token(seq_len, "decode"),
        },
        "num_layers": cfg.num_layers,
        "kv_heads": cfg.kv_heads,
        "head_dim": cfg.head_dim,
        "moe_layers": cfg.num_moe_layers,
        "provenance": "derived",
    }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Counts of an instantiated model: total / trainable parameters, buffer bytes and bytes per linear kind."""
    seen: set[int] = set()
    total = trainable = 0
    nbytes = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        nbytes += p.numel() * p.element_size()
        if p.requires_grad:
            trainable += p.numel()
    buf = sum(b.numel() * b.element_size() for b in model.buffers())
    by_kind: dict[str, int] = {}
    for _, m in model.named_modules():
        if isinstance(m, PackedLinear):
            by_kind[m.kind] = by_kind.get(m.kind, 0) + m.packed_nbytes()
    return {"total": total, "trainable": trainable, "param_bytes": nbytes, "buffer_bytes": buf, **{f"bytes_{k}": v for k, v in by_kind.items()}}
