"""Tiny configs / models for tests and smoke runs everywhere in Forge (never downloads anything)."""
from __future__ import annotations

from typing import Any

import torch

from forge.models.config import ModelConfig, MoEConfig
from forge.models.modeling import ForgeLM


def make_tiny_config(**overrides: Any) -> ModelConfig:
    """2-layer, hidden-64 llama-style config (GQA 4/2 heads). Any ``ModelConfig`` field can be overridden.

    ``moe=True`` is shorthand for a small 4-expert top-2 MoE block (arch ``qwen3_moe``-style when not otherwise set).
    """
    ov = dict(overrides)
    moe = ov.pop("moe", None)
    base: dict[str, Any] = dict(
        arch="llama",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    if moe is True:
        moe = MoEConfig(num_experts=4, top_k=2, moe_intermediate_size=32)
        base.setdefault("arch", "qwen3_moe")
        if "arch" not in ov:
            ov["arch"] = "qwen3_moe"
            base.update(qk_norm=True)
    elif isinstance(moe, dict):
        moe = MoEConfig(**moe)
    base.update(ov)
    if moe is not None and moe is not False:
        base["moe"] = moe
    return ModelConfig(**base)


def build_tiny_model(seed: int = 0, dtype: torch.dtype = torch.float32, **overrides: Any) -> ForgeLM:
    """Deterministically initialised tiny :class:`ForgeLM` on CPU (float32 by default)."""
    cfg = make_tiny_config(**overrides)
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        model = ForgeLM(cfg, dtype=dtype)
    finally:
        torch.random.set_rng_state(gen_state)
    return model
