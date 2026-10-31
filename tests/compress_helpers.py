"""Shared helpers for the compress tests (kept out of pytest collection by name)."""
from __future__ import annotations

import torch

from forge.models import ForgeLM
from forge.models.config import ModelConfig, MoEConfig


def tiny_config(**ov) -> ModelConfig:
    base = dict(
        arch="llama", vocab_size=128, hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
        head_dim=16, max_position_embeddings=128,
    )
    base.update(ov)
    return ModelConfig(**base)


def tiny_moe_config(**ov) -> ModelConfig:
    base = dict(
        arch="qwen2_moe", vocab_size=128, hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=16,
        max_position_embeddings=128, moe=MoEConfig(num_experts=4, top_k=2, moe_intermediate_size=32, num_shared_experts=0, norm_topk_prob=True),
    )
    base.update(ov)
    return ModelConfig(**base)


def make_lm(seed: int = 0, outliers: bool = False, dtype=torch.float32, **cfg_ov) -> ForgeLM:
    torch.manual_seed(seed)
    cfg = tiny_moe_config(**cfg_ov) if cfg_ov.pop("_moe", False) else tiny_config(**cfg_ov)
    m = ForgeLM(cfg, dtype=dtype)
    # std 0.02 init is too small to make quantisation error visible next to rounding of tiny logits: rescale to ~0.1
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2 and "embed" not in n:
                p.mul_(4.0)
        if outliers:
            for blk in m.layers:
                blk.input_layernorm.weight[:3] = 12.0
                blk.post_attention_layernorm.weight[5:8] = 12.0
    return m.eval()


def make_moe_lm(seed: int = 0, **kw) -> ForgeLM:
    return make_lm(seed, _moe=True, **kw)


def rand_tokens(n=8, length=32, vocab=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n, length), generator=g)


@torch.no_grad()
def logits_of(model, tokens):
    return model(tokens).logits.float()


def rel_err(a, b):
    return ((a - b).pow(2).sum() / b.pow(2).sum()).item()
