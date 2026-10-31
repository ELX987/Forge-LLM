"""Shared fixtures/helpers for the train test-suite (no tests of its own)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 288  # >= ByteTokenizer.vocab_size (272)

TEXTS = [
    "the quick brown fox jumps over the lazy dog. " * 4,
    "pack my box with five dozen liquor jugs! " * 4,
    "how vexingly quick daft zebras jump. " * 4,
    "sphinx of black quartz, judge my vow. " * 4,
]


class _RMS(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class _Block(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.input_layernorm = _RMS(d)
        self.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, n, nn.Linear(d, d, bias=False))
        self.post_attention_layernorm = _RMS(d)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, 2 * d, bias=False)
        self.mlp.up_proj = nn.Linear(d, 2 * d, bias=False)
        self.mlp.down_proj = nn.Linear(2 * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        a = self.self_attn
        hn = self.input_layernorm(x)
        q, k, v = (p(hn).view(B, T, self.h, D // self.h).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + a.o_proj(o)
        hn = self.post_attention_layernorm(x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(hn)) * self.mlp.up_proj(hn))


class LocalTinyLM(nn.Module):
    """Independent minimal decoder with the same forward contract as ForgeLM (logits via forward(input_ids))."""

    def __init__(self, vocab: int = VOCAB, d: int = 64, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab, hidden_size=d, num_layers=layers)
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Block(d, heads) for _ in range(layers)])
        self.norm = _RMS(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: object):
        h = self.embed_tokens(input_ids)
        for blk in self.layers:
            h = blk(h)
        logits = self.lm_head(self.norm(h))
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss, aux_loss=None)


def make_model(seed: int = 0, kind: str = "local", layers: int = 3) -> nn.Module:
    """``kind='local'`` = independent fixture; ``'forge'`` = forge.models.tiny (skip if unavailable)."""
    torch.manual_seed(seed)
    if kind == "forge":
        from forge.models.tiny import build_tiny_model

        return build_tiny_model(seed, vocab_size=VOCAB, num_layers=layers)
    return LocalTinyLM(layers=layers)


def write_corpus(tmp: Path, n_repeat: int = 6) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "corpus.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in range(n_repeat):
            for i, t in enumerate(TEXTS):
                f.write(json.dumps({"text": f"[{r}.{i}] " + t}) + "\n")
    return p


def ingest_tiny(tmp: Path, name: str = "tiny_ds", **kw) -> Path:
    from forge.core.context import NullContext
    from forge.train.data import IngestConfig, ingest_dataset

    src = write_corpus(tmp)
    cfg = IngestConfig(name=name, sources=[{"path": str(src), "kind": "jsonl"}], tokenizer="byte", output_dir=str(tmp / name),
                       shard_size_mib=0.004, **kw)
    ingest_dataset(cfg, NullContext(run_dir=tmp / "ingest_run"))
    return tmp / name


def write_sft(tmp: Path, n: int = 24) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "sft.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"say {i % 4}"}, {"role": "assistant", "content": f"it is {i % 4}!"}]}) + "\n")
    return p


def small_train_cfg(ds_dir: Path, **over):
    """TrainConfig for a CPU smoke run."""
    from forge.train.config import TrainConfig

    cfg = {
        "name": "t", "seed": 3, "model": {"source": "tiny", "tokenizer": "byte"},
        "micro_batch_size": 2, "grad_accum_steps": 2,
        "data": {"dataset": [str(ds_dir)], "seq_len": 16, "val_sequences": 4, "num_workers": 0},
        "optim": {"lr": 3e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "schedule": {"name": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.1},
        "budget": {"max_steps": 6}, "checkpoint": {"every_steps": 0, "keep_last": 3},
        "eval": {"every_steps": 0, "num_batches": 2}, "logging": {"log_every": 1},
        "save_final_model": False, "register_artifact": False,
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return TrainConfig.model_validate(cfg)
