"""Test/demo helpers: a byte-level tokenizer and a tiny HF-llama-shaped decoder (independent full-recompute forward).

Used by the CPU tests and by ``tests_gpu``; contains no GPU code. The tiny model deliberately re-implements RoPE/attention
with plain torch ops so that "engine output == plain full-recompute greedy generate" is a real cross-check of the paged runner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, EOS, PAD = 256, 257, 258
VOCAB = 260


class ByteTokenizer:
    """Bytes 0..255 are ids 0..255; 256 = <bos>, 257 = <eos>, 258 = <pad>."""

    vocab_size = VOCAB
    bos_token_id = BOS
    eos_token_id = EOS
    pad_token_id = PAD

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return ([BOS] + ids) if add_special_tokens else ids

    def __call__(self, text: str, **kw: Any) -> dict:
        return {"input_ids": self.encode(text, kw.get("add_special_tokens", False))}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        bs = bytes(i for i in ids if i < 256) if skip_special_tokens else bytes(i if i < 256 else 63 for i in ids)
        return bs.decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: list[dict], tokenize: bool = True, add_generation_prompt: bool = True, **kw: Any):
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


@dataclass
class TinyConfig:
    vocab_size: int = VOCAB
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 8
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 512
    qk_norm: bool = False
    sliding_window: int | None = None
    tie_word_embeddings: bool = False
    eos_token_id: int = EOS


class TinyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = pos.float().unsqueeze(-1) * inv  # [T, d/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
    h = d // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + rot * sin


class TinyAttn(nn.Module):
    def __init__(self, c: TinyConfig) -> None:
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.hidden_size, c.num_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_heads * c.head_dim, c.hidden_size, bias=False)
        if c.qk_norm:
            self.q_norm = TinyRMSNorm(c.head_dim, c.rms_norm_eps)
            self.k_norm = TinyRMSNorm(c.head_dim, c.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None
        self.window = c.sliding_window
        self.scale = c.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,H] full-sequence causal attention
        c = self.c
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, c.num_heads, c.head_dim)
        k = self.k_proj(x).view(b, t, c.num_kv_heads, c.head_dim)
        v = self.v_proj(x).view(b, t, c.num_kv_heads, c.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        pos = torch.arange(t)
        q, k = _rope(q, pos, c.rope_theta), _rope(k, pos, c.rope_theta)
        g = c.num_heads // c.num_kv_heads
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2).repeat_interleave(g, dim=1)
        vh = v.transpose(1, 2).repeat_interleave(g, dim=1)
        i = torch.arange(t).unsqueeze(1)
        j = torch.arange(t).unsqueeze(0)
        mask = j <= i
        if self.window:
            mask = mask & ((i - j) < self.window)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=self.scale)
        return self.o_proj(o.transpose(1, 2).reshape(b, t, -1))


class TinyMLP(nn.Module):
    def __init__(self, c: TinyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, c: TinyConfig) -> None:
        super().__init__()
        self.input_layernorm = TinyRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attn = TinyAttn(c)
        self.post_attention_layernorm = TinyRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = TinyMLP(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyBackbone(nn.Module):
    def __init__(self, c: TinyConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([TinyLayer(c) for _ in range(c.num_layers)])
        self.norm = TinyRMSNorm(c.hidden_size, c.rms_norm_eps)


class TinyLM(nn.Module):
    def __init__(self, c: TinyConfig | None = None, seed: int = 0) -> None:
        super().__init__()
        c = c or TinyConfig()
        self.config = c
        g = torch.Generator().manual_seed(seed)
        self.model = TinyBackbone(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) * (0.6 / math.sqrt(p.shape[-1]) * 2))
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        h = self.model.norm(x)
        logits = self.lm_head(h)
        return (logits, h) if return_hidden else logits

    @torch.no_grad()
    def greedy_generate(self, prompt_ids: list[int], max_new: int, eos: int | None = EOS) -> list[int]:
        """Plain full-recompute greedy decoding (the oracle for engine tests)."""
        ids = list(prompt_ids)
        out: list[int] = []
        for _ in range(max_new):
            logits = self(torch.tensor([ids]))[0, -1]
            t = int(logits.argmax())
            ids.append(t)
            out.append(t)
            if eos is not None and t == eos:
                break
        return out

    @torch.no_grad()
    def logits_for(self, ids: list[int]) -> torch.Tensor:
        return self(torch.tensor([ids]))[0]


def tiny_model(seed: int = 0, **cfg_overrides: Any) -> TinyLM:
    m = TinyLM(TinyConfig(**cfg_overrides), seed=seed)
    m.eval()
    return m


def make_engine(model: Any | None = None, *, seed: int = 0, tokenizer: Any | None = None, model_overrides: dict | None = None,
                **cfg_kw: Any):
    """Build a CPU ``LLMEngine`` around a tiny model with test-friendly defaults (cfg_kw overrides ServeConfig fields)."""
    from forge.serve.config import ServeConfig
    from forge.serve.engine import LLMEngine

    model = model if model is not None else tiny_model(seed, **(model_overrides or {}))
    base: dict[str, Any] = dict(model="tiny", device="cpu", dtype="float32", block_size=4, num_gpu_blocks=96, max_model_len=160,
                                max_num_seqs=8, max_num_batched_tokens=64)
    base.update(cfg_kw)
    return LLMEngine(ServeConfig(**base), model=model, tokenizer=tokenizer if tokenizer is not None else ByteTokenizer())
