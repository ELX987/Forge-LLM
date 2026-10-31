"""``ForgeLM``: decoder-only transformer (dense or MoE) used by training, fine-tuning, compression and serving.

Module tree mirrors Hugging Face naming minus the leading ``model.``::

    embed_tokens, layers.<i>.{input_layernorm, self_attn.{q,k,v,o}_proj[, q_norm, k_norm], post_attention_layernorm,
    mlp.{gate,up,down}_proj | mlp.{gate, experts.<j>.*, shared_expert.*, shared_expert_gate}}, norm, lm_head
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from forge.core.errors import ConfigError
from forge.models.attention import Attention, AttnContext, AttnMeta
from forge.models.config import ModelConfig
from forge.models.kvcache import DenseKVCache
from forge.models.layers import MLP, RMSNorm, RotaryEmbedding
from forge.models.linear import make_linear
from forge.models.moe import MoEBlock

log = logging.getLogger("forge.models")


@dataclass
class ModelOutput:
    """Result of :meth:`ForgeLM.forward`."""

    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None

    def to_tuple(self) -> tuple:
        return (self.logits, self.loss, self.hidden_states, self.aux_loss)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, dtype=None, device=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = cfg.is_moe_layer(layer_idx)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_unit_offset, dtype=dtype, device=device)
        self.self_attn = Attention(cfg, layer_idx, dtype=dtype, device=device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_unit_offset, dtype=dtype, device=device)
        if self.is_moe:
            self.mlp: nn.Module = MoEBlock(cfg, dtype=dtype, device=device)
        else:
            self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act, dtype=dtype, device=device)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: AttnContext, want_aux: bool = False):
        h = h + self.self_attn(self.input_layernorm(h), cos, sin, ctx)
        x = self.post_attention_layernorm(h)
        if self.is_moe:
            y, aux = self.mlp(x, want_aux)
        else:
            y, aux = self.mlp(x), h.new_zeros((), dtype=torch.float32)
        return h + y, aux


class ForgeLM(nn.Module):
    """Decoder-only transformer. See module docstring and ``docs/ARCHITECTURE.md`` section 5.4."""

    def __init__(self, config: ModelConfig, dtype: torch.dtype | None = None, device: Any = None, init_weights: bool = True):
        super().__init__()
        self.config = config
        c = config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList([DecoderLayer(c, i, dtype=dtype, device=device) for i in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps, c.norm_unit_offset, dtype=dtype, device=device)
        self.lm_head = make_linear(c.hidden_size, c.vocab_size, False, dtype=dtype, device=device)
        self.rotary = RotaryEmbedding(c)
        self.attn_impl = "auto"
        self.gradient_checkpointing = False
        self._gc_kwargs: dict[str, Any] = {"use_reentrant": False}
        self.loss_chunk_size = 0
        if init_weights:
            self.init_weights()
        self.tie_weights()

    # ------------------------------------------------------------------ setup helpers
    def init_weights(self) -> None:
        std = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                if m.weight.is_meta:
                    continue
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm) and not m.weight.is_meta:
                with torch.no_grad():
                    m.weight.fill_(0.0 if m.unit_offset else 1.0)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen: set[int] = set()
        n = 0
        for p in self.parameters():
            if id(p) in seen or (trainable_only and not p.requires_grad):
                continue
            seen.add(id(p))
            n += p.numel()
        return n

    def set_attn_impl(self, impl: str) -> None:
        """``auto`` | ``sdpa`` | ``flash_attn`` | ``eager``."""
        if impl not in ("auto", "sdpa", "flash_attn", "eager"):
            raise ConfigError(f"unknown attention implementation {impl!r}")
        self.attn_impl = impl

    def gradient_checkpointing_enable(self, **kw: Any) -> None:
        self.gradient_checkpointing = True
        self._gc_kwargs = {"use_reentrant": False, **kw}

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def new_kv_cache(self, batch: int, max_len: int, kv_dtype: str | None = None, **kw: Any) -> DenseKVCache:
        """Dense KV cache sized for this model (dtype follows the model unless ``kv_dtype`` given)."""
        return DenseKVCache(batch, max_len, self.config, dtype=self.dtype, kv_dtype=kv_dtype, device=self.device, **kw)

    # ------------------------------------------------------------------ forward
    def _loss_from_hidden(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hs = h[:, :-1].reshape(-1, h.shape[-1])
        ls = labels[:, 1:].reshape(-1).to(h.device)
        count = (ls != -100).sum().clamp_min(1)
        chunk = self.loss_chunk_size or hs.shape[0]
        total = hs.new_zeros((), dtype=torch.float32)
        for s in range(0, hs.shape[0], chunk):
            hc, lc = hs[s : s + chunk], ls[s : s + chunk]
            if torch.is_grad_enabled() and self.loss_chunk_size and hc.requires_grad:
                total = total + checkpoint(self._ce_sum, hc, lc, use_reentrant=False)
            else:
                total = total + self._ce_sum(hc, lc)
        return total / count

    def _ce_sum(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.lm_head(h).float(), labels, ignore_index=-100, reduction="sum")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        attn_meta: AttnMeta | dict | None = None,
        labels: torch.Tensor | None = None,
        return_hidden: bool = False,
        logits_to_keep: int | torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> ModelOutput:
        """Run the model.

        ``input_ids [B,T]``; ``positions [B,T]`` absolute positions (default: cache length + arange); ``kv_cache``: dense or paged cache
        (updated and committed in place); ``labels [B,T]`` (-100 ignored) -> next-token CE loss (+ MoE aux loss while training);
        ``logits_to_keep``: only compute logits for the last N positions (int) or given indices; ``return_logits=False`` with labels
        computes the loss without materialising logits (use with ``loss_chunk_size``).
        """
        if isinstance(attn_meta, dict):
            attn_meta = AttnMeta(**attn_meta)
        meta = attn_meta or AttnMeta()
        b, t = input_ids.shape
        dev = input_ids.device
        am = meta.attention_mask
        n_new = meta.n_new
        if n_new is None and am is not None:
            n_new = am.long().sum(dim=1)
        if positions is None:
            base = kv_cache.seq_lens.to(dev).unsqueeze(1) if kv_cache is not None else 0
            positions = base + torch.arange(t, device=dev).unsqueeze(0).expand(b, t)
        kv_lens = kv_cache.end_lens(n_new, t) if kv_cache is not None else None
        ctx = AttnContext(
            positions=positions,
            cache=kv_cache,
            attention_mask=am,
            n_new=n_new,
            kv_lens=kv_lens,
            impl=self.attn_impl,
            use_paged_kernel=meta.use_paged_kernel,
        )
        h = self.embed_tokens(input_ids)
        if self.config.scale_embeddings:
            h = h * torch.tensor(self.config.hidden_size**0.5, dtype=h.dtype)
        cos, sin = self.rotary(positions)
        want_aux = labels is not None or self.training
        aux_total = None
        use_gc = self.gradient_checkpointing and self.training and kv_cache is None and torch.is_grad_enabled()
        for layer in self.layers:
            if use_gc:
                h, aux = checkpoint(layer, h, cos, sin, ctx, want_aux, **self._gc_kwargs)
            else:
                h, aux = layer(h, cos, sin, ctx, want_aux)
            if layer.is_moe:
                aux_total = aux if aux_total is None else aux_total + aux
        h = self.norm(h)
        if kv_cache is not None:
            kv_cache.commit(n_new if n_new is not None else t)

        aux_loss = None
        if aux_total is not None:
            aux_loss = aux_total / self.config.num_moe_layers

        loss = None
        if labels is not None:
            if logits_to_keep is not None:
                raise ConfigError("labels and logits_to_keep cannot be combined")
            loss = self._loss_from_hidden(h, labels)
            if aux_loss is not None and self.config.moe is not None:
                loss = loss + self.config.moe.router_aux_loss_coef * aux_loss

        logits = None
        if return_logits or labels is None:
            hk = h
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                hk = h[:, -logits_to_keep:]
            elif isinstance(logits_to_keep, torch.Tensor):
                hk = h[:, logits_to_keep]
            logits = self.lm_head(hk)
        return ModelOutput(logits=logits, loss=loss, hidden_states=h if return_hidden else None, aux_loss=aux_loss)

    # ------------------------------------------------------------------ generation (reference, uniform-length batches)
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        seed: int | None = None,
        kv_dtype: str | None = None,
    ) -> torch.Tensor:
        """Simple KV-cached autoregressive generation (greedy when ``temperature==0``). All rows must have equal length."""
        was_training = self.training
        self.eval()
        b, t = input_ids.shape
        cache = self.new_kv_cache(b, t + max_new_tokens, kv_dtype=kv_dtype)
        gen = torch.Generator(device=input_ids.device)
        gen.manual_seed(0 if seed is None else seed)
        out = self(input_ids, kv_cache=cache, logits_to_keep=1).logits[:, -1]
        seq = input_ids
        done = torch.zeros(b, dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            nxt = _sample(out.float(), temperature, top_k, top_p, gen)
            if eos_token_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_token_id), nxt)
                done = done | (nxt == eos_token_id)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            if eos_token_id is not None and bool(done.all()):
                break
            out = self(nxt.unsqueeze(1), kv_cache=cache).logits[:, -1]
        self.train(was_training)
        return seq


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, gen: torch.Generator) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cp = torch.softmax(sl, dim=-1).cumsum(-1)
        remove = cp - torch.softmax(sl, dim=-1) > top_p
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)
