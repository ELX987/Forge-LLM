"""Model execution over a paged KV pool: :class:`KVPool` (tensors) and :class:`ModelRunner` (ragged forward).

The runner drives a decoder-only model *module by module* (``embed_tokens -> [input_layernorm, self_attn.{q,k,v,o}_proj
(+ q_norm/k_norm), post_attention_layernorm, mlp] x L -> norm -> lm_head``), so quantised / LoRA / packed linears work
transparently - they are just called as modules on ``[tokens, hidden]`` activations. Both ``forge.models.ForgeLM``
(HF-style naming) and any HF-llama-like module tree are supported; MoE ``mlp`` modules may return ``(out, aux)``.

Hybrid models (lfm2: ``layers.<i>.conv`` instead of ``self_attn``): the KV pool only has layers for the attention layers (compact
index ``ModelDims.attn_layers.index(i)``); every conv layer keeps a per-sequence ``[H, K-1]`` state in the runner
(``SeqInput.state_key``; a chunk with ``start_pos == 0`` starts from zeros, so recompute-preemption needs no special handling).
That state is not a function of KV blocks, hence the engine disables prefix caching and refuses speculation / swap preemption.

Pool layout follows the kernel contract (``forge/kernels/ref/attn_kv.py``): per layer ``[num_blocks+1, Hkv, block_size, D]``
(head-major pages); the extra last block is a scratch block used for padding rows (CUDA graphs) and never allocated.

Attention paths
* **reference** (CPU / any device): per-sequence gather of the block table + torch SDPA with an explicit causal/window mask.
  Handles arbitrary chunk lengths (prefill chunks, speculative verification) and every KV dtype. This is the path all
  CPU tests exercise.
* **kernel** (CUDA, ``kernel_attention`` auto/on): when *every* sequence in the step has exactly one new token (pure decode)
  and the KV dtype is bf16/fp16/fp32/fp8_e4m3, ``forge.kernels.ops_attn.paged_decode_attention`` is used; any exception
  disables it for the rest of the process and falls back to the reference path (logged once). Implemented, **not validated
  on GPU hardware**.
* **CUDA graphs** (optional): decode steps are padded to power-of-two batch buckets and replayed. Same caveat.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

from forge.core.errors import ConfigError

log = logging.getLogger("forge.serve.runner")

KERNEL_KV_DTYPES = ("bf16", "fp16", "fp32", "fp8_e4m3")


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def resolve_device(name: str = "auto") -> Any:
    import torch

    if name in ("auto", "", None):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: Any) -> Any:
    import torch

    if name != "auto":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]
    if device.type == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:  # pragma: no cover
            major = 8
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def normalize_kv_name(kv_dtype: str, compute_dtype: Any) -> str:
    import torch

    if kv_dtype == "auto":
        return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[compute_dtype]
    return kv_dtype


def kv_bytes_per_block(num_layers: int, num_kv_heads: int, head_dim: int, block_size: int, kv_name: str) -> int:
    elem = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8_e4m3": 1, "int8": 1}[kv_name]
    b = 2 * num_layers * block_size * num_kv_heads * head_dim * elem
    if kv_name == "int8":
        b += 2 * num_layers * block_size * num_kv_heads * 4  # per-token-per-head fp32 scales
    return b


# --------------------------------------------------------------------------------------------------
# KV pool
# --------------------------------------------------------------------------------------------------
class KVPool:
    """Dense tensors backing the paged KV cache; implements the ``BlockStorage`` protocol of ``KVBlockManager``."""

    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int,
                 compute_dtype: Any, kv_name: str, device: Any) -> None:
        import torch

        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.compute_dtype = compute_dtype
        self.kv_name = kv_name
        self.device = device
        self.scratch = self.num_blocks  # extra block id
        if kv_name in ("bf16", "fp16", "fp32"):
            store = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[kv_name]
        elif kv_name == "fp8_e4m3":
            if not hasattr(torch, "float8_e4m3fn"):
                raise ConfigError("this torch build has no float8_e4m3fn; use kv_dtype bf16/fp16 or int8")
            store = torch.uint8  # bit view of float8_e4m3fn: avoids dtype gaps in index_put on some torch builds;
            #                       the kernel contract (kernels/ref/attn_kv.py) accepts the uint8 view
        elif kv_name == "int8":
            store = torch.int8
        else:
            raise ConfigError(f"unsupported kv_dtype {kv_name!r} for the paged pool (int4 KV is planner-only)")
        shape = (self.num_layers, self.num_blocks + 1, self.num_kv_heads, self.block_size, self.head_dim)
        self.k = torch.zeros(shape, dtype=store, device=device)
        self.v = torch.zeros(shape, dtype=store, device=device)
        self.quantised_scales = kv_name == "int8"
        if self.quantised_scales:
            sshape = (self.num_layers, self.num_blocks + 1, self.num_kv_heads, self.block_size)
            self.ks = torch.ones(sshape, dtype=torch.float32, device=device)
            self.vs = torch.ones(sshape, dtype=torch.float32, device=device)
        self.bytes_per_block = kv_bytes_per_block(num_layers, num_kv_heads, head_dim, block_size, kv_name)

    # ---- write / gather -------------------------------------------------------------------
    def _quant(self, x: Any) -> tuple[Any, Any]:
        import torch

        if self.kv_name == "fp8_e4m3":
            return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8), None
        if self.kv_name == "int8":
            xf = x.float()
            scale = (xf.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)  # [T,H]
            q = torch.round(xf / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
            return q, scale
        return x.to(self.k.dtype), None

    def write(self, layer: int, blk: Any, off: Any, k: Any, v: Any) -> None:
        """Store ``k, v [T, Hkv, D]`` at (block ``blk[t]``, offset ``off[t]``)."""
        kq, ks = self._quant(k)
        vq, vs = self._quant(v)
        self.k[layer][blk, :, off, :] = kq
        self.v[layer][blk, :, off, :] = vq
        if self.quantised_scales:
            self.ks[layer][blk, :, off] = ks
            self.vs[layer][blk, :, off] = vs

    def gather(self, layer: int, block_ids: Any, length: int) -> tuple[Any, Any]:
        """Return dequantised ``k, v [length, Hkv, D]`` in compute dtype for a block table."""
        kb = self.k[layer][block_ids]  # [nb, Hkv, bs, D]
        vb = self.v[layer][block_ids]
        nb = kb.shape[0]
        if self.kv_name == "fp8_e4m3":
            import torch

            kb, vb = kb.view(torch.float8_e4m3fn), vb.view(torch.float8_e4m3fn)
        kf = kb.permute(0, 2, 1, 3).reshape(nb * self.block_size, self.num_kv_heads, self.head_dim)[:length]
        vf = vb.permute(0, 2, 1, 3).reshape(nb * self.block_size, self.num_kv_heads, self.head_dim)[:length]
        if self.quantised_scales:
            ksb = self.ks[layer][block_ids].permute(0, 2, 1).reshape(nb * self.block_size, self.num_kv_heads)[:length]
            vsb = self.vs[layer][block_ids].permute(0, 2, 1).reshape(nb * self.block_size, self.num_kv_heads)[:length]
            return ((kf.float() * ksb.unsqueeze(-1)).to(self.compute_dtype),
                    (vf.float() * vsb.unsqueeze(-1)).to(self.compute_dtype))
        return kf.to(self.compute_dtype), vf.to(self.compute_dtype)

    # ---- BlockStorage protocol --------------------------------------------------------------
    def read_block(self, bid: int) -> dict[str, Any]:
        d = {"k": self.k[:, bid].detach().to("cpu", copy=True), "v": self.v[:, bid].detach().to("cpu", copy=True)}
        if self.quantised_scales:
            d["ks"] = self.ks[:, bid].detach().to("cpu", copy=True)
            d["vs"] = self.vs[:, bid].detach().to("cpu", copy=True)
        return d

    def write_block(self, bid: int, payload: dict[str, Any]) -> None:
        self.k[:, bid] = payload["k"].to(self.device)
        self.v[:, bid] = payload["v"].to(self.device)
        if self.quantised_scales:
            self.ks[:, bid] = payload["ks"].to(self.device)
            self.vs[:, bid] = payload["vs"].to(self.device)

    def copy_block(self, src: int, dst: int) -> None:
        self.k[:, dst] = self.k[:, src]
        self.v[:, dst] = self.v[:, src]
        if self.quantised_scales:
            self.ks[:, dst] = self.ks[:, src]
            self.vs[:, dst] = self.vs[:, src]


# --------------------------------------------------------------------------------------------------
# model adapter
# --------------------------------------------------------------------------------------------------
def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def _cfg_get(cfg: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for n in names:
        v = cfg.get(n) if isinstance(cfg, dict) else getattr(cfg, n, None)
        if v is not None:
            return v
    return default


def check_servable(cfg: Any) -> None:
    """Refuse configs whose block structure the paged runner does not implement (it would crash or silently misbehave)."""
    bad: list[str] = []
    arch = _cfg_get(cfg, ("arch", "model_type"), None)
    if isinstance(arch, str) and arch.startswith("gemma4"):
        bad.append(f"arch {arch}")
    if _cfg_get(cfg, ("sandwich_norm",), False):
        bad.append("sandwich_norm")
    if int(_cfg_get(cfg, ("per_layer_input_size", "hidden_size_per_layer_input"), 0) or 0):
        bad.append("per_layer_input_size")
    if int(_cfg_get(cfg, ("num_kv_shared_layers",), 0) or 0):
        bad.append("num_kv_shared_layers")
    ldim = getattr(cfg, "layer_head_dim", None)
    n = _cfg_get(cfg, ("num_layers", "num_hidden_layers"), 0)
    if callable(ldim) and n:
        try:
            if len({int(ldim(i)) for i in range(int(n))}) > 1:
                bad.append("non-uniform per-layer head_dim")
        except Exception:  # noqa: BLE001
            pass
    if bad:
        raise ConfigError("serving Gemma 4 style models is not supported yet (" + ", ".join(bad) + "); training, finetuning and "
                          "compress-free inspection work, but the paged runner lacks sandwich norms, PLE, layer_scalar, "
                          "KV sharing, per-layer head_dim/rope and is unsafe for them")


@dataclass
class ModelDims:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 4096
    sliding_window: int | None = None
    rotary_dim: int | None = None
    scale_embeddings: bool = False
    conv_layers: tuple[int, ...] = ()  # lfm2 gated-conv layers (no KV cache, per-sequence recurrent state)
    conv_kernel: int = 3

    @property
    def attn_layers(self) -> list[int]:
        """Absolute indices of the layers that own a KV cache; their position in this list is the KV-pool layer index."""
        c = set(self.conv_layers)
        return [i for i in range(self.num_layers) if i not in c]

    @property
    def num_kv_layers(self) -> int:
        return self.num_layers - len(self.conv_layers)

    @classmethod
    def from_config(cls, cfg: Any, model: Any = None) -> ModelDims:
        h = int(_cfg_get(cfg, ("hidden_size",)))
        nh = int(_cfg_get(cfg, ("num_heads", "num_attention_heads")))
        nkv = int(_cfg_get(cfg, ("num_kv_heads", "num_key_value_heads"), nh))
        hd = _cfg_get(cfg, ("head_dim",), None)
        n_layers = int(_cfg_get(cfg, ("num_layers", "num_hidden_layers")))
        lt = _cfg_get(cfg, ("layer_types",), None)
        fa = _cfg_get(cfg, ("full_attn_idxs",), None)
        if isinstance(lt, (list, tuple)):
            conv = tuple(i for i, t in enumerate(lt) if t == "conv")
        elif fa is not None:
            conv = tuple(i for i in range(n_layers) if i not in {int(x) for x in fa})
        else:
            conv = ()
        return cls(
            conv_layers=conv,
            conv_kernel=int(_cfg_get(cfg, ("conv_kernel", "conv_L_cache"), 3)),
            hidden_size=h,
            num_layers=int(_cfg_get(cfg, ("num_layers", "num_hidden_layers"))),
            num_heads=nh, num_kv_heads=nkv, head_dim=int(hd) if hd else h // nh,
            vocab_size=int(_cfg_get(cfg, ("vocab_size",))),
            rope_theta=float(_cfg_get(cfg, ("rope_theta",), 10000.0)),
            rope_scaling=_cfg_get(cfg, ("rope_scaling",), None),
            max_position_embeddings=int(_cfg_get(cfg, ("max_position_embeddings",), 4096)),
            sliding_window=_cfg_get(cfg, ("sliding_window",), None),
            scale_embeddings=bool(_cfg_get(cfg, ("scale_embeddings",), False)),
        )


class ModelAdapter:
    """Locates the pieces of a decoder-only model needed for the functional paged forward."""

    def __init__(self, model: Any, cfg: Any | None = None) -> None:
        self.model = model
        cfg = cfg if cfg is not None else getattr(model, "config", None)
        if cfg is None:
            raise ConfigError("model has no .config; pass model_cfg explicitly")
        check_servable(cfg)
        self.dims = ModelDims.from_config(cfg, model)
        self.cfg = cfg
        self.softcap = _cfg_get(cfg, ("final_logit_softcapping",), None)
        backbone = getattr(model, "model", None)
        if backbone is None or not hasattr(backbone, "layers"):
            backbone = model
        self.backbone = backbone
        self.embed = _first_attr(backbone, ("embed_tokens", "tok_embeddings", "wte")) or _first_attr(model, ("embed_tokens",))
        self.layers = _first_attr(backbone, ("layers", "h", "blocks"))
        self.norm = _first_attr(backbone, ("norm", "final_norm", "ln_f"))
        self.lm_head = getattr(model, "lm_head", None)
        if self.embed is None or self.layers is None or self.norm is None:
            raise ConfigError("unsupported model structure: need embed_tokens, layers and norm (HF-llama style naming)")
        if len(self.layers) != self.dims.num_layers:
            raise ConfigError(f"config says {self.dims.num_layers} layers but model has {len(self.layers)}")
        self.attn_names: list[dict[str, Any]] = []
        for i, layer in enumerate(self.layers):
            attn = _first_attr(layer, ("self_attn", "attn", "attention"))
            ln1 = _first_attr(layer, ("input_layernorm", "attn_norm", "ln1", "attention_norm", "operator_norm"))
            ln2 = _first_attr(layer, ("post_attention_layernorm", "mlp_norm", "ln2", "ffn_norm"))
            mlp = _first_attr(layer, ("mlp", "feed_forward", "ffn", "block_sparse_moe"))
            if i in self.dims.conv_layers:
                conv = getattr(layer, "conv", None)
                if conv is None or ln1 is None or ln2 is None or mlp is None or not hasattr(conv, "in_proj") or not hasattr(conv, "out_proj"):
                    raise ConfigError(f"layer {i}: conv layer needs conv.{{in_proj, conv, out_proj}}, layernorms and mlp")
                self.attn_names.append({"kind": "conv", "conv": conv, "ln1": ln1, "ln2": ln2, "mlp": mlp, "ci": self.dims.conv_layers.index(i)})
                continue
            if attn is None or ln1 is None or ln2 is None or mlp is None:
                raise ConfigError(f"layer {i}: could not find self_attn / layernorms / mlp")
            for need in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if not hasattr(attn, need):
                    raise ConfigError(f"layer {i}: attention lacks {need} (fused qkv is not supported by the paged runner)")
            self.attn_names.append({"kind": "attn", "pi": self.dims.attn_layers.index(i),
                                    "attn": attn, "ln1": ln1, "ln2": ln2, "mlp": mlp,
                                    "q_norm": getattr(attn, "q_norm", None), "k_norm": getattr(attn, "k_norm", None),
                                    "window": getattr(attn, "window", None) if hasattr(attn, "window") else self.dims.sliding_window,
                                    "scale": getattr(attn, "scale", None)})

    def logits(self, h: Any) -> Any:
        """``h [R, H]`` (post final norm) -> float32 logits ``[R, V]``."""
        import torch
        import torch.nn.functional as F

        if self.lm_head is not None:
            out = self.lm_head(h)
        else:
            out = F.linear(h, self.embed.weight)
        if isinstance(out, tuple):
            out = out[0]
        out = out.to(torch.float32)
        cap = getattr(self, "softcap", None)
        if cap:
            out = torch.tanh(out / float(cap)) * float(cap)
        return out


# --------------------------------------------------------------------------------------------------
# rotary table
# --------------------------------------------------------------------------------------------------
class RotaryTable:
    def __init__(self, dims: ModelDims, max_len: int, device: Any) -> None:
        import torch

        try:
            from forge.models.layers import compute_inv_freq

            inv, att = compute_inv_freq(dims.head_dim, dims.rope_theta, dims.rope_scaling, dims.max_position_embeddings)
        except ImportError:  # pragma: no cover - models package missing
            if dims.rope_scaling:
                raise
            inv = 1.0 / (dims.rope_theta ** (torch.arange(0, dims.head_dim, 2, dtype=torch.float32) / dims.head_dim))
            att = 1.0
        pos = torch.arange(max_len, dtype=torch.float32)
        freqs = pos.unsqueeze(-1) * inv.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos = (emb.cos() * att).to(device)  # [max_len, D]
        self.sin = (emb.sin() * att).to(device)
        self.max_len = max_len

    def apply(self, x: Any, positions: Any) -> Any:
        """``x [T, H, D]``; positions ``[T]`` long."""
        import torch

        cos = self.cos[positions].to(x.dtype).unsqueeze(1)
        sin = self.sin[positions].to(x.dtype).unsqueeze(1)
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * cos + rot * sin


# --------------------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------------------
@dataclass
class SeqInput:
    """One sequence's slice of an engine step."""

    token_ids: list[int]
    start_pos: int
    block_table: list[int]
    logits_from: int = -1  # first chunk index (0-based) to return logits for; -1 = last token only; len(token_ids) = none
    state_key: Any = None  # hybrid models: identity of the sequence's conv state (the request id); required when conv layers exist


@dataclass
class RunnerOutput:
    logits: list[Any] = field(default_factory=list)   # per sequence: [k, V] float32 or None
    hidden: list[Any] = field(default_factory=list)   # per sequence: [k, H] (post-norm) or None


class ModelRunner:
    def __init__(self, model: Any, pool: KVPool, *, model_cfg: Any | None = None, max_model_len: int = 4096,
                 kernel_attention: str = "auto", cuda_graphs: bool = False, cuda_graph_max_batch: int = 32) -> None:
        import torch

        self.ad = ModelAdapter(model, model_cfg)
        self.pool = pool
        self.dims = self.ad.dims
        self.device = pool.device
        self.bs = pool.block_size
        d = self.dims
        if pool.num_layers != d.num_kv_layers or pool.num_kv_heads != d.num_kv_heads or pool.head_dim != d.head_dim:
            raise ConfigError("KV pool shape does not match the model")
        self.rope = RotaryTable(d, max(max_model_len, 8), self.device)
        self._kernel_mode = kernel_attention
        self._kernel_ok = (
            kernel_attention != "off" and pool.kv_name in KERNEL_KV_DTYPES
            and os.environ.get("FORGE_FORCE_REF", "0") != "1"
            and (kernel_attention == "on" or self.device.type == "cuda")
        )
        self._graphs: dict[int, Any] = {}
        self.hybrid = bool(d.conv_layers)
        self._conv_state: dict[Any, Any] = {}  # state_key -> [n_conv_layers, H, K-1]
        self._conv_step: list[tuple[int, int, Any, bool]] = []  # per sequence of the running step: (offset, n_tokens, key, fresh)
        if cuda_graphs and self.hybrid:
            log.warning("CUDA graph decode is disabled for hybrid (conv) models: the conv state lives outside the captured graph")
        self._graph_ok = bool(cuda_graphs) and self.device.type == "cuda" and self._kernel_ok and not self.hybrid
        self._graph_max = int(cuda_graph_max_batch)
        self.stats = {"kernel_decode_steps": 0, "graph_replays": 0, "reference_steps": 0}
        self._torch = torch

    # ---- shared transformer body -----------------------------------------------------------------
    def _body(self, ids: Any, positions: Any, blk: Any, off: Any, attn_fn: Any) -> Any:
        torch = self._torch
        ad, d = self.ad, self.dims
        t = ids.shape[0]
        x = ad.embed(ids)
        if isinstance(x, tuple):
            x = x[0]
        if d.scale_embeddings:
            x = x * torch.tensor(d.hidden_size ** 0.5, dtype=x.dtype)
        for parts in ad.attn_names:
            h = parts["ln1"](x)
            if parts["kind"] == "conv":
                x = x + self._conv_mix(parts, h)
                h = parts["ln2"](x)
                m = parts["mlp"](h)
                if isinstance(m, tuple):
                    m = m[0]
                x = x + m
                continue
            li = parts["pi"]  # KV-pool layer index
            attn = parts["attn"]
            q = attn.q_proj(h).reshape(t, d.num_heads, d.head_dim)
            k = attn.k_proj(h).reshape(t, d.num_kv_heads, d.head_dim)
            v = attn.v_proj(h).reshape(t, d.num_kv_heads, d.head_dim)
            if parts["q_norm"] is not None:
                q = parts["q_norm"](q)
                k = parts["k_norm"](k)
            q = self.rope.apply(q, positions)
            k = self.rope.apply(k, positions)
            self.pool.write(li, blk, off, k, v)
            o = attn_fn(li, q, parts)
            x = x + attn.o_proj(o.reshape(t, d.num_heads * d.head_dim))
            h = parts["ln2"](x)
            m = parts["mlp"](h)
            if isinstance(m, tuple):
                m = m[0]
            x = x + m
        return ad.norm(x)

    # ---- lfm2 gated short conv over ragged tokens ---------------------------------------------------
    def _conv_mix(self, parts: dict, h: Any) -> Any:
        """``out_proj(C * conv1d(B * x))`` for ``h [T, H]`` holding several sequences back to back; updates their conv states."""
        torch = self._torch
        import torch.nn.functional as F

        conv, ci = parts["conv"], parts["ci"]
        hid, k1 = self.dims.hidden_size, self.dims.conv_kernel - 1
        w, bias = conv.conv.weight, conv.conv.bias  # [H,1,K]
        bb, cc, xx = conv.in_proj(h).chunk(3, dim=-1)
        u = bb * xx  # [T,H]

        def prev_of(key: Any, fresh: bool) -> Any:
            st = None if fresh else self._conv_state.get(key)
            return u.new_zeros(hid, k1) if st is None else st[ci].to(u.dtype)

        def store(key: Any, new: Any) -> None:
            st = self._conv_state.get(key)
            if st is None:
                st = u.new_zeros(len(self.dims.conv_layers), hid, k1)
                self._conv_state[key] = st
            st[ci] = new.to(st.dtype)

        step = self._conv_step
        if k1 == 0:
            y = F.conv1d(u.t().unsqueeze(0).to(w.dtype), w, bias, groups=hid)[0].t().to(u.dtype)
        elif all(n == 1 for _, n, _, _ in step):  # pure decode: one tap window per sequence, no python-level conv1d
            prev = torch.stack([prev_of(key, fresh) for _, _, key, fresh in step])  # [B,H,K-1]
            win = torch.cat([prev, u.unsqueeze(-1)], dim=-1)  # [B,H,K]
            y = (win.to(w.dtype) * w.squeeze(1)).sum(-1)
            if bias is not None:
                y = y + bias
            y = y.to(u.dtype)
            for bi, (_, _, key, _) in enumerate(step):
                store(key, win[bi, :, 1:])
        else:
            y = torch.empty_like(u)
            for off, n, key, fresh in step:
                z = torch.cat([prev_of(key, fresh), u[off: off + n].t()], dim=-1)  # [H, K-1+n]
                y[off: off + n] = F.conv1d(z.unsqueeze(0).to(w.dtype), w, bias, groups=hid)[0].t().to(u.dtype)
                store(key, z[:, z.shape[-1] - k1:])
        return conv.out_proj(cc * y)

    def drop_state(self, key: Any) -> None:
        self._conv_state.pop(key, None)

    def prune_state(self, live: Any) -> None:
        """Forget conv states of sequences that are no longer alive (finished / aborted requests)."""
        live = set(live)
        for k in [k for k in self._conv_state if k not in live]:
            del self._conv_state[k]

    # ---- attention implementations ---------------------------------------------------------------
    def _ref_attention_fn(self, inputs: list[SeqInput], offsets: list[int]) -> Any:
        torch = self._torch
        import torch.nn.functional as F

        d = self.dims
        g = d.num_heads // d.num_kv_heads
        dev = self.device
        tables = [torch.tensor(s.block_table[: -(-(s.start_pos + len(s.token_ids)) // self.bs)], dtype=torch.long, device=dev)
                  for s in inputs]

        def fn(li: int, q: Any, parts: dict) -> Any:
            out = torch.empty_like(q)
            scale = parts["scale"] or (d.head_dim ** -0.5)
            window = parts["window"]
            for si, s in enumerate(inputs):
                n = len(s.token_ids)
                total = s.start_pos + n
                kc, vc = self.pool.gather(li, tables[si], total)           # [L, Hkv, D]
                qi = q[offsets[si]: offsets[si] + n].transpose(0, 1)         # [Hq, n, D]
                kh = kc.transpose(0, 1)
                vh = vc.transpose(0, 1)
                if g > 1:
                    kh = kh.repeat_interleave(g, dim=0)
                    vh = vh.repeat_interleave(g, dim=0)
                if n == 1 and not window:
                    mask = None
                else:
                    qpos = torch.arange(s.start_pos, total, device=dev).unsqueeze(1)
                    kpos = torch.arange(total, device=dev).unsqueeze(0)
                    mask = kpos <= qpos
                    if window:
                        mask = mask & ((qpos - kpos) < window)
                o = F.scaled_dot_product_attention(qi.unsqueeze(0), kh.unsqueeze(0), vh.unsqueeze(0), attn_mask=mask, scale=scale)
                out[offsets[si]: offsets[si] + n] = o[0].transpose(0, 1)
            return out

        return fn

    def _kernel_attention_fn(self, block_tables: Any, seq_lens: Any) -> Any:
        from forge.kernels import ops_attn

        d = self.dims
        kvd = self.pool.kv_name

        def fn(li: int, q: Any, parts: dict) -> Any:
            scale = parts["scale"] or (d.head_dim ** -0.5)
            return ops_attn.paged_decode_attention(
                q, self.pool.k[li], self.pool.v[li], block_tables, seq_lens,
                softmax_scale=scale, kv_dtype=kvd, k_scale=None, v_scale=None, window=parts["window"],
            ).to(q.dtype)

        return fn

    # ---- forward ----------------------------------------------------------------------------------
    def forward(self, inputs: list[SeqInput], *, want_hidden: bool = False) -> RunnerOutput:
        torch = self._torch
        dev = self.device
        if not inputs:
            return RunnerOutput()
        ids: list[int] = []
        pos: list[int] = []
        slot_blk: list[int] = []
        slot_off: list[int] = []
        offsets: list[int] = []
        bs = self.bs
        for s in inputs:
            offsets.append(len(ids))
            ids.extend(s.token_ids)
            for j in range(len(s.token_ids)):
                p = s.start_pos + j
                pos.append(p)
                slot_blk.append(s.block_table[p // bs])
                slot_off.append(p % bs)
        ids_t = torch.tensor(ids, dtype=torch.long, device=dev)
        pos_t = torch.tensor(pos, dtype=torch.long, device=dev)
        blk_t = torch.tensor(slot_blk, dtype=torch.long, device=dev)
        off_t = torch.tensor(slot_off, dtype=torch.long, device=dev)
        decode_only = all(len(s.token_ids) == 1 for s in inputs)
        if self.hybrid:
            if any(s.state_key is None for s in inputs):
                raise ConfigError("hybrid (conv) models need SeqInput.state_key (one per sequence) to carry the conv state")
            self._conv_step = [(offsets[i], len(s.token_ids), s.state_key, s.start_pos == 0) for i, s in enumerate(inputs)]

        h = None
        if decode_only and self._kernel_ok:
            h = self._try_kernel_decode(inputs, ids_t, pos_t, blk_t, off_t)
        if h is None:
            self.stats["reference_steps"] += 1
            h = self._body(ids_t, pos_t, blk_t, off_t, self._ref_attention_fn(inputs, offsets))

        # select rows needing logits
        rows: list[int] = []
        spans: list[tuple[int, int]] = []
        for si, s in enumerate(inputs):
            n = len(s.token_ids)
            lf = n - 1 if s.logits_from < 0 else min(s.logits_from, n)
            a = offsets[si] + lf
            b = offsets[si] + n
            spans.append((len(rows), len(rows) + (b - a)))
            rows.extend(range(a, b))
        out = RunnerOutput()
        if rows:
            sel = torch.tensor(rows, dtype=torch.long, device=dev)
            hs = h[sel]
            logits = self.ad.logits(hs)
        for a, b in spans:
            if b > a:
                out.logits.append(logits[a:b])
                out.hidden.append(hs[a:b] if want_hidden else None)
            else:
                out.logits.append(None)
                out.hidden.append(None)
        return out

    # ---- kernel / graph decode ---------------------------------------------------------------------
    def _try_kernel_decode(self, inputs: list[SeqInput], ids_t: Any, pos_t: Any, blk_t: Any, off_t: Any) -> Any | None:
        torch = self._torch
        # _body writes conv state in place before a failure can surface; snapshot so the reference fallback does not advance it twice
        keys = [s.state_key for s in inputs if self.dims.conv_layers]
        snap = {k: self._conv_state[k].clone() for k in keys if k in self._conv_state}
        try:
            b = len(inputs)
            max_blocks = max(len(s.block_table) for s in inputs)
            bt = torch.zeros((b, max_blocks), dtype=torch.int32, device=self.device)
            for i, s in enumerate(inputs):
                bt[i, : len(s.block_table)] = torch.tensor(s.block_table, dtype=torch.int32, device=self.device)
            lens = torch.tensor([s.start_pos + 1 for s in inputs], dtype=torch.int32, device=self.device)
            if self._graph_ok and b <= self._graph_max:
                h = self._graph_decode(b, ids_t, pos_t, blk_t, off_t, bt, lens)
                if h is not None:
                    return h
            h = self._body(ids_t, pos_t, blk_t, off_t, self._kernel_attention_fn(bt, lens))
            self.stats["kernel_decode_steps"] += 1
            return h
        except Exception as e:
            if os.environ.get("FORGE_KERNEL_STRICT", "0") == "1":
                raise
            log.warning("paged decode kernel path failed (%s: %s); falling back to the torch reference path", type(e).__name__, e)
            self._kernel_ok = False
            self._graph_ok = False
            for k in keys:
                if k in snap:
                    self._conv_state[k] = snap[k]
                else:
                    self._conv_state.pop(k, None)
            return None

    def _graph_decode(self, b: int, ids_t: Any, pos_t: Any, blk_t: Any, off_t: Any, bt: Any, lens: Any) -> Any | None:
        """Replay a captured CUDA graph for batch bucket >= b (capturing it on first use)."""
        bucket = 1
        while bucket < b:
            bucket *= 2
        try:
            g = self._graphs.get(bucket)
            max_blocks = bt.shape[1]
            if g is None or g["max_blocks"] < max_blocks:
                g = self._capture(bucket, max(max_blocks, -(-self.rope.max_len // self.bs)))
                self._graphs[bucket] = g
            scratch = self.pool.scratch
            g["ids"].zero_()
            g["pos"].zero_()
            g["blk"].fill_(scratch)
            g["off"].zero_()
            g["bt"].fill_(scratch)
            g["lens"].fill_(1)
            g["ids"][:b] = ids_t
            g["pos"][:b] = pos_t
            g["blk"][:b] = blk_t
            g["off"][:b] = off_t
            g["bt"][:b, : bt.shape[1]] = bt
            g["lens"][:b] = lens
            g["graph"].replay()
            self.stats["graph_replays"] += 1
            return g["out"][:b]
        except Exception as e:
            log.warning("CUDA graph decode failed (%s: %s); disabling graphs", type(e).__name__, e)
            self._graph_ok = False
            self._graphs.clear()
            return None

    def _capture(self, bucket: int, max_blocks: int) -> dict[str, Any]:
        torch = self._torch
        dev = self.device
        scratch = self.pool.scratch
        buf = {
            "ids": torch.zeros(bucket, dtype=torch.long, device=dev),
            "pos": torch.zeros(bucket, dtype=torch.long, device=dev),
            "blk": torch.full((bucket,), scratch, dtype=torch.long, device=dev),
            "off": torch.zeros(bucket, dtype=torch.long, device=dev),
            "bt": torch.full((bucket, max_blocks), scratch, dtype=torch.int32, device=dev),
            "lens": torch.ones(bucket, dtype=torch.int32, device=dev),
            "max_blocks": max_blocks,
        }

        def run() -> Any:
            return self._body(buf["ids"], buf["pos"], buf["blk"], buf["off"], self._kernel_attention_fn(buf["bt"], buf["lens"]))

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            buf["out"] = run()
        buf["graph"] = graph
        return buf

    # ---- utilities -------------------------------------------------------------------------------
    def copy_blocks(self, copies: list[tuple[int, int]]) -> None:
        for src, dst in copies:
            self.pool.copy_block(src, dst)


# --------------------------------------------------------------------------------------------------
# memory planning
# --------------------------------------------------------------------------------------------------
def plan_num_blocks(*, dims: ModelDims, block_size: int, kv_name: str, device: Any, gpu_mem_util: float,
                    activation_reserve_gb: float, max_num_seqs: int, max_model_len: int,
                    num_gpu_blocks: int | None = None, cpu_cap_blocks: int = 2048) -> int:
    """How many KV blocks fit. On CUDA: ``total * util - allocated - activation reserve``; on CPU: enough for the
    configured concurrency but capped (so a laptop does not allocate gigabytes)."""
    import torch

    if num_gpu_blocks:
        return int(num_gpu_blocks)
    per = kv_bytes_per_block(dims.num_kv_layers, dims.num_kv_heads, dims.head_dim, block_size, kv_name)
    want = max_num_seqs * math.ceil(max_model_len / block_size)
    if device.type != "cuda":
        return max(8, min(want, cpu_cap_blocks))
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    budget = total * gpu_mem_util - allocated - activation_reserve_gb * 1e9
    budget = min(budget, free - activation_reserve_gb * 1e9)
    n = int(budget // per)
    if n < 8:
        raise ConfigError(f"not enough GPU memory for a KV cache (budget {budget / 1e9:.2f} GB, block {per / 1e6:.2f} MB); "
                          "lower max_model_len/weights or raise gpu_mem_util")
    return n
