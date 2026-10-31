"""Model configuration for the Forge decoder-only transformer family.

``ModelConfig`` is the single description of an architecture (dense or MoE).  It converts to / from Hugging Face
``config.json`` dictionaries for the supported families (llama, mistral, qwen2, qwen3, mixtral, qwen2_moe,
qwen3_moe, gemma) and exposes analytic accounting (parameters, FLOPs, KV bytes per token) that the planner,
serving capacity model and UI use without instantiating the model.

Every field carries a ``Field(description=...)`` because the UI renders forms from the JSON schema.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.core.errors import ConfigError
from forge.core.jsonschema import ui

SUPPORTED_ARCHS: tuple[str, ...] = (
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "mixtral",
    "qwen2_moe",
    "qwen3_moe",
    "gemma",
)

_HF_ARCHITECTURE = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "gemma": "GemmaForCausalLM",
}

_ACT_ALIASES = {
    "silu": "silu",
    "swish": "silu",
    "gelu": "gelu",
    "gelu_pytorch_tanh": "gelu_tanh",
    "gelu_tanh": "gelu_tanh",
    "gelu_new": "gelu_tanh",
    "relu": "relu",
}


class MoEConfig(BaseModel):
    """Mixture-of-experts block configuration (router, routed experts, optional shared expert)."""

    model_config = ConfigDict(extra="forbid")

    num_experts: int = Field(8, ge=2, description="Number of routed experts per MoE layer.", json_schema_extra=ui(group="MoE"))
    top_k: int = Field(2, ge=1, description="Experts activated per token.", json_schema_extra=ui(group="MoE"))
    moe_intermediate_size: int = Field(
        1408, ge=1, description="FFN hidden size of each routed expert.", json_schema_extra=ui(group="MoE")
    )
    num_shared_experts: int = Field(
        0,
        ge=0,
        description="Always-active shared experts (fused into one dense MLP of size num_shared_experts * moe_intermediate_size "
        "unless shared_expert_intermediate_size is set).",
        json_schema_extra=ui(group="MoE"),
    )
    shared_expert_intermediate_size: int | None = Field(
        None,
        ge=1,
        description="Explicit hidden size of the single shared expert MLP (qwen2_moe style). Overrides num_shared_experts sizing.",
        json_schema_extra=ui(group="MoE", advanced=True),
    )
    shared_expert_gate: bool = Field(
        False,
        description="Multiply shared-expert output by sigmoid(Linear(h)) (qwen2_moe).",
        json_schema_extra=ui(group="MoE", advanced=True),
    )
    norm_topk_prob: bool = Field(
        True,
        description="Renormalise the selected top-k routing probabilities to sum to 1 (mixtral/qwen3_moe True, qwen2_moe default False).",
        json_schema_extra=ui(group="MoE"),
    )
    router_aux_loss_coef: float = Field(
        0.001,
        ge=0.0,
        description="Weight of the Switch-style load-balancing auxiliary loss added to the LM loss while training.",
        json_schema_extra=ui(group="MoE"),
    )
    decoder_sparse_step: int = Field(
        1, ge=1, description="Every n-th layer is an MoE layer (1 = all layers).", json_schema_extra=ui(group="MoE", advanced=True)
    )
    mlp_only_layers: list[int] = Field(
        default_factory=list,
        description="Layer indices that use a plain dense MLP instead of MoE.",
        json_schema_extra=ui(group="MoE", advanced=True),
    )
    use_grouped_gemm: bool = Field(
        True,
        description="Use forge.kernels.ops_moe grouped GEMM for inference on CUDA when all experts are dense (falls back to a loop).",
        json_schema_extra=ui(group="MoE", advanced=True),
    )

    @model_validator(mode="after")
    def _check(self) -> "MoEConfig":
        if self.top_k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")
        return self

    @property
    def shared_size(self) -> int:
        """Hidden size of the fused shared-expert MLP (0 = none)."""
        if self.shared_expert_intermediate_size:
            return int(self.shared_expert_intermediate_size)
        return int(self.num_shared_experts * self.moe_intermediate_size)


def _normalize_rope(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if not d:
        return None
    d = dict(d)
    rt = d.pop("rope_type", None) or d.pop("type", None)
    d.pop("type", None)
    d.pop("rope_theta", None)
    if rt in (None, "default"):
        return None
    d["rope_type"] = rt
    return d


class ModelConfig(BaseModel):
    """Architecture description of a Forge decoder-only transformer (dense or MoE)."""

    model_config = ConfigDict(extra="forbid")

    arch: Literal["llama", "mistral", "qwen2", "qwen3", "mixtral", "qwen2_moe", "qwen3_moe", "gemma"] = Field(
        "llama", description="Architecture family (determines HF config/weight-name conventions).", json_schema_extra=ui(group="Architecture")
    )
    vocab_size: int = Field(32000, ge=1, description="Vocabulary size.", json_schema_extra=ui(group="Architecture"))
    hidden_size: int = Field(4096, ge=1, description="Residual stream width.", json_schema_extra=ui(group="Architecture"))
    intermediate_size: int = Field(11008, ge=1, description="Dense SwiGLU MLP hidden size.", json_schema_extra=ui(group="Architecture"))
    num_layers: int = Field(32, ge=1, description="Number of transformer blocks.", json_schema_extra=ui(group="Architecture"))
    num_heads: int = Field(32, ge=1, description="Number of query heads.", json_schema_extra=ui(group="Attention"))
    num_kv_heads: int | None = Field(
        None, ge=1, description="Number of key/value heads (GQA). None = num_heads (MHA).", json_schema_extra=ui(group="Attention")
    )
    head_dim: int | None = Field(
        None, ge=2, description="Per-head dimension. None = hidden_size // num_heads.", json_schema_extra=ui(group="Attention")
    )
    rope_theta: float = Field(10000.0, gt=0, description="RoPE base frequency.", json_schema_extra=ui(group="Attention"))
    rope_scaling: dict[str, Any] | None = Field(
        None,
        description="RoPE scaling: {rope_type: linear|llama3|yarn, factor, ...}. None = plain RoPE.",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    rms_norm_eps: float = Field(1e-5, gt=0, description="RMSNorm epsilon.", json_schema_extra=ui(group="Architecture", advanced=True))
    tie_word_embeddings: bool = Field(
        False, description="Share the input embedding matrix with the LM head.", json_schema_extra=ui(group="Architecture")
    )
    max_position_embeddings: int = Field(
        4096, ge=1, description="Maximum (original) context length of the checkpoint.", json_schema_extra=ui(group="Attention", unit="tokens")
    )
    attention_bias: bool = Field(
        False, description="Bias on the q/k/v projections (and o-proj for llama-style archs).", json_schema_extra=ui(group="Attention", advanced=True)
    )
    attention_out_bias: bool | None = Field(
        None,
        description="Bias on o_proj. None = follow attention_bias, except qwen2/qwen2_moe where it is False.",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    qk_norm: bool = Field(
        False, description="RMSNorm on per-head q and k before RoPE (qwen3).", json_schema_extra=ui(group="Attention", advanced=True)
    )
    sliding_window: int | None = Field(
        None, ge=1, description="Sliding-window attention size in tokens (None = full causal).", json_schema_extra=ui(group="Attention", unit="tokens")
    )
    moe: MoEConfig | None = Field(None, description="MoE settings; None = dense model.", json_schema_extra=ui(group="MoE"))
    hidden_act: str = Field("silu", description="MLP activation: silu | gelu | gelu_tanh | relu.", json_schema_extra=ui(group="Architecture", advanced=True))
    initializer_range: float = Field(
        0.02, gt=0, description="Std of the normal init for random initialisation.", json_schema_extra=ui(group="Architecture", advanced=True)
    )
    norm_unit_offset: bool = Field(
        False, description="RMSNorm uses (1 + weight) (gemma).", json_schema_extra=ui(group="Architecture", advanced=True)
    )
    scale_embeddings: bool = Field(
        False, description="Multiply embeddings by sqrt(hidden_size) (gemma).", json_schema_extra=ui(group="Architecture", advanced=True)
    )

    # ------------------------------------------------------------------ validation
    @model_validator(mode="after")
    def _defaults(self) -> "ModelConfig":
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            if self.hidden_size % self.num_heads:
                raise ValueError("hidden_size must be divisible by num_heads when head_dim is not given")
            self.head_dim = self.hidden_size // self.num_heads
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even (RoPE)")
        if self.hidden_act not in ("silu", "gelu", "gelu_tanh", "relu"):
            raise ValueError(f"unsupported hidden_act {self.hidden_act!r}")
        if self.arch in ("mixtral", "qwen2_moe", "qwen3_moe") and self.moe is None:
            raise ValueError(f"arch {self.arch} requires a moe section")
        if self.rope_scaling is not None:
            self.rope_scaling = _normalize_rope(self.rope_scaling)
        return self

    # ------------------------------------------------------------------ derived
    @property
    def kv_heads(self) -> int:
        return int(self.num_kv_heads or self.num_heads)

    @property
    def q_dim(self) -> int:
        return int(self.num_heads * self.head_dim)

    @property
    def kv_dim(self) -> int:
        return int(self.kv_heads * self.head_dim)

    @property
    def o_proj_bias(self) -> bool:
        if self.attention_out_bias is not None:
            return bool(self.attention_out_bias)
        if self.arch in ("qwen2", "qwen2_moe"):
            return False
        return bool(self.attention_bias)

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    def is_moe_layer(self, i: int) -> bool:
        m = self.moe
        if m is None:
            return False
        if i in set(m.mlp_only_layers):
            return False
        return (i + 1) % m.decoder_sparse_step == 0

    @property
    def num_moe_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if self.is_moe_layer(i))

    # ------------------------------------------------------------------ accounting
    def kv_bytes_per_token(self, kv_bits: float = 16, *, group_size: int | None = None, scale_bits: int = 16) -> int:
        """Bytes of K+V cache per token over all layers.

        ``2 * layers * kv_heads * head_dim * kv_bits / 8`` (Plan: Llama-3-8B 128 KiB, 70B 320 KiB, Qwen3-8B 144 KiB at 16 bit).
        With ``group_size`` set, one ``scale_bits`` scale per group is added (quantised KV).
        """
        elems = 2 * self.num_layers * self.kv_dim
        total = elems * kv_bits / 8.0
        if group_size:
            total += (elems / group_size) * scale_bits / 8.0
        return int(math.ceil(total))

    def _attn_params_per_layer(self) -> int:
        h = self.hidden_size
        n = h * self.q_dim + 2 * h * self.kv_dim + self.q_dim * h
        if self.attention_bias:
            n += self.q_dim + 2 * self.kv_dim
        if self.o_proj_bias:
            n += h
        if self.qk_norm:
            n += 2 * self.head_dim
        return n

    def _mlp_params(self, i: int, active: bool = False) -> int:
        h = self.hidden_size
        if not self.is_moe_layer(i):
            return 3 * h * self.intermediate_size
        m = self.moe
        assert m is not None
        n_exp = m.top_k if active else m.num_experts
        n = m.num_experts * h + n_exp * 3 * h * m.moe_intermediate_size
        if m.shared_size:
            n += 3 * h * m.shared_size
            if m.shared_expert_gate:
                n += h
        return n

    def param_count(self, *, active: bool = False, include_embeddings: bool = True) -> int:
        """Analytic parameter count (``active=True`` counts only top-k experts per token)."""
        h = self.hidden_size
        n = 0
        emb = self.vocab_size * h
        if include_embeddings:
            n += emb
            if not self.tie_word_embeddings:
                n += emb
        for i in range(self.num_layers):
            n += self._attn_params_per_layer() + self._mlp_params(i, active) + 2 * h
        n += h  # final norm
        return int(n)

    def active_param_count(self, include_embeddings: bool = True) -> int:
        return self.param_count(active=True, include_embeddings=include_embeddings)

    def flops_per_token(self, seq_len: int = 0, mode: Literal["train", "prefill", "decode"] = "prefill") -> float:
        """Approximate FLOPs per token.

        Matmul FLOPs = 2 * active non-embedding params + 2 * vocab * hidden (LM head) + attention
        ``4 * layers * q_dim * ctx`` where ``ctx`` is ``seq_len/2`` for train/prefill (causal average) and ``seq_len`` for decode.
        ``mode='train'`` multiplies by 3 (forward + backward).
        """
        non_emb = self.active_param_count(include_embeddings=False)
        f = 2.0 * non_emb + 2.0 * self.vocab_size * self.hidden_size
        ctx = seq_len if mode == "decode" else seq_len / 2.0
        f += 4.0 * self.num_layers * self.q_dim * ctx
        return f * (3.0 if mode == "train" else 1.0)

    def weight_bytes(self, bits: float = 16.0) -> int:
        return int(math.ceil(self.param_count() * bits / 8.0))

    # ------------------------------------------------------------------ HF conversion
    @classmethod
    def from_hf_config(cls, d: dict[str, Any]) -> "ModelConfig":
        """Build from a HF ``config.json`` dict (llama/mistral/qwen2/qwen3/mixtral/qwen2_moe/qwen3_moe/gemma)."""
        d = dict(d)
        if "text_config" in d and isinstance(d["text_config"], dict) and "hidden_size" not in d:
            d = {**d["text_config"]}
        mt = d.get("model_type")
        if mt is None:
            archs = d.get("architectures") or []
            for a, name in _HF_ARCHITECTURE.items():
                if name in archs:
                    mt = a
        if mt not in SUPPORTED_ARCHS:
            raise ConfigError(f"unsupported model_type {mt!r}; supported: {', '.join(SUPPORTED_ARCHS)}")
        heads = int(d["num_attention_heads"])
        params = d.get("rope_parameters") or {}
        theta = d.get("rope_theta", params.get("rope_theta", 10000.0))
        scaling = _normalize_rope(d.get("rope_scaling") or params)
        kw: dict[str, Any] = dict(
            arch=mt,
            vocab_size=int(d["vocab_size"]),
            hidden_size=int(d["hidden_size"]),
            intermediate_size=int(d["intermediate_size"]),
            num_layers=int(d["num_hidden_layers"]),
            num_heads=heads,
            num_kv_heads=int(d.get("num_key_value_heads") or heads),
            head_dim=int(d["head_dim"]) if d.get("head_dim") else None,
            rope_theta=float(theta),
            rope_scaling=scaling,
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", mt == "gemma")),  # HF GemmaConfig defaults to tied; its checkpoints ship no lm_head
            max_position_embeddings=int(d.get("max_position_embeddings", 4096)),
        )
        act = d.get("hidden_activation") or d.get("hidden_act") or "silu"
        if mt == "gemma":
            act = "gelu_pytorch_tanh" if act in ("gelu", "gelu_pytorch_tanh", None) else act
            kw.update(norm_unit_offset=True, scale_embeddings=True)
        kw["hidden_act"] = _ACT_ALIASES.get(act, act)
        sw = d.get("sliding_window")
        if mt in ("qwen2", "qwen2_moe", "qwen3", "qwen3_moe") and not d.get("use_sliding_window", False):
            sw = None
        kw["sliding_window"] = int(sw) if sw else None
        if mt in ("qwen2", "qwen2_moe"):
            kw["attention_bias"] = True
            kw["attention_out_bias"] = False
        else:
            kw["attention_bias"] = bool(d.get("attention_bias", False))
        if mt in ("qwen3", "qwen3_moe"):
            kw["qk_norm"] = True
        if mt == "mixtral":
            kw["moe"] = MoEConfig(
                num_experts=int(d["num_local_experts"]),
                top_k=int(d["num_experts_per_tok"]),
                moe_intermediate_size=int(d["intermediate_size"]),
                norm_topk_prob=True,
                router_aux_loss_coef=float(d.get("router_aux_loss_coef", 0.001)),
            )
        elif mt in ("qwen2_moe", "qwen3_moe"):
            shared = d.get("shared_expert_intermediate_size") or 0
            kw["moe"] = MoEConfig(
                num_experts=int(d.get("num_experts") or d["num_local_experts"]),
                top_k=int(d["num_experts_per_tok"]),
                moe_intermediate_size=int(d["moe_intermediate_size"]),
                shared_expert_intermediate_size=int(shared) if shared else None,
                shared_expert_gate=bool(shared) and mt == "qwen2_moe",
                norm_topk_prob=bool(d.get("norm_topk_prob", False)),
                router_aux_loss_coef=float(d.get("router_aux_loss_coef", 0.001)),
                decoder_sparse_step=int(d.get("decoder_sparse_step", 1) or 1),
                mlp_only_layers=[int(x) for x in (d.get("mlp_only_layers") or [])],
            )
        return cls(**kw)

    def to_hf_config(self) -> dict[str, Any]:
        """HF-compatible ``config.json`` dictionary (loadable by ``transformers`` for the same arch)."""
        a = self.arch
        d: dict[str, Any] = {
            "model_type": a,
            "architectures": [_HF_ARCHITECTURE[a]],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
        }
        if self.rope_scaling:
            d["rope_scaling"] = dict(self.rope_scaling)
        act = {"silu": "silu", "gelu": "gelu", "gelu_tanh": "gelu_pytorch_tanh", "relu": "relu"}[self.hidden_act]
        d["hidden_act"] = act
        if a == "gemma":
            d["hidden_activation"] = act
        if a in ("llama", "qwen3", "qwen3_moe"):
            d["attention_bias"] = bool(self.attention_bias)
        if a in ("mistral", "mixtral"):
            d["sliding_window"] = self.sliding_window
        elif a in ("qwen2", "qwen2_moe", "qwen3", "qwen3_moe"):
            d["use_sliding_window"] = self.sliding_window is not None
            d["sliding_window"] = self.sliding_window
        elif self.sliding_window:
            d["sliding_window"] = self.sliding_window
        m = self.moe
        if m is not None:
            d["router_aux_loss_coef"] = m.router_aux_loss_coef
            d["num_experts_per_tok"] = m.top_k
            if a == "mixtral":
                d["num_local_experts"] = m.num_experts
                d["intermediate_size"] = m.moe_intermediate_size
            else:
                d["num_experts"] = m.num_experts
                d["moe_intermediate_size"] = m.moe_intermediate_size
                d["norm_topk_prob"] = m.norm_topk_prob
                d["decoder_sparse_step"] = m.decoder_sparse_step
                d["mlp_only_layers"] = list(m.mlp_only_layers)
                if a == "qwen2_moe":
                    d["shared_expert_intermediate_size"] = m.shared_size
        return d
