"""Model configuration for the Forge decoder-only transformer family.

``ModelConfig`` is the single description of an architecture (dense or MoE).  It converts to / from Hugging Face
``config.json`` dictionaries for the supported families (llama, mistral, qwen2, qwen3, mixtral, qwen2_moe,
qwen3_moe, gemma, gemma4) and exposes analytic accounting (parameters, FLOPs, KV bytes per token) that the planner,
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
    "gemma4",
    "lfm2",
    "lfm2_moe",
)

# HF ``model_type`` values (top-level multimodal wrapper and its text decoder) that map to arch "gemma4"
_GEMMA4_TYPES = ("gemma4", "gemma4_text")
_GEMMA4_ARCHITECTURES = ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
# the 12B "unified" variants are not verified against a real checkpoint, so they are refused instead of silently aliased
_GEMMA4_UNVERIFIED_TYPES = ("gemma4_unified", "gemma4_unified_text")
_GEMMA4_UNVERIFIED_ARCHITECTURES = ("Gemma4UnifiedForConditionalGeneration",)

_HF_ARCHITECTURE = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma4": "Gemma4ForCausalLM",
    "lfm2": "Lfm2ForCausalLM",
    "lfm2_moe": "Lfm2MoeForCausalLM",
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
    router_type: Literal["softmax", "sigmoid"] = Field(
        "softmax",
        description="Router scoring: softmax over experts (default) or independent sigmoid scores (lfm2_moe).",
        json_schema_extra=ui(group="MoE", advanced=True),
    )
    use_expert_bias: bool = Field(
        False,
        description="Sigmoid router: a persistent fp32 `expert_bias` buffer is added to the scores for expert SELECTION only (lfm2_moe).",
        json_schema_extra=ui(group="MoE", advanced=True),
    )
    routed_scaling_factor: float = Field(
        1.0, gt=0, description="Sigmoid router: multiplier applied to the (renormalised) top-k weights.", json_schema_extra=ui(group="MoE", advanced=True)
    )

    @model_validator(mode="after")
    def _check(self) -> "MoEConfig":
        if self.top_k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")
        if self.use_expert_bias and self.router_type != "sigmoid":
            raise ValueError("use_expert_bias requires router_type='sigmoid'")
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

    arch: Literal["llama", "mistral", "qwen2", "qwen3", "mixtral", "qwen2_moe", "qwen3_moe", "gemma", "gemma4", "lfm2", "lfm2_moe"] = Field(
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

    # ---- gemma4 text decoder (all defaults leave the other archs untouched)
    layer_types: list[Literal["sliding_attention", "full_attention", "conv"]] | None = Field(
        None,
        description="Per-layer mixer kind. gemma4: sliding_attention uses sliding_window and rope_local_theta, full_attention uses "
        "global_head_dim and the rope_theta/rope_scaling table. lfm2: full_attention or conv (gated short convolution, no KV cache, "
        "only a small recurrent state). None = every layer identical.",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    conv_kernel: int = Field(
        3, ge=1, description="Short-convolution kernel size of `conv` layers (lfm2 conv_L_cache).", json_schema_extra=ui(group="Architecture", advanced=True)
    )
    conv_bias: bool = Field(
        False, description="Bias on the in_proj / depthwise conv / out_proj of `conv` layers (lfm2 conv_bias).", json_schema_extra=ui(group="Architecture", advanced=True)
    )
    global_head_dim: int | None = Field(
        None, ge=2, description="Head dimension of full_attention layers (gemma4). None = head_dim.", json_schema_extra=ui(group="Attention", advanced=True)
    )
    num_global_kv_heads: int | None = Field(
        None,
        ge=1,
        description="KV heads of full_attention layers; only used together with attention_k_eq_v (gemma4 12B/31B). None = num_kv_heads.",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    attention_k_eq_v: bool = Field(
        False,
        description="full_attention layers have no v_proj: V is the (pre-norm) output of k_proj (gemma4 12B/31B).",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    rope_local_theta: float | None = Field(
        None,
        gt=0,
        description="RoPE base of sliding_attention layers (gemma4: 10000); rope_theta is then the full_attention base. None = rope_theta.",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    attn_scale: float | None = Field(
        None,
        gt=0,
        description="Softmax scale of attention scores. None = head_dim**-0.5 (gemma4 uses 1.0: the scale lives in the q/k norms).",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    sandwich_norm: bool = Field(
        False,
        description="gemma4 block: post_attention_layernorm normalises the attention OUTPUT and the MLP is wrapped in "
        "pre_feedforward_layernorm / post_feedforward_layernorm.",
        json_schema_extra=ui(group="Architecture", advanced=True),
    )
    v_norm: bool = Field(
        False, description="Scale-less RMSNorm on the per-head values before attention (gemma4).", json_schema_extra=ui(group="Attention", advanced=True)
    )
    layer_scalar: bool = Field(
        False,
        description="Every block output is multiplied by a learned [1] buffer `layer_scalar` (gemma4).",
        json_schema_extra=ui(group="Architecture", advanced=True),
    )
    final_logit_softcapping: float | None = Field(
        None,
        gt=0,
        description="Logits are soft-capped as cap * tanh(logits / cap), also in the loss path (gemma4: 30).",
        json_schema_extra=ui(group="Architecture", advanced=True),
    )
    per_layer_input_size: int = Field(
        0,
        ge=0,
        description="Per-layer-embedding (PLE) width per layer (gemma4 E2B/E4B: 256). 0 = no PLE.",
        json_schema_extra=ui(group="Architecture", advanced=True),
    )
    vocab_size_per_layer_input: int = Field(
        0, ge=0, description="Rows of the PLE table. 0 = vocab_size.", json_schema_extra=ui(group="Architecture", advanced=True)
    )
    num_kv_shared_layers: int = Field(
        0,
        ge=0,
        description="The last N layers own no K/V projections and reuse the K/V of the last unshared layer of the same attention kind (gemma4 E2B/E4B).",
        json_schema_extra=ui(group="Attention", advanced=True),
    )
    double_wide_mlp_shared: bool = Field(
        False,
        description="KV-shared layers use a 2x wider MLP (gemma4 E2B use_double_wide_mlp).",
        json_schema_extra=ui(group="Architecture", advanced=True),
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
        if self.arch in ("mixtral", "qwen2_moe", "qwen3_moe", "lfm2_moe") and self.moe is None:
            raise ValueError(f"arch {self.arch} requires a moe section")
        if self.layer_types is not None and "conv" in self.layer_types and self.arch not in ("lfm2", "lfm2_moe"):
            raise ValueError("conv layers are only supported by the lfm2 / lfm2_moe archs")
        if self.rope_scaling is not None:
            self.rope_scaling = _normalize_rope(self.rope_scaling)
        if self.layer_types is not None and len(self.layer_types) != self.num_layers:
            raise ValueError(f"layer_types has {len(self.layer_types)} entries but num_layers is {self.num_layers}")
        if self.global_head_dim is not None and self.global_head_dim % 2:
            raise ValueError("global_head_dim must be even (RoPE)")
        if self.num_kv_shared_layers >= self.num_layers:
            raise ValueError("num_kv_shared_layers must be < num_layers")
        if self.num_kv_shared_layers and self.layer_types is not None:
            own = self.layer_types[: self.num_layers - self.num_kv_shared_layers]
            for t in set(self.layer_types[self.num_layers - self.num_kv_shared_layers :]):
                if t not in own:
                    raise ValueError(f"KV-shared {t} layers have no unshared {t} layer to read from")
        if self.attention_k_eq_v and self.num_global_kv_heads and self.num_heads % self.num_global_kv_heads:
            raise ValueError("num_heads must be a multiple of num_global_kv_heads")
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

    # ---- per-layer attention geometry (uniform for every arch except gemma4-style ``layer_types`` configs)
    def is_conv_layer(self, i: int) -> bool:
        """lfm2 gated short-convolution block (no attention, no KV cache; carries a ``[hidden, conv_kernel-1]`` recurrent state)."""
        return self.layer_types is not None and self.layer_types[i] == "conv"

    @property
    def is_hybrid(self) -> bool:
        return self.layer_types is not None and "conv" in self.layer_types

    @property
    def conv_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.is_conv_layer(i)]

    @property
    def attn_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_layers) if not self.is_conv_layer(i)]

    def layer_owns_kv(self, i: int) -> bool:
        """True when layer ``i`` allocates / writes KV cache (attention layer that is not KV-shared)."""
        return not self.is_conv_layer(i) and not self.is_kv_shared_layer(i)

    def recurrent_state_bytes_per_seq(self, dtype_bytes: float = 2) -> int:
        """Bytes of per-sequence conv state (independent of context length; 0 for pure-attention models)."""
        return int(len(self.conv_layer_ids) * self.hidden_size * (self.conv_kernel - 1) * dtype_bytes)

    def layer_is_sliding(self, i: int) -> bool:
        return self.layer_types is not None and self.layer_types[i] == "sliding_attention"

    def layer_is_full(self, i: int) -> bool:
        return self.layer_types is not None and self.layer_types[i] == "full_attention"

    def layer_window(self, i: int) -> int | None:
        """Sliding-window size of layer ``i`` (None = full causal)."""
        if self.layer_types is None:
            return self.sliding_window
        return self.sliding_window if self.layer_is_sliding(i) else None

    def layer_head_dim(self, i: int) -> int:
        if self.layer_is_full(i) and self.global_head_dim:
            return int(self.global_head_dim)
        return int(self.head_dim)  # type: ignore[arg-type]

    def layer_kv_heads(self, i: int) -> int:
        if self.layer_is_full(i) and self.attention_k_eq_v and self.num_global_kv_heads:
            return int(self.num_global_kv_heads)
        return self.kv_heads

    def layer_q_dim(self, i: int) -> int:
        return self.num_heads * self.layer_head_dim(i)

    def layer_kv_dim(self, i: int) -> int:
        return self.layer_kv_heads(i) * self.layer_head_dim(i)

    def layer_has_v_proj(self, i: int) -> bool:
        return not (self.attention_k_eq_v and self.layer_is_full(i))

    def layer_intermediate(self, i: int) -> int:
        return self.intermediate_size * (2 if (self.double_wide_mlp_shared and self.is_kv_shared_layer(i)) else 1)

    def is_kv_shared_layer(self, i: int) -> bool:
        return self.num_kv_shared_layers > 0 and i >= self.num_layers - self.num_kv_shared_layers

    def kv_source_layer(self, i: int) -> int | None:
        """Index of the layer whose K/V layer ``i`` reads (None = layer owns its K/V)."""
        if not self.is_kv_shared_layer(i):
            return None
        for j in range(self.num_layers - self.num_kv_shared_layers - 1, -1, -1):
            if self.layer_types is None or self.layer_types[j] == self.layer_types[i]:
                return j
        raise ValueError(f"layer {i} has no KV source")

    @property
    def norm_fp32_scale(self) -> bool:
        """RMSNorm weights are multiplied in fp32 before the final cast (HF Gemma4RMSNorm)."""
        return self.arch == "gemma4"

    @property
    def kv_layers(self) -> list[int]:
        """Layers that own (allocate / write) a KV cache."""
        return [i for i in range(self.num_layers) if self.layer_owns_kv(i)]

    @property
    def ple_vocab(self) -> int:
        return int(self.vocab_size_per_layer_input or self.vocab_size)

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
        elems = 2 * sum(self.layer_kv_dim(i) for i in self.kv_layers)  # KV-shared layers own no cache; gemma4 layers differ in width
        total = elems * kv_bits / 8.0
        if group_size:
            total += (elems / group_size) * scale_bits / 8.0
        return int(math.ceil(total))

    def _attn_params_per_layer(self, i: int = 0) -> int:
        h = self.hidden_size
        q, kv, hd = self.layer_q_dim(i), self.layer_kv_dim(i), self.layer_head_dim(i)
        if self.is_kv_shared_layer(i):  # q_proj / o_proj / q_norm only
            return 2 * h * q + (hd if self.qk_norm else 0)
        n_kv = 2 if self.layer_has_v_proj(i) else 1
        n = h * q + n_kv * h * kv + q * h
        if self.attention_bias:
            n += q + n_kv * kv
        if self.o_proj_bias:
            n += h
        if self.qk_norm:
            n += 2 * hd
        return n

    def _conv_params_per_layer(self) -> int:
        h = self.hidden_size
        n = 3 * h * h + h * self.conv_kernel + h * h  # in_proj, depthwise conv, out_proj
        if self.conv_bias:
            n += 3 * h + h + h
        return n

    def _mixer_params(self, i: int) -> int:
        return self._conv_params_per_layer() if self.is_conv_layer(i) else self._attn_params_per_layer(i)

    def _block_extra_params(self, i: int) -> int:
        """Norms / buffers / PLE tensors of block ``i`` beyond attention + MLP + the 2 base norms."""
        h, p = self.hidden_size, self.per_layer_input_size
        n = 0
        if self.sandwich_norm:
            n += 2 * h
        if self.layer_scalar:
            n += 1
        if p:
            n += 2 * h * p + h
        return n

    def _ple_params(self) -> int:
        """Model-level PLE tensors excluding the embedding table: projection h -> L*P and the P-wide norm."""
        p = self.per_layer_input_size
        return self.hidden_size * self.num_layers * p + p if p else 0

    def _mlp_params(self, i: int, active: bool = False) -> int:
        h = self.hidden_size
        if not self.is_moe_layer(i):
            return 3 * h * self.layer_intermediate(i)
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
            n += self.ple_table_params
        for i in range(self.num_layers):
            n += self._mixer_params(i) + self._mlp_params(i, active) + 2 * h + self._block_extra_params(i)
        n += h + self._ple_params()  # final norm (+ PLE projection)
        return int(n)

    @property
    def ple_table_params(self) -> int:
        """Parameters of the per-layer embedding table ``[ple_vocab, num_layers * per_layer_input_size]`` (0 without PLE)."""
        return self.ple_vocab * self.num_layers * self.per_layer_input_size if self.per_layer_input_size else 0

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
        if self.layer_types is None:
            f += 4.0 * self.num_layers * self.q_dim * ctx
        else:  # sliding layers only see `window` keys
            for i in range(self.num_layers):
                if self.is_conv_layer(i):  # depthwise conv taps (the projections are already in the matmul params)
                    f += 2.0 * self.hidden_size * self.conv_kernel
                    continue
                w = self.layer_window(i)
                f += 4.0 * self.layer_q_dim(i) * (min(ctx, float(w)) if w else ctx)
        return f * (3.0 if mode == "train" else 1.0)

    def weight_bytes(self, bits: float = 16.0) -> int:
        return int(math.ceil(self.param_count() * bits / 8.0))

    # ------------------------------------------------------------------ HF conversion
    @classmethod
    def from_hf_config(cls, d: dict[str, Any]) -> "ModelConfig":
        """Build from a HF ``config.json`` dict (llama/mistral/qwen2/qwen3/mixtral/qwen2_moe/qwen3_moe/gemma)."""
        d = dict(d)
        _tc = d.get("text_config") if isinstance(d.get("text_config"), dict) else {}
        if (d.get("model_type") in _GEMMA4_UNVERIFIED_TYPES or _tc.get("model_type") in _GEMMA4_UNVERIFIED_TYPES
                or any(a in _GEMMA4_UNVERIFIED_ARCHITECTURES for a in (d.get("architectures") or []))):
            raise ConfigError("gemma4_unified (12B 'unified' variant) is unverified against a real checkpoint and is not supported yet; "
                              "supported Gemma 4 model_types: " + ", ".join(_GEMMA4_TYPES))
        if "text_config" in d and isinstance(d["text_config"], dict) and "hidden_size" not in d:
            top = d.get("model_type")
            d = {**d["text_config"]}
            if top in _GEMMA4_TYPES and not d.get("model_type"):
                d["model_type"] = top
        mt = d.get("model_type")
        if mt is None:
            archs = d.get("architectures") or []
            for a, name in _HF_ARCHITECTURE.items():
                if name in archs:
                    mt = a
            if any(a in _GEMMA4_ARCHITECTURES for a in archs):
                mt = "gemma4"
        if mt in _GEMMA4_TYPES:
            return cls._from_gemma4_config(d)
        if mt in ("lfm2", "lfm2_moe"):
            return cls._from_lfm2_config(d, mt)
        if mt not in SUPPORTED_ARCHS:
            raise ConfigError(f"unsupported model_type {mt!r}; supported: {', '.join(SUPPORTED_ARCHS)}, gemma4_text")
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

    @classmethod
    def _from_gemma4_config(cls, d: dict[str, Any]) -> "ModelConfig":
        """Gemma 4 text decoder from the ``text_config`` of ``Gemma4ForConditionalGeneration`` or a plain ``gemma4_text`` config."""
        if d.get("enable_moe_block"):
            raise ConfigError("gemma4 MoE blocks (enable_moe_block, e.g. 26B-A4B) are not supported yet; use E2B / E4B / 12B / 31B")
        if d.get("use_bidirectional_attention") == "all":
            raise ConfigError("gemma4 use_bidirectional_attention='all' is not supported (text decoder is causal)")
        n = int(d["num_hidden_layers"])
        heads = int(d["num_attention_heads"])
        lt = list(d["layer_types"]) if d.get("layer_types") else [
            "sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(n)
        ]
        if lt[-1] != "full_attention":  # HF forces the last layer to be global
            lt[-1] = "full_attention"
        k_eq_v = bool(d.get("attention_k_eq_v", False))
        rp = d.get("rope_parameters") or {}
        if "full_attention" not in rp and "sliding_attention" not in rp:  # flat / absent -> HF defaults
            rp = {}
        sliding = {"rope_type": "default", "rope_theta": 10000.0, **(rp.get("sliding_attention") or {})}
        full = {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0, **(rp.get("full_attention") or {})}
        if sliding.get("rope_type", "default") != "default":
            raise ConfigError(f"unsupported gemma4 sliding-layer rope_type {sliding.get('rope_type')!r}")
        scaling: dict[str, Any] | None
        if full["rope_type"] == "proportional":
            scaling = {"rope_type": "proportional", "partial_rotary_factor": float(full.get("partial_rotary_factor", 1.0))}
            if full.get("factor") is not None:
                scaling["factor"] = float(full["factor"])
        elif full["rope_type"] == "default" and float(full.get("partial_rotary_factor", 1.0)) == 1.0:
            scaling = None
        else:
            raise ConfigError(f"unsupported gemma4 full-attention rope_type {full['rope_type']!r}")
        act = d.get("hidden_activation") or d.get("hidden_act") or "gelu_pytorch_tanh"
        sw = d.get("sliding_window", 512)
        kv_shared = int(d.get("num_kv_shared_layers") or 0)
        ple = d.get("hidden_size_per_layer_input")
        gkv = d.get("num_global_key_value_heads")
        ghd = d.get("global_head_dim")
        plc = d.get("per_layer_config") or {}  # what transformers writes back from to_dict() instead of global_head_dim
        for i, t in enumerate(lt):
            ov = plc.get(str(i)) or plc.get(i)
            if t == "full_attention" and ov:
                ghd = ghd or ov.get("head_dim")
                gkv = gkv or ov.get("num_key_value_heads")
                break
        cap = d.get("final_logit_softcapping")
        return cls(
            arch="gemma4",
            vocab_size=int(d["vocab_size"]),
            hidden_size=int(d["hidden_size"]),
            intermediate_size=int(d["intermediate_size"]),
            num_layers=n,
            num_heads=heads,
            num_kv_heads=int(d.get("num_key_value_heads") or heads),
            head_dim=int(d.get("head_dim") or 256),
            global_head_dim=int(ghd or 512),
            num_global_kv_heads=int(gkv) if (gkv and k_eq_v) else None,
            attention_k_eq_v=k_eq_v,
            layer_types=lt,
            sliding_window=int(sw) if sw else None,
            rope_theta=float(full["rope_theta"]),
            rope_local_theta=float(sliding["rope_theta"]),
            rope_scaling=scaling,
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", True)),
            max_position_embeddings=int(d.get("max_position_embeddings", 131072)),
            attention_bias=bool(d.get("attention_bias", False)),
            hidden_act=_ACT_ALIASES.get(act, act),
            initializer_range=float(d.get("initializer_range", 0.02)),
            norm_unit_offset=False,  # gemma4 RMSNorm is plain `x_norm * weight`
            scale_embeddings=True,
            qk_norm=True,
            v_norm=True,
            sandwich_norm=True,
            layer_scalar=True,
            attn_scale=1.0,
            final_logit_softcapping=float(cap) if cap else None,
            per_layer_input_size=int(ple) if ple else 0,
            vocab_size_per_layer_input=int(d.get("vocab_size_per_layer_input") or 0) if ple else 0,
            num_kv_shared_layers=kv_shared,
            double_wide_mlp_shared=bool(d.get("use_double_wide_mlp", False)) and kv_shared > 0,
        )

    @classmethod
    def _from_lfm2_config(cls, d: dict[str, Any], mt: str) -> "ModelConfig":
        """Liquid LFM2 (hybrid gated-conv + attention) and LFM2-MoE text decoders, incl. the text backbone of ``lfm2_vl``."""
        n = int(d["num_hidden_layers"])
        heads = int(d["num_attention_heads"])
        # layer plan: explicit layer_types, else attention exactly at full_attn_idxs, else all attention
        if d.get("layer_types"):
            lt = ["conv" if t == "conv" else "full_attention" for t in d["layer_types"]]
        elif d.get("full_attn_idxs") is not None:
            idx = {int(i) for i in d["full_attn_idxs"]}
            lt = ["full_attention" if i in idx else "conv" for i in range(n)]
        else:
            lt = ["full_attention"] * n
        if len(lt) != n:
            raise ConfigError(f"lfm2 layer_types has {len(lt)} entries but num_hidden_layers is {n}")
        inter = int(d.get("intermediate_size") or d["block_ff_dim"])
        if mt == "lfm2" and d.get("block_auto_adjust_ff_dim", True):  # Lfm2MLP width (the round-up only runs when a multiplier is set)
            inter = int(2 * inter / 3)
            mult = d.get("block_ffn_dim_multiplier", 1.0)
            if mult is not None:
                inter = int(float(mult) * inter)
                m_of = int(d.get("block_multiple_of", 256))
                inter = m_of * ((inter + m_of - 1) // m_of)
        params = d.get("rope_parameters") or {}
        theta = d.get("rope_theta", params.get("rope_theta", d.get("theta", 1000000.0)))
        kw: dict[str, Any] = dict(
            arch=mt,
            vocab_size=int(d["vocab_size"]),
            hidden_size=int(d["hidden_size"]),
            intermediate_size=inter,
            num_layers=n,
            num_heads=heads,
            num_kv_heads=int(d.get("num_key_value_heads") or heads),
            rope_theta=float(theta),
            rope_scaling=_normalize_rope(d.get("rope_scaling") or params),
            rms_norm_eps=float(d.get("norm_eps", d.get("rms_norm_eps", 1e-5))),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", d.get("tie_embedding", True))),
            max_position_embeddings=int(d.get("max_position_embeddings", 128000)),
            initializer_range=float(d.get("initializer_range", 0.02)),
            qk_norm=True,
            layer_types=lt,
            conv_kernel=int(d.get("conv_L_cache", 3)),
            conv_bias=bool(d.get("conv_bias", False)),
        )
        if mt == "lfm2_moe":
            kw["moe"] = MoEConfig(
                num_experts=int(d["num_experts"]),
                top_k=int(d["num_experts_per_tok"]),
                moe_intermediate_size=int(d["moe_intermediate_size"]),
                norm_topk_prob=bool(d.get("norm_topk_prob", True)),
                router_aux_loss_coef=float(d.get("router_aux_loss_coef", 0.001)),
                mlp_only_layers=list(range(int(d.get("num_dense_layers", 0)))),
                router_type="sigmoid",
                use_expert_bias=bool(d.get("use_expert_bias", True)),
                routed_scaling_factor=float(d.get("routed_scaling_factor", 1.0)),
            )
        return cls(**kw)

    def _to_lfm2_hf_config(self) -> dict[str, Any]:
        """``Lfm2Config`` / ``Lfm2MoeConfig`` dict; the MLP width is written resolved (``block_auto_adjust_ff_dim=False``)."""
        full = [i for i in range(self.num_layers) if not self.is_conv_layer(i)]
        d: dict[str, Any] = {
            "model_type": self.arch,
            "architectures": [_HF_ARCHITECTURE[self.arch]],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "block_ff_dim": self.intermediate_size,
            "block_auto_adjust_ff_dim": False,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.kv_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "initializer_range": self.initializer_range,
            "rope_theta": self.rope_theta,
            "conv_bias": self.conv_bias,
            "conv_L_cache": self.conv_kernel,
            "layer_types": ["conv" if self.is_conv_layer(i) else "full_attention" for i in range(self.num_layers)],
            "full_attn_idxs": full,
        }
        if self.rope_scaling:
            d["rope_scaling"] = dict(self.rope_scaling)
        m = self.moe
        if m is not None:
            d.pop("block_ff_dim"), d.pop("block_auto_adjust_ff_dim")
            d.update(
                num_experts=m.num_experts,
                num_experts_per_tok=m.top_k,
                moe_intermediate_size=m.moe_intermediate_size,
                num_dense_layers=len(m.mlp_only_layers),
                use_expert_bias=m.use_expert_bias,
                routed_scaling_factor=m.routed_scaling_factor,
                norm_topk_prob=m.norm_topk_prob,
                router_aux_loss_coef=m.router_aux_loss_coef,
            )
        return d

    def _to_gemma4_hf_config(self) -> dict[str, Any]:
        """Text-only ``gemma4_text`` / ``Gemma4ForCausalLM`` config (weights are saved with the plain ``model.`` prefix)."""
        full: dict[str, Any] = {"rope_type": "proportional", "partial_rotary_factor": 1.0, **(self.rope_scaling or {}), "rope_theta": self.rope_theta}
        if not self.rope_scaling:
            full = {"rope_type": "default", "rope_theta": self.rope_theta}
        d: dict[str, Any] = {
            "model_type": "gemma4_text",
            "architectures": ["Gemma4ForCausalLM"],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "global_head_dim": self.global_head_dim or self.head_dim,
            "attention_k_eq_v": self.attention_k_eq_v,
            "hidden_activation": {"silu": "silu", "gelu": "gelu", "gelu_tanh": "gelu_pytorch_tanh", "relu": "relu"}[self.hidden_act],
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "attention_bias": bool(self.attention_bias),
            "sliding_window": self.sliding_window,
            "layer_types": list(self.layer_types or ["full_attention"] * self.num_layers),
            "final_logit_softcapping": self.final_logit_softcapping,
            "hidden_size_per_layer_input": self.per_layer_input_size,
            "vocab_size_per_layer_input": self.ple_vocab if self.per_layer_input_size else 0,
            "num_kv_shared_layers": self.num_kv_shared_layers,
            "use_double_wide_mlp": self.double_wide_mlp_shared,
            "enable_moe_block": False,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": self.rope_local_theta or self.rope_theta},
                "full_attention": full,
            },
        }
        if self.attention_k_eq_v:
            d["num_global_key_value_heads"] = self.num_global_kv_heads
        return d

    def to_hf_config(self) -> dict[str, Any]:
        """HF-compatible ``config.json`` dictionary (loadable by ``transformers`` for the same arch)."""
        a = self.arch
        if a == "gemma4":
            return self._to_gemma4_hf_config()
        if a in ("lfm2", "lfm2_moe"):
            return self._to_lfm2_hf_config()
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
