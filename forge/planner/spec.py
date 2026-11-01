"""Planner input models: :class:`ModelSpec` (what a model costs in bytes/FLOPs), :class:`ChipSpec` (numbers the cost model
needs), :class:`Scheme` (a quantisation/KV/speculation scheme) and the preset tables.

* ``ModelSpec.from_config`` accepts a ``ModelSpec``, a Hugging Face ``config.json`` dict, a ``forge.models.ModelConfig`` (any
  object with the usual attribute names) or a preset name.
* ``ChipSpec`` is resolved from a preset name, a dict, a :class:`forge.core.schemas.ChipDescriptor` (measured values win over
  datasheet values) or a ``forge.hw.capabilities.Capabilities``-like object.
* Chip presets are **vendor datasheet** numbers (dense, not sparse), rounded; they are labelled ``Provenance.vendor`` and are
  replaced by measured descriptor values whenever the probe has run (Plan HW-2).  Consumer-Blackwell entries are approximate and must
  be confirmed by the probe.
"""
from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.core.errors import ConfigError
from forge.core.schemas import ChipDescriptor, Provenance

KiB = 1024
GB = 1e9
GiB = 1024**3


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------
class ModelSpec(_M):
    """The parts of a model the cost model needs (weights, KV cache, active parameters)."""

    name: str = Field(default="", description="Display name.")
    form: Literal["dense", "moe", "fleet"] = Field(default="dense", description="Model form (Plan 2.4): dense, moe, or a multi-agent fleet sharing one base.")
    params_b: float = Field(gt=0, description="Total parameters in billions.")
    active_params_b: float | None = Field(default=None, gt=0, description="Parameters active per token in billions (MoE); None = all.")
    num_layers: int = Field(ge=1, description="Transformer layers.")
    hidden_size: int = Field(default=4096, ge=1, description="Hidden size.")
    num_heads: int = Field(default=32, ge=1, description="Query heads.")
    num_kv_heads: int = Field(default=8, ge=1, description="KV heads (GQA groups).")
    head_dim: int = Field(default=128, ge=1, description="Head width.")
    intermediate_size: int | None = Field(default=None, ge=1, description="MLP width (dense) or per-expert width (MoE).")
    vocab_size: int = Field(default=32000, ge=1, description="Vocabulary size.")
    tie_embeddings: bool = Field(default=False, description="Input embedding shared with the output head.")
    attention: Literal["gqa", "mla", "hybrid"] = Field(default="gqa", description="gqa/mha: per-head KV; mla: latent KV; hybrid: only some layers keep a KV cache.")
    full_attn_layers: int | None = Field(default=None, ge=0, description="Layers that keep a KV cache (hybrids); None = all layers.")
    mla_kv_lora_rank: int | None = Field(default=None, ge=1, description="MLA latent width (e.g. 512).")
    mla_rope_dim: int | None = Field(default=None, ge=0, description="MLA decoupled rope width (e.g. 64).")
    sliding_window: int | None = Field(default=None, ge=1, description="Window of local-attention layers.")
    sliding_layers: int | None = Field(default=None, ge=0, description="Number of local (windowed) layers among the KV layers.")
    num_experts: int | None = Field(default=None, ge=1, description="Routed experts per MoE layer.")
    experts_per_token: int | None = Field(default=None, ge=1, description="Top-k experts per token.")
    expert_params_fraction: float | None = Field(default=None, gt=0, le=1, description="Fraction of parameters that live in routed experts (MoE).")
    max_position: int | None = Field(default=None, ge=1, description="Maximum context the model supports.")
    distill_allowed: bool | None = Field(default=None, description="Licence permits distillation (None = unknown).")
    source: str = Field(default="", description="Where these numbers come from.")
    params_estimated: bool = Field(default=False, description="True if params_b was estimated from the architecture rather than given.")

    @model_validator(mode="after")
    def _check(self) -> "ModelSpec":
        if self.form == "moe" and not self.num_experts:
            raise ValueError("MoE models need num_experts")
        if self.num_heads % self.num_kv_heads:
            # tolerated (some configs are odd) but g must be >=1
            pass
        return self

    # ---- derived quantities ----------------------------------------------------------------
    @property
    def kv_layers(self) -> int:
        return self.num_layers if self.full_attn_layers is None else self.full_attn_layers

    @property
    def group_size(self) -> float:
        """g = query heads per KV head (decode-attention arithmetic intensity is g FLOP/byte at 16 bit)."""
        return self.num_heads / self.num_kv_heads

    @property
    def active_b(self) -> float:
        return self.active_params_b if self.active_params_b is not None else self.params_b

    @property
    def embed_params_b(self) -> float:
        n = self.vocab_size * self.hidden_size * (1 if self.tie_embeddings else 2)
        return n / 1e9

    def kv_elems_per_token(self) -> float:
        """Cached elements per token over all KV layers (K and V; MLA latent + rope)."""
        if self.attention == "mla" or self.mla_kv_lora_rank:
            return self.kv_layers * ((self.mla_kv_lora_rank or 512) + (self.mla_rope_dim if self.mla_rope_dim is not None else 64))
        return 2.0 * self.kv_layers * self.num_kv_heads * self.head_dim

    def kv_bytes_per_token(self, kv_bits: float = 16.0) -> float:
        """KV bytes per token per sequence (Plan 5.1: ``2 L Hkv dh b/8``; MLA: latent width; hybrids: cache layers only).

        Windowed (local) layers are counted at full length here; use :meth:`kv_bytes_for_context` for exactness."""
        return self.kv_elems_per_token() * kv_bits / 8.0

    def kv_bytes_for_context(self, ctx: int, kv_bits: float = 16.0) -> float:
        """Total KV bytes of one sequence of ``ctx`` tokens (local layers cap at their window)."""
        per_layer = self.kv_elems_per_token() / max(1, self.kv_layers) * kv_bits / 8.0
        local = min(self.sliding_layers or 0, self.kv_layers) if self.sliding_window else 0
        glob = self.kv_layers - local
        return per_layer * (glob * ctx + local * min(ctx, self.sliding_window or ctx))

    @classmethod
    def from_config(cls, cfg: Any, **overrides: Any) -> "ModelSpec":
        """Build from a preset name, ModelSpec, HF config dict or any object with the usual attributes."""
        if isinstance(cfg, ModelSpec):
            return cfg.model_copy(update=overrides) if overrides else cfg
        if isinstance(cfg, str):
            if cfg.lower() in MODEL_PRESETS:
                spec = MODEL_PRESETS[cfg.lower()]
                return spec.model_copy(update=overrides) if overrides else spec
            raise ConfigError(f"unknown model preset {cfg!r}; presets: {', '.join(sorted(MODEL_PRESETS))}")
        if hasattr(cfg, "model_dump") and not isinstance(cfg, dict):
            try:
                cfg = cfg.model_dump()
            except Exception:
                pass
        get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))

        def first(*keys: str, default: Any = None) -> Any:
            for k in keys:
                v = get(k)
                if v is not None:
                    return v
            return default

        layers = first("num_hidden_layers", "num_layers", "n_layer")
        hidden = first("hidden_size", "n_embd", "d_model")
        heads = first("num_attention_heads", "num_heads", "n_head")
        if layers is None or hidden is None or heads is None:
            raise ConfigError("model config needs layers, hidden size and heads")
        kv_heads = first("num_key_value_heads", "num_kv_heads", "n_head_kv", default=heads)
        head_dim = first("head_dim", default=hidden // heads)
        inter = first("intermediate_size", "ffn_hidden_size")
        vocab = first("vocab_size", default=32000)
        tie = bool(first("tie_word_embeddings", "tie_embeddings", default=False))
        moe = get("moe")
        n_exp = first("num_experts", "num_local_experts", "n_routed_experts")
        top_k = first("num_experts_per_tok", "experts_per_token", "num_experts_per_token", "top_k")
        moe_inter = first("moe_intermediate_size")
        if moe is not None and n_exp is None:
            mg = (lambda k: moe.get(k)) if isinstance(moe, dict) else (lambda k: getattr(moe, k, None))
            n_exp = mg("num_experts") or mg("n_experts")
            top_k = top_k or mg("top_k") or mg("num_experts_per_tok")
            moe_inter = moe_inter or mg("intermediate_size") or mg("moe_intermediate_size")
        rank = first("kv_lora_rank")
        rope = first("qk_rope_head_dim")
        window = first("sliding_window")
        kw: dict[str, Any] = dict(
            name=str(first("name", "_name_or_path", "model_type", default="")), num_layers=int(layers), hidden_size=int(hidden), num_heads=int(heads),
            num_kv_heads=int(kv_heads), head_dim=int(head_dim), intermediate_size=int(inter) if inter else None, vocab_size=int(vocab), tie_embeddings=tie,
            mla_kv_lora_rank=int(rank) if rank else None, mla_rope_dim=int(rope) if rope else None, attention="mla" if rank else "gqa",
            sliding_window=int(window) if window else None, max_position=first("max_position_embeddings"), source="from config",
        )
        if n_exp:
            kw.update(form="moe", num_experts=int(n_exp), experts_per_token=int(top_k or 2))
        pb = first("params_b")
        hybrid = _hybrid_model_config(cfg) if isinstance(cfg, dict) else None
        if hybrid is not None:  # lfm2: conv layers keep no KV cache and have their own weights -> exact analytic counts
            kw.update(attention="hybrid", full_attn_layers=len(hybrid.kv_layers), intermediate_size=hybrid.intermediate_size,
                      tie_embeddings=hybrid.tie_word_embeddings)
            if pb is None:
                pb = hybrid.param_count() / 1e9
                if hybrid.moe is not None and first("active_params_b") is None:
                    kw["active_params_b"] = hybrid.active_param_count() / 1e9
        if pb is None:
            kw["params_b"] = _estimate_params_b(kw, moe_inter)
            kw["params_estimated"] = True
        else:
            kw["params_b"] = float(pb)
        if n_exp and first("active_params_b") is None and "active_params_b" not in kw:
            kw["active_params_b"] = _estimate_active_b(kw, moe_inter)
        elif first("active_params_b") is not None:
            kw["active_params_b"] = float(first("active_params_b"))
        kw.update(overrides)
        return cls(**kw)


def _hybrid_model_config(cfg: dict[str, Any]) -> Any:
    """The ``ModelConfig`` of a config dict with gated-conv layers (lfm2), else None."""
    lt = cfg.get("layer_types")
    if not ((isinstance(lt, (list, tuple)) and "conv" in lt) or cfg.get("conv_L_cache") is not None or cfg.get("full_attn_idxs") is not None):
        return None
    try:
        from forge.models.config import ModelConfig

        mc = ModelConfig.model_validate(cfg) if "arch" in cfg else ModelConfig.from_hf_config(cfg)
    except Exception:
        return None
    return mc if mc.is_hybrid else None


def _estimate_params_b(kw: dict[str, Any], moe_inter: int | None) -> float:
    h, L = kw["hidden_size"], kw["num_layers"]
    attn = h * (kw["num_heads"] * kw["head_dim"] + 2 * kw["num_kv_heads"] * kw["head_dim"]) + kw["num_heads"] * kw["head_dim"] * h
    inter = kw.get("intermediate_size") or int(h * 8 / 3)
    if kw.get("num_experts"):
        mlp = kw["num_experts"] * 3 * h * (moe_inter or inter) + h * kw["num_experts"]
    else:
        mlp = 3 * h * inter
    emb = kw["vocab_size"] * h * (1 if kw.get("tie_embeddings") else 2)
    return (L * (attn + mlp) + emb) / 1e9


def _estimate_active_b(kw: dict[str, Any], moe_inter: int | None) -> float:
    h, L = kw["hidden_size"], kw["num_layers"]
    attn = h * (kw["num_heads"] * kw["head_dim"] + 2 * kw["num_kv_heads"] * kw["head_dim"]) + kw["num_heads"] * kw["head_dim"] * h
    inter = moe_inter or kw.get("intermediate_size") or int(h * 8 / 3)
    mlp = kw["experts_per_token"] * 3 * h * inter
    emb = kw["vocab_size"] * h * (1 if kw.get("tie_embeddings") else 2)
    return (L * (attn + mlp) + emb) / 1e9


def _P(name: str, **kw: Any) -> ModelSpec:
    return ModelSpec(name=name, **kw)


MODEL_PRESETS: dict[str, ModelSpec] = {
    # dense GQA models whose KV/token the Plan states (128 / 144 / 320 KiB)
    "llama-3-8b": _P("Llama-3-8B", params_b=8.03, num_layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8, head_dim=128, intermediate_size=14336,
                     vocab_size=128256, source="public config; Plan 5.1 / P3 Table 2.1 (128 KiB/token at 16-bit)"),
    "qwen3-8b": _P("Qwen3-8B", params_b=8.2, num_layers=36, hidden_size=4096, num_heads=32, num_kv_heads=8, head_dim=128, intermediate_size=12288,
                   vocab_size=151936, source="public config; P3 Table 2.1 (144 KiB/token at 16-bit)"),
    "llama-3.1-70b": _P("Llama-3.1-70B", params_b=70.6, num_layers=80, hidden_size=8192, num_heads=64, num_kv_heads=8, head_dim=128, intermediate_size=28672,
                        vocab_size=128256, source="public config; Plan Example A (320 KiB/token at 16-bit)"),
    "llama-3.2-1b": _P("Llama-3.2-1B", params_b=1.24, num_layers=16, hidden_size=2048, num_heads=32, num_kv_heads=8, head_dim=64, intermediate_size=8192,
                       vocab_size=128256, tie_embeddings=True, source="public config"),
    "llama-3.2-3b": _P("Llama-3.2-3B", params_b=3.21, num_layers=28, hidden_size=3072, num_heads=24, num_kv_heads=8, head_dim=128, intermediate_size=8192,
                       vocab_size=128256, tie_embeddings=True, source="public config"),
    "mistral-7b": _P("Mistral-7B", params_b=7.24, num_layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8, head_dim=128, intermediate_size=14336,
                     vocab_size=32000, source="public config"),
    # hybrids / MLA (P3 Table 2.1: 24 KiB and 8 KiB per token)
    "qwen3-next-80b-a3b": _P("Qwen3-Next-80B-A3B", form="moe", params_b=80.0, active_params_b=3.0, num_layers=48, hidden_size=2048, num_heads=16,
                             num_kv_heads=2, head_dim=256, attention="hybrid", full_attn_layers=12, num_experts=512, experts_per_token=10, vocab_size=151936,
                             source="P3 Table 2.1: 12 full-attention layers -> 24 KiB/token (parameter counts from the model card; other fields approximate)"),
    "kimi-linear-48b-a3b": _P("Kimi-Linear-48B-A3B", form="moe", params_b=48.0, active_params_b=3.0, num_layers=27, hidden_size=2304, num_heads=32,
                              num_kv_heads=32, head_dim=128, attention="hybrid", full_attn_layers=7, mla_kv_lora_rank=512, mla_rope_dim=64, num_experts=256,
                              experts_per_token=8, vocab_size=163840,
                              source="P3 Table 2.1: 7 MLA layers -> ~8 KiB/token (parameter counts from the model card; other fields approximate)"),
    "deepseek-v3": _P("DeepSeek-V3", form="moe", params_b=671.0, active_params_b=37.0, num_layers=61, hidden_size=7168, num_heads=128, num_kv_heads=128,
                      head_dim=128, attention="mla", mla_kv_lora_rank=512, mla_rope_dim=64, num_experts=256, experts_per_token=8, vocab_size=129280,
                      source="P3 Finding 2: MLA caches 576 values per token per layer"),
    "qwen3-30b-a3b": _P("Qwen3-30B-A3B", form="moe", params_b=30.5, active_params_b=3.3, num_layers=48, hidden_size=2048, num_heads=32, num_kv_heads=4,
                        head_dim=128, num_experts=128, experts_per_token=8, vocab_size=151936, source="public config (approximate)"),
}


# --------------------------------------------------------------------------------------------
# chip
# --------------------------------------------------------------------------------------------
class ChipSpec(_M):
    """Numbers the cost model needs.  ``peak_tflops`` are DENSE tensor-core rates by dtype key (bf16, fp16, fp8, int8, fp4, tf32)."""

    name: str = Field(description="Chip name / preset id.")
    sm: int | None = Field(default=None, description="CUDA SM level (e.g. 90).")
    hbm_gb: float = Field(gt=0, description="Device memory in GB (1e9 bytes).")
    hbm_bw_gbs: float = Field(gt=0, description="Memory bandwidth in GB/s (measured if available).")
    peak_tflops: dict[str, float] = Field(default_factory=dict, description="Dense peak TFLOP/s (TOP/s) per tensor-core dtype.")
    alu_tops: float | None = Field(default=None, gt=0, description="Scalar/vector ALU simple-op rate in Tera-ops/s (dequantisation work).")
    launch_us: float = Field(default=5.0, ge=0, description="Kernel launch latency in microseconds.")
    l2_mb: float | None = Field(default=None, ge=0, description="L2 cache in MB.")
    smem_kib: float | None = Field(default=None, ge=0, description="Shared memory per SM (KiB).")
    num_sms: int | None = Field(default=None, ge=1, description="Number of SMs.")
    interconnect_gbs: float | None = Field(default=None, ge=0, description="GPU-GPU bandwidth in GB/s.")
    two_four_sparse: bool = Field(default=False, description="2:4 structured sparsity in tensor cores.")
    tdp_w: float | None = Field(default=None, gt=0, description="Board power (W) for energy estimates; None = unknown.")
    pcie_gbs: float | None = Field(default=None, gt=0, description="Host-link bandwidth in GB/s used when there is no GPU-GPU interconnect; None = 25.")
    int8_tc_fast: bool = Field(default=True, description="False when the INT8 tensor-core pipe is throttled (sm_103 B300/GB300: ~1/30 of FP8); int8 kernels should use the bf16 path.")
    provenance: Provenance = Field(default=Provenance.vendor, description="vendor (datasheet) or measured (probe).")
    fingerprint: str = Field(default="", description="Chip fingerprint for calibration storage.")
    notes: str = Field(default="", description="Caveats (e.g. approximate datasheet values).")

    @property
    def bw(self) -> float:
        """Bytes per second."""
        return self.hbm_bw_gbs * 1e9

    @property
    def hbm_bytes(self) -> float:
        return self.hbm_gb * 1e9

    def peak_flops(self, dtype: str) -> float | None:
        """Dense FLOP/s for ``dtype`` (aliases: fp8_e4m3, nvfp4, mxfp4 ...) or None if not natively supported."""
        d = _canon(dtype)
        v = self.peak_tflops.get(d)
        return None if v is None else v * 1e12

    def supports(self, dtype: str) -> bool:
        return self.peak_flops(dtype) is not None

    @property
    def alu_ops(self) -> float:
        if self.alu_tops:
            return self.alu_tops * 1e12
        sms = self.num_sms or 80
        return sms * 128 * 1.5e9  # 128 lanes/SM at ~1.5 GHz, a documented fallback

    def ridge(self, dtype: str = "bf16") -> float | None:
        """Roofline ridge point in FLOP/byte for ``dtype`` (Plan 5.1: 295 FLOP/B for H100 BF16)."""
        pf = self.peak_flops(dtype)
        return None if pf is None else pf / self.bw


def _canon(d: str) -> str:
    d = d.strip().lower().replace("float", "fp").replace("bfloat16", "bf16")
    d = {"fp8_e4m3": "fp8", "fp8_e5m2": "fp8", "e4m3": "fp8", "e5m2": "fp8", "nvfp4": "fp4", "mxfp4": "fp4", "fp4_e2m1": "fp4", "e2m1": "fp4",
         "bf16": "bf16", "bfp16": "bf16", "half": "fp16", "fp16": "fp16", "f16": "fp16", "int8": "int8", "tf32": "tf32", "fp32": "fp32", "int4": "int4"}.get(d, d)
    return d


def _chip(name: str, sm: int, hbm: float, bw: float, tf: dict[str, float], sms: int, **kw: Any) -> ChipSpec:
    return ChipSpec(name=name, sm=sm, hbm_gb=hbm, hbm_bw_gbs=bw, peak_tflops=tf, num_sms=sms, provenance=Provenance.vendor, **kw)


_V = "vendor datasheet, dense (non-sparse) rates, rounded"
CHIP_PRESETS: dict[str, ChipSpec] = {c.name: c for c in [
    _chip("h100-sxm-80g", 90, 80, 3350, {"bf16": 989, "fp16": 989, "tf32": 495, "fp8": 1979, "int8": 1979}, 132, l2_mb=50, smem_kib=228, interconnect_gbs=900,
          two_four_sparse=True, notes=_V + "; Plan Fig 5.1 uses 3.35 TB/s and 989 TFLOP/s (ridge 295 FLOP/B)"),
    _chip("h100-pcie-80g", 90, 80, 2000, {"bf16": 756, "fp16": 756, "fp8": 1513, "int8": 1513}, 114, l2_mb=50, smem_kib=228, two_four_sparse=True, notes=_V),
    _chip("h200-141g", 90, 141, 4800, {"bf16": 989, "fp16": 989, "fp8": 1979, "int8": 1979}, 132, l2_mb=50, smem_kib=228, interconnect_gbs=900, two_four_sparse=True, notes=_V),
    _chip("a100-80g", 80, 80, 2039, {"bf16": 312, "fp16": 312, "tf32": 156, "int8": 624}, 108, l2_mb=40, smem_kib=164, interconnect_gbs=600, two_four_sparse=True, notes=_V),
    _chip("a100-40g", 80, 40, 1555, {"bf16": 312, "fp16": 312, "tf32": 156, "int8": 624}, 108, l2_mb=40, smem_kib=164, interconnect_gbs=600, two_four_sparse=True, notes=_V),
    _chip("a10", 86, 24, 600, {"bf16": 125, "fp16": 125, "int8": 250}, 72, l2_mb=6, smem_kib=100, two_four_sparse=True, notes=_V),
    _chip("l4", 89, 24, 300, {"bf16": 121, "fp16": 121, "fp8": 242, "int8": 242}, 58, l2_mb=48, smem_kib=100, two_four_sparse=True, notes=_V),
    _chip("l40s", 89, 48, 864, {"bf16": 362, "fp16": 362, "fp8": 733, "int8": 733}, 142, l2_mb=96, smem_kib=100, two_four_sparse=True, notes=_V),
    _chip("rtx-4090", 89, 24, 1008, {"bf16": 165, "fp16": 165, "fp8": 330, "int8": 330}, 128, l2_mb=72, smem_kib=100, two_four_sparse=True,
          notes=_V + "; fp32-accumulate rates (fp16-accumulate is 2x on GeForce)"),
    _chip("rtx-3090", 86, 24, 936, {"bf16": 71, "fp16": 71, "int8": 142}, 82, l2_mb=6, smem_kib=100, two_four_sparse=True, notes=_V + "; fp32-accumulate rates"),
    _chip("t4", 75, 16, 320, {"fp16": 65, "int8": 130}, 40, l2_mb=6, smem_kib=64, notes=_V + "; no bf16 tensor cores"),
    _chip("v100-32g", 70, 32, 900, {"fp16": 125}, 80, l2_mb=6, smem_kib=96, interconnect_gbs=300, notes=_V + "; no bf16 tensor cores"),
    _chip("b200", 100, 192, 8000, {"bf16": 2250, "fp16": 2250, "fp8": 4500, "int8": 4500, "fp4": 9000}, 148, l2_mb=126, smem_kib=228, interconnect_gbs=1800, two_four_sparse=True,
          notes=_V + "; verify with the probe"),
    _chip("b300", 103, 288, 8000, {"bf16": 2500, "fp16": 2500, "fp8": 5000, "fp4": 15000}, 160, interconnect_gbs=1800, two_four_sparse=True, int8_tc_fast=False,
          notes="P5 Table 11.2 vendor numbers for Blackwell Ultra: F4=15 PFLOP/s, F8=5 PFLOP/s, B=8 TB/s; bf16 and SM count are approximate"),
    _chip("rtx-pro-6000-blackwell", 120, 96, 1792, {"bf16": 500, "fp16": 500, "int8": 1000}, 188, tdp_w=600,
          notes="Datasheet-derived values taken from the tern158 GPU table; the int8 rate is NOT confirmed (tern158 measured 346-411 TOPS with its own kernel on this chip, RTX PRO 6000 Blackwell Server Edition, CUDA 13.4). Confirm with the probe."),
    _chip("rtx-5090", 120, 32, 1792, {"bf16": 210, "fp16": 210, "fp8": 419, "fp4": 838}, 170, l2_mb=96, smem_kib=100,
          notes="APPROXIMATE datasheet-derived values for consumer Blackwell; confirm block-scaled FP4 support and rates with the probe (ARCHITECTURE 3)"),
]}
_ALIASES = {"h100": "h100-sxm-80g", "h100-sxm": "h100-sxm-80g", "h200": "h200-141g", "a100": "a100-80g", "4090": "rtx-4090", "3090": "rtx-3090", "5090": "rtx-5090",
            "b200": "b200", "b300": "b300", "l40s": "l40s", "l4": "l4", "t4": "t4", "v100": "v100-32g", "a10": "a10"}


def _lookup_chip_by_name(name: str) -> ChipSpec | None:
    """Preset by exact id, alias, or fuzzy match on the marketing name (longest family token wins; memory size and sxm/pcie break ties)."""
    key = name.strip().lower().replace(" ", "-").replace("_", "-")
    key = re.sub(r"^(nvidia|geforce)-", "", key)
    if key in CHIP_PRESETS:
        return CHIP_PRESETS[key]
    if key in _ALIASES:
        return CHIP_PRESETS[_ALIASES[key]]
    squashed = re.sub(r"[^a-z0-9]", "", key)
    if not squashed:
        return None
    best: list[tuple[int, ChipSpec]] = []
    for pname, spec in CHIP_PRESETS.items():
        parts = pname.split("-")
        family = "".join(parts[:2]) if parts[0] == "rtx" else parts[0]
        if family and family in squashed:
            best.append((len(family), spec))
    if not best:
        return None
    top = max(n for n, _ in best)
    cands = [s for n, s in best if n == top]
    if len(cands) > 1:
        def score(c: ChipSpec) -> tuple[int, int]:
            mem = int(c.hbm_gb)
            return (int(f"{mem}g" in squashed), int(("pcie" in squashed) == ("pcie" in c.name)))
        cands.sort(key=score, reverse=True)
    return cands[0]


def resolve_chip(chip: Any) -> ChipSpec:
    """ChipSpec from a preset name, dict, ChipSpec, ChipDescriptor or Capabilities-like object."""
    if isinstance(chip, ChipSpec):
        return chip
    if isinstance(chip, str):
        c = _lookup_chip_by_name(chip)
        if c is None:
            raise ConfigError(f"unknown chip {chip!r}; presets: {', '.join(sorted(CHIP_PRESETS))}")
        return c
    if isinstance(chip, ChipDescriptor):
        return chip_from_descriptor(chip)
    if isinstance(chip, dict):
        if "hbm_bw_gbs" in chip and "hbm_gb" in chip and not isinstance(chip.get("hbm_bw_gbs"), dict):
            return ChipSpec(**chip)
        return chip_from_descriptor(ChipDescriptor(**chip))
    # Capabilities-like
    name = getattr(chip, "name", "") or ""
    base = _lookup_chip_by_name(name) if name else None
    sm = getattr(chip, "sm", None)
    if base is None and sm is not None:
        cands = [c for c in CHIP_PRESETS.values() if c.sm == sm]
        base = max(cands, key=lambda c: c.hbm_gb) if cands else None
    if base is None:
        raise ConfigError(f"cannot resolve chip from {chip!r}")
    upd: dict[str, Any] = {}
    tm = getattr(chip, "total_mem_gb", None)
    if tm:
        upd["hbm_gb"] = float(tm)
    return base.model_copy(update=upd) if upd else base


def chip_from_descriptor(d: ChipDescriptor) -> ChipSpec:
    """Measured values (probe) win over datasheet values; missing numbers fall back to the closest preset for the same name/sm."""
    base = _lookup_chip_by_name(d.name) if d.name and d.name != "unknown" else None
    bw = d.memory.hbm_bw_gbs.best or (base.hbm_bw_gbs if base else None)
    if not bw:
        raise ConfigError(f"descriptor {d.id} has no memory bandwidth (run the probe)")
    peaks: dict[str, float] = dict(base.peak_tflops) if base else {}
    measured = False
    for mu in d.matrix_units:
        v = mu.peak_tflops.best
        if v:
            for part in mu.dtype.split("|"):
                peaks[_canon(part)] = float(v)
            measured = measured or mu.peak_tflops.measured is not None
    hbm = d.memory.hbm_gb or (base.hbm_gb if base else 0.0)
    if not hbm:
        raise ConfigError(f"descriptor {d.id} has no memory size")
    return ChipSpec(
        name=d.id or d.name, sm=d.sm, hbm_gb=hbm, hbm_bw_gbs=bw, peak_tflops=peaks, alu_tops=d.alu.simple_ops_per_s_t or (base.alu_tops if base else None),
        launch_us=d.launch_latency_us if d.launch_latency_us is not None else (base.launch_us if base else 5.0), l2_mb=d.memory.l2_mb or None,
        smem_kib=d.on_chip.smem_per_cu_kib or None, num_sms=d.compute_units or (base.num_sms if base else None), interconnect_gbs=d.interconnect.intra_gbs,
        two_four_sparse=d.sparsity.two_four, provenance=Provenance.measured if (measured or d.memory.hbm_bw_gbs.measured) else Provenance.vendor,
        fingerprint=d.fingerprint, notes="from chip descriptor" + ("" if base is None else f" (gaps filled from preset {base.name})"),
    )


# --------------------------------------------------------------------------------------------
# quantisation scheme
# --------------------------------------------------------------------------------------------
class Scheme(_M):
    """A weight / activation / KV precision scheme with its *effective* bits per element (scales included)."""

    name: str = Field(description="Scheme id, e.g. 'w4a16-g128+kv8'.")
    weight_bits: float = Field(default=16.0, gt=0, description="Nominal weight bits.")
    weight_scale_bits: float = Field(default=0.0, ge=0, description="Bits of scale (+zero) metadata per group.")
    weight_group: int | None = Field(default=None, ge=1, description="Group/block size for scales.")
    act: Literal["bf16", "fp16", "fp8", "int8", "fp4"] = Field(default="bf16", description="Activation / tensor-core compute format.")
    kv_bits: float = Field(default=16.0, gt=0, description="Effective KV bits per element (including scales).")
    method: str = Field(default="none", description="Calibration method family (rtn, gptq, awq, mr_gptq, qat ...).")
    activation_sparsity: float = Field(default=0.0, ge=0, lt=1, description="Fraction of activations skipped (TEAL-style).")
    weight_density: float = Field(default=1.0, gt=0, le=1, description="Fraction of weights stored/read (pruning ratio applied by the scheme).")
    lossless_ratio: float = Field(default=1.0, gt=0, le=1, description="Cold-storage compression ratio from entropy coding (does not change HBM bytes unless decoded in-kernel).")
    experimental: bool = Field(default=False, description="Gated / experimental track (Plan 1.3 Tier 3).")
    family: str = Field(default="", description="Quality-prior family key.")

    @property
    def weight_bits_eff(self) -> float:
        """Plan 5.1 ``b_eff``: nominal bits plus amortised scales (NVFP4 4.5, MXFP4 4.25, W4 g128 4.125)."""
        extra = self.weight_scale_bits / self.weight_group if (self.weight_group and self.weight_scale_bits) else 0.0
        return self.weight_bits + extra

    @property
    def compute_dtype(self) -> str:
        return self.act

    @property
    def weight_only(self) -> bool:
        return self.act in ("bf16", "fp16") and self.weight_bits < 16

    def with_kv(self, kv_bits: float, suffix: str = "") -> "Scheme":
        return self.model_copy(update={"kv_bits": kv_bits, "name": self.name.split("+kv")[0] + (suffix or f"+kv{kv_bits:g}")})


KV_BITS = {"16": 16.0, "bf16": 16.0, "fp16": 16.0, "fp8": 8.0, "8": 8.0, "int8": 8.0, "fp8_e4m3": 8.0, "4": 4.25, "int4": 4.25, "kv4": 4.25}


def _base_schemes() -> dict[str, Scheme]:
    S = Scheme
    return {
        "bf16": S(name="bf16", family="bf16"),
        "fp16": S(name="fp16", act="fp16", family="bf16"),
        "fp8": S(name="fp8", weight_bits=8, act="fp8", method="rtn", family="fp8"),  # W8A8-FP8
        "int8": S(name="int8", weight_bits=8, act="int8", method="smoothquant", family="int8"),
        "w8a16": S(name="w8a16", weight_bits=8, act="bf16", method="rtn", family="int8"),
        "w4a16-g128": S(name="w4a16-g128", weight_bits=4, weight_scale_bits=16, weight_group=128, method="gptq", family="w4a16"),
        "w4a16-g64": S(name="w4a16-g64", weight_bits=4, weight_scale_bits=16, weight_group=64, method="gptq", family="w4a16"),
        "w4a8": S(name="w4a8", weight_bits=4, weight_scale_bits=16, weight_group=128, act="fp8", method="gptq", family="w4a8"),
        "nvfp4": S(name="nvfp4", weight_bits=4, weight_scale_bits=8, weight_group=16, act="fp4", method="mr_gptq", family="nvfp4", experimental=False),
        "mxfp4": S(name="mxfp4", weight_bits=4, weight_scale_bits=8, weight_group=32, act="fp4", method="mr_gptq", family="mxfp4"),
        "w3a16-g128": S(name="w3a16-g128", weight_bits=3, weight_scale_bits=16, weight_group=128, method="gptq", family="sub4"),
        "w2a16-vq": S(name="w2a16-vq", weight_bits=2.0, weight_scale_bits=0, method="vq", family="sub4", experimental=True),
        # ternary {-1,0,+1}: 2 bits packed storage (ideal log2(3)=1.585) + fp32 scale per 128 weights, int8 activations (EXPERIMENTAL; docs/TERNARY.md)
        "t158": S(name="t158", weight_bits=2.0, weight_scale_bits=32, weight_group=128, act="int8", method="qat", family="t158", experimental=True),
    }


SCHEMES = _base_schemes()
_SCHEME_ALIASES = {"w4a16": "w4a16-g128", "w4": "w4a16-g128", "w8a8-fp8": "fp8", "fp8-w8a8": "fp8", "w8a8": "int8", "w4a4": "nvfp4", "fp4": "nvfp4", "w8a16-int8": "w8a16", "ternary": "t158", "bitnet": "t158", "bitnet158": "t158", "t158-g128": "t158"}


def parse_scheme(text: str | Scheme | None) -> Scheme:
    """'bf16', 'fp8', 'w4a16-g128+kv8', 'nvfp4+kv4', 'fp8+kv4+sp40' -> Scheme (``+kvN`` sets KV bits; ``+spNN`` activation sparsity)."""
    if isinstance(text, Scheme):
        return text
    if text is None or str(text).strip() == "":
        return SCHEMES["bf16"]
    parts = [p.strip().lower() for p in str(text).split("+") if p.strip()]
    base = parts[0]
    base = _SCHEME_ALIASES.get(base, base)
    if base not in SCHEMES:
        m = re.fullmatch(r"w(\d+(?:\.\d+)?)a16(?:-g(\d+))?", base)
        if not m:
            raise ConfigError(f"unknown quantisation scheme {parts[0]!r}; known: {', '.join(sorted(SCHEMES))}, or wNa16-gG")
        bits, g = float(m.group(1)), int(m.group(2) or 128)
        s = Scheme(name=base, weight_bits=bits, weight_scale_bits=16, weight_group=g, method="gptq", family="w4a16" if bits >= 4 else "sub4")
    else:
        s = SCHEMES[base]
    upd: dict[str, Any] = {}
    name = s.name
    for p in parts[1:]:
        m = re.fullmatch(r"kv(\w+)", p)
        if m:
            key = m.group(1)
            if key not in KV_BITS and key.replace("fp", "") not in KV_BITS:
                raise ConfigError(f"unknown KV precision {p!r}")
            upd["kv_bits"] = KV_BITS.get(key, KV_BITS.get(key.replace("fp", ""), 16.0))
            name += f"+kv{key}"
            continue
        m = re.fullmatch(r"sp(\d+)", p)
        if m:
            upd["activation_sparsity"] = int(m.group(1)) / 100.0
            name += f"+{p}"
            continue
        raise ConfigError(f"unknown scheme modifier {p!r}")
    return s.model_copy(update={**upd, "name": name}) if upd else s


def dtype_aliases() -> dict[str, str]:
    return {"fp8_e4m3": "fp8", "nvfp4": "fp4", "mxfp4": "fp4"}


def human_params(b: float) -> str:
    return f"{b:.3g}B" if b >= 1 else f"{b * 1000:.3g}M"


def finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)
