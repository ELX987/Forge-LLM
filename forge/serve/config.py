"""Serving configuration (Plan 3.5 SV-1..SV-7, ARCHITECTURE 5.8).

Every field carries a ``Field(description=...)`` because the web UI renders the deployment form from
the JSON Schema of :class:`ServeConfig` (``forge.core.jsonschema.schema_for``).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.core.jsonschema import ui


class SamplingDefaults(BaseModel):
    """Default sampling parameters applied when a request does not override them."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(1.0, ge=0.0, le=10.0, description="Softmax temperature. 0 means greedy decoding.",
                               json_schema_extra=ui(group="Sampling"))
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="Nucleus sampling mass (1.0 disables).",
                         json_schema_extra=ui(group="Sampling"))
    top_k: int = Field(0, ge=0, description="Keep only the k most likely tokens (0 disables).",
                       json_schema_extra=ui(group="Sampling"))
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Drop tokens whose probability is below min_p * max prob.",
                         json_schema_extra=ui(group="Sampling"))
    repetition_penalty: float = Field(1.0, gt=0.0, le=5.0,
                                      description="Multiplicative penalty for tokens already in prompt/output (1.0 disables).",
                                      json_schema_extra=ui(group="Sampling", advanced=True))
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Additive penalty for tokens already generated.",
                                    json_schema_extra=ui(group="Sampling", advanced=True))
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0,
                                     description="Additive penalty proportional to the count of a token in the output.",
                                     json_schema_extra=ui(group="Sampling", advanced=True))
    max_tokens: int = Field(256, ge=1, description="Default maximum number of generated tokens per request.",
                            json_schema_extra=ui(group="Sampling", unit="tokens"))


class SpeculationConfig(BaseModel):
    """Speculative decoding (Plan SV-3/SV-4, paper P7). Output distribution is exactly unchanged (rejection sampling)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "ngram", "draft", "eagle", "auto"] = Field(
        "off",
        description="Drafter: off; ngram (prompt lookup); draft (small draft model); eagle (feature-level head on target "
                    "hidden states); auto (ngram, falling back to draft/eagle when configured).",
        json_schema_extra=ui(group="Speculation"),
    )
    max_gamma: int = Field(5, ge=1, le=16, description="Maximum draft length per verification step.",
                           json_schema_extra=ui(group="Speculation"))
    draft_model: str | None = Field(None, description="Path/repo of the draft model (mode=draft or auto). Must share the target vocabulary.",
                                    json_schema_extra=ui(group="Speculation"))
    draft_quant: str | None = Field(None, description="Optional quantisation scheme for the draft model (same names as ServeConfig.quant).",
                                    json_schema_extra=ui(group="Speculation", advanced=True))
    eagle_head: str | None = Field(None, description="Path to a trained EAGLE-style head checkpoint (mode=eagle).",
                                   json_schema_extra=ui(group="Speculation", advanced=True))
    ngram_max: int = Field(3, ge=1, le=8, description="Longest suffix n-gram tried by the prompt-lookup drafter.",
                           json_schema_extra=ui(group="Speculation", advanced=True))
    ngram_min: int = Field(1, ge=1, le=8, description="Shortest suffix n-gram accepted by the prompt-lookup drafter.",
                           json_schema_extra=ui(group="Speculation", advanced=True))
    ngram_window: int = Field(4096, ge=16, description="How many trailing tokens of history the n-gram drafter searches.",
                              json_schema_extra=ui(group="Speculation", advanced=True, unit="tokens"))
    controller: Literal["load_aware", "fixed"] = Field(
        "load_aware", description="load_aware: choose gamma per step from measured acceptance and current batch (Plan 7.5); "
                                  "fixed: always propose max_gamma.",
        json_schema_extra=ui(group="Speculation", advanced=True))
    max_batch_for_spec: int = Field(64, ge=1, description="Never speculate when more than this many sequences are running (gamma=0).",
                                    json_schema_extra=ui(group="Speculation", advanced=True))
    hysteresis: float = Field(0.03, ge=0.0, le=2.0, description="A larger gamma must beat the current best modelled gain by this fraction.",
                              json_schema_extra=ui(group="Speculation", advanced=True))
    ema_decay: float = Field(0.05, gt=0.0, le=1.0, description="EMA step for per-position acceptance estimates.",
                             json_schema_extra=ui(group="Speculation", advanced=True))
    prior_acceptance: float = Field(0.5, ge=0.0, le=1.0, description="Acceptance assumed before any measurement exists.",
                                    json_schema_extra=ui(group="Speculation", advanced=True))
    draft_cost_ratio: float | None = Field(None, ge=0.0, le=2.0,
                                           description="Drafter time / target step time. None = derive (n-gram 0.01, draft model by parameter ratio).",
                                           json_schema_extra=ui(group="Speculation", advanced=True))


class ServeConfig(BaseModel):
    """Configuration of one inference deployment (OpenAI-compatible server + engine)."""

    model_config = ConfigDict(extra="forbid")

    # ---- model --------------------------------------------------------------------------
    model: str = Field("", description="Model directory, Hugging Face repo id, or Forge artifact reference to serve.",
                       json_schema_extra=ui(group="Model"))
    tokenizer: str | None = Field(None, description="Tokenizer path/repo (defaults to the model).",
                                  json_schema_extra=ui(group="Model", advanced=True))
    served_model_name: str | None = Field(None, description="Name reported by /v1/models (defaults to the model path basename).",
                                          json_schema_extra=ui(group="Model"))
    revision: str | None = Field(None, description="Hugging Face revision (branch/tag/commit).",
                                 json_schema_extra=ui(group="Model", advanced=True))
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field(
        "auto", description="Activation/compute dtype. auto = bf16 on sm>=80, fp16 on older GPUs, fp32 on CPU.",
        json_schema_extra=ui(group="Model"))
    quant: str | None = Field(None, description="Quantisation backend for weights that are not already packed "
                                                "(None = use whatever the artifact contains, e.g. w4a16, w8a16, fp8, nvfp4).",
                              json_schema_extra=ui(group="Model"))
    device: str = Field("auto", description="Device: auto, cuda, cuda:N or cpu.", json_schema_extra=ui(group="Model"))
    tp_size: int = Field(1, ge=1, le=16, description="Tensor-parallel size. Only 1 is implemented; >1 is rejected at engine start.",
                         json_schema_extra=ui(group="Model", advanced=True))
    eos_token_ids: list[int] | None = Field(None, description="Override end-of-sequence token ids (default: from tokenizer/model config).",
                                            json_schema_extra=ui(group="Model", advanced=True))
    seed: int = Field(0, ge=0, description="Base seed for requests that do not specify one (0 = nondeterministic).", json_schema_extra=ui(group="Model", advanced=True))

    # ---- KV cache -----------------------------------------------------------------------
    kv_dtype: Literal["auto", "bf16", "fp16", "fp32", "fp8_e4m3", "int8"] = Field(
        "auto", description="KV-cache storage dtype. auto = compute dtype. fp8_e4m3/int8 store per-token-per-head scales "
                            "(int4 KV exists in the capacity planner only).",
        json_schema_extra=ui(group="KV cache"))
    block_size: int = Field(16, ge=1, le=256, description="Tokens per KV block (page).", json_schema_extra=ui(group="KV cache", unit="tokens"))
    gpu_mem_util: float = Field(0.90, gt=0.05, le=0.99, description="Fraction of GPU memory the engine may use (weights + KV + activations).",
                                json_schema_extra=ui(group="KV cache"))
    num_gpu_blocks: int | None = Field(None, ge=2, description="Override the number of KV blocks instead of sizing from free memory.",
                                       json_schema_extra=ui(group="KV cache", advanced=True))
    activation_reserve_gb: float = Field(1.0, ge=0.0, description="Memory reserved for activations/logits when sizing the KV pool "
                                                                   "(no profiling run is performed).",
                                         json_schema_extra=ui(group="KV cache", advanced=True, unit="GB"))
    host_kv_gb: float = Field(0.0, ge=0.0, description="Host-DRAM KV tier (demote/promote + swap target). 0 disables.",
                              json_schema_extra=ui(group="KV cache", unit="GB"))
    nvme_kv_dir: str | None = Field(None, description="Directory for the NVMe KV tier (None disables).",
                                    json_schema_extra=ui(group="KV cache", advanced=True))
    nvme_kv_gb: float = Field(0.0, ge=0.0, description="Capacity of the NVMe KV tier.", json_schema_extra=ui(group="KV cache", advanced=True, unit="GB"))
    prefix_caching: bool = Field(True, description="Radix prefix cache keyed by rolling block hash (SV-1).", json_schema_extra=ui(group="KV cache"))
    cache_salt: str | None = Field(None, description="Salt mixed into block hashes (isolate tenants / adapters).",
                                   json_schema_extra=ui(group="KV cache", advanced=True))

    # ---- scheduler ----------------------------------------------------------------------
    max_model_len: int = Field(4096, ge=8, description="Maximum prompt+output length per request.", json_schema_extra=ui(group="Scheduler", unit="tokens"))
    max_num_seqs: int = Field(64, ge=1, description="Maximum concurrently running sequences.", json_schema_extra=ui(group="Scheduler"))
    max_num_batched_tokens: int = Field(2048, ge=1, description="Token budget per engine step (decode + prefill chunk tokens).",
                                        json_schema_extra=ui(group="Scheduler", unit="tokens"))
    chunked_prefill: bool = Field(True, description="Split long prompts across steps and mix them with decode (SV-1).", json_schema_extra=ui(group="Scheduler"))
    policy: Literal["fcfs", "priority"] = Field("fcfs", description="Queue order: first-come-first-served, or by request priority (lower = sooner).",
                                                json_schema_extra=ui(group="Scheduler"))
    priority_aging_s: float = Field(0.0, ge=0.0, description="With policy=priority: every this many seconds of waiting improves "
                                                              "a request's effective priority by 1 (0 disables) - prevents starvation.",
                                    json_schema_extra=ui(group="Scheduler", advanced=True, unit="s"))
    preemption_mode: Literal["recompute", "swap"] = Field(
        "recompute", description="What to do when KV runs out: drop blocks and recompute (cheap with prefix cache) or swap to host DRAM "
                                 "(requires host_kv_gb > 0).", json_schema_extra=ui(group="Scheduler"))

    # ---- execution ----------------------------------------------------------------------
    cuda_graphs: bool = Field(False, description="Capture CUDA graphs for decode steps (GPU only, requires the paged decode kernel; "
                                                   "implemented but not yet validated on hardware).",
                              json_schema_extra=ui(group="Execution"))
    cuda_graph_max_batch: int = Field(32, ge=1, description="Largest decode batch captured in a CUDA graph.", json_schema_extra=ui(group="Execution", advanced=True))
    kernel_attention: Literal["auto", "on", "off"] = Field(
        "auto", description="Use forge.kernels.ops_attn paged decode kernels (auto = on when CUDA is present, falling back to the "
                            "torch reference path on error).", json_schema_extra=ui(group="Execution", advanced=True))

    # ---- speculation --------------------------------------------------------------------
    speculation: SpeculationConfig = Field(default_factory=SpeculationConfig, description="Speculative decoding settings.",
                                           json_schema_extra=ui(group="Speculation"))

    # ---- server -------------------------------------------------------------------------
    host: str = Field("127.0.0.1", description="Bind address of the OpenAI-compatible HTTP server.", json_schema_extra=ui(group="Server"))
    port: int = Field(8000, ge=0, le=65535, description="TCP port of the HTTP server (0 = pick a free port).", json_schema_extra=ui(group="Server"))
    api_key: str | None = Field(None, description="If set, clients must send 'Authorization: Bearer <key>'.", json_schema_extra=ui(group="Server"))
    chat_template: str | None = Field(None, description="Override Jinja chat template (default: tokenizer's own).",
                                      json_schema_extra=ui(group="Server", advanced=True))
    stats_interval_s: float = Field(1.0, gt=0.0, description="Interval for metric emission to the job context.", json_schema_extra=ui(group="Server", advanced=True, unit="s"))
    sampling: SamplingDefaults = Field(default_factory=SamplingDefaults, description="Default sampling parameters.",
                                       json_schema_extra=ui(group="Sampling"))

    @model_validator(mode="after")
    def _check(self) -> ServeConfig:
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if self.nvme_kv_gb > 0 and not self.nvme_kv_dir:
            raise ValueError("nvme_kv_gb > 0 requires nvme_kv_dir")
        return self

    def display_name(self) -> str:
        if self.served_model_name:
            return self.served_model_name
        if self.model:
            return self.model.replace("\\", "/").rstrip("/").split("/")[-1]
        return "forge-model"
