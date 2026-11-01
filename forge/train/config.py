"""Pydantic configuration models for training, fine-tuning and distillation (Plan 3.4 TR-1..TR-8, 7.4).

These classes are rendered into forms by the web UI through ``forge.core.jsonschema.schema_for`` so *every* field
carries a ``description`` and (where useful) ``x-group`` / ``x-advanced`` / ``x-unit`` hints.

Top-level job configs
---------------------
* :class:`TrainConfig`    -- continued / small-scale pre-training on packed token shards.
* :class:`FinetuneConfig` -- full / LoRA / QLoRA supervised fine-tuning (+ optional DPO-style preference loss).
* :class:`DistillConfig`  -- offline-logit, sequence-level, on-policy and reasoning-trace distillation.

All three inherit :class:`CommonTrainConfig` (optimiser, schedule, precision rung, parallelism, checkpointing,
evaluation, stability and budget sections).  ``IngestConfig`` (dataset ingestion job) lives in ``forge.train.data``.

Conventions
-----------
Batches use *aligned* labels: ``labels[t]`` is the target of ``logits[t]`` and ``-100`` marks ignored positions
(the loaders shift by one themselves).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from forge.core.jsonschema import ui

__all__ = [
    "BudgetConfig", "CheckpointConfig", "CommonTrainConfig", "DistillConfig", "DistillLossConfig", "DiversityConfig",
    "EvalConfig", "FinetuneConfig", "GenerationConfig", "LoggingConfig", "LogitCacheConfig", "LoraConfig",
    "ModelSourceConfig", "OnPolicyConfig", "OptimConfig", "ParallelConfig", "PrecisionConfig", "PreferenceConfig",
    "PretrainDataConfig", "PromptSourceConfig", "QLoraConfig", "SFTDataConfig", "ScheduleConfig", "StabilityConfig",
    "TeacherConfig", "TernaryQATConfig", "TrainConfig",
]


def F(default: Any, desc: str, group: str | None = None, advanced: bool | None = None, unit: str | None = None, **kw: Any) -> Any:
    """``Field`` shortcut that always sets a description and the UI hints."""
    return Field(default=default, description=desc, json_schema_extra=ui(group=group, advanced=advanced, unit=unit), **kw)


def FF(factory: Any, desc: str, group: str | None = None, advanced: bool | None = None, **kw: Any) -> Any:
    """Like :func:`F` for ``default_factory`` fields (nested models, lists, dicts)."""
    return Field(default_factory=factory, description=desc, json_schema_extra=ui(group=group, advanced=advanced), **kw)


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), validate_assignment=False)


# ---------------------------------------------------------------------------------------------------------------
# model + data sources
# ---------------------------------------------------------------------------------------------------------------
class ModelSourceConfig(_Cfg):
    """Where the model to train comes from."""

    source: Literal["pretrained", "tiny", "scratch"] = F(
        "pretrained",
        "pretrained = load weights from `path` (local dir, HF repo id or Forge model id); tiny = build the tiny test model "
        "(CPU smoke runs); scratch = random-init a ForgeLM from `arch_overrides` (pre-training from zero).",
        "Model",
    )
    path: str = F("", "Local model directory, Hugging Face repo id, or Forge model id / artifact hash. Ignored for source=tiny/scratch.", "Model")
    revision: str | None = F(None, "Hugging Face revision (branch/tag/commit) when `path` is a repo id.", "Model", True)
    backend: Literal["forge", "hf"] = F("forge", "Model implementation: the Forge decoder (kernel-swappable linears) or Hugging Face transformers.", "Model", True)
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = F(
        "auto",
        "Parameter storage dtype. auto = float32 master weights for models under ~2B params when training all weights, "
        "bfloat16 otherwise (and for frozen LoRA bases); CPU always float32.",
        "Model",
    )
    tokenizer: str | None = F(None, "Tokenizer path / repo id / 'byte' (built-in byte-level tokenizer for tests). Default: same as the model.", "Model")
    arch_overrides: dict[str, Any] = FF(dict, "ModelConfig field overrides for source=scratch/tiny (e.g. hidden_size, num_layers, vocab_size).", "Model", True)
    parent_hash: str | None = F(None, "Artifact hash of the parent model (recorded in lineage). Resolved automatically for registered models.", "Model", True)
    trust_remote_code: bool = F(False, "Allow custom modelling code when using the Hugging Face backend.", "Model", True)


class PretrainDataConfig(_Cfg):
    """Packed-token data for pre-training / distillation on raw text."""

    dataset: list[str] = FF(list, "Dataset directories or manifest.json paths produced by the `dataset_ingest` job (or dataset ids under the Forge datasets dir). Several are concatenated.", "Data")
    seq_len: int = F(2048, "Training sequence length in tokens.", "Data", ge=2, unit="tokens")
    val_dataset: list[str] = FF(list, "Optional separate held-out dataset(s). If empty, `val_sequences` sequences are held out of the training data deterministically.", "Data")
    val_sequences: int = F(64, "Number of sequences held out for evaluation when no val dataset / val split exists.", "Data", ge=0)
    max_sequences: int | None = F(None, "Use only the first N sequences of the dataset (debugging / cache-bound distillation).", "Data", True, ge=1)
    shuffle: bool = F(True, "Shuffle sequence order each epoch (deterministic function of manifest, seed and epoch).", "Data")
    data_seed: int = F(0, "Added to the run seed for the data order only; change to see a different order with identical init.", "Data", True)
    num_workers: int = F(1, "Background prefetch threads for batch assembly (0 = synchronous).", "Data", True, ge=0, le=16)
    prefetch_steps: int = F(4, "Steps of batches prefetched ahead.", "Data", True, ge=1, le=64)


class SFTDataConfig(_Cfg):
    """Supervised fine-tuning data (chat / instruction / completion / preference JSONL or parquet)."""

    train_files: list[str] = FF(list, "Training files (.jsonl, .json, .parquet). Records may be chat `messages`, ShareGPT `conversations`, Alpaca `instruction/input/output`, `prompt/completion`, `prompt/chosen/rejected` or plain `text`.", "Data")
    val_files: list[str] = FF(list, "Held-out files. If empty, `val_fraction` of the training examples is held out deterministically.", "Data")
    val_fraction: float = F(0.02, "Fraction of examples held out for evaluation when val_files is empty.", "Data", ge=0.0, le=0.5)
    format: Literal["auto", "messages", "sharegpt", "alpaca", "prompt_completion", "text", "preference"] = F("auto", "Record format; auto-detected per record when 'auto'.", "Data")
    field_map: dict[str, str] = FF(dict, "Rename fields, e.g. {'messages': 'conv', 'prompt': 'question', 'completion': 'answer'}.", "Data", True)
    template: Literal["tokenizer", "chatml", "llama3", "mistral", "alpaca", "plain"] = F(
        "chatml", "Chat template used to render messages. 'tokenizer' uses the tokenizer's own template when it has one.", "Data")
    system_prompt: str | None = F(None, "Default system prompt prepended when a record has none.", "Data")
    max_seq_len: int = F(2048, "Maximum tokens per example/pack.", "Data", ge=8, unit="tokens")
    truncation: Literal["right", "left", "drop"] = F("right", "What to do with over-long examples: cut the end, cut the start, or drop the example.", "Data")
    train_on_prompt: bool = F(False, "Also compute loss on prompt/user tokens (default: assistant tokens only).", "Data")
    train_on_all_assistant_turns: bool = F(True, "In multi-turn chats train on every assistant turn (else only the last).", "Data")
    packing: bool = F(False, "Pack several short examples into one max_seq_len sequence (no cross-example attention mask; loss masks preserved).", "Data")
    shuffle: bool = F(True, "Shuffle examples each epoch (deterministic).", "Data")
    data_seed: int = F(0, "Added to the run seed for data order only.", "Data", True)
    min_supervised_tokens: int = F(1, "Drop examples with fewer supervised (loss) tokens than this after truncation.", "Data", True, ge=0)
    num_workers: int = F(0, "Reserved: batches are assembled in-process (tokenisation is cached up front).", "Data", True, ge=0)


class PromptSourceConfig(_Cfg):
    """Prompts for sequence-level / on-policy distillation."""

    files: list[str] = FF(list, "Prompt files (.jsonl/.txt/.parquet). jsonl records use `field`.", "Data")
    field: str = F("prompt", "Field holding the prompt text (or a `messages` list) in jsonl/parquet records.", "Data")
    max_prompts: int | None = F(None, "Cap the number of prompts.", "Data", ge=1)
    max_prompt_tokens: int = F(512, "Truncate prompts to this many tokens (left-truncate).", "Data", ge=1, unit="tokens")
    use_chat_template: bool = F(True, "Wrap each prompt as a user turn using the SFT template before generation.", "Data")
    template: Literal["tokenizer", "chatml", "llama3", "mistral", "alpaca", "plain"] = F("chatml", "Template used when use_chat_template is on.", "Data")


# ---------------------------------------------------------------------------------------------------------------
# optimisation
# ---------------------------------------------------------------------------------------------------------------
class OptimConfig(_Cfg):
    """Optimiser and gradient settings."""

    name: Literal["adamw", "adamw8bit", "lion", "sgd"] = F(
        "adamw", "adamw8bit needs bitsandbytes (falls back to adamw with a warning when absent). Lion is implemented locally.", "Optimiser")
    lr: float = F(3e-4, "Peak learning rate (before schedule).", "Optimiser", gt=0.0)
    weight_decay: float = F(0.1, "Decoupled weight decay (not applied to norms, biases, 1-D parameters).", "Optimiser", ge=0.0)
    betas: list[float] = F([0.9, 0.95], "(beta1, beta2) for adamw/lion.", "Optimiser", min_length=2, max_length=2)
    eps: float = F(1e-8, "Adam epsilon.", "Optimiser", True, gt=0.0)
    momentum: float = F(0.9, "SGD momentum.", "Optimiser", True, ge=0.0, lt=1.0)
    nesterov: bool = F(False, "Use Nesterov momentum (sgd).", "Optimiser", True)
    fused: bool = F(True, "Use the fused CUDA AdamW kernel when available.", "Optimiser", True)
    grad_clip: float = F(1.0, "Global gradient-norm clipping threshold (0 = disabled).", "Optimiser", ge=0.0)
    no_decay_patterns: list[str] = F(["bias", "norm", "ln_", "embed_scale"], "Parameter-name substrings excluded from weight decay (1-D params always excluded).", "Optimiser", True)
    lr_multipliers: dict[str, float] = FF(dict, "Per-parameter-name-substring learning-rate multipliers, e.g. {'embed': 0.1, 'lm_head': 0.5}.", "Optimiser", True)
    zero_grad_set_to_none: bool = F(True, "Set grads to None instead of zero (saves memory).", "Optimiser", True)

    @field_validator("betas")
    @classmethod
    def _betas_ok(cls, v: list[float]) -> list[float]:
        if not all(0.0 <= b < 1.0 for b in v):
            raise ValueError("betas must be in [0, 1)")
        return v


class ScheduleConfig(_Cfg):
    """Learning-rate schedule."""

    name: Literal["constant", "cosine", "linear", "wsd"] = F("cosine", "constant | cosine | linear decay | wsd (warmup-stable-decay).", "Schedule")
    warmup_steps: int = F(100, "Linear warmup length in optimiser steps.", "Schedule", ge=0, unit="steps")
    warmup_ratio: float | None = F(None, "If set, warmup = ratio x total steps (overrides warmup_steps).", "Schedule", True, ge=0.0, le=1.0)
    warmup_start_ratio: float = F(0.0, "LR at step 0 as a fraction of peak.", "Schedule", True, ge=0.0, le=1.0)
    min_lr_ratio: float = F(0.1, "Final LR as a fraction of peak.", "Schedule", ge=0.0, le=1.0)
    decay_ratio: float = F(0.2, "wsd: fraction of total steps spent in the final decay phase.", "Schedule", True, gt=0.0, le=1.0)
    wsd_decay_style: Literal["linear", "cosine", "sqrt", "exp"] = F("linear", "wsd: shape of the decay phase.", "Schedule", True)
    total_steps: int | None = F(None, "Override the schedule horizon (needed when only a time budget bounds the run).", "Schedule", True, ge=1)


class TernaryQATConfig(_Cfg):
    """Ternary (1.58-bit) quantisation-aware training settings (used when ``precision.quant_aware = ternary``; EXPERIMENTAL, CPU-validated only)."""

    group_size: int = F(128, "Scale group along K (>0), -1 = one scale per output row, 0 = one per tensor. Converted models need 128; scratch runs may use rows.", "Ternary", ge=-1)
    rule: Literal["absmean", "twn", "lloyd"] = F("lloyd", "Weight quantiser: absmean, TWN, or Lloyd (absmean start + refinement steps; the default).", "Ternary")
    iters: int = F(1, "Lloyd refinement steps per quantisation.", "Ternary", True, ge=1, le=16)
    quant_acts: bool = F(True, "Per-token int8 activation quantisation (BitNet-style A8).", "Ternary")
    warmup_steps: int = F(0, "Quantisation warm-up: W_eff = W + lam*(Q(W)-W) with lam = min(1, step/warmup). 0 = quantise from step 0 (scratch). Use ~500 when converting.", "Ternary", ge=0, unit="steps")
    collapse_warn: float = F(0.60, "Warn when the fraction of zero weights exceeds this (healthy range 0.40-0.52, TARGET from the tern158 report).", "Ternary", gt=0.0, le=1.0)
    collapse_stop: float = F(0.80, "Abort the run when the zero fraction exceeds this ('zero-density collapse').", "Ternary", gt=0.0, le=1.0)
    stop_on_collapse: bool = F(True, "Raise (abort) instead of only warning on collapse.", "Ternary")
    flip_warn: float = F(0.05, "Warn when more than this fraction of codes flip per step (noise-dominated updates).", "Ternary", True, gt=0.0, le=1.0)


class PrecisionConfig(_Cfg):
    """Precision ladder (TR-1): R0 BF16 reference, R1 FP8 mixed precision, R2 NVFP4 (gated, experimental)."""

    quant_aware: Literal["none", "ternary"] = F("none", "Quantisation-aware training mode, independent of the rung: ternary = latent-weight 1.58-bit QAT (EXPERIMENTAL; uses the `ternary` section; requires rung R0).", "Precision")
    ternary: TernaryQATConfig = FF(TernaryQATConfig, "Ternary QAT settings (only used when quant_aware = ternary).", "Ternary")
    rung: Literal["R0", "R1", "R2"] = F("R0", "R0 = BF16 autocast (reference). R1 = FP8 GEMMs (torchao / TransformerEngine, else emulated fake-quant with a warning). R2 = NVFP4 training pilot (EXPERIMENTAL, emulated).", "Precision")
    compute_dtype: Literal["auto", "bfloat16", "float16", "float32"] = F("auto", "Autocast dtype. auto = bfloat16 on CUDA sm>=80 (float16 + grad scaler on older GPUs), float32 on CPU.", "Precision")
    exempt_patterns: list[str] = F(["embed", "lm_head", "head", "norm", "router", "gate.weight"], "Module-name substrings kept in high precision on R1/R2 (embeddings, output head, norms, MoE routers).", "Precision")
    exempt_first_layers: int = F(1, "Keep the first N transformer blocks in high precision (R1/R2).", "Precision", ge=0)
    exempt_last_layers: int = F(2, "Keep the last N transformer blocks in high precision (the papers keep ~15% of layers, mostly the last ones).", "Precision", ge=0)
    healing_steps: int = F(0, "Final steps run back in high precision ('healing' phase). 0 = off. Overrides healing_fraction when > 0.", "Precision", ge=0, unit="steps")
    healing_fraction: float = F(0.0, "Alternatively: fraction of the total steps (e.g. 0.18 as in the NVIDIA 12B run).", "Precision", ge=0.0, le=0.9)
    healing_rung: Literal["R0", "R1"] = F("R0", "Rung used during the healing phase.", "Precision")
    backend: Literal["auto", "torchao", "transformer_engine", "emulated"] = F("auto", "R1 backend. auto prefers torchao float8, then TransformerEngine, then the emulated fake-quant fallback (logged as a warning).", "Precision", True)
    fp8_format: Literal["hybrid", "e4m3"] = F("hybrid", "FP8 formats: hybrid = E4M3 forward / E5M2 gradients; e4m3 = E4M3 everywhere.", "Precision", True)
    fp8_scaling: Literal["dynamic", "delayed"] = F("dynamic", "Emulated/torchao scaling recipe (delayed only where the backend supports it).", "Precision", True)
    fp4_stochastic_rounding: bool = F(True, "R2: stochastic rounding on gradient operands (dgrad / wgrad).", "Precision")
    fp4_rht_wgrad: bool = F(True, "R2: random Hadamard transform along the token dimension on the weight-gradient GEMM operands.", "Precision")
    fp4_rht_block: int = F(16, "R2: Hadamard block size (power of two).", "Precision", True, ge=2, le=256)
    fp4_weight_2d: bool = F(True, "R2: 16x16 tile scaling for weights (otherwise 1x16 blocks).", "Precision")
    fp4_block: int = F(16, "R2: NVFP4 block size along the contraction dimension.", "Precision", True, ge=4, le=128)
    fp4_matmul_dtype: Literal["auto", "float32", "bfloat16"] = F("auto", "R2 emulation: dtype of the matmul over fake-quantised operands (auto: float32 on CPU, bfloat16 on CUDA).", "Precision", True)
    allow_unvalidated_fp4: bool = F(False, "Gate: R2 refuses to start unless this is true (NVFP4 training is experimental; Plan R11).", "Precision")
    diagnostics_every: int = F(50, "Compute TR-2 diagnostics (zero-grad tiles, scale flush/occupancy, SR bias) every N steps (R1/R2 only; 0 = off).", "Precision", ge=0)
    loss_gap_every: int = F(50, "Every N steps run a shadow high-precision forward on the current batch and report fp4/loss_gap (R1/R2; 0 = off).", "Precision", ge=0)
    reference_loss_file: str | None = F(None, "JSONL/CSV of reference-run losses (step, loss) from an R0 run; adds fp4/loss_gap_ref against it.", "Precision", True)
    sr_bias_draws: int = F(32, "Stochastic-rounding draws used by the SR-bias test.", "Precision", True, ge=4, le=1024)
    diag_max_tensors: int = F(8, "Max number of quantised layers sampled per diagnostics pass.", "Precision", True, ge=1)
    seed: int = F(1234, "Seed of the stochastic-rounding / RHT generators (combined with the run seed).", "Precision", True)


class ParallelConfig(_Cfg):
    """Distributed training (torchrun) and compilation."""

    strategy: Literal["none", "ddp", "fsdp"] = F("none", "none = single process; ddp = DistributedDataParallel; fsdp = FullyShardedDataParallel.", "Parallelism")
    nproc_per_node: int = F(1, "Processes (GPUs) to launch with torchrun when started from a job (ignored if already under torchrun).", "Parallelism", ge=1, le=64)
    nnodes: int = F(1, "Nodes for torchrun (multi-node needs rdzv settings via the environment).", "Parallelism", True, ge=1)
    backend: Literal["auto", "nccl", "gloo"] = F("auto", "torch.distributed backend (auto: nccl on CUDA, gloo on CPU).", "Parallelism", True)
    fsdp_sharding: Literal["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"] = F("full_shard", "FSDP sharding strategy.", "Parallelism", True)
    fsdp_cpu_offload: bool = F(False, "Offload FSDP parameters to host memory.", "Parallelism", True)
    fsdp_min_params: int = F(1_000_000, "Size-based auto-wrap threshold when the block class cannot be detected.", "Parallelism", True, ge=0)
    ddp_find_unused: bool = F(False, "DDP find_unused_parameters.", "Parallelism", True)
    ddp_bucket_cap_mb: int = F(25, "DDP gradient bucket size.", "Parallelism", True, ge=1, unit="MiB")
    compile: bool = F(False, "Wrap the model in torch.compile (guarded: falls back to eager with a warning).", "Parallelism")
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = F("default", "torch.compile mode.", "Parallelism", True)
    device: Literal["auto", "cuda", "cpu"] = F("auto", "Compute device for single-process runs.", "Parallelism")


class CheckpointConfig(_Cfg):
    """Checkpointing (resume must reproduce data order and RNG; Plan 7.4 / 8.3 / 8.7)."""

    every_steps: int = F(500, "Save a checkpoint every N optimiser steps (0 = only at the end).", "Checkpointing", ge=0, unit="steps")
    keep_last: int = F(2, "Number of most-recent rolling checkpoints to keep on local disk.", "Checkpointing", ge=1)
    keep_best: bool = F(True, "Keep the best-validation checkpoint.", "Checkpointing")
    keyframe_every: int = F(4, "Store-backed checkpoints are delta-chained; a full keyframe every N saves.", "Checkpointing", True, ge=1)
    save_optimizer: bool = F(True, "Include optimiser + scaler state (required for exact resume).", "Checkpointing")
    save_initial: bool = F(True, "Save a step-0 checkpoint so a rollback (NaN / loss-spike recovery) is possible before the first periodic save. Disable only for very large models where the extra write matters.", "Checkpointing", True)
    pre_anneal: bool = F(True, "Keep a 'pre-anneal' checkpoint right before the LR decay / healing phase begins.", "Checkpointing", True)
    use_store: bool = F(True, "Mirror checkpoints into the content-addressed store (chunk-dedup + delta chain) when available.", "Checkpointing", True)
    dir: str | None = F(None, "Checkpoint directory (default: <run_dir>/checkpoints).", "Checkpointing", True)
    resume: str = F("none", "none | auto (latest in `dir`) | best | <step number> | <path to checkpoint dir>.", "Checkpointing")


class EvalConfig(_Cfg):
    """Held-out evaluation and early stopping."""

    every_steps: int = F(200, "Held-out loss every N steps (0 = off).", "Evaluation", ge=0, unit="steps")
    num_batches: int = F(16, "Micro-batches per evaluation (fixed set, so values are comparable).", "Evaluation", ge=1)
    in_high_precision: bool = F(True, "Evaluate with quantised GEMMs disabled (deployment-precision view; R1/R2).", "Evaluation")
    early_stopping: bool = F(False, "Stop when the held-out loss stops improving.", "Evaluation")
    patience: int = F(5, "Evaluations without improvement before stopping.", "Evaluation", ge=1)
    min_delta: float = F(0.0, "Minimum decrease that counts as an improvement.", "Evaluation", ge=0.0)
    restore_best: bool = F(True, "Reload the best checkpoint at the end of an early-stopped run.", "Evaluation", True)


class StabilityConfig(_Cfg):
    """NaN/Inf and loss-spike detectors with automatic rollback (Plan 7.4 failure handling)."""

    detect_nonfinite: bool = F(True, "Detect NaN/Inf loss or gradients.", "Stability")
    on_nonfinite: Literal["skip", "rollback", "abort"] = F("rollback", "Action after `nonfinite_patience` consecutive bad steps (the bad update is always skipped).", "Stability")
    nonfinite_patience: int = F(1, "Consecutive non-finite steps tolerated before acting.", "Stability", ge=1)
    detect_spikes: bool = F(True, "Detect loss / grad-norm spikes.", "Stability")
    spike_window: int = F(50, "Trailing window (steps) for spike statistics.", "Stability", True, ge=5)
    spike_zscore: float = F(6.0, "Loss z-score above the trailing mean that counts as a spike...", "Stability", True, gt=0.0)
    spike_ratio: float = F(1.5, "...and the loss must also exceed trailing-mean x ratio.", "Stability", True, gt=1.0)
    grad_spike_factor: float = F(10.0, "Grad-norm spike when above trailing median x factor (0 = off).", "Stability", True, ge=0.0)
    on_spike: Literal["ignore", "warn", "skip", "rollback"] = F("rollback", "Action after `spike_patience` consecutive spikes.", "Stability")
    spike_patience: int = F(2, "Consecutive spiking steps before acting.", "Stability", ge=1)
    max_restarts: int = F(3, "Maximum automatic rollbacks before the run fails.", "Stability", ge=0)
    rollback_lr_scale: float = F(0.5, "Multiply the LR by this after a rollback (decays back over `rollback_cooldown_steps`).", "Stability", True, gt=0.0, le=1.0)
    rollback_cooldown_steps: int = F(100, "Steps over which the post-rollback LR scale relaxes back to 1.", "Stability", True, ge=0)
    skip_batches_on_rollback: int = F(8, "Skip this many data steps after a rollback (so the offending batch is not replayed).", "Stability", True, ge=0)
    min_loss_scale: float = F(1.0, "fp16 grad scaler: abort if the dynamic scale collapses below this.", "Stability", True, gt=0.0)


class BudgetConfig(_Cfg):
    """Stopping budgets; the run ends at the first one reached."""

    max_steps: int | None = F(1000, "Maximum optimiser steps.", "Budget", ge=1, unit="steps")
    max_tokens: int | None = F(None, "Maximum tokens seen (global).", "Budget", ge=1, unit="tokens")
    max_epochs: float | None = F(None, "Maximum passes over the training data.", "Budget", gt=0.0)
    max_minutes: float | None = F(None, "Wall-clock budget (checked between steps).", "Budget", gt=0.0, unit="min")


class LoggingConfig(_Cfg):
    """Metric emission."""

    log_every: int = F(10, "Emit metrics every N steps.", "Logging", ge=1, unit="steps")
    peak_tflops_per_gpu: float | None = F(None, "Dense peak TFLOP/s per GPU for the compute dtype (for MFU). Default: built-in vendor table by GPU name.", "Logging", True, gt=0.0)
    update_ratio_max_numel: int = F(50_000_000, "Max parameter elements snapshotted to estimate the update/weight ratio on log steps (0 = off).", "Logging", True, ge=0)
    track_gpu_memory: bool = F(True, "Report peak GPU memory.", "Logging", True)
    metrics_file: str | None = F(None, "Also append metrics as JSONL to this path (default: <run_dir>/metrics.jsonl).", "Logging", True)
    stop_check_every: int = F(1, "Check ctx.should_stop() every N steps.", "Logging", True, ge=1)


# ---------------------------------------------------------------------------------------------------------------
# common base
# ---------------------------------------------------------------------------------------------------------------
class CommonTrainConfig(_Cfg):
    """Settings shared by training, fine-tuning and distillation."""

    name: str = F("run", "Human-readable run name (also the default output model name).", "General")
    seed: int = F(1337, "Run seed (init, dropout, SR, data order via data_seed).", "General")
    deterministic: bool = F(False, "Request deterministic algorithms (slower; bit-reproducible runs).", "General", True)
    model: ModelSourceConfig = FF(ModelSourceConfig, "Model to train (student for distillation).", "Model")
    micro_batch_size: int = F(4, "Sequences per forward/backward pass per process.", "Batching", ge=1)
    grad_accum_steps: int = F(8, "Micro-batches accumulated per optimiser step. Global batch = micro_batch x accum x world_size.", "Batching", ge=1)
    activation_checkpointing: bool = F(False, "Recompute block activations in backward (saves memory, ~30% more compute).", "Batching")
    optim: OptimConfig = FF(OptimConfig, "Optimiser.", "Optimiser")
    schedule: ScheduleConfig = FF(ScheduleConfig, "LR schedule.", "Schedule")
    precision: PrecisionConfig = FF(PrecisionConfig, "Precision rung (TR-1) and TR-2 diagnostics.", "Precision")
    parallel: ParallelConfig = FF(ParallelConfig, "Parallelism / compile / device.", "Parallelism")
    checkpoint: CheckpointConfig = FF(CheckpointConfig, "Checkpointing and resume.", "Checkpointing")
    eval: EvalConfig = FF(EvalConfig, "Held-out evaluation and early stopping.", "Evaluation")
    stability: StabilityConfig = FF(StabilityConfig, "NaN/spike detection and rollback.", "Stability", True)
    budget: BudgetConfig = FF(BudgetConfig, "Stopping budgets.", "Budget")
    logging: LoggingConfig = FF(LoggingConfig, "Metric emission.", "Logging", True)
    z_loss: float = F(0.0, "Auxiliary logit z-loss coefficient (PaLM-style stabiliser; 0 = off).", "Objective", True, ge=0.0)
    aux_loss_coef: float = F(1.0, "Extra multiplier on the MoE load-balancing loss, applied on top of the model's own router_aux_loss_coef (1.0 = exactly the model's configured weight; 0 = disable).", "Objective", True, ge=0.0)
    label_smoothing: float = F(0.0, "Label smoothing for cross-entropy.", "Objective", True, ge=0.0, lt=1.0)
    output_name: str | None = F(None, "Name of the produced model directory / artifact (default: `name`).", "Output")
    save_final_model: bool = F(True, "Export the final weights as a model directory (safetensors) and register it in the store.", "Output")
    final_dtype: Literal["auto", "float32", "bfloat16", "float16"] = F("bfloat16", "Dtype of the exported model weights.", "Output", True)
    register_artifact: bool = F(True, "Register outputs in the artifact store with lineage (needs a store on the context).", "Output", True)
    tags: list[str] = FF(list, "Free-form tags recorded with the run.", "General", True)

    @property
    def effective_output_name(self) -> str:
        return self.output_name or self.name

    @model_validator(mode="after")
    def _check_budget(self) -> CommonTrainConfig:
        b = self.budget
        if b.max_steps is None and b.max_tokens is None and b.max_epochs is None and b.max_minutes is None:
            raise ValueError("budget: set at least one of max_steps / max_tokens / max_epochs / max_minutes")
        return self


# ---------------------------------------------------------------------------------------------------------------
# job configs
# ---------------------------------------------------------------------------------------------------------------
class TrainConfig(CommonTrainConfig):
    """Continued / small-scale pre-training on packed token shards (job type `train`)."""

    data: PretrainDataConfig = FF(PretrainDataConfig, "Packed-token training data.", "Data")


class LoraConfig(_Cfg):
    """LoRA adapter hyper-parameters."""

    r: int = F(16, "Rank.", "LoRA", ge=1, le=1024)
    alpha: float = F(32.0, "Scaling numerator (scale = alpha / r, or alpha / sqrt(r) with rslora).", "LoRA", gt=0.0)
    dropout: float = F(0.0, "Dropout on the adapter input.", "LoRA", ge=0.0, lt=1.0)
    target_modules: list[str] = F(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "Linear-module name suffixes to adapt. ['all-linear'] adapts every Linear except the output head.", "LoRA")
    exclude_modules: list[str] = F(["lm_head"], "Name substrings never adapted.", "LoRA", True)
    rslora: bool = F(False, "Rank-stabilised scaling alpha/sqrt(r).", "LoRA", True)
    init: Literal["default", "zero_b", "gaussian"] = F("default", "A ~ Kaiming-uniform, B = 0 ('default'/'zero_b'); 'gaussian' uses N(0, 1/r) for A.", "LoRA", True)
    modules_to_save: list[str] = F([], "Extra modules trained fully and saved with the adapter (e.g. ['embed_tokens', 'lm_head']).", "LoRA", True)
    adapter_dtype: Literal["float32", "bfloat16"] = F("float32", "Adapter parameter dtype.", "LoRA", True)
    use_forge_lora: bool = F(True, "Use forge.models.lora when importable; otherwise (or on failure) the built-in implementation.", "LoRA", True)


class QLoraConfig(_Cfg):
    """Weight-only quantisation of the frozen base for QLoRA (built-in NF4 / int8 block quantiser)."""

    bits: Literal[4, 8] = F(4, "Base-weight bits (4 = NF4 normal-float codebook, 8 = absmax int8).", "QLoRA")
    block_size: int = F(64, "Quantisation block size along the input dimension.", "QLoRA", ge=8, le=1024)
    exclude_modules: list[str] = F(["lm_head"], "Linear names kept unquantised.", "QLoRA", True)
    compute_dtype: Literal["auto", "bfloat16", "float16", "float32"] = F("auto", "Dequantisation dtype.", "QLoRA", True)


class PreferenceConfig(_Cfg):
    """Optional preference (DPO-family) objective, used when records have chosen/rejected."""

    enabled: bool = F(False, "Train with a DPO-style preference loss on `prompt/chosen/rejected` records.", "Preference")
    beta: float = F(0.1, "DPO beta (inverse temperature of the implicit reward).", "Preference", gt=0.0)
    loss_type: Literal["sigmoid", "ipo", "hinge"] = F("sigmoid", "DPO variant.", "Preference")
    label_smoothing: float = F(0.0, "Conservative-DPO label smoothing.", "Preference", True, ge=0.0, lt=0.5)
    sft_weight: float = F(0.0, "Add this x NLL of the chosen response (RPO-style regulariser).", "Preference", True, ge=0.0)
    reference: Literal["auto", "adapter_disabled", "frozen_copy"] = F("auto", "Reference policy: LoRA with adapter disabled, or a frozen copy of the initial model. auto = adapter_disabled for LoRA, frozen_copy otherwise.", "Preference", True)
    length_normalise: bool = F(False, "Average (not sum) token log-probs per response (SimPO/IPO style).", "Preference", True)


class FinetuneConfig(CommonTrainConfig):
    """Supervised fine-tuning: full, LoRA or QLoRA (job type `finetune`)."""

    method: Literal["full", "lora", "qlora"] = F("lora", "full = update all weights; lora = low-rank adapters on a frozen base; qlora = LoRA on a block-quantised frozen base.", "Method")
    lora: LoraConfig = FF(LoraConfig, "LoRA settings (lora / qlora).", "LoRA")
    qlora: QLoraConfig = FF(QLoraConfig, "Base quantisation (qlora only).", "QLoRA")
    data: SFTDataConfig = FF(SFTDataConfig, "SFT data.", "Data")
    preference: PreferenceConfig = FF(PreferenceConfig, "Preference (DPO-family) objective.", "Preference", True)
    optim: OptimConfig = FF(lambda: OptimConfig(lr=2e-4, weight_decay=0.0, betas=[0.9, 0.999], eps=1e-8), "Optimiser (SFT defaults: lr 2e-4, no weight decay).", "Optimiser")
    schedule: ScheduleConfig = FF(lambda: ScheduleConfig(name="cosine", warmup_steps=20, min_lr_ratio=0.0), "LR schedule (SFT defaults).", "Schedule")
    freeze_patterns: list[str] = F([], "Full fine-tuning: parameter-name substrings to freeze (e.g. ['embed_tokens']).", "Method", True)
    save_adapter: bool = F(True, "Save the adapter separately (adapter_config.json + adapter weights).", "Output")
    save_merged: bool = F(True, "Also export a merged full model (LoRA merged into a high-precision reload of the base).", "Output")
    merged_dtype: Literal["auto", "float32", "bfloat16", "float16"] = F("bfloat16", "Dtype of the merged export.", "Output", True)

    @model_validator(mode="after")
    def _ft_checks(self) -> FinetuneConfig:
        if self.preference.enabled and self.method == "qlora" and self.preference.reference == "frozen_copy":
            raise ValueError("preference.reference=frozen_copy is not supported with qlora (use adapter_disabled)")
        return self


class TeacherConfig(_Cfg):
    """Teacher model and its licence status (TR-8 blocking gate)."""

    path: str = F("", "Teacher model dir / HF repo id / Forge model id.", "Teacher")
    name: str = F("", "Teacher display name recorded in lineage.", "Teacher")
    hash: str | None = F(None, "Teacher artifact hash (recorded in lineage and the cache key).", "Teacher", True)
    backend: Literal["forge", "hf"] = F("forge", "Model implementation.", "Teacher", True)
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = F("bfloat16", "Teacher load dtype.", "Teacher", True)
    tokenizer: str | None = F(None, "Teacher tokenizer (must match the student's vocabulary for logit distillation).", "Teacher", True)
    licence: str | None = F(None, "SPDX-style licence id of the teacher (e.g. apache-2.0).", "Teacher")
    locally_owned: bool = F(False, "Attest that the teacher is your own model (always allowed).", "Teacher")
    distill_allowed: bool | None = F(None, "Explicit attestation: true = you reviewed the teacher's terms and distillation is allowed; false = blocked; null = decide from the licence.", "Teacher")
    permissive_licences: list[str] = F(["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "unlicense"], "Licences treated as permitting distillation without attestation.", "Teacher", True)
    source: Literal["pretrained", "tiny"] = F("pretrained", "tiny builds the CPU test model (smoke runs).", "Teacher", True)
    arch_overrides: dict[str, Any] = FF(dict, "ModelConfig overrides when source=tiny.", "Teacher", True)


class DistillLossConfig(_Cfg):
    """Distillation objective."""

    kind: Literal["forward_kl", "reverse_kl", "jsd", "ce_only"] = F("forward_kl", "forward_kl = KL(teacher || student); reverse_kl = KL(student || teacher) (mode seeking); jsd = generalised Jensen-Shannon; ce_only = hard-label SFT.", "Loss")
    temperature: float = F(1.0, "Softmax temperature applied to teacher and student (loss is scaled by T^2).", "Loss", gt=0.0)
    jsd_beta: float = F(0.5, "Mixture weight of the generalised JSD (0.5 = symmetric).", "Loss", gt=0.0, lt=1.0)
    alpha: float = F(1.0, "Weight of the distillation term; the CE term gets (1 - alpha).", "Loss", ge=0.0, le=1.0)
    tail_mode: Literal["bin", "uniform", "renorm"] = F("bin", "Top-k cache tail handling: bin = residual mass is one extra category (student's own tail mass is its bin, works for all divergences); uniform = spread residual uniformly (forward KL only); renorm = renormalise over top-k.", "Loss", True)
    token_agreement_metric: bool = F(True, "Log teacher/student argmax agreement.", "Loss", True)


class LogitCacheConfig(_Cfg):
    """Offline top-k teacher-logit cache (Plan 8.5)."""

    use_cache: bool = F(True, "Offline mode: build/read a top-k cache. False = live teacher forward each step.", "Cache")
    k: int = F(32, "Top-k probabilities stored per position.", "Cache", ge=1, le=4096)
    lp_quant: Literal["fp16", "uint8"] = F("fp16", "Log-prob storage: float16, or 8-bit uniform quantisation on [lp_floor, 0].", "Cache")
    lp_floor: float = F(-20.0, "Lowest representable log-prob for uint8 quantisation.", "Cache", True, lt=0.0)
    max_sequences: int | None = F(None, "Cache only the first N sequences of the dataset.", "Cache", ge=1)
    shard_sequences: int = F(256, "Sequences per cache shard.", "Cache", True, ge=1)
    dir: str | None = F(None, "Cache directory (default: <cache>/distill/<cache key>). Same key => reuse across students.", "Cache", True)
    teacher_batch_size: int = F(4, "Teacher forward batch size while building the cache.", "Cache", ge=1)
    rebuild: bool = F(False, "Ignore an existing complete cache with the same key.", "Cache", True)
    min_coverage_warn: float = F(0.9, "Warn if mean top-k probability mass is below this (k too small).", "Cache", True, gt=0.0, le=1.0)


class GenerationConfig(_Cfg):
    """Sampling settings for teacher/student generation."""

    max_new_tokens: int = F(128, "Maximum generated tokens per sample.", "Generation", ge=1)
    temperature: float = F(1.0, "Sampling temperature (0 = greedy).", "Generation", ge=0.0)
    top_p: float = F(1.0, "Nucleus sampling threshold.", "Generation", gt=0.0, le=1.0)
    top_k: int = F(0, "Top-k sampling (0 = off).", "Generation", ge=0)
    num_samples: int = F(1, "Samples per prompt.", "Generation", ge=1, le=64)
    seed: int = F(0, "Generation seed offset.", "Generation", True)


class OnPolicyConfig(_Cfg):
    """On-policy distillation: the student samples, the teacher scores (live)."""

    prompts: PromptSourceConfig = FF(PromptSourceConfig, "Prompt source.", "On-policy")
    generation: GenerationConfig = FF(GenerationConfig, "Student sampling parameters.", "On-policy")
    prompts_per_step: int = F(2, "Prompts per micro-batch.", "On-policy", ge=1)
    log_samples_every: int = F(50, "Log a decoded sample (aggregates only otherwise) every N steps.", "On-policy", True, ge=0)


class DiversityConfig(_Cfg):
    """Diversity / collapse monitoring of generated data (Plan 7.4: n-gram entropy, self-BLEU, distinct-n)."""

    enabled: bool = F(True, "Compute diversity metrics on generated samples.", "Diversity")
    max_samples: int = F(64, "Samples used per measurement.", "Diversity", ge=2)
    ngram: int = F(2, "n for distinct-n and n-gram entropy.", "Diversity", ge=1, le=6)
    min_distinct_n: float = F(0.2, "Warn when distinct-n falls below this (possible collapse).", "Diversity", True, ge=0.0, le=1.0)
    self_bleu_pairs: int = F(200, "Random pairs for the self-BLEU estimate.", "Diversity", True, ge=1)


class DistillConfig(CommonTrainConfig):
    """Distillation (job type `distill`). `model` is the student."""

    mode: Literal["offline_logit", "sequence_level", "on_policy", "reasoning_sft"] = F(
        "offline_logit",
        "offline_logit = top-k cached (or live) teacher distributions on a token/SFT dataset; sequence_level = teacher "
        "generates completions, student does SFT on them; on_policy = student samples, teacher scores; reasoning_sft = SFT "
        "on (licensed) reasoning traces.", "Distillation")
    teacher: TeacherConfig = FF(TeacherConfig, "Teacher (licence gate TR-8).", "Teacher")
    loss: DistillLossConfig = FF(DistillLossConfig, "Divergence, temperature and mixing.", "Loss")
    cache: LogitCacheConfig = FF(LogitCacheConfig, "Top-k logit cache.", "Cache")
    data_kind: Literal["tokens", "sft"] = F("tokens", "Which data section feeds offline_logit: packed tokens or SFT examples.", "Data")
    token_data: PretrainDataConfig = FF(PretrainDataConfig, "Packed-token data (data_kind=tokens).", "Data")
    sft_data: SFTDataConfig = FF(SFTDataConfig, "SFT / reasoning-trace data (data_kind=sft, reasoning_sft). Trace records: prompt, trace/reasoning, answer, verified.", "Data")
    on_policy: OnPolicyConfig = FF(OnPolicyConfig, "On-policy and sequence-level generation settings.", "On-policy")
    generation: GenerationConfig = FF(GenerationConfig, "Teacher generation for sequence_level mode.", "Generation")
    diversity: DiversityConfig = FF(DiversityConfig, "Diversity / collapse monitoring.", "Diversity")
    trace_template: str = F("<think>\n{trace}\n</think>\n{answer}", "reasoning_sft: how a trace and answer are joined into the assistant turn.", "Data", True)
    verified_only: bool = F(True, "reasoning_sft: keep only traces whose `verified` field is true (records without the field are kept).", "Data", True)
    pruning_plan_hash: str | None = F(None, "Hash of the pruning plan that produced the student (Minitron-style prune-distil lineage).", "Lineage", True)
    mix_non_synthetic: float = F(0.0, "Fraction of each batch drawn from real (non-synthetic) data when generated data is used (collapse mitigation; sequence_level).", "Distillation", True, ge=0.0, le=1.0)
    eval_teacher_agreement: bool = F(True, "Report teacher/student next-token agreement on the held-out set.", "Distillation", True)
