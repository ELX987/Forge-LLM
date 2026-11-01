"""Built-in method descriptors (metadata only).  Status is deliberately conservative: nothing is ``gpu_validated`` unless a hardware run
is recorded in docs/KERNELS.md; ``planned`` entries are reserved slots that are NOT implemented."""
from __future__ import annotations

from .catalog import MethodSpec as M
from .catalog import register_method as reg

# --- pre-training / full fine-tuning -------------------------------------------------------------------------------------------
reg(M(id="pretrain-bf16", title="Pre-train / continue training (bf16)", family="pretrain", status="supported", job_type="train",
      summary="Decoder-only pre-training with bf16 autocast, AdamW, DDP/FSDP, resumable checkpoints.", template="train/llama-8b-full-bf16.yaml",
      produces="model", needs=["dataset"]))
reg(M(id="finetune-full", title="Full fine-tuning (SFT)", family="pretrain", status="supported", job_type="finetune", config_hint={"method": "full"},
      summary="Supervised fine-tuning of all weights.", produces="model", needs=["dataset"]))
# --- low-precision training ----------------------------------------------------------------------------------------------------
reg(M(id="train-fp8", title="FP8 training (R1)", family="low_precision_training", status="experimental", job_type="train",
      config_hint={"precision.rung": "R1"}, summary="FP8 GEMMs via torchao / TransformerEngine, else emulated fake-quant (no speed-up). Unvalidated on hardware.",
      template="train/qwen-fp8.yaml", needs=["sm>=89 for real FP8"], diagnostics_prefix="fp8", planner_scheme="fp8"))
reg(M(id="train-nvfp4", title="NVFP4 training pilot (R2)", family="low_precision_training", status="experimental", job_type="train",
      config_hint={"precision.rung": "R2", "precision.allow_unvalidated_fp4": True}, summary="Emulated NVFP4 training pilot; gated behind allow_unvalidated_fp4.",
      template="train/nvfp4-pilot.yaml", diagnostics_prefix="fp4", planner_scheme="nvfp4"))
# --- PEFT / preference ---------------------------------------------------------------------------------------------------------
reg(M(id="lora", title="LoRA / DoRA", family="peft", status="supported", job_type="finetune", config_hint={"method": "lora"},
      summary="Low-rank adapters (optionally DoRA / rsLoRA).", template="finetune/llama-8b-lora.yaml", produces="adapter", needs=["dataset"]))
reg(M(id="qlora", title="QLoRA (NF4 base)", family="peft", status="supported", job_type="finetune", config_hint={"method": "qlora"},
      summary="LoRA on a frozen 4-bit NF4 base.", produces="adapter", needs=["dataset"]))
reg(M(id="dpo", title="Preference optimisation (DPO)", family="peft", status="supported", job_type="finetune",
      summary="DPO-style preference loss on top of full / LoRA fine-tuning.", needs=["preference dataset"]))
# --- distillation --------------------------------------------------------------------------------------------------------------
reg(M(id="distill-offline", title="Offline logit distillation", family="distillation", status="supported", job_type="distill",
      template="distill/offline-logit-7b-to-1.5b.yaml", summary="Top-k cached teacher logits (forward/reverse KL, JSD).", needs=["teacher model", "dataset"]))
reg(M(id="distill-on-policy", title="On-policy distillation", family="distillation", status="supported", job_type="distill",
      template="distill/on-policy-7b-to-1.5b.yaml", summary="Student samples, teacher scores.", needs=["teacher model", "prompts"]))
# --- post-training quantization / compression ----------------------------------------------------------------------------------
reg(M(id="ptq-w4a16-gptq", title="GPTQ W4A16 (g128)", family="ptq", status="supported", job_type="compress", template="compress/w4a16-gptq-g128.yaml",
      produces="quantized", summary="Hessian-based 4-bit weight quantisation with Triton W4A16 kernels.", needs=["calibration data"], planner_scheme="w4a16-g128"))
reg(M(id="ptq-fp8-w8a8", title="FP8 W8A8 PTQ", family="ptq", status="supported", job_type="compress", template="compress/fp8-w8a8.yaml",
      produces="quantized", summary="FP8 weights and activations.", needs=["calibration data"], planner_scheme="fp8"))
reg(M(id="ptq-nvfp4", title="NVFP4 PTQ", family="ptq", status="supported", job_type="compress", template="compress/nvfp4-ptq.yaml",
      produces="quantized", summary="Block-scaled FP4 weights.", needs=["calibration data"], planner_scheme="nvfp4"))
reg(M(id="prune-distill", title="Prune + distill", family="compression", status="supported", job_type="compress",
      template="compress/prune-distill-8b-to-4b.yaml", summary="Structured pruning followed by distillation recovery.", needs=["calibration data", "teacher model"]))
reg(M(id="lowrank", title="Low-rank factorisation", family="compression", status="supported", job_type="compress", template="compress/lowrank.yaml",
      summary="Whitened-SVD low-rank layers with healing."))
reg(M(id="bitdelta", title="BitDelta (1-bit delta)", family="compression", status="supported", job_type="compress", template="compress/delta.yaml",
      produces="delta", summary="Sign + scale delta vs. a base model for multi-tenant serving.", needs=["base model"]))
# --- ternary (this merge; see docs/TERNARY.md) ---------------------------------------------------------------------------------
reg(M(id="ternary-train", title="Ternary (1.58-bit) QAT from scratch", family="ternary", status="experimental", job_type="ternary_train",
      template="ternary/scratch-125m.yaml", summary="Latent-weight QAT with {-1,0,+1} weights and int8 activations; health metrics ternary/*. "
      "CPU-validated only; no ternary kernel yet.", needs=["dataset"], diagnostics_prefix="ternary", planner_scheme="t158", docs="docs/TERNARY.md"))
reg(M(id="ternary-ptq", title="Ternary post-training quantisation", family="ternary", status="planned",
      summary="Absmean/Lloyd ternarisation of a pretrained model (recipe stage `ternarize`). Phase P3.", produces="quantized",
      diagnostics_prefix="ternary", planner_scheme="t158", docs="docs/TERNARY.md"))
reg(M(id="ternary-convert", title="Convert a pretrained model to ternary", family="ternary", status="planned",
      summary="HF -> ternary with warm-up, SubLN and KD recovery. Phase P3.", produces="quantized", needs=["teacher model", "dataset"],
      diagnostics_prefix="ternary", planner_scheme="t158", docs="docs/TERNARY.md"))
reg(M(id="ternary-lora", title="LoRA on a ternary base", family="ternary", status="planned", summary="Adapters on a packed ternary model. Phase P3.",
      produces="adapter", docs="docs/TERNARY.md"))
# --- reserved slots (not scheduled; each needs its own plan) -----------------------------------------------------------------------
for _id, _fam, _t, _s in [
    ("bitdistill-kd", "qat", "BitDistill-style hidden/attention-relation distillation", "Layer-wise KD for ternary students."),
    ("muon-latent", "qat", "Muon / 8-bit Adam on latent weights", "Optimiser variants for latent-weight QAT."),
    ("multi-plane-ternary", "ternary", "Multi-plane ternary (K=2)", "Two ternary planes per weight."),
    ("nm-sparse-ternary", "ternary", "N:M sparse ternary", "Structured sparsity on ternary weights."),
    ("vector-quantization", "ptq", "Vector quantization (AQLM/QuIP#-style)", "Codebook-based sub-3-bit PTQ."),
]:
    reg(M(id=_id, title=_t, family=_fam, status="planned", summary=_s))
