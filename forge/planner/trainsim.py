"""Training simulator: memory per GPU, step time, throughput, communication, cost and energy of a *training* configuration, before any GPU
time is spent.  Generalised from tern158's ``simulator.py`` (a first-order roofline) to several training schemes; chip numbers come from
:class:`forge.planner.spec.ChipSpec` (one chip table).

HONESTY: every output is an ESTIMATE from a first-order roofline with guessed efficiency factors (below).  tern158 validated it against
exactly one run (125M ternary model, one RTX PRO 6000 Blackwell); nothing has been calibrated in Forge.  Never record these numbers in the
claims ledger as measured.  :func:`calibrate` rescales the throughput from a measured step time.

Schemes
-------
``bf16`` mixed precision (fp32 master + bf16 autocast) | ``fp8`` / ``nvfp4_emulated`` (Forge R1 / R2: same memory, emulated fake-quant unless the chip
has the tensor cores) | ``lora`` / ``qlora`` (frozen base, adapters trained) | ``ternary_qat`` (what Forge runs today: fake-quant, bf16-sized state) |
``ternary_native`` (tern158 design: int8 forward on packed 2-bit codes, sharded latents, packed-code all-gather; NOT available in Forge before
phases P4/P5, so this models the design, not a runnable configuration).
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge.planner.spec import ChipSpec, resolve_chip

GB = 1024 ** 3
EFF = {"bf16": 0.45, "fp8": 0.40, "int8": 0.38, "attn": 0.40, "mem": 0.70}
OVERLAP = 0.7  # fraction of communication hidden behind the backward pass
SCHEMES = ("bf16", "fp8", "nvfp4_emulated", "lora", "qlora", "ternary_qat", "ternary_native")


class TrainSimConfig(BaseModel):
    """Inputs of the training simulator (dense decoder-only transformer)."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["bf16", "fp8", "nvfp4_emulated", "lora", "qlora", "ternary_qat", "ternary_native"] = Field("bf16", description="Training scheme (see module docstring).")
    hidden: int = Field(2048, ge=16, description="Residual width.")
    layers: int = Field(24, ge=1, description="Transformer blocks.")
    heads: int = Field(16, ge=1, description="Query heads.")
    kv_heads: int = Field(16, ge=1, description="KV heads (GQA).")
    head_dim: int | None = Field(None, ge=2, description="Head dim (default hidden // heads).")
    ffn: int = Field(5632, ge=16, description="MLP hidden size.")
    vocab: int = Field(32000, ge=1, description="Vocabulary size.")
    act: Literal["swiglu", "relu2"] = Field("swiglu", description="MLP activation (swiglu has gate+up).")
    tie_embeddings: bool = Field(True, description="LM head shares the embedding matrix.")
    seq_len: int = Field(2048, ge=16, description="Training sequence length.")
    global_batch_tokens: int = Field(262144, ge=16, description="Tokens per optimiser step across all GPUs.")
    max_tokens: float = Field(2e9, gt=0, description="Token budget (for time/cost).")
    grad_checkpoint: bool = Field(False, description="Recompute activations in backward.")
    shard_state: bool = Field(False, description="Shard master weights, gradients and optimiser state over the GPUs (FSDP full-shard / tern158 ZeRO-style).")
    master_dtype: Literal["fp32", "bf16"] = Field("fp32", description="Master-weight dtype (bf16 needs stochastic rounding below lr ~3e-4).")
    state_dtype: Literal["fp32", "bf16"] = Field("fp32", description="Adam moment dtype.")
    lora_rank: int = Field(16, ge=1, description="LoRA rank (lora / qlora).")
    group_size: int = Field(128, ge=1, description="Scale group for ternary / qlora (4 B fp32 scale per group; qlora uses 8-bit absmax per 64).")
    teacher_params_b: float = Field(0.0, ge=0, description="Frozen bf16 teacher resident on each GPU for distillation (billions of params; 0 = none).")
    micro_batch: int = Field(0, ge=0, description="Sequences per GPU per micro-step (0 = largest that fits).")


class TrainSimResult(BaseModel):
    ok: bool
    scheme: str
    params: dict[str, float]
    mem_gb: dict[str, float]
    vram_gb: float
    micro_batch: int
    accum: int
    step_s: float
    tokens_per_s: float
    tokens_per_s_gpu: float
    mfu: float
    total_hours: float
    cost_usd: float
    energy_kwh: float
    breakdown_s: dict[str, float]
    notes: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    estimate: bool = Field(True, description="Always true: a first-order estimate, uncalibrated unless a calibration factor was supplied.")


def param_counts(c: TrainSimConfig) -> dict[str, float]:
    hd = c.head_dim or c.hidden // c.heads
    qkv = (c.heads + 2 * c.kv_heads) * hd * c.hidden
    o = c.heads * hd * c.hidden
    up = c.ffn * c.hidden * (2 if c.act == "swiglu" else 1)
    down = c.ffn * c.hidden
    lin = c.layers * (qkv + o + up + down)
    embed = c.vocab * c.hidden
    head = 0 if c.tie_embeddings else c.vocab * c.hidden
    norms = c.layers * 2 * c.hidden + c.hidden
    return {"linear": float(lin), "embed": float(embed), "head": float(head), "norms": float(norms), "total": float(lin + embed + head + norms)}


def _b(d: str) -> float:
    return 4.0 if d == "fp32" else 2.0


def _adapter_params(c: TrainSimConfig) -> float:
    hd = c.head_dim or c.hidden // c.heads
    mlp_up = c.ffn * (2 if c.act == "swiglu" else 1)
    per_layer = c.lora_rank * ((c.hidden + (c.heads + 2 * c.kv_heads) * hd) + (c.heads * hd + c.hidden) + (c.hidden + mlp_up) + (c.ffn + c.hidden))
    return float(c.layers * per_layer)


def _linear_state_bytes(c: TrainSimConfig, n_lin: float, world: int) -> dict[str, float]:
    """Bytes per GPU for the *linear* (quantisable) parameters."""
    sh = world if c.shard_state else 1
    out: dict[str, float] = {}
    s = c.scheme
    if s in ("lora", "qlora"):
        ad = _adapter_params(c)
        base = (0.5 + 4.0 / 64 + 0.5 / 64) if s == "qlora" else 2.0
        out["frozen base weights"] = n_lin * base
        out["adapters + grads + adam"] = ad * (4 + 4 + 8)
        return out
    if s == "ternary_native":
        out["packed codes + scales"] = n_lin * 0.25 + n_lin / c.group_size * 4
        out["latent master"] = n_lin / sh * _b(c.master_dtype)
        out["adam states"] = n_lin / sh * 2 * _b(c.state_dtype)
        out["grads"] = n_lin / sh * _b(c.master_dtype)
        return out
    out["master weights"] = n_lin / sh * _b(c.master_dtype)
    out["grads"] = n_lin / sh * 4.0
    out["adam states"] = n_lin / sh * 2 * _b(c.state_dtype)
    if sh > 1:
        out["gathered bf16 working copy"] = n_lin * 2.0
    if s == "ternary_qat":
        out["fake-quant temporaries"] = n_lin / c.layers * 4.0 * 2  # one layer's quantised weight + scales at a time (rough)
    return out


def memory_per_gpu(c: TrainSimConfig, chip: ChipSpec, n_gpu: int, mb: int) -> dict[str, float]:
    pc = param_counts(c)
    out = {k: v / GB for k, v in _linear_state_bytes(c, pc["linear"], n_gpu).items()}
    dense = pc["embed"] + pc["head"] + pc["norms"]
    out["embeddings/head/norms (+grads+adam)"] = dense * (4 + 4 + 8) / GB
    hd = c.head_dim or c.hidden // c.heads
    toks = mb * c.seq_len
    qkv_dim = (c.heads + 2 * c.kv_heads) * hd
    attn_dim = c.heads * hd
    mlp_w = c.ffn * (3 if c.act == "swiglu" else 2)
    if c.grad_checkpoint:
        saved = toks * c.hidden * 2 * c.layers
    else:
        saved = toks * (c.hidden * 2 * 3 + qkv_dim * 2 + attn_dim * 2 * 2 + mlp_w * 2 + c.hidden * 2) * c.layers
    block_peak = toks * (c.hidden * 2 * 4 + qkv_dim * 2 * 3 + attn_dim * 2 * 3 + mlp_w * 2 * 2)
    out["activations"] = (saved + block_peak) / GB
    chunk = int(max(256, min(4096, (1 << 28) // max(c.vocab, 1))))
    out["loss buffers"] = (chunk * c.vocab * 2 * 3 + chunk * c.vocab * 4) / GB
    if c.teacher_params_b > 0:
        out["teacher (bf16)"] = c.teacher_params_b * 1e9 * 2 / GB
    out["runtime (ctx, comm, fragmentation)"] = 1.6 + 0.04 * n_gpu
    return out


def _fwd_rate(c: TrainSimConfig, chip: ChipSpec, notes: list[str]) -> tuple[float, float]:
    """(FLOP/s for the linear forward, extra-time factor)."""
    bf16 = (chip.peak_flops("bf16") or chip.peak_flops("fp16") or 0.0) * EFF["bf16"]
    if bf16 <= 0:
        raise ValueError(f"chip {chip.name} has no bf16/fp16 tensor-core rate")
    s = c.scheme
    if s == "ternary_native":
        i8 = chip.peak_flops("int8")
        if i8 and chip.int8_tc_fast and (chip.sm or 0) >= 80:
            return i8 * EFF["int8"], 1.0
        notes.append("INT8 tensor cores unavailable or throttled on this chip: ternary forward modelled on the bf16 path (+15% for dequantisation)")
        return bf16, 1.15
    if s == "fp8":
        f8 = chip.peak_flops("fp8")
        if f8 and (chip.sm or 0) >= 89:
            return f8 * EFF["fp8"], 1.0
        notes.append("no FP8 tensor cores on this chip: R1 FP8 runs the emulated fake-quant path (no speed-up, ~1.3x slower than bf16)")
        return bf16, 1.3
    if s == "nvfp4_emulated":
        notes.append("NVFP4 training is emulated in Forge (fake-quant): modelled ~1.5x slower than bf16, no memory saving")
        return bf16, 1.5
    if s == "ternary_qat":
        notes.append("ternary QAT in Forge today is fake-quantised bf16 compute (no ternary kernel): modelled ~1.2x slower than bf16, no speed-up")
        return bf16, 1.2
    if s == "qlora":
        return bf16, 1.15
    return bf16, 1.0


def simulate(c: TrainSimConfig, chip: ChipSpec | str, n_gpu: int = 1, price_per_gpu_hour: float = 0.0, link: Literal["auto", "nvlink", "pcie"] = "auto",
             calibration: float = 1.0) -> TrainSimResult:
    chip = resolve_chip(chip)
    notes: list[str] = []
    sug: list[str] = []
    pc = param_counts(c)
    vram = chip.hbm_bytes / GB * 0.93  # GiB usable (7% headroom, as in tern158)
    seqs_global = max(1, c.global_batch_tokens // c.seq_len)
    per_gpu = max(1, math.ceil(seqs_global / n_gpu))

    def total(mb: int) -> float:
        return sum(memory_per_gpu(c, chip, n_gpu, mb).values())

    mb = c.micro_batch
    if mb <= 0:
        mb = next((m for m in (64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1) if m <= per_gpu and total(m) <= vram), 1)
    mem = memory_per_gpu(c, chip, n_gpu, mb)
    ok = sum(mem.values()) <= vram
    accum = max(1, math.ceil(per_gpu / mb))
    if not ok:
        notes.append(f"does NOT fit: needs {sum(mem.values()):.1f} GB per GPU, usable {vram:.1f} GB")
        if not c.shard_state or n_gpu == 1:
            sug.append("shard master weights / gradients / optimiser state over more GPUs (shard_state)")
        if c.master_dtype == "fp32" and c.scheme == "ternary_native":
            sug.append("bf16 master weights with stochastic rounding save 2 B/param (and state_dtype bf16 saves 4 B/param more)")
        if not c.grad_checkpoint:
            sug.append("enable gradient checkpointing")
        sug.append("shorten seq_len or pick a smaller model")

    lin = pc["linear"]
    head = float(c.vocab * c.hidden)
    hd = c.head_dim or c.hidden // c.heads
    toks = accum * mb * c.seq_len  # tokens per GPU per optimiser step
    rate_fwd, slow = _fwd_rate(c, chip, notes)
    bf16 = (chip.peak_flops("bf16") or chip.peak_flops("fp16") or 1.0) * EFF["bf16"]
    frozen_base = c.scheme in ("lora", "qlora")
    t_fwd = 2.0 * lin * toks / rate_fwd * slow
    t_bwd = (2.0 if frozen_base else 4.0) * lin * toks / bf16 * (slow if c.scheme in ("nvfp4_emulated", "fp8", "ternary_qat") else 1.0)
    t_re = t_fwd if c.grad_checkpoint else 0.0
    t_head = 3.0 * 2.0 * head * toks / bf16 * (1.33 if c.teacher_params_b > 0 else 1.0)
    attn = 2.0 * c.seq_len * c.heads * hd * c.layers
    t_attn = 3.0 * attn * toks / ((chip.peak_flops("bf16") or bf16) * EFF["attn"]) * (1 + (1 / 3 if c.grad_checkpoint else 0))
    elem = toks * c.layers * (c.hidden * 24 + c.ffn * 14 + (c.heads + 2 * c.kv_heads) * hd * 8)
    t_elem = elem / (chip.bw * EFF["mem"]) * (1.35 if c.grad_checkpoint else 1.0)
    wbytes = lin * ({"ternary_native": 0.25 * 2 + 2 * 2, "qlora": 0.6 + 2 * 2}.get(c.scheme, 2.0 * 2)) * accum
    t_wt = wbytes / (chip.bw * EFF["mem"])
    t_teacher = (2.0 * c.teacher_params_b * 1e9 * toks / bf16) if c.teacher_params_b > 0 else 0.0
    t_compute = t_fwd + t_bwd + t_re + t_head + t_attn + t_elem + t_wt + t_teacher

    world = n_gpu if c.shard_state else 1
    trainable = _adapter_params(c) if frozen_base else lin
    opt_bytes = trainable / world * (_b(c.master_dtype) * 2 + 2 * _b(c.state_dtype) * 2 + 2 + (4.25 if c.scheme == "ternary_native" else 0.0))
    t_opt = opt_bytes / (chip.bw * EFF["mem"]) + (pc["embed"] + pc["head"]) * 16 / (chip.bw * EFF["mem"])
    t_comm = 0.0
    if n_gpu > 1:
        nvl = chip.interconnect_gbs or 0.0
        pcie = chip.pcie_gbs or 25.0
        bwl = (nvl if (nvl > 0 and link != "pcie") else pcie) * 1e9 * 0.5 * 0.6
        ring = (n_gpu - 1) / n_gpu
        gb = trainable
        if c.shard_state:
            ag = 0.25 if c.scheme == "ternary_native" else 2.0
            vol = gb * 2 * ring * accum + gb * ag * ring  # reduce-scatter per micro-batch + weight all-gather
        else:
            vol = gb * 2 * 2 * ring  # all-reduce of bf16 grads once per step
        vol += (pc["embed"] + pc["head"]) * 2 * 2 * ring
        t_comm = vol / bwl
        if nvl == 0:
            notes.append("no GPU-GPU interconnect in the chip description: gradient traffic modelled over PCIe")
        if t_comm * (1 - OVERLAP) > 0.25 * t_compute:
            sug.append("communication bound: raise global_batch_tokens (more accumulation) or reduce sync frequency")
    step = (t_compute + max(0.0, t_comm * (1 - OVERLAP)) + t_opt) / max(calibration, 1e-6)
    tokens_step = accum * mb * c.seq_len * n_gpu
    tps = tokens_step / step
    peak = (chip.peak_flops("bf16") or chip.peak_flops("fp16") or 1.0)
    mfu = 6.0 * (lin + head) * tps / (n_gpu * peak)
    hours = c.max_tokens / tps / 3600.0
    tdp = chip.tdp_w
    if tdp is None:
        notes.append("chip has no tdp_w: energy assumes 300 W")
        tdp = 300.0
    if mb * c.seq_len < 4096:
        notes.append("tiny micro-batch: the GPU will be under-utilised")
    if accum > 64:
        sug.append("very high gradient accumulation: reduce global_batch_tokens or add GPUs")
    if c.scheme == "ternary_native":
        notes.append("ternary_native models the tern158 design (int8 forward, packed-code all-gather); Forge has no such trainer before phases P4/P5")
    notes.append("ESTIMATE: first-order roofline with guessed efficiency factors; uncalibrated" + (f" (calibration x{calibration:.2f})" if calibration != 1.0 else ""))
    br = {"linear fwd": t_fwd, "linear bwd": t_bwd, "recompute": t_re, "head+loss": t_head, "attention": t_attn, "elementwise": t_elem, "weight traffic": t_wt,
          "teacher": t_teacher, "optimizer": t_opt, "exposed comm": max(0.0, t_comm * (1 - OVERLAP))}
    return TrainSimResult(ok=ok, scheme=c.scheme, params=pc, mem_gb={k: round(v, 3) for k, v in mem.items()}, vram_gb=round(chip.hbm_gb, 1), micro_batch=mb, accum=accum,
                          step_s=step, tokens_per_s=tps, tokens_per_s_gpu=tps / n_gpu, mfu=mfu, total_hours=hours, cost_usd=hours * n_gpu * price_per_gpu_hour,
                          energy_kwh=hours * n_gpu * tdp * 0.8 / 1000.0, breakdown_s={k: round(v, 4) for k, v in br.items()}, notes=notes, suggestions=sug)


def calibrate(predicted_tokens_per_s: float, measured_tokens_per_s: float) -> float:
    """Factor to pass as ``calibration`` so predictions match a measured throughput."""
    return measured_tokens_per_s / max(predicted_tokens_per_s, 1e-9)
