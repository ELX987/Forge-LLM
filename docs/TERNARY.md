# Ternary (1.58-bit) in Forge

> **EXPERIMENTAL. Validated on CPU only.** Nothing described here has run on a GPU through Forge. Design: `docs/adr/0007-ternary-and-method-layer.md`;
> original plan: `plan-and-docs/Forge_Ternary_Merge_Plan.pdf`; hardware validation ladder: `docs/runbooks/ternary-validation.md`.

## What it is

Weights constrained to {-1, 0, +1} with an fp32 scale per group (default group 128, so 2.25 bit/weight stored: 2 bits for the code plus 0.25 for the scale; the information
content of a code is log2(3) = 1.585 bits), activations quantised per token to int8. Forward is an int8 GEMM with int32 accumulation (exact); backward uses a
straight-through estimator with the scale detached (never learned). Latent full-precision weights exist during training only.

Storage: layout `ternary2_row`, 2-bit codes, 4 per byte along K (00 = 0, 01 = +1, 11 = -1, 10 reserved and decoded as 0), `K % 16 == 0`. This is byte-identical to tern158's
little-endian int32 words and to `formats.packing.pack_bits(t & 3, 2)`; golden vectors generated from tern158 are tested bit-for-bit (`tests/data/ternary_golden.safetensors`).

## What exists (all CPU-tested)

| Area | Where |
|------|-------|
| Numerics: quantisers (absmean, TWN, Lloyd), int8 activations, packing, flip-rate and warm-up helpers | `forge/formats/ternary.py` |
| Layout, oracle, GEMM reference | `forge/kernels/layouts/ternary.py`, `forge/kernels/ref/gemm_ternary.py` |
| Linear kinds `t158` (packed, inference) and `fq_t158` (latent-weight QAT) | `forge/ternary/linear.py` |
| Artifact: save / load / serve like any packed model, label "Ternary 1.58b"; `.t158.safetensors` import/export (tensor level) | `forge/ternary/export.py` |
| QAT as a precision plug-in (`precision.quant_aware: ternary`): warm-up blend, health metrics `ternary/*`, zero-density collapse abort | `forge/ternary/qat.py`, `forge/train/precision.py` |
| Job `ternary_train`, CLI `forge ternary`, templates | `forge/ternary/run.py`, `configs/ternary/` |
| Post-training: recipe stage `ternarize`, `ptq` with `format: ternary` (method rtn), `pack` validation; recipes `ternary-ptq`, `ternary-convert`; gate `g-t158` | `forge/ternary/convert.py`, `configs/recipes/`, `configs/gates/g-t158.yaml` |
| Kernels: Triton GEMM (planes + gather kernels) and tern158-exact activation quantiser; dispatcher `ops_gemm.ternary_linear` (used by `t158` layers); spec `ternary_w158a8_gemm`; chip flags `has_int8` / `int8_tc_fast` | `forge/kernels/triton/gemm_ternary.py`, `forge/kernels/ops_gemm.py` |
| Optional native CUDA tier (renamed `forge_ternary_C`; type-checked with stub headers; NOT wired into dispatch) | `forge/native/ternary/` |
| Training simulator `POST /planner/train-sim` (estimate, uncalibrated) and planner scheme `t158` | `forge/planner/trainsim.py` |
| Method catalog `GET /methods`; UI pages Methods and Ternary Lab; job-detail "Ternary health" panel | `forge/methods/`, `forge/web/static/js/pages/` |
| End-to-end smoke (CPU or GPU) and hardware ladder | `scripts/ternary_smoke.sh`, `scripts/ternary_validate.sh`, `scripts/bench_ternary.py` |

Quick try on any machine: `bash scripts/ternary_smoke.sh` (ingest the sample corpus, ternary QAT on the tiny model, `ternarize` + `pack`, a registered artifact labelled "Ternary 1.58b").

## Evidence status

| Claim | Status |
|-------|--------|
| Triton kernels match the torch oracle for every group layout (-1, 0, 32, 64, 128, 256) incl. tails and bias | interpreter-checked (CPU); not run on a GPU |
| Triton kernels compile for sm_80 / 89 / 120 and emit int8 `mma.sync m16n8k32` (with `satfinite`; accumulators cannot saturate at these magnitudes) | offline PTX only |
| Triton kernel speed vs tern158's CUDA kernel (346-411 TOPS at M=4096 on sm_120, bf16/cuBLAS route 332-359) | **not measured**; target >= 80%; `scripts/bench_ternary.py` |
| Ternary QAT training is faster or cheaper than bf16 | **no**: Forge runs fake-quantised bf16 compute, so it is not faster; the product is the 2-bit model |
| Converted pretrained models keep quality | **not tested**; one-shot ternarisation is lossy and expected to fail `g-t158`; QAT + distillation recipe is the intended route |
| Health thresholds (zero density healthy 40-52%, warn 60%, abort 80%; flip-rate equilibrium 0.59 x sqrt(2 x lr x wd)) | TARGET values from the tern158 report, not re-measured |
| INT8 throttled on sm_103 (B300/GB300) | datasheet-derived rule; unconfirmed on hardware |

## Simulator vs the one measured run (informational, NOT a validation)

tern158 trained a 125M model on 200M tokens on one RTX PRO 6000 Blackwell: ~170k tok/s, MFU ~22%, 9.0 GB, 19.7 min. The Forge simulator (same shape, uncalibrated) predicts
`ternary_native` ~214k tok/s / 12.4 GB, `ternary_qat` ~170k tok/s / 12.5 GB, `bf16` ~189k tok/s. The `ternary_qat` throughput match is a coincidence of guessed efficiency
factors and memory is over-estimated by ~38%. One data point cannot calibrate a model.

## Not implemented (by design or deferred)

- Sharded multi-GPU ternary trainer (`zero_ternary`: row-sharded latents, packed-code all-gather): developed separately; not described or tested here.
- bf16 stochastic-rounded optimizer state and the weight-decay-to-zero cooldown tern158 used; ternary-specific schedules.
- Planner rule `R-Q7-TERNARY` and `T158` claims-ledger entries (the seed tests tie every claim to a rule); see `docs/CLAIMS.md`.
- Converting an arbitrary pretrained HF model with SubLN (needs a ModelConfig switch); `ternary-convert` runs QAT + distillation on the loaded Forge model.
- Dedicated decode GEMV kernel, bf16 dequant+cuBLAS large-M route inside the dispatcher, Hopper wgmma / Blackwell tcgen05 ternary kernels.
- Simulator panel on New Job, per-layer health heatmap, Hardware-page probe rows.

## Provenance

Derived from tern158 v0.1 (same owner; relicensed Apache-2.0, see `NOTICE`). tern158's own results are single-GPU on one RTX PRO 6000 Blackwell (sm_120, CUDA 13.4): 23/23 kernel
selftests, 65/65 selfchecks, mma 346-411 TOPS, a 125M-parameter run to validation perplexity 141 at step 500. None of that has been reproduced through Forge.
