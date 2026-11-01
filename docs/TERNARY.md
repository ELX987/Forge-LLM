# Ternary (1.58-bit) in Forge

> Status: **skeleton (phase P0).** Nothing described below is implemented yet. See `docs/adr/0007-ternary-and-method-layer.md`
> and `plan-and-docs/Forge_Ternary_Merge_Plan.pdf` for the design and phased roadmap.

## What it is

Weights constrained to {-1, 0, +1} with a per-group scale (default group 128, +0.125 bit/weight), activations quantised per
token to int8. Packed storage is 2 bits per weight. Forward is an int8 GEMM with int32 accumulation (exact); backward uses a
straight-through estimator with the scale detached. Latent full-precision weights are kept for training only.

## Planned surfaces (by phase)

| Phase | Surface |
|-------|---------|
| P1 | `formats/ternary.py`, layout `ternary2_row`, linear kinds `t158` / `fq_t158`, `.t158.safetensors` import/export |
| P2 | `ternary.train` job, health diagnostics (`ternary/*` metrics), templates in `configs/ternary/` |
| P3 | `ternarize` / `ternary_qat` stages, HF conversion, ternary LoRA, gate `g-t158`, training simulator |
| P4 | Triton GEMM/GEMV/quantiser, optional native CUDA tier, chip int8 flags |
| P5 | `zero_ternary` sharded trainer |
| P6 | Ternary Lab and method picker in the UI |

## Validation ladder (carried over from tern158)

G0 kernels pass selftest per architecture; G1 smoke run converges on 1 and N GPUs; G2 TinyLlama conversion SQNR 5.5-7.5 dB;
G3 real run within the calibrated simulator band. No ternary row in `docs/KERNELS.md` may claim validation before its gate passes.

## Health reference (TARGET until re-measured in Forge)

Zero-weight fraction healthy 40-52%, warn 60%, stop 80%. Equilibrium flip rate per step ~ 0.59 * sqrt(2 * lr * wd).

## Provenance

Derived from tern158 v0.1 (single-GPU sm_120 results only: RTX PRO 6000 Blackwell, CUDA 13.4).
