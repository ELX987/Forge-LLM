# Ternary (1.58-bit) in Forge

> Status: **phases P1-P4 + P6 (UI) done on CPU; P4 kernels are interpreter-checked and PTX-compiled only.** P4 adds: Triton ternary GEMM (planes + gather kernels) and
> tern158-exact activation quantiser (`forge/kernels/triton/gemm_ternary.py`), `ops_gemm.ternary_linear` dispatcher (used by `t158` layers), spec
> `ternary_w158a8_gemm`, `ChipInfo.has_int8/int8_tc_fast`, catalog gating, bench case + `scripts/bench_ternary.py` (CUDA-graph timing, plan-only without GPU),
> `tests_gpu/test_kernels_ternary.py`, and the optional native CUDA tier `forge/native/ternary` (renamed `forge_ternary_C`, type-checked, NOT wired into dispatch).
> Note: Triton emits `mma.sync ... satfinite` for int8; accumulators cannot saturate at these magnitudes. No GPU run exists: all GPU/perf cells are 'no'.
> **P6 UI:** `GET /methods` + Methods page (honest status cards, planned slots disabled), Ternary Lab page (quick setup over `ternary_train`, flip-rate calculator, live
> train-sim panel via `components/trainsim.js`), nav entries, demo-mode handlers (DEMO DATA). Models page shows the 'Ternary 1.58b' badge and Kernel Lab shows the
> ternary kernel row automatically from the artifact label / catalog. Not done: simulator panel on New Job, per-layer heatmap, Hardware-page probe rows.
> **P1-P3:** P3 adds: `ternarize` recipe stage and `ptq format=ternary` (`forge/ternary/convert.py`), `pack` validation,
> gate `g-t158` (TARGET placeholders), recipes `ternary-ptq` / `ternary-convert`, LoRA-on-ternary test, planner Scheme `t158` + prior,
> training simulator `forge/planner/trainsim.py` + `POST /planner/train-sim`, chip table fields (`tdp_w`, `pcie_gbs`, `int8_tc_fast`) + RTX PRO 6000 preset.
> NOT in P3 (deferred): planner rule R-Q7-TERNARY and T158 claims (the claims-seed tests tie claims to rules), simulator UI panel (P6), HF conversion on
> a pretrained HF model with SubLN (needs a ModelConfig switch; `ternary-convert` uses QAT + distillation on the existing model), a separate `ternary_qat` stage
> (it is `finetune`/`distill` with `precision.quant_aware: ternary`). (P1/P2:) P2 adds: `precision.quant_aware = ternary` plug-in (`forge/ternary/qat.py`), health metrics
> `ternary/*`, collapse abort, job type `ternary_train` + templates in `configs/ternary/`, `forge ternary` CLI shortcut, job-detail health panel,
> and the `forge.methods` catalog. Not in P2: bf16-stochastic-rounded optimizer state, wd-to-zero schedule, sharded trainer, kernels. (P1:) Implemented and CPU-tested: numerics (`forge/formats/ternary.py`), layout `ternary2_row`, oracle,
> `t158` / `fq_t158` linears, artifact save/load/serve, `.t158.safetensors` tensor-level import/export. Everything else below is planned. No GPU path exists or is validated. See `docs/adr/0007-ternary-and-method-layer.md`
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

## Simulator vs the one measured run (informational, NOT a validation)

tern158 trained a 125M model on 200M tokens on one RTX PRO 6000 Blackwell: ~170k tok/s, MFU ~22%, 9.0 GB, 19.7 min. The Forge simulator (same shape,
uncalibrated) predicts: `ternary_native` ~214k tok/s / 15.6 min / 12.4 GB; `ternary_qat` ~170k tok/s / 19.6 min / 12.5 GB; `bf16` ~189k tok/s. The
`ternary_qat` throughput match is a coincidence of guessed efficiency factors, and memory is over-estimated by ~38%. One data point cannot calibrate a model.
