# Runbook: validating the ternary (1.58-bit) stack on hardware

Nothing in the ternary path has run on a GPU through Forge. This runbook is the ladder that turns each claim from "no" into a recorded result. It is carried over
from tern158's gates G0-G3. **Record what you measure; leave a cell at "no" until a run is archived.** Never extrapolate a result to another architecture.

Prerequisites: a rented host (see `docs/VAST_AI.md`), `bash scripts/vast_setup.sh`, torch with CUDA, Triton >= 3.3 for Blackwell (`python -m forge.hw.cuda_compat`
checks the stack). The optional native tier needs `nvcc` (CUDA 12.8+ for sm_100/120, 12.9+ for sm_103/121).

## G0 - kernels (per GPU architecture)

```bash
bash scripts/ternary_validate.sh                      # Triton kernels vs oracle, tern158 golden vectors, quantiser bit-exactness, CUDA-graph benchmark, smoke
FORGE_TERNARY_NATIVE=1 bash scripts/ternary_validate.sh   # + build and self-test the native CUDA tier (make -C forge/native/ternary selftest)
```

Pass: `tests_gpu/test_kernels_ternary.py` all green (`junit_ternary.xml`), native selftest 23/23 if built. Then fill, per architecture actually tested:

| Arch (device) | Triton == oracle | Quantiser bit-exact | M=1 GB/s | M=16 TOPS | M=512 TOPS | M=4096 TOPS (N=K=4096) | native selftest | vs tern158 sm_120 reference |
|---|---|---|---|---|---|---|---|---|
| sm_120 (RTX PRO 6000 Blackwell) | no | no | - | - | - | - | no | mma 346 / bf16-cuBLAS 332 / GEMV 4.8 TOPS (tern158 run, not Forge) |
| sm_80 / 86 / 89 / 90 / 100 / 103 | no | no | - | - | - | - | no | none |

Decision rule (ADR 0007): the Triton kernel ships as the default only if it matches the oracle; the native tier is wired into dispatch only if it passes its probe
AND beats Triton on that device by a margin worth a build dependency. On sm_103 the INT8 pipe is throttled (~1/30 of FP8, datasheet-derived): confirm with the
benchmark before trusting the `int8_tc_fast=False` rule.

## G1 - smoke run

Single GPU: `bash scripts/ternary_validate.sh` runs `scripts/ternary_smoke.sh`. Pass: the job succeeds, `ternary/zero_frac` stays inside 0.35-0.52 for the tiny model, the
exported artifact is labelled `Ternary 1.58b`. N GPUs: needs the sharded trainer (phase P5); parity criterion there: loss agrees with the 1-GPU run within noise and the
NCCL collectives check passes.

## G2 - conversion of a small pretrained model

```bash
forge job submit dataset_ingest -c <your tokenised corpus config>.yaml
forge job submit compress --set model=TinyLlama/TinyLlama-1.1B-Chat-v1.0 --set recipe_path=configs/recipes/ternary-ptq.yaml   # one-shot, lossy
# QAT + distillation recovery: edit configs/recipes/ternary-convert.yaml (data, max_steps) and run the same way
```

Record the per-layer SQNR from the `ternarize` stage report (tern158 expected a mean of roughly 5.5-7.5 dB for Lloyd / G=128: a TARGET). One-shot ternarisation is expected to
fail the `g-t158` gate; the recovery recipe is the test.

## G3 - a real run against the simulator and a bf16 baseline

1. `python -m forge.cli ternary -c configs/ternary/scratch-125m.yaml` (ingest a corpus first; tokenizer must match) and the same shape with `precision.quant_aware: none`.
2. Record: final eval loss for both (tern158 only recorded val loss 4.949 / ppl 141 at step 500 of a 763-step, 200M-token run and had no bf16 baseline), tokens/s, peak memory, zero fraction, flip rate.
3. Compare with `POST /planner/train-sim` (`ternary_qat` and `bf16`): tern158's single run vs the simulator was ~170k tok/s measured against ~170k predicted (coincidence) with memory over-estimated by
   ~38%. Feed the measured throughput to `forge.planner.trainsim.calibrate` and record the factor.
4. Update `docs/KERNELS.md` (validation rows) and the claims ledger only from archived reports.

## Results log

| Date | Host (GPU, driver, torch, triton) | Stage | Result | Report archive |
|---|---|---|---|---|
| - | - | - | none yet | - |
