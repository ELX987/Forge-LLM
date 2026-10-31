# ADR 0005: Fake-quant FP4 (NVFP4/MXFP4) on hardware without block-scaled MMA

- Status: accepted
- Date: 2026-10-30
- Related Plan items: R2, R11, C-033, C-090, C-091, C-092, C-093; NVFP4 paper (P5)

## Context

Native block-scaled FP4 matrix instructions exist only on Blackwell-class parts (`sm_100`, `sm_103`, and per
the architecture matrix `sm_120`/`sm_121`, the latter two "verify by probe"). The Plan says not to promise FP4
speed-ups without measurement (R2) and to treat FP4 *training* as a gated experiment (R11, evidence thin).
Most rented consumer and datacenter GPUs (Ampere, Ada, Hopper) cannot execute FP4 natively.

## Decision

1. **Numerics are defined once** in `forge.formats` (E2M1 grid, E4M3 block scales of 16, FP32 per-tensor scale,
   round-to-nearest-even, stochastic rounding, optional random Hadamard transform and 2D weight scaling) and are
   shared by PTQ, QAT/training, and the kernel oracle.
2. **On non-Blackwell GPUs and on CPU, FP4 is emulated ("fake-quant")**: values are quantised then dequantised
   into BF16/FP16 and the matmul runs in the higher precision. Memory savings can still be realised for storage
   (packed payload), but **no speed-up is claimed**.
3. **Native path only when the probe confirms it** (`has_fp4_block_scaled` plus a successful oracle comparison
   at startup). The decision is recorded in the hardware descriptor (`probe_provenance`).
4. NVFP4 training (`precision: nvfp4`) is marked **experimental and gated** in configs and the UI: it requires an
   explicit acknowledgement flag, keeps BF16 exemptions for sensitive layers, and logs the diagnostics listed in
   Plan TR-2 (`fp4/zero_grad_tile_frac`, `fp4/scale_flush_frac`, `fp4/sr_bias`, `fp4/loss_gap`).
5. Paper numbers about FP4 quality or speed are stored in the claims ledger as `paper_reported` / `untested`
   until reproduced here.

## Consequences

* Users on A100/H100/L40S/RTX 40xx can study FP4 accuracy impact and run the full pipeline; they cannot
  measure FP4 throughput benefits. The UI labels such results "emulated".
* A fake-quant training step is slower than BF16 (extra quantise/dequantise work). That is expected and
  documented, not a bug.
* Risk: fake-quant and native kernels may differ in rounding corner cases. The GPU test for the native path
  compares against the same oracle used by fake-quant.
