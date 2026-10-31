# ADR 0002: Triton-first kernels with pure-torch oracles

- Status: accepted
- Date: 2026-10-30
- Related Plan items: KR-1, KR-2, KR-9, NFR-V1, C-040, C-042, C-043

## Context

The Plan requires kernels (K1-K9) that are correct first, fast second, and portable across chips, with
chip-specific layouts and autotuning. The target is "most consumer and datacenter NVIDIA GPUs" from `sm_70`
to `sm_121`. Candidate implementation routes:

| Route | Pros | Cons |
|-------|------|------|
| Hand-written CUDA / CUTLASS | Peak performance, full control of tcgen05/wgmma | Needs `nvcc` on the rented box, long compile, one code path per arch, hard to unit test off-GPU |
| Triton | Pure Python, JIT per device, `tl.dot` maps to the right MMA per arch, autotuning built in | Less control over async rings and fp4 block-scaled MMA; version-sensitive |
| torch-only | Runs everywhere | Not fast, not bandwidth-bound for packed formats |

## Decision

1. **Triton is the primary kernel language.** Kernels stay simple and standard (`tl.dot`, masked
   `tl.load/store`, `tl.constexpr` tile parameters). Exotic APIs are version-guarded.
2. **Every kernel has a pure-torch reference (the "oracle")** in `forge/kernels/ref/`. It runs on CPU, is the
   definition of correct, and is the fallback path when CUDA, Triton or the required capability is missing.
3. **Launchers in `ops_*.py` dispatch**: Triton kernel if (CUDA and Triton importable and capability OK and
   shape supported and `FORGE_FORCE_REF` not set), otherwise the reference path. `FORGE_KERNEL_DEBUG=1` logs
   which path ran.
4. **The autotuner refuses to cache a candidate that fails the oracle** (KR-9).
5. Vendor GEMMs (cuBLAS FP8 via `torch._scaled_mm`, SDPA/flash) are used where they are already optimal; the
   Plan's "wrap vendor FP8 GEMM" (C-044) is followed literally.
6. Native fp4 block-scaled MMA (tcgen05) is not reachable from portable Triton today on all versions; it is
   used only when `has_fp4_block_scaled` is true and the installed Triton exposes it, otherwise ADR 0005 applies.

## Consequences

* Everything is unit-testable on CPU against the oracle; GPU tests (`tests_gpu/`) compare Triton against the
  same oracle over the Plan section 6.7 edge-case list.
* We accept that tuned Triton kernels may not reach the Plan KPI targets (85% bandwidth, 3.2x vs BF16). These
  are `target` provenance until measured. See `docs/KERNELS.md` validation table.
* Adding a CUDA/CUTLASS backend later is possible: the launcher dispatch is the only place that changes.
* **No Triton kernel has been executed on hardware at the time of this ADR.**
