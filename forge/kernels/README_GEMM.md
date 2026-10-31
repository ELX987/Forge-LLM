# Forge kernels: GEMM family (K1 / K2 / K3) -- design notes and validation status

Owner: `kernels_gemm`.  Plan references: chapter 6 (6.3 layout deriver, 6.4 K1-K3, 6.6 autotuning, 6.7 edge cases),
Appendix C.2 (kernel spec), phase-3 items C-040 ... C-044.  Paper references: P1 section 14 (Marlin), P5 sections 3.3 / 11
(NVFP4, scale layout, fused quantisers).

**Read the status table first.** Nothing in `forge/kernels/triton/gemm_*.py` has been compiled or run on an NVIDIA GPU by
the developer (the dev machine has none).  "Interpreter-checked" means the kernel *logic* was executed by the Triton CPU
interpreter (`TRITON_INTERPRET=1`) and compared with the torch oracle -- it proves indexing, masking, packing
conventions and scale handling, not that the kernel compiles for sm_80/89/90/100 or how fast it is.  The GPU gate is
`tests_gpu/test_kernels_gemm_*.py` (`scripts/run_gpu_tests.sh` on the vast.ai host).

## 1. Validation status

| Component | Where | CPU unit tests (`tests/`) | Triton CPU interpreter | Real GPU (`tests_gpu/`) |
|---|---|---|---|---|
| Kernel specs (YAML + pydantic) | `spec.py`, `specs/*.yaml` | yes | n/a | n/a |
| Layout deriver (6.3) | `layouts/layout.py` | yes (11 chip families) | n/a | n/a |
| W4 packer / unpacker / repacker, both layouts | `layouts/w4.py` | yes, property tests (473 cases) | n/a | pack on GPU == pack on CPU: written, not run |
| W8 packer, RTN quantisers | `layouts/w8.py`, `rtn.py` | yes | n/a | -- |
| FP4 containers, 128x4 scale swizzle | `layouts/fp4.py` | yes, vs independent view/permute reference | n/a | -- |
| Oracles (W4/W8/FP8/INT8/FP4 + compare()) | `ref/gemm_*.py` | yes, edge-case list of 6.7 | n/a | -- |
| Autotuner (gate, cache, throttle, IQR) | `autotune/` | yes, fake run_fn / clock | n/a | real-kernel test written, not run |
| Triton W4A16 / W8A16 GEMM (+split-K) | `triton/gemm_wq.py` | -- | **yes** (fp16; groups, zeros, g_idx, per-channel, odd N, K tail, split-K det./atomic) | written, **not run** |
| Triton INT8 W8A8 GEMM + act quantiser | `triton/gemm_int8.py` | -- | **yes**, bit exact | written, **not run** |
| Triton FP8 GEMM (3 scale modes) | `triton/gemm_fp8.py` | -- | **yes** (fp8 loads/dot work in the interpreter) | written, **not run** (needs sm_89+) |
| Triton fused RMSNorm+absmax->FP8 quantiser | `triton/gemm_fp8.py` | -- | scales yes; **fp8 codes not checked** (interpreter's fp32->fp8 cast is wrong) | written, **not run** |
| Triton FP4 dequant GEMM (NVFP4/MXFP4, logical+swizzled scales) | `triton/gemm_fp4.py` | -- | **yes** | written, **not run** |
| Triton fused FP4 quantiser | `triton/gemm_fp4.py` | -- | **yes, bit exact vs oracle** (incl. swizzled scales) | written, **not run** |
| Native block-scaled MMA (`tl.dot_scaled`, MXFP4) | `triton/gemm_fp4.py` | -- | yes where the interpreter supports it (Triton 3.9: packing convention verified) | **EXPERIMENTAL, not run**; API changes between Triton releases |
| bf16 activations through `tl.dot` | all | -- | **not checkable** (interpreter bf16 dot is broken) | written, **not run** |
| Dispatchers (`ops_gemm`), packed-linear modules | `ops_gemm.py` | yes (reference paths); triton path via interpreter | yes (`last_path == 'triton'`) | written, not run |
| `torch._scaled_mm` FP8 fast path | `ops_gemm.fp8_linear` | -- | -- | written, signature-guarded, **not run** |
| Bench harness / catalog | `bench.py`, `catalog.py` | yes (CPU smoke, DB, roofline math) | -- | written, not run |

Performance KPIs of the spec (>= 85% of HBM bandwidth and 3.2x vs BF16 at M<=16 for W4A16; 90% of vendor GEMM for FP8)
are **targets**, not measurements.  `tests_gpu/` prints the measured numbers and never fails on them; `bench.py` reports
them with `kpi` met/not-met flags and *measured* peak bandwidth / TFLOP/s of the device.

## 2. Layouts (Plan 6.3) -- what is real

`derive_layout(spec, chip, executor="triton"|"native", shape=...)`:

* picks the MMA atom from the chip (`m16n8k16` sm80+, `m16n8k8` sm75, Volta -> portable only), the vector width, a tile /
  stage / warp schedule that satisfies `stages * (A + B + scales) <= 0.9 * smem` and the accumulator register bound,
  the XOR swizzle for 16-byte vector access, the striped partition and split-K (`choose_split_k`, never below 2 K tiles
  per split).
* returns a `Layout` whose `id` is `<base>-<6 hex>` of the **storage-relevant** description only (kind, bits, storage tile,
  nibble interleave, scale orientation).  Schedule fields (`Layout.schedule`) are *not* in the id, so two chips that
  need identical bytes share artifacts (deviation from the Plan pseudo-code, which hashes the tile; it avoids needless
  re-packs).  Layouts are registered (`get_layout(id)`) so an artifact manifest's `layout_id` is enough to unpack.

Layouts shipped:

| id prefix | what | consumed by |
|---|---|---|
| `w4.portable_row.v1` | `qweight int32[N, K/8]` sequential nibbles; scales `[G, N]`; zeros `int32[G, N/8]`; optional `g_idx[K]` | the Triton kernels |
| `w4.mma_fragment_k16n64.v1` | Marlin-style: 16(K)x64(N) tiles, lane-major `m16n8k16` B-fragment order (32 nibbles = one 16-byte vector per lane), bit-interleaved `[0,2,4,6,1,3,5,7]` so `q & 0x000f000f` is a ready fp16x2 pair; `qweight int32[K/16, N*2]` | **no kernel yet** (packer + repacker + tests only); `executor="native"` |
| `w8.portable_row.v1` | `qweight int8[N,K]`, scales `[G,N]` | Triton W8A16 |
| `fp4.{nvfp4,mxfp4}.{logical,swz128x4}.v1` | payload `uint8[rows,K/2]`, E4M3/E8M0 scale bytes (logical `[rows,K/B]` or padded + swizzled 128x4 atoms), fp32 tensor scale | Triton FP4 kernels |

`repack(packed, new_layout)` is unpack -> pack of the *same logical* `(q, scales, zeros)` -- a pure permutation, property
tested to be bit exact in both directions.  The Triton path auto-repacks fragment-layout inputs to `portable_row` once
and caches the result on the object.  Honest gap: the fragment layout is produced for a future hand-tuned (CUTLASS/CUDA
class) kernel; the shipped Triton kernel is the portable baseline (C-042), not the tuned Marlin-class kernel (C-043).

## 3. Kernels

* **W4A16 / W8A16** (`gemm_wq.py`): in-kernel dequant `(q - zero) * scale` to the activation dtype, `tl.dot`, fp32 accumulate.
  Per-channel scales are applied once in the epilogue (they commute with the K sum); grouped scales (and GPTQ `g_idx`)
  before the MMA.  M-bucketed default configs (`default_wq_config`, BLOCK_M=16 for decode), split-K to fill the SMs with a
  fixed-order workspace reduction (`deterministic=True`, bit reproducible) or fp32 atomics.  bf16 is computed in fp16 on
  sm < 80.  Simple on purpose (gather-style loads of the packed words, no cp.async tricks): correct first.
* **FP8** (`gemm_fp8.py`): E4M3 `tl.dot` on fp8 operands, scale modes per-tensor / per-token x per-channel / 1x128 x
  128x128 block; fixed reduction order (reproducible).  Fused per-token RMSNorm + absmax -> FP8 quantiser (3 passes over
  the row, normalised activation never written in bf16), block and static-scale variants.  Fallback chain in
  `ops_gemm.fp8_linear`: Triton (sm89+) -> `torch._scaled_mm` -> dequantise + 16-bit matmul -> fp32 oracle.
* **INT8 W8A8** (`gemm_int8.py`): int8 `tl.dot` -> int32, exact; per-token dynamic quantiser with round-half-even.
* **FP4** (`gemm_fp4.py`): (1) weight-only dequant GEMM for any sm; (2) fused quantiser (RTN, saturating, optional
  swizzled scale output); (3) MXFP4 `tl.dot_scaled` native path, version- and capability-guarded.  NVFP4 (16 x E4M3 + fp32
  tensor scale) has **no** `tl.dot_scaled` mapping in released Triton, so NVFP4 runs through (1)/(2) -- documented gap.
  E4M3/E8M0 conversions use integer + bit-cast arithmetic (no fp8 dtype), so these kernels have no sm >= 89 requirement.
  Conventions: all-zero block -> codes 0; E4M3 scale underflow -> block flushed to 0 (never 0/0); NaN/Inf **poison their
  block** in the oracle (scale byte = NaN) -- the Triton quantiser assumes finite input and does not implement that.

## 4. Autotuner (Plan 6.6)

`Autotuner(cache_path).tune(spec, shape(s), run_fn, ref_fn, ...)`: analytic infeasibility pruning, optional cost-model
`rank_fn` (top-N + seeded random sample), warm-up, L2 flush (CUDA), median/IQR, telemetry before/after, throttle detection
(thermal/hw slowdown reasons, clock drop > 10%, last-third drift > 20%) with re-measurement, **correctness gate before any
caching** (float: spec tolerance, dtype-aware; integer outputs: bit exact), JSON cache keyed by (chip fingerprint,
driver/cuda/triton versions, op, dtype config, M-bucket + exact N/K, layout id), regression check (> 5% slower median).
`ops_gemm` consults the cache (`tuned_config`, 5 s memo) and otherwise uses the heuristic; it never tunes implicitly --
tune through the `bench` job (`autotune: true`) or the API.

## 5. Tolerances

`ref.compare()` (spec `tolerance`): `max|out-ref| <= abs + max(rel, 2*eps_out) * max|ref|` **and**
`<= abs + rel_norm * max|ref|`; non-finite entries must match positionally; integer paths bit exact.  The Plan's
`rel 2e-3` cannot hold for bf16 *outputs* (bf16 eps is 7.8e-3), hence the dtype-aware floor.

## 6. Environment variables

`FORGE_FORCE_REF=1` (always use oracles/fallbacks), `FORGE_KERNEL_DEBUG=1` (log chosen path), `FORGE_KERNEL_STRICT=1`
(kernel failure raises instead of falling back), `FORGE_AUTOTUNE_USE=0` (ignore the tuned-config cache).

## 7. Known gaps / not done

* No tuned (cp.async ring / persistent / TMA) W4A16 kernel; the Triton kernel is the portable baseline.
* No Hadamard / stochastic-rounding GEMM-epilogue fusion for FP4 *training* (K8 and training extras belong to the
  utility kernels); only RTN quantisation is fused.  Row- and column-wise (TN) FP4 operand copies are not produced here.
* No W3/W6 bit-plane layouts, LUT/vector-quantised kernels, 2:4-sparse x quantised combos, W4A8.
* `tl.dot_scaled` kernel is unvalidated and may need edits for your Triton version.
* FP8 `e5m2` operands use the dequantise fallback (no Triton path).
* The interpreter cannot validate bf16 `tl.dot` or the fp32->fp8 cast; those are covered only by `tests_gpu/`.

## 8. Independent review changes (kernels-gemm reviewer)

Found and fixed by reading the code, emulating index math and compiling every GEMM kernel offline with `triton.compile`
for sm_80/89/90/120 (still **not** run on a GPU):

* Split-K reduce kernel did not compile for M == 1 (Triton turns integer args equal to 1 into `constexpr`; `M.to(...)`
  failed) -> every decode-shape W4A16/W8A16 call silently took the slow reference path.  `total = M*N` is now a host arg.
* Default W4 schedules needed up to 148 KiB shared memory (asymmetric, 3 stages) and would fail on sm_86/89/120 (~99 KiB).
  `launch_fitting` retries with fewer stages / smaller `tile_k` and memoises the schedule; consumer sm default to 2 stages.
* FP8 GEMM read scales out of bounds for mixed granularities (per-token x per-tensor weights, per-tensor x per-channel,
  block x per-channel ...).  `prepare_fp8_scales` normalises every combination with explicit (possibly zero) strides and
  validates sizes.  On sm_90 fp8 `tl.dot` now promotes the accumulator every <=128 K elements (`max_num_imprecise_acc`).
* INT8 launcher used the strides of a non-contiguous weight for its contiguous copy; FP4 dequant GEMM did not decode
  NaN-poisoned scale blocks and did not validate container shapes; the fused FP4 quantiser silently turned NaN/Inf into
  numbers (now poisons the block like the oracle) and used a non-finite tensor amax.
* `tl.dot_scaled` MXFP4 path is now gated by a one-time on-device numerical self-test (falls back if it fails).
* Triton failures are memoised per (op, dtype, shape bucket) instead of re-compiling on every call; the repack cache of
  fragment-layout W4 modules is stamped with tensor identity/version (was rebuilt every forward, then unsafe to cache).
* `load_spec` rejects path-like names; `pack_w4` rejects negative zero-points.
