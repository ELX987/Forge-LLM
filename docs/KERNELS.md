# Kernels

Forge's kernel layer follows Plan section 6 ("Kernel catalogue", K1-K9) and Appendix C.2 (kernel spec).
The governing decisions are in `docs/adr/0002-triton-first-kernels-with-torch-oracles.md` and
`docs/adr/0005-fake-quant-fp4-on-non-blackwell.md`.

> **Honesty statement.** At the time of writing, *no Triton kernel in this repository has been executed on an
> NVIDIA GPU.* The development machine has none. The pure-torch oracles and the packing/layout code run and are
> unit-tested on CPU; the Triton code is written to be simple and standard but is *unvalidated*. Plan KPIs
> (e.g. 85% of HBM bandwidth, 3.2x vs BF16 for W4A16) are **targets**, not results. See section 6.

## 1. How an op executes

```
ops_gemm.w4a16_linear(x, packed)           (forge/kernels/ops_*.py)
   |-- capability gate: CUDA available? Triton importable? sm >= min_sm? shape supported?
   |        and env FORGE_FORCE_REF != 1
   |-- yes -> Triton kernel (forge/kernels/triton/*), tile config from the autotune cache or spec defaults
   `-- no  -> reference path (forge/kernels/ref/*): dequantise + torch matmul; always available, runs on CPU
```

* `FORGE_FORCE_REF=1` forces the reference path everywhere (use it to separate "kernel bug" from "environment bug").
* `FORGE_KERNEL_DEBUG=1` logs, per call, which path ran and why.
* The reference path is the **definition of correct**: tests compare Triton output to it, and the autotuner refuses
  to cache any candidate whose output fails that comparison.

## 2. Catalogue (Plan section 6.4)

| ID | Kernel family | Ops (launcher names) | Oracle | Regime | Min capability | Notes |
|----|---------------|----------------------|--------|--------|----------------|-------|
| K1 | Weight-only mixed-precision GEMM W4A16 / W8A16 | `w4a16_linear`, `w8a16_linear` | `w4a16_gemm_ref`, `w8a16_gemm_ref` | decode, bandwidth-bound, M 1-128 | sm_70 (fp16 activations), bf16 on sm_80+ | Layout-dependent: weights are repacked offline (`layout_id`). Per-column scales may be applied once in the epilogue. |
| K2 | Block-scaled FP4 GEMM (NVFP4 / MXFP4) + fused quantiser | `fp4_linear` | `fp4_block_gemm_ref` | compute-bound prefill/training | native: sm_100/103 (and sm_120/121 if the probe confirms); elsewhere dequant-to-higher-precision (ADR 0005) | NVFP4: 16 x E2M1 + E4M3 scale + FP32 tensor scale. Scale tensor is padded to 128x4 and swizzled; artifact stores payload / scales / tensor-scale separately. |
| K3 | FP8 GEMM (per-tensor / per-block), INT8 W8A8 | `fp8_linear`, `quantize_act_fp8` | `fp8_gemm_ref` | prefill and batched decode | FP8: sm_89+; INT8: sm_75+ | Wraps the vendor FP8 GEMM where possible (C-044). Scaling granularity is a parameter. |
| K4 | Decode attention: paged, GQA-grouped, split-KV, quantised KV, MLA | `paged_decode_attention`, `prefill_attention`, `mla_decode_attention` | `paged_decode_attention_ref` | decode, bandwidth-bound | sm_70+ | All query heads of a GQA group share one KV read. KV dtypes: bf16/fp16/fp8_e4m3/int8/int4. Prefill falls back to torch SDPA. |
| K5 | Sparse attention: indexer, top-k, gather-attention | `sparse_topk_attention` | `sparse_attention_ref` | long context | sm_70+ | Experimental (single-source evidence, Plan C-066). |
| K6 | Speculative-decoding kernels | `tree_attention`, `rejection_verify`, `kv_compact` | `tree_attention_ref`, rejection oracle | decode | sm_70+ | Verification must be distribution-exact; tested statistically. |
| K7 | MoE kernels | `moe_route`, `moe_permute`/`unpermute`, `grouped_gemm` | `moe_route_ref`, `moe_grouped_gemm_ref` | decode + prefill | sm_70+ | Variable per-expert M including 0. |
| K8 | Utility kernels | `hadamard_transform`, `quantize_fp8_rowwise`, `stochastic_round`, `rmsnorm_quant`, `rope_and_kv_write`, `topk_select` | `hadamard_ref`, `stochastic_round_ref`, `rmsnorm_quant_ref`, `rope_kv_write_ref` | memory-bound | sm_70+ | Counter-based RNG for reproducible stochastic rounding. |
| K9 | Low-rank / multi-adapter | LoRA / low-rank linear paths in `forge.models` | torch reference | decode | sm_70+ | Segmented-batched adapter kernel (SV-7) is a roadmap item; check `docs/ROADMAP_STATUS.md`. |

Run `GET /api/v1/kernels` (or the *Kernel lab* page, or `forge.kernels.catalog.list_kernels()`) to see which of
these are *available on the current machine* and why not (`reasons`).

## 3. `layout_id`: layouts are part of the artifact

A quantised weight tensor is *numerics* (the integer codes and scales) plus a *layout* (how bits are permuted and
interleaved so a given chip's MMA fragments are loaded with coalesced 16-byte reads). Forge keeps them separable
(Plan section 6.3):

* `derive_layout(spec, chip) -> Layout(id, perm, swizzle, scales, partition, tile defaults)` decides the layout.
* `pack_w4(...) -> PackedW4{qweight:int32, scales, zeros, layout_id, meta}` packs; `unpack_w4` is its exact inverse
  (property-tested: pack then unpack is identity for random shapes, including K not a multiple of the group size).
* The `layout_id` is stored per tensor in the artifact manifest (`TensorEntry.layout_id`).
* A chip that cannot run that layout efficiently triggers a **bit-exact offline repack** (`repack(packed, new_layout)`)
  - a permutation, not a re-quantisation - so the same artifact moves between GPU generations without changing numerics.
* NVFP4 scale tensors use the same idea (`pack_nvfp4_scales` / unpack, 128x4 swizzle).

## 4. GPU support matrix

From `docs/ARCHITECTURE.md` section 3. "Native" entries are hypotheses until `forge probe` confirms them
(Plan HW-2). Unknown or older chips fall back to the portable tier C.

| sm | Examples | bf16 | fp8 mma | fp4 block-scaled mma | Notes |
|----|----------|------|---------|----------------------|-------|
| 70 | V100 | emulated | no | no | fp16 only; Triton OK |
| 75 | T4, RTX 20xx | emulated | no | no | fp16 only |
| 80 | A100, A30 | yes | no | no | cp.async, 2:4 sparsity |
| 86 | RTX 30xx, A10, A40, RTX A-series | yes | no | no | |
| 87 | Jetson Orin | yes | no | no | |
| 89 | RTX 40xx, L4, L40S, RTX 6000 Ada | yes | yes | no | |
| 90 | H100, H200, GH200 | yes | yes | no | TMA, wgmma |
| 100 | B200, GB200 | yes | yes | yes (NVFP4/MXFP4) | tcgen05, tensor memory |
| 103 | B300 | yes | yes | yes | |
| 120 | RTX 50xx, RTX PRO 6000 Blackwell | yes | yes | yes (verify by probe) | consumer Blackwell |
| 121 | GB10 (DGX Spark) | yes | yes | yes (verify by probe) | |

Tiers (`ChipDescriptor.tier`): **A** = fully probed with a tuned schedule available; **B** = known family, portable
schedule with autotuned tiles; **C** = unknown/old/CPU, reference or portable fallback only.

## 5. Autotuning protocol

Implemented in `forge/kernels/autotune` (Plan KR-9, C-043). The protocol, in order:

1. **Cache key**: (chip fingerprint, driver version, Triton version, op, dtypes, shape bucket, `layout_id`).
   A hit returns the stored best config; any key component change is a miss.
2. **Enumerate candidates** from the kernel spec `search:` block (`tile_m/n/k`, `stages`, `split_k`, warps),
   pruned by shared-memory / register budgets from the chip descriptor.
3. **Correctness first**: each candidate runs against the oracle on the shape set (edge cases of Plan section 6.7).
   A candidate that fails is discarded and **never cached**.
4. **Timing**: warm-up runs, L2 flush between repeats, N timed repeats; record the **median and IQR**, not the minimum.
5. **Throttle detection**: if clocks/power drop between repeats, results are flagged and re-run.
6. **Select** the best by median; ties within IQR prefer the simpler (fewer stages / no split-K) config.
7. **Persist** to `$FORGE_HOME/autotune/*.json` with provenance `measured`. Results are visible under
   `GET /api/v1/kernels/autotune-cache` and in the Kernel lab heatmap.

Run it as a job: `bench` job type with `configs/bench/bench-gemm-sweep.yaml` (UI: *New Job -> bench*), or the quick
sweep in `scripts/gpu_smoke.sh`.

## 6. Validation status (HONEST)

Three levels, in increasing strength:

1. **CPU-tested**: oracle, packers/layouts, dispatchers' reference paths run in the normal `tests/` suite.
2. **Interpreter-checked**: the Triton kernel *logic* was executed by the Triton CPU interpreter (`TRITON_INTERPRET=1`,
   run in subprocesses by `tests/test_kernels_*_interp*.py`, skipped when `triton` is not installed) and compared with
   the oracle. This proves indexing, masking, packing conventions and scale handling. It does **not** prove that the
   kernel compiles for `sm_80/89/90/100`, uses the tensor cores, or is fast; bf16 `tl.dot` and the fp32->fp8 cast are
   not even checkable in the interpreter.
3. **GPU-validated**: a `tests_gpu/` run on real hardware passed and its `gpu_report/report.md` is archived.
   **No kernel has reached level 3.** No kernel has been timed on hardware, so every performance number is a target.

| Kernel | CPU-tested (oracle, layouts, fallback) | Interpreter-checked | GPU-validated | Perf-validated | Hardware tried |
|--------|----------------------------------------|---------------------|---------------|----------------|----------------|
| K1 W8A16 and the legacy W4A16 kernel (`triton/gemm_wq.py`) | yes | yes (fp16) | compiled + ran on sm_120 (correct; **0.23x-0.40x of bf16**, see section 10) | **no, slow** | RTX PRO 6000 Blackwell (sm_120) |
| K1 W4A16 nibble-split kernels (`triton/gemm_w4_fast.py`, `impl=dot` / `impl=gemv`) | yes (config policy, act-order sort) | yes (fp16 dot, fp16/bf16/fp32 gemv; all group sizes, zeros, act-order, odd N/K, split-K) | **no** (offline-compiled for sm_80/89/90/120 only; `tests_gpu/test_kernels_w4a16_fast.py` not run) | **no** (`scripts/bench_w4a16.py` not run) | none |
| K3 INT8 W8A8 + activation quantiser (`gemm_int8.py`) | yes | yes, bit exact | **no** (sm_120 run: quantiser mismatched the oracle; root cause 1: Triton approximate `/` (fixed with `div_rn`); root cause 2: the torch oracle's `amax / 127.0` is `amax * (1/127)` on CUDA (CPU-scalar divisor), 1 ulp off IEEE; oracle now divides by a tensor. GEMM exactness and quantiser agreement are separate tests; awaiting re-run) | **no** | RTX PRO 6000 Blackwell (sm_120) |
| K3 FP8 GEMM (`gemm_fp8.py`, `torch._scaled_mm` fast path) | yes | yes; fp8 codes of the fused RMSNorm quantiser not checkable | **no** | **no** | none |
| K2 NVFP4 / MXFP4 dequant GEMM, fused quantiser (`gemm_fp4.py`) | yes | yes (quantiser bit exact vs oracle) | **no** | **no** | none |
| K2 native block-scaled MMA (`tl.dot_scaled`) | oracle | partly | **no (experimental, API changes between Triton releases)** | **no** | none |
| K4 paged / split-KV / quantised-KV decode attention, prefill (`attn_decode.py`, `attn_prefill.py`) | yes | yes | **no** | **no** | none |
| K4 MLA decode (`attn_mla.py`) | yes | see `tests/test_kernels_attn_triton_interp.py` | **no** | **no** | none |
| K5 sparse attention (`attn_sparse.py`) | yes | see interp suite | **no** | **no** | none |
| K6 tree attention / speculation kernels (`attn_spec.py`) | yes (+ statistical exactness tests in `tests/test_serve_spec.py`) | see interp suite | **no** | **no** | none |
| K7 MoE routing / permute / grouped GEMM (`moe_kernels.py`) | yes | see interp suite | **no** | **no** | none |
| K8 utility (Hadamard, row-wise FP8 quant, RoPE+KV write, GEMV) | yes | see interp suite | **no** | **no** | none |
| K9 low-rank / LoRA | torch reference (`forge.models`) | n/a | **no** | **no** | none |
| Autotuner (`forge/kernels/autotune`) | yes (fake run function and clock) | n/a | real-kernel test written, **not run** | n/a | none |
| Bench harness (`forge/kernels/bench.py`) | yes (CPU smoke, DB, roofline arithmetic) | n/a | **not run** | n/a | none |

"see interp suite" means the kernel module exists and is exercised by `tests/interp_attn_cases.py` /
`tests/interp_moe_util_cases.py`; read those files for the exact cases, because the owner's tests are the source of
truth. Per-component detail for the GEMM family is in `forge/kernels/README_GEMM.md`.

Other honest limits: `FORGE_FORCE_REF=1` and the reference paths are the only fully trusted paths today; CUDA graphs
(`serve.cuda_graphs`), the FP8 `torch._scaled_mm` path, bf16 tensor-core dots and native FP4 have never executed.
The Plan KPIs (85% of HBM bandwidth, 3.2x vs BF16 at M<=16, 90% of vendor FP8 GEMM, 2.2x FP4 vs FP8) are **targets**.

**Update rule:** change a "GPU-validated" cell to `yes (<chip>, <date>, torch X / triton Y)` only when you can attach the
`gpu_report/report.md` showing that test file passing on that chip. Perf cells additionally need the bench output.

Not implemented in this release (Plan items outside scope): CUDA/CUTLASS backends, FP4 *training* kernels beyond
fake-quant, LUT/vector-quantised codebook kernels, entropy-decode-fused GEMM, expert-parallel quantised all-to-all,
disaggregated prefill/decode, segmented-batched adapter kernel for fleets.

## 7. Running GPU tests

On a GPU host (see `docs/VAST_AI.md`):

```bash
bash scripts/run_gpu_tests.sh                       # everything, writes ./gpu_report/report.md
bash scripts/run_gpu_tests.sh -k "w4a16 or fp8" -x  # pytest args pass through
python -m pytest tests_gpu -v -rs                   # directly; -rs prints why tests were skipped
FORGE_FORCE_REF=1 python -m pytest tests_gpu -q     # the reference path only (should pass anywhere)
```

Tests in `tests_gpu/` are auto-skipped when CUDA is unavailable (`conftest.py`). Capability-gated tests skip with a
reason on chips that lack the feature (e.g. FP8 on `sm_86`); a skip is not a pass.

## 7b. W4A16 decode GEMM (K1): measured problem, diagnosis, redesign

**What was measured** (RTX PRO 6000 Blackwell Max-Q, sm_120, Triton 3.8, `gpu_report` bench, bf16, N=K=4096, group 128): the
legacy kernel took 88.6 us at M=1 and 86.0 us at M=16 (identical, i.e. not scaling with work), 199 us at M=256, achieving
~98 GB/s = 6.7% of the measured copy bandwidth; `path: triton` (not the reference fallback), results correct. The
"speed-up" column is `baseline_ms / kernel_ms` with `baseline = F.linear(x, p.dequantize(bf16))` (a fair bf16 baseline:
dense bf16 weight, cuBLAS), both timed **eagerly** (one call between two CUDA events, L2 flushed). That is 0.34x / 0.40x /
0.23x at M = 1 / 16 / 256.

**Diagnosis** (static, from the code and from compiling the kernels offline with Triton for sm_120 and reading the PTX):

1. *Scalar, redundant weight loads.* The legacy kernel loads, for every `(k, n)` element of the tile, the int32 word that
   contains it (`offs_k // 8` addresses; 8 identical 4-byte loads per word, 2.5 KiB stride between channels). Triton cannot
   vectorise that pattern (constancy 8, no contiguity). In the sm_120 PTX of the legacy kernel: **384 four-byte
   `cp.async.ca` weight loads per tile body vs 6 sixteen-byte `cp.async.cg`** in the new kernel (asserted by
   `tests/test_kernels_w4_fast_compile.py`).
2. *Per-element scale and zero-point loads* inside the K loop (`s_ptr + g[:, None] * stride + offs_n[None, :]`, plus a zero
   word load and shift per element), and a scale multiply on every weight before the MMA. For g128 and `tile_k=128` the
   scale/zero are constant over the tile: one `[BLOCK_N]` vector load is enough, and the scale can multiply the
   `[BM, BN]` partial sum instead of the `[BN, BK]` weights.
3. *act-order (`g_idx`)*: an external GPTQ checkpoint quantised with `desc_act` carries a `g_idx`. The legacy kernel then
   also gathers `g_idx[k]` and a scale/zero **per element**: the worst case of (2). Artifacts produced by Forge's own
   recipe `w4a16-gptq-g128` (`act_order: true`) do **not** carry a `g_idx`: Forge's GPTQ uses static groups (group
   parameters computed up-front, so the stored layout is the plain `k // group_size` rule). The `g_idx` / group-sorted
   path is for externally produced GPTQ weights.
4. *Integer to float*: shift, mask, `int32 -> f32`, subtract, multiply, `f32 -> bf16` per weight element. Not the
   bottleneck at 4-bit (estimate: a few us of ALU per 4096x4096 layer vs ~6 us of DRAM time) once the loads are fixed.
5. *Split-K at decode*: legacy `default_wq_config` only splits to fill the SMs with a 64-wide N tile; reasonable, but the
   fp32 workspace + second reduce kernel are two extra launches, which matters for the next point.
6. *Host overhead and how the number is measured.* An eager Triton launch costs 15-50 us of host time (argument
   processing, `torch.empty` for the workspace, the split-K reduce launch, the Python dispatcher in `ops_gemm`). A 4096x4096
   W4 decode GEMM needs ~7 us of device time at the measured bandwidth, so **an eager benchmark cannot show more than
   ~0.5-1x no matter how good the kernel is**: bf16 `F.linear` is one cuBLAS launch. Real decode loops hide launch cost
   with CUDA graphs; `scripts/bench_w4a16.py` therefore times CUDA-graph replays over a ring of weight copies larger than L2
   (and also prints the eager numbers). The built-in `bench` job (`forge/kernels/bench.py`) is still eager and keeps
   under-reporting the speed-up until it adopts the same method.
7. *Prefill (M = 256)*: with in-kernel dequant every M tile re-dequantises its weights, so large M is compute/ALU bound;
   dequantising once (cached bf16 weight + cuBLAS) wins there. Not changed here; the decode regime is what 4-bit buys.

**Redesign** (`forge/kernels/triton/gemm_w4_fast.py`, policy in `forge/kernels/w4_tuning.py`, launcher/fallback in
`triton/gemm_w4a16.py`):

* *Nibble-split loads.* The packed words are loaded as a `[KW, BLOCK_N]` (dot) or `[BLOCK_N, KW]` (gemv) tile, `KW = BLOCK_K/8`,
  contiguous along the word axis: one 16-byte vector per thread, each word read once. Slice `j` of the tile is
  `(word >> 4j) & 15` and pairs with the activations `x[:, k0 + 8c + j]` (stride-8 gather of the tiny activation tile,
  L1-resident), so the K sum is eight independent products and no nibble interleaving is needed (the portable_row layout is
  unchanged; no repack).
* *Hoisted scale / zero.* `SINGLE_GROUP` (g128 with `tile_k <= 128`, or per-channel): one `[BLOCK_N]` scale and zero vector per K
  tile; the MMA multiplies the exact integers `q - z` (|.| <= 15) and the scale is applied once to the partial sum, so the
  result is *at least as accurate* as before (the old kernel rounded `(q - z) * s` to 16 bit). Multi-group tiles (g32/g64, or
  `tile_k > group`) gather one `[KW, BN]` scale tile per K tile, hoisted out of the 8-slice loop.
* *`EVEN_K` / `EVEN_N` specialisation.* The offline PTX showed that a K-axis mask on the word loads (`wcol < nwords`) kills
  vectorisation unless the compiler can prove the divisibility; the host passes constexpr `EVEN_K` / `EVEN_N` (`K % BLOCK_K`,
  `N % BLOCK_N`) and the kernels use unmasked loads for full tiles. All Qwen3-14B shapes are even for the default tiles.
* *act-order.* `sort_columns_by_group` (layouts/w4.py) permutes the K columns once at first use so that every group is a
  contiguous run (`x @ W^T == x[:, perm] @ W_sorted^T`), cached in `packed.meta`; the kernel gathers `x` through `perm`.
  Irregular `g_idx` (unequal groups) falls back to the legacy per-column kernel.
* *Two implementations.* `impl=dot`: tensor cores (`mma.sync m16n8k16`, BLOCK_M >= 16, works for every M).
  `impl=gemv` (M <= 8): fp32 FMA, all M rows in one program, the K reduction deferred to a single `tl.sum` after the loop; no
  tensor cores and no 16-row padding. Which is faster per chip is a measurement, not a derivation: both are autotune candidates.
* *Split-K for decode* targets `num_sms * ctas_per_sm` programs (chip profile table in `w4_tuning.py`; 2-3 CTAs/SM) and never
  gives a slice fewer than two K tiles. The deterministic fixed-order reduction is unchanged.
* *Chip profiles* (`w4_tuning.chip_family`): `pre_ampere` (sm<80, 2 stages), `ampere_dc` (sm_80/87: 163 KiB smem, 4 stages),
  `ampere_consumer` (sm_86), `ada` (sm_89: ~99 KiB, 3 stages), `hopper` (sm_90: 227 KiB, 4 stages), `blackwell_dc` (sm_100/103),
  `blackwell_consumer` (sm_120/121: ~99 KiB, GDDR7, `mma.sync`, 3 stages). These numbers are first-principles defaults, not
  measurements.
* *Autotune keying.* The cache key still buckets M to powers of two; the dispatcher (`ops_gemm.tuned_config`) now also borrows
  the nearest tuned bucket of the same regime (decode <= 16 / prefill >= 32) and, for decode, the closest tuned (N, K) of the same
  chip (split-K recomputed). Configs tuned for the legacy kernel or without the `dequant_matmul` candidate are ignored (`dtype_cfg` carries the tag `k4|ao=<act-order>|sym=<symmetric>`).
* *Fallbacks.* `config={'impl': 'legacy'}` or `FORGE_W4_IMPL=legacy` (also `dot` / `gemv` / `dequant_matmul`) selects a kernel explicitly. If the
  nibble-split kernel raises and the caller did not pin an `impl`, the launcher logs once and uses the legacy Triton kernel
  for that (sm, dtype) instead of dropping to the torch reference. Explicit candidates (the autotuner, tests) never fall back.
  The pure-torch oracle `w4a16_gemm_ref` and all tolerances (`w4a16_gemm.yaml`) are unchanged.

### 7b.1 After the redesign: sm_120 measurements (GPU report 6) and the large-M path

**Measured** (same RTX PRO 6000 Blackwell Max-Q, sm_120, 300 W cap, torch 2.14, Triton 3.8.0). GPU suite on that host: 894 passed,
1 failed (the interpreter had no `zstandard`: an environment problem, not a kernel one), 46 skipped, so `dot` and `gemv` are
numerically validated on sm_120. Quick bench W4A16 4096x4096 (default heuristic, vs bf16 `F.linear`): M=1 1.56x, M=16 1.70x
(legacy was 0.35x / 0.40x), **M=256 0.31x**. (Probably flattering the W4 side: an 8 MiB W4 weight fits the 96 MiB L2 unless the bench flushes it; the
sweep below uses the L2-defeating ring.) `scripts/bench_w4a16.py --act-order` on Qwen3-14B shapes, graph timing, microseconds
(bf16 / legacy / default / tuned) and tuned speed-up vs bf16:

| shape (N x K) | M | bf16 | legacy | dflt | tuned | tuned vs bf16 |
|---|---|---|---|---|---|---|
| down_proj 5120 x 17408 | 1 | 113.8 | 167.8 | 72.7 | 52.6 | 2.16x (883 GB/s, 60% of peak) |
| | 4 | 112.3 | 167.9 | 96.9 | 63.3 | 1.78x |
| | 8 | 112.7 | 170 | 147.8 | 66.5 | 1.69x |
| | 16 | 115.6 | 173.5 | 254.4 | 89.5 | 1.29x |
| | 32 | 124 | 433 | 444 | 161 | 0.77x |
| | 64 | 140.6 | 451 | 889 | 431 | 0.33x |
| | 128 | 135 | 943 | 2154 | 850 | 0.16x |
| | 256 | 241 | 968 | 2957 | 1567 | 0.15x |
| | 1024 | 967 | 3726 | 12161 | 6240 | 0.15x |
| lm_head 151936 x 5120 | 1 | 1107 | 1283 | 500 | 446 | 2.48x |
| | 16 | 1069 | 1300 | 2305 | 818 | 1.31x |
| | 32 | 1630 | 3470 | 3861 | 1320 | 1.23x |
| | 64 | 1592 | 3653 | 7312 | 3713 | 0.43x |
| | 256 | 1960 | 8222 | 26087 | 13966 | 0.14x |

**Reading the table (reasoned from these numbers and from the offline PTX, not re-measured).**

* Decode (M <= 8) is solved: 1.7-2.5x, and the tuned time at M=1 is already 60% of the copy bandwidth.
* The default (`dflt`, the `dot` kernel) grows almost linearly with M at the *same* tile (72.7 / 96.9 / 147.8 / 254.4 us at
  M = 1 / 4 / 8 / 16): the tile does not change, only the number of valid activation rows does. That is the cost of the
  stride-8 activation gather, one scalar 16-bit load per element per nibble slice. Offline PTX for sm_120 (bf16, tile_k 128, 4
  warps; `tests/test_kernels_w4_dequant_compile.py`): the `dot` kernel issues 16-18 scalar `ld.global.b16` per thread per K tile
  for tile_n = 64 / 128 / 256 while the `mma.sync` count is 16 / 32 / 64, i.e. a wider N tile amortises the gather over
  proportionally more tensor-core work. The default for 9 <= M < 32 therefore moved from tile_n 64 to 128 (tile_n 256 stays an
  autotune candidate: it under-fills the SMs for the 1024-wide k/v projections). Not measured.
* For M >= 32 the fused kernels cannot win: even the tuned `dot` is 0.15x of bf16 at M >= 128, and the default is *worse than the
  legacy kernel* from M = 16 (254 vs 173 us) on down_proj. Every M tile re-dequantises the same weights (docs 7b item 7).

**Routing (`w4_tuning.default_w4_config`, per chip in `_PROFILES[...]["dequant_min_m"]`, env `FORGE_W4_DEQUANT_MIN_M`):**

| M | impl | notes |
|---|---|---|
| 1-8 | `gemv` | fp32 FMA, `tile_m = next_pow2(M)`, `tile_n` 32 (M <= 4) / 16, `tile_k` 128, split-K to fill the SMs (sm_120: gemv only at M = 1, measured tiles; `dot` t16x128x128 at M 2-4, t16x256x128 at M 5-16) |
| 9-63 | `dot` | tensor cores, `tile_n` 128 (sm_120: t16x256x128 up to M = 16, t32x256x128 at M 17-32) |
| >= 64 | `dequant_matmul` | Triton dequantisation to bf16/fp16 + cuBLAS (below); the threshold is per chip and overridable (`<= 0` disables) |

The autotune cache still wins over the heuristic per (chip, M bucket, N, K), so the *effective* threshold is moved by the sweep: every
bucket >= 8 now has `dequant_matmul` (and, from M = 9, the legacy kernel) among its candidates and the measured winner is cached.
`bench_w4a16.py` prints `suggested_dequant_min_m` (smallest M from which `dequant_matmul` wins on every swept shape). The cache tag is
now `k4` (act-order and symmetry are part of the key): entries from earlier sweeps (`k2`/`k3`) are ignored; re-run `--quick`.
The default threshold is 64 (32 only on a chip whose measured dot speed is below the dequant cost model at M = 32; the sm_120 sweep measured dot at x0.74-0.80 of bf16 at M = 32, x0.33-0.44 at M = 64).
Explicitly configured kernels are untouched: `config={'impl': ...}` and a hand-written config without `impl` still mean the `dot`
kernel, `FORGE_W4_IMPL=legacy|dot|gemv|dequant_matmul` forces one everywhere.

**`dequant_matmul`** (`triton/gemm_w4_dequant.py`). A memory-bound kernel expands the packed weight to a dense bf16/fp16 tile and
cuBLAS (`F.linear`) does the GEMM, so the cost is `W4 read + dense write + cuBLAS(M, N, K)` and no longer grows with the number of M tiles.

* One program = a `[BLOCK_N, BLOCK_K]` tile of the output weight (default 32 x 128). It loads `[BLOCK_N, KW]` packed words (KW = BLOCK_K/8,
  contiguous along the word axis), expands the 8 nibbles along a new innermost axis `[BLOCK_N, KW, 8]` and stores
  `out[n, 8c + j]`: the flattened `(c, j)` axis *is* the contiguous K axis, so stores are coalesced. Compiled offline for sm_80/89/90/120: one
  `ld.global.v4.b32` (16 bytes of packed words) and `st.global.v4.b32` (8 output elements) per thread, no scalar 16-bit stores.
* `(q - zero) * scale` in fp32, rounded once to the output dtype (what the legacy kernel did per element). Symmetric weights use zero 8.
  Scale access: one `[BLOCK_N]` vector per tile when the tile lies in one group (g128), a `[BLOCK_N, KW]` gather per word for g32/g64, and
  a per-element gather for act-order. K not a multiple of 8 and N not a multiple of the tile are masked (pad nibbles are never stored).
* **Act-order**: the original-order columns are dequantised straight from `g_idx` (per-element scale/zero gather). No column sort, no
  activation permutation, irregular groups work, and nothing needs to be prepared.
* **Memory.** The dequantised weight is a *temporary* of `rows x K x itemsize`. It is produced in row chunks bounded by
  `FORGE_W4_DEQUANT_MB` (or `config['chunk_mb']`; default **1024 MiB**; `w4_tuning.plan_dequant_chunks`, chunk starts multiples of 64 rows), one
  buffer reused by all chunks. The 151936 x 5120 lm_head (1.45 GiB in bf16) is therefore done in 2 chunks and peaks at ~1 GiB + the output;
  a 5120 x 17408 down_proj (170 MiB) is one chunk. Budget this next to the KV cache when sizing `gpu_memory_fraction`. The buffer comes from
  the torch caching allocator (also what CUDA graph capture uses), so the path is graph-capturable; with several layers captured in one graph the
  graph's private pool holds one such buffer per concurrently-live chunk (not per layer).
* **Expected cost (a model, to be confirmed by `bench_w4a16.py --quick`):** down_proj = 44.6 MB read + 178 MB written at 0.9-1.4 TB/s = ~160-250 us for
  the dequantisation plus the bf16 cuBLAS time of the table (135-241 us at M = 128-256, ~1 ms at M = 1024), i.e. roughly 3x faster than the legacy
  kernel and 2.5-5x faster than the tuned `dot` at M >= 128 (850 / 1567 / 6240 us vs a modelled ~335 / ~440 / ~1170 us), but still slower than bf16 `F.linear` on a pre-dequantised weight (that is the price of keeping
  the weight 4-bit: 0.15-0.2x becomes ~0.4-0.5x at M = 128-256 and approaches 0.8-0.9x at M = 1024). It does not beat the tuned `dot` at M = 32
  (161 us vs a modelled ~320 us), which is why the sweep, not the default, decides that bucket (the default there is a floor: it is not slower than legacy or the old default).

**Launch path is host-sync free (CUDA graph capture with act-order).** The `--act-order` sweep logged `CUDA graph capture failed ... Cannot copy between
CPU and CUDA tensors` and fell back to eager timing: `sort_columns_by_group` (a `.cpu()` and `int(tensor)` calls) ran lazily on the first launch of every
ring copy. Now `layouts.cached_group_sort` is the only launch-path entry (a stamped cache lookup, no sync); the sort itself runs at prepare time:
`ops_gemm.prepare_w4` / `KernelW4A16Linear.prepare()` (called by `from_packed`, after `load_state_dict` and after `.to()` / `.half()`), and by the bench before
capture. A cold cache while a CUDA graph is being captured uses the legacy `g_idx` kernel (sync-free) and logs once instead of failing. The
`dequant_matmul` path never needs the sort. Guarded by `tests/test_kernels_w4_dequant_policy.py` (AST scan of every launch function for
`.cpu/.item/.tolist/.numpy/synchronize`, plus a run with those methods monkeypatched to raise) and, on the GPU, by
`tests_gpu/test_kernels_w4_dequant_gpu.py::test_actorder_cuda_graph_capture_without_host_sync`.

**Sweep cost.** The full sweep takes hours on a 300 W card because every candidate is timed to completion, a throttled candidate is timed twice and a
throttled winner was never cached. `scripts/bench_w4a16.py --quick` runs M in {1, 8, 16, 64, 256, 1024} on q/o (5120 x 5120) and down_proj (5120 x 17408) with <= 8 candidates per bucket, no
refinement and fewer repeats; `--time-budget-min N` stops gracefully; the JSON and the autotune cache are written after every row, so a killed run leaves usable
configs; a candidate whose cheap probe is > 2x slower than the best so far is dropped (`--prune-factor`, `--no-prune`); a throttled candidate is re-measured at
most once and the row is marked `throttled` (the winner is still cached, flagged in the entry, because a power-capped card is "throttled" in steady state and
the ranking stays valid; `--strict-throttle` refuses to cache such rows). The summary prints the best impl per M, the decode (M <= 16) geometric mean, the
prefill (M >= 32) ratio vs bf16 and the suggested `dequant_min_m`.

**Status of 7b.1 (HONEST).** Verified without a GPU: the dequantisation kernel and the chunked `dequant_matmul` in the Triton CPU interpreter against the
fp32 oracle (group sizes 8..K, per-channel, odd N/K, sym/asym, act-order, row chunks, fp16 and bf16, routing and `FORGE_W4_IMPL`); offline compilation for sm_80/89/90/120
with the PTX structure above; the routing policy, chunk planner, candidate lists, early abort / deadline / throttle handling in the tuner, the bench plan, budget and
incremental report, and the absence of host syncs in the launch path. **Not verified:** any speed of `dequant_matmul`, the tuned threshold, the `gemv` default tiles
at M = 1-8 (the table's tuned winner at M <= 8 is unknown, and `dflt` there was `dot`), the `dot` tile_n = 128 default at M = 9-31, the cuBLAS behaviour on
the chunked `[M, rows]` outputs, bf16 weight rounding against the GPU tolerance (`tests_gpu/test_kernels_w4_dequant_gpu.py` is the gate), the temp-memory peak under
graph capture. Not done: a rewrite of the `dot` kernel (a contiguous activation load plus an in-register nibble split with `tl.split` would remove the gather, but it
cannot be validated without hardware).

**How to measure on the GPU host**

```bash
pytest tests_gpu/test_kernels_w4a16_fast.py -q -s      # compile + numerics + a printed legacy-vs-new speed table
python scripts/bench_w4a16.py --quick                  # minutes: M {1,8,16,64,256,1024} on 2 shapes, <= 8 candidates/bucket
python scripts/bench_w4a16.py --quick --time-budget-min 30   # same, stops gracefully after 30 minutes (partial results kept)
python scripts/bench_w4a16.py                          # full Qwen3-14B x M sweep, writes the autotune cache + w4a16_bench.json
pytest tests_gpu/test_kernels_w4_dequant_gpu.py -q -s  # dequant kernel + dequant_matmul + graph capture + temp-memory gate
python scripts/bench_w4a16.py --act-order --m 1,8,16   # external GPTQ desc_act layout (shuffled g_idx); Forge's own artifacts use static groups
python scripts/bench_w4a16.py --dry-run                # plan only, works without a GPU
```

The sweep prints, per (shape, M): bf16, legacy, default-heuristic and tuned device times, speed-up vs bf16, achieved GB/s and
% of the measured copy bandwidth, the winning config, and whether the dispatcher actually picks the cached config up. The
cache is `<FORGE_HOME>/autotune/<chip fingerprint>.json`, the file `ops_gemm.tuned_config` reads.

**Status of this change (HONEST).** Verified without a GPU: the kernel logic in the Triton CPU interpreter against the fp32
oracle (all group sizes, sym/asym, act-order, odd N/K, split-K deterministic and atomic, fp16/fp32 dot, fp16/bf16/fp32 gemv);
offline compilation to PTX/cubin for sm_80, 89, 90 and 120 (fp16 and bf16, all config shapes in the sweep); the memory-access
structure in the generated PTX (16-byte weight vectors, shared-memory use 5-75 KiB, zero local-memory spills at sm_120);
the config policy, M bucketing and cache fallbacks on CPU. **Not verified:** that any of this is *faster* on hardware, the
achieved bandwidth, the gemv-vs-dot winner on any chip, register spills of the gemv kernel at runtime, the tuned split-K, the
sm_120 behaviour of Triton 3.8 (the user's version) vs the 3.9 used offline, and every number the sweep will print. The
expectation "approaches 3-4x at small M" is the Plan target (spec KPI 3.2x), not a result; with CUDA-graph timing the ceiling is
roughly the traffic ratio (16 / 4.19 bits per weight incl. scales and zeros, i.e. ~3.8x for the user's artifact) times
the achieved-bandwidth ratio of the two kernels.

## 8. Adding a kernel

1. **Spec**: add `forge/kernels/specs/<name>.yaml` (Plan C.2: `kernel`, `op`, `dtypes`, `quant`, `shape_hints`,
   `search`, `fusion`, `layout`, `oracle`, `tolerance`, `kpi`). Spec files are package data.
2. **Oracle**: add a pure-torch function in `forge/kernels/ref/` that is obviously correct (fp32 accumulate,
   readable over fast). It must run on CPU.
3. **Layout** (if the weight/KV format needs one): extend `forge/kernels/layouts` with `pack_*`/`unpack_*` and a
   property test that unpack(pack(x)) is identity.
4. **Triton kernel** in `forge/kernels/triton/<name>.py`: keep to `tl.dot`, masked `tl.load/store`, `tl.constexpr`
   tile parameters; document minimum Triton/torch versions in the module docstring; import Triton lazily.
5. **Launcher** in `forge/kernels/ops_*.py`: capability gate, shape-support check, `FORGE_FORCE_REF`/`FORGE_KERNEL_DEBUG`
   handling, reference fallback; re-export from `forge/kernels/ops.py`.
6. **Catalog**: make it appear in `forge.kernels.catalog.list_kernels()` with `min_sm` and availability reasons.
7. **Tests**: CPU test (oracle + layout + fallback launcher) in `tests/test_kernels_<name>.py`; GPU test in
   `tests_gpu/test_kernels_<name>.py` over the Plan section 6.7 edge-case list (K not a multiple of group size, odd
   shapes, M in {1..256}, empty experts, extreme scales, all-zero blocks...).
8. **Bench**: add the kernel to `BenchConfig.kernels` so the sweep and roofline fields cover it.
9. **Docs**: add a row to the catalogue and to the validation table here - validation cells start at **no**.

## 9. Adding a chip

Follow `docs/runbooks/adding-a-chip-descriptor.md`: add an entry to `forge/hw/chips.py::KNOWN_CHIPS`, extend the
capability rules, add an example descriptor under `configs/chips/`, run `forge probe` on the real device, and only
then upgrade the chip's tier from C to B/A.
