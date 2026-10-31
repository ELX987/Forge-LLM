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
| K1 W4A16 / W8A16 (`triton/gemm_wq.py`) | yes | yes (fp16) | **no** | **no** | none |
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
