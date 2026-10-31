# Attention / MoE / utility kernels (`kernels_attn`)

Plan references: **K4-K9**, **KR-4 ... KR-9**, §6.7 (fusion map + edge-case list), P6 Ch. 3 (selector bytes).
Owner files: `ref/attn_*.py ref/moe_*.py ref/util_*.py`, `triton/attn_*.py moe_*.py util_*.py`, `ops_attn.py ops_moe.py ops_util.py`.

> **Honest status.** Every Triton kernel here is *implemented but not yet run on an NVIDIA GPU* (the development machine has none).
> What has been verified, and how:
>
> | Level | What it proves | Where |
> |---|---|---|
> | Oracle tests (CPU, torch) | the reference maths: GQA, ragged/non-page-multiple lengths, quantised KV, split-KV merge, windows, MLA, tree masks, exact rejection sampling (chi-square style), MoE incl. M = 0 experts, P6 byte-accounting numbers, Philox known-answer tests ... | `tests/test_kernels_attn_ref_*.py`, `tests/test_kernels_attn_ops.py` |
> | **Triton CPU interpreter** | the *kernel code* (indexing, masks, online softmax, LSE merge, top-k bisection, Philox, grouped-GEMM tile schedule ...) gives the oracle's answer on small shapes (`TRITON_INTERPRET=1`, no GPU involved) | `tests/interp_*_cases.py` (run by `tests/test_kernels_attn_triton_interp.py`) |
> | **Offline compilation** | the *real* Triton compiler accepts every kernel/specialisation for sm_70, 75, 80, 89, 90, 100, 120 (`triton.compile` to cubin, no device) and the shared-memory footprint fits each target | `tests/test_kernels_attn_triton_compile.py` (`FORGE_COMPILE_TARGETS=70,75,80,89,90,100,120`) |
> | **GPU tests** (not yet run) | numerics vs the oracles on hardware over the Plan 6.7 edge-case list, determinism, reproducibility | `tests_gpu/test_kernels_attn_*.py` |
>
> Performance has **not** been measured: none of the Plan's bandwidth KPIs (e.g. 85 % of peak for decode attention at 32-128K) is claimed.
> Run `pytest tests_gpu/test_kernels_attn_*.py -q` first on the vast.ai host, then `forge.kernels.bench` with the `bench_cases()` of each ops file.

## Dispatch

Each `ops_*` function uses Triton only if **CUDA + `import triton` + compute capability + supported dtype/shape** (and `FORGE_FORCE_REF` is not `1`), else the torch oracle in `forge.kernels.ref`.
`FORGE_KERNEL_DEBUG=1` logs the path taken. `TRITON_INTERPRET=1` makes CPU tensors take the (slow) Triton path - used by the tests only.
Other knobs: `FORGE_DECODE_SPLITS` (force split-KV count), `FORGE_TRITON_TOPK=1` (prefer the Triton top-k kernel; default is a vectorised `torch.topk` implementation, see below).
Minimum versions: **Triton 2.1** (developed against 3.x; `tl.cumsum`, `tl.randint4x`, 3-arg `tl.dot`), PyTorch 2.1.
Hardware: fp16 kernels sm_70+, bf16 kernels sm_80+, FP8 *stores* sm_89+ (FP8 KV/latent/index-key *loads* are portable: integer E4M3 decode on any sm, native `float8e4nv` bitcast on sm_89+), MLA sm_80+ (shared-memory tile), grouped GEMM / LoRA sm_80+.

## Conventions (shared by oracle and kernels)

* **KV pages** are head-major: `pool[num_pages, Hkv, page_size, D]` (INT4: last dim `D/2`; two offset-binary nibbles per byte, low nibble = even channel). `block_tables[B, max_pages]` int32, `seq_lens[B]` count *all* cached tokens including the current one and the residual window.
  The decode query sits at position `seq_len - 1`; query head `h` uses KV head `h // (Hq // Hkv)`.
* **Quantised KV** (KIVI recipe): keys per page, per KV head, **per channel** (`k_scale [P, Hkv, D]`; `[P, Hkv, 1]`, `[Hkv, D]`, scalar and `None` accepted); values **per token** (`v_scale [P, Hkv, page_size]`; `[P, Hkv, 1]`, scalar, `None`). Optional additive zero points `k_zero / v_zero` (asymmetric INT8/INT4). `dequant = q * scale + zero`. Helpers: `ref.attn_kv.quantize_kv_page` (residual-window flush), `build_paged_pools`.
* **Residual window**: `k_res, v_res [B, Hkv, R, D]` + `res_lens[B]` hold the *last* `res_len` tokens in full precision; the pool holds the first `seq_len - res_len`.
* **MLA**: absorbed form - `q_nope [B,H,Dc]` already multiplied by `W_UK`, latent cache `c_pool [P, ps, Dc]` (bf16/fp16/FP8 + per-token `c_scale`), rotary keys `r_pool [P, ps, Dr]`; output in latent space, `w_uv [H, Dc, Dv]` applied afterwards.
* **Prefill** causal mask is *bottom-right aligned* (flash-attn convention): `k_len > q_len` = prefix cache / chunked prefill. Layouts: `bshd` (default), `bhsd`, or packed varlen with `cu_seqlens`.
* **RNG**: Philox-4x32-10, counter `(offset, 0, 0, 0)`, key = 64-bit seed split in two words, uniform = `(r0 >> 8) * 2^-24`; bit-identical to `tl.randint4x` (known-answer tested). 32-bit offsets; fold step/layer into the seed with `util_rng.derive_seed`.
* **Rounding**: INT8/INT4 quantisers round half *away* from zero (kernel convention, mirrored in the oracle); FP8 uses RNE after clamping to +-448; all-zero or scale-underflowing groups get `scale = 1` (flush to 0, never 0/0); the absmax ignores non-finite values and non-finite inputs are unspecified in the kernels.

## What is where

| Plan item | Public API (`forge.kernels.ops_*`) | Triton | Oracle |
|---|---|---|---|
| **KR-4 / K4** paged decode, GQA-grouped, split-KV + LSE merge, FP8/INT8/INT4 KV, residual window, sliding window, ragged lengths | `ops_attn.paged_decode_attention` | `triton/attn_decode.py` (`run_paged_attention`), `attn_common.py` | `ref/attn_decode.py`, `attn_kv.py` |
| K4 MLA latent decode (+ sparse/window) | `ops_attn.mla_decode_attention` | `triton/attn_mla.py` | `ref/attn_decode.mla_decode_attention_ref` |
| prefill / chunked prefill, paged KV write | `ops_attn.prefill_attention`, `chunked_prefill_attention`, `paged_kv_write` | `triton/attn_prefill.py` (torch SDPA fallback on CUDA) | `ref/attn_prefill.py` |
| **KR-5 / K5** lightning indexer scan, top-k, token/block gather attention, cross-layer sharing, selector-bytes accounting | `lightning_indexer_scores`, `select_sparse_indices`, `sparse_topk_attention`, `sparse_decode_step`, `SharedSelector`, `selector_bytes`, `ops_util.topk_select` | `triton/attn_sparse.py` (+ `topk_idx` mode of decode/MLA kernels) | `ref/attn_sparse.py` |
| **KR-6/7 / K6** tree attention, fused exact rejection verification, KV accept/rollback compaction | `ops_attn.tree_attention`, `ops_util.rejection_verify`, `greedy_verify`, `kv_compact` | `triton/attn_spec.py` | `ref/attn_spec.py`, `ref/util_rng.py` |
| **KR-9 / K7** fused routing, deterministic permutation, grouped GEMM (M = 0 experts, W4A16 hook), combine, expert indirection | `ops_moe.moe_route`, `moe_permute`, `grouped_gemm`, `moe_unpermute_combine`, `apply_expert_map`, `moe_experts_forward` | `triton/moe_kernels.py` | `ref/moe_ref.py` |
| **K8** Walsh-Hadamard (block 16/32/64/128/any pow2), absmax/scale, FP8/INT8 row-group quant, Philox stochastic rounding (INT8/INT4/FP4), RMSNorm(+residual)+quant, RoPE + KV-quantise + page write, TEAL gather-GEMV | `ops_util.hadamard_transform`, `absmax`, `quantize_rowwise` / `quantize_fp8_rowwise`, `stochastic_round`, `rmsnorm_quant`, `rope_and_kv_write`, `teal_gemv` | `triton/util_*.py` | `ref/util_*.py` |
| **K9** segmented-batched LoRA / delta on a mixed batch | `ops_util.segmented_lora` | `triton/moe_kernels.py` grouped GEMM (adapters = experts) | `ref/util_adapters.segmented_lora_ref` |

Every ops file exposes `KERNELS` (family / min_sm / module names, for the catalog) and `bench_cases()` (`{name, family, make_inputs(cfg), run, ref, shapes, bytes_moved(cfg), flops(cfg)}`, for `bench.py`).

## Design notes and deviations worth knowing

* **Decode kernel**: one program per (sequence, KV head, split) serves *all* `g` query heads (rows of the `tl.dot` M tile, padded to 16) so each KV byte feeds `g` heads. Tokens are addressed individually (`pos -> page_table[pos // ps]`), so `BLOCK_N` (64, or 32 for D > 128) is independent of the page size. The same kernel runs in `SPARSE` mode over a token list and, with several draft-tree nodes packed into the M tile, in tree-verification mode. Dequantisation is fused into the K/V loads; scales are gathered per token (stride-0 expanded views), which costs extra load instructions but no extra HBM traffic. Split count defaults to `ceil(2 * SMs / (B * Hkv))` rounded down to a power of two (<= 64); the merge is a second tiny kernel. For a fixed split count the result is bit-reproducible (no atomics).
* **Top-k**: the Triton bisection kernel is exact and deterministic but does 32 passes over the scores per row, so `ops_util.topk_select(method="auto")` defaults to a vectorised `torch.topk` implementation with *identical* semantics (ties -> lower position, ascending output, `-1` padding) and uses the Triton kernel only with `FORGE_TRITON_TOPK=1` / `method="triton"`. A radix/bitonic shared-memory selector is future work.
* **Block-granular sparse attention** expands selected block ids into token positions and reuses the token gather kernel (rows of one page are contiguous so loads still coalesce); a dedicated block-contiguous kernel is not implemented.
* **Grouped GEMM**: the tile schedule is computed on device by `build_block_schedule_ref` (no host sync, static upper bound `ceil(M/BM) + E` tiles). The activation row-gather, scatter-accumulate epilogue and per-expert scale are fused; the permutation itself is a stable `torch.sort` (deterministic, but not a Triton kernel). W4A16 uses a *portable local format* (`pack_w4_experts`: uint8 nibbles along K + group scales/zeros) - convert from the layouts of `forge.kernels.layouts` before use. Expert all-to-all / EP overlap (K7 "Expert-parallel comms") and offload tiers are not part of this module.
* **Segmented LoRA** is two grouped-GEMM launches (shrink, then expand with scatter-accumulate into `y`), not one fused kernel: the rank-`R` intermediate makes one trip through L2. Ranks are zero-padded to a multiple of 16.
* **Rejection sampling** supports draft *chains* (and greedy tree verification in torch); multi-candidate tree rejection sampling (SpecInfer-style) is not implemented.
* `kv_compact` moves native-dtype rows; quantised pools keep per-page key scales so tree tokens should be accepted through the residual window (not implemented in this module).
* Not implemented: Hadamard fused with quantise-on-write (compose `hadamard_transform` -> `quantize_rowwise`), fused entropy-decode GEMM prologue (K8 item 4), 1-bit-sign BitDelta kernels, logit softcapping / attention sinks, INT4 fused RoPE+write (oracle only), autotuning hooks (launch configs are fixed heuristics: `BLOCK_*`, `num_warps`, `num_stages` in the launchers).

## Running the tests

```bash
pytest tests/test_kernels_attn_*.py -q                       # CPU: oracles, dispatchers, Triton CPU-interpreter, offline compile (~1.5 min)
FORGE_COMPILE_TARGETS=70,75,80,89,90,100,120 pytest tests/test_kernels_attn_triton_compile.py -q
pytest tests_gpu/test_kernels_attn_*.py -q                   # on the GPU host
```
