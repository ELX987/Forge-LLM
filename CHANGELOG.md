# Changelog

All notable changes are recorded here. Format: Keep a Changelog; versions follow `forge/__version__.py`.

## [Unreleased]

### Added (mixed-precision block-scaled attention, K4b; CPU + offline-compile validated, NOT run on a GPU)
- `ops_attn.mixed_attention`: one paged decode-attention path for every KV precision - bf16/fp16/fp32, fp8_e4m3, int8, int4,
  **nvfp4, mxfp4 and ternary (1.58-bit)** - with optional Gemma-2 softcap and per-head attention sinks. Block-scaled formats run
  `triton/attn_mixed.py` (two-pass flash, block payload dequantised once per tile and fused into the score / value matmuls); the
  fp8/int8/int4 family reuses the K4 kernel; everything falls back to the pure-torch oracle `ref/attn_mixed.py`.
- KV numerics in `ref/attn_kv.py`: NVFP4 / MXFP4 / ternary pools (packed E2M1 / 2-bit codes, per-channel-block key scales,
  per-token-block value scales), E4M3 / E8M0 scale codecs, `frexp`-based E8M0 exponent (FA4 trick), block quantise/dequantise and
  `build_paged_pools` support; `kv_bytes_per_element` now reports 4.5 / 4.25 / 3.0 bits for the new formats.
- Adopted from FlashAttention (2/3/4): GQA work partitioning + paged KV + split-KV with LSE merge (FA2), block-quantisation scale
  folding (FA3), `frexp` E8M0 exponent, rescale threshold, `track_unquantized_sum`, learnable sink and softcap (FA4). Ternary KV is
  Forge's own - FA has no 1.58-bit attention path.
- Tests: `tests/test_kernels_attn_mixed.py` (oracle + codecs + dispatch, 64 cases), `tests/interp_attn_mixed_cases.py` (kernel in the
  Triton CPU interpreter, tile-exact vs the oracle), offline compilation for sm_70/75/80/89/90/100/120, `tests_gpu/test_kernels_attn_mixed.py`
  (the GPU gate, not yet run). Docs: `docs/KERNELS.md` §7c, `forge/kernels/README_ATTN.md`.

### Added (ternary / 1.58-bit integration, ADR 0007; EXPERIMENTAL, CPU-validated only)
- Ternary {-1,0,+1} weights with int8 activations: numerics (`forge/formats/ternary.py`), layout `ternary2_row`, oracle, linear kinds `t158` (packed) and `fq_t158` (QAT),
  `.t158.safetensors` import/export (tern158 interchange, bit-exact; golden vectors from tern158).
- Training: `precision.quant_aware: ternary` plug-in (class-swap QAT, warm-up blend, `ternary/*` health metrics, zero-density collapse abort), job type `ternary_train`,
  `forge ternary`, templates `configs/ternary/`, job-detail "Ternary health" panel.
- Compression: recipe stage `ternarize`, `ptq` format `ternary`, `pack` validation, gate `g-t158` (TARGET placeholders), recipes `ternary-ptq` / `ternary-convert`.
- Kernels: Triton ternary GEMM (planes + gather) and tern158-exact activation quantiser, `ops_gemm.ternary_linear`, spec `ternary_w158a8_gemm`, `ChipInfo.has_int8` /
  `int8_tc_fast`, bench case + `scripts/bench_ternary.py` (CUDA-graph timing), optional native CUDA tier `forge/native/ternary` (not wired into dispatch).
- Planner: training simulator (`forge/planner/trainsim.py`, `POST /planner/train-sim`; estimate, uncalibrated), `t158` scheme + wide prior, chip table fields and an RTX PRO 6000 preset.
- Planner rule `R-Q7-TERNARY` and flow node Q7 (sub-3.5-bit + unthrottled INT8 pipe -> one-shot ternarize probe and QAT + distillation arms, gate `g-t158`), with `T158-*`
  claims seeded from the tern158 report and linked to the rule (`docs/CLAIMS.md`). Method catalog: `ternary-ptq` / `ternary-convert` are now `experimental` with real recipes.
- Planner: `budget.max_bits_per_weight` is enforced as a hard cap on ranked candidates (`over_budget` in `dropped`), and serving-level arms (KV, speculation, activation sparsity)
  stack on a base that satisfies the cap; the compress `eval` stage accepts `noise_repeats`, so planner recipes can run the noise-floor estimate.
- Method catalog `forge.methods` + `GET /methods`; UI pages Methods and Ternary Lab; training-simulator panel component.
- Scripts and docs: `scripts/ternary_smoke.sh` (CPU end-to-end), `scripts/ternary_validate.sh` + `docs/runbooks/ternary-validation.md` (hardware ladder), `docs/TERNARY.md`, `NOTICE`.

### Added
- Compress outputs are registered as selectable models (`models` table, `source='artifact'`, quantisation label / base model / size / layout) by the
  compress job, by `POST /models/import-artifacts` (backfill of `runs/*/artifact`) and by `POST /artifacts/{hash}/register-model`; the Inference
  start dialog has a grouped picker (quantised artifacts, base models) with W4A16 badges and a live memory estimate (`POST /deployments/estimate`).
- Serving loads packed artifacts as packed and verifies it (no silent dense fallback), resolves ids / names / paths / store digests, and reports what it
  loaded (`model`, `memory` in `/stats` and the deployment record). KV pool is sized explicitly (`kv_pool`, `gpu_mem_util` now 0.85 of total memory).
- Repository scaffold, `forge.core` (schemas, hashing, db, registry, run context, recipe engine).
- Operations tooling: `scripts/vast_setup.sh`, `run_server.sh`, `gpu_smoke.sh`, `run_gpu_tests.sh`,
  `gpu_report.py`, CPU dev setup scripts (bash + PowerShell).
- Docker: `docker/Dockerfile.cuda`, `docker/docker-compose.yml`, `.dockerignore`.
- CI: CPU tests on Python 3.10 / 3.12, lint, shellcheck, docker build check, manual GPU job for self-hosted runners.
- Documentation: VAST_AI, USER_GUIDE, KERNELS, API, CONFIG_REFERENCE, CLAIMS, ROADMAP_STATUS, runbooks, ADRs 0001-0006.
- Config templates under `configs/` validated by `tests/test_configs_templates.py`.

### Known limitations
- Ternary: no GPU run exists. Training uses fake-quantised bf16 compute (no training speed-up); the Triton kernels are interpreter-checked and PTX-compiled only; the native CUDA
  tier is type-checked only; the sharded multi-GPU ternary trainer has no NCCL/GPU run; planner rule R-Q7-TERNARY and the T158 claims are seeded but every T158 claim is untested
  (see docs/TERNARY.md, docs/CLAIMS.md).
- No GPU code path has been executed on hardware yet; see `docs/KERNELS.md` for the validation table.
