# Changelog

All notable changes are recorded here. Format: Keep a Changelog; versions follow `forge/__version__.py`.

## [Unreleased]

### Added (ternary / 1.58-bit integration, ADR 0007; EXPERIMENTAL, CPU-validated only)
- Ternary {-1,0,+1} weights with int8 activations: numerics (`forge/formats/ternary.py`), layout `ternary2_row`, oracle, linear kinds `t158` (packed) and `fq_t158` (QAT),
  `.t158.safetensors` import/export (tern158 interchange, bit-exact; golden vectors from tern158).
- Training: `precision.quant_aware: ternary` plug-in (class-swap QAT, warm-up blend, `ternary/*` health metrics, zero-density collapse abort), job type `ternary_train`,
  `forge ternary`, templates `configs/ternary/`, job-detail "Ternary health" panel.
- Compression: recipe stage `ternarize`, `ptq` format `ternary`, `pack` validation, gate `g-t158` (TARGET placeholders), recipes `ternary-ptq` / `ternary-convert`.
- Kernels: Triton ternary GEMM (planes + gather) and tern158-exact activation quantiser, `ops_gemm.ternary_linear`, spec `ternary_w158a8_gemm`, `ChipInfo.has_int8` /
  `int8_tc_fast`, bench case + `scripts/bench_ternary.py` (CUDA-graph timing), optional native CUDA tier `forge/native/ternary` (not wired into dispatch).
- Planner: training simulator (`forge/planner/trainsim.py`, `POST /planner/train-sim`; estimate, uncalibrated), `t158` scheme + wide prior, chip table fields and an RTX PRO 6000 preset.
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
  tier is type-checked only; the sharded multi-GPU ternary trainer, planner rule R-Q7-TERNARY and T158 claims-ledger entries are not implemented (see docs/TERNARY.md).
- No GPU code path has been executed on hardware yet; see `docs/KERNELS.md` for the validation table.
