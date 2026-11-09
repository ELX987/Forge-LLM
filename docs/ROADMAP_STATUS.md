# Roadmap status: Plan checklist vs. this repository

Maps the Plan's work items (Plan section 12, C-001 .. C-124) to what the repository actually contains. Written from a
reading of the code and tests at the time of the first complete build. Much of the table is historical: later runs include
GPU reports from an RTX PRO 6000 Blackwell and a Qwen3-14B W4A16 compression job. See `docs/KERNELS.md` for measured
kernel results; real-model evaluation and most hardware gates remain open.

Status legend

* **done (CPU)**: implemented and covered by CPU tests in `tests/` with synthetic/tiny inputs. Says nothing about
  real-model quality or GPU behaviour.
* **partial**: implemented in part, or implemented but with a documented gap; the Notes say what is missing.
* **GPU-unvalidated**: code exists (often interpreter-checked) but no GPU run has happened. Counts as *partial* for
  "Done when" purposes that require hardware numbers.
* **not started**: not in the repository (or only a template/stub document).

"Done when" criteria in the Plan frequently require measurements on real hardware or real models; those are marked as
unmet even when the machinery exists.

## Phase 0 - Decisions, access, foundations

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-001 | Inventory hardware and access | partial | Chip database + support matrix (`forge/hw/chips.py`) and probe exist; the written inventory / "primary chip" decision is the owner's task (vast.ai instance chosen at rental time) |
| C-002 | Choose reference models | partial | Templates use Llama-3.1-8B (gated), Qwen2.5 family (Apache-2.0); licences are recorded by the model registry. No models downloaded/hashed; no 70B or MoE reference chosen |
| C-003 | Repo, CI, project instructions | done (CPU) | Monorepo layout, `.github/workflows/ci.yml` (lint, shellcheck, CPU tests on 3.10/3.12, docker build, manual GPU job), `CONTRIBUTING.md`. CI has not been executed on GitHub yet. No `CLAUDE.md` |
| C-010 | Define schemas | done (CPU) | `forge.core.schemas` (chip, recipe, manifest, EvalReport, Claim, Deployment...) with validation tests; "golden-file" tests are limited to round-trips |
| C-011 | Artifact store v0 | done (CPU) | Chunked content-addressed store, manifests, lineage, tiers, GC, delta-chained checkpoints, lossless codec; `python -m forge.store` CLI. Dedup demo exists as tests |
| C-012 | Determinism and environment capture | done (CPU) | `seed_everything`, `capture_environment`; re-run identity holds for deterministic stages on CPU |
| C-013 | Evaluation harness v0 | done (CPU) | KL / p99 / flip / ECE / perplexity, bootstrap CIs, noise-floor estimation, suites `distribution`, `mcq` |
| C-014 | Evaluation harness v1 | partial | `longgen`, `longctx`, `safety`, `contamination`, `throughput`, `agentic` suites exist. The "known-bad (heavily pruned) model fails as expected" acceptance check has not been demonstrated on real models |
| C-015 | Telemetry lake and job runner | partial | SQLite metrics store with rollups, subprocess job runner with log capture, optional Parquet export. No Parquet lake/embedded query engine (ADR 0003); local runner only, no SSH job runner |
| C-017 | Control-plane API and CLI | done (CPU) | FastAPI `/api/v1` (OpenAPI at `/docs`), bearer-token auth, `forge` CLI (`job`, `train`, `finetune`, `compress`, `eval`, `serve`, `bench`, `probe`, `store`, `models`, `schema`, `doctor`) |

## Phase 1 - Baselines and hardware profiling (Gate G1)

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-020 | Capability probe and micro-benchmarks | GPU-unvalidated | `forge.hw.probe`/`microbench` run on CPU with tier C fallback; GPU microbenchmarks written, untested. No measured descriptor for any chip yet |
| C-021 | Cost model v0 | done (CPU) | `forge.planner.costmodel` reproduces Examples A and B and the B* ~ 74 / 295 crossover in tests (within the tests' stated tolerances) |
| C-022 | BF16 serving baselines | not started | Needs hardware and an existing open engine to compare against |
| C-023 | Seed the claims ledger | partial | 53 claims seeded from P1-P7 (>= 30 required). The "at least 10 resolved" criterion is not met: all remain `untested` except possibly formula-level facts |
| C-024 | Define baseline arms | partial | Recipes list `baselines:` names; the arms themselves (fresh student, next-lower precision) are not defined as runnable recipes with token budgets |
| G1 | Gate: baselines trusted | not started | Needs real noise floors and descriptors |

## Phase 2 - PTQ compression core

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-030 | Calibration registry and statistics cache | done (CPU) | Named calibration sets, Hessian/channel-scale cache keyed by model/set/layer |
| C-031 | RTN + group quantisation, reference dequant | done (CPU) | INT4/INT8 sym/asym, groups, oracles; the ~6 dB/bit check is done on synthetic data |
| C-032 | GPTQ / AWQ / SmoothQuant | partial | Implemented (plus MR-GPTQ variant, mixed-precision bit allocation) and tested on tiny models. "W4A16 and W8A8 artifacts for the 8B model with eval reports" not done |
| C-033 | Fake-quant format library | done (CPU) | INT, FP8 E4M3/E5M2, E2M1 + E4M3/E8M0 block scales, stochastic rounding, conformance tests. NVFP4-vs-MXFP4 MSE ordering reproduction: see `tests/test_formats_numerics.py` |
| C-034 | KV-cache quantisation (reference) | partial | FP8 / int8 / KIVI-style with residual window in reference attention; "eval at full target context, accuracy-vs-bytes plot" not done |
| C-035 | Recipe engine | done (CPU) | Declarative stages, hashing, per-stage cache, YAML loader, lineage emission, `recipe` and `compress` jobs |
| C-036 | First gated artifacts (FP8, W4A16 for 8B) | partial | Pipeline + gates exist and run on a tiny model; no 8B EvalReports |
| C-037 | Rotation module | partial | Hadamard / learned rotations implemented; the per-format "when rotation helps" table has not been produced |

## Phase 3 - Kernel library v1 (Gate G2)

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-040 | Kernel spec, oracles, correctness harness | done (CPU) | Specs (YAML + pydantic), oracles with the Plan 6.7 edge cases, tolerance rules; autotuner rejects candidates failing the oracle |
| C-041 | Layout deriver and W4 packer | done (CPU) | `derive_layout`, pack/unpack/repack with property tests |
| C-042 | Portable W4A16 GEMM | GPU-unvalidated | Triton kernel interpreter-checked only; speed-up vs BF16 unrecorded |
| C-043 | Tuned W4A16 GEMM + autotuner | partial | Autotuner done (cache, IQR, throttle detection). No chip-specific hand-tuned schedule; 85% / 3.2x are targets |
| C-044 | FP8 GEMM and fused activation quantisation | GPU-unvalidated | `torch._scaled_mm` path and Triton kernels written; not run |
| C-045 | Paged decode attention | GPU-unvalidated | Paged/GQA/split-KV/quantised-KV/MLA kernels + oracles; interpreter-checked, not run on GPU |
| C-046 | Kernel benchmark DB and regression CI | partial | `kernel_bench` table, `bench` job, roofline fields. No automated >5% regression alert or CI job |
| C-047 | Calibrate the cost model | partial | `fit_eta` / residual machinery exists; no kernel data to fit |
| G2 | Gate: kernels pay off | not started | Needs hardware numbers |

## Phase 4 - Serving, KV management, speculation

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-050 | Serving engine or adapter | partial | Own engine (ADR 0004): continuous batching, paged KV, radix prefix cache, chunked prefill, sampling, OpenAI-compatible server; "outputs identical with features on/off" is tested on CPU with tiny models. No GPU run, no throughput comparison |
| C-051 | Capacity planner | partial | `forge.serve.capacity` + `/serve/simulate`; the "within 10% of measured concurrency" criterion needs a live engine on a GPU |
| C-052 | KV block manager with tiers | partial | HBM / host / NVMe tiers, hash-addressed blocks, demote/promote, stats (CPU-tested). The break-even experiment is not done |
| C-053 | Speculation | partial | N-gram and draft-model drafters, EAGLE-style head, load-aware controller (gamma = 0 allowed), distribution-exactness tests. G-SPEC "never slower than off" unmeasured |
| C-054 | Load tests and capacity curves | not started | `throughput` suite exists; no curves stored |
| C-055 | Disaggregated prefill/decode | not started | |

## Phase 5 - Distillation, pruning, QAT (Gate G3)

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-060 | Teacher top-k cache builder | done (CPU) | Build/read/verify with residual mass and coverage stats; reuse keyed by cache key |
| C-061 | Distillation trainer | partial | Offline-logit, sequence-level, on-policy, reasoning-SFT; forward/reverse KL, JSD. Works on tiny models; "improves over SFT-only on a pilot" not shown |
| C-062 | Structured pruning | done (CPU) | Importance estimation, GQA-aware width/head/depth/expert pruning; shapes verified on tiny models |
| C-063 | Prune-distil run 8B -> 4B | not started | Only a recipe template (`prune-distill-8b-to-4b`) with a pilot budget |
| C-064 | QAT / QAD | partial | Fake-quant linears with STE exist; no QAT/QAD stage or job and no QAD-vs-PTQ comparison |
| C-065 | Teacher licence gate and audit trail | done (CPU) | `LicenceBlocked` on non-permissive teachers without attestation; teacher identity recorded in lineage |
| C-066 | Low-rank, delta, sparse-attention experiments | partial | Whitened SVD with rank allocation and healing; BitDelta-style deltas; sparse-attention kernels/oracles. No same-budget dense-baseline experiments; no ledger resolutions |
| G3 | Gate: recipes beat baselines | not started | |

## Phase 6 - Web application

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-070 | Front-end scaffold and design system | done (CPU) | No-build ES modules, schema-driven forms, canvas charts, theming (ADR 0006); static-asset tests exist; not browser-tested on a live GPU box |
| C-071 | Core screens | partial | Dashboard, jobs, job monitor, models, datasets, hardware, new-job wizard, fine-tune, recipes, compare, inference/chat, kernel lab, storage, claims, settings exist. No recorded walkthroughs |
| C-072 | Provenance badges and uncertainty display | partial | Badges and CI / within-noise markers in compare tables; "no ranking of indistinguishable values" is implemented in the compare endpoint, audit across all tables not done |
| C-073 | Deep links, export, roles | partial | Hash routes and CSV/PNG export; no PDF export, no read/write roles (single shared token) |

## Phase 7 - Desktop application

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-080 | Desktop shell and native sidecar | not started | Deferred by ADR 0001 |
| C-081 | Kernel lab (M4) and local-hardware view (M9) | partial | Browser Kernel lab and Hardware pages cover part; no tray alerts/desktop |
| C-082 | Offline mode and sync | not started | |

## Phase 8 - Block-scaled FP4 and low-precision training

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-090 | Confirm FP4 hardware access | not started | Needs a Blackwell host and a probe run (`docs/VAST_AI.md`) |
| C-091 | Block-scaled GEMM, scale layouts, fused quantiser | GPU-unvalidated | Dequant-GEMM and fused quantiser kernels interpreter-checked; native `tl.dot_scaled` experimental; 2.2x vs FP8 unmeasured |
| C-092 | Format-aware FP4 PTQ | partial | NVFP4/MXFP4 PTQ (incl. MR-GPTQ-style) implemented; no per-model recommendation table |
| C-093 | FP4 training pilot with diagnostics | partial | R2 rung with SR, RHT, 2D scaling, exemptions, healing, TR-2 diagnostics, **emulated** (fake-quant); gated by `allow_unvalidated_fp4`. No pilot run |

## Phase 9 - Mixture-of-Experts

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-100 | MoE profiling | partial | Router statistics collection in calibration; not run on a real MoE |
| C-101 | Routing and grouped-GEMM kernels | GPU-unvalidated | Oracles + Triton kernels (interpreter-checked) |
| C-102 | Expert prune/merge and MoE quantisation | partial | Expert prune/merge options and per-layer bit allocation exist; no tail evaluation on a real MoE |
| C-103 | Parallelism planner and quantised all-to-all | not started | |
| C-104 | Speculation under MoE | partial | Cost model has expert-touch growth; no controller validation |
| G4 | Gate: FP4 training and MoE defaults | not started | |

## Phase 10 - Multi-agent fleets

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-110 | Fleet manifest, runtime, prompt linter | not started | `configs/fleets/support-triage-v3.yaml` documents the format only |
| C-111 | Adapter/delta kernels and delta compression | partial | Delta compression (BitDelta/low-rank) artifact; no segmented-batched serving kernel |
| C-112 | Context manager | not started | |
| C-113 | Fleet scheduler, traces, M8 | not started | |
| C-114 | Agent evaluation, hypotheses H1-H5 | partial | `agentic` suite scaffold only; no hypotheses resolved |

## Phase 11 - Hardening and release (Gate G5)

| ID | Item | Status | Notes |
|----|------|--------|-------|
| C-120 | Security review and signing | not started | Token auth and `SECURITY.md` notes only; no signed manifests, audit log, secret scanning |
| C-121 | Regression and chaos suites | partial | Store GC / corruption / restore tests and CPU CI; no kernel/perf regression or planner replay suites |
| C-122 | Docs, tutorials, example projects | partial | Docs, runbooks, ADRs, templates exist; no outside-user reproduction of a baseline ("first baseline in an hour") |
| C-123 | Release rehearsal | not started | Runbooks drafted (`docs/runbooks/`); no canary/rollback exercised |
| C-124 | Retrospective and roadmap v2 | not started | |
| G5 | Gate: release | not started | |

## Summary

Done on CPU: foundations (schemas, store, recipe engine, determinism), evaluation machinery, PTQ/pruning/distillation
algorithms on tiny models, planner/cost model, job system, API, CLI, web UI scaffolding. Some non-ternary kernels and a
Qwen3-14B W4A16 compression run have since been exercised on hardware; their results do not close the full model-quality,
serving-throughput, multi-GPU or G1-G5 requirements. For ternary the next step is the G0/G1 validation ladder in
`docs/runbooks/ternary-validation.md`, followed by a real-model conversion and measured baseline.

## Ternary (1.58-bit) integration - ADR 0007 / docs/TERNARY.md

Not a Plan work item; tracked here because it adds a method family. Ternary-specific paths are **CPU-validated only** (kernels: interpreter + offline PTX compile); no GPU run of the ternary path through Forge is recorded. Other Forge kernels have GPU results documented in `docs/KERNELS.md`.

| Phase | Item | Status | Notes |
|-------|------|--------|-------|
| P0 | ADR 0007, TERNARY.md, branch hygiene | done (CPU) | |
| P1 | Format, layout, oracle, `t158` / `fq_t158`, artifact round trip, `.t158` interchange | done (CPU) | golden vectors from tern158; serves on CPU through the engine |
| P2 | Quant-mode plug-in, ternary QAT, health metrics, `ternary_train` job, method catalog | done (CPU) | emulated fake-quant: no training speed-up; wd-to-zero schedule and bf16-SR optimizer state not done |
| P3 | `ternarize` stage, ptq format, g-t158 gate, recipes, train simulator, t158 scheme, chip table | partial | planner rule R-Q7-TERNARY, T158 claims, HF conversion with SubLN, simulator panel on New Job not done |
| P4 | Triton GEMM + quantiser, dispatcher, chip int8 flags, bench, native tier | GPU-unvalidated | interpreter + PTX only; native tier not wired |
| P5 | Sharded multi-GPU trainer (`zero_ternary`) | partial | Merged; CPU/gloo tests cover training and checkpoint resharding. Full latent weights remain replicated; NCCL/GPU G1 validation is outstanding (`docs/ZERO_TERNARY.md`) |
| P6 | Methods page, Ternary Lab, sim panel | done (CPU) | checked visually in demo mode only |
| P7 | Smoke script, hardware validation ladder, docs | partial | ladder written (`docs/runbooks/ternary-validation.md`); every hardware cell is "no" |
