# Configuration reference

## 1. How configuration works

* Every job type has a **pydantic v2 config model** (`extra="forbid"`): unknown fields are errors, never ignored.
  Every field has a description, bounds and (for the UI) `x-group` / `x-advanced` / `x-unit` hints.

  | Job type | Config model |
  |----------|--------------|
  | `train` | `forge.train.config.TrainConfig` |
  | `ternary_train` | `forge.ternary.config.TernaryTrainConfig` (experimental) |
  | `finetune` | `forge.train.config.FinetuneConfig` |
  | `distill` | `forge.train.config.DistillConfig` |
  | `dataset_ingest` | `forge.train.data.IngestConfig` |
  | `compress` | `forge.compress.run.CompressJobConfig` |
  | `eval` | `forge.eval.run.EvalConfig` |
  | `serve` | `forge.serve.config.ServeConfig` |
  | `bench` | `forge.kernels.bench.BenchConfig` |
  | `probe`, `recipe`, `model_download`, `gc`, `checkpoint_export`, `selftest` | `forge.jobs.builtin.*` |

  The registry that maps type to model and entrypoint is `forge/jobs/registry.py`.
* **Schema-driven forms.** `GET /api/v1/schemas/job-types` returns, per type, the JSON Schema
  (`forge.core.jsonschema.schema_for`), defaults (`defaults_for`) and template names. The web UI builds the form from
  that: types, enums, ranges, descriptions, nested objects, arrays, an "advanced" toggle. **Adding a field to a
  pydantic model adds it to the UI automatically.** There is no per-field front-end code.
* **Templates** are read-only seeds: `configs/<dir>/*.yaml` is the *params* of a job (exactly what you would POST as
  `params`). They appear in the UI's template selector and `GET /api/v1/configs/templates?type=`. Pick one, edit,
  validate, run, or **save as** a named config (table `configs`, editable) - saved configs live in the database,
  templates live in the repository.
* Partial templates are normal: omitted fields take the model defaults. The header comment of each template says what
  matters and what is untested.
* **YAML gotchas**: bare `off/on/yes/no` are booleans in YAML 1.1 (write `mode: "off"`); keep numbers as numbers
  (`3e-4` parses as a string in YAML 1.1 - write `0.0003`); `null` is `null`, `~` also works.
* Environment variables (server level, not job level): `FORGE_HOME`, `FORGE_API_TOKEN`, `FORGE_API_HOST`,
  `FORGE_API_PORT`, `HF_TOKEN`, `FORGE_MAX_CONCURRENT_JOBS`, `FORGE_LOG_LEVEL`, `FORGE_JOB_KILL_GRACE_S`,
  `FORGE_TELEMETRY_INTERVAL_S`, `FORGE_CORS_ORIGINS`, `FORGE_SERVE_BASE_PORT`, `FORGE_FORCE_REF=1` (reference kernels
  only), `FORGE_KERNEL_DEBUG=1`, `FORGE_CONFIGS_DIR` (alternative templates directory).
* The authoritative per-field documentation is the model source and the live schema (`/docs`, the form tooltips):
  this page explains the templates and decisions, not every field.

## 2. Template catalogue

Validated by `tests/test_configs_templates.py` (loads each template into its model, rejects unknown fields at any
nesting level). "Validated" here means *parses and passes the model's checks*, not *has run on hardware*.

### configs/train (job type `train`)

| Template | Purpose | Notes |
|----------|---------|-------|
| `tiny-smoke` | CPU end-to-end smoke test | Built-in tiny model, byte tokenizer, dataset `tiny-corpus` from `configs/dataset/dataset-jsonl.yaml` |
| `llama-8b-full-bf16` | Continued pre-training, all weights, R0 BF16 | FSDP over 8 GPUs by default; ~100 GB state; gated model |
| `qwen-fp8` | Rung R1 FP8 mixed precision | Real speed-up only on sm_89+; emulated otherwise |
| `nvfp4-pilot` | Rung R2 NVFP4 pilot | **Experimental, gated**: needs `allow_unvalidated_fp4`; emulated except on Blackwell; healing phase + diagnostics |

### configs/ternary (job type `ternary_train`, EXPERIMENTAL)

| Template | Purpose | Notes |
|----------|---------|-------|
| `tiny-smoke` | CPU smoke test of ternary QAT | Tiny model, group 32, dataset `tiny-corpus`; watch `ternary/*` metrics |
| `scratch-125m` | 125M model from scratch with ternary QAT | **Not validated on GPU through Forge.** Emulated fake-quant path: no speed-up; thresholds are TARGET values from tern158 |

### configs/finetune (job type `finetune`)

| Template | Purpose |
|----------|---------|
| `llama-8b-lora` | LoRA SFT of an 8B instruct model on chat JSONL; saves adapter and merged model |
| `tiny-lora-smoke` | CPU smoke test of the LoRA path |

### configs/distill (job type `distill`)

| Template | Purpose | Notes |
|----------|---------|-------|
| `offline-logit-7b-to-1.5b` | Top-k cached teacher logits, forward KL | Reusable cache; licence gate (TR-8) |
| `on-policy-7b-to-1.5b` | Student samples, teacher scores, reverse KL | Diversity/collapse monitor on |

### configs/serve (job type `serve`)

| Template | Purpose | Notes |
|----------|---------|-------|
| `serve-8b-w4a16` | OpenAI-compatible server for a W4A16 artifact, FP8 KV | K1 kernel unvalidated; falls back to reference |
| `serve-spec-ngram` | Same plus n-gram speculative decoding | Benefit regime-dependent; verify with G-SPEC |

### configs/dataset (job type `dataset_ingest`)

| Template / file | Purpose |
|-----------------|---------|
| `dataset-jsonl` | Ingest the bundled `sample-corpus.jsonl` as `tiny-corpus` (byte tokenizer) |
| `dataset-hf-streaming` | Example of a streamed Hugging Face source (needs network and the `datasets` extra: `pip install 'forge[hf]'`; check licence). Sources: a local path under `kind: hf` is read as files (`hf_config` ignored); `kind: auto` falls back to a Hub id when the path does not exist; legacy bare ids (`wikitext`) are redirected to `Salesforce/wikitext`; unknown config/split/`text_field` errors list the valid values; the Hub token comes from Settings/`HF_TOKEN`, never the job config; `POST /datasets/source-preview` (UI: **Check source**) resolves a source without ingesting |
| `sample-corpus.jsonl`, `sample-sft.jsonl` | Tiny hand-written sample data for smoke tests (CC0) |

### configs/chips (Plan Appendix B examples)

`h100-sxm-80g`, `a100-sxm-80g`, `rtx-4090-24g`, `b200-blackwell-hypothesis`: **datasheet-only** descriptors
(provenance `vendor`, `tier: C`, measured slots null). They are examples of the schema and inputs for the planner
before a probe exists; they are not measurements. The B200 file lists FP4 block-scaled support as `untested`.

### configs/fleets (Plan Appendix C.4)

`support-triage-v3`: a fleet manifest example. The fleet runtime is not implemented in this release; the file is
documentation of the intended format and is shape-checked only.

### configs/gates

Gate definitions (Plan Appendix C.3) are owned by the evaluation module; see the files in that directory and
`GET /api/v1/gates`.

### configs/recipes (Plan Appendix C.1) and configs/compress (job type `compress`)

`configs/recipes/*.yaml` are **recipes**: `recipe:` name, `inputs:`, an ordered `stages:` list of `- <stage>: {params}`,
`baselines`, `budget`, `policy`, `seeds`. They load with `forge.core.recipe.load_recipe`, are listed by
`GET /api/v1/recipes/templates` (Recipes page) and are validated stage by stage against
`forge.compress.config.STAGE_PARAM_MODELS`. `configs/compress/*.yaml` are the matching **job configs**
(`CompressJobConfig`): they name the model to compress and point at the recipe with `recipe_path` (or embed it inline,
as `tiny-smoke` does).

| Recipe | What it does | Status / caveats |
|--------|--------------|------------------|
| `w4a16-gptq-g128` | calibrate -> GPTQ 4-bit group 128 act-order -> pack for the chip -> FP8 KV -> gate `g-w4` | Plan Appendix C.1. K1 kernel unvalidated on hardware |
| `fp8-w8a8` | calibrate -> FP8 quantisation -> FP8 KV -> gate `g-fp8` | Speed-up only on sm_89+ |
| `nvfp4-ptq` | calibrate -> NVFP4 GPTQ -> pack -> gate `g-w4` | **Experimental**; emulated off Blackwell; gate reused from W4 |
| `ternary-ptq` | ternarize (data-free, group 128) -> pack -> gate `g-t158` | **Experimental**; one-shot and lossy (no recovery training); gate thresholds are TARGET placeholders |
| `ternary-convert` | distill (delegate, ternary QAT + KD) -> ternarize -> pack -> gate `g-t158` | **Experimental**; fill `config.data`; not validated on hardware |
| `prune-distill-8b-to-4b` | calibrate -> structured prune -> built-in recovery distillation -> gate `g-struct` | Pilot budget; real run needs billions of tokens and `mode: delegate` |
| `lowrank` | calibrate -> whitened SVD with rank allocation (+ healing) -> gate `g-struct` | **Experimental**; single-source evidence |
| `delta` | BitDelta-style delta of a fine-tune vs its base -> gate `g-lossless` | **Experimental**; no serving kernel for deltas |

Stage vocabulary: `calibrate, ptq, pack, kv, rotate, prune, lowrank, delta, lossless, eval, gate, distill, finetune`
(`GET /api/v1/recipes/stages` lists what is registered on the running server, with parameter schemas). Appendix C.1
spellings are accepted as aliases (`ptq.group` -> `group_size`, `gate.suite` -> `gate`).

Notes on specific fields: `calib` must name a registered calibration set (`builtin-mix`, `builtin-prose|code|math|chat`,
`synthetic`, `file:<path>`, or one registered through the job's `calibration_texts` / `calibration_files`); `pack.target`
is a measured chip descriptor id or null; `gate.suite` accepts the preset ids `g-lossless, g-fp8, g-w4, g-struct,
g-long, g-spec, g-fp4train, g-t158` (Appendix-C names like `G-W4-v2` resolve to their base gate); `compress.fail_on_gate` marks
the job failed *after* the artifact is registered.

`configs/compress/tiny-smoke.yaml` is the only compression template that runs on a laptop CPU (random tiny model).

### configs/eval (job type `eval`)

| Template | Purpose |
|----------|---------|
| `eval-gw4` | Candidate (W4A16 artifact) vs BF16 parent over distribution, mcq, longctx, longgen and safety, gate `g-w4`, three seeds, parent noise floor from 5 repeats |
| `eval-tiny-smoke` | CPU plumbing check: distribution suite, identical candidate and parent |

The gate presets live in `configs/gates/` (owned by the evaluation module). **Their thresholds are targets to be
recalibrated at Plan Gate G1**, not validated constants. The `candidate.ref` hash in `eval-gw4` is a placeholder you
must replace.

### configs/bench (job type `bench`)

| Template | Purpose |
|----------|---------|
| `bench-gemm-sweep` | Full GEMM sweep over M at 4096x4096 with roofline fields and a dense baseline; set `autotune: true` for the slow search |
| `bench-quick` | Three kernels x three batch sizes, few repeats; used by `scripts/gpu_smoke.sh` / `run_gpu_tests.sh` |
| `bench-cpu-smoke` | Reference paths on CPU to test the harness; **not** kernel measurements |

## 3. Adding or changing a template

1. Put the YAML in `configs/<dir>/` with a header comment (first line must be a comment): what it is, what it needs,
   what is untested or experimental.
2. Use only fields that exist in the model; run `python -m pytest tests/test_configs_templates.py -q`. The test rejects
   unknown fields at every nesting level, missing header comments, unknown recipe stages and invalid stage parameters.
3. Keep secrets out (the test greps for token-like strings). Use `HF_TOKEN` in the environment for gated models.
4. Experimental or gated features must say so in the header comment and, where the model has a gate flag
   (`allow_unvalidated_fp4`), set it explicitly so the user's acknowledgement is visible.
