# User guide

End-to-end walkthroughs for the Forge web app. Everything here is done from the browser; the same operations are
available over the REST API (`docs/API.md`) and, for a few, the `forge` CLI.

> **Status reminder.** The CPU path (tiny models, reference kernels) is unit-tested. Everything that needs an NVIDIA
> GPU - real 8B models, Triton kernels, FP8/NVFP4 speed, CUDA graphs - is **implemented but not yet validated on
> hardware**. Run `bash scripts/run_gpu_tests.sh` on your instance first and read `gpu_report/report.md`
> (`docs/VAST_AI.md`). Treat every first GPU run as a validation run and keep the logs.

Contents: 0 First run - 1 Add a model - 2 Ingest a dataset - 3 Fine-tune with LoRA - 4 Train - 5 Compress
(W4A16 / FP8 / NVFP4) - 6 Evaluate with gates - 7 Serve and chat - 8 Compare runs - 9 Storage and GC -
10 Hardware, kernels and planning - 11 Troubleshooting.

## 0. First run (5 minutes, works on CPU)

```bash
bash scripts/run_server.sh --host 127.0.0.1 --tmux     # prints the API token
# laptop:  ssh -N -L 8765:127.0.0.1:8765 -p <port> root@<host>    -> http://localhost:8765
```

1. Open the UI, go to **Settings**, paste the API token, pick a theme.
2. **Dashboard** shows GPUs (live utilisation, memory, temperature, power), active and queued jobs, storage
   compaction and claims that need attention. No GPU: the GPU panel is empty and jobs run on CPU.
3. Run the end-to-end smoke test, no downloads needed:
   * **New Job -> dataset_ingest**, template `dataset-jsonl` -> Run. (Dataset `tiny-corpus`.)
   * **New Job -> train**, template `tiny-smoke` -> Run. Watch **Job detail**: loss curve, learning rate,
     tokens/s, log tail, progress bar. It finishes in about a minute on CPU.
   * **New Job -> finetune**, template `tiny-lora-smoke` -> Run.
   If these pass, the pipeline (ingest, trainer, metrics streaming, checkpoints, store) works on your host. The
   loss numbers are meaningless - the model is a few hundred thousand random-init parameters.

Everything you launch is a **job**: a subprocess with its own log (`runs/<id>/log.txt`), metrics, progress and
outputs. Jobs can be cancelled (cooperative stop, then kill) and cloned.

## 1. Add a model

Models page -> **Add model**. Three ways:

* **Download from Hugging Face**: enter a repo id (for example `Qwen/Qwen2.5-1.5B`), revision (optional). This
  starts a `model_download` job into `FORGE_HOME/models`. Gated repos (Llama and others) need you to accept the
  licence on the Hub and have `HF_TOKEN` set in the server's environment. Disk: an 8B bf16 model is ~16 GB.
* **Register a local directory**: a HF-style folder (`config.json`, `*.safetensors`, tokenizer files). Nothing is copied.
* **From a checkpoint or compression output**: Jobs produce models automatically (fine-tune `save_merged`, compress
  output) and register them; or *Checkpoints -> Export*.

Open a model -> **Inspect**: architecture, parameter count, KV bytes per token (this drives serving capacity),
licence and *distill allowed* flag. **Set the licence and distillation permission correctly**: the distillation
gate refuses teachers that are not permissive or attested (Plan TR-8). Architectures Forge's native decoder does
not implement load through the Hugging Face backend with fewer features (ADR 0004).

## 2. Ingest a dataset

Datasets page -> **Upload** (jsonl / txt / parquet; chunked upload is fine for big files) or point at files already
on the server, then **Ingest** (a `dataset_ingest` job). Key settings (hover the field labels for descriptions):

| Setting | Why it matters |
|---------|----------------|
| `tokenizer` | Must be the tokenizer of the model you will train/distill. The dataset records the tokenizer hash; mismatches are caught at train time. |
| `sources[].text_field` | Which JSON field holds the text (dotted paths ok). |
| `dedup_exact` / `dedup_minhash` | Exact and near-duplicate removal. Report is saved in the manifest. |
| `val_fraction` | Held-out documents chosen by content hash, stable across re-ingests. |
| `contamination.eval_files` | n-gram overlap scan against your evaluation sets. `action: drop` removes hits. |
| `licence` | Recorded in the manifest and in lineage. |

Output: 256 MiB token shards plus a manifest (token counts, mixture, dedup report) stored content-addressed.
**Preview** shows the first records and statistics. For SFT you do not ingest: chat/instruction JSONL files are
used directly by the fine-tune job (`data.train_files`; formats: `messages`, ShareGPT, Alpaca, prompt/completion,
prompt/chosen/rejected).

## 3. Fine-tune with LoRA

**New Job -> finetune**, start from template `llama-8b-lora` (or **Fine-tune** shortcut wizard). Fill:

1. `model.path`: a registered model id, local dir or HF repo id.
2. `data.train_files`: your chat JSONL; `data.template`: the chat template matching the model family
   (`llama3`, `chatml`, `mistral`, or `tokenizer` to use the tokenizer's own template).
3. `method`: `lora` (bf16 base), `qlora` (4-bit NF4 base - least memory), or `full`.
4. `lora.r` / `alpha` / `target_modules`; `optim.lr` (2e-4 is a common LoRA starting point);
   `micro_batch_size` x `grad_accum_steps` = effective batch.
5. `budget` (steps / epochs / minutes), `checkpoint.every_steps`, `resume: auto` on rented/interruptible hardware.
6. Press **Validate** (schema + cross-field checks), pick GPU(s), **Run**.

Before launching on a big model use **Recipes -> estimate** (or `POST /planner/estimate`) to see whether weights +
optimiser + activations fit your VRAM. In the monitor watch loss, grad norm, tokens/s, MFU, GPU memory, and the
update/weight ratio. NaN or loss spikes trigger an automatic rollback to the last checkpoint (settings under `stability`).

Outputs: the adapter (`save_adapter`) and a merged full model (`save_merged`) registered under Models, both with
lineage back to the base model and dataset. Cancel with the button; the trainer stops at the next step boundary
and writes a checkpoint. **Clone** the job to resume with edits.

## 4. Train (continued pre-training, precision ladder)

**New Job -> train**. Same monitor, plus the precision ladder (Plan TR-1):

* `precision.rung: R0` - BF16 autocast (reference; start here and keep an R0 run as the comparison).
* `R1` - FP8 mixed precision (needs `sm_89`+ for speed; otherwise emulated and slower). Template `qwen-fp8`.
* `R2` - NVFP4 pilot, **experimental and gated** (`allow_unvalidated_fp4`): emulated on non-Blackwell GPUs.
  Template `nvfp4-pilot`. The monitor adds the FP4 diagnostics panel: zero-gradient tile fraction, scale flush
  fraction, stochastic-rounding bias, loss gap vs the high-precision shadow.

Multi-GPU: `parallel.strategy: ddp | fsdp`, `nproc_per_node` = number of GPUs. A full 8B BF16 AdamW run needs
roughly 100 GB for state before activations (see the `llama-8b-full-bf16` template comments).

## 5. Compress: W4A16, FP8, NVFP4 (and others)

A compression run is a **recipe**: an ordered list of stages, each cached by content hash, so re-running with a
changed last stage only recomputes that stage. Templates (see `docs/CONFIG_REFERENCE.md`):

| Goal | Template | Pipeline |
|------|----------|----------|
| 4-bit weights, 16-bit activations (decode speed + memory) | `w4a16-gptq-g128` | calibrate -> GPTQ (4-bit, group 128, act-order) -> pack (layout for your chip) -> optional FP8 KV -> gate G-W4 |
| FP8 weights and activations (production default on sm_89+) | `fp8-w8a8` | calibrate -> FP8 quantise -> gate G-FP8 |
| NVFP4 post-training quantisation | `nvfp4-ptq` | calibrate -> NVFP4 PTQ -> pack -> gate. On non-Blackwell GPUs this is storage + emulated compute |
| 8B -> ~4B | `prune-distill-8b-to-4b` | calibrate -> importance -> structured prune -> distill (licence-gated) -> gate G-STRUCT |
| Low-rank factorisation | `lowrank` | calibrate -> whitened SVD with rank allocation -> heal -> gate |
| Fine-tune deltas | `delta` | low-rank or 1-bit-sign delta against a base model |

Steps: **New Job -> compress** (or **Recipes -> builder**), choose the template, set `model` (the parent), `calib`
(calibration set: use text that resembles production prompts - calibration sensitivity is a documented risk),
and the target chip (`FORGE_HOME` descriptor from *Hardware -> Probe*, or `auto`). **Run**.

Outputs are **artifacts** in the content-addressed store: packed weights, scales, zeros and a manifest carrying the
`layout_id`, recipe hash, calibration hash and parents. Compare artifact sizes in *Storage*; dedup shows shared tensors.

Expect: W4A16 ~3.5-4x smaller weights than BF16; FP8 W8A8 2x; NVFP4 ~3.5x (4.5 bits/value). Quality and speed
claims from the papers are **paper-reported** until you evaluate and benchmark them here (section 6, section 10).

## 6. Evaluate with gates

**New Job -> eval**, template `eval-gw4`. An eval job compares a **candidate** artifact with its **parent** (the
uncompressed model) over suites and applies a gate:

* Suites: `distribution` (KL, p99 KL, top-1 flip rate, ECE on held-out text), `mcq` (multiple choice from local
  jsonl), `longgen` (long reasoning/generation), `longctx` (needle / multi-key retrieval at your deployment length),
  `safety` (probe set), `contamination` (n-gram overlap), `throughput` (serving engine).
* **Noise floor**: the parent is evaluated repeatedly (`n` seeds) so the report can say whether a difference exceeds
  run-to-run noise. Differences inside the floor are marked **within noise** and are not ranked.
* **Gates** (`G-LOSSLESS`, `G-FP8`, `G-W4`, `G-STRUCT`, `G-LONG`, `G-SPEC`, `G-FP4TRAIN`) turn metrics into a
  verdict: **pass**, **fail** or **inconclusive** with reasons. Inconclusive means "not enough evidence" (for example
  CIs too wide) - collect more samples rather than treating it as a pass.

Results are `EvalReport`s under **Compare / Evals** with confidence intervals and provenance badges.
The bundled sample jsonl and tiny models only exercise the machinery; a real verdict needs real models, real eval
sets and enough examples.

## 7. Serve and chat

**Inference -> Start deployment** (a `serve` job; templates `serve-8b-w4a16`, `serve-spec-ngram`):

1. `model`: a registered model or artifact (quantised artifacts are used as packed).
2. KV settings: `kv_dtype` (`auto`, `fp8_e4m3`, `int8`...), `max_model_len`, `max_num_seqs`, `gpu_mem_util`,
   `prefix_caching`, `chunked_prefill`, `host_kv_gb` for a host-memory tier.
3. Speculation: `off`, `ngram` (no extra model), `draft` (small draft model sharing the vocabulary), `auto`.
   Output distribution is unchanged; speed benefit depends on batch size and workload (check the stats panel).
4. Use the **capacity simulator** first: it computes weights + KV bytes per token from the model's config and tells
   you how many concurrent sequences fit and the bandwidth-bound ideal tokens/s.

Open the **chat playground**: pick the deployment, set temperature / top-p / top-k / max tokens, chat with streaming.
The side panel shows running and waiting sequences, KV block usage, prefix-cache hit rate, tokens/s, TTFT and
inter-token latency (p50/p99) and, with speculation, acceptance per draft position.

Scripts and apps can call the deployment directly: it speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`) on its own port. Tunnel that port too, and set `api_key` if it is reachable by others. Stop the
deployment from the Inference page (cancels the job and frees the GPU).

## 8. Compare runs

* **Training runs**: Jobs list -> select several -> **Compare**: overlaid loss/lr/throughput curves (zoomable,
  exportable as PNG/CSV) keyed by step.
* **Artifacts / evals**: **Compare** page takes eval ids and an optional parent: table of metrics with confidence
  intervals, *within noise* markers, gate verdicts, size and bandwidth columns, and a Pareto chart (quality vs bytes vs
  speed). Values with overlapping uncertainty are not ordered.
* **Kernel runs**: Kernel lab shows benchmark rows per chip/kernel/shape.
* Every table cell carries a **provenance badge**: measured here, reproduced, paper-reported, derived, vendor,
  target, untested. Do not treat `paper_reported`/`target` numbers as results.
* Deep links: the URL captures the page and selection; share it with someone connected to the same server.

## 9. Manage storage and GC

**Storage** page: logical vs unique bytes (dedup ratio), bytes per kind and per tier (hot/warm/cold), and a lineage
graph for any artifact (which model, recipe, calibration set and teacher produced it).

* **Pin** artifacts you must keep (final models, best checkpoints, eval baselines). Unpinned, unreferenced data is
  garbage.
* **Tiers**: demote cold artifacts to the cold directory (can be a slower disk) and promote when needed.
* **GC**: *dry run* first - it lists exactly what would be deleted; then run it. See
  `docs/runbooks/store-gc-drill.md` and `docs/runbooks/restore-drill.md`.
* Checkpoints are delta-chained and chunk-deduplicated, so keeping many is cheap; retention (`keep_last`,
  `keep_best`, keyframes, pre-anneal) is configured per job.
* Free disk shows on the Dashboard; model/dataset deletion on their pages removes the record and (optionally) files.

## 10. Hardware, kernels, planning

* **Hardware**: *Probe* runs the capability probe and micro-benchmarks (HBM bandwidth, matrix throughput per dtype,
  shared-memory bandwidth, launch latency) and saves a measured chip descriptor with provenance. Measured values
  replace datasheet values in the planner. The support matrix shows what each GPU generation can do natively.
* **Kernel lab**: catalogue with availability on this machine, benchmark results with roofline fields, autotune cache.
  Start a **bench** job (`bench-gemm-sweep`). Remember that nothing is performance-validated until you have run it
  (`docs/KERNELS.md`, validation table).
* **Recipes -> planner**: given model, chip and workload it estimates fit, bandwidth bound and a ceiling for tokens/s,
  and ranks candidate recipes. These are models, not measurements; the Claims ledger tracks which paper claims
  they rely on and whether we have reproduced them.

## 11. Troubleshooting

* Job stuck `queued`: another job holds the GPU (`FORGE_MAX_CONCURRENT_JOBS=1`), or no GPU is free; check Dashboard.
* Job `failed` right away: open the log tail; configuration errors are reported field by field. Unknown fields are
  rejected.
* YAML gotcha: bare `off`, `on`, `yes`, `no` become booleans. Quote them (`mode: "off"`).
* CUDA out of memory: lower `micro_batch_size`, enable `activation_checkpointing`, use QLoRA, FSDP, or a smaller `seq_len`.
* After a server restart, jobs that were running are marked failed; clone them and set `checkpoint.resume: auto`.
* Anything GPU-specific that behaves strangely: reproduce with `FORGE_FORCE_REF=1` (reference kernels) and send the
  report as described in `docs/VAST_AI.md` section 7.
