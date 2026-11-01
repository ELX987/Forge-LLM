# REST API reference

Source of truth for the endpoint list is `docs/ARCHITECTURE.md` section 7; the running server publishes the exact,
machine-readable contract at **`/docs`** (Swagger UI) and **`/openapi.json`**. This page adds conventions and the
SSE event formats. All paths below are relative to the prefix **`/api/v1`**.

## Conventions

* **Auth**: if the server was started with `FORGE_API_TOKEN` (or `forge serve-ui --token`), every `/api/v1` request must
  carry the token as `Authorization: Bearer <token>`, or header `X-Forge-Token: <token>`, or the query parameter
  `?token=<token>` (also `access_token`) - the query form exists because browser `EventSource` cannot set headers, but it
  puts the token in URLs and logs, so prefer headers. `GET /api/v1/system/health` is public (liveness). Empty token = no auth.
* **Content type**: JSON in, JSON out, UTF-8. Uploads use `multipart/form-data`.
* **Errors**: non-2xx responses carry JSON `{"detail": ...}`. Typical codes: `400/422` invalid config or
  validation error, `401` missing/bad token, `404` unknown id, `409` conflict (e.g. artifact pinned, GPU busy),
  `403` licence/gate blocked (`LicenceBlocked`, `GateFailed`), `503` hardware unsupported / engine not ready.
* **Time**: ISO-8601 UTC strings unless a field says `*_ms`/`*_s`.
* **Ids**: job ids, config ids and run ids are opaque strings. Artifact ids are content hashes
  (`blake3:<64 hex>`, or `b2:<...>` when `blake3` is not installed).
* **Config validation**: `POST /jobs` validates `params` against the job type's pydantic model
  (`extra="forbid"`): unknown fields are rejected, not ignored.
* **Pagination**: list endpoints accept `limit` (default varies) and `offset` where meaningful.

## Endpoints

### System and hardware

| Method & path | Purpose |
|---------------|---------|
| `GET /system/info` | Versions, `FORGE_HOME`, store stats summary, capabilities of the current GPU, job counts, server time |
| `GET /system/cuda` `?probe=0\|1&refresh=0\|1&force=0\|1` | CUDA stack check (driver / torch / GPU / Triton compatibility): `{ok, worst, stack, findings[{level, code, message, fix}], recommend}`; cached ~60 s, `refresh=1` bypasses. `probe=1` (default 0) also runs a per-GPU allocation test in a subprocess; it is skipped (static report plus a `probe_skipped` note) while Forge jobs hold GPUs unless `force=1`. Level `info` findings (e.g. `torch_cuda_newer_minor`) are informational only; never 5xx |
| `GET /system/gpus` | Live GPU telemetry list (one sample per GPU) |
| `GET /system/health` | Public liveness probe (no token needed) |
| `GET /system/gpus/stream` | SSE of GPU telemetry |
| `GET /hardware/chips` | Known-chips table and support matrix |
| `GET /hardware/descriptors`, `GET /hardware/descriptors/{id}` | Measured chip descriptors (from `forge probe`) |
| `POST /hardware/probe` | Start a `probe` job, returns a JobRecord |

### Models and datasets

| Method & path | Purpose |
|---------------|---------|
| `GET /models`, `POST /models` | List / create model records |
| `GET/PATCH/DELETE /models/{id}` | Read / update / remove a model record |
| `POST /models/{id}/download` | Start a Hugging Face download job into the models dir |
| `POST /models/register-local` `{path}` | Register an existing local model directory |
| `GET /models/{id}/inspect` | Config, parameter count, KV bytes/token, licence |
| `GET /datasets`, `POST /datasets` | List / create dataset records |
| `GET/DELETE /datasets/{id}` | Read / delete |
| `POST /datasets/upload` | Multipart upload (chunked allowed) |
| `PUT /datasets/{id}/content` | Raw (non-multipart) upload/append of bytes into a managed dataset |
| `POST /datasets/{id}/ingest` | Start a `dataset_ingest` job (tokenise, pack, shard, manifest) |
| `GET /datasets/{id}/preview` | First records / token statistics |

### Schemas, configs, jobs

| Method & path | Purpose |
|---------------|---------|
| `GET /schemas/job-types` | For every job type: `{type, title, description, schema (JSON Schema), defaults, templates:[names], groups:[...]}` |
| `GET /schemas/job-types/{type}` | Same for one type |
| `GET /configs`, `POST /configs` | Saved configs (`id, job_type, name, params_json, created_at, updated_at`) |
| `GET/PUT/DELETE /configs/{id}` | CRUD (saved configs are validated on save) |
| `POST /configs/validate` | Validate `params` against a job type's config model without running anything |
| `GET /configs/templates?type=` | Read-only templates seeded from `configs/<type>/*.yaml` |
| `GET /jobs?status=&type=&limit=` | List jobs |
| `POST /jobs` `{type, name, params, gpus, tags}` | Create and enqueue a job; returns JobRecord |
| `GET /jobs/{id}` | JobRecord: `status` in `queued/running/succeeded/failed/cancelled`, `progress` 0..1, `progress_msg`, `outputs`, `exit_code`, `error`, `run_dir` |
| `POST /jobs/{id}/cancel` | Cooperative stop (SIGTERM, then SIGKILL after the grace period) |
| `DELETE /jobs/{id}` | Remove a finished job record |
| `POST /jobs/{id}/clone` | New queued job with the same spec (use for resume after a failure) |
| `GET /jobs/{id}/logs?tail=&offset=` | Log text from `runs/<id>/log.txt` |
| `GET /jobs/{id}/stream` | SSE for one job (events below) |
| `GET /jobs/{id}/metrics?keys=&since_step=&max_points=` | Time series, see "Metrics response" |

Job types: `probe`, `bench`, `dataset_ingest`, `train`, `finetune`, `distill`, `compress`, `eval`, `serve`, `recipe`,
`model_download`, `gc`. Each type's JSON Schema comes from the pydantic config class listed in ARCHITECTURE section 6.

### Recipes, planner, claims

| Method & path | Purpose |
|---------------|---------|
| `GET /recipes/stages`, `GET /recipes/templates` | Registered recipe stages (the palette); recipe templates from `configs/recipes/` |
| `GET/POST /recipes`, `GET/PUT/DELETE /recipes/{id}` | Saved recipes (YAML in Plan Appendix C.1 form) |
| `POST /recipes/validate` `{yaml\|json}` | Validate and return the canonical form + hash |
| `POST /recipes/{id}/run` | Start a `recipe` job |
| `POST /planner/estimate` `{model, chip\|descriptor_id, quant, kv_dtype, workload}` | Quick fit / bandwidth bound / throughput ceiling |
| `POST /planner/plan` `{model, chip, workload, budget}` | Ranked candidate recipes |
| `POST /planner/train-sim` `{config, chip, n_gpu, price_per_gpu_hour, link, calibration}` | Training memory / step time / cost estimate for a scheme (bf16, fp8, nvfp4_emulated, lora, qlora, ternary_qat, ternary_native). First-order and uncalibrated: an estimate, never a measurement |
| `GET /claims`, `GET /claims/summary`, `GET /claims/{id}`, `PUT /claims` | Claims ledger (see `docs/CLAIMS.md`); `PUT` updates status / notes / linked experiments |

### Artifacts, storage, evals, gates

| Method & path | Purpose |
|---------------|---------|
| `GET /artifacts?kind=` | List artifacts |
| `GET /artifacts/{hash}`, `GET /artifacts/{hash}/lineage` | Manifest; lineage graph `{"nodes": [...], "edges": [...]}` |
| `POST /artifacts/{hash}/pin`, `/unpin`, `/tier` `{to}` | Pin, unpin, move between hot/warm/cold tiers |
| `GET /storage/stats`, `GET /storage/tiers` | Logical vs unique bytes, dedup ratio, per-kind/per-tier |
| `POST /storage/gc` `{dry_run}` | Mark-and-sweep from pins; `dry_run=true` only reports |
| `GET /evals`, `GET /evals/{id}` | EvalReports |
| `POST /evals/compare` `{ids:[...], parent?}` | Table with CIs, within-noise flags, Pareto set |
| `GET /gates`, `GET /gates/{id}` | Gate definitions (G-LOSSLESS, G-FP8, G-W4, G-STRUCT, G-LONG, G-SPEC, G-FP4TRAIN, G-T158) |

### Kernels

| Method & path | Purpose |
|---------------|---------|
| `GET /kernels` | Catalog with `available`, `reasons`, `min_sm` for this machine |
| `GET /kernels/results?kernel=&chip=` | Stored benchmark rows (roofline fields included) |
| `POST /kernels/bench` | Start a `bench` job |
| `GET /kernels/autotune-cache` | Contents of the autotune cache |

### Deployments (inference)

| Method & path | Purpose |
|---------------|---------|
| `GET/POST /deployments` | List / start a deployment (creates a `serve` job) |
| `GET/DELETE /deployments/{id}` | Read / stop and remove |
| `POST /deployments/{id}/stop` | Stop the serve job but keep the record; `GET /deployments/{id}/health` |
| `GET /deployments/{id}/stats` | Engine stats (below) |
| `POST /deployments/{id}/chat` | SSE proxy to the engine's chat streaming |
| `POST /serve/simulate` | Capacity simulator (weights, KV, max concurrency, ideal tokens/s) |

### Checkpoints and global events

| Method & path | Purpose |
|---------------|---------|
| `GET /checkpoints?run=` | Checkpoints of a run |
| `POST /checkpoints/{id}/export` | Export to a safetensors model dir and register it in `/models` |
| `GET /events` | Global SSE (job status, GPU samples) for the dashboard |

The static front end is served at `/`; interactive docs at `/docs`.

## Job creation example

```bash
TOKEN=...   # printed by scripts/run_server.sh
curl -s -X POST http://localhost:8765/api/v1/jobs \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"type":"train","name":"smoke","params":{ "...": "see configs/train/tiny-smoke.yaml" },"gpus":null,"tags":["smoke"]}'
```

`params` is exactly the YAML template converted to JSON. `gpus: null` lets the scheduler choose (CPU jobs share;
GPU jobs get exclusive assignment); `gpus: [0,1]` pins devices via `CUDA_VISIBLE_DEVICES`.

## Metrics response

`GET /jobs/{id}/metrics` returns

```json
{
  "keys": ["loss", "lr", "grad_norm", "tokens_per_s"],
  "points": { "loss": [[step, value, ts], ...], "lr": [[step, value, ts], ...] }
}
```

`ts` is a unix timestamp. Series longer than `max_points` are downsampled server-side (rollups of 1/10/100 steps).
Training jobs emit these keys every N steps: `loss, lr, grad_norm, tokens_per_s, tokens_seen, step_time_ms,
gpu_mem_gb, mfu, update_weight_ratio`, plus FP4 diagnostics `fp4/zero_grad_tile_frac`, `fp4/scale_flush_frac`,
`fp4/sr_bias`, `fp4/loss_gap` when NVFP4 training is on. System series while a job runs:
`sys/gpu{i}/util|mem|temp|power`.

## SSE formats

Streams use `text/event-stream`; each message is `event: <name>` + `data: <json>` + blank line. Keep-alive comment
lines (`: ping`) may appear. Payload shapes for `/jobs/{id}/stream` below were taken from `forge/api/routers/jobs.py`; **the code and the OpenAPI document
are authoritative if this page drifts**.

### `GET /jobs/{id}/stream`

Query parameters: `since_seq` (resume metrics after this sequence number), `replay` (metric rows replayed on connect,
default 5000), `log_tail` (log lines sent on connect, default 100) or `log_offset` (resume the log at a byte offset).

| event | data | Notes |
|-------|------|-------|
| `status` | the public JobRecord (`id, type, status, progress, error, outputs, ...`) | Sent on connect and on every change of status / error / pid / outputs |
| `progress` | `{"progress": 0.42, "message": "...", "status": "running"}` | From `ctx.progress` |
| `metric` | `{"points": {"loss": [[step, value, ts], ...], ...}, "seq": 1234}` | Batched points since the previous event; use `seq` to resume |
| `log` | `{"lines": ["..."], "offset": 4096}` | New log lines; `offset` is the byte offset for `log_offset` |
| `end` | `{"status": "succeeded", "error": null, "exit_code": 0}` | Final event after a terminal status; the server then closes the stream |

Messages also carry an `id:` line (sequence) and `: ping` heartbeat comments every 15 s.

### `GET /system/gpus/stream` and `GET /events`

* `/system/gpus/stream`: GPU telemetry samples (list of per-GPU dicts: index, name, util, memory used/total, temperature,
  power, clock) as events.
* `/events` (dashboard): `hello` on connect, `job` on any job row change, `gpus` when a new telemetry sample arrives.

### `POST /deployments/{id}/chat`

Request: `{"messages": [{"role":"user","content":"..."}], "max_tokens": 256, "temperature": 0.7, "top_p": 0.95, ...}`
(sampling parameters of the engine's `SamplingParams`). Response is an SSE stream of OpenAI-style chunks:
`data: {"id":..., "object":"chat.completion.chunk", "choices":[{"delta":{"content":"..."}, "finish_reason":null}]}`
ending with a chunk carrying `finish_reason` and `data: [DONE]`.

### Deployed model server (OpenAI-compatible, runs in the `serve` job)

`GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions` (`"stream": true` for SSE), `GET /health`,
`GET /stats`, `POST /abort`. `GET /stats` returns running/waiting sequence counts, KV block usage, prefix-cache hit
rate, tokens/s, speculative-acceptance per position, and TTFT/ITL p50/p99. If `api_key` is set in the serve
config, send it as `Authorization: Bearer <api_key>`.
