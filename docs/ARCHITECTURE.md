# Forge LLM — Architecture & Build Contract

This is the binding contract between all modules. Source of truth for *why*: `plan-and-docs/Forge_LLM_System_Plan.pdf`
(referred to as "the Plan", with requirement IDs such as KR-2, CP-1, TR-3, SV-3, EV-5, DT-1, C-0xx).
Source of truth for *what/how to import*: this file. If you must deviate, record it in `docs/adr/NNNN-title.md`.

## 0. Goals (from the user)

1. **Kernel setup** for training, inferencing and fine-tuning given model weights, optimal on most consumer + datacenter
   NVIDIA GPUs (sm_70 … sm_121). Triton kernels (portable, autotuned) + pure-torch reference oracles + fallbacks.
2. **Full-stack Python web app** (FastAPI + dependency-free vanilla-JS front end, no node build) to: configure every
   parameter of every process, launch/monitor training / fine-tuning / compression / eval / inference, show live progress
   and inference status, and store/manage models, datasets, checkpoints, artifacts, runs, configs.
3. Supporting components from the Plan: content-addressed store, hardware profiler, cost model/planner, eval harness
   with gates, compression engine, serving engine, claims ledger, docs, config templates, deploy scripts.

Runs on **Linux + NVIDIA** (user rents vast.ai). Development machine is Windows w/o NVIDIA ⇒ **everything must import
and unit-test on CPU-only torch**; GPU-only code is guarded (`torch.cuda.is_available()`, lazy `import triton`) and has
tests under `tests_gpu/` that only run on a GPU host. Never claim GPU code was run locally. Python ≥ 3.10 syntax only
(no 3.12+ only features), `from __future__ import annotations` at top of every module.

## 1. Repository layout & ownership (one owner per directory — do not edit others' files)

```
pyproject.toml README.md LICENSE(Apache-2.0) .gitignore .gitattributes Makefile   [scaffold]
forge/
  __init__.py  __version__.py                                                      [core]
  core/        schemas, hashing, paths, settings, errors, registry, db, context, recipe engine, env capture   [core]
  store/       content-addressed store, chunking, manifests, lineage, lossless codec, checkpoints, GC          [store]
  hw/          chip DB, descriptor, capability probe, microbench, gpu telemetry                                 [hw]
  kernels/     specs/*.yaml, ref/ (oracles), triton/ (kernels), layouts/ (packers+deriver), autotune/, ops_*.py, bench.py
                 gemm side: [kernels_gemm]   attention/moe/fp4/utility side: [kernels_attn]
  formats/     fake-quant numerics: int, fp8, e2m1, nvfp4, mxfp4, stochastic rounding                           [compress]
  models/      forge decoder-only transformer (dense+MoE), HF safetensors loader/saver, linear-module registry,
               kv-cache classes, LoRA                                                                           [models]
  compress/    calibrate, ptq (rtn/gptq/awq/smoothquant), kvquant, rotation, prune, lowrank, delta, lossless,
               qat stage wrappers, recipe stage registration                                                    [compress]
  train/       data (tokenise/shard/loaders), trainer (precision ladder), finetune (full/LoRA), distill, diagnostics [train]
  serve/       engine, scheduler, kv block manager, prefix cache, speculative decoding, capacity planner, http server [serve]
  eval/        metrics, stats, suites, gates, harness                                                           [eval_planner]
  planner/     costmodel, rulebase, search, claims ledger (+ seeded claims)                                     [eval_planner]
  telemetry/   metrics store (sqlite + optional parquet), log capture                                           [jobs_api]
  jobs/        job manager, worker process, job-type registry, handlers                                         [jobs_api]
  api/         FastAPI app + routers                                                                            [jobs_api]
  web/static/  index.html, css, ES-module JS pages, canvas charts                                               [web]
  cli.py       `forge` CLI (typer-free: argparse)                                                               [jobs_api]
configs/       templates: recipes/, train/, finetune/, distill/, serve/, eval/, gates/, chips/, fleets/, datasets/  [docs_ops]
scripts/       vast_setup.sh, run_server.sh, gpu_smoke.sh, run_gpu_tests.sh, tunnel notes                       [docs_ops]
docker/        Dockerfile(.cuda), docker-compose.yml                                                            [docs_ops]
docs/          this file, adr/, runbooks/, VAST_AI.md, USER_GUIDE.md, KERNELS.md, API.md, CONFIG_REFERENCE.md   [docs_ops]
tests/         CPU-only pytest, files named test_<module>_*.py, owner = module owner                            [each]
tests_gpu/     GPU-only pytest (skipped w/o CUDA), test_<module>_*.py                                           [each]
.github/workflows/ci.yml                                                                                        [docs_ops]
```

## 2. Conventions (all modules)

- Typed, `pydantic` v2 models for every config; every config class has docstring + `Field(description=..., ge/le...)`
  because **the UI renders forms from the JSON Schema** of these models (see §6). Defaults must be sane.
- No global mutable state except explicit registries. No network calls at import time. Heavy imports (torch, triton,
  transformers, pynvml, pyarrow) are lazy inside functions or guarded by try/except so `import forge.<mod>` is cheap and
  works without them (torch itself may be assumed present in CPU test env; triton / pynvml / pyarrow may NOT be).
- Logging via `logging.getLogger("forge.<module>")`. Long-running work receives a `RunContext` (§4.3) and reports through it.
- Determinism: anything claimed "deterministic stage" must be bit-reproducible; seeds go through `forge.core.env.seed_everything`.
- Every number surfaced to the UI that comes from a paper is tagged with a `Provenance` (measured/reproduced/paper_reported/derived/vendor).
- Errors: raise subclasses of `forge.core.errors.ForgeError`.
- Tests: each module ships CPU tests for everything that does not need an NVIDIA GPU; use tiny synthetic models
  (e.g. 2-layer, hidden 64) — never download weights in tests. `pytest -q` from repo root must pass on CPU.
- Kernel correctness: every Triton kernel has (a) a pure-torch reference in `kernels/ref/` (the oracle, runnable on CPU),
  (b) a launcher in `kernels/ops_*.py` that falls back to the reference/dequant path when CUDA/Triton/capability is missing,
  (c) a GPU test in `tests_gpu/` comparing against the oracle over the edge-case list of Plan §6.7.
- Code written for a GPU that cannot be run here must be extra careful: keep Triton kernels simple/standard
  (`tl.dot`, `tl.load/store` with masks, `tl.constexpr` tile params), avoid exotic APIs unless version-guarded, and
  document the minimum triton/torch version in the module docstring. Prefer correctness over cleverness.

## 3. GPU support matrix (kernels MUST consult `forge.hw.capabilities`)

| sm   | examples                                   | bf16 | fp8 mma | fp4 block-scaled mma | notes |
|------|--------------------------------------------|------|---------|----------------------|-------|
| 70   | V100                                       | emu  | no      | no                   | fp16 only; Triton ok |
| 75   | T4, RTX 20xx                               | emu  | no      | no                   | fp16 only |
| 80   | A100, A30                                  | yes  | no      | no                   | cp.async, 2:4 sparse |
| 86   | RTX 30xx, A10, A40, RTX A-series           | yes  | no      | no                   | |
| 87   | Jetson Orin                                | yes  | no      | no                   | |
| 89   | RTX 40xx, L4, L40S, RTX 6000 Ada           | yes  | yes     | no                   | |
| 90   | H100, H200, GH200                          | yes  | yes     | no                   | TMA, wgmma |
| 100  | B200, GB200                                | yes  | yes     | yes (NVFP4/MXFP4)    | tcgen05, tensor memory |
| 103  | B300                                       | yes  | yes     | yes                  | |
| 120  | RTX 50xx, RTX PRO 6000 Blackwell           | yes  | yes     | yes (verify by probe)| consumer Blackwell |
| 121  | GB10 (DGX Spark)                           | yes  | yes     | yes (verify by probe)| |

Everything "native" is a hypothesis confirmed by the probe (Plan HW-2). Unknown / older ⇒ Tier C portable fallback.

## 4. Core interfaces (module `forge.core`, owner: core). Others import these exactly.

### 4.1 `forge.core.hashing`
`hash_bytes(b: bytes) -> str`, `hash_file(path) -> str`, `hash_json(obj) -> str` (canonical: sorted keys, no whitespace,
floats via repr), class `Hasher` (`update(b)`, `hexdigest() -> str`). Digest strings look like `blake3:<64hex>`; if the `blake3`
package is missing fall back to `blake2b-256` and prefix `b2:` — never mix inside one store (store records its algo).

### 4.2 `forge.core.schemas` (pydantic v2; all `model_config = ConfigDict(extra="forbid")` unless noted)
- `Provenance(str, Enum)`: measured, reproduced, paper_reported, derived, vendor, target, untested.
- `MetricValue{value: float, ci_low: float|None, ci_high: float|None, n: int|None, provenance: Provenance, within_noise: bool|None}`.
- `ChipDescriptor` (Plan Appendix B): id, fingerprint, vendor, family, name, sm: int|None (e.g. 90), compute_capability: str,
  compute_units, clock_mhz, memory{hbm_gb, hbm_bw_gbs{datasheet, measured}, l2_mb, transaction_bytes, max_vector_load_bytes},
  on_chip{smem_per_cu_kib, smem_banks, bank_width_bytes, regs_per_cu, max_regs_per_thread, async_copy: list[str]},
  matrix_units: list[MatrixUnit{dtype, atom, peak_tflops{datasheet, measured}, block_scaled: dict|None}],
  sparsity{two_four}, alu{...}, launch_latency_us, interconnect{...}, derived{ridge_flop_per_byte{dtype: float}},
  power_limit_w, tier: Literal["A","B","C"], probe_provenance: dict[str, Provenance]. Helper `supports(dtype:str)->bool`.
- `ModelRef{id, name, source: Literal["local","hf"], path_or_repo, revision, hash, licence, distill_allowed: bool|None, params_b}`.
- `Workload{context_lengths: list[int], batch_sizes: list[int], ttft_slo_ms, itl_slo_ms, prefix_reuse: float, task_mix: dict}`.
- `StageSpec{name: str, params: dict}`, `Recipe{name, inputs: dict, stages: list[StageSpec], baselines: list[str], budget: dict, policy: dict, seeds: list[int]}`
  with `Recipe.hash()` (pure function of canonical content).
- `TensorEntry{name, dtype, shape, hash, layout_id: str|None, role: Literal["payload","scale","zero","tensor_scale","perm","codebook","other"]}`,
  `ArtifactManifest{hash (computed), kind, model_config: dict, tensors: list[TensorEntry], recipe_hash, parents: list[str], calib_hash, licence, hardware_families: list[str], eval_report_hashes: list[str], created_at, extra: dict}`.
- `EvalReport{id, artifact_hash, parent_hash, suite_versions, metrics: dict[str, MetricValue], gate: GateVerdict|None, noise_floor: dict, created_at, env: dict}`.
  `GateVerdict{gate_id, verdict: Literal["pass","fail","inconclusive"], reasons: list[str], details: dict}`.
- `Claim{id, source_paper: str, location: str, statement: str, status: Literal["untested","reproduced","contradicted","partial"], prior: Literal["trusted","untrusted"], linked_experiments: list[str], depends_rules: list[str], notes}`.
- `JobStatus(str, Enum)`: queued, running, succeeded, failed, cancelled. `JobSpec{type: str, name: str, params: dict, gpus: list[int]|None, tags: list[str]}`.
  `JobRecord{id, spec, status, created_at, started_at, finished_at, pid, exit_code, progress: float, progress_msg, error, outputs: dict, run_dir}`.
- `Deployment{id, name, artifact_ref, serve_config: dict, job_id, port, status, created_at}`.
- `DatasetManifest{id, name, tokenizer_hash, shards: list[{hash, tokens, docs}], mixture: dict, filtering_version, licence, dedup_report: dict, total_tokens}`.

### 4.3 `forge.core.context`
```python
class RunContext(Protocol):
    run_id: str; run_dir: Path; params: dict
    def log(self, msg: str, level: str = "info") -> None
    def metric(self, step: int, **values: float) -> None          # streamed to telemetry + UI
    def progress(self, frac: float, msg: str = "") -> None         # 0..1
    def should_stop(self) -> bool                                  # cooperative cancel
    def set_output(self, key: str, value: Any) -> None            # shown in job record `outputs`
    @property store(self) -> "ArtifactStore | None"
class NullContext(RunContext)   # in-memory, for tests & scripts: .logs, .metrics (list of (step, dict)), .outputs, .stop flag
```

### 4.4 other core modules
- `forge.core.paths`: `forge_home()` (env `FORGE_HOME`, default `~/.forge`), `data_dir(name)`, `runs_dir()`, `store_dir()`, `models_dir()`, `datasets_dir()`, `cache_dir()` — all create-on-demand.
- `forge.core.settings`: `get_settings()` → dataclass: home, db_url (`sqlite:///<home>/forge.db`), api_host/port, api_token (env `FORGE_API_TOKEN`, optional), hf_token (env `HF_TOKEN`), max_concurrent_jobs, etc.
- `forge.core.errors`: `ForgeError`, `StoreError`, `IntegrityError`, `GateFailed`, `LicenceBlocked`, `HardwareUnsupported`, `KernelCorrectnessError`, `ConfigError`, `NotFound`.
- `forge.core.registry`: `register(kind: str, name: str, *, meta: dict|None=None)` decorator, `get(kind, name)`, `names(kind)`, `describe(kind)`. Kinds in use: `"stage"` (recipe stages), `"job_type"`, `"kernel"`, `"gate"`, `"suite"`, `"chip"`.
- `forge.core.db`: `get_db() -> Database` (sqlite3 stdlib, WAL, thread-safe via per-thread connections), methods `execute(sql, params=())`, `query(sql, params=()) -> list[dict]`, `query_one`, `executemany`, `transaction()` ctx-manager; `register_migration(module: str, version: int, sql: str)`; `migrate()` applies all registered, idempotent, tracked in `_migrations`. Each module registers its own tables at import time of its `db.py`.
- `forge.core.env`: `seed_everything(seed:int)`, `capture_environment() -> dict` (python, torch, cuda, triton, driver, GPU fingerprint, git sha if any, pip versions of key libs).
- `forge.core.recipe`: `RecipeEngine(store, *, cache=True)`; `engine.run(recipe: Recipe, ctx: RunContext) -> dict` executes stages in order: stage fn signature
  `fn(inputs: dict[str, Any], params: dict, ctx: RunContext) -> dict[str, Any]` (registered with `@register("stage", "<name>")`).
  Stage output hash cache key = hash(stage name, params, input hashes, code_version attr). Re-run of a cached stage is instant; changing one stage reruns downstream only. Emits lineage edges to the store when given. `load_recipe(path|str) -> Recipe` (YAML), `dump_recipe`.
- `forge.core.jsonschema`: `schema_for(model_cls) -> dict` (pydantic JSON schema with `$defs` inlined + `x-group`, `x-advanced`, `x-unit` hints from `Field(json_schema_extra=...)`) and `defaults_for(model_cls) -> dict`. The UI/API use ONLY these.

## 5. Module public APIs (cross-module imports must use exactly these names)

### 5.1 `forge.store` [store]
```python
class ArtifactStore:            # content-addressed, chunk-deduplicated, tiered(hot/warm/cold dirs), sqlite lineage
    def __init__(self, root: Path|None=None)               # default store_dir()
    def put_bytes(data: bytes, *, kind="blob") -> str      # returns content hash; chunked if > 4 MiB (content-defined, 1–4 MiB)
    def put_file(path, *, kind="blob") -> str
    def get_bytes(h) -> bytes ; def open(h) -> BinaryIO ; def materialize(h, dest: Path) -> Path
    def has(h) -> bool ; def verify(h) -> bool ; def size(h) -> int ; def stored_size(h) -> int   # logical vs unique-on-disk
    def put_manifest(m: ArtifactManifest) -> str ; def get_manifest(h) -> ArtifactManifest
    def put_tensor_dir(dir: Path, *, kind, recipe_hash=None, parents=(), extra=None) -> str   # safetensors dir -> per-tensor dedup + manifest
    def materialize_artifact(manifest_hash, dest: Path) -> Path                                # reconstruct a loadable dir
    def add_edge(child, parent, recipe_hash=None, relation="derived_from") ; def lineage(h, depth=None) -> {"nodes": [...], "edges": [...]}
    def pin(h, reason) ; def unpin(h) ; def list_artifacts(kind=None, limit=..., offset=...) -> list[dict]
    def stats() -> dict   # logical bytes, unique bytes, dedup ratio, per-kind, per-tier, compaction vs naive
    def gc(dry_run=True) -> GCReport   # mark-and-sweep from pins; never collects delta-chain keyframes with live deltas
    def demote(h, to_tier) ; def promote(h) ; def tier_of(h)
class CheckpointWriter/CheckpointReader (store/checkpoints.py): save(step, state_dict(s), meta, keyframe_every=N) delta-chained (xor+zstd) with chunk dedup; load(step|"latest"|"best") bit-exact; retention policy (keyframes, best-val, pre-anneal, final).
store/lossless.py: encode_tensor(t: torch.Tensor | bytes+dtype) -> bytes, decode_tensor(...) bit-exact; BF16/FP16/FP8 exponent-split + zstd/ANS-ish; byte-grouped zstd baseline; `verify_roundtrip`.
```
`get_store()` singleton helper in `forge.store.__init__`.

### 5.2 `forge.hw` [hw]
```python
forge.hw.chips: KNOWN_CHIPS: dict[str, dict]  # name patterns -> datasheet specs (sm, SMs, hbm_gb, bw, tflops per dtype, smem, l2, nvlink...) for ALL families in §3 incl. consumer
forge.hw.capabilities: 
    @dataclass Capabilities: sm:int; name; has_bf16; has_fp8; has_fp4_block_scaled; has_cp_async; has_tma; has_two_four; smem_per_sm_kib; num_sms; total_mem_gb; tier:str
    def current_capabilities(device=0) -> Capabilities     # works w/o GPU (returns cpu caps, tier "C")
    def list_gpus() -> list[dict]
forge.hw.probe: def run_probe(ctx: RunContext|None=None, device=0, quick=False) -> ChipDescriptor   # fills descriptor, runs microbench; no GPU -> descriptor tier C from CPU info
forge.hw.microbench: stream_copy_bw(), mma_throughput(dtype), smem_bw(), launch_latency(), dequant_alu_throughput() — all GPU, guarded
forge.hw.descriptor_store: save_descriptor(d) / load_descriptor(id) / list_descriptors()  (sqlite + yaml in home)
forge.hw.telemetry: GpuSampler: sample() -> list[dict]{index,name,util,mem_used_mb,mem_total_mb,temp_c,power_w,clock_mhz,...} via pynvml, falling back to `nvidia-smi --query-gpu`, falling back to [] 
```

### 5.3 `forge.formats` [compress] — numerics shared by PTQ, QAT, kernel oracles
`formats.int_quant`: `quantize_int(w, bits, group_size, sym=True, dim=-1) -> (q, scale, zero)`, `dequantize_int(...)`;
`formats.fp8`: `to_fp8(x, fmt="e4m3"|"e5m2", scale=None) -> (x8, scale)`, `from_fp8`; `formats.fp4`: `E2M1_GRID`, `quantize_nvfp4(x, ...) -> NVFP4Tensor{payload_u8, block_scales_e4m3, tensor_scale, shape}`, `dequantize_nvfp4`, `quantize_mxfp4`, `dequantize_mxfp4`, `pack_fp4/unpack_fp4`, `stochastic_round_fp4`; round-to-nearest-even exactly as Plan §6 / paper P5.

### 5.4 `forge.models` [models]
```python
ModelConfig (pydantic): arch, vocab_size, hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads, head_dim, rope_theta, rope_scaling|None,
    rms_norm_eps, tie_word_embeddings, max_position_embeddings, attention_bias, qk_norm, sliding_window|None, moe: MoEConfig|None, hidden_act="silu";
    ModelConfig.from_hf_config(dict) ; .to_hf_config() ; .kv_bytes_per_token(kv_bits=16) ; .param_count()
ForgeLM(nn.Module): forward(input_ids, positions=None, kv_cache=None, attn_meta=None, labels=None, return_hidden=False, logits_to_keep=None) -> ModelOutput(logits, loss, hidden_states, aux_loss)
   supports archs: llama, mistral, qwen2, qwen3, (mixtral / qwen2_moe / qwen3_moe style MoE), gemma optional
   .layers[i].self_attn / .mlp modules use Linear modules from the linear registry so they can be swapped
models.loader: load_model(path_or_repo, *, dtype="bfloat16", device="cuda"|"cpu"|"meta", revision=None, quant=None, backend="forge"|"hf") -> (model, tokenizer_or_None)
    save_model(model, dir, *, as_dtype=None, shard_gb=4) (safetensors + config.json HF-compatible), load_tokenizer(path_or_repo)
models.linear: LinearBase protocol; `DenseLinear`, `register_linear(kind)(cls)`; `replace_linears(model, factory: Callable[[str, nn.Linear], nn.Module|None])`; `PackedLinear` kinds registered by compress/kernels: "w4a16", "w8a16", "fp8", "nvfp4", "lora"
models.lora: LoRALinear, add_lora(model, r, alpha, dropout, target_modules) ; merge_lora(model) ; lora_state_dict(model) ; load_lora_state_dict
models.kvcache: KVCacheBase(Protocol): update(layer, k, v, positions...) ; DenseKVCache(batch, max_len, cfg, dtype, kv_dtype in {bf16,fp16,fp8_e4m3,int8,int4}) ; PagedKVView(block_tables, seq_lens, pool)  (pool itself in serve)
models.tiny: make_tiny_config(**ov), build_tiny_model(seed) -> ForgeLM  # for tests everywhere
```
### 5.5 `forge.kernels` [kernels_gemm + kernels_attn]
```python
kernels/ref/*.py   torch oracles:  w4a16_gemm_ref, w8a16_gemm_ref, fp8_gemm_ref, fp4_block_gemm_ref, paged_decode_attention_ref, sparse_attention_ref,
                   tree_attention_ref, moe_grouped_gemm_ref, moe_route_ref, hadamard_ref, stochastic_round_ref, rmsnorm_quant_ref, rope_kv_write_ref, ...
kernels/layouts:   derive_layout(spec: KernelSpec, chip: ChipDescriptor|Capabilities) -> Layout(id, perm, swizzle, scales, partition, tile defaults)
                   pack_w4(q_int4[N,K] (uint8 values 0..15), scales[N,K/G], zeros|None, group_size, layout) -> PackedW4{qweight:int32, scales, zeros, layout_id, meta}
                   unpack_w4(PackedW4) -> (q, scales, zeros)   # exact inverse (property-tested)
                   pack_nvfp4_scales(...)/unpack (128x4 swizzle) ; repack(packed, new_layout)
kernels/spec.py:   KernelSpec (pydantic; Appendix C.2 YAML), load_spec(name) from kernels/specs/*.yaml
kernels/autotune:  Autotuner(cache_path).tune(spec, shapes, run_fn, ref_fn) -> best config; cache key (chip fingerprint, driver/triton version, op, dtypes, shape bucket, layout_id);
                   refuses to cache candidates failing oracle; median/IQR over runs, warmup, L2 flush; throttle detect; cache JSON in home/autotune/
kernels/ops_gemm.py   w4a16_linear(x, packed: PackedW4, bias=None) ; w8a16_linear ; fp8_linear(x, w_fp8, w_scale, ...) ; quantize_act_fp8(x, ...) ; fp4_linear(x, w: NVFP4Tensor|packed, ...)
kernels/ops_attn.py   paged_decode_attention(q, k_pool, v_pool, block_tables, seq_lens, *, softmax_scale, kv_dtype, k_scale=None, v_scale=None, window=None) ;
                      prefill_attention(q,k,v,...) (flash-style causal GQA; wraps torch SDPA fallback) ; sparse_topk_attention ; tree_attention ; mla_decode_attention
kernels/ops_moe.py    moe_route(logits, top_k) ; moe_permute / unpermute ; grouped_gemm(x_sorted, expert_weights, offsets, ...)
kernels/ops_util.py   hadamard_transform(x, block) ; quantize_fp8_rowwise ; stochastic_round ; rmsnorm_quant ; rope_and_kv_write ; rejection_verify ; kv_compact ; topk_select
kernels/ops.py       re-exports all of the above (integration owner = core? no: kernels_gemm writes it AFTER reading ops_attn names listed here)
kernels/bench.py:  run_bench(cfg: BenchConfig, ctx) ; BenchConfig{kernels: list[str], shapes: ..., dtypes, repeats, ...}; results stored via telemetry/db table `kernel_bench`; roofline fields (achieved GB/s, % of measured peak, bound type, residual vs cost model)
kernels/catalog.py: list_kernels() -> [{name, family, spec, available(bool), reasons, min_sm, ...}] for the UI
Every op dispatches: triton kernel if (cuda & triton importable & capability ok & shape supported & not env FORGE_FORCE_REF=1) else reference path. `FORGE_KERNEL_DEBUG=1` logs which path ran.
```
### 5.6 `forge.compress` [compress]
`compress.calibrate`: `CalibrationSet` registry (`register_calibration_set(name, texts|dataset_ref)`), `collect_stats(model, tokenizer, set, layers, stats=["hessian","channel_scale","router"]) -> StatsCache` keyed (model hash, set hash, layer);
`compress.ptq`: `quantize_model(model, method: "rtn"|"gptq"|"awq"|"smoothquant", bits, group_size, sym, act_order, calib, ...) -> QuantizedModel info`, produces `PackedLinear` modules ("w4a16"/"w8a16"/"fp8"/"nvfp4") via kernels.layouts; `save_quantized/load_quantized` through models.loader w/ manifest;
`compress.kvquant`: FP8 KV, KIVI-style per-channel-key/per-token-value quant + residual window (reference attention);
`compress.rotation`: randomized Hadamard / orthogonal fuse into weights; `compress.prune`: importance estimation (activation/Fisher), GQA-aware width pruning, head/depth pruning, expert prune/merge hooks → returns new smaller `ModelConfig` + weights; `compress.lowrank` (whitened SVD, rank allocation, break-even guard); `compress.delta` (low-rank & 1-bit-sign BitDelta; apply/serve); `compress.lossless` (artifact-level entropy coding via store.lossless);
`compress.stages`: registers recipe stages with `forge.core.registry` (`calibrate`, `ptq`, `pack`, `kv`, `prune`, `distill` (delegates to train), `lowrank`, `delta`, `lossless`, `gate` (delegates to eval), `eval`, `finetune`). `compress.run.run_compress(cfg: CompressJobConfig, ctx)` = job entry (runs recipe from YAML/inline).
### 5.7 `forge.train` [train]
`train.data`: `ingest_dataset(cfg: IngestConfig, ctx)` (text/jsonl/parquet/HF dataset → tokenise → pack → 256 MiB-shard npy/bin with offsets sidecar → manifest in store; dedup exact + MinHash optional; contamination scan hook); `ShardedTokenDataset`/`DataLoader` with deterministic order = f(manifest, seed, step), resume exact; SFT formats (chat templates, loss masks); preference pairs optional.
`train.config`: `TrainConfig`, `FinetuneConfig`, `DistillConfig` (pydantic, huge but well-described — optimiser, schedule, precision rung R0/R1/R2, parallelism (ddp/fsdp via torch), grad-accum, checkpointing, eval-every, early stop, LoRA params, etc.).
`train.run`: `run_training(cfg: TrainConfig, ctx)`, `run_finetune(cfg: FinetuneConfig, ctx)`, `run_distill(cfg: DistillConfig, ctx)` — emit `ctx.metric` every N steps with keys: `loss, lr, grad_norm, tokens_per_s, tokens_seen, step_time_ms, gpu_mem_gb, mfu, update_weight_ratio` (+ fp4 diagnostics `fp4/zero_grad_tile_frac`, `fp4/scale_flush_frac`, `fp4/sr_bias`, `fp4/loss_gap`); `ctx.progress`; cooperative stop; resume; NaN/Inf/loss-spike detectors with auto-restart-from-checkpoint; save via store checkpoints.
`train.precision`: R0 bf16 autocast, R1 FP8 (torchao/transformer-engine if importable, else fake-quant fallback with clear warning), R2 NVFP4 (fake-quant via `forge.formats` + SR + RHT + 2D weight scaling, native kernels when `has_fp4_block_scaled`); per-layer high-precision exemptions; healing phase.
`train.distill`: offline logit (top-k cache build/read, residual mass), sequence-level, on-policy (student samples, teacher scores), reasoning-trace SFT; forward/reverse KL, JSD, temperature; teacher licence gate (`LicenceBlocked`).
### 5.8 `forge.serve` [serve]
`serve.config.ServeConfig` (model/artifact, dtype, quant, kv_dtype, max_model_len, max_num_seqs, block_size, gpu_mem_util, prefix_caching, chunked_prefill, speculation{mode: off|ngram|draft|auto, max_gamma}, host/port, api_key, sampling defaults, cuda_graphs, tp_size(1 default)). `model` accepts a registered model id/name (base models *and* compress outputs, registered automatically as `source='artifact'` rows with `info.quantization` / `info.base_model`), a model or compress-artifact directory, an artifact-store digest or an HF repo id (`core.modelref.resolve_model_ref`); a packed artifact loads packed (`models.loader.load_model` verifies the packed layers exist, else raises; never a dense stand-in). Memory is explicit (`serve.memory.plan_kv_memory`): `weights` (measured on the loaded model) + `KV pool` (`kv_pool=fit_workload|fill`, bounded by `gpu_mem_util` x total GPU memory, default 0.85) + activation reserve; the plan is logged, returned by `/stats` (`memory`, `model`) and shown on the Inference page.
`serve.kv`: `KVBlockManager` (alloc/match_prefix/fork/commit/demote/promote/stats; radix tree w/ rolling block hash; HBM→host tier), `BlockPool`.
`serve.engine`: `LLMEngine(cfg)`: `add_request(prompt_ids, SamplingParams, request_id) `, `step() -> list[RequestOutput]`, `abort`, `stats()`; continuous batching scheduler (`serve.scheduler`), chunked prefill, prefix cache, sampling (temperature/top-p/top-k/min-p/penalties/seed/logprobs/stop), guided max-tokens; uses `forge.models.ForgeLM` with PagedKVView + `kernels.ops_attn.paged_decode_attention`; falls back to dense HF-style generate on CPU for tests.
`serve.spec`: `NgramDrafter`, `DraftModelDrafter`, `SpecController.choose_gamma(...)` (Plan §7.5 pseudocode, γ=0 allowed, 3% hysteresis), `rejection_sample(p, q, draft_tokens, ...)` exact (P7 App. A.1) with a distribution-exactness test; KV rollback.
`serve.capacity`: `capacity_plan(model_cfg, chip: ChipDescriptor|Capabilities, weight_bits, kv_dtype, context_dist, ...) -> CapacityReport` (Plan Examples A/B: weights/KV/free, max concurrency, ideal decode tok/s, byte breakdown).
`serve.server`: `create_app(engine) -> FastAPI` OpenAI-compatible (`/v1/models`, `/v1/completions`, `/v1/chat/completions` incl. SSE streaming, `/health`, `/stats` (running/waiting seqs, KV usage, prefix hit rate, tokens/s, spec acceptance per position, TTFT/ITL p50/p99), `/abort`); `serve.run.run_server(cfg: ServeConfig, ctx)` = job entry that starts uvicorn and reports stats via `ctx.metric` every second.
### 5.9 `forge.eval` & `forge.planner` [eval_planner]
`eval.metrics`: `kl_divergence(parent_logits, cand_logits, p99=True)`, `top1_flip_rate`, `ece`, `entropy_shift`, `perplexity`; `eval.stats`: `paired_bootstrap(a, b, n=2000)`, `binomial_ci`, `benjamini_hochberg`, `noise_floor_from_repeats`, `distinguishable(...)`; `eval.suites`: registry (`register("suite", name)`) with suites: `distribution` (KL/flip/ECE on held-out text), `mcq` (generic multiple-choice log-likelihood scorer from local jsonl), `longgen`, `longctx` (synthetic needle / multi-key / variable-tracking at length L — RULER-style generators, no downloads), `safety` (refusal/jailbreak probe harness from local jsonl w/ small bundled seed set), `contamination` (n-gram overlap scan), `throughput` (calls serve engine); `eval.gates`: `Gate` classes from YAML (Appendix C.3) -> `GateVerdict`, incl. G-LOSSLESS, G-FP8, G-W4, G-STRUCT, G-LONG, G-SPEC, G-FP4TRAIN presets in `configs/gates/*.yaml`; `eval.run.run_eval(cfg: EvalConfig, ctx)` (parent vs candidate; stores `EvalReport` in db + store).
`planner.costmodel`: `weight_bytes, kv_bytes_per_token, decode_step_time, batch_crossover, quantizer_overhead, spec_speedup, amdahl, t_pred(...)` (Plan §5.1/§5.4, reproduces Examples A, B & P1 B*≈74/295 within rounding — unit tests!); `planner.rulebase`: Fig 5.3 decision flow + Table 5.1 as **data** (YAML-like python dicts) with ledger links; `planner.search.plan(model, chip, workload, budget) -> ranked recipes` (+ successive-halving harness that can call eval); `planner.claims`: ledger CRUD in db + `seed_claims()` with ≥30 claims from P1–P7 (statement, paper, status=untested, prior) ; `planner.estimate(model_cfg, chip, recipe|quant_scheme, workload)` quick fit/bound/ceiling for the UI.

## 6. Jobs, metrics and schema-driven config (owner: jobs_api)

- **Job types** (registered in `forge/jobs/registry.py`; each maps to (ConfigModel, entrypoint) lazily imported):
  `probe`→`forge.hw.probe`, `bench`→`forge.kernels.bench.run_bench` (BenchConfig), `dataset_ingest`→`forge.train.data.ingest_dataset` (IngestConfig),
  `train`→`forge.train.run.run_training` (TrainConfig), `finetune`→`run_finetune` (FinetuneConfig), `distill`→`run_distill` (DistillConfig),
  `compress`→`forge.compress.run.run_compress` (CompressJobConfig), `eval`→`forge.eval.run.run_eval` (EvalConfig), `serve`→`forge.serve.run.run_server` (ServeConfig),
  `recipe`→ run a saved recipe via RecipeEngine, `model_download`→ HF download into models dir + registered in db, `gc`→ store gc.
  Config model location: `forge.<module>.config` (or the module named above) — class names exactly as listed.
- **Worker**: `python -m forge.jobs.worker <job_id>` (subprocess per job, own process group, `CUDA_VISIBLE_DEVICES` set from `spec.gpus`), stdout/stderr → `runs/<id>/log.txt`,
  a `RunContext` implementation writes metrics to `forge.telemetry.metrics`, progress/outputs to the `jobs` table; cancel = SIGTERM→cooperative `should_stop()` then SIGKILL after grace.
  A `JobManager` in the API process (thread) schedules queued jobs under GPU availability (exclusive GPU assignment, or shared if `gpus=None` & cpu job), recovers state on restart (jobs left `running` whose pid is dead → failed).
- **Metrics** (`forge.telemetry.metrics`): `MetricsStore.write(run_id, step, values: dict, ts=None)`, `.query(run_id, keys=None, since_step=None, max_points=2000, rollup="auto") -> {"keys":[...], "points": {key: [[step, value, ts], ...]}}`, `.keys(run_id)`, `.latest(run_id)`; sqlite table `metrics`; optional `export_parquet(run_id)` when pyarrow exists; retention rollups (1/10/100-step) helper.
  System telemetry (GPU) is sampled by the API process every 1–2 s into a ring buffer (+ per-job series `sys/gpu{i}/util|mem|temp|power` while a job runs).
- **Schema-driven forms**: `GET /api/v1/schemas/job-types` returns for every job type `{type, title, description, schema (JSON Schema), defaults, templates:[names], groups:[...]}`.
  The front end builds forms from the schema (types, enums, ranges, descriptions, nested objects, arrays, advanced toggles) — so *adding a config field = automatically in UI*.
  Saved configs: table `configs(id, job_type, name, params_json, created_at, updated_at)`; templates from `configs/<type>/*.yaml` are read-only seeds.

## 7. REST API (prefix `/api/v1`, JSON; optional `Authorization: Bearer $FORGE_API_TOKEN`; SSE for streams) [jobs_api]

```
GET  /system/info                      versions, home, store stats summary, capabilities, job counts, server time
GET  /system/gpus                      live telemetry list (GpuSampler)
GET  /system/gpus/stream               SSE of telemetry (1 Hz)
GET  /hardware/chips                   KNOWN_CHIPS summary + support matrix
GET  /hardware/descriptors | /hardware/descriptors/{id}    measured descriptors
POST /hardware/probe                   -> JobRecord (type probe)
GET/POST /models   GET/PATCH/DELETE /models/{id}   POST /models/{id}/download | /models/register-local {path} | /models/{id}/inspect (config, params, kv bytes/token, licence)
GET/POST /datasets  GET/DELETE /datasets/{id}  POST /datasets/upload (multipart, chunked allowed)  POST /datasets/{id}/ingest -> job  GET /datasets/{id}/preview
GET  /schemas/job-types ; GET /schemas/job-types/{type}
GET/POST /configs ; GET/PUT/DELETE /configs/{id} ; GET /configs/templates?type=
GET  /jobs?status=&type=&limit= ; POST /jobs {type,name,params,gpus,tags} ; GET /jobs/{id} ; POST /jobs/{id}/cancel ; DELETE /jobs/{id}
GET  /jobs/{id}/logs?tail=&offset= ; GET /jobs/{id}/stream (SSE: events log|metric|progress|status) ; GET /jobs/{id}/metrics?keys=&since_step=&max_points=
POST /jobs/{id}/clone
GET/POST /recipes ; GET/PUT/DELETE /recipes/{id} ; POST /recipes/validate {yaml|json} ; POST /recipes/{id}/run -> job
POST /planner/estimate {model, chip|descriptor_id, quant, kv_dtype, workload} ; POST /planner/plan {model, chip, workload, budget}
GET/PUT /claims ; GET /claims/{id}
GET  /artifacts?kind= ; GET /artifacts/{hash} ; GET /artifacts/{hash}/lineage ; POST /artifacts/{hash}/pin|unpin ; POST /artifacts/{hash}/tier {to}
GET  /storage/stats ; POST /storage/gc {dry_run} ; GET /storage/tiers
GET  /evals ; GET /evals/{id} ; POST /evals/compare {ids:[...], parent?} -> table with CIs/noise flags/Pareto
GET  /gates ; GET /gates/{id}
GET  /kernels ; GET /kernels/results?kernel=&chip= ; POST /kernels/bench -> job ; GET /kernels/autotune-cache
GET/POST /deployments ; GET/DELETE /deployments/{id} ; GET /deployments/{id}/stats ; POST /deployments/{id}/chat (SSE proxy to engine) ; POST /serve/simulate {capacity inputs}
GET  /checkpoints?run=  ; POST /checkpoints/{id}/export (to safetensors model dir, registers in /models)
GET  /events (SSE: global job/status/gpu events for dashboard)
```
Static front end served at `/` (SPA with hash routing) from `forge/web/static`; `/docs` = FastAPI OpenAPI.

## 8. Front end (owner: web) — vanilla ES modules, no build, no CDN dependencies (works offline on a rented box)

Pages (Plan §9.3 adapted to a Python full-stack tool): **Dashboard** (GPUs live, active/queued jobs, gates, storage compaction, claims needing attention),
**Models**, **Datasets** (upload/ingest/preview), **Hardware** (descriptor, probe, roofline plot, support matrix), **New Job** wizard for every type with schema-driven form +
template/saved-config selector + YAML/JSON toggle + validation + GPU picker, **Jobs** list, **Job detail / Training monitor** (live loss/lr/grad-norm/tokens-s charts via SSE, FP4 diagnostics panel, log tail, progress, cancel/clone, outputs),
**Fine-tune** (shortcut wizard), **Recipes** (builder: stage palette + YAML + planner suggestion + fit/bound/ceiling panel), **Compare** (table w/ CIs, noise flags, Pareto chart),
**Inference** (deployments list/start, **chat playground** with streaming + sampling params + stats side-panel: running/waiting, KV usage, tok/s, TTFT/ITL, prefix hit rate, spec acceptance, serving simulator + capacity curve),
**Kernel lab** (catalog with availability, bench results, autotune surface heatmap, roofline residuals), **Storage** (bytes breakdown, tiers, lineage graph, GC dry-run → run), **Claims ledger**, **Settings** (API token, theme, defaults).
Features: light/dark, deep-linkable hash routes, provenance badges + CI/"within noise" markers, keyboard accessible, mobile-friendly, CSV/PNG export of charts, toast errors.
Own tiny helpers: `api.js` (fetch wrapper + SSE), `charts.js` (canvas line/scatter/bar/heatmap with zoom, 100k points), `forms.js` (JSON-Schema → DOM), `router.js`, `state.js`, `components/*.js`.

## 9. Config templates (owner: docs_ops) — real, validated by tests against the pydantic models

`configs/{train,finetune,distill,compress,eval,serve,bench,dataset,recipes,gates,chips,fleets}/*.yaml` e.g. tiny-smoke, llama-8b-lora, llama-8b-full-bf16, qwen-fp8, nvfp4-pilot, w4a16-gptq-g128, serve-8b-w4a16, gate-g-w4, … Each has a header comment explaining every non-obvious field. A CPU test loads each template through its pydantic model.

## 10. Priorities when time is short
Correctness & honesty > breadth. Every module must be importable and CPU-tested. Mark clearly in docstrings/`docs/KERNELS.md` which GPU paths are "implemented, not yet validated on hardware".
