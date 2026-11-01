// In-browser fake backend for DEMO mode (?mock=1). EVERYTHING here is synthetic demo data and is labelled
// "DEMO DATA" in the UI. It exists only so the interface can be reviewed without the Python server.
// The shapes follow docs/ARCHITECTURE.md section 6/7; the front end is written defensively so real payloads may differ slightly.

import { ApiError } from './api.js';

const T0 = Date.now() / 1000;
const now = () => Date.now() / 1000;
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function hnoise(step, k = 1) { return rng(step * 2654435761 + k * 97)() - 0.5; }
const clone = (o) => JSON.parse(JSON.stringify(o));
const hex = (seed, n = 64) => { const r = rng(seed); let s = ''; while (s.length < n) s += Math.floor(r() * 16).toString(16); return s; };
const H = (seed) => 'blake3:' + hex(seed);

// ------------------------------------------------------------------ schema builders
const ui = (group, advanced, unit) => ({ ...(group ? { 'x-group': group } : {}), ...(advanced ? { 'x-advanced': true } : {}), ...(unit ? { 'x-unit': unit } : {}) });
const S = {
  str: (description, def, extra = {}) => ({ type: 'string', description, ...(def !== undefined ? { default: def } : {}), ...extra }),
  int: (description, def, minimum, maximum, extra = {}) => ({ type: 'integer', description, default: def, ...(minimum != null ? { minimum } : {}), ...(maximum != null ? { maximum } : {}), ...extra }),
  num: (description, def, minimum, maximum, extra = {}) => ({ type: 'number', description, default: def, ...(minimum != null ? { minimum } : {}), ...(maximum != null ? { maximum } : {}), ...extra }),
  bool: (description, def, extra = {}) => ({ type: 'boolean', description, default: def, ...extra }),
  enm: (description, values, def, extra = {}) => ({ type: 'string', enum: values, description, default: def, ...extra }),
  arr: (description, items, def = [], extra = {}) => ({ type: 'array', items, description, default: def, ...extra }),
  map: (description, valueSchema = {}, def = {}, extra = {}) => ({ type: 'object', additionalProperties: valueSchema, description, default: def, ...extra }),
  nul: (inner) => ({ ...inner, 'x-nullable': true, default: null }),
  obj: (description, properties, extra = {}) => ({ type: 'object', description, properties, ...extra }),
};
const withGroup = (schema, group, advanced, unit) => ({ ...schema, ...ui(group, advanced, unit) });

function trainSchema(kind) {
  const ft = kind === 'finetune';
  const props = {
    run_name: withGroup(S.str('Human-readable run name (also the run directory name).', ''), 'General'),
    model: withGroup(S.str(ft ? 'Base model id (registered in Models) or local path.' : 'Model config preset or checkpoint to start from.', 'tiny-llama-demo', { examples: ['tiny-llama-demo', 'llama-3.1-8b'] }), 'Model'),
    dtype: withGroup(S.enm('Parameter / activation dtype.', ['bfloat16', 'float16', 'float32'], 'bfloat16'), 'Model'),
    data: withGroup(S.obj('Data loading.', {
      dataset: S.str('Dataset id from the Datasets page.', ''),
      seq_len: { ...S.int('Training sequence length.', 2048, 16, 131072), ...ui(null, false, 'tokens') },
      shuffle: S.bool('Shuffle documents deterministically from (manifest, seed, step).', true),
      mix: { ...S.map('Source name -> sampling weight.', S.num('Weight', 1, 0), { web: 0.7, code: 0.3 }), ...ui(null, true) },
    }, { required: ['dataset'] }), 'Data'),
    optimizer: withGroup(S.obj('Optimiser settings.', {
      name: S.enm('Optimiser.', ['adamw', 'lion', 'muon', 'sgd'], 'adamw'),
      lr: { ...S.num('Peak learning rate.', ft ? 2e-5 : 3e-4, 1e-7, 1e-2), ...ui(null, false) },
      weight_decay: S.num('Decoupled weight decay.', 0.1, 0, 1),
      betas: { ...S.arr('Adam betas.', { type: 'number', minimum: 0, maximum: 1 }, [0.9, 0.95], { minItems: 2, maxItems: 2 }), ...ui(null, true) },
      grad_clip: S.nul(S.num('Global gradient-norm clip (unset = no clipping).', 1.0, 0)),
    }), 'Optimisation'),
    schedule: withGroup(S.obj('Learning-rate schedule.', {
      type: S.enm('Schedule shape.', ['cosine', 'linear', 'constant', 'wsd'], 'cosine'),
      warmup_steps: S.int('Linear warm-up steps.', 200, 0, 100000),
      min_lr_ratio: S.num('Final LR as a fraction of peak.', 0.1, 0, 1),
    }), 'Optimisation'),
    max_steps: withGroup(S.int('Total optimiser steps.', 5000, 1, 10000000), 'Optimisation'),
    precision: withGroup(S.obj('Precision ladder (Plan TR-3).', {
      rung: S.enm('R0 = bf16, R1 = FP8, R2 = NVFP4 (experimental, gated).', ['R0', 'R1', 'R2'], 'R0'),
      exempt_layers: S.arr('Layers kept in high precision.', { type: 'string' }, ['embed', 'lm_head']),
      healing_steps: S.int('Final high-precision healing steps (R2 only).', 0, 0, 100000),
    }), 'Precision'),
    parallelism: withGroup(S.obj('Parallelism and batching.', {
      strategy: S.enm('Parallel strategy.', ['single', 'ddp', 'fsdp'], 'single'),
      micro_batch_size: S.int('Sequences per micro-batch per GPU.', 4, 1, 4096),
      grad_accum: S.int('Gradient accumulation steps.', 8, 1, 4096),
      activation_checkpointing: S.bool('Recompute activations to save memory.', false),
    }), 'Hardware'),
    checkpoint: withGroup(S.obj('Checkpointing.', {
      every: S.int('Save a checkpoint every N steps.', 500, 1, 1000000, ui(null, false, 'steps')),
      keep_last: S.int('Retention: keep the last N.', 3, 0, 1000),
      keyframe_every: { ...S.int('Delta-chain keyframe interval.', 4, 1, 1000), ...ui(null, true) },
    }), 'Checkpoints'),
    eval: withGroup(S.obj('In-training evaluation.', {
      every: S.int('Evaluate every N steps (0 disables).', 250, 0, 1000000),
      suites: S.arr('Suites to run.', { type: 'string', enum: ['distribution', 'mcq', 'longctx', 'safety'] }, ['distribution']),
    }), 'Evaluation'),
    seed: withGroup(S.int('Global seed.', 1234, 0, 2 ** 31 - 1), 'General'),
    extra: withGroup(S.map('Free-form extra options passed to the trainer.', {}, {}), 'General', true),
  };
  if (ft) {
    props.method = withGroup(S.enm('Fine-tuning method.', ['lora', 'full', 'qlora'], 'lora'), 'Model');
    props.lora = withGroup(S.obj('LoRA adapter settings.', {
      r: S.int('Rank.', 16, 1, 512),
      alpha: S.num('Scaling alpha.', 32, 0.1, 1024),
      dropout: S.num('Dropout.', 0.05, 0, 0.5),
      target_modules: S.arr('Linear modules to adapt.', { type: 'string' }, ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
    }), 'LoRA');
    props.dataset = withGroup(S.str('Fine-tuning dataset id (convenience alias for data.dataset).', ''), 'Data');
  }
  return S.obj(kind === 'train' ? 'Train from scratch / continue pre-training.' : 'Fine-tune an existing model.', props, { required: ['model'], additionalProperties: false, 'x-groups': ['General', 'Model', 'Data', 'Optimisation', 'Precision', 'Hardware', 'Checkpoints', 'Evaluation'].concat(ft ? ['LoRA'] : []) });
}

const JOB_TYPES = [
  { type: 'probe', title: 'Hardware probe', description: 'Measure the GPU: bandwidth, tensor-core throughput, launch latency. Fills a ChipDescriptor.', schema: S.obj('Probe', { quick: S.bool('Skip long micro-benchmarks.', false), device: S.int('CUDA device index.', 0, 0, 63) }, { additionalProperties: false }), templates: ['probe-quick'] },
  { type: 'ternary_train', title: 'Train ternary (1.58-bit) - experimental', description: 'DEMO DATA. Latent-weight ternary QAT with health diagnostics (zero density, flip rate).', schema: trainSchema('train'), templates: ['tiny-smoke'] },
  { type: 'bench', title: 'Kernel benchmark', description: 'Benchmark kernels against their reference oracles with roofline fields.', schema: S.obj('Bench', { kernels: S.arr('Kernel names.', { type: 'string' }, ['w4a16_gemm']), dtypes: S.arr('Dtypes.', { type: 'string', enum: ['fp16', 'bf16', 'fp8', 'fp4'] }, ['bf16']), repeats: S.int('Timed repeats.', 20, 1, 1000), warmup: S.int('Warm-up iterations.', 5, 0, 100), l2_flush: S.bool('Flush L2 between runs.', true) }), templates: [] },
  { type: 'dataset_ingest', title: 'Dataset ingest', description: 'Tokenise, pack and shard a text/jsonl/parquet/HF dataset.', schema: S.obj('Ingest', { source: S.str('Path, upload id or HF dataset.', ''), tokenizer: S.str('Tokenizer id or path.', 'llama-3.1-8b'), shard_mb: S.int('Shard size.', 256, 1, 4096, ui(null, true, 'MiB')), dedup: S.enm('Deduplication.', ['none', 'exact', 'minhash'], 'exact') }, { required: ['source'] }), templates: [] },
  { type: 'train', title: 'Train', description: 'Pre-train or continue training with the precision ladder, checkpoints and diagnostics.', schema: trainSchema('train'), templates: ['tiny-smoke', 'nvfp4-pilot'] },
  { type: 'finetune', title: 'Fine-tune', description: 'Full or LoRA/QLoRA fine-tuning on an SFT dataset.', schema: trainSchema('finetune'), templates: ['llama-8b-lora', 'llama-8b-full-bf16'] },
  { type: 'distill', title: 'Distill', description: 'Offline-logit, sequence-level or on-policy distillation (licence gated).', schema: S.obj('Distill', { teacher: S.str('Teacher model id.', ''), student: S.str('Student model id.', ''), loss: S.enm('Divergence.', ['forward_kl', 'reverse_kl', 'jsd'], 'forward_kl'), temperature: S.num('Temperature.', 1, 0.1, 10), topk: S.int('Cached top-k logits.', 64, 1, 4096) }, { required: ['teacher', 'student'] }), templates: [] },
  { type: 'compress', title: 'Compress', description: 'Quantise / prune / low-rank a model through a recipe with eval gates.', schema: S.obj('Compress', { model: S.str('Input model.', ''), method: S.enm('PTQ method.', ['rtn', 'gptq', 'awq', 'smoothquant'], 'gptq'), bits: S.enm('Weight bits.', ['4', '8'], '4'), group_size: S.int('Group size.', 128, 16, 1024), act_order: S.bool('Activation ordering (GPTQ).', true), calibration: S.str('Calibration set id.', ''), recipe: S.nul(S.str('Optional recipe YAML id (overrides the above).', '')) }, { required: ['model'] }), templates: ['w4a16-gptq-g128', 'qwen-fp8'] },
  { type: 'eval', title: 'Evaluate', description: 'Run suites on a candidate vs parent, with noise floors and gates.', schema: S.obj('Eval', { candidate: S.str('Candidate artifact/model.', ''), parent: S.nul(S.str('Parent for paired comparison.', '')), suites: S.arr('Suites.', { type: 'string', enum: ['distribution', 'mcq', 'longgen', 'longctx', 'safety', 'contamination', 'throughput'] }, ['distribution']), gate: S.nul(S.enm('Gate preset.', ['G-LOSSLESS', 'G-FP8', 'G-W4', 'G-STRUCT', 'G-LONG', 'G-SPEC', 'G-FP4TRAIN'], 'G-W4')), bootstrap: S.int('Bootstrap resamples.', 2000, 100, 100000) }, { required: ['candidate'] }), templates: ['gate-g-w4'] },
  { type: 'serve', title: 'Serve', description: 'Start the OpenAI-compatible serving engine.', schema: S.obj('Serve', { model: S.str('Model / artifact.', ''), dtype: S.enm('Dtype.', ['bfloat16', 'float16'], 'bfloat16'), kv_dtype: S.enm('KV cache dtype.', ['bf16', 'fp8_e4m3', 'int8', 'int4'], 'bf16'), max_model_len: S.int('Max context.', 8192, 128, 1048576), max_num_seqs: S.int('Max concurrent sequences.', 64, 1, 4096), gpu_mem_util: S.num('Fraction of GPU memory for weights + KV.', 0.9, 0.1, 1), prefix_caching: S.bool('Radix prefix cache.', true), speculation: S.obj('Speculative decoding.', { mode: S.enm('Mode.', ['off', 'ngram', 'draft', 'auto'], 'off'), max_gamma: S.int('Max draft tokens.', 4, 1, 16) }), port: S.int('Port.', 8000, 1, 65535) }, { required: ['model'] }), templates: ['serve-8b-w4a16'] },
  { type: 'recipe', title: 'Run recipe', description: 'Run a saved compression/training recipe through the cached stage engine.', schema: S.obj('Recipe run', { recipe_id: S.str('Saved recipe id.', ''), overrides: S.map('Per-stage param overrides.', {}) }, { required: ['recipe_id'] }), templates: [] },
  { type: 'model_download', title: 'Download model', description: 'Download a Hugging Face model into the models directory.', schema: S.obj('Download', { repo: S.str('HF repo id.', ''), revision: S.nul(S.str('Revision.', '')), allow_patterns: S.arr('File patterns.', { type: 'string' }, ['*.safetensors', '*.json']) }, { required: ['repo'] }), templates: [] },
  { type: 'gc', title: 'Storage GC', description: 'Mark-and-sweep garbage collection of the artifact store.', schema: S.obj('GC', { dry_run: S.bool('Only report what would be removed.', true) }), templates: [] },
].map((t) => ({ ...t, defaults: schemaDefaults(t.schema), groups: t.schema['x-groups'] || [] }));

function schemaDefaults(s) {
  if (s.default !== undefined) return clone(s.default);
  if (s.type === 'object' && s.properties) { const o = {}; for (const [k, v] of Object.entries(s.properties)) { const d = schemaDefaults(v); if (d !== undefined) o[k] = d; } return o; }
  return undefined;
}

// ------------------------------------------------------------------ state
const state = {
  models: [
    { id: 'm_llama31_8b', name: 'Llama-3.1-8B (demo)', source: 'hf', path_or_repo: 'meta-llama/Llama-3.1-8B', revision: 'main', hash: H(1), licence: 'llama3.1', distill_allowed: true, params_b: 8.03, status: 'ready', created_at: T0 - 86400 * 3 },
    { id: 'm_qwen25_7b', name: 'Qwen2.5-7B (demo)', source: 'hf', path_or_repo: 'Qwen/Qwen2.5-7B', revision: 'main', hash: H(2), licence: 'apache-2.0', distill_allowed: true, params_b: 7.6, status: 'ready', created_at: T0 - 86400 * 2 },
    { id: 'm_gemma_restricted', name: 'Restricted-licence-9B (demo)', source: 'hf', path_or_repo: 'example/restricted-9b', revision: 'main', hash: H(3), licence: 'custom-no-distill', distill_allowed: false, params_b: 9.2, status: 'ready', created_at: T0 - 86400 },
    { id: 'm_tiny', name: 'tiny-llama-demo', source: 'local', path_or_repo: '~/.forge/models/tiny-llama-demo', revision: null, hash: H(4), licence: 'apache-2.0', distill_allowed: true, params_b: 0.0021, status: 'ready', created_at: T0 - 3600 * 5 },
  ],
  datasets: [
    { id: 'ds_web_mix', name: 'web-mix-v1 (demo)', tokenizer_hash: H(11), total_tokens: 4.2e9, shards: Array.from({ length: 16 }, (_, i) => ({ hash: H(100 + i), tokens: 2.6e8, docs: 120000 })), mixture: { web: 0.7, code: 0.2, math: 0.1 }, filtering_version: 'f3', licence: 'mixed', dedup_report: { exact_removed: 0.083, minhash_removed: 0.041 }, status: 'ready', created_at: T0 - 86400 * 4 },
    { id: 'ds_sft_chat', name: 'sft-chat-small (demo)', tokenizer_hash: H(12), total_tokens: 3.1e7, shards: [{ hash: H(120), tokens: 3.1e7, docs: 41000 }], mixture: { chat: 1 }, filtering_version: 'f3', licence: 'cc-by-4.0', dedup_report: { exact_removed: 0.01 }, status: 'ready', created_at: T0 - 86400 },
  ],
  jobs: [],
  configs: [],
  recipes: [
    { id: 'rcp_w4_gptq', name: 'w4a16-gptq-g128', created_at: T0 - 86400, yaml: 'name: w4a16-gptq-g128\ninputs:\n  model: m_llama31_8b\nstages:\n  - name: calibrate\n    params: {samples: 128, seq_len: 2048}\n  - name: ptq\n    params: {method: gptq, bits: 4, group_size: 128, act_order: true}\n  - name: pack\n    params: {layout: auto}\n  - name: eval\n    params: {suites: [distribution], gate: G-W4}\nbaselines: [rtn_w4, bf16]\nbudget: {gpu_hours: 2}\npolicy: {licence_check: true}\nseeds: [0, 1, 2]\n' },
    { id: 'rcp_fp8', name: 'fp8-weights-acts', created_at: T0 - 7200, yaml: 'name: fp8-weights-acts\ninputs:\n  model: m_qwen25_7b\nstages:\n  - name: calibrate\n    params: {samples: 64}\n  - name: ptq\n    params: {method: rtn, bits: 8, fmt: fp8_e4m3}\n  - name: eval\n    params: {suites: [distribution, mcq], gate: G-FP8}\nseeds: [0]\n' },
  ],
  deployments: [
    { id: 'dep_demo', name: 'llama-3.1-8b-w4a16 (demo)', artifact_ref: 'm_llama31_8b', serve_config: { kv_dtype: 'fp8_e4m3', max_model_len: 8192, max_num_seqs: 64 }, job_id: 'job_serve_demo', port: 8000, status: 'running', created_at: T0 - 1800 },
  ],
  claims: [],
  evals: [],
  artifacts: [],
  pins: new Set(),
  tierOverrides: {},
  gcDone: false,
};

function initJobs() {
  const mk = (id, type, name, status, ageS, durS, extra = {}) => ({
    id, spec: { type, name, params: clone(JOB_TYPES.find((t) => t.type === type).defaults), gpus: extra.gpus || null, tags: extra.tags || [] }, status,
    created_at: T0 - ageS, started_at: status === 'queued' ? null : T0 - ageS + 5, finished_at: ['succeeded', 'failed', 'cancelled'].includes(status) ? T0 - ageS + 5 + durS : null,
    pid: status === 'running' ? 4242 : null, exit_code: status === 'succeeded' ? 0 : status === 'failed' ? 1 : null,
    progress: status === 'succeeded' ? 1 : status === 'queued' ? 0 : extra.progress ?? 0.5, progress_msg: extra.msg || '', error: extra.error || null, outputs: extra.outputs || {}, run_dir: `~/.forge/runs/${id}`, total_steps: extra.total_steps,
  });
  state.jobs = [
    mk('job_demo_run', 'train', 'tiny-llama pretrain (demo)', 'running', 900, 0, { gpus: [0], total_steps: 6000, tags: ['demo', 'pretrain'] }),
    mk('job_demo_queue', 'finetune', 'llama-8b lora sft (demo)', 'queued', 120, 0, { tags: ['sft'] }),
    mk('job_demo_done', 'compress', 'llama-8b w4a16 gptq (demo)', 'succeeded', 7200, 1380, { outputs: { artifact: H(41), compression_ratio: 3.6, gate: 'G-W4: pass' }, tags: ['w4'] }),
    mk('job_demo_eval', 'eval', 'w4 vs bf16 distribution (demo)', 'succeeded', 6000, 600, { outputs: { report_id: 'ev_w4', gate_verdict: 'pass' } }),
    mk('job_demo_fail', 'train', 'nvfp4 pilot (demo)', 'failed', 20000, 3100, { error: 'Loss spike detector tripped 3 times (step 2210); auto-restart budget exhausted.', total_steps: 4000, tags: ['fp4'] }),
    mk('job_demo_cancel', 'bench', 'gemm bench (demo)', 'cancelled', 30000, 80),
    mk('job_serve_demo', 'serve', 'serve llama-3.1-8b (demo)', 'running', 1800, 0, { gpus: [1], progress: 0, msg: 'serving on :8000' }),
  ];
}

function initClaims() {
  const C = (id, paper, loc, statement, status, prior, notes = '', rules = []) => ({ id, source_paper: paper, location: loc, statement, status, prior, linked_experiments: status === 'untested' ? [] : ['ev_w4'], depends_rules: rules, notes });
  state.claims = [
    C('P1-W4-001', 'P1', 'Sec 4.2', 'GPTQ W4 g128 keeps perplexity within 1% of BF16 on 7-8B models.', 'reproduced', 'trusted', 'Reproduced on tiny model suite; needs full-size confirmation.', ['R-W4-DEFAULT']),
    C('P1-W4-002', 'P1', 'Table 3', 'AWQ outperforms GPTQ at W4 on instruction-tuned models.', 'untested', 'untrusted', '', ['R-W4-AWQ']),
    C('P1-KV-003', 'P3', 'Fig 5', 'Per-channel key / per-token value 2-bit KV quantisation is near lossless with a residual window.', 'partial', 'untrusted', '4-bit confirmed, 2-bit unstable on long contexts.', ['R-KV-2BIT']),
    C('P2-SP-004', 'P2', 'Sec 3', '2:4 sparsity with fine-tuning recovers 98% of dense accuracy.', 'untested', 'untrusted', '', ['R-SPARSE-24']),
    C('P2-PR-005', 'P2', 'Table 5', 'GQA-aware width pruning beats head pruning at equal parameter count.', 'contradicted', 'untrusted', 'Our run shows equal quality within noise.', ['R-PRUNE-WIDTH']),
    C('P3-AT-006', 'P3', 'Sec 6', 'Top-k sparse attention with k=2048 matches dense on RULER up to 128k.', 'untested', 'untrusted', '', ['R-SPARSE-ATTN']),
    C('P4-LR-007', 'P4', 'Eq 9', 'Whitened SVD beats plain SVD at equal rank for MLP layers.', 'reproduced', 'trusted', '', ['R-LOWRANK']),
    C('P4-LL-008', 'P4', 'Sec 5', 'Exponent-split + ANS gives ~15-20% lossless savings on BF16 weights.', 'partial', 'trusted', 'Measured 14% on Llama weights.', ['R-LOSSLESS']),
    C('P5-FP4-009', 'P5', 'Sec 3.1', 'NVFP4 pretraining with RHT + SR + 2D scaling tracks BF16 loss within 1.5%.', 'untested', 'untrusted', 'Experimental / gated (TR-3).', ['R-FP4TRAIN']),
    C('P6-SA-010', 'P6', 'Table 2', 'Learned sparse-attention gates retain 99% quality at 4x fewer KV reads.', 'untested', 'untrusted', '', ['R-SPARSE-ATTN']),
    C('P7-SD-011', 'P7', 'App A.1', 'Rejection sampling preserves the target distribution exactly.', 'reproduced', 'trusted', 'Chi-square test on toy distributions passes.', ['R-SPEC']),
    C('P7-SD-012', 'P7', 'Fig 4', 'Speculative speedup at batch 1 is ~2-3x with n-gram drafters on code.', 'untested', 'untrusted', '', ['R-SPEC']),
  ];
}

function mv(value, rel, n, prov = 'measured', within = null) {
  return { value, ci_low: value * (1 - rel), ci_high: value * (1 + rel), n, provenance: prov, within_noise: within };
}
function initEvals() {
  const mkE = (id, label, art, parent, m, gate) => ({ id, label, artifact_hash: art, parent_hash: parent, suite_versions: { distribution: 'v2', mcq: 'v1' }, metrics: m, gate, noise_floor: { kl_p99: 0.0006, mmlu_acc: 0.004 }, created_at: T0 - 5000, env: { gpu: 'H100 (demo)' } });
  const gate = (id, v, reasons) => ({ gate_id: id, verdict: v, reasons, details: {} });
  state.evals = [
    mkE('ev_bf16', 'bf16 baseline', H(40), null, { kl_p99: mv(0, 0, 3), mmlu_acc: mv(0.664, 0.006, 3), decode_tok_s: mv(86, 0.03, 5), weight_gb: { value: 16.06, provenance: 'derived' } }, null),
    mkE('ev_w4', 'w4a16 gptq g128', H(41), H(40), { kl_p99: mv(0.0112, 0.12, 3), mmlu_acc: mv(0.659, 0.006, 3, 'measured', true), decode_tok_s: mv(231, 0.03, 5), weight_gb: { value: 4.5, provenance: 'derived' } }, gate('G-W4-v2', 'pass', ['KL p99 below 0.02', 'MMLU delta within noise'])),
    mkE('ev_w4_rtn', 'w4a16 rtn g128', H(42), H(40), { kl_p99: mv(0.031, 0.1, 3), mmlu_acc: mv(0.641, 0.007, 3, 'measured', false), decode_tok_s: mv(233, 0.03, 5), weight_gb: { value: 4.5, provenance: 'derived' } }, gate('G-W4-v2', 'fail', ['KL p99 0.031 > 0.02'])),
    mkE('ev_fp8', 'fp8 w8a8', H(43), H(40), { kl_p99: mv(0.0021, 0.15, 3), mmlu_acc: mv(0.663, 0.006, 3, 'measured', true), decode_tok_s: mv(142, 0.03, 5), weight_gb: { value: 8.1, provenance: 'derived' } }, gate('G-FP8-v1', 'pass', ['all metrics within thresholds'])),
    mkE('ev_w4_awq', 'w4a16 awq g128', H(44), H(40), { kl_p99: mv(0.0108, 0.12, 3, 'measured', true), mmlu_acc: mv(0.661, 0.006, 3, 'measured', true), decode_tok_s: mv(229, 0.03, 5), weight_gb: { value: 4.5, provenance: 'derived' } }, gate('G-W4-v2', 'inconclusive', ['KL difference vs GPTQ within noise'])),
  ];
}

function initArtifacts() {
  const A = (seed, kind, name, bytes, stored, parents = [], tier = 'hot') => ({ hash: H(seed), kind, name, size: bytes, stored_size: stored, tier, pinned: false, parents: parents.map(H), created_at: T0 - seed * 600, recipe_hash: seed > 40 ? H(900 + seed) : null });
  state.artifacts = [
    A(40, 'model', 'llama-3.1-8b bf16', 16.06e9, 16.06e9), A(41, 'model', 'llama-3.1-8b w4a16-gptq', 4.5e9, 4.5e9, [40]), A(42, 'model', 'llama-3.1-8b w4a16-rtn', 4.5e9, 1.2e9, [40]),
    A(43, 'model', 'llama-3.1-8b fp8', 8.1e9, 3.1e9, [40]), A(44, 'model', 'llama-3.1-8b w4a16-awq', 4.5e9, 1.0e9, [40], 'warm'), A(45, 'checkpoint', 'tiny-llama step 3000', 8.4e6, 2.0e6, [], 'hot'),
    A(46, 'checkpoint', 'tiny-llama step 3500 (delta)', 8.4e6, 0.6e6, [45]), A(47, 'adapter', 'lora r16 sft', 4.2e7, 4.2e7, [40], 'cold'), A(48, 'dataset', 'web-mix-v1', 8.4e9, 7.7e9), A(49, 'calib', 'calib-128x2048', 1.1e6, 1.1e6, [], 'warm'),
    A(50, 'eval', 'eval report ev_w4', 2.1e4, 2.1e4, [41, 40]),
  ];
  state.artifacts[0].pinned = true; state.artifacts[1].pinned = true;
}

initJobs(); initClaims(); initEvals(); initArtifacts();

// ------------------------------------------------------------------ telemetry
function gpuSample() {
  const t = now() - T0;
  return [0, 1].map((i) => {
    const busy = i === 0 ? 0.88 + 0.08 * Math.sin(t / 3) : 0.35 + 0.25 * Math.sin(t / 5 + 1);
    return { index: i, name: 'NVIDIA H100 80GB HBM3 (demo)', util: Math.round(busy * 100 + hnoise(Math.floor(t * 2), i) * 6), mem_used_mb: Math.round((i === 0 ? 61000 : 28000) + 800 * Math.sin(t / 7)), mem_total_mb: 81559, temp_c: Math.round(58 + 14 * busy + hnoise(Math.floor(t), i) * 3), power_w: Math.round(180 + 480 * busy), clock_mhz: 1755, processes: [] };
  });
}

// ------------------------------------------------------------------ metrics synthesis
function jobById(id) { return state.jobs.find((j) => j.id === id); }
function jobStepNow(j) {
  const total = j.total_steps || 5000;
  if (j.status === 'running') { const el = now() - j.started_at; return Math.min(total, Math.floor(el * 6)); }
  if (j.status === 'succeeded') return total;
  if (j.status === 'failed') return Math.floor(total * 0.55);
  return Math.floor(total * 0.3);
}
function metricAt(key, step, total) {
  const x = step / total;
  switch (key) {
    case 'loss': return 2.1 + 6.4 * Math.exp(-step / (total * 0.12)) + 0.05 * hnoise(step, 1) + (step > 0 && step % 997 === 0 ? 0.15 : 0);
    case 'val_loss': return 2.2 + 6.2 * Math.exp(-step / (total * 0.12)) + 0.01 * hnoise(step, 2);
    case 'lr': return step < 200 ? 3e-4 * (step / 200) : 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + Math.cos(Math.PI * Math.min(1, x)));
    case 'grad_norm': return 0.4 + 2.5 * Math.exp(-step / (total * 0.05)) + 0.08 * Math.abs(hnoise(step, 3));
    case 'tokens_per_s': return 182000 + 9000 * hnoise(step, 4);
    case 'tokens_seen': return step * 262144;
    case 'step_time_ms': return 1440 + 60 * hnoise(step, 5);
    case 'gpu_mem_gb': return 58.4 + 0.2 * hnoise(step, 6);
    case 'mfu': return 0.41 + 0.015 * hnoise(step, 7);
    case 'update_weight_ratio': return 1.5e-3 * Math.exp(-x) * (1 + 0.1 * hnoise(step, 8));
    case 'fp4/zero_grad_tile_frac': return 0.012 + 0.004 * hnoise(step, 9) + 0.01 * x;
    case 'fp4/scale_flush_frac': return 0.0004 + 0.0002 * Math.abs(hnoise(step, 10));
    case 'fp4/sr_bias': return 0.0002 * hnoise(step, 11);
    case 'fp4/loss_gap': return 0.02 + 0.03 * x + 0.004 * hnoise(step, 12);
    case 'ternary/zero_frac': return 0.43 + 0.012 * hnoise(step, 13) + 0.02 * Math.exp(-x * 6);
    case 'ternary/zero_frac_min': return 0.40 + 0.01 * hnoise(step, 14);
    case 'ternary/zero_frac_max': return 0.47 + 0.01 * hnoise(step, 15) + 0.03 * x;
    case 'ternary/flip_rate': return 0.0006 + 0.006 * Math.exp(-x * 8) + 0.0002 * Math.abs(hnoise(step, 16));
    case 'ternary/flip_rate_max': return 0.0012 + 0.01 * Math.exp(-x * 8) + 0.0004 * Math.abs(hnoise(step, 17));
    case 'ternary/scale_mean': return 0.021 + 0.003 * x + 0.0002 * hnoise(step, 18);
    case 'ternary/lam': return Math.min(1, step / 500);
    case 'ternary/loss_gap': return 0.06 * Math.exp(-x * 2) + 0.02 + 0.004 * hnoise(step, 19);
    default: return 0;
  }
}
const METRIC_KEYS = ['loss', 'val_loss', 'lr', 'grad_norm', 'tokens_per_s', 'tokens_seen', 'step_time_ms', 'gpu_mem_gb', 'mfu', 'update_weight_ratio'];
const FP4_KEYS = ['fp4/zero_grad_tile_frac', 'fp4/scale_flush_frac', 'fp4/sr_bias', 'fp4/loss_gap'];
const TERNARY_KEYS = ['ternary/zero_frac', 'ternary/zero_frac_min', 'ternary/zero_frac_max', 'ternary/flip_rate', 'ternary/flip_rate_max', 'ternary/scale_mean', 'ternary/lam', 'ternary/loss_gap'];
function jobKeys(j) { const k = METRIC_KEYS.slice(); if (j.spec.type === 'ternary_train') k.push(...TERNARY_KEYS); if (j.id === 'job_demo_fail' || (j.spec.params.precision && j.spec.params.precision.rung === 'R2')) k.push(...FP4_KEYS); return k; }

const MOCK_METHODS = [
  { id: 'pretrain-bf16', title: 'Pre-train / continue training (bf16) (demo)', family: 'pretrain', status: 'supported', summary: 'DEMO DATA. Decoder-only pre-training with bf16 autocast.', job_type: 'train', template: null, config_hint: {}, produces: 'model', needs: ['dataset'], diagnostics_prefix: null, planner_scheme: null, gpu_validated: false, docs: null },
  { id: 'lora', title: 'LoRA / DoRA (demo)', family: 'peft', status: 'supported', summary: 'DEMO DATA. Low-rank adapters.', job_type: 'finetune', template: null, config_hint: {}, produces: 'adapter', needs: ['dataset'], diagnostics_prefix: null, planner_scheme: null, gpu_validated: false, docs: null },
  { id: 'ptq-w4a16-gptq', title: 'GPTQ W4A16 (demo)', family: 'ptq', status: 'supported', summary: 'DEMO DATA. 4-bit weight quantisation.', job_type: 'compress', template: null, config_hint: {}, produces: 'quantized', needs: ['calibration data'], diagnostics_prefix: null, planner_scheme: 'w4a16-g128', gpu_validated: false, docs: null },
  { id: 'ternary-train', title: 'Ternary (1.58-bit) QAT from scratch (demo)', family: 'ternary', status: 'experimental', summary: 'DEMO DATA. Latent-weight QAT with {-1,0,+1} weights and int8 activations; CPU-validated only.', job_type: 'ternary_train', template: null, config_hint: {}, produces: 'model', needs: ['dataset'], diagnostics_prefix: 'ternary', planner_scheme: 't158', gpu_validated: false, docs: 'docs/TERNARY.md' },
  { id: 'ternary-ptq', title: 'Ternary post-training quantisation (demo)', family: 'ternary', status: 'planned', summary: 'DEMO DATA. Reserved slot.', job_type: null, template: null, config_hint: {}, produces: 'quantized', needs: [], diagnostics_prefix: 'ternary', planner_scheme: 't158', gpu_validated: false, docs: null },
];
function mockTrainSim(b) {
  const c = b.config || {}; const hidden = c.hidden || 768, layers = c.layers || 12, ffn = c.ffn || 2048, vocab = c.vocab || 32000;
  const lin = layers * (4 * hidden * hidden + 3 * ffn * hidden); const gib = 1024 ** 3; const world = b.n_gpu || 1;
  const state = lin * (c.scheme === 'ternary_native' ? 0.25 + 4 + 4 + 2 : 16) / (c.shard_state ? world : 1) / gib;
  const mem = { 'DEMO weights+optimiser': state, 'DEMO activations': 0.002 * layers * hidden * (c.seq_len || 2048) / 1e6, 'DEMO embeddings/head': vocab * hidden * 16 / gib, 'DEMO runtime': 1.6 };
  const total = Object.values(mem).reduce((a, x) => a + x, 0); const tps = 180000 * (768 / hidden) * (12 / layers) ** 0.5 * world * 0.9;
  return { ok: total < 80, scheme: c.scheme || 'bf16', params: { linear: lin, total: lin + vocab * hidden }, mem_gb: mem, vram_gb: 80, micro_batch: 8, accum: 4, step_s: (c.global_batch_tokens || 262144) / tps, tokens_per_s: tps, tokens_per_s_gpu: tps / world, mfu: 0.22, total_hours: (c.max_tokens || 2e8) / tps / 3600, cost_usd: 0, energy_kwh: 1.2, breakdown_s: { 'DEMO linear fwd': 0.3, 'DEMO linear bwd': 0.6, 'DEMO attention': 0.1 }, notes: ['DEMO DATA: crude closed-form numbers, no relation to measurements. ESTIMATE only.'], suggestions: [], estimate: true };
}

function metricsFor(j, q) {
  if (!['train', 'finetune', 'distill', 'ternary_train'].includes(j.spec.type)) return { keys: [], points: {} };
  const total = j.total_steps || 5000; const stepNow = jobStepNow(j);
  const maxPts = Math.max(10, Math.min(5000, Number(q.max_points) || 2000));
  const since = Number(q.since_step) || 0;
  const keys = q.keys ? String(q.keys).split(',').filter(Boolean) : jobKeys(j);
  const span = Math.max(1, stepNow - since); const stride = Math.max(1, Math.ceil(span / maxPts));
  const points = {};
  for (const k of keys) {
    const arr = [];
    for (let s = since; s <= stepNow; s += stride) {
      if (k === 'val_loss' && s % 250 !== 0 && stride < 250) continue;
      arr.push([s, metricAt(k, s, total), j.started_at + s / 6]);
    }
    points[k] = arr;
  }
  return { keys: jobKeys(j), points, step: stepNow };
}

// ------------------------------------------------------------------ estimate / plan / simulate
function estimate(b) {
  const params = (b.model && b.model.params_b) || (state.models.find((m) => m.id === (b.model && b.model.id || b.model)) || {}).params_b || 8;
  const bits = ({ bf16: 16, fp16: 16, fp8: 8, w8a16: 8, w4a16: 4.25, nvfp4: 4.5, mxfp4: 4.25 })[b.quant || 'bf16'] || 16;
  const kvBits = ({ bf16: 16, fp16: 16, fp8_e4m3: 8, int8: 8, int4: 4 })[b.kv_dtype || 'bf16'] || 16;
  const chip = { hbm_gb: 80, bw: 3350, tflops: 989 };
  const w = params * 1e9 * bits / 8 / 1e9;
  const ctx = (b.workload && b.workload.context_lengths && b.workload.context_lengths[0]) || 4096;
  const kvPerTok = 2 * 32 * 8 * 128 * kvBits / 8; // 32 layers, 8 kv heads, 128 hd
  const kvSeq = kvPerTok * ctx / 1e9;
  const free = chip.hbm_gb * 0.9 - w;
  const maxConc = Math.max(0, Math.floor(free / kvSeq));
  const ideal = chip.bw / (w + kvSeq);
  return {
    fit: { fits: free > 0, weights_gb: w, kv_gb_per_seq: kvSeq, free_gb: free, max_concurrency: maxConc, provenance: 'derived' },
    bound: { kind: 'memory-bandwidth', batch_crossover: Math.round(chip.tflops * 1e3 / chip.bw * bits / 16 * 0.5 * 16), note: 'Decode is bandwidth bound below the crossover batch.', provenance: 'derived' },
    ceiling: { decode_tok_s_batch1: ideal, speedup_vs_bf16: 16 / bits, provenance: 'derived', note: 'Roofline ceiling, not a measurement.' },
    chip: 'H100 SXM 80GB (demo datasheet numbers)',
  };
}
function simulate(b) {
  const base = estimate({ ...b, model: b.model }); const conc = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  const ideal = base.ceiling.decode_tok_s_batch1; const maxC = Math.max(1, base.fit.max_concurrency);
  const curve = conc.filter((c) => c <= maxC * 2).map((c) => {
    const eff = Math.min(c, maxC);
    const tok = ideal * eff * (1 / (1 + (eff / 90))) * 0.82;
    return { concurrency: c, tok_s: tok, itl_ms: 1000 * eff / Math.max(tok, 1), kv_used_frac: Math.min(1, c / maxC), feasible: c <= maxC, provenance: 'derived' };
  });
  return { capacity: base.fit, ceiling: base.ceiling, curve, byte_breakdown: { weights_gb: base.fit.weights_gb, kv_gb_total: base.fit.kv_gb_per_seq * Math.min(maxC, 32), free_gb: base.fit.free_gb }, note: 'DEMO analytic model; not a measurement.' };
}
function plan(b) {
  const base = (q, kv, risk) => ({ ...estimate({ ...b, quant: q, kv_dtype: kv }), quant: q, kv_dtype: kv, risk });
  return {
    candidates: [
      { name: 'W4A16 GPTQ + FP8 KV', rationale: 'Weight bytes dominate decode traffic at batch 1-8; FP8 KV halves long-context reads.', est: base('w4a16', 'fp8_e4m3', 'low'), claims: ['P1-W4-001', 'P3-KV-003'], recipe: 'name: plan-w4-fp8kv\nstages:\n  - name: calibrate\n    params: {samples: 128}\n  - name: ptq\n    params: {method: gptq, bits: 4, group_size: 128}\n  - name: kv\n    params: {dtype: fp8_e4m3}\n  - name: eval\n    params: {gate: G-W4}\n' },
      { name: 'FP8 W8A8', rationale: 'Near-lossless on Hopper; halves weight bytes with native FP8 MMA.', est: base('fp8', 'bf16', 'very low'), claims: ['P1-FP8-020'], recipe: 'name: plan-fp8\nstages:\n  - name: ptq\n    params: {method: rtn, bits: 8, fmt: fp8_e4m3}\n  - name: eval\n    params: {gate: G-FP8}\n' },
      { name: 'NVFP4 weights (Blackwell only)', rationale: 'Needs block-scaled FP4 MMA; this chip does not have it - tier C fallback would dequantise.', est: base('nvfp4', 'fp8_e4m3', 'medium'), claims: ['P5-FP4-009'], recipe: 'name: plan-nvfp4\nstages:\n  - name: ptq\n    params: {method: rtn, fmt: nvfp4}\n', experimental: true },
    ],
    note: 'DEMO planner output. Ranking is by derived ceiling; no measurements back these numbers.',
  };
}

// ------------------------------------------------------------------ router
function notFound(path) { return new ApiError(`Not found: ${path}`, { status: 404, path }); }
function bad(msg, fieldErrors = {}) { return new ApiError(msg, { status: 422, fieldErrors }); }
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
function newId(p) { return `${p}_${Math.floor(Math.random() * 1e9).toString(36)}`; }

function validateParams(type, params) {
  const jt = JOB_TYPES.find((t) => t.type === type);
  if (!jt) throw bad(`Unknown job type ${type}`);
  const errs = {};
  for (const r of jt.schema.required || []) {
    if (r === 'dataset' || r === 'model') { /* ok if provided or default */ }
    if (params[r] === undefined || params[r] === '') errs[r] = 'is required';
  }
  if (type === 'train' || type === 'finetune') {
    const ds = params.data && params.data.dataset;
    if (!ds && !params.dataset && !params.data?.dataset) errs['data.dataset'] = 'is required (pick a dataset)';
  }
  if (Object.keys(errs).length) throw bad('Invalid job parameters', errs);
}

export async function mockRequest(method, path, { query = {}, body, form } = {}) {
  await delay(60 + Math.random() * 90);
  const [p0, qs] = path.split('?');
  const q = { ...Object.fromEntries(new URLSearchParams(qs || '')), ...(query || {}) };
  const p = p0.replace(/\/+$/, '');
  let m;
  const M = method.toUpperCase();
  const route = (re, ms) => { if (ms && !ms.includes(M)) return null; return re.exec(p); };

  // ---- system
  if (route(/^\/system\/info$/)) return { version: '0.1.0-demo', home: '~/.forge (demo)', server_time: now(), python: '3.11', torch: '2.x (demo)', cuda: '12.x (demo)', capabilities: { tier: 'A', sm: 90, name: 'H100 (demo)' }, store: { logical_bytes: 41e9, unique_bytes: 12.5e9, dedup_ratio: 3.3 }, jobs: countJobs() };
  if (route(/^\/system\/gpus$/)) return gpuSample();
  // ---- hardware
  if (route(/^\/hardware\/chips$/)) return mockChips();
  if (route(/^\/hardware\/descriptors$/)) return [mockDescriptor()];
  if ((m = route(/^\/hardware\/descriptors\/([^/]+)$/))) return mockDescriptor();
  if (route(/^\/hardware\/probe$/, ['POST'])) return addJob('probe', 'hardware probe', { quick: !!(body && body.quick) });
  // ---- models
  if (route(/^\/models$/, ['GET'])) return state.models;
  if (route(/^\/models$/, ['POST'])) { const mdl = { id: newId('m'), name: body.name || body.path_or_repo, source: body.source || 'hf', path_or_repo: body.path_or_repo, revision: body.revision || 'main', hash: null, licence: body.licence || null, distill_allowed: null, params_b: null, status: 'registered', created_at: now() }; state.models.push(mdl); return mdl; }
  if (route(/^\/models\/register-local$/, ['POST'])) { const mdl = { id: newId('m'), name: (body.path || '').split(/[\\/]/).pop() || 'local-model', source: 'local', path_or_repo: body.path, revision: null, hash: H(Math.floor(Math.random() * 1e6)), licence: 'unknown', distill_allowed: null, params_b: 1.2, status: 'ready', created_at: now() }; state.models.push(mdl); return mdl; }
  if ((m = route(/^\/models\/([^/]+)\/download$/, ['POST']))) return addJob('model_download', 'download ' + m[1], { repo: m[1] });
  if ((m = route(/^\/models\/([^/]+)\/inspect$/))) { const mdl = state.models.find((x) => x.id === m[1]); if (!mdl) throw notFound(p); return { id: mdl.id, config: { arch: 'llama', hidden_size: 4096, num_layers: 32, num_heads: 32, num_kv_heads: 8, head_dim: 128, vocab_size: 128256, max_position_embeddings: 131072 }, params_b: mdl.params_b, kv_bytes_per_token: { 16: 131072, 8: 65536, 4: 32768 }, licence: mdl.licence, distill_allowed: mdl.distill_allowed, provenance: 'derived' }; }
  if ((m = route(/^\/models\/([^/]+)$/))) {
    const i = state.models.findIndex((x) => x.id === m[1]); if (i < 0) throw notFound(p);
    if (M === 'GET') return state.models[i];
    if (M === 'PATCH') { Object.assign(state.models[i], body); return state.models[i]; }
    if (M === 'DELETE') { state.models.splice(i, 1); return { ok: true }; }
  }
  // ---- datasets
  if (route(/^\/datasets$/, ['GET'])) return state.datasets;
  if (route(/^\/datasets\/upload$/, ['POST'])) { const f = form && form.get && form.get('file'); const d = { id: newId('ds'), name: (f && f.name) || 'uploaded', total_tokens: 0, shards: [], mixture: {}, status: 'uploaded', created_at: now(), licence: null, dedup_report: {}, size_bytes: (f && f.size) || 0 }; state.datasets.push(d); return d; }
  if ((m = route(/^\/datasets\/([^/]+)\/ingest$/, ['POST']))) return addJob('dataset_ingest', 'ingest ' + m[1], { source: m[1] });
  if ((m = route(/^\/datasets\/([^/]+)\/preview$/))) return { id: m[1], kind: 'text', rows: ['The quick brown fox jumps over the lazy dog. (demo sample 1)', 'def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)  # demo sample 2', 'Forge LLM demo document: tokens are packed into 256 MiB shards.'], tokens_per_doc_p50: 412, tokens_per_doc_p99: 3980 };
  if ((m = route(/^\/datasets\/([^/]+)$/))) {
    const i = state.datasets.findIndex((x) => x.id === m[1]); if (i < 0) throw notFound(p);
    if (M === 'DELETE') { state.datasets.splice(i, 1); return { ok: true }; }
    return state.datasets[i];
  }
  // ---- schemas / configs
  if (route(/^\/schemas\/job-types$/)) return JOB_TYPES;
  if ((m = route(/^\/schemas\/job-types\/([^/]+)$/))) { const t = JOB_TYPES.find((x) => x.type === m[1]); if (!t) throw notFound(p); return t; }
  if (route(/^\/configs\/templates$/)) return templatesFor(q.type);
  if (route(/^\/configs$/, ['GET'])) return q.job_type || q.type ? state.configs.filter((c) => c.job_type === (q.job_type || q.type)) : state.configs;
  if (route(/^\/configs$/, ['POST'])) { const c = { id: newId('cfg'), job_type: body.job_type, name: body.name, params_json: body.params || body.params_json, created_at: now(), updated_at: now() }; state.configs.push(c); return c; }
  if ((m = route(/^\/configs\/([^/]+)$/))) { const i = state.configs.findIndex((x) => x.id === m[1]); if (i < 0) throw notFound(p); if (M === 'DELETE') { state.configs.splice(i, 1); return { ok: true }; } if (M === 'PUT') { Object.assign(state.configs[i], body, { updated_at: now() }); } return state.configs[i]; }
  // ---- jobs
  if (route(/^\/jobs$/, ['GET'])) {
    let list = state.jobs.slice().sort((a, b) => b.created_at - a.created_at);
    if (q.status) list = list.filter((j) => j.status === q.status);
    if (q.type) list = list.filter((j) => j.spec.type === q.type);
    return list.slice(0, Number(q.limit) || 200).map(pubJob);
  }
  if (route(/^\/jobs$/, ['POST'])) { validateParams(body.type, body.params || {}); const j = addJobRaw(body.type, body.name || body.type, body.params || {}, body.gpus, body.tags); return pubJob(j); }
  if ((m = route(/^\/jobs\/([^/]+)\/cancel$/, ['POST']))) { const j = jobById(m[1]); if (!j) throw notFound(p); if (['queued', 'running'].includes(j.status)) { j.status = 'cancelled'; j.finished_at = now(); } return pubJob(j); }
  if ((m = route(/^\/jobs\/([^/]+)\/clone$/, ['POST']))) { const j = jobById(m[1]); if (!j) throw notFound(p); return pubJob(addJobRaw(j.spec.type, j.spec.name + ' (clone)', clone(j.spec.params), j.spec.gpus, j.spec.tags)); }
  if ((m = route(/^\/jobs\/([^/]+)\/logs$/))) { const j = jobById(m[1]); if (!j) throw notFound(p); const lines = logLines(j); return { lines, offset: lines.length, text: lines.join('\n') }; }
  if ((m = route(/^\/jobs\/([^/]+)\/metrics$/))) { const j = jobById(m[1]); if (!j) throw notFound(p); return metricsFor(j, q); }
  if ((m = route(/^\/jobs\/([^/]+)$/))) { const j = jobById(m[1]); if (!j) throw notFound(p); if (M === 'DELETE') { state.jobs.splice(state.jobs.indexOf(j), 1); return { ok: true }; } return pubJob(j); }
  // ---- recipes
  if (route(/^\/recipes$/, ['GET'])) return state.recipes.map((r) => ({ id: r.id, name: r.name, created_at: r.created_at, n_stages: (r.yaml.match(/^\s*- name:/gm) || []).length }));
  if (route(/^\/recipes$/, ['POST'])) { const r = { id: newId('rcp'), name: body.name || 'untitled', yaml: body.yaml || '', created_at: now() }; state.recipes.push(r); return r; }
  if (route(/^\/recipes\/validate$/, ['POST'])) { const y = String(body.yaml || ''); const errs = []; if (!/^name:/m.test(y)) errs.push('missing "name"'); if (!/stages:/.test(y)) errs.push('missing "stages"'); return { valid: !errs.length, errors: errs, hash: errs.length ? null : H(y.length * 31), warnings: /nvfp4/.test(y) ? ['nvfp4 stages are experimental and gated'] : [] }; }
  if ((m = route(/^\/recipes\/([^/]+)\/run$/, ['POST']))) return addJob('recipe', 'recipe ' + m[1], { recipe_id: m[1] });
  if ((m = route(/^\/recipes\/([^/]+)$/))) { const i = state.recipes.findIndex((x) => x.id === m[1]); if (i < 0) throw notFound(p); if (M === 'DELETE') { state.recipes.splice(i, 1); return { ok: true }; } if (M === 'PUT') Object.assign(state.recipes[i], body); return state.recipes[i]; }
  // ---- methods + training simulator (DEMO DATA: crude closed-form numbers, no relation to measurements)
  if (route(/^\/methods$/, ['GET'])) return MOCK_METHODS.filter((x) => (!q.family || x.family === q.family) && (!q.status || x.status === q.status));
  if ((m = route(/^\/methods\/([^/]+)$/, ['GET']))) { const x = MOCK_METHODS.find((y) => y.id === m[1]); if (!x) throw notFound(p); return x; }
  if (route(/^\/planner\/train-sim$/, ['POST'])) return mockTrainSim(body || {});
  // ---- planner
  if (route(/^\/planner\/estimate$/, ['POST'])) return estimate(body || {});
  if (route(/^\/planner\/plan$/, ['POST'])) return plan(body || {});
  // ---- claims
  if (route(/^\/claims$/, ['GET'])) return state.claims;
  if (route(/^\/claims$/, ['PUT'])) { const i = state.claims.findIndex((c) => c.id === body.id); if (i < 0) throw notFound(p); Object.assign(state.claims[i], body); return state.claims[i]; }
  if ((m = route(/^\/claims\/([^/]+)$/))) { const c = state.claims.find((x) => x.id === m[1]); if (!c) throw notFound(p); if (M === 'PUT') Object.assign(c, body); return c; }
  // ---- artifacts / storage
  if (route(/^\/artifacts$/, ['GET'])) return state.artifacts.filter((a) => !q.kind || a.kind === q.kind).map((a) => ({ ...a, tier: state.tierOverrides[a.hash] || a.tier, pinned: state.pins.has(a.hash) || a.pinned }));
  if ((m = route(/^\/artifacts\/([^/]+)\/lineage$/))) return lineage(decodeURIComponent(m[1]));
  if ((m = route(/^\/artifacts\/([^/]+)\/(pin|unpin)$/, ['POST']))) { const a = state.artifacts.find((x) => x.hash === decodeURIComponent(m[1])); if (!a) throw notFound(p); a.pinned = m[2] === 'pin'; return a; }
  if ((m = route(/^\/artifacts\/([^/]+)\/tier$/, ['POST']))) { const a = state.artifacts.find((x) => x.hash === decodeURIComponent(m[1])); if (!a) throw notFound(p); a.tier = body.to; return a; }
  if ((m = route(/^\/artifacts\/([^/]+)$/))) { const a = state.artifacts.find((x) => x.hash === decodeURIComponent(m[1])); if (!a) throw notFound(p); return a; }
  if (route(/^\/storage\/stats$/)) return storageStats();
  if (route(/^\/storage\/tiers$/)) return [{ name: 'hot', path: '~/.forge/store/hot', bytes: 6.4e9, capacity_bytes: 200e9 }, { name: 'warm', path: '~/.forge/store/warm', bytes: 4.1e9, capacity_bytes: 1e12 }, { name: 'cold', path: '~/.forge/store/cold', bytes: 2.0e9, capacity_bytes: 4e12 }];
  if (route(/^\/storage\/gc$/, ['POST'])) {
    const dry = body && body.dry_run !== false;
    const rep = { dry_run: dry, candidates: [{ hash: H(60), kind: 'checkpoint', size: 8.4e6, reason: 'unreferenced delta (no pin)' }, { hash: H(61), kind: 'blob', size: 3.2e8, reason: 'orphan chunks' }, { hash: H(62), kind: 'checkpoint', size: 8.4e6, reason: 'unreferenced' }], bytes_freed: 3.4e8, kept_keyframes: 2, notes: ['delta-chain keyframes with live deltas are never collected'] };
    if (!dry) state.gcDone = true;
    return rep;
  }
  // ---- evals / gates
  if (route(/^\/evals$/, ['GET'])) return state.evals;
  if (route(/^\/evals\/compare$/, ['POST'])) return compare(body || {});
  if ((m = route(/^\/evals\/([^/]+)$/))) { const e = state.evals.find((x) => x.id === m[1]); if (!e) throw notFound(p); return e; }
  if (route(/^\/gates$/)) return [{ id: 'G-LOSSLESS', title: 'Lossless', description: 'Bit-exact roundtrip.' }, { id: 'G-FP8', title: 'FP8', description: 'KL p99 < 0.01; no flip-rate regression.' }, { id: 'G-W4', title: 'W4', description: 'KL p99 < 0.02; MMLU within noise.' }, { id: 'G-STRUCT', title: 'Structured pruning', description: 'Quality loss budget after healing.' }, { id: 'G-LONG', title: 'Long context', description: 'Needle / multi-key at 32k-128k.' }, { id: 'G-SPEC', title: 'Speculative decoding', description: 'Distribution exactness.' }, { id: 'G-FP4TRAIN', title: 'FP4 training', description: 'Loss gap vs BF16 within budget (experimental).' }].map((g) => ({ ...g, latest: state.evals.find((e) => e.gate && e.gate.gate_id.startsWith(g.id)) ? state.evals.find((e) => e.gate && e.gate.gate_id.startsWith(g.id)).gate : null }));
  if ((m = route(/^\/gates\/([^/]+)$/))) return { id: m[1], title: m[1], description: 'Demo gate' };
  // ---- kernels
  if (route(/^\/kernels$/, ['GET'])) return kernelCatalog();
  if (route(/^\/kernels\/results$/)) return kernelResults(q);
  if (route(/^\/kernels\/bench$/, ['POST'])) return addJob('bench', 'kernel bench', body && body.params || body || {});
  if (route(/^\/kernels\/autotune-cache$/)) return autotuneCache();
  // ---- deployments
  if (route(/^\/deployments$/, ['GET'])) return state.deployments;
  if (route(/^\/deployments$/, ['POST'])) { const d = { id: newId('dep'), name: body.name || 'deployment', artifact_ref: body.artifact_ref, serve_config: body.serve_config || {}, job_id: null, port: 8000 + state.deployments.length, status: 'starting', created_at: now() }; state.deployments.push(d); setTimeout(() => { d.status = 'running'; }, 2500); return d; }
  if ((m = route(/^\/deployments\/([^/]+)\/stats$/))) return depStats(m[1]);
  if ((m = route(/^\/deployments\/([^/]+)$/))) { const i = state.deployments.findIndex((x) => x.id === m[1]); if (i < 0) throw notFound(p); if (M === 'DELETE') { state.deployments.splice(i, 1); return { ok: true }; } return state.deployments[i]; }
  if (route(/^\/serve\/simulate$/, ['POST'])) return simulate(body || {});
  if (route(/^\/checkpoints$/)) return [{ id: 'ck_3500', run: 'job_demo_run', step: 3500, hash: H(46), kind: 'delta', size: 6e5, best: false }, { id: 'ck_3000', run: 'job_demo_run', step: 3000, hash: H(45), kind: 'keyframe', size: 8.4e6, best: true }];
  if ((m = route(/^\/checkpoints\/([^/]+)\/export$/, ['POST']))) return addJob('gc', 'export checkpoint ' + m[1], {});
  throw notFound(p);
}

export async function mockUpload(path, formData, onProgress) {
  for (let i = 1; i <= 10; i++) { await delay(80); if (onProgress) onProgress(i / 10); }
  return mockRequest('POST', path, { form: formData });
}

// ------------------------------------------------------------------ helpers for the router above
function countJobs() { const c = {}; for (const j of state.jobs) c[j.status] = (c[j.status] || 0) + 1; return c; }
function pubJob(j) {
  const out = clone(j);
  if (j.status === 'running' && j.total_steps) { const s = jobStepNow(j); out.progress = s / j.total_steps; out.progress_msg = `step ${s}/${j.total_steps}`; if (s >= j.total_steps) { j.status = 'succeeded'; j.finished_at = now(); out.status = 'succeeded'; } }
  if (j.status === 'running' && !j.total_steps && !['serve'].includes(j.spec.type)) { const el = now() - j.started_at; out.progress = Math.min(0.99, el / 25); out.progress_msg = 'working...'; if (el > 25) { j.status = 'succeeded'; j.finished_at = now(); j.progress = 1; out.status = 'succeeded'; out.progress = 1; } }
  if (j.status === 'queued' && j.id !== 'job_demo_queue' && now() - j.created_at > 3) { j.status = 'running'; j.started_at = now(); }
  return out;
}
function addJobRaw(type, name, params, gpus, tags) {
  const j = { id: newId('job'), spec: { type, name, params, gpus: gpus || null, tags: tags || [] }, status: 'queued', created_at: now(), started_at: null, finished_at: null, pid: null, exit_code: null, progress: 0, progress_msg: '', error: null, outputs: {}, run_dir: '~/.forge/runs/new', total_steps: ['train', 'finetune'].includes(type) ? Math.min(Number(params.max_steps) || 1200, 3000) : undefined };
  state.jobs.push(j);
  return j;
}
function addJob(type, name, params) { return pubJob(addJobRaw(type, name, params)); }
function logLines(j) {
  const lines = [`[info] job ${j.id} (${j.spec.type}) started`, '[info] environment captured: torch (demo), cuda (demo)', '[info] loaded config; seed=1234'];
  if (['train', 'finetune'].includes(j.spec.type)) {
    const step = jobStepNow(j); const total = j.total_steps || 5000;
    for (let s = 0; s <= step; s += Math.max(50, Math.floor(step / 40))) lines.push(`[info] step ${s}/${total} loss=${metricAt('loss', s, total).toFixed(4)} lr=${metricAt('lr', s, total).toExponential(2)} tok/s=${Math.round(metricAt('tokens_per_s', s, total))}`);
    if (j.status === 'failed') lines.push('[warn] loss spike detected at step 2210', '[warn] restart from checkpoint step 2000', '[error] ' + j.error);
    if (j.status === 'succeeded') lines.push('[info] training finished');
  } else if (j.status === 'failed') lines.push('[error] ' + j.error);
  else if (j.status === 'succeeded') lines.push('[info] done', ...Object.entries(j.outputs).map(([k, v]) => `[info] output ${k}=${v}`));
  return lines;
}
function templatesFor(type) {
  const jt = JOB_TYPES.find((t) => t.type === type);
  if (!jt) return [];
  return (jt.templates || []).map((name, i) => {
    const params = clone(jt.defaults);
    if (type === 'train' || type === 'finetune') { params.max_steps = i === 0 ? 200 : 20000; params.data = { ...(params.data || {}), dataset: 'ds_sft_chat' }; if (name.includes('fp4')) params.precision = { ...params.precision, rung: 'R2', healing_steps: 500 }; }
    return { name, type, description: 'Read-only template (demo)', params };
  });
}
function mockChips() {
  const C = (name, sm, family, hbm, bw, bf16, fp8, fp4, smem, l2, tier) => ({ name, sm, family, hbm_gb: hbm, hbm_bw_gbs: bw, tflops: { bf16, fp8, fp4 }, smem_kib: smem, l2_mb: l2, tier, provenance: 'vendor' });
  return {
    chips: [C('V100', 70, 'volta', 32, 900, 125, null, null, 96, 6, 'C'), C('T4', 75, 'turing', 16, 320, 65, null, null, 64, 4, 'C'), C('A100 80GB', 80, 'ampere', 80, 2039, 312, null, null, 164, 40, 'B'), C('RTX 3090', 86, 'ampere', 24, 936, 71, null, null, 100, 6, 'B'), C('L40S', 89, 'ada', 48, 864, 362, 733, null, 100, 96, 'B'), C('RTX 4090', 89, 'ada', 24, 1008, 165, 330, null, 100, 72, 'B'), C('H100 SXM', 90, 'hopper', 80, 3350, 989, 1979, null, 227, 50, 'A'), C('B200', 100, 'blackwell', 192, 8000, 2250, 4500, 9000, 227, 126, 'A'), C('RTX 5090', 120, 'blackwell', 32, 1792, 209, 419, 838, 100, 96, 'A')],
    support_matrix: [
      { sm: 70, examples: 'V100', bf16: 'emulated', fp8: false, fp4_block_scaled: false, cp_async: false, tma: false, two_four: false, tier: 'C' }, { sm: 75, examples: 'T4, RTX 20xx', bf16: 'emulated', fp8: false, fp4_block_scaled: false, cp_async: false, tma: false, two_four: false, tier: 'C' },
      { sm: 80, examples: 'A100, A30', bf16: true, fp8: false, fp4_block_scaled: false, cp_async: true, tma: false, two_four: true, tier: 'B' }, { sm: 86, examples: 'RTX 30xx, A10, A40', bf16: true, fp8: false, fp4_block_scaled: false, cp_async: true, tma: false, two_four: true, tier: 'B' },
      { sm: 89, examples: 'RTX 40xx, L4, L40S', bf16: true, fp8: true, fp4_block_scaled: false, cp_async: true, tma: false, two_four: true, tier: 'B' }, { sm: 90, examples: 'H100, H200, GH200', bf16: true, fp8: true, fp4_block_scaled: false, cp_async: true, tma: true, two_four: true, tier: 'A' },
      { sm: 100, examples: 'B200, GB200', bf16: true, fp8: true, fp4_block_scaled: true, cp_async: true, tma: true, two_four: true, tier: 'A' }, { sm: 120, examples: 'RTX 50xx (verify by probe)', bf16: true, fp8: true, fp4_block_scaled: 'verify by probe', cp_async: true, tma: true, two_four: true, tier: 'A' },
    ],
  };
}
function mockDescriptor() {
  const pair = (d, mm) => ({ datasheet: d, measured: mm, src: mm ? 'microbench (demo)' : undefined });
  return { id: 'h100-sxm-80g-demo', fingerprint: hex(7, 16), vendor: 'nvidia', family: 'hopper', name: 'NVIDIA H100 80GB HBM3 (demo)', sm: 90, compute_capability: '9.0', compute_units: 132, clock_mhz: 1755,
    memory: { hbm_gb: 80, hbm_bw_gbs: pair(3350, 2890), l2_mb: 50, transaction_bytes: 128, max_vector_load_bytes: 16 },
    on_chip: { smem_per_cu_kib: 227, smem_banks: 32, bank_width_bytes: 4, regs_per_cu: 65536, max_regs_per_thread: 255, async_copy: ['cp.async', 'tma'] },
    matrix_units: [{ dtype: 'bf16', atom: [64, 256, 16], peak_tflops: pair(989, 701) }, { dtype: 'fp8_e4m3', atom: [64, 256, 32], peak_tflops: pair(1979, 1390) }, { dtype: 'int8', atom: [64, 256, 32], peak_tflops: pair(1979, null) }],
    sparsity: { two_four: true }, launch_latency_us: 4.2, interconnect: { intra_gbs: 900, topology: 'switch' },
    derived: { ridge_flop_per_byte: { bf16: 242.6, fp8_e4m3: 480.9 } }, power_limit_w: 700, tier: 'A',
    probe_provenance: { 'memory.hbm_bw_gbs': 'measured', 'matrix_units.bf16': 'measured', 'matrix_units.int8': 'vendor', launch_latency_us: 'measured' } };
}
function lineage(hash) {
  const byHash = new Map(state.artifacts.map((a) => [a.hash, a]));
  const seen = new Set(); const nodes = []; const edges = [];
  const visit = (h) => { if (seen.has(h) || !byHash.has(h)) return; seen.add(h); const a = byHash.get(h); nodes.push({ id: h, hash: h, kind: a.kind, label: a.name }); for (const par of a.parents) { edges.push({ child: h, parent: par, relation: 'derived_from' }); visit(par); } for (const c of state.artifacts) if (c.parents.includes(h) && !seen.has(c.hash)) { visit(c.hash); } };
  visit(hash);
  return { root: hash, nodes, edges };
}
function storageStats() {
  const kinds = {}; let logical = 0; let stored = 0;
  for (const a of state.artifacts) { const k = kinds[a.kind] || (kinds[a.kind] = { logical_bytes: 0, unique_bytes: 0, count: 0 }); k.logical_bytes += a.size; k.unique_bytes += a.stored_size; k.count++; logical += a.size; stored += a.stored_size; }
  return { logical_bytes: logical, unique_bytes: stored, dedup_ratio: logical / stored, per_kind: kinds, per_tier: { hot: 6.4e9, warm: 4.1e9, cold: 2.0e9 }, naive_bytes: logical * 1.0, compaction_vs_naive: logical / stored, algo: 'blake3', provenance: 'measured', gc_last: state.gcDone ? now() - 5 : null };
}
function compare(b) {
  const ids = b.ids && b.ids.length ? b.ids : state.evals.map((e) => e.id);
  const rows = ids.map((id) => state.evals.find((e) => e.id === id)).filter(Boolean).map((e) => ({ id: e.id, label: e.label, artifact_hash: e.artifact_hash, metrics: e.metrics, gate: e.gate }));
  const metrics = Array.from(new Set(rows.flatMap((r) => Object.keys(r.metrics))));
  return { parent: b.parent || 'ev_bf16', metrics, rows, directions: { kl_p99: 'min', mmlu_acc: 'max', decode_tok_s: 'max', weight_gb: 'min' }, noise_floor: { kl_p99: 0.0006, mmlu_acc: 0.004 } };
}
function kernelCatalog() {
  const K = (name, family, minsm, avail, reasons = []) => ({ name, family, spec: `kernels/specs/${name}.yaml`, available: avail, reasons, min_sm: minsm, triton: true, validated_on_hardware: false });
  return [K('w4a16_gemm', 'gemm', 70, true), K('w8a16_gemm', 'gemm', 70, true), K('fp8_gemm', 'gemm', 89, true), K('fp4_block_gemm', 'gemm', 100, false, ['needs block-scaled FP4 MMA (sm_100+); this GPU is sm_90, falling back to dequant path']), K('paged_decode_attention', 'attention', 70, true), K('sparse_topk_attention', 'attention', 80, true), K('tree_attention', 'attention', 80, true), K('moe_grouped_gemm', 'moe', 80, true), K('hadamard_transform', 'utility', 70, true), K('stochastic_round', 'utility', 70, true), K('rmsnorm_quant', 'utility', 70, true), K('rejection_verify', 'utility', 70, true)];
}
function kernelResults(q) {
  const out = [];
  const names = ['w4a16_gemm', 'fp8_gemm', 'paged_decode_attention', 'hadamard_transform', 'rmsnorm_quant'];
  names.forEach((n, i) => {
    for (const shape of ['M=1,N=4096,K=4096', 'M=16,N=4096,K=4096', 'M=256,N=4096,K=4096']) {
      const memBound = !shape.startsWith('M=256');
      const ai = memBound ? 2 + i : 120 + i * 20;
      out.push({ kernel: n, chip: 'h100-sxm-80g-demo', shape, dtype: n.includes('fp8') ? 'fp8' : 'bf16', time_us: 18 + i * 3 + (memBound ? 0 : 40), iqr_us: 0.6, runs: 20, achieved_gbs: memBound ? 2300 + i * 80 : 900, pct_of_peak: memBound ? 0.80 + i * 0.02 : 0.31, achieved_tflops: memBound ? ai * 2.3 : 520 + i * 10, ai_flop_per_byte: ai, bound: memBound ? 'memory' : 'compute', residual_vs_model: (hnoise(i * 7 + shape.length) * 0.16), oracle_ok: true, provenance: 'measured' });
    }
  });
  return q.kernel ? out.filter((r) => r.kernel === q.kernel) : out;
}
function autotuneCache() {
  const entries = [];
  for (const op of ['w4a16_gemm', 'fp8_gemm']) {
    const cand = [];
    for (const bm of [16, 32, 64, 128]) for (const bn of [32, 64, 128, 256]) for (const w of [4, 8]) {
      const t = 30 + Math.abs(Math.log2(bm / 32)) * 6 + Math.abs(Math.log2(bn / 128)) * 5 + (w === 8 ? 2 : 0) + hnoise(bm * 31 + bn + w, 9) * 3 + (op === 'fp8_gemm' ? -4 : 0);
      cand.push({ config: { BLOCK_M: bm, BLOCK_N: bn, num_warps: w }, time_us: t, oracle_ok: !(bm === 128 && bn === 256 && w === 4) });
    }
    entries.push({ key: `${op}|h100-sxm|bf16|M<=16,N=4096,K=4096|layout=auto`, op, chip: 'h100-sxm-80g-demo', shape_bucket: 'M<=16,N=4096,K=4096', best: cand.filter((c) => c.oracle_ok).sort((a, b) => a.time_us - b.time_us)[0], candidates: cand, throttle_detected: false });
  }
  return { path: '~/.forge/autotune/cache.json', entries };
}
function depStats(id) {
  const d = state.deployments.find((x) => x.id === id); if (!d) throw notFound('/deployments/' + id);
  const t = now() - T0;
  const run = d.status === 'running';
  return { running: run ? Math.round(12 + 6 * Math.sin(t / 4)) : 0, waiting: run ? Math.max(0, Math.round(3 + 4 * Math.sin(t / 6))) : 0, kv_usage: run ? 0.52 + 0.1 * Math.sin(t / 9) : 0, prefix_hit_rate: 0.38, tokens_per_s: run ? 2600 + 300 * Math.sin(t / 3) : 0, ttft_ms: { p50: 112, p99: 410 }, itl_ms: { p50: 11.8, p99: 26.4 }, spec_acceptance: [0.82, 0.64, 0.47, 0.31], spec_mode: 'ngram', requests_total: 1423, uptime_s: Math.round(now() - d.created_at) };
}

// ------------------------------------------------------------------ streaming
const CANNED = ['This is a ', 'DEMO ', 'response ', 'from ', 'the ', 'in-browser ', 'fake ', 'backend. ', 'No ', 'model ', 'is ', 'running. ', 'When ', 'connected ', 'to ', 'a ', 'real ', 'Forge ', 'server, ', 'tokens ', 'stream ', 'here ', 'from ', 'the ', 'serving ', 'engine ', 'one ', 'by ', 'one.'];
export function mockStream(path, opts = {}) {
  const [p0] = path.split('?'); let timer = null; let closed = false; let n = 0;
  const emit = (ev) => { if (!closed && opts.onEvent) opts.onEvent(ev); };
  const status = (s) => { if (!closed && opts.onStatus) opts.onStatus(s); };
  status('open');
  let m;
  if (p0.endsWith('/system/gpus/stream')) {
    timer = setInterval(() => emit({ event: 'gpus', data: gpuSample() }), 1000); emit({ event: 'gpus', data: gpuSample() });
  } else if (p0 === '/events') {
    timer = setInterval(() => emit({ event: 'heartbeat', data: { t: now() } }), 5000);
  } else if ((m = /^\/jobs\/([^/]+)\/stream$/.exec(p0))) {
    const j = jobById(m[1]); let last = j ? jobStepNow(j) : 0;
    timer = setInterval(() => {
      if (!j) return;
      const pj = pubJob(j);
      if (j.status === 'running') {
        const cur = jobStepNow(j); const total = j.total_steps || 5000;
        for (let s = last + 1; s <= cur; s += Math.max(1, Math.floor((cur - last) / 4))) {
          const values = {}; for (const k of jobKeys(j)) if (k !== 'val_loss' || s % 250 === 0) values[k] = metricAt(k, s, total);
          emit({ event: 'metric', data: { step: s, values } });
        }
        if (cur > last) emit({ event: 'log', data: { line: `[info] step ${cur}/${total} loss=${metricAt('loss', cur, total).toFixed(4)} tok/s=${Math.round(metricAt('tokens_per_s', cur, total))}` } });
        last = cur;
        emit({ event: 'progress', data: { progress: pj.progress, msg: pj.progress_msg } });
      }
      if (pj.status !== 'running') { emit({ event: 'status', data: { status: pj.status } }); }
    }, 1000);
  } else if ((m = /^\/deployments\/([^/]+)\/chat$/.exec(p0))) {
    const lastUser = (opts.body && opts.body.messages && opts.body.messages.slice(-1)[0] && opts.body.messages.slice(-1)[0].content) || '';
    const toks = [`(you said: "${String(lastUser).slice(0, 40)}") `, ...CANNED];
    timer = setInterval(() => {
      if (n < toks.length) emit({ event: 'message', data: { choices: [{ delta: { content: toks[n] }, index: 0 }] } });
      else { emit({ event: 'message', data: { choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 24, completion_tokens: toks.length } } }); emit({ event: 'message', data: '[DONE]', raw: '[DONE]' }); clearInterval(timer); status('closed'); }
      n++;
    }, 70);
  }
  return { close() { closed = true; if (timer) clearInterval(timer); } };
}
