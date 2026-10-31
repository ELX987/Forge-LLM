# ADR 0004: Forge-native decoder implementation with a Hugging Face fallback

- Status: accepted
- Date: 2026-10-30
- Related Plan items: SV-1, SV-3, KR-7, C-050 ("build vs adapt via ADR"), C-111

## Context

The Plan asks (C-050) for an ADR deciding whether the serving engine is built or an existing engine is
adapted. We need to: swap `Linear` layers for packed quantised linears that call our kernels, use a paged KV
cache with our block manager, attach LoRA, run MoE with our grouped GEMM, and unit-test on CPU with tiny
models. The `transformers` library is excellent for loading and tokenisation but its modelling code is not
designed around paged KV views, per-layer module replacement contracts or our kernel launchers.

## Decision

* **`forge.models.ForgeLM` is a compact, self-contained decoder-only transformer** (llama, mistral, qwen2,
  qwen3 style, plus MoE variants) whose linear layers come from a registry, so compression and LoRA are plain
  module replacement.
* The serving engine (`forge.serve`) drives `ForgeLM` with a `PagedKVView` and our attention kernels.
* **HF fallback**: `load_model(..., backend="hf")` loads through `transformers` for architectures ForgeLM does
  not implement. HF-backed models can be trained/fine-tuned and evaluated, but the paged serving fast path and
  packed-kernel replacement are only guaranteed for the native backend.
* Tokenisers always come from `transformers`/`tokenizers`.
* Weights are exchanged as HF-compatible safetensors + `config.json`, so artifacts remain usable outside Forge.

## Consequences

* We own correctness of the architecture implementation. Mitigation: parity tests of tiny random models
  against `transformers` when it is installed, and logit-level tests during GPU smoke.
* Unsupported architectures need the fallback and lose some features; the UI marks the backend per model.
* We do not claim to match vLLM / TensorRT-LLM throughput. Plan C-022 baselines against an existing engine
  remain an experiment to run on hardware.
