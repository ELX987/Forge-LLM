# Forge LLM

Unified **training, fine-tuning, compression and serving** system for LLMs, built from the Forge LLM System Plan
(`plan-and-docs/`). It combines:

- a **kernel layer** (Triton kernels with pure-PyTorch reference oracles and automatic fallbacks) targeting consumer and
  datacenter NVIDIA GPUs from `sm_70` (V100) to `sm_121` (GB10);
- a **Python full-stack web app** (FastAPI + dependency-free vanilla-JS front end) to configure, launch and monitor
  training / fine-tuning / compression / evaluation / inference, and to manage models, datasets, checkpoints and artifacts;
- supporting components from the Plan: content-addressed artifact store, hardware profiler, cost model and planner,
  evaluation harness with gates, compression engine, serving engine, and a claims ledger.

> Status: early. All GPU code paths are written to run on Linux + NVIDIA but are validated only on a GPU host
> (see `docs/KERNELS.md` for what is implemented vs. validated). CPU unit tests cover everything that does not need a GPU.

## Quickstart: local CPU development (any OS)

> Do **not** use this on a GPU host: the CPU-only torch wheel replaces CUDA torch, and on some Linux builds it fails on
> `import torch` (`AttributeError: module 'torch._C' has no attribute '_has_gds'`). On vast.ai use the section below.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
make test                                               # or: python -m pytest tests -q
export FORGE_HOME=$PWD/.forge                           # where runs, store, db and models live (default ~/.forge)
forge serve-ui                                          # web UI + API on http://127.0.0.1:8765
```

## Quickstart: vast.ai (Linux + NVIDIA)

1. Rent an instance with a recent PyTorch + CUDA image and an exposed port (or use the SSH tunnel).
2. `git clone <your private repo> && cd Forge-LLM && bash scripts/vast_setup.sh` (keeps the image's CUDA torch; do not create a venv with CPU torch)
3. `bash scripts/run_server.sh` then open the forwarded port (or `ssh -L 8765:localhost:8765 ...`).
4. `bash scripts/gpu_smoke.sh` and `bash scripts/run_gpu_tests.sh` to validate kernels on your GPU.

Details: [`docs/VAST_AI.md`](docs/VAST_AI.md).

## Documentation

| Doc | Content |
|-----|---------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module layout, ownership, public APIs, conventions |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Using the UI and CLI |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog and validation status |
| [`docs/CUDA_COMPAT.md`](docs/CUDA_COMPAT.md) | CUDA driver / torch / GPU / Triton compatibility check (CUDA 12.x, 13.0-13.4) |
| [`docs/API.md`](docs/API.md) | REST API reference |
| [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) | Every configuration option |
| [`configs/`](configs) | Ready-to-use config and recipe templates |

## Layout

```
forge/core      schemas, hashing, db, registry, run context, recipe engine
forge/store     content-addressed artifact store
forge/hw        chip database, capability probe, microbenchmarks
forge/kernels   Triton kernels, reference oracles, layouts, autotuner
forge/models    decoder-only transformer, loaders, LoRA, KV caches
forge/compress  PTQ, KV quantisation, pruning, low-rank, delta, lossless
forge/train     data pipeline, trainer, fine-tuning, distillation
forge/serve     inference engine, scheduler, OpenAI-compatible server
forge/eval      metrics, suites, gates     forge/planner  cost model, claims ledger
forge/jobs  forge/api  forge/telemetry  forge/web   job manager, REST API, metrics, front end
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
