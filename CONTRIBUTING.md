# Contributing to Forge LLM

Thanks for helping. Read `docs/ARCHITECTURE.md` first: it is the contract for module layout, public APIs and
conventions. Decisions that deviate from it are recorded as ADRs in `docs/adr/`.

## Setup (CPU is enough for most work)

```bash
bash scripts/dev_setup_cpu.sh            # Linux / macOS / Git-Bash
powershell -File scripts/dev_setup_cpu.ps1   # Windows
python -m pytest tests -q
```

GPU code can only be validated on an NVIDIA host (see `docs/VAST_AI.md`, `docs/KERNELS.md`).

## Ground rules

1. **Honesty over polish.** Never claim a GPU path works unless `scripts/run_gpu_tests.sh` passed on real
   hardware and you can attach the report. Update the validation table in `docs/KERNELS.md` only with evidence.
2. **No placeholders.** No `TODO`/`pass` bodies, no fake data presented as measurements. Experimental or gated
   features are implemented and clearly marked.
3. **Oracle before kernel.** Every Triton kernel has a pure-torch reference in `forge/kernels/ref/`, a launcher
   with fallback in `forge/kernels/ops_*.py`, and a GPU test in `tests_gpu/`.
4. **CPU tests for everything testable without NVIDIA**, with tiny synthetic models. Never download weights or hit
   the network in tests. Keep each test file under ~30 s.
5. **Typed configs.** Config models are pydantic v2 with a `Field(description=...)` on every field; the UI builds
   forms from the JSON Schema. Adding a field should need no front-end change.
6. **Lazy heavy imports** (torch, triton, transformers, pynvml, pyarrow) so `import forge.<module>` stays cheap and
   works without them. `from __future__ import annotations` at the top of every module; Python >= 3.10 syntax.
7. **No CDN or remote URLs in the front end** (ADR 0006).
8. **Config templates** in `configs/` must load through their pydantic model; `tests/test_configs_templates.py`
   enforces that and rejects unknown fields.
9. Shell scripts: LF line endings, `set -euo pipefail`, shellcheck-clean.

## Workflow

* Branch from `main`, keep changes focused, run `python -m pytest tests -q` and `python -m ruff check forge tests tests_gpu`.
* Update `CHANGELOG.md` (Unreleased section) and any affected doc (`docs/API.md` for endpoints,
  `docs/CONFIG_REFERENCE.md` for new templates, `docs/ROADMAP_STATUS.md` when a Plan item changes status).
* For performance claims, include chip, driver, torch/triton versions, shapes and the `gpu_report/report.md`.

## Adding things

* A kernel or chip: `docs/KERNELS.md` ("Adding a kernel") and `docs/runbooks/adding-a-chip-descriptor.md`.
* A recipe stage: register it with `@register("stage", "<name>")` and give it a `code_version`.
* An ADR: copy the shape of `docs/adr/0001-python-first-stack.md`, number sequentially.

## Licence

By contributing you agree your contributions are licensed under Apache-2.0 (see `LICENSE`).
