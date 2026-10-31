# ADR 0001: Python-first stack instead of Rust core + React front end

- Status: accepted
- Date: 2026-10-30
- Deciders: project owner
- Related Plan items: UI-1, C-017, C-070, C-080 (desktop/Rust sidecar), NFR-V1

## Context

The Plan (section 9, section 11.4) sketches a Rust sidecar/daemon for native access and a typed single-page
application built with a JavaScript toolchain. The stated goals for this implementation are narrower:

1. A kernel setup and a full-stack **Python** web app that run on a Linux host rented on vast.ai.
2. The development machine is Windows without an NVIDIA GPU, so everything must import and unit-test on CPU.
3. The user wants to open-source the result; a small, readable code base lowers the barrier to contribution.

A Rust daemon plus a Node build would add two more toolchains that must be installed on every rented instance
(instances are short-lived and billed by the hour), and none of the Rust-only capabilities are required for
the first release (GPU telemetry is available from `pynvml` / `nvidia-smi`).

## Decision

* The whole system is **Python >= 3.10**: FastAPI for the control plane, `sqlite3` from the standard library for
  state, subprocess-per-job workers, PyTorch + Triton for compute.
* The front end is **vanilla ES modules served as static files** by the API process (see ADR 0006). There is no
  Node, no bundler, no `package.json`.
* The Rust sidecar, desktop shell and offline-sync items of Plan Phase 7 (C-080..C-082) are **deferred**. The
  browser UI served from the rented box (through an SSH tunnel) covers the same screens.

## Consequences

* Positive: one `pip install -e .` gets a working system; setup scripts stay short; CPU CI is simple.
* Positive: contributors need only Python.
* Negative: Python is slower than Rust for the control plane. Mitigation: the control plane is not on the hot
  path; all hot loops are in torch/Triton kernels or in worker subprocesses.
* Negative: no native desktop integration (tray alerts, local file pickers). Browser equivalents are used.
* The module boundaries in `docs/ARCHITECTURE.md` stay narrow, so a Rust sidecar can be added later behind the
  same REST API without changing the Python modules.
