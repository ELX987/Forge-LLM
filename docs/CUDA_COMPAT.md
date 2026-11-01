# CUDA compatibility check

`forge/hw/cuda_compat.py` (stdlib-only) inspects the whole CUDA stack of a host and reports what is wrong and how to fix it,
instead of letting a job fail later with an opaque `cudaErrorDevicesUnavailable`. The motivating case: a host with driver 580
(CUDA 13.0) running a CUDA 13.4 torch, where every CUDA call failed deep inside a job.

## What is checked

- NVIDIA driver version and the newest CUDA the driver supports (NVML, falling back to the `nvidia-smi` header); GPUs, their
  compute capability (sm) and compute mode.
- torch build: CUDA version, ROCm/CPU-only builds, arch list, NVIDIA container builds (`...nvNN`; never pip-replaced).
- torch CUDA vs driver CUDA (major mismatch = fail, newer minor = warn), driver vs the per-family floor.
- GPU architecture vs the torch build (CUDA floor per sm, arch list contents, Volta and older with CUDA 13).
- Triton (present; >= 3.3 for Blackwell), `nvidia-ml-py` (present, major not older than the driver's CUDA major), `nvcc` vs torch.
- Compute mode Prohibited/Exclusive_Process.
- A real allocation on every torch-visible device, run in a throwaway subprocess so the calling process gets no CUDA context.

Findings have a level (`ok`, `info`, `warn`, `fail`), a code, a message and a fix. Only `fail` stops anything.

## Rule constants (from `forge/hw/cuda_compat.py`)

Minimum Linux driver for any toolkit of a CUDA major family (`DRIVER_FLOOR`):

| CUDA family | driver |
|---|---|
| 11.x | 450.80.02 |
| 12.x | 525.60.13 |
| 13.x | 580.65.06 |

Driver that natively supports a CUDA release (`NATIVE_DRIVER`):

| CUDA | driver | source |
|---|---|---|
| 12.0 | 525.60.13 | spec |
| 12.1 | 530.30.02 | spec |
| 12.2 | 535.54.03 | spec |
| 12.3 | 545.23.06 | spec |
| 12.4 | 550.54.14 | spec |
| 12.5 | 555.42.02 | spec |
| 12.6 | 560.28.03 | spec |
| 12.8 | 570.26.00 | spec |
| 12.9 | 575.51.03 | spec |
| 13.0 | 580.65.06 | spec |
| 13.1 | 590.44.01 | spec |
| 13.4 | 595.91.07 | observed |

CUDA releases not listed (for example 13.2, 13.3) use the nearest older entry and the message says the value is not exact.

Oldest CUDA toolkit that can target an architecture (`ARCH_MIN_CUDA`): sm_75 10.0, sm_80 11.0, sm_86 11.1, sm_87 11.4,
sm_89 11.8, sm_90 11.8, sm_100 12.8, sm_101 12.8, sm_103 12.9, sm_110 13.0, sm_120 12.8, sm_121 12.9.

Other constants: CUDA 13 torch needs sm >= 75 (`CUDA13_MIN_SM`); Triton >= 3.3 for sm_100 and newer (`TRITON_BLACKWELL_MIN`).

Recommended torch wheel line by driver CUDA (`recommend`): >= 13.0 `cu130`, >= 12.9 `cu129`, >= 12.8 `cu128`, >= 12.6 `cu126`,
>= 12.4 `cu124`. A GPU older than Turing downgrades `cu130` to `cu129`; Blackwell with driver CUDA < 12.8 gives no wheel
(the driver must be upgraded).

## CUDA 13.0 - 13.4 notes

- All CUDA 13.x toolkits share one driver floor (580.65.06) through minor-version compatibility, so a CUDA 13.4 torch can
  start on a driver that only supports CUDA 13.0. This is reported as `torch_cuda_newer_minor` (warn), not a failure,
  because it is not guaranteed to work: PTX JIT of newer code, newer-driver features and some multi-GPU/graph paths can fail at first use.
- If CUDA calls then fail with "busy or unavailable", that mismatch is the first thing to rule out; the fix is a host with
  a newer driver. A torch with a newer CUDA *major* than the driver supports is a hard `fail`.
- CUDA 13 dropped offline compilation for Maxwell/Pascal/Volta (sm_50..sm_72): a CUDA 13 torch cannot run on a V100 or older
  (`cuda13_dropped_arch`, fail); use a CUDA 12.x build for those GPUs.
- Observed vs spec: driver numbers for 13.0 and 13.1 are from NVIDIA release notes. 595.91.07 for CUDA 13.4 is only the driver a
  CUDA 13.4 torch was seen running correctly on. No driver minimum is recorded for 13.2 or 13.3. Verify against the CUDA
  toolkit release notes before relying on a number.

## Where it runs

- `scripts/vast_setup.sh`: prints the report, picks the torch wheel index from the driver when installing torch
  (`FORGE_TORCH_INDEX` overrides), never replaces a torch that already sees CUDA (or an NVIDIA container build), runs the
  full check after installing Forge, installs the recommended `nvidia-ml-py` / `triton` without touching torch, and exits 3 on a failure.
- `scripts/gpu_smoke.sh`: runs the check before the torch step and stops with the report on failure.
- `scripts/run_gpu_tests.sh`: `gpu_report.py cuda-health` uses the check; a failure stops the run with exit 3.
  The findings and stack are written to `gpu_report/cuda_health.json` and a "CUDA stack" section of `report.md`.
- Job workers can call `forge.hw.cuda_compat.assert_cuda_usable()`, which raises `CudaUnusable` with the diagnosis.

## Overrides

- `FORGE_SKIP_CUDA_CHECK=1`: continue in `vast_setup.sh`, `gpu_smoke.sh` and `run_gpu_tests.sh` even when the check fails.
  (`FORGE_ALLOW_UNUSABLE_CUDA=1` still works for `run_gpu_tests.sh`.)
- `FORGE_TORCH_INDEX=<url>`: force the torch wheel index in `vast_setup.sh`.

## Commands

```
python -m forge.hw.cuda_compat                 # report, exit 1 on a fail finding
python -m forge.hw.cuda_compat --json          # full machine-readable report
python -m forge.hw.cuda_compat --no-probe      # skip the allocation test (does not import torch)
python -m forge.hw.cuda_compat --strict        # exit 1 on warnings too
python -m forge.hw.cuda_compat --torch-index   # e.g. cu130, empty if unknown
python -m forge.hw.cuda_compat --requirements  # recommended companion requirements
```
