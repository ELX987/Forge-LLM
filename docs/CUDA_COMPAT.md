# CUDA compatibility check

`forge/hw/cuda_compat.py` (stdlib-only) inspects the whole CUDA stack of a host and reports what is wrong and how to fix it,
instead of letting a job fail later with an opaque `AcceleratorError` / `cudaErrorDevicesUnavailable`.

The field case that motivated it: a 4x RTX 5090 host where every CUDA call failed with `cudaErrorDevicesUnavailable`. That
error is not a version problem. It means the GPUs cannot be used by this process: compute mode (Exclusive_Process or
Prohibited), another process holding the device, a container that was not given the GPUs it lists, or a driver/fabric fault.
The check therefore allocates on every device and, when that fails, lists exactly those causes.

## What is checked

- NVIDIA driver version and the newest CUDA the driver supports (NVML, falling back to the `nvidia-smi` header); GPUs, their
  compute capability (sm) and compute mode.
- torch build: CUDA version, ROCm/CPU-only builds, arch list, NVIDIA container builds (`...nvNN`; never pip-replaced).
- torch CUDA vs driver CUDA, and the driver vs the per-family floor.
- GPU architecture vs the torch build (CUDA floor per sm, arch list contents, pre-Turing GPUs with CUDA 13).
- Triton (present; >= 3.3 for Blackwell), `nvidia-ml-py` (major not older than the driver's CUDA major), `nvcc` vs torch.
- Compute mode Prohibited / Exclusive_Process.
- A real allocation on every torch-visible device, run in a throwaway subprocess so the calling process gets no CUDA context.

## Findings and levels

Each finding has a level (`ok`, `info`, `warn`, `fail`), a code, a message and a fix. Only `fail` stops anything
(`--strict` also exits 1 on `warn`).

| code | level | when |
|---|---|---|
| `nvidia_smi_failed` | fail | `nvidia-smi` exists but cannot query the GPUs |
| `gpus_hidden` | info | `CUDA_VISIBLE_DEVICES` is empty or `-1` (hidden on purpose) |
| `no_nvidia_driver` | warn | no `nvidia-smi` or no devices listed (fine for the UI and CPU tests) |
| `driver`, `torch`, `driver_torch_match`, `triton`, `devices_usable` | ok | the layer looks right |
| `torch_missing`, `torch_rocm`, `torch_cpu_only` | fail | no torch, a ROCm build, or a CPU-only build on a GPU host |
| `driver_below_family_floor` | fail | driver older than the floor of the torch's CUDA major (table below) |
| `torch_cuda_newer_major` | fail | torch needs a CUDA major the driver does not support (e.g. torch 13.x on a CUDA 12 driver) |
| `torch_cuda_newer_minor` | info | torch's CUDA minor is newer than the driver's, same major: supported (see below) |
| `cuda13_dropped_arch` | fail | CUDA 13 torch with a pre-Turing GPU (sm < 75) |
| `arch_needs_newer_cuda` | fail | the GPU's sm needs a newer CUDA than the torch was built with |
| `arch_family_sass` | info | the GPU runs on same-family SASS in this torch (compatible, not tuned) |
| `arch_ptx_only` | warn | no native code for the GPU, PTX JIT at first use |
| `arch_not_in_torch` | fail | the torch arch list has no code for the GPU |
| `triton_missing` | warn | Forge kernels fall back to slower torch paths |
| `triton_too_old_for_blackwell` | warn | Triton < 3.3 with an sm_100+ GPU |
| `nvml_missing` | info | `nvidia-ml-py` absent, telemetry falls back to `nvidia-smi` |
| `nvml_bindings_old` | warn | `nvidia-ml-py` major older than the driver's CUDA major |
| `nvcc_major_mismatch` | warn | `nvcc` major differs from torch's (matters only when building extensions) |
| `nvcc_minor_mismatch` | info | same major, different minor (extension builds normally work) |
| `compute_mode_prohibited` | fail | no process can create a CUDA context |
| `compute_mode_exclusive` | warn | one process at a time; Forge runs a worker per job plus the server probe |
| `cuda_probe_failed` | fail | the allocation test itself could not run |
| `torch_sees_no_gpu` | fail | CUDA torch sees no device although the driver lists GPUs |
| `devices_unusable` | fail | one or more devices failed the allocation test |
| `some_devices_usable` | info | names the working devices and the `CUDA_VISIBLE_DEVICES` value to use |

## Rule constants (from `forge/hw/cuda_compat.py`)

Minimum Linux driver for any toolkit of a CUDA major family (`DRIVER_FLOOR`):

| CUDA family | driver |
|---|---|
| 11.x | 450.80.02 |
| 12.x | 525.60.13 |
| 13.x | 580.65.06 |

Driver that natively supports a CUDA release (`NATIVE_DRIVER`). `spec` = NVIDIA release-note GA minimum; `branch` = NVIDIA
documents only the driver branch (the value is "that branch or newer"; there is no patch-level minimum):

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
| 13.0 | 580.65.06 | branch (R580) |
| 13.1 | R590 | branch |
| 13.2 | R595 | branch |
| 13.3 | R610 | branch |
| 13.4 | R615 | branch |

CUDA releases not listed (for example 12.7) use the nearest older entry and the message says the value is not exact.

Oldest CUDA toolkit that can target an architecture (`ARCH_MIN_CUDA`): sm_75 10.0, sm_80 11.0, sm_86 11.1, sm_87 11.4,
sm_89 11.8, sm_90 11.8, sm_100 12.8, sm_101 12.8, sm_103 12.9, sm_110 13.0, sm_120 12.8, sm_121 12.9.

Other constants: a CUDA 13 torch needs sm >= 75 (`CUDA13_MIN_SM`); Triton >= 3.3 for sm_100 and newer (`TRITON_BLACKWELL_MIN`).

## CUDA 13.0 - 13.4

- **Minor-version compatibility.** NVIDIA documents that any CUDA 13.x toolkit or torch runs on any driver >= 580. Their own
  example is a CUDA 13.4 application on a 580.65.06 driver. So a torch built for CUDA 13.4 on a host whose driver reports
  CUDA 13.0 is supported. Forge reports it as `torch_cuda_newer_minor` at level `info`, never `warn` or `fail`, and never
  names it as a probable cause of a failure.
- **What the info means.** The driver's branch decides the full feature set, not whether things start. With an older branch,
  PTX JIT of newer code and features added after that driver's release are unavailable. For the full feature set use a driver
  from the branch of the toolkit: 13.0 R580, 13.1 R590, 13.2 R595, 13.3 R610, 13.4 R615. NVIDIA documents branches only, so
  there is no patch-level minimum (595.91.07, for instance, is not "the CUDA 13.4 driver").
- **Major mismatch is different.** A torch whose CUDA major is newer than the driver supports (torch 13.x on a CUDA 12 driver)
  is a hard `fail` (`torch_cuda_newer_major`), and so is a driver below the family floor.
- **What actually causes `cudaErrorDevicesUnavailable`.** Compute mode Exclusive_Process or Prohibited
  (`nvidia-smi -c DEFAULT` needs root or a privileged container); another process holding the GPU
  (`nvidia-smi --query-compute-apps=pid,process_name --format=csv`); a container that was not given the GPUs it lists
  (`ls /dev/nvidia*`, `NVIDIA_VISIBLE_DEVICES`); a driver or fabric fault, where the fix is to recreate the instance or try
  another host. When a device fails with "busy or unavailable", the `devices_unusable` fix lists these, and adds the
  minor-version difference only as something to rule out by trying a matching torch/host.
- **Volta and older.** CUDA 13 dropped offline compilation for Maxwell/Pascal/Volta (sm_50..sm_72): a CUDA 13 torch cannot run
  on a V100 or older (`cuda13_dropped_arch`, fail). Use a CUDA 12.x build (cu126) for those GPUs.
- **Blackwell.** sm_100/sm_101/sm_103/sm_120/sm_121 need a torch built with CUDA >= 12.8 (12.9 for sm_103 and sm_121), and
  Triton >= 3.3 (`triton_too_old_for_blackwell`, warn). A Blackwell GPU on a driver supporting less than CUDA 12.8 gets no wheel
  recommendation: the driver must be upgraded.

## Wheel index recommendation

`recommend()` picks the newest PyTorch wheel line the driver's CUDA version can run: driver CUDA >= 13.0 `cu130`, >= 12.9
`cu129`, >= 12.8 `cu128`, otherwise `cu126`.

- It deliberately caps at `cu130`. The PyTorch index also has `cu132` and `cu134`, but `cu130` runs on every 13.x driver and
  is published for every torch release that has CUDA 13.
- GPUs older than Turing (sm < 75) get `cu126`: CUDA 13 dropped them and `cu128` wheels of torch >= 2.11 dropped Volta too.
- Blackwell with a driver CUDA < 12.8 gives no recommendation.
- Companion requirements: `nvidia-ml-py>=12.535` (or `>=13` for a CUDA 13 driver), and `triton>=3.3` with Blackwell GPUs,
  otherwise `triton>=2.2`.
- `reinstall_torch` is true only for a pip-installed torch that is CPU-only or has a newer CUDA major than the driver. A
  container torch is never replaced: the report says to fix the host or driver instead.

## CUDA_VISIBLE_DEVICES

The rules look only at the GPUs `CUDA_VISIBLE_DEVICES` exposes, when it is a list of integer indices. UUID and MIG forms, and
ids that do not exist, leave the list untouched so a rule never fires on a guess. Empty or `-1` means the GPUs are hidden on
purpose: `gpus_hidden` (info) and no failure for "torch sees no GPU". When only some devices pass the allocation test,
`some_devices_usable` names the indices to use (relative to the current `CUDA_VISIBLE_DEVICES`).

## Where it runs

- `scripts/vast_setup.sh`: prints the report, picks the torch wheel index from the driver when installing torch
  (`FORGE_TORCH_INDEX` overrides), and with `FORGE_INSTALL_TORCH=1` replaces a pip-installed torch that cannot see CUDA
  (a CPU-only PyPI build) using `--upgrade --force-reinstall`, since a plain install would be a no-op. It never replaces a torch
  that already sees CUDA or an NVIDIA container build (exit 2). It writes `~/.forge_env` before the strict CUDA check, so a
  failed check does not leave the host half-configured; it then runs the full check, installs the recommended `nvidia-ml-py` /
  `triton` without touching torch, and exits 3 on a failure.
- `scripts/gpu_smoke.sh`: runs the check before the torch step and stops with the report on failure.
- `scripts/run_gpu_tests.sh`: `gpu_report.py cuda-health` uses the check; a failure stops the run with exit 3. The findings and
  stack are written to `gpu_report/cuda_health.json` and a "CUDA stack" section of `report.md`.
- Job workers call `forge.hw.cuda_compat.assert_cuda_usable()`, which raises `CudaUnusable` with the diagnosis. Success is
  cached per process; failures are re-checked on every call.
- Dashboard probe: the dashboard's CUDA status (`/system/cuda`) loads static by default (`probe=0`: driver, torch build, arch,
  Triton, compute mode; no allocation test). The allocation test runs only on demand ("Re-check (test GPUs)"), in a
  subprocess, so the server process never gets a CUDA context.

## Overrides

- `FORGE_SKIP_CUDA_CHECK=1`: continue in `vast_setup.sh`, `gpu_smoke.sh` and `run_gpu_tests.sh` even when the check fails.
  (`FORGE_ALLOW_UNUSABLE_CUDA=1` still works for `run_gpu_tests.sh`.)
- `FORGE_TORCH_INDEX=<url>`: force the torch wheel index in `vast_setup.sh`.
- `FORGE_INSTALL_TORCH=1`: let `vast_setup.sh` install (or replace a CPU-only) torch.

## Commands

```
forge cuda-check                               # same report through the CLI
python -m forge.hw.cuda_compat                 # report, exit 1 on a fail finding
python -m forge.hw.cuda_compat --json          # full machine-readable report
python -m forge.hw.cuda_compat --no-probe      # skip the allocation test (does not import torch)
python -m forge.hw.cuda_compat --strict        # exit 1 on warnings too
python -m forge.hw.cuda_compat --torch-index   # e.g. cu130, empty if unknown
python -m forge.hw.cuda_compat --requirements  # recommended companion requirements
```
