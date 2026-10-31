# Runbook: adding a chip descriptor

A chip descriptor (Plan Appendix B, `forge.core.schemas.ChipDescriptor`) tells kernels, the layout deriver, the cost
model and the capacity planner what hardware they target. There are two sources: the **datasheet table**
(`forge/hw/chips.py::KNOWN_CHIPS`, provenance `vendor`) and **measured descriptors** produced by the probe
(provenance `measured`). Measured values override datasheet values.

## A. A GPU that is not yet in `KNOWN_CHIPS`

1. Run the probe on the real device: `python -m forge.cli probe` (or *Hardware -> Probe*). It reads `torch.cuda`
   properties and `nvidia-smi`, runs the micro-benchmarks (HBM stream-copy bandwidth, MMA throughput per dtype,
   shared-memory bandwidth, launch latency, dequant ALU throughput) and stores a descriptor under `FORGE_HOME`.
2. Inspect it on the *Hardware* page or `GET /api/v1/hardware/descriptors`. Check `sm`, `compute_units`,
   `memory.hbm_bw_gbs` (datasheet vs measured; measured is normally somewhat below datasheet), `matrix_units` and
   their `block_scaled` entries.
3. Add the datasheet entry to `forge/hw/chips.py`: name pattern(s) matching `torch.cuda.get_device_name()`,
   `sm`, SM count, HBM size and bandwidth, L2, shared memory per SM, peak TFLOPS per dtype, 2:4 sparsity, NVLink.
   Use vendor datasheet numbers only (provenance `vendor`).
4. If the architecture is new, extend `forge/hw/capabilities.py` (`has_bf16`, `has_fp8`, `has_fp4_block_scaled`,
   `has_tma`, ...) and the support matrix in `docs/ARCHITECTURE.md` section 3 and `docs/KERNELS.md` section 4.
   **Native features stay false until the probe confirms them** (Plan HW-2).
5. Save an example under `configs/chips/<name>.yaml`. `tests/test_configs_templates.py` checks that it parses as a
   `ChipDescriptor`.
6. Tests: add a name-resolution case to the `forge.hw` CPU tests (no GPU needed), then run `tests_gpu` on the device.
7. Tier: leave `tier: "C"` until kernels pass `tests_gpu` there; then `B` (portable schedule, autotuned); `A` only once
   a tuned schedule is validated and benchmarked.
8. Update the validation table in `docs/KERNELS.md` with the chip and evidence (attach `gpu_report/report.md`).

## B. A non-NVIDIA device

Out of scope for this release (Triton kernels here target NVIDIA). A descriptor can still be written by hand for the
cost model and capacity planner; leave `tier: "C"` and mark `probe_provenance` entries `vendor` or `untested`.

## Pass criteria

`python -m pytest tests -q -k "hw or configs"` passes on CPU; a probe on the device produces a descriptor whose
measured bandwidth and TFLOPS are plausible against the datasheet; the capacity planner gives sensible numbers
for a known model on the new chip.
