"""Triton kernels used ONLY by :mod:`forge.hw.microbench` (imported lazily; requires ``triton`` + a CUDA GPU).

These kernels are micro-benchmark probes, not production kernels.  Status: implemented, NOT validated on
hardware in this repository's development environment (no NVIDIA GPU was available).  Minimum: triton >= 2.1.

Honesty note on "shared memory bandwidth": Triton cannot address shared memory explicitly, so
``onchip_reread_kernel`` measures the throughput of repeatedly re-reading a small per-program working set that
fits in the SM's L1/shared-memory SRAM.  The probe reports it as an *on-chip SRAM re-read proxy*
(``smem_bw_proxy``), not as a certified shared-memory bank-conflict-free peak.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def onchip_reread_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, NT: tl.constexpr, REPS: tl.constexpr):
    """Each program owns NT tiles of BLOCK fp32 (NT*BLOCK*4 bytes, 16 KiB for 1024x4) and re-reads them REPS
    times round-robin (address depends on the loop index so the loads cannot be hoisted)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = x_ptr + pid * (BLOCK * NT)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, REPS):
        v = tl.load(base + (i % NT) * BLOCK + offs)
        acc += v
    tl.store(out_ptr + pid * BLOCK + offs, acc)


@triton.jit
def dequant_alu_kernel(w_ptr, out_ptr, scale, BLOCK: tl.constexpr, REPS: tl.constexpr):
    """ALU-bound int4 -> fp32 dequantisation: each int32 word holds eight 4-bit values; per repetition the word is
    perturbed (xor with the loop index) so the work cannot be hoisted.  Per element: shift, and, sub, convert,
    multiply-add (five simple ops)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    w = tl.load(w_ptr + pid * BLOCK + offs)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for r in range(0, REPS):
        ww = w ^ r
        for j in tl.static_range(8):
            q = (ww >> (4 * j)) & 0xF
            acc += (q - 8).to(tl.float32) * scale
    tl.store(out_ptr + pid * BLOCK + offs, acc)
