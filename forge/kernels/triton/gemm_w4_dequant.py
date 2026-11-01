"""Large-M W4A16 path: memory-bound Triton dequantisation to bf16/fp16 + one cuBLAS GEMM per row chunk (``dequant_matmul``).

Why (docs/KERNELS.md 7b): for M >= 32 the fused tensor-core kernels are limited by the stride-8 activation gather and the
in-loop dequantisation, not by DRAM; measured on sm_120 they are 3-7x slower than a bf16 cuBLAS GEMM at M >= 64.  At large M the
GEMM is compute-bound, so it is cheaper to pay the dequantisation once per call at memory speed and let cuBLAS run the GEMM
on the dense weight: ``time ~ (W4 read + bf16 write) + cuBLAS(M, N, K)``, i.e. roughly constant in M instead of linear.

Kernel ``_w4_dequant_kernel``: each program owns a ``[BLOCK_N, BLOCK_K]`` tile of the *output* weight.  It loads the
``[BLOCK_N, KW]`` packed words (``KW = BLOCK_K // 8``, contiguous along the word axis -> 16-byte vector loads, every word read
once), expands the 8 nibbles of each word along a new innermost axis ``[BLOCK_N, KW, 8]`` and stores them to
``out[n, k0 + 8*c + j]`` -- the flattened ``(c, j)`` axis is the contiguous K axis, so the stores are fully coalesced
(8 x 16-bit = 16 bytes per word).  ``(q - zero) * scale`` is evaluated in fp32 and rounded once to the output dtype, which is
what the legacy kernel did per element too (so the numerics are those of a "bf16 dequantised weight" baseline: relative
weight error 2^-9, fp32 accumulation inside cuBLAS).

Scale / zero handling (``SCALE_MODE``): 0 = the K tile lies inside one quantisation group (``group % BLOCK_K == 0``, the g128
default): one ``[BLOCK_N]`` vector per tile; 1 = groups are multiples of 8: one ``[BLOCK_N, KW]`` gather per word;
2 = per-element group index (group not a multiple of 8, or act-order ``g_idx`` given: columns of the *original* order are
dequantised straight from ``g_idx``, so no column sort, no activation permutation and irregular groups work).
Symmetric weights (no ``zeros`` tensor) use the zero-point 8.  K that is not a multiple of 8 is handled by the masks (the pad
nibbles are never stored).

Memory: the dequantised weight is a TEMPORARY of ``rows * K * itemsize`` bytes.  It is produced in row chunks bounded by
``FORGE_W4_DEQUANT_MB`` / ``config['chunk_mb']`` (default 1 GiB, see :func:`forge.kernels.w4_tuning.plan_dequant_chunks`),
so even the 151936 x 5120 lm_head (1.45 GiB in bf16) is never materialised at once under the default budget.  One buffer is
allocated per call and reused by every chunk (same stream -> ordered); it comes from the torch caching allocator, which
is also what CUDA graph capture uses (graph-private pool), so the path is capturable; no host synchronisation anywhere.

Status: logic verified in the Triton CPU interpreter (tests/test_kernels_w4_dequant_interp.py); the GPU run is
tests_gpu/test_kernels_w4_dequant_gpu.py and scripts/bench_w4a16.py on the GPU host.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from forge.kernels.w4_tuning import dequant_budget_bytes, normalise_w4_config, plan_dequant_chunks

from .gemm_common import HAS_TRITON, cdiv, require_triton, tl, triton

log = logging.getLogger(__name__)
# per (device, K, itemsize): the largest chunk working set (bytes) that did not OOM-out; shrunk on every OOM so the next call does not retry 1 GiB
_BUDGET_CAP: dict[tuple, int] = {}
_WARNED: set[tuple] = set()

if HAS_TRITON:

    @triton.jit
    def _w4_dequant_kernel(
        qw_ptr, s_ptr, z_ptr, gidx_ptr, out_ptr,
        row0, rows, K,
        stride_qn, stride_sg, stride_zg, stride_on,
        GROUP: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_GIDX: tl.constexpr, SCALE_MODE: tl.constexpr,
        EVEN_R: tl.constexpr, EVEN_K: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, KW: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_r = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # row inside the chunk
        offs_n = row0 + offs_r  # output channel
        offs_c = tl.arange(0, KW)
        offs_j = tl.arange(0, 8)
        r_mask = offs_r < rows
        k0 = pid_k * BLOCK_K
        wcol = pid_k * KW + offs_c
        nwords = tl.cdiv(K, 8)
        offs_n64 = offs_n.to(tl.int64)

        wptr = qw_ptr + offs_n64[:, None] * stride_qn + wcol[None, :]
        if EVEN_R and EVEN_K:  # no predicate on the contiguous axis: lets the compiler prove the 16-byte vector loads
            word = tl.load(wptr)
        else:
            w_mask = r_mask[:, None] & (wcol < nwords)[None, :]
            word = tl.load(wptr, mask=w_mask, other=0)
        q = ((word[:, :, None] >> (offs_j * 4)[None, None, :]) & 0xF).to(tl.float32)  # [BN, KW, 8]

        kk = k0 + offs_c[:, None] * 8 + offs_j[None, :]  # [KW, 8]: the contiguous K axis of the output tile
        if SCALE_MODE == 0:  # whole tile inside one group
            gi = k0 // GROUP
            if EVEN_R:
                s = tl.load(s_ptr + gi * stride_sg + offs_n).to(tl.float32)
            else:
                s = tl.load(s_ptr + gi * stride_sg + offs_n, mask=r_mask, other=0.0).to(tl.float32)
            s3 = s[:, None, None]
            if HAS_ZEROS:
                if EVEN_R:
                    zw = tl.load(z_ptr + gi * stride_zg + offs_n // 8)
                else:
                    zw = tl.load(z_ptr + gi * stride_zg + offs_n // 8, mask=r_mask, other=0)
                z3 = ((zw >> ((offs_n % 8) * 4)) & 0xF).to(tl.float32)[:, None, None]
        elif SCALE_MODE == 1:  # one group per word
            gc = (k0 + offs_c * 8) // GROUP
            gmask = r_mask[:, None] & (wcol < nwords)[None, :]
            s = tl.load(s_ptr + gc[None, :] * stride_sg + offs_n[:, None], mask=gmask, other=0.0).to(tl.float32)
            s3 = s[:, :, None]
            if HAS_ZEROS:
                zw = tl.load(z_ptr + gc[None, :] * stride_zg + (offs_n // 8)[:, None], mask=gmask, other=0)
                z3 = ((zw >> ((offs_n % 8) * 4)[:, None]) & 0xF).to(tl.float32)[:, :, None]
        else:  # per-element group index (act-order g_idx, or a group size that is not a multiple of 8)
            k_ok = kk < K
            if HAS_GIDX:
                gk = tl.load(gidx_ptr + kk, mask=k_ok, other=0)
            else:
                gk = kk // GROUP
            emask = r_mask[:, None, None] & k_ok[None, :, :]
            s3 = tl.load(s_ptr + gk[None, :, :] * stride_sg + offs_n[:, None, None], mask=emask, other=0.0).to(tl.float32)
            if HAS_ZEROS:
                zw = tl.load(z_ptr + gk[None, :, :] * stride_zg + (offs_n // 8)[:, None, None], mask=emask, other=0)
                z3 = ((zw >> ((offs_n % 8) * 4)[:, None, None]) & 0xF).to(tl.float32)

        if HAS_ZEROS:
            v = (q - z3) * s3
        else:
            v = (q - 8.0) * s3

        optr = out_ptr + offs_r[:, None, None].to(tl.int64) * stride_on + kk[None, :, :]
        if EVEN_R and EVEN_K:
            tl.store(optr, v.to(out_ptr.dtype.element_ty))
        else:
            tl.store(optr, v.to(out_ptr.dtype.element_ty), mask=r_mask[:, None, None] & (kk < K)[None, :, :])


def _dequant_params(config: dict[str, Any] | None) -> tuple[int, int, int]:
    """``(BLOCK_N, BLOCK_K, num_warps)`` of a ``dequant_matmul`` config (clamped to what the kernel supports)."""
    c = normalise_w4_config({**(config or {}), "impl": "dequant_matmul"}, 1)
    return int(c["tile_n"]), int(c["tile_k"]), int(c["warps"])


def scale_mode(group_size: int, k: int, n_groups: int, block_k: int, has_gidx: bool) -> int:
    """Scale/zero access pattern of the dequant kernel (see module docstring)."""
    gsz = k if group_size <= 0 else int(group_size)
    if has_gidx:
        return 2
    if n_groups == 1 or gsz % block_k == 0:
        return 0
    return 1 if gsz % 8 == 0 else 2


def w4_dequant(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    *,
    n: int,
    k: int,
    group_size: int,
    dtype: torch.dtype,
    row0: int = 0,
    rows: int | None = None,
    out: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Dequantise output channels ``row0 .. row0 + rows`` of a ``portable_row`` weight to a dense ``[rows, k]`` tensor of
    ``dtype`` (natural column order; ``g_idx`` is honoured directly, a group-sorted weight is dequantised in its sorted order).
    ``out`` (optional) is a caller-owned buffer with at least ``rows`` rows of ``k`` contiguous elements; the first ``rows``
    rows are filled and returned."""
    require_triton()
    rows = (n - row0) if rows is None else int(rows)
    if rows < 0 or row0 < 0 or row0 + rows > n:
        raise ValueError(f"row range [{row0}, {row0 + rows}) outside 0..{n}")
    if row0 % 8:
        raise ValueError("row0 must be a multiple of 8 (packed zero-point words hold 8 channels)")
    if out is None:
        out = torch.empty((rows, k), dtype=dtype, device=qweight.device)
    else:
        if out.dtype != dtype or out.shape[1] != k or out.shape[0] < rows or not out.is_contiguous():
            raise ValueError("out must be a contiguous [>=rows, k] tensor of the requested dtype")
        out = out[:rows]
    if rows == 0 or k == 0:
        return out
    bn, bk, warps = _dequant_params(config)
    has_zeros = zeros is not None
    mode = scale_mode(group_size, k, int(scales.shape[0]), bk, g_idx is not None)
    dummy = qweight
    gsz = k if group_size <= 0 else int(group_size)
    grid = (cdiv(rows, bn), cdiv(k, bk))
    _w4_dequant_kernel[grid](
        qweight, scales, zeros if has_zeros else dummy, g_idx if g_idx is not None else dummy, out,
        row0, rows, k,
        qweight.stride(0), scales.stride(0), zeros.stride(0) if has_zeros else 0, out.stride(0),
        GROUP=gsz, HAS_ZEROS=has_zeros, HAS_GIDX=g_idx is not None, SCALE_MODE=mode,
        EVEN_R=rows % bn == 0, EVEN_K=k % bk == 0,
        BLOCK_N=bn, BLOCK_K=bk, KW=bk // 8,
        num_warps=warps, num_stages=1,
    )
    return out


def w4_dequant_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    bias: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    *,
    n: int,
    k: int,
    group_size: int,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """``Y = X @ dequant(W)^T + bias`` for 2-D ``x [M, K]`` (fp16/bf16): chunked Triton dequantisation + cuBLAS ``F.linear``.
    Returns ``[M, N]`` in ``x.dtype``.  Peak temporary memory = ``min(budget, N * K * itemsize)`` (+ ``M * rows`` for the output)."""
    require_triton()
    if x.dim() != 2 or x.shape[1] != k:
        raise ValueError(f"x must be [M, {k}], got {tuple(x.shape)}")
    m = x.shape[0]
    if m == 0:
        return torch.empty((0, n), dtype=x.dtype, device=x.device)
    dt = x.dtype
    b = None if bias is None else bias.to(dt)
    item = x.element_size()
    cap_key = (str(x.device), k, item)
    floor = 64 * (k + m) * item  # one minimum chunk (64 rows): below this a retry cannot help
    while True:
        budget = dequant_budget_bytes(config)
        cap = _BUDGET_CAP.get(cap_key)
        if cap is not None:
            budget = min(budget, cap)
        try:
            return _dequant_matmul_chunks(x, qweight, scales, zeros, b, g_idx, n, k, group_size, config, budget)
        except torch.cuda.OutOfMemoryError as e:
            if budget <= floor:
                raise
            new = max(floor, budget // 2)
            _BUDGET_CAP[cap_key] = new
            if (cap_key, new) not in _WARNED and len(_WARNED) < 64:
                _WARNED.add((cap_key, new))
                log.warning("W4A16 dequant_matmul ran out of memory with a %d MiB chunk budget (%s); retrying with %d MiB and "
                            "using that cap for later calls (set FORGE_W4_DEQUANT_MB to silence)", budget >> 20, str(e).splitlines()[0][:120], new >> 20)
            torch.cuda.empty_cache()


def _dequant_matmul_chunks(x, qweight, scales, zeros, b, g_idx, n, k, group_size, config, budget) -> torch.Tensor:
    """One attempt of :func:`w4_dequant_matmul` under a chunk working-set ``budget`` (bytes): the dequantised ``[rows, k]`` buffer
    + the chunk's ``[m, rows]`` output.  Every chunk's GEMM writes straight into its slice of the preallocated ``y``."""
    m, dt = x.shape[0], x.dtype
    chunks = plan_dequant_chunks(n, k, x.element_size(), budget, m=m)
    kw = dict(n=n, k=k, group_size=group_size, dtype=dt, config=config)
    if len(chunks) == 1:
        w = w4_dequant(qweight, scales, zeros, g_idx, **kw)
        return F.linear(x, w, b)
    y = torch.empty((m, n), dtype=dt, device=x.device)
    buf = torch.empty((max(r1 - r0 for r0, r1 in chunks), k), dtype=dt, device=x.device)  # one buffer for all chunks
    for r0, r1 in chunks:
        w = w4_dequant(qweight, scales, zeros, g_idx, row0=r0, rows=r1 - r0, out=buf, **kw)
        if b is None:
            torch.mm(x, w.t(), out=y[:, r0:r1])
        else:
            torch.addmm(b[r0:r1], x, w.t(), out=y[:, r0:r1])
    return y


__all__ = ["w4_dequant", "w4_dequant_matmul", "scale_mode"]
