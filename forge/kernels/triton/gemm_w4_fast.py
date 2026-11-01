"""Nibble-split W4A16 Triton kernels (``dot`` = tensor cores, ``gemv`` = fp32 FMA) for the ``portable_row`` layout.

Why this exists (diagnosis of the legacy kernel in :mod:`forge.kernels.triton.gemm_wq`, see docs/KERNELS.md):

* the legacy kernel loads, for every ``(k, n)`` element of the tile, the int32 word that contains it: 8 redundant
  4-byte loads per word with a 2.5 KiB stride between channels, which Triton cannot vectorise (the address pattern
  ``offs_k // 8`` has constancy 8, not contiguity) -> scalar, uncoalesced weight traffic;
* it also loads a scale and a zero-point word *per element* inside the K loop;
* it multiplies by the scale before the MMA for every element.

Here the packed words are loaded as a ``[KW, BLOCK_N]`` (dot) / ``[BLOCK_N, KW]`` (gemv) tile, ``KW = BLOCK_K // 8``,
**contiguous along the word axis** (16-byte vector loads, each word read once).  Nibble ``j`` of word ``c`` is the
weight for ``k = k0 + 8*c + j``, so the nibble-``j`` slice ``(word >> 4j) & 15`` pairs with the activations
``x[:, k0 + 8*c + j]`` (a stride-8 gather of the tiny activation tile, L1-resident); the K sum is simply the sum over
the 8 nibble slices of 8 independent products, which needs no register shuffling to interleave nibbles.

* scale / zero loads are per ``[BLOCK_N]`` vector per K tile when a tile lies in one quantisation group
  (``SINGLE_GROUP``: g128 with ``tile_k <= 128``); the scale is then applied once to the ``[BM, BN]`` partial sum
  instead of every weight (``dot``): the MMA multiplies exact small integers ``q - z`` (|.| <= 15, exact in bf16/fp16).
  Otherwise one ``[.., KW]`` scale tile is gathered per K tile (hoisted out of the 8-slice loop).
* act-order (GPTQ ``g_idx``) weights are handled by sorting the K columns by group once at load time
  (:func:`forge.kernels.layouts.w4.sort_columns_by_group`) and gathering ``x`` through ``perm`` inside the kernel.
* split-K, bias, deterministic/atomic reduction are identical to the legacy kernel (same reduce kernel).

Numerics: fp32 accumulation, dequantisation exact in fp32 (``gemv``) or integer-exact operands + fp32 scale (``dot``,
single group) -- equal or closer to the fp32 oracle than the legacy kernel, which rounded ``(q - z) * s`` to 16 bit.

Status: logic verified in the Triton CPU interpreter (tests/test_kernels_w4_fast_interp.py); NOT compiled or benchmarked
on a GPU.  Use ``scripts/bench_w4a16.py`` on the GPU host.
"""
from __future__ import annotations

from typing import Any

import torch

from forge.kernels.w4_tuning import GEMV_MAX_M, MIN_TILE_K, resolve_w4_config

from .gemm_common import HAS_TRITON, cdiv, launch_fitting, require_triton, tl, triton

if HAS_TRITON:
    from .gemm_wq import _splitk_reduce_kernel

    @triton.jit
    def _w4_dot_kernel(
        x_ptr, qw_ptr, s_ptr, z_ptr, perm_ptr, bias_ptr, y_ptr, ws_ptr,
        M, N, K,
        stride_xm, stride_xk, stride_qn, stride_sg, stride_zg, stride_ym, stride_yn,
        GROUP: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_PERM: tl.constexpr, HAS_BIAS: tl.constexpr,
        SINGLE_GROUP: tl.constexpr, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, SPLIT_K: tl.constexpr,
        DETERMINISTIC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, KW: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, KW)
        m_mask = offs_m < M
        n_mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        nwords = tl.cdiv(K, 8)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k_tiles = tl.cdiv(K, BLOCK_K)
        tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
        k_tile_start = pid_k * tiles_per_split
        k_tile_end = tl.minimum(k_tile_start + tiles_per_split, k_tiles)

        for kt in range(k_tile_start, k_tile_end):
            k0 = kt * BLOCK_K
            wcol = k0 // 8 + offs_c
            if EVEN_K:  # no K-axis predicate: lets the compiler prove the 16-byte contiguity of the word loads
                w_mask = n_mask[None, :]
            else:
                w_mask = (wcol < nwords)[:, None] & n_mask[None, :]
            # [KW, BN] packed words; contiguous along dim 0 (the K axis of the row-major [N, K/8] matrix)
            wptr = qw_ptr + offs_n64[None, :] * stride_qn + wcol[:, None]
            if EVEN_K and EVEN_N:
                word = tl.load(wptr)
            else:
                word = tl.load(wptr, mask=w_mask, other=0)

            if SINGLE_GROUP:
                gi = k0 // GROUP
                if EVEN_N:
                    s = tl.load(s_ptr + gi * stride_sg + offs_n).to(tl.float32)
                else:
                    s = tl.load(s_ptr + gi * stride_sg + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                if HAS_ZEROS:
                    if EVEN_N:
                        zw = tl.load(z_ptr + gi * stride_zg + offs_n // 8)
                    else:
                        zw = tl.load(z_ptr + gi * stride_zg + offs_n // 8, mask=n_mask, other=0)
                    z = ((zw >> ((offs_n % 8) * 4)) & 0xF).to(tl.float32)
                part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            else:
                gc = (k0 + offs_c * 8) // GROUP
                s = tl.load(s_ptr + gc[:, None] * stride_sg + offs_n[None, :], mask=w_mask, other=0.0).to(tl.float32)
                if HAS_ZEROS:
                    zw = tl.load(z_ptr + gc[:, None] * stride_zg + (offs_n // 8)[None, :], mask=w_mask, other=0)
                    z = ((zw >> ((offs_n % 8) * 4)[None, :]) & 0xF).to(tl.float32)

            for j in tl.static_range(8):
                kk = k0 + offs_c * 8 + j
                if EVEN_K:
                    if HAS_PERM:
                        src = tl.load(perm_ptr + kk)
                    else:
                        src = kk
                    x_mask = m_mask[:, None]
                else:
                    k_mask = kk < K
                    if HAS_PERM:
                        src = tl.load(perm_ptr + kk, mask=k_mask, other=0)
                    else:
                        src = kk
                    x_mask = m_mask[:, None] & k_mask[None, :]
                x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + src[None, :] * stride_xk, mask=x_mask, other=0.0)
                q = ((word >> (4 * j)) & 0xF).to(tl.float32)
                if SINGLE_GROUP:
                    if HAS_ZEROS:
                        v = q - z[None, :]
                    else:
                        v = q - 8.0
                    part = tl.dot(x, v.to(x.dtype), part)
                else:
                    if HAS_ZEROS:
                        v = (q - z) * s
                    else:
                        v = (q - 8.0) * s
                    acc = tl.dot(x, v.to(x.dtype), acc)
            if SINGLE_GROUP:
                acc += part * s[None, :]

        out_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT_K == 1:
            if HAS_BIAS:
                b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                acc = acc + b[None, :]
            tl.store(y_ptr + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :] * stride_yn, acc, mask=out_mask)
        else:
            if DETERMINISTIC:
                ws = ws_ptr + (pid_k * M + offs_m[:, None].to(tl.int64)) * N + offs_n[None, :]
                tl.store(ws, acc, mask=out_mask)
            else:
                ws = ws_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
                tl.atomic_add(ws, acc, mask=out_mask)

    @triton.jit
    def _w4_gemv_kernel(
        x_ptr, qw_ptr, s_ptr, z_ptr, perm_ptr, bias_ptr, y_ptr, ws_ptr,
        M, N, K,
        stride_xm, stride_xk, stride_qn, stride_sg, stride_zg, stride_ym, stride_yn,
        GROUP: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_PERM: tl.constexpr, HAS_BIAS: tl.constexpr,
        SINGLE_GROUP: tl.constexpr, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, SPLIT_K: tl.constexpr,
        DETERMINISTIC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, KW: tl.constexpr,
    ):
        """Decode GEMV (M <= 8): all M rows in one program, fp32 FMA, the K reduction is deferred to the very end
        (``acc3 [BM, BN, KW]`` collects element-wise products, one ``tl.sum`` after the loop)."""
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, KW)
        m_mask = offs_m < M
        n_mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        nwords = tl.cdiv(K, 8)

        acc3 = tl.zeros((BLOCK_M, BLOCK_N, KW), dtype=tl.float32)

        k_tiles = tl.cdiv(K, BLOCK_K)
        tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
        k_tile_start = pid_k * tiles_per_split
        k_tile_end = tl.minimum(k_tile_start + tiles_per_split, k_tiles)

        for kt in range(k_tile_start, k_tile_end):
            k0 = kt * BLOCK_K
            wcol = k0 // 8 + offs_c
            if EVEN_K:  # no K-axis predicate: lets the compiler prove the 16-byte contiguity of the word loads
                w_mask = n_mask[:, None]
            else:
                w_mask = n_mask[:, None] & (wcol < nwords)[None, :]
            # [BN, KW] packed words, contiguous along dim 1 (16-byte vector loads)
            wptr = qw_ptr + offs_n64[:, None] * stride_qn + wcol[None, :]
            if EVEN_K and EVEN_N:
                word = tl.load(wptr)
            else:
                word = tl.load(wptr, mask=w_mask, other=0)

            if SINGLE_GROUP:
                gi = k0 // GROUP
                if EVEN_N:
                    s = tl.load(s_ptr + gi * stride_sg + offs_n).to(tl.float32)[:, None]
                else:
                    s = tl.load(s_ptr + gi * stride_sg + offs_n, mask=n_mask, other=0.0).to(tl.float32)[:, None]
                if HAS_ZEROS:
                    if EVEN_N:
                        zw = tl.load(z_ptr + gi * stride_zg + offs_n // 8)
                    else:
                        zw = tl.load(z_ptr + gi * stride_zg + offs_n // 8, mask=n_mask, other=0)
                    z = ((zw >> ((offs_n % 8) * 4)) & 0xF).to(tl.float32)[:, None]
            else:
                gc = (k0 + offs_c * 8) // GROUP
                s = tl.load(s_ptr + gc[None, :] * stride_sg + offs_n[:, None], mask=w_mask, other=0.0).to(tl.float32)
                if HAS_ZEROS:
                    zw = tl.load(z_ptr + gc[None, :] * stride_zg + (offs_n // 8)[:, None], mask=w_mask, other=0)
                    z = ((zw >> ((offs_n % 8) * 4)[:, None]) & 0xF).to(tl.float32)

            for j in tl.static_range(8):
                kk = k0 + offs_c * 8 + j
                if EVEN_K:
                    if HAS_PERM:
                        src = tl.load(perm_ptr + kk)
                    else:
                        src = kk
                    x_mask = m_mask[:, None]
                else:
                    k_mask = kk < K
                    if HAS_PERM:
                        src = tl.load(perm_ptr + kk, mask=k_mask, other=0)
                    else:
                        src = kk
                    x_mask = m_mask[:, None] & k_mask[None, :]
                x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + src[None, :] * stride_xk, mask=x_mask, other=0.0).to(tl.float32)
                q = ((word >> (4 * j)) & 0xF).to(tl.float32)
                if HAS_ZEROS:
                    w = (q - z) * s
                else:
                    w = (q - 8.0) * s
                acc3 += x[:, None, :] * w[None, :, :]

        acc = tl.sum(acc3, axis=2)  # [BM, BN]
        out_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT_K == 1:
            if HAS_BIAS:
                b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                acc = acc + b[None, :]
            tl.store(y_ptr + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :] * stride_yn, acc, mask=out_mask)
        else:
            if DETERMINISTIC:
                ws = ws_ptr + (pid_k * M + offs_m[:, None].to(tl.int64)) * N + offs_n[None, :]
                tl.store(ws, acc, mask=out_mask)
            else:
                ws = ws_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
                tl.atomic_add(ws, acc, mask=out_mask)


_RESOLVED: dict[tuple, tuple[dict[str, Any], tuple]] = {}  # per-call Python overhead matters at decode (~10 us budget)


def _resolve(m: int, n: int, k: int, config: dict[str, Any] | None, sm: int | None, num_sms: int, gsz: int) -> tuple[dict[str, Any], tuple]:
    ck = tuple(sorted(config.items())) if config else None
    key = (m, n, k, sm, num_sms, gsz, ck)
    hit = _RESOLVED.get(key)
    if hit is None:
        if len(_RESOLVED) > 4096:
            _RESOLVED.clear()
        c = resolve_w4_config(m, n, k, config, sm=sm, num_sms=num_sms, group=gsz)
        _RESOLVED[key] = hit = (c, tuple(sorted(c.items())))
    return hit


def w4_fast_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    bias: torch.Tensor | None,
    perm: torch.Tensor | None,
    *,
    n: int,
    k: int,
    group_size: int,
    config: dict[str, Any] | None = None,
    deterministic: bool = True,
    sm: int | None = None,
    num_sms: int = 0,
) -> torch.Tensor:
    """``Y = X @ dequant(W)^T + bias`` on 2-D ``x [M, K]`` with the nibble-split kernels.  ``perm`` (int32 ``[K]``) is
    the column permutation of a group-sorted act-order weight (``x`` is gathered through it) or ``None``.
    Returns ``[M, N]`` in ``x.dtype``."""
    require_triton()
    if x.dim() != 2 or x.shape[1] != k:
        raise ValueError(f"x must be [M, {k}], got {tuple(x.shape)}")
    m = x.shape[0]
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return y
    gsz = k if group_size <= 0 else int(group_size)
    cfg, cfg_items = _resolve(m, n, k, config, sm, num_sms, gsz)
    if cfg["impl"] == "legacy":
        raise ValueError("impl='legacy' is launched by gemm_wq.wq_gemm, not by w4_fast_gemm")
    n_groups = int(scales.shape[0])
    has_zeros = zeros is not None
    dummy = qweight

    def _launch(c: dict[str, Any]) -> torch.Tensor:
        impl = c["impl"]
        bm, bn, bk = int(c["tile_m"]), int(c["tile_n"]), int(c["tile_k"])
        split = min(max(1, int(c.get("split_k", 1))), max(1, cdiv(k, bk)))
        single = n_groups == 1 or gsz % bk == 0
        ws = None
        if split > 1:
            ws = torch.empty((split, m, n), dtype=torch.float32, device=x.device) if deterministic else torch.zeros(
                (m, n), dtype=torch.float32, device=x.device)
        if impl == "gemv":
            kernel, grid = _w4_gemv_kernel, (cdiv(n, bn), split)
        else:
            kernel, grid = _w4_dot_kernel, (cdiv(m, bm) * cdiv(n, bn), split)
        kernel[grid](
            x, qweight, scales, zeros if has_zeros else dummy, perm if perm is not None else dummy,
            bias if bias is not None else dummy, y, ws if ws is not None else dummy,
            m, n, k,
            x.stride(0), x.stride(1), qweight.stride(0), scales.stride(0), zeros.stride(0) if has_zeros else 0,
            y.stride(0), y.stride(1),
            GROUP=gsz, HAS_ZEROS=has_zeros, HAS_PERM=perm is not None, HAS_BIAS=bias is not None,
            SINGLE_GROUP=single, EVEN_K=k % bk == 0, EVEN_N=n % bn == 0, SPLIT_K=split, DETERMINISTIC=deterministic,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, KW=bk // 8,
            num_warps=int(c.get("warps", 4)), num_stages=int(c.get("stages", 3)),
        )
        if split > 1:
            block = 1024
            _splitk_reduce_kernel[(cdiv(m * n, block),)](
                ws, bias if bias is not None else dummy, y, n, m * n, y.stride(0), y.stride(1),
                SPLITS=split if deterministic else 1, HAS_BIAS=bias is not None, BLOCK=block, num_warps=4,
            )
        return y

    key = ("w4fast", has_zeros, perm is not None, n_groups == 1, x.dtype, sm, gsz, cfg_items)
    return launch_fitting(key, cfg, _launch, min_tile_k=MIN_TILE_K[cfg["impl"]])


__all__ = ["w4_fast_gemm", "GEMV_MAX_M"]
