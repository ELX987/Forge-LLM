"""Utility-kernel launchers / dispatchers (Plan K8, K9, §6.7): Hadamard, row/group quantisation, stochastic rounding,
RMSNorm+quant, RoPE+KV write, speculative verification helpers, top-k selection, TEAL gather-GEMV, segmented LoRA.

Every op dispatches to a Triton kernel only if CUDA + Triton + capability + shape are OK (and ``FORGE_FORCE_REF`` is not set),
otherwise to the pure-torch oracle in ``forge.kernels.ref``.  ``FORGE_KERNEL_DEBUG=1`` logs the chosen path.
**Triton paths are implemented but not yet validated on NVIDIA hardware** (see ``docs`` / ``README_ATTN.md``).
"""
from __future__ import annotations

import os
from typing import Any

import torch

from forge.kernels.ref import util_adapters as _ad
from forge.kernels.ref import util_hadamard as _had
from forge.kernels.ref import util_quant as _uq
from forge.kernels.ref import util_rope as _rope
from forge.kernels.ref.attn_kv import normalize_kv_dtype
from forge.kernels.ref.attn_sparse import topk_select_ref
from forge.kernels.ref.attn_spec import (
    greedy_verify_ref,
    kv_compact_ref,
    rejection_verify_ref,
)
from forge.kernels.ref.util_rng import derive_seed  # noqa: F401  (re-exported)
from forge.kernels.triton import util_dispatch as _d

# re-exports of pure helpers
from forge.kernels.ref.util_adapters import pad_adapters, teal_threshold  # noqa: F401
from forge.kernels.ref.util_hadamard import hadamard_matrix, random_signs  # noqa: F401
from forge.kernels.ref.util_rope import make_cos_sin_cache  # noqa: F401

KERNELS: dict[str, dict[str, Any]] = {
    "hadamard": {"family": "util", "min_sm": 70, "triton": "forge.kernels.triton.util_hadamard", "ref": "util_hadamard.hadamard_ref"},
    "quantize_rowwise": {"family": "util", "min_sm": 70, "triton": "forge.kernels.triton.util_quant", "ref": "util_quant.quantize_rowwise_ref"},
    "stochastic_round": {"family": "util", "min_sm": 70, "triton": "forge.kernels.triton.util_quant", "ref": "util_quant.stochastic_round_ref"},
    "rmsnorm_quant": {"family": "util", "min_sm": 70, "triton": "forge.kernels.triton.util_quant", "ref": "util_quant.rmsnorm_quant_ref"},
    "rope_kv_write": {"family": "util", "min_sm": 70, "triton": "forge.kernels.triton.util_rope", "ref": "util_rope.rope_and_kv_write_ref"},
    "rejection_verify": {"family": "spec", "min_sm": 70, "triton": "forge.kernels.triton.attn_spec", "ref": "attn_spec.rejection_verify_ref"},
    "kv_compact": {"family": "spec", "min_sm": 70, "triton": "forge.kernels.triton.attn_spec", "ref": "attn_spec.kv_compact_ref"},
    "topk_select": {"family": "sparse", "min_sm": 70, "triton": "forge.kernels.triton.attn_sparse", "ref": "attn_sparse.topk_select_ref"},
    "teal_gemv": {"family": "util", "min_sm": 70, "triton": "forge.kernels.triton.util_gemv", "ref": "util_adapters.teal_gemv_ref"},
    "segmented_lora": {"family": "util", "min_sm": 80, "triton": "forge.kernels.triton.moe_kernels", "ref": "util_adapters.segmented_lora_ref"},
}


def _tri(*tensors, min_sm: int = 70, op: str = "") -> bool:
    ok = _d.use_triton(*tensors, min_sm=min_sm, op=op)
    if ok:
        for t in tensors:
            if t is not None and t.dtype == torch.bfloat16 and not _d.bf16_ok(t.dtype, t.device):
                ok = False
    _d.note_path(op, "triton" if ok else "reference")
    return ok


# ------------------------------------------------------------------------------------------------------------
# Hadamard
# ------------------------------------------------------------------------------------------------------------
def hadamard_transform(x: torch.Tensor, block: int | None = None, signs: torch.Tensor | None = None,
                       normalize: bool = True) -> torch.Tensor:
    """Block-diagonal (randomised) Walsh-Hadamard transform along the last dim (block 16/32/64/128/head_dim ...).

    ``y = H_block (signs * x)`` per contiguous ``block`` elements; orthonormal when ``normalize``. Triton for fp16/bf16 and
    block in {16, 32, 64, 128}; other power-of-two blocks use the torch butterfly (also on GPU).
    """
    d = x.shape[-1]
    blk = d if block is None else block
    if blk in (16, 32, 64, 128) and x.dtype in (torch.float16, torch.bfloat16) and _tri(x, op="hadamard"):
        from forge.kernels.triton.util_hadamard import hadamard_triton

        return hadamard_triton(x, blk, signs, normalize)
    return _had.hadamard_ref(x, blk, signs, normalize)


# ------------------------------------------------------------------------------------------------------------
# quantisation
# ------------------------------------------------------------------------------------------------------------
def absmax(x: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Per-row / per-group absolute max over finite values (torch reduction - already bandwidth-optimal)."""
    return _uq.absmax_ref(x, group_size)


def quantize_rowwise(x: torch.Tensor, qdtype: str = "fp8", group_size: int | None = None):
    """Symmetric row/group quantisation -> ``(q, scale)`` (``qdtype`` ``"fp8"`` E4M3 or ``"int8"``; scale = absmax / qmax)."""
    ok_dtype = x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if qdtype == "fp8":
        triton_ok = ok_dtype and _d.fp8_native(x.device) and _tri(x, min_sm=89, op="quantize_rowwise")
    else:
        triton_ok = ok_dtype and _tri(x, op="quantize_rowwise")
    if triton_ok:
        from forge.kernels.triton.util_quant import quantize_rowwise_triton

        return quantize_rowwise_triton(x, qdtype, group_size)
    return _uq.quantize_rowwise_ref(x, qdtype, group_size)


def quantize_fp8_rowwise(x: torch.Tensor, group_size: int | None = None):
    """FP8-E4M3 row (or group) quantisation: ``(x_fp8, scale float32)``."""
    return quantize_rowwise(x, "fp8", group_size)


def dequantize_rowwise(q: torch.Tensor, scale: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    return _uq.dequantize_rowwise_ref(q, scale, group_size)


def stochastic_round(x: torch.Tensor, fmt: str = "int8", *, group_size: int | None = None,
                     scale: torch.Tensor | None = None, seed: int = 0, offset: int = 0):
    """Counter-based-RNG stochastic-rounding quantiser (``fmt`` int8 | int4 | fp4) -> ``(codes, scale)``.

    Element ``i`` (row-major) uses ``philox(seed, offset + i)`` - reproducible across launches, devices and chunking.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative (see util_rng.derive_seed)")
    if _tri(x, op="stochastic_round"):
        from forge.kernels.triton.util_quant import stochastic_round_triton

        g = x.shape[-1] if group_size is None else group_size
        if x.shape[-1] % g:
            raise ValueError(f"last dim {x.shape[-1]} not divisible by group_size {g}")
        sc = scale if scale is not None else _uq.scale_from_absmax(_uq.absmax_ref(x, group_size), fmt)
        # the kernel indexes one scale per group in row-major order: broadcast scalar / [rows, 1] scales so it never reads
        # past the end of a short scale tensor
        sc_k = torch.broadcast_to(sc.to(torch.float32), (*x.shape[:-1], x.shape[-1] // g))
        return stochastic_round_triton(x, sc_k, g, fmt, seed, offset), sc
    return _uq.stochastic_round_ref(x, fmt, group_size=group_size, scale=scale, seed=seed, offset=offset)


def rmsnorm_quant(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, qdtype: str = "fp8",
                  residual: torch.Tensor | None = None, return_norm: bool = False):
    """Fused (residual add) + RMSNorm + per-row quantise -> ``(q, scale[R], new_residual | None[, normed_fp32])``."""
    x2 = x.reshape(-1, x.shape[-1])
    r2 = residual.reshape(-1, x.shape[-1]) if residual is not None else None
    ok_dtype = x.dtype in (torch.float16, torch.bfloat16, torch.float32) and x.shape[-1] <= 16384
    if qdtype == "fp8":
        triton_ok = ok_dtype and _d.fp8_native(x.device) and _tri(x, min_sm=89, op="rmsnorm_quant")
    else:
        triton_ok = ok_dtype and _tri(x, op="rmsnorm_quant")
    if triton_ok:
        from forge.kernels.triton.util_quant import rmsnorm_quant_triton

        out = rmsnorm_quant_triton(x2, weight, eps, qdtype, r2, return_norm)
    else:
        out = _uq.rmsnorm_quant_ref(x2, weight, eps, qdtype, r2, return_norm)
    q, s, res = out[0].reshape(x.shape), out[1].reshape(x.shape[:-1]), out[2]
    res = res.reshape(x.shape) if res is not None else None
    return (q, s, res, out[3].reshape(x.shape)) if return_norm else (q, s, res)


# ------------------------------------------------------------------------------------------------------------
# RoPE + KV write
# ------------------------------------------------------------------------------------------------------------
def rope_and_kv_write(q, k, v, positions, cos_sin_cache, slot_mapping, k_pool, v_pool, *, kv_dtype: str = "bf16",
                      k_scale=None, v_scale_pool=None, rotary_dim: int | None = None, interleaved: bool = False):
    """Rotate ``q``/``k``, quantise ``k`` (static per-channel ``k_scale``) and ``v`` (dynamic per-token scale written to
    ``v_scale_pool``) and scatter both into the paged pools at ``slot_mapping``.  Returns the rotated queries."""
    kv = normalize_kv_dtype(kv_dtype)
    triton_ok = (kv != "int4" and q.dtype in (torch.float16, torch.bfloat16, torch.float32) and
                 (kv != "fp8_e4m3" or _d.fp8_native(q.device)) and
                 _tri(q, k, v, k_pool, v_pool, min_sm=89 if kv == "fp8_e4m3" else 70, op="rope_and_kv_write"))
    if triton_ok:
        from forge.kernels.triton.util_rope import rope_and_kv_write_triton

        return rope_and_kv_write_triton(q, k, v, positions, cos_sin_cache, slot_mapping, k_pool, v_pool, kv_dtype=kv,
                                        k_scale=k_scale, v_scale_pool=v_scale_pool, rotary_dim=rotary_dim,
                                        interleaved=interleaved)
    return _rope.rope_and_kv_write_ref(q, k, v, positions, cos_sin_cache, slot_mapping, k_pool, v_pool, kv_dtype=kv,
                                       k_scale=k_scale, v_scale_pool=v_scale_pool, rotary_dim=rotary_dim,
                                       interleaved=interleaved)


# ------------------------------------------------------------------------------------------------------------
# speculative decoding helpers
# ------------------------------------------------------------------------------------------------------------
def rejection_verify(draft_tokens: torch.Tensor, q_probs: torch.Tensor, p_probs: torch.Tensor, seed: int,
                     offset: int = 0):
    """Exact speculative-sampling verification of a draft chain (accept ``min(1, p/q)``, resample from ``(p-q)+/Z``,
    bonus token).  Returns ``(tokens [B, G+1] int32 (-1 padded), n_accepted [B] int32)``.  ``seed``/``offset`` select the
    counter-based random stream (reproducible; vary ``offset`` per step)."""
    if seed < 0:
        raise ValueError("seed must be non-negative (see util_rng.derive_seed)")
    if _tri(draft_tokens, q_probs, p_probs, op="rejection_verify"):
        from forge.kernels.triton.attn_spec import rejection_verify_triton

        return rejection_verify_triton(draft_tokens, q_probs, p_probs, seed, offset)
    return rejection_verify_ref(draft_tokens, q_probs, p_probs, seed, offset)


def greedy_verify(draft_tokens: torch.Tensor, target_argmax: torch.Tensor):
    """Temperature-0 verification (pure tensor ops, no kernel needed)."""
    return greedy_verify_ref(draft_tokens, target_argmax)


def kv_compact(k_pool, v_pool, block_tables, ctx_lens, accepted_idx, accept_lens) -> None:
    """In-place KV accept/rollback compaction (tree node ``accepted_idx[b, t]`` moves to position ``ctx + t``)."""
    if k_pool.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        # quantised pools carry per-token value scales (and per-page key scales) that this row copy does not move: moving the
        # codes alone would silently corrupt the cache.  Speculative tokens live in the full-precision residual window.
        raise ValueError("kv_compact supports native-float KV pools only (flush the residual window / re-quantise instead)")
    if accepted_idx.shape[1] <= 128 and _tri(k_pool, v_pool, op="kv_compact"):
        from forge.kernels.triton.attn_spec import kv_compact_triton

        kv_compact_triton(k_pool, v_pool, block_tables, ctx_lens, accepted_idx, accept_lens)
        return
    kv_compact_ref(k_pool, v_pool, block_tables, ctx_lens, accepted_idx, accept_lens)


# ------------------------------------------------------------------------------------------------------------
# top-k selection
# ------------------------------------------------------------------------------------------------------------
def topk_select_torch(scores: torch.Tensor, k: int, valid_lens: torch.Tensor | None = None) -> torch.Tensor:
    """Vectorised deterministic top-k with the exact semantics of :func:`topk_select_ref` (ties -> lower position, output
    ascending, ``-1`` padded): ``torch.topk`` finds the k-th value, ties are resolved with a cumulative rank."""
    b, t = scores.shape
    dev = scores.device
    s = torch.nan_to_num(scores.float(), nan=float("-inf"))
    s = torch.where(s == 0, torch.zeros_like(s), s)  # -0.0 == 0.0
    pos = torch.arange(t, device=dev).unsqueeze(0)
    valid = pos < (valid_lens.to(dev).long().unsqueeze(1) if valid_lens is not None else t)
    sm = torch.where(valid, s, torch.full_like(s, float("-inf")))
    kk = min(k, t)
    thr = torch.topk(sm, kk, dim=1).values[:, -1:]
    gt = (sm > thr) & valid
    eq = (sm == thr) & valid
    need = kk - gt.sum(1, keepdim=True)
    rank = torch.cumsum(eq.to(torch.int64), dim=1)
    sel = gt | (eq & (rank <= need))
    key = torch.where(sel, pos.expand(b, t), torch.full_like(pos.expand(b, t), t))
    idx = torch.topk(key, kk, dim=1, largest=False).values
    idx = torch.where(idx >= t, torch.full_like(idx, -1), idx).to(torch.int32)
    if kk < k:
        idx = torch.cat([idx, torch.full((b, k - kk), -1, dtype=torch.int32, device=dev)], dim=1)
    return idx


def topk_select(scores: torch.Tensor, k: int, valid_lens: torch.Tensor | None = None, method: str = "auto") -> torch.Tensor:
    """Deterministic top-k positions per row -> ``idx [B, k]`` int32, ascending, ``-1`` padded.

    ``method``: ``"auto"`` (default: vectorised torch implementation - validated vendor ``torch.topk`` underneath; set
    ``FORGE_TRITON_TOPK=1`` to prefer the Triton bisection kernel), ``"torch"``, ``"triton"``, ``"ref"`` (python loop oracle).
    """
    if method == "ref":
        return topk_select_ref(scores, k, valid_lens)
    use_tri = method == "triton" or (method == "auto" and _d.env_flag("FORGE_TRITON_TOPK"))
    if use_tri and _tri(scores, op="topk_select"):
        from forge.kernels.triton.attn_sparse import topk_select_triton

        return topk_select_triton(scores, k, valid_lens)
    _d.note_path("topk_select", "torch")
    return topk_select_torch(scores, k, valid_lens)


# ------------------------------------------------------------------------------------------------------------
# TEAL + LoRA
# ------------------------------------------------------------------------------------------------------------
def teal_gemv(x: torch.Tensor, w_t: torch.Tensor, threshold: float, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Activation-sparsity gather-GEMV (TEAL): channels with ``|x| <= threshold`` are skipped. ``w_t`` is ``[K, N]``."""
    if x.dtype in (torch.float16, torch.bfloat16) and x.shape[0] <= 16 and _tri(x, w_t, op="teal_gemv"):
        from forge.kernels.triton.util_gemv import teal_gemv_triton

        return teal_gemv_triton(x, w_t, threshold, bias)
    return _ad.teal_gemv_ref(x, w_t, threshold, bias)


def segmented_lora(x: torch.Tensor, y: torch.Tensor, adapter_ids: torch.Tensor, lora_a: torch.Tensor,
                   lora_b: torch.Tensor, scalings: torch.Tensor | float = 1.0, *, inplace: bool = False) -> torch.Tensor:
    """Segmented-batched LoRA/delta (K9): ``y[t] += scaling[a] * (x[t] @ A[a]^T) @ B[a]^T`` with ``a = adapter_ids[t]``
    (``-1`` = base model only) for a mixed batch.  Tokens are grouped by adapter with a stable sort and both thin GEMMs run as
    grouped GEMMs (adapters play the role of experts; the expand step scatters + accumulates into ``y`` in place, so the
    delta is never materialised).  ``lora_a [n, R, K]`` / ``lora_b [n, N, R]`` must share a (zero padded) rank ``R``;
    ranks below 16 are padded automatically on the Triton path.  The rank-``R`` intermediate makes one trip through L2.
    """
    from forge.kernels.ops_moe import grouped_gemm

    t = x.shape[0]
    n_ad = lora_a.shape[0]
    out = y if inplace else y.clone()
    if t == 0 or n_ad == 0:
        return out
    ids = adapter_ids.to(x.device).long()
    ids = torch.where((ids >= 0) & (ids < n_ad), ids, torch.full_like(ids, n_ad))
    order = torch.sort(ids, stable=True).indices
    counts = torch.zeros(n_ad + 1, dtype=torch.long, device=x.device).scatter_add_(0, ids, torch.ones_like(ids))[:n_ad]
    offsets = torch.zeros(n_ad + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    sc = torch.as_tensor(scalings, dtype=torch.float32, device=x.device)
    if sc.dim() == 0:
        sc = sc.expand(n_ad).contiguous()
    r = lora_a.shape[1]
    a_st, b_st = lora_a, lora_b
    if r % 16 and _d.use_triton(x, min_sm=80):
        pad = ((r + 15) // 16) * 16 - r
        a_st = torch.nn.functional.pad(lora_a, (0, 0, 0, pad))
        b_st = torch.nn.functional.pad(lora_b, (0, pad))
    a_st = a_st.to(x.dtype)
    b_st = b_st.to(x.dtype)
    order32 = order.to(torch.int32)
    u = grouped_gemm(x, a_st, offsets, gather_idx=order32, total_rows=t)
    grouped_gemm(u, b_st, offsets, out_idx=order32, out=out, accumulate=True, alpha=sc, total_rows=t)
    return out


# ------------------------------------------------------------------------------------------------------------
# benchmark cases
# ------------------------------------------------------------------------------------------------------------
def _dev(cfg: dict) -> torch.device:
    return torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))


def bench_cases() -> list[dict]:
    """Benchmark descriptors for ``forge.kernels.bench``: ``{name, family, make_inputs(cfg), run, ref, shapes,
    bytes_moved(cfg), flops(cfg)}`` - ``run`` / ``ref`` take the dict returned by ``make_inputs``."""
    dt = lambda cfg: getattr(torch, cfg.get("dtype", "bfloat16"))  # noqa: E731

    def had_inputs(cfg):
        return {"x": torch.randn(cfg["rows"], cfg["dim"], device=_dev(cfg), dtype=dt(cfg)), "block": cfg["block"]}

    def quant_inputs(cfg):
        return {"x": torch.randn(cfg["rows"], cfg["dim"], device=_dev(cfg), dtype=dt(cfg))}

    def rms_inputs(cfg):
        return {"x": torch.randn(cfg["rows"], cfg["dim"], device=_dev(cfg), dtype=dt(cfg)),
                "w": torch.rand(cfg["dim"], device=_dev(cfg))}

    def teal_inputs(cfg):
        d = _dev(cfg)
        x = torch.randn(cfg["batch"], cfg["k"], device=d, dtype=dt(cfg))
        thr = teal_threshold(x, cfg.get("sparsity", 0.5))
        return {"x": x, "w": torch.randn(cfg["k"], cfg["n"], device=d, dtype=dt(cfg)), "thr": thr}

    def topk_inputs(cfg):
        return {"s": torch.randn(cfg["batch"], cfg["t"], device=_dev(cfg)), "k": cfg["k"]}

    nbytes = lambda cfg: 2 * cfg["rows"] * cfg["dim"] * 2  # noqa: E731
    return [
        {"name": "hadamard", "family": "util", "make_inputs": had_inputs,
         "run": lambda i: hadamard_transform(i["x"], i["block"]),
         "ref": lambda i: _had.hadamard_ref(i["x"], i["block"]),
         "shapes": [{"rows": r, "dim": 4096, "block": b} for r in (1, 64, 4096) for b in (16, 128)],
         "bytes_moved": nbytes, "flops": lambda cfg: 2 * cfg["rows"] * cfg["dim"] * cfg["block"]},
        {"name": "quantize_fp8_rowwise", "family": "util", "make_inputs": quant_inputs,
         "run": lambda i: quantize_rowwise(i["x"], "fp8"), "ref": lambda i: _uq.quantize_rowwise_ref(i["x"], "fp8"),
         "shapes": [{"rows": r, "dim": 4096} for r in (1, 64, 4096)],
         "bytes_moved": lambda cfg: cfg["rows"] * cfg["dim"] * 3, "flops": lambda cfg: 3 * cfg["rows"] * cfg["dim"]},
        {"name": "rmsnorm_quant", "family": "util", "make_inputs": rms_inputs,
         "run": lambda i: rmsnorm_quant(i["x"], i["w"], 1e-6, "int8"),
         "ref": lambda i: _uq.rmsnorm_quant_ref(i["x"], i["w"], 1e-6, "int8"),
         "shapes": [{"rows": r, "dim": 4096} for r in (1, 64, 4096)],
         "bytes_moved": lambda cfg: cfg["rows"] * cfg["dim"] * 3, "flops": lambda cfg: 6 * cfg["rows"] * cfg["dim"]},
        {"name": "teal_gemv", "family": "util", "make_inputs": teal_inputs,
         "run": lambda i: teal_gemv(i["x"], i["w"], i["thr"]), "ref": lambda i: _ad.teal_gemv_ref(i["x"], i["w"], i["thr"]),
         "shapes": [{"batch": 1, "k": 4096, "n": 4096, "sparsity": s} for s in (0.0, 0.5)],
         "bytes_moved": lambda cfg: int(cfg["k"] * cfg["n"] * 2 * (1 - cfg.get("sparsity", 0.5))),
         "flops": lambda cfg: 2 * cfg["batch"] * cfg["k"] * cfg["n"] * (1 - cfg.get("sparsity", 0.5))},
        {"name": "topk_select", "family": "sparse", "make_inputs": topk_inputs,
         "run": lambda i: topk_select(i["s"], i["k"]), "ref": lambda i: topk_select_ref(i["s"], i["k"]),
         "shapes": [{"batch": 1, "t": 131072, "k": 2048}, {"batch": 8, "t": 32768, "k": 2048}],
         "bytes_moved": lambda cfg: cfg["batch"] * cfg["t"] * 4, "flops": lambda cfg: cfg["batch"] * cfg["t"]},
    ]
