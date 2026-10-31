"""MoE launchers / dispatchers (Plan K7, KR-9): routing, deterministic permutation, grouped GEMM, combine, expert indirection.

Typical pipeline (all functions are differentiable-free inference/forward ops)::

    w, ids = moe_route(router_logits, top_k)                    # fused softmax/top-k
    ids, w = apply_expert_map(ids, w, expert_map)               # optional: pruned / merged experts (indirection table)
    pr = moe_permute(ids, num_experts)                          # deterministic grouping of token slots by expert
    gu = grouped_gemm(x, w_gate_up, pr.expert_offsets, gather_idx=pr.token_idx, total_rows=T * k)
    y  = grouped_gemm(silu(gu[:, :I]) * gu[:, I:], w_down, pr.expert_offsets, total_rows=T * k)
    out = moe_unpermute_combine(y, pr.inv_perm, w)

or just :func:`moe_experts_forward`.  Triton is used only when CUDA + Triton + capability allow (and ``FORGE_FORCE_REF`` is
unset); **the Triton kernels are implemented but not yet validated on NVIDIA hardware**.
"""
from __future__ import annotations

from typing import Any

import torch

from forge.kernels.ref import moe_ref as _ref
from forge.kernels.ref.moe_ref import (  # noqa: F401  (public re-exports)
    PermuteResult,
    apply_expert_map_ref,
    build_block_schedule_ref,
    dequant_w4_experts,
    max_blocks_bound,
    moe_experts_forward_ref,
    moe_grouped_gemm_ref,
    moe_permute_ref,
    moe_route_ref,
    moe_unpermute_combine_ref,
    pack_w4_experts,
)
from forge.kernels.triton import util_dispatch as _d

KERNELS: dict[str, dict[str, Any]] = {
    "moe_route": {"family": "moe", "min_sm": 70, "triton": "forge.kernels.triton.moe_kernels", "ref": "moe_ref.moe_route_ref"},
    "moe_grouped_gemm": {"family": "moe", "min_sm": 80, "triton": "forge.kernels.triton.moe_kernels", "ref": "moe_ref.moe_grouped_gemm_ref"},
    "moe_combine": {"family": "moe", "min_sm": 70, "triton": "forge.kernels.triton.moe_kernels", "ref": "moe_ref.moe_unpermute_combine_ref"},
}


def _tri(*tensors, min_sm: int = 70, op: str = "") -> bool:
    ok = _d.use_triton(*tensors, min_sm=min_sm, op=op)
    if ok:
        for t in tensors:
            if t is not None and t.dtype == torch.bfloat16 and not _d.bf16_ok(t.dtype, t.device):
                ok = False
    _d.note_path(op, "triton" if ok else "reference")
    return ok


def moe_route(logits: torch.Tensor, top_k: int, *, scoring: str = "softmax", norm_topk_prob: bool = True,
              routed_scaling: float = 1.0, score_bias: torch.Tensor | None = None):
    """Router: ``logits [T, E]`` -> ``(weights [T, k] float32, ids [T, k] int32)``; ties go to the lower expert id."""
    if (logits.dtype in (torch.float16, torch.bfloat16, torch.float32) and logits.shape[1] <= 2048 and top_k <= 32
            and _tri(logits, op="moe_route")):
        from forge.kernels.triton.moe_kernels import moe_route_triton

        return moe_route_triton(logits, top_k, scoring=scoring, norm_topk_prob=norm_topk_prob,
                                routed_scaling=routed_scaling, score_bias=score_bias)
    return moe_route_ref(logits, top_k, scoring=scoring, norm_topk_prob=norm_topk_prob, routed_scaling=routed_scaling,
                         score_bias=score_bias)


def apply_expert_map(topk_ids: torch.Tensor, topk_weights: torch.Tensor, expert_map: torch.Tensor,
                     renormalize: bool = False):
    """Indirection table for pruned/merged experts: original id -> kept slot (``-1`` = pruned, dropped). The same kernels
    then run unchanged on the smaller expert set. Returns ``(ids, weights)``."""
    return apply_expert_map_ref(topk_ids, topk_weights, expert_map, renormalize)


def moe_permute(topk_ids: torch.Tensor, num_experts: int) -> PermuteResult:
    """Deterministic token permutation (stable sort on the flat assignment index; dropped ids ``<0`` go last)."""
    return moe_permute_ref(topk_ids, num_experts)


def build_block_schedule(offsets: torch.Tensor, block_m: int, max_blocks: int):
    """Host-sync-free tile schedule for the grouped GEMM (see ``ref.moe_ref.build_block_schedule_ref``)."""
    return build_block_schedule_ref(offsets, block_m, max_blocks)


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor, *, gather_idx: torch.Tensor | None = None,
                 out_idx: torch.Tensor | None = None, out: torch.Tensor | None = None, accumulate: bool = False,
                 alpha: torch.Tensor | None = None, w_scales: torch.Tensor | None = None,
                 w_zeros: torch.Tensor | None = None, group_size: int | None = None, total_rows: int | None = None,
                 num_rows_out: int | None = None) -> torch.Tensor:
    """Grouped GEMM over expert slabs: ``y[r] = x[gather_idx[r]] @ W[e(r)]^T`` for sorted rows ``r`` of expert ``e``
    (``offsets [E+1]`` int32; experts with zero rows are free).  ``w [E, N, K]`` bf16/fp16 or packed W4A16 (``uint8 [E, N, K/2]``
    + ``w_scales`` + optional ``w_zeros`` + ``group_size``).  Fusions: ``gather_idx`` (activation row gather),
    ``out_idx`` (+ ``out`` / ``accumulate``) scatter-accumulate epilogue, ``alpha [E]`` per-expert output scale.
    ``total_rows`` = static number of sorted rows (``T * k``); needed to avoid host syncs on the Triton path."""
    w4 = w.dtype == torch.uint8
    dtype_ok = x.dtype in (torch.float16, torch.bfloat16, torch.float32) and (w4 or w.dtype == x.dtype)
    if dtype_ok and x.shape[0] > 0 and _tri(x, w, offsets, min_sm=80, op="moe_grouped_gemm"):
        from forge.kernels.triton.moe_kernels import moe_grouped_gemm_triton

        return moe_grouped_gemm_triton(x, w, offsets, gather_idx=gather_idx, out_idx=out_idx, out=out,
                                       accumulate=accumulate, alpha=alpha, w_scales=w_scales, w_zeros=w_zeros,
                                       group_size=group_size, total_rows=total_rows, num_rows_out=num_rows_out)
    return moe_grouped_gemm_ref(x, w, offsets, gather_idx=gather_idx, out_idx=out_idx, out=out, accumulate=accumulate,
                                alpha=alpha, w_scales=w_scales, w_zeros=w_zeros, group_size=group_size)


def moe_unpermute_combine(y_sorted: torch.Tensor, inv_perm: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Weighted combine ``out[t] = sum_s w[t, s] * y[inv_perm[t * k + s]]`` (fp32 accumulation)."""
    if y_sorted.dtype in (torch.float16, torch.bfloat16, torch.float32) and _tri(y_sorted, op="moe_combine"):
        from forge.kernels.triton.moe_kernels import moe_combine_triton

        return moe_combine_triton(y_sorted, inv_perm, topk_weights)
    return moe_unpermute_combine_ref(y_sorted, inv_perm, topk_weights)


def moe_experts_forward(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, w_gate_up: torch.Tensor,
                        w_down: torch.Tensor, *, activation: str = "silu", expert_map: torch.Tensor | None = None,
                        renormalize_after_prune: bool = False, gate_up_scales=None, gate_up_zeros=None,
                        down_scales=None, down_zeros=None, group_size: int | None = None) -> torch.Tensor:
    """Complete MoE FFN. ``x [T, H]``; ``w_gate_up [E, 2I, H]`` (gate rows then up rows), ``w_down [E, H, I]`` (``E`` = number
    of *kept* experts when ``expert_map`` is given); W4A16 experts via the ``*_scales/zeros`` + ``group_size`` arguments."""
    t = x.shape[0]
    k = topk_ids.shape[1]
    e = w_gate_up.shape[0]
    ids, w = topk_ids, topk_weights
    if expert_map is not None:
        ids, w = apply_expert_map(ids, w, expert_map, renormalize_after_prune)
    pr = moe_permute(ids, e)
    gu = grouped_gemm(x, w_gate_up, pr.expert_offsets, gather_idx=pr.token_idx, total_rows=t * k,
                      w_scales=gate_up_scales, w_zeros=gate_up_zeros, group_size=group_size)
    inter = gu.shape[1] // 2
    actf = {"silu": torch.nn.functional.silu, "gelu": torch.nn.functional.gelu}[activation]
    h = (actf(gu[:, :inter].float()) * gu[:, inter:].float()).to(x.dtype)
    y = grouped_gemm(h, w_down, pr.expert_offsets, total_rows=t * k, w_scales=down_scales, w_zeros=down_zeros,
                     group_size=group_size)
    return moe_unpermute_combine(y, pr.inv_perm, w)


def bench_cases() -> list[dict]:
    """Benchmark descriptors (see ``ops_util.bench_cases``)."""
    dt = lambda cfg: getattr(torch, cfg.get("dtype", "bfloat16"))  # noqa: E731
    dev = lambda cfg: torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))  # noqa: E731

    def route_inputs(cfg):
        return {"logits": torch.randn(cfg["tokens"], cfg["experts"], device=dev(cfg), dtype=dt(cfg)), "k": cfg["top_k"]}

    def gg_inputs(cfg):
        d = dev(cfg)
        t, e, k = cfg["tokens"], cfg["experts"], cfg["top_k"]
        logits = torch.randn(t, e, device=d)
        if cfg.get("skew", 0):
            logits[:, : max(1, e // 8)] += cfg["skew"]
        _, ids = moe_route_ref(logits, k)
        pr = moe_permute_ref(ids, e)
        return {"x": torch.randn(t, cfg["hidden"], device=d, dtype=dt(cfg)),
                "w": torch.randn(e, cfg["inter"], cfg["hidden"], device=d, dtype=dt(cfg)) * 0.05, "pr": pr, "m": t * k}

    def gg_flops(cfg):
        return 2 * cfg["tokens"] * cfg["top_k"] * cfg["hidden"] * cfg["inter"]

    def gg_bytes(cfg):
        touched = min(cfg["experts"], cfg["tokens"] * cfg["top_k"])
        return 2 * (touched * cfg["inter"] * cfg["hidden"] + cfg["tokens"] * cfg["top_k"] * (cfg["hidden"] + cfg["inter"]))

    return [
        {"name": "moe_route", "family": "moe", "make_inputs": route_inputs,
         "run": lambda i: moe_route(i["logits"], i["k"]), "ref": lambda i: moe_route_ref(i["logits"], i["k"]),
         "shapes": [{"tokens": t, "experts": e, "top_k": 8} for t in (1, 256, 8192) for e in (64, 128)],
         "bytes_moved": lambda cfg: cfg["tokens"] * cfg["experts"] * 2, "flops": lambda cfg: 4 * cfg["tokens"] * cfg["experts"]},
        {"name": "moe_grouped_gemm", "family": "moe", "make_inputs": gg_inputs,
         "run": lambda i: grouped_gemm(i["x"], i["w"], i["pr"].expert_offsets, gather_idx=i["pr"].token_idx, total_rows=i["m"]),
         "ref": lambda i: moe_grouped_gemm_ref(i["x"], i["w"], i["pr"].expert_offsets, gather_idx=i["pr"].token_idx),
         "shapes": [{"tokens": t, "experts": 64, "top_k": 8, "hidden": 2048, "inter": 1408, "skew": s}
                    for t in (1, 16, 256, 4096) for s in (0, 2)],
         "bytes_moved": gg_bytes, "flops": gg_flops},
    ]


#: ARCHITECTURE 5.5 name for the combine step
moe_unpermute = moe_unpermute_combine
