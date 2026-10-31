"""Mixed-precision bit allocation under a byte budget (Plan CP-12 "per-layer / per-expert bit-widths from sensitivity").

* :func:`layer_sensitivity` -- per linear layer and candidate bit width, the Hessian-weighted output error of round-to-nearest
  quantisation, ``tr(dW H dW^T)`` with ``dW = W - Q_b(W)`` and ``H = 2/n X^T X`` (the GPTQ layer-wise objective; no extra forward passes
  beyond one block-wise sweep).  A layer that never saw calibration tokens (cold expert) gets the plain weight error.
* :func:`allocate_bits` -- choose one bit width per layer minimising the summed sensitivity subject to a byte budget.  ``dp`` is an exact
  multiple-choice knapsack over a discretised byte axis; ``greedy`` upgrades along each layer's lower convex hull by best marginal gain per
  byte (optimal for convex curves, used when there are very many layers).
* :func:`allocate_for_model` glue used by ``ptq.quantize_model(mixed_precision=True)``.

Bytes of a layer at ``b`` bits: ``n_params * (b + (scale_bits + zero_bits) / group) / 8``.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from forge.compress.modelutil import (
    HessianAccumulator,
    capture_block_inputs,
    find_blocks,
    iter_dense_linears,
    model_device,
    run_with_hooks,
)
from forge.core.errors import ConfigError
from forge.formats.int_quant import fake_quant_int


def layer_bytes(n_params: int, bits: int, group_size: int, scale_bits: float = 16.0, zero_bits: float = 0.0) -> float:
    g = group_size if group_size and group_size > 0 else max(n_params, 1)
    return n_params * (bits + (scale_bits + zero_bits) / g) / 8.0


def allocate_bits(
    sens: dict[str, dict[int, float]],
    n_params: dict[str, int],
    budget_bytes: float,
    *,
    group_size: int = 128,
    scale_bits: float = 16.0,
    zero_bits: float = 0.0,
    method: str = "auto",
    resolution: int = 4000,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Pick ``bits[name]`` from the keys of ``sens[name]`` minimising ``sum sens[name][bits]`` s.t. ``sum bytes <= budget_bytes``."""
    names = list(sens)
    opts: dict[str, list[tuple[float, float, int]]] = {}
    for n in names:
        o = sorted((layer_bytes(n_params[n], b, group_size, scale_bits, zero_bits), float(l), int(b)) for b, l in sens[n].items())
        opts[n] = o
    min_total = sum(o[0][0] for o in opts.values())
    if min_total > budget_bytes + 1e-6:
        raise ConfigError(f"byte budget {budget_bytes:.0f} is below the cheapest allocation ({min_total:.0f} bytes)")
    if method == "auto":
        method = "dp" if len(names) <= 600 else "greedy"
    if method == "dp":
        choice = _dp(names, opts, budget_bytes, resolution)
    elif method == "greedy":
        choice = _greedy(names, opts, budget_bytes)
    else:
        raise ConfigError(f"unknown allocation method {method!r} (auto|dp|greedy)")
    bits = {n: opts[n][choice[n]][2] for n in names}
    total_b = sum(opts[n][choice[n]][0] for n in names)
    total_l = sum(opts[n][choice[n]][1] for n in names)
    npar = sum(n_params.values())
    return bits, {"method": method, "bytes": total_b, "budget": budget_bytes, "loss": total_l, "avg_bits": 8 * total_b / max(npar, 1),
                  "histogram": {b: sum(1 for v in bits.values() if v == b) for b in sorted(set(bits.values()))}}


def _dp(names: list[str], opts: dict, budget: float, resolution: int) -> dict[str, int]:
    min_total = sum(opts[n][0][0] for n in names)
    spare = max(budget - min_total, 0.0)
    unit = spare / resolution if spare > 0 else 1.0
    inf = float("inf")
    # dp over the extra bytes above the cheapest option, in units; costs rounded UP so the result never exceeds the budget
    dp = np.zeros(resolution + 1)
    back: list[np.ndarray] = []
    for n in names:
        base = opts[n][0][0]
        extra = [int(np.ceil((o[0] - base) / unit - 1e-9)) if spare > 0 else (0 if o[0] == base else resolution + 1) for o in opts[n]]
        new = np.full(resolution + 1, inf)
        arg = np.zeros(resolution + 1, dtype=np.int32)
        for j, (e, o) in enumerate(zip(extra, opts[n])):
            if e > resolution:
                continue
            cand = np.full(resolution + 1, inf)
            cand[e:] = dp[: resolution + 1 - e] + o[1]
            better = cand < new
            new = np.where(better, cand, new)
            arg = np.where(better, j, arg)
        dp = new
        back.append(arg)
    u = int(np.argmin(dp))
    choice: dict[str, int] = {}
    for n, arg in zip(reversed(names), reversed(back)):
        j = int(arg[u])
        choice[n] = j
        base = opts[n][0][0]
        e = int(np.ceil((opts[n][j][0] - base) / unit - 1e-9)) if spare > 0 else 0
        u -= e
    return choice


def _hull(o: list[tuple[float, float, int]]) -> list[int]:
    """Indices of the lower convex hull of (bytes, loss) points (sorted by bytes), keeping only loss-decreasing steps."""
    pts = [(i, o[i][0], o[i][1]) for i in range(len(o))]
    pts_f = [pts[0]]
    for p in pts[1:]:
        if p[2] < pts_f[-1][2]:
            pts_f.append(p)
    hull: list[tuple[int, float, float]] = []
    for p in pts_f:
        while len(hull) >= 2:
            (_, x1, y1), (_, x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[1] - x2) >= (p[2] - y2) * (x2 - x1):  # hull[-1] is above the chord -> not convex
                hull.pop()
            else:
                break
        hull.append(p)
    return [h[0] for h in hull]


def _greedy(names: list[str], opts: dict, budget: float) -> dict[str, int]:
    import heapq

    hulls = {n: _hull(opts[n]) for n in names}
    pos = {n: 0 for n in names}
    choice = {n: hulls[n][0] for n in names}
    spent = sum(opts[n][choice[n]][0] for n in names)
    heap: list[tuple[float, str]] = []

    def push(n: str) -> None:
        h = hulls[n]
        if pos[n] + 1 < len(h):
            a, b = opts[n][h[pos[n]]], opts[n][h[pos[n] + 1]]
            heapq.heappush(heap, ((b[1] - a[1]) / max(b[0] - a[0], 1e-12), n))  # negative slope: most negative first

    for n in names:
        push(n)
    while heap:
        _, n = heapq.heappop(heap)
        h = hulls[n]
        a, b = opts[n][h[pos[n]]], opts[n][h[pos[n] + 1]]
        if spent + (b[0] - a[0]) <= budget + 1e-9:
            spent += b[0] - a[0]
            pos[n] += 1
            choice[n] = h[pos[n]]
            push(n)
    return choice


def brute_force(sens: dict[str, dict[int, float]], n_params: dict[str, int], budget_bytes: float, group_size: int = 128, scale_bits: float = 16.0) -> dict[str, int]:
    """Exhaustive optimum (testing aid; exponential in the number of layers)."""
    names = list(sens)
    best, best_l = None, float("inf")
    for combo in itertools.product(*[sorted(sens[n]) for n in names]):
        b = sum(layer_bytes(n_params[n], c, group_size, scale_bits) for n, c in zip(names, combo))
        if b > budget_bytes + 1e-9:
            continue
        l = sum(sens[n][c] for n, c in zip(names, combo))
        if l < best_l:
            best_l, best = l, combo
    if best is None:
        raise ConfigError("no feasible allocation")
    return dict(zip(names, best))


# ------------------------------------------------------------------------------------------------ model glue
def layer_sensitivity(
    model: nn.Module,
    tokens: torch.Tensor,
    candidate_bits: list[int],
    *,
    group_size: int = 128,
    sym: bool = True,
    batch_size: int = 4,
    exclude: list[str] | None = None,
    include: list[str] | None = None,
) -> tuple[dict[str, dict[int, float]], dict[str, int]]:
    """``(sens[name][bits], n_params[name])`` from one unquantised block-wise sweep (see module docstring)."""
    from forge.compress.modelutil import DEFAULT_EXCLUDE, discover_input_groups

    prefix, blocks = find_blocks(model)
    dev = model_device(model)
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    binp = capture_block_inputs(model, tokens, batch_size=batch_size)
    sens: dict[str, dict[int, float]] = {}
    npar: dict[str, int] = {}
    for bi, block in enumerate(blocks):
        bprefix = f"{prefix}.{bi}"
        lins = {n[len(bprefix) + 1 :]: m for n, m in iter_dense_linears(block, bprefix, exclude, include)}
        if lins:
            groups = discover_input_groups(block, binp, lins)
            accs, cbs = {}, {}
            for grp in groups:
                a = HessianAccumulator(lins[grp[0]].in_features, dev)
                accs[grp[0]] = a
                cbs[grp[0]] = (lins[grp[0]], lambda x, _o, a=a: a.add(x))
            run_with_hooks(block, binp, cbs)
            for grp in groups:
                H = accs[grp[0]].H
                for rel in grp:
                    W = lins[rel].weight.detach().float()
                    full = f"{bprefix}.{rel}"
                    sens[full], npar[full] = {}, W.numel()
                    for b in candidate_bits:
                        g = group_size if group_size > 0 and W.shape[1] % group_size == 0 else W.shape[1]
                        dW = W - fake_quant_int(W, b, g, sym)
                        sens[full][b] = float(((dW @ H) * dW).sum()) if accs[grp[0]].n > 0 else float(dW.pow(2).sum())
        binp.propagate(block)
    return sens, npar


def allocate_for_model(model: nn.Module, tokens: torch.Tensor, cfg: Any, ctx: Any = None) -> tuple[dict[str, int], dict[str, Any]]:
    """Allocation for ``ptq.PTQConfig`` (``avg_bits`` / ``byte_budget``, ``candidate_bits``, group/sym)."""
    if tokens is None:
        raise ConfigError("mixed_precision needs calibration tokens")
    if cfg.byte_budget is None and cfg.avg_bits is None:
        raise ConfigError("mixed_precision needs avg_bits or byte_budget")
    cands = sorted(set(int(b) for b in cfg.candidate_bits))
    sens, npar = layer_sensitivity(model, tokens, cands, group_size=cfg.group_size, sym=cfg.sym, batch_size=cfg.batch_size,
                                   exclude=cfg.exclude, include=cfg.include)
    total = sum(npar.values())
    budget = float(cfg.byte_budget) if cfg.byte_budget is not None else cfg.avg_bits * total / 8.0
    zero_bits = 0.0 if cfg.sym else float(max(cands))
    bits, info = allocate_bits(sens, npar, budget, group_size=cfg.group_size, zero_bits=zero_bits)
    if ctx is not None:
        ctx.log(f"bit allocation ({info['method']}): avg {info['avg_bits']:.3f} bits, histogram {info['histogram']}")
    info["sensitivity_sum_uniform_min_bits"] = sum(s[cands[0]] for s in sens.values())
    return bits, info
