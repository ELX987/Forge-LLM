"""Low-rank compression (Plan CP-9): activation-whitened SVD (SVD-LLM style), heterogeneous rank allocation, break-even guard, healing.

For a layer ``y = W x`` with calibration activations ``X`` and ``C = X X^T = S S^T`` (Cholesky), truncating the SVD of ``W S`` gives the
rank-r matrix minimising the **output** error ``||(W - W_r) X||_F``; that error equals the energy of the discarded singular values of ``W S``
(``sum_{i>r} sigma_i^2``) -- exactly the quantity the rank allocator trades off across layers (and the property the unit test verifies).

Break-even guard: ``W_r = A B`` with ``A [m,r]``, ``B [r,n]`` stores ``r (m+n)`` numbers instead of ``m n`` and costs the same number of
multiply-adds per token, so it only helps when ``r < m n / (m + n)`` (square matrices: ``r < n/2``).  The planner therefore refuses larger ranks unless
``allow_above_breakeven`` is set (memory-only goals; it then no longer saves bytes either, which is reported).

Statistics come from the **dense** model in one block-wise sweep (non-sequential): ranks are only known after all layers have been seen.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from forge.compress.calibrate import get_calibration_tokens
from forge.compress.modelutil import (
    DEFAULT_EXCLUDE,
    HessianAccumulator,
    capture_block_inputs,
    discover_input_groups,
    find_blocks,
    iter_dense_linears,
    model_device,
    model_dtype,
    run_with_hooks,
    set_submodule,
)
from forge.compress.serialize import register_module_kind
from forge.core.context import NullContext, RunContext
from forge.core.errors import ConfigError

log = logging.getLogger("forge.compress.lowrank")


class LowRankConfig(BaseModel):
    """Low-rank factorisation settings (stage ``lowrank``)."""

    model_config = ConfigDict(extra="forbid")

    target_ratio: float = Field(0.7, gt=0, le=1.0, description="Fraction of the targeted layers' parameters to keep after factorisation.")
    rank_alloc: Literal["heterogeneous", "uniform"] = Field("heterogeneous", description="heterogeneous = water-filling over whitened singular values across layers; uniform = same keep ratio everywhere.")
    normalize: Literal["relative", "absolute"] = Field("relative", description="Heterogeneous allocation: weigh each layer's discarded energy relative to its total energy (relative) or in raw output units (absolute).")
    min_rank: int = Field(4, ge=1, description="Minimum rank of any factorised layer.")
    allow_above_breakeven: bool = Field(False, description="Permit ranks >= m*n/(m+n) (no byte/FLOP saving; memory-neutral experiments only).")
    damp: float = Field(1e-4, ge=0, description="Relative diagonal damping of the activation covariance before Cholesky.")
    whiten: bool = Field(True, description="Use activation-whitened SVD (false = plain SVD of W, the weak baseline).")
    exclude: list[str] = Field(list(DEFAULT_EXCLUDE), description="Regexes of modules left dense.")
    include: list[str] | None = Field(None, description="If set, only modules matching one of these regexes are factorised.")
    n_seqs: int = Field(64, ge=1, description="Calibration sequences.")
    seq_len: int = Field(256, ge=8, description="Calibration sequence length.")
    batch_size: int = Field(4, ge=1, description="Calibration batch size.")
    seed: int = Field(0, description="Calibration seed.")
    heal_steps: int = Field(0, ge=0, description="Short healing stage: gradient steps on the low-rank factors after truncation (0 = skip).")
    heal_lr: float = Field(1e-4, gt=0, description="Healing learning rate.")


def breakeven_rank(m: int, n: int) -> int:
    """Largest integer rank ``r`` with ``r < m n / (m + n)``."""
    return max(math.ceil(m * n / (m + n)) - 1, 0)


@register_module_kind("lowrank")
class LowRankLinear(nn.Module):
    """``y = (x B^T) A^T + bias`` with trainable factors ``A [out, r]`` and ``B [r, in]``."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.A = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False) if bias else None

    @classmethod
    def from_factors(cls, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor | None = None, dtype: torch.dtype | None = None) -> LowRankLinear:
        m = cls(B.shape[1], A.shape[0], A.shape[1], bias is not None, dtype or A.dtype, A.device)
        with torch.no_grad():
            m.A.copy_(A)
            m.B.copy_(B)
            if bias is not None:
                m.bias.copy_(bias)
        return m

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, rank: int = 1, **_: Any) -> LowRankLinear:
        return cls(in_features, out_features, rank, bias, dtype, device)

    def quant_spec(self) -> dict[str, Any]:
        return {"rank": self.rank}

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.A.float() @ self.B.float()
        return w.to(dtype) if dtype is not None else w

    @property
    def weight(self) -> torch.Tensor:  # read-only convenience (allocates), like PackedLinear
        return self.dequantize()

    def packed_nbytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(F.linear(x, self.B.to(x.dtype)), self.A.to(x.dtype))
        return y + self.bias.to(x.dtype) if self.bias is not None else y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


# ------------------------------------------------------------------------------------------------ SVD
def whitened_svd(
    W: torch.Tensor, cov: torch.Tensor | None, rank: int | None = None, damp: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(A [m,r], B [r,n], sigma [min(m,n)])`` with ``A B = U_r Sigma_r V_r^T S^-1`` for ``W S = U Sigma V^T`` (``cov = S S^T``).

    ``cov=None`` -> plain SVD (``S = I``).  ``rank=None`` -> keep everything up to ``min(m, n)``.  ``sigma`` is the full singular spectrum of
    ``W S`` (discarded output-error energy = sum of squares of the dropped tail when ``cov = X X^T``).
    """
    Wd = W.double()
    m, n = Wd.shape
    if cov is None:
        U, S, Vh = torch.linalg.svd(Wd, full_matrices=False)
        Sinv_t = None
    else:
        C = cov.double()
        C = C + damp * torch.diagonal(C).mean().clamp(min=1e-12) * torch.eye(n, dtype=torch.float64, device=C.device)
        L = torch.linalg.cholesky(C)
        U, S, Vh = torch.linalg.svd(Wd @ L, full_matrices=False)
        Sinv_t = L
    r = min(m, n) if rank is None else min(rank, min(m, n))
    A = U[:, :r] * S[:r].unsqueeze(0)
    Vr = Vh[:r]  # [r, n]; B = Vr S^-1 = Vr L^-1
    B = Vr if Sinv_t is None else torch.linalg.solve_triangular(Sinv_t, Vr, upper=False, left=False)
    return A.float(), B.float(), S.float()


def allocate_ranks(
    spectra: dict[str, torch.Tensor],
    shapes: dict[str, tuple[int, int]],
    target_ratio: float,
    *,
    mode: str = "heterogeneous",
    normalize: str = "relative",
    min_rank: int = 4,
    allow_above_breakeven: bool = False,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Ranks per layer under a global parameter budget ``target_ratio * sum(m n)``; every rank is capped below break-even (see module docstring)."""
    names = list(spectra)
    rmax = {n: (min(shapes[n]) if allow_above_breakeven else min(breakeven_rank(*shapes[n]), min(shapes[n]))) for n in names}
    rmin = {n: min(min_rank, rmax[n]) for n in names}
    dense = {n: shapes[n][0] * shapes[n][1] for n in names}
    budget = target_ratio * sum(dense.values())
    cost = {n: shapes[n][0] + shapes[n][1] for n in names}
    ranks = dict(rmin)
    if mode == "uniform":
        for n in names:
            ranks[n] = int(max(rmin[n], min(rmax[n], math.floor(target_ratio * dense[n] / cost[n]))))
    else:
        used = sum(ranks[n] * cost[n] for n in names)
        # candidate increments (gain per parameter), best first; spectra are descending so greedy == optimal for this separable concave problem
        cands: list[tuple[float, str]] = []
        for n in names:
            s2 = spectra[n].double().pow(2)
            tot = float(s2.sum()) if normalize == "relative" else 1.0
            for k in range(rmin[n], rmax[n]):
                cands.append((float(s2[k]) / max(tot, 1e-30) / cost[n], n))
        cands.sort(key=lambda t: -t[0])
        for _g, n in cands:
            if ranks[n] >= rmax[n]:
                continue
            if used + cost[n] > budget:
                continue
            ranks[n] += 1
            used += cost[n]
    kept = sum(ranks[n] * cost[n] for n in names)
    lost = {n: float(spectra[n].double().pow(2)[ranks[n]:].sum() / max(float(spectra[n].double().pow(2).sum()), 1e-30)) for n in names}
    info = {
        "params_dense": sum(dense.values()), "params_kept": kept, "ratio": kept / max(sum(dense.values()), 1), "budget_ratio": target_ratio,
        "mode": mode, "mean_rel_energy_lost": sum(lost.values()) / max(len(lost), 1), "max_rel_energy_lost": max(lost.values()) if lost else 0.0,
        "rel_energy_lost": lost, "breakeven_capped": [n for n in names if ranks[n] >= rmax[n] and not allow_above_breakeven],
    }
    return ranks, info


# ------------------------------------------------------------------------------------------------ model driver
def lowrank_compress_model(
    model: nn.Module,
    cfg: LowRankConfig | dict | None = None,
    calib: Any = "synthetic",
    *,
    tokenizer: Any = None,
    ctx: RunContext | None = None,
    teacher: nn.Module | None = None,
) -> dict[str, Any]:
    """Factorise the selected linears of ``model`` **in place**. Returns a report dict (ranks, byte accounting, optional healing curve)."""
    cfg = LowRankConfig.model_validate(cfg) if isinstance(cfg, dict) else (cfg or LowRankConfig())
    ctx = ctx or NullContext()
    t0 = time.perf_counter()
    prefix, blocks = find_blocks(model)
    dev = model_device(model)
    dtype = model_dtype(model)
    tokens, _ = get_calibration_tokens(calib, model, tokenizer, n_seqs=cfg.n_seqs, seq_len=cfg.seq_len, seed=cfg.seed)
    binp = capture_block_inputs(model, tokens, batch_size=cfg.batch_size) if cfg.whiten else None
    store: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}  # name -> (A, B, sigma) at rank <= rmax
    shapes: dict[str, tuple[int, int]] = {}
    for bi, block in enumerate(blocks):
        bprefix = f"{prefix}.{bi}"
        lins = {n[len(bprefix) + 1 :]: m for n, m in iter_dense_linears(block, bprefix, cfg.exclude, cfg.include)}
        covs: dict[str, torch.Tensor | None] = {r: None for r in lins}
        if cfg.whiten and lins:
            groups = discover_input_groups(block, binp, lins)
            accs, cbs = {}, {}
            for g in groups:
                a = HessianAccumulator(lins[g[0]].in_features, dev)
                accs[g[0]] = a
                cbs[g[0]] = (lins[g[0]], lambda x, _o, a=a: a.add(x))
            run_with_hooks(block, binp, cbs)
            for g in groups:
                for r in g:
                    covs[r] = accs[g[0]].H if accs[g[0]].n else None
        for rel, lin in lins.items():
            W = lin.weight.detach().float()
            m, n = W.shape
            rcap = min(m, n) if cfg.allow_above_breakeven else min(breakeven_rank(m, n), min(m, n))
            if rcap < 1:
                continue
            A, B, S = whitened_svd(W, covs[rel], rcap, cfg.damp)
            full = f"{bprefix}.{rel}"
            store[full] = (A.cpu(), B.cpu(), S.cpu())
            shapes[full] = (m, n)
        if binp is not None:
            binp.propagate(block)
        ctx.progress(0.5 * (bi + 1) / len(blocks), f"svd block {bi + 1}/{len(blocks)}")
    if not store:
        raise ConfigError("no layer can be factorised below the break-even rank")
    ranks, info = allocate_ranks({n: v[2] for n, v in store.items()}, shapes, cfg.target_ratio, mode=cfg.rank_alloc, normalize=cfg.normalize,
                                 min_rank=cfg.min_rank, allow_above_breakeven=cfg.allow_above_breakeven)
    bytes_before = sum(shapes[n][0] * shapes[n][1] * model.get_submodule(n).weight.element_size() for n in store)
    bytes_after = 0
    for name, (A, B, S) in store.items():
        r = ranks[name]
        lin = model.get_submodule(name)
        mod = LowRankLinear.from_factors(A[:, :r].to(dev), B[:r].to(dev), lin.bias.detach() if lin.bias is not None else None, dtype)
        set_submodule(model, name, mod)
        bytes_after += mod.packed_nbytes()
    info.update({"ranks": ranks, "layers": len(ranks), "bytes_before": bytes_before, "bytes_after": bytes_after, "seconds": time.perf_counter() - t0,
                 "whitened": cfg.whiten, "provenance": "measured", "stats_source": "dense model, non-sequential"})
    if cfg.heal_steps > 0:
        info["healing"] = lowrank_heal(model, tokens, steps=cfg.heal_steps, lr=cfg.heal_lr, teacher=teacher, batch_size=cfg.batch_size, ctx=ctx)
    ctx.progress(1.0, "lowrank done")
    return info


def lowrank_heal(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    steps: int = 50,
    lr: float = 1e-4,
    teacher: nn.Module | None = None,
    batch_size: int = 4,
    temperature: float = 1.0,
    ctx: RunContext | None = None,
) -> dict[str, Any]:
    """LoRA-style healing hook: fine-tune only the low-rank factors (``A``, ``B``) with KL to ``teacher`` logits (or next-token CE if no teacher).

    A short stage (tens of steps) -- the Plan pairs low-rank compression with such a healing stage.  Returns the loss curve.
    """
    ctx = ctx or NullContext()
    dev = model_device(model)
    params = [p for m in model.modules() if isinstance(m, LowRankLinear) for p in (m.A, m.B)]
    if not params:
        raise ConfigError("model has no LowRankLinear layers to heal")
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    losses: list[float] = []
    batches = list(tokens.split(batch_size))
    try:
        for step in range(steps):
            if ctx.should_stop():
                break
            ids = batches[step % len(batches)].to(dev)
            with torch.enable_grad():
                logits = model(ids).logits.float()
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(ids).logits.float()
                    loss = F.kl_div(F.log_softmax(logits / temperature, -1).reshape(-1, logits.shape[-1]),
                                    F.softmax(tl / temperature, -1).reshape(-1, logits.shape[-1]), reduction="batchmean") * temperature**2
                else:
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss.detach()))
            ctx.metric(step, heal_loss=losses[-1])
    finally:
        model.eval()
        for p, f in flags:
            p.requires_grad_(f)
    return {"steps": len(losses), "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None, "losses": losses,
            "objective": "kl_to_teacher" if teacher is not None else "next_token_ce"}
