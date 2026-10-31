"""Delta compression of fine-tuned model families (Plan CP-10): low-rank deltas and BitDelta (1-bit sign + scale).

A fine-tune ``W_ft = W_base + Delta``.  Instead of storing ``W_ft`` we store a compressed ``Delta`` next to the shared base:

``bitdelta``  ``Delta ~ alpha * sign(Delta)`` per weight matrix: 1 bit per weight + one scale (``alpha = mean|Delta|`` minimises ``||Delta - alpha sign||_2``);
              then the scales are *calibrated* by logit distillation against the fine-tuned model (BitDelta's second step) with the sign
              bits frozen (:func:`calibrate_bitdelta`).
``lowrank``   ``Delta ~ A B`` from a truncated SVD (rank ``r`` => ``r (m+n)`` numbers per matrix).
Tensors that are not 2-D transformer-block weights (embeddings, norms, LM head, biases) are stored as exact fp16 differences, as BitDelta does.

Serving: :func:`apply_delta` builds a model whose targeted linears are :class:`BitDeltaLinear` / :class:`LowRankDeltaLinear` sharing the base weight;
:func:`merge_delta` writes ``W_base + Delta`` back (dense).  :class:`MultiTenantLinear` evaluates many tenants' deltas on one shared base in a single
forward (reference implementation of the batched kernel: ``y = x W_base^T + alpha_t (x sign_t^T)``).  Reference numerics only.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from forge.compress.modelutil import model_device
from forge.core.context import NullContext, RunContext
from forge.core.errors import ConfigError
from forge.formats.packing import pack_bits, unpack_bits

log = logging.getLogger("forge.compress.delta")


class DeltaConfig(BaseModel):
    """Delta-compression settings (stage ``delta``)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["bitdelta", "lowrank"] = Field("bitdelta", description="1-bit sign + scale (BitDelta) or truncated-SVD low-rank delta.")
    rank: int = Field(16, ge=1, description="lowrank: rank of the delta factors.")
    calibrate_steps: int = Field(40, ge=0, description="bitdelta: scale-distillation steps against the fine-tuned model's logits (0 = keep alpha=mean|delta|).")
    calibrate_lr: float = Field(0.02, gt=0, description="bitdelta: RELATIVE learning rate of the scale calibration (each scale moves by about this fraction per step).")
    base: str | None = Field(None, description="Recipe stage: digest of the base model (default: the recipe's original input model).")
    finetuned: str | None = Field(None, description="Recipe stage: digest of the fine-tuned model (default: the current model).")
    calib_set: str = Field("synthetic", description="Calibration set for scale calibration.")
    n_seqs: int = Field(16, ge=1, description="Calibration sequences for scale calibration.")
    seq_len: int = Field(128, ge=8, description="Calibration sequence length.")
    batch_size: int = Field(4, ge=1, description="Calibration batch size.")
    seed: int = Field(0, description="Calibration seed.")


@dataclass
class DeltaEntry:
    kind: str  # bitdelta | lowrank | full
    shape: tuple[int, ...]
    bits: torch.Tensor | None = None  # packed sign bits uint8 [N, ceil(K/8)]
    alpha: torch.Tensor | None = None  # scalar fp32
    A: torch.Tensor | None = None
    B: torch.Tensor | None = None
    diff: torch.Tensor | None = None  # fp16 exact difference (kind=full)

    def nbytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in (self.bits, self.alpha, self.A, self.B, self.diff) if t is not None))


@dataclass
class Delta:
    """A compressed fine-tune delta relative to a base model."""

    method: str
    entries: dict[str, DeltaEntry] = field(default_factory=dict)  # module name (matrices) or parameter name (full diffs) -> entry
    meta: dict[str, Any] = field(default_factory=dict)

    def nbytes(self) -> int:
        return sum(e.nbytes() for e in self.entries.values())

    def size_report(self, dense_bytes: int) -> dict[str, Any]:
        mat = sum(e.nbytes() for e in self.entries.values() if e.kind != "full")
        full = sum(e.nbytes() for e in self.entries.values() if e.kind == "full")
        return {"delta_bytes": self.nbytes(), "matrix_bytes": mat, "full_diff_bytes": full, "dense_finetune_bytes": dense_bytes,
                "compression_vs_dense": dense_bytes / max(self.nbytes(), 1), "method": self.method}

    # -- persistence ---------------------------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        index: dict[str, Any] = {}
        for name, e in self.entries.items():
            index[name] = {"kind": e.kind, "shape": list(e.shape)}
            for field_name in ("bits", "alpha", "A", "B", "diff"):
                t = getattr(e, field_name)
                if t is not None:
                    tensors[f"{name}::{field_name}"] = t.detach().cpu().contiguous()
        save_file(tensors, str(out / "delta.safetensors"), metadata={"format": "pt"})
        (out / "delta.json").write_text(json.dumps({"method": self.method, "meta": self.meta, "entries": index}, indent=1, default=str))
        return out

    @classmethod
    def load(cls, path: str | Path) -> Delta:
        from safetensors.torch import load_file

        p = Path(path)
        info = json.loads((p / "delta.json").read_text())
        tensors = load_file(str(p / "delta.safetensors"))
        d = cls(info["method"], meta=info.get("meta", {}))
        for name, ent in info["entries"].items():
            kw = {f: tensors.get(f"{name}::{f}") for f in ("bits", "alpha", "A", "B", "diff")}
            d.entries[name] = DeltaEntry(ent["kind"], tuple(ent["shape"]), **kw)
        return d


def _is_target(name: str, w: torch.Tensor) -> bool:
    return w.ndim == 2 and name.startswith("layers.") and name.endswith(".weight") and "norm" not in name and "router" not in name and not name.endswith("mlp.gate.weight")


def compute_delta(base_sd: dict[str, torch.Tensor], ft_sd: dict[str, torch.Tensor], cfg: DeltaConfig | dict | None = None) -> Delta:
    """Compress ``ft_sd - base_sd`` (state dicts with identical keys/shapes)."""
    cfg = DeltaConfig.model_validate(cfg) if isinstance(cfg, dict) else (cfg or DeltaConfig())
    if set(base_sd) != set(ft_sd):
        raise ConfigError("base and fine-tuned state dicts have different keys")
    d = Delta(cfg.method, meta={"config": cfg.model_dump()})
    for k, wb in base_sd.items():
        wf = ft_sd[k]
        if wb.shape != wf.shape:
            raise ConfigError(f"shape mismatch for {k}: {tuple(wb.shape)} vs {tuple(wf.shape)}")
        if not wb.is_floating_point():
            continue
        diff = wf.float() - wb.to(wf.device).float()
        if not bool(diff.abs().max() > 0):
            continue
        if _is_target(k, wb):
            name = k[: -len(".weight")]
            if cfg.method == "bitdelta":
                bits = pack_bits((diff > 0).to(torch.uint8), 1)
                d.entries[name] = DeltaEntry("bitdelta", tuple(wb.shape), bits=bits, alpha=diff.abs().mean().reshape(()).float())
            else:
                r = min(cfg.rank, min(wb.shape))
                if min(wb.shape) > 4 * r:  # randomized SVD: seed it so the stage is reproducible (Plan: deterministic stages)
                    with torch.random.fork_rng(devices=[diff.device] if diff.device.type == "cuda" else []):
                        torch.manual_seed(int(cfg.seed))
                        U, S, V = torch.svd_lowrank(diff, q=min(r + 8, min(wb.shape)), niter=3)
                else:
                    U, S, V = None, None, None
                if U is None:
                    U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
                    V = Vh.t()
                s = S[:r].sqrt()
                d.entries[name] = DeltaEntry("lowrank", tuple(wb.shape), A=(U[:, :r] * s).float(), B=(V[:, :r] * s).t().float())
        else:
            # "exact" difference: fp16 only when that is lossless (fp16 overflows above 65504, flushes tiny diffs, has 11 mantissa bits)
            d16 = diff.to(torch.float16)
            d.entries[k] = DeltaEntry("full", tuple(wb.shape), diff=d16 if bool(torch.equal(d16.float(), diff)) else diff.float())
    return d


def reconstruct_matrix(e: DeltaEntry) -> torch.Tensor:
    """Dense ``Delta`` of a matrix entry (fp32)."""
    if e.kind == "bitdelta":
        n, k = e.shape
        sgn = unpack_bits(e.bits, 1, k).float() * 2.0 - 1.0
        return e.alpha.float() * sgn
    if e.kind == "lowrank":
        return e.A.float() @ e.B.float()
    return e.diff.float()


# ------------------------------------------------------------------------------------------------ modules
class BitDeltaLinear(nn.Module):
    """``W = W_base + alpha * sign`` with frozen ``W_base`` / sign bits and a trainable scale ``alpha`` (BitDelta calibration)."""

    kind = "bitdelta"

    def __init__(self, base: nn.Linear, bits: torch.Tensor, alpha: torch.Tensor) -> None:
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_buffer("base_weight", base.weight.detach())
        self.register_buffer("sign_bits", bits.to(base.weight.device))
        self.alpha = nn.Parameter(alpha.clone().float().to(base.weight.device))
        self.bias = base.bias

    def sign(self) -> torch.Tensor:
        return unpack_bits(self.sign_bits, 1, self.in_features).float() * 2.0 - 1.0

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.base_weight.float() + self.alpha * self.sign()
        return w.to(dtype) if dtype is not None else w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype), self.bias)


class LowRankDeltaLinear(nn.Module):
    """``y = x W_base^T + (x B^T) A^T``."""

    kind = "lowrank_delta"

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor) -> None:
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_buffer("base_weight", base.weight.detach())
        self.register_buffer("A", A.to(base.weight.device))
        self.register_buffer("B", B.to(base.weight.device))
        self.bias = base.bias

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.base_weight.float() + self.A.float() @ self.B.float()
        return w.to(dtype) if dtype is not None else w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base_weight.to(x.dtype)) + F.linear(F.linear(x, self.B.to(x.dtype)), self.A.to(x.dtype))
        return y + self.bias.to(x.dtype) if self.bias is not None else y


def apply_delta(base_model: nn.Module, delta: Delta, *, inplace: bool = False) -> nn.Module:
    """Model computing ``base + delta``: matrices become delta modules sharing the base weights; full diffs are added to parameters."""
    model = base_model if inplace else copy.deepcopy(base_model)
    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    for name, e in delta.entries.items():
        if e.kind == "full":
            t = params.get(name, bufs.get(name))
            if t is None:
                raise ConfigError(f"delta refers to unknown tensor {name}")
            with torch.no_grad():
                # add in fp32 and round once (casting the diff to a bf16 base dtype first would lose mantissa bits)
                t.copy_((t.float() + e.diff.to(t.device).float()).to(t.dtype))
            continue
        lin = model.get_submodule(name)
        if not isinstance(lin, nn.Linear):
            raise ConfigError(f"{name} is not a dense linear in the base model")
        new = BitDeltaLinear(lin, e.bits, e.alpha) if e.kind == "bitdelta" else LowRankDeltaLinear(lin, e.A, e.B)
        parent_name, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, leaf, new)
    return model


def merge_delta(model: nn.Module) -> nn.Module:
    """Replace delta modules by dense ``nn.Linear`` holding ``base + delta`` (in place)."""
    for name, m in list(model.named_modules()):
        if isinstance(m, (BitDeltaLinear, LowRankDeltaLinear)):
            lin = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None, device=m.base_weight.device, dtype=m.base_weight.dtype)
            with torch.no_grad():
                lin.weight.copy_(m.dequantize(m.base_weight.dtype))
                if m.bias is not None:
                    lin.bias.copy_(m.bias)
            parent_name, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, leaf, lin)
    return model


def calibrate_bitdelta(
    base_model: nn.Module,
    ft_model: nn.Module,
    delta: Delta,
    tokens: torch.Tensor,
    *,
    steps: int = 40,
    lr: float = 0.02,
    batch_size: int = 4,
    ctx: RunContext | None = None,
) -> dict[str, Any]:
    """BitDelta step 2: tune the per-matrix scales ``alpha`` to minimise the logit MSE to the fine-tuned model (signs frozen).
    Updates ``delta`` in place and returns the loss curve."""
    ctx = ctx or NullContext()
    if delta.method != "bitdelta":
        raise ConfigError("scale calibration applies to bitdelta deltas")
    student = apply_delta(base_model, delta)
    dev = model_device(student)
    alphas = [m.alpha for m in student.modules() if isinstance(m, BitDeltaLinear)]
    for p in student.parameters():
        p.requires_grad_(False)
    for a in alphas:
        a.requires_grad_(True)
    # relative step size: Adam moves a parameter by ~lr per step, so give every scale lr proportional to its own magnitude
    opt = torch.optim.Adam([{"params": [a], "lr": lr * float(a.detach().abs())} for a in alphas])
    batches = list(tokens.split(batch_size))
    ft_model.eval()
    student.eval()
    losses = []
    with torch.no_grad():
        targets = [ft_model(b.to(dev)).logits.float() for b in batches]

    def full_loss() -> float:
        with torch.no_grad():
            return sum(float(F.mse_loss(student(b.to(dev)).logits.float(), t)) for b, t in zip(batches, targets)) / len(batches)

    loss_before = full_loss()
    for step in range(steps):
        i = step % len(batches)
        with torch.enable_grad():
            loss = F.mse_loss(student(batches[i].to(dev)).logits.float(), targets[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        ctx.metric(step, bitdelta_loss=losses[-1])
    loss_after = full_loss()
    if loss_after <= loss_before:  # keep the calibrated scales only if they help on the calibration set
        for name, m in student.named_modules():
            if isinstance(m, BitDeltaLinear):
                delta.entries[name].alpha = m.alpha.detach().cpu().clone()
    return {"steps": steps, "loss_before": loss_before, "loss_after": min(loss_after, loss_before), "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None, "losses": losses}


# ------------------------------------------------------------------------------------------------ multi-tenant serving
class MultiTenantLinear(nn.Module):
    """One shared base weight + per-tenant BitDelta deltas; ``forward(x, tenant_ids)`` with ``tenant_ids [B]`` (``-1`` = base model).

    Reference implementation of the batched multi-tenant kernel: groups rows by tenant and adds ``alpha_t * (x sign_t^T)``.
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_buffer("base_weight", base.weight.detach())
        self.bias = base.bias
        self.tenants: list[tuple[torch.Tensor, torch.Tensor]] = []

    def add_tenant(self, bits: torch.Tensor, alpha: torch.Tensor) -> int:
        self.tenants.append((bits.to(self.base_weight.device), alpha.float().to(self.base_weight.device)))
        return len(self.tenants) - 1

    def forward(self, x: torch.Tensor, tenant_ids: torch.Tensor | None = None) -> torch.Tensor:
        y = F.linear(x, self.base_weight.to(x.dtype))
        if tenant_ids is not None:
            for t in torch.unique(tenant_ids).tolist():
                if t < 0:
                    continue
                bits, alpha = self.tenants[t]
                sel = tenant_ids == t
                sgn = unpack_bits(bits, 1, self.in_features).float() * 2.0 - 1.0
                y[sel] = y[sel] + (alpha * F.linear(x[sel].float(), sgn)).to(y.dtype)
        return y + self.bias.to(x.dtype) if self.bias is not None else y


def delta_size_report(delta: Delta, ft_sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    dense = sum(t.numel() * 2 for t in ft_sd.values() if t.is_floating_point())  # bf16/fp16 fine-tune
    return delta.size_report(dense)
