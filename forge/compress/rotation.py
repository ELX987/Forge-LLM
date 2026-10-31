"""Orthogonal rotations fused into adjacent linears (QuaRot / SpinQuant style; Plan CP-8).

Applied *before* quantisation, only where the planner's per-format model says it helps (Plan: P5 found random Hadamard does **not** reduce RTN
error for NVFP4 but does reduce SR-product variance; for INT4/W8A8 outlier-taming it usually helps).  All transforms are exact in infinite
precision -- the unit tests check logits equivalence of the rotated model.

R1 (residual stream, ``hidden x hidden`` orthogonal ``Q``)
    1. RMSNorm weights are folded into the following linears (norm becomes weightless, hence rotation invariant),
    2. embedding rows ``E <- E Q``; every reader of the stream (q/k/v, gate/up, router, shared experts, lm_head) ``W <- W Q``;
       every writer (o_proj, down_proj) ``W <- Q^T W`` (and bias ``b <- b Q``).
    ``Q`` is a randomised Hadamard (``H diag(s)``; for sizes that are not a power of two: block Hadamard of the largest power-of-two divisor
    mixed by a random permutation) or a *learned* orthogonal matrix (Cayley parametrisation initialised at the Hadamard, optimised so that the
    rotated weights/activations are easier to quantise; a local objective, not SpinQuant's end-to-end loss).
R2 (per head, ``head_dim``): value rows ``W_v <- Q2^T W_v`` per KV head and ``W_o <- W_o Q2`` per query head (exact, free at run time).
R4 (optional, online): a block Hadamard before ``down_proj`` (``OnlineHadamardLinear``); PTQ keeps it as ``rot_block`` of the packed module.

Tied embeddings: the output head is untied (cloned) because the final norm must be folded into it.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from forge.compress.modelutil import find_blocks, is_dense_linear
from forge.compress.packed import apply_block_hadamard, hadamard_matrix
from forge.compress.serialize import register_module_kind
from forge.core.errors import ConfigError

log = logging.getLogger("forge.compress.rotation")


class RotationConfig(BaseModel):
    """Rotation settings (stage ``rotate``)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["hadamard", "learned"] = Field("hadamard", description="Residual-stream rotation: randomised Hadamard, or learned orthogonal (Cayley, initialised at the Hadamard).")
    seed: int = Field(0, description="Seed for the random signs / permutation.")
    r2: bool = Field(True, description="Also rotate each attention head's value/output subspace (exact, no run-time cost).")
    online_hadamard_down: bool = Field(False, description="Insert an online block-Hadamard before every down_proj (run-time cost; helps INT4/FP8 outliers).")
    online_block: int = Field(128, ge=2, description="Maximum block size of the online Hadamard (largest power-of-two divisor of the intermediate size, capped).")
    untie_embeddings: bool = Field(True, description="Clone the LM head if it is tied to the embeddings (required to fold the final norm).")
    learn_steps: int = Field(100, ge=1, description="Learned mode: optimisation steps.")
    learn_lr: float = Field(0.01, gt=0, description="Learned mode: Adam learning rate on the skew-symmetric parameter.")
    learn_bits: int = Field(4, ge=2, le=8, description="Learned mode: bit width of the quantiser inside the objective.")
    learn_group: int = Field(64, description="Learned mode: group size of the quantiser inside the objective.")
    learn_layers: int = Field(4, ge=1, description="Learned mode: number of layers whose weights enter the objective.")
    learn_act_weight: float = Field(0.5, ge=0, description="Learned mode: weight of the activation-quantisation term (needs calibration tokens).")


# ------------------------------------------------------------------------------------------------ matrices
def largest_pow2_divisor(n: int, cap: int | None = None) -> int:
    p = n & -n
    return min(p, cap) if cap else p


def random_orthogonal_hadamard(n: int, seed: int = 0, device: Any = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Randomised orthonormal Hadamard-structured matrix of size ``n`` (see module docstring for non power-of-two sizes)."""
    g = torch.Generator().manual_seed(int(seed))
    signs = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float64)
    p = largest_pow2_divisor(n)
    if p == 1:
        raise ConfigError(f"no Hadamard structure for odd size {n}; use mode='learned' with a random orthogonal init")
    h = hadamard_matrix(p, dtype=torch.float64)
    blocks = torch.block_diag(*([h] * (n // p)))
    q = blocks * signs.unsqueeze(0)  # H diag(s): columns signed
    if n // p > 1:
        perm = torch.randperm(n, generator=g)
        q = q[perm]
    return q.to(device=device, dtype=dtype)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last dim (power of two), orthonormal. O(n log n); matches ``x @ hadamard_matrix(n)``."""
    n = x.shape[-1]
    if n & (n - 1):
        raise ConfigError("fwht needs a power-of-two length")
    y = x.reshape(-1, n).float()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack([a + b, a - b], dim=2).reshape(-1, n)
        h *= 2
    return (y / n**0.5).reshape(x.shape).to(x.dtype)


def cayley(a: torch.Tensor) -> torch.Tensor:
    """Orthogonal matrix ``(I - S)^-1 (I + S)`` with ``S = a - a^T`` (skew-symmetric)."""
    s = a - a.t()
    eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    return torch.linalg.solve(eye - s, eye + s)


@register_module_kind("online_hadamard")
class OnlineHadamardLinear(nn.Linear):
    """Dense linear whose weight already contains ``blockdiag(H)`` on the input side; applies the block Hadamard to ``x`` first."""

    online_rot_block: int = 0

    def quant_spec(self) -> dict[str, Any]:
        return {"block": self.online_rot_block}

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, block: int = 2, **_: Any) -> OnlineHadamardLinear:
        m = cls(in_features, out_features, bias=bias, dtype=dtype, device=device)
        m.online_rot_block = int(block)
        return m

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = apply_block_hadamard(self.weight.detach().float(), self.online_rot_block)  # back to the original input basis
        return w.to(dtype) if dtype is not None else w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(apply_block_hadamard(x.float(), self.online_rot_block).to(x.dtype), self.weight, self.bias)

    @classmethod
    def wrap(cls, lin: nn.Linear, block: int) -> OnlineHadamardLinear:
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_features, m.out_features = lin.in_features, lin.out_features
        w = apply_block_hadamard(lin.weight.detach().float(), block).to(lin.weight.dtype)
        m.weight = nn.Parameter(w, requires_grad=lin.weight.requires_grad)
        m.bias = lin.bias
        m.online_rot_block = int(block)
        return m


# ------------------------------------------------------------------------------------------------ structure discovery
def _need(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            raise ConfigError(f"rotation: model has no '{path}' (expects HF/ForgeLM-style names)")
        cur = getattr(cur, part)
    return cur


def _roles(model: nn.Module) -> dict[str, Any]:
    prefix, blocks = find_blocks(model)
    return {"prefix": prefix, "blocks": blocks, "embed": _need(model, "embed_tokens"), "norm": _need(model, "norm"), "head": _need(model, "lm_head")}


def _is_norm(m: nn.Module) -> bool:
    return "norm" in type(m).__name__.lower() and getattr(m, "weight", None) is not None and m.weight.ndim == 1


def _norm_scale(norm: nn.Module) -> torch.Tensor:
    w = norm.weight.detach().double()
    return 1.0 + w if getattr(norm, "unit_offset", False) else w


def _reset_norm(norm: nn.Module) -> None:
    with torch.no_grad():
        norm.weight.fill_(0.0 if getattr(norm, "unit_offset", False) else 1.0)


def _scale_cols(lin: nn.Module, s: torch.Tensor) -> None:
    if not is_dense_linear(lin):
        raise ConfigError("rotation must run on a dense (unquantised) model")
    with torch.no_grad():
        lin.weight.copy_((lin.weight.double() * s.unsqueeze(0)).to(lin.weight.dtype))


def _stream_readers(block: nn.Module, which: str) -> list[nn.Linear]:
    if which == "attn":
        a = block.self_attn
        return [a.q_proj, a.k_proj, a.v_proj]
    return [m for n, m in block.mlp.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] != "down_proj"]


def _stream_writers(block: nn.Module) -> list[nn.Linear]:
    out = [block.self_attn.o_proj]
    out += [m for n, m in block.mlp.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] == "down_proj"]
    return out


def fuse_norms(model: nn.Module, untie: bool = True) -> dict[str, Any]:
    """Fold every RMSNorm weight into its consumers and reset the norm weights to identity (exact)."""
    r = _roles(model)
    cfg = getattr(model, "config", None)
    if getattr(cfg, "tie_word_embeddings", False) or r["head"].weight is r["embed"].weight:
        if not untie:
            raise ConfigError("tied embeddings: set untie_embeddings=True (the final norm must be folded into the LM head)")
        r["head"].weight = nn.Parameter(r["head"].weight.detach().clone())
        if cfg is not None and hasattr(cfg, "tie_word_embeddings"):
            try:
                cfg.tie_word_embeddings = False
            except Exception:  # frozen/validated config -> leave, model.tie_weights() would re-tie: warn
                log.warning("could not set config.tie_word_embeddings=False; do not call tie_weights() on the rotated model")
    n = 0
    for blk in r["blocks"]:
        for norm_name, which in (("input_layernorm", "attn"), ("post_attention_layernorm", "mlp")):
            norm = _need(blk, norm_name)
            s = _norm_scale(norm)
            for lin in _stream_readers(blk, which):
                _scale_cols(lin, s)
            _reset_norm(norm)
            n += 1
    s = _norm_scale(r["norm"])
    _scale_cols(r["head"], s)
    _reset_norm(r["norm"])
    return {"norms_folded": n + 1}


# ------------------------------------------------------------------------------------------------ fusion
def _right(lin: nn.Module, q: torch.Tensor) -> None:  # W <- W Q  (reader)
    with torch.no_grad():
        lin.weight.copy_((lin.weight.double() @ q.double()).to(lin.weight.dtype))


def _left_t(lin: nn.Module, q: torch.Tensor) -> None:  # W <- Q^T W, b <- b Q  (writer)
    with torch.no_grad():
        lin.weight.copy_((q.double().t() @ lin.weight.double()).to(lin.weight.dtype))
        if lin.bias is not None:
            lin.bias.copy_((lin.bias.double() @ q.double()).to(lin.bias.dtype))


def apply_r1(model: nn.Module, q: torch.Tensor) -> None:
    """Rotate the residual stream by orthogonal ``q [hidden, hidden]`` (norms must already be fused)."""
    r = _roles(model)
    h = r["embed"].weight.shape[1]
    if q.shape != (h, h):
        raise ConfigError(f"rotation matrix {tuple(q.shape)} does not match hidden size {h}")
    with torch.no_grad():
        r["embed"].weight.copy_((r["embed"].weight.double() @ q.double()).to(r["embed"].weight.dtype))
    for blk in r["blocks"]:
        for lin in _stream_readers(blk, "attn") + _stream_readers(blk, "mlp"):
            _right(lin, q)
        for lin in _stream_writers(blk):
            _left_t(lin, q)
    _right(r["head"], q)


def apply_r2(model: nn.Module, seed: int = 0) -> None:
    """Per-head rotation of the value / output-projection subspace (exact)."""
    r = _roles(model)
    for bi, blk in enumerate(r["blocks"]):
        a = blk.self_attn
        d = int(getattr(a, "head_dim"))
        v, o = a.v_proj, a.o_proj
        q2 = random_orthogonal_hadamard(d, seed + 7919 * (bi + 1), device=v.weight.device, dtype=torch.float64)
        with torch.no_grad():
            wv = v.weight.double().reshape(-1, d, v.weight.shape[1])  # [kvh, D, H]
            v.weight.copy_((q2.t().unsqueeze(0) @ wv).reshape(v.weight.shape).to(v.weight.dtype))
            if v.bias is not None:
                v.bias.copy_((v.bias.double().reshape(-1, d) @ q2).reshape(-1).to(v.bias.dtype))
            wo = o.weight.double().reshape(o.weight.shape[0], -1, d)  # [H, qh, D]
            o.weight.copy_((wo @ q2).reshape(o.weight.shape).to(o.weight.dtype))


def add_online_hadamard(model: nn.Module, max_block: int = 128) -> int:
    """Wrap every ``down_proj`` with :class:`OnlineHadamardLinear`. Returns the number of wrapped layers."""
    n = 0
    for name, m in list(model.named_modules()):
        if name.split(".")[-1] == "down_proj" and is_dense_linear(m) and not isinstance(m, OnlineHadamardLinear):
            b = largest_pow2_divisor(m.in_features, max_block)
            if b < 2:
                continue
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, OnlineHadamardLinear.wrap(m, b))
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ learned rotation
def _fq_int_ste(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    n, k = w.shape
    g = group if group > 0 and k % group == 0 else k
    wg = w.reshape(n, k // g, g)
    s = (wg.detach().abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)).clamp(min=1e-8)
    return (torch.round(wg.detach() / s).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s).reshape(n, k)


def learn_rotation(
    weights: list[torch.Tensor],
    q0: torch.Tensor,
    *,
    acts: torch.Tensor | None = None,
    steps: int = 100,
    lr: float = 0.01,
    bits: int = 4,
    group: int = 64,
    act_weight: float = 0.5,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimise ``Q = Q0 cayley(A)`` so that ``W Q`` (and ``X Q``) quantise with less error.

    Objective: ``sum_W ||W Q - sg(fq(W Q))||^2 / ||W||^2  +  act_weight * ||X Q - sg(fq_tok(X Q))||^2 / ||X||^2`` where ``sg`` stops gradients
    (rounding is treated as locally constant) and ``fq_tok`` is per-token symmetric quantisation of the activations.  Returns
    ``(Q, {"before": obj0, "after": obj_final})`` with ``Q`` exactly orthogonal up to solver precision.
    """
    torch.manual_seed(seed)
    dt = torch.float32
    q0 = q0.to(dt)
    ws = [w.to(dt) for w in weights]
    x = None if acts is None else acts.to(dt)
    n = q0.shape[0]
    a = torch.zeros(n, n, dtype=dt, device=q0.device, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)

    def objective(q: torch.Tensor) -> torch.Tensor:
        tot = q.new_zeros(())
        for w in ws:
            wr = w @ q
            tot = tot + (wr - _fq_int_ste(wr, bits, group)).pow(2).sum() / w.pow(2).sum().clamp(min=1e-12)
        if x is not None and act_weight > 0:
            xr = x @ q
            s = (xr.detach().abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)).clamp(min=1e-8)
            xq = torch.round(xr.detach() / s).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s
            tot = tot + act_weight * (xr - xq).pow(2).sum() / x.pow(2).sum().clamp(min=1e-12)
        return tot

    with torch.no_grad():
        before = float(objective(q0))
    best, best_q = before, q0
    for _ in range(steps):
        opt.zero_grad()
        q = q0 @ cayley(a)
        loss = objective(q)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        if v < best:
            best, best_q = v, q.detach().clone()
    with torch.no_grad():
        # re-orthogonalise (QR with positive diagonal of R) to remove accumulated solver error
        qq, rr = torch.linalg.qr(best_q.double())
        final = (qq * torch.sign(torch.diagonal(rr)).unsqueeze(0)).to(dt)
        after = float(objective(final))
    return final, {"before": before, "after": after}


# ------------------------------------------------------------------------------------------------ driver
@torch.no_grad()
def rotate_model(model: nn.Module, cfg: RotationConfig | dict | None = None, tokens: torch.Tensor | None = None) -> dict[str, Any]:
    """Fuse norms and rotate ``model`` in place (R1 [+ R2] [+ online R4]). Returns a report dict. ``tokens`` feed the learned objective."""
    cfg = RotationConfig.model_validate(cfg) if isinstance(cfg, dict) else (cfg or RotationConfig())
    report: dict[str, Any] = {"mode": cfg.mode, "r2": cfg.r2, "online_hadamard_down": cfg.online_hadamard_down}
    report.update(fuse_norms(model, cfg.untie_embeddings))
    r = _roles(model)
    h = r["embed"].weight.shape[1]
    q = random_orthogonal_hadamard(h, cfg.seed, device=r["embed"].weight.device, dtype=torch.float32)
    if cfg.mode == "learned":
        ws = []
        for blk in list(r["blocks"])[: cfg.learn_layers]:
            ws += [lin.weight.detach().float() for lin in _stream_readers(blk, "attn") + _stream_readers(blk, "mlp") if lin.in_features == h]
        acts = None
        if tokens is not None and cfg.learn_act_weight > 0:
            acts = _residual_samples(model, tokens)
        with torch.enable_grad():
            q, info = learn_rotation(ws, q, acts=acts, steps=cfg.learn_steps, lr=cfg.learn_lr, bits=cfg.learn_bits, group=cfg.learn_group,
                                     act_weight=cfg.learn_act_weight, seed=cfg.seed)
        report["objective"] = info
    apply_r1(model, q)
    if cfg.r2:
        apply_r2(model, cfg.seed)
    if cfg.online_hadamard_down:
        report["online_hadamard_layers"] = add_online_hadamard(model, cfg.online_block)
    report["orthogonality_error"] = float((q.double().t() @ q.double() - torch.eye(h, dtype=torch.float64, device=q.device)).abs().max())
    return report


def _residual_samples(model: nn.Module, tokens: torch.Tensor, max_rows: int = 1024) -> torch.Tensor:
    """Inputs of the first block's q_proj (= normalised residual stream) as activation samples."""
    prefix, blocks = find_blocks(model)
    rows: list[torch.Tensor] = []
    h = blocks[0].self_attn.q_proj.register_forward_hook(lambda _m, i, _o: rows.append(i[0].detach().reshape(-1, i[0].shape[-1]).float()))
    try:
        dev = next(model.parameters()).device
        model(tokens[:4].to(dev))
    finally:
        h.remove()
    x = torch.cat(rows, 0)
    return x[:: max(1, x.shape[0] // max_rows)][:max_rows]
