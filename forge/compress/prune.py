"""Pruning (Plan CP-5, CP-6, CP-7): structured width / head / depth / expert pruning with importance estimation, plus research-only
unstructured and N:M sparsity.

Structured pruning returns a **new, physically smaller model** (new :class:`ModelConfig` + sliced weights), because only structure removed
from the dense tensors yields a wall-clock win on GPUs (Plan 7.1).  Because ``ModelConfig`` has one value per field, pruning is *uniform
across layers in count* (every kept layer has the same ``intermediate_size`` / ``num_heads`` / ``num_experts``) while the *indices* kept differ per
layer.  Plan rule: structured pruning is **always followed by a distillation recovery stage** -- the ``prune`` recipe stage sets
``needs_recovery`` and the report carries the parent for use as teacher.

Components
----------
``estimate_importance``
    ``activation``  mean ``|a|`` of the quantity being pruned (MLP hidden activations, per-head attention outputs, normalised residual channels),
    ``fisher``      diagonal empirical Fisher of a unit mask on that quantity: ``sum_batches (dL/dm)^2``,
    ``taylor``      first-order Taylor ``sum_batches |dL/dm|``  (``m`` = 1 gate multiplying the activation; gradients come from the LM loss),
    ``magnitude``   data-free weight norms.
GQA-aware head pruning
    ``group``       remove whole KV groups (a KV head and its ``q_per_kv`` query heads together),
    ``per_group``   keep the ``r`` most important query heads in every group (``num_kv_heads`` unchanged).
Depth
    block influence ``BI_i = 1 - E cos(h_i, h_{i+1})`` (ShortGPT); lowest-BI blocks or the best contiguous window (Minitron) are removed.
Hidden size
    prune residual channels by the summed magnitude of normalised stream activations (embedding, norms, all readers/writers sliced).
Experts (MoE, REAP-style)
    saliency = routed-token frequency x mean gate weight x mean expert output norm (from :class:`~forge.compress.calibrate.RouterStats`);
    ``prune`` drops the least salient experts (router rows too), ``merge`` folds each dropped expert into its most similar kept expert by
    frequency-weighted weight averaging.  Per-layer floor ``max(top_k, expert_min_keep)`` and optional per-domain protection.
Sparsity (research)
    magnitude / Wanda unstructured or N:M masks written into the dense weights, always with an explicit "no GPU speed-up expected" warning.
"""
from __future__ import annotations

import copy
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from forge.compress.modelutil import (
    ChannelStats,
    count_params,
    find_blocks,
    iter_dense_linears,
    model_device,
    model_dtype,
    require_attention_only,
)
from forge.core.errors import ConfigError

log = logging.getLogger("forge.compress.prune")

NO_SPEEDUP_WARNING = (
    "unstructured / N:M sparsity stores dense tensors: NO GPU speed-up is expected unless a 2:4-capable sparse kernel (cuSPARSELt-class, sm80+) "
    "and sparse fine-tuning are used, and generative tasks are validated (Plan 7.1, P2)"
)


class SparsityConfig(BaseModel):
    """Unstructured / N:M sparsity (research option, stage ``prune`` with ``sparsity`` params)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["magnitude", "wanda"] = Field("wanda", description="Score: |W| or |W| * ||x_j||_2 (Wanda).")
    sparsity: float = Field(0.5, ge=0, lt=1, description="Fraction of weights set to zero (unstructured, per output row).")
    pattern: Literal["unstructured", "2:4", "4:8"] = Field("unstructured", description="Sparsity pattern; N:M keeps N of every M consecutive input weights.")


class PruneConfig(BaseModel):
    """Structured pruning settings (stage ``prune``)."""

    model_config = ConfigDict(extra="forbid")

    importance: Literal["activation", "fisher", "taylor", "magnitude"] = Field("activation", description="Importance estimator. fisher/taylor use the LM loss gradient on the calibration set.")
    mlp_keep: float = Field(1.0, gt=0, le=1.0, description="Fraction of dense-MLP intermediate channels (and MoE expert hidden channels) to keep.")
    mlp_round_to: int = Field(8, ge=1, description="Round the kept intermediate size to a multiple of this.")
    head_mode: Literal["none", "group", "per_group"] = Field("none", description="Attention pruning: none, group = drop whole KV groups, per_group = keep r query heads in every group.")
    head_keep: float = Field(1.0, gt=0, le=1.0, description="Fraction of KV groups (group) or of query heads per group (per_group) to keep.")
    hidden_keep: float = Field(1.0, gt=0, le=1.0, description="Fraction of residual-stream channels to keep (embedding width). Aggressive: needs strong distillation recovery.")
    hidden_round_to: int = Field(8, ge=1, description="Round the kept hidden size to a multiple of this.")
    depth_remove: int = Field(0, ge=0, description="Number of transformer blocks to remove.")
    depth_mode: Literal["block_influence", "contiguous", "explicit"] = Field("block_influence", description="How blocks are chosen: lowest block influence, best contiguous window, or explicit layers_to_remove.")
    layers_to_remove: list[int] | None = Field(None, description="Explicit block indices to remove (depth_mode=explicit).")
    protect_ends: bool = Field(True, description="Never remove the first and last block.")
    expert_keep: float = Field(1.0, gt=0, le=1.0, description="Fraction of routed experts to keep per MoE layer.")
    expert_mode: Literal["prune", "merge"] = Field("prune", description="Dropped experts are removed (prune) or merged into the most similar kept expert (merge).")
    expert_min_keep: int = Field(2, ge=1, description="Per-layer floor on the number of experts (never below the router top_k).")
    domain_protect_frac: float = Field(0.0, ge=0, le=1.0, description="Protect, per calibration domain, this fraction of experts with the highest routing frequency in that domain.")
    sparsity: SparsityConfig | None = Field(None, description="Research option: also zero weights (unstructured / N:M). Dense storage -> NO GPU speed-up expected; always emits a warning.")
    n_seqs: int = Field(64, ge=1, description="Calibration sequences for importance estimation.")
    seq_len: int = Field(256, ge=8, description="Calibration sequence length.")
    batch_size: int = Field(4, ge=1, description="Calibration batch size.")
    seed: int = Field(0, description="Calibration sampling seed.")

    def any_structure(self) -> bool:
        return (self.mlp_keep < 1 or self.head_mode != "none" or self.hidden_keep < 1 or self.depth_remove > 0 or self.expert_keep < 1
                or (self.depth_mode == "explicit" and bool(self.layers_to_remove)))


# ------------------------------------------------------------------------------------------------ importance
@dataclass
class Importance:
    """Importance scores. ``mlp[(layer, expert|None)]`` -> ``[I]``; ``heads[layer]`` -> ``[num_heads]``; ``hidden`` -> ``[H]``; ``block_influence`` -> ``[L]``."""

    method: str
    mlp: dict[tuple[int, int | None], torch.Tensor] = field(default_factory=dict)
    heads: dict[int, torch.Tensor] = field(default_factory=dict)
    hidden: torch.Tensor | None = None
    block_influence: torch.Tensor | None = None


def _mlp_down_modules(block: nn.Module) -> list[tuple[int | None, nn.Module]]:
    mlp = block.mlp
    experts = getattr(mlp, "experts", None)
    if isinstance(experts, nn.ModuleList):
        return [(e, ex.down_proj) for e, ex in enumerate(experts)]
    return [(None, mlp.down_proj)]


def _lm_loss(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1))


def estimate_importance(
    model: nn.Module,
    tokens: torch.Tensor,
    method: str = "activation",
    *,
    batch_size: int = 4,
    heads: bool = True,
    mlp: bool = True,
    hidden: bool = True,
    depth: bool = True,
) -> Importance:
    """Importance of MLP channels / attention heads / hidden channels / blocks (see module docstring)."""
    require_attention_only(model, "structured pruning")
    prefix, blocks = find_blocks(model)
    cfg = getattr(model, "config", None)
    dev = model_device(model)
    imp = Importance(method)
    nheads = int(getattr(cfg, "num_heads", 0))
    hd = int(getattr(cfg, "head_dim", 0))
    was_training = model.training
    model.eval()
    handles = []
    acc: dict[Any, torch.Tensor] = {}
    cnt: dict[Any, int] = {}

    def add_abs(key: Any, x: torch.Tensor, reshape_heads: bool = False) -> None:
        xf = x.detach().float().reshape(-1, x.shape[-1])
        if reshape_heads:
            v = xf.reshape(xf.shape[0], nheads, hd).abs().mean(-1).sum(0)  # per-head mean |a| summed over tokens
        else:
            v = xf.abs().sum(0)
        acc[key] = acc.get(key, torch.zeros_like(v)) + v
        cnt[key] = cnt.get(key, 0) + xf.shape[0]

    # ---- hooks common to all estimators
    stream_key = "hidden"
    if depth:
        bi_sum = torch.zeros(len(blocks), dtype=torch.float64)
        for i, blk in enumerate(blocks):
            def bi_hook(_m, inp, out, i=i):
                h_in = inp[0].detach().float()
                h_out = (out[0] if isinstance(out, (tuple, list)) else out).detach().float()
                cos = F.cosine_similarity(h_in.reshape(-1, h_in.shape[-1]), h_out.reshape(-1, h_out.shape[-1]), dim=-1)
                bi_sum[i] += float((1.0 - cos).sum())
                if i == 0:
                    nonlocal_n[0] += cos.numel()

            handles.append(blk.register_forward_hook(bi_hook))
    nonlocal_n = [0]
    if hidden:
        for i, blk in enumerate(blocks):
            for lin in (blk.self_attn.q_proj,):
                handles.append(lin.register_forward_hook(lambda _m, inp, _o: add_abs(stream_key, inp[0])))
    if method in ("activation", "magnitude"):
        if method == "activation":
            for i, blk in enumerate(blocks):
                if mlp:
                    for e, dp in _mlp_down_modules(blk):
                        handles.append(dp.register_forward_hook(lambda _m, inp, _o, k=(i, e): add_abs(("mlp",) + k, inp[0])))
                if heads:
                    handles.append(blk.self_attn.o_proj.register_forward_hook(lambda _m, inp, _o, i=i: add_abs(("head", i), inp[0], True)))
        with torch.no_grad():
            for b in tokens.split(batch_size):
                model(b.to(dev))
    else:  # fisher / taylor: unit masks on the pruned quantities, gradients of the LM loss
        masks: dict[Any, torch.Tensor] = {}
        for i, blk in enumerate(blocks):
            if mlp:
                for e, dp in _mlp_down_modules(blk):
                    m = torch.ones(dp.in_features, device=dev, requires_grad=True)
                    masks[("mlp", i, e)] = m
                    handles.append(dp.register_forward_pre_hook(lambda _mod, inp, m=m: (inp[0] * m.to(inp[0].dtype),)))
            if heads:
                m = torch.ones(nheads, device=dev, requires_grad=True)
                masks[("head", i)] = m
                handles.append(blk.self_attn.o_proj.register_forward_pre_hook(
                    lambda _mod, inp, m=m: ((inp[0].reshape(*inp[0].shape[:-1], nheads, hd) * m.to(inp[0].dtype).unsqueeze(-1)).reshape(inp[0].shape),)
                ))
        stats: dict[Any, torch.Tensor] = {k: torch.zeros_like(m, dtype=torch.float64) for k, m in masks.items()}
        flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        for b in tokens.split(batch_size):
            for m in masks.values():
                m.grad = None
            with torch.enable_grad():
                _lm_loss(model, b.to(dev)).backward()
            for k, m in masks.items():
                if m.grad is None:
                    continue
                g = m.grad.detach().double()
                stats[k] += g.pow(2) if method == "fisher" else g.abs()
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)
        for k, v in stats.items():
            if k[0] == "mlp":
                imp.mlp[(k[1], k[2])] = v.float().cpu()
            else:
                imp.heads[k[1]] = v.float().cpu()
    for h in handles:
        h.remove()
    model.train(was_training)

    # ---- collect
    if method == "activation":
        for k, v in acc.items():
            if k == stream_key:
                continue
            n = cnt[k]
            if k[0] == "mlp":
                imp.mlp[(k[1], k[2])] = (v / n).float().cpu()
            else:
                imp.heads[k[1]] = (v / n).float().cpu()
    if method == "magnitude":
        for i, blk in enumerate(blocks):
            if mlp:
                for e, dp in _mlp_down_modules(blk):
                    imp.mlp[(i, e)] = dp.weight.detach().float().norm(dim=0).cpu()
            if heads:
                w = blk.self_attn.o_proj.weight.detach().float()
                imp.heads[i] = w.reshape(w.shape[0], nheads, hd).norm(dim=(0, 2)).cpu()
    if hidden and stream_key in acc:
        imp.hidden = (acc[stream_key] / cnt[stream_key]).float().cpu()
    elif hidden:
        # magnitude fallback / gradient estimators: accumulate stream magnitude in a cheap extra pass
        h_acc = torch.zeros(int(cfg.hidden_size), dtype=torch.float64)
        n = 0
        hh = [blk.self_attn.q_proj.register_forward_hook(lambda _m, inp, _o: h_acc.add_(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).abs().sum(0).double().cpu()))
              for blk in blocks]
        with torch.no_grad():
            for b in tokens.split(batch_size):
                model(b.to(dev))
                n += b.numel()
        for h in hh:
            h.remove()
        imp.hidden = (h_acc / max(n, 1)).float()
    if depth:
        imp.block_influence = (bi_sum / max(nonlocal_n[0], 1)).float()
    return imp


# ------------------------------------------------------------------------------------------------ planning
@dataclass
class PrunePlan:
    layers_keep: list[int]
    mlp_keep: dict[tuple[int, int | None], list[int]]  # (old layer, expert old idx|None) -> kept intermediate channels
    mlp_size: int | None
    q_heads_keep: dict[int, list[int]]  # old layer -> kept query heads
    kv_heads_keep: dict[int, list[int]]
    hidden_keep: list[int] | None
    experts_keep: dict[int, list[int]]  # old layer -> kept expert ids
    expert_merge: dict[int, dict[int, list[int]]]  # old layer -> kept expert -> merged-in expert ids
    notes: list[str] = field(default_factory=list)


def _topk_sorted(scores: torch.Tensor, k: int) -> list[int]:
    return sorted(torch.topk(scores.float(), k).indices.tolist())


def _round_keep(n: int, frac: float, mult: int) -> int:
    k = max(int(round(n * frac)), 1)
    k = max(int(round(k / mult)) * mult, mult)
    return min(k, n)


def plan_pruning(model: nn.Module, cfg: PruneConfig, imp: Importance | None, router_stats: dict[str, Any] | None = None) -> PrunePlan:
    require_attention_only(model, "structured pruning")
    mc = model.config
    prefix, blocks = find_blocks(model)
    n_layers = len(blocks)
    notes: list[str] = []
    # ---- depth
    remove: set[int] = set()
    if cfg.depth_mode == "explicit" and cfg.layers_to_remove:
        remove = set(int(i) for i in cfg.layers_to_remove)
    elif cfg.depth_remove > 0:
        if imp is None or imp.block_influence is None:
            raise ConfigError("depth pruning needs block-influence estimates (calibration tokens)")
        bi = imp.block_influence.clone()
        lo, hi = (1, n_layers - 1) if cfg.protect_ends else (0, n_layers)
        if cfg.depth_remove > hi - lo:
            raise ConfigError(f"cannot remove {cfg.depth_remove} of {hi - lo} removable blocks")
        if cfg.depth_mode == "contiguous":
            best, best_s = lo, float("inf")
            for s in range(lo, hi - cfg.depth_remove + 1):
                v = float(bi[s : s + cfg.depth_remove].sum())
                if v < best_s:
                    best, best_s = s, v
            remove = set(range(best, best + cfg.depth_remove))
        else:
            order = [i for i in torch.argsort(bi).tolist() if lo <= i < hi]
            remove = set(order[: cfg.depth_remove])
    if len(remove) >= n_layers:
        raise ConfigError("cannot remove every block")
    layers_keep = [i for i in range(n_layers) if i not in remove]

    # ---- MLP width (dense layers and per-expert hidden)
    mlp_keep: dict[tuple[int, int | None], list[int]] = {}
    mlp_size = None
    if cfg.mlp_keep < 1.0:
        sizes = {(i, e): dp.in_features for i in layers_keep for e, dp in _mlp_down_modules(blocks[i])}
        if len(set(sizes.values())) != 1:
            raise ConfigError(f"MLP widths differ between layers/experts ({sorted(set(sizes.values()))}); cannot prune uniformly")
        full = next(iter(sizes.values()))
        mlp_size = _round_keep(full, cfg.mlp_keep, cfg.mlp_round_to)
        for key in sizes:
            sc = imp.mlp[key] if imp is not None and key in imp.mlp else torch.arange(full, dtype=torch.float32)
            mlp_keep[key] = _topk_sorted(sc, mlp_size)
        if mlp_size == full:
            mlp_keep, mlp_size = {}, None
    # ---- heads (GQA aware)
    q_keep: dict[int, list[int]] = {}
    kv_keep: dict[int, list[int]] = {}
    if cfg.head_mode != "none" and cfg.head_keep < 1.0:
        nh, nkv = int(mc.num_heads), int(mc.kv_heads)
        gs = nh // nkv
        if cfg.head_mode == "group":
            kg = max(int(round(nkv * cfg.head_keep)), 1)
        else:
            r = max(int(round(gs * cfg.head_keep)), 1)
        for i in layers_keep:
            hs = imp.heads[i] if imp is not None and i in imp.heads else torch.ones(nh)
            if cfg.head_mode == "group":
                gsc = hs.reshape(nkv, gs).sum(1)
                keep_g = _topk_sorted(gsc, kg)
                kv_keep[i] = keep_g
                q_keep[i] = [g * gs + j for g in keep_g for j in range(gs)]
            else:
                kv_keep[i] = list(range(nkv))
                q_keep[i] = [g * gs + j for g in range(nkv) for j in sorted(torch.topk(hs[g * gs : (g + 1) * gs], r).indices.tolist())]
    # ---- hidden
    hidden_keep = None
    if cfg.hidden_keep < 1.0:
        if imp is None or imp.hidden is None:
            raise ConfigError("hidden pruning needs activation statistics")
        k = _round_keep(int(mc.hidden_size), cfg.hidden_keep, cfg.hidden_round_to)
        hidden_keep = _topk_sorted(imp.hidden, k)
    # ---- experts
    experts_keep: dict[int, list[int]] = {}
    merge: dict[int, dict[int, list[int]]] = {}
    if cfg.expert_keep < 1.0:
        moe = getattr(mc, "moe", None)
        if moe is None:
            raise ConfigError("expert pruning requested for a dense model")
        E = int(moe.num_experts)
        k_exp = min(E, max(int(round(E * cfg.expert_keep)), cfg.expert_min_keep, int(moe.top_k)))
        for i in layers_keep:
            if not isinstance(getattr(blocks[i].mlp, "experts", None), nn.ModuleList):
                continue
            st = (router_stats or {}).get(f"{prefix}.{i}.mlp")
            if st is None:
                notes.append(f"layer {i}: no router statistics, experts ranked by weight norm")
                sal = torch.stack([sum(float(m.weight.norm()) for m in ex.modules() if isinstance(m, nn.Linear)) for ex in blocks[i].mlp.experts])
            else:
                mo = st.mean_out_norm
                sal = st.freq * st.mean_gate_when_routed * (mo if mo is not None else 1.0)
            protected: set[int] = set()
            if cfg.domain_protect_frac > 0 and st is not None and st.domain_counts:
                for d in st.domain_counts:
                    protected |= set(torch.topk(st.domain_freq(d), max(1, int(round(E * cfg.domain_protect_frac)))).indices.tolist())
            order = torch.argsort(sal, descending=True).tolist()
            keep = list(dict.fromkeys(list(protected) + order))[:max(k_exp, 1)] if len(protected) <= k_exp else sorted(protected)[:k_exp]
            keep = sorted(keep)
            experts_keep[i] = keep
            if cfg.expert_mode == "merge":
                merge[i] = _cluster_experts(blocks[i].mlp, keep, st, cfg.seed)
    return PrunePlan(layers_keep, mlp_keep, mlp_size, q_keep, kv_keep, hidden_keep, experts_keep, merge, notes)


def _cluster_experts(moe: nn.Module, keep: list[int], st: Any, seed: int) -> dict[int, list[int]]:
    """Assign every dropped expert to the kept expert with the highest cosine similarity of (random-projected) weights."""
    vecs = []
    g = torch.Generator().manual_seed(seed)
    n = sum(m.weight.numel() for m in moe.experts[0].modules() if isinstance(m, nn.Linear))
    dim = min(n, 8192)
    proj = torch.randn(n, dim, generator=g) / math.sqrt(dim) if n > dim else None
    for ex in moe.experts:
        v = torch.cat([m.weight.detach().float().reshape(-1).cpu() for m in ex.modules() if isinstance(m, nn.Linear)])
        vecs.append(v @ proj if proj is not None else v)
    V = F.normalize(torch.stack(vecs), dim=-1)
    out: dict[int, list[int]] = {k: [] for k in keep}
    for e in range(len(vecs)):
        if e in out:
            continue
        sims = V[e] @ V[keep].t()
        out[keep[int(torch.argmax(sims))]].append(e)
    return out


# ------------------------------------------------------------------------------------------------ application
def _updated_config(cfg: Any, **upd: Any) -> Any:
    if hasattr(cfg, "model_dump"):
        d = cfg.model_dump()
        for k, v in upd.items():
            if k == "moe" and isinstance(v, dict) and isinstance(d.get("moe"), dict):
                d["moe"].update(v)
            else:
                d[k] = v
        return type(cfg).model_validate(d)
    new = copy.deepcopy(cfg)
    for k, v in upd.items():
        setattr(new, k, v)
    return new


def _rows(t: torch.Tensor, idx: list[int], block: int = 1) -> torch.Tensor:
    if block == 1:
        return t[idx]
    return t.reshape(-1, block, *t.shape[1:])[idx].reshape(-1, *t.shape[1:])


def _cols(t: torch.Tensor, idx: list[int], block: int = 1) -> torch.Tensor:
    if block == 1:
        return t[:, idx]
    return t.reshape(t.shape[0], -1, block)[:, idx].reshape(t.shape[0], -1)


def _slice_state_dict(sd: dict[str, torch.Tensor], plan: PrunePlan, cfg_old: Any) -> dict[str, torch.Tensor]:
    D = int(cfg_old.head_dim)
    out: dict[str, torch.Tensor] = {}
    hk = plan.hidden_keep
    layer_re = re.compile(r"^layers\.(\d+)\.(.*)$")
    new_idx = {old: j for j, old in enumerate(plan.layers_keep)}

    def hid(t: torch.Tensor, dim: int) -> torch.Tensor:
        if hk is None:
            return t
        return t[hk] if dim == 0 else t[:, hk]

    for k, v in sd.items():
        m = layer_re.match(k)
        if not m:
            if k == "embed_tokens.weight" or k == "lm_head.weight":
                out[k] = hid(v, 1)
            elif k == "norm.weight":
                out[k] = hid(v, 0)
            else:
                out[k] = v
            continue
        li, rest = int(m.group(1)), m.group(2)
        if li not in new_idx:
            continue
        nk = f"layers.{new_idx[li]}.{rest}"
        t = v
        qh, kvh = plan.q_heads_keep.get(li), plan.kv_heads_keep.get(li)
        if rest.startswith("self_attn.q_proj"):
            if rest.endswith("weight"):
                t = hid(_rows(t, qh, D) if qh is not None else t, 1)
            else:
                t = _rows(t, qh, D) if qh is not None else t
        elif rest.startswith(("self_attn.k_proj", "self_attn.v_proj")):
            if rest.endswith("weight"):
                t = hid(_rows(t, kvh, D) if kvh is not None else t, 1)
            else:
                t = _rows(t, kvh, D) if kvh is not None else t
        elif rest.startswith("self_attn.o_proj"):
            if rest.endswith("weight"):
                t = hid(_cols(t, qh, D) if qh is not None else t, 0)
            else:
                t = hid(t, 0)
        elif rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            t = hid(t, 0)
        elif rest.startswith("mlp."):
            res = _slice_mlp(li, rest, t, plan, hid)
            if res is None:
                continue
            rel, t = res
            nk = f"layers.{new_idx[li]}.{rel}"
        out[nk] = t
    return out


def _slice_mlp(li: int, rest: str, t: torch.Tensor, plan: PrunePlan, hid) -> tuple[str, torch.Tensor] | None:
    """Slice one tensor under ``mlp.``; returns ``(new key relative to the layer, tensor)`` or ``None`` to drop it (pruned expert)."""
    m = re.match(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$", rest)
    if m:
        e, proj, kind = int(m.group(1)), m.group(2), m.group(3)
        ek = plan.experts_keep.get(li)
        if ek is not None and e not in ek:
            return None
        e_new = ek.index(e) if ek is not None else e
        ch = plan.mlp_keep.get((li, e))
        return (f"mlp.experts.{e_new}.{proj}.{kind}", _slice_ffn_tensor(t, proj, kind, ch, hid))
    m = re.match(r"^mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$", rest)
    if m:
        return (rest, _slice_ffn_tensor(t, m.group(1), m.group(2), plan.mlp_keep.get((li, None)), hid))
    if rest == "mlp.gate.weight":
        ek = plan.experts_keep.get(li)
        return (rest, hid(t[ek] if ek is not None else t, 1))
    if rest == "mlp.shared_expert_gate.weight":
        return (rest, hid(t, 1))
    m = re.match(r"^mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.(weight|bias)$", rest)
    if m:
        return (rest, _slice_ffn_tensor(t, m.group(1), m.group(2), None, hid))
    return (rest, t)


def _slice_ffn_tensor(t: torch.Tensor, proj: str, kind: str, ch: list[int] | None, hid) -> torch.Tensor:
    if proj in ("gate_proj", "up_proj"):  # [I, H] / bias [I]
        if kind == "weight":
            return hid(t[ch] if ch is not None else t, 1)
        return t[ch] if ch is not None else t
    if kind == "weight":  # down_proj [H, I]
        return hid(t[:, ch] if ch is not None else t, 0)
    return hid(t, 0)


def _merge_experts(model: nn.Module, plan: PrunePlan, sd: dict[str, torch.Tensor], router_stats: dict[str, Any] | None) -> None:
    """Frequency-weighted merge of dropped experts into their kept targets -- applied to the *unsliced* weights before slicing."""
    prefix, blocks = find_blocks(model)
    for li, groups in plan.expert_merge.items():
        st = (router_stats or {}).get(f"{prefix}.{li}.mlp")
        freq = st.freq if st is not None else torch.ones(len(blocks[li].mlp.experts))
        gate_key = f"layers.{li}.mlp.gate.weight"
        for kept, members in groups.items():
            if not members:
                continue
            allm = [kept] + members
            w = torch.tensor([float(freq[e]) for e in allm]).clamp(min=1e-8)
            w = w / w.sum()
            for name, mod in blocks[li].mlp.experts[kept].named_modules():
                if isinstance(mod, nn.Linear):
                    for pname in ("weight", "bias"):
                        key = f"layers.{li}.mlp.experts.{kept}.{name}.{pname}"
                        if key not in sd:
                            continue
                        acc = sum(w[j] * sd[f"layers.{li}.mlp.experts.{e}.{name}.{pname}"].float() for j, e in enumerate(allm))
                        sd[key] = acc.to(sd[key].dtype)
            if gate_key in sd:
                rows = torch.stack([sd[gate_key][e].float() for e in allm])
                sd[gate_key][kept] = (w.unsqueeze(1) * rows).sum(0).to(sd[gate_key].dtype)


@dataclass
class PruneResult:
    model: nn.Module
    config: Any
    report: dict[str, Any]


def _build_like(model: nn.Module, cfg_new: Any) -> nn.Module:
    dev, dt = model_device(model), model_dtype(model)
    cls = type(model)
    for kw in ({"dtype": dt, "device": dev, "init_weights": False}, {"dtype": dt, "device": dev}, {}):
        try:
            return cls(cfg_new, **kw)
        except TypeError:
            continue
    raise ConfigError(f"cannot construct {cls.__name__} from a ModelConfig")


def prune_model(
    model: nn.Module,
    cfg: PruneConfig | dict | None = None,
    tokens: torch.Tensor | None = None,
    *,
    router_stats: dict[str, Any] | None = None,
    importance: Importance | None = None,
) -> PruneResult:
    """Return a **new** pruned model (the input is untouched), its :class:`ModelConfig` and a report.

    ``tokens``: calibration token tensor ``[n, L]`` (needed for everything except ``importance='magnitude'`` without depth pruning).
    ``router_stats``: ``{moe module name: RouterStats}`` from :func:`forge.compress.calibrate.collect_stats` (``cache.get(..., 'router')``).
    """
    cfg = PruneConfig.model_validate(cfg) if isinstance(cfg, dict) else (cfg or PruneConfig())
    mc = getattr(model, "config", None)
    if mc is None:
        raise ConfigError("prune_model needs model.config (a ModelConfig)")
    if not cfg.any_structure():
        raise ConfigError("PruneConfig prunes nothing (all keep fractions are 1 and depth_remove is 0)")
    require_attention_only(model, "structured pruning")
    prefix, blocks = find_blocks(model)
    need_data = cfg.importance != "magnitude" or cfg.depth_remove > 0 and cfg.depth_mode != "explicit" or cfg.hidden_keep < 1
    imp = importance
    if imp is None and (need_data or cfg.importance != "magnitude"):
        if tokens is None:
            raise ConfigError("importance estimation needs calibration tokens")
        imp = estimate_importance(
            model, tokens, cfg.importance, batch_size=cfg.batch_size,
            heads=cfg.head_mode != "none", mlp=cfg.mlp_keep < 1, hidden=cfg.hidden_keep < 1,
            depth=cfg.depth_remove > 0 and cfg.depth_mode != "explicit",
        )
    elif imp is None:
        imp = estimate_importance(model, torch.zeros(1, 8, dtype=torch.long), "magnitude", hidden=False, depth=False)
    plan = plan_pruning(model, cfg, imp, router_stats)
    # ---- new config
    upd: dict[str, Any] = {"num_layers": len(plan.layers_keep)}
    moe_upd: dict[str, Any] = {}
    if plan.mlp_size is not None:
        if any(k[1] is None for k in plan.mlp_keep):
            upd["intermediate_size"] = plan.mlp_size
        if any(k[1] is not None for k in plan.mlp_keep):
            moe_upd["moe_intermediate_size"] = plan.mlp_size
    if plan.q_heads_keep:
        any_i = next(iter(plan.q_heads_keep))
        upd["num_heads"] = len(plan.q_heads_keep[any_i])
        upd["num_kv_heads"] = len(plan.kv_heads_keep[any_i])
    if plan.hidden_keep is not None:
        upd["hidden_size"] = len(plan.hidden_keep)
    if plan.experts_keep:
        k_e = len(next(iter(plan.experts_keep.values())))
        moe_upd["num_experts"] = k_e
        moe_upd["top_k"] = min(int(mc.moe.top_k), k_e)
    if moe_upd:
        upd["moe"] = moe_upd
    if getattr(mc, "moe", None) is not None and len(plan.layers_keep) != len(blocks):
        mlp_only = [plan.layers_keep.index(i) for i in mc.moe.mlp_only_layers if i in plan.layers_keep]
        upd.setdefault("moe", {})["mlp_only_layers"] = mlp_only
        if mc.moe.decoder_sparse_step != 1:
            raise ConfigError("depth pruning of models with decoder_sparse_step > 1 is not supported")
    cfg_new = _updated_config(mc, **upd)
    # ---- weights
    sd_full = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if plan.expert_merge:
        _merge_experts(model, plan, sd_full, router_stats)
    sliced = _slice_state_dict(sd_full, plan, mc)
    new_model = _build_like(model, cfg_new)
    res = new_model.load_state_dict(sliced, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith("lm_head.weight") or not getattr(cfg_new, "tie_word_embeddings", False)]
    if missing or res.unexpected_keys:
        raise ConfigError(f"pruned state dict does not fit the new model: missing {missing[:5]} unexpected {res.unexpected_keys[:5]}")
    if hasattr(new_model, "tie_weights"):
        new_model.tie_weights()
    new_model.eval()
    before, after = count_params(model), count_params(new_model)
    rep = {
        "params_before": before, "params_after": after, "ratio": after / max(before, 1), "importance": cfg.importance,
        "layers_removed": [i for i in range(len(blocks)) if i not in plan.layers_keep],
        "intermediate_size": plan.mlp_size, "num_heads": upd.get("num_heads"), "num_kv_heads": upd.get("num_kv_heads"),
        "hidden_size": upd.get("hidden_size"), "num_experts": (upd.get("moe") or {}).get("num_experts"),
        "expert_mode": cfg.expert_mode if plan.experts_keep else None,
        "kept": {"layers": plan.layers_keep, "hidden": plan.hidden_keep,
                 "q_heads": {str(k): v for k, v in plan.q_heads_keep.items()}, "kv_heads": {str(k): v for k, v in plan.kv_heads_keep.items()},
                 "experts": {str(k): v for k, v in plan.experts_keep.items()}},
        "needs_recovery": True,
        "recovery_hint": "structured pruning must be followed by a distillation recovery stage with the parent as teacher (Plan CP-6)",
        "notes": plan.notes, "new_config": cfg_new.model_dump(mode="json") if hasattr(cfg_new, "model_dump") else None,
    }
    return PruneResult(new_model, cfg_new, rep)


# ------------------------------------------------------------------------------------------------ sparsity (research)
@torch.no_grad()
def sparsify_model(model: nn.Module, cfg: SparsityConfig | dict | None = None, tokens: torch.Tensor | None = None, *, exclude: list[str] | None = None) -> dict[str, Any]:
    """Zero weights in place (dense storage). Returns a report whose ``warnings`` always contain the no-speed-up notice."""
    cfg = SparsityConfig.model_validate(cfg) if isinstance(cfg, dict) else (cfg or SparsityConfig())
    from forge.compress.modelutil import DEFAULT_EXCLUDE

    lins = dict(iter_dense_linears(model, "", exclude if exclude is not None else DEFAULT_EXCLUDE))
    norms: dict[str, torch.Tensor] = {}
    if cfg.method == "wanda":
        if tokens is None:
            raise ConfigError("wanda needs calibration tokens")
        stats = {n: ChannelStats(m.in_features, model_device(model)) for n, m in lins.items()}
        hs = [m.register_forward_hook(lambda _m, i, _o, n=n: stats[n].add(i[0])) for n, m in lins.items()]
        model.eval()
        for b in tokens.split(4):
            model(b.to(model_device(model)))
        for h in hs:
            h.remove()
        norms = {n: (s.mean_sq * s.n).sqrt().to(model_device(model)) for n, s in stats.items()}
    total = zeros = 0
    for n, m in lins.items():
        w = m.weight
        score = w.detach().abs().float()
        if cfg.method == "wanda" and n in norms:
            score = score * norms[n].unsqueeze(0)
        if cfg.pattern == "unstructured":
            k = int(w.shape[1] * cfg.sparsity)
            if k > 0:
                idx = torch.topk(score, k, dim=1, largest=False).indices
                mask = torch.ones_like(score, dtype=torch.bool)
                mask.scatter_(1, idx, False)
            else:
                mask = torch.ones_like(score, dtype=torch.bool)
        else:
            nkeep, mm = (int(x) for x in cfg.pattern.split(":"))
            if w.shape[1] % mm:
                continue
            sg = score.reshape(w.shape[0], -1, mm)
            idx = torch.topk(sg, nkeep, dim=-1).indices
            mask = torch.zeros_like(sg, dtype=torch.bool).scatter_(-1, idx, True).reshape(w.shape)
        w.mul_(mask.to(w.dtype))
        total += w.numel()
        zeros += int((w == 0).sum())
    return {"method": cfg.method, "pattern": cfg.pattern, "target_sparsity": cfg.sparsity if cfg.pattern == "unstructured" else 1 - int(cfg.pattern.split(":")[0]) / int(cfg.pattern.split(":")[1]),
            "achieved_sparsity": zeros / max(total, 1), "layers": len(lins), "warnings": [NO_SPEEDUP_WARNING], "provenance": "measured"}
