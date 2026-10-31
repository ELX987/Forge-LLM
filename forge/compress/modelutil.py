"""Model plumbing shared by the compression algorithms: block discovery, activation capture, hooks, fingerprints.

Works on ``forge.models.ForgeLM`` (``layers.<i>.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj | mlp.experts.<j>.*}``) and on any
module tree that has a ``layers`` / ``model.layers`` / ``transformer.h`` ``ModuleList`` of blocks whose first positional argument is
the hidden state and whose output is a tensor or a tuple with the hidden state first.

Block-wise algorithms (GPTQ, AWQ, SmoothQuant, whitened SVD, pruning importance, sensitivity) all follow the same loop:

1. :func:`capture_block_inputs` runs the embedding part of the model once per calibration batch and records the exact
   ``(args, kwargs)`` the first block receives (rotary tables, attention context, ...),
2. every block is then replayed on the stored hidden states (:meth:`BlockInputs.run`), so a block can be modified (quantised)
   and the *modified* block's outputs feed the next block ("sequential" calibration, as in GPTQ).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError, ForgeError
from forge.core.hashing import hash_json

#: modules never quantised/pruned by default: output head, MoE routers / gates (Plan: routers stay high precision)
DEFAULT_EXCLUDE: tuple[str, ...] = (r"(^|\.)lm_head$", r"(^|\.)(gate|router)$", r"shared_expert_gate$", r"embed")


class _StopForward(Exception):
    """Raised by the capture hook to abort the forward pass after the first block's inputs were recorded."""


# ------------------------------------------------------------------------------------------------ linear helpers
def _packed_base() -> type | None:
    try:
        from forge.models.linear import PackedLinear

        return PackedLinear
    except Exception:  # pragma: no cover
        return None


def is_dense_linear(m: nn.Module) -> bool:
    """A plain (unquantised) linear: any ``nn.Linear`` (``DenseLinear`` and fake-quant linears included)."""
    return isinstance(m, nn.Linear)


def is_packed_linear(m: nn.Module) -> bool:
    """A weight-compressed linear (``forge.models.linear.PackedLinear`` or any module exposing ``quant_spec`` + ``dequantize``)."""
    pb = _packed_base()
    return (pb is not None and isinstance(m, pb)) or (hasattr(m, "quant_spec") and hasattr(m, "dequantize") and not isinstance(m, nn.Linear))


def excluded(name: str, exclude: Iterable[str] = DEFAULT_EXCLUDE, include: Iterable[str] | None = None) -> bool:
    for pat in exclude:
        if re.search(pat, name):
            return True
    if include:
        return not any(re.search(p, name) for p in include)
    return False


def iter_dense_linears(
    module: nn.Module, prefix: str = "", exclude: Iterable[str] = DEFAULT_EXCLUDE, include: Iterable[str] | None = None
) -> Iterator[tuple[str, nn.Linear]]:
    exclude = tuple(exclude)
    for name, m in module.named_modules():
        if m is module:
            continue
        full = f"{prefix}.{name}" if prefix else name
        if is_dense_linear(m) and not excluded(full, exclude, include):
            yield full, m  # type: ignore[misc]


def iter_packed_linears(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, m in module.named_modules():
        if m is not module and is_packed_linear(m):
            yield name, m


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    return model.get_submodule(name) if name else model


def set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf, new)


def linear_weight(m: nn.Module, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dense ``[out, in]`` weight of any linear-like module."""
    if is_packed_linear(m):
        return m.dequantize(dtype)  # type: ignore[attr-defined]
    w = m.weight.detach()
    return w.to(dtype) if dtype is not None else w


def model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def model_dtype(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


# ------------------------------------------------------------------------------------------------ blocks
def find_blocks(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """``(qualified name, ModuleList)`` of the transformer blocks."""
    for path in ("layers", "model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return path, obj
    raise ConfigError("could not find the transformer blocks (expected model.layers / model.model.layers / model.transformer.h)")


def model_config(model: nn.Module) -> Any:
    return getattr(model, "config", None)


def gemma4_features(model: nn.Module) -> list[str]:
    """Names of the Gemma 4 style block features the model config enables (empty for every other model)."""
    cfg = getattr(model, "config", None)
    return [n for n, on in (
        ("arch gemma4", getattr(cfg, "arch", None) == "gemma4"),
        ("sandwich_norm", bool(getattr(cfg, "sandwich_norm", False))),
        ("num_kv_shared_layers", bool(getattr(cfg, "num_kv_shared_layers", 0))),
        ("per_layer_input_size", bool(getattr(cfg, "per_layer_input_size", 0))),
        ("non-uniform layer head_dim", _non_uniform_head_dim(cfg)),
    ) if on]


def require_attention_only(model: nn.Module, what: str) -> None:
    """Refuse hybrid (lfm2 conv + attention) models for algorithms that assume ``self_attn`` in every block (pruning, rotation)."""
    cfg = getattr(model, "config", None)
    if getattr(cfg, "is_hybrid", False):
        raise ConfigError(
            f"{what} is not supported for hybrid conv/attention models ({getattr(cfg, 'arch', 'lfm2')}): it assumes attention in every layer "
            "and would have to slice the depthwise conv consistently. Quantisation (PTQ/GPTQ/AWQ) and LoRA work."
        )
    gemma4 = gemma4_features(model)
    if gemma4:
        raise ConfigError(
            f"{what} is not supported for Gemma 4 style models ({', '.join(gemma4)}): it assumes uniform attention blocks and would "
            "mishandle sandwich norms, per-layer embeddings, KV sharing and per-layer head dims. Weight-only quantisation (PTQ) and LoRA work."
        )


def _non_uniform_head_dim(cfg: Any) -> bool:
    fn = getattr(cfg, "layer_head_dim", None)
    n = getattr(cfg, "num_layers", 0)
    if not callable(fn) or not n:
        return False
    try:
        return len({int(fn(i)) for i in range(int(n))}) > 1
    except Exception:  # noqa: BLE001
        return False


@dataclass
class BlockInputs:
    """Recorded inputs of the first transformer block for each calibration batch (see module docstring)."""

    hidden: list[torch.Tensor]
    args: list[tuple]
    kwargs: list[dict[str, Any]]
    hidden_in_kwargs: str | None = None
    offload: bool = False
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __len__(self) -> int:
        return len(self.hidden)

    @property
    def n_tokens(self) -> int:
        return int(sum(h.shape[0] * h.shape[1] for h in self.hidden))

    def run(self, block: nn.Module, i: int, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Run ``block`` on batch ``i`` (optionally with a different hidden state) and return its hidden-state output."""
        h = (self.hidden[i] if hidden is None else hidden).to(self.device)
        args, kwargs = self.args[i], self.kwargs[i]
        if self.hidden_in_kwargs is not None:
            out = block(*args, **{**kwargs, self.hidden_in_kwargs: h})
        else:
            out = block(h, *args[1:], **kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def set_hidden(self, i: int, h: torch.Tensor) -> None:
        self.hidden[i] = h.cpu() if self.offload else h

    def propagate(self, block: nn.Module) -> None:
        """Replace every stored hidden state by the output of ``block`` (inputs of the next block)."""
        with torch.no_grad():
            for i in range(len(self)):
                self.set_hidden(i, self.run(block, i))


def capture_block_inputs(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int = 4,
    offload: bool = False,
    forward: Callable[[nn.Module, torch.Tensor], Any] | None = None,
) -> BlockInputs:
    """Record the inputs of block 0 for ``tokens [N, L]`` split into batches of ``batch_size``."""
    _, blocks = find_blocks(model)
    dev = model_device(model)
    store: dict[str, Any] = {}

    def pre_hook(_m: nn.Module, args: tuple, kwargs: dict) -> None:
        store["args"], store["kwargs"] = args, kwargs
        raise _StopForward

    handle = blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    hs: list[torch.Tensor] = []
    all_args: list[tuple] = []
    all_kwargs: list[dict] = []
    hidden_kw: str | None = None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for b in tokens.split(batch_size):
                ids = b.to(dev)
                try:
                    (forward or (lambda m, x: m(x)))(model, ids)
                except _StopForward:
                    pass
                if "args" not in store:
                    raise ForgeError("block-0 pre-hook never fired; model forward does not call model.layers[0]")
                args, kwargs = store.pop("args"), store.pop("kwargs")
                if args:
                    h = args[0]
                else:
                    hidden_kw = next((k for k in ("hidden_states", "h", "x") if k in kwargs), None)
                    if hidden_kw is None:
                        raise ForgeError("cannot locate the hidden state among block-0 call arguments")
                    h = kwargs[hidden_kw]
                hs.append(h.detach().cpu() if offload else h.detach())
                all_args.append(tuple(a.detach() if torch.is_tensor(a) else a for a in args))
                all_kwargs.append({k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()})
    finally:
        handle.remove()
        model.train(was_training)
    return BlockInputs(hs, all_args, all_kwargs, hidden_kw, offload, dev)


# ------------------------------------------------------------------------------------------------ statistics hooks
class HessianAccumulator:
    """Running ``H = 2/n * sum x x^T`` (GPTQ normalisation) for inputs ``x [..., K]``, kept in float32."""

    def __init__(self, k: int, device: torch.device | str = "cpu") -> None:
        self.k = k
        self.H = torch.zeros(k, k, dtype=torch.float32, device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, self.k).float()
        t = x.shape[0]
        if t == 0:
            return
        self.H *= self.n / (self.n + t)
        self.n += t
        x = math.sqrt(2.0 / self.n) * x
        self.H += x.t() @ x


class ChannelStats:
    """Per-input-channel running ``mean|x|``, ``max|x|`` and ``mean x^2`` (for AWQ / SmoothQuant / Wanda / whitening diag)."""

    def __init__(self, k: int, device: torch.device | str = "cpu") -> None:
        self.k = k
        self.abs_sum = torch.zeros(k, dtype=torch.float64, device=device)
        self.sq_sum = torch.zeros(k, dtype=torch.float64, device=device)
        self.abs_max = torch.zeros(k, dtype=torch.float32, device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, self.k).float()
        self.abs_sum += x.abs().sum(0).double()
        self.sq_sum += x.pow(2).sum(0).double()
        self.abs_max = torch.maximum(self.abs_max, x.abs().amax(0))
        self.n += x.shape[0]

    @property
    def mean_abs(self) -> torch.Tensor:
        return (self.abs_sum / max(self.n, 1)).float()

    @property
    def mean_sq(self) -> torch.Tensor:
        return (self.sq_sum / max(self.n, 1)).float()


class SampleBuffer:
    """Reservoir-free bounded buffer of activation rows (first ``max_rows`` rows seen) for output-error searches."""

    def __init__(self, k: int, max_rows: int = 2048) -> None:
        self.k, self.max_rows = k, max_rows
        self.rows: list[torch.Tensor] = []
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        if self.n >= self.max_rows:
            return
        x = x.detach().reshape(-1, self.k)
        take = min(self.max_rows - self.n, x.shape[0])
        # stride through the batch so one long sequence does not dominate
        idx = torch.linspace(0, x.shape[0] - 1, take, device=x.device).long()
        self.rows.append(x[idx].float().cpu())
        self.n += take

    def get(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.cat(self.rows, 0).to(device) if self.rows else torch.zeros(0, self.k, device=device)


def run_with_hooks(
    block: nn.Module,
    binp: BlockInputs,
    callbacks: dict[str, tuple[nn.Module, Callable[[torch.Tensor, torch.Tensor], None]]],
    *,
    collect_outputs: bool = False,
) -> list[torch.Tensor] | None:
    """Replay ``block`` on all stored batches with ``callbacks[name] = (module, fn(x_in, y_out))`` forward hooks attached.

    Returns the block outputs per batch when ``collect_outputs`` (they are *not* stored into ``binp``).
    """
    handles = []
    for _name, (mod, fn) in callbacks.items():

        def hook(_m: nn.Module, inp: tuple, out: Any, fn=fn) -> None:
            x = inp[0] if inp else None
            if x is not None:
                fn(x, out)

        handles.append(mod.register_forward_hook(hook))
    outs: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for i in range(len(binp)):
                o = binp.run(block, i)
                if collect_outputs:
                    outs.append(o.cpu() if binp.offload else o)
    finally:
        for h in handles:
            h.remove()
    return outs if collect_outputs else None


def discover_input_groups(block: nn.Module, binp: BlockInputs, linears: dict[str, nn.Module]) -> list[list[str]]:
    """Probe one batch: group linears (in execution order) that receive the *same input tensor* (q/k/v, gate/up, ...).

    Linears not reached by the probe batch (e.g. experts that got no tokens) are returned as singleton groups at the end.
    """
    order: list[tuple[str, int, tuple]] = []
    # Every probed input is kept alive until the probe ends: otherwise the allocator may hand the address of a freed input to the next,
    # unrelated tensor and two independent linears would be grouped (sharing one Hessian / one AWQ scale) -- silent mis-quantisation.
    keepalive: list[torch.Tensor] = []
    handles = []
    for name, m in linears.items():

        def pre(_m: nn.Module, inp: tuple, name=name) -> None:
            x = inp[0]
            if all(o[0] != name for o in order):
                keepalive.append(x)
                order.append((name, x.data_ptr(), tuple(x.shape)))

        handles.append(m.register_forward_pre_hook(pre))
    try:
        with torch.no_grad():
            binp.run(block, 0)
    finally:
        for h in handles:
            h.remove()
    groups: list[list[str]] = []
    last_key = None
    for name, ptr, shape in order:
        key = (ptr, shape)
        if groups and key == last_key:
            groups[-1].append(name)
        else:
            groups.append([name])
        last_key = key
    seen = {n for g in groups for n in g}
    groups += [[n] for n in linears if n not in seen]
    return groups


def sequential_stages(groups: list[list[str]], mode: str = "auto") -> list[list[list[str]]]:
    """Arrange input-groups into sequential stages. ``mode``: ``full`` (one stage per input group), ``block`` (a single
    stage for the whole block), ``auto`` (like ``full`` but all MoE expert linears collapse into two stages: gate/up, down)."""
    if mode == "block":
        return [groups]
    if mode == "full":
        return [[g] for g in groups]
    stages: list[list[list[str]]] = []
    exp_in: list[list[str]] = []
    exp_down: list[list[str]] = []
    for g in groups:
        if ".experts." in g[0] or g[0].startswith("experts."):
            (exp_down if re.search(r"(down|w2)", g[0].split(".")[-1]) else exp_in).append(g)
        else:
            stages.append([g])
    if exp_in:
        stages.append(exp_in)
    if exp_down:
        stages.append(exp_down)
    return stages


# ------------------------------------------------------------------------------------------------ fingerprints
def model_fingerprint(model: nn.Module, sample: int = 64) -> str:
    """Cheap deterministic digest of a model's parameters/buffers: names, shapes, dtypes, sum, abs-sum and ``sample`` strided
    values per tensor.  Identifies a model for calibration-cache keys; it is *not* a cryptographic content hash."""
    ent = []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        d = t.detach()
        if d.numel() == 0:
            ent.append([name, list(d.shape), str(d.dtype)])
            continue
        f = d.reshape(-1).float().cpu() if d.dtype != torch.float8_e4m3fn else d.reshape(-1).float().cpu()
        step = max(1, f.numel() // sample)
        ent.append([name, list(d.shape), str(d.dtype), round(float(f.sum()), 4), round(float(f.abs().sum()), 4),
                    [round(float(v), 5) for v in f[::step][:sample].tolist()]])
    return hash_json(ent)


def count_params(model: nn.Module) -> int:
    seen: set[int] = set()
    n = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            n += p.numel()
    return n


def model_nbytes(model: nn.Module) -> int:
    """Bytes of parameters + buffers (shared tensors counted once)."""
    seen: set[int] = set()
    n = 0
    for t in list(model.parameters()) + list(model.buffers()):
        k = t.data_ptr()
        if k in seen:
            continue
        seen.add(k)
        n += t.numel() * t.element_size()
    return n
