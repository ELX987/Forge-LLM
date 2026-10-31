"""Linear-module registry: swappable linear layers for dense, quantised and fake-quantised weights.

Every projection inside :class:`forge.models.ForgeLM` is an ``nn.Linear``-compatible module taken from this registry so
that PTQ (``forge.compress``), kernels (``forge.kernels``) and LoRA can swap them without touching the model code.

Conventions for a *packed* (weight-compressed) linear (subclass :class:`PackedLinear`):

* attributes ``in_features`` / ``out_features``; ``kind`` class attribute equal to its registry name;
* all packed tensors are **buffers** (never ``Parameter``) so they appear in ``state_dict`` under
  ``<module path>.<buffer name>`` (e.g. ``qweight``, ``scales``, ``zeros``, ``weight_fp8``, ``weight_scale``);
  the optional bias is a ``Parameter`` named ``bias`` (so it is saved in the usual place);
* ``from_dense(linear, **kw)`` builds the module from a dense ``nn.Linear``;
* ``from_shape(in_features, out_features, bias, **kw)`` builds an *empty* module with correctly shaped buffers so that a
  saved checkpoint can be loaded (``kw`` is what :meth:`quant_spec` returned at save time);
* ``dequantize(dtype)`` returns the dense ``[out, in]`` weight (reference path; used by LoRA-on-packed, DoRA, merge);
* ``quant_spec()`` returns a JSON-able dict of constructor kwargs (stored in ``config.json`` -> ``forge_quantization``).

Registered here: ``dense`` (:class:`DenseLinear`), ``fp8`` (:class:`Fp8Linear`, weight-only-FP8 storage with dynamic
per-token activation FP8, native path through ``forge.kernels.ops_gemm.fp8_linear`` on sm89+), ``fq_int`` and
``fq_nvfp4`` (fake-quant wrappers with STE for QAT / numerics studies).  ``lora`` is registered by ``forge.models.lora``.
Kinds ``w4a16``, ``w8a16``, ``nvfp4`` are expected from ``forge.compress`` / ``forge.kernels`` via ``register_linear``.
"""
from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.core.errors import ConfigError

log = logging.getLogger("forge.models.linear")

_LINEAR_KINDS: dict[str, type] = {}

#: name fragments never converted by default when quantising a whole model
DEFAULT_QUANT_SKIP: tuple[str, ...] = ("lm_head", "gate", "router", "shared_expert_gate")

FP8_E4M3_MAX = 448.0


@runtime_checkable
class LinearBase(Protocol):
    """Anything that behaves like ``nn.Linear`` for the model: ``in_features``, ``out_features`` and ``forward(x)``."""

    in_features: int
    out_features: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def register_linear(kind: str) -> Callable[[type], type]:
    """Class decorator: register a linear module class under ``kind`` (also mirrored in ``forge.core.registry['linear']``)."""

    def deco(cls: type) -> type:
        cls.kind = kind  # type: ignore[attr-defined]
        _LINEAR_KINDS[kind] = cls
        try:
            from forge.core import registry

            registry.register_obj("linear", kind, cls)
        except Exception:  # pragma: no cover
            pass
        return cls

    return deco


def get_linear_cls(kind: str) -> type:
    try:
        return _LINEAR_KINDS[kind]
    except KeyError:
        raise ConfigError(
            f"unknown linear kind {kind!r}; registered: {', '.join(sorted(_LINEAR_KINDS))}", details={"kind": kind}
        ) from None


def linear_kinds() -> list[str]:
    return sorted(_LINEAR_KINDS)


def _strict() -> bool:
    return os.environ.get("FORGE_KERNEL_STRICT", "0") == "1"


# --------------------------------------------------------------------------------------------- dense
@register_linear("dense")
class DenseLinear(nn.Linear):
    """Plain ``nn.Linear`` (``weight [out, in]``, optional ``bias``)."""

    kind: ClassVar[str] = "dense"

    @classmethod
    def from_dense(cls, linear: nn.Linear, **_: Any) -> "DenseLinear":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_features = linear.in_features
        m.out_features = linear.out_features
        m.weight = linear.weight
        m.bias = linear.bias
        return m

    def quant_spec(self) -> dict[str, Any]:
        return {}


class PackedLinear(nn.Module):
    """Base class for weight-compressed linear layers. See the module docstring for the conventions."""

    kind: ClassVar[str] = ""
    in_features: int
    out_features: int

    @classmethod
    def from_dense(cls, linear: nn.Linear, **kw: Any) -> "PackedLinear":  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def from_shape(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: Any = None,
        **kw: Any,
    ) -> "PackedLinear":  # pragma: no cover - interface
        raise NotImplementedError

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def quant_spec(self) -> dict[str, Any]:
        return {}

    def packed_nbytes(self) -> int:
        """Bytes of all buffers and parameters (the memory this layer really occupies)."""
        n = sum(b.numel() * b.element_size() for b in self.buffers())
        n += sum(p.numel() * p.element_size() for p in self.parameters())
        return int(n)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantised dense weight (read-only convenience; allocates)."""
        return self.dequantize()

    def extra_repr(self) -> str:
        return f"kind={self.kind}, in_features={self.in_features}, out_features={self.out_features}, bias={getattr(self, 'bias', None) is not None}"


def is_linear_like(m: nn.Module) -> bool:
    return isinstance(m, (nn.Linear, PackedLinear))


def linear_dense_weight(m: nn.Module, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dense ``[out, in]`` weight of any linear-like module (dequantises packed ones)."""
    if isinstance(m, PackedLinear):
        return m.dequantize(dtype)
    w = getattr(m, "weight", None)
    if w is None:
        base = getattr(m, "base", None)
        if base is not None:
            return linear_dense_weight(base, dtype)
        raise ConfigError(f"{type(m).__name__} has no weight")
    w = w.detach()
    return w.to(dtype) if dtype is not None else w


# --------------------------------------------------------------------------------------------- replacement
def replace_linears(
    model: nn.Module,
    factory: Callable[[str, nn.Linear], nn.Module | None],
    *,
    prefix: str = "",
) -> list[str]:
    """Replace every ``nn.Linear`` for which ``factory(qualified_name, linear)`` returns a module.

    ``None`` from the factory keeps the layer.  Does not descend into the freshly inserted module nor into LoRA
    adapter matrices; LoRA wrappers are traversed through their ``base``.  Returns the list of replaced names.
    """
    replaced: list[str] = []
    for name, child in list(model.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if getattr(child, "kind", None) == "lora" and hasattr(child, "base"):
            replaced += replace_linears(child, factory, prefix=full)
            continue
        if isinstance(child, nn.Linear) and not isinstance(child, PackedLinear):
            new = factory(full, child)
            if new is not None and new is not child:
                setattr(model, name, new)
                replaced.append(full)
            continue
        if isinstance(child, PackedLinear):
            continue
        replaced += replace_linears(child, factory, prefix=full)
    return replaced


def iter_linears(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for every linear-like leaf (dense, packed, LoRA-wrapped excluded adapters)."""
    skip_prefixes: list[str] = []
    for name, m in model.named_modules():
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        if getattr(m, "kind", None) == "lora":
            skip_prefixes.append(name + ".lora_")
            yield name, m
            continue
        if is_linear_like(m):
            yield name, m


def linear_report(model: nn.Module) -> dict[str, str]:
    """``{module name: kind}`` for every linear-like module of the model."""
    out = {}
    for n, m in iter_linears(model):
        out[n] = getattr(m, "kind", "dense")
    return out


def _skipped(name: str, skip: Iterable[str]) -> bool:
    last = name.split(".")[-1]
    return any(s == last or s in name.split(".") or name.endswith(s) for s in skip)


def quantize_linears(
    model: nn.Module,
    kind: str,
    *,
    skip: Iterable[str] = DEFAULT_QUANT_SKIP,
    include: Iterable[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Convert dense linears of ``model`` to registered ``kind`` through ``cls.from_dense(linear, **kwargs)``."""
    cls = get_linear_cls(kind)
    skip = tuple(skip)
    inc = tuple(include) if include is not None else None

    def factory(name: str, lin: nn.Linear) -> nn.Module | None:
        if _skipped(name, skip):
            return None
        if inc is not None and not any(i in name for i in inc):
            return None
        if lin.weight.is_meta:
            return None
        return cls.from_dense(lin, **kwargs)  # type: ignore[attr-defined]

    return replace_linears(model, factory)


def quantization_manifest(model: nn.Module) -> dict[str, dict[str, Any]]:
    """``{name: {"kind": ..., "spec": {...}}}`` for every non-dense packed layer (saved in config.json)."""
    out: dict[str, dict[str, Any]] = {}
    for n, m in iter_linears(model):
        if isinstance(m, PackedLinear) or (isinstance(m, nn.Linear) and getattr(m, "kind", "dense") not in ("dense", "lora")):
            out[n] = {
                "kind": m.kind,  # type: ignore[attr-defined]
                "spec": m.quant_spec() if hasattr(m, "quant_spec") else {},
                "in_features": m.in_features,
                "out_features": m.out_features,
                "bias": getattr(m, "bias", None) is not None,
            }
    return out


def apply_quantization_manifest(model: nn.Module, manifest: dict[str, dict[str, Any]], dtype: torch.dtype | None = None) -> None:
    """Swap linears named in ``manifest`` for empty modules of the right kind/shape (then load the state dict)."""
    lookup = dict(model.named_modules())
    for name, ent in manifest.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = lookup[parent_name] if parent_name else model
        cls = get_linear_cls(ent["kind"])
        old = getattr(parent, leaf)
        dev = old.weight.device if hasattr(old, "weight") and isinstance(old.weight, torch.Tensor) else None
        new = cls.from_shape(  # type: ignore[attr-defined]
            ent["in_features"], ent["out_features"], bool(ent.get("bias")), dtype=dtype, device=dev, **ent.get("spec", {})
        )
        setattr(parent, leaf, new)


# --------------------------------------------------------------------------------------------- numerics helpers
def _fake_quant_int(w: torch.Tensor, bits: int, group_size: int, sym: bool) -> torch.Tensor:
    """Group-wise integer fake quantisation along the last dim (round-half-even), returns same shape/dtype as ``w``."""
    out_f, in_f = w.shape[-2], w.shape[-1]
    g = in_f if group_size <= 0 or group_size > in_f else group_size
    pad = (-in_f) % g
    x = w.float()
    if pad:
        x = F.pad(x, (0, pad))
    xg = x.reshape(*x.shape[:-1], -1, g)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = (xg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-12)
        q = torch.round(xg / scale).clamp(-qmax - 1, qmax)
        deq = q * scale
    else:
        qmax = 2**bits - 1
        mn = xg.amin(dim=-1, keepdim=True)
        mx = xg.amax(dim=-1, keepdim=True)
        scale = ((mx - mn) / qmax).clamp_min(1e-12)
        zero = torch.round(-mn / scale)
        q = torch.round(xg / scale + zero).clamp(0, qmax)
        deq = (q - zero) * scale
    deq = deq.reshape(*x.shape)
    if pad:
        deq = deq[..., :in_f]
    return deq.to(w.dtype)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest(-even on ties) onto the signed E2M1 grid; input already scaled to [-6, 6]."""
    a = x.abs().clamp(max=6.0)
    # midpoints between grid points; ties go to the even mantissa (grid index even)
    mids = a.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    idx = torch.bucketize(a, mids, right=False)  # a == mid -> lower index
    tie = (a.unsqueeze(-1) == mids).any(-1)
    odd_lower = tie & (idx % 2 == 1)  # lower index odd -> mantissa odd -> round up to the even neighbour
    idx = torch.where(odd_lower, idx + 1, idx)
    grid = a.new_tensor(_E2M1)
    return torch.sign(x) * grid[idx]


def _fake_quant_nvfp4_local(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Local NVFP4 reference: 16-element blocks, E4M3 block scales, FP32 tensor scale, E2M1 RTN values."""
    in_f = w.shape[-1]
    pad = (-in_f) % block
    x = w.float()
    if pad:
        x = F.pad(x, (0, pad))
    xb = x.reshape(*x.shape[:-1], -1, block)
    amax = x.abs().amax().clamp_min(1e-12)
    tensor_scale = amax / (6.0 * FP8_E4M3_MAX)
    bs = (xb.abs().amax(dim=-1, keepdim=True) / 6.0) / tensor_scale
    bs = bs.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    scale = (bs * tensor_scale).clamp_min(1e-30)
    q = _round_e2m1(xb / scale)
    deq = (q * scale).reshape(*x.shape)
    if pad:
        deq = deq[..., :in_f]
    return deq.to(w.dtype)


def fake_quant_nvfp4(w: torch.Tensor) -> torch.Tensor:
    """NVFP4 quantise->dequantise. Uses ``forge.formats.fp4`` when importable and consistent, else the local reference."""
    try:  # pragma: no cover - depends on the compress module being present
        from forge.formats import fp4 as _fp4

        t = _fp4.quantize_nvfp4(w.detach().float())
        d = _fp4.dequantize_nvfp4(t)
        if tuple(d.shape) == tuple(w.shape):
            return d.to(w.dtype)
    except Exception:
        pass
    return _fake_quant_nvfp4_local(w)


def fake_quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 quantise->dequantise given a broadcastable ``scale`` (value = fp8 * scale)."""
    y = (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
    return y.to(x.dtype)


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, wq):  # noqa: D401
        return wq

    @staticmethod
    def backward(ctx, g):
        return g, None


# --------------------------------------------------------------------------------------------- fake quant linears
class _FakeQuantLinear(nn.Linear):
    """``nn.Linear`` whose weight (and optionally activations) are fake-quantised in forward with a straight-through estimator."""

    kind: ClassVar[str] = ""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, *, dtype=None, device=None, quant_acts: bool = False):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype, device=device)
        self.quant_acts = quant_acts
        self._cache: tuple[int, torch.Tensor] | None = None

    def _wq(self, w: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _aq(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def quantized_weight(self) -> torch.Tensor:
        w = self.weight
        if w.requires_grad and torch.is_grad_enabled():
            return _STE.apply(w, self._wq(w.detach()))
        key = w._version
        if self._cache is None or self._cache[0] != key or self._cache[1].device != w.device or self._cache[1].dtype != w.dtype:
            self._cache = (key, self._wq(w.detach()))
        return self._cache[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_acts:
            xq = self._aq(x.detach())
            x = x + (xq - x).detach() if x.requires_grad else xq
        return F.linear(x, self.quantized_weight(), self.bias)

    @classmethod
    def from_dense(cls, linear: nn.Linear, **kw: Any):
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, dtype=linear.weight.dtype, device=linear.weight.device, **kw)
        m.weight = linear.weight
        m.bias = linear.bias
        return m

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, **kw):
        return cls(in_features, out_features, bias, dtype=dtype, device=device, **kw)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self._wq(self.weight.detach())
        return w.to(dtype) if dtype is not None else w


@register_linear("fq_int")
class FakeQuantIntLinear(_FakeQuantLinear):
    """Group-wise integer fake-quant weights (``bits``, ``group_size``, symmetric/asymmetric); STE for QAT."""

    def __init__(self, in_features, out_features, bias=False, *, dtype=None, device=None, bits: int = 4, group_size: int = 128, sym: bool = True, quant_acts: bool = False):
        super().__init__(in_features, out_features, bias, dtype=dtype, device=device, quant_acts=quant_acts)
        self.bits, self.group_size, self.sym = int(bits), int(group_size), bool(sym)

    def _wq(self, w):
        return _fake_quant_int(w, self.bits, self.group_size, self.sym)

    def quant_spec(self):
        return {"bits": self.bits, "group_size": self.group_size, "sym": self.sym, "quant_acts": self.quant_acts}


@register_linear("fq_nvfp4")
class FakeQuantNVFP4Linear(_FakeQuantLinear):
    """NVFP4 fake-quant weights (and optionally activations) with STE; numerics via ``forge.formats.fp4`` when present."""

    def _wq(self, w):
        return fake_quant_nvfp4(w)

    def _aq(self, x):
        return fake_quant_nvfp4(x)

    def quant_spec(self):
        return {"quant_acts": self.quant_acts}


# --------------------------------------------------------------------------------------------- fp8
@register_linear("fp8")
class Fp8Linear(PackedLinear):
    """FP8-E4M3 weights with per-output-channel scales and optional dynamic per-token FP8 activations.

    Buffers: ``weight_fp8`` (float8_e4m3fn ``[out, in]``), ``weight_scale`` (fp32 ``[out, 1]``, value = fp8 * scale).
    Native path: ``forge.kernels.ops_gemm.fp8_linear`` on CUDA sm>=89 (written, *not validated on hardware*; assumed signature
    ``fp8_linear(x, w_fp8, w_scale, bias=None)``).  Reference path: dequantise + (optionally fake-quantised activations) matmul.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, *, act_quant: str = "dynamic", dtype=None, device=None):
        super().__init__()
        if act_quant not in ("dynamic", "none"):
            raise ConfigError("act_quant must be 'dynamic' or 'none'")
        self.in_features, self.out_features, self.act_quant = in_features, out_features, act_quant
        self.register_buffer("weight_fp8", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("weight_scale", torch.ones(out_features, 1, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False) if bias else None
        self._kernel_ok = True

    @classmethod
    def from_dense(cls, linear: nn.Linear, act_quant: str = "dynamic", **_: Any) -> "Fp8Linear":
        w = linear.weight.detach().float()
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, act_quant=act_quant, dtype=linear.weight.dtype, device=w.device)
        scale = (w.abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-12)
        m.weight_fp8.copy_((w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn))
        m.weight_scale.copy_(scale)
        if linear.bias is not None:
            m.bias.data.copy_(linear.bias.detach())
        return m

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, act_quant: str = "dynamic", **_: Any):
        return cls(in_features, out_features, bias, act_quant=act_quant, dtype=dtype, device=device)

    def quant_spec(self) -> dict[str, Any]:
        return {"act_quant": self.act_quant}

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.weight_fp8.float() * self.weight_scale
        return w.to(dtype) if dtype is not None else w

    def _native(self, x: torch.Tensor) -> torch.Tensor | None:
        if not (self._kernel_ok and x.is_cuda and os.environ.get("FORGE_FORCE_REF", "0") != "1"):
            return None
        try:
            if torch.cuda.get_device_capability(x.device)[0] * 10 + torch.cuda.get_device_capability(x.device)[1] < 89:
                return None
            from forge.kernels import ops_gemm

            return ops_gemm.fp8_linear(x, self.weight_fp8, self.weight_scale, bias=self.bias)
        except Exception as e:  # kernel missing / signature mismatch / unsupported -> reference path
            if _strict():
                raise
            log.warning("fp8 native path unavailable (%s); using reference path", e)
            self._kernel_ok = False
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._native(x)
        if y is not None:
            return y
        w = self.dequantize(x.dtype)
        if self.act_quant == "dynamic":
            xs = (x.detach().float().abs().amax(dim=-1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-12)
            x = fake_quant_fp8(x, xs)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)


def make_linear(in_features: int, out_features: int, bias: bool = False, *, dtype=None, device=None, kind: str = "dense", **kw: Any) -> nn.Module:
    """Construct an (empty, randomly initialised) linear of a registered kind."""
    cls = get_linear_cls(kind)
    if issubclass(cls, PackedLinear):
        return cls.from_shape(in_features, out_features, bias, dtype=dtype, device=device, **kw)  # type: ignore[attr-defined]
    return cls(in_features, out_features, bias=bias, dtype=dtype, device=device, **kw)


__all__ = [
    "LinearBase",
    "DenseLinear",
    "PackedLinear",
    "Fp8Linear",
    "FakeQuantIntLinear",
    "FakeQuantNVFP4Linear",
    "register_linear",
    "get_linear_cls",
    "linear_kinds",
    "replace_linears",
    "iter_linears",
    "linear_report",
    "quantize_linears",
    "quantization_manifest",
    "apply_quantization_manifest",
    "linear_dense_weight",
    "make_linear",
    "fake_quant_nvfp4",
    "fake_quant_fp8",
    "DEFAULT_QUANT_SKIP",
]
