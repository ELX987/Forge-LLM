"""Packed (weight-compressed) linear modules produced by :mod:`forge.compress.ptq`.

All classes follow the ``forge.models.linear.PackedLinear`` conventions (buffers only, ``from_dense``, ``from_shape``,
``dequantize``, ``quant_spec``) and are registered in the linear registry under the canonical kinds

======================  =====================================================================================
kind                    storage (portable "simple" layout, ``layout_id = "forge.simple.v1"``)
======================  =====================================================================================
``w4a16`` / ``w8a16``   ``qweight uint8 [N, K*bits/8]`` (codes packed little-endian along K), ``scales [N, K/G]``,
                        optional ``zeros uint8 [N, K/G]`` (absent = symmetric, zero point ``2**(bits-1)``).
                        Other bit widths register as ``w2a16``, ``w3a16``, ``w5a16``, ``w6a16`` (3/5/6 bit codes are
                        stored one per byte -- accuracy experiments, not memory savings).
``w8a8_int8``           SmoothQuant W8A8: ``qweight int8 [N, K]``, per-channel ``scales [N, 1]`` fp32; dynamic per-token
                        int8 activation quantisation (emulated in float; no int8 GEMM kernel is wired up).
``nvfp4`` / ``mxfp4``   ``qweight uint8 [N, K'/2]`` (E2M1 codes, 2 per byte), ``block_scales`` (float8_e4m3fn for
                        NVFP4, uint8 E8M0 for MXFP4), ``tensor_scale`` fp32 (NVFP4).  ``act_quant="dynamic"`` fake-
                        quantises activations to the same format (W4A4 emulation).
======================  =====================================================================================

Optional per-layer input transforms (all exact, applied before the matmul): ``input_scale`` buffer (AWQ / SmoothQuant
channel scaling that could not be folded into the preceding norm) and ``rot_block`` (block-diagonal orthonormal Hadamard
of that size, MR-GPTQ style).

Kernel dispatch (CUDA only, ``FORGE_FORCE_REF=1`` disables): ``w4a16`` packs the codes with ``forge.kernels.layouts.pack_w4`` (portable row
layout unless the ``pack`` stage chose another) and calls ``ops_gemm.w4a16_linear``; ``nvfp4`` calls ``ops_gemm.fp4_linear`` with the
``NVFP4Tensor`` view of its buffers.  Those entry points fall back to their own torch oracles when Triton is missing.  The dispatch is
*written against the kernels' signatures and not validated on hardware*; any exception falls back (once, with a warning) to the
dequantise-then-matmul reference path in this file, which is what the CPU tests exercise.  The kernel-owner classes in
``forge.kernels.ops_gemm`` (``KernelW4A16Linear`` ...) use the same *kind* names but their own buffer layouts; they are not mixed with these.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.core.errors import ConfigError
from forge.formats.fp4 import (
    NVFP4Tensor,
    decode_e2m1,
    fake_quant_mxfp4,
    fake_quant_nvfp4,
)
from forge.formats.int_quant import _div_const
from forge.formats.packing import pack_bits, unpack_bits
from forge.formats.fp4 import unpack_fp4

log = logging.getLogger("forge.compress.packed")

try:  # the models package is written concurrently; fall back to a minimal base so this module always imports
    from forge.models.linear import PackedLinear, get_linear_cls, linear_kinds, register_linear

    _HAVE_MODELS = True
except Exception:  # pragma: no cover - exercised only when forge.models is absent
    _HAVE_MODELS = False

    class PackedLinear(nn.Module):  # type: ignore[no-redef]
        kind: ClassVar[str] = ""

        @property
        def weight(self) -> torch.Tensor:
            return self.dequantize()

        def packed_nbytes(self) -> int:
            return int(sum(b.numel() * b.element_size() for b in self.buffers()))

    def register_linear(kind: str):  # type: ignore[no-redef]
        def deco(cls):
            cls.kind = kind
            _LOCAL_KINDS[kind] = cls
            return cls

        return deco

    def get_linear_cls(kind: str):  # type: ignore[no-redef]
        return _LOCAL_KINDS[kind]

    def linear_kinds():  # type: ignore[no-redef]
        return sorted(_LOCAL_KINDS)


_LOCAL_KINDS: dict[str, type] = {}
LAYOUT_ID = "forge.simple.v1"
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(msg)


def _force_ref() -> bool:
    return os.environ.get("FORGE_FORCE_REF", "0") == "1"


# ------------------------------------------------------------------------------------------------ Hadamard helper
def hadamard_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix (``n`` must be a power of two)."""
    if n < 1 or n & (n - 1):
        raise ConfigError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / (n**0.5)).to(dtype)


def apply_block_hadamard(x: torch.Tensor, block: int) -> torch.Tensor:
    """Apply the orthonormal ``block x block`` Hadamard to consecutive blocks of the last dim (``x @ blockdiag(H)``)."""
    k = x.shape[-1]
    if k % block:
        raise ConfigError(f"last dim {k} not divisible by rotation block {block}")
    h = hadamard_matrix(block, x.device, x.dtype)
    return (x.reshape(*x.shape[:-1], k // block, block) @ h).reshape(x.shape)


class _InputTransform:
    """Mixin: optional ``input_scale`` buffer (multiplied into the activations) and block-Hadamard rotation."""

    in_features: int

    def _init_transform(self, has_input_scale: bool, rot_block: int, dtype, device) -> None:
        self.rot_block = int(rot_block)
        self.has_input_scale = bool(has_input_scale)
        if has_input_scale:
            self.register_buffer("input_scale", torch.ones(self.in_features, dtype=torch.float32, device=device))  # type: ignore[attr-defined]
        else:
            self.input_scale = None

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_input_scale:
            x = x * self.input_scale.to(x.dtype)  # type: ignore[union-attr]
        if self.rot_block:
            x = apply_block_hadamard(x.float(), self.rot_block).to(x.dtype)
        return x

    def _transform_spec(self) -> dict[str, Any]:
        return {"has_input_scale": self.has_input_scale, "rot_block": self.rot_block}


# ------------------------------------------------------------------------------------------------ integer weights
class IntLinear(_InputTransform, PackedLinear):
    """Group-wise integer weight-only linear (W{2..8}A16). See module docstring for the storage format."""

    kind: ClassVar[str] = "w4a16"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        has_input_scale: bool = False,
        rot_block: int = 0,
        dtype: torch.dtype | None = None,
        device: Any = None,
    ) -> None:
        super().__init__()
        if not 2 <= bits <= 8:
            raise ConfigError(f"IntLinear supports 2..8 bits, got {bits}")
        g = in_features if group_size is None or group_size <= 0 or group_size > in_features else int(group_size)
        if in_features % g:
            raise ConfigError(f"in_features {in_features} not divisible by group_size {g}")
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size, self.sym = int(bits), g, bool(sym)
        self._per_byte = 8 // bits if bits in (2, 4, 8) else 1
        self._pack_bits = bits if bits in (2, 4, 8) else 8
        ng = in_features // g
        qcols = -(-in_features // self._per_byte)
        dt = dtype or torch.float16
        self.register_buffer("qweight", torch.zeros(out_features, qcols, dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.ones(out_features, ng, dtype=dt, device=device))
        if sym:
            self.zeros = None
        else:
            self.register_buffer("zeros", torch.zeros(out_features, ng, dtype=torch.uint8, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dt, device=device), requires_grad=False) if bias else None
        self._init_transform(has_input_scale, rot_block, dt, device)
        self._kernel_cache: tuple[Any, Any] | None = None
        self._kernel_ok = True
        self.kernel_layout: Any = None  # device layout chosen by the `pack` stage (None = portable default)

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def from_codes(
        cls,
        q: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        *,
        bits: int,
        group_size: int,
        bias: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        rot_block: int = 0,
        sym: bool | None = None,
    ) -> IntLinear:
        """Build from unsigned codes ``q [N,K]`` (``0..2**bits-1``), ``scales [N,G]`` and ``zeros [N,G]`` (``None`` = symmetric)."""
        n, k = q.shape
        sym = zeros is None if sym is None else sym
        sub = _int_class_for_bits(bits)
        m = sub(
            k, n, bias is not None, bits=bits, group_size=group_size, sym=bool(sym) and zeros is None,
            has_input_scale=input_scale is not None, rot_block=rot_block, dtype=dtype or scales.dtype, device=q.device,
        )
        m.qweight.copy_(pack_bits(q, m._pack_bits) if m._per_byte > 1 or bits == 8 else q.to(torch.uint8))
        m.scales.copy_(scales.to(m.scales.dtype))
        if m.zeros is not None:
            m.zeros.copy_(zeros.to(torch.uint8))  # type: ignore[union-attr]
        if bias is not None:
            m.bias.data.copy_(bias.detach().to(m.bias.dtype))
        if input_scale is not None:
            m.input_scale.copy_(input_scale.float())
        return m

    @classmethod
    def from_dense(cls, linear: nn.Linear, bits: int = 4, group_size: int = 128, sym: bool = True, **_: Any) -> IntLinear:
        from forge.formats.int_quant import quantize_int

        q, s, z = quantize_int(linear.weight.detach(), bits, group_size, sym)
        return cls.from_codes(
            q, s.to(linear.weight.dtype), None if sym else z, bits=bits, group_size=group_size if group_size > 0 else q.shape[1],
            bias=linear.bias, dtype=linear.weight.dtype,
        )

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, **kw: Any) -> IntLinear:
        kw = {k: v for k, v in kw.items() if k not in ("layout",)}
        bits = int(kw.get("bits", 4))
        return _int_class_for_bits(bits)(in_features, out_features, bias, dtype=dtype, device=device, **kw)

    # -- views ------------------------------------------------------------------------------------
    def codes(self) -> torch.Tensor:
        """Unsigned codes ``uint8 [N, K]``."""
        if self._per_byte > 1:
            return unpack_bits(self.qweight, self._pack_bits, self.in_features)
        return self.qweight

    def zero_points(self) -> torch.Tensor:
        if self.zeros is not None:
            return self.zeros
        return torch.full_like(self.scales, 1 << (self.bits - 1), dtype=torch.uint8)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dense ``[N, K]`` weight in the *quantised-weight space* (input transforms are not folded in)."""
        g = self.group_size
        q = self.codes().float()
        z = self.zero_points().float().repeat_interleave(g, dim=1)
        s = self.scales.float().repeat_interleave(g, dim=1)
        w = (q - z) * s
        return w.to(dtype) if dtype is not None else w.to(self.scales.dtype)

    def quant_spec(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "sym": self.sym, "layout": LAYOUT_ID, **self._transform_spec()}

    # -- kernel path ------------------------------------------------------------------------------
    def _kernel_packed(self) -> Any:
        from forge.kernels.layouts import pack_w4

        dev = self.qweight.device
        if self._kernel_cache is not None and self._kernel_cache[0] == (dev, self.qweight._version):
            return self._kernel_cache[1]
        z = None if self.sym else self.zeros
        pk = pack_w4(self.codes(), self.scales, z, self.group_size, getattr(self, "kernel_layout", None))
        self._kernel_cache = ((dev, self.qweight._version), pk)
        return pk

    def _native(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.bits != 4 or not (self._kernel_ok and x.is_cuda) or _force_ref():
            return None
        try:
            from forge.kernels import ops_gemm

            return ops_gemm.w4a16_linear(x, self._kernel_packed(), self.bias.to(x.dtype) if self.bias is not None else None)
        except Exception as e:  # missing kernel / signature mismatch / unsupported -> reference
            if os.environ.get("FORGE_KERNEL_STRICT", "0") == "1":
                raise
            _warn_once("w4a16_native", f"w4a16 native path unavailable ({type(e).__name__}: {e}); using reference path")
            self._kernel_ok = False
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre(x)
        y = self._native(x)
        if y is not None:
            return y
        return F.linear(x, self.dequantize(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, group={self.group_size}, sym={self.sym}, bias={self.bias is not None}"


_INT_CLASSES: dict[int, type] = {}


def _int_class_for_bits(bits: int) -> type:
    cls = _INT_CLASSES.get(bits)
    if cls is None:
        raise ConfigError(f"no packed linear for {bits}-bit weights (supported: {sorted(_INT_CLASSES)})")
    return cls


def _make_int_class(bits: int) -> type:
    name = f"w{bits}a16"
    cls = type(f"W{bits}A16Linear", (IntLinear,), {"kind": name, "__doc__": f"{bits}-bit weight-only linear (kind '{name}')."})
    _INT_CLASSES[bits] = cls
    return cls


for _b in (2, 3, 4, 5, 6, 7, 8):
    _make_int_class(_b)


# ------------------------------------------------------------------------------------------------ int8 W8A8 (SmoothQuant)
class Int8W8A8Linear(_InputTransform, PackedLinear):
    """SmoothQuant-style W8A8: per-output-channel int8 weights, dynamic per-token int8 activations (emulated in float)."""

    kind: ClassVar[str] = "w8a8_int8"

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, *, act_quant: str = "dynamic", has_input_scale: bool = False,
        rot_block: int = 0, dtype: torch.dtype | None = None, device: Any = None,
    ) -> None:
        super().__init__()
        if act_quant not in ("dynamic", "none"):
            raise ConfigError("act_quant must be 'dynamic' or 'none'")
        self.in_features, self.out_features, self.act_quant = in_features, out_features, act_quant
        dt = dtype or torch.float16
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scales", torch.ones(out_features, 1, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dt, device=device), requires_grad=False) if bias else None
        self._init_transform(has_input_scale, rot_block, dt, device)

    @classmethod
    def from_weight(cls, w: torch.Tensor, bias: torch.Tensor | None = None, *, act_quant: str = "dynamic", input_scale: torch.Tensor | None = None, dtype=None) -> Int8W8A8Linear:
        wf = w.detach().float()
        s = _div_const(wf.abs().amax(dim=1, keepdim=True), 127.0).clamp_min(1e-12)
        m = cls(w.shape[1], w.shape[0], bias is not None, act_quant=act_quant, has_input_scale=input_scale is not None, dtype=dtype or w.dtype, device=w.device)
        m.qweight.copy_(torch.round(wf / s).clamp(-128, 127).to(torch.int8))
        m.scales.copy_(s)
        if bias is not None:
            m.bias.data.copy_(bias.detach().to(m.bias.dtype))
        if input_scale is not None:
            m.input_scale.copy_(input_scale.float())
        return m

    @classmethod
    def from_dense(cls, linear: nn.Linear, act_quant: str = "dynamic", **_: Any) -> Int8W8A8Linear:
        return cls.from_weight(linear.weight, linear.bias, act_quant=act_quant)

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, **kw: Any) -> Int8W8A8Linear:
        kw = {k: v for k, v in kw.items() if k not in ("layout",)}
        return cls(in_features, out_features, bias, dtype=dtype, device=device, **kw)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.qweight.float() * self.scales
        return w.to(dtype) if dtype is not None else w

    def quant_spec(self) -> dict[str, Any]:
        return {"act_quant": self.act_quant, "layout": LAYOUT_ID, **self._transform_spec()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre(x)
        if self.act_quant == "dynamic":
            xf = x.float()
            s = _div_const(xf.abs().amax(dim=-1, keepdim=True), 127.0).clamp_min(1e-12)
            x = (torch.round(xf / s).clamp(-128, 127) * s).to(x.dtype)
        return F.linear(x, self.dequantize(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


# ------------------------------------------------------------------------------------------------ FP4 weights
class NVFP4Linear(_InputTransform, PackedLinear):
    """NVFP4 weights (16-block E4M3 scales + FP32 tensor scale). ``act_quant='dynamic'`` emulates W4A4."""

    kind: ClassVar[str] = "nvfp4"
    BLOCK = 16

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, *, act_quant: str = "none", has_input_scale: bool = False,
        rot_block: int = 0, dtype: torch.dtype | None = None, device: Any = None,
    ) -> None:
        super().__init__()
        if act_quant not in ("dynamic", "none"):
            raise ConfigError("act_quant must be 'dynamic' or 'none'")
        self.in_features, self.out_features, self.act_quant = in_features, out_features, act_quant
        kb = -(-in_features // self.BLOCK)
        dt = dtype or torch.bfloat16
        self.register_buffer("qweight", torch.zeros(out_features, kb * self.BLOCK // 2, dtype=torch.uint8, device=device))
        self.register_buffer("block_scales", torch.ones(out_features, kb, dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("tensor_scale", torch.ones((), dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dt, device=device), requires_grad=False) if bias else None
        self._init_transform(has_input_scale, rot_block, dt, device)
        self._kernel_ok = True

    @classmethod
    def from_tensor(cls, t: NVFP4Tensor, bias: torch.Tensor | None = None, *, act_quant: str = "none", input_scale: torch.Tensor | None = None, rot_block: int = 0, dtype=None) -> NVFP4Linear:
        n, k = t.shape
        m = cls(k, n, bias is not None, act_quant=act_quant, has_input_scale=input_scale is not None, rot_block=rot_block, dtype=dtype or torch.bfloat16, device=t.payload_u8.device)
        m.qweight.copy_(t.payload_u8)
        m.block_scales.copy_(t.block_scales_e4m3)
        m.tensor_scale.copy_(t.tensor_scale.reshape(()))
        if bias is not None:
            m.bias.data.copy_(bias.detach().to(m.bias.dtype))
        if input_scale is not None:
            m.input_scale.copy_(input_scale.float())
        return m

    @classmethod
    def from_dense(cls, linear: nn.Linear, act_quant: str = "none", **kw: Any) -> NVFP4Linear:
        from forge.formats.fp4 import quantize_nvfp4

        t = quantize_nvfp4(linear.weight.detach().float(), **{k: v for k, v in kw.items() if k in ("four_over_six", "scale_2d")})
        return cls.from_tensor(t, linear.bias, act_quant=act_quant, dtype=linear.weight.dtype)

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, **kw: Any) -> NVFP4Linear:
        kw = {k: v for k, v in kw.items() if k not in ("layout",)}
        return cls(in_features, out_features, bias, dtype=dtype, device=device, **kw)

    def as_nvfp4_tensor(self) -> NVFP4Tensor:
        return NVFP4Tensor(self.qweight, self.block_scales, self.tensor_scale, (self.out_features, self.in_features), self.BLOCK)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.as_nvfp4_tensor().dequantize(torch.float32)
        return w.to(dtype) if dtype is not None else w

    def quant_spec(self) -> dict[str, Any]:
        return {"act_quant": self.act_quant, "layout": LAYOUT_ID, **self._transform_spec()}

    def _native(self, x: torch.Tensor) -> torch.Tensor | None:
        """CUDA path: ``ops_gemm.fp4_linear`` (Triton dequant-GEMM on any sm; W4A4 activation quantiser emulated/fused by the kernel owner)."""
        if not (self._kernel_ok and x.is_cuda) or _force_ref():
            return None
        try:
            from forge.kernels import ops_gemm

            b = self.bias.to(x.dtype) if self.bias is not None else None
            return ops_gemm.fp4_linear(x, self.as_nvfp4_tensor(), b, act_quant="fp4" if self.act_quant == "dynamic" else "none")
        except Exception as e:
            if os.environ.get("FORGE_KERNEL_STRICT", "0") == "1":
                raise
            _warn_once("nvfp4_native", f"nvfp4 native path unavailable ({type(e).__name__}: {e}); using reference path")
            self._kernel_ok = False
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre(x)
        y = self._native(x)  # (activation quantisation, if enabled, happens inside the kernel path)
        if y is not None:
            return y
        if self.act_quant == "dynamic":
            x = fake_quant_nvfp4(x.float()).to(x.dtype)
        return F.linear(x, self.dequantize(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class MXFP4Linear(_InputTransform, PackedLinear):
    """MXFP4 weights (32-block E8M0 power-of-two scales)."""

    kind: ClassVar[str] = "mxfp4"
    BLOCK = 32

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, *, act_quant: str = "none", has_input_scale: bool = False,
        rot_block: int = 0, dtype: torch.dtype | None = None, device: Any = None,
    ) -> None:
        super().__init__()
        if act_quant not in ("dynamic", "none"):
            raise ConfigError("act_quant must be 'dynamic' or 'none'")
        self.in_features, self.out_features, self.act_quant = in_features, out_features, act_quant
        kb = -(-in_features // self.BLOCK)
        dt = dtype or torch.bfloat16
        self.register_buffer("qweight", torch.zeros(out_features, kb * self.BLOCK // 2, dtype=torch.uint8, device=device))
        self.register_buffer("block_scales", torch.full((out_features, kb), 127, dtype=torch.uint8, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dt, device=device), requires_grad=False) if bias else None
        self._init_transform(has_input_scale, rot_block, dt, device)

    @classmethod
    def from_parts(cls, payload: torch.Tensor, scales_u8: torch.Tensor, shape: tuple[int, int], bias=None, *, act_quant="none", input_scale=None, rot_block=0, dtype=None) -> MXFP4Linear:
        n, k = shape
        m = cls(k, n, bias is not None, act_quant=act_quant, has_input_scale=input_scale is not None, rot_block=rot_block, dtype=dtype or torch.bfloat16, device=payload.device)
        m.qweight.copy_(payload)
        m.block_scales.copy_(scales_u8)
        if bias is not None:
            m.bias.data.copy_(bias.detach().to(m.bias.dtype))
        if input_scale is not None:
            m.input_scale.copy_(input_scale.float())
        return m

    @classmethod
    def from_dense(cls, linear: nn.Linear, act_quant: str = "none", scale_mode: str = "ceil", **_: Any) -> MXFP4Linear:
        from forge.formats.fp4 import quantize_mxfp4

        t = quantize_mxfp4(linear.weight.detach().float(), scale_mode=scale_mode)
        return cls.from_parts(t.payload_u8, t.scales_e8m0_u8, tuple(t.shape), linear.bias, act_quant=act_quant, dtype=linear.weight.dtype)  # type: ignore[arg-type]

    @classmethod
    def from_shape(cls, in_features, out_features, bias=False, dtype=None, device=None, **kw: Any) -> MXFP4Linear:
        kw = {k: v for k, v in kw.items() if k not in ("layout",)}
        return cls(in_features, out_features, bias, dtype=dtype, device=device, **kw)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        vals = decode_e2m1(unpack_fp4(self.qweight))
        n = vals.shape[0]
        vb = vals.reshape(n, -1, self.BLOCK)
        w = (vb * torch.pow(2.0, self.block_scales.float() - 127.0).unsqueeze(-1)).reshape(n, -1)[:, : self.in_features]
        return w.to(dtype) if dtype is not None else w

    def quant_spec(self) -> dict[str, Any]:
        return {"act_quant": self.act_quant, "layout": LAYOUT_ID, **self._transform_spec()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre(x)
        if self.act_quant == "dynamic":
            x = fake_quant_mxfp4(x.float()).to(x.dtype)
        return F.linear(x, self.dequantize(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


# ------------------------------------------------------------------------------------------------ registration
def register_packed_kinds(force: bool = False) -> list[str]:
    """Register the classes above in the linear registry (kinds already registered by someone else are left alone unless
    ``force``).  Returns the kinds registered.  Idempotent; safe to call from any module."""
    done = []
    existing = set(linear_kinds())
    table: dict[str, type] = {c.kind: c for c in _INT_CLASSES.values()}
    table.update({"w8a8_int8": Int8W8A8Linear, "nvfp4": NVFP4Linear, "mxfp4": MXFP4Linear})
    for kind, cls in table.items():
        if kind in existing and not force:
            try:
                if get_linear_cls(kind) is cls:
                    continue
            except Exception:
                pass
            continue
        register_linear(kind)(cls)
        done.append(kind)
    return done


try:
    register_packed_kinds()
except Exception as _e:  # pragma: no cover
    log.debug("could not register packed kinds: %s", _e)

PACKED_CLASSES: dict[str, type] = {c.kind: c for c in _INT_CLASSES.values()}
PACKED_CLASSES.update({"w8a8_int8": Int8W8A8Linear, "nvfp4": NVFP4Linear, "mxfp4": MXFP4Linear})


def packed_class(kind: str) -> type:
    """Our own class for ``kind`` (independent of what the registry currently holds)."""
    try:
        return PACKED_CLASSES[kind]
    except KeyError:
        raise ConfigError(f"unknown packed kind {kind!r}; known: {sorted(PACKED_CLASSES)}") from None
