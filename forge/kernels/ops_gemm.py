"""GEMM-side kernel dispatchers (Plan 6.4 K1-K3): capability-based routing between the Triton kernels, vendor/torch
fast paths and the pure-torch oracles.

Public API (``docs/ARCHITECTURE.md`` 5.5)::

    w4a16_linear(x, packed: PackedW4, bias=None)            w8a16_linear(x, packed: PackedW8, bias=None)
    fp8_linear(x, w_fp8, w_scale, bias=None, ...)           quantize_act_fp8(x, ...)  -> (q_fp8, scale)
    int8_linear(x, w_int8, w_scale, bias=None, ...)         quantize_act_int8(x) -> (q_int8, scale)
    fp4_linear(x, w: PackedFP4 | NVFP4Tensor, bias=None, act_quant="none"|"fp4")      quantize_act_fp4(x, fmt)
    make_packed_linear(kind, ...)  -> nn.Module             bench_cases() -> list[dict]   (see bench.py)

Dispatch rule (every op)::

    Triton kernel   if  tensor on CUDA (or TRITON_INTERPRET=1)  and  triton importable  and  capability OK
                        and  not FORGE_FORCE_REF=1  and  shape/dtype supported
    otherwise       a torch fallback (vendor ``_scaled_mm`` / dequantise-then-matmul) or the oracle.
``FORGE_KERNEL_DEBUG=1`` logs the path taken; :func:`last_path` returns it programmatically (used by tests).
``FORGE_KERNEL_STRICT=1`` turns kernel failures into errors instead of silent fallback.

HONESTY: nothing in the Triton branches has been executed on an NVIDIA GPU in this repository's development
environment; see ``forge/kernels/README_GEMM.md`` for the validation status table.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.core.errors import ConfigError
from forge.kernels.layouts import (
    PORTABLE_W4,
    PORTABLE_W8,
    Layout,
    PackedFP4,
    PackedW4,
    PackedW8,
    get_layout,
    pack_w4,
    pack_w8,
    quantize_w4_rtn,
    quantize_w8_rtn,
    repack,
)
from forge.kernels.layouts.chip import ChipInfo, chip_info
from forge.kernels.w4_tuning import DECODE_MAX_M, W4_TUNE_TAG, neighbor_buckets, nearest_entry
from forge.kernels.ref import (
    dequantize_fp4_ref,
    dequantize_fp8_ref,
    fp4_block_gemm_ref,
    fp8_gemm_ref,
    int8_gemm_ref,
    quantize_fp4_ref,
    quantize_fp8_ref,
    quantize_int8_ref,
    rmsnorm_act_quant_fp8_ref,
    w4a16_gemm_ref,
    w8a16_gemm_ref,
)

try:  # subclass the models' PackedLinear when available so replace_linears / LoRA treat us as packed layers
    from forge.models.linear import PackedLinear as _PackedBase  # type: ignore
except Exception:  # models not importable (partial install): plain nn.Module
    _PackedBase = nn.Module  # type: ignore[assignment,misc]

log = logging.getLogger("forge.kernels.ops_gemm")

_LAST_PATH: dict[str, str] = {}
_WARNED: set[str] = set()
_LOCK = threading.Lock()


# ------------------------------------------------------------------------------------------------------------------
# environment / capability helpers
# ------------------------------------------------------------------------------------------------------------------
def force_ref() -> bool:
    return os.environ.get("FORGE_FORCE_REF", "0") == "1"


def kernel_debug() -> bool:
    return os.environ.get("FORGE_KERNEL_DEBUG", "0") == "1"


def kernel_strict() -> bool:
    return os.environ.get("FORGE_KERNEL_STRICT", "0") == "1"


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def last_path(op: str) -> str | None:
    """Path (``'triton'``, ``'scaled_mm'``, ``'dequant_matmul'``, ``'reference'``...) taken by the last call of ``op``."""
    return _LAST_PATH.get(op)


def _note(op: str, path: str, why: str = "") -> None:
    _LAST_PATH[op] = path
    if kernel_debug():
        log.info("%s -> %s%s", op, path, f" ({why})" if why else "")


def _warn_once(key: str, msg: str) -> None:
    with _LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    log.warning(msg)


_CHIPS: dict[Any, ChipInfo] = {}


def chip_of(x: torch.Tensor) -> ChipInfo:
    """:class:`ChipInfo` of the device holding ``x`` (CPU info for CPU tensors)."""
    if not x.is_cuda:
        return chip_info(None) if _interpret() else ChipInfo()
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    c = _CHIPS.get(idx)
    if c is None:
        c = chip_info(None, idx)
        _CHIPS[idx] = c
    return c


def _triton_modules() -> Any:
    """Lazily import the Triton launchers; ``None`` if Triton is unavailable."""
    try:
        from forge.kernels.triton import gemm_common

        if not gemm_common.HAS_TRITON:
            return None
        return gemm_common
    except Exception:
        return None


_FAILED: set[tuple] = set()


def _fkey(op: str, x: torch.Tensor, *dims: int) -> tuple:
    """Failure-memo key: op, dtype, device and shape bucket (rows rounded up to a power of two)."""
    rows = 1
    for s in x.shape[:-1]:
        rows *= int(s)
    return (op, str(x.dtype), x.device.index if x.is_cuda else -1, 1 if rows <= 1 else 1 << (rows - 1).bit_length(), *dims)


def reset_failure_memo() -> None:
    """Forget which (op, shape) combinations failed in Triton (they are skipped to avoid re-compiling every call)."""
    _FAILED.clear()


def triton_usable(x: torch.Tensor, fkey: tuple | None = None) -> tuple[bool, str]:
    """(usable, reason-if-not) for running a Triton kernel on ``x``'s device.  ``fkey`` (see :func:`_fkey`): a key whose
    Triton launch already failed in this process is skipped (the reference/fallback path is used directly) instead of
    re-compiling and re-failing on every forward call."""
    if force_ref():
        return False, "FORGE_FORCE_REF=1"
    if fkey is not None and fkey in _FAILED:
        return False, "Triton launch failed earlier for this op/shape (see log); use reset_failure_memo() to retry"
    if not (x.is_cuda or _interpret()):
        return False, "tensor is not on a CUDA device"
    if _triton_modules() is None:
        return False, "triton is not installed"
    if x.is_cuda:
        c = chip_of(x)
        if c.sm is not None and c.sm < 70:
            return False, f"sm_{c.sm} < sm_70"
    return True, ""


def _fail(op: str, e: Exception, fkey: tuple | None = None) -> None:
    if kernel_strict():
        raise e
    from forge.kernels.triton.gemm_common import is_compile_error

    if fkey is not None and is_compile_error(e):  # transient errors (OOM, ...) are never memoised: only this call falls back
        _FAILED.add(fkey)
    _warn_once(f"{op}:{type(e).__name__}", f"{op}: Triton path failed ({type(e).__name__}: {e}); falling back to reference path")


# ------------------------------------------------------------------------------------------------------------------
# tuned-config lookup (autotune cache) with a short-lived memo to keep the decode path cheap
# ------------------------------------------------------------------------------------------------------------------
_TUNERS: dict[str, Any] = {}
_CFG_MEMO: dict[tuple, tuple[float, dict | None]] = {}
_MEMO_TTL_S = 5.0


def dtype_cfg(kernel: str, dtype: torch.dtype, group_size: int = 0, extra: str = "") -> str:
    """Canonical ``dtype_cfg`` string of the autotune cache key (shared by dispatchers, benchmarks and tuning)."""
    return f"{kernel}|x={str(dtype).replace('torch.', '')}|g={group_size}{('|' + extra) if extra else ''}"


def clear_config_memo() -> None:
    _CFG_MEMO.clear()


def tuned_config(kernel: str, chip: ChipInfo, m: int, n: int, k: int, *, dtype_cfg_str: str, layout_id: str = "") -> dict | None:
    """Best cached config for this shape bucket or ``None`` (never tunes, never raises)."""
    if os.environ.get("FORGE_AUTOTUNE_USE", "1") == "0":
        return None
    key = (kernel, chip.fingerprint, 1 if m <= 1 else 1 << (m - 1).bit_length(), n, k, dtype_cfg_str, layout_id)
    now = time.monotonic()
    hit = _CFG_MEMO.get(key)
    if hit is not None and now - hit[0] < _MEMO_TTL_S:
        return hit[1]
    cfg = None
    try:
        from forge.kernels.autotune import Autotuner
        from forge.kernels.spec import load_spec

        tuner = _TUNERS.get(chip.fingerprint)
        if tuner is None:
            tuner = _TUNERS[chip.fingerprint] = Autotuner(chip=chip)
        spec = load_spec(kernel)
        # W4A16: an untuned M bucket borrows the nearest tuned bucket of the same regime (decode / prefill) ...
        buckets = neighbor_buckets(m) if kernel == "w4a16_gemm" else [key[2]]
        for b in buckets:
            cfg = tuner.lookup(spec, {"M": b, "N": n, "K": k}, dtype_cfg=dtype_cfg_str, layout_id=layout_id)
            if cfg:
                break
        # ... and an untuned (N, K) borrows the closest tuned shape (decode only; split-K is recomputed for the new shape)
        if not cfg and kernel == "w4a16_gemm" and buckets[0] <= DECODE_MAX_M:
            entries = tuner.cache.entries()
            for b in buckets:
                want = tuner.key_fields(spec, {"M": b, "N": n, "K": k}, dtype_cfg=dtype_cfg_str, layout_id=layout_id)
                near = nearest_entry(entries, want, n, k)
                if near:
                    cfg = {kk: v for kk, v in near["config"].items() if kk != "split_k"}
                    break
    except Exception as e:  # corrupt cache etc. must never break inference
        log.debug("autotune lookup failed: %s", e)
    _CFG_MEMO[key] = (now, cfg)
    return cfg


# ------------------------------------------------------------------------------------------------------------------
# W4A16 / W8A16
# ------------------------------------------------------------------------------------------------------------------
def _data_stamp(*ts: torch.Tensor | None) -> tuple:
    """Identity + in-place-modification stamp of tensors (``data_ptr`` and autograd ``_version``)."""
    return tuple((t.data_ptr(), t._version, tuple(t.shape), str(t.device)) for t in ts if t is not None)


def _portable(packed: PackedW4) -> PackedW4:
    """``packed`` in the portable_row layout (repacked once and cached in ``packed.meta``).

    The cache is stamped with the identity and version counter of every source tensor, so ``load_state_dict`` /
    in-place weight edits / ``.to()`` can never serve a stale repack (silent wrong results)."""
    if packed.layout.id == PORTABLE_W4.id:
        return packed
    stamp = _data_stamp(packed.qweight, packed.scales, packed.zeros, packed.g_idx)
    cache = packed.meta.setdefault("_repack_cache", {})
    hit = cache.get(PORTABLE_W4.id)
    if hit is None or hit[0] != stamp:
        hit = cache[PORTABLE_W4.id] = (stamp, repack(packed, PORTABLE_W4))
    return hit[1]


def _compute_dtype(x: torch.Tensor, chip: ChipInfo) -> torch.dtype:
    """bf16 has no tensor-core path before sm_80: compute in fp16 there."""
    if x.dtype == torch.bfloat16 and x.is_cuda and not chip.has_bf16:
        return torch.float16
    return x.dtype


def w4a16_linear(
    x: torch.Tensor,
    packed: PackedW4,
    bias: torch.Tensor | None = None,
    *,
    config: dict[str, Any] | None = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """``y = x @ dequant(W)^T + bias`` for ``x [..., K]`` fp16/bf16.  Triton kernel on CUDA, oracle elsewhere."""
    fk = _fkey("w4a16_linear", x, packed.n, packed.k)
    ok, why = triton_usable(x, fk)
    if ok and x.dtype in (torch.float16, torch.bfloat16):
        try:
            from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton

            chip = chip_of(x)
            p = _portable(packed)
            xc = x.to(_compute_dtype(x, chip))
            m = xc.reshape(-1, xc.shape[-1]).shape[0]
            cfg = config or tuned_config("w4a16_gemm", chip, m, p.n, p.k,
                                         dtype_cfg_str=dtype_cfg("w4a16_gemm", xc.dtype, p.group_size, W4_TUNE_TAG),
                                         layout_id=p.layout_id)
            y = w4a16_gemm_triton(xc, p, bias, config=cfg, deterministic=deterministic, sm=chip.sm, num_sms=chip.num_sms,
                                  allow_fallback=True if (cfg and not config) else None)
            _note("w4a16_linear", "triton")
            return y.to(x.dtype)
        except Exception as e:
            _fail("w4a16_linear", e, fk)
    _note("w4a16_linear", "reference", why)
    return w4a16_gemm_ref(x, packed, bias, out_dtype=x.dtype)


def w8a16_linear(
    x: torch.Tensor,
    packed: PackedW8,
    bias: torch.Tensor | None = None,
    *,
    config: dict[str, Any] | None = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """``y = x @ dequant(Wq)^T + bias`` with int8 weights.  Triton kernel on CUDA, oracle elsewhere."""
    fk = _fkey("w8a16_linear", x, packed.n, packed.k)
    ok, why = triton_usable(x, fk)
    if ok and x.dtype in (torch.float16, torch.bfloat16):
        try:
            from forge.kernels.triton.gemm_w8a16 import w8a16_gemm_triton

            chip = chip_of(x)
            xc = x.to(_compute_dtype(x, chip))
            m = xc.reshape(-1, xc.shape[-1]).shape[0]
            cfg = config or tuned_config("w8a16_gemm", chip, m, packed.n, packed.k,
                                         dtype_cfg_str=dtype_cfg("w8a16_gemm", xc.dtype, packed.group_size), layout_id=packed.layout_id)
            y = w8a16_gemm_triton(xc, packed, bias, config=cfg, deterministic=deterministic, sm=chip.sm, num_sms=chip.num_sms)
            _note("w8a16_linear", "triton")
            return y.to(x.dtype)
        except Exception as e:
            _fail("w8a16_linear", e, fk)
    _note("w8a16_linear", "reference", why)
    return w8a16_gemm_ref(x, packed, bias, out_dtype=x.dtype)


# ------------------------------------------------------------------------------------------------------------------
# FP8
# ------------------------------------------------------------------------------------------------------------------
def _flat_scale(s: torch.Tensor, rows: int | None = None) -> torch.Tensor:
    s = s.float()
    if s.dim() == 2 and s.numel() > 1 and ((rows is not None and s.shape in ((rows, 1), (1, rows))) or (rows is None and s.shape[1] == 1)):
        return s.reshape(-1)  # [N,1] / [1,N] vectors
    return s


def quantize_act_fp8(
    x: torch.Tensor,
    *,
    granularity: str = "per_token",
    block: int = 128,
    scale: torch.Tensor | None = None,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float | None = None,
    fmt: str = "e4m3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic (or static ``scale=``) FP8 activation quantisation -> ``(q fp8, scale fp32)``.

    ``rms_eps`` set -> RMSNorm (times optional ``rms_weight``) is fused in front of the absmax (per-token only).
    Triton kernel on CUDA (E4M3, sm_89+ for the fp8 cast), oracle otherwise.
    """
    ok, why = triton_usable(x)
    if ok and fmt == "e4m3" and x.is_floating_point() and (not x.is_cuda or chip_of(x).has_fp8 or _interpret()):
        try:
            from forge.kernels.triton.gemm_fp8 import act_quant_fp8_triton

            r = act_quant_fp8_triton(x, granularity=granularity, block=block, scale=scale, rms_weight=rms_weight, rms_eps=rms_eps)
            _note("quantize_act_fp8", "triton")
            return r
        except Exception as e:
            _fail("quantize_act_fp8", e)
    _note("quantize_act_fp8", "reference", why)
    if rms_eps is not None:
        q, s, _ = rmsnorm_act_quant_fp8_ref(x, rms_weight, rms_eps, granularity=granularity, block=block, fmt=fmt, scale=scale)
        return q, s
    return quantize_fp8_ref(x, granularity, block=block, scale=scale, fmt=fmt)


def quantize_weight_fp8(w: torch.Tensor, granularity: str = "per_channel", block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline weight quantisation: ``per_tensor`` | ``per_channel`` | ``block2d`` -> ``(w_fp8 [N,K], scale)``."""
    return quantize_fp8_ref(w, granularity, block=block)


def fp8_linear(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    x_scale: torch.Tensor | None = None,
    act_granularity: str | None = None,
    block: int = 128,
    out_dtype: torch.dtype | None = None,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """``y = dq(quant(x)) @ dq(w)^T + bias``.

    ``x`` is a float tensor (dynamically quantised, default per-token; 1x128 blocks if ``w_scale`` is 2-D) or an
    already quantised fp8 tensor together with ``x_scale``.  Paths: native Triton fp8 ``tl.dot`` (sm_89+) ->
    ``torch._scaled_mm`` -> dequantise + 16-bit matmul (any GPU) -> fp32 oracle (CPU).
    """
    ws = _flat_scale(w_scale, w_fp8.shape[0])
    pre_quantised = x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    odt = out_dtype or (torch.bfloat16 if pre_quantised else x.dtype)
    if odt not in (torch.float16, torch.bfloat16, torch.float32):
        odt = torch.bfloat16
    gran = act_granularity or ("block" if ws.dim() == 2 else ("per_tensor" if ws.numel() == 1 and x_scale is not None and x_scale.numel() == 1 else "per_token"))
    if pre_quantised:
        if x_scale is None:
            raise ConfigError("pre-quantised fp8 activations need x_scale")
        xq, sx = x, _flat_scale(x_scale)
    else:
        xq, sx = quantize_act_fp8(x, granularity=gran, block=block, scale=x_scale)
    chip = chip_of(xq)
    fk = _fkey("fp8_linear", xq, w_fp8.shape[0], ws.dim(), sx.dim())
    ok, why = triton_usable(xq, fk)
    if ok and (chip.has_fp8 or _interpret()) and xq.dtype == torch.float8_e4m3fn and w_fp8.dtype == torch.float8_e4m3fn:
        try:
            from forge.kernels.triton.gemm_fp8 import fp8_gemm_triton

            y = fp8_gemm_triton(xq, sx, w_fp8, ws, bias, out_dtype=odt, block=block, config=config, sm=chip.sm)
            _note("fp8_linear", "triton")
            return y
        except Exception as e:
            _fail("fp8_linear", e, fk)
    if xq.is_cuda and chip.has_fp8 and not force_ref() and ws.dim() <= 1 and sx.dim() <= 1 and hasattr(torch, "_scaled_mm"):
        try:
            lead = xq.shape[:-1]
            a = xq.reshape(-1, xq.shape[-1]).contiguous()
            m = a.shape[0]
            sa = sx.reshape(1, 1) if sx.numel() == 1 else sx.reshape(m, 1)
            sb = ws.reshape(1, 1) if ws.numel() == 1 else ws.reshape(1, -1)
            y = torch._scaled_mm(a, w_fp8.t(), scale_a=sa.contiguous(), scale_b=sb.contiguous(), bias=bias.to(odt) if bias is not None else None, out_dtype=odt)
            y = y[0] if isinstance(y, tuple) else y
            _note("fp8_linear", "scaled_mm")
            return y.reshape(*lead, w_fp8.shape[0])
        except Exception as e:
            if kernel_strict():
                raise
            _warn_once("fp8_scaled_mm", f"fp8_linear: torch._scaled_mm failed ({type(e).__name__}: {e}); dequantising instead")
    if xq.is_cuda:  # dequantise -> 16-bit tensor-core matmul (weight-only FP8 storage win on any GPU)
        cdt = torch.float16 if (odt == torch.bfloat16 and not chip.has_bf16) else (odt if odt != torch.float32 else torch.float16)
        xd = dequantize_fp8_ref(xq, sx, block).to(cdt)
        wd = dequantize_fp8_ref(w_fp8, ws, block).to(cdt)
        y = F.linear(xd, wd, bias.to(cdt) if bias is not None else None)
        _note("fp8_linear", "dequant_matmul")
        return y.to(odt)
    _note("fp8_linear", "reference", why)
    return fp8_gemm_ref(xq, sx, w_fp8, ws, bias, block=block, out_dtype=odt)


# ------------------------------------------------------------------------------------------------------------------
# INT8 W8A8
# ------------------------------------------------------------------------------------------------------------------
def quantize_act_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 quantisation -> ``(q int8, scale fp32 [M])``."""
    ok, why = triton_usable(x)
    if ok and x.is_floating_point():
        try:
            from forge.kernels.triton.gemm_int8 import int8_quant_triton

            r = int8_quant_triton(x)
            _note("quantize_act_int8", "triton")
            return r
        except Exception as e:
            _fail("quantize_act_int8", e)
    _note("quantize_act_int8", "reference", why)
    return quantize_int8_ref(x, "per_token")


def int8_linear(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    x_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """SmoothQuant-style W8A8: ``x`` float (quantised per token on the fly) or int8 with ``x_scale``."""
    pre = x.dtype == torch.int8
    odt = out_dtype or (torch.bfloat16 if pre else x.dtype)
    if pre:
        if x_scale is None:
            raise ConfigError("int8 activations need x_scale")
        xq, sx = x, x_scale.float()
    else:
        xq, sx = quantize_act_int8(x)
    sw = _flat_scale(w_scale, w_int8.shape[0])
    fk = _fkey("int8_linear", xq, w_int8.shape[0])
    ok, why = triton_usable(xq, fk)
    if ok and (not xq.is_cuda or (chip_of(xq).sm or 0) >= 75):
        try:
            from forge.kernels.triton.gemm_int8 import int8_gemm_triton

            y = int8_gemm_triton(xq, sx, w_int8, sw, bias, out_dtype=odt, config=config, sm=chip_of(xq).sm)
            _note("int8_linear", "triton")
            return y
        except Exception as e:
            _fail("int8_linear", e, fk)
    _note("int8_linear", "reference", why)
    return int8_gemm_ref(xq, sx, w_int8, sw, bias, out_dtype=odt)


# ------------------------------------------------------------------------------------------------------------------
# FP4 (NVFP4 / MXFP4) -- experimental
# ------------------------------------------------------------------------------------------------------------------
def as_packed_fp4(w: Any) -> PackedFP4:
    """Accept :class:`PackedFP4` or a duck-typed ``forge.formats.fp4.NVFP4Tensor``."""
    if isinstance(w, PackedFP4):
        return w
    if hasattr(w, "payload_u8") and hasattr(w, "block_scales_e4m3"):
        return PackedFP4.from_nvfp4_tensor(w)
    raise ConfigError(f"cannot interpret {type(w).__name__} as a block-scaled FP4 tensor")


def quantize_act_fp4(
    x: torch.Tensor, fmt: str = "nvfp4", *, scale_layout: str = "logical", tensor_scale: float | torch.Tensor | None = None
) -> PackedFP4:
    """Fused FP4 quantiser (RTN) -> :class:`PackedFP4`.  Triton on CUDA (any sm: scale conversion is integer arithmetic)."""
    ok, why = triton_usable(x)
    if ok and x.is_floating_point():
        try:
            from forge.kernels.triton.gemm_fp4 import quantize_act_fp4_triton

            r = quantize_act_fp4_triton(x, fmt, scale_layout=scale_layout, tensor_scale=tensor_scale)
            _note("quantize_act_fp4", "triton")
            return r
        except Exception as e:
            _fail("quantize_act_fp4", e)
    _note("quantize_act_fp4", "reference", why)
    kw: dict[str, Any] = {"scale_layout": scale_layout}
    if tensor_scale is not None and fmt == "nvfp4":
        kw["tensor_scale"] = tensor_scale
    return quantize_fp4_ref(x, fmt, **kw)


def quantize_weight_fp4(w: torch.Tensor, fmt: str = "nvfp4", scale_layout: str = "logical") -> PackedFP4:
    """Offline weight quantisation (RTN, oracle quantiser)."""
    return quantize_fp4_ref(w, fmt, scale_layout=scale_layout)


_NATIVE_FP4_OK: dict[Any, bool] = {}


def _native_fp4_selftest(device: torch.device) -> bool:
    """One-time numerical check of the (unvalidated) ``tl.dot_scaled`` MXFP4 kernel against the oracle on a tiny problem.

    The block-scaled-MMA path is a *hypothesis* about the installed Triton + device (operand packing, scale layout and
    API changed between releases).  A kernel that compiles but computes garbage must never be used silently, so the
    first use on each device runs this check; any exception or mismatch disables the native path for the process."""
    try:
        from forge.kernels.triton.gemm_fp4 import mxfp4_gemm_native_triton

        g = torch.Generator().manual_seed(1234)
        x = torch.randn(32, 256, generator=g).to(device)
        w = torch.randn(128, 256, generator=g).to(device)
        xq, wq = quantize_fp4_ref(x, "mxfp4"), quantize_fp4_ref(w, "mxfp4")
        y = mxfp4_gemm_native_triton(xq, wq, None, out_dtype=torch.float32)
        ref = fp4_block_gemm_ref(xq, wq, None, out_dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        err = float((y.float() - ref).abs().max())
        scale = float(ref.abs().max()) + 1e-6
        ok = bool(torch.isfinite(y).all()) and err <= 1e-2 * scale
        if not ok:
            log.warning("native MXFP4 tl.dot_scaled self-test FAILED on %s (max abs err %.3g vs scale %.3g): "
                        "native path disabled, using the dequantise kernel", device, err, scale)
        return ok
    except Exception as e:  # noqa: BLE001
        log.warning("native MXFP4 tl.dot_scaled self-test raised %s: %s; native path disabled", type(e).__name__, e)
        return False


def native_fp4_available(x: torch.Tensor) -> bool:
    """True if ``tl.dot_scaled`` exists, the device reports block-scaled FP4 MMA (sm_100/120 hypothesis) AND the
    one-time numerical self-test of the native kernel passed on this device."""
    if not x.is_cuda or force_ref():
        return False
    try:
        from forge.kernels.triton.gemm_fp4 import mxfp4_dot_scaled_available

        if not (mxfp4_dot_scaled_available() and chip_of(x).has_fp4_block_scaled):
            return False
        key = (x.device.type, x.device.index)
        if key not in _NATIVE_FP4_OK:
            _NATIVE_FP4_OK[key] = _native_fp4_selftest(x.device)
        return _NATIVE_FP4_OK[key]
    except Exception:
        return False


def fp4_linear(
    x: torch.Tensor,
    w: Any,
    bias: torch.Tensor | None = None,
    *,
    act_quant: str = "none",
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """``y = x @ dq(W)^T + bias`` with FP4 weights.

    ``act_quant="none"``: weight-only (dequantise-in-kernel, any GPU).  ``act_quant="fp4"``: activations are
    quantised with the weight's format (fused quantiser); MXFP4 on a block-scaled-MMA device uses ``tl.dot_scaled``
    (experimental), everything else emulates W4A4 by dequantising the quantised activations (same numerics up to
    the rounding of ``e2m1*scale*tensor_scale`` in 16 bit).
    """
    wp = as_packed_fp4(w)
    if x.shape[-1] != wp.k:
        raise ConfigError(f"x has K={x.shape[-1]} but the FP4 weight has K={wp.k}")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if act_quant == "fp4":
        if wp.fmt == "mxfp4" and native_fp4_available(x2):
            try:
                from forge.kernels.triton.gemm_fp4 import mxfp4_gemm_native_triton

                xq = quantize_act_fp4(x2, "mxfp4")
                y = mxfp4_gemm_native_triton(xq, wp, bias, out_dtype=x.dtype)
                _note("fp4_linear", "triton_dot_scaled")
                return y.reshape(*lead, wp.rows)
            except Exception as e:
                _fail("fp4_linear", e)
        xq = quantize_act_fp4(x2, wp.fmt)
        x2 = dequantize_fp4_ref(xq, x.dtype if x.dtype != torch.float32 else torch.float32)
    fk = _fkey("fp4_linear", x2, wp.rows, wp.fmt)
    ok, why = triton_usable(x2, fk)
    if ok and x2.dtype in (torch.float16, torch.bfloat16):
        try:
            from forge.kernels.triton.gemm_fp4 import fp4_dequant_gemm_triton

            chip = chip_of(x2)
            xc = x2.to(_compute_dtype(x2, chip))
            y = fp4_dequant_gemm_triton(xc, wp, bias, config=config, sm=chip.sm)
            _note("fp4_linear", "triton_dequant" + ("+act_quant" if act_quant == "fp4" else ""))
            return y.to(x.dtype).reshape(*lead, wp.rows)
        except Exception as e:
            _fail("fp4_linear", e, fk)
    _note("fp4_linear", "reference", why)
    return fp4_block_gemm_ref(x2, wp, bias, out_dtype=x.dtype).reshape(*lead, wp.rows)


# ------------------------------------------------------------------------------------------------------------------
# packed linear modules + factory
# ------------------------------------------------------------------------------------------------------------------
def _w4_shapes(n: int, k: int, group_size: int, layout: Layout, has_zeros: bool, n_groups: int | None = None) -> dict[str, tuple[int, ...]]:
    from forge.kernels.layouts import effective_group
    from forge.kernels.layouts.bits import roundup

    g = n_groups if n_groups is not None else -(-k // effective_group(group_size, k))
    if layout.kind == "portable_row":
        n_pad, qshape = n, (n, roundup(k, 8) // 8)
    else:
        n_pad = roundup(n, 64)
        qshape = (roundup(k, 16) // 16, n_pad * 2)
    out = {"qweight": qshape, "scales": (g, n_pad)}
    if has_zeros:
        out["zeros"] = (g, roundup(n_pad, 8) // 8)
    return out


class _KernelLinear(_PackedBase):  # type: ignore[misc,valid-type]
    """Common plumbing of the kernel-backed packed linears (buffers + optional bias parameter)."""

    kind: ClassVar[str] = ""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype or torch.float32, device=device))
        else:
            self.register_parameter("bias", None)

    def _bias(self, x: torch.Tensor) -> torch.Tensor | None:
        return None if self.bias is None else self.bias.to(x.dtype)

    def quant_spec(self) -> dict[str, Any]:
        return {}

    def extra_repr(self) -> str:
        return f"kind={self.kind}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class KernelW4A16Linear(_KernelLinear):
    """4-bit weight-only linear backed by :func:`w4a16_linear` (portable_row or fragment layout)."""

    kind: ClassVar[str] = "w4a16"

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, *,
                 group_size: int = 128, sym: bool = True, layout: str | Layout | None = None, has_g_idx: bool = False, n_groups: int | None = None, **_: Any):
        super().__init__(in_features, out_features, bias, dtype, device)
        self.group_size, self.sym = int(group_size), bool(sym)
        self.layout = PORTABLE_W4 if layout is None else (get_layout(layout) if isinstance(layout, str) else layout)
        shapes = _w4_shapes(out_features, in_features, group_size, self.layout, not sym, n_groups)
        self.register_buffer("qweight", torch.zeros(shapes["qweight"], dtype=torch.int32, device=device))
        self.register_buffer("scales", torch.zeros(shapes["scales"], dtype=dtype or torch.float16, device=device))
        self.register_buffer("zeros", torch.zeros(shapes["zeros"], dtype=torch.int32, device=device) if not sym else None)
        self.register_buffer("g_idx", torch.zeros(in_features, dtype=torch.int32, device=device) if has_g_idx else None)
        self._packed_meta: dict[str, Any] = {}  # holds the (stamped) portable-repack cache across forward calls

    @classmethod
    def from_packed(cls, packed: PackedW4, bias: torch.Tensor | None = None) -> KernelW4A16Linear:
        m = cls(packed.k, packed.n, bias is not None, packed.scales.dtype, packed.qweight.device, group_size=packed.group_size,
                sym=packed.zeros is None, layout=packed.layout, has_g_idx=packed.g_idx is not None, n_groups=packed.num_groups)
        with torch.no_grad():
            m.qweight.copy_(packed.qweight)
            m.scales.copy_(packed.scales)
            if packed.zeros is not None:
                m.zeros.copy_(packed.zeros)
            if packed.g_idx is not None:
                m.g_idx.copy_(packed.g_idx)
            if bias is not None:
                m.bias.data = bias.detach().clone()
        return m

    @classmethod
    def from_dense(cls, linear: nn.Linear, *, group_size: int = 128, sym: bool = True, layout: str | Layout | None = None,
                   scale_dtype: torch.dtype = torch.float16, **_: Any) -> KernelW4A16Linear:
        w = linear.weight.detach()
        q, s, z = quantize_w4_rtn(w, group_size, sym=sym, scale_dtype=scale_dtype)
        return cls.from_packed(pack_w4(q, s, z, group_size, layout), linear.bias.detach() if linear.bias is not None else None)

    @classmethod
    def from_shape(cls, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, **kw: Any):
        return cls(in_features, out_features, bias, dtype, device, **kw)

    def packed(self) -> PackedW4:
        return PackedW4(self.qweight, self.scales, self.zeros, self.layout, self.out_features, self.in_features, self.group_size, self.g_idx,
                        self._packed_meta)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.packed().dequantize(dtype or torch.float32)

    def quant_spec(self) -> dict[str, Any]:
        return {"bits": 4, "group_size": self.group_size, "sym": self.sym, "layout": self.layout.id, "has_g_idx": self.g_idx is not None, "n_groups": int(self.scales.shape[0])}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w4a16_linear(x, self.packed(), self._bias(x))


class KernelW8A16Linear(_KernelLinear):
    """8-bit weight-only linear backed by :func:`w8a16_linear`."""

    kind: ClassVar[str] = "w8a16"

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, *,
                 group_size: int = -1, **_: Any):
        super().__init__(in_features, out_features, bias, dtype, device)
        from forge.kernels.layouts import num_groups

        self.group_size = int(group_size)
        g = num_groups(in_features, group_size)
        self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("scales", torch.zeros((g, out_features), dtype=dtype or torch.float16, device=device))

    @classmethod
    def from_packed(cls, packed: PackedW8, bias: torch.Tensor | None = None) -> KernelW8A16Linear:
        m = cls(packed.k, packed.n, bias is not None, packed.scales.dtype, packed.qweight.device, group_size=packed.group_size)
        with torch.no_grad():
            m.qweight.copy_(packed.qweight)
            m.scales.copy_(packed.scales)
            if bias is not None:
                m.bias.data = bias.detach().clone()
        return m

    @classmethod
    def from_dense(cls, linear: nn.Linear, *, group_size: int = -1, scale_dtype: torch.dtype = torch.float16, **_: Any) -> KernelW8A16Linear:
        q, s = quantize_w8_rtn(linear.weight.detach(), group_size, scale_dtype)
        return cls.from_packed(pack_w8(q, s, group_size), linear.bias.detach() if linear.bias is not None else None)

    @classmethod
    def from_shape(cls, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, **kw: Any):
        return cls(in_features, out_features, bias, dtype, device, **kw)

    def packed(self) -> PackedW8:
        return PackedW8(self.qweight, self.scales, PORTABLE_W8, self.out_features, self.in_features, self.group_size)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.packed().dequantize(dtype or torch.float32)

    def quant_spec(self) -> dict[str, Any]:
        return {"bits": 8, "group_size": self.group_size, "layout": PORTABLE_W8.id}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w8a16_linear(x, self.packed(), self._bias(x))


class KernelFp8Linear(_KernelLinear):
    """FP8 (E4M3) weights with per-channel (default) / per-tensor / 128x128-block scales and dynamic activation FP8."""

    kind: ClassVar[str] = "fp8"

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, *,
                 granularity: str = "per_channel", block: int = 128, **_: Any):
        super().__init__(in_features, out_features, bias, dtype, device)
        if granularity not in ("per_tensor", "per_channel", "block2d"):
            raise ConfigError(f"unknown fp8 weight granularity {granularity!r}")
        self.granularity, self.block = granularity, int(block)
        sshape = {"per_tensor": (1,), "per_channel": (out_features,),
                  "block2d": (-(-out_features // block), -(-in_features // block))}[granularity]
        self.register_buffer("weight_fp8", torch.zeros((out_features, in_features), dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("weight_scale", torch.ones(sshape, dtype=torch.float32, device=device))

    @classmethod
    def from_dense(cls, linear: nn.Linear, *, granularity: str = "per_channel", block: int = 128, **_: Any) -> KernelFp8Linear:
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.dtype, linear.weight.device,
                granularity=granularity, block=block)
        q, s = quantize_weight_fp8(linear.weight.detach(), granularity, block)
        with torch.no_grad():
            m.weight_fp8.copy_(q)
            m.weight_scale.copy_(s)
            if linear.bias is not None:
                m.bias.copy_(linear.bias.detach().to(m.bias.dtype))
        return m

    @classmethod
    def from_shape(cls, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, **kw: Any):
        return cls(in_features, out_features, bias, dtype, device, **kw)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_fp8_ref(self.weight_fp8, self.weight_scale, self.block).to(dtype or torch.float32)

    def quant_spec(self) -> dict[str, Any]:
        return {"granularity": self.granularity, "block": self.block}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_linear(x, self.weight_fp8, self.weight_scale, self._bias(x), block=self.block)


class KernelInt8Linear(_KernelLinear):
    """INT8 W8A8 (per-channel weight scale, dynamic per-token activation scale)."""

    kind: ClassVar[str] = "int8"

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, **_: Any):
        super().__init__(in_features, out_features, bias, dtype, device)
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32, device=device))

    @classmethod
    def from_dense(cls, linear: nn.Linear, **_: Any) -> KernelInt8Linear:
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.dtype, linear.weight.device)
        q, s = quantize_int8_ref(linear.weight.detach(), "per_channel")
        with torch.no_grad():
            m.weight_int8.copy_(q)
            m.weight_scale.copy_(s)
            if linear.bias is not None:
                m.bias.copy_(linear.bias.detach().to(m.bias.dtype))
        return m

    @classmethod
    def from_shape(cls, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, **kw: Any):
        return cls(in_features, out_features, bias, dtype, device, **kw)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return (self.weight_int8.float() * self.weight_scale.float().unsqueeze(1)).to(dtype or torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_linear(x, self.weight_int8, self.weight_scale, self._bias(x))


class KernelFp4Linear(_KernelLinear):
    """Block-scaled FP4 weights (``fmt`` = ``'nvfp4'`` | ``'mxfp4'``); experimental."""

    kind: ClassVar[str] = "nvfp4"

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, *,
                 fmt: str = "nvfp4", act_quant: str = "none", scale_layout: str = "logical", **_: Any):
        super().__init__(in_features, out_features, bias, dtype, device)
        from forge.kernels.layouts import FP4_BLOCK, padded_scale_shape
        from forge.kernels.layouts.bits import roundup

        if fmt not in FP4_BLOCK:
            raise ConfigError(f"unknown fp4 format {fmt!r}")
        self.fmt, self.act_quant, self.scale_layout = fmt, act_quant, scale_layout
        kp = roundup(in_features, FP4_BLOCK[fmt])
        nb = kp // FP4_BLOCK[fmt]
        sshape = (out_features, nb) if scale_layout == "logical" else (int(padded_scale_shape(out_features, nb)[0] * padded_scale_shape(out_features, nb)[1]),)
        self.register_buffer("payload", torch.zeros((out_features, kp // 2), dtype=torch.uint8, device=device))
        self.register_buffer("block_scales", torch.zeros(sshape, dtype=torch.uint8, device=device))
        self.register_buffer("tensor_scale", torch.ones((), dtype=torch.float32, device=device))

    @classmethod
    def from_dense(cls, linear: nn.Linear, *, fmt: str = "nvfp4", act_quant: str = "none", scale_layout: str = "logical", **_: Any) -> KernelFp4Linear:
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.dtype, linear.weight.device,
                fmt=fmt, act_quant=act_quant, scale_layout=scale_layout)
        p = quantize_weight_fp4(linear.weight.detach(), fmt, scale_layout)
        with torch.no_grad():
            m.payload.copy_(p.payload)
            m.block_scales.copy_(p.scales)
            m.tensor_scale.copy_(p.tensor_scale)
            if linear.bias is not None:
                m.bias.copy_(linear.bias.detach().to(m.bias.dtype))
        return m

    @classmethod
    def from_shape(cls, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype | None = None, device: Any = None, **kw: Any):
        return cls(in_features, out_features, bias, dtype, device, **kw)

    def packed(self) -> PackedFP4:
        return PackedFP4(self.fmt, self.payload, self.block_scales, self.tensor_scale, self.out_features, self.in_features, self.scale_layout)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_fp4_ref(self.packed(), dtype or torch.float32)

    def quant_spec(self) -> dict[str, Any]:
        return {"fmt": self.fmt, "act_quant": self.act_quant, "scale_layout": self.scale_layout}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp4_linear(x, self.packed(), self._bias(x), act_quant=self.act_quant)


PACKED_LINEAR_CLASSES: dict[str, type] = {
    "w4a16": KernelW4A16Linear,
    "w8a16": KernelW8A16Linear,
    "fp8": KernelFp8Linear,
    "int8": KernelInt8Linear,
    "nvfp4": KernelFp4Linear,
    "mxfp4": KernelFp4Linear,
}


def make_packed_linear(
    kind: str,
    linear: nn.Linear | None = None,
    *,
    packed: PackedW4 | PackedW8 | PackedFP4 | None = None,
    bias: torch.Tensor | None = None,
    in_features: int | None = None,
    out_features: int | None = None,
    **kw: Any,
) -> nn.Module:
    """Factory for kernel-backed packed linear layers (the models owner calls this lazily).

    * ``make_packed_linear("w4a16", linear, group_size=128, sym=True)``     RTN-quantise a dense ``nn.Linear``
    * ``make_packed_linear("w4a16", packed=pack_w4(...), bias=b)``          wrap pre-quantised data (GPTQ/AWQ output)
    * ``make_packed_linear("fp8", linear, granularity="per_channel")``, ``"int8"``, ``"w8a16"``,
      ``"nvfp4"`` / ``"mxfp4"`` (``act_quant="none"|"fp4"``)
    * ``make_packed_linear("w4a16", in_features=.., out_features=.., group_size=..)``  empty module for ``load_state_dict``
    """
    cls = PACKED_LINEAR_CLASSES.get(kind)
    if cls is None:
        raise ConfigError(f"unknown packed linear kind {kind!r}; choose from {sorted(PACKED_LINEAR_CLASSES)}")
    if kind in ("nvfp4", "mxfp4"):
        kw.setdefault("fmt", kind)
    if packed is not None:
        if isinstance(packed, PackedW4) and cls is KernelW4A16Linear:
            return cls.from_packed(packed, bias)
        if isinstance(packed, PackedW8) and cls is KernelW8A16Linear:
            return cls.from_packed(packed, bias)
        if isinstance(packed, PackedFP4) and cls is KernelFp4Linear:
            m = cls(packed.k, packed.rows, bias is not None, None, packed.payload.device, fmt=packed.fmt,
                    scale_layout=packed.scale_layout, act_quant=kw.get("act_quant", "none"))
            with torch.no_grad():
                m.payload.copy_(packed.payload)
                m.block_scales.copy_(packed.scales)
                m.tensor_scale.copy_(packed.tensor_scale)
                if bias is not None:
                    m.bias.data = bias.detach().clone()
            return m
        raise ConfigError(f"packed data of type {type(packed).__name__} does not match kind {kind!r}")
    if linear is not None:
        return cls.from_dense(linear, **kw)
    if in_features is None or out_features is None:
        raise ConfigError("give a dense `linear`, `packed` data, or in_features/out_features")
    return cls.from_shape(in_features, out_features, bias is not None, **kw)


def register_packed_linears(prefix: str = "kernel_") -> list[str]:
    """Optionally register the kernel-backed classes with ``forge.models.linear`` under ``<prefix><kind>``
    (kept explicit: ``forge.compress`` already owns the plain ``w4a16`` / ``nvfp4`` names)."""
    from forge.models.linear import register_linear

    done = []
    for kind, cls in PACKED_LINEAR_CLASSES.items():
        if kind == "mxfp4":
            continue  # same class as nvfp4 (fmt kwarg)
        name = f"{prefix}{kind}"
        sub = type(cls.__name__, (cls,), {})
        register_linear(name)(sub)
        done.append(name)
    return done


# ------------------------------------------------------------------------------------------------------------------
# benchmark registration (consumed by forge.kernels.bench)
# ------------------------------------------------------------------------------------------------------------------
def bench_cases() -> list[dict[str, Any]]:
    """Benchmark cases of the GEMM family (see :class:`forge.kernels.bench.BenchCase`)."""
    from forge.kernels.bench import gemm_bench_cases

    return gemm_bench_cases()


__all__ = [
    "KernelFp4Linear", "KernelFp8Linear", "KernelInt8Linear", "KernelW4A16Linear", "KernelW8A16Linear",
    "PACKED_LINEAR_CLASSES", "as_packed_fp4", "bench_cases", "chip_of", "clear_config_memo", "dtype_cfg", "force_ref",
    "fp4_linear", "fp8_linear", "int8_linear", "kernel_debug", "last_path", "make_packed_linear", "native_fp4_available",
    "quantize_act_fp4", "quantize_act_fp8", "quantize_act_int8", "quantize_weight_fp4", "quantize_weight_fp8",
    "register_packed_linears", "reset_failure_memo", "triton_usable", "tuned_config", "w4a16_linear", "w8a16_linear",
]
