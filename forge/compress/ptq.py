"""Post-training quantisation: RTN, GPTQ, AWQ-style scale search, SmoothQuant W8A8, MR-GPTQ-style FP4 (Plan CP-1..CP-3, CP-5, CP-6).

Methods (``PTQConfig.method``)
------------------------------
``rtn``
    Round-to-nearest, no calibration (the baseline every other method is judged against).
``gptq``
    Layer-wise second-order reconstruction, exactly the algorithm of Frantar et al.: ``H = 2/n X X^T`` accumulated over the calibration set
    with the activations produced by the *already quantised* earlier layers (sequential, block by block, optionally "true sequential"
    inside a block), dead-column handling, optional *act-order* (columns processed by decreasing ``H_ii``), percentage damping with retry,
    Cholesky factor of ``H^-1`` (``cholesky -> cholesky_inverse -> cholesky(upper)``), and **lazy block updates** (errors are propagated
    inside a block of ``block_size`` columns and applied to the trailing columns once per block).  Groups are contiguous in the original
    column order; with act-order the group parameters are computed up-front from the original weights ("static groups") so the stored
    layout needs no ``g_idx`` and stays kernel-compatible.
``awq``
    Activation-aware per-input-channel scaling: ``s = mean|x|^alpha`` (normalised), ``alpha`` grid-searched per input-sharing layer group
    to minimise ``|| X W^T - (X/s) Q(W diag(s))^T ||``; the scale is folded into the preceding RMSNorm / ``v_proj`` / ``up_proj`` when the
    structure allows (exact), otherwise kept as an ``input_scale`` buffer; then RTN with optional MSE clip search.
``smoothquant``
    W8A8: ``s_j = max|X_j|^a / max|W_j|^(1-a)`` migrates activation outliers into the weights (folded like AWQ), weights int8 per output
    channel, activations dynamic per-token int8 (``Int8W8A8Linear``; emulated, no int8 GEMM kernel is wired).
``mrgptq``
    Format-aware GPTQ for block-scaled FP4 (NVFP4 / MXFP4): the GPTQ loop with the E2M1 grid + block scale as the column quantiser (block
    scale from the *current* error-compensated weights), optional ``four_over_six`` per-block scale choice, optional block-Hadamard
    rotation of the weight/Hessian ("micro-rotation", applied online to activations at run time).  ``gptq`` with ``format`` nvfp4/mxfp4
    is the same code path.  Plan note (P5): RHT does not lower RTN error for NVFP4 -- rotation is off by default.

Everything here runs on CPU and CUDA with plain torch; no kernel is required.  GPU numerics are *not validated on hardware*.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.compress.calibrate import get_calibration_tokens
from forge.compress.modelutil import (
    DEFAULT_EXCLUDE,
    BlockInputs,
    ChannelStats,
    HessianAccumulator,
    SampleBuffer,
    capture_block_inputs,
    discover_input_groups,
    find_blocks,
    gemma4_features,
    iter_dense_linears,
    model_device,
    model_dtype,
    model_nbytes,
    run_with_hooks,
    sequential_stages,
    set_submodule,
)
from forge.compress.packed import (
    Int8W8A8Linear,
    IntLinear,
    MXFP4Linear,
    NVFP4Linear,
    apply_block_hadamard,
)
from forge.core.context import NullContext, RunContext
from forge.core.errors import ConfigError, ForgeError
from forge.formats.fp4 import (
    E2M1_MAX,
    E4M3_MAX,
    NVFP4Tensor,
    decode_e2m1,
    e8m0_scale,
    encode_e2m1,
    pack_fp4,
    quantize_mxfp4,
    quantize_nvfp4,
)
from forge.formats.fp8 import round_e4m3
from forge.formats.int_quant import compute_qparams, mse_clip_search, quantize_int

log = logging.getLogger("forge.compress.ptq")


# ------------------------------------------------------------------------------------------------ config
class PTQConfig(BaseModel):
    """Post-training quantisation settings (stage ``ptq``)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["rtn", "gptq", "awq", "smoothquant", "mrgptq"] = Field(
        "gptq", description="Algorithm. rtn = baseline, gptq = Hessian reconstruction, awq = activation-aware scaling + RTN, "
        "smoothquant = W8A8 with outlier migration, mrgptq = format-aware GPTQ for NVFP4/MXFP4."
    )
    format: Literal["int", "fp8", "nvfp4", "mxfp4"] = Field(
        "int", description="Weight format: int (group-wise INT, W{bits}A16), fp8 (per-channel FP8 E4M3 W8A8), nvfp4 (16-block E4M3 scales), mxfp4 (32-block E8M0)."
    )
    bits: int = Field(4, ge=2, le=8, description="Weight bits for format=int (2..8).")
    group_size: int = Field(128, description="Quantisation group size along the input dim (format=int). 0 or -1 = one group per output channel.")
    sym: bool = Field(True, description="Symmetric quantisation (zero point 2^(bits-1)); false = asymmetric with per-group zero points.")
    act_order: bool = Field(False, description="GPTQ: process columns by decreasing Hessian diagonal (better accuracy; uses static groups).")
    damp: float = Field(0.01, gt=0, le=1.0, description="GPTQ: Hessian damping as a fraction of the mean diagonal (retried x10 if Cholesky fails).")
    block_size: int = Field(128, ge=1, description="GPTQ: lazy-batch block size (columns updated together).")
    static_groups: bool | None = Field(None, description="GPTQ: compute group parameters up-front from the original weights. None = on iff act_order.")
    true_sequential: Literal["auto", "full", "block"] = Field(
        "auto", json_schema_extra={"x-advanced": True},
        description="GPTQ: quantise input-sharing layer groups one after another inside a block (full), all layers of a block in one pass (block), "
        "or full except MoE experts collapsed into two stages (auto).",
    )
    clip_search: bool = Field(False, description="Search a per-group clipping ratio by MSE (RTN/AWQ; GPTQ uses it for the initial scales).")
    n_seqs: int = Field(128, ge=1, description="Calibration sequences.")
    seq_len: int = Field(512, ge=8, description="Calibration sequence length (tokens).")
    batch_size: int = Field(4, ge=1, description="Calibration batch size.")
    seed: int = Field(0, description="Calibration sampling seed.")
    exclude: list[str] = Field(list(DEFAULT_EXCLUDE), description="Regexes of module names that stay in high precision (lm_head, MoE routers, embeddings).")
    include: list[str] | None = Field(None, description="If set, only modules whose name matches one of these regexes are quantised.")
    per_layer_bits: dict[str, int] = Field(default_factory=dict, description="Regex -> bits overrides (mixed precision), first match wins.")
    # mixed-precision allocation
    mixed_precision: bool = Field(False, description="Allocate bits per layer from Hessian sensitivity under a byte budget (needs avg_bits or byte_budget).")
    avg_bits: float | None = Field(None, gt=0, description="Mixed precision: target average bits per quantised weight including scale overhead.")
    byte_budget: int | None = Field(None, ge=1, description="Mixed precision: byte budget for all quantised weights (overrides avg_bits).")
    candidate_bits: list[int] = Field([3, 4, 8], description="Mixed precision: candidate bit widths per layer.")
    # awq
    awq_n_grid: int = Field(20, ge=2, json_schema_extra={"x-advanced": True}, description="AWQ: points of the alpha grid.")
    awq_samples: int = Field(1024, ge=16, json_schema_extra={"x-advanced": True}, description="AWQ: activation rows per layer group used in the scale search.")
    fold_scales: bool = Field(True, description="AWQ/SmoothQuant: fold channel scales into the preceding norm / v_proj / up_proj when possible.")
    # smoothquant
    sq_alpha: float = Field(0.5, ge=0.0, le=1.0, description="SmoothQuant migration strength alpha.")
    act_quant: Literal["auto", "none", "dynamic"] = Field(
        "auto", description="Activation quantisation of the produced modules: auto = dynamic per-token for W8A8 (smoothquant, fp8) and weight-only for INT/FP4; "
        "dynamic = also fake-quantise activations of NVFP4/MXFP4 modules (W4A4 emulation); none = weight-only."
    )
    # fp4
    four_over_six: bool = Field(False, description="NVFP4: per block choose block-max -> 6 or 4 by lowest MSE ('Four Over Six').")
    mx_scale_mode: Literal["ceil", "floor", "near"] = Field("ceil", description="MXFP4: E8M0 scale rounding (ceil never saturates).")
    rotate: Literal["none", "block_hadamard"] = Field("none", description="MR-GPTQ: block-Hadamard rotation of weights/activations (block = format block size, or group size for INT).")
    offload_activations: bool = Field(False, json_schema_extra={"x-advanced": True}, description="Keep stored calibration activations on the CPU.")

    @model_validator(mode="after")
    def _check_combo(self) -> PTQConfig:
        if self.method == "awq" and self.format != "int":
            raise ValueError("method 'awq' searches integer group scales: use format='int' (for FP4 use method gptq/mrgptq)")
        if self.method == "smoothquant" and self.format not in ("int",):
            raise ValueError("method 'smoothquant' produces INT8 W8A8; leave format='int' (FP8 W8A8 is format='fp8' with method rtn)")
        if self.mixed_precision and self.format != "int":
            raise ValueError("mixed_precision allocates integer bit widths: use format='int'")
        return self

    def uses_calibration(self) -> bool:
        return self.method in ("gptq", "awq", "smoothquant", "mrgptq") or self.mixed_precision

    def resolved_act_quant(self) -> str:
        if self.act_quant != "auto":
            return self.act_quant
        return "dynamic" if self.method == "smoothquant" or self.format == "fp8" else "none"


# ------------------------------------------------------------------------------------------------ results
@dataclass
class QuantizedWeight:
    """Result of quantising one weight matrix. ``deq`` is the float32 dequantised weight *in the transformed space* (after scaling/rotation)."""

    fmt: str
    deq: torch.Tensor
    bits: int = 4
    group_size: int = 128
    sym: bool = True
    q: torch.Tensor | None = None  # uint8 codes [N,K] (int format)
    scales: torch.Tensor | None = None
    zeros: torch.Tensor | None = None
    nvfp4: NVFP4Tensor | None = None
    mx_payload: torch.Tensor | None = None
    mx_scales: torch.Tensor | None = None
    input_scale: torch.Tensor | None = None
    rot_block: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class LayerRecord:
    name: str
    kind: str
    bits: float
    group_size: int
    shape: tuple[int, int]
    rel_err: float
    proxy_loss: float | None
    seconds: float
    bytes_before: int
    bytes_after: int
    notes: list[str] = field(default_factory=list)


@dataclass
class QuantReport:
    """What :func:`quantize_model` did. ``to_dict()`` is JSON-able (stored in the artifact manifest ``extra``)."""

    method: str
    config: dict[str, Any]
    layers: list[LayerRecord] = field(default_factory=list)
    seconds: float = 0.0
    bytes_before: int = 0
    bytes_after: int = 0
    calib: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def quantized_params(self) -> int:
        return sum(r.shape[0] * r.shape[1] for r in self.layers if r.kind != "dense")

    @property
    def avg_bits(self) -> float:
        n = self.quantized_params
        return 8.0 * sum(r.bytes_after for r in self.layers if r.kind != "dense") / n if n else 16.0

    @property
    def compression(self) -> float:
        return self.bytes_before / max(self.bytes_after, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "config": self.config, "seconds": self.seconds, "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after, "compression": self.compression, "avg_bits_quantized": self.avg_bits,
            "n_layers": len(self.layers), "calib": self.calib, "warnings": self.warnings,
            "layers": [asdict(r) for r in self.layers],
        }


@dataclass
class QuantizedModel:
    """``model`` (modified in place and returned here) + the :class:`QuantReport`."""

    model: nn.Module
    report: QuantReport

    @property
    def info(self) -> QuantReport:
        return self.report

    def save(self, out_dir: str, **kw: Any) -> str:
        from forge.compress.serialize import save_quantized

        return str(save_quantized(self.model, out_dir, report=self.report.to_dict(), **kw))


# ------------------------------------------------------------------------------------------------ GPTQ core
def _chol_inv_upper(H: torch.Tensor, damp: float) -> torch.Tensor:
    mean_diag = torch.mean(torch.diag(H)).clamp(min=1e-12)
    d = damp
    last: Exception | None = None
    for _ in range(7):
        Hd = H.clone()
        Hd.diagonal().add_(d * mean_diag)
        try:
            L = torch.linalg.cholesky(Hd)
            return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
        except Exception as e:  # not positive definite -> damp more
            last = e
            d *= 10.0
    raise ForgeError(f"Hessian not positive definite even with damping {d:g}: {last}")


def gptq_core(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    group: int,
    block_size: int,
    act_order: bool,
    damp: float,
    static_groups: bool,
    params_fn: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    quant_fn: Callable[[torch.Tensor, tuple[torch.Tensor, ...]], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, float]:
    """Generic GPTQ loop.

    ``params_fn(W_slice [N, G]) -> tuple of [N] tensors`` computes per-row group parameters; ``quant_fn(w [N], params) -> (code [N] uint8,
    dequantised [N])`` quantises one column.  Returns ``(codes [N,K] in original column order, params stacked as [N, n_groups] each,
    dequantised weight [N,K], proxy loss sum)``.  The proxy loss is GPTQ's ``sum err^2 / 2`` (diagnostic).
    """
    W = W.float().clone()
    H = H.float().clone()
    n, k = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    g = k if group is None or group <= 0 or group > k else int(group)
    if k % g:
        raise ConfigError(f"K={k} not divisible by group size {g}")
    ngroups = k // g
    static = static_groups or act_order
    static_params = [params_fn(W[:, j * g : (j + 1) * g]) for j in range(ngroups)] if static else None
    perm_list: list[int] | None = None
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        perm_list = perm.tolist()
    U = _chol_inv_upper(H, damp)
    del H
    bs = max(1, int(block_size))
    bs = int(math.ceil(bs / g) * g) if g <= bs else g  # a block always holds whole groups
    codes = torch.zeros(n, k, dtype=torch.uint8, device=W.device)
    deq = torch.zeros(n, k, dtype=torch.float32, device=W.device)
    dyn: dict[int, tuple[torch.Tensor, ...]] = {}
    loss = 0.0
    for i1 in range(0, k, bs):
        i2 = min(i1 + bs, k)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        cur: tuple[torch.Tensor, ...] | None = None
        for i in range(cnt):
            col = i1 + i
            if static_params is not None:
                orig = perm_list[col] if perm_list is not None else col
                p = static_params[orig // g]
            else:
                if col % g == 0:
                    cur = params_fn(W1[:, i : i + g] if i + g <= cnt else W[:, col : col + g])
                    dyn[col // g] = cur
                p = cur  # type: ignore[assignment]
            w = W1[:, i]
            d = U1[i, i]
            c, dq = quant_fn(w, p)
            codes[:, col] = c
            deq[:, col] = dq
            err = (w - dq) / d
            loss += float((err * err).sum()) * 0.5
            W1[:, i:] -= err.unsqueeze(1) * U1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        if i2 < k:
            W[:, i2:] -= Err1 @ U[i1:i2, i2:]
    plist = static_params if static_params is not None else [dyn[j] for j in range(ngroups)]
    stacked = tuple(torch.stack([p[j] for p in plist], dim=1) for j in range(len(plist[0])))
    if perm is not None:
        inv = torch.argsort(perm)
        codes, deq = codes[:, inv], deq[:, inv]
    return codes, stacked, deq, loss


# -- column quantisers ---------------------------------------------------------------------------
def _int_quantizers(bits: int, sym: bool, sdtype: torch.dtype, clip_search: bool):
    maxq = float((1 << bits) - 1)

    def params_fn(ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xg = ws.unsqueeze(1)  # [N,1,G]
        ratio: Any = 1.0
        if clip_search:
            ratio = mse_clip_search(xg, bits, sym)
        s, z = compute_qparams(xg, bits, sym, ratio)
        s = s.squeeze(1).squeeze(-1).to(sdtype).float()  # stored scale dtype: codes must be consistent with what is saved
        s = torch.where(s > 0, s, torch.ones_like(s))
        return s, z.squeeze(1).squeeze(-1)

    def quant_fn(w: torch.Tensor, p: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        s, z = p
        q = torch.clamp(torch.round(w / s) + z, 0, maxq)
        return q.to(torch.uint8), (q - z) * s

    return params_fn, quant_fn


def _nvfp4_quantizers(s_dec: torch.Tensor, s_enc: torch.Tensor, four_over_six: bool):
    """Column quantiser for NVFP4 with fixed tensor scale. params = (stored block scale,) float32 values of the E4M3 scale."""

    def stored_for(amax: torch.Tensor, m: float) -> torch.Tensor:
        return round_e4m3((amax / m) * s_enc)

    def params_fn(ws: torch.Tensor) -> tuple[torch.Tensor]:
        amax = ws.abs().amax(dim=1)
        st6 = stored_for(amax, E2M1_MAX)
        if not four_over_six:
            return (st6,)
        st4 = stored_for(amax, 4.0)

        def err(st: torch.Tensor) -> torch.Tensor:
            sc = (st * s_dec).clamp(min=1e-30).unsqueeze(1)
            dq = decode_e2m1(encode_e2m1(ws / sc)) * sc
            dq = torch.where(st.unsqueeze(1) > 0, dq, torch.zeros_like(dq))
            return (dq - ws).pow(2).sum(1)

        return (torch.where(err(st4) < err(st6), st4, st6),)

    def quant_fn(w: torch.Tensor, p: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        st = p[0]
        sc = (st * s_dec).clamp(min=1e-30)
        code = encode_e2m1(w / sc)
        dq = decode_e2m1(code) * sc
        ok = st > 0
        return torch.where(ok, code, torch.zeros_like(code)), torch.where(ok, dq, torch.zeros_like(dq))

    return params_fn, quant_fn


def _mxfp4_quantizers(mode: str):
    def params_fn(ws: torch.Tensor) -> tuple[torch.Tensor]:
        amax = ws.abs().amax(dim=1)
        if mode == "floor":
            s = torch.pow(2.0, (torch.floor(torch.log2(amax.clamp(min=2.0**-126))) - 2.0).clamp(-127, 127))
        else:
            s = e8m0_scale(amax / E2M1_MAX, mode)
        return (torch.where(amax == 0, torch.ones_like(s), s),)

    def quant_fn(w: torch.Tensor, p: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        s = p[0]
        code = encode_e2m1(w / s)
        return code, decode_e2m1(code) * s

    return params_fn, quant_fn


# ------------------------------------------------------------------------------------------------ per-weight quantisation
def _largest_divisor_group(k: int, g: int) -> int:
    if g <= 0 or g >= k:
        return k
    for cand in range(g, 0, -1):
        if k % cand == 0 and cand >= 8:
            return cand
    return k


def _rotation_block(cfg: PTQConfig, k: int) -> int:
    if cfg.rotate != "block_hadamard":
        return 0
    b = 16 if cfg.format == "nvfp4" else 32 if cfg.format == "mxfp4" else (cfg.group_size if cfg.group_size > 0 else 64)
    while b > 1 and (k % b or b & (b - 1)):
        b //= 2
    return b if b > 1 else 0


def quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor | None,
    cfg: PTQConfig,
    *,
    bits: int | None = None,
    sdtype: torch.dtype = torch.float16,
) -> QuantizedWeight:
    """Quantise one ``[N, K]`` weight under ``cfg`` using Hessian ``H`` ([K,K]) when the method needs it (None -> RTN)."""
    W = W.float()
    n, k = W.shape
    bits = int(bits or cfg.bits)
    fmt = cfg.format
    use_hessian = cfg.method in ("gptq", "mrgptq") and H is not None
    notes: list[str] = []
    rot = _rotation_block(cfg, k) if cfg.method in ("gptq", "mrgptq", "rtn") else 0
    if rot:
        W = apply_block_hadamard(W, rot)
        if H is not None:
            H = apply_block_hadamard(apply_block_hadamard(H.float(), rot).t(), rot).t()
    if fmt == "int":
        g = _largest_divisor_group(k, cfg.group_size)
        if cfg.group_size > 0 and g != cfg.group_size:
            notes.append(f"group_size {cfg.group_size} does not divide K={k}; used {g}")
        if use_hessian:
            pf, qf = _int_quantizers(bits, cfg.sym, sdtype, cfg.clip_search)
            codes, (s, z), deq, _ = gptq_core(
                W, H, group=g, block_size=cfg.block_size, act_order=cfg.act_order, damp=cfg.damp,
                static_groups=bool(cfg.static_groups) if cfg.static_groups is not None else cfg.act_order, params_fn=pf, quant_fn=qf,
            )
            q, scales, zeros = codes, s.to(sdtype), z.to(torch.uint8)
        else:
            q, scales, zeros = quantize_int(W, bits, g, cfg.sym, clip_search=cfg.clip_search)
            scales = scales.to(sdtype)
            # re-quantise with the *stored* (rounded) scales so codes are consistent
            q, scales, zeros = _requantize_with_stored_scales(W, scales, zeros, bits, g)
        from forge.formats.int_quant import dequantize_int

        deq = dequantize_int(q, scales.float(), zeros, g, bits=bits)
        if rot:
            deq = apply_block_hadamard(deq, rot)  # effective weight in the original input space (Hadamard is symmetric orthonormal)
        return QuantizedWeight("int", deq, bits, g, cfg.sym, q=q, scales=scales, zeros=None if cfg.sym else zeros, rot_block=rot, notes=notes)
    if fmt in ("nvfp4", "mxfp4"):
        block = 16 if fmt == "nvfp4" else 32
        if k % block:
            notes.append(f"K={k} not a multiple of {block}: RTN with zero padding instead of GPTQ")
            use_hessian = False
        if fmt == "nvfp4":
            if use_hessian:
                gamax = W.abs().amax().clamp(min=1e-30)
                m_t = 256.0 if cfg.four_over_six else E4M3_MAX
                s_enc = (E2M1_MAX * m_t) / gamax
                s_dec = 1.0 / s_enc
                pf, qf = _nvfp4_quantizers(s_dec, s_enc, cfg.four_over_six)
                codes, (stored,), deq, _ = gptq_core(
                    W, H, group=block, block_size=cfg.block_size, act_order=cfg.act_order, damp=cfg.damp,
                    static_groups=bool(cfg.static_groups) if cfg.static_groups is not None else cfg.act_order, params_fn=pf, quant_fn=qf,
                )
                t = NVFP4Tensor(pack_fp4(codes), stored.to(torch.float8_e4m3fn), s_dec.reshape(()).float(), (n, k), 16)
            else:
                t = quantize_nvfp4(W, four_over_six=cfg.four_over_six)
            d4 = t.dequantize(torch.float32)
            return QuantizedWeight("nvfp4", apply_block_hadamard(d4, rot) if rot else d4, 4, 16, True, nvfp4=t, rot_block=rot, notes=notes)
        if use_hessian:
            pf, qf = _mxfp4_quantizers(cfg.mx_scale_mode)
            codes, (s,), deq, _ = gptq_core(
                W, H, group=block, block_size=cfg.block_size, act_order=cfg.act_order, damp=cfg.damp,
                static_groups=bool(cfg.static_groups) if cfg.static_groups is not None else cfg.act_order, params_fn=pf, quant_fn=qf,
            )
            payload = pack_fp4(codes)
            scales_u8 = (torch.log2(s).round() + 127).clamp(0, 254).to(torch.uint8)
        else:
            t = quantize_mxfp4(W, scale_mode=cfg.mx_scale_mode)
            payload, scales_u8 = t.payload_u8, t.scales_e8m0_u8
        from forge.formats.fp4 import MXFP4Tensor, dequantize_mxfp4

        deq = dequantize_mxfp4(MXFP4Tensor(payload, scales_u8, (n, k), 32))
        if rot:
            deq = apply_block_hadamard(deq, rot)
        return QuantizedWeight("mxfp4", deq, 4, 32, True, mx_payload=payload, mx_scales=scales_u8, rot_block=rot, notes=notes)
    raise ConfigError(f"quantize_weight: unsupported format {fmt!r}")


def _requantize_with_stored_scales(W, scales, zeros, bits, g):
    """RTN codes for given (already rounded) group scales / zero points."""
    n, k = W.shape
    maxq = (1 << bits) - 1
    s = scales.float().repeat_interleave(g, dim=1)
    z = zeros.float().repeat_interleave(g, dim=1)
    q = torch.clamp(torch.round(W / s) + z, 0, maxq).to(torch.uint8)
    return q, scales, zeros


def quantize_layer(lin: nn.Module, W: torch.Tensor, H: torch.Tensor | None, cfg: PTQConfig, *, bits: int | None, sdtype: torch.dtype) -> QuantizedWeight:
    """:func:`quantize_weight` that honours an ``online_rot_block`` attribute of ``lin`` (``rotation.OnlineHadamardLinear``): the weight already
    contains the rotation, the Hessian is rotated to match and the packed module re-applies the rotation to its inputs."""
    orb = int(getattr(lin, "online_rot_block", 0) or 0)
    if not orb:
        return quantize_weight(W, H, cfg, bits=bits, sdtype=sdtype)
    if H is not None:
        H = apply_block_hadamard(apply_block_hadamard(H.float(), orb).t(), orb).t()
    qw = quantize_weight(W, H, cfg.model_copy(update={"rotate": "none"}), bits=bits, sdtype=sdtype)
    qw.rot_block = orb
    qw.deq = apply_block_hadamard(qw.deq, orb)
    qw.notes.append(f"online Hadamard block {orb}")
    return qw


def build_module(qw: QuantizedWeight, lin: nn.Module, cfg: PTQConfig, dtype: torch.dtype) -> nn.Module:
    """Turn a :class:`QuantizedWeight` into a packed module with ``lin``'s bias."""
    bias = getattr(lin, "bias", None)
    if qw.fmt == "int":
        return IntLinear.from_codes(
            qw.q, qw.scales, qw.zeros, bits=qw.bits, group_size=qw.group_size, bias=bias, dtype=dtype, input_scale=qw.input_scale,
            rot_block=qw.rot_block, sym=qw.sym,
        )
    act = cfg.resolved_act_quant()
    if qw.fmt == "nvfp4":
        return NVFP4Linear.from_tensor(qw.nvfp4, bias, act_quant=act, input_scale=qw.input_scale, rot_block=qw.rot_block, dtype=dtype)
    if qw.fmt == "mxfp4":
        return MXFP4Linear.from_parts(qw.mx_payload, qw.mx_scales, tuple(qw.deq.shape), bias, act_quant=act,
                                      input_scale=qw.input_scale, rot_block=qw.rot_block, dtype=dtype)
    raise ConfigError(f"cannot build module for {qw.fmt}")


def _make_fp8_module(lin: nn.Linear, cfg: PTQConfig) -> nn.Module:
    try:
        from forge.models.linear import Fp8Linear
    except Exception as e:  # pragma: no cover
        raise ConfigError(f"format=fp8 needs forge.models.linear.Fp8Linear ({e})") from e
    return Fp8Linear.from_dense(lin, act_quant="dynamic" if cfg.resolved_act_quant() == "dynamic" else "none")


# ------------------------------------------------------------------------------------------------ AWQ / SmoothQuant helpers
def _norm_modules(block: nn.Module) -> dict[str, nn.Module]:
    return {n: m for n, m in block.named_modules() if "norm" in type(m).__name__.lower() and getattr(m, "weight", None) is not None
            and getattr(m, "weight").ndim == 1}


def _find_producer_norm(block: nn.Module, binp: BlockInputs, first_linear: nn.Module) -> nn.Module | None:
    """The norm module whose output *is* the input of ``first_linear`` (same storage), or None."""
    norms = _norm_modules(block)
    if not norms:
        return None
    outs: dict[int, nn.Module] = {}
    handles = []
    inp_ptr: list[int] = []
    keepalive: list[torch.Tensor] = []  # tensors stay alive so a recycled address can never be mistaken for a norm output

    def norm_hook(mod: nn.Module, _i: tuple, o: torch.Tensor) -> None:  # hooks must return None (a value would replace the module output)
        keepalive.append(o)
        outs[o.data_ptr()] = mod

    def lin_hook(_mod: nn.Module, i: tuple) -> None:
        keepalive.append(i[0])
        inp_ptr.append(i[0].data_ptr())

    for _n, m in norms.items():
        handles.append(m.register_forward_hook(norm_hook))
    handles.append(first_linear.register_forward_pre_hook(lin_hook))
    try:
        with torch.no_grad():
            binp.run(block, 0)
    finally:
        for h in handles:
            h.remove()
    return outs.get(inp_ptr[0]) if inp_ptr else None


def _scale_norm(norm: nn.Module, s: torch.Tensor) -> None:
    """Divide the output of an RMSNorm/LayerNorm by ``s`` (folds ``X -> X/s``)."""
    w = norm.weight
    with torch.no_grad():
        if getattr(norm, "unit_offset", False):
            w.copy_(((1.0 + w.float()) / s - 1.0).to(w.dtype))
        else:
            w.copy_((w.float() / s).to(w.dtype))
        b = getattr(norm, "bias", None)
        if b is not None:
            b.copy_((b.float() / s).to(b.dtype))


def _fold_group_scale(
    block: nn.Module, binp: BlockInputs, names: list[str], mods: dict[str, nn.Module], s: torch.Tensor, allow: bool
) -> bool:
    """Try to absorb ``X -> X / s`` for the input shared by ``names`` into the producer. Returns True if folded (exact)."""
    if not allow:
        return False
    first = mods[names[0]]
    leaf = names[0].split(".")[-1]
    parent_name = names[0].rpartition(".")[0]
    parent = block.get_submodule(parent_name) if parent_name else block
    if ".experts." not in names[0]:
        norm = _find_producer_norm(block, binp, first)
        if norm is not None and norm.weight.numel() == s.numel():
            # every consumer of this norm output must be in the group (else we would rescale their input too)
            consumers = _consumers_of(block, binp, norm)
            if set(consumers) <= set(names):
                _scale_norm(norm, s)
                return True
    # o_proj <- v_proj (multi-head attention only), down_proj <- up_proj
    if len(names) == 1:
        sib = {"o_proj": "v_proj", "down_proj": "up_proj"}.get(leaf)
        prod = getattr(parent, sib, None) if sib else None
        if isinstance(prod, nn.Linear) and prod.out_features == first.in_features == s.numel() and prod is not first:
            if leaf == "o_proj":
                cfg_heads = (getattr(parent, "num_heads", None), getattr(parent, "num_kv_heads", None))
                if cfg_heads[0] is None or cfg_heads[0] != cfg_heads[1]:
                    return False
            with torch.no_grad():
                prod.weight.copy_((prod.weight.float() / s.unsqueeze(1)).to(prod.weight.dtype))
                if prod.bias is not None:
                    prod.bias.copy_((prod.bias.float() / s).to(prod.bias.dtype))
            return True
    return False


def _consumers_of(block: nn.Module, binp: BlockInputs, norm: nn.Module) -> list[str]:
    """Names of all dense linears whose input is the output tensor of ``norm``."""
    ptr: list[int] = []
    keepalive: list[torch.Tensor] = []  # keep the norm output alive: a freed-and-recycled address would fake a consumer

    def norm_hook(_m: nn.Module, _i: tuple, o: torch.Tensor) -> None:  # must return None: a returned value would replace the output
        keepalive.append(o)
        ptr.append(o.data_ptr())

    h0 = norm.register_forward_hook(norm_hook)
    hits: list[str] = []
    handles = []
    for n, m in block.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(lambda mod, i, n=n: hits.append(n) if ptr and i[0].data_ptr() == ptr[-1] else None))
    try:
        with torch.no_grad():
            binp.run(block, 0)
    finally:
        h0.remove()
        for h in handles:
            h.remove()
    return hits


def _awq_search(
    xs: torch.Tensor, xmean: torch.Tensor, weights: list[torch.Tensor], cfg: PTQConfig, bits: int
) -> tuple[torch.Tensor, float, float]:
    """Return (best scale [K], best alpha, relative error improvement vs alpha=0)."""
    from forge.formats.int_quant import fake_quant_int

    g = _largest_divisor_group(xs.shape[1], cfg.group_size)
    ref = [xs @ w.t() for w in weights]
    best_err, best_s, best_a = float("inf"), torch.ones_like(xmean), 0.0
    base_err = None
    for gi in range(cfg.awq_n_grid):
        a = gi / cfg.awq_n_grid
        s = xmean.clamp(min=1e-5).pow(a)
        s = (s / (s.max() * s.min()).sqrt()).clamp(min=1e-4)
        err = 0.0
        for w, r in zip(weights, ref):
            wq = fake_quant_int(w * s.unsqueeze(0), bits, g, cfg.sym, clip_search=cfg.clip_search) / s.unsqueeze(0)
            err += float((xs @ wq.t() - r).pow(2).sum())
        if gi == 0:
            base_err = err
        if err < best_err:
            best_err, best_s, best_a = err, s, a
    return best_s, best_a, (base_err - best_err) / max(base_err or 1e-30, 1e-30)


# ------------------------------------------------------------------------------------------------ the driver
def _layer_bits(name: str, cfg: PTQConfig, alloc: dict[str, int] | None) -> int:
    for pat, b in cfg.per_layer_bits.items():
        if re.search(pat, name):
            return int(b)
    if alloc and name in alloc:
        return alloc[name]
    return cfg.bits


def _proxy_metrics(W: torch.Tensor, deq: torch.Tensor, H: torch.Tensor | None) -> tuple[float, float | None]:
    wf = W.float()
    d = wf - deq.float()
    rel = float(d.pow(2).sum() / wf.pow(2).sum().clamp(min=1e-30)) ** 0.5
    if H is None:
        return rel, None
    num = float(((d @ H) * d).sum())
    den = float(((wf @ H) * wf).sum())
    return rel, num / max(den, 1e-30)


def _replace(model: nn.Module, name: str, new: nn.Module, lin: nn.Module) -> None:
    dev = lin.weight.device
    new.to(dev)
    set_submodule(model, name, new)


def quantize_model(
    model: nn.Module,
    method: str | None = None,
    bits: int | None = None,
    group_size: int | None = None,
    sym: bool | None = None,
    act_order: bool | None = None,
    calib: Any = None,
    *,
    config: PTQConfig | dict | None = None,
    tokenizer: Any = None,
    ctx: RunContext | None = None,
    **overrides: Any,
) -> QuantizedModel:
    """Quantise ``model`` **in place** (dense linears -> packed modules) and return ``QuantizedModel(model, report)``.

    ``calib``: calibration-set name / :class:`CalibrationSet` / token tensor ``[n, L]`` (default ``cfg``-driven ``"synthetic"``).
    Explicit keyword args override ``config``; any further ``overrides`` are :class:`PTQConfig` fields.
    """
    cfg = PTQConfig.model_validate(config) if isinstance(config, dict) else (config or PTQConfig())
    upd: dict[str, Any] = dict(overrides)
    for k, v in (("method", method), ("bits", bits), ("group_size", group_size), ("sym", sym), ("act_order", act_order)):
        if v is not None:
            upd[k] = v
    if upd:
        cfg = cfg.model_copy(update=upd)
        cfg = PTQConfig.model_validate(cfg.model_dump())
    ctx = ctx or NullContext()
    t_start = time.perf_counter()
    dtype = model_dtype(model)
    sdtype = dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
    report = QuantReport(cfg.method, cfg.model_dump())
    report.bytes_before = model_nbytes(model)
    dev = model_device(model)
    if cfg.method == "smoothquant":
        cfg = cfg.model_copy(update={"format": "int", "bits": 8})
    g4 = gemma4_features(model)
    if g4 and (cfg.uses_calibration() or cfg.mixed_precision):
        raise ConfigError(
            f"PTQ method {cfg.method!r} replays transformer blocks on calibration data, which is not supported for Gemma 4 style models "
            f"({', '.join(g4)}: per-layer inputs, KV sharing, sandwich norms). Use method='rtn' (data-free weight-only) or fp8."
        )
    if cfg.format == "fp8":
        return _quantize_fp8(model, cfg, report, ctx, t_start)

    # ---- calibration data
    binp = None
    tokens = None
    if cfg.uses_calibration():
        tokens, _ = get_calibration_tokens(calib, model, tokenizer, n_seqs=cfg.n_seqs, seq_len=cfg.seq_len, seed=cfg.seed)
        report.calib = {"n_seqs": int(tokens.shape[0]), "seq_len": int(tokens.shape[1]),
                        "set": getattr(calib, "name", calib if isinstance(calib, str) else "tokens"), "seed": cfg.seed}

    # ---- mixed precision allocation (sensitivity from RTN error weighted by Hessian)
    alloc: dict[str, int] | None = None
    if cfg.mixed_precision:
        from forge.compress.bitalloc import allocate_for_model

        alloc, alloc_info = allocate_for_model(model, tokens, cfg, ctx=ctx)
        report.calib["bit_allocation"] = alloc_info

    prefix, blocks = (None, None)
    try:
        prefix, blocks = find_blocks(model)
    except ConfigError:
        if cfg.uses_calibration():
            raise
    in_block_names: set[str] = set()
    if blocks is not None:
        for i, blk in enumerate(blocks):
            in_block_names |= {n for n, _ in iter_dense_linears(blk, f"{prefix}.{i}", cfg.exclude, cfg.include)}

    def record(name: str, lin: nn.Module, qw: QuantizedWeight, mod: nn.Module, Wt: torch.Tensor, H: torch.Tensor | None, secs: float) -> None:
        rel, proxy = _proxy_metrics(Wt, qw.deq, H)
        nb = sum(b.numel() * b.element_size() for b in mod.buffers()) + sum(p.numel() * p.element_size() for p in mod.parameters())
        n0 = lin.weight.numel() * lin.weight.element_size() + (lin.bias.numel() * lin.bias.element_size() if lin.bias is not None else 0)
        report.layers.append(LayerRecord(name, getattr(mod, "kind", qw.fmt), qw.bits, qw.group_size, (lin.out_features, lin.in_features),
                                         rel, proxy, secs, int(n0), int(nb), list(qw.notes)))

    def record_w8a8(name: str, lin: nn.Module, mod: nn.Module, Wt: torch.Tensor, secs: float, notes: list[str]) -> None:
        rel, _ = _proxy_metrics(Wt, mod.dequantize(torch.float32), None)
        nb = sum(b.numel() * b.element_size() for b in mod.buffers()) + sum(p.numel() * p.element_size() for p in mod.parameters())
        n0 = lin.weight.numel() * lin.weight.element_size() + (lin.bias.numel() * lin.bias.element_size() if lin.bias is not None else 0)
        report.layers.append(LayerRecord(name, "w8a8_int8", 8, -1, (lin.out_features, lin.in_features), rel, None, secs, int(n0), int(nb), notes))

    def finish_block_layer(
        name: str, lin: nn.Linear, W: torch.Tensor, H: torch.Tensor | None, scale_info: tuple | None = None, note: str | None = None
    ) -> QuantizedWeight:
        t0 = time.perf_counter()
        b = _layer_bits(name, cfg, alloc)
        qw = quantize_layer(lin, W, H, cfg, bits=b, sdtype=sdtype)
        if note:
            qw.notes.append(note)
        if scale_info is not None:
            qw.input_scale = scale_info[0]
        mod = build_module(qw, lin, cfg, dtype)
        _replace(model, name, mod, lin)
        # report metrics against the original (unscaled / unrotated) weight space via the transformed weight
        record(name, lin, qw, mod, W, H, time.perf_counter() - t0)
        return qw

    n_done = 0
    total_lin = max(1, len(in_block_names))
    # ---- block-wise methods
    if cfg.uses_calibration() and blocks is not None and cfg.method != "rtn":
        binp = capture_block_inputs(model, tokens, batch_size=cfg.batch_size, offload=cfg.offload_activations)
        for bi, block in enumerate(blocks):
            if ctx.should_stop():
                report.warnings.append("cancelled before all blocks were quantised")
                break
            bprefix = f"{prefix}.{bi}"
            lins = {n[len(bprefix) + 1 :]: m for n, m in iter_dense_linears(block, bprefix, cfg.exclude, cfg.include)}
            if lins:
                if cfg.method in ("gptq", "mrgptq"):
                    _gptq_block(model, block, bprefix, binp, lins, cfg, finish_block_layer, dev)
                elif cfg.method == "awq":
                    _awq_block(model, block, bprefix, binp, lins, cfg, finish_block_layer, alloc)
                elif cfg.method == "smoothquant":
                    _smoothquant_block(model, block, bprefix, binp, lins, cfg, record_w8a8)
            n_done += len(lins)
            binp.propagate(block)
            ctx.progress(min(0.99, n_done / total_lin), f"quantised block {bi + 1}/{len(blocks)}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    # ---- everything not handled above (RTN, mixed-precision leftovers, linears outside the blocks)
    for name, lin in list(iter_dense_linears(model, "", cfg.exclude, cfg.include)):
        if name in {r.name for r in report.layers}:
            continue
        t0 = time.perf_counter()
        W = lin.weight.detach().float()
        qw = quantize_layer(lin, W, None, cfg.model_copy(update={"method": "rtn"}), bits=_layer_bits(name, cfg, alloc), sdtype=sdtype)
        if cfg.method in ("gptq", "mrgptq", "awq", "smoothquant") and blocks is not None and name not in in_block_names:
            qw.notes.append("outside transformer blocks: RTN")
        mod = build_module(qw, lin, cfg, dtype)
        _replace(model, name, mod, lin)
        record(name, lin, qw, mod, W, None, time.perf_counter() - t0)
        n_done += 1
        ctx.progress(min(0.99, n_done / total_lin), f"quantised {name}")
    report.seconds = time.perf_counter() - t_start
    report.bytes_after = model_nbytes(model)
    ctx.log(f"ptq[{cfg.method}] done: {len(report.layers)} layers, {report.bytes_before / 2**20:.1f} -> {report.bytes_after / 2**20:.1f} MiB "
            f"({report.compression:.2f}x), avg {report.avg_bits:.2f} bits/weight on quantised layers, {report.seconds:.1f}s")
    ctx.progress(1.0, "ptq done")
    return QuantizedModel(model, report)


def _quantize_fp8(model: nn.Module, cfg: PTQConfig, report: QuantReport, ctx: RunContext, t_start: float) -> QuantizedModel:
    for name, lin in list(iter_dense_linears(model, "", cfg.exclude, cfg.include)):
        t0 = time.perf_counter()
        W = lin.weight.detach().float()
        mod = _make_fp8_module(lin, cfg)
        set_submodule(model, name, mod.to(lin.weight.device))
        deq = mod.dequantize(torch.float32)
        rel, _ = _proxy_metrics(W, deq, None)
        nb = sum(b.numel() * b.element_size() for b in mod.buffers())
        report.layers.append(LayerRecord(name, "fp8", 8, -1, (lin.out_features, lin.in_features), rel, None, time.perf_counter() - t0,
                                         lin.weight.numel() * lin.weight.element_size(), int(nb)))
    if cfg.method != "rtn":
        report.warnings.append("format=fp8 is calibration-free (per-channel absmax scales); method ignored")
    report.seconds = time.perf_counter() - t_start
    report.bytes_after = model_nbytes(model)
    ctx.progress(1.0, "fp8 done")
    return QuantizedModel(model, report)


# ------------------------------------------------------------------------------------------------ per-block drivers
def _gptq_block(model, block, bprefix, binp, lins, cfg, finish, dev) -> None:
    groups = discover_input_groups(block, binp, lins)
    stages = sequential_stages(groups, cfg.true_sequential)
    for stage in stages:
        # one Hessian per distinct input (q/k/v share one)
        accs: dict[str, HessianAccumulator] = {}
        cbs: dict[str, tuple[nn.Module, Callable]] = {}
        for grp in stage:
            lead = grp[0]
            acc = HessianAccumulator(lins[lead].in_features, dev)
            accs[lead] = acc
            cbs[lead] = (lins[lead], lambda x, _o, acc=acc: acc.add(x))
        run_with_hooks(block, binp, cbs)
        for grp in stage:
            H = accs[grp[0]].H
            for rel in grp:
                lin = lins[rel]
                full = f"{bprefix}.{rel}"
                W = lin.weight.detach().float()
                if accs[grp[0]].n == 0:
                    finish(full, lin, W, None, None, "no calibration tokens reached this layer: RTN")  # cold MoE expert
                    continue
                finish(full, lin, W, H)
                # keep the block numerics in sync for later stages (dense fake-quant weights until the module swap below)
                lin_new = model.get_submodule(full)
                lins[rel] = lin_new
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # subsequent stages must see the quantised layers: modules were already swapped by `finish`; refresh hooks' module refs
        for rel in list(lins):
            lins[rel] = block.get_submodule(rel)


def _group_samples(block, binp, lins, groups, dev, max_rows) -> tuple[dict[str, SampleBuffer], dict[str, ChannelStats]]:
    bufs, stats, cbs = {}, {}, {}
    for grp in groups:
        lead = grp[0]
        k = lins[lead].in_features
        bufs[lead], stats[lead] = SampleBuffer(k, max_rows), ChannelStats(k, dev)

        def cb(x, _o, lead=lead):
            bufs[lead].add(x)
            stats[lead].add(x)

        cbs[lead] = (lins[lead], cb)
    run_with_hooks(block, binp, cbs)
    return bufs, stats


def _awq_block(model, block, bprefix, binp, lins, cfg, finish, alloc) -> None:
    dev = model_device(block)
    groups = discover_input_groups(block, binp, lins)
    # AWQ searches on the original block; scaling groups are processed in execution order, outputs are not re-propagated between groups
    # (as in the reference implementation) -- activations were collected once from the unmodified block.
    bufs, stats = _group_samples(block, binp, lins, groups, dev, cfg.awq_samples)
    scales: dict[str, tuple[torch.Tensor, bool]] = {}
    for grp in groups:
        lead = grp[0]
        xs = bufs[lead].get(dev)
        if xs.shape[0] == 0:
            continue
        b = _layer_bits(f"{bprefix}.{lead}", cfg, alloc)
        Ws = [lins[r].weight.detach().float() for r in grp]
        s, alpha, gain = _awq_search(xs, stats[lead].mean_abs.to(dev), Ws, cfg, b)
        if gain <= 0:
            continue
        for r in grp:
            lins[r].weight.data = (lins[r].weight.data.float() * s.unsqueeze(0)).to(lins[r].weight.dtype)
        folded = _fold_group_scale(block, binp, grp, lins, s, cfg.fold_scales)
        for r in grp:
            scales[r] = (s, folded)
    for rel, lin in lins.items():
        full = f"{bprefix}.{rel}"
        s_info = scales.get(rel)
        if s_info is None:
            finish(full, lin, lin.weight.detach().float(), None)
        else:
            s, folded = s_info
            finish(full, lin, lin.weight.detach().float(), None, None if folded else (1.0 / s,), "awq scale " + ("folded" if folded else "kept as input_scale"))


def _smoothquant_block(model, block, bprefix, binp, lins, cfg, record_w8a8) -> None:
    dev = model_device(block)
    groups = discover_input_groups(block, binp, lins)
    _, stats = _group_samples(block, binp, lins, groups, dev, 16)
    a = cfg.sq_alpha
    scales: dict[str, tuple[torch.Tensor, bool]] = {}
    for grp in groups:
        lead = grp[0]
        if stats[lead].n == 0:
            continue
        xmax = stats[lead].abs_max.to(dev).clamp(min=1e-5)
        wmax = torch.stack([lins[r].weight.detach().float().abs().amax(dim=0) for r in grp]).amax(0).clamp(min=1e-5)
        s = (xmax.pow(a) / wmax.pow(1 - a)).clamp(min=1e-5)
        for r in grp:
            lins[r].weight.data = (lins[r].weight.data.float() * s.unsqueeze(0)).to(lins[r].weight.dtype)
        folded = _fold_group_scale(block, binp, grp, lins, s, cfg.fold_scales)
        for r in grp:
            scales[r] = (s, folded)
    for rel, lin in lins.items():
        t0 = time.perf_counter()
        full = f"{bprefix}.{rel}"
        W = lin.weight.detach().float()
        s_info = scales.get(rel)
        inp = None if s_info is None or s_info[1] else (1.0 / s_info[0])
        act = cfg.resolved_act_quant()
        mod = Int8W8A8Linear.from_weight(W, lin.bias, act_quant=act, input_scale=inp, dtype=lin.weight.dtype)
        _replace(model, full, mod, lin)
        note = "smoothquant scale " + ("none" if s_info is None else "folded" if s_info[1] else "kept as input_scale")
        record_w8a8(full, lin, mod, W, time.perf_counter() - t0, [note])
