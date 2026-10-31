"""Oracles for the FP8 GEMM (K3): quantisers with per-tensor / per-token(channel) / block scaling, dequantisation,
the GEMM itself and the fused RMSNorm + absmax + FP8-quantise activation producer.

Scale-tensor conventions (all fp32), ``x`` is ``[M, K]`` (leading dims flattened), ``w`` is ``[N, K]``:

==============  =========================  =======================================
granularity     activations ``x``          weights ``w``
==============  =========================  =======================================
per_tensor      ``[1]``                    ``[1]``
per_token       ``[M]``                    --
per_channel     --                         ``[N]``
block           ``[M, ceil(K/B)]`` (1xB)   ``[ceil(N/B), ceil(K/B)]`` (BxB)
==============  =========================  =======================================

``dequant = q.float() * scale``.  E4M3 here is the OCP *FN* variant (max 448, no infinities, NaN = 0x7F);
inputs are clamped to +-max before the cast (``torch`` casts do not saturate: 449 -> NaN).
"""
from __future__ import annotations

import torch

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


def fp8_dtype(fmt: str = "e4m3") -> torch.dtype:
    if fmt == "e4m3":
        return torch.float8_e4m3fn
    if fmt == "e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unknown fp8 format {fmt!r}")


def _amax_all(x: torch.Tensor) -> torch.Tensor:
    """max |x| over everything as a shape-[1] tensor (0 for empty tensors, e.g. M=0 batches)."""
    return x.abs().amax().reshape(1) if x.numel() else torch.zeros(1, dtype=x.dtype, device=x.device)


def _safe_scale(amax: torch.Tensor, fmax: float) -> torch.Tensor:
    """``amax / fmax`` with 1.0 where the group is all-zero (avoids 0/0; q is 0 there anyway)."""
    return torch.where(amax > 0, amax / fmax, torch.ones_like(amax))


def quantize_fp8_ref(
    x: torch.Tensor,
    granularity: str = "per_token",
    *,
    block: int = 128,
    scale: torch.Tensor | None = None,
    fmt: str = "e4m3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``x`` (``[..., K]``) to FP8 -> ``(q fp8 same shape, scale fp32)``.

    ``granularity``: ``per_tensor`` | ``per_token`` (= ``per_channel`` for weights) | ``block`` (1xB for a single
    leading dim ``x[M,K]`` activations) | ``block2d`` (BxB, weights).  ``scale`` forces a static scale
    (broadcastable per the table above).
    """
    fmax = FP8_MAX[fmt]
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).float()
    m, k = x2.shape
    g = granularity
    if g == "per_channel":
        g = "per_token"
    if g == "per_tensor":
        s = _safe_scale(_amax_all(x2), fmax) if scale is None else scale.float().reshape(1)
        sb = s.reshape(1, 1)
    elif g == "per_token":
        s = _safe_scale(x2.abs().amax(dim=-1), fmax) if scale is None else scale.float().reshape(m)
        sb = s.reshape(m, 1)
    elif g == "block":
        kb = -(-k // block)
        xp = torch.nn.functional.pad(x2, (0, kb * block - k))
        amax = xp.view(m, kb, block).abs().amax(dim=-1)
        s = _safe_scale(amax, fmax) if scale is None else scale.float().reshape(m, kb)
        sb = s.repeat_interleave(block, dim=1)[:, :k]
    elif g == "block2d":
        nb, kb = -(-m // block), -(-k // block)
        xp = torch.nn.functional.pad(x2, (0, kb * block - k, 0, nb * block - m))
        amax = xp.view(nb, block, kb, block).abs().amax(dim=(1, 3))
        s = _safe_scale(amax, fmax) if scale is None else scale.float().reshape(nb, kb)
        sb = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:m, :k]
    else:
        raise ValueError(f"unknown granularity {granularity!r}")
    q = (x2 / sb).clamp(-fmax, fmax).to(fp8_dtype(fmt))
    return q.reshape(shape), s.contiguous()


def expand_scale(scale: torch.Tensor, rows: int, k: int, block: int = 128) -> torch.Tensor:
    """Broadcast a scale tensor (conventions above) to ``[rows, k]`` fp32."""
    s = scale.float()
    if s.numel() == 1:
        return s.reshape(1, 1).expand(rows, k)
    if s.dim() == 1:
        if s.numel() != rows:
            raise ValueError(f"1-D scale has {s.numel()} entries, expected {rows}")
        return s.reshape(rows, 1).expand(rows, k)
    if s.dim() == 2:
        if s.shape[0] == rows:  # 1xB blocks
            return s.repeat_interleave(block, dim=1)[:, :k]
        return s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:rows, :k]
    raise ValueError(f"unsupported scale shape {tuple(s.shape)}")


def dequantize_fp8_ref(q: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """fp32 dequantisation of ``q [..., K]`` (granularity inferred from the scale shape)."""
    q2 = q.reshape(-1, q.shape[-1])
    return (q2.float() * expand_scale(scale, q2.shape[0], q2.shape[1], block)).reshape(q.shape)


def fp8_gemm_ref(
    xq: torch.Tensor,
    sx: torch.Tensor,
    wq: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block: int = 128,
    out_dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """``Y = dq(Xq) @ dq(Wq)^T + bias`` in fp32.  Works for every scale granularity of the table above."""
    lead = xq.shape[:-1]
    x = dequantize_fp8_ref(xq, sx, block).reshape(-1, xq.shape[-1])
    w = dequantize_fp8_ref(wq, sw, block)
    y = x @ w.t()
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*lead, wq.shape[0]).to(out_dtype or torch.float32)


def rmsnorm_act_quant_fp8_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-6,
    *,
    granularity: str = "per_token",
    block: int = 128,
    fmt: str = "e4m3",
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused producer oracle: ``y = x * rsqrt(mean(x^2)+eps) * weight`` in fp32, then FP8 quantisation of ``y``.

    Returns ``(q, scale, y_fp32)``.  (HF Llama rounds the normalised value to bf16 before multiplying by the
    weight; here everything stays fp32 until the single FP8 rounding, which is what the fused kernel does.)
    """
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    q, s = quantize_fp8_ref(y, granularity, block=block, scale=scale, fmt=fmt)
    return q, s, y
