"""Oracle for INT8 W8A8 (K3 INT8 branch): symmetric per-token activation / per-channel weight quantisation and an
exact integer GEMM (int32 accumulation emulated exactly in float64; |sum| < 2**53 for any K < 5e8)."""
from __future__ import annotations

import torch


def quantize_int8_ref(x: torch.Tensor, granularity: str = "per_token", scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8: ``q = clamp(round(x / s), -127, 127)``, ``s = amax / 127`` (round-half-even).

    ``granularity``: ``per_token`` / ``per_channel`` (scale ``[rows]``) or ``per_tensor`` (scale ``[1]``).
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).float()
    if granularity in ("per_token", "per_channel"):
        amax = x2.abs().amax(dim=-1)
        s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)) if scale is None else scale.float().reshape(-1)
        sb = s.reshape(-1, 1)
    elif granularity == "per_tensor":
        amax = x2.abs().amax().reshape(1) if x2.numel() else torch.zeros(1, device=x2.device)
        s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)) if scale is None else scale.float().reshape(1)
        sb = s.reshape(1, 1)
    else:
        raise ValueError(f"unknown granularity {granularity!r}")
    q = torch.round(x2 / sb).clamp(-127, 127).to(torch.int8)
    return q.reshape(shape), s.contiguous()


def int8_matmul_exact(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """``int32 [M, N] = xq[M,K] @ wq[N,K]^T`` computed exactly (float64 holds every partial sum)."""
    acc = xq.reshape(-1, xq.shape[-1]).double() @ wq.double().t()
    return acc.round().to(torch.int64).to(torch.int32)


def int8_gemm_ref(
    xq: torch.Tensor,
    sx: torch.Tensor,
    wq: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """``Y = (Xq @ Wq^T)_int32 * sx[:,None] * sw[None,:] + bias`` (fp32 epilogue)."""
    lead = xq.shape[:-1]
    acc = int8_matmul_exact(xq, wq).float()
    sxf = sx.float().reshape(-1, 1) if sx.numel() != 1 else sx.float().reshape(1, 1)
    swf = sw.float().reshape(1, -1) if sw.numel() != 1 else sw.float().reshape(1, 1)
    y = acc * sxf * swf
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*lead, wq.shape[0]).to(out_dtype or torch.float32)
