"""Oracle for the ternary W1.58A8 GEMM (``ternary_w158a8_gemm``).

``Y[m, n] = sx[m] * sum_g sw[n, g] * sum_{k in g} xq[m, k] * T[n, k]``: exact integer accumulation per scale group, then fp32 scale
multiply per group and a per-token epilogue -- the math of every tern158 backend.  Integer sums stay far below 2**24 per group, so
fp32 accumulation here is exact (``group * 127``), matching the int32 accumulators of the kernels.
"""
from __future__ import annotations

import torch

from forge.formats import ternary as _tn


def ternary_gemm_ref(xq: torch.Tensor, sx: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, group_size: int, k: int,
                     out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``xq int8 [M, K]``, ``sx fp32 [M]``, packed weight ``uint8 [N, K/4]`` + scales -> ``[M, N]``."""
    t = _tn.unpack_ternary(qweight, k).float()
    n = t.shape[0]
    m = xq.shape[0]
    xf = xq.float()
    if group_size == 0:
        y = (xf @ t.t()) * scales.reshape(1).float()
    else:
        g = _tn.effective_group(group_size, k)
        ngr = k // g
        part = torch.einsum("mgk,ngk->mng", xf.reshape(m, ngr, g), t.reshape(n, ngr, g))
        y = (part * scales.float().reshape(n, ngr)[None]).sum(dim=2)
    return (y * sx.float()[:, None]).to(out_dtype)


def ternary_linear_ref(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, group_size: int, k: int, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Float ``x [..., K]`` -> per-token int8 quant -> :func:`ternary_gemm_ref` -> ``[..., N]`` in ``x.dtype``."""
    lead = x.shape[:-1]
    q, s = _tn.act_quant_int8(x)
    y = ternary_gemm_ref(q, s, qweight, scales, group_size, k, torch.float32)
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype).reshape(*lead, -1)


def ternary_bwd_data_ref(dy: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, group_size: int, k: int, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``dX = dY . (T * sw)`` (straight-through, scale detached)."""
    t = _tn.unpack_ternary(qweight, k)
    w = _tn.dequantize_ternary(t, scales, group_size)
    return (dy.float() @ w).to(out_dtype)
