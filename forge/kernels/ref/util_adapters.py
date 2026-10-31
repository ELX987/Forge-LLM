"""Oracles for activation-sparsity GEMV (TEAL-style, Plan K8) and segmented-batched LoRA/delta (Plan K9).

TEAL (training-free activation sparsity, P2 F5): per token, input channels with ``|x_k| <= threshold`` are skipped, so
only the surviving columns of the weight matrix are read.  The weight is stored transposed ``w_t [K, N]`` so that a
surviving input channel is one contiguous row of ``N`` weights (coalesced gather).

Segmented LoRA (SV-7 multi-tenant serving): token ``t`` of a mixed batch belongs to adapter ``adapter_ids[t]``
(``-1`` = base model only) and receives ``scaling[a] * (x[t] @ A[a]^T) @ B[a]^T`` added to its output row.
"""
from __future__ import annotations

import torch


def teal_threshold(samples: torch.Tensor, sparsity: float) -> float:
    """Magnitude threshold such that a fraction ``sparsity`` of the entries of ``samples`` fall below it (calibration)."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    flat = samples.float().abs().flatten()
    if sparsity == 0.0:
        return 0.0
    k = max(1, int(round(sparsity * flat.numel())))
    return float(torch.kthvalue(flat.cpu(), k).values)


def teal_gemv_ref(x: torch.Tensor, w_t: torch.Tensor, threshold: float, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``y[b] = sum_{k: |x[b,k]| > thr} x[b,k] * w_t[k]`` (+ bias). ``x [B, K]``, ``w_t [K, N]`` -> ``[B, N]`` in ``x.dtype``."""
    xf = x.float()
    xs = torch.where(xf.abs() > threshold, xf, torch.zeros_like(xf))
    y = xs @ w_t.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype)


def teal_active_fraction(x: torch.Tensor, threshold: float) -> float:
    return float((x.float().abs() > threshold).float().mean())


def segmented_lora_ref(x: torch.Tensor, y: torch.Tensor, adapter_ids: torch.Tensor, lora_a: torch.Tensor,
                       lora_b: torch.Tensor, scalings: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Return ``y + delta`` (new tensor).

    ``x [T, K]``, ``y [T, N]``, ``adapter_ids [T]`` int (``-1`` = none), ``lora_a [n_adapters, R, K]``,
    ``lora_b [n_adapters, N, R]`` (ranks smaller than ``R`` are zero padded), ``scalings`` scalar or ``[n_adapters]``
    (``alpha / r`` of each adapter).
    """
    out = y.float().clone()
    n_ad = lora_a.shape[0]
    sc = torch.as_tensor(scalings, dtype=torch.float32, device=x.device)
    for a in range(n_ad):
        sel = (adapter_ids == a).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        s = sc if sc.dim() == 0 else sc[a]
        delta = (x[sel].float() @ lora_a[a].float().T) @ lora_b[a].float().T
        out[sel] += s * delta
    return out.to(y.dtype)


def pad_adapters(a_list: list[torch.Tensor], b_list: list[torch.Tensor], multiple: int = 16):
    """Stack adapters of different ranks (``A_i [r_i, K]``, ``B_i [N, r_i]``) into zero-padded tensors with a common
    rank ``R`` rounded up to ``multiple`` (tensor-core friendly)."""
    r_max = max(a.shape[0] for a in a_list)
    r = ((r_max + multiple - 1) // multiple) * multiple
    k = a_list[0].shape[1]
    n = b_list[0].shape[0]
    a_st = torch.zeros(len(a_list), r, k, dtype=a_list[0].dtype, device=a_list[0].device)
    b_st = torch.zeros(len(b_list), n, r, dtype=b_list[0].dtype, device=b_list[0].device)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        a_st[i, : a.shape[0]] = a
        b_st[i, :, : b.shape[1]] = b
    return a_st, b_st
