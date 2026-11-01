"""Explicit GPU-memory budgeting of a deployment: weights, KV pool, activation reserve, headroom.

The engine used to size its KV pool as ``total * gpu_mem_util - allocated - reserve`` and always filled that budget, so a 10 GB W4A16 model on a 96 GB
card ended up occupying ~87 GB regardless of what was loaded: the number on the GPU said nothing about the weights.  This module makes the split
explicit and reportable::

    weights  = bytes of the loaded parameters + buffers (packed int4 codes + scales count as they are stored)
    other    = whatever else torch has allocated at planning time (kernel-layout copies, LoRA, ...)
    budget   = total * gpu_mem_util - weights - other - activation_reserve       (and <= free - activation_reserve)
    KV pool  = min(budget, workload)   with ``kv_pool='fit_workload'`` (default): workload = max_num_seqs * ceil(max_model_len / block_size) blocks
             = budget                  with ``kv_pool='fill'`` (vLLM-style: everything the budget allows)
             = num_gpu_blocks          when given explicitly

:func:`plan_kv_memory` is a pure function over numbers (tests need no GPU); :func:`estimate_memory` answers "what would this model + config cost" for
the deployment form before anything is loaded.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from forge.core.errors import ConfigError

GB = 1e9
MIN_BLOCKS = 8


@dataclass
class MemoryPlan:
    device: str
    gpu_mem_util: float
    weight_bytes: int
    packed_weight_bytes: int
    other_allocated_bytes: int
    activation_reserve_bytes: int
    bytes_per_block: int
    block_size: int
    blocks_by_memory: int | None  # None on CPU (no memory budget)
    blocks_for_workload: int
    num_blocks: int
    limited_by: str  # explicit | workload | workload_cap | memory | cpu_cap
    kv_pool: str  # fit_workload | fill
    workload_uncapped_blocks: int = 0
    extra_bytes: int = 0  # draft-model weights + its private KV pool, reserved before the target pool is sized
    scratch_blocks: int = 1  # KVPool allocates one extra scratch block beyond num_blocks
    total_bytes: int | None = None
    free_bytes: int | None = None
    budget_bytes: int | None = None
    max_model_len: int = 0
    max_num_seqs: int = 0
    blocks_per_max_len_seq: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def kv_pool_bytes(self) -> int:
        return self.num_blocks * self.bytes_per_block

    @property
    def planned_total_bytes(self) -> int:
        return self.weight_bytes + self.other_allocated_bytes + self.extra_bytes + self.kv_pool_bytes + self.scratch_blocks * self.bytes_per_block + self.activation_reserve_bytes

    @property
    def headroom_bytes(self) -> int | None:
        """Total GPU memory not claimed by this plan (``total - planned``); ``None`` without a GPU."""
        return None if self.total_bytes is None else self.total_bytes - self.planned_total_bytes

    @property
    def max_concurrent_at_full_len(self) -> int:
        return self.num_blocks // max(self.blocks_per_max_len_seq, 1)

    def summary(self) -> str:
        s = (f"weights {self.weight_bytes / GB:.2f} GB"
             + (f" (packed {self.packed_weight_bytes / GB:.2f} GB)" if self.packed_weight_bytes else "")
             + f", KV pool {self.kv_pool_bytes / GB:.2f} GB ({self.kv_pool_bytes / max(self.weight_bytes, 1):.1f}x weights; {self.num_blocks} blocks x {self.block_size} tokens, limited by {self.limited_by})")
        if self.headroom_bytes is not None:
            s += f", free {self.headroom_bytes / GB:.2f} GB of {self.total_bytes / GB:.1f} GB (gpu_mem_util={self.gpu_mem_util:g})"
        return s

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(kv_pool_bytes=self.kv_pool_bytes, planned_total_bytes=self.planned_total_bytes, headroom_bytes=self.headroom_bytes,
                 weights_gb=round(self.weight_bytes / GB, 3), kv_pool_gb=round(self.kv_pool_bytes / GB, 3),
                 free_gb=None if self.headroom_bytes is None else round(self.headroom_bytes / GB, 3),
                 max_concurrent_at_full_len=self.max_concurrent_at_full_len, summary=self.summary())
        return d


def plan_kv_memory(*, device: str, bytes_per_block: int, block_size: int, weight_bytes: int, packed_weight_bytes: int = 0,
                   gpu_mem_util: float, activation_reserve_gb: float, max_num_seqs: int, max_model_len: int,
                   total_bytes: int | None = None, free_bytes: int | None = None, allocated_bytes: int | None = None,
                   num_gpu_blocks: int | None = None, kv_pool: str = "fit_workload", cpu_cap_blocks: int = 2048,
                   kv_pool_cap_tokens: int = 0, extra_bytes: int = 0) -> MemoryPlan:
    """Decide the number of KV blocks. ``allocated_bytes`` = everything torch holds on the device right now (weights included); on CUDA the three
    byte arguments ``total_bytes`` / ``free_bytes`` / ``allocated_bytes`` come from ``mem_get_info`` / ``memory_allocated``."""
    if kv_pool not in ("fit_workload", "fill"):
        raise ConfigError(f"kv_pool must be 'fit_workload' or 'fill', got {kv_pool!r}")
    per = int(bytes_per_block)
    reserve = int(activation_reserve_gb * GB)
    one_seq = math.ceil(max_model_len / block_size)
    want_full = max(1, max_num_seqs) * one_seq
    want = want_full
    if kv_pool == "fit_workload" and kv_pool_cap_tokens > 0:
        want = min(want_full, max(one_seq, math.ceil(kv_pool_cap_tokens / block_size)))  # a single max-length request must always fit
    cuda = device.startswith("cuda") and total_bytes is not None
    other = max(int((allocated_bytes or 0) - weight_bytes), 0) if cuda else 0
    plan = MemoryPlan(device=device, gpu_mem_util=gpu_mem_util, weight_bytes=int(weight_bytes), packed_weight_bytes=int(packed_weight_bytes),
                      other_allocated_bytes=other, activation_reserve_bytes=reserve if cuda else 0, bytes_per_block=per, block_size=block_size,
                      blocks_by_memory=None, blocks_for_workload=want, workload_uncapped_blocks=want_full, num_blocks=0, limited_by="", kv_pool=kv_pool, total_bytes=total_bytes,
                      free_bytes=free_bytes, extra_bytes=int(extra_bytes) if cuda else 0, max_model_len=max_model_len, max_num_seqs=max_num_seqs, blocks_per_max_len_seq=one_seq)
    if num_gpu_blocks:
        plan.num_blocks, plan.limited_by = int(num_gpu_blocks), "explicit"
    elif not cuda:
        plan.num_blocks = max(MIN_BLOCKS, min(want, cpu_cap_blocks))
        plan.limited_by = "cpu_cap" if want > cpu_cap_blocks else ("workload_cap" if want < want_full else "workload")
    else:
        assert total_bytes is not None
        used = max(int(allocated_bytes or 0), int(weight_bytes)) + plan.extra_bytes
        ceiling = int(total_bytes * gpu_mem_util)
        if used + reserve >= ceiling:
            raise ConfigError(
                f"the model does not fit the memory budget: weights {weight_bytes / GB:.2f} GB (+{other / GB:.2f} GB other allocations, +{plan.extra_bytes / GB:.2f} GB draft model) + activation "
                f"reserve {reserve / GB:.2f} GB >= gpu_mem_util {gpu_mem_util:g} x {total_bytes / GB:.1f} GB = {ceiling / GB:.1f} GB; raise gpu_mem_util, "
                "use a smaller / more strongly quantised model, or a bigger GPU")
        budget = ceiling - used - reserve
        if free_bytes is not None:
            budget = min(budget, int(free_bytes) - reserve)
        plan.budget_bytes = budget
        plan.blocks_by_memory = max(int(budget // per) - plan.scratch_blocks, 0) if budget > 0 else 0  # the scratch block is part of the budget
        if plan.blocks_by_memory < MIN_BLOCKS:
            raise ConfigError(
                f"not enough GPU memory for a KV cache: budget {max(budget, 0) / GB:.2f} GB, block {per / 1e6:.2f} MB (weights {weight_bytes / GB:.2f} GB, "
                f"total {total_bytes / GB:.1f} GB, free {(free_bytes or 0) / GB:.1f} GB); lower max_model_len, use a smaller model or raise gpu_mem_util")
        if kv_pool == "fit_workload" and want < plan.blocks_by_memory:
            plan.num_blocks, plan.limited_by = want, ("workload_cap" if want < want_full else "workload")
        else:
            plan.num_blocks, plan.limited_by = plan.blocks_by_memory, "memory"
    if plan.num_blocks < one_seq:
        plan.warnings.append(f"the KV pool holds {plan.num_blocks * block_size} tokens, less than max_model_len={max_model_len}: a single maximum-length "
                             "request can never be served; lower max_model_len or raise gpu_mem_util / num_gpu_blocks")
    if plan.limited_by == "workload_cap":
        plan.warnings.append(f"the default KV pool is capped at {plan.num_blocks * block_size} tokens (kv_pool_cap_tokens={kv_pool_cap_tokens}) although "
                             f"max_num_seqs x max_model_len would use {want_full * block_size}: sequences beyond the pool wait or are preempted; "
                             "raise kv_pool_cap_tokens or use kv_pool='fill' for more concurrency")
    if plan.weight_bytes > 0 and plan.kv_pool_bytes > 2 * plan.weight_bytes:
        plan.warnings.append(f"the KV pool ({plan.kv_pool_bytes / GB:.2f} GB) is more than 2x the model weights ({plan.weight_bytes / GB:.2f} GB): "
                             "lower max_num_seqs / max_model_len / kv_pool_cap_tokens, or quantise the KV cache (kv_dtype) if that is not intended")
    return plan


def draft_pool_blocks(*, max_num_seqs: int, max_model_len: int, block_size: int, kv_pool: str = "fit_workload", kv_pool_cap_tokens: int = 0) -> int:
    """Blocks of a draft model's private KV pool: a quarter of the target's workload pool (at least 16)."""
    one_seq = math.ceil(max_model_len / block_size)
    want = max(1, max_num_seqs) * one_seq
    if kv_pool == "fit_workload" and kv_pool_cap_tokens > 0:
        want = min(want, max(one_seq, math.ceil(kv_pool_cap_tokens / block_size)))
    return max(16, want // 4)


SUPPORTED_ARTIFACT_KV = ("fp8_e4m3", "int8")


def artifact_kv_choice(kv_dtype: str, artifact_kv: Any) -> tuple[str | None, list[str]]:
    """``(scheme to serve with | None, warnings)`` for the KV scheme an artifact recorded (``artifact_kv`` = the artifact summary's ``kv`` dict).

    Only ``fp8_e4m3`` / ``int8`` exist in the paged pool. Under ``kv_dtype='auto'`` an unsupported scheme (fp8_e5m2, kivi, ...) or a ``residual_window``
    (full-precision recent tokens, a KIVI feature) is *not* silently dropped: the caller reports these warnings in the deployment status."""
    if not isinstance(artifact_kv, dict) or not artifact_kv.get("scheme"):
        return None, []
    scheme = str(artifact_kv["scheme"])
    warns: list[str] = []
    chosen: str | None = None
    if scheme in SUPPORTED_ARTIFACT_KV:
        if kv_dtype == "auto":
            chosen = scheme
        elif kv_dtype != scheme:
            warns.append(f"the artifact was compressed/gated with KV scheme {scheme!r} but kv_dtype={kv_dtype!r} was requested: quality was not validated for it")
        rw = artifact_kv.get("residual_window")
        if rw:
            warns.append(f"the artifact recorded residual_window={rw}; the paged KV pool has no full-precision residual window, so the whole KV cache "
                         f"is stored as {scheme} (quality may differ from the gated recipe)")
    else:
        warns.append(f"the artifact's KV scheme {scheme!r} is not implemented by the serving KV pool (supported: {', '.join(SUPPORTED_ARTIFACT_KV)}, "
                     f"plus plain bf16/fp16/fp32); the KV cache is stored as {kv_dtype if kv_dtype != 'auto' else 'the compute dtype'} instead, "
                     "so the quality gated at compression time does not apply to the KV cache")
    return chosen, warns


def device_memory(device: Any) -> tuple[int | None, int | None, int | None]:
    """``(total, free, allocated)`` bytes for a CUDA device, ``(None, None, None)`` otherwise."""
    import torch

    if getattr(device, "type", str(device)) != "cuda" or not torch.cuda.is_available():
        return None, None, None
    free, total = torch.cuda.mem_get_info(device)
    return int(total), int(free), int(torch.cuda.memory_allocated(device))


def estimate_memory(model_cfg: Any, *, weight_bytes: int, packed_weight_bytes: int = 0, kv_name: str = "bf16", block_size: int = 16,
                    gpu_mem_util: float = 0.85, activation_reserve_gb: float = 1.0, max_num_seqs: int = 16, max_model_len: int = 4096,
                    kv_pool: str = "fit_workload", total_gpu_bytes: int | None = None, num_gpu_blocks: int | None = None,
                    kv_pool_cap_tokens: int = 0, extra_bytes: int = 0) -> dict[str, Any]:
    """What a deployment would cost, before anything is loaded (the deployment form calls this). ``total_gpu_bytes=None``: no GPU known -> only sizes."""
    from forge.serve.runner import ModelDims, kv_bytes_per_block

    d = ModelDims.from_config(model_cfg)
    per = kv_bytes_per_block(d.num_kv_layers, d.num_kv_heads, d.head_dim, block_size, kv_name)
    dev = "cuda" if total_gpu_bytes else "cpu"
    try:
        plan = plan_kv_memory(device=dev, bytes_per_block=per, block_size=block_size, weight_bytes=weight_bytes, packed_weight_bytes=packed_weight_bytes,
                              gpu_mem_util=gpu_mem_util, activation_reserve_gb=activation_reserve_gb, max_num_seqs=max_num_seqs,
                              max_model_len=max_model_len, total_bytes=total_gpu_bytes, free_bytes=total_gpu_bytes, allocated_bytes=weight_bytes,
                              num_gpu_blocks=num_gpu_blocks, kv_pool=kv_pool, kv_pool_cap_tokens=kv_pool_cap_tokens, extra_bytes=extra_bytes)
        out = plan.to_dict()
        out["fits"] = True
    except ConfigError as e:
        out = {"fits": False, "error": str(e), "weights_gb": round(weight_bytes / GB, 3), "device": dev}
    out["kv_bytes_per_token"] = per / block_size
    out["kv_gb_per_max_len_seq"] = round(per / block_size * max_model_len / GB, 4)
    out["kv_dtype"] = kv_name
    return out
