"""forge.compress -- calibration, PTQ, KV quantisation, rotation, pruning, low-rank, delta and lossless compression, plus the recipe stages.

Importing this package is cheap: it registers the packed linear kinds (``w4a16``, ``w8a16``, ``nvfp4``, ... via :mod:`forge.compress.packed`) and
registers the recipe stages *lazily* (their implementation module is imported on first use).  Public entry points::

    forge.compress.run.run_compress(cfg: CompressJobConfig, ctx)        # job entry
    forge.compress.ptq.quantize_model(model, method, bits, group_size, sym, act_order, calib, ...)
    forge.compress.calibrate.register_calibration_set / collect_stats / StatsCache
    forge.compress.kvquant / rotation / prune / lowrank / delta / lossless / bitalloc / serialize

See ``docs/ARCHITECTURE.md`` section 5.6 and ``docs/KERNELS.md`` for which paths are validated on GPU hardware (none of the compression code
needs a GPU; the packed linears' CUDA kernels are *written, not validated on hardware*).
"""
from __future__ import annotations

from typing import Any

from forge.core.registry import register_lazy

_STAGE_DOCS = {
    "calibrate": "Choose and fingerprint the calibration set; optionally pre-collect statistics.",
    "ptq": "Post-training quantisation: rtn / gptq / awq / smoothquant / mrgptq (FP4).",
    "pack": "Derive and validate the device layout of packed weights for a target chip.",
    "ternarize": "Data-free ternary (1.58-bit) post-training quantisation to packed t158 layers (EXPERIMENTAL).",
    "kv": "KV-cache quantisation scheme (FP8 / int8 / KIVI) with accuracy-vs-bytes report.",
    "rotate": "Fuse norms and rotate the residual stream (randomised Hadamard or learned orthogonal).",
    "prune": "Structured pruning (width, GQA heads, depth, hidden, experts) and research sparsity.",
    "lowrank": "Activation-whitened SVD with heterogeneous rank allocation and break-even guard.",
    "delta": "Delta compression of fine-tunes: BitDelta (1-bit sign + scale) or low-rank.",
    "lossless": "Bit-exact entropy coding report/container for the model tensors.",
    "eval": "Parent-vs-candidate evaluation (delegates to forge.eval).",
    "gate": "Pass / fail / inconclusive verdict (delegates to forge.eval gates).",
    "distill": "Distillation recovery (builtin logit-KL, or delegate to forge.train).",
    "finetune": "Fine-tune the current model (delegates to forge.train).",
}

for _name, _doc in _STAGE_DOCS.items():
    register_lazy("stage", _name, f"forge.compress.stages:stage_{_name}", meta={"doc": _doc, "module": "compress"})

try:  # register w4a16 / w8a16 / nvfp4 ... in the linear registry (needs torch only)
    from forge.compress import packed as _packed  # noqa: F401
except Exception:  # pragma: no cover - never block package import
    pass

_LAZY = {
    "quantize_model": "forge.compress.ptq",
    "PTQConfig": "forge.compress.ptq",
    "run_compress": "forge.compress.run",
    "CompressJobConfig": "forge.compress.config",
    "register_calibration_set": "forge.compress.calibrate",
    "collect_stats": "forge.compress.calibrate",
    "save_quantized": "forge.compress.serialize",
    "load_quantized": "forge.compress.serialize",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'forge.compress' has no attribute {name!r}")


__all__ = list(_LAZY)
