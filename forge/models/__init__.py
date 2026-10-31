"""forge.models: decoder-only transformer, HF IO, linear registry, LoRA, KV caches (see docs/ARCHITECTURE.md 5.4).

Imports are lazy (PEP 562) so ``import forge.models`` is cheap; attribute access loads the defining submodule.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    # config
    "ModelConfig": "config",
    "MoEConfig": "config",
    "SUPPORTED_ARCHS": "config",
    # model
    "ForgeLM": "modeling",
    "ModelOutput": "modeling",
    "AttnMeta": "attention",
    # linear registry
    "LinearBase": "linear",
    "DenseLinear": "linear",
    "PackedLinear": "linear",
    "Fp8Linear": "linear",
    "FakeQuantIntLinear": "linear",
    "FakeQuantNVFP4Linear": "linear",
    "register_linear": "linear",
    "get_linear_cls": "linear",
    "linear_kinds": "linear",
    "replace_linears": "linear",
    "quantize_linears": "linear",
    "linear_report": "linear",
    # lora
    "LoRALinear": "lora",
    "add_lora": "lora",
    "merge_lora": "lora",
    "remove_lora": "lora",
    "lora_state_dict": "lora",
    "load_lora_state_dict": "lora",
    "save_lora_adapter": "lora",
    "load_lora_adapter": "lora",
    "set_lora_enabled": "lora",
    "trainable_parameters": "lora",
    # kv cache
    "KVCacheBase": "kvcache",
    "DenseKVCache": "kvcache",
    "PagedKVView": "kvcache",
    "SimplePagedPool": "kvcache",
    # loader
    "load_model": "loader",
    "save_model": "loader",
    "load_tokenizer": "loader",
    "materialize": "loader",
    "hf_to_forge_name": "loader",
    "forge_to_hf_name": "loader",
    # tiny / accounting
    "make_tiny_config": "tiny",
    "build_tiny_model": "tiny",
    "kv_bytes_per_token": "accounting",
    "model_summary": "accounting",
    "count_parameters": "accounting",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'forge.models' has no attribute {name!r}")
    return getattr(importlib.import_module(f"forge.models.{mod}"), name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
