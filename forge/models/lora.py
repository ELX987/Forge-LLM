"""LoRA / DoRA adapters over any linear module (dense, fake-quant, or packed for QLoRA-style training).

``LoRALinear`` wraps a base linear (``base``) and adds ``scaling * B(A(dropout(x)))``.  The adapter matrices are plain
``nn.Linear`` modules named ``lora_A`` (in -> r) and ``lora_B`` (r -> out) so the on-disk adapter format is PEFT-compatible
(``base_model.model.<hf module path>.lora_A.weight``).  With ``use_dora`` a learned per-output magnitude ``lora_magnitude``
rescales the row-normalised merged weight (DoRA, Liu et al. 2024; the norm is detached as in the paper's efficient variant).

Merging into a packed base is impossible without dequantising: :func:`merge_lora` raises unless ``dequantize_packed=True``.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.core.errors import ConfigError
from forge.models.linear import (
    DEFAULT_QUANT_SKIP,
    PackedLinear,
    linear_dense_weight,
    register_linear,
    replace_linears,
)

log = logging.getLogger("forge.models.lora")

DEFAULT_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
ALL_LINEAR = "all-linear"


@register_linear("lora")
class LoRALinear(nn.Module):
    """Low-rank adapter around ``base`` (any module with ``in_features`` / ``out_features``)."""

    def __init__(
        self,
        base: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_dora: bool = False,
        lora_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if r <= 0:
            raise ConfigError("LoRA rank must be positive")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.r, self.alpha = int(r), float(alpha)
        self.scaling = self.alpha / self.r
        self.use_dora = use_dora
        self.enabled = True
        self.merged = False
        bw = getattr(base, "weight", None)
        bdt = bw.dtype if isinstance(bw, torch.Tensor) and bw.is_floating_point() and bw.dtype != torch.float8_e4m3fn else torch.float32
        dev = bw.device if isinstance(bw, torch.Tensor) else None
        if lora_dtype is None:
            lora_dtype = torch.float32 if bdt in (torch.float16, torch.bfloat16) else bdt
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False, dtype=lora_dtype, device=dev)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False, dtype=lora_dtype, device=dev)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if not self.lora_A.weight.is_meta:
            self.reset_lora()
        if use_dora:
            self.lora_magnitude = nn.Parameter(torch.ones(self.out_features, dtype=lora_dtype, device=dev))
            self.init_dora_magnitude()

    # ---- init
    def reset_lora(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @torch.no_grad()
    def init_dora_magnitude(self) -> None:
        w = linear_dense_weight(self.base, torch.float32)
        self.lora_magnitude.copy_(w.norm(dim=1).to(self.lora_magnitude.dtype))

    # ---- helpers
    @property
    def weight(self) -> torch.Tensor:
        return linear_dense_weight(self.base)

    @property
    def bias(self):
        return getattr(self.base, "bias", None)

    def _delta(self) -> torch.Tensor:
        return (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.merged:
            return self.base(x)
        if not self.use_dora:
            lora = self.lora_B(self.lora_A(self.lora_dropout(x).to(self.lora_A.weight.dtype)))
            return self.base(x) + lora.to(x.dtype) * self.scaling
        w = linear_dense_weight(self.base, torch.float32)
        wp = w + self._delta()
        norm = wp.norm(dim=1).detach()
        mag = (self.lora_magnitude.float() / norm).to(x.dtype)
        y = (F.linear(x, w.to(x.dtype)) + F.linear(self.lora_dropout(x), self._delta().to(x.dtype))) * mag
        bias = self.bias
        return y + bias.to(x.dtype) if bias is not None else y

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        w = linear_dense_weight(self.base, torch.float32)
        wp = w + self._delta()
        if self.use_dora:
            wp = wp * (self.lora_magnitude.float() / wp.norm(dim=1)).unsqueeze(1)
        return wp

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, dora={self.use_dora}, in={self.in_features}, out={self.out_features}"


# ----------------------------------------------------------------------------------------------- selection
def _matches(name: str, targets: Iterable[str], exclude: Iterable[str]) -> bool:
    parts = name.split(".")
    for e in exclude:
        if e in parts or name.endswith(e):
            return False
    for t in targets:
        if t.startswith("re:"):
            if re.search(t[3:], name):
                return True
        elif t == name or t == parts[-1] or name.endswith("." + t):
            return True
    return False


def add_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] | str = DEFAULT_TARGETS,
    *,
    use_dora: bool = False,
    freeze_base: bool = True,
    exclude: Iterable[str] = (),
    lora_dtype: torch.dtype | None = None,
) -> list[str]:
    """Wrap matching linears with :class:`LoRALinear`. Returns the wrapped module names.

    ``target_modules``: last-name-component/suffix strings (e.g. ``"q_proj"``), ``"re:<regex>"`` patterns, or ``"all-linear"``
    (every linear except lm_head and MoE routers).  ``freeze_base`` sets ``requires_grad=False`` on every non-LoRA parameter.
    """
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    targets = list(target_modules)
    all_lin = ALL_LINEAR in targets
    excl = tuple(exclude) + (DEFAULT_QUANT_SKIP if all_lin else ())

    def factory(name: str, lin: nn.Module) -> nn.Module | None:
        if all_lin:
            ok = not any(e == name.split(".")[-1] or e in name.split(".") for e in excl)
        else:
            ok = _matches(name, targets, excl)
        return LoRALinear(lin, r, alpha, dropout, use_dora, lora_dtype) if ok else None

    # packed modules are not nn.Linear: handle them in a second pass
    names = replace_linears(model, factory)
    names += _wrap_packed(model, targets, excl, all_lin, r, alpha, dropout, use_dora, lora_dtype)
    if not names:
        raise ConfigError(f"no module matched LoRA target_modules={targets}")
    if freeze_base:
        for n, p in model.named_parameters():
            p.requires_grad = ".lora_A." in n or ".lora_B." in n or n.endswith("lora_magnitude")
    return names


def _wrap_packed(model, targets, excl, all_lin, r, alpha, dropout, use_dora, lora_dtype, prefix="") -> list[str]:
    out: list[str] = []
    for name, child in list(model.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, PackedLinear):
            ok = (not any(e == name or e in full.split(".") for e in excl)) if all_lin else _matches(full, targets, excl)
            if ok:
                setattr(model, name, LoRALinear(child, r, alpha, dropout, use_dora, lora_dtype))
                out.append(full)
        elif isinstance(child, LoRALinear):
            continue
        else:
            out += _wrap_packed(child, targets, excl, all_lin, r, alpha, dropout, use_dora, lora_dtype, full)
    return out


def iter_lora(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield n, m


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle all adapters (disabled = base model behaviour; useful for reference-policy logprobs)."""
    for _, m in iter_lora(model):
        m.enabled = enabled


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    out = []
    for _, m in iter_lora(model):
        out += [m.lora_A.weight, m.lora_B.weight]
        if m.use_dora:
            out.append(m.lora_magnitude)
    return out


def _swap(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf, new)


@torch.no_grad()
def merge_lora(model: nn.Module, *, dequantize_packed: bool = False) -> nn.Module:
    """Fold every adapter into its base weight and remove the wrappers (in place). Returns ``model``.

    Dense bases are merged in fp32 and cast back to the base dtype.  Packed bases raise :class:`ConfigError` unless
    ``dequantize_packed=True`` (then they are replaced by a dense ``nn.Linear``).
    """
    from forge.models.linear import DenseLinear

    for name, m in list(iter_lora(model)):
        base = m.base
        w = m.merged_weight()
        if isinstance(base, PackedLinear):
            if not dequantize_packed:
                raise ConfigError(f"cannot merge LoRA into packed layer {name} ({base.kind}); pass dequantize_packed=True")
            dense = DenseLinear(base.in_features, base.out_features, bias=getattr(base, "bias", None) is not None, dtype=w.dtype, device=w.device)
            dense.weight.data.copy_(w)
            if getattr(base, "bias", None) is not None:
                dense.bias.data.copy_(base.bias.data.to(w.dtype))
            _swap(model, name, dense)
        else:
            base.weight.data.copy_(w.to(base.weight.dtype))
            _swap(model, name, base)
    return model


def remove_lora(model: nn.Module) -> nn.Module:
    """Strip adapters without merging (restores the base model)."""
    for name, m in list(iter_lora(model)):
        _swap(model, name, m.base)
    return model


# ----------------------------------------------------------------------------------------------- state dict io
def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapter tensors keyed ``<module>.lora_A.weight`` / ``.lora_B.weight`` / ``.lora_magnitude`` (Forge module names)."""
    sd: dict[str, torch.Tensor] = {}
    for n, m in iter_lora(model):
        sd[f"{n}.lora_A.weight"] = m.lora_A.weight.detach().cpu().clone()
        sd[f"{n}.lora_B.weight"] = m.lora_B.weight.detach().cpu().clone()
        if m.use_dora:
            sd[f"{n}.lora_magnitude"] = m.lora_magnitude.detach().cpu().clone()
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor], strict: bool = True) -> tuple[list[str], list[str]]:
    """Copy adapter tensors in. Returns ``(missing, unexpected)``; raises on mismatch if ``strict``."""
    mods = dict(iter_lora(model))
    seen: set[str] = set()
    unexpected: list[str] = []
    for k, v in sd.items():
        if k.endswith(".lora_A.weight"):
            mod_name, tgt = k[: -len(".lora_A.weight")], "A"
        elif k.endswith(".lora_B.weight"):
            mod_name, tgt = k[: -len(".lora_B.weight")], "B"
        elif k.endswith(".lora_magnitude"):
            mod_name, tgt = k[: -len(".lora_magnitude")], "M"
        else:
            unexpected.append(k)
            continue
        m = mods.get(mod_name)
        if m is None:
            unexpected.append(k)
            continue
        dst = {"A": m.lora_A.weight, "B": m.lora_B.weight, "M": getattr(m, "lora_magnitude", None)}[tgt]
        if dst is None or tuple(dst.shape) != tuple(v.shape):
            unexpected.append(k)
            continue
        with torch.no_grad():
            dst.copy_(v.to(dst.device, dst.dtype))
        seen.add(k)
    expected = set(lora_state_dict(model))
    missing = sorted(expected - seen)
    if strict and (missing or unexpected):
        raise ConfigError(f"LoRA state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return missing, unexpected


def save_lora_adapter(model: nn.Module, directory: str | Path, *, base_model: str | None = None) -> Path:
    """Write ``adapter_model.safetensors`` + ``adapter_config.json`` (PEFT-style key names; Forge names are HF names minus ``model.``)."""
    from safetensors.torch import save_file

    from forge.models.loader import forge_to_hf_name

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    mods = dict(iter_lora(model))
    cfg = getattr(model, "config", None)
    tensors = {}
    for k, v in lora_state_dict(model).items():
        hf = forge_to_hf_name(k, cfg) if cfg is not None else k
        tensors["base_model.model." + hf] = v.contiguous()
    save_file(tensors, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
    any_m = next(iter(mods.values()))
    targets = sorted({n.split(".")[-1] for n in mods})
    (d / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": any_m.r,
                "lora_alpha": any_m.alpha,
                "lora_dropout": getattr(any_m.lora_dropout, "p", 0.0),
                "target_modules": targets,
                "use_dora": any_m.use_dora,
                "base_model_name_or_path": base_model,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "forge": {"num_modules": len(mods)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return d


def load_lora_adapter(model: nn.Module, directory: str | Path, *, strict: bool = True, add_if_missing: bool = True) -> list[str]:
    """Load an adapter written by :func:`save_lora_adapter`; wraps the target linears first when the model has no adapters."""
    from safetensors.torch import load_file

    from forge.models.loader import hf_to_forge_name

    d = Path(directory)
    cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    raw = load_file(str(d / "adapter_model.safetensors"))
    cfgm = getattr(model, "config", None)
    sd = {}
    for k, v in raw.items():
        kk = k[len("base_model.model.") :] if k.startswith("base_model.model.") else k
        sd[hf_to_forge_name(kk, cfgm) or kk if cfgm is not None else kk] = v
    if add_if_missing and not any(True for _ in iter_lora(model)):
        add_lora(model, cfg["r"], cfg["lora_alpha"], cfg.get("lora_dropout", 0.0), cfg["target_modules"], use_dora=cfg.get("use_dora", False))
    load_lora_state_dict(model, sd, strict=strict)
    return sorted(dict(iter_lora(model)))
