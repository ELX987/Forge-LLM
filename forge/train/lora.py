"""Built-in LoRA / QLoRA implementation used by :mod:`forge.train.finetune`.

* :class:`LoRALinear` wraps a frozen base linear (``nn.Linear`` or :class:`QuantizedLinear`) with a trainable low-rank update.
* :class:`QuantizedLinear` stores weights block-quantised -- NF4 (QLoRA normal-float codebook, 4 bit, two codes per byte) or absmax
  int8 -- and de-quantises on the fly.  Pure torch: works on CPU and every CUDA GPU (a fused kernel would be faster; this path
  is correct, not fast).
* :class:`LoraBackend` hides whether ``forge.models.lora`` (preferred when importable and compatible) or this built-in
  implementation is used.  Adapter-disabled reference passes (DPO) and QLoRA always use the built-in implementation.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from forge.core.errors import ConfigError
from forge.train.config import LoraConfig, QLoraConfig

log = logging.getLogger("forge.train.lora")

__all__ = ["LoRALinear", "LoraBackend", "NF4_CODE", "QuantizedLinear", "adapter_disabled", "merge_adapter_state"]

NF4_CODE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141025543, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236954, 1.0,
)


# ---------------------------------------------------------------------------------------------------------------
# quantised frozen base
# ---------------------------------------------------------------------------------------------------------------
class QuantizedLinear(nn.Module):
    """Frozen block-quantised linear (weight-only). ``bits`` 4 = NF4, 8 = absmax int8."""

    def __init__(self, in_features: int, out_features: int, bits: int = 4, block_size: int = 64, bias: torch.Tensor | None = None,
                 compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ConfigError("QuantizedLinear bits must be 4 or 8")
        self.in_features, self.out_features, self.bits, self.block_size = in_features, out_features, bits, block_size
        self.compute_dtype = compute_dtype
        self.n_blocks = -(-in_features // block_size)
        padded = self.n_blocks * block_size
        self.register_buffer("qweight", torch.zeros(out_features, padded // 2 if bits == 4 else padded, dtype=torch.uint8 if bits == 4 else torch.int8))
        self.register_buffer("absmax", torch.zeros(out_features, self.n_blocks, dtype=torch.float32))
        self.register_buffer("code", torch.tensor(NF4_CODE, dtype=torch.float32), persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int = 4, block_size: int = 64, compute_dtype: torch.dtype = torch.float32) -> QuantizedLinear:
        q = cls(lin.in_features, lin.out_features, bits, block_size, lin.bias, compute_dtype).to(lin.weight.device)
        q._quantize(lin.weight.detach().float())
        return q

    @torch.no_grad()
    def _quantize(self, w: torch.Tensor) -> None:
        n_out, K = w.shape
        padded = self.n_blocks * self.block_size
        if padded != K:
            w = Fn.pad(w, (0, padded - K))
        wb = w.reshape(n_out, self.n_blocks, self.block_size)
        amax = wb.abs().amax(-1)
        self.absmax.copy_(amax)
        norm = wb / amax.clamp(min=1e-12).unsqueeze(-1)
        if self.bits == 8:
            self.qweight.copy_((norm * 127.0).round().clamp(-127, 127).to(torch.int8).reshape(n_out, padded))
            return
        code = self.code.to(w.device)
        idx = torch.empty(n_out, self.n_blocks, self.block_size, dtype=torch.uint8, device=w.device)
        step = max(1, (1 << 24) // (self.n_blocks * self.block_size * 16))
        for i in range(0, n_out, step):
            seg = norm[i:i + step].unsqueeze(-1)
            idx[i:i + step] = (seg - code).abs().argmin(-1).to(torch.uint8)
        flat = idx.reshape(n_out, padded)
        self.qweight.copy_(flat[:, 0::2] | (flat[:, 1::2] << 4))

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        n_out = self.out_features
        padded = self.n_blocks * self.block_size
        if self.bits == 8:
            norm = self.qweight.float().reshape(n_out, self.n_blocks, self.block_size) / 127.0
        else:
            lo = self.qweight & 0x0F
            hi = self.qweight >> 4
            idx = torch.stack([lo, hi], dim=-1).reshape(n_out, padded).long()
            norm = self.code[idx].reshape(n_out, self.n_blocks, self.block_size)
        w = (norm * self.absmax.unsqueeze(-1)).reshape(n_out, padded)[:, : self.in_features]
        return w.to(dtype or self.compute_dtype)

    @property
    def weight(self) -> torch.Tensor:  # read-only convenience (dequantised, compute dtype)
        return self.dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cd = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else self.compute_dtype
        w = self.dequantize(cd)
        b = self.bias.to(cd) if self.bias is not None else None
        return Fn.linear(x.to(cd), w, b)


# ---------------------------------------------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """``y = base(x) + (alpha/r) * B(A(dropout(x)))`` with a frozen ``base``."""

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0, rslora: bool = False, init: str = "default",
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        in_f, out_f = base.in_features, base.out_features  # type: ignore[attr-defined]
        self.base = base
        self.r, self.alpha = int(r), float(alpha)
        self.scaling = self.alpha / (math.sqrt(self.r) if rslora else self.r)
        dev = next((p.device for p in base.parameters()), next((b.device for b in base.buffers()), torch.device("cpu")))
        self.lora_A = nn.Linear(in_f, self.r, bias=False, device=dev, dtype=dtype)
        self.lora_B = nn.Linear(self.r, out_f, bias=False, device=dev, dtype=dtype)
        if init == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1.0 / self.r)
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True
        self.in_features, self.out_features = in_f, out_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        a = self.lora_A.weight.dtype
        upd = self.lora_B(self.lora_A(self.drop(x).to(a))) * self.scaling
        return y + upd.to(y.dtype)

    @torch.no_grad()
    def merged_linear(self, dtype: torch.dtype | None = None) -> nn.Linear:
        """A plain ``nn.Linear`` with the update merged into the (de-quantised) base weight."""
        base = self.base
        w = base.dequantize(torch.float32) if isinstance(base, QuantizedLinear) else base.weight.detach().float()
        w = w + self.scaling * (self.lora_B.weight.float() @ self.lora_A.weight.float())
        bias = base.bias
        out = nn.Linear(self.in_features, self.out_features, bias=bias is not None, device=w.device, dtype=dtype or (base.weight.dtype if hasattr(base, "weight") else torch.float32))
        out.weight.copy_(w.to(out.weight.dtype))
        if bias is not None:
            out.bias.copy_(bias.detach().to(out.bias.dtype))
        return out


@contextlib.contextmanager
def adapter_disabled(model: nn.Module):  # type: ignore[no-untyped-def]
    """Temporarily bypass all built-in LoRA adapters (reference policy for DPO)."""
    mods = [m for m in model.modules() if hasattr(m, "lora_A") and hasattr(m, "enabled")]
    prev = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, p in zip(mods, prev, strict=False):
            m.enabled = p


def _match(name: str, targets: list[str], excludes: list[str]) -> bool:
    if any(e and e in name for e in excludes):
        return False
    if "all-linear" in targets:
        return True
    last = name.rsplit(".", 1)[-1]
    return any(last == t or name.endswith("." + t) or name == t for t in targets)


def merge_adapter_state(model: nn.Module, adapter_sd: dict[str, torch.Tensor], scaling_by_module: dict[str, float]) -> int:
    """Merge a saved adapter state dict (``<module>.lora_A.weight`` / ``lora_B.weight``) into a *plain* model's linears in place."""
    n = 0
    mods = dict(model.named_modules())
    for key in list(adapter_sd.keys()):
        if not key.endswith(".lora_A.weight"):
            continue
        mname = key[: -len(".lora_A.weight")]
        a, b = adapter_sd[key].float(), adapter_sd[mname + ".lora_B.weight"].float()
        mod = mods.get(mname)
        if mod is None or not hasattr(mod, "weight"):
            raise ConfigError(f"adapter module {mname!r} not found in the base model")
        with torch.no_grad():
            mod.weight.add_((scaling_by_module[mname] * (b @ a)).to(device=mod.weight.device, dtype=mod.weight.dtype))
        n += 1
    return n


class LoraBackend:
    """Applies / saves / merges LoRA with either ``forge.models.lora`` or the built-in implementation."""

    def __init__(self, lcfg: LoraConfig, *, prefer_builtin: bool = False) -> None:
        self.lcfg = lcfg
        self.impl = "builtin"
        self._forge: Any = None
        if lcfg.use_forge_lora and not prefer_builtin and not lcfg.rslora and lcfg.init == "default":
            try:
                import forge.models.lora as fl  # type: ignore

                if all(hasattr(fl, n) for n in ("add_lora", "merge_lora", "lora_state_dict", "load_lora_state_dict")):
                    self._forge = fl
            except Exception:  # noqa: BLE001
                self._forge = None
        self.replaced: list[str] = []
        self.scalings: dict[str, float] = {}

    # -- apply -----------------------------------------------------------------------------------------
    def apply(self, model: nn.Module) -> list[str]:
        lc = self.lcfg
        targets = list(lc.target_modules)
        if getattr(getattr(model, "config", None), "is_hybrid", False) and "all-linear" not in targets:
            # lfm2 conv blocks have no q/k/v/o: adapt their gated-conv projections whenever the attention projections are targeted
            targets += [t for t in ("in_proj", "out_proj") if t not in targets] if "o_proj" in targets else []
        self._targets = targets
        if self._forge is not None:
            try:
                keep = tuple(lc.modules_to_save)
                names = self._forge.add_lora(
                    model, r=lc.r, alpha=lc.alpha, dropout=lc.dropout, target_modules=targets,
                    exclude=tuple(lc.exclude_modules), lora_dtype=torch.float32 if lc.adapter_dtype == "float32" else torch.bfloat16)
                self.impl = "forge"
                self._freeze(model, lambda n: (".lora_A." in n or ".lora_B." in n or n.endswith("lora_magnitude")) or any(k in n for k in keep))
                self.replaced = list(names)
                return self.replaced
            except Exception as e:  # noqa: BLE001
                log.warning("forge.models.lora.add_lora failed (%r); using the built-in LoRA", e)
                self._forge = None
        dtype = torch.float32 if lc.adapter_dtype == "float32" else torch.bfloat16
        names = [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, QuantizedLinear)) and not isinstance(m, LoRALinear) and _match(n, targets, list(lc.exclude_modules))]
        if not names:
            raise ConfigError("LoRA: no linear modules matched target_modules", details={"targets": list(lc.target_modules)})
        for n in names:
            parent_name, _, child = n.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            base = getattr(parent, child)
            wrapped = LoRALinear(base, lc.r, lc.alpha, lc.dropout, lc.rslora, lc.init, dtype)
            setattr(parent, child, wrapped)
            self.scalings[n] = wrapped.scaling
        self.impl = "builtin"
        self.replaced = names
        keep = tuple(lc.modules_to_save)
        self._freeze(model, lambda n: (".lora_A." in n or ".lora_B." in n) or any(k in n for k in keep))
        return names

    @staticmethod
    def _freeze(model: nn.Module, trainable: Any) -> None:
        for n, p in model.named_parameters():
            p.requires_grad = bool(trainable(n))

    def trainable_names(self, model: nn.Module) -> list[str]:
        return [n for n, p in model.named_parameters() if p.requires_grad]

    # -- state --------------------------------------------------------------------------------------------
    def state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        if self.impl == "forge":
            sd = {k: v.detach().cpu() for k, v in self._forge.lora_state_dict(model).items()}
            for k, v in model.state_dict().items():
                if any(s in k for s in self.lcfg.modules_to_save) and ".lora_" not in k and ".base." not in k:
                    sd[k] = v.detach().cpu()
            return sd
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_A." in k or ".lora_B." in k}
        for k, v in model.state_dict().items():
            if any(s in k for s in self.lcfg.modules_to_save) and ".lora_" not in k and ".base." not in k:
                sd[k] = v.detach().cpu()
        return sd

    def load_state_dict(self, model: nn.Module, sd: dict[str, torch.Tensor]) -> None:
        if self.impl == "forge":
            ad = {k: v for k, v in sd.items() if ".lora_" in k}
            self._forge.load_lora_state_dict(model, ad)
            rest = {k: v for k, v in sd.items() if ".lora_" not in k}
            if rest:
                model.load_state_dict(rest, strict=False)
            return
        res = model.load_state_dict(sd, strict=False)
        unexpected = [k for k in res.unexpected_keys]
        if unexpected:
            raise ConfigError("adapter has keys not present in the model", details={"unexpected": unexpected[:10]})

    def save_adapter(self, model: nn.Module, out_dir: str | Path, base_model: str = "") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.impl == "forge":
            try:
                self._forge.save_lora_adapter(model, out, base_model=base_model or None)
                extra = {k: v for k, v in self.state_dict(model).items() if ".lora_" not in k}
                if extra:
                    torch.save(extra, out / "modules_to_save.pt")
                return out
            except Exception as e:  # noqa: BLE001
                log.warning("forge.models.lora.save_lora_adapter failed (%r); writing the built-in adapter format", e)
        sd = self.state_dict(model)
        try:
            from safetensors.torch import save_file

            save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
            fmt = "safetensors"
        except Exception:  # noqa: BLE001
            torch.save(sd, out / "adapter_model.pt")
            fmt = "pt"
        cfg = {
            "peft_type": "LORA", "impl": self.impl, "format": fmt, "r": self.lcfg.r, "lora_alpha": self.lcfg.alpha,
            "lora_dropout": self.lcfg.dropout, "rslora": self.lcfg.rslora, "target_modules": list(getattr(self, "_targets", self.lcfg.target_modules)),
            "modules_to_save": list(self.lcfg.modules_to_save), "base_model_name_or_path": base_model,
            "scaling": {n: s for n, s in self.scalings.items()},
        }
        (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        return out

    def merge(self, model: nn.Module, dtype: torch.dtype | None = None) -> nn.Module:
        """Merge adapters into the base weights in place; returns the (now adapter-free) model."""
        if self.impl == "forge":
            out = self._forge.merge_lora(model)
            return out if out is not None else model
        for n, m in list(model.named_modules()):
            if isinstance(m, LoRALinear):
                parent_name, _, child = n.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child, m.merged_linear(dtype))
        return model


def quantize_base_linears(model: nn.Module, q: QLoraConfig, compute_dtype: torch.dtype) -> int:
    """Replace every ``nn.Linear`` (except excluded names) by a frozen :class:`QuantizedLinear` (QLoRA base)."""
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and not any(e and e in name for e in q.exclude_modules):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, QuantizedLinear.from_linear(m, q.bits, q.block_size, compute_dtype))
            n += 1
    return n


__all__.append("quantize_base_linears")
