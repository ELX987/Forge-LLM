"""Precision ladder (Plan 7.4 / TR-1): R0 BF16 autocast, R1 FP8 mixed precision, R2 NVFP4 pilot (experimental, emulated).

* **R0** -- autocast (bf16 on CUDA sm>=80, fp16 + grad scaler on older GPUs, fp32 on CPU).
* **R1** -- FP8 GEMMs.  Backends, in order of preference: ``torchao`` float8 training (sm>=89), ``transformer_engine`` (sm>=89),
  else the *emulated* fake-quant fallback (per-tensor dynamic scaling, E4M3 forward / E5M2 gradients) -- always announced
  with a WARNING because it gives no speed-up and is only numerically indicative.  The torchao / TE paths are written
  against the public APIs but have NOT been run in this repository (no GPU available at authoring time).
* **R2** -- NVFP4 training pilot, emulated in pure torch (:mod:`forge.train.fp4sim`): activations 1x16 blocks, weights 16x16 tiles
  (2D scaling), stochastic rounding + optional random Hadamard transform (token dimension) on the gradient GEMM operands,
  E4M3 block scales with an FP32 tensor scale.  Native FP4 training kernels are not wired; ``has_fp4_block_scaled`` hardware is
  only logged.  Gated behind ``allow_unvalidated_fp4``.
* Per-layer exemptions (embeddings / norms / head / routers by name pattern, first and last N blocks) and a final
  high-precision *healing* phase.

Modules are converted by swapping ``nn.Linear.__class__`` to :class:`QuantLinear`, so parameter names, shapes, optimiser state
and checkpoints are unchanged and conversion is reversible.
"""
from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError
from forge.train.config import PrecisionConfig
from forge.train.diagnostics import scale_stats, sr_bias_test, zero_grad_tile_stats
from forge.train.fp4sim import FP4Rng, fake_quant_fp8, fake_quant_nvfp4, random_hadamard

log = logging.getLogger("forge.train.precision")

__all__ = ["PrecisionManager", "QuantLinear", "is_exempt", "make_precision_manager", "register_quant_mode", "resolve_compute_dtype"]

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")


# ---------------------------------------------------------------------------------------------------------------
# autograd functions
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class _FP4Settings:
    block: int = 16
    weight_2d: bool = True
    sr: bool = True
    rht: bool = True
    rht_block: int = 16
    mm_dtype: torch.dtype = torch.float32
    seed: int = 0
    rng: FP4Rng | None = None


def _gemm_nt(a: torch.Tensor, b: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """``a [M,K] @ b[N,K].T`` computed in ``dt`` (operands are already-dequantised fake-quant values)."""
    return a.to(dt) @ b.to(dt).t()


class _FP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, w: torch.Tensor, st: _FP4Settings) -> torch.Tensor:  # type: ignore[override]
        B = st.block
        xq = fake_quant_nvfp4(x, (1, B), "rtn")
        wq = fake_quant_nvfp4(w, (B, B) if st.weight_2d else (1, B), "rtn")
        ctx.save_for_backward(x, w)
        ctx.st = st
        return _gemm_nt(xq, wq, st.mm_dtype).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, gy: torch.Tensor):  # type: ignore[override]
        x, w = ctx.saved_tensors
        st: _FP4Settings = ctx.st
        B = st.block
        rounding = "sr" if st.sr else "rtn"
        gy = gy.contiguous()
        # --- dgrad: dX = Q(gY; 1xB over O) @ Q(W; blocks over O) ---------------------------------------
        gq = fake_quant_nvfp4(gy, (1, B), rounding, st.rng)
        if st.weight_2d:
            w_dg = fake_quant_nvfp4(w, (B, B), "rtn")  # [O,K]; 2D tiles are transpose-symmetric
        else:
            w_dg = fake_quant_nvfp4(w.t().contiguous(), (1, B), "rtn").t()  # blocks along O
        dx = (gq.to(st.mm_dtype) @ w_dg.to(st.mm_dtype)).to(x.dtype)
        # --- wgrad: dW = Q(gY^T) @ Q(X^T), contraction over tokens N (RHT along N) --------------------------
        N = gy.shape[0]
        mult = max(B, st.rht_block if st.rht else 1)
        pad = (-N) % mult
        a = gy.t().contiguous().to(torch.float32)  # [O,N]
        b = x.t().contiguous().to(torch.float32)  # [K,N]
        if pad:
            a = torch.nn.functional.pad(a, (0, pad))
            b = torch.nn.functional.pad(b, (0, pad))
        if st.rht:
            a = random_hadamard(a, st.rht_block, st.seed)
            b = random_hadamard(b, st.rht_block, st.seed)
        qa = fake_quant_nvfp4(a, (1, B), rounding, st.rng)
        qb = fake_quant_nvfp4(b, (1, B), "rtn")
        dw = _gemm_nt(qa, qb, st.mm_dtype).to(w.dtype)
        return dx, dw, None


class _FP8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, w: torch.Tensor, grad_fmt: str, mm_dtype: torch.dtype) -> torch.Tensor:  # type: ignore[override]
        xq = fake_quant_fp8(x, "e4m3")
        wq = fake_quant_fp8(w, "e4m3")
        ctx.save_for_backward(xq.to(mm_dtype), wq.to(mm_dtype))
        ctx.grad_fmt, ctx.mm, ctx.xdt, ctx.wdt = grad_fmt, mm_dtype, x.dtype, w.dtype
        return (xq.to(mm_dtype) @ wq.to(mm_dtype).t()).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, gy: torch.Tensor):  # type: ignore[override]
        xq, wq = ctx.saved_tensors
        gq = fake_quant_fp8(gy, ctx.grad_fmt).to(ctx.mm)
        dx = (gq @ wq).to(ctx.xdt)
        dw = (gq.t() @ xq).to(ctx.wdt)
        return dx, dw, None, None


# ---------------------------------------------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------------------------------------------
class QuantLinear(nn.Linear):
    """``nn.Linear`` whose GEMMs are fake-quantised according to ``qmode`` in {'off', 'fp8', 'fp4'}.

    Instances are produced by swapping ``__class__`` of an existing ``nn.Linear`` (see :meth:`PrecisionManager.prepare`);
    attributes ``qmode``, ``_fp4`` (:class:`_FP4Settings`), ``_fp8_grad_fmt`` and ``_mm_dtype`` are set there.
    """

    qmode: str = "off"

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        mode = self.qmode
        if mode == "off":
            return nn.Linear.forward(self, x)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if mode == "fp4":
            y = _FP4LinearFn.apply(x2, self.weight.to(x2.dtype) if self.weight.dtype != x2.dtype else self.weight, self._fp4)
        elif mode == "fp8":
            w = self.weight.to(x2.dtype) if self.weight.dtype != x2.dtype else self.weight
            y = _FP8LinearFn.apply(x2, w, self._fp8_grad_fmt, self._mm_dtype)
        else:
            raise RuntimeError(f"bad qmode {mode!r}")
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*lead, y.shape[-1])


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def resolve_compute_dtype(pref: str, device_type: str) -> torch.dtype | None:
    """Autocast dtype, or None for no autocast (pure fp32)."""
    if pref == "float32":
        return None
    if pref == "bfloat16":
        return torch.bfloat16
    if pref == "float16":
        return torch.float16
    if device_type == "cuda":
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:  # noqa: BLE001
            return torch.float16
    return None


def _num_layers(names: list[str]) -> int:
    idx = [int(m.group(1)) for n in names if (m := _LAYER_RE.search(n))]
    return max(idx) + 1 if idx else 0


def is_exempt(name: str, cfg: PrecisionConfig, n_layers: int) -> bool:
    """True if module ``name`` stays in high precision under the exemption rules."""
    lname = name.lower()
    if any(p.lower() in lname for p in cfg.exempt_patterns):
        return True
    # MoE routers: module names carry no ".weight" suffix, so the "gate.weight" pattern alone never matched ``...mlp.gate``.
    # Match the path component exactly so expert/dense ``gate_proj`` stays eligible.
    last = lname.rsplit(".", 1)[-1]
    if any(p.lower().endswith(".weight") and last == p.lower()[: -len(".weight")] for p in cfg.exempt_patterns) or last == "shared_expert_gate":
        return True
    m = _LAYER_RE.search(name)
    if m and n_layers:
        i = int(m.group(1))
        if i < cfg.exempt_first_layers or i >= n_layers - cfg.exempt_last_layers:
            return True
    return False


def _cuda_cc() -> tuple[int, int]:
    try:
        return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    except Exception:  # noqa: BLE001
        return (0, 0)


def _has(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001
        return False


class PrecisionManager:
    """Owns the precision rung of one run: converts the model, switches phases (healing), supplies autocast and diagnostics."""

    def __init__(self, cfg: PrecisionConfig, *, device: torch.device | str = "cpu", total_steps: int = 0, seed: int = 0, rank: int = 0) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.autocast_dtype = resolve_compute_dtype(cfg.compute_dtype, self.device.type)
        self.needs_scaler = self.autocast_dtype == torch.float16 and self.device.type == "cuda"
        self.rng = FP4Rng(cfg.seed + seed * 7919 + rank)
        self.backend = "none"
        self.phase = "main"
        self._hp_depth = 0
        self.summary: dict[str, Any] = {}
        self._quant_names: list[str] = []
        self._te_active = False
        self._te_mod: Any = None
        self._te_recipe: Any = None
        self._model: Any = None
        self._converted_torchao = False
        if cfg.rung == "R2" and not cfg.allow_unvalidated_fp4:
            raise ConfigError(
                "precision.rung=R2 (NVFP4 training) is experimental and gated: set precision.allow_unvalidated_fp4=true after "
                "reviewing the TR-2 diagnostics plan (Plan R11).")

    # -- phases --------------------------------------------------------------------------------------
    @property
    def rung_mode(self) -> str:
        return {"R0": "off", "R1": "fp8", "R2": "fp4"}[self.cfg.rung]

    @property
    def healing_start(self) -> int | None:
        c = self.cfg
        if self.cfg.rung == "R0" or self.total_steps <= 0:
            return None
        if c.healing_steps > 0:
            return max(0, self.total_steps - c.healing_steps)
        if c.healing_fraction > 0:
            return max(0, self.total_steps - int(round(c.healing_fraction * self.total_steps)))
        return None

    @property
    def metric_prefix(self) -> str:
        return "fp4" if self.cfg.rung == "R2" else "fp8"

    @property
    def quantised(self) -> bool:
        return bool(self._quant_names) or self._te_active or self._converted_torchao

    @property
    def wants_diagnostics(self) -> bool:
        """Whether the trainer should call :meth:`diagnostics` on its cadence (quant-mode plug-ins override)."""
        return self.cfg.rung != "R0"

    @property
    def wants_loss_gap(self) -> bool:
        """Whether the trainer should run the shadow high-precision loss-gap forward on its cadence."""
        return self.cfg.rung != "R0"

    def drain_events(self) -> list[tuple[str, str]]:
        """``[(level, message)]`` raised since the last call (level 'warning' | 'error'); the trainer logs them and aborts on 'error'."""
        return []

    def set_step(self, step: int) -> bool:
        """Switch phase if needed (healing). Returns True when the phase changed."""
        hs = self.healing_start
        new = "healing" if (hs is not None and step >= hs) else "main"
        if new == self.phase:
            return False
        self.phase = new
        self._apply_mode(self._current_mode())
        log.info("precision phase -> %s at step %d (mode=%s)", new, step, self._current_mode())
        return True

    def _current_mode(self) -> str:
        if self._hp_depth > 0:
            return "off"
        if self.phase == "healing":
            return {"R0": "off", "R1": "fp8"}[self.cfg.healing_rung]
        return self.rung_mode

    def _apply_mode(self, mode: str) -> None:
        if self._model is None:
            return
        for _, m in self._model.named_modules():
            if isinstance(m, QuantLinear) and getattr(m, "_forge_q", False):
                m.qmode = mode if (mode != "fp4" or getattr(m, "_fp4", None) is not None) else "off"
        if self._te_active:
            self._te_enabled = mode != "off"
        if self._converted_torchao and mode == "off" and self.phase == "healing":
            for _, m in self._model.named_modules():
                if type(m).__name__ == "Float8Linear":
                    m.__class__ = nn.Linear

    # -- conversion -----------------------------------------------------------------------------------
    def _resolve_r1_backend(self) -> str:
        want = self.cfg.backend
        cc = _cuda_cc()
        hw_ok = self.device.type == "cuda" and cc >= (8, 9)
        if want in ("auto", "torchao") and hw_ok and _has("torchao"):
            return "torchao"
        if want in ("auto", "transformer_engine") and hw_ok and _has("transformer_engine"):
            return "transformer_engine"
        if want in ("torchao", "transformer_engine"):
            log.warning("FP8 backend %r requested but unavailable (cuda=%s cc=%s importable=%s); using the emulated fallback",
                        want, self.device.type == "cuda", cc, _has(want))
        return "emulated"

    def prepare(self, model: nn.Module) -> dict[str, Any]:
        """Convert eligible linears for the configured rung (call before creating the optimiser). Returns a summary dict."""
        self._model = model
        cfg = self.cfg
        names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        n_layers = _num_layers(names)
        exempt, skipped_shape, eligible = [], [], []
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if cfg.rung == "R0":
                continue
            if "lora_" in n or ".lora" in n:  # adapter matrices (tiny, high-precision by design)
                exempt.append(n)
            elif is_exempt(n, cfg, n_layers):
                exempt.append(n)
            elif cfg.rung == "R2" and (m.in_features % cfg.fp4_block or (cfg.fp4_weight_2d and m.out_features % cfg.fp4_block)):
                skipped_shape.append(n)
            elif m.in_features < 16 or m.out_features < 16:
                skipped_shape.append(n)
            else:
                eligible.append(n)
        self.summary = {"rung": cfg.rung, "n_linear": len(names), "n_layers": n_layers, "eligible": len(eligible), "exempt": len(exempt),
                        "skipped_shape": len(skipped_shape), "backend": "none", "exempt_names": exempt[:50]}
        if cfg.rung == "R0":
            self.backend = "autocast"
            self.summary["backend"] = self.backend
            return self.summary
        if cfg.rung == "R1":
            self.backend = self._resolve_r1_backend()
            if self.backend == "emulated":
                log.warning("R1 FP8: running the EMULATED fake-quant fallback (no FP8 tensor-core backend). Numerics only -- no speed-up.")
            elif self.backend == "torchao":
                self._convert_torchao(model, set(eligible))
            else:
                self._convert_te(model, set(eligible))
        else:  # R2
            self.backend = "emulated"
            self._log_fp4_hardware()
            log.warning("R2 NVFP4: running the EMULATED fake-quant pilot (experimental, unvalidated numerics, no speed-up).")
        if self.backend == "emulated":
            self._convert_emulated(model, set(eligible))
        self.summary["backend"] = self.backend
        self.summary["quantised"] = len(self._quant_names) if self.backend == "emulated" else len(eligible)
        self._apply_mode(self._current_mode())
        return self.summary

    def _log_fp4_hardware(self) -> None:
        try:
            from forge.hw.capabilities import current_capabilities

            caps = current_capabilities()
            log.info("hardware: sm=%s has_fp4_block_scaled=%s (native FP4 training kernels are not wired; emulating)",
                     getattr(caps, "sm", "?"), getattr(caps, "has_fp4_block_scaled", "?"))
        except Exception:  # noqa: BLE001
            pass

    def _mm_dtype(self) -> torch.dtype:
        pref = self.cfg.fp4_matmul_dtype
        if pref == "float32":
            return torch.float32
        if pref == "bfloat16":
            return torch.bfloat16
        return torch.bfloat16 if self.device.type == "cuda" else torch.float32

    def _convert_emulated(self, model: nn.Module, eligible: set[str]) -> None:
        cfg = self.cfg
        mm = self._mm_dtype()
        for n, m in model.named_modules():
            if n in eligible and isinstance(m, nn.Linear) and not isinstance(m, QuantLinear):
                m._forge_orig_cls = type(m)
                m.__class__ = QuantLinear
                m._forge_q = True
                m.qmode = "off"
                m._mm_dtype = mm
                m._fp8_grad_fmt = "e5m2" if cfg.fp8_format == "hybrid" else "e4m3"
                if cfg.rung == "R2":
                    m._fp4 = _FP4Settings(block=cfg.fp4_block, weight_2d=cfg.fp4_weight_2d, sr=cfg.fp4_stochastic_rounding,
                                          rht=cfg.fp4_rht_wgrad, rht_block=cfg.fp4_rht_block, mm_dtype=mm,
                                          seed=cfg.seed + self.seed, rng=self.rng)
                else:
                    m._fp4 = None
                self._quant_names.append(n)

    def _convert_torchao(self, model: nn.Module, eligible: set[str]) -> None:
        try:
            from torchao.float8 import convert_to_float8_training  # type: ignore

            def flt(mod: nn.Module, fqn: str) -> bool:
                return isinstance(mod, nn.Linear) and fqn in eligible and mod.in_features % 16 == 0 and mod.out_features % 16 == 0

            convert_to_float8_training(model, module_filter_fn=flt)
            self._converted_torchao = True
            log.info("R1 FP8: torchao float8 training enabled (NOT validated in this repo; verify on hardware)")
        except Exception as e:  # noqa: BLE001
            log.warning("torchao float8 conversion failed (%r); using the emulated fallback", e)
            self.backend = "emulated"

    def _convert_te(self, model: nn.Module, eligible: set[str]) -> None:
        try:
            import transformer_engine.pytorch as te  # type: ignore
            from transformer_engine.common import recipe as te_recipe  # type: ignore

            fmt = te_recipe.Format.HYBRID if self.cfg.fp8_format == "hybrid" else te_recipe.Format.E4M3
            self._te_recipe = te_recipe.DelayedScaling(fp8_format=fmt) if self.cfg.fp8_scaling == "delayed" else getattr(te_recipe, "Float8CurrentScaling", te_recipe.DelayedScaling)(fp8_format=fmt)
            for n, m in list(model.named_modules()):
                if n in eligible and isinstance(m, nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0:
                    new = te.Linear(m.in_features, m.out_features, bias=m.bias is not None, params_dtype=m.weight.dtype, device=m.weight.device)
                    with torch.no_grad():
                        new.weight.copy_(m.weight)
                        if m.bias is not None:
                            new.bias.copy_(m.bias)
                    parent_name, _, child = n.rpartition(".")
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child, new)
            self._te_mod = te
            self._te_active = True
            self._te_enabled = True
            log.info("R1 FP8: TransformerEngine Linear + fp8 autocast enabled (NOT validated in this repo; verify on hardware)")
        except Exception as e:  # noqa: BLE001
            log.warning("TransformerEngine FP8 conversion failed (%r); using the emulated fallback", e)
            self.backend = "emulated"

    def restore(self, model: nn.Module | None = None) -> int:
        """Undo the ``QuantLinear`` class swap (e.g. before exporting a model). Returns the number of restored modules."""
        n = 0
        for _, m in (model or self._model).named_modules():
            if isinstance(m, QuantLinear) and getattr(m, "_forge_q", False):
                m.__class__ = getattr(m, "_forge_orig_cls", nn.Linear)
                for attr in ("_forge_q", "qmode", "_fp4", "_fp8_grad_fmt", "_mm_dtype", "_forge_orig_cls"):
                    if attr in m.__dict__:
                        del m.__dict__[attr]
                n += 1
        self._quant_names = []
        return n

    # -- contexts -----------------------------------------------------------------------------------
    @contextlib.contextmanager
    def autocast(self):  # type: ignore[no-untyped-def]
        with contextlib.ExitStack() as st:
            if self.autocast_dtype is not None:
                st.enter_context(torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype))
            if self._te_active and getattr(self, "_te_enabled", False):
                st.enter_context(self._te_mod.fp8_autocast(enabled=True, fp8_recipe=self._te_recipe))
            yield

    @contextlib.contextmanager
    def high_precision(self):  # type: ignore[no-untyped-def]
        """Temporarily disable all quantised GEMMs (evaluation / shadow reference forward)."""
        if not self.quantised:
            yield
            return
        self._hp_depth += 1
        self._apply_mode("off")
        try:
            yield
        finally:
            self._hp_depth -= 1
            self._apply_mode(self._current_mode())

    # -- diagnostics (TR-2) ---------------------------------------------------------------------------
    def diagnostics(self, model: nn.Module) -> dict[str, float]:
        """Sample quantised layers and report TR-2 statistics. Call after backward (gradients present), before zero_grad."""
        cfg = self.cfg
        if cfg.rung == "R0" or not self._quant_names:
            return {}
        mods = [(n, m) for n, m in model.named_modules() if isinstance(m, QuantLinear) and getattr(m, "_forge_q", False) and m.weight.ndim == 2]
        if not mods:
            return {}
        k = min(cfg.diag_max_tensors, len(mods))
        step = max(1, len(mods) // k)
        sample = mods[::step][:k]
        p = self.metric_prefix
        acc: dict[str, list[float]] = {}

        def add(key: str, v: float) -> None:
            acc.setdefault(key, []).append(float(v))

        for _, m in sample:
            w = m.weight.detach()
            if cfg.rung == "R2":
                wblock = (cfg.fp4_block, cfg.fp4_block) if cfg.fp4_weight_2d else (1, cfg.fp4_block)
                ss = scale_stats(w.float(), wblock)
                add(f"{p}/scale_flush_frac", ss["flush_frac"])
                add(f"{p}/scale_subnormal_frac", ss["subnormal_frac"])
                add(f"{p}/scale_occupancy", ss["occupancy"])
                add(f"{p}/saturation_frac", ss["saturated_frac"])
            g = m.weight.grad
            if g is not None and g.ndim == 2:
                zs = zero_grad_tile_stats(g)
                add(f"{p}/zero_grad_tile_frac", zs["zero_tile_frac"])
                add(f"{p}/zero_grad_elem_frac", zs["zero_elem_frac"])
                if cfg.rung == "R2" and g.shape[1] >= 16 and torch.isfinite(g).all():
                    rows = max(16, min(g.shape[0], max(16, (1 << 20) // g.shape[1])) // 16 * 16)
                    bt = sr_bias_test(g[:rows].float(), cfg.sr_bias_draws, (1, cfg.fp4_block), seed=self.seed)
                    add(f"{p}/sr_bias", bt["rel_bias"])
                    add(f"{p}/sr_bias_z", bt["z"])
                    add(f"{p}/rtn_rel_err", bt["rtn_rel_err"])
        out = {k_: sum(v) / len(v) for k_, v in acc.items()}
        zk = f"{p}/zero_grad_tile_frac"
        if zk in acc:
            out[zk + "_max"] = max(acc[zk])
        return out


# ---------------------------------------------------------------------------------------------------------------
# quant-aware mode plug-ins (ADR 0007): modes subclass PrecisionManager (same interface: prepare, set_step, autocast, high_precision,
# diagnostics, restore, quantised, metric_prefix, wants_*, drain_events) and register under a ``PrecisionConfig.quant_aware`` value.
# ---------------------------------------------------------------------------------------------------------------
_QUANT_MODES: dict[str, str] = {"ternary": "forge.ternary.qat:TernaryPrecisionManager"}


def register_quant_mode(name: str, target: str) -> None:
    """Register ``name`` -> ``"module:Class"`` (lazy import) for ``PrecisionConfig.quant_aware``."""
    _QUANT_MODES[name] = target


def make_precision_manager(cfg: PrecisionConfig, **kw: Any) -> PrecisionManager:
    """The manager for ``cfg``: the registered quant-aware plug-in when ``cfg.quant_aware != 'none'``, else :class:`PrecisionManager`."""
    mode = getattr(cfg, "quant_aware", "none")
    if mode == "none":
        return PrecisionManager(cfg, **kw)
    if mode not in _QUANT_MODES:
        raise ConfigError(f"unknown precision.quant_aware mode {mode!r}; registered: {sorted(_QUANT_MODES)}")
    import importlib

    mod, _, cls = _QUANT_MODES[mode].partition(":")
    return getattr(importlib.import_module(mod), cls)(cfg, **kw)

