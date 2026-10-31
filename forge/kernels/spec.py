"""Kernel specifications (Plan Appendix C.2) as typed pydantic models + YAML loader.

A :class:`KernelSpec` is *chip independent*: it states the math (``op``), dtypes, quantisation parameters,
the autotuner search space, fusion boundaries, tolerance rules and KPIs.  The layout deriver
(:mod:`forge.kernels.layouts`) and the autotuner (:mod:`forge.kernels.autotune`) instantiate it per chip.

Specs live in ``forge/kernels/specs/*.yaml`` and are loaded with :func:`load_spec`.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from forge.core.errors import ConfigError, NotFound

SPECS_DIR = Path(__file__).resolve().parent / "specs"

SplitK = Union[int, Literal["auto"]]


class QuantSpec(BaseModel):
    """Quantisation parameters of a kernel."""

    model_config = ConfigDict(extra="forbid")

    weight_bits: float = Field(16, description="Bits per stored weight element (4, 8, 16; FP4 formats count the payload only).")
    group: list[int] = Field(default_factory=list, description="Allowed group sizes along K (-1 = per output channel).")
    zero_point: Literal["none", "optional", "required"] = Field("none", description="Whether an asymmetric zero-point is stored.")
    scale_apply: str = Field("epilogue_if_per_column", description="Where scales are applied: 'prologue' (dequant before MMA), 'epilogue' or 'epilogue_if_per_column'.")
    act_scaling: str = Field("none", description="Activation scaling granularity: none | per_tensor | per_token | block_1x128 | block_16 | block_32.")
    weight_scaling: str = Field("per_group", description="Weight scaling granularity: per_group | per_channel | per_tensor | block_128x128 | block_16 | block_32.")


class SearchSpace(BaseModel):
    """Autotuner search space (Plan 6.6 step 1)."""

    model_config = ConfigDict(extra="forbid")

    tile_m: list[int] = Field(default_factory=lambda: [16, 32, 64], description="Candidate BLOCK_M values (multiples of the MMA atom M=16).")
    tile_n: list[int] = Field(default_factory=lambda: [64, 128], description="Candidate BLOCK_N values.")
    tile_k: list[int] = Field(default_factory=lambda: [64, 128], description="Candidate BLOCK_K values.")
    stages: list[int] = Field(default_factory=lambda: [2, 3, 4], description="Software-pipeline depths (num_stages).")
    warps: list[int] = Field(default_factory=lambda: [4, 8], description="num_warps candidates.")
    split_k: list[SplitK] = Field(default_factory=lambda: [1, 2, 4], description="Split-K factors; 'auto' lets the heuristic fill the SMs.")


class Tolerance(BaseModel):
    """Per-kernel tolerance rules vs the fp32 oracle (Plan 6.6 step 4)."""

    model_config = ConfigDict(extra="forbid")

    rel: float = Field(2e-3, ge=0, description="Relative tolerance for float paths (floored by 2 ulp of the output dtype).")
    abs: float = Field(1e-3, ge=0, description="Absolute tolerance for float paths.")
    rel_norm: float = Field(2e-2, ge=0, description="Max allowed max|out-ref| / max|ref| (outlier-robust whole-tensor check).")
    bitexact_int_paths: bool = Field(True, description="Integer accumulation / pack-unpack paths must be bit-exact.")


class Kpi(BaseModel):
    """Performance targets (TARGETS, not measurements)."""

    model_config = ConfigDict(extra="forbid")

    bw_fraction_min: float | None = Field(None, description="Min fraction of measured HBM bandwidth at the listed M.")
    speedup_vs_bf16_min: float | None = Field(None, description="Min speed-up vs a BF16 GEMM at the listed M.")
    tflops_fraction_min: float | None = Field(None, description="Min fraction of vendor GEMM throughput on large shapes.")
    at_M: list[int] = Field(default_factory=list, description="Batch sizes the bandwidth/speed-up KPIs apply to.")


class KernelSpec(BaseModel):
    """One kernel family specification (Plan Appendix C.2)."""

    model_config = ConfigDict(extra="forbid")

    kernel: str = Field(..., description="Unique kernel name, e.g. 'w4a16_gemm'.")
    family: str = Field("gemm", description="Kernel family used by the catalog/UI grouping (gemm, attention, moe, util).")
    op: str = Field(..., description="Mathematical definition of the operation.")
    description: str = Field("", description="Human readable summary.")
    dtypes: dict[str, str] = Field(default_factory=dict, description="Operand dtypes: x, w, scale, acc, y.")
    compute_dtype: str = Field("bf16", description="Matrix-unit operand dtype used by the MMA.")
    quant: QuantSpec = Field(default_factory=QuantSpec, description="Quantisation parameters.")
    shape_hints: dict[str, Any] = Field(default_factory=dict, description="Typical shapes; 'model_native' means taken from the model.")
    search: SearchSpace = Field(default_factory=SearchSpace, description="Autotuner search space.")
    fusion: dict[str, Any] = Field(default_factory=dict, description="Fusion boundaries: prologue / epilogue ops.")
    layout: dict[str, Any] = Field(default_factory=dict, description="Layout requirements: weights, scales, align_bytes.")
    oracle: str = Field(..., description="Dotted name of the pure-torch oracle in forge.kernels.ref.")
    tolerance: Tolerance = Field(default_factory=Tolerance, description="Tolerance rules.")
    kpi: Kpi = Field(default_factory=Kpi, description="Performance KPIs (targets).")
    min_sm: int = Field(70, ge=0, description="Minimum compute capability (e.g. 89) for the native Triton path; refs run anywhere.")
    experimental: bool = Field(False, description="Plan-gated / not yet hardware-validated feature; the UI flags it.")
    notes: list[str] = Field(default_factory=list, description="Free-form honest notes (validation status etc.).")

    @field_validator("kernel")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if not v or not v.replace("_", "").isalnum():
            raise ValueError("kernel name must be alphanumeric/underscore")
        return v

    # -- helpers -----------------------------------------------------------------------------------
    @property
    def weight_bits(self) -> float:
        return float(self.quant.weight_bits)

    def split_k_values(self, default_auto: int = 0) -> list[int]:
        """Integer split-K candidates; ``'auto'`` becomes ``default_auto`` (0 = decide per shape)."""
        return [default_auto if s == "auto" else int(s) for s in self.search.split_k]


_LOCK = threading.Lock()
_CACHE: dict[str, KernelSpec] = {}


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"kernel spec {path} must be a YAML mapping")
    return data


def list_specs() -> list[str]:
    """Names of all bundled specs (file stems of ``specs/*.yaml``)."""
    if not SPECS_DIR.is_dir():
        return []
    return sorted(p.stem for p in SPECS_DIR.glob("*.yaml"))


def load_spec(name: str, *, reload: bool = False) -> KernelSpec:
    """Load ``specs/<name>.yaml`` (cached).  Raises :class:`NotFound` / :class:`ConfigError`."""
    with _LOCK:
        if not reload and name in _CACHE:
            return _CACHE[name]
    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
        raise NotFound(f"no kernel spec named {name!r} (names are alphanumeric/underscore; available: {', '.join(list_specs()) or 'none'})")
    path = SPECS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise NotFound(f"no kernel spec named {name!r} (available: {', '.join(list_specs()) or 'none'})")
    try:
        spec = KernelSpec(**_read_yaml(path))
    except ConfigError:
        raise
    except Exception as e:  # pydantic ValidationError, yaml errors
        raise ConfigError(f"invalid kernel spec {path.name}: {e}") from e
    with _LOCK:
        _CACHE[name] = spec
    return spec


def load_spec_from_dict(data: dict) -> KernelSpec:
    """Validate an in-memory spec (used by tests / UI previews)."""
    try:
        return KernelSpec(**data)
    except Exception as e:
        raise ConfigError(f"invalid kernel spec: {e}") from e
