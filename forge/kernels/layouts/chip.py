"""Light-weight chip view used by the layout deriver / autotuner / dispatchers.

Accepts (duck-typed, so it works before/without :mod:`forge.hw` is importable):
``None`` (current device, or CPU), a :class:`ChipInfo`, a ``forge.hw.capabilities.Capabilities``,
a ``forge.core.schemas.ChipDescriptor`` or a plain ``dict``.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger("forge.kernels.layouts")

# datasheet-level defaults per compute capability (used only when the descriptor/probe has no number)
_SM_DEFAULTS: dict[int, dict[str, Any]] = {
    70: dict(family="volta", smem=96, sms=80, cp_async=False, tma=False, fp8=False, fp4=False),
    75: dict(family="turing", smem=64, sms=40, cp_async=False, tma=False, fp8=False, fp4=False),
    80: dict(family="ampere", smem=164, sms=108, cp_async=True, tma=False, fp8=False, fp4=False),
    86: dict(family="ampere", smem=100, sms=82, cp_async=True, tma=False, fp8=False, fp4=False),
    87: dict(family="ampere", smem=164, sms=16, cp_async=True, tma=False, fp8=False, fp4=False),
    89: dict(family="ada", smem=100, sms=128, cp_async=True, tma=False, fp8=True, fp4=False),
    90: dict(family="hopper", smem=227, sms=132, cp_async=True, tma=True, fp8=True, fp4=False),
    100: dict(family="blackwell", smem=227, sms=148, cp_async=True, tma=True, fp8=True, fp4=True),
    103: dict(family="blackwell", smem=227, sms=160, cp_async=True, tma=True, fp8=True, fp4=True),
    120: dict(family="blackwell_consumer", smem=100, sms=170, cp_async=True, tma=True, fp8=True, fp4=True),
    121: dict(family="blackwell_consumer", smem=100, sms=48, cp_async=True, tma=True, fp8=True, fp4=True),
}


@dataclass(frozen=True)
class ChipInfo:
    """Everything the layout deriver and dispatchers need to know about a device."""

    sm: int | None = None  # None => CPU / unknown
    name: str = "cpu"
    family: str = "cpu"
    fingerprint: str = "cpu"
    smem_per_sm_kib: int = 0
    num_sms: int = 0
    vec_bytes: int = 16
    smem_banks: int = 32
    bank_width_bytes: int = 4
    regs_per_sm: int = 65536
    max_regs_per_thread: int = 255
    has_bf16: bool = False
    has_fp8: bool = False
    has_fp4_block_scaled: bool = False
    has_cp_async: bool = False
    has_tma: bool = False
    hbm_gbs: float | None = None
    l2_mb: float | None = None
    total_mem_gb: float | None = None
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def is_cuda(self) -> bool:
        return self.sm is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra", None)
        return d


def _get(obj: Any, path: str, default: Any = None) -> Any:
    """getattr / dict lookup following dotted ``path`` (works on pydantic models, dataclasses, dicts)."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def chip_from_sm(sm: int | None, name: str | None = None, **over: Any) -> ChipInfo:
    """Build a :class:`ChipInfo` from a compute capability using datasheet defaults."""
    if sm is None:
        return ChipInfo(**over)
    d = _SM_DEFAULTS.get(sm)
    if d is None:  # unknown future/older sm: use the nearest lower known one
        lower = [k for k in _SM_DEFAULTS if k <= sm]
        d = _SM_DEFAULTS[max(lower)] if lower else _SM_DEFAULTS[70]
    kw: dict[str, Any] = dict(
        sm=int(sm),
        name=name or f"sm_{sm}",
        family=d["family"],
        fingerprint=f"sm{sm}-{(name or 'generic').replace(' ', '_')}",
        smem_per_sm_kib=d["smem"],
        num_sms=d["sms"],
        has_bf16=sm >= 80,
        has_fp8=bool(d["fp8"]),
        has_fp4_block_scaled=bool(d["fp4"]),
        has_cp_async=bool(d["cp_async"]),
        has_tma=bool(d["tma"]),
    )
    kw.update(over)
    return ChipInfo(**kw)


def _from_cuda_device(device: int = 0) -> ChipInfo | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device)
        sm = props.major * 10 + props.minor
        return chip_from_sm(
            sm,
            name=props.name,
            num_sms=int(props.multi_processor_count),
            total_mem_gb=round(props.total_memory / 2**30, 2),
        )
    except Exception as e:  # pragma: no cover - needs a GPU
        log.debug("cuda device probe failed: %s", e)
        return None


def chip_info(chip: Any = None, device: int = 0) -> ChipInfo:
    """Normalise ``chip`` to a :class:`ChipInfo`.  ``None`` means "the current device" (CPU info when no CUDA)."""
    if isinstance(chip, ChipInfo):
        return chip
    if chip is None:
        try:  # prefer the hw module (it may know measured numbers)
            from forge.hw.capabilities import current_capabilities  # type: ignore

            caps = current_capabilities(device)
            if getattr(caps, "sm", None):
                return chip_info(caps)
        except Exception:
            pass
        return _from_cuda_device(device) or ChipInfo()
    sm = _get(chip, "sm")
    if sm is None:
        cc = _get(chip, "compute_capability")
        if isinstance(cc, str) and "." in cc:
            try:
                major, minor = cc.split(".")[:2]
                sm = int(major) * 10 + int(minor)
            except ValueError:
                sm = None
    if sm is None:
        return ChipInfo(name=str(_get(chip, "name", "cpu")))
    sm = int(sm)
    base = chip_from_sm(sm, name=_get(chip, "name"))
    over: dict[str, Any] = {}
    smem = _get(chip, "smem_per_sm_kib", _get(chip, "on_chip.smem_per_cu_kib"))
    if smem:
        over["smem_per_sm_kib"] = int(smem)
    nsm = _get(chip, "num_sms", _get(chip, "compute_units"))
    if nsm:
        over["num_sms"] = int(nsm)
    vec = _get(chip, "memory.max_vector_load_bytes", _get(chip, "vec_bytes"))
    if vec:
        over["vec_bytes"] = int(vec)
    banks = _get(chip, "on_chip.smem_banks")
    if banks:
        over["smem_banks"] = int(banks)
    bw = _get(chip, "bank_width_bytes", _get(chip, "on_chip.bank_width_bytes"))
    if bw:
        over["bank_width_bytes"] = int(bw)
    for flag, attr in (
        ("has_bf16", "has_bf16"),
        ("has_fp8", "has_fp8"),
        ("has_fp4_block_scaled", "has_fp4_block_scaled"),
        ("has_cp_async", "has_cp_async"),
        ("has_tma", "has_tma"),
    ):
        v = _get(chip, attr)
        if v is not None:
            over[flag] = bool(v)
    fp = _get(chip, "fingerprint", _get(chip, "id"))
    if fp:
        over["fingerprint"] = str(fp)
    hbm = _get(chip, "memory.hbm_bw_gbs.measured", _get(chip, "memory.hbm_bw_gbs.datasheet", _get(chip, "mem_bw_gbs_datasheet")))
    if isinstance(hbm, (int, float)):
        over["hbm_gbs"] = float(hbm)
    tm = _get(chip, "total_mem_gb", _get(chip, "memory.hbm_gb"))
    if isinstance(tm, (int, float)):
        over["total_mem_gb"] = float(tm)
    fam = _get(chip, "family")
    if isinstance(fam, str) and fam:
        over["family"] = fam
    name = _get(chip, "name")
    if isinstance(name, str) and name:
        over["name"] = name
    return chip_from_sm(sm, **{**{"name": over.get("name", base.name)}, **{k: v for k, v in over.items() if k != "name"}})
