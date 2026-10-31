"""GPU capability detection (ARCHITECTURE.md section 3 / 5.2).

``current_capabilities(device)`` is what kernels consult before choosing a path.  It works on CPU-only hosts
(returns a tier-"C" CPU record) and under a mocked ``torch.cuda`` (unit tests): it only needs
``torch.cuda.is_available()``, ``device_count()`` and ``get_device_properties(i)``.

Feature flags are *hypotheses from the SM level* (Plan HW-2: "nothing is enabled on the strength of the table
alone"); a probe that ran on the same device model can veto them (``fp8_confirmed`` / ``fp4_confirmed`` are
read from the stored probe report, ``None`` = never probed).  Set ``FORGE_DISABLE_FEATURES=fp8,fp4,tma,...`` to
force features off (escape hatch for a misbehaving driver/toolchain).

Tier definitions (documented deviation-free reading of Plan HW-4):

* ``A`` - Hopper / Blackwell class (sm 90, 100, 103, 110, 120, 121): FP8 (and FP4 on Blackwell) native path.
* ``B`` - Ampere / Ada / Orin (sm 80, 86, 87, 89): full Triton path, bf16, cp.async, 2:4; FP8 on 89 only.
* ``C`` - everything else (Volta/Turing, pre-Volta, unknown, CPU, non-NVIDIA): portable fallback paths.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from forge.core.errors import HardwareUnsupported
from forge.hw import chips as _chips

log = logging.getLogger("forge.hw.capabilities")

#: SM levels with a native block-scaled FP4 tensor-core MMA (hypothesis; probe confirms)
FP4_SMS = frozenset({100, 101, 103, 110, 120, 121})
TIER_A_SMS = frozenset({90, 100, 101, 103, 110, 120, 121})
TIER_B_SMS = frozenset({80, 86, 87, 88, 89})


# --------------------------------------------------------------------------------------------
# pure SM -> feature mapping
# --------------------------------------------------------------------------------------------
def sm_features(sm: int | None) -> dict[str, bool]:
    """Hypothesised hardware features for an SM level (ARCHITECTURE.md section 3)."""
    s = int(sm or 0)
    return {
        "fp16_tensor": s >= 70,
        "bf16": s >= 80,            # sm70/75 only emulate bf16 (no tensor-core path)
        "tf32": s >= 80,
        "int8_tensor": s >= 75,
        "fp8": s >= 89,
        "fp4_block_scaled": s in FP4_SMS,
        "cp_async": s >= 80,
        "tma": s >= 90,
        "wgmma": s == 90,
        "tcgen05": s in (100, 101, 103, 110),
        "two_four": s >= 80,
        "cluster_launch": s in (90, 100, 101, 103, 110),
        "triton_ok": s >= 70,
    }


def tier_for_sm(sm: int | None) -> str:
    """Support tier A/B/C for an SM level (see module docstring)."""
    s = int(sm or 0)
    if s in TIER_A_SMS:
        return "A"
    if s in TIER_B_SMS:
        return "B"
    return "C"


def demote_tier(tier: str) -> str:
    return {"A": "B", "B": "C", "C": "C"}.get(tier, "C")


def _disabled_features() -> set[str]:
    raw = os.environ.get("FORGE_DISABLE_FEATURES", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


# --------------------------------------------------------------------------------------------
# Capabilities record
# --------------------------------------------------------------------------------------------
@dataclass
class Capabilities:
    """What the current (or a hypothetical) device can do.  The first twelve fields are the contract of
    ARCHITECTURE.md 5.2; the rest are additive.  ``smem_per_sm_kib`` is the **usable opt-in shared memory per
    block** (e.g. 227 on H100) - the number kernels must stay below - ``smem_total_per_sm_kib`` is the physical
    per-SM size.  ``total_mem_gb`` is GiB."""

    sm: int
    name: str
    has_bf16: bool
    has_fp8: bool
    has_fp4_block_scaled: bool
    has_cp_async: bool
    has_tma: bool
    has_two_four: bool
    smem_per_sm_kib: float
    num_sms: int
    total_mem_gb: float
    tier: str
    # additive fields -------------------------------------------------------------------------
    index: int = 0
    device_type: str = "cuda"            # "cuda" | "cpu"
    vendor: str = "nvidia"
    family: str = "unknown"
    compute_capability: str = ""
    has_fp16: bool = True
    has_tf32: bool = False
    has_int8: bool = False
    has_wgmma: bool = False
    has_tcgen05: bool = False
    has_cluster_launch: bool = False
    bf16_emulated: bool = False
    triton_ok: bool = False
    smem_total_per_sm_kib: float = 0.0
    l2_mb: float | None = None
    regs_per_sm: int = 65536
    warp_size: int = 32
    mem_bw_gbs_datasheet: float | None = None
    chip_key: str | None = None
    spec_match: str = "none"             # exact | name_adjusted | sm_fallback | unknown | none
    fp8_confirmed: bool | None = None    # from a stored probe report; None = never probed
    fp4_confirmed: bool | None = None
    disabled: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- helpers -----------------------------------------------------------------------------
    @property
    def is_gpu(self) -> bool:
        return self.device_type == "cuda" and self.sm >= 70 and self.vendor == "nvidia"

    @property
    def sm_tag(self) -> str:
        return f"sm_{self.sm}" if self.sm else "cpu"

    def at_least(self, sm: int) -> bool:
        """True for a CUDA device at SM level >= ``sm``."""
        return self.device_type == "cuda" and self.sm >= sm

    def supports(self, feature: str) -> bool:
        """``feature`` in {bf16, fp16, tf32, int8, fp8, fp4 (block scaled), cp_async, tma, wgmma, tcgen05,
        two_four, cluster, triton}."""
        f = feature.lower().replace("-", "_")
        table = {
            "bf16": self.has_bf16, "fp16": self.has_fp16, "tf32": self.has_tf32, "int8": self.has_int8,
            "fp8": self.has_fp8, "fp4": self.has_fp4_block_scaled, "nvfp4": self.has_fp4_block_scaled,
            "mxfp4": self.has_fp4_block_scaled, "fp4_block_scaled": self.has_fp4_block_scaled,
            "cp_async": self.has_cp_async, "tma": self.has_tma, "wgmma": self.has_wgmma,
            "tcgen05": self.has_tcgen05, "two_four": self.has_two_four, "sparse_2_4": self.has_two_four,
            "cluster": self.has_cluster_launch, "triton": self.triton_ok,
        }
        if f not in table:
            raise KeyError(f"unknown capability {feature!r}")
        return bool(table[f])

    def preferred_dtype(self) -> str:
        """Best 16-bit compute dtype name for this device ('bfloat16' if native, else 'float16' on GPU)."""
        if self.device_type == "cuda" and self.is_gpu:
            return "bfloat16" if self.has_bf16 else "float16"
        return "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------------
# software environment (cheap, cached)
# --------------------------------------------------------------------------------------------
_SW_LOCK = threading.Lock()
_SW: dict[str, Any] = {}


def software_versions(refresh: bool = False) -> dict[str, Any]:
    """torch / CUDA runtime / triton versions (None when absent). Cached."""
    with _SW_LOCK:
        if _SW and not refresh:
            return dict(_SW)
        info: dict[str, Any] = {"torch": None, "cuda": None, "hip": None, "triton": None}
        try:
            import torch

            info["torch"] = getattr(torch, "__version__", None)
            info["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
            info["hip"] = getattr(getattr(torch, "version", None), "hip", None)
        except Exception:
            pass
        try:
            import importlib.metadata as md

            info["triton"] = md.version("triton")
        except Exception:
            info["triton"] = None
        _SW.clear()
        _SW.update(info)
        return dict(_SW)


def triton_available() -> bool:
    """True if a triton distribution is installed (import is NOT attempted here)."""
    return software_versions().get("triton") is not None


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# CPU info
# --------------------------------------------------------------------------------------------
def cpu_info() -> dict[str, Any]:
    """Best-effort CPU description (no hard dependency on psutil / py-cpuinfo)."""
    name = ""
    flags: set[str] = set()
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    low = line.lower()
                    if not name and (low.startswith("model name") or low.startswith("hardware")
                                     or low.startswith("cpu model")):
                        name = line.split(":", 1)[-1].strip()
                    if low.startswith("flags") or low.startswith("features"):
                        flags.update(line.split(":", 1)[-1].split())
                    if name and flags:
                        break
        except OSError:
            pass
    if not name and platform.system() == "Darwin" and shutil.which("sysctl"):
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=3).stdout.strip()
        except Exception:
            pass
    if not name:
        name = platform.processor() or platform.machine() or "unknown cpu"
    logical = os.cpu_count() or 1
    physical = logical
    ram_gb = 0.0
    freq = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or logical
        ram_gb = psutil.virtual_memory().total / 2**30
        f = psutil.cpu_freq()
        freq = float(f.max or f.current) if f else None
    except Exception:
        if hasattr(os, "sysconf"):
            try:
                ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
            except (ValueError, OSError):
                pass
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") or machine.startswith("arm"):
        family = "cpu_arm"
    elif "avx2" in flags or "avx512f" in flags or machine in ("x86_64", "amd64", "x86"):
        family = "cpu_avx"
    else:
        family = "cpu_generic"
    isa = sorted(f for f in flags if f.startswith(("avx", "amx", "fma", "neon", "sve", "sse4")))
    return {
        "name": re.sub(r"\s+", " ", name).strip(), "family": family, "machine": machine,
        "system": platform.system(), "logical_cores": logical, "physical_cores": physical,
        "ram_gb": round(ram_gb, 2), "max_freq_mhz": freq, "isa_flags": isa,
        "has_avx2": "avx2" in flags, "has_avx512": any(f.startswith("avx512") for f in flags),
        "has_amx": any(f.startswith("amx") for f in flags),
    }


def cpu_capabilities() -> Capabilities:
    """Tier-C capabilities of the host CPU (used when no CUDA device is present)."""
    ci = cpu_info()
    return Capabilities(
        sm=0, name=ci["name"], has_bf16=False, has_fp8=False, has_fp4_block_scaled=False, has_cp_async=False,
        has_tma=False, has_two_four=False, smem_per_sm_kib=0.0, num_sms=int(ci["logical_cores"]),
        total_mem_gb=float(ci["ram_gb"]), tier="C", index=-1, device_type="cpu", vendor="cpu", family=ci["family"],
        has_fp16=False, notes=["no CUDA device: portable CPU fallback paths only"],
    )


# --------------------------------------------------------------------------------------------
# GPU detection
# --------------------------------------------------------------------------------------------
_MATCH_CACHE: dict[tuple, tuple[_chips.ChipMatch, bool | None, bool | None]] = {}
_MATCH_LOCK = threading.Lock()


def clear_cache() -> None:
    """Forget cached chip matches / probe lookups (tests, after a new probe)."""
    with _MATCH_LOCK:
        _MATCH_CACHE.clear()
    with _SW_LOCK:
        _SW.clear()


def _stored_confirmations(name: str, sm: int, num_sms: int, mem_mb: int) -> tuple[bool | None, bool | None]:
    """(fp8_confirmed, fp4_confirmed) from the newest stored probe report for this device model."""
    try:
        from forge.hw import descriptor as _d
        from forge.hw import descriptor_store as _s

        fp = _d.chip_fingerprint(name, sm, num_sms, mem_mb)
        rep = _s.latest_report_for_fingerprint(fp)
        if not rep:
            return None, None
        feats = rep.get("features", {})
        return feats.get("fp8"), feats.get("fp4")
    except Exception as e:  # DB not writable, yaml missing ...
        log.debug("no stored probe confirmation: %s", e)
        return None, None


def _props_to_caps(index: int, p: Any) -> Capabilities:
    name = str(getattr(p, "name", "") or "unknown")
    major = int(getattr(p, "major", 0) or 0)
    minor = int(getattr(p, "minor", 0) or 0)
    sm = major * 10 + minor
    total = int(getattr(p, "total_memory", 0) or 0)
    nsm = int(getattr(p, "multi_processor_count", 0) or 0)
    mem_gib = total / 2**30
    hip = software_versions().get("hip")
    if hip:
        return Capabilities(sm=sm, name=name, has_bf16=False, has_fp8=False, has_fp4_block_scaled=False,
                            has_cp_async=False, has_tma=False, has_two_four=False, smem_per_sm_kib=0.0,
                            num_sms=nsm, total_mem_gb=mem_gib, tier="C", index=index, vendor="amd",
                            notes=["ROCm/HIP build of torch: Forge GPU kernels are NVIDIA-only; portable paths used"])
    key = (name, sm, nsm, total)
    with _MATCH_LOCK:
        cached = _MATCH_CACHE.get(key)
    if cached is None:
        match = _chips.find_chip(name, sm=sm, num_sms=nsm or None, mem_gb=mem_gib or None)
        conf = _stored_confirmations(name, sm, nsm, round(total / 2**20))
        cached = (match, conf[0], conf[1])
        with _MATCH_LOCK:
            _MATCH_CACHE[key] = cached
    match, fp8_conf, fp4_conf = cached
    spec = match.spec
    feats = sm_features(sm)
    off = _disabled_features()
    fp8 = feats["fp8"] and fp8_conf is not False
    fp4 = feats["fp4_block_scaled"] and fp4_conf is not False
    notes: list[str] = []
    if "fp8" in off:
        fp8 = False
        notes.append("fp8 disabled by FORGE_DISABLE_FEATURES")
    if "fp4" in off:
        fp4 = False
        notes.append("fp4 disabled by FORGE_DISABLE_FEATURES")
    if feats["fp4_block_scaled"] and fp4_conf is None and fp4:
        notes.append("fp4 block-scaled MMA is a hypothesis from the SM level - run the probe to confirm")
    smem_block = getattr(p, "shared_memory_per_block_optin", None)
    smem_sm = getattr(p, "shared_memory_per_multiprocessor", None)
    smem_usable = (smem_block / 1024.0) if smem_block else float(spec.get("smem_per_sm_kib") or 0.0)
    smem_total = (smem_sm / 1024.0) if smem_sm else float(spec.get("smem_total_per_sm_kib") or smem_usable)
    l2 = getattr(p, "L2_cache_size", None)
    l2_mb = (l2 / 2**20) if l2 else spec.get("l2_mb")
    tier = tier_for_sm(sm)
    caps = Capabilities(
        sm=sm, name=name, has_bf16=feats["bf16"] and "bf16" not in off, has_fp8=fp8,
        has_fp4_block_scaled=fp4, has_cp_async=feats["cp_async"] and "cp_async" not in off,
        has_tma=feats["tma"] and "tma" not in off, has_two_four=feats["two_four"], smem_per_sm_kib=round(smem_usable, 2),
        num_sms=nsm, total_mem_gb=round(mem_gib, 3), tier=tier, index=index, device_type="cuda", vendor="nvidia",
        family=_chips._FAMILY_BY_SM.get(sm, spec.get("family", "unknown")), compute_capability=f"{major}.{minor}",
        has_fp16=feats["fp16_tensor"], has_tf32=feats["tf32"], has_int8=feats["int8_tensor"],
        has_wgmma=feats["wgmma"] and "wgmma" not in off, has_tcgen05=feats["tcgen05"],
        has_cluster_launch=feats["cluster_launch"], bf16_emulated=(sm in (70, 72, 75)), triton_ok=feats["triton_ok"],
        smem_total_per_sm_kib=round(smem_total, 2), l2_mb=l2_mb, regs_per_sm=int(
            getattr(p, "regs_per_multiprocessor", 0) or spec.get("regs_per_sm") or 65536),
        warp_size=int(getattr(p, "warp_size", 32) or 32), mem_bw_gbs_datasheet=spec.get("mem_bw_gbs"),
        chip_key=match.key, spec_match=match.kind, fp8_confirmed=fp8_conf, fp4_confirmed=fp4_conf,
        disabled=sorted(off), notes=notes,
    )
    if sm < 70:
        caps.notes.append(f"sm_{sm} is older than Volta: Triton/tensor-core paths unavailable (tier C)")
    return caps


def current_capabilities(device: int = 0) -> Capabilities:
    """Capabilities of CUDA device ``device``; a tier-"C" CPU record when CUDA is unavailable.

    Never raises on CPU-only hosts.  Raises :class:`HardwareUnsupported` if CUDA is available but ``device`` is
    out of range."""
    try:
        import torch

        if not torch.cuda.is_available():
            return cpu_capabilities()
        n = int(torch.cuda.device_count())
        if not 0 <= device < n:
            raise HardwareUnsupported(f"CUDA device {device} not found ({n} visible)")
        return _props_to_caps(device, torch.cuda.get_device_properties(device))
    except HardwareUnsupported:
        raise
    except Exception as e:
        log.debug("capability detection failed (%s); using CPU capabilities", e)
        return cpu_capabilities()


def capabilities_for_chip(key_or_name: str) -> Capabilities:
    """Hypothetical capabilities for a catalogue chip / device name (no hardware needed) - for the planner,
    docs and kernel-dispatch tests.  Never uses stored probe results."""
    m = _chips.resolve_key_or_name(key_or_name)
    s = m.spec
    sm = int(s["sm"] or 0)
    f = sm_features(sm)
    return Capabilities(
        sm=sm, name=s["display"], has_bf16=f["bf16"], has_fp8=f["fp8"], has_fp4_block_scaled=f["fp4_block_scaled"],
        has_cp_async=f["cp_async"], has_tma=f["tma"], has_two_four=f["two_four"] and bool(s.get("tensor_cores")),
        smem_per_sm_kib=float(s.get("smem_per_sm_kib") or 0.0), num_sms=int(s.get("num_sms") or 0),
        total_mem_gb=float(s.get("mem_gb") or 0.0), tier=tier_for_sm(sm), index=-1, family=s["family"],
        compute_capability=f"{sm // 10}.{sm % 10}" if sm else "", has_fp16=f["fp16_tensor"] and bool(s.get("tensor_cores")),
        has_tf32=f["tf32"], has_int8=f["int8_tensor"], has_wgmma=f["wgmma"], has_tcgen05=f["tcgen05"],
        has_cluster_launch=f["cluster_launch"], bf16_emulated=sm in (70, 72, 75), triton_ok=f["triton_ok"],
        smem_total_per_sm_kib=float(s.get("smem_total_per_sm_kib") or 0.0), l2_mb=s.get("l2_mb"),
        mem_bw_gbs_datasheet=s.get("mem_bw_gbs"), chip_key=m.key, spec_match=m.kind,
        notes=["hypothetical: derived from the catalogue, not from a live device"],
    )


# --------------------------------------------------------------------------------------------
# GPU listing
# --------------------------------------------------------------------------------------------
def _smi_gpus() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        if out.returncode != 0:
            out = subprocess.run([exe, "--query-gpu=index,name,uuid,memory.total,driver_version",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                                 check=False)
        if out.returncode != 0:
            return []
    except Exception:
        return []
    return parse_smi_gpu_list(out.stdout)


def parse_smi_gpu_list(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version[,compute_cap]`` CSV."""
    rows = []
    for line in text.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        cc = parts[5] if len(parts) > 5 else ""
        sm = None
        m = re.match(r"^(\d+)\.(\d+)$", cc)
        if m:
            sm = int(m.group(1)) * 10 + int(m.group(2))
        try:
            mem_mib = float(parts[3])
        except ValueError:
            mem_mib = 0.0
        match = _chips.find_chip(parts[1], sm=sm, mem_gb=mem_mib / 1024 or None)
        rows.append({
            "index": int(parts[0]), "name": parts[1], "uuid": parts[2], "total_mem_gb": round(mem_mib / 1024, 2),
            "driver": parts[4] if len(parts) > 4 else None, "sm": sm if sm is not None else match.spec.get("sm"),
            "num_sms": match.spec.get("num_sms"), "compute_capability": cc or None,
            "tier": tier_for_sm(sm if sm is not None else match.spec.get("sm")), "chip_key": match.key,
            "source": "nvidia-smi", "torch_visible": False,
        })
    return rows


def list_gpus() -> list[dict[str, Any]]:
    """Visible GPUs: via ``torch.cuda`` when available, else ``nvidia-smi``, else ``[]``. Never raises."""
    try:
        import torch

        if torch.cuda.is_available():
            rows = []
            for i in range(int(torch.cuda.device_count())):
                c = _props_to_caps(i, torch.cuda.get_device_properties(i))
                p = torch.cuda.get_device_properties(i)
                rows.append({
                    "index": i, "name": c.name, "uuid": str(getattr(p, "uuid", "") or "") or None, "sm": c.sm,
                    "compute_capability": c.compute_capability, "num_sms": c.num_sms, "total_mem_gb": c.total_mem_gb,
                    "tier": c.tier, "chip_key": c.chip_key, "source": "torch", "torch_visible": True,
                    "features": {"bf16": c.has_bf16, "fp8": c.has_fp8, "fp4_block_scaled": c.has_fp4_block_scaled,
                                 "cp_async": c.has_cp_async, "tma": c.has_tma, "two_four": c.has_two_four},
                })
            return rows
    except Exception as e:
        log.debug("torch GPU listing failed: %s", e)
    return _smi_gpus()


# --------------------------------------------------------------------------------------------
# support matrix (ARCHITECTURE.md section 3) for the UI
# --------------------------------------------------------------------------------------------
_MATRIX_ROWS = [
    (70, "V100", "emulated", "no", "no", "fp16 only; Triton ok"),
    (75, "T4, RTX 20xx", "emulated", "no", "no", "fp16 only"),
    (80, "A100, A30", "yes", "no", "no", "cp.async, 2:4 sparse"),
    (86, "RTX 30xx, A10, A40, RTX A-series", "yes", "no", "no", ""),
    (87, "Jetson Orin", "yes", "no", "no", ""),
    (89, "RTX 40xx, L4, L40/L40S, RTX 6000 Ada", "yes", "yes", "no", ""),
    (90, "H100, H200, GH200", "yes", "yes", "no", "TMA, wgmma"),
    (100, "B200, GB200", "yes", "yes", "yes (NVFP4/MXFP4)", "tcgen05, tensor memory"),
    (103, "B300", "yes", "yes", "yes", ""),
    (120, "RTX 50xx, RTX PRO 6000 Blackwell", "yes", "yes", "yes (verify by probe)", "consumer Blackwell"),
    (121, "GB10 (DGX Spark)", "yes", "yes", "yes (verify by probe)", ""),
]


def support_matrix() -> list[dict[str, Any]]:
    """Rows of the section-3 GPU support matrix with tier and the catalogue chips of each SM level."""
    rows = []
    for sm, examples, bf16, fp8, fp4, notes in _MATRIX_ROWS:
        f = sm_features(sm)
        rows.append({
            "sm": sm, "examples": examples, "bf16": bf16, "fp8_mma": fp8, "fp4_block_scaled_mma": fp4,
            "cp_async": f["cp_async"], "tma": f["tma"], "two_four": f["two_four"], "tier": tier_for_sm(sm),
            "notes": notes, "chips": [c["key"] for c in _chips.list_chips(sm=sm)],
            "provenance": "vendor (hypothesis until confirmed by the probe)",
        })
    return rows
