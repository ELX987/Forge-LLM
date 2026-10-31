"""NVIDIA chip database: datasheet specifications for datacenter, workstation, consumer and edge GPUs.

HONESTY NOTICE
--------------
Everything here is *vendor datasheet / whitepaper data as known to the author, with no network access when this
file was written* - none of it is measured.  Each entry carries a ``confidence`` map:

``datasheet``     value matches the public NVIDIA datasheet/whitepaper as far as the author knows.
``derived``       computed from ``num_sms x clock x per-SM tensor rate`` of the architecture (see ``_RATE``).
                  Used mainly for consumer cards where NVIDIA publishes marketing "AI TOPS" rather than a table.
``approximate``   the author is not sure of the number (new part, several SKUs sharing one name, third-party
                  figure ...).  The ``notes`` field says why.

The probe (:mod:`forge.hw.probe`) replaces datasheet numbers with measured ones wherever it can; planners must
prefer measured values (Plan HW-2).  Throughput numbers are *dense* (no 2:4 sparsity) tensor-core TFLOP/s (TOP/s
for integer types) at the quoted clock, with FP32 accumulation where the card distinguishes (GeForce parts run
FP16 with FP32 accumulate at half the FP16-accumulate rate).  ``mem_gb`` is the nominal (marketing) capacity.
``nvlink_gbs`` is the *total bidirectional* NVLink bandwidth per GPU.

Lookups
-------
``find_chip(name, sm=, num_sms=, mem_gb=)`` -> :class:`ChipMatch` (exact name match, name match adjusted to the
real SM count, SM-architecture fallback, or unknown).  ``KNOWN_CHIPS`` maps a stable key to a spec dict.
"""
from __future__ import annotations

import copy
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("forge.hw.chips")

TENSOR_DTYPES = ("tf32", "fp16", "bf16", "fp8", "int8", "int4", "fp4")

# --------------------------------------------------------------------------------------------
# per-architecture constants.  rates = dense tensor FLOP (or OP) per SM per clock, FP32 accumulate.
# --------------------------------------------------------------------------------------------
_RATE: dict[str, dict[str, Any]] = {
    # name: family, default sm, fp32 lanes/SM, smem total per SM (KiB), usable per block opt-in (KiB), rates
    "volta":            dict(family="volta",     sm=70,  fp32=64,  smem_total=96,  smem_block=96,  rates={"fp16": 1024}),
    "turing_geforce":   dict(family="turing",    sm=75,  fp32=64,  smem_total=64,  smem_block=64,
                             rates={"fp16": 512, "int8": 2048, "int4": 4096}),
    "turing_pro":       dict(family="turing",    sm=75,  fp32=64,  smem_total=64,  smem_block=64,
                             rates={"fp16": 1024, "int8": 2048, "int4": 4096}),
    "ga100":            dict(family="ampere",    sm=80,  fp32=64,  smem_total=164, smem_block=163,
                             rates={"fp16": 2048, "bf16": 2048, "tf32": 1024, "int8": 4096, "int4": 8192}),
    "ga10x_pro":        dict(family="ampere",    sm=86,  fp32=128, smem_total=100, smem_block=99,
                             rates={"fp16": 1024, "bf16": 1024, "tf32": 512, "int8": 2048, "int4": 4096}),
    "ga10x_geforce":    dict(family="ampere",    sm=86,  fp32=128, smem_total=100, smem_block=99,
                             rates={"fp16": 512, "bf16": 512, "tf32": 256, "int8": 2048, "int4": 4096}),
    "ga10b":            dict(family="ampere",    sm=87,  fp32=128, smem_total=164, smem_block=163,
                             rates={"fp16": 2048, "bf16": 2048, "tf32": 1024, "int8": 4096, "int4": 8192}),
    "ada_pro":          dict(family="ada",       sm=89,  fp32=128, smem_total=100, smem_block=99,
                             rates={"fp16": 1024, "bf16": 1024, "tf32": 512, "fp8": 2048, "int8": 2048}),
    "ada_geforce":      dict(family="ada",       sm=89,  fp32=128, smem_total=100, smem_block=99,
                             rates={"fp16": 512, "bf16": 512, "tf32": 256, "fp8": 2048, "int8": 2048}),
    "hopper":           dict(family="hopper",    sm=90,  fp32=128, smem_total=228, smem_block=227,
                             rates={"fp16": 4096, "bf16": 4096, "tf32": 2048, "fp8": 8192, "int8": 8192}),
    "blackwell_dc":     dict(family="blackwell", sm=100, fp32=128, smem_total=228, smem_block=227,
                             rates={"fp16": 8192, "bf16": 8192, "tf32": 4096, "fp8": 16384, "int8": 16384,
                                    "fp4": 32768}),
    "blackwell_geforce": dict(family="blackwell", sm=120, fp32=128, smem_total=100, smem_block=99,
                              rates={"fp16": 512, "bf16": 512, "tf32": 256, "fp8": 2048, "int8": 2048,
                                     "fp4": 4096}),
}

# conservative per-SM-per-clock rate set to use when only the SM level of an unknown part is known
_SM_TO_RATE = {
    70: "volta", 72: "volta", 75: "turing_geforce", 80: "ga100", 86: "ga10x_geforce", 87: "ga10b",
    89: "ada_geforce", 90: "hopper", 100: "blackwell_dc", 101: "blackwell_dc", 103: "blackwell_dc",
    110: "blackwell_dc", 120: "blackwell_geforce", 121: "blackwell_geforce",
}

_FAMILY_BY_SM = {70: "volta", 72: "volta", 75: "turing", 80: "ampere", 86: "ampere", 87: "ampere", 88: "ampere",
                 89: "ada", 90: "hopper", 100: "blackwell", 101: "blackwell", 103: "blackwell", 110: "blackwell",
                 120: "blackwell", 121: "blackwell"}

KNOWN_CHIPS: dict[str, dict[str, Any]] = {}


def _c(key: str, display: str, arch: str, sms: int | None, mhz: float | None, mem_gb: float, mem_type: str,
       bw: float | None, l2: float | None, tdp: float | None, *, pat: list[str], ex: list[str] | None = None,
       sm: int | None = None, nv: float | None = None, pcie: int = 4, form: str = "pcie",
       tf: dict[str, float] | None = None, conf: str = "datasheet", note: str = "", prio: int = 0,
       tensor: bool = True, boost_known: bool = True) -> None:
    """Register one chip. ``tf`` holds explicit (datasheet) dense TFLOP/s; the rest is derived from ``arch``."""
    a = _RATE[arch]
    sm_level = sm if sm is not None else a["sm"]
    tflops: dict[str, float] = {}
    basis: dict[str, str] = {}
    if tensor and sms and mhz:
        for dt, rate in a["rates"].items():
            tflops[dt] = round(sms * rate * mhz / 1e6, 2)
            basis[dt] = "derived"
    for dt, v in (tf or {}).items():
        if v is None:
            tflops.pop(dt, None)
            basis.pop(dt, None)
            continue
        tflops[dt] = float(v)
        basis[dt] = "datasheet" if conf == "datasheet" else "approximate"
    if not tensor:
        tflops = {k: v for k, v in tflops.items() if k == "fp32"}
        basis = {k: basis[k] for k in tflops}
    fp32 = None
    if sms and mhz:
        fp32 = round(sms * a["fp32"] * 2 * mhz / 1e6, 2)
        tflops.setdefault("fp32", fp32)
        basis.setdefault("fp32", "derived")
    if tf and "fp32" in tf and tf["fp32"] is not None:
        basis["fp32"] = "datasheet" if conf == "datasheet" else "approximate"
    tb = {v for k, v in basis.items() if k != "fp32"}
    if conf != "datasheet":
        tconf = "approximate"
    elif tb == {"datasheet"}:
        tconf = "datasheet"
    elif tb == {"derived"}:
        tconf = "derived"
    else:
        tconf = "mixed"
    confidence = {"specs": conf, "tflops": tconf}  # specs: SMs, clock, memory, bandwidth, L2, TDP
    KNOWN_CHIPS[key] = {
        "key": key,
        "display": display,
        "vendor": "nvidia",
        "family": a["family"],
        "rate_set": arch,
        "sm": sm_level,
        "num_sms": sms,
        "boost_mhz": mhz,
        "mem_gb": mem_gb,
        "mem_type": mem_type,
        "mem_bw_gbs": bw,
        "l2_mb": l2,
        "smem_per_sm_kib": a["smem_block"],      # usable per block (opt-in max) - what kernels may allocate
        "smem_total_per_sm_kib": a["smem_total"],  # physical per SM
        "regs_per_sm": 65536,
        "fp32_lanes_per_sm": a["fp32"],
        "tdp_w": tdp,
        "nvlink_gbs": nv,
        "pcie_gen": pcie,
        "form": form,
        "tensor_cores": bool(tensor),
        "two_four": bool(tensor and sm_level >= 80),
        "tflops": tflops,
        "tflops_basis": basis,
        "confidence": confidence,
        "patterns": pat,
        "exclude": ex or [],
        "prio": prio,
        "notes": note,
    }


_GF = ("tflops for GeForce parts are derived (SMs x boost MHz x per-SM rate) with FP32 accumulate (half the "
       "FP16-accumulate rate); real sustained clocks are lower than boost.")
_LAPTOP = r"laptop|mobile|max q|maxq"

# ---- Volta ----------------------------------------------------------------------------------------------------
_c("v100-sxm2-16g", "Tesla V100 SXM2 16GB", "volta", 80, 1530, 16, "HBM2", 900, 6, 300, pat=[r"\bv100\b.*sxm2?"],
   ex=[r"32gb", r"v100s"], nv=300, pcie=3, form="sxm", tf={"fp16": 125.0, "fp32": 15.7})
_c("v100-sxm2-32g", "Tesla V100 SXM2 32GB", "volta", 80, 1530, 32, "HBM2", 900, 6, 300, pat=[r"\bv100\b.*sxm[23]?"],
   ex=[r"16gb", r"v100s"], nv=300, pcie=3, form="sxm", tf={"fp16": 125.0, "fp32": 15.7},
   note="V100 SXM3 (DGX-2, 350 W) shares this entry; same SM count and clocks.")
_c("v100-pcie-16g", "Tesla V100 PCIe 16GB", "volta", 80, 1380, 16, "HBM2", 900, 6, 250, pat=[r"\bv100\b"],
   ex=[r"sxm", r"32gb", r"v100s", r"fhhl"], pcie=3, tf={"fp16": 112.0, "fp32": 14.0})
_c("v100-pcie-32g", "Tesla V100 PCIe 32GB", "volta", 80, 1380, 32, "HBM2", 900, 6, 250, pat=[r"\bv100\b.*32gb"],
   ex=[r"sxm", r"v100s"], pcie=3, tf={"fp16": 112.0, "fp32": 14.0})
_c("v100s-pcie-32g", "Tesla V100S PCIe 32GB", "volta", 80, 1597, 32, "HBM2", 1134, 6, 250, pat=[r"\bv100s\b"],
   pcie=3, tf={"fp16": 130.0, "fp32": 16.4})
_c("titan-v", "NVIDIA Titan V", "volta", 80, 1455, 12, "HBM2", 652.8, 4.5, 250, pat=[r"titan v\b"], pcie=3,
   conf="approximate", note="tensor TFLOPs derived from SMs x boost; NVIDIA quotes ~110 TFLOPs.")

# ---- Turing ---------------------------------------------------------------------------------------------------
_c("t4", "Tesla T4", "turing_pro", 40, 1590, 16, "GDDR6", 320, 4, 70, pat=[r"\bt4\b"], pcie=3,
   tf={"fp16": 65.0, "int8": 130.0, "int4": 260.0, "fp32": 8.1})
_c("titan-rtx", "NVIDIA Titan RTX", "turing_geforce", 72, 1770, 24, "GDDR6", 672, 6, 280, pat=[r"titan rtx"],
   pcie=3, nv=100, note=_GF, conf="approximate")
_c("rtx-2080-ti", "GeForce RTX 2080 Ti", "turing_geforce", 68, 1545, 11, "GDDR6", 616, 5.5, 250,
   pat=[r"rtx 2080 ti\b"], ex=[_LAPTOP], pcie=3, nv=100, note=_GF)
_c("rtx-2080-super", "GeForce RTX 2080 SUPER", "turing_geforce", 48, 1815, 8, "GDDR6", 496, 4, 250,
   pat=[r"rtx 2080 super"], ex=[_LAPTOP], pcie=3, nv=50, note=_GF)
_c("rtx-2080", "GeForce RTX 2080", "turing_geforce", 46, 1710, 8, "GDDR6", 448, 4, 215,
   pat=[r"rtx 2080\b"], ex=[r"ti\b", r"super", _LAPTOP], pcie=3, nv=50, note=_GF)
_c("rtx-2070-super", "GeForce RTX 2070 SUPER", "turing_geforce", 40, 1770, 8, "GDDR6", 448, 4, 215,
   pat=[r"rtx 2070 super"], ex=[_LAPTOP], pcie=3, nv=50, note=_GF)
_c("rtx-2070", "GeForce RTX 2070", "turing_geforce", 36, 1620, 8, "GDDR6", 448, 4, 175,
   pat=[r"rtx 2070\b"], ex=[r"super", _LAPTOP], pcie=3, note=_GF)
_c("rtx-2060-super", "GeForce RTX 2060 SUPER", "turing_geforce", 34, 1650, 8, "GDDR6", 448, 4, 175,
   pat=[r"rtx 2060 super"], ex=[_LAPTOP], pcie=3, note=_GF)
_c("rtx-2060", "GeForce RTX 2060", "turing_geforce", 30, 1680, 6, "GDDR6", 336, 3, 160,
   pat=[r"rtx 2060\b"], ex=[r"super", _LAPTOP], pcie=3, note=_GF + " 12 GB refresh variant has 34 SMs.")
_c("gtx-16xx", "GeForce GTX 16xx (Turing, no tensor cores)", "turing_geforce", None, None, 6, "GDDR5/6", None, None,
   None, pat=[r"gtx 16\d\d"], pcie=3, tensor=False, conf="approximate",
   note="Turing TU116/TU117 have NO tensor cores: only CUDA-core fallbacks apply (tier C).")

# ---- Ampere datacenter ----------------------------------------------------------------------------------------
_A100_TF = {"fp16": 312.0, "bf16": 312.0, "tf32": 156.0, "int8": 624.0, "int4": 1248.0, "fp32": 19.5}
_c("a100-sxm4-40g", "A100 SXM4 40GB", "ga100", 108, 1410, 40, "HBM2", 1555, 40, 400, pat=[r"\ba100\b"],
   ex=[r"80gb", r"pcie"], nv=600, pcie=4, form="sxm", tf=_A100_TF)
_c("a100-sxm4-80g", "A100 SXM4 80GB", "ga100", 108, 1410, 80, "HBM2e", 2039, 40, 400, pat=[r"\ba100\b"],
   ex=[r"pcie", r"40gb"], nv=600, pcie=4, form="sxm", tf=_A100_TF,
   note="Some SXM boards run at 500 W.")
_c("a100-pcie-40g", "A100 PCIe 40GB", "ga100", 108, 1410, 40, "HBM2", 1555, 40, 250, pat=[r"\ba100\b.*pcie"],
   ex=[r"80gb"], nv=600, pcie=4, tf=_A100_TF, note="NVLink via 2-way bridge.")
_c("a100-pcie-80g", "A100 PCIe 80GB", "ga100", 108, 1410, 80, "HBM2e", 1935, 40, 300, pat=[r"\ba100\b.*pcie"],
   ex=[r"40gb"], nv=600, pcie=4, tf=_A100_TF, note="NVLink via 2-way bridge.")
_c("a30", "A30", "ga100", 56, 1440, 24, "HBM2", 933, 24, 165, pat=[r"\ba30\b"], nv=200, pcie=4,
   tf={"fp16": 165.0, "bf16": 165.0, "tf32": 82.0, "int8": 330.0, "int4": 661.0, "fp32": 10.3})
_c("a2", "A2", "ga10x_pro", 10, 1770, 16, "GDDR6", 200, 4, 60, pat=[r"\ba2\b"], pcie=4, conf="approximate",
   note="tflops derived (10 SMs x 1770 MHz); NVIDIA quotes 18 FP16 tensor TFLOPs.")

# ---- Ampere GA10x pro -----------------------------------------------------------------------------------------
_c("a10", "A10", "ga10x_pro", 72, 1695, 24, "GDDR6", 600, 6, 150, pat=[r"\ba10\b"], pcie=4,
   tf={"fp16": 125.0, "bf16": 125.0, "tf32": 62.5, "int8": 250.0, "int4": 500.0, "fp32": 31.2})
_c("a10g", "A10G", "ga10x_pro", 80, 1710, 24, "GDDR6", 600, 6, 300, pat=[r"\ba10g\b"], pcie=4, conf="approximate",
   tf={"fp16": 70.0, "bf16": 70.0, "tf32": 35.0, "int8": 140.0, "int4": 280.0},
   note="NVIDIA does not publish A10G tensor numbers; ~70 FP16 TFLOPs is a third-party figure and uncertain "
        "(a full-rate GA10x pro part would be ~125). SM count 80 is as reported by torch on AWS g5. Trust the probe.")
_c("a40", "A40", "ga10x_pro", 84, 1740, 48, "GDDR6", 696, 6, 300, pat=[r"\ba40\b"], nv=112.5, pcie=4,
   tf={"fp16": 149.7, "bf16": 149.7, "tf32": 74.8, "int8": 299.3, "int4": 598.7, "fp32": 37.4})
_c("rtx-a6000", "RTX A6000", "ga10x_pro", 84, 1800, 48, "GDDR6", 768, 6, 300, pat=[r"rtx a6000"], nv=112.5, pcie=4,
   tf={"fp16": 154.8, "bf16": 154.8, "tf32": 77.4, "int8": 309.7, "int4": 619.4, "fp32": 38.7})
_c("rtx-a5000", "RTX A5000", "ga10x_pro", 64, 1695, 24, "GDDR6", 768, 6, 230, pat=[r"rtx a5000"], nv=112.5,
   pcie=4, conf="approximate", note="tflops derived from SMs x boost.")
_c("rtx-a4000", "RTX A4000", "ga10x_pro", 48, 1560, 16, "GDDR6", 448, 4, 140, pat=[r"rtx a4000"], pcie=4,
   conf="approximate", note="tflops derived from SMs x boost.")

# ---- Ampere GA10x GeForce -------------------------------------------------------------------------------------
_c("rtx-3090-ti", "GeForce RTX 3090 Ti", "ga10x_geforce", 84, 1860, 24, "GDDR6X", 1008, 6, 450,
   pat=[r"rtx 3090 ti"], ex=[_LAPTOP], nv=112.5, note=_GF)
_c("rtx-3090", "GeForce RTX 3090", "ga10x_geforce", 82, 1695, 24, "GDDR6X", 936.2, 6, 350, pat=[r"rtx 3090\b"],
   ex=[r"\bti\b", _LAPTOP], nv=112.5, note=_GF)
_c("rtx-3080-ti", "GeForce RTX 3080 Ti", "ga10x_geforce", 80, 1665, 12, "GDDR6X", 912.4, 6, 350,
   pat=[r"rtx 3080 ti"], ex=[_LAPTOP], note=_GF)
_c("rtx-3080-12g", "GeForce RTX 3080 12GB", "ga10x_geforce", 70, 1710, 12, "GDDR6X", 912.4, 6, 350,
   pat=[r"rtx 3080\b"], ex=[r"\bti\b", _LAPTOP], note=_GF)
_c("rtx-3080-10g", "GeForce RTX 3080 10GB", "ga10x_geforce", 68, 1710, 10, "GDDR6X", 760, 5, 320,
   pat=[r"rtx 3080\b"], ex=[r"\bti\b", _LAPTOP], note=_GF)
_c("rtx-3070-ti", "GeForce RTX 3070 Ti", "ga10x_geforce", 48, 1770, 8, "GDDR6X", 608, 4, 290,
   pat=[r"rtx 3070 ti"], ex=[_LAPTOP], note=_GF)
_c("rtx-3070", "GeForce RTX 3070", "ga10x_geforce", 46, 1725, 8, "GDDR6", 448, 4, 220, pat=[r"rtx 3070\b"],
   ex=[r"\bti\b", _LAPTOP], note=_GF)
_c("rtx-3060-ti", "GeForce RTX 3060 Ti", "ga10x_geforce", 38, 1665, 8, "GDDR6", 448, 4, 200,
   pat=[r"rtx 3060 ti"], ex=[_LAPTOP], note=_GF)
_c("rtx-3060", "GeForce RTX 3060", "ga10x_geforce", 28, 1777, 12, "GDDR6", 360, 3, 170, pat=[r"rtx 3060\b"],
   ex=[r"\bti\b", _LAPTOP], conf="approximate", note=_GF + " The 8 GB variant has 240 GB/s (different bandwidth).")

# ---- Jetson Orin (ga10b, sm_87) -------------------------------------------------------------------------------
_ORIN_NOTE = ("Jetson Orin: LPDDR5 shared with the CPU; clocks depend on the nvpmodel power mode; tflops derived "
              "from SMs x GPU max clock x GA10B per-SM rate (275 TOPS marketing figure includes DLAs).")
_c("jetson-agx-orin-64g", "Jetson AGX Orin 64GB", "ga10b", 16, 1301, 64, "LPDDR5", 204.8, 4, 60,
   pat=[r"\borin\b"], pcie=4, form="module", conf="approximate", note=_ORIN_NOTE)
_c("jetson-agx-orin-32g", "Jetson AGX Orin 32GB", "ga10b", 14, 939, 32, "LPDDR5", 204.8, 4, 40,
   pat=[r"\borin\b"], pcie=4, form="module", conf="approximate", note=_ORIN_NOTE)
_c("jetson-orin-nx-16g", "Jetson Orin NX 16GB", "ga10b", 8, 918, 16, "LPDDR5", 102.4, 4, 25,
   pat=[r"\borin\b"], pcie=4, form="module", conf="approximate", note=_ORIN_NOTE)
_c("jetson-orin-nx-8g", "Jetson Orin NX 8GB", "ga10b", 8, 765, 8, "LPDDR5", 102.4, 4, 20,
   pat=[r"\borin\b"], pcie=4, form="module", conf="approximate", note=_ORIN_NOTE)
_c("jetson-orin-nano-8g", "Jetson Orin Nano 8GB", "ga10b", 8, 625, 8, "LPDDR5", 68, 4, 15,
   pat=[r"\borin\b"], pcie=4, form="module", conf="approximate", note=_ORIN_NOTE)
_c("jetson-orin-nano-4g", "Jetson Orin Nano 4GB", "ga10b", 4, 625, 4, "LPDDR5", 34, 2, 10,
   pat=[r"\borin\b"], pcie=4, form="module", conf="approximate", note=_ORIN_NOTE)

# ---- Ada ------------------------------------------------------------------------------------------------------
_c("rtx-4090", "GeForce RTX 4090", "ada_geforce", 128, 2520, 24, "GDDR6X", 1008, 72, 450,
   pat=[r"rtx 4090\b"], ex=[r"4090 ?d", _LAPTOP], form="desktop", note=_GF,
   tf={"fp16": 165.2, "bf16": 165.2, "fp8": 660.6, "int8": 660.6, "tf32": 82.6})
_c("rtx-4090d", "GeForce RTX 4090 D", "ada_geforce", 114, 2520, 24, "GDDR6X", 1008, 72, 425,
   pat=[r"rtx 4090 ?d"], ex=[_LAPTOP], note=_GF, conf="approximate")
_c("rtx-4080-super", "GeForce RTX 4080 SUPER", "ada_geforce", 80, 2550, 16, "GDDR6X", 736, 64, 320,
   pat=[r"rtx 4080 super"], ex=[_LAPTOP], note=_GF)
_c("rtx-4080", "GeForce RTX 4080", "ada_geforce", 76, 2505, 16, "GDDR6X", 716.8, 64, 320,
   pat=[r"rtx 4080\b"], ex=[r"super", _LAPTOP], note=_GF)
_c("rtx-4070-ti-super", "GeForce RTX 4070 Ti SUPER", "ada_geforce", 66, 2610, 16, "GDDR6X", 672, 48, 285,
   pat=[r"rtx 4070 ti super"], ex=[_LAPTOP], note=_GF)
_c("rtx-4070-ti", "GeForce RTX 4070 Ti", "ada_geforce", 60, 2610, 12, "GDDR6X", 504, 48, 285,
   pat=[r"rtx 4070 ti\b"], ex=[r"super", _LAPTOP], note=_GF)
_c("rtx-4070-super", "GeForce RTX 4070 SUPER", "ada_geforce", 56, 2475, 12, "GDDR6X", 504, 48, 220,
   pat=[r"rtx 4070 super"], ex=[_LAPTOP], note=_GF)
_c("rtx-4070", "GeForce RTX 4070", "ada_geforce", 46, 2475, 12, "GDDR6X", 504, 36, 200,
   pat=[r"rtx 4070\b"], ex=[r"\bti\b", r"super", _LAPTOP], note=_GF)
_c("rtx-4060-ti", "GeForce RTX 4060 Ti", "ada_geforce", 34, 2535, 16, "GDDR6", 288, 32, 160,
   pat=[r"rtx 4060 ti"], ex=[_LAPTOP], note=_GF + " 8 GB and 16 GB variants share the SM count.")
_c("rtx-4060", "GeForce RTX 4060", "ada_geforce", 24, 2460, 8, "GDDR6", 272, 24, 115,
   pat=[r"rtx 4060\b"], ex=[r"\bti\b", _LAPTOP], note=_GF)
_c("l4", "L4", "ada_pro", 58, 2040, 24, "GDDR6", 300, 48, 72, pat=[r"\bl4\b"], pcie=4,
   tf={"fp16": 121.0, "bf16": 121.0, "tf32": 60.0, "fp8": 242.0, "int8": 242.0, "fp32": 30.3})
_c("l40", "L40", "ada_pro", 142, 2490, 48, "GDDR6", 864, 96, 300, pat=[r"\bl40\b"], pcie=4,
   tf={"fp16": 181.0, "bf16": 181.0, "tf32": 90.5, "fp8": 362.0, "int8": 362.0, "fp32": 90.5})
_c("l40s", "L40S", "ada_pro", 142, 2520, 48, "GDDR6", 864, 96, 350, pat=[r"\bl40s\b"], pcie=4,
   tf={"fp16": 362.0, "bf16": 362.0, "tf32": 183.0, "fp8": 733.0, "int8": 733.0, "fp32": 91.6})
_c("rtx-6000-ada", "RTX 6000 Ada Generation", "ada_pro", 142, 2505, 48, "GDDR6", 960, 96, 300,
   pat=[r"rtx 6000 ada"], pcie=4, conf="approximate",
   note="tflops derived from SMs x boost (matches the 1457 AI-TOPS FP8-sparse marketing figure within rounding).")

# ---- Hopper ---------------------------------------------------------------------------------------------------
_H100_SXM_TF = {"fp16": 989.0, "bf16": 989.0, "tf32": 494.0, "fp8": 1979.0, "int8": 1979.0, "fp32": 67.0}
_c("h100-sxm5-80g", "H100 SXM5 80GB", "hopper", 132, 1980, 80, "HBM3", 3350, 50, 700,
   pat=[r"\bh100\b"], ex=[r"pcie", r"nvl", r"gh200"], nv=900, pcie=5, form="sxm", tf=_H100_SXM_TF,
   note="Tensor rates are quoted at ~1830 MHz; max boost is 1980 MHz.")
_c("h100-pcie-80g", "H100 PCIe 80GB", "hopper", 114, 1755, 80, "HBM2e", 2000, 50, 350, pat=[r"\bh100\b.*pcie"],
   nv=600, pcie=5, tf={"fp16": 756.0, "bf16": 756.0, "tf32": 378.0, "fp8": 1513.0, "int8": 1513.0, "fp32": 51.0},
   note="NVLink via bridge.")
_c("h100-nvl-94g", "H100 NVL 94GB", "hopper", 132, 1785, 94, "HBM3", 3938, 50, 400, pat=[r"\bh100\b.*nvl"],
   nv=600, pcie=5, tf={"fp16": 1671.0, "bf16": 1671.0, "tf32": 835.0, "fp8": 3341.0, "int8": 3341.0, "fp32": 60.0},
   conf="approximate", note="SM count and clock for the NVL SKU are not firmly documented; bandwidth/TFLOPs from "
   "the H100 NVL datasheet (per GPU of the pair).")
_c("h200-sxm-141g", "H200 SXM 141GB", "hopper", 132, 1980, 141, "HBM3e", 4800, 50, 700, pat=[r"\bh200\b"],
   ex=[r"nvl"], nv=900, pcie=5, form="sxm", tf=_H100_SXM_TF, note="Same compute as H100 SXM; more/faster HBM.")
_c("h200-nvl-141g", "H200 NVL 141GB", "hopper", 132, 1785, 141, "HBM3e", 4800, 50, 600, pat=[r"\bh200\b.*nvl"],
   nv=900, pcie=5, tf={"fp16": 1671.0, "bf16": 1671.0, "tf32": 835.0, "fp8": 3341.0, "int8": 3341.0},
   conf="approximate", note="Clock and exact TFLOPs of the NVL SKU are approximate.")
_c("gh200-96g", "GH200 Grace Hopper (96GB HBM3)", "hopper", 132, 1980, 96, "HBM3", 4000, 50, 900,
   pat=[r"\bgh200\b"], nv=900, pcie=5, form="module", tf=_H100_SXM_TF, conf="approximate",
   note="Module power is configurable (450-1000 W incl. CPU). Also exists with 144 GB HBM3e (~4900 GB/s) - see "
        "gh200-144g; NVLink-C2C to the Grace CPU is 900 GB/s.")
_c("gh200-144g", "GH200 Grace Hopper (144GB HBM3e)", "hopper", 132, 1980, 144, "HBM3e", 4900, 50, 900,
   pat=[r"\bgh200\b"], nv=900, pcie=5, form="module", tf=_H100_SXM_TF, conf="approximate",
   note="See gh200-96g.")

# ---- Blackwell datacenter -------------------------------------------------------------------------------------
_c("b200", "B200", "blackwell_dc", 148, None, 192, "HBM3e", 8000, 126, 1000, pat=[r"\bb200\b"], ex=[r"gb200"],
   nv=1800, pcie=5, form="sxm", conf="approximate",
   tf={"fp16": 2250.0, "bf16": 2250.0, "tf32": 1100.0, "fp8": 4500.0, "int8": 4500.0, "fp4": 9000.0},
   note="HGX B200 datasheet figures (dense = half of the quoted sparse numbers). L2 size, SM count and clock are "
        "approximate; 180-192 GB usable depending on SKU.")
_c("gb200", "GB200 (Blackwell GPU in Grace Blackwell)", "blackwell_dc", 148, None, 186, "HBM3e", 8000, 126, 1200,
   pat=[r"\bgb200\b"], nv=1800, pcie=5, form="module", conf="approximate",
   tf={"fp16": 2500.0, "bf16": 2500.0, "tf32": 1250.0, "fp8": 5000.0, "int8": 5000.0, "fp4": 10000.0},
   note="Per-GPU figures derived from the GB200 NVL72 rack numbers divided by 72 (dense = half of sparse).")
_c("b300", "B300 (Blackwell Ultra)", "blackwell_dc", 160, None, 288, "HBM3e", 8000, 126, 1100,
   pat=[r"\bb300\b", r"\bgb300\b"], sm=103, nv=1800, pcie=6, form="sxm", conf="approximate",
   tf={"fp16": 2250.0, "bf16": 2250.0, "tf32": None, "fp8": 4500.0, "int8": None, "fp4": 15000.0},
   note="Blackwell Ultra: FP4 dense ~15 PFLOPs per datasheet figures; TF32/INT8 reportedly cut and left unspecified; "
        "SM count and TDP approximate.")

# ---- Blackwell consumer / workstation / edge (sm_120 / sm_121) -----------------------------------------------
_BW_GF = ("tflops derived from the marketing 'AI TOPS' (FP4 with sparsity) and SMs x boost x per-SM rate; GeForce "
          "FP32-accumulate rates may be half of FP8/FP4 figures - confirm with the probe (sm_120 FP4 is a hypothesis "
          "until then).")
_c("rtx-5090", "GeForce RTX 5090", "blackwell_geforce", 170, 2407, 32, "GDDR7", 1792, 96, 575,
   pat=[r"rtx 5090\b"], ex=[_LAPTOP, r"5090 ?d"], pcie=5, form="desktop", conf="approximate", note=_BW_GF)
_c("rtx-5080", "GeForce RTX 5080", "blackwell_geforce", 84, 2617, 16, "GDDR7", 960, 64, 360,
   pat=[r"rtx 5080\b"], ex=[_LAPTOP], pcie=5, form="desktop", conf="approximate", note=_BW_GF)
_c("rtx-5070-ti", "GeForce RTX 5070 Ti", "blackwell_geforce", 70, 2452, 16, "GDDR7", 896, 48, 300,
   pat=[r"rtx 5070 ti"], ex=[_LAPTOP], pcie=5, form="desktop", conf="approximate", note=_BW_GF)
_c("rtx-5070", "GeForce RTX 5070", "blackwell_geforce", 48, 2512, 12, "GDDR7", 672, 48, 250,
   pat=[r"rtx 5070\b"], ex=[r"\bti\b", _LAPTOP], pcie=5, form="desktop", conf="approximate", note=_BW_GF)
_c("rtx-pro-6000-blackwell", "RTX PRO 6000 Blackwell (Workstation/Server)", "blackwell_geforce", 188, 2617, 96,
   "GDDR7", 1792, 128, 600, pat=[r"rtx pro 6000"], ex=[r"max q|maxq|max-q"], pcie=5, conf="approximate",
   note="Consumer-class Blackwell (sm_120) workstation part; " + _BW_GF)
_c("rtx-pro-6000-blackwell-maxq", "RTX PRO 6000 Blackwell Max-Q", "blackwell_geforce", 188, 2300, 96, "GDDR7",
   1792, 128, 300, pat=[r"rtx pro 6000.*(max q|maxq|max-q)"], pcie=5, conf="approximate",
   note="Clock is a guess (power-limited to 300 W); " + _BW_GF)
_c("gb10", "GB10 (DGX Spark)", "blackwell_geforce", 48, None, 128, "LPDDR5X", 273, 24, 140, pat=[r"\bgb10\b"],
   sm=121, pcie=5, form="module", conf="approximate",
   tf={"fp4": 500.0, "fp8": 250.0, "fp16": 125.0, "bf16": 125.0, "tf32": None, "int8": None},
   note="128 GB unified LPDDR5X shared with the Grace CPU; '1 PFLOP FP4' marketing figure is with sparsity, so dense "
        "FP4 ~500; lower-precision rows derived from it. SM count/L2/clocks approximate.")

# register in the generic registry (no-op if core.registry is unavailable)
try:  # pragma: no cover - trivial
    from forge.core.registry import register_obj as _register_obj

    for _k, _v in KNOWN_CHIPS.items():
        _register_obj("chip", _k, _v, meta={"family": _v["family"], "sm": _v["sm"], "display": _v["display"]})
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------------------------------------
# lookups
# --------------------------------------------------------------------------------------------
_STOP_TOKENS = re.compile(r"\b(nvidia|geforce|tesla|quadro|corporation|graphics|generation|edition|workstation|server)\b")


def normalize_name(name: str) -> str:
    """Lower-case device name with vendor words/punctuation removed: ``'NVIDIA A100-SXM4-80GB' -> 'a100 sxm4 80gb'``."""
    s = (name or "").lower()
    s = s.replace("(r)", " ").replace("(tm)", " ").replace("®", " ").replace("™", " ")
    s = re.sub(r"[^a-z0-9.+ ]+", " ", s)
    s = _STOP_TOKENS.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class ChipMatch:
    """Result of :func:`find_chip`."""

    kind: str  # exact | name_adjusted | sm_fallback | unknown
    spec: dict[str, Any]  # always a private deep copy (safe to mutate)
    key: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.kind in ("exact", "name_adjusted")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "key": self.key, "notes": list(self.notes),
                "display": self.spec.get("display"), "sm": self.spec.get("sm")}


def _mem_score(nominal_gb: float | None, mem_gb: float | None) -> float:
    if not nominal_gb or not mem_gb:
        return 0.0
    # torch reports binary GiB slightly below the nominal capacity (ECC / reserved)
    r = mem_gb / nominal_gb
    err = abs(math.log(max(r, 1e-6) / 0.985))
    return max(0.0, 3.0 - 6.0 * err)


def _candidates(norm: str, sm: int | None, use_exclude: bool = True) -> list[dict[str, Any]]:
    out = []
    for spec in KNOWN_CHIPS.values():
        if sm is not None and spec["sm"] != sm:
            # allow the sm_103 / sm_100 ambiguity for datacenter Blackwell only
            if not (spec["sm"] in (100, 103) and sm in (100, 103)):
                continue
        if use_exclude and any(re.search(e, norm) for e in spec["exclude"]):
            continue
        if any(re.search(p, norm) for p in spec["patterns"]):
            out.append(spec)
    return out


def find_chip(name: str, *, sm: int | None = None, num_sms: int | None = None, mem_gb: float | None = None,
              clock_mhz: float | None = None) -> ChipMatch:
    """Match a device name (as reported by torch / nvidia-smi) to a datasheet spec.

    ``sm`` (e.g. 90), ``num_sms`` and ``mem_gb`` (GiB as reported by the driver) disambiguate SKUs sharing a name
    (A100 40/80, Orin variants ...).  When the name is unknown but ``sm`` is known an architecture-level generic
    spec is returned (``kind='sm_fallback'``); otherwise ``kind='unknown'`` with an empty-ish spec.
    The returned spec is a deep copy whose ``num_sms`` / ``mem_gb`` are overridden by the actual values and whose
    derived TFLOPs are rescaled when the real SM count differs from the catalogue entry.
    """
    norm = normalize_name(name)
    cands = _candidates(norm, sm)
    if cands:

        def score(s: dict[str, Any]) -> tuple[float, ...]:
            sc = 0.0
            if num_sms and s["num_sms"]:
                sc += 4.0 if s["num_sms"] == num_sms else -min(3.0, abs(s["num_sms"] - num_sms) / max(num_sms, 1) * 6)
            sc += _mem_score(s["mem_gb"], mem_gb)
            return (s["prio"], sc)

        best = max(cands, key=score)
        if mem_gb and best["mem_gb"] and not 0.8 <= mem_gb / best["mem_gb"] <= 1.25:
            # the name carried no capacity hint (e.g. plain 'A100'): retry ignoring the exclusion patterns
            wider = _candidates(norm, sm, use_exclude=False)
            alt = max(wider, key=score) if wider else best
            if score(alt) > score(best):
                cands, best = wider, alt
        spec = copy.deepcopy(best)
        notes: list[str] = []
        kind = "exact"
        if len(cands) > 1:
            notes.append("name matches several SKUs: " + ", ".join(c["key"] for c in cands) + f"; picked {best['key']}")
        if num_sms and spec["num_sms"] and num_sms != spec["num_sms"]:
            kind = "name_adjusted"
            ratio = num_sms / spec["num_sms"]
            notes.append(f"driver reports {num_sms} SMs but catalogue entry has {spec['num_sms']}: "
                         f"tflops/fp32 rescaled by {ratio:.3f} and marked approximate")
            for k in list(spec["tflops"]):
                spec["tflops"][k] = round(spec["tflops"][k] * ratio, 2)
                spec["tflops_basis"][k] = "approximate"
            spec["num_sms"] = num_sms
            spec["confidence"]["specs"] = "approximate"
            spec["confidence"]["tflops"] = "approximate"
        elif num_sms and not spec["num_sms"]:
            spec["num_sms"] = num_sms
        if clock_mhz and not spec["boost_mhz"]:
            spec["boost_mhz"] = float(clock_mhz)
            notes.append("clock taken from the driver (catalogue has none)")
        if spec.get("notes"):
            notes.append(spec["notes"])
        return ChipMatch(kind=kind, spec=spec, key=best["key"], notes=notes)
    if sm is not None:
        return ChipMatch(kind="sm_fallback", spec=generic_spec_for_sm(sm, num_sms=num_sms, clock_mhz=clock_mhz,
                                                                      mem_gb=mem_gb, name=name),
                         key=None, notes=[f"unknown device name {name!r}: using generic sm_{sm} architecture rates "
                                          "(approximate, conservative)"])
    return ChipMatch(kind="unknown", spec=generic_spec_for_sm(None, num_sms=num_sms, clock_mhz=clock_mhz,
                                                              mem_gb=mem_gb, name=name),
                     key=None, notes=[f"unknown device {name!r} and no SM level"])


def generic_spec_for_sm(sm: int | None, *, num_sms: int | None = None, clock_mhz: float | None = None,
                        mem_gb: float | None = None, name: str = "") -> dict[str, Any]:
    """Architecture-level fallback spec.  Tensor TFLOPs are only filled when ``num_sms`` and ``clock_mhz`` are
    known; GeForce-class (half-rate FP32 accumulate) rates are used for the ambiguous sm_86/89/120 levels so that
    the estimate is conservative.  Everything is flagged ``approximate``."""
    rs = _SM_TO_RATE.get(sm) if sm is not None else None
    a = _RATE.get(rs) if rs else None
    tflops: dict[str, float] = {}
    basis: dict[str, str] = {}
    if a and num_sms and clock_mhz:
        for dt, rate in a["rates"].items():
            tflops[dt] = round(num_sms * rate * clock_mhz / 1e6, 2)
            basis[dt] = "derived"
        tflops["fp32"] = round(num_sms * a["fp32"] * 2 * clock_mhz / 1e6, 2)
        basis["fp32"] = "derived"
    return {
        "key": None,
        "display": name or f"generic sm_{sm}",
        "vendor": "nvidia",
        "family": _FAMILY_BY_SM.get(sm, "unknown") if sm is not None else "unknown",
        "rate_set": rs,
        "sm": sm,
        "num_sms": num_sms,
        "boost_mhz": clock_mhz,
        "mem_gb": mem_gb,
        "mem_type": None,
        "mem_bw_gbs": None,
        "l2_mb": None,
        "smem_per_sm_kib": a["smem_block"] if a else None,
        "smem_total_per_sm_kib": a["smem_total"] if a else None,
        "regs_per_sm": 65536,
        "fp32_lanes_per_sm": a["fp32"] if a else None,
        "tdp_w": None,
        "nvlink_gbs": None,
        "pcie_gen": None,
        "form": "unknown",
        "tensor_cores": bool(a),
        "two_four": bool(sm is not None and sm >= 80),
        "tflops": tflops,
        "tflops_basis": basis,
        "confidence": {"specs": "approximate", "tflops": "approximate"},
        "patterns": [],
        "exclude": [],
        "prio": 0,
        "notes": "generic architecture fallback",
    }


def get_chip(key: str) -> dict[str, Any]:
    """Deep copy of a catalogue entry by key. Raises ``KeyError``."""
    return copy.deepcopy(KNOWN_CHIPS[key])


def list_chips(sm: int | None = None, family: str | None = None) -> list[dict[str, Any]]:
    """Catalogue entries (deep copies), optionally filtered, ordered by sm then key."""
    rows = [copy.deepcopy(v) for v in KNOWN_CHIPS.values()
            if (sm is None or v["sm"] == sm) and (family is None or v["family"] == family)]
    return sorted(rows, key=lambda r: (r["sm"] or 0, r["key"]))


def chip_summary() -> list[dict[str, Any]]:
    """Compact rows for the UI (``GET /hardware/chips``)."""
    rows = []
    for s in list_chips():
        rows.append({
            "key": s["key"], "name": s["display"], "family": s["family"], "sm": s["sm"], "num_sms": s["num_sms"],
            "mem_gb": s["mem_gb"], "mem_type": s["mem_type"], "mem_bw_gbs": s["mem_bw_gbs"], "l2_mb": s["l2_mb"],
            "tdp_w": s["tdp_w"], "nvlink_gbs": s["nvlink_gbs"], "boost_mhz": s["boost_mhz"],
            "tflops": dict(s["tflops"]), "tflops_basis": dict(s["tflops_basis"]), "confidence": dict(s["confidence"]),
            "notes": s["notes"], "provenance": "vendor",
        })
    return rows


def resolve_key_or_name(key_or_name: str) -> ChipMatch:
    """Accept either a catalogue key (``'h100-sxm5-80g'``) or a free-form device name."""
    if key_or_name in KNOWN_CHIPS:
        s = copy.deepcopy(KNOWN_CHIPS[key_or_name])
        return ChipMatch(kind="exact", spec=s, key=key_or_name, notes=[s["notes"]] if s["notes"] else [])
    # allow case/underscore variants of keys
    k = key_or_name.strip().lower().replace("_", "-").replace(" ", "-")
    if k in KNOWN_CHIPS:
        return resolve_key_or_name(k)
    return find_chip(key_or_name)
