"""CUDA stack detection and compatibility rules (driver <-> torch build <-> GPU architecture <-> Triton <-> NVML bindings).

Why this exists: ``torch.cuda.is_available()`` only says the driver loaded.  A torch built for CUDA 13.4 on a host whose driver
speaks CUDA 13.0, a CUDA 12 torch on a Blackwell card, or a GPU that is busy/exclusive/not passed through all fail later, deep
inside a job, with an opaque ``AcceleratorError``.  This module looks at the whole stack up front, applies the rules below and
returns findings that say what is wrong and what to do about it.

Layers (everything is stdlib-only and import-cheap; torch is imported lazily and CUDA is never initialised in the calling
process - device usability is probed in a short-lived subprocess so a server process does not grow a CUDA context):

* :func:`detect_stack` - gather facts (JSON-serialisable dict).  Never raises.
* :func:`evaluate`     - pure function ``stack -> [finding]``; this is where the rules live (unit-tested with synthetic stacks).
* :func:`recommend`    - which torch wheel index / companion requirements fit this host.
* :func:`check`        - detect + evaluate + recommend; :func:`format_report` renders it; :func:`assert_cuda_usable` is the gate
  job workers call before touching a GPU.

Rule sources.  ``driver_cuda`` (the newest CUDA the *driver* supports) comes from NVML / the ``nvidia-smi`` header, so the
main torch-vs-driver rule needs no table.  The tables below are only used for GPU-architecture floors and for the "which driver
do I need" wording.  NVIDIA's release notes name the *driver branch* each CUDA 13.x ships with (13.0 R580, 13.1 R590, 13.2 R595,
13.3 R610, 13.4 R615) but no patch-level minimum, so those entries hold the branch (``source == "branch"``).  Any 13.x toolkit runs on
any driver >= 580 through minor-version compatibility (NVIDIA's own example is a CUDA 13.4 application on a 580.65.06 driver); the
branch is what gives the full feature set, not what makes it start.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

Version = tuple[int, ...]

LEVELS = ("ok", "info", "warn", "fail")

#: Minimum Linux driver for any toolkit of a CUDA major family (CUDA minor-version compatibility floor).
DRIVER_FLOOR: dict[int, Version] = {11: (450, 80, 2), 12: (525, 60, 13), 13: (580, 65, 6)}

#: Driver that ships with (natively supports) a CUDA release.  ``spec`` = NVIDIA release-note minimum (GA); ``branch`` = only the
#: driver branch is documented (a one-element tuple is "that branch or newer").
NATIVE_DRIVER: dict[Version, tuple[Version, str]] = {
    (12, 0): ((525, 60, 13), "spec"), (12, 1): ((530, 30, 2), "spec"), (12, 2): ((535, 54, 3), "spec"),
    (12, 3): ((545, 23, 6), "spec"), (12, 4): ((550, 54, 14), "spec"), (12, 5): ((555, 42, 2), "spec"),
    (12, 6): ((560, 28, 3), "spec"), (12, 8): ((570, 26, 0), "spec"), (12, 9): ((575, 51, 3), "spec"),
    (13, 0): ((580, 65, 6), "branch"), (13, 1): ((590,), "branch"), (13, 2): ((595,), "branch"),
    (13, 3): ((610,), "branch"), (13, 4): ((615,), "branch"),
}

#: Oldest CUDA toolkit that can target an SM architecture (kept to architectures Forge's kernels care about).
ARCH_MIN_CUDA: dict[int, Version] = {
    75: (10, 0), 80: (11, 0), 86: (11, 1), 87: (11, 4), 89: (11, 8), 90: (11, 8),
    100: (12, 8), 101: (12, 8), 103: (12, 9), 110: (13, 0), 120: (12, 8), 121: (12, 9),
}

#: CUDA 13 dropped offline compilation for Maxwell/Pascal/Volta (sm_50..sm_72): a CUDA 13 torch cannot run on a V100 or older.
CUDA13_MIN_SM = 75

#: First Triton release with Blackwell (sm_100/sm_120) code generation.
TRITON_BLACKWELL_MIN: Version = (3, 3)

#: The failure that makes every later CUDA call fail identically (reported in the field as cudaErrorDevicesUnavailable).
_BUSY = "busy or unavailable"


# --------------------------------------------------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------------------------------------------------
def parse_version(text: Any) -> Version | None:
    """``'13.4'`` -> ``(13, 4)``, ``'580.178.04'`` -> ``(580, 178, 4)``; ``None`` for anything without a leading number."""
    m = re.match(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(text or ""))
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


def fmt_version(v: Version | None) -> str:
    return ".".join(str(x) for x in v) if v else "?"


def fmt_driver(v: Version | None) -> str:
    """NVIDIA driver strings are ``major.minor.patch`` with two-digit tails: ``(580, 65, 6)`` -> ``580.65.06``."""
    if not v:
        return "?"
    if len(v) == 1:
        return f"R{v[0]}"  # a bare driver branch
    return ".".join([str(v[0]), *(f"{x:02d}" for x in v[1:])])


def cuda_int_to_version(n: int) -> Version:
    """NVML encodes CUDA versions as ``major*1000 + minor*10`` (13000 -> 13.0, 12080 -> 12.8)."""
    return (int(n) // 1000, (int(n) % 1000) // 10)


def sm_from_compute_cap(cc: str) -> int | None:
    """``'12.0'`` -> 120."""
    m = re.match(r"\s*(\d+)\.(\d+)", str(cc or ""))
    return int(m.group(1)) * 10 + int(m.group(2)) if m else None


def native_driver_for(cuda: Version) -> tuple[Version, str, bool]:
    """Driver that natively supports ``cuda``: ``(driver, 'spec'|'branch', exact)``; ``exact`` is False when the nearest older
    known release is used (the true requirement is then at least that high)."""
    key = tuple(cuda[:2])
    if key in NATIVE_DRIVER:
        d, src = NATIVE_DRIVER[key]
        return d, src, True
    older = [k for k in NATIVE_DRIVER if k <= key]
    if older:
        k = max(older)
        d, src = NATIVE_DRIVER[k]
        return d, src, False
    return DRIVER_FLOOR.get(cuda[0], (0,)), "spec", False


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - missing binary, timeout, permission
        return None


# --------------------------------------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------------------------------------
def _pkg(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def _nvml_facts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        try:
            v = pynvml.nvmlSystemGetDriverVersion()
            out["driver_version"] = v.decode() if isinstance(v, bytes) else str(v)
            fn = getattr(pynvml, "nvmlSystemGetCudaDriverVersion_v2", None) or getattr(pynvml, "nvmlSystemGetCudaDriverVersion", None)
            if fn is not None:
                out["driver_cuda"] = fmt_version(cuda_int_to_version(fn()))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - no pynvml, no driver, no permission
        pass
    return out


def _smi_facts() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    out: dict[str, Any] = {"present": bool(exe), "devices": []}
    if not exe:
        return out
    text = _run([exe, "--query-gpu=index,name,driver_version,compute_cap,compute_mode,memory.total", "--format=csv,noheader,nounits"])
    if text is None:  # older drivers lack compute_cap
        text = _run([exe, "--query-gpu=index,name,driver_version,compute_mode,memory.total", "--format=csv,noheader,nounits"])
        has_cc = False
    else:
        has_cc = True
    out["ok"] = text is not None
    for line in (text or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            if has_cc:
                idx, name, drv, cc, mode, mem = p[:6]
                sm = sm_from_compute_cap(cc)
            else:
                idx, name, drv, mode, mem = p[:5]
                sm = None
            out["devices"].append({"index": int(idx), "name": name, "driver": drv, "sm": sm, "compute_mode": mode,
                                   "mem_mib": int(float(mem)) if mem.replace(".", "", 1).isdigit() else None})
        except Exception:  # noqa: BLE001 - malformed row
            continue
    header = _run([exe])
    m = re.search(r"CUDA Version:\s*([\d.]+)", header or "")
    if m:
        out["driver_cuda"] = m.group(1)
    return out


def _nvcc_version() -> str | None:
    for exe in (shutil.which("nvcc"), os.path.join(os.environ.get("CUDA_HOME") or "/usr/local/cuda", "bin", "nvcc")):
        if exe and os.path.exists(exe):
            m = re.search(r"release\s+([\d.]+)", _run([exe, "--version"]) or "")
            if m:
                return m.group(1)
    return None


def _torch_facts() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False}
    try:
        torch = importlib.import_module("torch")
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    ver = str(getattr(torch, "__version__", ""))
    info.update(installed=True, version=ver, cuda=getattr(torch.version, "cuda", None), hip=getattr(torch.version, "hip", None),
                # NVIDIA NGC/PyTorch container builds ("2.14.0a0+...nv26.08"): pinned to the image, never pip-replaceable
                container_build=bool(re.search(r"\.nv\d", ver)))
    try:
        info["arch_list"] = list(torch.cuda.get_arch_list())  # reads the build, does not initialise CUDA
    except Exception:  # noqa: BLE001
        info["arch_list"] = []
    return info


_PROBE_SNIPPET = r"""
import json, sys
out = []
try:
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(n):
        try:
            x = torch.ones(1024, device=f"cuda:{i}")
            torch.cuda.synchronize(i)
            out.append({"torch_index": i, "ok": True, "name": torch.cuda.get_device_name(i), "sum": float(x.sum())})
        except Exception as e:
            out.append({"torch_index": i, "ok": False, "error": (str(e).splitlines() or [repr(e)])[0]})
    print(json.dumps({"available": bool(torch.cuda.is_available()), "device_count": n, "devices": out}))
except Exception as e:
    print(json.dumps({"available": False, "device_count": 0, "devices": [], "error": f"{type(e).__name__}: {e}"}))
"""


def probe_devices(timeout: float = 120.0) -> dict[str, Any]:
    """Allocate on every torch-visible device in a throwaway subprocess (so the caller never gets a CUDA context).

    Returns ``{available, device_count, devices: [{torch_index, ok, error?}], error?}``; never raises."""
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE_SNIPPET], capture_output=True, text=True, timeout=timeout, check=False)
        for line in reversed((out.stdout or "").strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"available": False, "device_count": 0, "devices": [], "error": (out.stderr or "probe produced no output").strip()[-400:]}
    except subprocess.TimeoutExpired:
        return {"available": False, "device_count": 0, "devices": [], "error": f"device probe timed out after {timeout:.0f}s (driver hang?)"}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "device_count": 0, "devices": [], "error": f"{type(e).__name__}: {e}"}


def detect_stack(*, probe: bool = True, probe_timeout: float = 120.0) -> dict[str, Any]:
    """Collect the facts :func:`evaluate` needs.  ``probe=False`` skips the (torch-importing) device allocation test."""
    smi = _smi_facts()
    nvml = _nvml_facts()
    stack: dict[str, Any] = {
        "driver_version": nvml.get("driver_version") or (smi["devices"][0]["driver"] if smi.get("devices") else None),
        "driver_cuda": nvml.get("driver_cuda") or smi.get("driver_cuda"),
        "smi_present": bool(smi.get("present")),
        "smi_ok": bool(smi.get("ok")),
        "devices": smi.get("devices", []),
        "torch": _torch_facts(),
        "triton": _pkg("triton"),
        "nvidia_ml_py": _pkg("nvidia-ml-py"),
        "nvcc": _nvcc_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "platform": sys.platform,
    }
    if probe and stack["torch"].get("installed") and stack["smi_present"]:
        stack["probe"] = probe_devices(probe_timeout)
    return stack


# --------------------------------------------------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------------------------------------------------
def _f(level: str, code: str, message: str, fix: str | None = None) -> dict[str, Any]:
    return {"level": level, "code": code, "message": message, "fix": fix}


def worst(findings: list[dict[str, Any]]) -> str:
    return max((f["level"] for f in findings), key=LEVELS.index, default="ok")


def _visible_devices(devices: list[dict[str, Any]], cvd: str | None) -> list[dict[str, Any]]:
    """Restrict nvidia-smi's physical device list to what ``CUDA_VISIBLE_DEVICES`` exposes (integer lists only; UUID/MIG forms and
    out-of-range ids leave the list untouched so a rule never fires on a guess)."""
    if cvd is None or not cvd.strip() or cvd.strip() == "-1":
        return devices
    try:
        ids = [int(x) for x in cvd.split(",")]
    except ValueError:
        return devices
    by_index = {d["index"]: d for d in devices}
    picked = [by_index[i] for i in ids if i in by_index]
    return picked if picked else devices


def evaluate(stack: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: C901 - one flat list of independent rules
    """Apply every compatibility rule to a detected (or synthetic) stack.  Pure; the order is stable."""
    F: list[dict[str, Any]] = []
    drv = parse_version(stack.get("driver_version"))
    dcuda = parse_version(stack.get("driver_cuda"))
    torch = stack.get("torch") or {}
    tcuda = parse_version(torch.get("cuda"))
    devices = _visible_devices(stack.get("devices") or [], stack.get("cuda_visible_devices"))
    sms = sorted({d["sm"] for d in devices if d.get("sm")})
    probe = stack.get("probe") or {}
    probe_devs = probe.get("devices") or []
    bad_probe = [d for d in probe_devs if not d.get("ok")]
    busy = any(_BUSY in (d.get("error") or "") for d in bad_probe)

    # -- driver present ------------------------------------------------------------------------------------------------
    if stack.get("smi_present") and not stack.get("smi_ok", True):
        F.append(_f("fail", "nvidia_smi_failed", "nvidia-smi is installed but fails to query the GPUs (driver not loaded, GPU fault, or no device nodes in this container)",
                    "run `nvidia-smi` and read its error; recreate the instance if the driver/GPU is not reachable"))
        return F
    if (stack.get("cuda_visible_devices") is not None and stack["cuda_visible_devices"].strip() in ("", "-1")
            and stack.get("devices")):
        F.append(_f("info", "gpus_hidden", "all GPUs are hidden from this process by CUDA_VISIBLE_DEVICES"))
        return F
    if not stack.get("smi_present") or not devices:
        F.append(_f("warn", "no_nvidia_driver", "no NVIDIA driver / GPU visible (nvidia-smi missing or listing no devices)",
                    "fine for the UI and CPU tests; training and serving need an NVIDIA GPU instance with the driver passed through"))
        return F
    F.append(_f("ok", "driver", f"driver {fmt_driver(drv)}" + (f", supports up to CUDA {fmt_version(dcuda)}" if dcuda else "")
                + f", {len(devices)} GPU(s): " + ", ".join(sorted({d['name'] for d in devices}))))

    # -- torch build -------------------------------------------------------------------------------------------------------
    if not torch.get("installed"):
        F.append(_f("fail", "torch_missing", "torch is not installed", "install the CUDA build that matches this driver (see `recommend`)"))
    elif torch.get("hip"):
        F.append(_f("fail", "torch_rocm", "torch is a ROCm build; Forge targets NVIDIA CUDA", "install a CUDA build of torch"))
    elif tcuda is None:
        F.append(_f("fail", "torch_cpu_only", f"torch {torch.get('version')} is a CPU-only build but this host has NVIDIA GPUs",
                    "install a CUDA build: " + (recommend(stack).get("torch_install_hint") or "pip install torch --index-url https://download.pytorch.org/whl/cu128")))
    else:
        F.append(_f("ok", "torch", f"torch {torch.get('version')} built for CUDA {fmt_version(tcuda)}"
                    + (" (container build, do not pip-replace)" if torch.get("container_build") else "")))
        # driver must be new enough for the torch build
        floor = DRIVER_FLOOR.get(tcuda[0])
        if floor and drv and drv < tuple(floor[:len(drv)]):  # compare at the precision the driver string has ("580.65" is not below 580.65.06)
            F.append(_f("fail", "driver_below_family_floor",
                        f"driver {fmt_driver(drv)} is older than {fmt_driver(floor)}, the minimum for any CUDA {tcuda[0]}.x build",
                        f"use an instance with driver >= {fmt_driver(native_driver_for(tcuda)[0])}, or a torch built for CUDA {dcuda[0] if dcuda else tcuda[0] - 1}.x"))
        elif dcuda and tcuda[0] > dcuda[0]:
            F.append(_f("fail", "torch_cuda_newer_major",
                        f"torch needs CUDA {tcuda[0]}.x but the driver only supports up to CUDA {fmt_version(dcuda)}",
                        f"pick an instance with driver >= {fmt_driver(native_driver_for((tcuda[0], 0))[0])}, or install a CUDA {dcuda[0]}.x torch build"))
        elif dcuda and tcuda > dcuda:
            nd, src, exact = native_driver_for(tcuda)
            F.append(_f("info", "torch_cuda_newer_minor",
                        f"torch is built for CUDA {fmt_version(tcuda)} but the driver ({fmt_driver(drv)}) only supports up to CUDA {fmt_version(dcuda)}. "
                        f"NVIDIA supports this through minor-version compatibility (any {tcuda[0]}.x on driver >= {fmt_driver(DRIVER_FLOOR[tcuda[0]])}); "
                        "limits: PTX JIT of newer code and features added after the driver's release are unavailable",
                        f"for the full feature set use a host with driver {fmt_driver(nd)} or newer"
                        + ("" if exact else f" (CUDA {fmt_version(tcuda)}'s own branch is not in Forge's table; at least that of CUDA {fmt_version(max(k for k in NATIVE_DRIVER if k <= tuple(tcuda[:2])))})")))
        elif dcuda and tcuda <= dcuda:
            F.append(_f("ok", "driver_torch_match", f"driver CUDA {fmt_version(dcuda)} >= torch CUDA {fmt_version(tcuda)}"))

        # CUDA 13 dropped pre-Turing architectures
        if tcuda[0] >= 13:
            old = [d for d in devices if d.get("sm") and d["sm"] < CUDA13_MIN_SM]
            if old:
                F.append(_f("fail", "cuda13_dropped_arch",
                            "CUDA 13 builds do not support GPUs older than Turing (sm_75): " + ", ".join(f"{d['name']} (sm_{d['sm']})" for d in old),
                            "install a torch built for CUDA 12.x (cu126/cu128) for these GPUs"))

        # GPU architecture vs torch build
        archs = [a for a in (torch.get("arch_list") or [])]
        sass = {int(m.group(1)) for a in archs if (m := re.fullmatch(r"sm_(\d+)[a-z]?", a))}
        ptx = {int(m.group(1)) for a in archs if (m := re.fullmatch(r"compute_(\d+)[a-z]?", a))}
        for sm in sms:
            need = ARCH_MIN_CUDA.get(sm)
            if need and tcuda < need:
                F.append(_f("fail", "arch_needs_newer_cuda", f"sm_{sm} GPUs need a CUDA >= {fmt_version(need)} torch build; this torch is CUDA {fmt_version(tcuda)}",
                            "install a newer torch build (" + (recommend(stack).get("torch_install_hint") or "cu128 or newer") + ")"))
            elif archs:
                if sm in sass:
                    continue
                family = {s for s in sass if s // 10 == sm // 10 and s <= sm}  # same-major-arch SASS runs on later minors (e.g. sm_100 on sm_103)
                if family:
                    F.append(_f("info", "arch_family_sass", f"sm_{sm} runs on sm_{max(family)} SASS shipped in this torch (family compatible, not tuned)"))
                elif any(p <= sm for p in ptx):
                    F.append(_f("warn", "arch_ptx_only", f"torch ships no native code for sm_{sm}; it will JIT PTX at first use (slow start, may miss kernels)",
                                "install a torch build whose arch list includes sm_%d" % sm))
                else:
                    F.append(_f("fail", "arch_not_in_torch", f"torch's arch list {archs} has no code for sm_{sm} ({', '.join(sorted({d['name'] for d in devices if d.get('sm') == sm}))})",
                                "install a torch build that includes this architecture"))

    # -- triton ----------------------------------------------------------------------------------------------------------------
    trv = parse_version(stack.get("triton"))
    if trv is None:
        F.append(_f("warn", "triton_missing", "triton is not installed; Forge kernels fall back to slower torch reference paths", "pip install 'triton>=3.3'"))
    elif any(sm >= 100 for sm in sms) and trv < TRITON_BLACKWELL_MIN:
        F.append(_f("warn", "triton_too_old_for_blackwell", f"triton {fmt_version(trv)} predates Blackwell (sm_100/sm_120) code generation (needs >= {fmt_version(TRITON_BLACKWELL_MIN)})",
                    "pip install -U 'triton>=3.3' (or a torch build that ships a matching triton)"))
    else:
        F.append(_f("ok", "triton", f"triton {fmt_version(trv)}"))

    # -- nvml bindings ------------------------------------------------------------------------------------------------------------
    nv = parse_version(stack.get("nvidia_ml_py"))
    if nv is None:
        F.append(_f("info", "nvml_missing", "nvidia-ml-py not installed; GPU telemetry falls back to nvidia-smi", "pip install nvidia-ml-py"))
    elif dcuda and nv[0] < dcuda[0]:
        F.append(_f("warn", "nvml_bindings_old", f"nvidia-ml-py {fmt_version(nv)} predates this driver's CUDA {dcuda[0]}.x NVML API (new fields read as unsupported)",
                    f"pip install -U 'nvidia-ml-py>={dcuda[0]}'"))

    # -- nvcc (only matters when building extensions; Forge's own kernels are Triton) --------------------------------------------------
    nvcc = parse_version(stack.get("nvcc"))
    if nvcc and tcuda and nvcc[0] != tcuda[0]:
        F.append(_f("warn", "nvcc_major_mismatch", f"nvcc is CUDA {fmt_version(nvcc)} but torch is built for CUDA {fmt_version(tcuda)}; compiling torch extensions will fail",
                    "only matters if you build CUDA extensions (flash-attn, etc.): install the matching toolkit or use prebuilt wheels"))
    elif nvcc and tcuda and nvcc != tcuda:
        F.append(_f("info", "nvcc_minor_mismatch", f"nvcc {fmt_version(nvcc)} vs torch CUDA {fmt_version(tcuda)} (same major: extension builds normally work)"))

    # -- compute mode ---------------------------------------------------------------------------------------------------------------
    modes = {str(d.get("compute_mode") or "") for d in devices} - {""}
    if any("Prohibited" in m for m in modes):
        F.append(_f("fail", "compute_mode_prohibited", "a GPU is in Prohibited compute mode: no process can create a CUDA context", "nvidia-smi -c DEFAULT (needs root/privileged container)"))
    elif any("Exclusive" in m for m in modes):
        F.append(_f("warn", "compute_mode_exclusive", "a GPU is in Exclusive_Process compute mode: only one process may use it, but Forge runs a worker process per job plus the server probe",
                    "nvidia-smi -c DEFAULT (needs root/privileged container)"))

    # -- device usability (allocation test) ----------------------------------------------------------------------------------------------
    if probe.get("error") and not probe_devs:
        F.append(_f("fail", "cuda_probe_failed", f"could not test the GPUs through torch: {probe['error']}"))
    elif (probe and probe.get("device_count", 0) == 0 and torch.get("installed") and tcuda is not None
          and (stack.get("cuda_visible_devices") is None or stack["cuda_visible_devices"].strip() not in ("", "-1"))):  # empty/-1 = hidden on purpose
        F.append(_f("fail", "torch_sees_no_gpu", "torch built with CUDA support sees no device although the driver lists GPUs",
                    "check CUDA_VISIBLE_DEVICES, container GPU pass-through (/dev/nvidia*) and that the driver is not older than the torch build requires"))
    elif bad_probe:
        causes = []
        if busy:
            if dcuda and tcuda and tcuda > dcuda:
                causes.append(f"driver CUDA {fmt_version(dcuda)} is older than torch's CUDA {fmt_version(tcuda)} (supported by NVIDIA via minor-version compatibility, "
                              "but rule it out by trying a torch/host with matching versions)")
            causes.append("another process holds the GPU, or compute mode is Exclusive_Process/Prohibited (`nvidia-smi --query-compute-apps=pid,process_name --format=csv`, `nvidia-smi -c DEFAULT`)")
            causes.append("the container was not given the GPUs it lists (`ls /dev/nvidia*`, NVIDIA_VISIBLE_DEVICES) or the machine has a driver/fabric fault: recreate the instance or try another host")
        F.append(_f("fail", "devices_unusable",
                    f"{len(bad_probe)} of {len(probe_devs)} CUDA device(s) failed an allocation test: " + "; ".join(sorted({d.get('error', '?') for d in bad_probe}))[:300],
                    "; ".join(f"{i + 1}) {c}" for i, c in enumerate(causes)) or None))
        good = [d["torch_index"] for d in probe_devs if d.get("ok")]
        if good:
            F.append(_f("info", "some_devices_usable", f"devices {good} work; run with CUDA_VISIBLE_DEVICES={','.join(map(str, good))}"
                    + (f" (indices are relative to the current CUDA_VISIBLE_DEVICES={stack['cuda_visible_devices']})" if stack.get("cuda_visible_devices") else "")))
    elif probe_devs:
        F.append(_f("ok", "devices_usable", f"{len(probe_devs)} CUDA device(s) passed an allocation test"))
    return F


def recommend(stack: dict[str, Any]) -> dict[str, Any]:
    """Which torch wheel index and companion requirements fit this host.  Advisory; never touches the environment."""
    dcuda = parse_version(stack.get("driver_cuda"))
    sms = [d["sm"] for d in stack.get("devices") or [] if d.get("sm")]
    torch = stack.get("torch") or {}
    rec: dict[str, Any] = {"torch_index": None, "torch_index_url": None, "torch_install_hint": None, "requirements": [], "notes": [],
                           "reinstall_torch": False}
    if not dcuda:
        rec["requirements"] = ["nvidia-ml-py>=12.535", "triton>=2.2"]
        rec["notes"].append("driver CUDA version unknown (no NVML / nvidia-smi): cannot choose a torch build")
        return rec
    cap = dcuda
    # newest wheel line this driver can run (PyTorch publishes cu126 / cu128 / cu129 / cu130)
    # (cu132/cu134 also exist on the index, but cu130 runs on every 13.x driver and is published for every torch release that has CUDA 13)
    lines = [(13, 0, "cu130"), (12, 9, "cu129"), (12, 8, "cu128"), (12, 0, "cu126")]
    pick = next((name for ma, mi, name in lines if cap >= (ma, mi)), None)
    if pick and sms and min(sms) < CUDA13_MIN_SM and pick in ("cu130", "cu129", "cu128"):
        pick = "cu126"  # newer wheel lines have dropped Volta (cu128 in torch >= 2.11, all of CUDA 13)
        rec["notes"].append("a GPU older than Turing is present: CUDA 13 (and recent cu128+) wheels do not support it, using the CUDA 12.6 line")
    if pick and sms and max(sms) >= 100 and cap < (12, 8):
        rec["notes"].append("Blackwell GPUs need CUDA >= 12.8 but this driver only supports %s: the driver must be upgraded" % fmt_version(cap))
        pick = None
    if pick:
        rec["torch_index"] = pick
        rec["torch_index_url"] = f"https://download.pytorch.org/whl/{pick}"
        rec["torch_install_hint"] = f"pip install torch --index-url {rec['torch_index_url']}"
    rec["requirements"] = [f"nvidia-ml-py>={max(dcuda[0], 12)}" if dcuda[0] >= 13 else "nvidia-ml-py>=12.535"]
    if sms and max(sms) >= 100:
        rec["requirements"].append("triton>=3.3")
    else:
        rec["requirements"].append("triton>=2.2")
    if torch.get("container_build"):
        rec["notes"].append("torch is an NVIDIA container build pinned to this image: do not pip-replace it; fix the host/driver instead")
    else:
        tcuda = parse_version(torch.get("cuda"))
        rec["reinstall_torch"] = bool(torch.get("installed") and (tcuda is None or tcuda[0] > dcuda[0]))
    return rec


# --------------------------------------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------------------------------------
def check(*, probe: bool = True, probe_timeout: float = 120.0, stack: dict[str, Any] | None = None) -> dict[str, Any]:
    """detect + evaluate + recommend.  ``ok`` is False when any finding is ``fail``."""
    st = stack if stack is not None else detect_stack(probe=probe, probe_timeout=probe_timeout)
    findings = evaluate(st)
    return {"ok": worst(findings) != "fail", "worst": worst(findings), "stack": st, "findings": findings, "recommend": recommend(st)}


def format_report(rep: dict[str, Any]) -> str:
    tag = {"ok": "[ ok ]", "info": "[info]", "warn": "[warn]", "fail": "[FAIL]"}
    lines = []
    for f in rep["findings"]:
        lines.append(f"{tag[f['level']]} {f['message']}")
        if f.get("fix") and f["level"] in ("warn", "fail"):
            lines.append(f"       -> {f['fix']}")
    rec = rep.get("recommend") or {}
    for n in rec.get("notes", []):
        lines.append(f"[note] {n}")
    if not rep["ok"] and rec.get("torch_install_hint") and rec.get("reinstall_torch"):
        lines.append(f"[hint] {rec['torch_install_hint']}")
    return "\n".join(lines)


class CudaUnusable(Exception):
    """Raised by :func:`assert_cuda_usable`; ``findings`` carries the failing rules."""

    def __init__(self, message: str, findings: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.findings = findings


_CACHE: dict[str, Any] = {}


def assert_cuda_usable(*, refresh: bool = False, probe_timeout: float = 120.0) -> dict[str, Any]:
    """Gate for code about to use a GPU: raise :class:`CudaUnusable` (with the diagnosis) when the stack is broken.

    A successful result is cached for the process (the stack does not change under a running worker); a failure is re-checked
    every call because the operator may have fixed the host in the meantime.  Hosts with no GPU at all do not raise here - callers
    that need a GPU decide that themselves."""
    if not refresh and _CACHE.get("ok"):
        return _CACHE["ok"]
    rep = check(probe=True, probe_timeout=probe_timeout)
    fails = [f for f in rep["findings"] if f["level"] == "fail"]
    if fails:
        raise CudaUnusable("CUDA stack problem: " + " | ".join(f["message"] + (f" -> {f['fix']}" if f.get("fix") else "") for f in fails), fails)
    _CACHE["ok"] = rep
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m forge.hw.cuda_compat", description="Check the driver / torch / GPU / Triton CUDA stack.")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--no-probe", action="store_true", help="skip the per-device allocation test (no torch import)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--torch-index", action="store_true", help="print only the recommended torch wheel suffix (e.g. cu130), empty if unknown")
    ap.add_argument("--requirements", action="store_true", help="print only the recommended companion requirements, one per line")
    args = ap.parse_args(argv)
    rep = check(probe=not (args.no_probe or args.torch_index or args.requirements))
    if args.torch_index:
        print(rep["recommend"].get("torch_index") or "")
        return 0
    if args.requirements:
        print("\n".join(rep["recommend"]["requirements"]))
        return 0
    print(json.dumps(rep, indent=2) if args.json else format_report(rep))
    if not rep["ok"] or (args.strict and rep["worst"] == "warn"):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
