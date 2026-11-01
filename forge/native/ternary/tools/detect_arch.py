#!/usr/bin/env python3
"""Print the CUDA architectures (e.g. "80 86 90") to compile tern158 for.

  detect_arch.py            -> architectures of the GPUs visible via nvidia-smi (or all supported if none)
  detect_arch.py --all      -> every architecture the installed nvcc can target (for redistributable builds)
  detect_arch.py --gencode  -> the matching nvcc -gencode flags (plus PTX for the newest arch)
Environment override: TERN158_ARCHS="86 89".
"""
import os, re, subprocess, sys

# arch -> minimum CUDA toolkit that can target it
MIN_CUDA = {80: (11, 0), 86: (11, 1), 89: (11, 8), 90: (11, 8), 100: (12, 8), 103: (12, 9), 120: (12, 8), 121: (12, 9)}
# consumer / server families the project targets
FALLBACK_ALL = [80, 86, 89, 90, 100, 103, 120]


def nvcc_version():
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None


def gpu_caps():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        caps = set()
        for line in out.split():
            if re.match(r"^\d+\.\d+$", line):
                a, b = line.split(".")
                caps.add(int(a) * 10 + int(b))
        return sorted(caps)
    except Exception:
        return []


def supported(ver):
    if ver is None:
        return list(FALLBACK_ALL)
    return [a for a, mv in sorted(MIN_CUDA.items()) if ver >= mv]


def pick():
    env = os.environ.get("TERN158_ARCHS")
    ver = nvcc_version()
    sup = supported(ver)
    if env:
        return [int(x) for x in env.replace(",", " ").split()], sup
    if "--all" in sys.argv:
        return sup, sup
    caps = gpu_caps()
    if not caps:
        return sup, sup
    out = []
    for c in caps:
        if c in sup:
            out.append(c)
        else:  # newer GPU than this nvcc knows: fall back to the best older SASS + PTX (JIT at load time)
            lower = [a for a in sup if a <= c]
            if lower and lower[-1] not in out:
                out.append(lower[-1])
            sys.stderr.write(f"[detect_arch] warning: sm_{c} needs a newer CUDA toolkit; using sm_{lower[-1] if lower else '?'} + PTX\n")
    return sorted(set(out)), sup


def main():
    archs, sup = pick()
    if "--gencode" in sys.argv:
        flags = [f"-gencode arch=compute_{a},code=sm_{a}" for a in archs]
        top = max(archs)
        flags.append(f"-gencode arch=compute_{top},code=compute_{top}")
        print(" ".join(flags))
    else:
        print(" ".join(str(a) for a in archs))


if __name__ == "__main__":
    main()
