"""Build the OPTIONAL native ternary CUDA extension ``forge_ternary_C`` (ADR 0007, tier 3).

    cd forge/native/ternary && pip install -e . --no-build-isolation        (or: python setup.py build_ext --inplace)

Needs nvcc (CUDA 11.8+ for sm_89/90, 12.8+ for sm_100/120, 12.9+ for sm_103/121), PyTorch with CUDA, cuBLAS, C++17.  Forge works without it: the Triton kernels
and torch oracle are the portable paths; this extension is only used where its validation probe passes AND it beats Triton (not wired yet: phase P4 follow-up on hardware).
"""
import glob
import os
import subprocess
import sys

from setuptools import setup

here = os.path.dirname(os.path.abspath(__file__))


def gencode():
    out = subprocess.run([sys.executable, os.path.join(here, "tools", "detect_arch.py"), "--gencode"], capture_output=True, text=True).stdout.split()
    flags, i = [], 0
    while i < len(out):  # "-gencode arch=..,code=.." pairs
        flags += [out[i], out[i + 1]]
        i += 2
    return flags


if __name__ == "__main__":  # guarded: this file sits inside a python package that tests import module-by-module
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

    setup(
        name="forge-ternary-native",
        version="0.1.0",
        description="Optional native CUDA kernels for Forge's ternary (1.58-bit) path (from tern158 v0.1)",
        ext_modules=[CUDAExtension(
            name="forge_ternary_C",
            sources=sorted(glob.glob(os.path.join("kernel", "src", "*.cu"))) + [os.path.join("kernel", "torch_ext", "bindings.cpp")],
            include_dirs=[os.path.join(here, "kernel", "include")],
            libraries=["cublas"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"] + gencode()},
        )],
        cmdclass={"build_ext": BuildExtension},
        python_requires=">=3.10",
    )
