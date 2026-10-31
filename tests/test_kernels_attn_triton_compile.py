"""Offline compilation of every attention / MoE / utility Triton kernel for several NVIDIA targets - no GPU needed.

Uses ``triton.compile`` with an explicit ``GPUTarget`` (Triton ships ``ptxas``), recording the launches of our launcher
functions instead of executing them.  This proves the kernels are accepted by the *real* Triton compiler (types, shapes,
constexpr rules, supported ops) and that their shared-memory footprint fits the target - the failure classes the CPU interpreter
does not catch.  It says nothing about numerical correctness on hardware (see ``tests_gpu/``).
Targets: ``FORGE_COMPILE_TARGETS`` (default ``"80,90"``; e.g. ``"70,75,80,89,90,100,120"`` for the full matrix).  Skipped when
``triton`` or its bundled ``ptxas`` is unavailable.
"""
from __future__ import annotations

import importlib.util
import os

import pytest
import torch

if importlib.util.find_spec("triton") is None:  # pragma: no cover
    pytest.skip("triton not installed", allow_module_level=True)

import kernels_attn_compile_cases as cases  # noqa: E402
import kernels_attn_compile_harness as harness  # noqa: E402

SMEM_LIMIT = {70: 96 * 1024, 75: 64 * 1024, 80: 163 * 1024, 86: 99 * 1024, 87: 163 * 1024, 89: 99 * 1024,
              90: 227 * 1024, 100: 227 * 1024, 103: 227 * 1024, 120: 99 * 1024, 121: 99 * 1024}
TARGETS = [int(x) for x in os.environ.get("FORGE_COMPILE_TARGETS", "80,90").split(",") if x.strip()]


@pytest.fixture(scope="module", autouse=True)
def _compiler_available():
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    @triton.jit
    def _probe(x_ptr, B: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, B), tl.zeros([B], tl.float32))

    try:
        triton.compile(ASTSource(fn=_probe, signature={"x_ptr": "*fp32", "B": "constexpr"}, constexprs={"B": 16}),
                       target=GPUTarget("cuda", 80, 32))
    except Exception as e:  # pragma: no cover
        pytest.skip(f"offline Triton compilation unavailable: {e}")


def _uses_fp8e4nv(rec) -> bool:
    """True when any launch argument is an E4M3 tensor, i.e. the variant lowers to ``float8e4nv`` (hardware: sm_89+ only)."""
    f8 = getattr(torch, "float8_e4m3fn", None)
    return f8 is not None and any(isinstance(v, torch.Tensor) and v.dtype == f8
                                  for v in (*rec.args, *rec.kwargs.values()))


def _check(rec, sm):
    if sm < 89 and _uses_fp8e4nv(rec):
        return  # fp8e4nv does not exist on Ampere and older; the dispatchers never launch these variants there
    try:
        ck = harness.compile_record(rec, sm)
    except Exception as e:
        raise AssertionError(f"{rec.kernel.__name__} failed to compile for sm_{sm}: {str(e)[-800:]}") from e
    shared = getattr(ck.metadata, "shared", 0) or 0
    limit = SMEM_LIMIT.get(sm)
    if limit is not None and sm >= 75:
        assert shared <= limit, f"{rec.kernel.__name__} uses {shared} B shared memory > {limit} B limit of sm_{sm}"


def _compile_group(name, dtype, native_fp8=False):
    from forge.kernels.triton import attn_decode, attn_mla, attn_sparse

    ctx = cases.native_fp8(attn_decode, attn_mla, attn_sparse) if native_fp8 else None
    with harness.recording() as recs:
        if ctx is not None:
            with ctx:
                cases.all_groups()[name](dtype)
        else:
            cases.all_groups()[name](dtype)
    recs = harness.unique_by_specialisation(recs)
    assert recs, f"group {name} recorded no kernel launches"
    for sm in TARGETS:
        if (dtype == torch.bfloat16 or name == "mla") and sm < 80:
            continue  # bf16 tensor-core dots and the MLA kernel (min_sm 80) are not dispatched below sm_80
        for rec in recs:
            _check(rec, sm)
    return len(recs)


@pytest.mark.slow
@pytest.mark.parametrize("group", ["decode", "mla", "prefill", "sparse_spec", "moe", "util"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_kernels_compile_offline(group, dtype):
    n = _compile_group(group, dtype)
    assert n > 0


@pytest.mark.slow
@pytest.mark.parametrize("group", ["decode", "mla", "sparse_spec"])
def test_native_fp8_decode_paths_compile_offline(group):
    """The sm_89+ code paths (`float8e4nv` bitcast loads) - compiled for the Ada/Hopper/Blackwell targets only."""
    from forge.kernels.triton import attn_decode, attn_mla, attn_sparse

    with harness.recording() as recs, cases.native_fp8(attn_decode, attn_mla, attn_sparse):
        cases.all_groups()[group](torch.bfloat16)
    for sm in [t for t in TARGETS if t >= 89] or [90]:
        for rec in harness.unique_by_specialisation(recs):
            _check(rec, sm)
