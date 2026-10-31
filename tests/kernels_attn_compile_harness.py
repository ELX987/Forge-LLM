"""Offline Triton *compilation* harness (no GPU, no kernel is executed).

Triton can lower a kernel all the way to PTX/cubin for an arbitrary NVIDIA ``sm`` target without a device present.  This module
records the launches made by our launcher functions (``JITFunction.run`` is replaced by a recorder, so nothing runs) and then
compiles each recorded kernel/specialisation for the requested targets with ``triton.compile``.  It catches the class of
bugs the CPU interpreter tolerates but the real compiler rejects (type mismatches, bad shapes, unsupported ops, non-constexpr
``tl.arange`` sizes, loop-carried type changes, ...).  It validates *compilability only* - not numerics, not performance.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

_DTYPE = {
    torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.int32: "i32", torch.int64: "i64",
    torch.int8: "i8", torch.uint8: "u8", torch.bool: "i1",
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE[torch.float8_e4m3fn] = "fp8e4nv"


@dataclass
class Record:
    kernel: object
    args: tuple
    kwargs: dict


@contextlib.contextmanager
def recording():
    """Record kernel launches instead of executing them."""
    from triton.runtime.jit import JITFunction

    recs: list[Record] = []
    orig = JITFunction.run

    def fake_run(self, *args, grid=None, warmup=False, **kwargs):  # noqa: ANN001
        recs.append(Record(self, args, kwargs))
        return None

    JITFunction.run = fake_run
    try:
        yield recs
    finally:
        JITFunction.run = orig


def _sig_entry(v) -> str:
    if isinstance(v, torch.Tensor):
        return "*" + _DTYPE[v.dtype]
    if isinstance(v, bool):
        return "i1"
    if isinstance(v, int):
        return "i32" if -(2 ** 31) <= v < 2 ** 31 else "i64"
    if isinstance(v, float):
        return "fp32"
    raise TypeError(f"cannot type launch argument {v!r}")


def compile_record(rec: Record, sm: int, warp_size: int = 32):
    """Compile one recorded launch for ``sm_<sm>``; returns the ``CompiledKernel`` (raises on any compiler error)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = rec.kernel
    names = list(fn.arg_names)
    values = dict(zip(names, rec.args))
    values.update({k: v for k, v in rec.kwargs.items() if k in names})
    opts = {k: v for k, v in rec.kwargs.items() if k in ("num_warps", "num_stages")}
    constexpr_names = {p.name for p in fn.params if p.is_constexpr}
    signature, constexprs = {}, {}
    for n in names:
        if n in constexpr_names:
            signature[n] = "constexpr"
            constexprs[n] = values[n]
        else:
            signature[n] = _sig_entry(values[n])
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("cuda", sm, warp_size), options=opts)


def unique_by_specialisation(recs: list[Record]) -> list[Record]:
    """Drop launches whose (kernel, constexpr values, argument types) were already seen."""
    seen, out = set(), []
    for r in recs:
        fn = r.kernel
        names = list(fn.arg_names)
        values = dict(zip(names, r.args))
        values.update({k: v for k, v in r.kwargs.items() if k in names})
        constexpr_names = {p.name for p in fn.params if p.is_constexpr}
        key = (fn.__name__, tuple(sorted((n, str(values[n])) for n in constexpr_names)),
               tuple((n, _sig_entry(values[n])) for n in names if n not in constexpr_names),
               tuple(sorted((k, v) for k, v in r.kwargs.items() if k in ("num_warps", "num_stages"))))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
