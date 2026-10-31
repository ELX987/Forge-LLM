"""Regression for the sm_120 failure of test_int8_w8a8_is_bit_exact_on_integer_path: the Triton quantiser used plain
``/`` (approximate ``div.full.f32`` on GPU) and so differed from ``torch.round(x / s)`` on rare elements/scales."""
import inspect
import re

import pytest

pytest.importorskip("triton")


def test_int8_quantiser_uses_ieee_division():
    from forge.kernels.triton import gemm_int8

    src = inspect.getsource(gemm_int8._int8_quant_rowwise_kernel.fn)
    assert "_div_rn(amax, 127.0)" in src and "_div_rn(xv, scale)" in src
    assert not re.search(r"[\w\)]\s*/\s*(scale|127\.0)", src)
