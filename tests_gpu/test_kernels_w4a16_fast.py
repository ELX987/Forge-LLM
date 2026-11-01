"""GPU tests (auto-skipped without CUDA): nibble-split W4A16 kernels (impl 'dot' / 'gemv') vs the fp32 oracle.

NEVER run by the developer (no NVIDIA GPU on the dev machine); logic is covered by the Triton CPU interpreter tests in
tests/test_kernels_w4_fast_interp.py.  These are the compile + numerics + sanity-speed gate for the GPU host:

    pytest tests_gpu/test_kernels_w4a16_fast.py -q -s

then run ``python scripts/bench_w4a16.py`` to tune and write the autotune cache.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
triton = pytest.importorskip("triton")

from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.layouts import PORTABLE_W4, pack_w4, quantize_w4_rtn  # noqa: E402
from forge.kernels.ref import compare, w4a16_gemm_ref  # noqa: E402
from forge.kernels.spec import load_spec  # noqa: E402
from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton  # noqa: E402
from forge.kernels.w4_tuning import GEMV_MAX_M, default_w4_config, w4_candidates  # noqa: E402

DEV = "cuda"
TOL = load_spec("w4a16_gemm").tolerance
MS = [1, 2, 4, 8, 16, 64, 256]


def _chip():
    from forge.kernels.layouts.chip import chip_info

    return chip_info(None, 0)


def _dtypes():
    out = [torch.float16]
    if torch.cuda.get_device_capability()[0] >= 8:
        out.append(torch.bfloat16)
    return out


def make(n, k, gs, sym, act_order=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.05).to(DEV)
    q, s, z = quantize_w4_rtn(w, gs, sym=sym)
    gsz = k if gs <= 0 else gs
    gi = (torch.arange(k, device=DEV) // gsz)[torch.randperm(k, generator=g).to(DEV)].to(torch.int32) if act_order else None
    return pack_w4(q, s, z, gs, PORTABLE_W4, gi)


def run(x, p, bias=None, **cfg):
    chip = _chip()
    return w4a16_gemm_triton(x, p, bias, config=cfg or None, sm=chip.sm, num_sms=chip.num_sms)


@pytest.mark.parametrize("impl", ["dot", "gemv", "legacy", None])
@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("n,k,gs", [(4096, 4096, 128), (1000, 4096, 64), (4096, 1000, 32), (777, 2056, 128), (256, 128, -1), (5120, 5120, 128)])
@pytest.mark.parametrize("sym", [True, False])
def test_impls_match_oracle(impl, m, n, k, gs, sym):
    if impl == "gemv" and m > GEMV_MAX_M:
        pytest.skip("gemv is decode only")
    p = make(n, k, gs, sym)
    for dtype in _dtypes():
        x = torch.randn(m, k, device=DEV, dtype=dtype)
        bias = torch.randn(n, device=DEV, dtype=dtype)
        y = run(x, p, bias, **({"impl": impl} if impl else {}))
        res = compare(y, w4a16_gemm_ref(x, p, bias), TOL, out_dtype=dtype)
        assert res.ok, (impl, dtype, res.detail)


@pytest.mark.parametrize("impl", ["dot", "gemv"])
@pytest.mark.parametrize("k", [100, 131, 1001])
def test_k_not_multiple_of_eight_or_group(impl, k):
    p = make(333, k, 32, False)
    for dtype in _dtypes():
        x = torch.randn(3, k, device=DEV, dtype=dtype)
        assert compare(run(x, p, impl=impl), w4a16_gemm_ref(x, p), TOL, out_dtype=dtype).ok


@pytest.mark.parametrize("impl", ["dot", "gemv", None])
@pytest.mark.parametrize("m", [1, 4, 16, 64])
@pytest.mark.parametrize("gs", [128, 64, 32])
def test_act_order_g_idx_weights(impl, m, gs):
    if impl == "gemv" and m > GEMV_MAX_M:
        pytest.skip("gemv is decode only")
    p = make(1024, 2048, gs, False, act_order=True)
    for dtype in _dtypes():
        x = torch.randn(m, 2048, device=DEV, dtype=dtype)
        y = run(x, p, **({"impl": impl} if impl else {}))
        assert compare(y, w4a16_gemm_ref(x, p), TOL, out_dtype=dtype).ok


@pytest.mark.parametrize("impl", ["dot", "gemv"])
@pytest.mark.parametrize("split_k", [1, 2, 4, 8, 16])
def test_split_k_is_deterministic_and_correct(impl, split_k):
    p = make(2048, 4096, 128, False)
    x = torch.randn(1, 4096, device=DEV, dtype=torch.float16)
    cfg = dict(impl=impl, tile_n=32, tile_k=128, split_k=split_k, stages=3, warps=4)
    chip = _chip()
    runs = [w4a16_gemm_triton(x, p, None, config=cfg, deterministic=True, sm=chip.sm, num_sms=chip.num_sms) for _ in range(5)]
    assert all(torch.equal(runs[0], r) for r in runs[1:])
    assert compare(runs[0], w4a16_gemm_ref(x, p), TOL, out_dtype=torch.float16).ok
    nd = w4a16_gemm_triton(x, p, None, config=cfg, deterministic=False, sm=chip.sm, num_sms=chip.num_sms)
    assert compare(nd, runs[0], TOL, out_dtype=torch.float16).ok


def test_every_candidate_of_the_sweep_compiles_and_is_correct():
    """The bench script's candidate lists must all be valid on this chip (a rejected candidate is simply never cached,
    but a systematic failure here means the search space or the kernel is wrong for this architecture)."""
    chip = _chip()
    p = make(2048, 4096, 128, False)
    bad = []
    for m in (1, 8, 16, 64):
        x = torch.randn(m, 4096, device=DEV, dtype=torch.float16)
        ref = w4a16_gemm_ref(x, p)
        for cfg in w4_candidates(m, 2048, 4096, sm=chip.sm, num_sms=chip.num_sms, group=128):
            try:
                y = w4a16_gemm_triton(x, p, None, config=cfg, sm=chip.sm, num_sms=chip.num_sms)
                if not compare(y, ref, TOL, out_dtype=torch.float16).ok:
                    bad.append((m, cfg, "wrong result"))
            except Exception as e:  # noqa: BLE001
                bad.append((m, cfg, f"{type(e).__name__}: {str(e)[:120]}"))
    assert not bad, bad[:10]


def test_dispatcher_uses_triton_and_default_config_is_not_the_legacy_fallback():
    p = make(4096, 4096, 128, False)
    x = torch.randn(1, 4096, device=DEV, dtype=torch.float16)
    y = ops_gemm.w4a16_linear(x, p)
    assert ops_gemm.last_path("w4a16_linear") == "triton"
    from forge.kernels.triton import gemm_w4a16

    assert not gemm_w4a16._FAST_BROKEN, f"nibble-split kernel failed and fell back to legacy: {gemm_w4a16._FAST_BROKEN}"
    assert compare(y, w4a16_gemm_ref(x, p), TOL, out_dtype=torch.float16).ok


def _graph_ms(fn, iters=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def test_decode_speed_vs_bf16_is_reported_not_asserted():
    """Prints device-time (CUDA graph) numbers for the legacy kernel, the default nibble-split config and bf16 F.linear.
    Only a sanity bound is asserted: the new default must not be slower than the legacy kernel it replaces."""
    import torch.nn.functional as F  # noqa: N812

    chip = _chip()
    n, k = 5120, 5120
    p = make(n, k, 128, False)
    dense = p.dequantize(torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    # ring of copies larger than L2 so nothing is served from cache
    ring = max(2, min(16, int(3 * 128 * 2**20 // dense.numel() // 2) + 1))
    denses = [dense] + [dense.clone() for _ in range(ring - 1)]
    for m in (1, 8, 16):
        x = torch.randn(m, k, device=DEV, dtype=dense.dtype)
        i = {"v": 0}

        def bf16():
            i["v"] += 1
            return F.linear(x, denses[i["v"] % ring])

        t_bf = _graph_ms(bf16)
        t_leg = _graph_ms(lambda: w4a16_gemm_triton(x, p, None, config={"impl": "legacy"}, sm=chip.sm, num_sms=chip.num_sms))
        t_new = _graph_ms(lambda: w4a16_gemm_triton(x, p, None, sm=chip.sm, num_sms=chip.num_sms))
        cfg = default_w4_config(m, n, k, sm=chip.sm, num_sms=chip.num_sms)
        print(f"M={m:<3} bf16 {t_bf * 1e3:7.1f}us  legacy {t_leg * 1e3:7.1f}us (x{t_bf / t_leg:.2f})  "
              f"nibble-split default {t_new * 1e3:7.1f}us (x{t_bf / t_new:.2f})  cfg={cfg}")
        assert t_new <= 1.25 * t_leg, "default nibble-split config slower than the legacy kernel: run scripts/bench_w4a16.py"
