"""GPU tests (skipped without CUDA): lightning indexer, top-k selection, fused rejection-sampling verification (exactness +
reproducibility), KV accept/rollback compaction.  Run on the target GPU: ``pytest tests_gpu/test_kernels_attn_sparse_spec.py -q``."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from forge.kernels import ops_attn, ops_util  # noqa: E402
from forge.kernels.ref.attn_kv import build_paged_pools  # noqa: E402
from forge.kernels.ref.attn_sparse import indexer_scores_ref, topk_select_ref  # noqa: E402
from forge.kernels.ref.attn_spec import kv_compact_ref, rejection_verify_ref  # noqa: E402
from forge.kernels.triton.attn_sparse import topk_select_triton  # noqa: E402
from forge.kernels.triton.util_dispatch import device_sm  # noqa: E402

DEV = torch.device("cuda")
SM = device_sm(DEV) or 0
DT = torch.bfloat16 if SM >= 80 else torch.float16


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("lens", [[8192, 1000, 17], [131072]])
def test_indexer_scan_vs_oracle(fp8, lens):
    b, hi, di, ps = len(lens), 64, 128, 64
    g = torch.Generator().manual_seed(1)
    ik = torch.randn(b, 1, max(lens), di, generator=g)
    p = build_paged_pools(ik, ik, lens, ps, "fp32", generator=g)
    ik_pool = p["k_pool"][:, 0].contiguous()
    iq = torch.randn(b, hi, di, generator=g)
    iw = torch.rand(b, hi, generator=g)
    scale = None
    if fp8:
        scale = ik_pool.abs().amax(-1).clamp_min(1e-6) / 448.0
        store = (ik_pool / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    else:
        store = ik_pool.to(DT)
    out = ops_attn.lightning_indexer_scores(iq.to(DEV, DT), iw.to(DEV), store.to(DEV), scale.to(DEV) if scale is not None else None,
                                            p["block_tables"].to(DEV), p["seq_lens"].to(DEV))
    ref = indexer_scores_ref(iq.to(DT).float(), iw, store.float() if not fp8 else store, scale, p["block_tables"], p["seq_lens"])
    ok = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(out.cpu()), ok)
    err = (out.cpu()[ok] - ref[ok]).abs().max() / ref[ok].abs().max()
    assert err < 3e-2


@pytest.mark.parametrize("t,k,b", [(5000, 64, 3), (131072, 2048, 2), (40000, 2048, 1)])
def test_topk_triton_exact_vs_reference(t, k, b):
    s = torch.randn(b, t)
    s[0, ::13] = 3.0  # ties at the top
    lens = torch.tensor([t] + [t - 7] * (b - 1), dtype=torch.int32)
    ref = topk_select_ref(s, k, lens)
    out = topk_select_triton(s.to(DEV), k, lens.to(DEV))
    assert torch.equal(out.cpu(), ref)
    assert torch.equal(ops_util.topk_select(s.to(DEV), k, lens.to(DEV), method="torch").cpu(), ref)


@pytest.mark.parametrize("seed,offset", [(1, 0), (2 ** 40 + 7, 123)])
def test_rejection_verify_matches_reference_and_is_reproducible(seed, offset):
    torch.manual_seed(0)
    b, g, v = 256, 4, 32000
    p = torch.softmax(torch.randn(b, g + 1, v) * 2, -1)
    q = torch.softmax(torch.randn(b, g, v) * 2, -1)
    draft = torch.multinomial(q.reshape(-1, v), 1).reshape(b, g)
    rt, rn = rejection_verify_ref(draft, q, p, seed, offset)
    tt, tn = ops_util.rejection_verify(draft.to(DEV), q.to(DEV), p.to(DEV), seed, offset)
    assert torch.equal(rn, tn.cpu())
    assert (tt.cpu() == rt).float().mean() > 0.999  # only a CDF-boundary rounding difference can change a resampled token
    tt2, _ = ops_util.rejection_verify(draft.to(DEV), q.to(DEV), p.to(DEV), seed, offset)
    assert torch.equal(tt, tt2)


def test_rejection_verify_statistical_exactness_on_device():
    v, gamma, n = 6, 2, 400_000
    torch.manual_seed(5)
    p0 = torch.softmax(torch.randn(v), -1)
    q0 = torch.softmax(torch.randn(v), -1)
    p1 = torch.softmax(torch.randn(v, v), -1)
    q1 = torch.softmax(torch.randn(v, v), -1)
    p2 = torch.softmax(torch.randn(v, v, v), -1)
    x0 = torch.multinomial(q0.expand(n, v), 1).squeeze(1)
    x1 = torch.multinomial(q1[x0], 1).squeeze(1)
    draft = torch.stack([x0, x1], 1)
    qp = torch.stack([q0.expand(n, v), q1[x0]], 1)
    pp = torch.stack([p0.expand(n, v), p1[x0], p2[x0, x1]], 1)
    toks, _ = ops_util.rejection_verify(draft.to(DEV), qp.to(DEV), pp.to(DEV), seed=77)
    freq = torch.bincount(toks[:, 0].long().cpu(), minlength=v).float() / n
    assert (freq - p0).abs().max() < 0.004


def test_kv_compact_matches_reference():
    torch.manual_seed(2)
    ps, h, d, b = 16, 4, 128, 3
    ctx = torch.tensor([37, 64, 5], dtype=torch.int32)
    bt = torch.arange(b * 8, dtype=torch.int32).view(b, 8)
    kp = torch.randn(b * 8, h, ps, d).to(DT)
    vp = torch.randn(b * 8, h, ps, d).to(DT)
    acc = torch.tensor([[0, 2, 3, 6, 7, -1, -1, -1], [1, 2, 3, 4, 5, 6, 7, 8], [-1] * 8], dtype=torch.int32)
    alen = torch.tensor([5, 8, 0], dtype=torch.int32)
    k1, v1 = kp.clone(), vp.clone()
    kv_compact_ref(k1, v1, bt, ctx, acc, alen)
    k2, v2 = kp.clone().to(DEV), vp.clone().to(DEV)
    ops_util.kv_compact(k2, v2, bt.to(DEV), ctx.to(DEV), acc.to(DEV), alen.to(DEV))
    assert torch.equal(k1, k2.cpu()) and torch.equal(v1, v2.cpu())


def test_kv_compact_overlapping_ranges_are_race_free():
    """Source and destination ranges overlap heavily (node j -> position j - s); a single-kernel read-then-write races across
    warps on real hardware, the two-phase launch must reproduce the oracle every time."""
    torch.manual_seed(7)
    ps, h, d, b, a = 16, 8, 128, 4, 96
    ctx = torch.tensor([3, 40, 0, 17], dtype=torch.int32)
    bt = torch.arange(b * 16, dtype=torch.int32).view(b, 16)
    kp = torch.randn(b * 16, h, ps, d).to(DT)
    vp = torch.randn(b * 16, h, ps, d).to(DT)
    acc = torch.full((b, a), -1, dtype=torch.int32)
    alen = torch.tensor([90, 64, 33, 1], dtype=torch.int32)
    for bi in range(b):
        n = int(alen[bi])
        # accepted path in tree order: node index >= path position, strictly increasing (sources overlap destinations)
        acc[bi, :n] = torch.arange(n, dtype=torch.int32) + torch.randint(0, 2, (n,)).cumsum(0).int()
    k1, v1 = kp.clone(), vp.clone()
    kv_compact_ref(k1, v1, bt, ctx, acc, alen)
    for _ in range(25):
        k2, v2 = kp.clone().to(DEV), vp.clone().to(DEV)
        ops_util.kv_compact(k2, v2, bt.to(DEV), ctx.to(DEV), acc.to(DEV), alen.to(DEV))
        assert torch.equal(k1, k2.cpu()) and torch.equal(v1, v2.cpu())


def test_rejection_verify_device_offset_fresh_randoms_under_cuda_graph():
    """A device-tensor ``offset`` must stay a load inside a captured graph: bumping it between replays draws new randoms."""
    from forge.kernels.triton.attn_spec import rejection_verify_triton

    torch.manual_seed(1)
    b, g, v = 64, 3, 512
    p = torch.softmax(torch.randn(b, g + 1, v, device=DEV), -1)
    q = torch.softmax(torch.randn(b, g, v, device=DEV), -1)
    draft = torch.multinomial(q.reshape(-1, v), 1).reshape(b, g)
    counter = torch.zeros((), dtype=torch.int64, device=DEV)
    rejection_verify_triton(draft, q, p, 5, counter)  # warm-up / compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_t, out_n = rejection_verify_triton(draft, q, p, 5, counter)
    graph.replay()
    first = out_t.clone()
    counter += 10_000
    graph.replay()
    second = out_t.clone()
    ref_first = rejection_verify_ref(draft.cpu(), q.cpu(), p.cpu(), 5, 0)[0]
    ref_second = rejection_verify_ref(draft.cpu(), q.cpu(), p.cpu(), 5, 10_000)[0]
    assert (first.cpu() == ref_first).float().mean() > 0.995
    assert (second.cpu() == ref_second).float().mean() > 0.995
    assert not torch.equal(first, second)
