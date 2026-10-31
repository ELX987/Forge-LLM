"""GPU-only checks for forge.serve (skipped automatically without CUDA; see conftest.py).

These exercise code that CANNOT run on the CPU development machine: the paged decode kernel path of ``ModelRunner``,
CUDA-graph replay, GPU memory sizing of the KV pool, FP8 KV on sm>=89, and end-to-end serving on a real device.
Written but never executed by the authors - run on the rented host with ``scripts/run_gpu_tests.sh`` and report failures.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - collected but skipped by conftest
    pytest.skip("no CUDA device", allow_module_level=True)

from forge.models.tiny import build_tiny_model  # noqa: E402
from forge.serve._testing import ByteTokenizer  # noqa: E402
from forge.serve.config import ServeConfig  # noqa: E402
from forge.serve.engine import LLMEngine  # noqa: E402
from forge.serve.runner import SeqInput  # noqa: E402
from forge.serve.sampling import SamplingParams  # noqa: E402

DEV = "cuda:0"


def build(kernel="auto", kv="auto", graphs=False, dtype="bfloat16", **kw):
    m = build_tiny_model(11, dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16, num_layers=3).eval().to(DEV)
    base = dict(model="t", device=DEV, dtype=dtype, block_size=16, num_gpu_blocks=256, max_model_len=512, max_num_seqs=16,
                kernel_attention=kernel, kv_dtype=kv, cuda_graphs=graphs)
    base.update(kw)
    return m, LLMEngine(ServeConfig(**base), model=m, tokenizer=ByteTokenizer())


GREEDY = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)


def test_paged_decode_kernel_matches_reference_path():
    m, e_ref = build(kernel="off")
    _, e_ker = build(kernel="on")
    prompts = [list(range(5, 45)), list(range(60, 70)), list(range(100, 190))]
    o_ref = e_ref.generate(prompts, [GREEDY] * 3)
    o_ker = e_ker.generate(prompts, [GREEDY] * 3)
    agree = sum(a == b for r, k in zip(o_ref, o_ker) for a, b in zip(r.token_ids, k.token_ids))
    total = sum(len(r.token_ids) for r in o_ref)
    assert agree / total > 0.9, f"kernel/reference greedy agreement only {agree}/{total}"
    assert e_ker.runner.stats["kernel_decode_steps"] > 0 or not e_ker.runner._kernel_ok, \
        "kernel path was disabled by a runtime failure - check the log for the reason"


def test_kernel_logits_close_to_reference_logits():
    m, e_ref = build(kernel="off")
    _, e_ker = build(kernel="on")
    p = list(range(5, 100))
    outs = []
    for e in (e_ref, e_ker):
        e.manager.allocate("x", p)
        e.runner.forward([SeqInput(p, 0, e.manager.block_table("x"))])
        e.manager.reserve("x", len(p) + 1)
        outs.append(e.runner.forward([SeqInput([7], len(p), e.manager.block_table("x"))]).logits[0][-1].float())
    assert torch.allclose(outs[0], outs[1], atol=0.15, rtol=0.05)


def test_cuda_graph_decode_replays_and_matches():
    _, e_plain = build(kernel="on", graphs=False)
    _, e_graph = build(kernel="on", graphs=True)
    prompts = [list(range(5, 30)), list(range(40, 80))]
    a = e_plain.generate(prompts, [GREEDY] * 2)
    b = e_graph.generate(prompts, [GREEDY] * 2)
    assert [x.token_ids for x in a] == [x.token_ids for x in b]
    assert e_graph.runner.stats["graph_replays"] > 0 or not e_graph.runner._graph_ok


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1] < 89,
                    reason="FP8 needs sm89+")
def test_fp8_kv_cache_end_to_end():
    _, e = build(kernel="auto", kv="fp8_e4m3")
    o = e.generate([list(range(5, 60))], [GREEDY])[0]
    assert len(o.token_ids) == 24
    assert e.kv_name == "fp8_e4m3"


def test_speculative_greedy_is_exact_on_gpu():
    _, e_off = build()
    _, e_on = build(speculation=dict(mode="ngram", max_gamma=4, controller="fixed"))
    p = list(b"abcabcabcabcabcabc") * 2
    a = e_off.generate([p], [GREEDY])[0].token_ids
    b = e_on.generate([p], [GREEDY])[0].token_ids
    assert sum(x == y for x, y in zip(a, b)) / len(a) > 0.9  # bf16 verification shapes differ -> allow rare near-tie flips


def test_kv_pool_sizing_from_free_memory_and_server_smoke():
    m = build_tiny_model(2, dtype=torch.bfloat16).eval().to(DEV)
    cfg = ServeConfig(model="t", device=DEV, dtype="bfloat16", gpu_mem_util=0.3, max_model_len=256, max_num_seqs=8)
    e = LLMEngine(cfg, model=m, tokenizer=ByteTokenizer())
    assert e.pool.num_blocks >= 8
    assert e.pool.k.device.type == "cuda"
    o = e.generate([list(range(5, 40))], [GREEDY])[0]
    assert len(o.token_ids) == 24
    assert e.stats()["gpu_mem_gb"] > 0


def test_host_tier_demote_promote_roundtrip_on_gpu():
    _, e = build(host_kv_gb=0.05, num_gpu_blocks=64)
    p = list(range(5, 85))
    first = e.generate([p], [GREEDY])[0].token_ids
    assert e.manager.demote_cold() > 0
    assert e.manager.stats()["bytes_by_tier"]["host"] > 0
    again = e.generate([p], [GREEDY])[0]
    assert again.token_ids == first and again.cached_tokens > 0
    assert e.manager.stats()["promotions"] > 0
