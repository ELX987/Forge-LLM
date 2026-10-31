"""LLMEngine on CPU with a tiny random model: exactness vs full-recompute greedy, caching/chunking invariance,
ragged continuous batching, sampling features, preemption, abort, priorities, KV dtypes."""
from __future__ import annotations

import pytest
import torch

from forge.serve._testing import EOS, ByteTokenizer, make_engine, tiny_model
from forge.serve.sampling import SamplingParams

GREEDY = dict(temperature=0.0, ignore_eos=True)


def sp(**kw):
    base = dict(GREEDY)
    base.update(kw)
    return SamplingParams(**base)


def ref_tokens(model, prompt, n):
    return model.greedy_generate(prompt, n, eos=None)


PROMPTS = [list(b"hello world, hello"), list(b"the quick brown fox jumps over the lazy dog"), list(b"a"), list(range(10, 41))]


def test_greedy_matches_full_recompute_ragged_batch():
    m = tiny_model(0)
    e = make_engine(m)
    outs = e.generate(PROMPTS, [sp(max_tokens=18) for _ in PROMPTS])
    for p, o in zip(PROMPTS, outs):
        assert o.token_ids == ref_tokens(m, p, 18)
        assert o.finish_reason == "length" and o.finished
        assert o.prompt_tokens == len(p) and o.completion_tokens == 18


def test_batch1_vs_batched_logits_close():
    from forge.serve.runner import SeqInput

    m = tiny_model(1)
    e = make_engine(m)
    # allocate two sequences by hand and compare ragged batched logits with single-sequence runs
    a, b = PROMPTS[1], PROMPTS[3]
    for sid, p in (("a", a), ("b", b)):
        assert e.manager.allocate(sid, p) == 0
    both = e.runner.forward([SeqInput(a, 0, e.manager.block_table("a")), SeqInput(b, 0, e.manager.block_table("b"))])
    ra, rb = m.logits_for(a)[-1], m.logits_for(b)[-1]
    assert torch.allclose(both.logits[0][-1], ra, atol=1e-4, rtol=1e-4)
    assert torch.allclose(both.logits[1][-1], rb, atol=1e-4, rtol=1e-4)


def test_prefix_caching_on_off_identical_and_hits():
    m = tiny_model(2)
    shared = list(b"You are a helpful assistant. Answer briefly. ")
    prompts = [shared + list(b"Q1: what is 2+2?"), shared + list(b"Q2: name a colour"), shared + list(b"Q3: hi")]
    res = {}
    for pc in (True, False):
        e = make_engine(m, prefix_caching=pc)
        outs = []
        for p in prompts:  # sequential so later requests can hit
            outs.append(e.generate([p], [sp(max_tokens=12)])[0])
        res[pc] = (outs, e.stats())
    for a, b in zip(res[True][0], res[False][0]):
        assert a.token_ids == b.token_ids  # G-LOSSLESS: caching never changes outputs
    assert res[True][0][1].cached_tokens >= 40
    assert res[True][1]["prefix_hit_rate"] > 0.3
    assert res[False][1]["prefix_hit_rate"] == 0.0
    for p, o in zip(prompts, res[True][0]):
        assert o.token_ids == ref_tokens(m, p, 12)


def test_identical_prompt_twice_hits_cache_but_same_output():
    m = tiny_model(3)
    e = make_engine(m)
    p = PROMPTS[1]
    o1 = e.generate([p], [sp(max_tokens=10)])[0]
    o2 = e.generate([p], [sp(max_tokens=10)])[0]
    assert o1.token_ids == o2.token_ids
    assert o2.cached_tokens == ((len(p) - 1) // 4) * 4


def test_chunked_prefill_identical():
    m = tiny_model(4)
    long_prompt = list(range(30, 130))
    for chunk_budget in (7, 16, 64):
        e = make_engine(m, chunked_prefill=True, max_num_batched_tokens=chunk_budget, max_model_len=256, num_gpu_blocks=128)
        outs = e.generate([long_prompt, PROMPTS[0]], [sp(max_tokens=10), sp(max_tokens=10)])
        assert outs[0].token_ids == ref_tokens(m, long_prompt, 10), chunk_budget
        assert outs[1].token_ids == ref_tokens(m, PROMPTS[0], 10)
    # non-chunked with oversize prompt still works (runs alone)
    e = make_engine(m, chunked_prefill=False, max_num_batched_tokens=16, max_model_len=256, num_gpu_blocks=128)
    o = e.generate([long_prompt], [sp(max_tokens=6)])[0]
    assert o.token_ids == ref_tokens(m, long_prompt, 6)


def test_chunked_prefill_mixes_with_decode():
    m = tiny_model(5)
    e = make_engine(m, max_num_batched_tokens=12, max_model_len=256, num_gpu_blocks=128)
    e.add_request(PROMPTS[0], sp(max_tokens=30), "short")
    for _ in range(3):
        e.step()
    e.add_request(list(range(20, 90)), sp(max_tokens=4), "long")
    saw_mixed = False
    while e.has_unfinished():
        e.step()
        info = e.last_step_info
        if info.get("scheduled", 0) >= 2:
            saw_mixed = True
    assert saw_mixed
    assert e.stats()["total_finished"] == 2


@pytest.mark.parametrize("overrides", [
    dict(sliding_window=8),
    dict(qk_norm=True),
    dict(num_kv_heads=4),
    dict(tie_word_embeddings=True),
    dict(rope_theta=500.0),
])
def test_model_variants_match_reference(overrides):
    m = tiny_model(6, **overrides)
    e = make_engine(m)
    p = list(range(5, 45))
    o = e.generate([p], [sp(max_tokens=16)])[0]
    assert o.token_ids == ref_tokens(m, p, 16)


def test_sampling_seed_determinism_and_diversity():
    m = tiny_model(7)
    e = make_engine(m)
    p = PROMPTS[0]
    a = e.generate([p], [SamplingParams(temperature=1.0, max_tokens=20, seed=11, ignore_eos=True)])[0].token_ids
    b = e.generate([p], [SamplingParams(temperature=1.0, max_tokens=20, seed=11, ignore_eos=True)])[0].token_ids
    c = e.generate([p], [SamplingParams(temperature=1.0, max_tokens=20, seed=12, ignore_eos=True)])[0].token_ids
    assert a == b and a != c


def test_topk1_and_tiny_topp_equal_greedy():
    m = tiny_model(8)
    e = make_engine(m)
    p = PROMPTS[1]
    ref = ref_tokens(m, p, 12)
    for kw in (dict(top_k=1), dict(top_p=1e-6), dict(min_p=0.999999)):
        o = e.generate([p], [SamplingParams(temperature=1.0, max_tokens=12, seed=1, ignore_eos=True, **kw)])[0]
        assert o.token_ids == ref, kw


def test_repetition_and_frequency_penalties_change_output():
    m = tiny_model(9, tie_word_embeddings=True)  # tied model repeats a lot -> penalties matter
    e = make_engine(m)
    p = PROMPTS[0]
    base = e.generate([p], [sp(max_tokens=24)])[0].token_ids
    pen = e.generate([p], [sp(max_tokens=24, repetition_penalty=1.8, frequency_penalty=1.0, presence_penalty=0.5)])[0].token_ids
    assert base != pen
    assert len(set(pen)) >= len(set(base))


def test_logit_bias_forces_token():
    m = tiny_model(10)
    e = make_engine(m)
    o = e.generate([PROMPTS[0]], [sp(max_tokens=5, logit_bias={77: 100.0})])[0]
    assert o.token_ids == [77] * 5


def test_stop_token_ids_eos_and_max_tokens():
    m = tiny_model(11)
    e = make_engine(m)
    p = PROMPTS[0]
    ref = ref_tokens(m, p, 10)
    o = e.generate([p], [sp(max_tokens=10, stop_token_ids=[ref[3]])])[0]
    assert o.token_ids == ref[:4] and o.finish_reason == "stop"
    # eos suppressed by min_tokens
    o = e.generate([p], [SamplingParams(temperature=0, max_tokens=6, min_tokens=6, logit_bias={EOS: 100.0})])[0]
    assert EOS not in o.token_ids and len(o.token_ids) == 6
    # eos stops generation when not suppressed
    o = e.generate([p], [SamplingParams(temperature=0, max_tokens=6, logit_bias={EOS: 100.0})])[0]
    assert o.token_ids == [EOS] and o.finish_reason == "stop" and o.text == ""


def test_stop_strings_and_streaming_deltas():
    m = tiny_model(12)
    tok = ByteTokenizer()
    e = make_engine(m, tokenizer=tok)
    p = PROMPTS[0]
    full = tok.decode(ref_tokens(m, p, 30))
    stop = full[9:11]
    assert stop and full.find(stop) >= 0
    cut = full[: full.find(stop)]
    e.add_request(p, sp(max_tokens=30, stop=[stop]), "r1")
    deltas, final = [], None
    while e.has_unfinished():
        for o in e.step():
            deltas.append(o.new_text)
            if o.finished:
                final = o
    assert final.finish_reason == "stop"
    assert final.text == cut and "".join(deltas) == cut


def test_logprobs():
    m = tiny_model(13)
    e = make_engine(m)
    p = PROMPTS[1]
    o = e.generate([p], [sp(max_tokens=3, logprobs=2)])[0]
    assert len(o.logprobs) == 3
    ref_logits = m.logits_for(p)[-1]
    lp = torch.log_softmax(ref_logits, -1)
    assert o.logprobs[0]["token_id"] == o.token_ids[0]
    assert abs(o.logprobs[0]["logprob"] - float(lp[o.token_ids[0]])) < 1e-3
    assert len(o.logprobs[0]["top"]) == 2 and o.logprobs[0]["top"][0][0] == int(lp.argmax())


def test_n_greater_than_one():
    m = tiny_model(14)
    e = make_engine(m)
    rid = e.add_request(PROMPTS[0], SamplingParams(temperature=1.0, max_tokens=8, n=3, seed=5, ignore_eos=True))
    finals = {}
    while e.has_unfinished():
        for o in e.step():
            assert o.request_id == rid
            if o.finished:
                finals[o.index] = o.token_ids
    assert sorted(finals) == [0, 1, 2]
    assert len({tuple(v) for v in finals.values()}) >= 2


def test_abort_waiting_and_running():
    m = tiny_model(15)
    e = make_engine(m, max_num_seqs=1)
    e.add_request(PROMPTS[0], sp(max_tokens=50), "a")
    e.add_request(PROMPTS[1], sp(max_tokens=50), "b")
    e.step()
    e.step()
    e.abort("b")  # waiting
    outs = e.step()
    assert any(o.request_id == "b" and o.finished and o.finish_reason == "abort" for o in outs)
    e.abort("a")  # running
    outs = e.step()
    assert any(o.request_id == "a" and o.finish_reason == "abort" for o in outs)
    assert not e.has_unfinished()
    st = e.manager.stats()
    assert st["blocks_in_use"] == 0


def test_preemption_recompute_preserves_outputs():
    m = tiny_model(16)
    # pool too small for all sequences at once -> forced preemptions
    e = make_engine(m, num_gpu_blocks=14, max_num_seqs=4, max_model_len=64, preemption_mode="recompute")
    prompts = [list(range(i, i + 9)) for i in (3, 40, 80, 120)]
    outs = e.generate(prompts, [sp(max_tokens=14) for _ in prompts])
    for p, o in zip(prompts, outs):
        assert o.token_ids == ref_tokens(m, p, 14)
    assert e.stats()["preemptions"] > 0


def test_preemption_swap_preserves_outputs():
    m = tiny_model(17)
    e = make_engine(m, num_gpu_blocks=14, max_num_seqs=4, max_model_len=64, preemption_mode="swap", host_kv_gb=0.0002)
    prompts = [list(range(i, i + 9)) for i in (3, 40, 80, 120)]
    outs = e.generate(prompts, [sp(max_tokens=14) for _ in prompts])
    for p, o in zip(prompts, outs):
        assert o.token_ids == ref_tokens(m, p, 14)
    assert e.stats()["preemptions"] > 0
    assert e.scheduler.total_swaps > 0


def test_priority_policy_orders_admission():
    m = tiny_model(18)
    e = make_engine(m, max_num_seqs=1, policy="priority")
    e.add_request(PROMPTS[0], sp(max_tokens=3), "low", priority=5)
    e.add_request(PROMPTS[1], sp(max_tokens=3), "high", priority=0)
    order = []
    while e.has_unfinished():
        for o in e.step():
            if o.finished:
                order.append(o.request_id)
    assert order == ["high", "low"]


def test_priority_aging_prevents_starvation():
    from forge.serve.kv import KVBlockManager
    from forge.serve.scheduler import Request, Scheduler

    t = [0.0]
    mgr = KVBlockManager(32, 4)
    s = Scheduler(mgr, max_num_seqs=1, max_num_batched_tokens=32, max_model_len=64, policy="priority", priority_aging_s=10.0,
                  clock=lambda: t[0])
    old_low = Request("old", [1, 2, 3], SamplingParams(), priority=5, arrival_time=0.0)
    new_high = Request("new", [4, 5, 6], SamplingParams(), priority=0, arrival_time=100.0)
    t[0] = 100.0
    assert s._key(old_low) < s._key(new_high)  # 100 s of waiting = 10 priority levels of aging


def test_admission_limits_and_errors():
    m = tiny_model(19)
    e = make_engine(m, max_model_len=32)
    from forge.core.errors import ConfigError

    with pytest.raises(ConfigError):
        e.add_request([], sp())
    with pytest.raises(ConfigError):
        e.add_request(list(range(40)), sp())
    with pytest.raises(ConfigError):
        e.add_request([999], sp())
    e.add_request([1, 2, 3], sp(max_tokens=2), "dup")
    with pytest.raises(ConfigError):
        e.add_request([1, 2, 3], sp(max_tokens=2), "dup")


def test_max_model_len_truncates_generation():
    m = tiny_model(20)
    e = make_engine(m, max_model_len=24)
    o = e.generate([list(range(1, 21))], [sp(max_tokens=100)])[0]
    assert len(o.token_ids) == 4 and o.finish_reason == "length"


def test_kv_fp8_and_int8_close_to_reference():
    from forge.serve.runner import SeqInput

    m = tiny_model(21)
    p = list(range(10, 60))
    ref_last = m.logits_for(p)[-1]
    for kvd, tol in (("fp8_e4m3", 0.6), ("int8", 0.05)):
        e = make_engine(m, kv_dtype=kvd)
        e.manager.allocate("x", p)
        out = e.runner.forward([SeqInput(p, 0, e.manager.block_table("x"))])
        err = (out.logits[0][-1] - ref_last).abs().max().item()
        assert err < tol, (kvd, err)
        # and the engine runs end-to-end with this dtype
        o = e.generate([p], [sp(max_tokens=6)])[0]
        assert len(o.token_ids) == 6


def test_stats_shape():
    m = tiny_model(22)
    e = make_engine(m)
    e.generate([PROMPTS[0], PROMPTS[1]], [sp(max_tokens=6)] * 2)
    st = e.stats()
    for k in ("running", "waiting", "kv_usage", "prefix_hit_rate", "tokens_per_s", "ttft_ms_p50", "itl_ms_p50", "gpu_mem_gb",
              "spec_accept_rate", "kv"):
        assert k in st
    assert st["ttft_ms_p50"] is not None and st["total_generated_tokens"] == 12
    assert st["kv"]["bytes_by_tier"]["hbm_cached"] >= 0


def test_cuda_graphs_flag_is_harmless_on_cpu():
    m = tiny_model(23)
    e = make_engine(m, cuda_graphs=True, kernel_attention="auto")
    o = e.generate([PROMPTS[0]], [sp(max_tokens=5)])[0]
    assert o.token_ids == ref_tokens(m, PROMPTS[0], 5)
    assert e.runner.stats["graph_replays"] == 0


def test_cow_fork_beam_style_blocks_stay_consistent():
    """Manager-level fork + runner write to the shared partial block must not corrupt the sibling (copy-on-write)."""
    from forge.serve.runner import SeqInput

    m = tiny_model(24)
    e = make_engine(m)
    p = list(range(1, 11))  # 10 tokens -> last block partial (2/4)
    e.manager.allocate("a", p)
    e.runner.forward([SeqInput(p, 0, e.manager.block_table("a"), 0)])
    e.manager.fork("a", "b")
    copies_b = e.manager.reserve("b", 11)
    assert copies_b
    e.runner.copy_blocks(copies_b)
    ob = e.runner.forward([SeqInput([77], 10, e.manager.block_table("b"))])
    assert e.manager.reserve("a", 11) == []
    oa = e.runner.forward([SeqInput([99], 10, e.manager.block_table("a"))])
    assert torch.allclose(ob.logits[0][-1], m.logits_for(p + [77])[-1], atol=1e-4)
    assert torch.allclose(oa.logits[0][-1], m.logits_for(p + [99])[-1], atol=1e-4)


def test_tied_head_without_lm_head_module():
    m = tiny_model(25, tie_word_embeddings=True)
    ref = ref_tokens(m, PROMPTS[0], 8)
    m.lm_head = None  # runner must fall back to embed_tokens.weight
    e = make_engine(m)
    o = e.generate([PROMPTS[0]], [sp(max_tokens=8)])[0]
    assert o.token_ids == ref


def test_batched_greedy_fast_path_mixed_with_sampling_rows():
    m = tiny_model(26)
    e = make_engine(m)
    prm = [sp(max_tokens=10), SamplingParams(temperature=1.0, max_tokens=10, seed=3, ignore_eos=True), sp(max_tokens=10)]
    outs = e.generate([PROMPTS[0], PROMPTS[1], PROMPTS[3]], prm)
    assert outs[0].token_ids == ref_tokens(m, PROMPTS[0], 10)
    assert outs[2].token_ids == ref_tokens(m, PROMPTS[3], 10)
    assert len(outs[1].token_ids) == 10


def test_sequence_outgrowing_the_whole_kv_pool_finishes_instead_of_hanging():
    m = tiny_model(27)
    e = make_engine(m, num_gpu_blocks=6, max_model_len=64, max_num_seqs=2)   # 24 tokens of KV in total
    e.add_request(list(range(1, 11)), sp(max_tokens=50), "big")
    outs, steps = [], 0
    while e.has_unfinished() and steps < 200:
        outs += e.step()
        steps += 1
    assert not e.has_unfinished(), "engine livelocked on an unfittable sequence"
    fin = [o for o in outs if o.finished][0]
    assert fin.finish_reason == "length" and 0 < len(fin.token_ids) < 50
    assert fin.token_ids == ref_tokens(m, list(range(1, 11)), len(fin.token_ids))
    assert e.manager.stats()["blocks_in_use"] == 0
