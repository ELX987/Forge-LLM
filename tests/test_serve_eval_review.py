"""Regression tests written by the adversarial review of forge/serve, forge/eval and forge/planner.

Each test demonstrates a bug that existed before the fix (see the review report)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from forge.serve.kv import KVBlockManager

BS = 4


class FakeStorage:
    def __init__(self):
        self.data = {}

    def read_block(self, bid):
        return ("p", self.data.get(bid))

    def write_block(self, bid, payload):
        self.data[bid] = payload[1]

    def copy_block(self, s, d):
        self.data[d] = self.data.get(s)


# ---------------------------------------------------------------------------------------------- KV manager
def test_rollback_below_duplicate_blocks_keeps_correct_hash_chain():
    """A sequence whose leading blocks duplicate already-cached content (so its own blocks are unregistered) must,
    after rollback + regrow, register new blocks under the *correct* parent chain, never under the root."""
    m = KVBlockManager(16, BS)
    x = list(range(12))
    m.allocate("c", x)            # no cache yet -> own blocks
    m.allocate("a", x)            # also own blocks (same content, concurrently admitted)
    m.commit_tokens("a", x)       # a registers the chain
    m.commit_tokens("c", x)       # c finds duplicates: its blocks stay unregistered but its chain pointer follows a's nodes
    m.rollback("c", 6)            # drop below the hashed prefix
    new = x[:6] + [100, 101, 102, 103, 104, 105]   # block 1 now holds different tokens (4,5,100,101)
    assert m.reserve("c", 12) is not None
    m.commit_tokens("c", new)
    # a fresh sequence starting with c's *block 1 content* must NOT hit a cached block: that block's KV depends on block 0.
    probe = [4, 5, 100, 101, 7, 7, 7, 7, 7]
    hit = m.peek_prefix(probe)
    assert hit == 0, "block registered under the wrong parent chain (KV aliasing)"
    # and the legitimate full prefix still hits
    assert m.peek_prefix(new + [9]) >= 8


def test_promote_does_not_evict_the_node_being_promoted(tmp_path):
    """With a tiny host tier, promoting a host node used to evict that very node while making room -> KeyError."""
    st = FakeStorage()
    a = list(range(8))
    b = list(range(100, 108))
    m = KVBlockManager(2, BS, host_blocks=1, storage=st, bytes_per_block=8)
    assert m.allocate("a", a) == 0
    for bl in m.block_table("a"):
        st.data[bl] = "A"
    m.commit_tokens("a", a)
    m.free("a")
    assert m.allocate("b", b) == 0       # evicts/demotes a's blocks
    m.commit_tokens("b", b)
    m.free("b")
    got = m.allocate("a2", a + [1])      # needs a's (possibly lower-tier) prefix back; must not raise
    assert got is None or got in (0, 4, 8)


def test_reserve_is_atomic_when_cached_free_blocks_are_not_evictable():
    m = KVBlockManager(5, BS)
    x = list(range(8))
    assert m.allocate("c", x) == 0
    assert m.allocate("a", x) == 0
    m.commit_tokens("a", x)
    m.commit_tokens("c", x)
    assert m.reserve("c", 12) is not None
    m.commit_tokens("c", x + [50, 51, 52, 53])   # c2 registered as a child of a's chain
    m.free("a")                                   # a's nodes become cached-free but are pinned by the child node
    assert m.pool.num_free == 0
    before_blocks = m.block_table("c")
    before_slots = m.num_slots("c")
    assert m.reserve("c", 16) is None             # must report "no memory", not raise
    assert m.block_table("c") == before_blocks and m.num_slots("c") == before_slots


def test_rollback_refusal_leaves_state_untouched():
    m = KVBlockManager(8, BS)
    x = list(range(8))
    m.allocate("p", x + [1])
    m.commit_tokens("p", x + [1])
    m.fork("p", "q")
    snap_nodes = m.stats()["num_radix_nodes"]
    with pytest.raises(Exception):
        m.rollback("q", 2)
    assert m.stats()["num_radix_nodes"] == snap_nodes
    assert m.num_slots("q") == 9
    # both sequences can still be freed cleanly
    m.free("p")
    m.free("q")
    st = m.stats()
    assert st["blocks_in_use"] == 0


# ---------------------------------------------------------------------------------------------- stats
def test_t_quantile_is_exact_for_small_df():
    from forge.eval.stats import t_quantile

    exact = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 9: 2.2622, 30: 2.0423}
    for df, v in exact.items():
        assert t_quantile(0.975, df) == pytest.approx(v, rel=2e-3), df
    assert t_quantile(0.5, 7) == pytest.approx(0.0, abs=1e-9)
    assert t_quantile(0.025, 3) == pytest.approx(-3.1824, rel=2e-3)
    assert t_quantile(0.95, 4) == pytest.approx(2.1318, rel=2e-3)


def test_noninferiority_lower_is_better_reports_bounds_in_original_units():
    from forge.eval.stats import noninferiority_paired

    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 1.0, 400)
    b = a + 0.1 + rng.normal(0, 0.05, 400)      # candidate slightly *higher* (worse for lower-is-better)
    r = noninferiority_paired(a, b, margin=0.5, higher_is_better=False)
    assert r.diff == pytest.approx(0.1, abs=0.02)
    assert r.lower <= r.diff <= r.upper
    assert r.lower > 0 or r.lower == pytest.approx(0.1, abs=0.05)
    assert r.noninferior


# ---------------------------------------------------------------------------------------------- metrics
def test_kl_and_entropy_with_masked_vocab_are_finite():
    import torch

    from forge.eval.metrics import entropy_shift, kl_divergence

    p = torch.randn(5, 12)
    p[:, 8:] = float("-inf")     # padded vocabulary rows masked by -inf (common in HF checkpoints)
    c = p + 0.05 * torch.randn(5, 12)
    c[:, 8:] = float("-inf")
    s = kl_divergence(p, c)
    assert math.isfinite(s.mean) and math.isfinite(s.p99) and s.mean >= 0
    assert math.isfinite(entropy_shift(p, c))
    # candidate puts mass where the parent has none: KL(parent||cand) is still finite (parent mass is zero there)
    c2 = p.clone()
    c2[:, 9] = 0.0
    assert math.isfinite(kl_divergence(p, c2).mean)
    # parent mass where candidate has none: genuinely infinite KL
    p3 = torch.randn(3, 6)
    c3 = p3.clone()
    c3[:, 2] = float("-inf")
    assert kl_divergence(p3, c3).mean == float("inf")


# ---------------------------------------------------------------------------------------------- scheduler
def _mk_sched(policy="fcfs", aging=0.0, blocks=4, clock=None):
    from forge.serve.scheduler import Scheduler

    m = KVBlockManager(blocks, BS, enable_prefix_caching=False)
    kw = {} if clock is None else {"clock": clock}
    s = Scheduler(m, max_num_seqs=8, max_num_batched_tokens=64, max_model_len=64, policy=policy, priority_aging_s=aging,
                  prefix_caching=False, **kw)
    return m, s


def _finish_prefill(s, reqs):
    for r in reqs:
        r.num_computed = len(r.prompt_ids)
        r.tokens.append(1)  # first sampled token -> next step is a decode step (pending == 1)


def _req(rid, n=8, priority=0, arrival=0.0):
    from forge.serve.sampling import SamplingParams
    from forge.serve.scheduler import Request

    return Request(rid, list(range(100 * len(rid), 100 * len(rid) + n)), SamplingParams(temperature=0, max_tokens=8),
                   priority=priority, arrival_time=arrival)


def test_scheduler_preempts_latest_arrival_under_fcfs_and_keeps_its_arrival_time():
    m, s = _mk_sched(blocks=5)
    a, b = _req("a", arrival=0.0), _req("b", arrival=1.0)
    s.add(a)
    s.add(b)
    out = s.schedule()
    assert {x.req.request_id for x in out.scheduled} == {"a", "b"}
    _finish_prefill(s, [a, b])
    # grow both until the pool (5 blocks) cannot hold 2 x 3 blocks
    for r in (a, b):
        r.tokens.extend([2] * 3)
        r.num_computed += 3
    a.tokens.append(3)
    b.tokens.append(3)
    a.num_computed, b.num_computed = len(a.tokens) - 1, len(b.tokens) - 1   # 12 tokens each in KV -> next token needs block 4
    for r in (a, b):
        m.reserve(r.request_id, r.num_computed)
    out = s.schedule()
    ids = [x.req.request_id for x in out.scheduled]
    assert "a" in ids and "b" not in ids
    assert [r.request_id for r in out.preempted] == ["b"]
    assert b.status.value == "waiting" and b.arrival_time == 1.0 and b.num_computed == 0
    assert not m.has_seq("b")


def test_scheduler_priority_policy_preempts_low_priority_not_latecomer():
    m, s = _mk_sched(policy="priority", blocks=5)
    a, b = _req("a", priority=5, arrival=0.0), _req("b", priority=0, arrival=1.0)   # b is more important but arrived later
    s.add(a)
    s.add(b)
    s.schedule()
    _finish_prefill(s, [a, b])
    for r in (a, b):
        r.tokens.extend([2] * 3)
        r.tokens.append(3)
        r.num_computed = len(r.tokens) - 1
        m.reserve(r.request_id, r.num_computed)
    out = s.schedule()
    assert [r.request_id for r in out.preempted] == ["a"]
    assert [x.req.request_id for x in out.scheduled] == ["b"]


def test_scheduler_aging_lets_old_low_priority_request_overtake():
    t = [0.0]
    m, s = _mk_sched(policy="priority", aging=1.0, blocks=8, clock=lambda: t[0])
    old_low = _req("old", priority=10, arrival=0.0)
    new_high = _req("new", priority=0, arrival=100.0)
    t[0] = 100.0
    # without aging the new high-priority request sorts first; after 100 s of waiting the old one has aged by 100 levels
    assert s._key(old_low, 100.0) < s._key(new_high, 100.0)
    s.aging = 0.0
    assert s._key(old_low, 100.0) > s._key(new_high, 100.0)


# ---------------------------------------------------------------------------------------------- server hardening
def test_server_rejects_absurd_n_and_bad_priority_and_checks_api_key():
    from fastapi.testclient import TestClient

    from forge.serve._testing import make_engine, tiny_model
    from forge.serve.server import create_app

    eng = make_engine(tiny_model(1))
    app = create_app(eng, api_key="s3cret", model_name="t")
    try:
        with TestClient(app) as c:
            h = {"Authorization": "Bearer s3cret"}
            assert c.get("/stats").status_code == 401
            assert c.get("/stats", headers={"Authorization": "Bearer s3cre"}).status_code == 401
            assert c.get("/stats", headers={"Authorization": "Bearer s3cret"}).status_code == 200
            r = c.post("/v1/completions", headers=h, json={"prompt": [1, 2, 3], "max_tokens": 2, "n": 100000})
            assert r.status_code == 400 and "n must be between" in r.json()["error"]["message"]
            r = c.post("/v1/completions", headers=h, json={"prompt": [1, 2, 3], "max_tokens": 2, "priority": "high"})
            assert r.status_code == 400
            r = c.post("/v1/completions", headers=h, json={"prompt": [1, 2, 3], "max_tokens": 2, "n": 2, "temperature": 0.7})
            assert r.status_code == 200 and len(r.json()["choices"]) == 2
    finally:
        app.state.loop.stop()


def test_seeded_sampling_is_invariant_to_batching_and_recompute_preemption():
    import random

    from forge.serve._testing import make_engine, tiny_model
    from forge.serve.sampling import SamplingParams

    m = tiny_model(31, tie_word_embeddings=True)
    rng = random.Random(1)
    prompts = [[rng.randint(1, 200) for _ in range(rng.randint(5, 20))] for _ in range(8)]

    def run(**kw):
        e = make_engine(m, **kw)
        prm = [SamplingParams(temperature=0.9, top_p=0.9, top_k=40, max_tokens=16, seed=100 + i, ignore_eos=True) for i in range(len(prompts))]
        return [o.token_ids for o in e.generate(prompts, prm)], e.stats()["preemptions"]

    roomy, _ = run(max_num_seqs=16, max_model_len=96)
    tight, pre = run(num_gpu_blocks=14, max_num_seqs=6, max_model_len=96, max_num_batched_tokens=10)
    assert pre > 0, "scenario must actually preempt"
    assert roomy == tight
