"""Randomised invariants for the KV manager (with tiers) and end-to-end engine stress with preemption + caching + speculation."""
from __future__ import annotations

import random

import pytest

from forge.serve._testing import make_engine, tiny_model
from forge.serve.kv import KVBlockManager
from forge.serve.sampling import SamplingParams

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


def check_invariants(m: KVBlockManager):
    pool = m.pool
    refs = [0] * pool.num_blocks
    for st in m._seqs.values():
        for b in st.blocks:
            refs[b] += 1
    assert refs == pool.ref_count, "refcounts diverged from sequence block tables"
    free = set(pool._free)
    cached = m._cached_free
    inuse = {i for i, r in enumerate(refs) if r > 0}
    assert not (free & cached) and not (free & inuse) and not (cached & inuse)
    assert len(free) + len(cached) + len(inuse) == pool.num_blocks, "leaked blocks"
    for b in cached:
        assert b in m._block_node and refs[b] == 0
    for b, node in m._block_node.items():
        assert node.block_id == b and node.tier == 0
    for key, node in m._nodes.items():
        assert node.key == key
        if node.parent is not None and node.parent.parent is not None:
            assert node.parent.children.get(key) is node
        n_hbm = sum(1 for c in node.children.values() if c.tier == 0)
        assert node.hbm_children == n_hbm, "hbm_children counter drifted"
    if m.host is not None:
        host_nodes = [n for n in m._nodes.values() if n.tier == 1]
        assert len(host_nodes) == len(m.host) <= m.host.capacity
    if m.nvme is not None:
        nv = [n for n in m._nodes.values() if n.tier == 2]
        assert len(nv) == len(m.nvme) <= m.nvme.capacity


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_kv_manager_random_ops_keep_invariants(seed, tmp_path):
    rng = random.Random(seed)
    stor = FakeStorage()
    m = KVBlockManager(24, BS, host_blocks=6, nvme_dir=str(tmp_path / f"nv{seed}"), nvme_blocks=6, swap_blocks=8, storage=stor,
                       bytes_per_block=64)
    seqs: dict[str, list[int]] = {}
    swapped: dict[str, list[int]] = {}
    vocab_pool = [list(range(i, i + 40)) for i in (0, 100, 200)]
    n = 0
    for _ in range(400):
        op = rng.choice(["alloc", "alloc", "grow", "free", "fork", "rollback", "swap", "unswap", "demote"])
        if op == "alloc":
            base = rng.choice(vocab_pool)[: rng.randint(3, 30)]
            toks = base + [rng.randint(300, 400) for _ in range(rng.randint(0, 5))]
            sid = f"s{n}"
            n += 1
            got = m.allocate(sid, toks)
            if got is not None:
                for b in m.block_table(sid)[got // BS:]:
                    stor.data[b] = ("seq", sid)
                seqs[sid] = list(toks)
                m.commit_tokens(sid, toks)
        elif op == "grow" and seqs:
            sid = rng.choice(list(seqs))
            k = rng.randint(1, 6)
            tot = m.num_slots(sid) + k
            cp = m.reserve(sid, tot)
            if cp is not None:
                for s_, d_ in cp:
                    stor.copy_block(s_, d_)
                seqs[sid] = seqs[sid] + [rng.randint(0, 250) for _ in range(tot - len(seqs[sid]))]
                if rng.random() < 0.7:
                    m.commit_tokens(sid, seqs[sid][: m.num_slots(sid)])
        elif op == "free" and seqs:
            sid = rng.choice(list(seqs))
            m.free(sid)
            seqs.pop(sid)
        elif op == "fork" and seqs:
            sid = rng.choice(list(seqs))
            child = f"s{n}"
            n += 1
            m.fork(sid, child)
            seqs[child] = list(seqs[sid])
        elif op == "rollback" and seqs:
            sid = rng.choice(list(seqs))
            cur = m.num_slots(sid)
            if cur > 1:
                try:
                    m.rollback(sid, rng.randint(1, cur))
                except Exception as e:  # rolling back into a shared registered block is refused by design
                    assert "shared registered" in str(e)
                else:
                    seqs[sid] = seqs[sid][: m.num_slots(sid)]
        elif op == "swap" and seqs:
            sid = rng.choice(list(seqs))
            if m.swap_out(sid):
                swapped[sid] = seqs.pop(sid)
        elif op == "unswap" and swapped:
            sid = rng.choice(list(swapped))
            toks = swapped[sid]
            if m.swap_in(sid, toks[: m.swapped_num_blocks(sid) * BS]):
                seqs[sid] = swapped.pop(sid)
        elif op == "demote":
            m.demote_cold(rng.randint(1, 4))
        check_invariants(m)
    for sid in list(seqs):
        m.free(sid)
    for sid in list(swapped):
        m.drop_swapped(sid)
    check_invariants(m)
    st = m.stats()
    assert st["blocks_in_use"] == 0 and st["blocks_free"] + st["blocks_cached_free"] == 24
    m.shutdown()


def test_prefix_hit_content_integrity_across_tiers(tmp_path):
    """Whatever tier a cached block lives in, a hit must return exactly the content that was stored."""
    stor = FakeStorage()
    m = KVBlockManager(10, BS, host_blocks=3, nvme_dir=str(tmp_path / "nv"), nvme_blocks=10, storage=stor)
    prompts = [[i * 50 + j for j in range(13)] for i in range(5)]
    for i, p in enumerate(prompts):
        sid = f"p{i}"
        assert m.allocate(sid, p) == 0
        for b in m.block_table(sid):
            stor.data[b] = (i, "blk")
        # tag per-block content by index so mixups are detectable
        for idx, b in enumerate(m.block_table(sid)):
            stor.data[b] = (i, idx)
        m.commit_tokens(sid, p)
        m.free(sid)
    for i, p in enumerate(prompts):
        sid = f"q{i}"
        got = m.allocate(sid, p)
        assert got is not None
        for idx, b in enumerate(m.block_table(sid)[: got // BS]):
            assert stor.data[b] == (i, idx), (i, idx, stor.data[b])
        m.free(sid)
        check_invariants(m)


@pytest.mark.parametrize("spec", [None, dict(mode="ngram", max_gamma=3, controller="fixed")])
def test_engine_stress_matches_reference(spec):
    rng = random.Random(5)
    m = tiny_model(31, tie_word_embeddings=True)
    kw = dict(num_gpu_blocks=26, max_num_seqs=5, max_model_len=96, max_num_batched_tokens=12, preemption_mode="swap",
              host_kv_gb=0.0003)
    if spec:
        kw["speculation"] = spec
    e = make_engine(m, **kw)
    shared = list(range(20, 36))
    prompts = []
    for i in range(14):
        base = shared if rng.random() < 0.6 else list(range(100 + i * 7, 116 + i * 7))
        prompts.append(base[: rng.randint(4, 16)] + [rng.randint(1, 200) for _ in range(rng.randint(0, 6))])
    prm = [SamplingParams(temperature=0, max_tokens=rng.randint(3, 16), ignore_eos=True) for _ in prompts]
    outs = e.generate(prompts, prm)
    for p, pr, o in zip(prompts, prm, outs):
        assert o.token_ids == m.greedy_generate(p, pr.max_tokens, eos=None)
    check_invariants(e.manager)
    st = e.manager.stats()
    assert st["blocks_in_use"] == 0
    assert e.stats()["total_finished"] == len(prompts)
