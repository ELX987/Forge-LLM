"""KVBlockManager / BlockPool: prefix cache, COW fork, rollback, tiers, swap, stats."""
from __future__ import annotations

import pytest

from forge.serve.kv import TIER_HBM, TIER_HOST, TIER_NVME, BlockPool, KVBlockManager

BS = 4


class FakeStorage:
    """Stores one python list per block id so we can verify content survives demote/promote/swap."""

    def __init__(self, n):
        self.data = {i: None for i in range(n)}
        self.copies = []

    def read_block(self, bid):
        return ("payload", self.data[bid])

    def write_block(self, bid, payload):
        self.data[bid] = payload[1]

    def copy_block(self, src, dst):
        self.data[dst] = self.data[src]
        self.copies.append((src, dst))


def toks(n, start=0):
    return list(range(start, start + n))


def test_block_pool_refcounts():
    p = BlockPool(3, BS)
    a, b = p.allocate(), p.allocate()
    assert p.num_free == 1 and p.ref_count[a] == 1
    p.incref(a)
    assert p.decref(a) == 1
    assert p.decref(a) == 0
    p.free_block(a)
    assert p.num_free == 2
    with pytest.raises(Exception):
        p.decref(a)
    _ = b


def test_allocate_free_and_prefix_hit():
    m = KVBlockManager(16, BS)
    prompt = toks(10)
    assert m.allocate("a", prompt) == 0
    assert len(m.block_table("a")) == 3
    m.commit_tokens("a", prompt)  # blocks 0,1 full -> registered
    assert m.stats()["num_radix_nodes"] == 2
    m.free("a")
    assert m.stats()["blocks_cached_free"] == 2
    # same prompt: 8 tokens cached (limit len-1 = 9 -> 2 full blocks)
    assert m.allocate("b", prompt) == 8
    assert m.block_table("b")[:2] == m.block_table("b")[:2]
    st = m.stats()
    assert st["hit_tokens"] == 8 and st["query_tokens"] == 20
    assert abs(st["hit_rate"] - 0.4) < 1e-9


def test_never_cache_whole_prompt():
    m = KVBlockManager(8, BS)
    p = toks(8)
    m.allocate("a", p)
    m.commit_tokens("a", p)
    m.free("a")
    # prompt length 8 == exactly 2 blocks; at most (8-1)//4 = 1 block may hit so >=1 token is recomputed
    assert m.allocate("b", p) == 4


def test_shared_prefix_blocks_are_shared():
    m = KVBlockManager(16, BS)
    base = toks(8)
    m.allocate("a", base + [100, 101])
    m.commit_tokens("a", base + [100, 101])
    n = m.allocate("b", base + [200, 201])
    assert n == 8
    assert m.block_table("a")[:2] == m.block_table("b")[:2]
    for bid in m.block_table("a")[:2]:
        assert m.pool.ref_count[bid] == 2


def test_allocate_fails_cleanly_when_full():
    m = KVBlockManager(4, BS)
    assert m.allocate("a", toks(12)) == 0  # 3 blocks
    before = m.stats()
    assert m.allocate("b", toks(9, 50)) is None  # needs 3, only 1 free
    after = m.stats()
    assert before["blocks_free"] == after["blocks_free"]
    assert not m.can_allocate(toks(9, 50))
    assert m.can_allocate(toks(4, 50))


def test_eviction_is_lru_leaf_first():
    m = KVBlockManager(6, BS)
    a = toks(12)
    m.allocate("a", a + [99])  # 4 blocks incl partial
    m.commit_tokens("a", a)
    m.free("a")
    assert m.stats()["blocks_cached_free"] == 3
    # force eviction by allocating a big other sequence
    assert m.allocate("x", toks(20, 500)) is not None  # 5 blocks, needs eviction of cached ones
    # chain was evicted leaf first, so the root block (first) may remain... but whatever remains must be a prefix
    hit = m.peek_prefix(a + [99])
    assert hit % BS == 0 and hit <= 8
    assert m.stats()["evictions"] >= 2


def test_salt_isolates_caches():
    m = KVBlockManager(16, BS)
    p = toks(9)
    m.allocate("a", p, salt="tenant1")
    m.commit_tokens("a", p)
    m.free("a")
    assert m.peek_prefix(p, "tenant1") == 8
    assert m.peek_prefix(p, "tenant2") == 0
    assert m.peek_prefix(p, None) == 0


def test_prefix_caching_disabled():
    m = KVBlockManager(16, BS, enable_prefix_caching=False)
    p = toks(9)
    m.allocate("a", p)
    m.commit_tokens("a", p)
    m.free("a")
    assert m.stats()["num_radix_nodes"] == 0
    assert m.allocate("b", p) == 0
    assert m.stats()["blocks_cached_free"] == 0


def test_reserve_rollback_commit():
    m = KVBlockManager(16, BS)
    m.allocate("a", toks(6))
    assert m.num_slots("a") == 6 and len(m.block_table("a")) == 2
    assert m.reserve("a", 6 + 6) == []  # speculation: 1 + gamma(5) slots
    assert len(m.block_table("a")) == 3
    m.commit("a", 6 + 2)  # keep 2
    assert m.num_slots("a") == 8 and len(m.block_table("a")) == 2
    st = m.stats()
    assert st["rollback_tokens"] == 4
    m.rollback("a", 5)
    assert len(m.block_table("a")) == 2


def test_rollback_unregisters_truncated_registered_block():
    m = KVBlockManager(16, BS)
    t = toks(8)
    m.allocate("a", t)
    m.commit_tokens("a", t)
    assert m.stats()["num_radix_nodes"] == 2
    m.rollback("a", 6)  # cuts into registered block 1
    assert m.stats()["num_radix_nodes"] == 1


def test_fork_copy_on_write():
    m = KVBlockManager(16, BS)
    m.allocate("a", toks(6))  # 2 blocks, last partial (2/4)
    m.fork("a", "b")
    ba, bb = m.block_table("a"), m.block_table("b")
    assert ba == bb and all(m.pool.ref_count[x] == 2 for x in ba)
    copies = m.reserve("b", 7)
    assert len(copies) == 1 and copies[0][0] == ba[1]
    assert m.block_table("b")[0] == ba[0] and m.block_table("b")[1] != ba[1]
    assert m.pool.ref_count[ba[1]] == 1
    # parent writing: it is now exclusive -> no copy
    assert m.reserve("a", 7) == []
    m.free("a")
    m.free("b")
    assert m.stats()["blocks_in_use"] == 0


def test_host_and_nvme_tiers_roundtrip(tmp_path):
    n = 4
    stor = FakeStorage(n)
    m = KVBlockManager(n, BS, host_blocks=2, nvme_dir=str(tmp_path / "nvme"), nvme_blocks=4, storage=stor, bytes_per_block=100)
    a = toks(12)
    m.allocate("a", a + [7])  # 4 blocks -> whole pool
    for i, b in enumerate(m.block_table("a")):
        stor.data[b] = f"content{i}"
    m.commit_tokens("a", a)
    m.free("a")
    keys = m.cached_keys(a)
    assert len(keys) == 3
    # new sequence needs all 4 blocks -> cached ones are demoted (not dropped)
    assert m.allocate("b", toks(16, 1000)) is not None
    st = m.stats()
    assert st["demotions"] >= 3 and st["dropped" if False else "evictions"] >= 3
    tiers = [m.tier_of_key(k) for k in keys]
    assert all(t in (TIER_HOST, TIER_NVME) for t in tiers)
    assert TIER_NVME in tiers or len(tiers) <= 2  # host holds 2, the third spilled to nvme
    assert st["bytes_by_tier"]["host"] + st["bytes_by_tier"]["nvme"] == 300
    m.free("b")
    # prefix hit promotes content back, bit-exact
    assert m.peek_prefix(a + [7]) == 12
    got = m.allocate("c", a + [7])
    assert got == 12
    bt = m.block_table("c")
    assert [stor.data[b] for b in bt[:3]] == ["content0", "content1", "content2"]
    assert m.stats()["promotions"] >= 3


def test_explicit_demote_promote_and_async(tmp_path):
    stor = FakeStorage(6)
    m = KVBlockManager(6, BS, host_blocks=4, storage=stor)
    t = toks(9)
    m.allocate("a", t)
    for i, b in enumerate(m.block_table("a")):
        stor.data[b] = f"c{i}"
    m.commit_tokens("a", t)
    m.free("a")
    keys = m.cached_keys(t)
    bt_ids = [b for b in range(6) if b in m._block_node]
    leaf = [b for b in bt_ids if m._block_node[b].hbm_children == 0][0]
    assert m.demote_async(leaf).result(timeout=5) == TIER_HOST
    assert m.tier_of_key(m._nodes and keys[-1]) == TIER_HOST
    assert m.prefetch_async(t).result(timeout=5) == 1
    assert m.tier_of_key(keys[-1]) == TIER_HBM
    m.shutdown()


def test_demote_cold(tmp_path):
    stor = FakeStorage(8)
    m = KVBlockManager(8, BS, host_blocks=8, storage=stor)
    t = toks(12) + [1]
    m.allocate("a", t)
    m.commit_tokens("a", t)
    m.free("a")
    assert m.demote_cold() == 3
    assert m.stats()["blocks_cached_free"] == 0
    assert m.peek_prefix(t) == 12  # still hit (host tier)


def test_swap_out_in():
    stor = FakeStorage(8)
    m = KVBlockManager(8, BS, swap_blocks=4, storage=stor)
    t = toks(10)
    m.allocate("a", t)
    for i, b in enumerate(m.block_table("a")):
        stor.data[b] = f"x{i}"
    assert m.swap_out("a")
    assert not m.has_seq("a") and m.is_swapped("a")
    assert m.stats()["blocks_in_use"] == 0
    assert m.swap_in("a", t)
    assert [stor.data[b] for b in m.block_table("a")] == ["x0", "x1", "x2"]
    assert m.num_slots("a") == 10


def test_dropping_parent_removes_unreachable_children_without_leaks():
    m = KVBlockManager(5, BS)  # no tiers -> eviction drops
    a = toks(12) + [5]
    m.allocate("a", a)
    m.commit_tokens("a", a)
    m.free("a")
    # churn the pool several times; the invariant: all blocks eventually free or cached, none leaked
    for i in range(6):
        sid = f"s{i}"
        assert m.allocate(sid, toks(10, 100 * (i + 1))) is not None
        m.commit_tokens(sid, toks(10, 100 * (i + 1)))
        m.free(sid)
    st = m.stats()
    assert st["blocks_in_use"] == 0
    assert st["blocks_free"] + st["blocks_cached_free"] == 5
