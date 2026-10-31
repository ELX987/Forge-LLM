"""Paged KV-cache bookkeeping (Plan 7.5 "KV block manager", SV-1/SV-2, paper P3).

* :class:`BlockPool`        - fixed-size block allocator with reference counts (no tensors; pure bookkeeping).
* :class:`KVBlockManager`   - per-sequence block tables, **radix prefix cache keyed by a rolling block hash**,
  copy-on-write ``fork`` (beam / speculation trees), ``reserve``/``rollback``/``commit`` for speculative decoding,
  and a **HBM -> host DRAM -> NVMe** tier hierarchy with ``demote``/``promote``/``prefetch`` (synchronous core,
  ``*_async`` wrappers returning futures) plus sequence swap-out/in for preemption.

The manager never touches GPU memory itself. Data movement goes through a *storage* object implementing
``read_block(bid) -> payload``, ``write_block(bid, payload)`` and ``copy_block(src, dst)`` (``forge.serve.runner.KVPool``
implements it). Without storage, tiering and swap are disabled and eviction simply drops cached blocks.

Block-hash chain: ``key_i = H(key_{i-1}, tokens of block i)`` (blake2b-64) with a per-``salt`` root, so two sequences
share block ``i`` iff their first ``(i+1)*block_size`` tokens are identical. Token tuples are stored in the node and
compared on lookup, so a hash collision can never alias different content.
"""
from __future__ import annotations

import hashlib
import heapq
import os
import struct
import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from forge.core.errors import ForgeError

TIER_HBM, TIER_HOST, TIER_NVME = 0, 1, 2
TIER_NAMES = {0: "hbm", 1: "host", 2: "nvme"}


class OutOfBlocks(ForgeError):
    code = "out_of_blocks"


class BlockStorage(Protocol):  # pragma: no cover - protocol
    def read_block(self, bid: int) -> Any: ...
    def write_block(self, bid: int, payload: Any) -> None: ...
    def copy_block(self, src: int, dst: int) -> None: ...


def hash_block(parent_key: int, tokens: tuple[int, ...]) -> int:
    """Rolling block hash (64-bit)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(struct.pack("<q", parent_key & 0x7FFFFFFFFFFFFFFF))
    h.update(struct.pack(f"<{len(tokens)}q", *tokens))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFFFFFFFFFF


def root_key(salt: str | None) -> int:
    if not salt:
        return 0
    return int.from_bytes(hashlib.blake2b(salt.encode("utf-8"), digest_size=8).digest(), "little") & 0x7FFFFFFFFFFFFFFF


# --------------------------------------------------------------------------------------------------
# block pool
# --------------------------------------------------------------------------------------------------
class BlockPool:
    """Allocator over ``num_blocks`` ids ``0..num_blocks-1`` with reference counts."""

    def __init__(self, num_blocks: int, block_size: int, name: str = "hbm", bytes_per_block: int = 0) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.name = name
        self.bytes_per_block = int(bytes_per_block)
        self.ref_count = [0] * self.num_blocks
        self._free: deque[int] = deque(range(self.num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int | None:
        if not self._free:
            return None
        bid = self._free.popleft()
        self.ref_count[bid] = 1
        return bid

    def incref(self, bid: int) -> int:
        self.ref_count[bid] += 1
        return self.ref_count[bid]

    def decref(self, bid: int) -> int:
        if self.ref_count[bid] <= 0:
            raise ForgeError(f"double free of block {bid}")
        self.ref_count[bid] -= 1
        return self.ref_count[bid]

    def free_block(self, bid: int) -> None:
        """Return a ref==0 block to the free list."""
        if self.ref_count[bid] != 0:
            raise ForgeError(f"free_block on referenced block {bid}")
        self._free.append(bid)


# --------------------------------------------------------------------------------------------------
# lower tiers
# --------------------------------------------------------------------------------------------------
class HostTier:
    """Host-DRAM payload store (slot -> payload)."""

    name = "host"

    def __init__(self, capacity_blocks: int) -> None:
        self.capacity = int(capacity_blocks)
        self._data: dict[int, Any] = {}
        self._next = 0
        self.lru: OrderedDict[int, None] = OrderedDict()  # node key -> None (oldest first)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def full(self) -> bool:
        return len(self._data) >= self.capacity

    def put(self, payload: Any) -> int:
        slot = self._next
        self._next += 1
        self._data[slot] = payload
        return slot

    def get(self, slot: int) -> Any:
        return self._data[slot]

    def remove(self, slot: int) -> None:
        self._data.pop(slot, None)


class NvmeTier(HostTier):
    """File-backed payload store (one ``torch.save`` file per block)."""

    name = "nvme"

    def __init__(self, directory: str, capacity_blocks: int) -> None:
        super().__init__(capacity_blocks)
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        self._slots: set[int] = set()

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def full(self) -> bool:
        return len(self._slots) >= self.capacity

    def _path(self, slot: int) -> str:
        return os.path.join(self.dir, f"kvblock_{slot}.pt")

    def put(self, payload: Any) -> int:
        import torch

        slot = self._next
        self._next += 1
        torch.save(payload, self._path(slot))
        self._slots.add(slot)
        return slot

    def get(self, slot: int) -> Any:
        import torch

        return torch.load(self._path(slot), map_location="cpu", weights_only=True)  # never unpickle arbitrary objects from disk

    def remove(self, slot: int) -> None:
        if slot in self._slots:
            self._slots.discard(slot)
            try:
                os.remove(self._path(slot))
            except OSError:
                pass


# --------------------------------------------------------------------------------------------------
# radix tree
# --------------------------------------------------------------------------------------------------
class RadixNode:
    __slots__ = ("key", "parent", "tokens", "children", "tier", "block_id", "slot", "last_used", "hbm_children", "version")

    def __init__(self, key: int, parent: RadixNode | None, tokens: tuple[int, ...]) -> None:
        self.key = key
        self.parent = parent
        self.tokens = tokens
        self.children: dict[int, RadixNode] = {}
        self.tier = TIER_HBM
        self.block_id = -1
        self.slot = -1
        self.last_used = 0
        self.hbm_children = 0
        self.version = 0


@dataclass
class SeqBlocks:
    blocks: list[int] = field(default_factory=list)
    num_slots: int = 0
    hashed_blocks: int = 0
    parent_key: int = 0
    salt: str | None = None
    cached_tokens: int = 0
    hash_disabled: bool = False
    chain: list[int] = field(default_factory=list)  # radix keys of blocks [0, hashed_blocks) along THIS sequence's prefix


@dataclass
class _Swapped:
    payloads: list[Any]
    num_slots: int
    salt: str | None


class KVBlockManager:
    """See module docstring. All public methods are thread-safe (single re-entrant lock)."""

    def __init__(self, num_blocks: int, block_size: int, *, enable_prefix_caching: bool = True,
                 host_blocks: int = 0, nvme_dir: str | None = None, nvme_blocks: int = 0,
                 swap_blocks: int = 0, bytes_per_block: int = 0, storage: BlockStorage | None = None) -> None:
        self.block_size = int(block_size)
        self.pool = BlockPool(num_blocks, block_size, "hbm", bytes_per_block)
        self.bytes_per_block = int(bytes_per_block)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.storage = storage
        self.host = HostTier(host_blocks) if host_blocks > 0 else None
        self.nvme = NvmeTier(nvme_dir, nvme_blocks) if (nvme_dir and nvme_blocks > 0) else None
        self.swap_capacity = int(swap_blocks)
        self._lock = threading.RLock()
        self._roots: dict[int, RadixNode] = {}
        self._nodes: dict[int, RadixNode] = {}
        self._block_node: dict[int, RadixNode] = {}
        self._cached_free: set[int] = set()  # ref==0 registered HBM blocks
        self._heap: list[tuple[int, int, int, int]] = []  # (last_used, counter, bid, version)
        self._counter = 0
        self._tick = 0
        self._seqs: dict[str, SeqBlocks] = {}
        self._swapped: dict[str, _Swapped] = {}
        self._swapped_blocks = 0
        self._executor: ThreadPoolExecutor | None = None
        # statistics
        self.hit_tokens = 0
        self.query_tokens = 0
        self.rollback_tokens = 0
        self.evictions = 0
        self.demotions = 0
        self.promotions = 0
        self.dropped_cached_blocks = 0

    # ------------------------------------------------------------------ helpers
    def attach_storage(self, storage: BlockStorage) -> None:
        self.storage = storage

    @property
    def tiering_enabled(self) -> bool:
        return self.storage is not None and (self.host is not None or self.nvme is not None)

    @property
    def swap_enabled(self) -> bool:
        return self.storage is not None and self.swap_capacity > 0

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def _root(self, salt: str | None) -> RadixNode:
        k = root_key(salt)
        r = self._roots.get(k)
        if r is None:
            r = RadixNode(k, None, ())
            self._roots[k] = r
        return r

    def _push_evictable(self, bid: int) -> None:
        node = self._block_node.get(bid)
        if node is None or self.pool.ref_count[bid] != 0 or node.hbm_children > 0:
            return
        node.version += 1
        self._counter += 1
        heapq.heappush(self._heap, (node.last_used, self._counter, bid, node.version))

    # ------------------------------------------------------------------ capacity
    @property
    def num_free_blocks(self) -> int:
        """Blocks obtainable right now: free list plus evictable cached blocks."""
        with self._lock:
            return self.pool.num_free + len(self._cached_free)

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    def _alloc_block(self) -> int | None:
        bid = self.pool.allocate()
        if bid is not None:
            return bid
        if not self._evict_one():
            return None
        return self.pool.allocate()

    def _evict_one(self) -> bool:
        while self._heap:
            last_used, _, bid, ver = heapq.heappop(self._heap)
            node = self._block_node.get(bid)
            if node is None or node.version != ver or self.pool.ref_count[bid] != 0 or node.hbm_children > 0:
                continue
            self.evictions += 1
            if self.tiering_enabled and self._demote_node(node):
                return True
            self._drop_hbm_node(node)
            self.dropped_cached_blocks += 1
            return True
        # heap exhausted but cached_free non-empty (parents blocked by children): evict leaves-first by scan
        for bid in sorted(self._cached_free, key=lambda b: self._block_node[b].last_used):
            node = self._block_node[bid]
            if node.hbm_children == 0:
                self.evictions += 1
                if self.tiering_enabled and self._demote_node(node):
                    return True
                self._drop_hbm_node(node)
                self.dropped_cached_blocks += 1
                return True
        return False

    # ------------------------------------------------------------------ node removal / demotion
    def _drop_hbm_node(self, node: RadixNode) -> None:
        """Remove an HBM node (and its lower-tier descendants) from the tree and free its block."""
        self._remove_subtree(node, free_hbm=True)

    def _remove_subtree(self, node: RadixNode, *, free_hbm: bool) -> None:
        # iterative post-order
        stack = [node]
        order: list[RadixNode] = []
        while stack:
            n = stack.pop()
            order.append(n)
            stack.extend(n.children.values())
        for n in reversed(order):
            self._remove_single(n, free_hbm=free_hbm)

    def _remove_single(self, n: RadixNode, *, free_hbm: bool) -> None:
        p = n.parent
        if n.tier == TIER_HBM and n.block_id >= 0:
            bid = n.block_id
            self._block_node.pop(bid, None)
            was_cached = bid in self._cached_free
            self._cached_free.discard(bid)
            n.version += 1
            n.block_id = -1
            if was_cached and free_hbm:
                self.pool.free_block(bid)
            # referenced blocks simply become unregistered; they return to the free list on release
            if p is not None and p.parent is not None:
                p.hbm_children = max(0, p.hbm_children - 1)
        elif n.tier == TIER_HOST and self.host is not None:
            self.host.remove(n.slot)
            self.host.lru.pop(n.key, None)
        elif n.tier == TIER_NVME and self.nvme is not None:
            self.nvme.remove(n.slot)
            self.nvme.lru.pop(n.key, None)
        self._nodes.pop(n.key, None)
        if p is not None:
            p.children.pop(n.key, None)
            if p.parent is not None and p.tier == TIER_HBM and p.block_id >= 0 and p.hbm_children == 0:
                self._push_evictable(p.block_id)

    def _demote_node(self, node: RadixNode) -> bool:
        """HBM (ref==0) -> next lower tier. Returns False if no lower tier has room."""
        bid = node.block_id
        if self.storage is None or self.pool.ref_count[bid] != 0:
            return False
        tier = None
        if self.host is not None:
            if self.host.full:
                self._make_host_room()
            if not self.host.full:
                tier = self.host
        if tier is None and self.nvme is not None:
            if self.nvme.full:
                self._make_nvme_room()
            if not self.nvme.full:
                tier = self.nvme
        if tier is None:
            return False
        if node.block_id != bid or self._nodes.get(node.key) is not node:
            return True  # making room dropped this very node (a descendant of the victim): its block is already free
        payload = self.storage.read_block(bid)
        slot = tier.put(payload)
        # unlink from HBM
        self._block_node.pop(bid, None)
        self._cached_free.discard(bid)
        node.version += 1
        node.block_id = -1
        node.slot = slot
        node.tier = TIER_HOST if tier is self.host else TIER_NVME
        tier.lru[node.key] = None
        p = node.parent
        if p is not None and p.parent is not None:
            p.hbm_children = max(0, p.hbm_children - 1)
            if p.tier == TIER_HBM and p.block_id >= 0 and p.hbm_children == 0:
                self._push_evictable(p.block_id)
        self.pool.free_block(bid)
        self.demotions += 1
        return True

    def _pick_lru(self, tier: HostTier) -> RadixNode | None:
        scanned = 0
        first = None
        for key in tier.lru:
            node = self._nodes.get(key)
            if node is None:
                continue
            if first is None:
                first = node
            if not node.children:
                return node
            scanned += 1
            if scanned > 64:
                break
        return first

    def _make_host_room(self) -> None:
        assert self.host is not None
        victim = self._pick_lru(self.host)
        if victim is None:
            return
        if self.nvme is not None:
            if self.nvme.full:
                self._make_nvme_room()
            if not self.nvme.full:
                payload = self.host.get(victim.slot)
                slot = self.nvme.put(payload)
                self.host.remove(victim.slot)
                self.host.lru.pop(victim.key, None)
                victim.tier, victim.slot = TIER_NVME, slot
                self.nvme.lru[victim.key] = None
                return
        self._remove_subtree(victim, free_hbm=True)

    def _make_nvme_room(self) -> None:
        assert self.nvme is not None
        victim = self._pick_lru(self.nvme)
        if victim is not None:
            self._remove_subtree(victim, free_hbm=True)

    def _promote_node(self, node: RadixNode) -> int | None:
        """Lower tier -> HBM. Returns the new block id (ref==1, held by the caller) or None."""
        if node.tier == TIER_HBM:
            return node.block_id
        if self.storage is None:
            return None
        tier = self.host if node.tier == TIER_HOST else self.nvme
        assert tier is not None
        payload = tier.get(node.slot)  # read first: making room below may move (or even drop) this very node
        bid = self._alloc_block()
        if bid is None:
            return None
        if self._nodes.get(node.key) is not node or node.tier == TIER_HBM:
            self.pool.decref(bid)  # the node vanished while evicting for it: give the block back
            self.pool.free_block(bid)
            return None
        tier = self.host if node.tier == TIER_HOST else self.nvme  # it may have moved host -> nvme meanwhile
        assert tier is not None
        self.storage.write_block(bid, payload)
        tier.remove(node.slot)
        tier.lru.pop(node.key, None)
        node.tier, node.slot, node.block_id = TIER_HBM, -1, bid
        node.version += 1
        self._block_node[bid] = node
        p = node.parent
        if p is not None and p.parent is not None:
            p.hbm_children += 1
        self.promotions += 1
        return bid

    # ------------------------------------------------------------------ prefix lookup
    def _walk(self, token_ids: list[int], salt: str | None, max_blocks: int):
        """Yield matching nodes (any tier) along the chain."""
        cur = self._root(salt)
        bs = self.block_size
        for i in range(max_blocks):
            toks = tuple(token_ids[i * bs:(i + 1) * bs])
            key = hash_block(cur.key, toks)
            node = self._nodes.get(key)
            if node is None or node.tokens != toks or node.parent is not cur:
                return
            yield node
            cur = node

    def peek_prefix(self, token_ids: list[int], salt: str | None = None, limit_tokens: int | None = None) -> int:
        """Number of leading tokens that would be cache hits (all tiers), without changing any state."""
        if not self.enable_prefix_caching:
            return 0
        with self._lock:
            lim = len(token_ids) if limit_tokens is None else min(limit_tokens, len(token_ids))
            n = sum(1 for _ in self._walk(token_ids, salt, lim // self.block_size))
            return n * self.block_size

    def match_prefix(self, token_ids: list[int], salt: str | None = None, limit_tokens: int | None = None,
                     promote: bool = True) -> tuple[list[int], int]:
        """Plan API: ``(block_ids, n_hit_tokens)``. Acquires one reference per returned block (call ``release_blocks``
        or hand them to ``allocate`` semantics yourself); lower-tier hits are promoted to HBM when ``promote``."""
        with self._lock:
            return self._match_acquire(token_ids, salt, limit_tokens, promote)

    def _match_acquire(self, token_ids: list[int], salt: str | None, limit_tokens: int | None,
                       promote: bool) -> tuple[list[int], int]:
        if not self.enable_prefix_caching:
            return [], 0
        lim = len(token_ids) if limit_tokens is None else min(limit_tokens, len(token_ids))
        hit: list[int] = []
        nodes: list[RadixNode] = []
        for node in self._walk(token_ids, salt, lim // self.block_size):
            if node.tier != TIER_HBM:
                if not promote:
                    break
                bid = self._promote_node(node)  # returns with ref==1
                if bid is None:
                    break
                hit.append(bid)
            else:
                bid = node.block_id
                if self.pool.ref_count[bid] == 0:
                    self._cached_free.discard(bid)
                self.pool.incref(bid)
                hit.append(bid)
            nodes.append(node)
        # touch leaf-first so the deepest block is the oldest (evicted first)
        for node in reversed(nodes):
            node.last_used = self._next_tick()
            if node.tier == TIER_HBM and node.block_id >= 0 and self.pool.ref_count[node.block_id] == 0:
                self._push_evictable(node.block_id)
        return hit, len(hit) * self.block_size

    def release_blocks(self, block_ids: list[int]) -> None:
        with self._lock:
            for bid in reversed(block_ids):
                self._release(bid)

    def _release(self, bid: int) -> None:
        if self.pool.decref(bid) == 0:
            node = self._block_node.get(bid)
            if node is not None:
                node.last_used = self._next_tick()
                self._cached_free.add(bid)
                self._push_evictable(bid)
            else:
                self.pool.free_block(bid)

    # ------------------------------------------------------------------ sequence lifecycle
    def can_allocate(self, token_ids: list[int], salt: str | None = None, total_slots: int | None = None) -> bool:
        with self._lock:
            n = len(token_ids) if total_slots is None else total_slots
            need_total = -(-n // self.block_size)
            hit_tokens = self.peek_prefix(token_ids, salt, limit_tokens=max(len(token_ids) - 1, 0))
            hit_nodes = list(self._walk(token_ids, salt, hit_tokens // self.block_size))
            hbm_hit = [nd for nd in hit_nodes if nd.tier == TIER_HBM]
            reclaimed_by_hit = sum(1 for nd in hbm_hit if nd.block_id in self._cached_free)
            need_new = need_total - len(hbm_hit)
            avail = self.pool.num_free + len(self._cached_free) - reclaimed_by_hit
            return need_new <= avail

    def allocate(self, seq_id: str, token_ids: list[int], salt: str | None = None,
                 total_slots: int | None = None) -> int | None:
        """Admit a sequence: reuse the longest cached prefix (always leaving >=1 token to compute), allocate the rest.

        Returns the number of cached prompt tokens, or ``None`` when there are not enough blocks (no state changed)."""
        with self._lock:
            if seq_id in self._seqs:
                raise ForgeError(f"sequence {seq_id} already allocated")
            n = len(token_ids) if total_slots is None else total_slots
            need_total = -(-n // self.block_size)
            hit, n_hit = self._match_acquire(token_ids, salt, max(len(token_ids) - 1, 0), True)
            need_new = need_total - len(hit)
            new_blocks: list[int] = []
            for _ in range(max(need_new, 0)):
                b = self._alloc_block()
                if b is None:
                    for x in new_blocks:
                        self.pool.decref(x)
                        self.pool.free_block(x)
                    for x in reversed(hit):
                        self._release(x)
                    return None
                new_blocks.append(b)
            st = SeqBlocks(blocks=hit + new_blocks, num_slots=n, hashed_blocks=len(hit), salt=salt, cached_tokens=n_hit,
                           chain=[self._block_node[b].key for b in hit])
            if hit:
                st.parent_key = self._block_node[hit[-1]].key
            else:
                st.parent_key = self._root(salt).key
            self._seqs[seq_id] = st
            self.query_tokens += len(token_ids)
            self.hit_tokens += n_hit
            return n_hit

    def has_seq(self, seq_id: str) -> bool:
        return seq_id in self._seqs

    def block_table(self, seq_id: str) -> list[int]:
        return list(self._seqs[seq_id].blocks)

    def num_slots(self, seq_id: str) -> int:
        return self._seqs[seq_id].num_slots

    def reserve(self, seq_id: str, total_slots: int) -> list[tuple[int, int]] | None:
        """Grow the sequence to ``total_slots`` token slots. Returns copy-on-write ``(src, dst)`` block copies the
        caller must execute before writing (usually empty), or ``None`` if blocks ran out (nothing changed)."""
        with self._lock:
            st = self._seqs[seq_id]
            old = st.num_slots
            if total_slots <= old:
                return []
            bs = self.block_size
            need_blocks = -(-total_slots // bs) - len(st.blocks)
            cow = old % bs != 0 and old // bs < len(st.blocks) and self.pool.ref_count[st.blocks[old // bs]] > 1
            need = max(need_blocks, 0) + (1 if cow else 0)
            if need > self.pool.num_free + len(self._cached_free):
                return None
            # allocate everything first so that a failure (cached blocks pinned by descendants are not evictable)
            # leaves the sequence exactly as it was
            fresh: list[int] = []
            for _ in range(need):
                b = self._alloc_block()
                if b is None:
                    for x in fresh:
                        self.pool.decref(x)
                        self.pool.free_block(x)
                    return None
                fresh.append(b)
            copies: list[tuple[int, int]] = []
            if cow:
                idx = old // bs
                src = st.blocks[idx]
                dst = fresh.pop(0)
                copies.append((src, dst))
                st.blocks[idx] = dst
                self._release(src)
            st.blocks.extend(fresh)
            st.num_slots = total_slots
            return copies

    append_slot = reserve  # plan-API alias

    def commit_tokens(self, seq_id: str, token_ids: list[int]) -> int:
        """Declare that ``token_ids`` (the sequence's valid tokens, prefix first) are now in the KV cache, and register
        every newly *completed* block in the radix tree. Returns the number of blocks registered."""
        with self._lock:
            st = self._seqs[seq_id]
            if not self.enable_prefix_caching or st.hash_disabled:
                return 0
            bs = self.block_size
            full = min(len(token_ids) // bs, len(st.blocks))
            n_new = 0
            for i in range(st.hashed_blocks, full):
                toks = tuple(token_ids[i * bs:(i + 1) * bs])
                if st.parent_key == root_key(st.salt):
                    parent = self._root(st.salt)
                else:
                    parent = self._nodes.get(st.parent_key)  # type: ignore[assignment]
                    if parent is None:  # ancestor was dropped from the tree: stop registering for this sequence
                        st.hash_disabled = True
                        break
                key = hash_block(st.parent_key, toks)
                existing = self._nodes.get(key)
                if existing is not None:
                    if existing.tokens != toks or existing.parent is not parent:
                        st.hash_disabled = True  # collision: stop registering for this sequence
                        break
                    st.parent_key = key  # duplicate content already cached: leave our block unregistered
                    st.hashed_blocks = i + 1
                    st.chain.append(key)
                    continue
                bid = st.blocks[i]
                if bid in self._block_node:  # already registered (e.g. restored); skip
                    st.parent_key = self._block_node[bid].key
                    st.hashed_blocks = i + 1
                    st.chain.append(st.parent_key)
                    continue
                node = RadixNode(key, parent, toks)
                node.block_id = bid
                node.last_used = self._next_tick()
                parent.children[key] = node
                if parent.parent is not None:
                    parent.hbm_children += 1
                    parent.version += 1  # invalidate parent's heap entry (no longer a leaf)
                self._nodes[key] = node
                self._block_node[bid] = node
                st.parent_key = key
                st.hashed_blocks = i + 1
                st.chain.append(key)
                n_new += 1
            return n_new

    def rollback(self, seq_id: str, new_num_slots: int) -> None:
        """Shrink to ``new_num_slots`` (speculation: drop rejected positions). Frees trailing blocks."""
        with self._lock:
            st = self._seqs[seq_id]
            if new_num_slots >= st.num_slots:
                return
            bs = self.block_size
            keep_blocks = -(-new_num_slots // bs)
            # registered blocks that would be truncated must be unregistered (their content changes)
            if new_num_slots < st.hashed_blocks * bs:
                first_bad = new_num_slots // bs
                hi = min(st.hashed_blocks, len(st.blocks))
                for i in range(first_bad, hi):  # validate first: a refusal must leave every structure untouched
                    node = self._block_node.get(st.blocks[i])
                    if node is not None and self.pool.ref_count[st.blocks[i]] > 1:
                        raise ForgeError("cannot roll back into a shared registered block")
                for i in range(first_bad, hi):
                    node = self._block_node.get(st.blocks[i])
                    if node is not None:
                        self._remove_subtree(node, free_hbm=True)
                st.hashed_blocks = first_bad
                del st.chain[first_bad:]
                st.parent_key = st.chain[-1] if st.chain else self._root(st.salt).key
            self.rollback_tokens += st.num_slots - new_num_slots
            for bid in reversed(st.blocks[keep_blocks:]):
                self._release(bid)
            del st.blocks[keep_blocks:]
            st.num_slots = new_num_slots

    def commit(self, seq_id: str, keep_tokens: int) -> None:
        """Plan API: keep the accepted path, free rejected positions (``rollback`` to ``keep_tokens``)."""
        self.rollback(seq_id, keep_tokens)

    def fork(self, seq_id: str, new_seq_id: str) -> list[int]:
        """Copy-on-write fork: the child shares every block; the first write into a shared partial block copies it."""
        with self._lock:
            st = self._seqs[seq_id]
            if new_seq_id in self._seqs:
                raise ForgeError(f"sequence {new_seq_id} already exists")
            for b in st.blocks:
                self.pool.incref(b)
            self._seqs[new_seq_id] = SeqBlocks(blocks=list(st.blocks), num_slots=st.num_slots,
                                               hashed_blocks=st.hashed_blocks, parent_key=st.parent_key, salt=st.salt,
                                               cached_tokens=st.cached_tokens, hash_disabled=st.hash_disabled,
                                               chain=list(st.chain))
            return list(st.blocks)

    def free(self, seq_id: str) -> None:
        """Release a sequence. Registered (full) blocks stay cached for reuse; the rest return to the free list."""
        with self._lock:
            st = self._seqs.pop(seq_id, None)
            if st is None:
                return
            for bid in reversed(st.blocks):  # leaf first -> oldest -> evicted first
                self._release(bid)

    # ------------------------------------------------------------------ swap (preemption)
    def can_swap_out(self, seq_id: str) -> bool:
        with self._lock:
            st = self._seqs.get(seq_id)
            return bool(st) and self.swap_enabled and self._swapped_blocks + len(st.blocks) <= self.swap_capacity

    def swap_out(self, seq_id: str) -> bool:
        with self._lock:
            if not self.can_swap_out(seq_id):
                return False
            st = self._seqs[seq_id]
            assert self.storage is not None
            payloads = [self.storage.read_block(b) for b in st.blocks]
            self._swapped[seq_id] = _Swapped(payloads, st.num_slots, st.salt)
            self._swapped_blocks += len(payloads)
            self.free(seq_id)
            return True

    def can_swap_in(self, seq_id: str) -> bool:
        with self._lock:
            sw = self._swapped.get(seq_id)
            return sw is not None and len(sw.payloads) <= self.pool.num_free + len(self._cached_free)

    def swap_in(self, seq_id: str, token_ids: list[int]) -> bool:
        """Restore a swapped sequence; ``token_ids`` are its KV-resident tokens (re-registers completed blocks)."""
        with self._lock:
            sw = self._swapped.get(seq_id)
            if sw is None or not self.can_swap_in(seq_id):
                return False
            assert self.storage is not None
            blocks: list[int] = []
            for pl in sw.payloads:
                b = self._alloc_block()
                if b is None:
                    for x in blocks:
                        self.pool.decref(x)
                        self.pool.free_block(x)
                    return False
                self.storage.write_block(b, pl)
                blocks.append(b)
            self._swapped.pop(seq_id)
            self._swapped_blocks -= len(sw.payloads)
            self._seqs[seq_id] = SeqBlocks(blocks=blocks, num_slots=sw.num_slots, salt=sw.salt,
                                           parent_key=self._root(sw.salt).key)
            self.commit_tokens(seq_id, token_ids)
            return True

    def drop_swapped(self, seq_id: str) -> None:
        with self._lock:
            sw = self._swapped.pop(seq_id, None)
            if sw is not None:
                self._swapped_blocks -= len(sw.payloads)

    def swapped_num_blocks(self, seq_id: str) -> int:
        sw = self._swapped.get(seq_id)
        return len(sw.payloads) if sw else 0

    def is_swapped(self, seq_id: str) -> bool:
        return seq_id in self._swapped

    # ------------------------------------------------------------------ explicit tier API
    def tier_of_key(self, key: int) -> int | None:
        """Tier index of the cached node with this hash key (None if not cached)."""
        node = self._nodes.get(key)
        return None if node is None else node.tier

    def cached_keys(self, token_ids: list[int], salt: str | None = None) -> list[int]:
        """Hash keys of the cached blocks along ``token_ids`` (for tests / tier management)."""
        with self._lock:
            return [n.key for n in self._walk(token_ids, salt, len(token_ids) // self.block_size)]

    def demote(self, block_id: int) -> int | None:
        """Demote an unreferenced cached HBM block one tier down. Returns the new tier index, or None if not possible."""
        with self._lock:
            node = self._block_node.get(block_id)
            if node is None or self.pool.ref_count[block_id] != 0 or node.hbm_children > 0 or not self.tiering_enabled:
                return None
            return node.tier if self._demote_node(node) else None

    def demote_key(self, key: int) -> int | None:
        """Move a host-tier node to NVMe (host -> nvme)."""
        with self._lock:
            node = self._nodes.get(key)
            if node is None or node.tier != TIER_HOST or self.nvme is None or self.host is None:
                return None
            if self.nvme.full:
                self._make_nvme_room()
            if self.nvme.full:
                return None
            payload = self.host.get(node.slot)
            slot = self.nvme.put(payload)
            self.host.remove(node.slot)
            self.host.lru.pop(key, None)
            node.tier, node.slot = TIER_NVME, slot
            self.nvme.lru[key] = None
            return TIER_NVME

    def promote(self, key: int) -> int | None:
        """Promote a lower-tier node to HBM (left cached, ref==0). Returns its block id."""
        with self._lock:
            node = self._nodes.get(key)
            if node is None:
                return None
            if node.tier == TIER_HBM:
                return node.block_id
            bid = self._promote_node(node)
            if bid is None:
                return None
            self._release(bid)
            return bid

    def prefetch(self, token_ids: list[int], salt: str | None = None) -> int:
        """Promote the lower-tier part of the cached prefix of ``token_ids`` into HBM. Returns blocks promoted."""
        with self._lock:
            n = 0
            nodes = list(self._walk(token_ids, salt, len(token_ids) // self.block_size))
            for node in nodes:
                if node.tier != TIER_HBM:
                    bid = self._promote_node(node)
                    if bid is None:
                        break
                    self._release(bid)
                    n += 1
            return n

    def _exec(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forge-kv-tier")
        return self._executor

    def demote_async(self, block_id: int) -> Future:
        return self._exec().submit(self.demote, block_id)

    def promote_async(self, key: int) -> Future:
        return self._exec().submit(self.promote, key)

    def prefetch_async(self, token_ids: list[int], salt: str | None = None) -> Future:
        return self._exec().submit(self.prefetch, list(token_ids), salt)

    def demote_cold(self, max_blocks: int = 1 << 30) -> int:
        """Proactively demote the coldest evictable cached blocks (e.g. when idle). Returns blocks demoted."""
        n = 0
        with self._lock:
            if not self.tiering_enabled:
                return 0
            for bid in sorted(self._cached_free, key=lambda b: self._block_node[b].last_used):
                if n >= max_blocks:
                    break
                if self.demote(bid) is not None:
                    n += 1
        return n

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.pool.num_blocks
            in_use = sum(1 for r in self.pool.ref_count if r > 0)
            cached = len(self._cached_free)
            used_slots = sum(s.num_slots for s in self._seqs.values())
            alloc_slots = sum(len(s.blocks) for s in self._seqs.values()) * self.block_size
            frag = 1.0 - (used_slots / alloc_slots) if alloc_slots else 0.0
            bpb = self.bytes_per_block
            return {
                "num_blocks": total,
                "block_size": self.block_size,
                "blocks_in_use": in_use,
                "blocks_cached_free": cached,
                "blocks_free": self.pool.num_free,
                "usage": in_use / total,
                "hit_rate": (self.hit_tokens / self.query_tokens) if self.query_tokens else 0.0,
                "hit_tokens": self.hit_tokens,
                "query_tokens": self.query_tokens,
                "fragmentation": frag,
                "bytes_by_tier": {
                    "hbm_active": in_use * bpb,
                    "hbm_cached": cached * bpb,
                    "hbm_free": self.pool.num_free * bpb,
                    "host": (len(self.host) if self.host else 0) * bpb,
                    "nvme": (len(self.nvme) if self.nvme else 0) * bpb,
                    "swapped": self._swapped_blocks * bpb,
                },
                "rollback_bytes": self.rollback_tokens * (bpb // self.block_size if self.block_size else 0),
                "rollback_tokens": self.rollback_tokens,
                "evictions": self.evictions,
                "demotions": self.demotions,
                "promotions": self.promotions,
                "num_seqs": len(self._seqs),
                "num_radix_nodes": len(self._nodes),
                "swapped_seqs": len(self._swapped),
            }
