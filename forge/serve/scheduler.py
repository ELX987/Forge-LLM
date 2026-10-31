"""Continuous-batching scheduler (Plan SV-1): token budget per step, chunked prefill, priorities with aging,
preemption by recompute or host swap, abort.

The scheduler owns the request queues and talks to the :class:`~forge.serve.kv.KVBlockManager`; it never runs the model.
Per-request invariant: ``tokens = prompt + output``; ``num_computed`` tokens are in the KV cache; the remaining
``pending = len(tokens) - num_computed`` tokens are fed in the next step(s). A step that consumes *all* pending tokens
produces logits for the next token (so decode is simply "pending == 1" and prefill is "pending == many").
"""
from __future__ import annotations

import enum
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from forge.serve.kv import KVBlockManager
from forge.serve.sampling import SamplingParams


class RequestStatus(str, enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


_seq_counter = itertools.count()


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    priority: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    group_id: str | None = None
    index: int = 0
    salt: str | None = None
    # lifecycle
    status: RequestStatus = RequestStatus.WAITING
    tokens: list[int] = field(default_factory=list)       # prompt + output
    num_computed: int = 0
    num_cached_prompt: int = 0
    finish_reason: str | None = None
    num_preemptions: int = 0
    # runtime objects (engine-owned)
    generator: Any = None
    detok: Any = None
    logprobs: list[dict] = field(default_factory=list)
    last_hidden: Any = None
    # metrics
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None
    spec_proposed: int = 0
    spec_accepted: int = 0
    order: int = field(default_factory=lambda: next(_seq_counter))
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = list(self.prompt_ids)

    @property
    def output_ids(self) -> list[int]:
        return self.tokens[len(self.prompt_ids):]

    @property
    def num_output(self) -> int:
        return len(self.tokens) - len(self.prompt_ids)

    @property
    def pending(self) -> int:
        return len(self.tokens) - self.num_computed

    @property
    def finished(self) -> bool:
        return self.status == RequestStatus.FINISHED


@dataclass
class ScheduledSeq:
    req: Request
    num_tokens: int          # tokens fed this step (for decode: 1 + gamma reserved draft slots)
    gamma: int = 0           # reserved draft slots (decode only)
    will_sample: bool = True  # this step consumes all pending tokens -> sample next token
    cow_copies: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_decode(self) -> bool:
        return self.req.pending == 1


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    dropped: list[Request] = field(default_factory=list)  # can never fit in the KV pool -> engine finishes them ("length")
    num_batched_tokens: int = 0


class Scheduler:
    def __init__(self, manager: KVBlockManager, *, max_num_seqs: int, max_num_batched_tokens: int, max_model_len: int,
                 chunked_prefill: bool = True, policy: str = "fcfs", priority_aging_s: float = 0.0,
                 preemption_mode: str = "recompute", prefix_caching: bool = True, max_prefill_chunk: int | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.manager = manager
        self.max_num_seqs = max_num_seqs
        self.max_batched = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.chunked_prefill = chunked_prefill
        self.policy = policy
        self.aging = priority_aging_s
        self.preemption_mode = preemption_mode
        self.prefix_caching = prefix_caching
        self.max_prefill_chunk = max_prefill_chunk
        self.clock = clock
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.swapped: list[Request] = []
        self.total_preemptions = 0
        self.total_swaps = 0

    # ------------------------------------------------------------------ queues
    def add(self, req: Request) -> None:
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def _key(self, req: Request, now: float | None = None) -> tuple:
        if self.policy == "fcfs":
            return (req.arrival_time, req.order)
        eff = req.priority
        if self.aging > 0:
            now = self.clock() if now is None else now
            eff = req.priority - int(max(0.0, now - req.arrival_time) / self.aging)
        return (eff, req.arrival_time, req.order)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def find(self, request_id: str) -> Request | None:
        for q in (self.running, self.waiting, self.swapped):
            for r in q:
                if r.request_id == request_id:
                    return r
        return None

    def abort(self, request_id: str, reason: str = "abort") -> Request | None:
        r = self.find(request_id)
        if r is None:
            return None
        self.finish(r, reason)
        return r

    def finish(self, req: Request, reason: str) -> None:
        for q in (self.running, self.waiting, self.swapped):
            if req in q:
                q.remove(req)
        if self.manager.has_seq(req.request_id):
            self.manager.free(req.request_id)
        self.manager.drop_swapped(req.request_id)
        req.status = RequestStatus.FINISHED
        req.finish_reason = req.finish_reason or reason
        req.finish_time = self.clock()

    # ------------------------------------------------------------------ preemption
    def _preempt(self, victim: Request, out: SchedulerOutput) -> None:
        self.total_preemptions += 1
        victim.num_preemptions += 1
        if victim in self.running:
            self.running.remove(victim)
        out.preempted.append(victim)
        if self.preemption_mode == "swap" and self.manager.swap_out(victim.request_id):
            victim.status = RequestStatus.SWAPPED
            self.swapped.append(victim)
            self.total_swaps += 1
            return
        if self.manager.has_seq(victim.request_id):
            self.manager.free(victim.request_id)
        victim.num_computed = 0
        victim.num_cached_prompt = 0
        victim.last_hidden = None
        victim.status = RequestStatus.WAITING
        self.waiting.append(victim)

    # ------------------------------------------------------------------ main entry
    def schedule(self, gamma_fn: Callable[[Request], int] | None = None) -> SchedulerOutput:
        now = self.clock()
        out = SchedulerOutput()
        budget = self.max_batched
        scheduled: list[ScheduledSeq] = []
        by_id: dict[str, ScheduledSeq] = {}

        # ---- swap-in (keep headroom so running sequences can still grow) --------------------------------
        for r in sorted(self.swapped, key=lambda x: self._key(x, now)):
            if len(self.running) >= self.max_num_seqs:
                break
            headroom = len(self.running)
            sw_blocks = self.manager.swapped_num_blocks(r.request_id)
            if self.manager.num_free_blocks - sw_blocks < headroom:
                break
            if self.manager.swap_in(r.request_id, r.tokens[:r.num_computed]):
                self.swapped.remove(r)
                r.status = RequestStatus.RUNNING
                self.running.append(r)

        # ---- running ------------------------------------------------------------------------------------
        order = sorted(self.running, key=lambda x: self._key(x, now))
        queue = list(order)
        while queue:
            req = queue.pop(0)
            if req.status != RequestStatus.RUNNING:
                continue
            pending = req.pending
            if pending <= 0:  # defensive: should not happen
                continue
            if pending == 1:
                g = gamma_fn(req) if gamma_fn else 0
                g = max(0, min(g, budget - 1))
                n = 1 + g
                if budget < 1:
                    continue
                while True:
                    copies = self.manager.reserve(req.request_id, req.num_computed + n)
                    if copies is not None:
                        break
                    if g > 0:
                        g, n = 0, 1
                        continue
                    victim = self._pick_victim(queue, scheduled, req)
                    if victim is None or victim is req:
                        self._preempt(req, out)
                        copies = None
                        break
                    if victim in queue:
                        queue.remove(victim)
                    sch = by_id.pop(victim.request_id, None)
                    if sch is not None:
                        scheduled.remove(sch)
                        budget += sch.num_tokens
                    self._preempt(victim, out)
                if copies is None:
                    continue
                sch = ScheduledSeq(req, n, g, True, copies)
            else:
                n = pending
                if self.chunked_prefill:
                    n = min(pending, budget)
                    if self.max_prefill_chunk:
                        n = min(n, self.max_prefill_chunk)
                elif pending > budget and scheduled:
                    continue
                if n <= 0:
                    continue
                sch = ScheduledSeq(req, n, 0, n == pending)
            scheduled.append(sch)
            by_id[req.request_id] = sch
            budget -= sch.num_tokens

        # ---- admit waiting ------------------------------------------------------------------------------
        if budget > 0 or not scheduled:
            for req in sorted(self.waiting, key=lambda x: self._key(x, now)):
                if -(-(len(req.tokens) + 1) // self.manager.block_size) > self.manager.num_blocks:
                    # the sequence outgrew the whole pool (even alone): it can never be scheduled again
                    self.waiting.remove(req)
                    req.finish_reason = "length"
                    out.dropped.append(req)
                    continue
                if len(self.running) >= self.max_num_seqs:
                    break
                tokens = req.tokens
                hit = self.manager.peek_prefix(tokens, req.salt, max(len(tokens) - 1, 0)) if self.prefix_caching else 0
                pending = len(tokens) - hit
                if self.chunked_prefill:
                    n = min(pending, budget)
                    if self.max_prefill_chunk:
                        n = min(n, self.max_prefill_chunk)
                    if n <= 0:
                        break
                else:
                    n = pending
                    if n > budget and scheduled:
                        break
                if not self.manager.can_allocate(tokens, req.salt):
                    break  # head-of-line: keep ordering fair, wait for memory
                got = self.manager.allocate(req.request_id, tokens, req.salt)
                if got is None:
                    break
                req.num_computed = got
                if req.num_preemptions == 0:
                    req.num_cached_prompt = got
                if self.chunked_prefill:
                    n = min(len(tokens) - got, budget, self.max_prefill_chunk or (1 << 60))
                else:
                    n = len(tokens) - got
                if n <= 0:  # budget exhausted after the cache hit changed the numbers
                    self.manager.free(req.request_id)
                    req.num_computed = 0
                    break
                self.waiting.remove(req)
                req.status = RequestStatus.RUNNING
                self.running.append(req)
                sch = ScheduledSeq(req, n, 0, n == len(tokens) - got)
                scheduled.append(sch)
                by_id[req.request_id] = sch
                budget -= n
                if budget <= 0:
                    break

        out.scheduled = scheduled
        out.num_batched_tokens = sum(s.num_tokens for s in scheduled)
        return out

    def _pick_victim(self, queue: list[Request], scheduled: list[ScheduledSeq], me: Request) -> Request | None:
        """Lowest-priority (largest key) running request other than ``me``; prefers ones not yet scheduled this step."""
        cands = [r for r in queue if r is not me and r.status == RequestStatus.RUNNING]
        if cands:
            return max(cands, key=self._key)
        cands = [s.req for s in scheduled if s.req is not me and s.req.status == RequestStatus.RUNNING]
        if cands:
            worst = max(cands, key=self._key)
            if self._key(worst) > self._key(me):
                return worst
        return me
