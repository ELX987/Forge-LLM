"""``LLMEngine``: continuous batching + paged KV + prefix cache + chunked prefill + speculative decoding.

``add_request`` / ``abort`` are thread-safe (they only touch lock-free queues); ``step`` must be called from one thread
(the HTTP server runs it in a background thread). Everything numerical goes through :class:`~forge.serve.runner.ModelRunner`,
which works on CPU (reference attention) and CUDA (paged decode kernel, optional CUDA graphs).

Exactness invariants (tested on CPU): greedy outputs equal a plain full-recompute greedy generate; prefix caching and
chunked prefill do not change outputs; speculation leaves the output distribution unchanged (rejection sampling), and
greedy outputs identical.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from forge.core.errors import ConfigError
from forge.serve.config import ServeConfig
from forge.serve.kv import KVBlockManager
from forge.serve.memory import artifact_kv_choice, draft_pool_blocks
from forge.serve.runner import (
    KVPool,
    ModelAdapter,
    ModelRunner,
    SeqInput,
    kv_bytes_per_block,
    normalize_kv_name,
    plan_num_blocks,
    plan_kv_pool,
    resolve_device,
    resolve_dtype,
)
from forge.serve.sampling import (
    IncrementalDetokenizer,
    SamplingParams,
    make_generator,
    sample_from_probs,
    sampling_probs,
    token_logprobs,
)
from forge.serve.scheduler import Request, ScheduledSeq, Scheduler
from forge.serve.spec import (
    CompositeDrafter,
    DraftModelDrafter,
    DraftRequest,
    EagleDrafter,
    NgramDrafter,
    SpecController,
    StepCostModel,
    _module_bytes,
    rejection_sample,
)

log = logging.getLogger("forge.serve.engine")

_auto_id = itertools.count()


@dataclass
class RequestOutput:
    request_id: str
    index: int = 0
    new_token_ids: list[int] = field(default_factory=list)
    new_text: str = ""
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finished: bool = False
    finish_reason: str | None = None
    logprobs: list[dict] | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


class LLMEngine:
    def __init__(self, cfg: ServeConfig, *, model: Any | None = None, tokenizer: Any | None = None,
                 model_cfg: Any | None = None, draft_model: Any | None = None, drafter: Any | None = None,
                 clock: Any = time.monotonic) -> None:
        import torch

        self._torch = torch
        self.cfg = cfg
        self.clock = clock
        if cfg.tp_size != 1:
            raise ConfigError("tensor parallelism (tp_size > 1) is not implemented in this engine")
        self.device = resolve_device(cfg.device)
        injected = model is not None
        if model is None:
            model, tok = self._load(cfg)
            tokenizer = tokenizer or tok
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        if injected:
            fp = next((p for p in model.parameters() if p.is_floating_point()), None)
            self.compute_dtype = fp.dtype if fp is not None else resolve_dtype(cfg.dtype, self.device)
            model.to(self.device)
        else:
            self.compute_dtype = resolve_dtype(cfg.dtype, self.device)
        self.adapter = ModelAdapter(model, model_cfg)
        d = self.adapter.dims
        self.dims = d
        # hybrid (lfm2) models carry a per-sequence conv state that is not part of the KV blocks: nothing that reuses / rolls back /
        # moves KV blocks (prefix cache, speculation, swap preemption) can reproduce it
        self.hybrid = bool(d.conv_layers)
        prefix_caching = cfg.prefix_caching
        if self.hybrid:
            if cfg.preemption_mode == "swap":
                raise ConfigError("preemption_mode='swap' is not supported for hybrid (conv) models: the conv state is not swapped; use 'recompute'")
            if cfg.speculation.mode != "off" or drafter is not None:
                raise ConfigError("speculative decoding is not supported for hybrid (conv) models: the conv state cannot be rolled back")
            if prefix_caching:
                log.info("prefix caching disabled: hybrid (conv) models need the conv state at the cached-prefix boundary")
                prefix_caching = False
        from forge.models.loader import describe_loaded_model

        info = describe_loaded_model(model)
        self.kv_name = normalize_kv_name(cfg.kv_dtype, self.compute_dtype)
        chosen_kv, self.kv_warnings = artifact_kv_choice(cfg.kv_dtype, info.get("kv"))
        if chosen_kv:
            self.kv_name = chosen_kv  # the artifact was compressed (and gated) with this KV scheme: serve it that way
            log.info("kv_dtype=auto -> %s (the artifact's recipe quantised the KV cache with this scheme)", self.kv_name)
        for w in self.kv_warnings:
            log.warning("%s", w)
        draft_extra = self._prepare_draft(draft_model, drafter)
        if self._draft_model is not None:
            draft_model = self._draft_model
        # explicit memory budget: weights are measured on the loaded model, the KV pool is what is left of gpu_mem_util (see forge.serve.memory)
        self.memory = plan_kv_pool(dims=d, block_size=cfg.block_size, kv_name=self.kv_name, device=self.device,
                                   gpu_mem_util=cfg.gpu_mem_util, activation_reserve_gb=cfg.activation_reserve_gb,
                                   max_num_seqs=cfg.max_num_seqs, max_model_len=cfg.max_model_len, weight_bytes=int(info["weight_bytes"]),
                                   packed_weight_bytes=int(info["packed_bytes"]), num_gpu_blocks=cfg.num_gpu_blocks, kv_pool=cfg.kv_pool,
                                   kv_pool_cap_tokens=cfg.kv_pool_cap_tokens, extra_bytes=draft_extra)
        num_blocks = self.memory.num_blocks
        for w in self.memory.warnings:
            log.warning("%s", w)
        log.info("memory plan: %s", self.memory.summary())
        self.identity = {
            "name": cfg.display_name(), "requested": cfg.model, "path": info.get("path"), "quantized": info["quantized"], "label": info.get("label"),
            "layer_kinds": info.get("layer_kinds"), "bits": info.get("bits"), "group_size": info.get("group_size"), "layout_id": info.get("layout_id"),
            "base_model": info.get("base_model"), "weight_gb": round(info["weight_bytes"] / 1e9, 3), "packed_gb": round(info["packed_bytes"] / 1e9, 3),
            "weight_bytes": int(info["weight_bytes"]), "packed_bytes": int(info["packed_bytes"]),
            "dtype": str(self.compute_dtype).replace("torch.", ""), "kv_dtype": self.kv_name, "device": str(self.device),
            "kernel_convert": info.get("kernel_convert"),
            "warnings": list(self.kv_warnings), "artifact_kv": info.get("kv"),
        }
        self.pool = KVPool(num_blocks, cfg.block_size, d.num_kv_layers, d.num_kv_heads, d.head_dim, self.compute_dtype,
                           self.kv_name, self.device)
        bpb = self.pool.bytes_per_block
        host_blocks = int(cfg.host_kv_gb * 1e9 // bpb) if cfg.host_kv_gb > 0 else 0
        swap_blocks = 0
        if cfg.preemption_mode == "swap" and host_blocks > 0:
            swap_blocks = host_blocks // 2
            host_blocks -= swap_blocks
        nvme_blocks = int(cfg.nvme_kv_gb * 1e9 // bpb) if (cfg.nvme_kv_dir and cfg.nvme_kv_gb > 0) else 0
        self.manager = KVBlockManager(num_blocks, cfg.block_size, enable_prefix_caching=prefix_caching,
                                      host_blocks=host_blocks, nvme_dir=cfg.nvme_kv_dir if nvme_blocks else None,
                                      nvme_blocks=nvme_blocks, swap_blocks=swap_blocks, bytes_per_block=bpb, storage=self.pool)
        self.runner = ModelRunner(model, self.pool, model_cfg=model_cfg, max_model_len=cfg.max_model_len,
                                  kernel_attention=cfg.kernel_attention, cuda_graphs=cfg.cuda_graphs,
                                  cuda_graph_max_batch=cfg.cuda_graph_max_batch)
        self.scheduler = Scheduler(self.manager, max_num_seqs=cfg.max_num_seqs, max_num_batched_tokens=cfg.max_num_batched_tokens,
                                   max_model_len=cfg.max_model_len, chunked_prefill=cfg.chunked_prefill, policy=cfg.policy,
                                   priority_aging_s=cfg.priority_aging_s, preemption_mode=cfg.preemption_mode,
                                   prefix_caching=prefix_caching, clock=clock)
        # eos
        eos: set[int] = set(cfg.eos_token_ids or [])
        if not eos:
            for src in (tokenizer, getattr(model, "config", None), model_cfg):
                v = getattr(src, "eos_token_id", None) if src is not None else None
                if isinstance(v, int):
                    eos.add(v)
                elif isinstance(v, (list, tuple)):
                    eos.update(int(x) for x in v)
        self.eos_ids = {t for t in eos if 0 <= t < d.vocab_size}
        self._eos_list = sorted(self.eos_ids)

        # speculation
        self.drafter: Any = drafter
        self.controller: SpecController | None = None
        self._need_hidden = False
        sc = cfg.speculation
        if drafter is None and sc.mode != "off":
            self.drafter = self._build_drafter(sc, draft_model)
        if self.drafter is not None:
            self._need_hidden = bool(getattr(self.drafter, "needs_hidden", False))
            self.controller = self._build_controller()

        # state
        self._incoming: deque[Request] = deque()
        self._aborts: deque[str] = deque()
        self._groups: dict[str, list[str]] = {}
        self._reqs: dict[str, Request] = {}
        self._lock = threading.Lock()
        self._sched_batch = 1
        self._ttft: deque[float] = deque(maxlen=2000)
        self._itl: deque[float] = deque(maxlen=5000)
        self._tok_events: deque[tuple[float, int]] = deque()
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0
        self.total_finished = 0
        self.num_steps = 0
        self.step_ms: deque[float] = deque(maxlen=500)
        self.spec_step_ms: deque[float] = deque(maxlen=500)
        self.plain_decode_step_ms: deque[float] = deque(maxlen=500)
        self.last_step_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ construction helpers
    @staticmethod
    def _load(cfg: ServeConfig) -> tuple[Any, Any]:
        if not cfg.model:
            raise ConfigError("ServeConfig.model is empty")
        from forge.models import loader

        dev = resolve_device(cfg.device)
        dt = resolve_dtype(cfg.dtype, dev)
        dtype_name = {"torch.bfloat16": "bfloat16", "torch.float16": "float16", "torch.float32": "float32"}[str(dt)]
        model, tok = loader.load_model(cfg.model, dtype=dtype_name, device=str(dev), revision=cfg.revision, quant=cfg.quant)
        li = getattr(model, "forge_load_info", None)
        if li and li.get("quantized"):
            log.info("loaded %s as packed %s: %s layers, %.2f GB weights (%.2f GB packed) from %s", cfg.display_name(), li.get("label"),
                     sum((li.get("layer_kinds") or {}).values()), li["weight_bytes"] / 1e9, li["packed_bytes"] / 1e9, li.get("path"))
            if cfg.kernel_linears == "auto" and dev.type == "cuda":
                conv = loader.convert_packed_for_kernels(model)
                li["kernel_convert"] = {k: conv[k] for k in ("converted", "kept", "avoided_duplicate_bytes")} | ({"errors": conv["errors"]} if conv["errors"] else {})
                log.info("kernel-layout W4A16 modules: %d converted (%.2f GB kernel layout replacing the portable copies; no lazy duplicate is cached), %d kept portable",
                         conv["converted"], conv["avoided_duplicate_bytes"] / 1e9, conv["kept"])
                for e in conv["errors"]:
                    log.warning("kernel conversion: %s", e)
        if cfg.tokenizer:
            tok = loader.load_tokenizer(cfg.tokenizer)
        return model, tok

    def _prepare_draft(self, draft_model: Any | None, drafter: Any | None) -> int:
        """Load the draft model (if one will be used) *before* the target KV pool is sized and return the bytes it reserves on the device
        (weights + its private KV pool), so the plan accounts for them instead of discovering the shortfall after the pool is allocated."""
        self._draft_model: Any | None = None
        self._draft_blocks = 0
        sc = self.cfg.speculation
        if drafter is not None or sc.mode not in ("draft", "auto") or (draft_model is None and not sc.draft_model):
            return 0
        if draft_model is None:
            from forge.models import loader

            draft_model, _ = loader.load_model(sc.draft_model, dtype=str(self.compute_dtype).replace("torch.", ""),
                                               device=str(self.device), quant=sc.draft_quant)
        draft_model.eval()
        draft_model.to(self.device)
        self._draft_model = draft_model
        dd = ModelAdapter(draft_model, None).dims
        cfg = self.cfg
        self._draft_blocks = draft_pool_blocks(max_num_seqs=cfg.max_num_seqs, max_model_len=cfg.max_model_len, block_size=cfg.block_size,
                                               kv_pool=cfg.kv_pool, kv_pool_cap_tokens=cfg.kv_pool_cap_tokens)
        per = kv_bytes_per_block(dd.num_layers, dd.num_kv_heads, dd.head_dim, cfg.block_size, self.kv_name)
        return _module_bytes(draft_model) + (self._draft_blocks + 1) * per

    def _build_drafter(self, sc: Any, draft_model: Any | None) -> Any:
        mode = sc.mode
        ngram = NgramDrafter(sc.ngram_max, sc.ngram_min, sc.ngram_window)
        model_drafter = None
        if mode in ("draft", "auto") and (draft_model is not None or sc.draft_model):
            if draft_model is None:  # defensive: _prepare_draft normally loaded it before the pool was sized
                draft_model = self._draft_model
            draft_model.eval()
            draft_model.to(self.device)
            model_drafter = DraftModelDrafter(
                draft_model, device=self.device, compute_dtype=self.compute_dtype, kv_name=self.kv_name,
                block_size=self.cfg.block_size, num_blocks=self._draft_blocks or max(16, self.pool.num_blocks // 4),
                max_model_len=self.cfg.max_model_len, kernel_attention=self.cfg.kernel_attention,
                expect_vocab=self.dims.vocab_size)
        elif mode == "draft":
            raise ConfigError("speculation.mode='draft' needs speculation.draft_model (or a draft_model instance)")
        if mode in ("eagle", "auto") and sc.eagle_head:
            from forge.serve.spec import EagleHead

            head = EagleHead.load(sc.eagle_head).to(self.device)
            model_drafter = model_drafter or EagleDrafter(head, self.adapter.embed, self.adapter.logits)
        elif mode == "eagle":
            raise ConfigError("speculation.mode='eagle' needs speculation.eagle_head")
        if mode == "ngram":
            return ngram
        if mode == "auto":
            return CompositeDrafter(ngram, model_drafter)
        return model_drafter

    def attach_eagle_head(self, head: Any) -> None:
        """Install an EAGLE head object at runtime (tests / online-trained heads) and switch speculation on."""
        if self.hybrid:
            raise ConfigError("speculative decoding is not supported for hybrid (conv) models: the conv state cannot be rolled back")
        self.drafter = EagleDrafter(head.to(self.device), self.adapter.embed, self.adapter.logits)
        self._need_hidden = True
        self.controller = self._build_controller()

    def _build_controller(self) -> SpecController:
        import torch

        sc = self.cfg.speculation
        wbytes = sum(p.numel() * p.element_size() for p in self.model.parameters()) + \
            sum(b.numel() * b.element_size() for b in self.model.buffers())
        mc = getattr(self.model, "config", None)
        params = None
        for attr in ("active_param_count", "param_count"):
            fn = getattr(mc, attr, None)
            if callable(fn):
                try:
                    params = float(fn())
                    break
                except Exception:
                    pass
        if params is None:
            params = float(sum(p.numel() for p in self.model.parameters()))
        if self.device.type == "cuda":
            from forge.serve.capacity import ridge_for_device_name

            ridge = ridge_for_device_name(torch.cuda.get_device_name(self.device))
        else:
            ridge = 20.0
        cost = StepCostModel(weight_bytes=float(wbytes), params=params, ridge_flop_per_byte=ridge)
        c = sc.draft_cost_ratio
        if c is None:
            dr = self.drafter
            if isinstance(dr, DraftModelDrafter):
                c = min(1.0, dr.weight_bytes / max(wbytes, 1))
            elif isinstance(dr, CompositeDrafter) and isinstance(dr.fallback, DraftModelDrafter):
                c = min(1.0, 0.5 * dr.fallback.weight_bytes / max(wbytes, 1))
            else:
                c = getattr(dr, "cost_ratio", 0.02)
        return SpecController(sc.max_gamma, cost=cost, draft_cost_ratio=float(c), hysteresis=sc.hysteresis,
                              ema_decay=sc.ema_decay, prior_acceptance=sc.prior_acceptance,
                              max_batch_for_spec=sc.max_batch_for_spec, fixed=(sc.controller == "fixed"))

    # ------------------------------------------------------------------ public API
    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None, request_id: str | None = None,
                    priority: int = 0, arrival_time: float | None = None, salt: str | None = None) -> str:
        prompt_ids = [int(t) for t in prompt_ids]
        if params is None:
            params = SamplingParams.from_defaults(self.cfg.sampling)
        if not prompt_ids:
            raise ConfigError("empty prompt")
        if len(prompt_ids) >= self.cfg.max_model_len:
            raise ConfigError(f"prompt has {len(prompt_ids)} tokens; max_model_len is {self.cfg.max_model_len}")
        if any(t < 0 or t >= self.dims.vocab_size for t in prompt_ids):
            raise ConfigError("prompt contains token ids outside the vocabulary")
        need_blocks = -(-(len(prompt_ids) + 1) // self.cfg.block_size)
        if need_blocks > self.pool.num_blocks:
            raise ConfigError(f"prompt needs {need_blocks} KV blocks but the pool only has {self.pool.num_blocks}")
        rid = request_id or f"req-{next(_auto_id)}"
        if rid in self._groups or rid in self._reqs:
            raise ConfigError(f"duplicate request id {rid}")
        now = arrival_time if arrival_time is not None else self.clock()
        children = []
        for i in range(params.n):
            cid = rid if params.n == 1 else f"{rid}:{i}"
            child_params = params.clone(n=1, seed=(params.seed + i) if params.seed is not None else None)
            req = Request(cid, prompt_ids, child_params, priority=priority, arrival_time=now, group_id=rid, index=i,
                          salt=salt if salt is not None else self.cfg.cache_salt)
            req.params.max_tokens = min(req.params.max_tokens, self.cfg.max_model_len - len(prompt_ids))
            req.params.min_tokens = min(req.params.min_tokens, req.params.max_tokens)
            children.append(req)
        with self._lock:
            self._groups[rid] = [c.request_id for c in children]
            for c in children:
                self._reqs[c.request_id] = c
                self._incoming.append(c)
        return rid

    def abort(self, request_id: str) -> None:
        with self._lock:
            for cid in self._groups.get(request_id, [request_id]):
                self._aborts.append(cid)

    def has_unfinished(self) -> bool:
        return bool(self._incoming or self._aborts or self.scheduler.num_unfinished)

    # ------------------------------------------------------------------ step
    def step(self) -> list[RequestOutput]:
        t_start = self.clock()
        acc: dict[str, dict[str, Any]] = {}
        self._drain(acc)
        self._sched_batch = max(1, len(self.scheduler.running))
        sched = self.scheduler.schedule(self._gamma_for)
        for dropped in sched.dropped:
            self._finish(dropped, dropped.finish_reason or "length", acc)
        for pre in sched.preempted:
            pre.last_hidden = None
            if self.drafter is not None:
                self.drafter.release(pre.request_id)
        if sched.scheduled:
            self._run(sched.scheduled, acc)
        else:
            self.last_step_info = {}  # idle step (e.g. waiting for KV memory): lets the serving loop back off
        outs = self._collect(acc)
        if self.hybrid:
            self.runner.prune_state(self._reqs)  # drop conv states of finished / aborted requests
        if sched.scheduled:
            dt = (self.clock() - t_start) * 1000.0
            self.step_ms.append(dt)
            self.num_steps += 1
            self.last_step_info = {"scheduled": len(sched.scheduled), "tokens": sched.num_batched_tokens,
                                   "preempted": len(sched.preempted), "step_ms": dt}
        return outs

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams] | None = None) -> list[RequestOutput]:
        """Blocking helper: run prompts to completion; returns the final output per prompt (n=1)."""
        ids = []
        for i, p in enumerate(prompts):
            sp = params[i] if isinstance(params, list) else params
            ids.append(self.add_request(p, sp))
        final: dict[tuple[str, int], RequestOutput] = {}
        lps: dict[tuple[str, int], list[dict]] = {}
        while self.has_unfinished():
            for o in self.step():
                key = (o.request_id, o.index)
                if o.logprobs:
                    lps.setdefault(key, []).extend(o.logprobs)
                if o.finished:
                    o.logprobs = lps.get(key) if o.logprobs is not None else None
                    o.new_token_ids, o.new_text = list(o.token_ids), o.text
                    final[key] = o
        return [final[(i, 0)] for i in ids]

    # ------------------------------------------------------------------ internals: queues
    def _drain(self, acc: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            incoming = list(self._incoming)
            self._incoming.clear()
            aborts = list(self._aborts)
            self._aborts.clear()
        for r in incoming:
            self.total_prompt_tokens += len(r.prompt_ids)
            r.detok = IncrementalDetokenizer(self.tokenizer, r.prompt_ids, r.params.stop,
                                             r.params.include_stop_in_output, r.params.skip_special_tokens) \
                if self.tokenizer is not None else None
            self.scheduler.add(r)
        for rid in aborts:
            r = self._reqs.get(rid)
            if r is None or r.finished:
                continue
            self._finish(r, "abort", acc)

    def _gamma_for(self, req: Request) -> int:
        if self.drafter is None or self.controller is None:
            return 0
        mg = self.cfg.speculation.max_gamma
        mg = min(mg, req.params.max_tokens - req.num_output - 1, self.cfg.max_model_len - len(req.tokens) - 1)
        if mg <= 0:
            return 0
        batch = self._sched_batch
        kv_bytes = 0.0
        if self.controller.cost is not None:
            per_tok = self.pool.bytes_per_block / self.pool.block_size
            kv_bytes = per_tok * sum(len(r.tokens) for r in self.scheduler.running)
        return self.controller.choose_gamma("greedy" if req.params.greedy else "sampled", batch, kv_bytes, mg)

    # ------------------------------------------------------------------ internals: run one step
    def _gen(self, req: Request) -> Any:
        if req.generator is None:
            seed = req.params.seed
            if seed is None and self.cfg.seed:
                seed = self.cfg.seed + req.order
            req.generator = make_generator(seed, self.device)
        return req.generator

    def _run(self, scheduled: list[ScheduledSeq], acc: dict[str, dict[str, Any]]) -> None:
        import torch

        for s in scheduled:
            if s.cow_copies:
                self.runner.copy_blocks(s.cow_copies)

        # --- drafting
        proposals: dict[str, Any] = {}
        dreqs = []
        for s in scheduled:
            if s.is_decode and s.gamma > 0:
                r = s.req
                dreqs.append(DraftRequest(r.request_id, r.tokens, s.gamma, r.params, len(r.prompt_ids), self._gen(r), r.last_hidden))
        t_draft = 0.0
        if dreqs:
            t0 = self.clock()
            for dr, prop in zip(dreqs, self.drafter.propose(dreqs)):
                proposals[dr.seq_id] = prop
            t_draft = self.clock() - t0

        inputs: list[SeqInput] = []
        drafts: dict[str, list[int]] = {}
        for s in scheduled:
            r = s.req
            if s.is_decode:
                d: list[int] = []
                prop = proposals.get(r.request_id)
                if prop is not None:
                    d = [int(x) for x in prop.tokens[: s.gamma]]
                if len(d) < s.gamma:
                    self.manager.rollback(r.request_id, r.num_computed + 1 + len(d))
                drafts[r.request_id] = d
                toks = [r.tokens[r.num_computed]] + d
                lf = 0 if d else -1
            else:
                toks = r.tokens[r.num_computed: r.num_computed + s.num_tokens]
                lf = -1 if s.will_sample else len(toks)
            inputs.append(SeqInput(toks, r.num_computed, self.manager.block_table(r.request_id), lf, state_key=r.request_id))

        t0 = self.clock()
        with torch.no_grad():
            ro = self.runner.forward(inputs, want_hidden=self._need_hidden)
        t_fwd = self.clock() - t0

        # --- batched greedy fast path: one argmax + one host sync for all plain-greedy rows
        pre: dict[str, int] = {}
        cand = [(s.req.request_id, lg[-1]) for s, lg in zip(scheduled, ro.logits)
                if lg is not None and not s.req.finished and not (s.is_decode and drafts.get(s.req.request_id))
                and self._plain_greedy(s.req)]
        if len(cand) > 1:
            pre = dict(zip([c[0] for c in cand], torch.stack([c[1] for c in cand]).argmax(-1).tolist()))

        # --- post-process
        any_spec = False
        for s, inp, logits, hid in zip(scheduled, inputs, ro.logits, ro.hidden):
            r = s.req
            if r.finished:
                continue
            if not s.is_decode:
                r.num_computed += len(inp.token_ids)
                self.manager.commit_tokens(r.request_id, r.tokens[: r.num_computed])
                if logits is not None:
                    row = logits[-1]
                    tok = pre[r.request_id] if r.request_id in pre else self._sample(r, row, [], 0)
                    if hid is not None:
                        r.last_hidden = hid[-1].detach().clone()
                    self._emit(r, [tok], [row], acc)
                continue
            d = drafts[r.request_id]
            if not d:
                row = logits[0]
                tok = pre[r.request_id] if r.request_id in pre else self._sample(r, row, [], 0)
                r.num_computed += 1
                self.manager.commit_tokens(r.request_id, r.tokens[: r.num_computed])
                if hid is not None:
                    r.last_hidden = hid[0].detach().clone()
                self._emit(r, [tok], [row], acc)
                continue
            any_spec = True
            emitted, n_acc = self._verify(r, logits, d, proposals[r.request_id])
            if self.controller is not None:
                self.controller.observe("greedy" if r.params.greedy else "sampled", len(d), n_acc)
            r.spec_proposed += len(d)
            r.spec_accepted += n_acc
            valid = r.tokens[: r.num_computed + 1] + d[:n_acc]  # pending token + accepted drafts are now in the KV cache
            r.num_computed += 1 + n_acc
            self.manager.rollback(r.request_id, r.num_computed)
            self.manager.commit_tokens(r.request_id, valid)
            if hid is not None:
                r.last_hidden = hid[n_acc].detach().clone()
            self._emit(r, emitted, [logits[i] for i in range(len(emitted))], acc)
        if dreqs and self.controller is not None and t_fwd > 0 and isinstance(self.drafter, (DraftModelDrafter, CompositeDrafter)):
            self.controller.update_draft_cost(t_draft / max(t_fwd, 1e-9) / max(1, max(s.gamma for s in scheduled)), ema=0.05)
        total_ms = (t_fwd + t_draft) * 1000.0
        if any_spec:
            self.spec_step_ms.append(total_ms)
        elif all(s.is_decode for s in scheduled):
            self.plain_decode_step_ms.append(total_ms)

    def _plain_greedy(self, req: Request) -> bool:
        p = req.params
        return p.greedy and not p.needs_history and not p.logit_bias and not self._suppressed(req, 0)

    def _suppressed(self, req: Request, n_extra: int) -> bool:
        return bool(self._eos_list) and (req.num_output + n_extra) < req.params.min_tokens

    def _prep(self, req: Request, row: Any, n_extra: int) -> Any:
        if self._suppressed(req, n_extra):
            row = row.clone()
            row[self._eos_list] = float("-inf")
        return row

    def _sample(self, req: Request, row: Any, extra: list[int], n_extra: int) -> int:
        p = req.params
        row2 = self._prep(req, row, n_extra)
        if p.greedy and not p.needs_history and not p.logit_bias:
            return int(row2.argmax().item())
        probs = sampling_probs(row2, p, req.prompt_ids, req.output_ids + extra)
        return sample_from_probs(probs, self._gen(req))

    def _verify(self, req: Request, logits: Any, draft: list[int], prop: Any) -> tuple[list[int], int]:
        """Exact verification of a draft chain. ``logits [len(draft)+1, V]``. Returns (emitted tokens, n_accepted)."""
        import torch

        p = req.params
        g = len(draft)
        simple = p.greedy and not p.needs_history and not p.logit_bias and not self._suppressed(req, g)
        if simple:
            top = torch.argmax(logits, dim=-1).tolist()
            n = 0
            while n < g and draft[n] == top[n]:
                n += 1
            return list(draft[:n]) + [int(top[n])], n
        rows = []
        for i in range(g + 1):
            rows.append(sampling_probs(self._prep(req, logits[i], i), p, req.prompt_ids, req.output_ids + draft[:i]))
        pmat = torch.stack(rows)
        q = prop.probs if prop is not None else None
        if q is not None:
            q = q[:g]
        return rejection_sample(pmat, q, draft, self._gen(req))

    def _emit(self, req: Request, tokens: list[int], rows: list[Any], acc: dict[str, dict[str, Any]]) -> None:
        """Append sampled tokens to the request, apply stop conditions, record outputs."""
        now = self.clock()
        a = acc.setdefault(req.request_id, {"ids": [], "text": "", "logprobs": []})
        want_lp = req.params.logprobs is not None
        n_before = req.num_output
        for i, tok in enumerate(tokens):
            req.tokens.append(int(tok))
            a["ids"].append(int(tok))
            if want_lp and i < len(rows):
                a["logprobs"].append(token_logprobs(rows[i], int(tok), req.params.logprobs or 0))
            self.total_generated_tokens += 1
            reason = None
            is_eos = tok in self.eos_ids and not req.params.ignore_eos
            if req.detok is not None and not is_eos:
                txt = req.detok.append([int(tok)])
                a["text"] += txt
                if req.detok.stopped:
                    reason = "stop"
            if is_eos and req.num_output >= req.params.min_tokens:
                reason = "stop"
            elif int(tok) in req.params.stop_token_ids and req.num_output >= req.params.min_tokens:
                reason = "stop"
            if reason is None:
                if req.num_output >= req.params.max_tokens or len(req.tokens) >= self.cfg.max_model_len:
                    reason = "length"
            if reason is not None:
                self._record_latency(req, now, req.num_output - n_before)
                self._finish(req, reason, acc)
                return
        self._record_latency(req, now, len(tokens))

    def _record_latency(self, req: Request, now: float, n_new: int) -> None:
        if n_new <= 0:
            return
        n_total = n_new
        if req.first_token_time is None:
            req.first_token_time = now
            self._ttft.append((now - req.arrival_time) * 1000.0)
            n_new -= 1
            req.last_token_time = now
        if n_new > 0 and req.last_token_time is not None:
            per = (now - req.last_token_time) * 1000.0 / n_new
            for _ in range(n_new):
                self._itl.append(per)
        req.last_token_time = now
        self._tok_events.append((now, n_total))

    def _finish(self, req: Request, reason: str, acc: dict[str, dict[str, Any]]) -> None:
        if req.finished:
            return
        if req.detok is not None:
            tail = req.detok.flush()
            if tail:
                acc.setdefault(req.request_id, {"ids": [], "text": "", "logprobs": []})["text"] += tail
        self.scheduler.finish(req, reason)
        if self.drafter is not None:
            self.drafter.release(req.request_id)
        self.total_finished += 1
        acc.setdefault(req.request_id, {"ids": [], "text": "", "logprobs": []})["finished"] = True

    def _collect(self, acc: dict[str, dict[str, Any]]) -> list[RequestOutput]:
        outs: list[RequestOutput] = []
        for rid, a in acc.items():
            r = self._reqs[rid]
            finished = bool(a.get("finished"))
            if not (a["ids"] or a["text"] or finished):
                continue
            text = r.detok.text[: r.detok.released] if r.detok is not None else ""
            ro = RequestOutput(
                request_id=r.group_id or r.request_id, index=r.index, new_token_ids=list(a["ids"]), new_text=a["text"],
                token_ids=list(r.output_ids), text=text, finished=finished, finish_reason=r.finish_reason if finished else None,
                logprobs=list(a["logprobs"]) if r.params.logprobs is not None else None, prompt_tokens=len(r.prompt_ids),
                cached_tokens=r.num_cached_prompt, completion_tokens=r.num_output,
            )
            if finished:
                ro.metrics = {
                    "ttft_ms": ((r.first_token_time - r.arrival_time) * 1000.0) if r.first_token_time else None,
                    "e2e_ms": ((r.finish_time - r.arrival_time) * 1000.0) if r.finish_time else None,
                    "preemptions": r.num_preemptions, "spec_proposed": r.spec_proposed, "spec_accepted": r.spec_accepted,
                }
                with self._lock:
                    self._reqs.pop(rid, None)
                    g = self._groups.get(r.group_id or "")
                    if g is not None:
                        if rid in g:
                            g.remove(rid)
                        if not g:
                            self._groups.pop(r.group_id or "", None)
            outs.append(ro)
        return outs

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        now = self.clock()
        while self._tok_events and now - self._tok_events[0][0] > 10.0:
            self._tok_events.popleft()
        win = (now - self._tok_events[0][0]) if self._tok_events else 0.0
        toks = sum(n for _, n in self._tok_events)
        tps = toks / max(win, 1.0) if self._tok_events else 0.0
        kv = self.manager.stats()
        torch = self._torch
        gpu_gb = 0.0
        if self.device.type == "cuda":
            try:
                gpu_gb = torch.cuda.memory_allocated(self.device) / 1e9
            except Exception:  # pragma: no cover
                pass
        spec = self.controller.stats() if self.controller is not None else None
        ttft, itl = self._snap(self._ttft), self._snap(self._itl)
        return {
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting) + len(self._incoming),
            "swapped": len(self.scheduler.swapped),
            "kv_usage": kv["usage"],
            "kv": kv,
            "prefix_hit_rate": kv["hit_rate"],
            "tokens_per_s": tps,
            "ttft_ms_p50": _pct(ttft, 0.5), "ttft_ms_p99": _pct(ttft, 0.99),
            "itl_ms_p50": _pct(itl, 0.5), "itl_ms_p99": _pct(itl, 0.99),
            "spec": spec,
            "spec_accept_rate": (spec or {}).get("overall_accept_rate"),
            "spec_acceptance_by_position": (spec or {}).get("acceptance_by_position"),
            "spec_verify_ms_avg": (sum(self.spec_step_ms) / len(self.spec_step_ms)) if self.spec_step_ms else None,
            "plain_decode_step_ms_avg": (sum(self.plain_decode_step_ms) / len(self.plain_decode_step_ms)) if self.plain_decode_step_ms else None,
            "gpu_mem_gb": gpu_gb,
            "model": dict(self.identity),
            "memory": self.memory.to_dict(),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "total_finished": self.total_finished,
            "preemptions": self.scheduler.total_preemptions,
            "steps": self.num_steps,
            "runner": dict(self.runner.stats),
            "num_blocks": self.pool.num_blocks,
            "block_size": self.pool.block_size,
            "kv_dtype": self.kv_name,
            "last_step": dict(self.last_step_info),
        }

    @staticmethod
    def _snap(dq: Any) -> list[float]:
        for _ in range(5):
            try:
                return list(dq)
            except RuntimeError:  # deque mutated by the engine thread
                continue
        return []

    def shutdown(self) -> None:
        self.manager.shutdown()
