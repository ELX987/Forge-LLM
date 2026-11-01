"""``parallel.strategy = zero_ternary``: row-sharded optimiser state and update for ternary QAT, with a packed 2-bit code all-gather.

Every ternary linear (``fq_t158``) is split into contiguous row ranges, one per rank.  Scale groups run along K, so a row range
quantises independently of the other rows and the result is bit-identical to the unsharded layer.  Per optimiser step:

1. every rank back-propagates its own micro-batches into the full-size ``weight.grad`` (the autograd leaf; local, no collectives);
2. :meth:`ZeroTernary.sync_grads` averages the gradients across ranks and keeps only each rank's row range
   (``reduce_scatter`` on NCCL, ``all_reduce`` + slice on gloo); non-ternary parameters (embeddings, norms, head) are all-reduced;
3. :meth:`ZeroTernary.clip_grad_norm` clips by the global norm with one scalar all-reduce and one host sync;
4. the optimiser updates the rank's latent rows and their Adam state (a view of ``weight`` -- the update is in place);
5. :meth:`ZeroTernary.post_step` re-quantises the rank's rows and all-gathers the **packed 2-bit codes** and group scales
   (8x less traffic than bf16 weights) so every rank holds identical codes; the forward decodes them
   (:class:`ZeroTernaryLinear`).

What is and is not sharded (v1, be precise): the optimiser state (Adam ``m`` and ``v``) and the update compute are sharded by ``1/world``.
The latent ``weight`` stays allocated at full size on every rank because it is the autograd leaf that receives ``dW``; the rows a rank
does not own are stale and unused (the forward reads the gathered codes).  Dropping that copy needs a grad-carrier-free autograd
function and is a documented follow-up.  While the quantisation warm-up blend ``W + lam * (Q(W) - W)`` is active (``lam < 1``) and
around high-precision shadow/eval forwards the latent rows are all-gathered too, so those paths equal the unsharded trainer.

Checkpoints are shard-aware: each rank writes ``shards/rank_XXXXX.safetensors`` (latent rows + optimiser state, no pickle) next to
``state.pt`` (everything replicated), with a manifest of row ranges, so a run can resume on a different number of ranks.

EXPERIMENTAL: verified on CPU with gloo (sharded == unsharded within float tolerance, resume on another world size).  The NCCL
``reduce_scatter`` path uses the same math but has NOT been executed on a GPU host (validation gate G1).
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError
from forge.formats import ternary as tn
from forge.train.optim import build_optimizer

from .linear import FakeQuantTernaryLinear

log = logging.getLogger("forge.ternary.zero")

__all__ = ["FORMAT", "SHARD_DIR", "ZeroOptimizer", "ZeroTernary", "ZeroTernaryLinear", "row_ranges", "verify_shard_files"]

FORMAT = "forge-zero-ternary-1"
SHARD_DIR = "shards"


def row_ranges(n: int, world: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal row ranges ``[(start, end), ...]`` covering ``n`` rows for ``world`` ranks (earlier ranks take the remainder)."""
    base, rem = divmod(int(n), int(world))
    out, a = [], 0
    for r in range(world):
        b = a + base + (1 if r < rem else 0)
        out.append((a, b))
        a = b
    return out


class ZeroTernaryLinear(FakeQuantTernaryLinear):
    """``fq_t158`` whose quantised weight is decoded from the engine's synchronised packed codes instead of re-quantising ``weight``."""

    @property
    def enabled(self) -> bool:
        return self.__dict__.get("_enabled", True)

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.__dict__["_enabled"] = bool(value)
        if not value:  # the high-precision forward reads ``weight`` itself: all rows must be current
            ref = self.__dict__.get("_zero")
            eng = ref() if ref is not None else None
            if eng is not None:
                eng.ensure_full_latents()

    def _wq(self, w: torch.Tensor) -> torch.Tensor:
        q = self.__dict__.get("_zero_q")
        if q is None:
            return super()._wq(w)
        t = tn.unpack_ternary(q, self.in_features)
        wq = tn.dequantize_ternary(t, self.__dict__["_zero_s"], self.group_size).to(w.dtype)
        return wq if self.lam >= 1.0 else w + self.lam * (wq - w)

    def codes(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.__dict__.get("_zero_q")
        if q is None:
            return super().codes()
        return tn.unpack_ternary(q, self.in_features), self.__dict__["_zero_s"]


@dataclass
class _Entry:
    name: str  # parameter name, "<module>.weight"
    mod: ZeroTernaryLinear
    n: int
    k: int
    ranges: list[tuple[int, int]]
    maxrows: int
    shard: nn.Parameter  # rows [r0, r1) of mod.weight, sharing its storage
    r0: int
    r1: int


def _dist() -> Any:
    import torch.distributed as dist

    return dist


def verify_shard_files(stage: Path, world: int, *, wait_s: float = 10.0) -> None:
    """Every rank's shard file must be visible: ranks write into one directory, so the checkpoint directory has to be shared storage."""
    import time

    want = [f"rank_{r:05d}.safetensors" for r in range(world)]
    deadline = time.time() + wait_s  # network filesystems may lag a moment behind the barrier
    while True:
        missing = [w for w in want if not (Path(stage) / w).exists()]
        if not missing:
            return
        if time.time() >= deadline:
            raise ConfigError("zero_ternary checkpoint shards from some ranks are not visible: checkpoint.dir must be on storage shared by all ranks "
                              "(multi-node runs need a shared filesystem)", details={"missing": missing, "dir": str(stage)})
        time.sleep(0.25)


class ZeroOptimizer:
    """Optimiser facade the Trainer drives like any other: replicated parameters + row-sharded ternary latents + the post-step code sync."""

    def __init__(self, engine: ZeroTernary, rep: Any | None, shard: Any) -> None:
        self.engine, self.rep, self.shard = engine, rep, shard

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return (list(self.rep.param_groups) if self.rep is not None else []) + list(self.shard.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.rep is not None:
            self.rep.zero_grad(set_to_none=set_to_none)
        self.shard.zero_grad(set_to_none=True)
        for e in self.engine.entries:
            e.mod.weight.grad = None

    def step(self, closure: Any = None) -> None:
        if self.rep is not None:
            self.rep.step()
        self.shard.step()
        self.engine.post_step()

    def state_dict(self) -> dict[str, Any]:
        """Replicated-parameter state only; the sharded state lives in ``shards/`` (see :meth:`ZeroTernary.write_shards`)."""
        return {"replicated": self.rep.state_dict() if self.rep is not None else None}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if self.rep is not None and sd.get("replicated") is not None:
            self.rep.load_state_dict(sd["replicated"])


class ZeroTernary:
    def __init__(self, model: nn.Module, precision: Any, dinfo: Any, parallel: Any, ternary: Any, *, save_optimizer: bool = True) -> None:
        self.model, self.precision, self.dinfo = model, precision, dinfo
        self.world, self.rank = int(dinfo.world), int(dinfo.rank)
        self.dist_on = bool(dinfo.initialized and self.world > 1)
        self.tc = ternary
        self.save_optimizer = save_optimizer
        self.entries: list[_Entry] = []
        self.opt: ZeroOptimizer | None = None
        self._latents_full = True
        self._rep_params: list[tuple[str, nn.Parameter]] = []
        self.comm = self._resolve_comm(parallel.zero_comm)
        self._attach()

    # ---------------------------------------------------------------------------------------------- setup
    def _resolve_comm(self, mode: str) -> str:
        if not self.dist_on or mode != "auto":
            return mode if mode != "auto" else "all_reduce"
        return "reduce_scatter" if _dist().get_backend() == "nccl" else "all_reduce"

    def _attach(self) -> None:
        if self.tc.group_size == 0:
            raise ConfigError("zero_ternary cannot shard a per-tensor scale (precision.ternary.group_size=0); use -1 (per row) or a group size")
        mods = self.precision._ternary_modules()
        if not mods:
            raise ConfigError("zero_ternary found no ternary layers to shard (precision.quant_aware must be 'ternary')")
        if self.dist_on:  # identical start state on every rank regardless of how the model was built
            dist = _dist()
            with torch.no_grad():
                for t in [*self.model.parameters(), *self.model.buffers()]:
                    dist.broadcast(t.data, src=0)
        seen: set[int] = set()
        for name, m in mods:
            w = m.weight
            if w.ndim != 2 or not w.is_contiguous():
                raise ConfigError(f"zero_ternary needs a contiguous 2-D weight, {name!r} has shape {tuple(w.shape)}")
            if w.data_ptr() in seen:
                raise ConfigError(f"zero_ternary does not support tied ternary weights ({name!r})")
            seen.add(w.data_ptr())
            n, k = w.shape
            if n < self.world:
                raise ConfigError(f"zero_ternary: layer {name!r} has {n} output rows, fewer than world_size={self.world}")
            ranges = row_ranges(n, self.world)
            r0, r1 = ranges[self.rank]
            m.__class__ = ZeroTernaryLinear
            m.__dict__["_zero"] = weakref.ref(self)
            m.__dict__["_zero_q"] = None
            m.__dict__["_zero_s"] = None
            m.__dict__["_enabled"] = m.__dict__.pop("enabled", True)
            shard = nn.Parameter(w.data[r0:r1], requires_grad=True)
            self.entries.append(_Entry(f"{name}.weight", m, n, k, ranges, max(b - a for a, b in ranges), shard, r0, r1))
        self._by_name = {e.name: e for e in self.entries}
        self._requantize_and_gather()

    def local_named_params(self, named: list[tuple[str, nn.Parameter]]) -> list[tuple[str, nn.Parameter]]:
        """``named_parameters()`` with every sharded ternary weight replaced by this rank's row shard."""
        return [(n, self._by_name[n].shard if n in self._by_name else p) for n, p in named]

    @property
    def param_names(self) -> set[str]:
        return set(self._by_name)

    def build_optimizer(self, named: list[tuple[str, nn.Parameter]], optim_cfg: Any, *, cuda: bool = False) -> ZeroOptimizer:
        rep = [(n, p) for n, p in named if n not in self._by_name and p.requires_grad]
        self._rep_params = rep
        shard = build_optimizer([(e.name, e.shard) for e in self.entries], optim_cfg, cuda=cuda)
        self.opt = ZeroOptimizer(self, build_optimizer(rep, optim_cfg, cuda=cuda) if rep else None, shard)
        return self.opt

    def describe(self) -> dict[str, Any]:
        numel = sum(e.n * e.k for e in self.entries)
        return {"strategy": "zero_ternary", "world": self.world, "comm": self.comm, "sharded_layers": len(self.entries), "sharded_params": numel,
                "replicated_params": sum(p.numel() for _, p in self._rep_params),
                "optimizer_state_bytes_per_rank_fp32_adam": int(8 * numel / self.world), "optimizer_state_bytes_unsharded_fp32_adam": 8 * numel,
                "code_bytes": int(numel // 4)}

    # ---------------------------------------------------------------------------------------------- collectives
    def _all_gather(self, buf: torch.Tensor) -> list[torch.Tensor]:
        if not self.dist_on:
            return [buf]
        outs = [torch.empty_like(buf) for _ in range(self.world)]
        _dist().all_gather(outs, buf.contiguous())
        return outs

    def _requantize_and_gather(self) -> None:
        """Quantise this rank's rows and all-gather packed codes + scales (one collective per layer, uint8 payload)."""
        for e in self.entries:
            t, s = tn.quantize_ternary(e.shard.data, e.mod.group_size, self.tc.rule, self.tc.iters)
            q = tn.pack_ternary(t)  # uint8 [rows, K/4]
            rows, kq = q.shape
            g = s.shape[1]
            pad = e.maxrows - rows
            if pad:
                q = torch.cat([q, q.new_zeros(pad, kq)])
                s = torch.cat([s, s.new_zeros(pad, g)])
            buf = torch.cat([q.reshape(-1), s.contiguous().view(torch.uint8).reshape(-1)])
            parts = self._all_gather(buf)
            qs, ss = [], []
            for (a, b), part in zip(e.ranges, parts):
                qs.append(part[: e.maxrows * kq].view(e.maxrows, kq)[: b - a])
                ss.append(part[e.maxrows * kq:].view(torch.float32).view(e.maxrows, g)[: b - a])
            e.mod.__dict__["_zero_q"] = torch.cat(qs).contiguous()
            e.mod.__dict__["_zero_s"] = torch.cat(ss).contiguous()
            e.mod._cache = None

    def gather_latents(self) -> None:
        """All-gather the latent rows so every rank's ``weight`` is current (warm-up blend, high-precision forwards, export)."""
        if self.dist_on:
            for e in self.entries:
                x = e.shard.data
                if x.shape[0] < e.maxrows:
                    x = torch.cat([x, x.new_zeros(e.maxrows - x.shape[0], e.k)])
                parts = self._all_gather(x.contiguous())
                with torch.no_grad():
                    for (a, b), part in zip(e.ranges, parts):
                        if (a, b) != (e.r0, e.r1):
                            e.mod.weight.data[a:b].copy_(part[: b - a])
                e.mod._cache = None
        self._latents_full = True

    def ensure_full_latents(self) -> None:
        if not self._latents_full:
            self.gather_latents()

    def _needs_latents_next(self) -> bool:
        w = int(getattr(self.tc, "warmup_steps", 0) or 0)
        return w > 0 and tn.warmup_lambda(int(getattr(self.precision, "_step", 0)) + 1, w) < 1.0

    def post_step(self) -> None:
        self._requantize_and_gather()
        self._latents_full = False
        if self._needs_latents_next():
            self.gather_latents()

    def resync(self) -> None:
        """Rebuild the codes (and gather the latents) after the module weights were loaded from a full, non-sharded checkpoint."""
        self._requantize_and_gather()
        self.gather_latents()

    def finalize(self) -> None:
        """Make the module's ``weight`` complete on every rank (call before exporting the model)."""
        self.ensure_full_latents()

    # ---------------------------------------------------------------------------------------------- gradients
    def sync_grads(self) -> None:
        """Average gradients over ranks; ternary weights keep only this rank's rows (``shard.grad``), the rest are all-reduced."""
        w = self.world
        for e in self.entries:
            full = e.mod.weight.grad
            if full is None:
                full = torch.zeros_like(e.mod.weight)
            if not self.dist_on:
                e.shard.grad = full[e.r0:e.r1].contiguous() if (e.r0, e.r1) != (0, e.n) else full
            elif self.comm == "reduce_scatter":
                if e.maxrows * w != e.n:
                    full = torch.cat([full, full.new_zeros(e.maxrows * w - e.n, e.k)])
                out = torch.empty(e.maxrows, e.k, dtype=full.dtype, device=full.device)
                _dist().reduce_scatter_tensor(out, full.contiguous(), op=_dist().ReduceOp.SUM)
                e.shard.grad = out[: e.r1 - e.r0].div_(w).contiguous()
            else:
                full = full.contiguous()
                _dist().all_reduce(full, op=_dist().ReduceOp.SUM)
                e.shard.grad = full[e.r0:e.r1].div(w).contiguous()
            e.mod.weight.grad = None
        if self.dist_on and self._rep_params:
            ps = [p for _, p in self._rep_params]
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in ps]
                             + [torch.tensor([0.0 if p.grad is None else 1.0 for p in ps], device=ps[0].device)])
            _dist().all_reduce(flat, op=_dist().ReduceOp.SUM)
            off = 0
            present = flat[-len(ps):]
            for i, p in enumerate(ps):
                nm = p.numel()
                if float(present[i]) > 0:
                    p.grad = (flat[off: off + nm] / w).view_as(p).to(p.dtype)
                off += nm

    def _all_grads(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        shard = [e.shard.grad for e in self.entries if e.shard.grad is not None]
        rep = [p.grad for _, p in self._rep_params if p.grad is not None]
        return shard, rep

    def clip_grad_norm(self, max_norm: float) -> float:
        """Global-norm clip over sharded + replicated gradients: one scalar all-reduce and one host sync.  Returns the pre-clip norm."""
        shard, rep = self._all_grads()
        if not shard and not rep:
            return 0.0
        dev = (shard or rep)[0].device
        sq = torch.zeros((), device=dev, dtype=torch.float32)
        for g in shard:
            sq = sq + g.float().pow(2).sum()
        if self.dist_on:
            _dist().all_reduce(sq, op=_dist().ReduceOp.SUM)
        for g in rep:
            sq = sq + g.float().pow(2).sum()
        norm = float(sq.sqrt())
        if max_norm > 0 and math.isfinite(norm):
            coef = max_norm / (norm + 1e-6)
            if coef < 1.0:
                for g in [*shard, *rep]:
                    g.mul_(coef)
        return norm

    # ---------------------------------------------------------------------------------------------- checkpoints
    def manifest(self) -> dict[str, Any]:
        keys: list[str] = []
        if self.save_optimizer and self.opt is not None:
            for e in self.entries:
                st = self.opt.shard.state.get(e.shard, {})
                keys = sorted(st)
                break
        return {"format": FORMAT, "world": self.world, "optim_keys": keys,
                "params": {e.name: {"n": e.n, "k": e.k, "ranges": [list(r) for r in e.ranges]} for e in self.entries}}

    def write_shards(self, root: Path, step: int) -> Path:
        """Collective: every rank writes its shard file into a staging directory under ``root``; returns the directory (to attach to the checkpoint)."""
        stage = Path(root) / f".zstage_{step:08d}"
        dist = _dist() if self.dist_on else None
        if self.rank == 0:
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True, exist_ok=True)
        if dist is not None:
            dist.barrier()
        stage.mkdir(parents=True, exist_ok=True)  # no-op on a shared directory; keeps a non-shared one from failing with an opaque I/O error
        from safetensors.torch import save_file

        man = self.manifest()
        tensors: dict[str, torch.Tensor] = {}
        for e in self.entries:
            tensors[f"latent/{e.name}"] = e.shard.data.detach().to("cpu", copy=True).contiguous()
            if man["optim_keys"]:
                st = self.opt.shard.state.get(e.shard, {})  # type: ignore[union-attr]
                for key in man["optim_keys"]:
                    v = st.get(key)
                    if v is None:
                        continue
                    v = torch.as_tensor(v).detach().to("cpu", copy=True)
                    if v.shape == e.shard.shape:
                        tensors[f"state/{key}/{e.name}"] = v.contiguous()
                    else:
                        tensors[f"scalar/{key}/{e.name}"] = v.reshape(1).contiguous()
        save_file(tensors, str(stage / f"rank_{self.rank:05d}.safetensors"), metadata={"format": FORMAT, "rank": str(self.rank), "world": str(self.world)})
        if self.rank == 0:
            (stage / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        if dist is not None:
            dist.barrier()
            verify_shard_files(stage, self.world)
        return stage

    def load_shards(self, ckpt_dir: Path, manifest: dict[str, Any], *, with_optim: bool = True) -> None:
        """Collective: restore this rank's latent rows (and optimiser state) from a checkpoint written with any number of ranks."""
        from safetensors import safe_open

        if manifest.get("format") != FORMAT:
            raise ConfigError("not a zero_ternary shard manifest", details={"format": manifest.get("format")})
        shard_dir = Path(ckpt_dir) / SHARD_DIR
        old_world = int(manifest["world"])
        params = manifest["params"]
        if set(params) != set(self._by_name):
            raise ConfigError("checkpoint shards do not match this model's ternary layers",
                              details={"missing": sorted(set(self._by_name) - set(params))[:8], "unexpected": sorted(set(params) - set(self._by_name))[:8]})
        files = [shard_dir / f"rank_{j:05d}.safetensors" for j in range(old_world)]
        for f in files:
            if not f.exists():
                raise ConfigError(f"checkpoint shard file missing: {f}")

        def read_rows(key: str, ranges: list[list[int]], r0: int, r1: int) -> torch.Tensor | None:
            out = None
            for j, (a, b) in enumerate(ranges):
                lo, hi = max(a, r0), min(b, r1)
                if lo >= hi:
                    continue
                with safe_open(str(files[j]), "pt") as fh:
                    piece = fh.get_slice(key)[lo - a: hi - a]
                if out is None:
                    out = torch.empty((r1 - r0, *piece.shape[1:]), dtype=piece.dtype)
                out[lo - r0: hi - r0] = piece
            return out

        def read_scalar(key: str) -> torch.Tensor:
            with safe_open(str(files[0]), "pt") as fh:
                return fh.get_tensor(key)

        keys = list(manifest.get("optim_keys") or []) if (with_optim and self.save_optimizer and self.opt is not None) else []
        osd = self.opt.shard.state_dict() if keys else None  # type: ignore[union-attr]
        index = {}
        if osd is not None:
            for gi, g in enumerate(self.opt.shard.param_groups):  # type: ignore[union-attr]
                for pi, p in enumerate(g["params"]):
                    index[id(p)] = osd["param_groups"][gi]["params"][pi]
        for e in self.entries:
            info = params[e.name]
            if (info["n"], info["k"]) != (e.n, e.k):
                raise ConfigError(f"checkpoint shape mismatch for {e.name}: {info['n']}x{info['k']} vs {e.n}x{e.k}")
            lat = read_rows(f"latent/{e.name}", info["ranges"], e.r0, e.r1)
            assert lat is not None
            with torch.no_grad():
                e.shard.data.copy_(lat.to(e.shard.device, e.shard.dtype))
            if osd is not None:
                st: dict[str, Any] = {}
                for key in keys:
                    sk = f"state/{key}/{e.name}"
                    try:
                        st[key] = read_rows(sk, info["ranges"], e.r0, e.r1)
                    except Exception:  # noqa: BLE001 - not a row-sliced entry
                        st[key] = None
                    if st[key] is None:
                        st[key] = read_scalar(f"scalar/{key}/{e.name}").reshape(())
                osd["state"][index[id(e.shard)]] = st
        if osd is not None:
            self.opt.shard.load_state_dict(osd)  # type: ignore[union-attr]
        self._requantize_and_gather()
        self.gather_latents()
