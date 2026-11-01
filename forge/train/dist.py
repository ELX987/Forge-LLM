"""Distributed training helpers: torchrun environment, DDP / FSDP wrapping, state-dict gathering, torchrun launcher.

Everything here degrades to a no-op for single-process runs.  The DDP / FSDP / NCCL paths are implemented against the public
``torch.distributed`` APIs but have NOT been executed in this repository (no multi-GPU host available at authoring time);
single-process CPU behaviour is unit-tested.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError, ForgeError
from forge.train.config import ParallelConfig

log = logging.getLogger("forge.train.dist")

__all__ = [
    "DistInfo", "all_reduce_flag", "build_torchrun_command", "get_model_state", "get_optim_state", "init_distributed",
    "launch_torchrun", "load_model_state", "load_optim_state", "shutdown_distributed", "under_torchrun", "unwrap", "wrap_model",
]


@dataclass
class DistInfo:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    initialized: bool = False
    strategy: str = "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def _pick_device(pref: str, local_rank: int) -> torch.device:
    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ConfigError("parallel.device=cuda but CUDA is not available")
    n = torch.cuda.device_count()
    return torch.device("cuda", local_rank % max(1, n))


def init_distributed(parallel: ParallelConfig) -> DistInfo:
    """Initialise ``torch.distributed`` when launched by torchrun (WORLD_SIZE>1); otherwise a single-process :class:`DistInfo`."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _pick_device(parallel.device, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if not under_torchrun():
        return DistInfo(0, 1, 0, device, False, "none" if parallel.strategy == "none" else parallel.strategy)
    import torch.distributed as dist

    backend = parallel.backend
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank, device, True, parallel.strategy)


def shutdown_distributed() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def all_reduce_flag(flag: bool, dinfo: DistInfo, op: str = "max") -> bool:
    """Logical OR across ranks (cooperative stop / anomaly agreement)."""
    if not dinfo.initialized:
        return bool(flag)
    import torch.distributed as dist

    t = torch.tensor([1.0 if flag else 0.0], device=dinfo.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.MIN)
    return bool(t.item() > 0.5)


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP / FSDP / torch.compile wrappers."""
    m = model
    for _ in range(4):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m, nn.Module) and type(m).__name__ in ("DistributedDataParallel", "FullyShardedDataParallel", "DataParallel"):
            m = m.module
        else:
            break
    return m


def _layer_class(model: nn.Module) -> type | None:
    for path in ("layers", "model.layers", "transformer.h", "h", "blocks"):
        cur: Any = model
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if isinstance(cur, nn.ModuleList) and len(cur):
                return type(cur[0])
        except AttributeError:
            continue
    return None


def wrap_model(model: nn.Module, parallel: ParallelConfig, dinfo: DistInfo, autocast_dtype: torch.dtype | None = None) -> nn.Module:
    """Wrap ``model`` in DDP or FSDP according to ``parallel.strategy`` (no-op when single process)."""
    if not dinfo.initialized or parallel.strategy in ("none", "zero_ternary"):
        if dinfo.initialized and parallel.strategy == "none":
            log.warning("running under torchrun with parallel.strategy=none: ranks train independent replicas")
        return model  # zero_ternary does its own gradient exchange (forge.ternary.zero), so the model stays unwrapped
    if parallel.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        kw: dict[str, Any] = dict(find_unused_parameters=parallel.ddp_find_unused, bucket_cap_mb=parallel.ddp_bucket_cap_mb)
        if dinfo.device.type == "cuda":
            kw["device_ids"] = [dinfo.device.index]
        return DDP(model, **kw)
    from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy

    layer_cls = _layer_class(model)
    if layer_cls is not None:
        policy = ModuleWrapPolicy({layer_cls})
    else:
        from functools import partial

        policy = partial(size_based_auto_wrap_policy, min_num_params=parallel.fsdp_min_params)
    mp = None
    if autocast_dtype is not None:
        mp = MixedPrecision(param_dtype=autocast_dtype, reduce_dtype=torch.float32, buffer_dtype=autocast_dtype)
    return FSDP(
        model, auto_wrap_policy=policy, sharding_strategy=getattr(ShardingStrategy, parallel.fsdp_sharding.upper()),
        cpu_offload=CPUOffload(offload_params=parallel.fsdp_cpu_offload), mixed_precision=mp, use_orig_params=True,
        device_id=dinfo.device if dinfo.device.type == "cuda" else None)


def _is_fsdp(model: nn.Module) -> bool:
    return type(model).__name__ == "FullyShardedDataParallel"


def get_model_state(model: nn.Module, names: set[str] | None = None) -> dict[str, torch.Tensor]:
    """CPU state dict of the underlying model (collective for FSDP: call on every rank; non-zero ranks get ``{}``).

    ``names`` restricts to the given keys (e.g. trainable adapter parameters).
    """
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            sd = model.state_dict()
        sd = {k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    else:
        sd = {k: v.detach().cpu() for k, v in unwrap(model).state_dict().items() if names is None or k in names}
    if names is not None:
        sd = {k: v for k, v in sd.items() if k in names}
    return sd


def load_model_state(model: nn.Module, sd: dict[str, torch.Tensor], strict: bool = True) -> Any:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            return model.load_state_dict(sd, strict=strict)
    return unwrap(model).load_state_dict(sd, strict=strict)


def get_optim_state(model: nn.Module, optimizer: Any) -> dict[str, Any]:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP.optim_state_dict(model, optimizer)
    return optimizer.state_dict()


def load_optim_state(model: nn.Module, optimizer: Any, osd: dict[str, Any]) -> None:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        osd = FSDP.optim_state_dict_to_load(model, optimizer, osd)
    optimizer.load_state_dict(osd)


# ---------------------------------------------------------------------------------------------------------------
# torchrun launcher
# ---------------------------------------------------------------------------------------------------------------
def build_torchrun_command(kind: str, config_path: str | Path, run_dir: str | Path, nproc_per_node: int, nnodes: int = 1,
                           python: str | None = None) -> list[str]:
    """argv that runs ``forge.train.launch`` under ``torch.distributed.run`` (``kind`` in train|finetune|distill)."""
    if kind not in ("train", "finetune", "distill"):
        raise ConfigError(f"bad launch kind {kind!r}")
    cmd = [python or sys.executable, "-m", "torch.distributed.run", "--standalone" if nnodes == 1 else "--rdzv_backend=c10d",
           f"--nproc_per_node={int(nproc_per_node)}", f"--nnodes={int(nnodes)}", "-m", "forge.train.launch",
           "--kind", kind, "--config", str(config_path), "--run-dir", str(run_dir)]
    return cmd


def launch_torchrun(kind: str, cfg: Any, ctx: Any, nproc_per_node: int | None = None) -> dict[str, Any]:
    """Run a job under torchrun in a subprocess, forwarding logs / metrics / outputs / cancellation through ``ctx``.

    Children write ``events.jsonl`` and ``result.json`` into the run dir (see :mod:`forge.train.launch`); this function tails
    them.  Cancellation: ``ctx.should_stop()`` creates ``<run_dir>/STOP`` which rank 0 polls cooperatively.
    """
    run_dir = Path(ctx.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "launch_config.json"
    cfg_path.write_text(cfg.model_dump_json(indent=1), encoding="utf-8")
    events = run_dir / "events.jsonl"
    for f in (events, run_dir / "STOP", run_dir / "result.json"):
        if f.exists():
            f.unlink()
    n = int(nproc_per_node or cfg.parallel.nproc_per_node)
    cmd = build_torchrun_command(kind, cfg_path, run_dir, n, cfg.parallel.nnodes)
    ctx.log("launching: " + " ".join(cmd))
    env = dict(os.environ)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            ctx.log(line.rstrip("\n"))

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    pos = 0
    stop_sent = False

    def drain() -> None:
        nonlocal pos
        if not events.exists():
            return
        with open(events, encoding="utf-8") as f:
            f.seek(pos)
            while True:
                line = f.readline()
                if not line or not line.endswith("\n"):
                    break
                pos = f.tell()
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("t")
                if t == "metric":
                    ctx.metric(int(ev["step"]), **ev["values"])
                elif t == "progress":
                    ctx.progress(float(ev["frac"]), ev.get("msg", ""))
                elif t == "output":
                    ctx.set_output(ev["key"], ev["value"])

    while proc.poll() is None:
        drain()
        if not stop_sent and ctx.should_stop():
            (run_dir / "STOP").write_text("stop")
            stop_sent = True
            ctx.log("stop requested; waiting for rank 0 to checkpoint and exit")
        time.sleep(0.5)
    th.join(timeout=5)
    drain()
    if proc.returncode != 0:
        raise ForgeError(f"torchrun exited with code {proc.returncode}", details={"cmd": cmd})
    res = run_dir / "result.json"
    return json.loads(res.read_text(encoding="utf-8")) if res.exists() else {}
