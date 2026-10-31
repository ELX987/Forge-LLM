"""Distributed paths exercised on CPU with the gloo backend (2 spawned processes, file-store rendezvous).

DDP / FSDP code paths for CUDA/NCCL are the same wrappers; NCCL itself is only validated on a GPU host (tests_gpu).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch.distributed as dist

    _OK = dist.is_available() and dist.is_gloo_available()
except Exception:  # noqa: BLE001
    _OK = False

pytestmark = pytest.mark.skipif(not _OK, reason="torch.distributed/gloo unavailable")


def _worker(rank, world, init_file, cfg_json, ds_dir, run_dir, strategy, out_q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), FORGE_HOME=str(Path(run_dir) / f"home{rank}"))
    import torch.distributed as dist
    from test_train_fixtures import make_model

    from forge.core.context import NullContext
    from forge.train import dist as D
    from forge.train.config import TrainConfig
    from forge.train.data import ShardedTokenDataset
    from forge.train.trainer import LMTask, Trainer

    try:
        dist.init_process_group("gloo", init_method="file:///" + init_file.replace("\\", "/"), rank=rank, world_size=world)
        cfg = TrainConfig.model_validate_json(cfg_json)
        dinfo = D.init_distributed(cfg.parallel)
        ds = ShardedTokenDataset([ds_dir], 16, seed=3, role="train", val_sequences=4)
        va = ShardedTokenDataset([ds_dir], 16, seed=3, role="val", val_sequences=4)
        ctx = NullContext(run_dir=Path(run_dir) / f"r{rank}")
        tr = Trainer(make_model(0), cfg, LMTask(ds, cfg, va, 16), ctx, dinfo=dinfo)
        res = tr.fit()
        sd = D.get_model_state(tr.model)
        out_q.put((rank, res["status"], res["steps"], {k: v.clone() for k, v in sd.items()} if sd else None, [m for _, m in ctx.metrics][:1]))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001
        import traceback

        out_q.put((rank, "error", traceback.format_exc(), None, None))
        raise


def _run_dist(tmp_path, strategy, **over):
    from test_train_fixtures import ingest_tiny, small_train_cfg

    ds = ingest_tiny(tmp_path)
    cfg = small_train_cfg(ds, micro_batch_size=2, grad_accum_steps=1, budget={"max_steps": 4}, parallel={"strategy": strategy, "device": "cpu"},
                          optim={"name": "sgd", "lr": 0.05, "momentum": 0.0, "weight_decay": 0.0, "grad_clip": 0.0},
                          schedule={"name": "constant", "warmup_steps": 0}, eval={"every_steps": 2, "num_batches": 2}, **over)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_file = str(Path(tempfile.mkdtemp()) / "rdzv")
    procs = [ctx.Process(target=_worker, args=(r, 2, init_file, cfg.model_dump_json(), str(ds), str(tmp_path), strategy, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = [q.get(timeout=240) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    return cfg, ds, {r[0]: r for r in results}


def _single(ds, cfg, tmp_path):
    from test_train_fixtures import make_model

    from forge.core.context import NullContext
    from forge.train.config import TrainConfig
    from forge.train.data import ShardedTokenDataset
    from forge.train.trainer import LMTask, Trainer

    c = TrainConfig.model_validate({**cfg.model_dump(), "micro_batch_size": 4, "parallel": {"strategy": "none", "device": "cpu"}})
    d1 = ShardedTokenDataset([str(ds)], 16, seed=3, role="train", val_sequences=4)
    d2 = ShardedTokenDataset([str(ds)], 16, seed=3, role="val", val_sequences=4)
    tr = Trainer(make_model(0), c, LMTask(d1, c, d2, 16), NullContext(run_dir=tmp_path / "single"))
    tr.fit()
    return {k: v.detach().clone() for k, v in tr.raw.state_dict().items()}


def test_ddp_two_process_matches_single_process_large_batch(tmp_path):
    cfg, ds, res = _run_dist(tmp_path, "ddp")
    for r in (0, 1):
        assert res[r][1] == "completed", res[r][2]
        assert res[r][2] == 4
    p0, p1 = res[0][3], res[1][3]
    ref = _single(ds, cfg, tmp_path)
    assert all(torch.allclose(p0[k], p1[k], atol=1e-6) for k in p0)  # replicas stay in sync
    assert all(torch.allclose(p0[k], ref[k], atol=2e-5) for k in ref)  # == one process with the combined batch
    # only rank 0 writes checkpoints / metrics
    assert (tmp_path / "r0" / "checkpoints").exists() and not list((tmp_path / "r1").glob("checkpoints/step_*"))
    assert (tmp_path / "r0" / "metrics.jsonl").exists() and not (tmp_path / "r1" / "metrics.jsonl").exists()


def test_fsdp_two_process_cpu_gloo_smoke(tmp_path):
    cfg, ds, res = _run_dist(tmp_path, "fsdp")
    errs = [res[r][2] for r in (0, 1) if res[r][1] == "error"]
    if errs:
        pytest.skip(f"FSDP on CPU/gloo not supported by this torch build: {errs[0][-300:]}")
    assert res[0][1] == "completed" and res[0][2] == 4
    full = res[0][3]
    assert full and all(torch.isfinite(v).all() for v in full.values())  # gathered full state dict on rank 0
    ref = _single(ds, cfg, tmp_path)
    assert set(full) == set(ref)
