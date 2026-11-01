#!/usr/bin/env python
"""Gate G1 harness: sharded (``zero_ternary``) vs unsharded ternary training on the same synthetic data, under torchrun.

    torchrun --standalone --nproc_per_node=2 scripts/zero_ternary_parity.py            # NCCL on a multi-GPU host
    torchrun --standalone --nproc_per_node=2 scripts/zero_ternary_parity.py --device cpu --backend gloo   # harness self-check (CPU)

Every rank trains the tiny ForgeLM sharded; rank 0 then trains the identical model unsharded on the combined batch and compares the loss
curve and the final weights.  Prints one JSON line (rank 0) and exits non-zero on a mismatch.  Tolerances are for fp32 summation-order noise
(SGD) - they are not a claim about bf16 runs.  The NCCL path has not been executed in this repository; the gloo path is CPU-tested.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.core.context import NullContext  # noqa: E402
from forge.models.tiny import build_tiny_model  # noqa: E402
from forge.train import dist as D  # noqa: E402
from forge.train.config import TrainConfig  # noqa: E402
from forge.train.trainer import LMTask, Trainer  # noqa: E402

VOCAB, SEQ = 288, 16


class _Data:
    """Deterministic synthetic next-token data: global row ``i`` of step ``s`` is the same for any world size."""

    def __len__(self) -> int:
        return 1 << 20

    def batch_at(self, step: int, global_batch: int, rank: int, world: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(1000 + step)
        ids = torch.randint(0, VOCAB, (global_batch, SEQ + 1), generator=g)
        per = global_batch // world
        rows = ids[rank * per:(rank + 1) * per]
        return {"input_ids": rows[:, :-1].contiguous(), "labels": rows[:, 1:].contiguous()}

    def eval_batches(self, mb: int, n: int) -> list[dict[str, torch.Tensor]]:
        return []

    def describe(self) -> dict[str, Any]:
        return {"synthetic": True}


def _cfg(strategy: str, device: str, steps: int, micro: int, ckpt_dir: str, seed: int) -> TrainConfig:
    return TrainConfig.model_validate({
        "name": "zparity", "seed": seed, "model": {"source": "tiny", "tokenizer": "byte"}, "micro_batch_size": micro, "grad_accum_steps": 1,
        "data": {"dataset": ["unused"], "seq_len": SEQ},
        "optim": {"name": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 0.0, "grad_clip": 0.1},
        "schedule": {"name": "constant", "warmup_steps": 0}, "budget": {"max_steps": steps},
        "precision": {"quant_aware": "ternary", "compute_dtype": "float32", "exempt_first_layers": 0, "exempt_last_layers": 0, "diagnostics_every": 0,
                      "loss_gap_every": 0, "ternary": {"group_size": 32}},
        "checkpoint": {"every_steps": steps, "keep_last": 1, "dir": ckpt_dir}, "eval": {"every_steps": 0}, "logging": {"log_every": 1},
        "parallel": {"strategy": strategy, "device": device, "backend": "auto"}, "save_final_model": False, "register_artifact": False})


def _model() -> torch.nn.Module:
    return build_tiny_model(7, vocab_size=VOCAB, num_layers=2, hidden_size=64, num_heads=4, num_kv_heads=4, intermediate_size=128)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--backend", default="auto", choices=["auto", "nccl", "gloo"])
    ap.add_argument("--atol", type=float, default=5e-5)
    ns = ap.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        print("run under torchrun with >= 2 processes", file=sys.stderr)
        return 2
    shared = os.environ.get("FORGE_PARITY_DIR") or str(Path(tempfile.gettempdir()) / f"forge_zero_parity_{os.environ.get('MASTER_PORT', '0')}")
    cfg = _cfg("zero_ternary", ns.device, ns.steps, ns.micro, str(Path(shared) / "ck"), 3)
    cfg.parallel.backend = ns.backend
    dinfo = D.init_distributed(cfg.parallel)
    try:
        ctx = NullContext(run_dir=Path(shared) / f"r{dinfo.rank}")
        t = Trainer(_model(), cfg, LMTask(_Data(), cfg, None, SEQ), ctx, dinfo=dinfo)
        res = t.fit()
        sharded = {k: v.detach().float().cpu() for k, v in t.raw.state_dict().items()}
        losses = [v["loss"] for _, v in ctx.metrics if "loss" in v]
        ok, report = True, {"world": world, "backend": cfg.parallel.backend, "steps": res["steps"], "status": res["status"], "sharded_losses": losses}
        if dinfo.is_main:
            ref_cfg = _cfg("none", ns.device, ns.steps, ns.micro * world, str(Path(shared) / "ck_ref"), 3)
            ref_dinfo = D.DistInfo(0, 1, 0, dinfo.device, False, "none")
            rctx = NullContext(run_dir=Path(shared) / "ref")
            rt = Trainer(_model(), ref_cfg, LMTask(_Data(), ref_cfg, None, SEQ), rctx, dinfo=ref_dinfo)
            rt.fit()
            ref = {k: v.detach().float().cpu() for k, v in rt.raw.state_dict().items()}
            rl = [v["loss"] for _, v in rctx.metrics if "loss" in v]
            wdiff = max(float((sharded[k] - ref[k]).abs().max()) for k in ref)
            ldiff = max(abs(a - b) for a, b in zip(losses, rl)) if losses and rl else float("inf")
            ok = wdiff <= ns.atol and ldiff <= 1e-3 and res["status"] == "completed"
            report.update({"reference_losses": rl, "max_weight_diff": wdiff, "max_loss_diff": ldiff, "atol": ns.atol, "ok": ok})
            print(json.dumps(report))
        flag = torch.tensor([1.0 if ok else 0.0], device=dinfo.device)
        import torch.distributed as dist

        dist.broadcast(flag, src=0)
        return 0 if float(flag) > 0.5 else 1
    finally:
        D.shutdown_distributed()


if __name__ == "__main__":
    sys.exit(main())
