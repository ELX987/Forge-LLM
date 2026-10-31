"""GPU validation of forge.train (skipped without CUDA). NOT run during development (no NVIDIA GPU available there).

Run on the GPU host:  pytest tests_gpu/test_train_gpu.py -q -x
Covers: CUDA autocast/bf16/fp16+GradScaler, fused AdamW, activation checkpointing memory, R1 backend selection, R2 emulation on GPU,
NF4 QLoRA on GPU, torch.compile smoke, CPU/GPU numerics parity of the FP4 emulation, 2-GPU DDP and FSDP (NCCL).
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():  # conftest also skips this directory; belt and braces for direct invocation
    pytest.skip("no CUDA device", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import torch.multiprocessing as mp  # noqa: E402
from test_train_fixtures import ingest_tiny, make_model, small_train_cfg  # noqa: E402

from forge.core.context import NullContext  # noqa: E402
from forge.train.data import ShardedTokenDataset  # noqa: E402
from forge.train.fp4sim import FP4Rng, fake_quant_nvfp4, round_e2m1_sr  # noqa: E402
from forge.train.trainer import LMTask, Trainer  # noqa: E402

pytestmark = pytest.mark.gpu
SM = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]


@pytest.fixture(scope="module")
def ds_dir(tmp_path_factory):
    return ingest_tiny(tmp_path_factory.mktemp("gpuds"))


def _run(ds_dir, tmp_path, model=None, **over):
    over.setdefault("parallel", {})["device"] = "cuda"
    cfg = small_train_cfg(ds_dir, **over)
    s = cfg.seed
    tr_ds = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="train", val_sequences=4)
    va_ds = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="val", val_sequences=4)
    ctx = NullContext(run_dir=tmp_path / "run")
    tr = Trainer(model or make_model(0, kind="forge", layers=4), cfg, LMTask(tr_ds, cfg, va_ds, cfg.data.seq_len), ctx)
    return tr, tr.fit(), ctx


def test_bf16_autocast_training_metrics(ds_dir, tmp_path):
    tr, res, ctx = _run(ds_dir, tmp_path, budget={"max_steps": 10})
    assert res["status"] == "completed" and math.isfinite(res["final_loss"])
    losses = [v for _, v in ctx.metric_series("loss")]
    assert losses[-1] < losses[0]
    assert all(v > 0 for _, v in ctx.metric_series("gpu_mem_gb"))
    mfu = ctx.metric_series("mfu")
    if mfu:  # present when the GPU is in the datasheet DB
        assert all(0 < v < 1.5 for _, v in mfu), mfu
    assert next(tr.raw.parameters()).device.type == "cuda"


def test_fused_adamw_on_cuda(ds_dir, tmp_path):
    tr, _, _ = _run(ds_dir, tmp_path, budget={"max_steps": 1})
    assert any(g.get("fused") for g in tr.opt.param_groups)


def test_fp16_uses_grad_scaler(ds_dir, tmp_path):
    tr, res, ctx = _run(ds_dir, tmp_path, budget={"max_steps": 6}, precision={"compute_dtype": "float16"})
    assert tr.scaler is not None and res["status"] == "completed"
    assert ctx.metric_series("loss_scale")


def test_activation_checkpointing_reduces_memory(ds_dir, tmp_path):
    def peak(ac):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _run(ds_dir, tmp_path / str(ac), budget={"max_steps": 2}, activation_checkpointing=ac, micro_batch_size=8, data={"seq_len": 128})
        return torch.cuda.max_memory_allocated()

    assert peak(True) < peak(False)


def test_r1_fp8_backend_and_training(ds_dir, tmp_path):
    tr, res, ctx = _run(ds_dir, tmp_path, budget={"max_steps": 8},
                        precision={"rung": "R1", "exempt_first_layers": 0, "exempt_last_layers": 0, "loss_gap_every": 4, "diagnostics_every": 0})
    be = res["precision"]["backend"]
    print("R1 backend:", be)
    if SM >= 89:
        assert be in ("torchao", "transformer_engine", "emulated")  # torchao/TE only if installed
    else:
        assert be == "emulated"
    assert math.isfinite(res["final_loss"]) and ctx.metric_series("fp8/loss_gap")


def test_r2_nvfp4_emulation_on_gpu(ds_dir, tmp_path):
    tr, res, ctx = _run(ds_dir, tmp_path, budget={"max_steps": 8}, logging={"log_every": 2},
                        precision={"rung": "R2", "allow_unvalidated_fp4": True, "diagnostics_every": 2, "loss_gap_every": 2, "healing_steps": 2,
                                   "exempt_first_layers": 1, "exempt_last_layers": 1})
    assert res["precision"]["backend"] == "emulated" and math.isfinite(res["final_loss"])
    keys = {k for _, m in ctx.metrics for k in m}
    assert {"fp4/zero_grad_tile_frac", "fp4/scale_flush_frac", "fp4/sr_bias", "fp4/loss_gap"} <= keys


def test_fp4_emulation_cpu_gpu_parity():
    torch.manual_seed(0)
    x = torch.randn(128, 128)
    cpu = fake_quant_nvfp4(x, (16, 16))
    gpu = fake_quant_nvfp4(x.cuda(), (16, 16)).cpu()
    assert torch.allclose(cpu, gpu, rtol=1e-5, atol=1e-6)
    z = torch.linspace(0, 6, 25, device="cuda").repeat(4000, 1)
    q = round_e2m1_sr(z, FP4Rng(0))
    assert float((q.mean(0) - z[0]).abs().max()) < 0.06  # SR unbiased with the CUDA generator


def test_nf4_qlora_on_gpu():
    from forge.train.config import LoraConfig, QLoraConfig
    from forge.train.lora import LoraBackend, QuantizedLinear, quantize_base_linears

    model = make_model(0, kind="forge").cuda()
    ids = torch.randint(0, 200, (2, 16), device="cuda")
    ref = model(ids).logits.detach()
    quantize_base_linears(model, QLoraConfig(bits=4, block_size=32), torch.bfloat16)
    assert isinstance(model.layers[0].self_attn.q_proj, QuantizedLinear) and model.layers[0].self_attn.q_proj.qweight.device.type == "cuda"
    LoraBackend(LoraConfig(r=4), prefer_builtin=True).apply(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids).logits
    assert torch.isfinite(out).all() and float((out.float() - ref).abs().mean()) < 0.5
    out.float().pow(2).mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_torch_compile_smoke(ds_dir, tmp_path):
    tr, res, _ = _run(ds_dir, tmp_path, budget={"max_steps": 3}, parallel={"compile": True})
    assert res["status"] == "completed" and math.isfinite(res["final_loss"])


# ----------------------------------------------------------------------------------------------- multi-GPU (NCCL)
def _worker(rank, world, init_file, cfg_json, ds_dir, run_dir, out_q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    import torch.distributed as dist

    from forge.train import dist as D
    from forge.train.config import TrainConfig

    try:
        dist.init_process_group("nccl", init_method=Path(init_file).as_uri(), rank=rank, world_size=world)
        cfg = TrainConfig.model_validate_json(cfg_json)
        dinfo = D.init_distributed(cfg.parallel)
        ds = ShardedTokenDataset([ds_dir], cfg.data.seq_len, seed=3, role="train", val_sequences=4)
        va = ShardedTokenDataset([ds_dir], cfg.data.seq_len, seed=3, role="val", val_sequences=4)
        ctx = NullContext(run_dir=Path(run_dir) / f"r{rank}")
        tr = Trainer(make_model(0, kind="forge", layers=4), cfg, LMTask(ds, cfg, va, cfg.data.seq_len), ctx, dinfo=dinfo)
        res = tr.fit()
        sd = D.get_model_state(tr.model)
        out_q.put((rank, res["status"], res["steps"], {k: v.clone() for k, v in sd.items()} if sd else None))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001
        import traceback

        out_q.put((rank, "error", traceback.format_exc(), None))
        raise


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_multi_gpu_nccl(tmp_path, strategy):
    ds = ingest_tiny(tmp_path)
    cfg = small_train_cfg(ds, micro_batch_size=2, grad_accum_steps=1, budget={"max_steps": 4}, parallel={"strategy": strategy, "device": "cuda"},
                          optim={"name": "sgd", "lr": 0.05, "momentum": 0.0, "weight_decay": 0.0, "grad_clip": 0.0},
                          schedule={"name": "constant", "warmup_steps": 0}, eval={"every_steps": 2, "num_batches": 2},
                          precision={"compute_dtype": "bfloat16"})
    c = mp.get_context("spawn")
    q = c.Queue()
    init_file = str(Path(tempfile.mkdtemp()) / "rdzv").replace("\\", "/")
    procs = [c.Process(target=_worker, args=(r, 2, init_file, cfg.model_dump_json(), str(ds), str(tmp_path), q)) for r in range(2)]
    for p in procs:
        p.start()
    res = {r[0]: r for r in (q.get(timeout=600) for _ in procs)}
    for p in procs:
        p.join(timeout=60)
    assert res[0][1] == "completed", res[0][2]
    assert res[0][2] == 4 and res[0][3] and all(torch.isfinite(v).all() for v in res[0][3].values())
