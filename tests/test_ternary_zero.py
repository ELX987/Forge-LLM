"""``parallel.strategy = zero_ternary`` (phase P5) on CPU: sharding maths, config rules, single-process equivalence, gloo multi-rank
equivalence with the unsharded trainer, shard-aware checkpoints and resume on a different world size, the OOM policy.

NCCL (``reduce_scatter``) and any GPU path are NOT exercised here; they belong to validation gate G1 on a multi-GPU host.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))
from test_train_fixtures import ingest_tiny, make_model, small_train_cfg

from forge.core.context import NullContext
from forge.core.errors import ConfigError
from forge.ternary.zero import FORMAT, ZeroTernaryLinear, row_ranges
from forge.train.config import ParallelConfig, TrainConfig
from forge.train.data import ShardedTokenDataset
from forge.train.trainer import LMTask, StepLoss, Trainer

try:
    import torch.distributed as dist

    _GLOO = dist.is_available() and dist.is_gloo_available()
except Exception:  # noqa: BLE001
    _GLOO = False

needs_gloo = pytest.mark.skipif(not _GLOO, reason="torch.distributed/gloo unavailable")


@pytest.fixture(scope="module")
def ds_dir(tmp_path_factory):
    return ingest_tiny(tmp_path_factory.mktemp("zerods"))


def _prec(**tern):
    return {"quant_aware": "ternary", "exempt_first_layers": 0, "exempt_last_layers": 0, "diagnostics_every": 2, "loss_gap_every": 2,
            "ternary": {"group_size": 32, **tern}}


def _cfg(ds_dir, strategy="zero_ternary", **over):
    over.setdefault("precision", _prec())
    over.setdefault("optim", {"name": "sgd", "lr": 0.05, "momentum": 0.0, "weight_decay": 0.0, "grad_clip": 0.0})
    over.setdefault("schedule", {"name": "constant", "warmup_steps": 0})
    over.setdefault("eval", {"every_steps": 2, "num_batches": 2})
    over["parallel"] = {"strategy": strategy, "device": "cpu", **over.get("parallel", {})}
    return small_train_cfg(ds_dir, **over)


def _datasets(cfg, ds_dir):
    s = cfg.seed + cfg.data.data_seed
    tr = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="train", val_sequences=cfg.data.val_sequences)
    va = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="val", val_sequences=cfg.data.val_sequences)
    return tr, va


def _fit(cfg, ds_dir, tmp_path, tag, task_cls=LMTask):
    tr, va = _datasets(cfg, ds_dir)
    ctx = NullContext(run_dir=tmp_path / tag)
    t = Trainer(make_model(0), cfg, task_cls(tr, cfg, va, cfg.data.seq_len), ctx)
    res = t.fit()
    return t, res, ctx


def _losses(ctx):
    return [(s, v["loss"]) for s, v in ctx.metrics if "loss" in v]


def _state(t):
    return {k: v.detach().clone().float() for k, v in t.raw.state_dict().items()}


def _assert_close(a, b, atol, what=""):
    assert set(a) == set(b), what
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst <= atol, f"{what}: max |diff| {worst:.3e} > {atol:.1e}"


# ---------------------------------------------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,world", [(64, 2), (64, 3), (10, 4), (17, 17), (5, 2)])
def test_row_ranges_cover_exactly_and_are_balanced(n, world):
    r = row_ranges(n, world)
    assert r[0][0] == 0 and r[-1][1] == n and all(r[i][1] == r[i + 1][0] for i in range(world - 1))
    sizes = [b - a for a, b in r]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == n


def test_config_requires_ternary_qat_and_rewrites_per_tensor_scale(ds_dir, caplog):
    with pytest.raises(ValueError, match="quant_aware"):
        small_train_cfg(ds_dir, parallel={"strategy": "zero_ternary", "device": "cpu"})
    with pytest.raises(ValueError, match="adamw\\|lion\\|sgd"):
        _cfg(ds_dir, optim={"name": "adamw8bit"})
    with pytest.raises(ValueError, match="float16"):
        _cfg(ds_dir, precision={**_prec(), "compute_dtype": "float16"})
    with caplog.at_level("WARNING"):
        c = _cfg(ds_dir, precision=_prec(group_size=0))
    assert c.precision.ternary.group_size == -1  # the saved config equals the effective config
    assert "per-tensor" in caplog.text and "-1" in caplog.text
    assert json.loads(c.model_dump_json())["precision"]["ternary"]["group_size"] == -1
    assert ParallelConfig().strategy == "none" and ParallelConfig().oom_halve_micro_batch is False


def test_per_tensor_scale_is_rejected_by_the_engine(ds_dir, tmp_path):
    cfg = _cfg(ds_dir)
    cfg.precision.ternary.group_size = 0  # bypass the validator
    tr, va = _datasets(cfg, ds_dir)
    t = Trainer(make_model(0), cfg, LMTask(tr, cfg, va, 16), NullContext(run_dir=tmp_path / "x"))
    with pytest.raises(ConfigError, match="per-tensor"):
        t.setup()


# ---------------------------------------------------------------------------------------------------------------
# single process (world 1): the shard is the whole layer, so results must equal the unsharded trainer
# ---------------------------------------------------------------------------------------------------------------
def test_world1_sgd_equals_plain_trainer(ds_dir, tmp_path):
    over = {"budget": {"max_steps": 5}, "optim": {"name": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 0.0, "grad_clip": 0.05}}
    t_ref, r_ref, c_ref = _fit(_cfg(ds_dir, "none", **over), ds_dir, tmp_path, "ref")
    t_z, r_z, c_z = _fit(_cfg(ds_dir, **over), ds_dir, tmp_path, "zero")
    assert t_z.zero is not None and r_z["zero_ternary"]["sharded_layers"] > 0
    _assert_close(_state(t_ref), _state(t_z), 1e-6, "weights")
    lr_, lz = _losses(c_ref), _losses(c_z)
    assert [s for s, _ in lr_] == [s for s, _ in lz] and all(abs(a - b) < 1e-5 for (_, a), (_, b) in zip(lr_, lz))
    gn = lambda ctx: [v["grad_norm"] for _, v in ctx.metrics if "grad_norm" in v]  # noqa: E731
    assert max(gn(c_ref)) > 0.05  # clipping was active, and the sharded global norm agrees
    assert all(abs(a - b) <= 1e-4 * max(1.0, a) for a, b in zip(gn(c_ref), gn(c_z)))
    ev = lambda ctx: [v["eval_loss"] for _, v in ctx.metrics if "eval_loss" in v]  # noqa: E731
    assert ev(c_ref) and all(abs(a - b) < 1e-5 for a, b in zip(ev(c_ref), ev(c_z)))
    z = lambda ctx: [v["ternary/zero_frac"] for _, v in ctx.metrics if "ternary/zero_frac" in v]  # noqa: E731
    assert z(c_ref) and z(c_ref) == pytest.approx(z(c_z))  # health diagnostics read the synchronised codes


def test_world1_adamw_with_warmup_equals_plain_trainer(ds_dir, tmp_path):
    over = {"budget": {"max_steps": 5}, "precision": _prec(warmup_steps=3),
            "optim": {"name": "adamw", "lr": 3e-3, "weight_decay": 0.1, "grad_clip": 1.0}}
    t_ref, _, _ = _fit(_cfg(ds_dir, "none", **over), ds_dir, tmp_path, "ref")
    t_z, _, _ = _fit(_cfg(ds_dir, **over), ds_dir, tmp_path, "zero")
    _assert_close(_state(t_ref), _state(t_z), 1e-6, "weights")


def test_forward_uses_the_synchronised_codes(ds_dir, tmp_path):
    cfg = _cfg(ds_dir, budget={"max_steps": 1})
    t, _, _ = _fit(cfg, ds_dir, tmp_path, "z")
    mods = [m for m in t.raw.modules() if isinstance(m, ZeroTernaryLinear)]
    assert mods and all(m.__dict__["_zero_q"] is not None for m in mods)
    m = mods[0]
    x = torch.randn(3, m.in_features)
    with torch.no_grad():
        y = m(x)
        m.__dict__["_zero_q"] = None  # fall back to quantising the latent weight: same numbers
        m._cache = None
        y2 = m(x)
    assert torch.equal(y, y2)


def test_checkpoint_is_sharded_safetensors_and_state_pt_holds_no_ternary_latents(ds_dir, tmp_path):
    from safetensors import safe_open

    cfg = _cfg(ds_dir, budget={"max_steps": 2}, optim={"name": "adamw", "lr": 3e-3, "weight_decay": 0.1, "grad_clip": 1.0},
               checkpoint={"every_steps": 2, "keep_last": 3, "dir": str(tmp_path / "ck")})
    t, res, _ = _fit(cfg, ds_dir, tmp_path, "z")
    d = Path(res["checkpoint_dir"]) / "step_00000002"
    assert (d / "state.pt").exists() and (d / "meta.json").exists()
    files = sorted(p.name for p in (d / "shards").iterdir())
    assert files == ["manifest.json", "rank_00000.safetensors"]
    man = json.loads((d / "shards" / "manifest.json").read_text())
    assert man["format"] == FORMAT and man["world"] == 1 and man["optim_keys"] == ["exp_avg", "exp_avg_sq", "step"]
    st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
    assert st["zero"] == man and not (set(st["model"]) & t.zero.param_names) and "lm_head.weight" in st["model"]
    with safe_open(str(d / "shards" / "rank_00000.safetensors"), "pt") as f:
        keys = set(f.keys())
        name = next(iter(t.zero.param_names))
        assert {f"latent/{name}", f"state/exp_avg/{name}", f"state/exp_avg_sq/{name}", f"scalar/step/{name}"} <= keys
        assert torch.equal(f.get_tensor(f"latent/{name}"), t.raw.get_parameter(name).detach())
    assert not list((tmp_path / "ck").glob(".zstage_*"))  # staging directory was moved into the checkpoint


def test_resume_world1_continues_exactly_and_adopts_plain_checkpoints(ds_dir, tmp_path):
    base = {"optim": {"name": "adamw", "lr": 3e-3, "weight_decay": 0.1, "grad_clip": 1.0}}
    t_full, _, _ = _fit(_cfg(ds_dir, budget={"max_steps": 6}, **base), ds_dir, tmp_path, "full")
    ck = {"every_steps": 3, "keep_last": 3, "dir": str(tmp_path / "ck")}
    _fit(_cfg(ds_dir, budget={"max_steps": 3}, checkpoint=ck, **base), ds_dir, tmp_path, "a")
    t_b, res_b, _ = _fit(_cfg(ds_dir, budget={"max_steps": 6}, checkpoint={**ck, "resume": "auto"}, **base), ds_dir, tmp_path, "b")
    assert res_b["steps"] == 6
    _assert_close(_state(t_full), _state(t_b), 1e-6, "resumed vs continuous")
    # a checkpoint written by the plain (unsharded) ternary trainer is adopted: latents load, optimiser state restarts
    ck2 = {"every_steps": 3, "keep_last": 3, "dir": str(tmp_path / "ck_plain")}
    t_plain, _, _ = _fit(_cfg(ds_dir, "none", budget={"max_steps": 3}, checkpoint=ck2, **base), ds_dir, tmp_path, "plain")
    t_c, res_c, c_c = _fit(_cfg(ds_dir, budget={"max_steps": 4}, checkpoint={**ck2, "resume": "auto"}, **base), ds_dir, tmp_path, "c")
    assert res_c["steps"] == 4 and all(torch.isfinite(v).all() for v in _state(t_c).values())


def test_unshared_checkpoint_directory_is_reported_clearly(tmp_path):
    from forge.ternary.zero import verify_shard_files

    (tmp_path / "rank_00000.safetensors").write_bytes(b"x")
    verify_shard_files(tmp_path, 1)
    with pytest.raises(ConfigError, match="shared by all ranks") as ei:
        verify_shard_files(tmp_path, 2, wait_s=0.2)
    assert ei.value.details["missing"] == ["rank_00001.safetensors"]


# ---------------------------------------------------------------------------------------------------------------
# OOM policy
# ---------------------------------------------------------------------------------------------------------------
class _OomOnBigBatches(LMTask):
    """Raises a CUDA OOM for any micro-batch with more than 2 rows (stand-in for a too-large micro-batch)."""

    def loss(self, model, batch, trainer):
        if batch["input_ids"].shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")
        return super().loss(model, batch, trainer)


def test_oom_policy_splits_micro_batches_and_keeps_the_gradient(ds_dir, tmp_path):
    over = {"micro_batch_size": 4, "grad_accum_steps": 1, "budget": {"max_steps": 3}, "eval": {"every_steps": 0}}
    t_ref, _, _ = _fit(_cfg(ds_dir, "none", **over), ds_dir, tmp_path, "ref")
    cfg = _cfg(ds_dir, "none", parallel={"oom_halve_micro_batch": True}, **over)
    t, res, ctx = _fit(cfg, ds_dir, tmp_path, "oom", task_cls=_OomOnBigBatches)
    assert res["oom_micro_batch_split"] == 2 and res["status"] == "completed"
    _assert_close(_state(t_ref), _state(t), 1e-5, "split micro-batches vs one big micro-batch")
    # disabled by default: the OOM propagates
    cfg2 = _cfg(ds_dir, "none", **over)
    tr, va = _datasets(cfg2, ds_dir)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Trainer(make_model(0), cfg2, _OomOnBigBatches(tr, cfg2, va, 16), NullContext(run_dir=tmp_path / "off")).fit()


def test_oom_policy_gives_up_at_one_row(ds_dir, tmp_path):
    class Always(LMTask):
        def loss(self, model, batch, trainer):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    cfg = _cfg(ds_dir, "none", parallel={"oom_halve_micro_batch": True}, micro_batch_size=2, grad_accum_steps=1, budget={"max_steps": 2})
    tr, va = _datasets(cfg, ds_dir)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Trainer(make_model(0), cfg, Always(tr, cfg, va, 16), NullContext(run_dir=tmp_path / "g")).fit()


# ---------------------------------------------------------------------------------------------------------------
# gloo multi-rank equivalence
# ---------------------------------------------------------------------------------------------------------------
def _zworker(rank, world, init_file, cfg_json, ds_dir, run_dir, out_q):
    import faulthandler

    faulthandler.dump_traceback_later(150, exit=False)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), FORGE_HOME=str(Path(run_dir) / f"home{rank}"))
    import torch.distributed as dist_

    from forge.train import dist as D

    try:
        dist_.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=world)
        cfg = TrainConfig.model_validate_json(cfg_json)
        dinfo = D.init_distributed(cfg.parallel)
        tr, va = _datasets(cfg, ds_dir)
        ctx = NullContext(run_dir=Path(run_dir) / f"r{rank}")
        t = Trainer(make_model(0), cfg, LMTask(tr, cfg, va, cfg.data.seq_len), ctx, dinfo=dinfo)
        res = t.fit()
        sd = {k: v.detach().cpu().float().numpy() for k, v in t.raw.state_dict().items()}
        out_q.put((rank, res["status"], res["steps"], sd, [(s, dict(v)) for s, v in ctx.metrics]))
        dist_.barrier()
        dist_.destroy_process_group()
    except BaseException:  # noqa: BLE001
        import traceback

        out_q.put((rank, "error", traceback.format_exc(), None, None))
        raise


def _run_ranks(cfg, ds_dir, tmp_path, world, tag):
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_file = str(Path(tempfile.mkdtemp()) / "rdzv")
    run_dir = tmp_path / tag
    procs = [ctx.Process(target=_zworker, args=(r, world, init_file, cfg.model_dump_json(), str(ds_dir), str(run_dir), q)) for r in range(world)]
    for p in procs:
        p.start()
    try:
        results = [q.get(timeout=240) for _ in procs]
    except Exception:  # noqa: BLE001
        for p in procs:
            if p.is_alive():
                p.terminate()
        pytest.fail(f"ranks did not report within 240 s (exit codes {[p.exitcode for p in procs]})")
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    out = {}
    for r in results:
        assert r[1] != "error", r[2]
        out[r[0]] = (r[1], r[2], {k: torch.from_numpy(v) for k, v in r[3].items()}, r[4])
    return out


def _metric(metrics, key):
    return [(s, v[key]) for s, v in metrics if key in v]


@needs_gloo
def test_two_ranks_sgd_with_warmup_clip_eval_equals_one_process(ds_dir, tmp_path):
    # 2 ranks x 2 sequences == one process x 4 sequences; warm-up (latent gather), active clipping, high-precision eval and loss-gap shadow forwards
    over = {"micro_batch_size": 2, "grad_accum_steps": 1, "budget": {"max_steps": 4}, "precision": _prec(warmup_steps=3),
            "optim": {"name": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 0.0, "grad_clip": 0.05},
            "checkpoint": {"every_steps": 2, "keep_last": 3, "dir": str(tmp_path / "ck2")}}
    res = _run_ranks(_cfg(ds_dir, **over), ds_dir, tmp_path, 2, "w2")
    assert all(r[0] == "completed" and r[1] == 4 for r in res.values())
    ref_cfg = _cfg(ds_dir, "none", **{**over, "micro_batch_size": 4, "checkpoint": {"every_steps": 0}})
    t_ref, _, c_ref = _fit(ref_cfg, ds_dir, tmp_path, "ref")
    ref = _state(t_ref)
    _assert_close(res[0][2], res[1][2], 0.0, "rank 0 vs rank 1 (replicas identical)")
    _assert_close(res[0][2], ref, 2e-5, "2-rank sharded vs 1-process")
    for key, tol in (("loss", 1e-4), ("eval_loss", 1e-4)):
        a, b = _metric(res[0][3], key), _metric(c_ref.metrics, key)
        assert a and [s for s, _ in a] == [s for s, _ in b], key
        assert all(abs(x - y) <= tol * max(1.0, abs(y)) for (_, x), (_, y) in zip(a, b)), (key, a, b)
    gn_a, gn_b = _metric(res[0][3], "grad_norm"), _metric(c_ref.metrics, "grad_norm")
    assert max(v for _, v in gn_b) > 0.05 and all(abs(x - y) <= 1e-3 * max(1.0, y) for (_, x), (_, y) in zip(gn_a, gn_b))
    # shard-aware checkpoint: one file per rank, only rank 0 writes the replicated state
    d = tmp_path / "ck2" / "step_00000004"
    assert sorted(p.name for p in (d / "shards").iterdir()) == ["manifest.json", "rank_00000.safetensors", "rank_00001.safetensors"]
    assert json.loads((d / "shards" / "manifest.json").read_text())["world"] == 2


@needs_gloo
def test_resume_on_a_different_number_of_ranks_continues_the_same_run(ds_dir, tmp_path):
    # global batch 6 throughout: 2 ranks x 3, then 3 ranks x 2, then 1 process x 6; SGD with momentum so the sharded optimiser state matters
    base = {"grad_accum_steps": 1, "optim": {"name": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 0.0, "grad_clip": 0.1}, "eval": {"every_steps": 0},
            "precision": _prec()}
    ck = {"every_steps": 2, "keep_last": 5, "dir": str(tmp_path / "ck")}
    r2 = _run_ranks(_cfg(ds_dir, micro_batch_size=3, budget={"max_steps": 2}, checkpoint=ck, **base), ds_dir, tmp_path, 2, "a")
    assert r2[0][1] == 2
    ref_t, _, _ = _fit(_cfg(ds_dir, "none", micro_batch_size=6, budget={"max_steps": 4}, checkpoint={"every_steps": 0}, **base), ds_dir, tmp_path, "ref")
    ref = _state(ref_t)
    # 3 ranks (uneven row split: 64 and 128 rows over 3 ranks) resume from the 2-rank checkpoint and run steps 3-4
    ck3 = {**ck, "resume": "auto"}
    r3 = _run_ranks(_cfg(ds_dir, micro_batch_size=2, budget={"max_steps": 4}, checkpoint=ck3, **base), ds_dir, tmp_path, 3, "b")
    assert all(r[1] == 4 for r in r3.values())
    _assert_close(r3[0][2], ref, 3e-5, "2 ranks -> 3 ranks vs continuous 1 process")
    _assert_close(r3[0][2], r3[2][2], 0.0, "ranks agree")
    # and the 3-rank checkpoint resumes in a single process
    m3 = json.loads((tmp_path / "ck" / "step_00000004" / "shards" / "manifest.json").read_text())
    assert m3["world"] == 3
    t1, res1, _ = _fit(_cfg(ds_dir, micro_batch_size=6, budget={"max_steps": 6}, checkpoint={**ck3}, **base), ds_dir, tmp_path, "c")
    assert res1["steps"] == 6
    ref6, _, _ = _fit(_cfg(ds_dir, "none", micro_batch_size=6, budget={"max_steps": 6}, checkpoint={"every_steps": 0}, **base), ds_dir, tmp_path, "ref6")
    _assert_close(_state(t1), _state(ref6), 5e-5, "3 ranks -> 1 process vs continuous")


@needs_gloo
def test_two_ranks_adamw_tracks_the_unsharded_trainer(ds_dir, tmp_path):
    # Adam normalises tiny gradients, so summation-order noise is amplified: assert the trajectory, not bit equality
    over = {"micro_batch_size": 2, "grad_accum_steps": 1, "budget": {"max_steps": 4}, "eval": {"every_steps": 0},
            "optim": {"name": "adamw", "lr": 3e-3, "weight_decay": 0.1, "grad_clip": 1.0},
            "checkpoint": {"every_steps": 0, "dir": str(tmp_path / "ck_adam")}}
    res = _run_ranks(_cfg(ds_dir, **over), ds_dir, tmp_path, 2, "adam")
    t_ref, _, c_ref = _fit(_cfg(ds_dir, "none", **{**over, "micro_batch_size": 4, "checkpoint": {"every_steps": 0}}), ds_dir, tmp_path, "ref")
    a, b = _metric(res[0][3], "loss"), _metric(c_ref.metrics, "loss")
    assert all(abs(x - y) <= 5e-3 * max(1.0, y) for (_, x), (_, y) in zip(a, b)), (a, b)
    ref = _state(t_ref)
    mean_diff = sum(float((res[0][2][k] - ref[k]).abs().mean()) for k in ref) / len(ref)
    assert mean_diff < 2e-4
    _assert_close(res[0][2], res[1][2], 0.0, "ranks agree")


# ---------------------------------------------------------------------------------------------------------------
# gate G1 harness + collectives check (gloo stands in for NCCL here)
# ---------------------------------------------------------------------------------------------------------------
def _parity_worker(rank, world, init_file, shared, out_q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), FORGE_PARITY_DIR=shared, FORGE_HOME=str(Path(shared) / f"home{rank}"))
    import importlib.util

    import torch.distributed as dist_

    try:
        dist_.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=world)
        spec = importlib.util.spec_from_file_location("zero_ternary_parity", Path(__file__).resolve().parents[1] / "scripts" / "zero_ternary_parity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main(["--device", "cpu", "--backend", "gloo", "--steps", "3"])
        out_q.put((rank, rc))
    except BaseException:  # noqa: BLE001
        import traceback

        out_q.put((rank, traceback.format_exc()))
        raise


@needs_gloo
def test_g1_parity_script_passes_on_gloo(tmp_path, capfd):
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_file = str(Path(tempfile.mkdtemp()) / "rdzv")
    procs = [ctx.Process(target=_parity_worker, args=(r, 2, init_file, str(tmp_path), q)) for r in range(2)]
    for p in procs:
        p.start()
    try:
        got = dict(q.get(timeout=240) for _ in procs)
    except Exception:  # noqa: BLE001
        for p in procs:
            if p.is_alive():
                p.terminate()
        pytest.fail("parity harness did not finish within 240 s")
    for p in procs:
        p.join(timeout=60)
    assert got == {0: 0, 1: 0}, got


@needs_gloo
def test_collectives_check_on_gloo_and_nccl_is_skipped_not_passed_without_gpus():
    from forge.train.collectives import run_collectives_check

    r = run_collectives_check("gloo", 2, size_mib=0.25, timeout=180)
    assert r["ok"] is True and r["world"] == 2
    assert r["checks"]["broadcast"] and r["checks"]["all_reduce"] and r["checks"]["all_gather_uint8"]
    assert r["checks"]["reduce_scatter"] in (True, None)  # gloo may lack reduce_scatter_tensor: unsupported, not a failure
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        n = run_collectives_check("nccl", 2)
        assert n["ok"] is None and "skipped" in n  # never reported as a pass
