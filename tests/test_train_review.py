"""Adversarial-review regression tests for forge.train (each test failed before the corresponding fix)."""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from test_train_fixtures import VOCAB, ingest_tiny, make_model, small_train_cfg

from forge.core.context import NullContext
from forge.models.tiny import build_tiny_model
from forge.train.config import CheckpointConfig, PrecisionConfig
from forge.train.data import ShardedTokenDataset
from forge.train.distill import build_topk_cache
from forge.train.losses import kd_loss_topk
from forge.train.precision import PrecisionManager, QuantLinear, is_exempt
from forge.train.trainer import LMTask, StepLoss, Trainer, split_batch

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="module")
def ds_dir(tmp_path_factory):
    return ingest_tiny(tmp_path_factory.mktemp("reviewds"))


def _task(ds_dir, cfg, cls=LMTask):
    s = cfg.seed + cfg.data.data_seed
    tr = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="train", val_sequences=cfg.data.val_sequences)
    va = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="val", val_sequences=cfg.data.val_sequences)
    return cls(tr, cfg, va, cfg.data.seq_len)


# ------------------------------------------------------------------------------------------ MoE router precision exemption
def test_moe_router_and_shared_gate_are_exempt_but_expert_gate_proj_is_not():
    cfg = PrecisionConfig(rung="R1", exempt_first_layers=0, exempt_last_layers=0)
    assert is_exempt("layers.1.mlp.gate", cfg, 4)  # MoE router
    assert is_exempt("layers.1.mlp.shared_expert_gate", cfg, 4)
    assert not is_exempt("layers.1.mlp.experts.0.gate_proj", cfg, 4)
    assert not is_exempt("layers.1.mlp.gate_proj", cfg, 4)


def test_router_stays_unquantised_in_a_real_moe_model():
    from forge.models.config import MoEConfig

    # 16 experts so the router (64 -> 16) is large enough to be an R1/R2 candidate at all
    model = build_tiny_model(0, moe=MoEConfig(num_experts=16, top_k=2, moe_intermediate_size=32), arch="qwen3_moe", qk_norm=True,
                             num_layers=3, vocab_size=VOCAB)
    pm = PrecisionManager(PrecisionConfig(rung="R1", exempt_first_layers=0, exempt_last_layers=0), device="cpu", total_steps=10)
    pm.prepare(model)
    for layer in model.layers:
        assert not isinstance(layer.mlp.gate, QuantLinear)
        assert isinstance(layer.mlp.experts[0].up_proj, QuantLinear)


# ------------------------------------------------------------------------------------------ MoE aux-loss coefficient
def test_lmtask_uses_the_models_router_aux_coefficient(ds_dir):
    from forge.models.config import MoEConfig

    moe = MoEConfig(num_experts=4, top_k=2, moe_intermediate_size=32, router_aux_loss_coef=0.01)
    model = build_tiny_model(0, moe=moe, arch="qwen3_moe", qk_norm=True, vocab_size=VOCAB).train()
    cfg = small_train_cfg(ds_dir)  # aux_loss_coef default (a multiplier on the model's own coefficient)
    task = _task(ds_dir, cfg)
    batch = task.get_batches(0, 0, 1)[0]
    out = task.loss(model, batch, SimpleNamespace(raw=model))
    ref = model(batch["input_ids"])
    ce = torch.nn.functional.cross_entropy(ref.logits.float().reshape(-1, VOCAB), batch["labels"].reshape(-1), reduction="sum")
    n = float((batch["labels"] != -100).sum())
    expected = ce + 0.01 * ref.aux_loss.float() * n
    assert torch.allclose(out.loss_sum, expected, rtol=1e-5), (float(out.loss_sum), float(expected))
    assert float(out.metrics["aux"]) == pytest.approx(float(ref.aux_loss) * n, rel=1e-5)  # aux metric is the raw balance loss


# ------------------------------------------------------------------------------------------ fp16 grad-scaler bookkeeping
class _NaNTask(LMTask):
    def loss(self, model, batch, trainer):
        sl = super().loss(model, batch, trainer)
        return StepLoss(sl.loss_sum * float("nan"), sl.metrics)


def test_skipped_steps_do_not_break_the_grad_scaler(ds_dir, tmp_path):
    cfg = small_train_cfg(ds_dir, stability={"on_nonfinite": "skip", "nonfinite_patience": 5})
    tr = Trainer(make_model(0), cfg, _task(ds_dir, cfg, _NaNTask), NullContext(run_dir=tmp_path / "r"))
    tr.setup()
    tr.scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    batches = tr.task.get_batches(0, 0, 1)
    before = {k: v.clone() for k, v in tr.raw.state_dict().items()}
    for s in range(3):  # used to raise "unscale_() has already been called on this optimizer since the last update()"
        res = tr._train_step(s, batches, False, False, False)
        assert res["anomaly"] == "nonfinite" and res["skipped"]
    assert all(torch.equal(before[k], v) for k, v in tr.raw.state_dict().items())  # no update was applied


def test_spike_skip_does_not_break_the_grad_scaler(ds_dir, tmp_path):
    cfg = small_train_cfg(ds_dir, stability={"on_spike": "skip", "spike_patience": 1})
    tr = Trainer(make_model(0), cfg, _task(ds_dir, cfg), NullContext(run_dir=tmp_path / "r"))
    tr.setup()
    tr.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    batches = tr.task.get_batches(0, 0, 1)
    tr.detector.check = lambda loss, gn=None: {"loss_spike": True, "grad_spike": False, "z": 99.0}  # force a spike verdict
    for s in range(3):
        res = tr._train_step(s, batches, False, False, False)
        assert res["anomaly"] == "spike" and res["skipped"]


# ------------------------------------------------------------------------------------------ DDP token-weighted normalisation
def _masked_batches(ds, step, rank, world, micro):
    b = ds.batch_at(step, 4, rank, world)
    per = 4 // world
    lab = b["labels"].clone()
    for i in range(per):
        if rank * per + i < 2:  # global rows 0,1 carry a single supervised token, rows 2,3 all of them
            lab[i, 1:] = -100
    b["labels"] = lab
    return split_batch(b, micro)


class MaskedTask(LMTask):
    def get_batches(self, step, rank, world):
        return _masked_batches(self.ds, step, rank, world, self.cfg.micro_batch_size)


def _norm_worker(rank, world, init_file, cfg_json, ds_dir, run_dir, out_q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), FORGE_HOME=str(Path(run_dir) / f"home{rank}"))
    import torch.distributed as dist

    from test_train_fixtures import make_model as mk
    from test_train_review import MaskedTask as MT

    from forge.core.context import NullContext as NC
    from forge.train import dist as D
    from forge.train.config import TrainConfig
    from forge.train.data import ShardedTokenDataset as SD
    from forge.train.trainer import Trainer as TR

    try:
        dist.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=world)
        cfg = TrainConfig.model_validate_json(cfg_json)
        dinfo = D.init_distributed(cfg.parallel)
        ds = SD([ds_dir], 16, seed=3, role="train", val_sequences=4)
        tr = TR(mk(0), cfg, MT(ds, cfg, None, 16), NC(run_dir=Path(run_dir) / f"r{rank}"), dinfo=dinfo)
        tr.fit()
        out_q.put((rank, {k: v.clone() for k, v in D.get_model_state(tr.model).items()}))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001
        import traceback

        out_q.put((rank, traceback.format_exc()))
        raise


@pytest.mark.skipif(not (torch.distributed.is_available() and torch.distributed.is_gloo_available()), reason="gloo unavailable")
def test_ddp_gradient_is_token_weighted_when_ranks_have_unequal_supervised_counts(ds_dir, tmp_path):
    over = dict(micro_batch_size=2, grad_accum_steps=1, budget={"max_steps": 3}, parallel={"strategy": "ddp", "device": "cpu"},
                optim={"name": "sgd", "lr": 0.05, "momentum": 0.0, "weight_decay": 0.0, "grad_clip": 0.0},
                schedule={"name": "constant", "warmup_steps": 0}, eval={"every_steps": 0, "num_batches": 1},
                stability={"detect_spikes": False})
    cfg = small_train_cfg(ds_dir, **over)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_file = str(Path(tempfile.mkdtemp()) / "rdzv")
    procs = [ctx.Process(target=_norm_worker, args=(r, 2, init_file, cfg.model_dump_json(), str(ds_dir), str(tmp_path), q)) for r in range(2)]
    for p in procs:
        p.start()
    res = dict(q.get(timeout=240) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    assert isinstance(res[0], dict), res[0]
    # reference: one process, the same 4 global rows in one micro-batch
    from forge.train.config import TrainConfig

    c1 = TrainConfig.model_validate({**cfg.model_dump(), "micro_batch_size": 4, "parallel": {"strategy": "none", "device": "cpu"}})
    ds = ShardedTokenDataset([str(ds_dir)], 16, seed=3, role="train", val_sequences=4)
    tr = Trainer(make_model(0), c1, MaskedTask(ds, c1, None, 16), NullContext(run_dir=tmp_path / "single"))
    tr.fit()
    ref = {k: v.detach().clone() for k, v in tr.raw.state_dict().items()}
    assert all(torch.allclose(res[0][k], ref[k], atol=2e-5) for k in ref), max(float((res[0][k] - ref[k]).abs().max()) for k in ref)


# ------------------------------------------------------------------------------------------ logit-cache rebuild / staleness
def _cds(ds_dir, n=12):
    ds = ShardedTokenDataset([str(ds_dir)], 16, seed=0, role="train", val_sequences=4)
    ds.restrict(n)
    return ds


def test_changing_k_in_an_existing_cache_dir_rebuilds_instead_of_mixing_shards(ds_dir, tmp_path):
    ds = _cds(ds_dir)
    t = make_model(5).eval()
    p = tmp_path / "c"
    c0 = build_topk_cache(p, t, ds, 12, k=4, shard_sequences=6, batch_size=3, device="cpu", meta_extra={"key": "a"})
    c0.close()
    c = build_topk_cache(p, t, ds, 12, k=8, shard_sequences=6, batch_size=3, device="cpu", meta_extra={"key": "a"})
    got = c.get(np.array([0, 7, 11]))
    assert got["idx"].shape[-1] == 8 and got["logp"].shape[-1] == 8
    lp = torch.log_softmax(t(ds.get_batch(np.array([7]))["input_ids"]).logits.float(), -1).topk(8, -1).values
    assert torch.allclose(got["logp"][1:2], lp.half().float(), atol=1e-3)
    c.close()


def test_rebuild_flag_really_recomputes_with_a_different_teacher(ds_dir, tmp_path):
    ds = _cds(ds_dir)
    p = tmp_path / "c"
    c1 = build_topk_cache(p, make_model(5).eval(), ds, 12, k=4, shard_sequences=6, batch_size=3, device="cpu")
    a = c1.get(np.array([3]))["logp"].clone()
    c1.close()
    c2 = build_topk_cache(p, make_model(6).eval(), ds, 12, k=4, shard_sequences=6, batch_size=3, device="cpu", rebuild=True)
    b = c2.get(np.array([3]))["logp"]
    assert not torch.allclose(a, b)
    c2.close()


def test_resume_after_cancel_is_still_supported(ds_dir, tmp_path):
    ds = _cds(ds_dir)
    t = make_model(5).eval()
    p = tmp_path / "c"

    class Stop(NullContext):
        def progress(self, frac, msg=""):
            super().progress(frac, msg)
            self.stop = True

    c = Stop(run_dir=tmp_path / "r")
    with pytest.raises(Exception, match="cancelled"):
        build_topk_cache(p, t, ds, 12, k=4, shard_sequences=6, batch_size=3, device="cpu", ctx=c, should_stop=c.should_stop, meta_extra={"key": "a"})
    assert (p / "done_00000").exists()
    cache = build_topk_cache(p, t, ds, 12, k=4, shard_sequences=6, batch_size=3, device="cpu", meta_extra={"key": "a"})
    assert cache.complete and (p / "done_00000").exists()
    cache.close()


# ------------------------------------------------------------------------------------------ top-k KD numerics
def test_topk_kd_tail_mass_is_accurate_for_a_peaked_student():
    torch.manual_seed(0)
    V, k = 1000, 4
    logits = torch.randn(1, 3, V) * 0.1
    logits[..., 7] = 40.0  # nearly all student mass on token 7
    idx = torch.tensor([[[7, 1, 2, 3]] * 3])
    lt = torch.tensor([0.6, 0.2, 0.1, 0.05]).log()
    t_lp = lt.expand(1, 3, k).clone()
    res = torch.full((1, 3), 1.0 - 0.95)
    mask = torch.ones(1, 3, dtype=torch.bool)
    got, _ = kd_loss_topk(logits, idx, t_lp, res, mask, "forward_kl", 1.0, tail_mode="bin")
    # float64 reference of the same binned forward KL
    ls = torch.log_softmax(logits.double(), -1)
    ls_k = ls.gather(-1, idx)
    mask_k = torch.ones_like(ls, dtype=torch.bool).scatter(-1, idx, False)
    s_tail = ls.exp().masked_fill(~mask_k, 0.0).sum(-1)
    pt = torch.cat([t_lp.double().exp(), res.double().unsqueeze(-1)], -1)
    lq = torch.cat([ls_k, s_tail.log().unsqueeze(-1)], -1)
    ref = (pt * (pt.log() - lq)).sum(-1).sum()
    assert float(got) == pytest.approx(float(ref), rel=1e-4, abs=1e-4), (float(got), float(ref))


def test_topk_kd_still_matches_dense_when_k_equals_vocab():
    torch.manual_seed(1)
    V = 16
    s, t = torch.randn(2, 5, V), torch.randn(2, 5, V)
    lp = torch.log_softmax(t, -1)
    v, i = lp.topk(V, -1)
    mask = torch.ones(2, 5, dtype=torch.bool)
    from forge.train.losses import kd_loss_dense

    for kind in ("forward_kl", "reverse_kl", "jsd"):
        a, _ = kd_loss_topk(s, i, v, (1 - v.exp().sum(-1)).clamp(min=0), mask, kind)
        b, _ = kd_loss_dense(s, t, mask, kind)
        assert float(a) == pytest.approx(float(b), rel=1e-3, abs=1e-4), kind


# ------------------------------------------------------------------------------------------ rollback availability
def test_checkpoint_defaults_allow_rollback_from_the_start():
    assert CheckpointConfig().save_initial is True


def test_rollback_before_first_periodic_checkpoint_recovers(ds_dir, tmp_path):
    """With the default config a NaN at step 2 must roll back to the step-0 checkpoint instead of aborting the run."""
    from forge.train.stability import TrainingDiverged

    class NaNOnce(LMTask):
        fired = False

        def loss(self, model, batch, trainer):
            sl = super().loss(model, batch, trainer)
            if trainer.step == 2 and not NaNOnce.fired:
                NaNOnce.fired = True
                return StepLoss(sl.loss_sum * float("nan"), sl.metrics)
            return sl

    cfg = small_train_cfg(ds_dir, budget={"max_steps": 5}, checkpoint={"every_steps": 0, "keep_last": 3, "save_initial": CheckpointConfig().save_initial})
    tr = Trainer(make_model(0), cfg, _task(ds_dir, cfg, NaNOnce), NullContext(run_dir=tmp_path / "r"))
    try:
        res = tr.fit()
    except TrainingDiverged as e:  # pragma: no cover - the regression
        pytest.fail(f"run aborted: {e}")
    assert res["status"] == "completed" and res["restarts"] == 1 and math.isfinite(res["final_loss"])
