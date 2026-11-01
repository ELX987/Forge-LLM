"""Ternary QAT through the real Trainer on CPU (phase P2): plug-in factory, class swap + restore, health metrics, collapse abort, warm-up."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_train_fixtures import ingest_tiny, make_model, small_train_cfg

from forge.core.context import NullContext
from forge.core.errors import ConfigError
from forge.train.config import PrecisionConfig
from forge.train.data import ShardedTokenDataset
from forge.train.precision import PrecisionManager, make_precision_manager
from forge.train.stability import TrainingDiverged
from forge.train.trainer import LMTask, Trainer


@pytest.fixture(scope="module")
def ds_dir(tmp_path_factory):
    return ingest_tiny(tmp_path_factory.mktemp("ternds"))


def _run(ds_dir, tmp_path, model=None, **over):
    cfg = small_train_cfg(ds_dir, **over)
    s = cfg.seed + cfg.data.data_seed
    tr = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="train", val_sequences=cfg.data.val_sequences)
    va = ShardedTokenDataset([str(ds_dir)], cfg.data.seq_len, seed=s, role="val", val_sequences=cfg.data.val_sequences)
    ctx = NullContext(run_dir=tmp_path / "run")
    t = Trainer(model or make_model(0), cfg, LMTask(tr, cfg, va, cfg.data.seq_len), ctx)
    return t, t.fit(), ctx


def _prec(**tern):
    return {"quant_aware": "ternary", "exempt_first_layers": 0, "exempt_last_layers": 0, "diagnostics_every": 2, "loss_gap_every": 2,
            "ternary": {"group_size": 32, **tern}}


def test_factory_returns_plugin_and_default_is_unchanged():
    assert type(make_precision_manager(PrecisionConfig())) is PrecisionManager
    m = make_precision_manager(PrecisionConfig.model_validate(_prec()))
    assert type(m).__name__ == "TernaryPrecisionManager" and m.metric_prefix == "ternary" and m.healing_start is None
    with pytest.raises(ConfigError):
        make_precision_manager(PrecisionConfig.model_validate({**_prec(), "rung": "R1"}))


def test_prepare_swaps_eligible_layers_and_restore_reverts():
    model = make_model(0)
    before = {n: type(m) for n, m in model.named_modules()}
    names = {n for n, _ in model.named_parameters()}
    m = make_precision_manager(PrecisionConfig.model_validate(_prec()))
    summ = m.prepare(model)
    assert summ["quantised"] > 0 and any(type(x).__name__ == "FakeQuantTernaryLinear" for x in model.modules())
    assert {n for n, _ in model.named_parameters()} == names  # names/optimiser state unchanged
    assert not any(type(x).__name__ == "FakeQuantTernaryLinear" for n, x in model.named_modules() if "lm_head" in n)  # exempt by default
    assert m.restore() == summ["quantised"]
    assert {n: type(x) for n, x in model.named_modules()} == before


def test_high_precision_context_disables_quantisation():
    model = make_model(0)
    m = make_precision_manager(PrecisionConfig.model_validate(_prec()))
    m.prepare(model)
    x = torch.randint(0, 100, (2, 8))
    q = model(x).logits
    with m.high_precision():
        hp = model(x).logits
    ref = make_model(0)(x).logits
    assert torch.allclose(hp, ref, atol=1e-6) and not torch.allclose(q, ref, atol=1e-3)


def test_trainer_trains_and_emits_ternary_health_metrics(ds_dir, tmp_path):
    t, res, ctx = _run(ds_dir, tmp_path, budget={"max_steps": 10}, precision=_prec())
    assert res["status"] == "completed" and type(t.precision).__name__ == "TernaryPrecisionManager"
    losses = [v for _, v in ctx.metric_series("loss")]
    assert losses[-1] < losses[0]
    keys = {k for _, mm in ctx.metrics for k in mm}
    for k in ("ternary/zero_frac", "ternary/zero_frac_min", "ternary/zero_frac_max", "ternary/scale_mean", "ternary/lam", "ternary/flip_rate", "ternary/loss_gap"):
        assert k in keys, k
    zf = [v for _, v in ctx.metric_series("ternary/zero_frac")]
    assert all(0.0 < z < 0.8 for z in zf)  # gaussian-like init: roughly 1/3 .. 1/2 zeros, far from collapse


def test_warmup_lambda_ramps(ds_dir, tmp_path):
    t, _, ctx = _run(ds_dir, tmp_path, budget={"max_steps": 6}, precision=_prec(warmup_steps=6))
    lam = [v for _, v in ctx.metric_series("ternary/lam")]
    assert lam and all(0.0 <= v <= 1.0 for v in lam) and lam[-1] > lam[0]


def test_collapse_aborts_the_run(ds_dir, tmp_path):
    with pytest.raises(TrainingDiverged, match="collapse"):
        _run(ds_dir, tmp_path, budget={"max_steps": 6}, precision=_prec(collapse_stop=0.05, collapse_warn=0.01))


def test_collapse_can_warn_only(ds_dir, tmp_path):
    _, res, ctx = _run(ds_dir, tmp_path, budget={"max_steps": 4}, precision=_prec(collapse_stop=0.05, collapse_warn=0.01, stop_on_collapse=False))
    assert res["status"] == "completed"


def test_exported_model_after_training_is_dense_and_convertible(ds_dir, tmp_path):
    from forge.models.linear import quantize_linears

    t, _, _ = _run(ds_dir, tmp_path, budget={"max_steps": 4}, precision=_prec(), model=None)
    t.precision.restore(t.raw)
    assert not any(type(x).__name__ == "FakeQuantTernaryLinear" for x in t.raw.modules())
    n = quantize_linears(t.raw, "t158", group_size=32)
    assert n


# ------------------------------------------------------------------------------------------------ job type / config / CLI
def test_ternary_train_config_defaults_and_validation():
    from forge.ternary.config import TernaryTrainConfig

    c = TernaryTrainConfig.model_validate({"data": {"dataset": ["x"]}})
    assert c.precision.quant_aware == "ternary" and c.precision.exempt_first_layers == 0 and c.precision.ternary.group_size == 128
    with pytest.raises(ValueError, match="quant_aware"):
        TernaryTrainConfig.model_validate({"data": {"dataset": ["x"]}, "precision": {"quant_aware": "none"}})


def test_run_ternary_train_job_entry(ds_dir, tmp_path):
    from forge.ternary.run import run_ternary_train

    base = small_train_cfg(ds_dir, budget={"max_steps": 4}, precision=_prec()).model_dump(mode="json")
    ctx = NullContext(run_dir=tmp_path / "run")
    res = run_ternary_train(base, ctx)
    assert res and any(k.startswith("ternary/") for _, m in ctx.metrics for k in m)


def test_job_type_registered_experimental_with_templates():
    from forge.jobs.registry import get_job_type

    spec = get_job_type("ternary_train")
    assert spec.experimental and spec.needs_gpu and "ternary" in spec.template_dirs


def test_cli_ternary_shortcut_maps_to_job_type():
    import forge.cli as cli

    assert cli._SHORTCUT_JOB_TYPES["ternary"] == "ternary_train"
