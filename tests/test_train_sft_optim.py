"""Chat templates / loss masks / packing / preference data; LR schedules; optimisers; losses."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from forge.core.errors import ConfigError
from forge.train.config import OptimConfig, ScheduleConfig, SFTDataConfig
from forge.train.losses import ce_loss_sum, dpo_loss, kd_loss_dense, kd_loss_topk, response_logprobs
from forge.train.optim import Lion, anneal_start_step, build_optimizer, lr_factor, set_lr
from forge.train.sft import (
    IGNORE_INDEX,
    SFTDataset,
    build_sft_datasets,
    encode_conversation,
    normalise_record,
    render_segments,
)
from forge.train.tokenization import ByteTokenizer, TokenizerAdapter

TOK = TokenizerAdapter(ByteTokenizer())
CONV = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "again"}, {"role": "assistant", "content": "ok"}]


# ----------------------------------------------------------------------------------------------- templates
@pytest.mark.parametrize("template", ["chatml", "llama3", "mistral", "alpaca", "plain"])
def test_assistant_only_mask(template):
    ids, mask = encode_conversation(CONV, TOK, template)
    assert len(ids) == len(mask) and any(mask) and not all(mask)
    text_trained = TOK.decode([i for i, m in zip(ids, mask, strict=False) if m])
    assert "yo" in text_trained and "ok" in text_trained
    assert "hi" not in text_trained and "again" not in text_trained


def test_chatml_mask_exact_and_last_turn_only():
    ids, mask = encode_conversation(CONV[:2], TOK, "chatml")
    assert TOK.decode([i for i, m in zip(ids, mask, strict=False) if m]) == "yo<|im_end|>"
    ids2, mask2 = encode_conversation(CONV, TOK, "chatml", train_all_assistant=False)
    assert TOK.decode([i for i, m in zip(ids2, mask2, strict=False) if m]) == "ok<|im_end|>"
    ids3, mask3 = encode_conversation(CONV, TOK, "chatml", train_on_prompt=True)
    assert all(mask3)


def test_system_prompt_and_generation_prompt():
    segs = render_segments("chatml", CONV[:1], TOK, system_prompt="be nice", add_generation_prompt=True)
    txt = "".join(s for s, _ in segs if isinstance(s, str))
    assert txt.startswith("<|im_start|>system\nbe nice") and txt.endswith("<|im_start|>assistant\n")
    with pytest.raises(ConfigError):
        render_segments("nope", CONV, TOK)


def test_record_formats():
    assert normalise_record({"messages": CONV})["kind"] == "chat"
    sg = normalise_record({"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]})
    assert [m["role"] for m in sg["messages"]] == ["user", "assistant"]
    al = normalise_record({"instruction": "do", "input": "x", "output": "done"})
    assert "do\n\nx" == al["messages"][0]["content"]
    assert normalise_record({"prompt": "p", "completion": "c"})["messages"][1]["content"] == "c"
    pr = normalise_record({"prompt": "p", "chosen": "good", "rejected": "bad"})
    assert pr["kind"] == "pref" and pr["chosen"][0]["content"] == "good"
    assert normalise_record({"text": "raw"})["kind"] == "text"
    assert normalise_record({"q": "p", "a": "c"}, "prompt_completion", {"prompt": "q", "completion": "a"})["kind"] == "chat"
    with pytest.raises(ConfigError):
        normalise_record({"weird": 1})


def _records(n=10):
    return [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"answer number {i}"}]} for i in range(n)]


def test_sft_dataset_batches_align_labels():
    cfg = SFTDataConfig(max_seq_len=64, shuffle=False)
    ds = SFTDataset(_records(6), TOK, cfg)
    b = ds.get_batch(np.array([0, 1]))
    x, y = b["input_ids"], b["labels"]
    assert x.shape == y.shape and b["lengths"].shape == (2,)
    for r in range(2):
        L = int(b["lengths"][r])
        sup = y[r, :L] != IGNORE_INDEX
        # supervised labels are exactly the next tokens of the input stream and cover the assistant answer
        assert sup.any()
        ids = x[r, :L].tolist()
        full, mask = encode_conversation(_records(6)[r]["messages"], TOK, "chatml")
        assert ids == full[:-1]
        assert [t for t, m in zip(full[1:], mask[1:], strict=False) if m] == y[r][sup].tolist()
        assert (y[r, L:] == IGNORE_INDEX).all()


def test_truncation_drop_pack_and_determinism():
    long_rec = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "z" * 200}]}
    ds_drop = SFTDataset([long_rec] + _records(3), TOK, SFTDataConfig(max_seq_len=48, truncation="drop"))
    assert len(ds_drop) == 3 and ds_drop.stats["dropped_long"] == 1
    ds_cut = SFTDataset([long_rec] + _records(3), TOK, SFTDataConfig(max_seq_len=48, truncation="right"))
    assert len(ds_cut) == 4 and ds_cut.stats["truncated"] == 1
    assert max(len(i[0]) for i in ds_cut.items) <= 49
    packed = SFTDataset(_records(12), TOK, SFTDataConfig(max_seq_len=128, packing=True))
    assert len(packed) < 12 and packed.stats["packed_sequences"] == len(packed)
    for ids, _mask in packed.items:
        assert len(ids) <= 129
    a = SFTDataset(_records(12), TOK, SFTDataConfig(max_seq_len=64), seed=1)
    b = SFTDataset(_records(12), TOK, SFTDataConfig(max_seq_len=64), seed=1)
    assert (a.batch_at(3, 4)["input_ids"] == b.batch_at(3, 4)["input_ids"]).all()


def test_preference_dataset():
    recs = [{"prompt": f"q{i}", "chosen": "good answer", "rejected": "bad"} for i in range(5)]
    ds = SFTDataset(recs, TOK, SFTDataConfig(max_seq_len=64))
    assert ds.kind == "preference"
    b = ds.get_batch(np.array([0, 1, 2]))
    assert b["n_pairs"] == 3 and b["input_ids"].shape[0] == 6
    assert (b["labels"][:3] != IGNORE_INDEX).any() and (b["labels"][3:] != IGNORE_INDEX).any()
    with pytest.raises(ConfigError):
        SFTDataset(recs + _records(1), TOK, SFTDataConfig())


def test_build_sft_datasets_val_split(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _records(40)), encoding="utf-8")
    tr, va = build_sft_datasets(SFTDataConfig(train_files=[str(p)], val_fraction=0.25, max_seq_len=64), TOK, 0)
    assert va is not None and len(tr) + len(va) == 40 and 0 < len(va) < 20
    tr2, va2 = build_sft_datasets(SFTDataConfig(train_files=[str(p)], val_fraction=0.25, max_seq_len=64), TOK, 5)
    assert len(va2) == len(va)  # split is a function of file content, not the seed


# ----------------------------------------------------------------------------------------------- schedules
def test_lr_schedules():
    s = ScheduleConfig(name="cosine", warmup_steps=10, min_lr_ratio=0.1)
    f = [lr_factor(i, s, 100) for i in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0) and f[10] == pytest.approx(1.0)
    assert f[-1] == pytest.approx(0.1, abs=5e-3) and all(a >= b - 1e-9 for a, b in zip(f[10:], f[11:], strict=False))
    lin = ScheduleConfig(name="linear", warmup_steps=0, min_lr_ratio=0.0)
    assert lr_factor(50, lin, 100) == pytest.approx(0.5, abs=0.02)
    const = ScheduleConfig(name="constant", warmup_steps=5)
    assert lr_factor(1000, const, 100) == 1.0
    w = ScheduleConfig(name="wsd", warmup_steps=5, decay_ratio=0.2, min_lr_ratio=0.0)
    assert anneal_start_step(w, 100) == 80
    assert lr_factor(50, w, 100) == 1.0 and lr_factor(79, w, 100) == 1.0 and lr_factor(99, w, 100) < 0.1
    for style in ("cosine", "sqrt", "exp"):
        ws = ScheduleConfig(name="wsd", warmup_steps=0, decay_ratio=0.5, min_lr_ratio=0.1, wsd_decay_style=style)
        vals = [lr_factor(i, ws, 100) for i in range(50, 100)]
        assert vals[0] == pytest.approx(1.0, abs=1e-6) and vals[-1] < 0.5 and all(a >= b - 1e-9 for a, b in zip(vals, vals[1:], strict=False))
    wr = ScheduleConfig(name="cosine", warmup_ratio=0.1, warmup_steps=0)
    assert lr_factor(9, wr, 100) == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------- optimisers
def _quad_model():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    return m


@pytest.mark.parametrize("name", ["adamw", "lion", "sgd", "adamw8bit"])
def test_optimizers_reduce_loss(name):
    m = _quad_model()
    cfg = OptimConfig(name=name, lr=0.05 if name != "lion" else 0.01, weight_decay=0.01)
    opt = build_optimizer(m.named_parameters(), cfg)
    x = torch.randn(16, 4)
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = m(x).pow(2).mean() + (m[0].weight ** 2).mean()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first


def test_param_groups_decay_and_multipliers():
    m = _quad_model()
    cfg = OptimConfig(weight_decay=0.2, lr_multipliers={"0.weight": 0.1})
    opt = build_optimizer(m.named_parameters(), cfg)
    by = {(g["weight_decay"], g["lr_mult"]): len(g["params"]) for g in opt.param_groups}
    assert by[(0.2, 0.1)] == 1  # the 2-D weight with multiplier & decay
    assert by[(0.0, 1.0)] == 3  # two biases + the norm weight: no decay
    set_lr(opt, 1.0)
    assert sorted(g["lr"] for g in opt.param_groups) == [0.1, 1.0]
    with pytest.raises(ValueError):
        build_optimizer([("w", torch.nn.Parameter(torch.zeros(2), requires_grad=False))], cfg)


def test_lion_update_is_signed():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
    p.grad = torch.tensor([5.0, -0.001])
    opt.step()
    assert torch.allclose(p, torch.tensor([0.9, -0.9]))


# ----------------------------------------------------------------------------------------------- losses
def test_ce_loss_sum_matches_reference_and_chunking():
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 11, requires_grad=True)
    labels = torch.randint(0, 11, (3, 7))
    labels[0, :3] = -100
    tot, n, _ = ce_loss_sum(logits, labels)
    ref = torch.nn.functional.cross_entropy(logits.view(-1, 11), labels.view(-1), ignore_index=-100, reduction="sum")
    assert float(n) == 18 and torch.allclose(tot, ref)
    tot2, _, _ = ce_loss_sum(logits, labels, chunk_tokens=5)
    assert torch.allclose(tot, tot2)
    tot2.backward()
    g1 = logits.grad.clone()
    logits.grad = None
    ce_loss_sum(logits, labels)[0].backward()
    assert torch.allclose(g1, logits.grad, atol=1e-6)
    z, _, parts = ce_loss_sum(logits, labels, z_loss=1e-2)
    assert float(parts["z_sum"]) > 0 and float(z) > float(tot)


def test_kd_losses_zero_when_equal_and_positive_otherwise():
    torch.manual_seed(1)
    s = torch.randn(2, 5, 9)
    t = torch.randn(2, 5, 9)
    mask = torch.ones(2, 5, dtype=torch.bool)
    for kind in ("forward_kl", "reverse_kl", "jsd"):
        same, n = kd_loss_dense(s, s, mask, kind)
        assert float(same) == pytest.approx(0.0, abs=1e-5) and float(n) == 10
        diff, _ = kd_loss_dense(s, t, mask, kind, temperature=2.0)
        assert float(diff) > 0
    # forward vs reverse KL are different (asymmetric), JSD is symmetric in swapping at beta=0.5
    f, _ = kd_loss_dense(s, t, mask, "forward_kl")
    r, _ = kd_loss_dense(s, t, mask, "reverse_kl")
    assert float(f) != pytest.approx(float(r))
    j1, _ = kd_loss_dense(s, t, mask, "jsd", jsd_beta=0.5)
    j2, _ = kd_loss_dense(t, s, mask, "jsd", jsd_beta=0.5)
    assert float(j1) == pytest.approx(float(j2), rel=1e-5)


def test_topk_kd_matches_dense_when_k_is_vocab_and_bounds_otherwise():
    torch.manual_seed(2)
    V = 12
    s = torch.randn(2, 4, V)
    t = torch.randn(2, 4, V)
    mask = torch.ones(2, 4, dtype=torch.bool)
    lt = torch.log_softmax(t, -1)
    v, i = lt.topk(V, -1)
    res = torch.zeros(2, 4)
    for kind in ("forward_kl", "reverse_kl", "jsd"):
        dense, _ = kd_loss_dense(s, t, mask, kind)
        topk, _ = kd_loss_topk(s, i, v, res, mask, kind)
        assert float(topk) == pytest.approx(float(dense), rel=1e-4, abs=1e-5)
    # k < V: coarse-grained bin KL is a lower bound of the full forward KL; uniform tail is close for mild tails
    k = 4
    v, i = lt.topk(k, -1)
    res = (1 - v.exp().sum(-1)).clamp(min=0)
    dense, _ = kd_loss_dense(s, t, mask, "forward_kl")
    binned, _ = kd_loss_topk(s, i, v, res, mask, "forward_kl", tail_mode="bin")
    assert 0 < float(binned) <= float(dense) + 1e-4
    uni, _ = kd_loss_topk(s, i, v, res, mask, "forward_kl", tail_mode="uniform")
    assert float(uni) >= float(binned) - 1e-4
    ren, _ = kd_loss_topk(s, i, v, res, mask, "forward_kl", tail_mode="renorm")
    assert float(ren) >= 0
    assert float(kd_loss_topk(s, i, v, res, mask, "reverse_kl", tail_mode="uniform")[0]) > 0  # falls back to bin
    # masked positions do not count
    m0 = torch.zeros(2, 4, dtype=torch.bool)
    assert float(kd_loss_topk(s, i, v, res, m0)[0]) == 0.0


def test_dpo_loss_properties():
    pc, pr = torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -1.0])
    rc, rr = torch.tensor([-1.5, -1.5]), torch.tensor([-2.0, -2.0])
    loss, d = dpo_loss(pc, pr, rc, rr, beta=0.5)
    assert loss.shape == (2,) and loss[0] < loss[1] and d["reward_acc"].tolist() == [1.0, 0.0]
    for lt in ("ipo", "hinge"):
        assert dpo_loss(pc, pr, rc, rr, 0.5, lt)[0].shape == (2,)
    # at policy == reference the sigmoid loss is log 2
    z, _ = dpo_loss(rc, rr, rc, rr, 0.3)
    assert torch.allclose(z, torch.full((2,), math.log(2)), atol=1e-6)
    lg = torch.randn(2, 5, 7)
    lb = torch.randint(0, 7, (2, 5))
    lb[:, :2] = -100
    s = response_logprobs(lg, lb)
    a = response_logprobs(lg, lb, average=True)
    assert torch.allclose(s / 3, a, atol=1e-6)
