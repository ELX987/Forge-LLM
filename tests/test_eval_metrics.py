from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from forge.eval import metrics as M


def _logp(p):
    return torch.log(torch.tensor(p, dtype=torch.float32))


def test_kl_known_value_and_zero():
    parent = _logp([[0.5, 0.5]])
    cand = _logp([[0.25, 0.75]])
    kl = M.kl_divergence(parent, cand)
    expect = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert kl.mean == pytest.approx(expect, abs=1e-6) and kl.n == 1 and kl.p99 == pytest.approx(expect, abs=1e-6)
    z = M.kl_divergence(parent, parent)
    assert z.mean == 0.0 and z.max == 0.0
    # asymmetry: KL(p||c) != KL(c||p)
    rev = M.kl_divergence(cand, parent)
    assert rev.mean != pytest.approx(kl.mean)


def test_kl_mask_p99_and_shapes():
    torch.manual_seed(0)
    p = torch.randn(4, 50, 17)
    c = p + 0.1 * torch.randn_like(p)
    mask = torch.ones(4, 50, dtype=torch.bool)
    mask[:, 25:] = False
    full = M.kl_divergence(p, c, keep_per_position=True)
    part = M.kl_divergence(p, c, mask=mask)
    assert full.n == 200 and part.n == 100
    assert full.p99 >= full.p50 and full.max >= full.p99
    assert full.p99 == pytest.approx(float(np.quantile(full.per_position, 0.99)))
    assert math.isnan(M.kl_divergence(p, c, p99=False).p99)
    with pytest.raises(ValueError):
        M.kl_divergence(p, c[:, :10])
    with pytest.raises(ValueError):
        M.kl_per_position(p, c, mask=torch.ones(3, 3, dtype=torch.bool))


def test_kl_chunking_matches_direct():
    torch.manual_seed(1)
    p = torch.randn(3000, 11)
    c = p + 0.3 * torch.randn_like(p)
    got = M.kl_per_position(p, c).numpy()
    lp, lc = torch.log_softmax(p.double(), -1), torch.log_softmax(c.double(), -1)
    ref = (lp.exp() * (lp - lc)).sum(-1).numpy()
    assert np.allclose(got, ref, atol=1e-12)


def test_flip_rate():
    p = torch.tensor([[[2.0, 0, 0]], [[0, 2.0, 0]], [[0, 0, 2.0]], [[2.0, 0, 0]]])
    c = torch.tensor([[[2.0, 0, 0]], [[2.0, 0, 0]], [[0, 0, 2.0]], [[0, 2.0, 0]]])
    fr = M.top1_flip_rate(p, c)
    assert fr.rate == 0.5 and fr.flips == 2 and fr.n == 4 and fr.ci_low < 0.5 < fr.ci_high
    assert M.top1_flip_rate(p, p).rate == 0.0
    mask = torch.tensor([[True], [False], [True], [False]])
    assert M.top1_flip_rate(p, c, mask=mask).rate == 0.0


def test_first_divergence():
    assert M.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert M.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert M.first_divergence([1, 2], [1, 2, 3]) == 2


def test_entropy_and_shift():
    uniform = torch.zeros(5, 8)
    peaked = torch.zeros(5, 8)
    peaked[:, 0] = 20.0
    h = M.entropy_per_position(uniform)
    assert torch.allclose(h, torch.full((5,), math.log(8), dtype=torch.float64))
    assert M.entropy_shift(uniform, uniform) == 0.0
    assert M.entropy_shift(uniform, peaked) < -1.5  # candidate more confident
    assert M.entropy_shift(peaked, uniform) > 1.5


def test_perplexity():
    V = 16
    lg = torch.zeros(2, 7, V)
    tgt = torch.randint(0, V, (2, 7))
    assert M.perplexity(lg, tgt) == pytest.approx(V, rel=1e-9)
    lg2 = torch.full((1, 3, V), -50.0)
    for t in range(3):
        lg2[0, t, t] = 50.0
    assert M.perplexity(lg2, torch.tensor([[0, 1, 2]])) == pytest.approx(1.0, abs=1e-6)
    mask = torch.tensor([[True, True, False]])
    assert M.perplexity(lg2, torch.tensor([[0, 1, 5]]), mask=mask) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        M.nll_per_position(lg, tgt[:, :3])


def test_ece():
    # perfectly calibrated: 80% confidence, 80% accuracy
    conf = [0.8] * 100
    corr = [True] * 80 + [False] * 20
    r = M.ece_from_confidences(conf, corr)
    assert r.ece == pytest.approx(0.0, abs=1e-9) and r.n == 100
    # overconfident: 100% confidence, 50% accuracy
    r = M.ece_from_confidences([1.0] * 10, [True] * 5 + [False] * 5)
    assert r.ece == pytest.approx(0.5) and r.mce == pytest.approx(0.5)
    assert math.isnan(M.ece_from_confidences([], []).ece)
    # from logits
    logits = torch.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0], [10.0, 0, 0]])
    labels = torch.tensor([0, 1, 2, 1])
    e = M.ece(logits, labels)
    assert e.accuracy == 0.75 and e.ece == pytest.approx(0.25, abs=1e-3)
    e2 = M.ece(torch.softmax(logits, -1), labels, from_probs=True)
    assert e2.ece == pytest.approx(e.ece)


def test_providers_and_batches():
    torch.manual_seed(0)
    table = torch.randn(20, 20)

    def fn(ids):
        return table[ids]

    prov = M.as_provider(fn)
    assert isinstance(prov, M.LogitsProvider)
    assert M.as_provider(prov) is prov
    lg = prov.logits(torch.tensor([[1, 2, 3]]))
    assert lg.shape == (1, 3, 20)

    emb = torch.nn.Embedding(20, 20)
    emb.weight.data = table.clone()
    mp = M.as_provider(emb, name="emb")
    assert torch.allclose(mp.logits(torch.tensor([[1, 2, 3]])), lg)
    assert mp.nbytes() == 20 * 20 * 4
    with pytest.raises(TypeError):
        M.as_provider(42)

    out = M.greedy_generate(prov, [3], 5)
    assert len(out) == 5 and all(0 <= t < 20 for t in out)
    assert out == M.generate(prov, [3], 5)
    s1 = M.greedy_generate(prov, [3], 6, temperature=1.0, seed=7)
    assert s1 == M.greedy_generate(prov, [3], 6, temperature=1.0, seed=7)
    assert M.greedy_generate(prov, [3], 5, stop_ids=[out[1]]) == out[:2]


def test_compare_distributions_identical_and_perturbed():
    torch.manual_seed(0)
    table = torch.randn(32, 32)
    parent = M.CallableProvider(lambda ids: table[ids], name="p")
    seqs = [list(range(i, i + 9)) for i in range(0, 12)]
    same = M.compare_distributions(parent, parent, seqs, batch_size=5)
    s = M.summarize_distribution(same)
    assert same.identical_logits and same.max_abs_logit_diff == 0.0
    assert s["kl_mean"] == 0.0 and s["top1_flip"] == 0.0 and s["entropy_shift"] == 0.0 and s["ppl_ratio"] == pytest.approx(1.0)
    assert same.n_sequences == 12 and same.n_tokens == 12 * 8

    table2 = table + 0.5 * torch.randn_like(table)
    cand = M.CallableProvider(lambda ids: table2[ids], name="c")
    d = M.compare_distributions(parent, cand, seqs, batch_size=4)
    s = M.summarize_distribution(d)
    assert not d.identical_logits and s["kl_mean"] > 0 and s["kl_p99"] >= s["kl_mean"] and 0 < s["top1_flip"] <= 1
    # batching must not change the result
    d2 = M.compare_distributions(parent, cand, seqs, batch_size=1)
    assert M.summarize_distribution(d2)["kl_mean"] == pytest.approx(s["kl_mean"], rel=1e-6)
    with pytest.raises(ValueError):
        M.compare_distributions(parent, M.CallableProvider(lambda ids: table2[ids][..., :10]), seqs)


def test_score_continuations_matches_manual():
    from forge.eval.suites.base import score_continuations

    torch.manual_seed(0)
    table = torch.randn(10, 10)
    prov = M.CallableProvider(lambda ids: table[ids])
    ctxs = [[1, 2], [3]]
    conts = [[4, 5], [6]]
    lp, n = score_continuations(prov, ctxs, conts, batch_size=2)
    lsm = torch.log_softmax(table.double(), -1)
    ref0 = lsm[2, 4] + lsm[4, 5]
    ref1 = lsm[3, 6]
    assert lp[0] == pytest.approx(float(ref0)) and lp[1] == pytest.approx(float(ref1)) and list(n) == [2, 1]
    # left-truncation keeps the continuation
    lp2, _ = score_continuations(prov, [[1, 2, 3, 4]], [[5]], max_len=3)
    assert lp2[0] == pytest.approx(float(lsm[4, 5]))
    with pytest.raises(ValueError):
        score_continuations(prov, [[]], [[1]])
