"""Speculative decoding: closed forms (P7), controller (Plan 7.5), exact rejection sampling, drafters, engine exactness."""
from __future__ import annotations

import copy
import math

import pytest
import torch

from forge.serve._testing import make_engine, tiny_model
from forge.serve.sampling import SamplingParams
from forge.serve.spec import (
    AcceptanceEstimator,
    NgramDrafter,
    SpecController,
    StepCostModel,
    expected_tokens,
    expected_tokens_iid,
    optimal_gamma_iid,
    rejection_sample,
    rejection_sample_batched,
    speedup_iid,
)


def sp(**kw):
    base = dict(temperature=0.0, ignore_eos=True)
    base.update(kw)
    return SamplingParams(**base)


# ---------------------------------------------------------------------------------------- closed forms
@pytest.mark.parametrize("alpha,gamma,speed", [(0.6, 2, 1.96), (0.7, 3, 2.53), (0.8, 2, 2.44), (0.8, 5, 3.69),
                                               (0.9, 2, 2.71), (0.9, 10, 6.86)])
def test_leviathan_table_free_draft(alpha, gamma, speed):
    # P7 Table 4.2: free draft (c = 0) speed-ups reproduced to two decimals
    assert round(speedup_iid(alpha, gamma, 0.0), 2) == speed


def test_expected_tokens_general_matches_iid_and_alpha_one():
    for a in (0.2, 0.5, 0.9):
        for g in (0, 1, 4, 8):
            assert expected_tokens([a] * g, g) == pytest.approx(expected_tokens_iid(a, g))
    assert expected_tokens_iid(1.0, 5) == 6.0
    # position dependent: 1 + a0 + a0*a1 + a0*a1*a2
    assert expected_tokens([0.9, 0.5, 0.2], 3) == pytest.approx(1 + 0.9 + 0.45 + 0.09)


def test_speculation_cannot_help_when_alpha_below_cost_ratio():
    # P7 Prop. 4.4: S > 1 iff alpha > c
    assert optimal_gamma_iid(0.3, 0.3) == 0
    assert optimal_gamma_iid(0.2, 0.5) == 0
    assert optimal_gamma_iid(0.8, 0.05) > 0
    assert all(speedup_iid(0.25, g, 0.25) <= 1.0 + 1e-12 for g in range(1, 12))


# ---------------------------------------------------------------------------------------- controller
def brute_force_choose(ctrl, alphas, batch, kv_bytes, max_gamma):
    """Literal transcription of the Plan 7.5 pseudocode."""
    c = ctrl.c
    base = ctrl.cost.base_step_time(batch, kv_bytes)
    best, best_gain = 0, 1.0
    for g in range(1, max_gamma + 1):
        e_tok = sum(math.prod(alphas[:k]) for k in range(g + 1))
        t_step = ctrl.cost.verify_time(batch * (g + 1), kv_bytes) + g * c * base
        gain = e_tok / (t_step / base)
        if gain > best_gain * 1.03:
            best, best_gain = g, gain
    return best


def make_ctrl(**kw):
    cost = StepCostModel(weight_bytes=16e9, params=8e9, ridge_flop_per_byte=250.0)
    base = dict(max_gamma=8, cost=cost, draft_cost_ratio=0.05, max_batch_for_spec=10**6)
    base.update(kw)
    g = base.pop("max_gamma")
    return SpecController(g, **base)


def test_choose_gamma_matches_plan_pseudocode_and_load_awareness():
    import random

    rng = random.Random(0)
    for _ in range(40):
        ctrl = make_ctrl(draft_cost_ratio=rng.choice([0.01, 0.05, 0.2]))
        alphas = sorted([rng.uniform(0.1, 0.95) for _ in range(8)], reverse=True)
        ctrl.est._a["w"] = list(alphas)
        for b in (1, 4, 16, 64, 256, 1024):
            kv = b * 1024 * 131072
            assert ctrl.choose_gamma("w", b, kv) == brute_force_choose(ctrl, alphas, b, kv, 8)
    ctrl = make_ctrl()
    ctrl.est._a["w"] = [0.85] * 8
    assert ctrl.choose_gamma("w", 1) >= 3            # memory bound: speculation pays
    assert ctrl.choose_gamma("w", 2048) == 0         # compute bound: verification costs (gamma+1)x, never pays
    gains = ctrl.gains("w", 2048)
    assert gains[0] == 1.0 and max(gains[1:]) < 1.0


def test_gamma_zero_under_high_batch_by_default_guard():
    ctrl = SpecController(5, cost=None, max_batch_for_spec=8)
    ctrl.est._a["w"] = [0.9] * 5
    assert ctrl.choose_gamma("w", 8) > 0
    assert ctrl.choose_gamma("w", 9) == 0


def test_hysteresis_requires_margin():
    ctrl = make_ctrl(hysteresis=0.0)
    ctrl.est._a["w"] = [0.6] * 8
    g0 = ctrl.choose_gamma("w", 1)
    ctrl2 = make_ctrl(hysteresis=1.2)
    ctrl2.est._a["w"] = [0.6] * 8
    assert g0 > 0 and ctrl2.choose_gamma("w", 1) == 0  # best modelled gain (~1.9) < 2.2 -> stay at gamma 0


def test_acceptance_estimator_learns_per_position():
    import random

    rng = random.Random(1)
    est = AcceptanceEstimator(4, decay=0.02, prior=0.5)
    true = [0.9, 0.7, 0.5, 0.3]
    for _ in range(6000):
        acc = 0
        for a in true:
            if rng.random() < a:
                acc += 1
            else:
                break
        est.observe("c", 4, acc)
    got = est.per_position("c")
    for g, t in zip(got, true):
        assert abs(g - t) < 0.08, (got, true)


def test_controller_stats_report_acceptance_by_position():
    ctrl = make_ctrl()
    ctrl.observe("x", 3, 2)
    ctrl.observe("x", 3, 0)
    st = ctrl.stats()
    assert st["proposed_by_position"][:3] == [2, 2, 2]
    assert st["accepted_by_position"][:3] == [1, 1, 0]
    assert st["acceptance_by_position"][0] == 0.5


# ---------------------------------------------------------------------------------------- rejection sampling exactness
def _softmax_rows(v, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(v, v, generator=g) * scale, dim=-1)


def _run_chain(P, Q, L, N, G, seed, deterministic_draft=False):
    """Generate L tokens from a Markov chain target P using draft chain Q and speculative verification."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.full((N, L), -1, dtype=torch.long)
    count = torch.zeros(N, dtype=torch.long)
    last = torch.zeros(N, dtype=torch.long)  # start state 0
    ar = torch.arange(N)
    while bool((count < L).any()):
        prev = last
        drafts, qs = [], []
        cur = prev
        for _ in range(G):
            qd = Q[cur]
            d = qd.argmax(-1) if deterministic_draft else torch.multinomial(qd, 1, generator=gen).squeeze(-1)
            drafts.append(d)
            qs.append(qd)
            cur = d
        draft = torch.stack(drafts, 1)
        chain = torch.cat([prev.unsqueeze(1), draft], 1)       # [N, G+1] contexts
        p = P[chain]                                           # p_i conditioned on (prefix, draft[:i])
        q = None if deterministic_draft else torch.stack(qs, 1)
        toks, n = rejection_sample_batched(p, q, draft, None, gen)
        active = count < L
        for j in range(G + 1):
            m = active & (j <= n) & (count + j < L)
            out[ar[m], (count + j)[m]] = toks[m, j]
        count = torch.where(active, torch.clamp(count + n + 1, max=L), count)
        last = out[ar, count - 1]
    return out


def _joint(out, v):
    idx = (out[:, 0] * v + out[:, 1]) * v + out[:, 2]
    return torch.bincount(idx, minlength=v ** 3).double() / out.shape[0]


def _exact_joint(P, v):
    j = torch.zeros(v, v, v, dtype=torch.double)
    for a in range(v):
        for b in range(v):
            for c in range(v):
                j[a, b, c] = P[0, a] * P[a, b] * P[b, c]
    return j.reshape(-1)


def test_rejection_sampling_matches_target_joint_distribution():
    v, n = 4, 60000
    P, Q = _softmax_rows(v, 1, 1.2), _softmax_rows(v, 2, 1.2)
    exact = _exact_joint(P.double(), v)
    q_joint = _exact_joint(Q.double(), v)
    for g in (1, 3):
        emp = _joint(_run_chain(P, Q, 3, n, g, seed=7), v)
        tv = 0.5 * (emp - exact).abs().sum().item()
        assert tv < 0.03, (g, tv)
    # power check: the draft distribution is genuinely different from the target
    assert 0.5 * (q_joint - exact).abs().sum().item() > 0.08


def test_rejection_sampling_exact_with_point_mass_drafts():
    v, n = 4, 60000
    P, Q = _softmax_rows(v, 3, 1.0), _softmax_rows(v, 4, 1.0)
    emp = _joint(_run_chain(P, Q, 3, n, 2, seed=9, deterministic_draft=True), v)
    tv = 0.5 * (emp - _exact_joint(P.double(), v)).abs().sum().item()
    assert tv < 0.03


def test_rejection_sampling_greedy_target_is_deterministic_argmax():
    v = 5
    p = torch.zeros(4, v)
    p[0, 2] = p[1, 3] = p[2, 1] = p[3, 4] = 1.0
    q = torch.softmax(torch.randn(3, v), -1)
    toks, n = rejection_sample(p, q, [2, 3, 0])
    assert toks == [2, 3, 1] and n == 2          # third draft rejected -> target's argmax emitted
    toks, n = rejection_sample(p, None, [2, 3, 1])
    assert toks == [2, 3, 1, 4] and n == 3       # all accepted -> bonus token from p[G]
    toks, n = rejection_sample(p, None, [0, 3, 1])
    assert toks == [2] and n == 0


def test_rejection_sample_zero_length_draft_is_plain_sampling():
    p = torch.softmax(torch.randn(1, 6), -1)
    toks, n = rejection_sample(p, None, [])
    assert n == 0 and len(toks) == 1


# ---------------------------------------------------------------------------------------- drafters
def test_ngram_drafter_prompt_lookup():
    d = NgramDrafter(ngram_max=3, ngram_min=1)
    toks = [5, 6, 7, 8, 9, 1, 2, 5, 6, 7]
    assert d.lookup(toks, 4) == [8, 9, 1, 2]
    assert d.lookup([1, 2, 3, 4], 3) == []
    assert d.lookup([9, 9, 9, 9, 9], 3) == [9, 9]
    assert d.lookup([9, 9, 9, 9, 9, 9, 9, 9], 3) == [9, 9, 9]
    # prefers the longest suffix match
    assert NgramDrafter(3, 1).lookup([1, 2, 3, 9, 2, 3, 4, 0, 1, 2, 3], 1) == [9]


# ---------------------------------------------------------------------------------------- engine integration
def ref_tokens(m, p, n):
    return m.greedy_generate(p, n, eos=None)


LOOPY = dict(tie_word_embeddings=True)  # tied tiny model falls into repetition loops -> n-gram drafts get accepted


def test_ngram_spec_greedy_exact_and_accepts():
    m = tiny_model(0, **LOOPY)
    e = make_engine(m, speculation=dict(mode="ngram", max_gamma=4, controller="fixed"))
    p = list(b"hello world, hello")
    o = e.generate([p], [sp(max_tokens=40)])[0]
    assert o.token_ids == ref_tokens(m, p, 40)
    st = e.stats()
    assert st["spec"]["steps"] > 0 and st["spec_accept_rate"] > 0.3
    assert st["spec"]["tokens_per_step"] > 1.3
    assert st["kv"]["rollback_tokens"] >= 0


def test_ngram_spec_no_proposals_still_exact():
    m = tiny_model(5)
    e = make_engine(m, speculation=dict(mode="ngram", max_gamma=3, controller="fixed"))
    p = list(range(10, 50))
    o = e.generate([p], [sp(max_tokens=20)])[0]
    assert o.token_ids == ref_tokens(m, p, 20)


def test_draft_model_identical_weights_accepts_everything():
    m = tiny_model(1)
    draft = copy.deepcopy(m)
    e = make_engine(m, draft_model=None, speculation=dict(mode="draft", max_gamma=4, controller="fixed")) if False else None
    from forge.serve._testing import ByteTokenizer
    from forge.serve.config import ServeConfig
    from forge.serve.engine import LLMEngine

    cfg = ServeConfig(model="tiny", device="cpu", dtype="float32", block_size=4, num_gpu_blocks=96, max_model_len=160,
                      max_num_seqs=8, max_num_batched_tokens=64, speculation=dict(mode="draft", max_gamma=4, controller="fixed"))
    e = LLMEngine(cfg, model=m, tokenizer=ByteTokenizer(), draft_model=draft)
    p = list(b"abcdefgh")
    o = e.generate([p], [sp(max_tokens=30)])[0]
    assert o.token_ids == ref_tokens(m, p, 30)
    st = e.stats()["spec"]
    assert st["overall_accept_rate"] > 0.95
    assert st["tokens_per_step"] > 3.5


def _draft_engine(m, draft, **spec):
    from forge.serve._testing import ByteTokenizer
    from forge.serve.config import ServeConfig
    from forge.serve.engine import LLMEngine

    base = dict(model="tiny", device="cpu", dtype="float32", block_size=4, num_gpu_blocks=96, max_model_len=160,
                max_num_seqs=8, max_num_batched_tokens=64)
    scfg = dict(mode="draft", max_gamma=3, controller="fixed")
    scfg.update(spec)
    return LLMEngine(ServeConfig(**base, speculation=scfg), model=m, tokenizer=ByteTokenizer(), draft_model=draft)


def test_draft_model_different_weights_greedy_exact_batched():
    m, d = tiny_model(2), tiny_model(99)
    e = _draft_engine(m, d)
    prompts = [list(b"abcdef"), list(range(20, 50)), list(b"zz")]
    outs = e.generate(prompts, [sp(max_tokens=16) for _ in prompts])
    for p, o in zip(prompts, outs):
        assert o.token_ids == ref_tokens(m, p, 16)


def test_spec_with_penalties_and_stop_mid_draft():
    m = tiny_model(3, **LOOPY)
    p = list(b"hello world, hello")
    base = make_engine(m).generate([p], [sp(max_tokens=20, repetition_penalty=1.3, frequency_penalty=0.2)])[0].token_ids
    e = make_engine(m, speculation=dict(mode="ngram", max_gamma=5, controller="fixed"))
    got = e.generate([p], [sp(max_tokens=20, repetition_penalty=1.3, frequency_penalty=0.2)])[0].token_ids
    assert got == base
    # max_tokens reached in the middle of an accepted draft
    o = e.generate([p], [sp(max_tokens=5)])[0]
    assert o.token_ids == ref_tokens(m, p, 5) and len(o.token_ids) == 5


def test_spec_under_chunked_prefill_prefix_cache_and_preemption():
    m = tiny_model(4, **LOOPY)
    spec = dict(mode="ngram", max_gamma=4, controller="fixed")
    e = make_engine(m, speculation=spec, num_gpu_blocks=24, max_num_seqs=4, max_model_len=96, max_num_batched_tokens=10)
    prompts = [list(range(i, i + 12)) for i in (5, 50, 90, 130)] + [list(b"abababababab")]
    outs = e.generate(prompts, [sp(max_tokens=20) for _ in prompts])
    for p, o in zip(prompts, outs):
        assert o.token_ids == ref_tokens(m, p, 20)
    assert e.manager.stats()["blocks_in_use"] == 0


def test_spec_sampling_preserves_marginal_distribution():
    """Sampled decoding with a (different) draft model has the same joint law over the first two tokens as plain sampling."""
    m, d = tiny_model(6), tiny_model(77)
    prompt = list(b"abc")
    top_k, temp = 3, 1.0
    lg1 = m.logits_for(prompt)[-1]
    p1 = torch.zeros(260, dtype=torch.double)
    idx = lg1.topk(top_k).indices
    p1[idx] = torch.softmax(lg1[idx].double() / temp, -1)
    exact = {}
    for a in idx.tolist():
        lg2 = m.logits_for(prompt + [a])[-1]
        i2 = lg2.topk(top_k).indices
        p2 = torch.softmax(lg2[i2].double() / temp, -1)
        for b, pb in zip(i2.tolist(), p2.tolist()):
            exact[(a, b)] = float(p1[a]) * pb
    n = 900
    for use_spec in (True, False):
        e = _draft_engine(m, d, max_gamma=2) if use_spec else make_engine(m)
        counts: dict[tuple, int] = {}
        outs = e.generate([prompt] * n, [SamplingParams(temperature=temp, top_k=top_k, max_tokens=3, seed=1000 + i, ignore_eos=True)
                                         for i in range(n)])
        for o in outs:
            key = tuple(o.token_ids[:2])  # the 3rd token only exists so that a draft of length 1 is possible
            counts[key] = counts.get(key, 0) + 1
        tv = 0.5 * (sum(abs(counts.get(k, 0) / n - p) for k, p in exact.items()) +
                    sum(c / n for k, c in counts.items() if k not in exact))
        assert tv < 0.09, (use_spec, tv)
        if use_spec:
            assert e.stats()["spec"]["steps"] > 0


# ---------------------------------------------------------------------------------------- eagle
def test_eagle_head_roundtrip_training_and_engine_exactness(tmp_path):
    from forge.serve.spec import EagleHead, eagle_training_loss

    m = tiny_model(8, **LOOPY)
    torch.manual_seed(0)
    head = EagleHead(m.config.hidden_size, num_blocks=1)
    ids = torch.randint(0, 250, (4, 24))
    with torch.no_grad():
        _, hid = m(ids, return_hidden=True)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss = eagle_training_loss(head, hid, ids, m.model.embed_tokens, m.lm_head)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8
    path = tmp_path / "head.pt"
    head.save(str(path))
    head2 = EagleHead.load(str(path))
    x = torch.randn(3, m.config.hidden_size)
    t = torch.randn(3, m.config.hidden_size)
    assert torch.allclose(head(x, t), head2(x, t))

    e = make_engine(m, speculation=dict(mode="off", max_gamma=3, controller="fixed"))
    e.attach_eagle_head(head2)
    p = list(b"hello world, hello")
    o = e.generate([p], [sp(max_tokens=24)])[0]
    assert o.token_ids == ref_tokens(m, p, 24)
    st = e.stats()["spec"]
    assert st["steps"] > 0 and st["proposed_by_position"][0] > 0
