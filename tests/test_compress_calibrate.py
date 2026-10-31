"""Calibration registry, statistics collection and the stats cache."""
from __future__ import annotations

import json

import pytest
import torch

from forge.compress.calibrate import (
    CalibrationSet,
    RouterStats,
    StatsCache,
    collect_stats,
    encode_text,
    get_calibration_set,
    get_calibration_tokens,
    list_calibration_sets,
    register_calibration_set,
)
from forge.compress.modelutil import model_fingerprint
from forge.core.errors import ConfigError, NotFound
from tests.compress_helpers import make_lm, make_moe_lm


class FakeTok:
    """HF-like tokenizer: whitespace words hashed into [0, vocab)."""

    eos_token_id = 1

    def __init__(self, vocab=128):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=True):
        return [2 + (sum(map(ord, w)) % (self.vocab - 2)) for w in text.split()]


def test_builtin_sets_listed_and_resolved():
    names = {s["name"] for s in list_calibration_sets()}
    assert {"synthetic", "builtin-mix", "builtin-prose", "builtin-code", "builtin-math", "builtin-chat"} <= names
    assert get_calibration_set("builtin-mix").kind == "mixture"
    with pytest.raises(NotFound):
        get_calibration_set("no-such-set")


def test_text_set_sampling_hash_and_tokenizers(tmp_path):
    cs = register_calibration_set("t-texts", ["alpha beta gamma delta " * 20, "epsilon zeta eta theta " * 20])
    a = cs.sample_tokens(5, 32, tokenizer=FakeTok(), vocab_size=128, seed=1)
    b = cs.sample_tokens(5, 32, tokenizer=FakeTok(), vocab_size=128, seed=1)
    c = cs.sample_tokens(5, 32, tokenizer=FakeTok(), vocab_size=128, seed=2)
    assert a.shape == (5, 32) and torch.equal(a, b) and not torch.equal(a, c) and int(a.max()) < 128
    h1 = cs.content_hash()
    h2 = register_calibration_set("t-texts2", ["alpha beta"]).content_hash()
    assert h1 != h2 and h1 == register_calibration_set("t-texts", ["alpha beta gamma delta " * 20, "epsilon zeta eta theta " * 20]).content_hash()
    # byte fallback without a tokenizer; vocab folding for tiny vocabularies
    fb = cs.sample_tokens(2, 16, tokenizer=None, vocab_size=64, seed=0)
    assert int(fb.max()) < 64
    # callable tokenizer and dict-style outputs
    assert encode_text(lambda t: [1, 2, 3], "x") == [1, 2, 3]
    assert encode_text(lambda t: {"input_ids": [4, 5]}, "x") == [4, 5]
    # tokenizer ids beyond the model vocabulary are rejected, not silently clipped
    with pytest.raises(ConfigError):
        cs.sample_tokens(2, 16, tokenizer=FakeTok(vocab=1000), vocab_size=128)


def test_file_and_token_sets(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("first paragraph of text " * 10 + "\n\n" + "second paragraph of text " * 10)
    js = tmp_path / "c.jsonl"
    js.write_text("\n".join(json.dumps({"text": "line number %d with words" % i}) for i in range(30)))
    for path in (txt, js):
        cs = get_calibration_set(f"file:{path}")
        t = cs.sample_tokens(3, 24, tokenizer=FakeTok(), vocab_size=128)
        assert t.shape == (3, 24)
    assert get_calibration_set(str(txt)).kind == "file"
    h = get_calibration_set(f"file:{txt}").content_hash()
    txt.write_text("changed")
    assert get_calibration_set(f"file:{txt}").content_hash() != h
    with pytest.raises(NotFound):
        get_calibration_set("file:/does/not/exist").sample_tokens(1, 8)
    toks = torch.randint(0, 100, (4, 50))
    ts = register_calibration_set("t-tokens", tokens=toks)
    s = ts.sample_tokens(3, 20, seed=3)
    assert s.shape == (3, 20) and ts.content_hash() != register_calibration_set("t-tokens2", tokens=toks + 1).content_hash()
    with pytest.raises(ConfigError):
        register_calibration_set("bad", ["a"], path="x")


def test_mixture_proportions_and_domains():
    mix = register_calibration_set("t-mix", components={"builtin-prose": 3, "builtin-code": 1})
    t, labels = mix.sample_labeled(8, 24, tokenizer=None, vocab_size=256, seed=0)
    assert t.shape == (8, 24) and labels.count("builtin-prose") == 6 and labels.count("builtin-code") == 2
    assert mix.content_hash() != get_calibration_set("builtin-mix").content_hash()
    syn, lab = get_calibration_tokens("synthetic", None, vocab_size=64, n_seqs=3, seq_len=10)
    assert syn.max() < 64 and lab == ["synthetic"] * 3
    with pytest.raises(ConfigError):
        CalibrationSet("s", "synthetic").sample_tokens(2, 8)  # synthetic needs a vocabulary size


def test_collect_stats_matches_manual_hessian_and_channel_stats(tmp_path):
    m = make_lm()
    toks = get_calibration_tokens("synthetic", m, n_seqs=6, seq_len=16, seed=4)[0]
    layer = "layers.0.self_attn.q_proj"
    xs = []
    h = m.layers[0].self_attn.q_proj.register_forward_hook(lambda _m, i, o: xs.append(i[0].detach().reshape(-1, 64)))
    with torch.no_grad():
        for b in toks.split(2):
            m(b)
    h.remove()
    X = torch.cat(xs)
    H_ref = 2.0 / X.shape[0] * X.t() @ X
    cache = collect_stats(m, None, "synthetic", stats=("hessian", "channel_scale"), n_seqs=6, seq_len=16, batch_size=2, seed=4)
    mh, sh = cache.model_hash, cache.set_hash
    assert mh == model_fingerprint(m)
    H = cache.get(mh, sh, layer, "hessian")
    assert torch.allclose(H, H_ref, rtol=1e-4, atol=1e-5)
    cs = cache.get(mh, sh, layer, "channel_scale")
    assert torch.allclose(cs["mean_abs"], X.abs().mean(0), rtol=1e-4) and torch.allclose(cs["abs_max"], X.abs().amax(0)) and cs["n"] == X.shape[0]
    assert torch.allclose(cs["mean_sq"], X.pow(2).mean(0), rtol=1e-4)
    # key semantics: (model fingerprint, set hash, layer)
    assert cache.has(mh, sh, layer, "hessian") and not cache.has(mh, "other-set", layer) and not cache.has("other-model", sh, layer)
    assert layer in cache.layers(mh, sh) and cache.get(mh, sh, "nope", "hessian", default=7) == 7
    with torch.no_grad():
        m.layers[0].mlp.up_proj.weight.add_(1e-3)
    assert model_fingerprint(m) != mh  # a modified model has a different cache namespace
    # layer filter and hessian size cap
    c2 = collect_stats(m, None, "synthetic", layers=[r"layers\.1\.mlp\.down_proj$"], stats=("hessian",), n_seqs=4, seq_len=16, max_hessian_dim=100)
    assert c2.layers(c2.model_hash, c2.set_hash) == ["__meta__", "layers.1.mlp.down_proj"] or c2.layers(c2.model_hash, c2.set_hash) == ["__meta__"]
    c3 = collect_stats(m, None, "synthetic", layers=[r"layers\.1\.mlp\.down_proj$"], stats=("hessian",), n_seqs=4, seq_len=16)
    assert c3.get(c3.model_hash, c3.set_hash, "layers.1.mlp.down_proj", "hessian").shape == (128, 128)
    with pytest.raises(ConfigError):
        collect_stats(m, None, "synthetic", stats=("bogus",))


def test_stats_cache_disk_spill(tmp_path):
    c = StatsCache(tmp_path / "sc")
    c.put("m", "s", "L", "channel_scale", {"mean_abs": torch.arange(3.0)})
    c.put("m", "s", "L", "hessian", torch.eye(4))
    c2 = StatsCache(tmp_path / "sc")  # fresh process view
    assert torch.equal(c2.get("m", "s", "L", "channel_scale")["mean_abs"], torch.arange(3.0))
    assert c2.get("m", "s", "L", "hessian") is None  # hessians are not spilled by default
    c3 = StatsCache(tmp_path / "sc2", spill_hessian=True)
    c3.put("m", "s", "L", "hessian", torch.eye(4))
    assert torch.equal(StatsCache(tmp_path / "sc2").get("m", "s", "L", "hessian"), torch.eye(4))
    assert c2.has("m", "s", "L") and len(c2) == 1


def test_router_stats_semantics_and_domains():
    m = make_moe_lm()
    mix = register_calibration_set("t-mix-router", components={"builtin-prose": 1, "builtin-code": 1})
    cache = collect_stats(m, None, mix, stats=("router",), n_seqs=8, seq_len=16, batch_size=2, track_coactivation=True)
    st = cache.get(cache.model_hash, cache.set_hash, "layers.1.mlp", "router")
    assert isinstance(st, RouterStats) and st.num_experts == 4 and st.top_k == 2
    assert st.tokens == 8 * 16 and int(st.counts.sum()) == st.tokens * 2
    assert torch.isclose(st.freq.sum(), torch.tensor(2.0), atol=1e-4)
    assert torch.allclose(st.gate_mass.sum(), torch.tensor(1.0), atol=1e-3)  # norm_topk: weights of a token sum to 1
    assert st.mean_out_norm is not None and (st.mean_out_norm > 0).all()
    assert set(st.domain_counts) == {"builtin-prose", "builtin-code"}
    assert torch.allclose(sum(st.domain_counts.values()), st.counts)
    assert st.coact.shape == (4, 4) and torch.allclose(torch.diag(st.coact), st.counts, atol=1e-3)
    d = st.to_dict()
    assert len(d["freq"]) == 4 and set(d["domains"]) == {"builtin-prose", "builtin-code"}
