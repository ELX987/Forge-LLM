"""Sampling math, incremental detokenisation / stop strings, ServeConfig schema (UI forms), package surface."""
from __future__ import annotations

import pytest
import torch

from forge.core.errors import ConfigError
from forge.serve._testing import ByteTokenizer
from forge.serve.sampling import (
    IncrementalDetokenizer,
    SamplingParams,
    make_generator,
    process_logits,
    sample_token,
    sampling_probs,
    token_logprobs,
)


def test_params_validation_and_defaults():
    with pytest.raises(ConfigError):
        SamplingParams(temperature=-1)
    with pytest.raises(ConfigError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ConfigError):
        SamplingParams(max_tokens=0)
    with pytest.raises(ConfigError):
        SamplingParams(min_tokens=5, max_tokens=3)
    p = SamplingParams(stop="x", logit_bias={"3": 1})
    assert p.stop == ["x"] and p.logit_bias == {3: 1.0}
    from forge.serve.config import SamplingDefaults

    q = SamplingParams.from_defaults(SamplingDefaults(temperature=0.3, max_tokens=7), top_k=5, seed=None, bogus=1)
    assert q.temperature == 0.3 and q.max_tokens == 7 and q.top_k == 5 and q.seed is None
    assert q.clone(n=3).n == 3 and q.n == 1


def test_top_k_top_p_min_p_support():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
    p = sampling_probs(logits, SamplingParams(temperature=1.0, top_k=2))
    assert (p > 0).sum() == 2 and p.sum().item() == pytest.approx(1.0)
    full = torch.softmax(logits, -1)
    # top-p keeps the smallest prefix whose mass >= top_p (always at least one token)
    pp = sampling_probs(logits, SamplingParams(temperature=1.0, top_p=float(full[0]) + 1e-4))
    assert (pp > 0).sum() == 2
    assert (sampling_probs(logits, SamplingParams(temperature=1.0, top_p=0.01)) > 0).sum() == 1
    mp = sampling_probs(logits, SamplingParams(temperature=1.0, min_p=0.5))   # keep p >= 0.5 * pmax
    assert set(torch.nonzero(mp).flatten().tolist()) == {i for i in range(5) if full[i] >= 0.5 * full.max()}
    g = sampling_probs(logits, SamplingParams(temperature=0.0))
    assert g.argmax().item() == 0 and g.max().item() == 1.0
    hot = sampling_probs(logits, SamplingParams(temperature=100.0))
    assert hot.max() - hot.min() < 0.05


def test_penalties_and_bias():
    logits = torch.tensor([2.0, 2.0, -2.0, 0.0])
    x = process_logits(logits, SamplingParams(repetition_penalty=2.0), prompt_ids=[0], output_ids=[2])
    assert x.tolist() == [1.0, 2.0, -4.0, 0.0]                      # >0 divided, <0 multiplied, only seen tokens
    x = process_logits(logits, SamplingParams(presence_penalty=1.0, frequency_penalty=0.5), prompt_ids=[], output_ids=[1, 1, 3])
    assert x.tolist() == [2.0, 2.0 - 1.0 - 1.0, -2.0, 0.0 - 1.0 - 0.5]
    x = process_logits(logits, SamplingParams(logit_bias={3: 5.0, 99: 1.0}))
    assert x.tolist() == [2.0, 2.0, -2.0, 5.0]


def test_sampling_distribution_matches_probs_and_is_seeded():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    prm = SamplingParams(temperature=1.0)
    g = make_generator(123)
    draws = torch.tensor([sample_token(logits, prm, generator=g) for _ in range(6000)])
    freq = torch.bincount(draws, minlength=3).float() / 6000
    assert torch.allclose(freq, torch.tensor([0.5, 0.3, 0.2]), atol=0.03)
    a = [sample_token(logits, prm, generator=make_generator(5)) for _ in range(3)]
    assert len(set(a)) == 1


def test_logprobs():
    lg = torch.tensor([1.0, 2.0, 3.0])
    d = token_logprobs(lg, 1, 2)
    lp = torch.log_softmax(lg, -1)
    assert d["logprob"] == pytest.approx(float(lp[1])) and [t for t, _ in d["top"]] == [2, 1]


def test_incremental_detok_utf8_boundaries_and_stop_holdback():
    tok = ByteTokenizer()
    text = "héllo wörld"
    ids = tok.encode(text)
    d = IncrementalDetokenizer(tok, [])
    out = "".join(d.append([i]) for i in ids) + d.flush()
    assert out == text                                   # multi-byte chars are never split
    d = IncrementalDetokenizer(tok, [], stop=["wör"])
    pieces = [d.append([i]) for i in ids]
    assert d.stopped and "".join(pieces) == "héllo " and d.stop_string == "wör"
    # text that merely *starts* like a stop string is held back, then released when it diverges
    d = IncrementalDetokenizer(tok, [], stop=["abc"])
    assert d.append(tok.encode("xa")) == "x"
    assert d.append(tok.encode("b")) == ""
    assert d.append(tok.encode("d")) == "abd"
    d = IncrementalDetokenizer(tok, [], stop=["abc"], include_stop=True)
    d.append(tok.encode("zzabc"))
    assert d.text == "zzabc" and d.stopped


def test_serve_config_schema_drives_the_ui():
    from forge.core.jsonschema import defaults_for, schema_for
    from forge.serve.config import ServeConfig

    sch = schema_for(ServeConfig)
    props = sch["properties"]
    assert "$defs" not in str(sch.keys())
    for key in ("model", "dtype", "kv_dtype", "block_size", "max_model_len", "max_num_seqs", "prefix_caching", "chunked_prefill",
                "speculation", "host", "port", "api_key", "sampling", "cuda_graphs", "tp_size", "gpu_mem_util", "policy"):
        assert key in props, key
    for name, spec in props.items():
        assert spec.get("description"), f"{name} lacks a description"
    assert props["speculation"]["properties"]["mode"]["enum"] == ["off", "ngram", "draft", "eagle", "auto"]
    assert all(v.get("description") for v in props["speculation"]["properties"].values())
    assert all(v.get("description") for v in props["sampling"]["properties"].values())
    d = defaults_for(ServeConfig)
    assert d["prefix_caching"] is True and d["speculation"]["mode"] == "off" and d["block_size"] == 16
    cfg = ServeConfig(**{**d, "model": "m"})
    assert cfg.display_name() == "m"
    with pytest.raises(Exception):
        ServeConfig(bogus=1)
    with pytest.raises(Exception):
        ServeConfig(nvme_kv_gb=1.0)  # needs nvme_kv_dir


def test_tp_size_is_rejected_honestly():
    from forge.serve._testing import make_engine, tiny_model

    with pytest.raises(ConfigError):
        make_engine(tiny_model(0), tp_size=2)


def test_package_lazy_exports():
    import forge.serve as s

    assert s.ServeConfig.__name__ == "ServeConfig" and callable(s.capacity_plan) and s.KVBlockManager.__name__ == "KVBlockManager"
    with pytest.raises(AttributeError):
        s.NotThere  # noqa: B018


def test_kv_pool_payload_roundtrip_all_dtypes():
    from forge.serve.runner import KVPool

    for kv in ("fp32", "bf16", "fp8_e4m3", "int8"):
        pool = KVPool(4, 4, 2, 2, 8, torch.float32, kv, torch.device("cpu"))
        k = torch.randn(3, 2, 8)
        v = torch.randn(3, 2, 8)
        blk, off = torch.tensor([1, 1, 1]), torch.tensor([0, 1, 2])
        for layer in range(2):
            pool.write(layer, blk, off, k, v)
        before = pool.gather(0, torch.tensor([1]), 3)
        payload = pool.read_block(1)
        pool.k[:, 1] = 0
        pool.v[:, 1] = 0
        pool.write_block(2, payload)
        after = pool.gather(0, torch.tensor([2]), 3)
        assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
        pool.copy_block(2, 3)
        assert torch.equal(pool.gather(0, torch.tensor([3]), 3)[0], after[0])
        assert pool.bytes_per_block > 0
    with pytest.raises(ConfigError):
        KVPool(2, 4, 1, 1, 8, torch.float32, "int4", torch.device("cpu"))
