"""Engine <-> forge.models.ForgeLM integration on CPU: the paged runner reproduces ForgeLM's own forward for every
supported structure (GQA, qk-norm, sliding window, RoPE scaling, MoE, tied head, quantised / LoRA linears)."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("forge.models.tiny")

from forge.models.tiny import build_tiny_model  # noqa: E402
from forge.serve._testing import ByteTokenizer  # noqa: E402
from forge.serve.config import ServeConfig  # noqa: E402
from forge.serve.engine import LLMEngine  # noqa: E402
from forge.serve.runner import SeqInput  # noqa: E402
from forge.serve.sampling import SamplingParams  # noqa: E402


def make(model, **kw):
    base = dict(model="t", device="cpu", dtype="float32", block_size=4, num_gpu_blocks=64, max_model_len=128, max_num_seqs=4)
    base.update(kw)
    return LLMEngine(ServeConfig(**base), model=model, tokenizer=ByteTokenizer())


def ref_greedy(model, prompt, n):
    ids, out = list(prompt), []
    with torch.no_grad():
        for _ in range(n):
            t = int(model(torch.tensor([ids])).logits[0, -1].argmax())
            ids.append(t)
            out.append(t)
    return out


GREEDY = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)

VARIANTS = [
    {},
    dict(moe=True),
    dict(sliding_window=8),
    dict(qk_norm=True),
    dict(tie_word_embeddings=True),
    dict(num_kv_heads=4, num_heads=4),
    dict(rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                       "original_max_position_embeddings": 64}),
    dict(rope_scaling={"rope_type": "linear", "factor": 2.0}),
    dict(rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64}),
]


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda k: ",".join(k) or "plain")
def test_engine_matches_forgelm_forward(kw):
    m = build_tiny_model(3, **kw).eval()
    e = make(m)
    p = list(range(5, 40))
    o = e.generate([p], [GREEDY])[0]
    assert o.token_ids == ref_greedy(m, p, 12)


def test_ragged_batch_and_logits_close():
    m = build_tiny_model(4).eval()
    e = make(m)
    a, b = list(range(3, 30)), list(range(50, 58))
    for sid, p in (("a", a), ("b", b)):
        e.manager.allocate(sid, p)
    out = e.runner.forward([SeqInput(a, 0, e.manager.block_table("a")), SeqInput(b, 0, e.manager.block_table("b"))])
    with torch.no_grad():
        ra = m(torch.tensor([a])).logits[0, -1]
        rb = m(torch.tensor([b])).logits[0, -1]
    assert torch.allclose(out.logits[0][-1], ra, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out.logits[1][-1], rb, atol=1e-4, rtol=1e-4)


def test_prefix_cache_chunked_prefill_and_spec_with_forgelm():
    m = build_tiny_model(5, tie_word_embeddings=True).eval()
    shared = list(range(10, 50))
    prompts = [shared + [60, 61], shared + [70], shared + [80, 81, 82]]
    refs = [ref_greedy(m, p, 10) for p in prompts]
    e = make(m, max_num_batched_tokens=16, speculation=dict(mode="ngram", max_gamma=3, controller="fixed"))
    for p, r in zip(prompts, refs):
        assert e.generate([p], [SamplingParams(temperature=0, max_tokens=10, ignore_eos=True)])[0].token_ids == r
    assert e.stats()["prefix_hit_rate"] > 0.3


def test_quantised_and_lora_linears_run_transparently():
    from forge.models.linear import quantize_linears
    from forge.models.lora import add_lora

    m = build_tiny_model(6).eval()
    names = quantize_linears(m, "fq_int", bits=8, group_size=32)
    assert names
    p = list(range(5, 35))
    e = make(m)
    assert e.generate([p], [GREEDY])[0].token_ids == ref_greedy(m, p, 12)

    m2 = build_tiny_model(7).eval()
    add_lora(m2, r=4, alpha=8.0)
    for mod in m2.modules():  # make the adapters non-trivial
        if hasattr(mod, "lora_B"):
            w = mod.lora_B.weight if hasattr(mod.lora_B, "weight") else mod.lora_B
            torch.nn.init.normal_(w, std=0.05)
    e2 = make(m2)
    assert e2.generate([p], [GREEDY])[0].token_ids == ref_greedy(m2, p, 12)


def test_fp8_packed_linear_close_to_model_forward():
    from forge.models.linear import quantize_linears

    m = build_tiny_model(8).eval()
    quantize_linears(m, "fp8")
    e = make(m)
    p = list(range(5, 35))
    e.manager.allocate("x", p)
    out = e.runner.forward([SeqInput(p, 0, e.manager.block_table("x"))])
    with torch.no_grad():
        ref = m(torch.tensor([p])).logits[0, -1]
    assert torch.allclose(out.logits[0][-1], ref, atol=2e-3, rtol=2e-3)


def test_model_without_expected_structure_is_rejected_clearly():
    from forge.core.errors import ConfigError
    from forge.serve.runner import ModelAdapter

    class Bad(torch.nn.Module):
        config = type("C", (), dict(hidden_size=8, num_layers=1, num_heads=2, num_kv_heads=2, vocab_size=10))()

    with pytest.raises(ConfigError):
        ModelAdapter(Bad())
