"""Packed T158 artifact load and autoregressive generation on an NVIDIA device.

This test is intentionally GPU-only; run it on the rented Linux host, never on the CPU development machine.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def test_t158_artifact_load_and_generate_on_cuda(tmp_path):
    from forge.compress.serialize import save_quantized
    from forge.core.artifactmeta import read_artifact_summary
    from forge.models.loader import describe_loaded_model, load_model
    from forge.models.tiny import build_tiny_model
    from forge.ternary.convert import ternarize_model

    model = build_tiny_model(seed=71).to("cuda").eval()
    report = ternarize_model(model, group_size=32, rule="lloyd", iters=1)
    assert report.layers
    artifact = tmp_path / "ternary-inference"
    save_quantized(model, artifact)

    summary = read_artifact_summary(artifact)
    assert summary and summary["is_quantized"] and summary["label"] == "Ternary 1.58b"
    loaded, _ = load_model(artifact, dtype="float32", device="cuda", with_tokenizer=False)
    info = describe_loaded_model(loaded)
    assert info["quantized"] and info["layer_kinds"].get("t158", 0) > 0

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        logits = loaded(prompt).logits
        generated = loaded.generate(prompt, max_new_tokens=2)
        torch.cuda.synchronize()
    assert logits.shape == (1, 3, loaded.config.vocab_size)
    assert torch.isfinite(logits).all()
    assert generated.shape == (1, 5)
    assert int(generated.min()) >= 0 and int(generated.max()) < loaded.config.vocab_size
