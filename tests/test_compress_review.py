"""Adversarial-review regression tests for forge.compress (GPTQ reference equivalence, input-group discovery, delta exactness/determinism,
lossless state-dict edge cases, calibration-cache deserialisation safety)."""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from forge.compress.delta import DeltaConfig, apply_delta, compute_delta
from forge.compress.lossless import LosslessConfig, lossless_decode_dir, lossless_encode_state_dict
from forge.compress.modelutil import discover_input_groups
from forge.compress.ptq import _int_quantizers, gptq_core


# ------------------------------------------------------------------------------------------------ GPTQ vs the textbook algorithm
def _textbook_gptq(W, H, bits, group, blocksize, damp):
    """Frantar et al. reference loop (sym int, groups taken from the error-updated weights), written independently of gptq_core."""
    W = W.clone().float()
    n, k = W.shape
    maxq = 2**bits - 1
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H[range(k), range(k)] += damp * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    s = z = None
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        W1, Q1, E1, Hi = W[:, i1:i2].clone(), torch.zeros(n, i2 - i1), torch.zeros(n, i2 - i1), Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, d = W1[:, i], Hi[i, i]
            if (i1 + i) % group == 0:
                amax = W1[:, i : i + group].abs().amax(1)
                s = amax / (2 ** (bits - 1) - 1)
                s = torch.where(s > 0, s, torch.ones_like(s))
                z = torch.full_like(s, 2 ** (bits - 1))
            q = (torch.clamp(torch.round(w / s) + z, 0, maxq) - z) * s
            Q1[:, i] = q
            e = (w - q) / d
            W1[:, i:] -= e.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_gptq_core_equals_textbook_gptq(bits, block_size):
    g = torch.Generator().manual_seed(0)
    n, k = 24, 128
    X = torch.randn(512, k, generator=g) * torch.rand(k, generator=g) * 3
    H = 2 * X.t() @ X / 512
    W = torch.randn(n, k, generator=g)
    pf, qf = _int_quantizers(bits, True, torch.float32, False)
    _, _, deq, _ = gptq_core(W, H, group=32, block_size=block_size, act_order=False, damp=0.01, static_groups=False, params_fn=pf, quant_fn=qf)
    assert torch.allclose(deq, _textbook_gptq(W, H, bits, 32, block_size, 0.01), atol=1e-6)


# ------------------------------------------------------------------------------------------------ input-group discovery
class _ChainBlock(nn.Module):
    """Three *independent* same-shape linears whose inputs are short-lived temporaries (the allocator happily recycles their addresses)."""

    def __init__(self) -> None:
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 64)

    def forward(self, x):
        h = self.l1(x * 1.0)
        h = self.l2(torch.relu(h) * 1.0)
        return self.l3(torch.relu(h) * 1.0)


class _Binp:
    def run(self, block, i):
        return block(torch.randn(2, 8, 64))


def test_input_group_discovery_is_not_fooled_by_recycled_addresses():
    b = _ChainBlock()
    for _ in range(5):  # allocator behaviour varies; every run must give three singleton groups
        assert discover_input_groups(b, _Binp(), {"l1": b.l1, "l2": b.l2, "l3": b.l3}) == [["l1"], ["l2"], ["l3"]]


def test_input_group_discovery_still_groups_true_siblings():
    class Sib(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.o = nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 64)

        def forward(self, x):
            h = x * 2.0
            return self.o(torch.relu(self.q(h) + self.k(h)) * 1.0)

    s = Sib()
    assert discover_input_groups(s, _Binp(), {"q": s.q, "k": s.k, "o": s.o}) == [["q", "k"], ["o"]]


# ------------------------------------------------------------------------------------------------ delta
class _Emb(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.embed_tokens = nn.Embedding(*v.shape)
        self.embed_tokens.weight.data = v.clone()


def test_full_diff_of_bf16_tensors_reconstructs_the_finetune_exactly():
    g = torch.Generator().manual_seed(0)
    base = (torch.randn(50, 32, generator=g) * 1e-3).to(torch.bfloat16)
    ft = (base.float() * (1 + 1e-2 * torch.randn(50, 32, generator=g))).to(torch.bfloat16)
    d = compute_delta({"embed_tokens.weight": base}, {"embed_tokens.weight": ft}, DeltaConfig())
    out = apply_delta(_Emb(base), d)
    assert torch.equal(out.embed_tokens.weight.view(torch.int16), ft.view(torch.int16))


def test_full_diff_survives_fp16_overflow():
    base = torch.zeros(4, 4, dtype=torch.float32)
    ft = base + 1e6
    d = compute_delta({"embed_tokens.weight": base}, {"embed_tokens.weight": ft})
    assert torch.equal(apply_delta(_Emb(base), d).embed_tokens.weight, ft)


def test_lowrank_delta_is_reproducible_and_independent_of_global_rng():
    g = torch.Generator().manual_seed(1)
    base = {"layers.0.mlp.up_proj.weight": torch.randn(64, 128, generator=g)}
    ft = {"layers.0.mlp.up_proj.weight": base["layers.0.mlp.up_proj.weight"] + 0.1 * torch.randn(64, 128, generator=g)}
    cfg = DeltaConfig(method="lowrank", rank=4, seed=3)
    torch.manual_seed(111)
    a = compute_delta(base, ft, cfg).entries["layers.0.mlp.up_proj"]
    torch.manual_seed(222)
    b = compute_delta(base, ft, cfg).entries["layers.0.mlp.up_proj"]
    assert torch.equal(a.A, b.A) and torch.equal(a.B, b.B)


# ------------------------------------------------------------------------------------------------ lossless state dict
def test_lossless_state_dict_handles_scalars_empty_bool_and_fp8(tmp_path):
    sd = {
        "a": torch.randn(2048).bfloat16(), "scale": torch.tensor(1.5), "empty": torch.empty(0), "mask": torch.tensor([True, False]),
        "ids": torch.arange(5000), "f8": torch.randn(3000).to(torch.float8_e4m3fn),
    }
    for kw in ({}, {"min_numel": 0}):
        rep = lossless_encode_state_dict(sd, LosslessConfig(**kw), tmp_path / f"o{len(kw)}")
        assert rep["tensors"] == len(sd)
        back = lossless_decode_dir(tmp_path / f"o{len(kw)}")
        for k, v in sd.items():
            assert back[k].shape == v.shape and back[k].dtype == v.dtype, k
            assert torch.equal(back[k].reshape(-1).view(torch.uint8) if v.numel() else back[k], v.reshape(-1).view(torch.uint8) if v.numel() else v), k


# ------------------------------------------------------------------------------------------------ calibration cache safety
def test_stats_cache_spill_does_not_unpickle_arbitrary_objects(tmp_path):
    from forge.compress.calibrate import StatsCache

    marker = tmp_path / "pwned.txt"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"echo pwned > {marker}",))

    c = StatsCache(tmp_path / "cache")
    c.put("m", "s", "layer", "hessian", torch.eye(2))
    torch.save({"key": ["m", "s", "layer"], "stats": {"hessian": Evil()}}, c._path(("m", "s", "layer")))
    fresh = StatsCache(tmp_path / "cache")
    assert fresh.get("m", "s", "layer", "hessian") is None  # treated as a cache miss
    assert fresh.has("m", "s", "layer", "hessian") is False
    assert not marker.exists()


def test_stats_cache_roundtrips_router_stats_with_restricted_loader(tmp_path):
    from forge.compress.calibrate import RouterStats, StatsCache

    rs = RouterStats("l", 4, 2, 10, torch.ones(4), torch.ones(4), torch.ones(4))
    StatsCache(tmp_path / "c").put("m", "s", "l", "router", rs)
    back = StatsCache(tmp_path / "c").get("m", "s", "l", "router")
    assert isinstance(back, RouterStats) and back.num_experts == 4 and torch.equal(back.counts, torch.ones(4))


# ------------------------------------------------------------------------------------------------ run_compress safety
def test_run_compress_refuses_to_delete_a_non_artifact_save_dir(tmp_path):
    from forge.compress.config import CompressJobConfig
    from forge.core.errors import ConfigError
    from forge.compress.run import run_compress

    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "thesis.docx").write_bytes(b"do not delete")
    cfg = CompressJobConfig(model="tiny:3", dtype="float32", device="cpu", use_store=False, save_dir=str(precious),
                            recipe="recipe: r\nstages:\n- ptq: {method: rtn, bits: 8, group: 64}\n")
    with pytest.raises(ConfigError, match="refusing to overwrite"):
        run_compress(cfg)
    assert (precious / "thesis.docx").read_bytes() == b"do not delete"
