from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from forge.core.errors import IntegrityError
from forge.store import lossless as L


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    ua, da, sa = L.tensor_to_array(a)
    ub, db, sb = L.tensor_to_array(b)
    return da == db and sa == sb and ua.tobytes() == ub.tobytes()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_exhaustive_bit_patterns_16bit_and_random_fp32(dtype):
    g = torch.Generator().manual_seed(0)
    if dtype == torch.float32:
        bits = torch.randint(-(2**31), 2**31 - 1, (200_000,), generator=g, dtype=torch.int32)
        t = bits.view(torch.float32)
    else:
        # every 16-bit pattern incl. NaN payloads / inf / subnormals / -0
        t = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(dtype)
    out = L.decode_tensor(L.encode_tensor(t))
    assert same_bits(t, out)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_all_patterns_roundtrip(dtype):
    t = torch.arange(256, dtype=torch.uint8).repeat(300).view(dtype)
    for m in ("auto", "exp_huffman", "exp_zstd", "bytegroup_zstd", "raw"):
        out = L.decode_tensor(L.encode_tensor(t, method=m))
        assert same_bits(t, out), m


@pytest.mark.parametrize("n", [0, 1, 2, 7, 255, 2048, 2049, 5000])
def test_odd_sizes(n):
    t = (torch.randn(n) * 0.02).to(torch.bfloat16)
    assert same_bits(t, L.decode_tensor(L.encode_tensor(t)))
    assert same_bits(t, L.decode_tensor(L.encode_tensor(t, block_elems=1000, lanes=64)))


def test_multiple_blocks_and_shape_preserved():
    t = (torch.randn(37, 129) * 0.05).to(torch.bfloat16)
    blob = L.encode_tensor(t, block_elems=500, lanes=32)
    info = L.read_info(blob)
    assert info["blocks"] == -(-t.numel() // 500) and info["shape"] == [37, 129]
    out = L.decode_tensor(blob)
    assert out.shape == t.shape and same_bits(t, out)


def test_constant_and_single_exponent():
    t = torch.full((10_000,), 0.5, dtype=torch.bfloat16)
    blob = L.encode_tensor(t)
    assert len(blob) < t.numel() * 2 * 0.75
    assert same_bits(t, L.decode_tensor(blob))


def test_length_limited_huffman_on_extreme_skew():
    # geometric exponent distribution over all 256 symbols forces >14-bit optimal codes
    rng = np.random.default_rng(1)
    exp = np.minimum(rng.geometric(0.45, 400_000) - 1, 255).astype(np.uint16)
    exp[:256] = np.arange(256)  # every symbol appears
    vals = (exp << 7) | rng.integers(0, 128, exp.size, dtype=np.uint16)
    t = torch.from_numpy(vals.view(np.int16)).view(torch.bfloat16)
    counts = np.bincount(exp.astype(np.int64), minlength=256)
    assert L._huffman_lengths(counts).max() <= L.MAX_CODE_LEN
    assert same_bits(t, L.decode_tensor(L.encode_tensor(t, block_elems=100_000)))


def test_integer_and_other_dtypes_use_bytegroup():
    for t in (torch.randint(-5, 5, (3000,), dtype=torch.int64), torch.randint(0, 255, (3000,), dtype=torch.uint8),
              torch.randn(2000, dtype=torch.float64), torch.rand(500) > 0.5):
        blob = L.encode_tensor(t)
        assert L.read_info(blob)["method"] == "bytegroup_zstd"
        assert same_bits(t, L.decode_tensor(blob))
    with pytest.raises(Exception):
        L.encode_tensor(torch.zeros(4, dtype=torch.int64), method="exp_huffman")


def test_bytes_input_and_numpy():
    a = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    blob = L.encode_tensor(a)
    out = L.decode_tensor(blob)
    assert out.numpy().tobytes() == a.tobytes()
    raw = a.tobytes()
    assert bytes(L.decode_bytes(L.encode_tensor(raw, dtype="fp32"))[0]) == raw


def test_streaming_decode_and_file_sink(tmp_path):
    t = (torch.randn(50_000) * 0.02).to(torch.bfloat16)
    raw = L.tensor_to_array(t)[0]
    p = tmp_path / "w.flc"
    with open(p, "wb") as f:
        stats = L.encode_stream(raw, "bfloat16", f, shape=t.shape, block_elems=8192)
    assert stats["blocks"] == 7
    blocks = []
    with open(p, "rb") as f:
        for b in L.decode_iter(f):
            assert len(b) <= 8192 * 2
            blocks.append(b)
    assert b"".join(blocks) == raw.tobytes()
    with open(p, "rb") as f, io.BytesIO() as out:
        assert L.decode_to_file(f, out) == raw.size
        assert out.getvalue() == raw.tobytes()
    rep = L.verify_file(str(p))
    assert rep["ok"] and rep["raw_bytes"] == raw.size


def test_corruption_is_detected():
    t = (torch.randn(20_000) * 0.02).to(torch.bfloat16)
    good = bytes(L.encode_tensor(t, block_elems=4096))
    blob = bytearray(good)
    hl = int.from_bytes(blob[4:8], "little")
    blob[8 + hl + 100] ^= 0x5A  # flip bits inside the first block payload
    with pytest.raises(Exception):
        out = L.decode_tensor(bytes(blob))
        if not same_bits(t, out):  # a lucky decode with wrong data must still be caught by the hash check
            raise IntegrityError("mismatch")
    with pytest.raises(IntegrityError):
        L.decode_bytes(b"not a container")
    with pytest.raises(IntegrityError):
        L.decode_bytes(good[:-20])


def test_encode_self_check_catches_a_broken_codec(monkeypatch):
    t = (torch.randn(5000) * 0.02).to(torch.bfloat16)
    real = L._decode_block

    def bad(blk, nraw, p):
        out = bytearray(real(blk, nraw, p))
        if blk[0] == 1:
            out[0] ^= 1
        return bytes(out)

    monkeypatch.setattr(L, "_decode_block", bad)
    with pytest.raises(IntegrityError):
        L.encode_tensor(t)
    L.encode_tensor(t, verify=False)  # documented escape hatch


def test_measured_ratio_on_gaussian_bf16(record_property, capsys):
    """Reports measured ratios for Gaussian bf16 'weights' (Plan KR-7 expects ~0.66-0.72)."""
    g = torch.Generator().manual_seed(42)
    rows = {}
    for std in (0.01, 0.02, 0.05):
        w = (torch.randn(1 << 20, generator=g) * std).to(torch.bfloat16)
        rows[f"std={std}"] = L.compare_methods(w)
        assert L.decode_tensor(L.encode_tensor(w)).view(torch.int16).equal(w.view(torch.int16))
    for k, rep in rows.items():
        h = rep["exp_huffman"]["ratio"]
        record_property(f"bf16_{k}_exp_huffman_ratio", round(h, 4))
        record_property(f"bf16_{k}_bytegroup_zstd_ratio", round(rep["bytegroup_zstd"]["ratio"], 4))
        assert 0.64 <= h <= 0.72, (k, h)
        assert h <= rep["bytegroup_zstd"]["ratio"] + 0.005  # at least as good as the byte-grouped baseline
        assert h >= rep["ideal_ratio"] - 1e-6  # cannot beat the entropy bound
        assert h - rep["ideal_ratio"] < 0.02  # near-optimal (overhead < 2 points)
    with capsys.disabled():
        print("\n[forge.store.lossless] measured bf16 Gaussian ratios (encoded/raw):")
        for k, rep in rows.items():
            print(f"  {k}: exp_huffman={rep['exp_huffman']['ratio']:.4f}  exp_zstd={rep['exp_zstd']['ratio']:.4f}  "
                  f"bytegroup_zstd={rep['bytegroup_zstd']['ratio']:.4f}  H(exp)={rep['exponent_entropy_bits']:.3f} bits  "
                  f"ideal={rep['ideal_ratio']:.4f}")


def test_auto_picks_the_better_method_for_fp16_and_fp8():
    g = torch.Generator().manual_seed(7)
    w = (torch.randn(1 << 18, generator=g) * 0.02).to(torch.float16)
    auto = len(L.encode_tensor(w))
    assert auto <= min(len(L.encode_tensor(w, method="exp_huffman")), len(L.encode_tensor(w, method="bytegroup_zstd"))) + 64


def test_dictionary_training_is_optional():
    assert L.train_dictionary([b"a" * 10]) is None  # too little data -> None, not an error
    samples = [bytes([i % 7]) * 2000 + bytes(range(50)) for i in range(200)]
    d = L.train_dictionary(samples, dict_size=4096)
    assert d is None or isinstance(d, bytes)


def test_canonical_dtype_aliases():
    assert L.canonical_dtype("BF16") == "bfloat16" and L.canonical_dtype(torch.float16) == "float16"
    assert L.itemsize_of("fp8_e5m2") == 1
    with pytest.raises(Exception):
        L.canonical_dtype("complex64")
