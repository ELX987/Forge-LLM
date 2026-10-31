"""GPU-host tests for forge.store (auto-skipped without CUDA by the root conftest).

The store itself is CPU code; what needs validating on a GPU box is the device->host path:
CUDA tensors checkpointed (sync + async), restored onto the GPU, and lossless-encoded.
These tests have NOT been run by the authors of this module (no NVIDIA hardware was available).
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture()
def store(tmp_path):
    from forge.store import ArtifactStore

    return ArtifactStore(tmp_path / "store")


def _bits(t: torch.Tensor) -> bytes:
    from forge.store.lossless import tensor_to_array

    return tensor_to_array(t)[0].tobytes()


def test_cuda_checkpoint_roundtrip_and_device_restore(store):
    from forge.store.checkpoints import CheckpointReader, CheckpointWriter

    w = CheckpointWriter(store, "gpu", keyframe_every=3)
    p = torch.randn(1 << 20, device="cuda").to(torch.bfloat16)
    snaps = {}
    for step in range(1, 6):
        p.add_(0.001)
        snaps[step] = p.clone()
        w.save(step, {"p": p, "opt": {"m": torch.randn(1 << 16, device="cuda")}})
    rd = CheckpointReader(store, "gpu")
    for step, s in snaps.items():
        got = rd.load(step)["p"]
        assert got.device.type == "cpu" and _bits(got) == _bits(s)
    on_gpu = rd.load(5, device="cuda")["p"]
    assert on_gpu.is_cuda and _bits(on_gpu) == _bits(snaps[5])


def test_cuda_async_save_snapshot_is_isolated_from_later_updates(store):
    from forge.store.checkpoints import CheckpointReader, CheckpointWriter

    w = CheckpointWriter(store, "gpu_async", keyframe_every=4)
    p = torch.randn(1 << 18, device="cuda")
    expect = p.clone()
    fut = w.save_async(1, {"p": p})
    p.mul_(2.0)  # training continues while the writer thread encodes
    fut.result()
    assert _bits(CheckpointReader(store, "gpu_async").load(1)["p"]) == _bits(expect)


def test_lossless_encode_cuda_tensor():
    from forge.store import lossless as L

    t = (torch.randn(1 << 20, device="cuda") * 0.02).to(torch.bfloat16)
    blob = L.encode_tensor(t)
    out = L.decode_tensor(blob)
    assert _bits(out) == _bits(t)
    assert len(blob) < 0.72 * t.numel() * 2
