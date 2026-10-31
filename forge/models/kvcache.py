"""KV-cache classes: dense (optionally quantised, with a full-precision residual window) and the paged view.

Cache protocol (:class:`KVCacheBase`).  Keys/values handed to a cache have shape ``[B, T, Hkv, D]`` (the layout the
attention module produces).  A forward call looks like::

    K, V = cache.update(layer, k, v, positions, n_new)   # write new tokens, return all keys/values [B, S, Hkv, D]
    ...                                                   # (after the LAST layer)
    cache.commit(n_new)                                   # advance sequence lengths once per forward

``n_new`` is the number of valid new tokens per sequence (``[B]`` tensor or int; right-padding only).  ``seq_lens``
always means "tokens already stored *before* the current forward call".

Quantised KV dtypes (``kv_dtype``): ``bf16 fp16 fp32 fp8_e4m3 int8 int4``.  Quantisation is per (token, head) over
``group_size`` channels of the head dimension (default: the whole head) with a symmetric scale; ``int4`` is packed two
values per byte.  ``residual_window=R`` keeps the most recent ``R`` tokens of every sequence in full precision (KIVI-style
residual) and overlays them on the dequantised view.  Per-channel key quantisation (KIVI) lives in ``forge.compress.kvquant``.
"""
from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import torch

from forge.core.errors import ConfigError
from forge.models.config import ModelConfig

KV_DTYPES = ("bf16", "fp16", "fp32", "fp8_e4m3", "int8", "int4")
_TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_QMAX = {"fp8_e4m3": 448.0, "int8": 127.0, "int4": 7.0}
_SCALE_MIN = 6.2e-5  # smallest normal fp16


_KV_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16", "fp16": "fp16", "float16": "fp16", "half": "fp16", "fp32": "fp32", "float32": "fp32",
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "e4m3": "fp8_e4m3", "float8_e4m3fn": "fp8_e4m3", "int8": "int8", "int4": "int4",
}


def kv_dtype_name(dtype: torch.dtype | str | None) -> str:
    """Normalise a dtype / alias to one of :data:`KV_DTYPES` (``None`` -> ``bf16``)."""
    if dtype is None:
        return "bf16"
    if isinstance(dtype, str):
        d = _KV_ALIASES.get(dtype.lower().replace("torch.", ""))
        if d is None:
            raise ConfigError(f"unsupported kv dtype {dtype!r}; choose from {KV_DTYPES}")
        return d
    if dtype == torch.float8_e4m3fn:
        return "fp8_e4m3"
    if dtype in (torch.bfloat16, torch.float16, torch.float32):
        return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
    raise ConfigError(f"unsupported kv dtype {dtype!r}")


def kv_bits(kv_dtype: str) -> int:
    return {"bf16": 16, "fp16": 16, "fp32": 32, "fp8_e4m3": 8, "int8": 8, "int4": 4}[kv_dtype]


def is_quantised(kv_dtype: str) -> bool:
    return kv_dtype in _QMAX


# ------------------------------------------------------------------------------------------- (de)quantisation
def quantize_kv(x: torch.Tensor, kv_dtype: str, group_size: int | None = None, scale_dtype: torch.dtype = torch.float16):
    """Quantise ``x [..., D]`` -> ``(data, scales[..., D/g])``. Quantisation uses the *stored* (rounded) scale."""
    d = x.shape[-1]
    g = d if not group_size or group_size > d else group_size
    if d % g:
        raise ConfigError(f"head_dim {d} not divisible by kv group size {g}")
    xg = x.float().reshape(*x.shape[:-1], d // g, g)
    qmax = _QMAX[kv_dtype]
    scale = (xg.abs().amax(-1, keepdim=True) / qmax).clamp_min(_SCALE_MIN).to(scale_dtype)
    sf = scale.float()
    if kv_dtype == "fp8_e4m3":
        q = (xg / sf).clamp(-qmax, qmax).to(torch.float8_e4m3fn)
    else:
        q = torch.round(xg / sf).clamp(-qmax, qmax).to(torch.int8)
    q = q.reshape(*x.shape)
    if kv_dtype == "int4":
        u = (q + 8).to(torch.uint8)
        q = u[..., 0::2] | (u[..., 1::2] << 4)
    return q, scale.squeeze(-1)


def dequantize_kv(q: torch.Tensor, scale: torch.Tensor, kv_dtype: str, out_dtype: torch.dtype) -> torch.Tensor:
    if kv_dtype == "int4":
        lo = (q & 0x0F).to(torch.int8) - 8
        hi = (q >> 4).to(torch.int8) - 8
        q = torch.stack([lo, hi], dim=-1).reshape(*q.shape[:-1], q.shape[-1] * 2)
    d = q.shape[-1]
    ng = scale.shape[-1]
    qf = q.float().reshape(*q.shape[:-1], ng, d // ng)
    return (qf * scale.float().unsqueeze(-1)).reshape(*q.shape[:-1], d).to(out_dtype)


# ------------------------------------------------------------------------------------------- protocol
@runtime_checkable
class KVCacheBase(Protocol):
    """Interface used by the attention module."""

    seq_lens: torch.Tensor

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor, n_new: torch.Tensor | None) -> None: ...

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor | None = None, n_new: torch.Tensor | int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def commit(self, n_new: torch.Tensor | int) -> None: ...


def _as_n_new(n_new: torch.Tensor | int | None, b: int, t: int, device: torch.device) -> torch.Tensor:
    if n_new is None:
        return torch.full((b,), t, dtype=torch.long, device=device)
    if isinstance(n_new, int):
        return torch.full((b,), n_new, dtype=torch.long, device=device)
    return n_new.to(device=device, dtype=torch.long)


# ------------------------------------------------------------------------------------------- dense cache
class DenseKVCache:
    """Pre-allocated dense KV cache ``[B, max_len, Hkv, D]`` per layer, optionally quantised."""

    def __init__(
        self,
        batch: int,
        max_len: int,
        cfg: ModelConfig,
        dtype: torch.dtype = torch.bfloat16,
        kv_dtype: str | None = None,
        *,
        residual_window: int = 0,
        group_size: int | None = None,
        device: torch.device | str | None = None,
        scale_dtype: torch.dtype = torch.float16,
    ):
        self.cfg = cfg
        self.batch, self.max_len = int(batch), int(max_len)
        self.dtype = dtype
        self.kv_dtype = kv_dtype_name(kv_dtype if kv_dtype is not None else dtype)
        self.quantised = is_quantised(self.kv_dtype)
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.residual_window = int(residual_window) if self.quantised else 0
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        h, d = cfg.kv_heads, int(cfg.head_dim)  # type: ignore[arg-type]
        self.num_layers, self.kv_heads, self.head_dim = cfg.num_layers, h, d
        self.seq_lens = torch.zeros(batch, dtype=torch.long, device=self.device)
        self._S: int | None = None
        shape = (batch, max_len, h, d)
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []
        self.k_scale: list[torch.Tensor] = []
        self.v_scale: list[torch.Tensor] = []
        self.k_res: list[torch.Tensor] = []
        self.v_res: list[torch.Tensor] = []
        g = d if not group_size or group_size > d else group_size
        for _ in range(cfg.num_layers):
            if not self.quantised:
                st = _TORCH_DTYPES[self.kv_dtype]
                self.k.append(torch.zeros(shape, dtype=st, device=self.device))
                self.v.append(torch.zeros(shape, dtype=st, device=self.device))
            else:
                if self.kv_dtype == "int4":
                    if d % 2:
                        raise ConfigError("int4 KV needs an even head_dim")
                    sshape = (batch, max_len, h, d // 2)
                    st = torch.uint8
                elif self.kv_dtype == "int8":
                    sshape, st = shape, torch.int8
                else:
                    sshape, st = shape, torch.float8_e4m3fn
                self.k.append(torch.zeros(sshape, dtype=st, device=self.device))
                self.v.append(torch.zeros(sshape, dtype=st, device=self.device))
                self.k_scale.append(torch.ones((batch, max_len, h, d // g), dtype=scale_dtype, device=self.device))
                self.v_scale.append(torch.ones((batch, max_len, h, d // g), dtype=scale_dtype, device=self.device))
                if self.residual_window:
                    rs = (batch, self.residual_window, h, d)
                    self.k_res.append(torch.zeros(rs, dtype=dtype, device=self.device))
                    self.v_res.append(torch.zeros(rs, dtype=dtype, device=self.device))

    # ---- bookkeeping
    def reset(self, batch_idx: torch.Tensor | list[int] | None = None) -> None:
        if batch_idx is None:
            self.seq_lens.zero_()
        else:
            self.seq_lens[torch.as_tensor(batch_idx, device=self.device)] = 0
        self._S = None

    def truncate(self, new_lens: torch.Tensor | list[int]) -> None:
        """Roll sequences back to ``new_lens`` (speculative-decoding rollback). Never extends."""
        nl = torch.as_tensor(new_lens, dtype=torch.long, device=self.device)
        self.seq_lens = torch.minimum(self.seq_lens, nl)
        self._S = None

    def commit(self, n_new: torch.Tensor | int) -> None:
        n = _as_n_new(n_new, self.batch, 0, self.device)
        self.seq_lens = self.seq_lens + n
        self._S = None

    def end_lens(self, n_new: torch.Tensor | int | None, t: int) -> torch.Tensor:
        return self.seq_lens + _as_n_new(n_new, self.batch, t, self.device)

    def nbytes(self) -> int:
        ts = [*self.k, *self.v, *self.k_scale, *self.v_scale, *self.k_res, *self.v_res]
        return int(sum(x.numel() * x.element_size() for x in ts))

    def bytes_per_token(self) -> float:
        """Allocated bytes per (sequence slot, token) over all layers (includes scales, excludes residual window)."""
        total = self.nbytes() - sum(x.numel() * x.element_size() for x in [*self.k_res, *self.v_res])
        return total / float(self.batch * self.max_len)

    # ---- core
    def _pos(self, positions: torch.Tensor | None, b: int, t: int) -> torch.Tensor:
        if positions is None:
            return self.seq_lens.unsqueeze(1) + torch.arange(t, device=self.device).unsqueeze(0)
        return positions.to(self.device, torch.long)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor | None, n_new: torch.Tensor | int | None = None) -> None:
        b, t = k.shape[0], k.shape[1]
        pos = self._pos(positions, b, t).clamp(max=self.max_len - 1)
        bi = torch.arange(b, device=self.device).unsqueeze(1).expand(b, t)
        if not self.quantised:
            st = self.k[layer].dtype
            self.k[layer][bi, pos] = k.to(st)
            self.v[layer][bi, pos] = v.to(st)
            return
        kq, ks = quantize_kv(k, self.kv_dtype, self.group_size, self.scale_dtype)
        vq, vs = quantize_kv(v, self.kv_dtype, self.group_size, self.scale_dtype)
        self.k[layer][bi, pos] = kq
        self.v[layer][bi, pos] = vq
        self.k_scale[layer][bi, pos] = ks
        self.v_scale[layer][bi, pos] = vs
        if self.residual_window:
            r = self.residual_window
            nn_ = _as_n_new(n_new, b, t, self.device)
            end = self.seq_lens.unsqueeze(1) + nn_.unsqueeze(1)
            tt = torch.arange(t, device=self.device).unsqueeze(0)
            keep = (tt < nn_.unsqueeze(1)) & (pos >= end - r)
            slots = pos % r
            self.k_res[layer][bi[keep], slots[keep]] = k.to(self.k_res[layer].dtype)[keep]
            self.v_res[layer][bi[keep], slots[keep]] = v.to(self.v_res[layer].dtype)[keep]

    def read(self, layer: int, s: int, out_dtype: torch.dtype, end: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        s = max(1, min(int(s), self.max_len))
        if not self.quantised:
            return self.k[layer][:, :s].to(out_dtype), self.v[layer][:, :s].to(out_dtype)
        k = dequantize_kv(self.k[layer][:, :s], self.k_scale[layer][:, :s], self.kv_dtype, out_dtype)
        v = dequantize_kv(self.v[layer][:, :s], self.v_scale[layer][:, :s], self.kv_dtype, out_dtype)
        if self.residual_window and end is not None:
            r = self.residual_window
            b = k.shape[0]
            j = end.unsqueeze(1) - r + torch.arange(r, device=self.device).unsqueeze(0)  # [B,R]
            valid = (j >= 0) & (j < s)
            jj = j.clamp(min=0, max=s - 1)
            bi = torch.arange(b, device=self.device).unsqueeze(1).expand(b, r)
            slots = jj % r
            k[bi[valid], jj[valid]] = self.k_res[layer][bi, slots][valid].to(out_dtype)
            v[bi[valid], jj[valid]] = self.v_res[layer][bi, slots][valid].to(out_dtype)
        return k, v

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor | None = None, n_new: torch.Tensor | int | None = None):
        b, t = k.shape[0], k.shape[1]
        self.write(layer, k, v, positions, n_new)
        end = self.end_lens(n_new, t)
        if self._S is None:
            self._S = int(end.max().item())
        return self.read(layer, self._S, k.dtype, end)


# ------------------------------------------------------------------------------------------- paged
class SimplePagedPool:
    """Minimal reference block pool (the production pool lives in ``forge.serve``). Duck-type used by :class:`PagedKVView`.

    ``k(layer)`` / ``v(layer)``: ``[num_blocks, block_size, Hkv, D]`` (storage dtype); for quantised pools
    ``k_scale(layer)`` / ``v_scale(layer)``: ``[num_blocks, block_size, Hkv, D/g]``.  Block 0 is a valid block (no sentinel).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        kv_dtype: str | None = None,
        device: torch.device | str | None = None,
        group_size: int | None = None,
    ):
        self.kv_dtype = kv_dtype_name(kv_dtype if kv_dtype is not None else dtype)
        if self.kv_dtype == "int4":
            raise ConfigError("SimplePagedPool does not support int4")
        self.block_size, self.num_blocks, self.dtype, self.group_size = block_size, num_blocks, dtype, group_size
        dev = torch.device(device) if device is not None else torch.device("cpu")
        h, d = cfg.kv_heads, int(cfg.head_dim)  # type: ignore[arg-type]
        shape = (num_blocks, block_size, h, d)
        st = _TORCH_DTYPES.get(self.kv_dtype) or (torch.int8 if self.kv_dtype == "int8" else torch.float8_e4m3fn)
        self._k = [torch.zeros(shape, dtype=st, device=dev) for _ in range(cfg.num_layers)]
        self._v = [torch.zeros(shape, dtype=st, device=dev) for _ in range(cfg.num_layers)]
        g = d if not group_size or group_size > d else group_size
        q = is_quantised(self.kv_dtype)
        self._ks = [torch.ones((num_blocks, block_size, h, d // g), dtype=torch.float16, device=dev) for _ in range(cfg.num_layers)] if q else None
        self._vs = [torch.ones((num_blocks, block_size, h, d // g), dtype=torch.float16, device=dev) for _ in range(cfg.num_layers)] if q else None

    def k(self, layer: int) -> torch.Tensor:
        return self._k[layer]

    def v(self, layer: int) -> torch.Tensor:
        return self._v[layer]

    def k_scale(self, layer: int) -> torch.Tensor | None:
        return self._ks[layer] if self._ks is not None else None

    def v_scale(self, layer: int) -> torch.Tensor | None:
        return self._vs[layer] if self._vs is not None else None


class PagedKVView:
    """Per-step view of a block pool for a batch of sequences.

    ``block_tables [B, max_blocks]`` (int, physical block id per logical block), ``seq_lens [B]`` = tokens already stored
    before this forward call, ``pool`` = object exposing ``k(layer) v(layer) k_scale(layer) v_scale(layer) block_size kv_dtype``
    (see :class:`SimplePagedPool`).  Only valid new tokens (``t < n_new``) are written, so padding never touches unallocated blocks.
    """

    def __init__(self, block_tables: torch.Tensor, seq_lens: torch.Tensor, pool: Any):
        self.block_tables = block_tables
        self.seq_lens = seq_lens.to(block_tables.device, torch.long)
        self.pool = pool
        self.block_size = int(pool.block_size)
        self.kv_dtype = kv_dtype_name(getattr(pool, "kv_dtype", "bf16"))
        self.device = block_tables.device
        self._S: int | None = None

    def commit(self, n_new: torch.Tensor | int) -> None:
        self.seq_lens = self.seq_lens + _as_n_new(n_new, self.seq_lens.shape[0], 0, self.device)
        self._S = None

    def end_lens(self, n_new: torch.Tensor | int | None, t: int) -> torch.Tensor:
        return self.seq_lens + _as_n_new(n_new, self.seq_lens.shape[0], t, self.device)

    def _pos(self, positions: torch.Tensor | None, t: int) -> torch.Tensor:
        if positions is None:
            return self.seq_lens.unsqueeze(1) + torch.arange(t, device=self.device).unsqueeze(0)
        return positions.to(self.device, torch.long)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor | None, n_new: torch.Tensor | int | None = None) -> None:
        b, t = k.shape[0], k.shape[1]
        pos = self._pos(positions, t)
        nn_ = _as_n_new(n_new, b, t, self.device)
        valid = torch.arange(t, device=self.device).unsqueeze(0) < nn_.unsqueeze(1)
        bi = torch.arange(b, device=self.device).unsqueeze(1).expand(b, t)
        blk = self.block_tables[bi, (pos // self.block_size).clamp(max=self.block_tables.shape[1] - 1)]
        off = pos % self.block_size
        blk, off = blk[valid].long(), off[valid]
        kk, vv = k[valid], v[valid]  # [N,Hkv,D]
        kp, vp = self.pool.k(layer), self.pool.v(layer)
        if is_quantised(self.kv_dtype):
            g = getattr(self.pool, "group_size", None)
            kq, ks = quantize_kv(kk, self.kv_dtype, g)
            vq, vs = quantize_kv(vv, self.kv_dtype, g)
            kp[blk, off] = kq
            vp[blk, off] = vq
            self.pool.k_scale(layer)[blk, off] = ks.to(self.pool.k_scale(layer).dtype)
            self.pool.v_scale(layer)[blk, off] = vs.to(self.pool.v_scale(layer).dtype)
        else:
            kp[blk, off] = kk.to(kp.dtype)
            vp[blk, off] = vv.to(vp.dtype)

    def read(self, layer: int, s: int, out_dtype: torch.dtype, end: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        s = max(1, int(s))
        nb = int(math.ceil(s / self.block_size))
        tbl = self.block_tables[:, :nb].clamp(min=0).long()
        b = tbl.shape[0]

        def gather(pool_t: torch.Tensor) -> torch.Tensor:
            x = pool_t[tbl]  # [B, nb, bs, ...]
            return x.reshape(b, nb * self.block_size, *x.shape[3:])[:, :s]

        kp, vp = gather(self.pool.k(layer)), gather(self.pool.v(layer))
        if is_quantised(self.kv_dtype):
            ks, vs = gather(self.pool.k_scale(layer)), gather(self.pool.v_scale(layer))
            return dequantize_kv(kp, ks, self.kv_dtype, out_dtype), dequantize_kv(vp, vs, self.kv_dtype, out_dtype)
        return kp.to(out_dtype), vp.to(out_dtype)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor | None = None, n_new: torch.Tensor | int | None = None):
        t = k.shape[1]
        self.write(layer, k, v, positions, n_new)
        end = self.end_lens(n_new, t)
        if self._S is None:
            self._S = int(end.max().item())
        return self.read(layer, self._S, k.dtype, end)
