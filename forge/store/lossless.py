"""Lossless weight codec: exponent-split entropy coding for BF16 / FP16 / FP32 / FP8 tensors (Plan KR-7 / P4).

Floating-point weights have a highly skewed exponent field (a handful of values carry >99% of the mass) while the
sign and mantissa bits are incompressible noise.  The codec therefore *splits* every element into

* an **exponent stream** (``eb`` bits/element) that is entropy coded, and
* a **sign+mantissa stream** (``1+mb`` bits/element) that is stored bit-packed and untouched,

so a BF16 tensor of Gaussian weights shrinks to ~ (H(exp) + 8) / 16 ~= 0.67-0.72 of its size (measured in
``tests/test_store_lossless.py``; the figures in the Plan come from the literature, ours are measured here).

Entropy coders (numpy only, no extension modules)
-------------------------------------------------
``exp_huffman``  canonical, length-limited (<=14) Huffman with **lane interleaving**: the symbols of a block
                 are dealt round-robin to ``lanes`` independent bit streams so that encode *and* decode run as
                 ``n/lanes`` vectorised numpy steps instead of a Python loop per symbol.
``exp_zstd``     zstd on the raw exponent byte stream (fast C baseline, a bit worse than Huffman).
``bytegroup_zstd`` baseline: byte planes (all byte-0s, then all byte-1s ...) + zstd; works for any dtype.
``raw``          stored verbatim.

``method="auto"`` = ``exp_huffman`` for the float formats above, ``bytegroup_zstd`` for everything else; any block
whose encoding is not smaller than the raw bytes is stored raw, so the codec never expands data by more than a
few header bytes.

Container ("FLC1")
------------------
``magic | u32 header_len | header_json | block_0 .. block_k | trailer_json | u32 trailer_len | b"CLF1"``.
The trailer (block lengths + hash of the raw bytes) lets the encoder stream blocks to a sink without buffering;
``decode_iter`` streams blocks back (bounded memory) and checks the hash at the end.

**Every encode is followed by a decode-and-compare per block and a whole-stream hash check** (``verify=True``
default); a mismatch raises :class:`forge.core.errors.IntegrityError` and nothing is returned.

Performance honesty: this is a portable CPU reference/archival codec (tens of MB/s in numpy).  A GPU decoder
(nvCOMP/dietgpu style) for load-time decompression is a separate, not-yet-implemented component.
"""
from __future__ import annotations

import heapq
import io
import json
import struct
import time
from collections.abc import Iterator
from typing import Any, BinaryIO

import numpy as np

from forge.core.errors import ConfigError, IntegrityError, StoreError
from forge.core.hashing import Hasher

try:  # pragma: no cover - environment dependent
    import zstandard as _zstd
except Exception:  # pragma: no cover
    _zstd = None

MAGIC = b"FLC1"
END_MAGIC = b"CLF1"
MAX_CODE_LEN = 14

# canonical torch-style dtype name -> (total bits, exponent bits, mantissa bits)
_FLOAT_FMT: dict[str, tuple[int, int, int]] = {
    "bfloat16": (16, 8, 7),
    "float16": (16, 5, 10),
    "float32": (32, 8, 23),
    "float8_e4m3fn": (8, 4, 3),
    "float8_e5m2": (8, 5, 2),
}
_ITEMSIZE: dict[str, int] = {
    "bfloat16": 2, "float16": 2, "float32": 4, "float64": 8, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "bool": 1,
}
_ALIASES: dict[str, str] = {
    "bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "f16": "float16", "half": "float16",
    "float16": "float16", "fp32": "float32", "f32": "float32", "float": "float32", "float32": "float32",
    "fp64": "float64", "f64": "float64", "double": "float64", "float64": "float64",
    "fp8_e4m3": "float8_e4m3fn", "float8_e4m3fn": "float8_e4m3fn", "e4m3": "float8_e4m3fn", "f8_e4m3": "float8_e4m3fn",
    "float8_e4m3fnuz": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2", "float8_e5m2": "float8_e5m2", "e5m2": "float8_e5m2",
    "f8_e5m2": "float8_e5m2", "float8_e5m2fnuz": "float8_e5m2", "fp8": "float8_e4m3fn",
    "int8": "int8", "i8": "int8", "uint8": "uint8", "u8": "uint8", "int16": "int16", "i16": "int16",
    "uint16": "uint16", "u16": "uint16", "int32": "int32", "i32": "int32", "uint32": "uint32", "u32": "uint32",
    "int64": "int64", "i64": "int64", "uint64": "uint64", "u64": "uint64", "bool": "bool",
}
_UINT = {1: "<u1", 2: "<u2", 4: "<u4"}


def canonical_dtype(name: Any) -> str:
    """Normalise ``'bf16'``, ``torch.bfloat16``, ``'BF16'`` ... to the torch-style canonical name."""
    s = str(name).strip().lower()
    if s.startswith("torch."):
        s = s[6:]
    if s.startswith("numpy."):
        s = s[6:]
    if s not in _ALIASES:
        raise ConfigError(f"unsupported dtype {name!r} for the lossless codec")
    return _ALIASES[s]


def itemsize_of(dtype: str) -> int:
    return _ITEMSIZE[canonical_dtype(dtype)]


def is_float_format(dtype: str) -> bool:
    try:
        return canonical_dtype(dtype) in _FLOAT_FMT
    except ConfigError:
        return False


# --------------------------------------------------------------------------------------------
# zstd wrapper (zlib fallback when the package is missing; recorded in the header)
# --------------------------------------------------------------------------------------------
def _zcodec_default() -> str:
    return "zstd" if _zstd is not None else "zlib"


def _z_compress(data: bytes | memoryview, level: int, codec: str) -> bytes:
    b = data if isinstance(data, bytes) else bytes(data)
    if codec == "zstd":
        if _zstd is None:
            raise StoreError("zstandard package is required to write zstd streams")
        return _zstd.ZstdCompressor(level=level).compress(b)
    import zlib

    return zlib.compress(b, min(max(level, 1), 9))


def _z_decompress(data: bytes | memoryview, codec: str, size_hint: int | None = None) -> bytes:
    b = bytes(data)
    if codec == "zstd":
        if _zstd is None:
            raise StoreError("zstandard package is required to read zstd streams")
        d = _zstd.ZstdDecompressor()
        if size_hint:
            return d.decompress(b, max_output_size=size_hint)
        return d.decompress(b)
    import zlib

    return zlib.decompress(b)


# --------------------------------------------------------------------------------------------
# tensor <-> bytes helpers (shared with checkpoints)
# --------------------------------------------------------------------------------------------
def tensor_to_array(t: Any) -> tuple[np.ndarray, str, tuple[int, ...]]:
    """Raw bytes of a torch tensor / numpy array as a flat ``uint8`` array (shares memory when possible).

    Returns ``(u8, canonical_dtype, shape)``."""
    if hasattr(t, "detach") and hasattr(t, "cpu"):  # torch tensor
        import torch

        x = t.detach().cpu().contiguous()
        shape = tuple(x.shape)
        name = canonical_dtype(str(x.dtype))
        flat = x.reshape(-1)
        u8 = flat.view(torch.uint8).numpy() if flat.numel() else np.zeros(0, dtype=np.uint8)
        return u8, name, shape
    if isinstance(t, np.ndarray):
        x = np.ascontiguousarray(t)
        return x.reshape(-1).view(np.uint8), canonical_dtype(x.dtype.name), tuple(x.shape)
    raise ConfigError(f"cannot take bytes of object of type {type(t).__name__}")


def bytes_to_tensor(buf: bytes | bytearray | memoryview, dtype: str, shape: tuple[int, ...] | list[int]) -> Any:
    """Inverse of :func:`tensor_to_array` for torch (always returns an owning CPU tensor)."""
    import torch

    name = canonical_dtype(dtype)
    tdt = getattr(torch, name)
    shape = tuple(int(s) for s in shape)
    n = 1
    for s in shape:
        n *= s
    if n == 0:
        return torch.empty(shape, dtype=tdt)
    ba = buf if isinstance(buf, bytearray) else bytearray(buf)
    return torch.frombuffer(ba, dtype=torch.uint8).view(tdt).reshape(shape)


# --------------------------------------------------------------------------------------------
# bit packing
# --------------------------------------------------------------------------------------------
def _pack_bits(v: np.ndarray, width: int) -> bytes:
    """Pack the low ``width`` bits of each element (MSB-first within the stream)."""
    if width % 8 == 0:
        planes = np.ascontiguousarray(v.astype("<u4")).view(np.uint8).reshape(-1, 4)[:, : width // 8]
        return planes.tobytes()
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint32)
    bits = ((v.astype(np.uint32)[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes()


def _unpack_bits(a: np.ndarray, n: int, width: int) -> np.ndarray:
    if width % 8 == 0:
        nb = width // 8
        out = np.zeros((n, 4), dtype=np.uint8)
        out[:, :nb] = a[: n * nb].reshape(n, nb)
        return out.reshape(-1).view("<u4").astype(np.uint32)
    bits = np.unpackbits(a)[: n * width].reshape(n, width).astype(np.uint32)
    weights = (np.uint32(1) << np.arange(width - 1, -1, -1, dtype=np.uint32)).astype(np.uint32)
    return bits @ weights


# --------------------------------------------------------------------------------------------
# canonical Huffman, lane interleaved
# --------------------------------------------------------------------------------------------
def _huffman_lengths(counts: np.ndarray, lmax: int = MAX_CODE_LEN) -> np.ndarray:
    nsym = len(counts)
    lengths = np.zeros(nsym, dtype=np.uint8)
    c = counts.astype(np.int64).copy()
    nz = np.flatnonzero(c)
    if nz.size == 0:
        return lengths
    if nz.size == 1:
        lengths[nz[0]] = 1
        return lengths
    while True:
        heap: list[tuple[int, int]] = [(int(c[i]), int(i)) for i in nz]
        heapq.heapify(heap)
        children: dict[int, tuple[int, int]] = {}
        nxt = nsym
        while len(heap) > 1:
            c1, n1 = heapq.heappop(heap)
            c2, n2 = heapq.heappop(heap)
            children[nxt] = (n1, n2)
            heapq.heappush(heap, (c1 + c2, nxt))
            nxt += 1
        root = heap[0][1]
        depth: dict[int, int] = {root: 0}
        stack = [root]
        while stack:
            node = stack.pop()
            if node in children:
                for ch in children[node]:
                    depth[ch] = depth[node] + 1
                    stack.append(ch)
        lens = np.zeros(nsym, dtype=np.int64)
        for i in nz:
            lens[i] = depth[int(i)]
        if lens.max() <= lmax:
            return lens.astype(np.uint8)
        c = np.where(c > 0, np.maximum(c >> 1, 1), 0)  # flatten the tail, retry


def _canonical_codes(lengths: np.ndarray) -> np.ndarray:
    codes = np.zeros(len(lengths), dtype=np.uint32)
    order = sorted((int(l), int(s)) for s, l in enumerate(lengths) if l > 0)
    code = 0
    prev = order[0][0] if order else 0
    for i, (l, s) in enumerate(order):
        if i > 0:
            code = (code + 1) << (l - prev)
        codes[s] = code
        prev = l
    return codes


def _decode_tables(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    codes = _canonical_codes(lengths)
    big_l = int(lengths.max())
    tsym = np.zeros(1 << big_l, dtype=np.uint8)
    tlen = np.zeros(1 << big_l, dtype=np.uint8)
    for s, l in enumerate(lengths):
        if l == 0:
            continue
        lo = int(codes[s]) << (big_l - int(l))
        hi = lo + (1 << (big_l - int(l)))
        tsym[lo:hi] = s
        tlen[lo:hi] = l
    return tsym, tlen, big_l


def _huff_encode(sym: np.ndarray, lengths: np.ndarray, lanes: int) -> tuple[np.ndarray, bytes]:
    """Return ``(lane_bytes[lanes], payload)`` for ``sym`` (uint8)."""
    n = len(sym)
    steps = -(-n // lanes)
    pad = steps * lanes - n
    codes = _canonical_codes(lengths)
    if pad:
        pad_sym = int(np.argmin(np.where(lengths > 0, lengths, 255)))
        sym = np.concatenate([sym, np.full(pad, pad_sym, dtype=np.uint8)])
    s2 = sym.reshape(steps, lanes)
    lens = lengths[s2].astype(np.int64)
    cds = codes[s2]
    lane_bits = lens.sum(axis=0)
    lane_bytes = (lane_bits + 7) >> 3
    base = np.concatenate([[0], np.cumsum(lane_bytes)[:-1]]).astype(np.int64)
    out = np.zeros(int(lane_bytes.sum()) + 4, dtype=np.uint8)
    pos = base * 8
    for s in range(steps):
        ln = lens[s]
        sh = (24 - ln - (pos & 7)).astype(np.uint32)
        w = cds[s] << sh
        b = pos >> 3
        out[b] |= (w >> 16).astype(np.uint8)
        out[b + 1] |= ((w >> 8) & 255).astype(np.uint8)
        out[b + 2] |= (w & 255).astype(np.uint8)
        pos += ln
    return lane_bytes, out[: int(lane_bytes.sum())].tobytes()


def _huff_decode(
    payload: bytes, lane_bytes: np.ndarray, n: int, lanes: int, tsym: np.ndarray, tlen: np.ndarray, big_l: int
) -> np.ndarray:
    steps = -(-n // lanes)
    buf = np.zeros(len(payload) + 4, dtype=np.uint8)
    buf[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    base = np.concatenate([[0], np.cumsum(lane_bytes)[:-1]]).astype(np.int64)
    pos = base * 8
    out = np.empty((steps, lanes), dtype=np.uint8)
    mask = (1 << big_l) - 1
    for s in range(steps):
        b = pos >> 3
        w = (buf[b].astype(np.uint32) << 16) | (buf[b + 1].astype(np.uint32) << 8) | buf[b + 2].astype(np.uint32)
        idx = (w >> (24 - big_l - (pos & 7)).astype(np.uint32)) & mask
        out[s] = tsym[idx]
        pos += tlen[idx]
    return out.reshape(-1)[:n]


# --------------------------------------------------------------------------------------------
# block codecs
# --------------------------------------------------------------------------------------------
def _split_fields(vals: np.ndarray, fmt: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    tot, eb, mb = fmt
    v = vals.astype(np.uint32)
    exp = ((v >> mb) & ((1 << eb) - 1)).astype(np.uint8)
    sign = v >> (tot - 1)
    sm = (sign << mb) | (v & ((1 << mb) - 1))
    return exp, sm


def _join_fields(exp: np.ndarray, sm: np.ndarray, fmt: tuple[int, int, int]) -> np.ndarray:
    tot, eb, mb = fmt
    sign = sm >> mb
    return (sign << (tot - 1)) | (exp.astype(np.uint32) << mb) | (sm & ((1 << mb) - 1))


class _Params:
    __slots__ = ("dtype", "itemsize", "method", "lanes", "zlevel", "zcodec")

    def __init__(self, dtype: str, method: str, lanes: int, zlevel: int, zcodec: str) -> None:
        self.dtype = dtype
        self.itemsize = _ITEMSIZE[dtype]
        self.method = method
        self.lanes = lanes
        self.zlevel = zlevel
        self.zcodec = zcodec


_LANE_DT = {1: "<u1", 2: "<u2", 4: "<u4"}


def _encode_block(raw: memoryview | bytes, p: _Params) -> bytes:
    """Encode one block; returns ``b'\\x00'+raw`` when nothing is gained."""
    nraw = len(raw)
    if nraw == 0:
        return b"\x00"
    body: bytes | None = None
    if p.method in ("exp_huffman", "exp_zstd"):
        fmt = _FLOAT_FMT[p.dtype]
        n = nraw // p.itemsize
        vals = np.frombuffer(raw, dtype=_UINT[p.itemsize])
        exp, sm = _split_fields(vals, fmt)
        smb = _pack_bits(sm, fmt[2] + 1)
        parts = [struct.pack("<I", n)]
        counts = np.bincount(exp, minlength=1 << fmt[1])
        if int((counts > 0).sum()) == 1:
            parts.append(struct.pack("<BB", 0, int(exp[0])))
        elif p.method == "exp_huffman":
            lengths = _huffman_lengths(counts)
            lanes = max(1, min(p.lanes, n))
            lane_bytes, payload = _huff_encode(exp, lengths, lanes)
            lo = int(lane_bytes.min())
            diff = lane_bytes - lo
            dmax = int(diff.max())
            lw = 1 if dmax < 256 else 2 if dmax < 65536 else 4
            nib = lengths.astype(np.uint8)
            if len(nib) % 2:
                nib = np.concatenate([nib, np.zeros(1, dtype=np.uint8)]).astype(np.uint8)
            packed_len = ((nib[0::2] << 4) | nib[1::2]).astype(np.uint8).tobytes()
            parts.append(struct.pack("<BHBI", 1, lanes, lw, lo))
            parts.append(packed_len)
            parts.append(diff.astype(_LANE_DT[lw]).tobytes())
            parts.append(struct.pack("<I", len(payload)))
            parts.append(payload)
        else:
            z = _z_compress(exp.tobytes(), p.zlevel, p.zcodec)
            parts.append(struct.pack("<BI", 2, len(z)))
            parts.append(z)
        parts.append(struct.pack("<I", len(smb)))
        parts.append(smb)
        body = b"".join(parts)
    elif p.method == "bytegroup_zstd":
        a = np.frombuffer(raw, dtype=np.uint8)
        if p.itemsize > 1 and nraw % p.itemsize == 0:
            planes = a.reshape(-1, p.itemsize).T.copy().tobytes()
        else:
            planes = a.tobytes()
        z = _z_compress(planes, p.zlevel, p.zcodec)
        body = struct.pack("<I", len(z)) + z
    if body is None or len(body) >= nraw:
        return b"\x00" + bytes(raw)
    return b"\x01" + body


def _decode_block(blk: bytes | memoryview, nraw: int, p: _Params) -> bytes:
    blk = memoryview(blk)
    if nraw == 0:
        return b""
    if blk[0] == 0:
        return bytes(blk[1 : 1 + nraw])
    off = 1
    if p.method in ("exp_huffman", "exp_zstd"):
        fmt = _FLOAT_FMT[p.dtype]
        (n,) = struct.unpack_from("<I", blk, off)
        off += 4
        emode = blk[off]
        if emode == 0:
            exp = np.full(n, blk[off + 1], dtype=np.uint8)
            off += 2
        elif emode == 1:
            _, lanes, lw, lo = struct.unpack_from("<BHBI", blk, off)
            off += struct.calcsize("<BHBI")
            nsym = 1 << fmt[1]
            nb = (nsym + 1) // 2
            packed = np.frombuffer(blk[off : off + nb], dtype=np.uint8)
            off += nb
            lengths = np.empty(nb * 2, dtype=np.uint8)
            lengths[0::2] = packed >> 4
            lengths[1::2] = packed & 15
            lengths = lengths[:nsym]
            diff = np.frombuffer(blk[off : off + lanes * lw], dtype=_LANE_DT[lw]).astype(np.int64)
            off += lanes * lw
            (plen,) = struct.unpack_from("<I", blk, off)
            off += 4
            payload = bytes(blk[off : off + plen])
            off += plen
            tsym, tlen, big_l = _decode_tables(lengths)
            exp = _huff_decode(payload, diff + lo, n, lanes, tsym, tlen, big_l)
        elif emode == 2:
            (zl,) = struct.unpack_from("<I", blk, off + 1)
            off += 5
            exp = np.frombuffer(_z_decompress(blk[off : off + zl], p.zcodec, n), dtype=np.uint8)
            off += zl
        else:
            raise IntegrityError(f"unknown exponent stream mode {emode}")
        (sl,) = struct.unpack_from("<I", blk, off)
        off += 4
        sm = _unpack_bits(np.frombuffer(blk[off : off + sl], dtype=np.uint8), n, fmt[2] + 1)
        v = _join_fields(exp, sm, fmt)
        return v.astype(_UINT[p.itemsize]).tobytes()
    if p.method == "bytegroup_zstd":
        (zl,) = struct.unpack_from("<I", blk, off)
        off += 4
        planes = np.frombuffer(_z_decompress(blk[off : off + zl], p.zcodec, nraw), dtype=np.uint8)
        if p.itemsize > 1 and nraw % p.itemsize == 0:
            return planes.reshape(p.itemsize, -1).T.copy().tobytes()
        return planes.tobytes()
    raise IntegrityError(f"unknown method {p.method!r}")


# --------------------------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------------------------
METHODS = ("auto", "exp_huffman", "exp_zstd", "bytegroup_zstd", "raw")


def _resolve_method(method: str, dtype: str) -> str:
    if method not in METHODS:
        raise ConfigError(f"unknown lossless method {method!r}; choose one of {METHODS}")
    if method == "auto":
        return "exp_huffman" if dtype in _FLOAT_FMT else "bytegroup_zstd"
    if method.startswith("exp_") and dtype not in _FLOAT_FMT:
        raise ConfigError(f"method {method} needs a float format (bf16/fp16/fp32/fp8), got {dtype}")
    return method


def _as_memoryview(data: Any) -> memoryview:
    if isinstance(data, np.ndarray):
        return memoryview(np.ascontiguousarray(data).reshape(-1).view(np.uint8))
    return memoryview(data).cast("B")


def encode_stream(
    data: bytes | bytearray | memoryview | np.ndarray,
    dtype: str,
    sink: BinaryIO,
    *,
    shape: tuple[int, ...] | list[int] | None = None,
    method: str = "auto",
    block_elems: int = 1 << 20,
    lanes: int = 2048,
    zstd_level: int = 3,
    verify: bool = True,
) -> dict[str, Any]:
    """Encode raw little-endian tensor bytes into ``sink`` block by block. Returns a stats dict.

    ``data`` may be any buffer (including a ``np.memmap`` of a huge file: only one block is touched at a time)."""
    dtype = canonical_dtype(dtype)
    isz = _ITEMSIZE[dtype]
    mv = _as_memoryview(data)
    nbytes = len(mv)
    if nbytes % isz:
        raise ConfigError(f"{nbytes} bytes is not a multiple of itemsize {isz} for {dtype}")
    n_el = nbytes // isz
    shape = tuple(int(s) for s in shape) if shape is not None else (n_el,)
    prod = 1
    for s in shape:
        prod *= s
    if prod != n_el:
        raise ConfigError(f"shape {shape} does not match {n_el} elements")
    requested = method
    method = _resolve_method(method, dtype)
    zcodec = _zcodec_default()
    p = _Params(dtype, method, int(lanes), int(zstd_level), zcodec)
    if requested == "auto" and dtype in _FLOAT_FMT and n_el:
        # per-tensor choice: compare exp_huffman with the byte-grouped zstd baseline on the first block
        # (for 8-bit / 16-bit non-bf16 formats the mantissa MSBs carry a little skew that zstd picks up).
        first = mv[: min(int(block_elems), n_el) * isz]
        best_len = None
        for cand in ("exp_huffman", "bytegroup_zstd"):
            pc = _Params(dtype, cand, int(lanes), int(zstd_level), zcodec)
            ln = len(_encode_block(first, pc))
            if best_len is None or ln < best_len:
                best_len, method, p = ln, cand, pc
    header = {
        "v": 1, "dtype": dtype, "shape": list(shape), "nbytes": nbytes, "itemsize": isz, "method": method,
        "block_elems": int(block_elems), "lanes": int(lanes), "zcodec": zcodec,
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    sink.write(MAGIC + struct.pack("<I", len(hb)) + hb)
    hasher = Hasher()
    dec_hasher = Hasher()
    block_lens: list[int] = []
    total = 8 + len(hb)
    be = max(1, int(block_elems))
    for start in range(0, n_el, be):
        raw = mv[start * isz : min(start + be, n_el) * isz]
        enc = b"\x00" + bytes(raw) if method == "raw" else _encode_block(raw, p)
        if verify:
            dec = bytes(raw) if enc[0] == 0 else _decode_block(enc, len(raw), p)
            if dec != raw:
                raise IntegrityError(f"lossless self-check failed in block at element {start} ({dtype}, {method})")
            dec_hasher.update(dec)
        hasher.update(raw)
        sink.write(enc)
        block_lens.append(len(enc))
        total += len(enc)
    digest = hasher.hexdigest()
    if verify and dec_hasher.hexdigest() != digest:
        raise IntegrityError("lossless self-check failed: decoded stream hash differs")
    tr = json.dumps({"blocks": block_lens, "hash": digest}, separators=(",", ":")).encode()
    sink.write(tr + struct.pack("<I", len(tr)) + END_MAGIC)
    total += len(tr) + 8
    return {
        "raw_bytes": nbytes, "encoded_bytes": total, "ratio": (total / nbytes) if nbytes else 1.0,
        "method": method, "dtype": dtype, "hash": digest, "blocks": len(block_lens),
    }


def encode_bytes(
    data: bytes | bytearray | memoryview | np.ndarray,
    dtype: str,
    shape: tuple[int, ...] | list[int] | None = None,
    **kw: Any,
) -> bytes:
    """Encode raw tensor bytes (``dtype`` required) into a container blob."""
    bio = io.BytesIO()
    encode_stream(data, dtype, bio, shape=shape, **kw)
    return bio.getvalue()


def encode_tensor(t: Any, dtype: str | None = None, shape: tuple[int, ...] | list[int] | None = None, **kw: Any) -> bytes:
    """Encode a ``torch.Tensor`` / ``np.ndarray`` (or ``bytes`` + ``dtype``) bit-exactly. See module docstring.

    Keyword arguments: ``method`` (``auto|exp_huffman|exp_zstd|bytegroup_zstd|raw``), ``block_elems``, ``lanes``,
    ``zstd_level``, ``verify`` (default True)."""
    if isinstance(t, (bytes, bytearray, memoryview)):
        if dtype is None:
            raise ConfigError("dtype is required when encoding raw bytes")
        return encode_bytes(t, dtype, shape, **kw)
    u8, name, shp = tensor_to_array(t)
    return encode_bytes(u8, dtype or name, shape if shape is not None else shp, **kw)


class _Source:
    def __init__(self, source: bytes | bytearray | memoryview | BinaryIO) -> None:
        self.f: BinaryIO = io.BytesIO(source) if isinstance(source, (bytes, bytearray, memoryview)) else source  # type: ignore[assignment]
        self.f.seek(0)
        head = self.f.read(8)
        if len(head) < 8 or head[:4] != MAGIC:
            raise IntegrityError("not a FLC1 lossless container")
        (hl,) = struct.unpack("<I", head[4:])
        self.header = json.loads(self.f.read(hl))
        self.data_start = 8 + hl
        self.f.seek(-8, 2)
        tail = self.f.read(8)
        if tail[4:] != END_MAGIC:
            raise IntegrityError("lossless container truncated (bad end magic)")
        (tl,) = struct.unpack("<I", tail[:4])
        self.f.seek(-8 - tl, 2)
        self.trailer = json.loads(self.f.read(tl))


def read_info(source: bytes | bytearray | memoryview | BinaryIO) -> dict[str, Any]:
    """Header + trailer summary without decoding any block."""
    s = _Source(source)
    enc = sum(s.trailer["blocks"]) + s.data_start
    return {**s.header, "hash": s.trailer["hash"], "blocks": len(s.trailer["blocks"]), "encoded_bytes": enc}


def decode_iter(source: bytes | bytearray | memoryview | BinaryIO, *, verify: bool = True) -> Iterator[bytes]:
    """Stream the raw bytes back block by block (bounded memory). Verifies the stream hash at the end."""
    s = _Source(source)
    h = s.header
    p = _Params(h["dtype"], h["method"], h["lanes"], 3, h.get("zcodec", "zstd"))
    isz = h["itemsize"]
    n_el = h["nbytes"] // isz
    be = h["block_elems"]
    hasher = Hasher()
    s.f.seek(s.data_start)
    remaining = n_el
    for blen in s.trailer["blocks"]:
        take = min(be, remaining)
        remaining -= take
        blk = s.f.read(blen)
        if len(blk) != blen:
            raise IntegrityError("lossless container truncated")
        raw = bytes(blk[1 : 1 + take * isz]) if (h["method"] == "raw" or blk[0] == 0) else _decode_block(blk, take * isz, p)
        if verify:
            hasher.update(raw)
        yield raw
    if remaining:
        raise IntegrityError("lossless container has fewer blocks than elements")
    if verify and hasher.hexdigest() != s.trailer["hash"]:
        raise IntegrityError("lossless decode hash mismatch")


def decode_bytes(source: bytes | bytearray | memoryview | BinaryIO, *, verify: bool = True) -> tuple[bytearray, str, tuple[int, ...]]:
    """Decode to ``(raw_bytes, dtype, shape)``."""
    s = _Source(source)
    out = bytearray(s.header["nbytes"])
    pos = 0
    for blk in decode_iter(s.f, verify=verify):
        out[pos : pos + len(blk)] = blk
        pos += len(blk)
    return out, s.header["dtype"], tuple(s.header["shape"])


def decode_tensor(source: bytes | bytearray | memoryview | BinaryIO, *, verify: bool = True) -> Any:
    """Decode to a ``torch.Tensor`` (bit-exact)."""
    raw, dtype, shape = decode_bytes(source, verify=verify)
    return bytes_to_tensor(raw, dtype, shape)


def decode_to_file(source: bytes | BinaryIO, dest: BinaryIO, *, verify: bool = True) -> int:
    n = 0
    for blk in decode_iter(source, verify=verify):
        dest.write(blk)
        n += len(blk)
    return n


# --------------------------------------------------------------------------------------------
# analysis / verification helpers
# --------------------------------------------------------------------------------------------
def _raw_and_dtype(t: Any, dtype: str | None) -> tuple[np.ndarray, str, tuple[int, ...] | None]:
    if isinstance(t, (bytes, bytearray, memoryview)):
        if dtype is None:
            raise ConfigError("dtype is required for raw bytes")
        return np.frombuffer(t, dtype=np.uint8), canonical_dtype(dtype), None
    u8, name, shp = tensor_to_array(t)
    return u8, name, shp


def exponent_entropy_bits(t: Any, dtype: str | None = None) -> float:
    """Shannon entropy (bits/element) of the exponent field of a float tensor."""
    u8, name, _ = _raw_and_dtype(t, dtype)
    fmt = _FLOAT_FMT[name]
    exp, _sm = _split_fields(np.frombuffer(u8, dtype=_UINT[_ITEMSIZE[name]]), fmt)
    c = np.bincount(exp, minlength=1 << fmt[1]).astype(np.float64)
    pr = c[c > 0] / c.sum()
    return float(-(pr * np.log2(pr)).sum())


def verify_roundtrip(t: Any, dtype: str | None = None, shape: Any = None, **kw: Any) -> dict[str, Any]:
    """Encode, decode and compare. Returns a report; raises IntegrityError on any mismatch."""
    raw, name, shp = _raw_and_dtype(t, dtype)
    t0 = time.perf_counter()
    blob = encode_bytes(raw, name, shape if shape is not None else shp, **kw)
    t1 = time.perf_counter()
    out, d2, _ = decode_bytes(blob)
    t2 = time.perf_counter()
    if bytes(out) != raw.tobytes() or d2 != name:
        raise IntegrityError("verify_roundtrip: decoded bytes differ from the input")
    info = read_info(blob)
    return {
        "ok": True, "dtype": name, "method": info["method"], "raw_bytes": int(raw.size), "encoded_bytes": len(blob),
        "ratio": len(blob) / max(1, raw.size), "encode_s": t1 - t0, "decode_s": t2 - t1,
        "encode_MBps": raw.size / 1e6 / max(1e-9, t1 - t0), "decode_MBps": raw.size / 1e6 / max(1e-9, t2 - t1),
    }


def compare_methods(t: Any, dtype: str | None = None, **kw: Any) -> dict[str, Any]:
    """Measured ratios of every applicable method on one tensor (for reports / the UI)."""
    raw, name, _ = _raw_and_dtype(t, dtype)
    out: dict[str, Any] = {"dtype": name, "raw_bytes": int(raw.size)}
    for m in ("exp_huffman", "exp_zstd", "bytegroup_zstd"):
        if m.startswith("exp_") and name not in _FLOAT_FMT:
            continue
        r = verify_roundtrip(raw.tobytes(), name, method=m, **kw)
        out[m] = {"ratio": r["ratio"], "encode_MBps": r["encode_MBps"], "decode_MBps": r["decode_MBps"]}
    if name in _FLOAT_FMT:
        fm = _FLOAT_FMT[name]
        h = exponent_entropy_bits(raw.tobytes(), name)
        out["exponent_entropy_bits"] = h
        out["ideal_ratio"] = (h + fm[2] + 1) / fm[0]
    return out


def train_dictionary(samples: list[bytes], dict_size: int = 112_640) -> bytes | None:
    """Optional trained zstd dictionary (used by checkpoint delta compression). ``None`` if unavailable/too little data."""
    if _zstd is None:
        return None
    samples = [s for s in samples if s]
    if len(samples) < 8:
        return None
    try:
        return _zstd.train_dictionary(dict_size, samples).as_bytes()
    except Exception:
        return None


def verify_file(path: str) -> dict[str, Any]:
    """CLI helper: verify a ``.flc`` container, or measure/round-trip every float tensor of a ``.safetensors`` file."""
    from pathlib import Path

    p = Path(path)
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic == MAGIC:
        n = 0
        with open(p, "rb") as f:
            for blk in decode_iter(f, verify=True):
                n += len(blk)
        with open(p, "rb") as f:
            info = read_info(f)
        return {"path": str(p), "kind": "flc", "ok": True, "raw_bytes": n, **info}
    from forge.store.tensordir import parse_safetensors_header

    hdr = parse_safetensors_header(p)
    rows = []
    raw_total = enc_total = 0
    with open(p, "rb") as f:
        for t in hdr["tensors"]:
            if t["end"] == t["begin"]:
                continue
            try:
                name = canonical_dtype(t["dtype"])
            except ConfigError:
                continue
            f.seek(hdr["data_start"] + t["begin"])
            buf = f.read(t["end"] - t["begin"])
            r = verify_roundtrip(buf, name)
            rows.append({"name": t["name"], "dtype": name, "ratio": r["ratio"]})
            raw_total += r["raw_bytes"]
            enc_total += r["encoded_bytes"]
    return {
        "path": str(p), "kind": "safetensors", "ok": True, "tensors": len(rows), "raw_bytes": raw_total,
        "encoded_bytes": enc_total, "ratio": enc_total / max(1, raw_total), "per_tensor": rows,
    }
