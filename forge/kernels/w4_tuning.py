"""Launch-config policy of the W4A16 GEMM: per-chip defaults, normalisation, candidate enumeration, M buckets.

Pure Python (no torch / triton import), so every decision in here is unit-tested on CPU-only machines.  The Triton
launcher (:mod:`forge.kernels.triton.gemm_w4_fast`) consumes :func:`resolve_w4_config`; the sweep script
``scripts/bench_w4a16.py`` consumes :func:`w4_candidates` / :func:`refine_candidates` and writes the winners into the
autotune cache, which :func:`forge.kernels.ops_gemm.tuned_config` reads at run time.

Config schema (all ints except ``impl``)::

    impl      "dot"    nibble-split kernel, tl.dot (tensor cores) on 16-bit operands, fp32 accumulate
              "gemv"   nibble-split kernel, fp32 FMA, deferred reduction (M <= 8 only; no tensor cores)
              "legacy" the original per-element-gather kernel of gemm_wq.py (fallback; unchanged)
    tile_m    BLOCK_M (dot: >= 16; gemv: derived from M, the config value is ignored)
    tile_n    BLOCK_N (output channels per program)
    tile_k    BLOCK_K (>= 128 for dot, >= 64 for gemv; a multiple of 8 nibbles x 16 words = KW words per row per tile)
    stages    num_stages        warps  num_warps        split_k  K slices (partial sums reduced by a 2nd tiny kernel)

Honesty: the numbers in the chip profiles below are first-principles estimates (datasheet smem sizes, SM counts,
memory type), NOT measurements; ``scripts/bench_w4a16.py`` replaces them with measured winners.
"""
from __future__ import annotations

from typing import Any

W4_TUNE_TAG = "k2"  # part of the autotune cache key (dtype_cfg): invalidates configs tuned for the legacy kernel
GEMV_MAX_M = 8
DECODE_MAX_M = 16
IMPLS = ("dot", "gemv", "legacy")
MIN_TILE_K = {"dot": 128, "gemv": 64, "legacy": 16}

Config = dict[str, Any]


def cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


# ------------------------------------------------------------------------------------------------------------------
# M buckets (cache keying)
# ------------------------------------------------------------------------------------------------------------------
def m_bucket(m: int) -> int:
    """Cache bucket of ``M`` rows: rounded up to a power of two (1, 2, 4, 8, 16, 32, ...)."""
    return next_pow2(max(1, int(m)))


def neighbor_buckets(m: int) -> list[int]:
    """Buckets to try when the exact bucket of ``m`` was never tuned: itself, then outwards (up first) but never
    across the decode (<= 16) / prefill (>= 32) boundary, because the best kernel differs between the regimes."""
    b = m_bucket(m)
    lo, hi = (1, DECODE_MAX_M) if b <= DECODE_MAX_M else (2 * DECODE_MAX_M, 1 << 16)
    out = [b]
    step = 1
    while True:
        up, dn = b << step, b >> step
        added = False
        if up <= hi:
            out.append(up)
            added = True
        if dn >= lo:
            out.append(dn)
            added = True
        if not added:
            return out
        step += 1


# ------------------------------------------------------------------------------------------------------------------
# chip profiles
# ------------------------------------------------------------------------------------------------------------------
_PROFILES: dict[str, dict[str, Any]] = {
    # smem_block_kib: max dynamic shared memory of ONE block; ctas: resident programs per SM wanted by split-K to hide
    # DRAM latency; stages_*: software-pipeline depth (cp.async needs sm_80+).
    "pre_ampere": dict(smem_block_kib=48, ctas=2, stages_dot=2, stages_gemv=2, mem="hbm_or_gddr"),
    "ampere_dc": dict(smem_block_kib=163, ctas=3, stages_dot=4, stages_gemv=4, mem="hbm"),  # sm_80/87: A100, 108 SMs
    "ampere_consumer": dict(smem_block_kib=99, ctas=2, stages_dot=3, stages_gemv=3, mem="gddr"),  # sm_86
    "ada": dict(smem_block_kib=99, ctas=2, stages_dot=3, stages_gemv=3, mem="gddr"),  # sm_89
    "hopper": dict(smem_block_kib=227, ctas=3, stages_dot=4, stages_gemv=4, mem="hbm"),  # sm_90: H100, 132 SMs
    "blackwell_dc": dict(smem_block_kib=227, ctas=3, stages_dot=4, stages_gemv=4, mem="hbm"),  # sm_100/103
    # sm_120/121 (RTX PRO 6000 / RTX 50xx): ~99 KiB per block, GDDR7, mma.sync (no tcgen05/wgmma in Triton)
    "blackwell_consumer": dict(smem_block_kib=99, ctas=2, stages_dot=3, stages_gemv=3, mem="gddr"),
    "generic": dict(smem_block_kib=48, ctas=2, stages_dot=2, stages_gemv=2, mem="gddr"),
}


def chip_family(sm: int | None) -> str:
    if sm is None:
        return "generic"
    if sm < 80:
        return "pre_ampere"
    if sm in (80, 87):
        return "ampere_dc"
    if sm == 86:
        return "ampere_consumer"
    if sm == 89:
        return "ada"
    if sm == 90:
        return "hopper"
    if sm in (100, 101, 103):
        return "blackwell_dc"
    if sm >= 120:
        return "blackwell_consumer"
    return "hopper"


def chip_profile(sm: int | None) -> dict[str, Any]:
    return dict(_PROFILES[chip_family(sm)])


# ------------------------------------------------------------------------------------------------------------------
# split-K for skinny decode
# ------------------------------------------------------------------------------------------------------------------
def choose_split_k_decode(programs: int, k_tiles: int, num_sms: int, ctas_per_sm: int = 2, max_split: int = 16) -> int:
    """Smallest power-of-two split-K with ``programs * split >= num_sms * ctas_per_sm`` (enough bytes in flight to
    saturate DRAM), never giving a slice fewer than two K tiles.  ``num_sms <= 0`` (unknown) -> 1."""
    if num_sms <= 0:
        return 1
    target = num_sms * max(1, ctas_per_sm)
    split = 1
    while programs * split < target and split < max_split and k_tiles // (split * 2) >= 2:
        split *= 2
    return split


# ------------------------------------------------------------------------------------------------------------------
# defaults / normalisation
# ------------------------------------------------------------------------------------------------------------------
def default_w4_config(m: int, n: int, k: int, *, sm: int | None = None, num_sms: int = 0, group: int = 128) -> Config:
    """Heuristic config for the nibble-split kernel (``impl='dot'`` everywhere; ``gemv`` is an autotune candidate).

    Rationale (first principles, see docs/KERNELS.md "W4A16 decode"): decode is DRAM-bound, so the kernel needs
    (1) coalesced 16-byte weight loads, (2) enough bytes in flight (programs x stages), (3) scale/zero loads hoisted
    out of the per-element dequant.  ``tile_k=128`` keeps one quantisation group per K tile for g128 (vector scale).
    """
    m = max(1, int(m))
    prof = chip_profile(sm)
    stages = int(prof["stages_dot"])
    if m <= DECODE_MAX_M:
        cfg: Config = dict(impl="dot", tile_m=16, tile_n=64, tile_k=128, warps=4, stages=stages)
    elif m <= 32:
        cfg = dict(impl="dot", tile_m=32, tile_n=64, tile_k=128, warps=4, stages=stages)
    elif m <= 64:
        cfg = dict(impl="dot", tile_m=64, tile_n=64, tile_k=128, warps=4, stages=stages)
    elif prof["smem_block_kib"] < 96:  # <= 64 KiB parts (sm_70/75, unknown): a 128-row tile does not fit
        cfg = dict(impl="dot", tile_m=64, tile_n=64, tile_k=128, warps=4, stages=2)
    else:
        big = prof["smem_block_kib"] >= 160
        cfg = dict(impl="dot", tile_m=128, tile_n=128 if big else 64, tile_k=128, warps=8, stages=max(2, stages - 1))
    if m <= 64:
        programs = cdiv(m, cfg["tile_m"]) * cdiv(n, cfg["tile_n"])
        cfg["split_k"] = choose_split_k_decode(programs, cdiv(k, cfg["tile_k"]), num_sms, int(prof["ctas"]))
    else:
        cfg["split_k"] = 1
    return normalise_w4_config(cfg, m)


def _pow2_at_least(v: Any, lo: int) -> int:
    return max(lo, next_pow2(int(v)))


def normalise_w4_config(cfg: Config, m: int) -> Config:
    """Clamp a config to what the kernels support (also repairs configs tuned for the legacy kernel)."""
    c = dict(cfg)
    impl = str(c.get("impl") or "dot")
    if impl not in IMPLS:
        impl = "dot"
    if impl == "gemv" and m > GEMV_MAX_M:
        impl = "dot"  # the FMA kernel keeps all M rows in one program: decode only
    c["impl"] = impl
    if impl == "legacy":
        for key, dflt in (("tile_m", 16), ("tile_n", 64), ("tile_k", 64), ("warps", 4), ("stages", 3), ("split_k", 1)):
            c[key] = int(c.get(key) or dflt)
        return c
    if impl == "gemv":
        c["tile_m"] = next_pow2(m)
    else:
        c["tile_m"] = _pow2_at_least(c.get("tile_m") or 16, 16)
    c["tile_n"] = _pow2_at_least(c.get("tile_n") or 64, 16)
    c["tile_k"] = _pow2_at_least(c.get("tile_k") or 128, MIN_TILE_K[impl])
    c["warps"] = min(8, _pow2_at_least(c.get("warps") or 4, 1))
    c["stages"] = min(6, max(1, int(c.get("stages") or 3)))
    c["split_k"] = max(1, int(c.get("split_k") or 1))
    return c


def resolve_w4_config(m: int, n: int, k: int, config: Config | None = None, *, sm: int | None = None, num_sms: int = 0,
                      group: int = 128) -> Config:
    """Defaults for (m, n, k, chip), overridden by the non-None entries of ``config``, then normalised.

    A config that names no ``impl`` (e.g. a hand-written test config such as ``{'tile_m':16,'tile_n':32,'tile_k':32}``)
    is interpreted for the nibble-split dot kernel with its tile sizes clamped to the supported minimum."""
    cfg = default_w4_config(m, n, k, sm=sm, num_sms=num_sms, group=group)
    if config:
        cfg.update({kk: v for kk, v in config.items() if v is not None})
    return normalise_w4_config(cfg, m)


# ------------------------------------------------------------------------------------------------------------------
# feasibility + candidates
# ------------------------------------------------------------------------------------------------------------------
def w4_feasible(cfg: Config, *, sm: int | None = None, group: int = 128) -> bool:
    """Analytic pruning: shared memory of the pipeline ring, accumulator registers, async-copy availability."""
    impl = cfg.get("impl", "dot")
    if impl == "legacy":
        return True
    tm, tn, tk = int(cfg["tile_m"]), int(cfg["tile_n"]), int(cfg["tile_k"])
    stages, warps = int(cfg.get("stages", 2)), int(cfg.get("warps", 4))
    if tk < MIN_TILE_K[impl] or tn < 16 or (impl == "dot" and tm < 16):
        return False
    prof = chip_profile(sm)
    if sm is not None and sm < 80 and stages > 2:
        return False
    budget = int(prof["smem_block_kib"] * 1024)
    kw = tk // 8
    if impl == "gemv":
        if tm > GEMV_MAX_M:
            return False
        if tm * tn * kw // (32 * warps) > 96:  # [MB, BN, KW] fp32 deferred-reduction accumulator
            return False
        smem = stages * (tn * kw * 4 + tm * tk * 2)
        return smem <= budget
    if tm * tn // (32 * warps) > 128:
        return False
    # packed W ring + X ring + double-buffered staging of ONE dequantised [KW, BN] operand slice (the 8 nibble dots are sequential)
    smem = stages * (tn * tk // 2 + tm * tk * 2) + 2 * tn * kw * 2
    return smem <= budget


def _key(c: Config) -> tuple:
    return tuple(c[k] for k in ("impl", "tile_m", "tile_n", "tile_k", "warps", "stages", "split_k"))


def _dedupe(cands: list[Config], m: int, sm: int | None, group: int) -> list[Config]:
    seen: set[tuple] = set()
    out: list[Config] = []
    for c in cands:
        c = normalise_w4_config(c, m)
        kk = _key(c)
        if kk in seen or not w4_feasible(c, sm=sm, group=group):
            continue
        seen.add(kk)
        out.append(c)
    return out


def w4_candidates(m: int, n: int, k: int, *, sm: int | None = None, num_sms: int = 0, group: int = 128,
                  level: str = "quick") -> list[Config]:
    """Candidate configs for one (M bucket, N, K, chip).  ``quick``: tile-shape sweep at heuristic stages / warps /
    split-K (the bench refines the winner with :func:`refine_candidates`); ``full``: also sweeps warps / stages / split."""
    m = max(1, int(m))
    prof = chip_profile(sm)
    st_dot, st_gemv, ctas = int(prof["stages_dot"]), int(prof["stages_gemv"]), int(prof["ctas"])
    out: list[Config] = [default_w4_config(m, n, k, sm=sm, num_sms=num_sms, group=group)]

    def split_for(tm: int, tn: int, tk: int) -> int:
        return choose_split_k_decode(cdiv(m, tm) * cdiv(n, tn), cdiv(k, tk), num_sms, ctas) if m <= 64 else 1

    full = level == "full"
    warps_sweep = (2, 4, 8) if full else (4,)
    if m <= GEMV_MAX_M:
        for tn in (16, 32, 64):
            for tk in (64, 128, 256):
                for w in warps_sweep:
                    for stg in ((2, 3, 4) if full else (st_gemv,)):
                        tm = next_pow2(m)
                        sp = split_for(tm, tn, tk)
                        for s in ((1, sp, sp * 2) if full else (sp,)):
                            out.append(dict(impl="gemv", tile_m=tm, tile_n=tn, tile_k=tk, warps=w, stages=stg, split_k=s))
    tms = [t for t in (16, 32, 64, 128) if t <= max(16, next_pow2(m))]
    if m > 64:
        tms = [t for t in tms if t >= 64]
    for tm in tms:
        for tn in (32, 64, 128) + ((256,) if m <= 16 else ()):
            for tk in (128, 256):
                for w in ((4, 8) if (full or tm * tn >= 8192) else (4,)):
                    for stg in ((2, 3, 4, 5) if full else (st_dot,)):
                        sp = split_for(tm, tn, tk)
                        for s in ((1, sp, sp * 2) if (full and m <= 64) else (sp,)):
                            out.append(dict(impl="dot", tile_m=tm, tile_n=tn, tile_k=tk, warps=w, stages=stg, split_k=s))
    return _dedupe(out, m, sm, group)


def refine_candidates(best: Config, m: int, n: int, k: int, *, sm: int | None = None, group: int = 128) -> list[Config]:
    """One-parameter-at-a-time neighbours of ``best``: warps, stages, split-K, tile_n and tile_k halved or doubled."""
    b = normalise_w4_config(best, m)
    out: list[Config] = []
    for w in (1, 2, 4, 8):
        out.append({**b, "warps": w})
    for s in (2, 3, 4, 5, 6):
        out.append({**b, "stages": s})
    for s in {max(1, b["split_k"] // 2), b["split_k"] * 2, 1, 2, 4, 8, 16}:
        out.append({**b, "split_k": s})
    for tn in (b["tile_n"] // 2, b["tile_n"] * 2):
        out.append({**b, "tile_n": tn})
    for tk in (b["tile_k"] // 2, b["tile_k"] * 2):
        out.append({**b, "tile_k": tk})
    out = [c for c in out if _key(normalise_w4_config(c, m)) != _key(b)]
    return _dedupe(out, m, sm, group)


# ------------------------------------------------------------------------------------------------------------------
# nearest-entry lookup (untuned N / K fall back to the closest tuned shape of the same chip, op and M bucket)
# ------------------------------------------------------------------------------------------------------------------
def nearest_entry(entries: list[dict[str, Any]], want_fields: dict[str, Any], n: int, k: int) -> dict[str, Any] | None:
    """Among cache ``entries`` whose key fields equal ``want_fields`` except for the bucket's N / K, return the one
    closest in log(N) + log(K) (``None`` if none).  Only the memory-bound decode regime is shape-insensitive enough
    for this to be sensible; callers use it for M buckets <= 16 only."""
    import math

    best: tuple[float, dict[str, Any]] | None = None
    ref_b = dict(want_fields.get("bucket") or {})
    for e in entries:
        f = e.get("fields") or {}
        if any(f.get(key) != want_fields.get(key) for key in want_fields if key != "bucket"):
            continue
        b = f.get("bucket") or {}
        if b.get("M") != ref_b.get("M") or "N" not in b or "K" not in b:
            continue
        dist = abs(math.log(max(1, b["N"]) / max(1, n))) + abs(math.log(max(1, b["K"]) / max(1, k)))
        if best is None or dist < best[0]:
            best = (dist, e)
    return best[1] if best else None
