"""Launch-config policy of the W4A16 GEMM: per-chip defaults, normalisation, candidate enumeration, M buckets.

Pure Python (no torch / triton import), so every decision in here is unit-tested on CPU-only machines.  The Triton
launcher (:mod:`forge.kernels.triton.gemm_w4_fast`) consumes :func:`resolve_w4_config`; the sweep script
``scripts/bench_w4a16.py`` consumes :func:`w4_candidates` / :func:`refine_candidates` and writes the winners into the
autotune cache, which :func:`forge.kernels.ops_gemm.tuned_config` reads at run time.

Config schema (all ints except ``impl``)::

    impl      "dot"    nibble-split kernel, tl.dot (tensor cores) on 16-bit operands, fp32 accumulate
              "gemv"   nibble-split kernel, fp32 FMA, deferred reduction (M <= 8 only; no tensor cores)
              "legacy" the original per-element-gather kernel of gemm_wq.py (fallback; unchanged)
              "dequant_matmul"  large M: dequantise the weight to bf16/fp16 in row chunks with a memory-bound Triton
                       kernel (gemm_w4_dequant.py), then one cuBLAS GEMM per chunk.  tile_n = weight rows per dequant program,
                       tile_k = K columns per program, warps; tile_m / stages / split_k are fixed placeholders.  The
                       temporary bf16 buffer is bounded by :func:`dequant_budget_bytes` (default 1 GiB).
    tile_m    BLOCK_M (dot: >= 16; gemv: derived from M, the config value is ignored)
    tile_n    BLOCK_N (output channels per program)
    tile_k    BLOCK_K (>= 128 for dot, >= 64 for gemv; a multiple of 8 nibbles x 16 words = KW words per row per tile)
    stages    num_stages        warps  num_warps        split_k  K slices (partial sums reduced by a 2nd tiny kernel)

Default routing by M (``default_w4_config``): ``gemv`` for M <= 8, ``dot`` for 9 <= M < ``dequant_min_m`` (64 unless a chip's
measured dot speed says dequantise-then-cuBLAS already wins at 32, see :func:`_derived_dequant_min_m`), and ``dequant_matmul``
from ``dequant_min_m`` up -- the tensor-core ``dot`` kernel is gather-bound and measured 2-3x slower than bf16 cuBLAS at M = 64
on sm_120 (x0.74-0.80 of bf16 speed at M = 32), where dequantise-then-cuBLAS (about 2.25x the bf16 time: it writes and re-reads
the bf16 weight) is the better option.  ``dequant_min_m`` is per chip (``_PROFILES``), overridable with
``FORGE_W4_DEQUANT_MIN_M`` (<= 0 disables the routing), and the sweep moves it per (N, K, M bucket) by caching whichever
candidate wins.

Honesty: the numbers in the chip profiles below are first-principles estimates (datasheet smem sizes, SM counts,
memory type), NOT measurements -- except the ``blackwell_consumer`` entries marked "measured" (one user sweep on an sm_120
card: dot-vs-bf16 speeds and the winning tiles of M = 1..32 on 4096-5120 shaped linears); ``scripts/bench_w4a16.py`` replaces
all of them with winners measured on the actual host.
"""
from __future__ import annotations

import os
from typing import Any

W4_TUNE_TAG = "k4"  # part of the autotune cache key (dtype_cfg): k4 = k3 (dequant_matmul / legacy candidates) + act-order / symmetry in the key (older entries ignored)
GEMV_MAX_M = 8
DECODE_MAX_M = 16
DEQUANT_MIN_M = 64  # default M from which dequant_matmul is the heuristic choice (32 only where the cost model beats the measured dot)
DEQUANT_TIME_VS_BF16 = 2.25  # cost model of dequant_matmul at large N*K: (0.5 B read + 2 B write + 2 B re-read) / 2 B of a bf16 GEMM
DEQUANT_BUDGET_MB = 1024  # default cap of the temporary dequantised-weight buffer
IMPLS = ("dot", "gemv", "legacy", "dequant_matmul")
MIN_TILE_K = {"dot": 128, "gemv": 64, "legacy": 16, "dequant_matmul": 64}

Config = dict[str, Any]


def w4_tune_tag(act_order: bool = False, sym: bool = False) -> str:
    """``extra`` of the W4A16 ``dtype_cfg`` cache key: tag + act-order (``g_idx`` present) + symmetry (no zero-points) -- the
    tuned winner of those variants differs (act-order gathers x, symmetric skips the zero loads), so they must not share entries."""
    return f"{W4_TUNE_TAG}|ao={int(bool(act_order))}|sym={int(bool(sym))}"


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
    # measured on an sm_120 card (4096-5120 shaped linears): dot speed relative to bf16 cuBLAS = x0.74-0.80 at M=32 and
    # x0.44 / 0.33 / 0.43 at M=64; legacy kernel slower than dot at every M >= 16; winning tiles -> _measured_blackwell_config
    "blackwell_consumer": dict(smem_block_kib=99, ctas=2, stages_dot=3, stages_gemv=3, mem="gddr", measured=True,
                               dot_vs_bf16={32: 0.77, 64: 0.40}),
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
    prof = dict(_PROFILES[chip_family(sm)])
    prof.setdefault("dequant_min_m", _derived_dequant_min_m(prof))  # per-chip override point (set it once a chip is measured)
    return prof


def _derived_dequant_min_m(prof: dict[str, Any]) -> int:
    """32 only if the dequant cost model is below the measured dot time at M=32 (``dot_vs_bf16`` = dot speed relative to bf16),
    else :data:`DEQUANT_MIN_M` (64).  Chips without measurements always get 64."""
    sp = (prof.get("dot_vs_bf16") or {}).get(32)
    return 32 if (sp and DEQUANT_TIME_VS_BF16 < 1.0 / sp) else DEQUANT_MIN_M


def dequant_min_m(sm: int | None = None) -> int:
    """First M at which the default heuristic routes to ``dequant_matmul`` (``FORGE_W4_DEQUANT_MIN_M`` overrides the chip
    profile; a value <= 0 disables the routing -> returns a huge M)."""
    raw = os.environ.get("FORGE_W4_DEQUANT_MIN_M", "").strip()
    if raw:
        try:
            v = int(raw)
            return v if v > 0 else 1 << 30
        except ValueError:
            pass
    return int(chip_profile(sm)["dequant_min_m"])


def dequant_budget_bytes(config: dict[str, Any] | None = None) -> int:
    """Cap in bytes of the temporary dequantised-weight buffer of one ``dequant_matmul`` chunk: ``config['chunk_mb']`` >
    ``FORGE_W4_DEQUANT_MB`` > 1024 MiB."""
    if config and config.get("chunk_mb"):
        return max(1, int(config["chunk_mb"])) << 20
    raw = os.environ.get("FORGE_W4_DEQUANT_MB", "").strip()
    if raw:
        try:
            return max(1, int(float(raw))) << 20
        except ValueError:
            pass
    return DEQUANT_BUDGET_MB << 20


def plan_dequant_chunks(n: int, k: int, itemsize: int, budget_bytes: int, align: int = 64, m: int = 0) -> list[tuple[int, int]]:
    """Row ranges ``[(r0, r1), ...]`` covering ``0..n`` so that each chunk's working set -- the dequantised ``[rows, k]`` buffer
    plus the chunk's ``[m, rows]`` GEMM output (``m`` > 0) -- takes at most ``budget_bytes`` (at least ``align`` rows per chunk,
    rows a multiple of ``align`` except the last -- the packed zero-point words hold 8 channels each, so chunk starts must be
    multiples of 8)."""
    n, k = int(n), int(k)
    if n <= 0:
        return []
    per_row = max(1, (k + max(0, int(m))) * int(itemsize))
    rows = max(align, (int(budget_bytes) // per_row) // align * align)
    if rows >= n:
        return [(0, n)]
    return [(r, min(n, r + rows)) for r in range(0, n, rows)]


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
    """Heuristic config: ``gemv`` (M <= 8), ``dot`` (9 <= M < ``dequant_min_m``), ``dequant_matmul`` (larger M).

    Rationale (first principles, see docs/KERNELS.md "W4A16 decode"): decode is DRAM-bound, so the kernel needs
    (1) coalesced 16-byte weight loads, (2) enough bytes in flight (programs x stages), (3) scale/zero loads hoisted
    out of the per-element dequant.  ``tile_k=128`` keeps one quantisation group per K tile for g128 (vector scale).
    """
    m = max(1, int(m))
    prof = chip_profile(sm)
    if m >= dequant_min_m(sm):
        return normalise_w4_config(dict(impl="dequant_matmul"), m)
    measured = _measured_blackwell_config(m, n, k, prof, num_sms)
    if measured is not None:
        return normalise_w4_config(measured, m)
    if m <= GEMV_MAX_M:
        return gemv_default_config(m, n, k, sm=sm, num_sms=num_sms)
    return dot_default_config(m, n, k, sm=sm, num_sms=num_sms)


def gemv_default_config(m: int, n: int, k: int, *, sm: int | None = None, num_sms: int = 0) -> Config:
    """Heuristic config of the FMA ``gemv`` kernel (M <= 8; larger M is normalised to ``dot``)."""
    m = max(1, int(m))
    prof = chip_profile(sm)
    if m > GEMV_MAX_M:
        return dot_default_config(m, n, k, sm=sm, num_sms=num_sms)
    measured = _measured_blackwell_config(m, n, k, prof, num_sms)
    if measured is not None and measured["impl"] == "gemv":
        return normalise_w4_config(measured, m)
    tm = next_pow2(m)
    cfg: Config = dict(impl="gemv", tile_m=tm, tile_n=32 if tm <= 4 else 16, tile_k=128, warps=4, stages=int(prof["stages_gemv"]))
    programs = cdiv(n, cfg["tile_n"])
    cfg["split_k"] = choose_split_k_decode(programs, cdiv(k, cfg["tile_k"]), num_sms, int(prof["ctas"]))
    return normalise_w4_config(cfg, m)


def _clamped_split(programs: int, k_tiles: int, num_sms: int, ctas: int, lo: int, hi: int) -> int:
    """Heuristic split-K clamped to the measured winning range ``[lo, hi]`` (never below two K tiles per slice)."""
    cap = hi
    while cap > 1 and k_tiles // cap < 2:
        cap //= 2
    return max(1, min(max(choose_split_k_decode(programs, k_tiles, num_sms, ctas), lo), cap))


def _measured_blackwell_config(m: int, n: int, k: int, prof: dict[str, Any], num_sms: int) -> Config | None:
    """Winning tiles of one user sweep on sm_120 (M=1: gemv t1x64x256 w4 s5 sk1-2 for big N, t1x32x128 w4 s3 sk8 for N=5120; M=4 dot
    t16x128x128 w4 s2-3 sk4-16; M=8-16 dot t16x256x128 w4/w8 s2-3 sk8-16; M=32 dot t32x256x128 s3 sk4-16).  ``None`` outside
    the measured range / on unmeasured chips (-> the first-principles heuristics)."""
    if not prof.get("measured") or m > 32:
        return None
    ctas = int(prof["ctas"])
    kt = lambda tk: cdiv(k, tk)  # noqa: E731
    if m <= 1:
        if n >= 8192:
            sp = _clamped_split(cdiv(n, 64), kt(256), num_sms, ctas, 1, 2)
            return dict(impl="gemv", tile_m=1, tile_n=64, tile_k=256, warps=4, stages=5, split_k=sp)
        return dict(impl="gemv", tile_m=1, tile_n=32, tile_k=128, warps=4, stages=3, split_k=_clamped_split(cdiv(n, 32), kt(128), num_sms, ctas, 8, 8))
    if m <= 4:
        sp = _clamped_split(cdiv(n, 128), kt(128), num_sms, ctas, 4, 16)
        return dict(impl="dot", tile_m=16, tile_n=128, tile_k=128, warps=4, stages=3, split_k=sp)
    if m <= 16:
        sp = _clamped_split(cdiv(n, 256), kt(128), num_sms, ctas, 8, 16)
        return dict(impl="dot", tile_m=16, tile_n=256, tile_k=128, warps=4, stages=3, split_k=sp)
    sp = _clamped_split(cdiv(m, 32) * cdiv(n, 256), kt(128), num_sms, ctas, 4, 16)
    return dict(impl="dot", tile_m=32, tile_n=256, tile_k=128, warps=4, stages=3, split_k=sp)


def dot_default_config(m: int, n: int, k: int, *, sm: int | None = None, num_sms: int = 0) -> Config:
    """Heuristic config of the tensor-core ``dot`` kernel (also what a hand-written config without an ``impl`` is merged into)."""
    m = max(1, int(m))
    prof = chip_profile(sm)
    stages = int(prof["stages_dot"])
    measured = _measured_blackwell_config(m, n, k, prof, num_sms)
    if measured is not None and measured["impl"] == "dot":
        return normalise_w4_config(measured, m)
    if m <= DECODE_MAX_M:
        cfg: Config = dict(impl="dot", tile_m=16, tile_n=128, tile_k=128, warps=4, stages=stages)
    elif m <= 64:  # 17..dequant_min_m-1 (or up to 64 when the dequant routing is disabled): dot with a wide N tile
        cfg = dict(impl="dot", tile_m=32 if m <= 32 else 64, tile_n=128, tile_k=128, warps=4, stages=stages)
    elif prof["smem_block_kib"] < 96:  # only reachable with the dequant routing disabled; <= 64 KiB parts: a 128-row tile does not fit
        cfg = dict(impl="dot", tile_m=64, tile_n=64, tile_k=128, warps=4, stages=2)
    else:
        big = prof["smem_block_kib"] >= 160
        cfg = dict(impl="dot", tile_m=128, tile_n=128 if big else 64, tile_k=128, warps=8, stages=max(2, stages - 1))
    while cfg["tile_n"] > 16 and not w4_feasible({**cfg, "split_k": 1}, sm=sm):  # small-smem parts (reachable now that dot covers M < 64)
        cfg["tile_n"] //= 2
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
    if impl == "dequant_matmul":
        c["tile_m"], c["stages"], c["split_k"] = 16, 1, 1  # fixed placeholders: keeps the cache schema uniform
        c["tile_n"] = min(128, _pow2_at_least(c.get("tile_n") or 32, 8))
        c["tile_k"] = min(512, _pow2_at_least(c.get("tile_k") or 128, MIN_TILE_K[impl]))
        c["warps"] = min(8, _pow2_at_least(c.get("warps") or 4, 1))
        return c
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
    is interpreted for the nibble-split dot kernel with its tile sizes clamped to the supported minimum (never routed to
    gemv / dequant_matmul behind the caller's back)."""
    impl = (config or {}).get("impl")
    if not config:
        cfg = default_w4_config(m, n, k, sm=sm, num_sms=num_sms, group=group)
    else:  # seed from the DEFAULT OF THE NAMED impl, not from the heuristic's (an M=256 dot must not inherit dequant's 16x32 tile)
        cfg = default_for_impl(impl, m, n, k, sm=sm, num_sms=num_sms)
    if config:
        cfg.update({kk: v for kk, v in config.items() if v is not None})
    return normalise_w4_config(cfg, m)


def default_for_impl(impl: str | None, m: int, n: int, k: int, *, sm: int | None = None, num_sms: int = 0) -> Config:
    """The default config of one specific ``impl`` (``None`` = the dot kernel, for hand-written configs without an ``impl``)."""
    if impl == "gemv":
        return gemv_default_config(m, n, k, sm=sm, num_sms=num_sms)
    if impl == "dequant_matmul":
        return normalise_w4_config(dict(impl="dequant_matmul"), m)
    if impl == "legacy":
        return normalise_w4_config(dict(impl="legacy"), m)
    return dot_default_config(m, n, k, sm=sm, num_sms=num_sms)


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
    if impl == "dequant_matmul":  # no pipeline ring; bound the [tn, tk/8, 8] fp32 working set per thread
        return tk >= MIN_TILE_K[impl] and tn >= 8 and tn * tk // (32 * warps) <= 64
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
        for tn in (32, 64, 128) + ((256,) if m <= 32 else ()):
            for tk in (128, 256):
                for w in ((4, 8) if (full or tm * tn >= 8192) else (4,)):
                    for stg in ((2, 3, 4, 5) if full else (st_dot,)):
                        sp = split_for(tm, tn, tk)
                        for s in ((1, sp, sp * 2) if (full and m <= 64) else (sp,)):
                            out.append(dict(impl="dot", tile_m=tm, tile_n=tn, tile_k=tk, warps=w, stages=stg, split_k=s))
    # large-M floor: dequantise + cuBLAS, and the legacy kernel (a bucket never ends up slower than either)
    if m >= 8:
        dq = [(32, 128, 4), (64, 128, 4), (16, 256, 4), (32, 256, 8)] if full else [(32, 128, 4), (64, 128, 4)]
        for tn, tk, w in dq:
            out.append(dict(impl="dequant_matmul", tile_m=16, tile_n=tn, tile_k=tk, warps=w, stages=1, split_k=1))
    if m >= 9:
        out.append(dict(impl="legacy"))
    return _dedupe(out, m, sm, group)


def limit_candidates(cands: list[Config], limit: int) -> list[Config]:
    """At most ``limit`` candidates for the ``--quick`` sweep: the heuristic default (first entry) and the first entry of
    every other impl are always kept, the remainder is an evenly spaced sample of the rest (deterministic)."""
    if limit <= 0 or len(cands) <= limit:
        return list(cands)
    keep: list[Config] = [cands[0]]
    seen = {cands[0]["impl"]}
    for c in cands[1:]:
        if c["impl"] not in seen and len(keep) < limit:
            seen.add(c["impl"])
            keep.append(c)
    rest = [c for c in cands[1:] if c not in keep]
    room = limit - len(keep)
    if room > 0 and rest:
        step = len(rest) / room
        keep.extend(rest[int(i * step)] for i in range(room))
    return keep[:limit]


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
