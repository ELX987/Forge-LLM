"""Search-space enumeration, analytic feasibility pruning (Plan 6.6 step 1) and shape bucketing (step 5)."""
from __future__ import annotations

from typing import Any

from forge.kernels.layouts.chip import ChipInfo, chip_info
from forge.kernels.layouts.layout import choose_split_k, tile_acc_regs_per_thread, tile_smem_bytes

Config = dict[str, Any]


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


def normalise_shape(shape: Any) -> dict[str, int]:
    """``{'M':..,'N':..,'K':..}`` from a dict or an ``(M, N, K)`` tuple (extra keys such as 'group' are kept)."""
    if isinstance(shape, dict):
        return {k: int(v) for k, v in shape.items()}
    m, n, k = shape
    return {"M": int(m), "N": int(n), "K": int(k)}


def shape_bucket(shape: Any) -> dict[str, int]:
    """Shape bucket of the cache key: ``M`` rounded up to a power of two, everything else model-native (exact)."""
    s = normalise_shape(shape)
    out = dict(s)
    out["M"] = next_pow2(max(1, s.get("M", 1)))
    return out


def is_feasible(cfg: Config, spec: Any, chip: ChipInfo, shape: dict[str, int] | None = None, group: int = 128) -> bool:
    """Discard configs violating shared-memory, register, async-copy or split-K constraints."""
    tile = (cfg["tile_m"], cfg["tile_n"], cfg["tile_k"])
    stages, warps = cfg.get("stages", 2), cfg.get("warps", 4)
    budget = int((chip.smem_per_sm_kib or 48) * 1024 * 0.95)
    if tile_smem_bytes(tile, stages, spec.quant.weight_bits, group) > budget:
        return False
    if tile_acc_regs_per_thread(tile, warps) > 160:
        return False
    if chip.sm is not None and not chip.has_cp_async and stages > 2:
        return False
    if tile[0] < 16 or tile[1] < 16 or tile[2] < 16:
        return False
    if shape is not None:
        k_tiles = -(-shape["K"] // cfg["tile_k"])
        if cfg.get("split_k", 1) > max(1, k_tiles):
            return False
    return True


def enumerate_configs(spec: Any, chip: Any = None, shape: Any = None, *, group: int = 128) -> list[Config]:
    """All feasible configs of ``spec.search`` for ``shape`` (``split_k='auto'`` resolved with the SM count)."""
    c = chip_info(chip)
    sh = normalise_shape(shape) if shape is not None else None
    m_cap = max(16, next_pow2(sh["M"])) if sh else None
    out: list[Config] = []
    seen: set[tuple] = set()
    for tm in spec.search.tile_m:
        if m_cap is not None and tm > m_cap:
            continue
        for tn in spec.search.tile_n:
            for tk in spec.search.tile_k:
                for stages in spec.search.stages:
                    for warps in spec.search.warps:
                        for sk in spec.search.split_k:
                            if sk == "auto":
                                skv = choose_split_k(sh["M"], sh["N"], sh["K"], tm, tn, tk, c.num_sms) if sh else 1
                            else:
                                skv = int(sk)
                            cfg = {"tile_m": tm, "tile_n": tn, "tile_k": tk, "stages": stages, "warps": warps, "split_k": skv}
                            key = tuple(cfg.values())
                            if key in seen:
                                continue
                            seen.add(key)
                            if is_feasible(cfg, spec, c, sh, group):
                                out.append(cfg)
    return out
