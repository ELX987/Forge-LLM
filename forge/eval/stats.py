"""Statistics for evaluation (Plan EV-5, 10.1): bootstrap CIs, binomial CIs, FDR, noise floors, non-inferiority.

Pure numpy + stdlib (no scipy).  Everything is seeded and deterministic given ``seed``.

Conventions
-----------
* "paired" means item ``i`` of ``a`` (parent) and ``b`` (candidate) are the same test item.
* ``alpha`` is the two-sided level for CIs (default 0.05 -> 95% CI) and the one-sided level for
  non-inferiority tests.
* A *noise floor* is the half-width of the interval in which the difference of two independent
  evaluations of the **same** model is expected to fall (95% by default); differences inside it must never
  be ranked (EV-5 / "Forge's UI must never rank such values").
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Sequence

import numpy as np

_ND = NormalDist()


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(0 if seed is None else int(seed))


def _as1d(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a


# --------------------------------------------------------------------------------------------
# quantiles of reference distributions
# --------------------------------------------------------------------------------------------
def normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF."""
    return _ND.inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


def t_quantile(p: float, df: float) -> float:
    """Student-t quantile, exact (bisection on the regularised-incomplete-beta CDF :func:`_t_cdf`).

    (An earlier Cornish-Fisher shortcut under-estimated the quantile by ~6% at df=3, which is exactly the regime of the
    default 4-repeat noise floors.)  Accurate to ~1e-12; ``df -> inf`` converges to the normal quantile."""
    if df <= 0:
        raise ValueError("df must be positive")
    p = min(max(p, 1e-12), 1 - 1e-12)
    if p == 0.5:
        return 0.0
    if df > 1e7:
        return normal_quantile(p)
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 10.0
    while _t_cdf(hi, df) < p and hi < 1e12:
        hi *= 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """Result of a (paired) bootstrap comparison ``b`` (candidate) versus ``a`` (parent)."""

    estimate: float  # statistic on the full sample (diff or ratio)
    ci_low: float
    ci_high: float
    p_value: float  # two-sided bootstrap p-value for H0: no difference (statistic == null_value)
    n_items: int
    a_mean: float
    b_mean: float
    stat: str = "mean_diff"
    alpha: float = 0.05
    n_boot: int = 0
    se: float = 0.0
    excludes_null: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boot_indices(n: int, n_boot: int, rng: np.random.Generator, chunk_elems: int = 4_000_000):
    """Yield index blocks of shape (m, n) with m*n <= chunk_elems."""
    m_max = max(1, chunk_elems // max(1, n))
    done = 0
    while done < n_boot:
        m = min(m_max, n_boot - done)
        yield rng.integers(0, n, size=(m, n))
        done += m


def bootstrap_distribution(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray | None = None,
    *,
    stat: str = "mean_diff",
    n: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution (length ``n``) of the paired statistic.

    ``stat``: ``mean_diff`` (mean(b)-mean(a)), ``ratio`` (mean(b)/mean(a)), ``mean`` (mean(a), unpaired, ``b`` ignored).
    Items are resampled jointly (paired), which is what gives paired comparisons their power.
    """
    A = _as1d(a)
    N = A.size
    if N == 0:
        raise ValueError("empty sample")
    B = _as1d(b) if b is not None else None
    if B is not None and B.size != N:
        raise ValueError(f"paired samples must have equal length ({N} vs {B.size})")
    rng = _rng(seed)
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for idx in _boot_indices(N, n, rng):
        m = idx.shape[0]
        ma = A[idx].mean(axis=1)
        if stat == "mean":
            out[pos : pos + m] = ma
        else:
            if B is None:
                raise ValueError(f"stat={stat!r} needs b")
            mb = B[idx].mean(axis=1)
            if stat == "mean_diff":
                out[pos : pos + m] = mb - ma
            elif stat == "ratio":
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[pos : pos + m] = np.where(ma != 0, mb / ma, np.nan)
            else:
                raise ValueError(f"unknown stat {stat!r}")
        pos += m
    return out


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n: int = 2000,
    *,
    stat: str = "mean_diff",
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Paired percentile bootstrap of ``b`` (candidate) versus ``a`` (parent) over items.

    Returns the point estimate, a ``1-alpha`` percentile CI, and a two-sided bootstrap p-value
    (``2*min(P(boot<=null), P(boot>=null))`` with +1 smoothing).  ``null`` is 0 for ``mean_diff`` and 1 for ``ratio``.
    """
    A, B = _as1d(a), _as1d(b)
    if A.size != B.size:
        raise ValueError(f"paired samples must have equal length ({A.size} vs {B.size})")
    if A.size == 0:
        raise ValueError("empty sample")
    am, bm = float(A.mean()), float(B.mean())
    if stat == "mean_diff":
        est, null = bm - am, 0.0
    elif stat == "ratio":
        est, null = (bm / am if am != 0 else float("nan")), 1.0
    else:
        raise ValueError(f"unknown stat {stat!r}")
    dist = bootstrap_distribution(A, B, stat=stat, n=n, seed=seed)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return BootstrapResult(est, float("nan"), float("nan"), 1.0, A.size, am, bm, stat, alpha, n)
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    k = dist.size
    p_le = (np.sum(dist <= null) + 1) / (k + 1)
    p_ge = (np.sum(dist >= null) + 1) / (k + 1)
    p = float(min(1.0, 2 * min(p_le, p_ge)))
    return BootstrapResult(
        estimate=float(est), ci_low=float(lo), ci_high=float(hi), p_value=p, n_items=int(A.size),
        a_mean=am, b_mean=bm, stat=stat, alpha=alpha, n_boot=int(k), se=float(dist.std(ddof=1)) if k > 1 else 0.0,
        excludes_null=bool(lo > null or hi < null),
    )


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    n: int = 2000,
    *,
    stat: Callable[[np.ndarray], float] | str = "mean",
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(estimate, low, high) percentile bootstrap CI of a one-sample statistic (default mean)."""
    X = _as1d(x)
    if X.size == 0:
        raise ValueError("empty sample")
    if stat == "mean":
        dist = bootstrap_distribution(X, None, stat="mean", n=n, seed=seed)
        est = float(X.mean())
    else:
        fn = {"median": np.median, "p99": lambda v: np.quantile(v, 0.99)}.get(stat, stat) if isinstance(stat, str) else stat
        rng = _rng(seed)
        dist = np.array([fn(X[rng.integers(0, X.size, X.size)]) for _ in range(n)], dtype=np.float64)
        est = float(fn(X))
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def cluster_bootstrap_ratio(
    numer: Sequence[float] | np.ndarray,
    denom: Sequence[float] | np.ndarray,
    n: int = 1000,
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI of ``sum(numer)/sum(denom)`` resampling *clusters* (e.g. sequences; tokens inside are correlated).

    ``numer[i]`` / ``denom[i]`` are per-cluster sums (e.g. flipped tokens, total tokens)."""
    N, D = _as1d(numer), _as1d(denom)
    if N.size != D.size or N.size == 0:
        raise ValueError("numer/denom must be equal non-empty")
    est = float(N.sum() / max(D.sum(), 1e-300))
    rng = _rng(seed)
    vals = np.empty(n)
    pos = 0
    for idx in _boot_indices(N.size, n, rng):
        m = idx.shape[0]
        vals[pos : pos + m] = N[idx].sum(axis=1) / np.maximum(D[idx].sum(axis=1), 1e-300)
        pos += m
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def cluster_bootstrap_quantile(
    values_by_cluster: Sequence[np.ndarray],
    q: float = 0.99,
    n: int = 200,
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI for a pooled quantile (e.g. p99 of per-token KL) resampling clusters (sequences)."""
    clusters = [np.asarray(v, dtype=np.float64).reshape(-1) for v in values_by_cluster if len(v)]
    if not clusters:
        raise ValueError("no data")
    est = float(np.quantile(np.concatenate(clusters), q))
    rng = _rng(seed)
    K = len(clusters)
    vals = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, K, K)
        vals[i] = np.quantile(np.concatenate([clusters[j] for j in pick]), q)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


# --------------------------------------------------------------------------------------------
# binomial / proportion tests
# --------------------------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes, modified Lentz)."""
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse of :func:`betainc` by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binomial_ci(k: int, n: int, alpha: float = 0.05, method: str = "wilson") -> tuple[float, float, float]:
    """(p_hat, low, high) CI for a binomial proportion.  ``method``: wilson (default), clopper_pearson, jeffreys, wald."""
    if n <= 0:
        return float("nan"), 0.0, 1.0
    k = int(min(max(k, 0), n))
    p = k / n
    if method == "wilson":
        z = normal_quantile(1 - alpha / 2)
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return p, max(0.0, centre - half), min(1.0, centre + half)
    if method == "wald":
        z = normal_quantile(1 - alpha / 2)
        half = z * math.sqrt(p * (1 - p) / n)
        return p, max(0.0, p - half), min(1.0, p + half)
    if method == "clopper_pearson":
        lo = 0.0 if k == 0 else beta_quantile(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta_quantile(1 - alpha / 2, k + 1, n - k)
        return p, lo, hi
    if method == "jeffreys":
        lo = 0.0 if k == 0 else beta_quantile(alpha / 2, k + 0.5, n - k + 0.5)
        hi = 1.0 if k == n else beta_quantile(1 - alpha / 2, k + 0.5, n - k + 0.5)
        return p, lo, hi
    raise ValueError(f"unknown method {method!r}")


def binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum of pmf <= pmf(k))."""
    if n == 0:
        return 1.0
    pk = binom_pmf(k, n, p)
    tot = sum(binom_pmf(i, n, p) for i in range(n + 1) if binom_pmf(i, n, p) <= pk * (1 + 1e-9))
    return float(min(1.0, tot))


def mcnemar_exact(a_correct: Sequence[bool] | np.ndarray, b_correct: Sequence[bool] | np.ndarray) -> dict[str, Any]:
    """Exact McNemar test on paired binary outcomes.

    Returns counts ``b_only`` (candidate right, parent wrong), ``a_only``, the discordant total and the
    two-sided exact p-value."""
    A = np.asarray(a_correct, dtype=bool).reshape(-1)
    B = np.asarray(b_correct, dtype=bool).reshape(-1)
    if A.size != B.size:
        raise ValueError("paired samples must have equal length")
    b_only = int(np.sum(B & ~A))
    a_only = int(np.sum(A & ~B))
    d = a_only + b_only
    p = 1.0 if d == 0 else binom_test_two_sided(b_only, d, 0.5)
    return {"b_only": b_only, "a_only": a_only, "discordant": d, "p_value": p}


# --------------------------------------------------------------------------------------------
# multiple comparisons
# --------------------------------------------------------------------------------------------
def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR control.  Returns ``(rejected, adjusted_p)`` in the input order.

    ``adjusted_p`` are the monotone BH step-up adjusted p-values (q-values); ``rejected[i]`` is
    ``adjusted_p[i] <= q``."""
    p = np.asarray(list(pvalues), dtype=np.float64)
    m = p.size
    if m == 0:
        return [], []
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m)
    out[order] = adj
    return [bool(v <= q) for v in out], [float(v) for v in out]


# --------------------------------------------------------------------------------------------
# noise floor
# --------------------------------------------------------------------------------------------
@dataclass
class NoiseFloor:
    """Noise floor of one metric estimated from repeats of the *parent*."""

    mean: float
    std: float
    n: int
    floor: float  # half-width: |difference| below this is "within noise"
    level: float = 0.95
    pair_factor: float = math.sqrt(2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def noise_floor_from_repeats(
    repeats: Sequence[float] | np.ndarray,
    *,
    level: float = 0.95,
    pair_factor: float = math.sqrt(2.0),
) -> NoiseFloor:
    """Noise floor from ``n`` repeated evaluations of the same model (different seeds / batch composition / kernels).

    The difference between a single parent run and a single candidate run of an *identical* model has standard
    deviation ``sigma * sqrt(2)``; the floor is ``t_{level, n-1} * sigma_hat * pair_factor``.  With a single repeat the
    variance is unknown: ``floor=inf`` (nothing can be called distinguishable).  With zero variance the floor is 0.0
    (deterministic evaluation - then any difference is real, but note this is also what a *too small* number of
    repeats of a noisy stack can look like; the harness records ``n``)."""
    x = _as1d(repeats)
    n = int(x.size)
    if n == 0:
        raise ValueError("no repeats")
    if n == 1:
        return NoiseFloor(mean=float(x[0]), std=float("nan"), n=1, floor=float("inf"), level=level, pair_factor=pair_factor)
    sd = float(x.std(ddof=1))
    t = t_quantile(0.5 + level / 2, n - 1)
    return NoiseFloor(mean=float(x.mean()), std=sd, n=n, floor=float(t * sd * pair_factor), level=level, pair_factor=pair_factor)


@dataclass
class Distinguishable:
    distinguishable: bool
    reason: str
    diff: float
    noise_floor: float | None
    ci_excludes_zero: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def distinguishable(
    diff: float,
    *,
    noise_floor: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    null: float = 0.0,
) -> Distinguishable:
    """Is ``diff`` distinguishable from run-to-run noise?  True only if it exceeds the noise floor (when known)
    **and** its CI (when known) excludes the null.  Unknown evidence is conservative (False)."""
    ci_ex: bool | None = None
    if ci_low is not None and ci_high is not None:
        ci_ex = bool(ci_low > null or ci_high < null)
    nf_ex: bool | None = None
    if noise_floor is not None:
        nf_ex = bool(abs(diff - null) > noise_floor)
    if ci_ex is None and nf_ex is None:
        return Distinguishable(False, "no CI and no noise floor available", diff, noise_floor, ci_ex)
    reasons = []
    ok = True
    if nf_ex is not None:
        ok &= nf_ex
        reasons.append(f"|diff|={abs(diff - null):.4g} {'>' if nf_ex else '<='} noise floor {noise_floor:.4g}")
    if ci_ex is not None:
        ok &= ci_ex
        reasons.append("CI excludes null" if ci_ex else "CI includes null")
    return Distinguishable(bool(ok), "; ".join(reasons), diff, noise_floor, ci_ex)


# --------------------------------------------------------------------------------------------
# non-inferiority
# --------------------------------------------------------------------------------------------
@dataclass
class NonInferiority:
    """One-sided non-inferiority result for ``candidate - parent >= -margin`` (higher is better)."""

    diff: float
    margin: float
    lower: float  # lower one-sided (1-alpha) confidence bound of the diff
    upper: float  # upper one-sided bound
    p_value: float  # H0: diff <= -margin
    p_inferior: float  # H0': diff >= -margin (small => confidently inferior)
    alpha: float
    noninferior: bool
    inferior: bool
    n_items: int = 0
    relative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def noninferiority_paired(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    margin: float,
    *,
    alpha: float = 0.05,
    n: int = 2000,
    seed: int = 0,
    higher_is_better: bool = True,
) -> NonInferiority:
    """Paired bootstrap non-inferiority test of candidate ``b`` against parent ``a`` with absolute ``margin``
    (same units as the items, e.g. 0.005 = 0.5 points on 0/1 accuracy).

    H0: mean(b) - mean(a) <= -margin   (candidate is inferior by at least the margin).
    For lower-is-better metrics pass ``higher_is_better=False`` (the sign is flipped internally)."""
    A, B = _as1d(a), _as1d(b)
    if not higher_is_better:
        A, B = -A, -B
    dist = bootstrap_distribution(A, B, stat="mean_diff", n=n, seed=seed)
    diff = float(B.mean() - A.mean())
    p = float((np.sum(dist <= -margin) + 1) / (dist.size + 1))
    p_inf = float((np.sum(dist >= -margin) + 1) / (dist.size + 1))
    lower = float(np.quantile(dist, alpha))
    upper = float(np.quantile(dist, 1 - alpha))
    if not higher_is_better:  # report diff/bounds in the metric's own units: candidate - parent
        diff, lower, upper = -diff, -upper, -lower
    return NonInferiority(
        diff=diff, margin=margin, lower=lower, upper=upper, p_value=p, p_inferior=p_inf,
        alpha=alpha, noninferior=bool(p < alpha), inferior=bool(p_inf < alpha), n_items=int(A.size),
    )


def noninferiority_from_summary(
    diff: float, se: float, margin: float, *, alpha: float = 0.05, df: float | None = None
) -> NonInferiority:
    """Normal/t-approximation non-inferiority test from a difference and its standard error."""
    if se <= 0 or not math.isfinite(se):
        noninf = diff > -margin
        return NonInferiority(diff, margin, diff, diff, 0.0 if noninf else 1.0, 0.0 if not noninf else 1.0, alpha, noninf, not noninf)
    stat = (diff + margin) / se
    cdf = (lambda z: _ND.cdf(z)) if df is None else (lambda z: _t_cdf(z, df))
    p = 1.0 - cdf(stat)
    p_inf = cdf(stat)
    q = normal_quantile(1 - alpha) if df is None else t_quantile(1 - alpha, df)
    return NonInferiority(diff, margin, diff - q * se, diff + q * se, float(p), float(p_inf), alpha, bool(p < alpha), bool(p_inf < alpha))


def _t_cdf(x: float, df: float) -> float:
    """Student-t CDF via the incomplete beta."""
    if df > 1e6:
        return _ND.cdf(x)
    xt = df / (df + x * x)
    ib = betainc(df / 2.0, 0.5, xt)
    return 1.0 - 0.5 * ib if x >= 0 else 0.5 * ib


def noninferiority_ratio(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    min_ratio: float,
    *,
    alpha: float = 0.05,
    n: int = 2000,
    seed: int = 0,
) -> NonInferiority:
    """Paired bootstrap test that ``mean(b)/mean(a) >= min_ratio`` (e.g. 0.98 of the parent's score).
    Reported ``diff`` is the ratio; ``margin`` is ``1-min_ratio`` (relative)."""
    A, B = _as1d(a), _as1d(b)
    dist = bootstrap_distribution(A, B, stat="ratio", n=n, seed=seed)
    dist = dist[np.isfinite(dist)]
    am, bm = float(A.mean()), float(B.mean())
    ratio = bm / am if am else float("nan")
    if dist.size == 0:
        return NonInferiority(ratio, 1 - min_ratio, float("nan"), float("nan"), 1.0, 1.0, alpha, False, False, int(A.size), True)
    p = float((np.sum(dist <= min_ratio) + 1) / (dist.size + 1))
    p_inf = float((np.sum(dist >= min_ratio) + 1) / (dist.size + 1))
    return NonInferiority(
        diff=ratio, margin=1 - min_ratio, lower=float(np.quantile(dist, alpha)), upper=float(np.quantile(dist, 1 - alpha)),
        p_value=p, p_inferior=p_inf, alpha=alpha, noninferior=bool(p < alpha), inferior=bool(p_inf < alpha),
        n_items=int(A.size), relative=True,
    )


@dataclass
class PairedSummary:
    """Everything the harness needs about one paired metric."""

    name: str
    parent: float
    candidate: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_items: int
    boot: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("boot", None)
        return d


def paired_summary(name: str, a: Sequence[float], b: Sequence[float], *, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> PairedSummary:
    r = paired_bootstrap(a, b, n=n, alpha=alpha, seed=seed)
    return PairedSummary(name, r.a_mean, r.b_mean, r.estimate, r.ci_low, r.ci_high, r.p_value, r.n_items,
                         boot=bootstrap_distribution(a, b, stat="mean_diff", n=n, seed=seed))
