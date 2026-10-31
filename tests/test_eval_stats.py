from __future__ import annotations

import math

import numpy as np
import pytest

from forge.eval import stats


def test_t_quantile_close_to_exact():
    assert stats.t_quantile(0.975, 4) == pytest.approx(2.776, abs=0.07)
    assert stats.t_quantile(0.975, 30) == pytest.approx(2.042, abs=0.01)
    assert stats.t_quantile(0.975, 1e6) == pytest.approx(1.96, abs=0.005)
    assert stats.t_quantile(0.975, 1) > 10


def test_betainc_known_values():
    assert stats.betainc(1, 1, 0.3) == pytest.approx(0.3, abs=1e-9)
    assert stats.betainc(2, 2, 0.5) == pytest.approx(0.5, abs=1e-9)
    assert stats.betainc(2, 5, 0.0) == 0.0 and stats.betainc(2, 5, 1.0) == 1.0


def test_binomial_ci_methods():
    p, lo, hi = stats.binomial_ci(5, 10, 0.05, "wilson")
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=2e-3) and hi == pytest.approx(0.7634, abs=2e-3)
    _, lo, hi = stats.binomial_ci(0, 10, 0.05, "clopper_pearson")
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-6)
    _, lo, hi = stats.binomial_ci(10, 10, 0.05, "clopper_pearson")
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10), abs=1e-6)
    _, lo, hi = stats.binomial_ci(5, 10, 0.05, "jeffreys")
    assert 0.2 < lo < 0.3 < 0.7 < hi < 0.8
    assert math.isnan(stats.binomial_ci(0, 0)[0])
    with pytest.raises(ValueError):
        stats.binomial_ci(1, 2, method="nope")


def test_binom_test_and_mcnemar():
    assert stats.binom_test_two_sided(5, 10) == pytest.approx(1.0)
    assert stats.binom_test_two_sided(0, 10) == pytest.approx(2 * 0.5**10)
    a = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], bool)
    b = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1], bool)
    r = stats.mcnemar_exact(a, b)
    assert r["a_only"] == 1 and r["b_only"] == 1 and r["p_value"] == pytest.approx(1.0)
    r = stats.mcnemar_exact(np.ones(20, bool), np.r_[np.zeros(10, bool), np.ones(10, bool)])
    assert r["a_only"] == 10 and r["p_value"] < 0.01
    assert stats.mcnemar_exact([True], [True])["p_value"] == 1.0


def test_benjamini_hochberg():
    rej, adj = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.005], 0.05)
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert rej == [True, True, True, True]
    rej, adj = stats.benjamini_hochberg([0.5, 0.8, 0.01], 0.05)
    assert rej == [False, False, True]
    assert adj[2] == pytest.approx(0.03)
    assert stats.benjamini_hochberg([]) == ([], [])
    # adjusted values are monotone in p and capped at 1
    _, adj = stats.benjamini_hochberg([0.9, 0.95, 0.99])
    assert max(adj) <= 1.0


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    r = stats.paired_bootstrap(a, a + 0.3, n=1000, seed=1)
    assert r.estimate == pytest.approx(0.3, abs=1e-9)
    assert r.ci_low > 0 and r.excludes_null and r.p_value < 0.01
    assert r.ci_low == pytest.approx(0.3, abs=1e-9)  # constant shift: paired CI collapses (that is the point of pairing)
    r0 = stats.paired_bootstrap(a, a, n=500, seed=1)
    assert r0.estimate == 0 and not r0.excludes_null and r0.p_value == pytest.approx(1.0)
    noisy = a + rng.normal(0, 1, 300)
    rn = stats.paired_bootstrap(a, noisy, n=1000, seed=2)
    assert rn.ci_low < 0 < rn.ci_high
    # determinism
    assert stats.paired_bootstrap(a, noisy, n=200, seed=5).ci_low == stats.paired_bootstrap(a, noisy, n=200, seed=5).ci_low
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1, 2], [1])


def test_paired_bootstrap_ratio():
    a = np.array([1.0] * 80 + [0.0] * 20)
    b = np.array([1.0] * 78 + [0.0] * 22)
    r = stats.paired_bootstrap(a, b, n=800, stat="ratio", seed=0)
    assert r.estimate == pytest.approx(0.975)
    assert r.ci_low < 0.975 < r.ci_high


def test_bootstrap_ci_and_cluster_helpers():
    x = np.arange(100, dtype=float)
    est, lo, hi = stats.bootstrap_ci(x, 500, seed=0)
    assert lo < est < hi and est == pytest.approx(49.5)
    est, lo, hi = stats.bootstrap_ci(x, 200, stat="median", seed=0)
    assert lo <= est <= hi
    e, lo, hi = stats.cluster_bootstrap_ratio(np.array([1, 2, 3.0]), np.array([10, 10, 10.0]), 300, seed=0)
    assert e == pytest.approx(0.2) and lo <= e <= hi
    e, lo, hi = stats.cluster_bootstrap_quantile([np.arange(50.0), np.arange(50.0, 100)], 0.99, 100, seed=0)
    assert e == pytest.approx(np.quantile(np.arange(100.0), 0.99)) and lo <= hi


def test_noise_floor_from_repeats():
    nf = stats.noise_floor_from_repeats([0.50, 0.52, 0.48, 0.51, 0.49])
    assert nf.n == 5 and nf.mean == pytest.approx(0.5)
    assert 0.05 < nf.floor < 0.07
    assert stats.noise_floor_from_repeats([0.3, 0.3, 0.3]).floor == 0.0
    assert stats.noise_floor_from_repeats([0.3]).floor == math.inf
    with pytest.raises(ValueError):
        stats.noise_floor_from_repeats([])


def test_distinguishable_logic():
    d = stats.distinguishable(0.02, noise_floor=0.05, ci_low=0.01, ci_high=0.03)
    assert not d.distinguishable and "noise floor" in d.reason  # CI excludes 0 but within the noise floor
    d = stats.distinguishable(0.1, noise_floor=0.05, ci_low=-0.01, ci_high=0.2)
    assert not d.distinguishable  # exceeds floor but CI includes 0
    d = stats.distinguishable(0.1, noise_floor=0.05, ci_low=0.02, ci_high=0.2)
    assert d.distinguishable
    assert not stats.distinguishable(0.5).distinguishable  # no evidence -> conservative
    assert stats.distinguishable(0.5, ci_low=0.1, ci_high=0.9).distinguishable


def test_noninferiority_paired():
    rng = np.random.default_rng(3)
    a = (rng.random(600) < 0.8).astype(float)
    same = a.copy()
    r = stats.noninferiority_paired(a, same, margin=0.02, n=800, seed=0)
    assert r.noninferior and not r.inferior and r.diff == 0
    worse = a.copy()
    worse[a == 1][:100] = 0
    idx = np.where(a == 1)[0][:90]
    worse[idx] = 0.0
    r = stats.noninferiority_paired(a, worse, margin=0.02, n=800, seed=0)
    assert r.inferior and not r.noninferior and r.diff < -0.1
    # lower-is-better flips the sign
    r = stats.noninferiority_paired(a, worse, margin=0.02, n=800, seed=0, higher_is_better=False)
    assert r.noninferior


def test_noninferiority_ratio_and_summary():
    a = np.array([1.0] * 90 + [0.0] * 10)
    r = stats.noninferiority_ratio(a, a.copy(), 0.95, n=500, seed=0)
    assert r.noninferior and r.diff == pytest.approx(1.0)
    b = np.array([1.0] * 70 + [0.0] * 30)
    r = stats.noninferiority_ratio(a, b, 0.95, n=500, seed=0)
    assert r.inferior
    s = stats.noninferiority_from_summary(0.0, 0.01, 0.05, alpha=0.05)
    assert s.noninferior
    s = stats.noninferiority_from_summary(-0.08, 0.01, 0.05, alpha=0.05)
    assert s.inferior
    s = stats.noninferiority_from_summary(0.0, 0.0, 0.05)
    assert s.noninferior
    s = stats.noninferiority_from_summary(0.0, 0.03, 0.02, alpha=0.05, df=20)
    assert not s.noninferior and not s.inferior


def test_paired_summary_has_boot_but_serialises_without():
    s = stats.paired_summary("m", [0, 1, 1, 0] * 20, [0, 1, 1, 1] * 20, n=200)
    assert s.boot is not None and "boot" not in s.to_dict()
    assert s.diff == pytest.approx(0.25)
