"""#3222 / #3051: clopper_pearson_lower_bound is the exact beta quantile at every n."""

from __future__ import annotations

import pytest

from traigent.tvl import statistics as stats
from traigent.tvl.statistics import clopper_pearson_lower_bound as lower

# (successes, trials, confidence, exact beta.ppf((1-conf)/2, k, n-k+1))
# Values from the #3222 table (scipy 1.17.1 oracle) plus #3051 reproductions.
_TABLE = [
    (1, 40, 0.95, 0.000633),
    (3, 60, 0.99, 0.005711),
    (30, 30, 0.95, 0.884297),
    (29, 30, 0.95, 0.827831),
    (45, 50, 0.95, 0.781865),
    (100, 100, 0.95, 0.963783),
    (15, 29, 0.90, 0.352005),
    (2, 30, 0.95, 0.008178),
    (3, 18, 0.99, 0.019703),
]


def _scipy_exact(k: int, n: int, conf: float) -> float:
    scipy_stats = pytest.importorskip("scipy.stats")
    return float(scipy_stats.beta.ppf((1 - conf) / 2, k, n - k + 1))


@pytest.fixture
def no_scipy(monkeypatch):
    monkeypatch.setattr(stats, "_scipy_stats", None)


@pytest.mark.parametrize(("k", "n", "conf", "expected"), _TABLE)
def test_matches_exact_table_with_scipy(k, n, conf, expected):
    assert lower(k, n, conf) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(("k", "n", "conf", "expected"), _TABLE)
def test_matches_exact_table_without_scipy(no_scipy, k, n, conf, expected):
    assert lower(k, n, conf) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("conf", [0.9, 0.95, 0.99])
def test_fallback_agrees_with_scipy_over_a_sweep(monkeypatch, conf):
    cases = [(k, n) for n in (2, 5, 9, 18, 29, 30, 60, 200) for k in range(n + 1)]
    exact = {(k, n): _scipy_exact(k, n, conf) for k, n in cases if 0 < k < n}
    monkeypatch.setattr(stats, "_scipy_stats", None)
    for (k, n), want in exact.items():
        assert lower(k, n, conf) == pytest.approx(want, abs=1e-6), (k, n, conf)


def test_3051_three_of_eighteen_no_longer_clamps_to_0999(no_scipy):
    assert lower(3, 18, 0.99) == pytest.approx(0.0197, abs=1e-4)
    assert lower(2, 18, 0.99) == pytest.approx(0.0059, abs=1e-4)


@pytest.mark.parametrize("conf", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("use_scipy", [True, False])
def test_3051_monotone_and_below_observed_rate(monkeypatch, conf, use_scipy):
    if not use_scipy:
        monkeypatch.setattr(stats, "_scipy_stats", None)
    for n in range(2, 60):
        previous = -1.0
        for k in range(n + 1):
            value = lower(k, n, conf)
            assert value >= previous - 1e-12, (k, n, conf)
            assert value <= k / n + 1e-12, (k, n, conf)
            previous = value


@pytest.mark.parametrize("use_scipy", [True, False])
def test_edges_are_exact(monkeypatch, use_scipy):
    if not use_scipy:
        monkeypatch.setattr(stats, "_scipy_stats", None)
    assert lower(0, 10, 0.95) == 0.0
    assert lower(0, 1, 0.5) == 0.0
    assert lower(10, 10, 0.95) == pytest.approx(0.025 ** (1 / 10), abs=1e-12)
    assert lower(1, 1, 0.95) == pytest.approx(0.025, abs=1e-12)
    assert lower(500, 500, 0.99) == pytest.approx(0.005 ** (1 / 500), abs=1e-12)


def test_large_n_terminates_and_agrees(no_scipy):
    scipy_stats = pytest.importorskip("scipy.stats")
    want = float(scipy_stats.beta.ppf(0.025, 4000, 6001))
    assert lower(4000, 10000, 0.95) == pytest.approx(want, abs=1e-6)


def test_validation_unchanged():
    for args in [(1, 0, 0.9), (-1, 5, 0.9), (6, 5, 0.9), (1, 5, 0.0), (1, 5, 1.0)]:
        with pytest.raises(ValueError):
            lower(*args)


def test_fallback_extreme_tail_is_relatively_accurate(no_scipy):
    scipy_stats = pytest.importorskip("scipy.stats")
    conf = 0.9999999999999999
    want = float(scipy_stats.beta.ppf((1 - conf) / 2, 1, 2))
    assert lower(1, 2, conf) == pytest.approx(want, rel=1e-6)
