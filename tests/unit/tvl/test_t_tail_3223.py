"""#3223: t-test tail p-values are exact for every df (no normal switch, no 1 - CDF)."""

from __future__ import annotations

import pytest

from traigent.tvl import statistics as stats
from traigent.tvl.statistics import _t_cdf_approx, _t_cdf_lower, _t_cdf_upper

scipy_stats = pytest.importorskip("scipy.stats")


@pytest.fixture(params=["scipy", "fallback"])
def backend(request, monkeypatch):
    if request.param == "fallback":
        monkeypatch.setattr(stats, "_scipy_stats", None)
    return request.param


def test_issue_case_n200_upper_tail(backend):
    want = float(scipy_stats.t.sf(6.1599, 199))  # 1.98e-9
    assert _t_cdf_upper(6.1599, 199) == pytest.approx(want, rel=1e-6, abs=0)


def test_issue_case_no_underflow_to_zero(backend):
    got = _t_cdf_upper(7612.0, 39)
    assert got > 0.0
    assert got == pytest.approx(2.816e-122, rel=1e-3, abs=0)
    assert got == pytest.approx(float(scipy_stats.t.sf(7612.0, 39)), rel=1e-6, abs=0)


def test_paired_comparison_p_value_n200(backend):
    # end to end: a large-n one-sided test no longer reports the normal-tail value
    got = stats._compute_test_statistics(
        0.5 - 0.3 + 0.05, 0.05, 0.2 / 6.1599 * 1.0, "greater", 199
    )
    assert got[1] == pytest.approx(
        float(scipy_stats.t.sf(got[0], 199)), rel=1e-6, abs=0
    )


@pytest.mark.parametrize("df", [1, 2, 3, 5, 10, 30, 100, 101, 150, 200, 499, 500])
@pytest.mark.parametrize("t", [-40.0, -8.0, -3.0, -1.0, -0.1, 0.1, 1.0, 3.0, 8.0, 40.0])
def test_sweep_matches_scipy_both_tails(backend, df, t):
    assert _t_cdf_upper(t, df) == pytest.approx(
        float(scipy_stats.t.sf(t, df)), rel=1e-6
    )
    assert _t_cdf_lower(t, df) == pytest.approx(
        float(scipy_stats.t.cdf(t, df)), rel=1e-6
    )
    assert _t_cdf_approx(t, df) == pytest.approx(
        float(scipy_stats.t.cdf(t, df)), rel=1e-6
    )


@pytest.mark.parametrize("df", range(1, 501, 7))
def test_dense_df_sweep(backend, df):
    for t in (-12.5, -2.2, 2.2, 12.5):
        assert _t_cdf_upper(t, df) == pytest.approx(
            float(scipy_stats.t.sf(t, df)), rel=1e-6
        )


def test_central_values_unchanged(backend):
    for df in (2, 10, 50, 100):
        for t in (-1.5, 0.0, 0.7, 2.0):
            assert _t_cdf_approx(t, df) == pytest.approx(
                float(scipy_stats.t.cdf(t, df)), abs=1e-9
            )
    assert _t_cdf_upper(0.0, 30) == 0.5


def test_edges(backend):
    assert _t_cdf_upper(float("inf"), 10) == 0.0
    assert _t_cdf_upper(float("-inf"), 10) == 1.0
    assert _t_cdf_upper(1.0, 0) == 0.5


def test_representable_tail_beyond_t_squared_overflow(backend):
    # df=1 tail at t=1e155 is 1/(pi*t) = 3.18e-156; t*t overflows a float.
    assert _t_cdf_upper(1e155, 1) == pytest.approx(
        1 / (3.141592653589793 * 1e155), rel=1e-6
    )
    assert _t_cdf_upper(-1e155, 1) == pytest.approx(1.0, abs=1e-12)
