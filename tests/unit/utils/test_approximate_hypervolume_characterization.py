"""Golden characterization tests for ParetoFrontCalculator._approximate_hypervolume.

The calculator is seeded, so Monte Carlo results are deterministic; the golden
values pin the sampling-box bounds and the draw order.
"""

import pytest

from traigent.utils.multi_objective import ParetoFrontCalculator, ParetoPoint

MAX_A_MIN_CL = {"a": True, "c": False, "l": False}


def _p(**objectives):
    return ParetoPoint(config={}, objectives=objectives, trial=None)


@pytest.fixture
def front():
    return [
        _p(a=0.9, c=5.0, l=3.0),
        _p(a=0.7, c=2.0, l=4.0),
        _p(a=0.5, c=1.0, l=1.0),
    ]


def _calc(maximize):
    return ParetoFrontCalculator(
        maximize=maximize, random_seed=7, monte_carlo_samples=500
    )


def test_empty_front_is_zero():
    assert _calc(MAX_A_MIN_CL)._approximate_hypervolume([], None) == 0.0


def test_default_bounds_golden(front):
    hv = _calc(MAX_A_MIN_CL)._approximate_hypervolume(front, None)
    assert hv == pytest.approx(20.304, rel=1e-12)


def test_full_reference_point_golden(front):
    ref = {"a": 0.0, "c": 10.0, "l": 10.0}
    hv = _calc(MAX_A_MIN_CL)._approximate_hypervolume(front, ref)
    assert hv == pytest.approx(57.76, rel=1e-12)


def test_partial_reference_point_ignores_unknown_and_missing_keys(front):
    hv = _calc(MAX_A_MIN_CL)._approximate_hypervolume(front, {"c": 10.0, "zz": 1.0})
    assert hv == pytest.approx(40.56, rel=1e-12)


def test_explicit_objective_subset_golden(front):
    hv = _calc(MAX_A_MIN_CL)._approximate_hypervolume(front, None, ["a", "c"])
    assert hv == pytest.approx(6.3072, rel=1e-12)


def test_single_point_front_golden(front):
    hv = _calc(MAX_A_MIN_CL)._approximate_hypervolume(front[:1], None)
    assert hv == pytest.approx(0.928, rel=1e-12)


def test_rng_consumption_is_pinned(front):
    calc = _calc(MAX_A_MIN_CL)
    calc._approximate_hypervolume(front, None)
    assert calc._rng.random() == pytest.approx(0.960613538890678, rel=1e-12)


def test_undeclared_orientation_raises_before_sampling(front):
    calc = _calc({})
    state = calc._rng.getstate()
    with pytest.raises(ValueError, match="no declared orientation"):
        calc._approximate_hypervolume(front, None)
    assert calc._rng.getstate() == state


def test_objective_missing_from_points_raises_key_error(front):
    calc = _calc(MAX_A_MIN_CL)
    state = calc._rng.getstate()
    with pytest.raises(KeyError):
        calc._approximate_hypervolume(front, None, ["a", "nope"])
    assert calc._rng.getstate() == state
