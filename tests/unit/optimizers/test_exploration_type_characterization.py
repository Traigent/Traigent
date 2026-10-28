"""Characterization tests for MockRemoteService._determine_exploration_type."""

from types import SimpleNamespace

import pytest

from traigent.optimizers.remote_services import MockRemoteService


@pytest.fixture
def service():
    return object.__new__(MockRemoteService)


def _t(metrics):
    return SimpleNamespace(metrics=metrics)


def _decide(service, history):
    return service._determine_exploration_type(history, None)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_early_trials_explore(service, count):
    assert _decide(service, [_t({"a": 1.0})] * count) == "exploration"


def test_mid_stage_two_recent_improvements_exploit(service):
    # Compares history[-i] to history[-i-1] for i in 1..4.
    history = [_t({"a": float(i)}) for i in range(8)]  # strictly increasing
    assert _decide(service, history) == "exploitation"


def test_mid_stage_exactly_two_improvements_exploit(service):
    scores = [0, 0, 0, 0, 0, 1, 0, 1]  # last index 7
    # i=1: s[7]=1 > s[6]=0 yes; i=2: s[6]=0 > s[5]=1 no;
    # i=3: s[5]=1 > s[4]=0 yes; i=4: s[4]=0 > s[3]=0 no -> 2 improvements
    history = [_t({"a": float(v)}) for v in scores]
    assert _decide(service, history) == "exploitation"


def test_mid_stage_one_improvement_explores(service):
    scores = [0, 0, 0, 0, 0, 0, 0, 1]
    history = [_t({"a": float(v)}) for v in scores]
    assert _decide(service, history) == "exploration"


def test_mid_stage_flat_or_decreasing_explores(service):
    history = [_t({"a": float(10 - i)}) for i in range(8)]
    assert _decide(service, history) == "exploration"


def test_empty_or_missing_metrics_are_skipped(service):
    history = [_t({"a": float(i)}) for i in range(5)] + [_t({}), _t(None), _t({})]
    # i=1,2,3 hit falsy metrics (skipped), only i=4 can count.
    assert _decide(service, history) == "exploration"


def test_primary_objective_is_first_key_of_current_trial(service):
    history = [_t({"a": 0.0, "b": 100.0}) for _ in range(6)]
    history += [_t({"b": 1.0, "a": 50.0}), _t({"b": 2.0, "a": -1.0})]
    # current's first key: i=1 -> "b": 2.0 vs prev 1.0 (improved); i=2 -> "b": 1.0
    # vs prev 100.0 (no); i=3 -> "a" 0 vs 0 (no); i=4 -> "a" 0 vs 0 (no)
    assert _decide(service, history) == "exploration"
    history[-1] = _t({"b": 200.0})
    history[-2] = _t({"b": 150.0})
    # i=1: 200>150 yes; i=2: 150>100 yes
    assert _decide(service, history) == "exploitation"


def test_missing_primary_in_previous_defaults_to_zero(service):
    history = [_t({"a": 1.0}) for _ in range(5)]
    history += [_t({"z": 5.0}), _t({"z": 7.0}), _t({"z": 9.0})]
    # i=1: z 9>7, i=2: z 7>5; i=3: a 1.0 > prev... history[-3]= z=5.0, prev a=1.0
    assert _decide(service, history) == "exploitation"


def test_previous_none_metrics_raises_attribute_error(service):
    history = (
        [_t({"a": 1.0})] * 5 + [_t(None), _t({"a": 1.0})] * 1 + [_t({"a": 2.0})] * 2
    )
    with pytest.raises(AttributeError):
        _decide(service, history)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (15, "verification"),
        (16, "refinement"),
        (17, "refinement"),
        (18, "verification"),
    ],
)
def test_late_stage_verification_every_third(service, count, expected):
    assert _decide(service, [_t({"a": 1.0})] * count) == expected
