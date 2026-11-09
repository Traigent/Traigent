"""Tests for the degenerate knob variation warning (issue #2025)."""

import warnings

import pytest

import traigent
from traigent.api.decorators import optimize
from traigent.utils.exceptions import TraigentWarning

_MARKER = "may not differ enough to compare"


def _decorate(space):
    @optimize(configuration_space=space, objectives=["accuracy"], algorithm="grid")
    def agent(q):
        return traigent.get_config().get("temperature")

    return agent


def _hits(record):
    return [str(w.message) for w in record if _MARKER in str(w.message)]


def test_duplicate_values_in_a_knob_warn() -> None:
    with pytest.warns(TraigentWarning, match=_MARKER) as record:
        _decorate({"temperature": [0.7, 0.7, 0.9]})
    (msg,) = _hits(record)
    assert "'temperature' lists 3 values but only 2 are distinct" in msg
    assert "'agent'" in msg


def test_indistinguishable_values_of_a_preset_knob_warn() -> None:
    with pytest.warns(TraigentWarning, match=_MARKER) as record:
        _decorate({"temperature": [0.0001, 0.0002]})
    assert "very narrow gap" in _hits(record)[0]


def test_cluster_of_three_close_values_warns() -> None:
    with pytest.warns(TraigentWarning, match=_MARKER):
        _decorate({"temperature": [0.70, 0.701, 0.702]})


def test_issue_example_with_float_error_gap_warns() -> None:
    # 0.71 - 0.70 == 0.010000000000000009, which a strict `< floor` compare misses.
    assert 0.71 - 0.70 > 0.01
    with pytest.warns(TraigentWarning, match=_MARKER) as record:
        _decorate({"temperature": [0.70, 0.71, 0.72]})
    assert "very narrow gap" in _hits(record)[0]


def test_gap_just_above_the_floor_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _decorate({"temperature": [0.70, 0.7101]})
    assert _hits(record) == []


@pytest.mark.parametrize(
    "space",
    [
        {"temperature": [0.0, 0.5, 1.0]},
        {"temperature": [0.1, 0.2]},  # timid but distinguishable
        {"model": ["a", "b"], "temperature": [0.7]},  # pinned knob is fine
        {"flag": [True, False]},  # bool must not be read as 1/0 numbers
        {"custom": [1000, 1010]},  # unknown knob, relative gap 1%, above the 0.1% floor
    ],
)
def test_healthy_spaces_do_not_warn(space) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _decorate(space)
    assert _hits(record) == []


def test_one_message_names_every_problem_knob() -> None:
    with pytest.warns(TraigentWarning, match=_MARKER) as record:
        _decorate({"temperature": [0.7, 0.7], "top_p": [0.5, 0.501]})
    (msg,) = _hits(record)
    assert "'temperature'" in msg and "'top_p'" in msg


def test_equal_int_and_float_count_as_duplicates_but_bool_does_not() -> None:
    with pytest.warns(TraigentWarning, match=_MARKER) as record:
        _decorate({"custom": [1, 1.0, 5]})
    assert "only 2 are distinct" in _hits(record)[0]
    with warnings.catch_warnings(record=True) as other:
        warnings.simplefilter("always")
        _decorate({"flag": [True, 1]})
    assert _hits(other) == []


def test_close_pair_is_flagged_even_with_a_distant_third_value() -> None:
    with pytest.warns(TraigentWarning, match=_MARKER):
        _decorate({"temperature": [0.7, 0.701, 1.0]})
