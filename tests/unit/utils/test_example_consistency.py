"""Classifier and helpers for per-example consistency labels."""

from __future__ import annotations

import pytest

from traigent.api.types import ExampleResult
from traigent.utils.example_consistency import (
    ExampleIdCollisionError,
    cell_outcome,
    classify_consistency,
    format_consistency_line,
    summarize_example_outcomes,
    trial_example_outcomes,
)


def _bools(passes: int, fails: int) -> list[bool]:
    return [True] * passes + [False] * fails


def test_five_evaluations_give_the_three_labels() -> None:
    assert classify_consistency(_bools(0, 5)) == "consistent_fail"
    assert classify_consistency(_bools(2, 3)) == "flaky"
    assert classify_consistency(_bools(5, 0)) == "consistent_pass"


@pytest.mark.parametrize("outcomes", [[], [True], [False], [None, None, True]])
def test_fewer_than_two_known_evaluations_gets_no_label(outcomes) -> None:
    assert classify_consistency(outcomes) is None


def test_unknown_evaluations_never_count_as_a_pass() -> None:
    # Two passes and three unknowns: still consistent_pass over the known two,
    # and with a fail among them the unknowns cannot rescue the example.
    assert classify_consistency([True, True, None, None, None]) == "consistent_pass"
    assert classify_consistency([None, None, False, False]) == "consistent_fail"
    assert classify_consistency([None, True, False]) == "flaky"


def test_cell_outcome_follows_the_failing_examples_rule() -> None:
    assert cell_outcome({"score": 0.5}) == "pass"  # equality passes
    assert cell_outcome({"score": 0.49}) == "fail"
    assert cell_outcome({"score": None, "accuracy": 1.0}) == "pass"
    assert cell_outcome({"success": False, "score": 1.0}) == "fail"
    assert cell_outcome({"score": float("nan")}) == "unknown"
    assert cell_outcome({"score": None, "accuracy": None}) == "unknown"
    assert cell_outcome({"score": True}) == "unknown"


def test_error_cell_is_unknown_even_with_a_passing_score() -> None:
    assert cell_outcome({"score": 1.0, "error": "timeout"}) == "unknown"
    assert cell_outcome({"score": 0.0, "error": "timeout"}) == "unknown"


def test_trial_outcomes_read_example_results_and_mark_errors_unknown() -> None:
    class _Trial:
        metadata = {
            "example_results": [
                ExampleResult("a", {}, "x", "x", {"score": 1.0}, 0.1, True).to_dict(),
                ExampleResult("b", {}, "x", "y", {"score": 0.0}, 0.1, True).to_dict(),
                ExampleResult("c", {}, "x", None, {}, 0.1, False, "boom").to_dict(),
                ExampleResult("d", {}, "x", "x", {}, 0.1, True).to_dict(),
            ]
        }

    assert trial_example_outcomes(_Trial()) == {
        "a": "pass",
        "b": "fail",
        "c": "unknown",
        "d": "unknown",
    }


def test_trial_without_example_results_is_none_not_empty() -> None:
    class _Bare:
        metadata: dict = {}

    assert trial_example_outcomes(_Bare()) is None


def test_summary_counts_unknowns_and_drops_unlabelled_examples() -> None:
    reps = [
        {"f": "fail", "k": "pass", "u": "unknown", "m": "pass"},
        {"f": "fail", "k": "fail", "u": "pass", "m": "pass"},
        {"f": "fail", "k": "pass", "u": "unknown"},
    ]
    summary = summarize_example_outcomes(reps)

    assert summary["f"] == {
        "label": "consistent_fail",
        "passes": 0,
        "fails": 3,
        "unknowns": 0,
    }
    assert summary["k"]["label"] == "flaky"
    # "u" has one known evaluation only: no label. "m" is missing from rep 3,
    # which counts as unknown there.
    assert "u" not in summary
    assert summary["m"] == {
        "label": "consistent_pass",
        "passes": 2,
        "fails": 0,
        "unknowns": 1,
    }


def test_consistency_line_is_absent_without_labels() -> None:
    empty = {"counts": {}, "examples": {}, "reps": 0, "config_hash": None}
    assert format_consistency_line(empty, "winner") is None


def test_consistency_line_text() -> None:
    report = {
        "counts": {"consistent_pass": 41, "consistent_fail": 2, "flaky": 3},
        "examples": {"x": {}},
        "reps": 5,
    }
    assert format_consistency_line(report, "winner") == (
        "41 consistent_pass, 2 consistent_fail, 3 flaky (winner, 5 reps)"
    )


def test_integer_one_and_string_one_are_never_merged() -> None:
    class _Trial:
        metadata = {
            "example_results": [
                ExampleResult(1, {}, "x", "y", {"score": 0.0}, 0.1, True).to_dict(),
                ExampleResult("1", {}, "x", "x", {"score": 1.0}, 0.1, True).to_dict(),
            ]
        }

    with pytest.raises(ExampleIdCollisionError) as excinfo:
        trial_example_outcomes(_Trial())
    message = str(excinfo.value)
    assert "1 (int)" in message and "'1' (str)" in message


def test_the_same_id_twice_is_not_a_collision() -> None:
    class _Trial:
        metadata = {
            "example_results": [
                ExampleResult(7, {}, "x", "x", {"score": 1.0}, 0.1, True).to_dict(),
                ExampleResult(7, {}, "x", "x", {"score": 1.0}, 0.1, True).to_dict(),
            ]
        }

    assert trial_example_outcomes(_Trial()) == {"7": "pass"}
