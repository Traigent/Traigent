"""Classifier and helpers for per-example consistency labels."""

from __future__ import annotations

import pytest

from traigent.api.types import ExampleResult
from traigent.utils.example_consistency import (
    ExampleIdCollisionError,
    build_consistency_report,
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


@pytest.mark.parametrize(
    "outcomes",
    [[], [True], [False], [None, True], [True, None, None], [True, True, None, None]],
)
def test_too_few_attempts_or_half_errored_gets_no_label(outcomes) -> None:
    assert classify_consistency(outcomes) is None


def test_pass_pass_then_three_errors_is_not_a_clean_pass() -> None:
    assert classify_consistency([True, True, None, None, None]) is None
    # Fewer than half errored: the scored attempts decide.
    assert classify_consistency([True, True, True, None, None]) == "consistent_pass"
    assert classify_consistency([None, False, False, False]) == "consistent_fail"
    assert classify_consistency([None, True, False, True]) == "flaky"


def test_cell_outcome_follows_the_failing_examples_rule() -> None:
    assert cell_outcome({"score": 0.5}) == "pass"  # equality passes
    assert cell_outcome({"score": 0.49}) == "fail"
    assert cell_outcome({"score": None, "accuracy": 1.0}) == "pass"
    assert cell_outcome({"success": False, "score": 1.0}) == "unknown"
    assert cell_outcome({"score": float("nan")}) == "unknown"
    assert cell_outcome({"score": None, "accuracy": None}) == "unknown"
    assert cell_outcome({"score": True}) == "unknown"


def test_error_cell_is_unknown_even_with_a_passing_score() -> None:
    assert cell_outcome({"score": 1.0, "error": "timeout"}) == "unknown"
    assert cell_outcome({"score": 0.0, "error": "timeout"}) == "unknown"


def test_both_error_shapes_are_errored_attempts() -> None:
    """One rule: no judgment (error present, or success=False) is errored."""
    assert cell_outcome({"score": 1.0, "error": ""}) == "unknown"
    assert cell_outcome({"score": 0.0, "success": False, "error": None}) == "unknown"
    assert cell_outcome({"score": 1.0, "success": False}) == "unknown"
    # A cell that ran and scored low is a fail; an empty error text is not
    # absent: ``error is not None`` is the presence test everywhere.
    assert cell_outcome({"score": 0.0, "success": True, "error": None}) == "fail"


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


def test_summary_counts_every_attempt_and_keeps_errored_examples() -> None:
    reps = [
        {"f": "fail", "k": "pass", "u": "unknown", "m": "pass"},
        {"f": "fail", "k": "fail", "u": "pass", "m": "pass"},
        {"f": "fail", "k": "pass", "u": "unknown"},
    ]
    summary = summarize_example_outcomes(reps)

    assert summary["f"] == {
        "label": "consistent_fail",
        "attempted": 3,
        "passes": 0,
        "fails": 3,
        "errored": 0,
    }
    assert summary["k"]["label"] == "flaky"
    # "u" errored in two of three attempts: no label, but it is kept and counted.
    assert summary["u"] == {
        "label": None,
        "attempted": 3,
        "passes": 1,
        "fails": 0,
        "errored": 2,
    }
    # "m" is missing from attempt 3, which counts as errored there.
    assert summary["m"] == {
        "label": "consistent_pass",
        "attempted": 3,
        "passes": 2,
        "fails": 0,
        "errored": 1,
    }


def test_a_crashed_attempt_is_errored_for_every_example() -> None:
    summary = summarize_example_outcomes(
        [{"a": "pass", "b": "fail"}, None, {"a": "pass", "b": "fail"}]
    )

    assert summary["a"]["attempted"] == 3 and summary["a"]["errored"] == 1
    assert summary["b"]["attempted"] == 3 and summary["b"]["errored"] == 1
    assert summary["a"]["label"] == "consistent_pass"
    assert summary["b"]["label"] == "consistent_fail"


def test_one_attempt_reports_nothing() -> None:
    assert summarize_example_outcomes([{"a": "pass"}]) == {}
    assert summarize_example_outcomes([None]) == {}


def test_report_counts_errored_examples_separately() -> None:
    summary = summarize_example_outcomes(
        [{"a": "pass", "e": "unknown"}, {"a": "pass", "e": "unknown"}]
    )
    report = build_consistency_report(summary, reps=2)

    assert report["counts"] == {
        "consistent_pass": 1,
        "consistent_fail": 0,
        "flaky": 0,
        "errored": 1,
    }
    assert report["examples"]["e"]["label"] is None


def test_consistency_line_is_absent_without_labels() -> None:
    empty = {"counts": {}, "examples": {}, "reps": 0, "config_hash": None}
    assert format_consistency_line(empty, "winner") is None


def test_consistency_line_text() -> None:
    report = {
        "counts": {
            "consistent_pass": 41,
            "consistent_fail": 2,
            "flaky": 3,
            "errored": 1,
        },
        "examples": {"x": {}},
        "reps": 5,
    }
    assert format_consistency_line(report, "winner") == (
        "41 consistent_pass, 2 consistent_fail, 3 flaky, 1 errored (winner, 5 reps)"
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
