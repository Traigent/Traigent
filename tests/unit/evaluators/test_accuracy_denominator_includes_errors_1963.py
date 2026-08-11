"""Regression tests for Traigent#1963: accuracy denominator dropped errored examples.

Both accuracy paths reachable by a live evaluator shared the SAME bug: an
example whose provider call errored was excluded from BOTH the numerator and
the denominator, so a config that fails on half its inputs could report a
perfect accuracy computed only over the surviving half.

* ``BaseEvaluator._compute_accuracy`` -- the metric-registry default, used
  directly by any evaluator that does not override "accuracy".
* ``LocalEvaluator._compute_accuracy_aggregated`` -- the value that actually
  reaches users of the default (local) evaluator: its result OVERWRITES
  whatever ``_compute_accuracy`` computed via ``compute_metrics()``. Fixing
  only the first function would have left the second one's identical bug
  live for the evaluator most users hit.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


class _DummyEvaluator(BaseEvaluator):
    async def evaluate(self, func, config, dataset, **kwargs):  # noqa: D401, ANN001
        raise NotImplementedError


def test_compute_accuracy_counts_errored_examples_against_the_denominator():
    """The issue's own worked example: 100 examples, 50 error, the other 50
    are all correct.

    Old (buggy): 50/50 = 1.0 -- a perfect score reported to the user.
    Fixed: 50/100 = 0.5 -- the honest number.
    """
    evaluator = _DummyEvaluator()
    outputs = ["a"] * 50 + [None] * 50
    expected = ["a"] * 100
    errors = [None] * 50 + ["boom"] * 50

    accuracy = evaluator._compute_accuracy(outputs, expected, errors)

    assert accuracy == pytest.approx(0.5)
    assert accuracy != pytest.approx(1.0)


def test_compute_accuracy_still_excludes_missing_expected_output():
    """The pre-existing, legitimate exclusion (empty/whitespace expected
    output) must survive untouched -- only the error-based exclusion was
    the bug.
    """
    evaluator = _DummyEvaluator()
    outputs = ["a", "b", "c"]
    expected = ["a", "", "c"]
    errors = [None, None, None]

    accuracy = evaluator._compute_accuracy(outputs, expected, errors)

    # Example 2 (empty expected) is excluded; 1 and 3 both match, so 2/2.
    assert accuracy == pytest.approx(1.0)


def test_compute_accuracy_all_errored_is_zero_not_undefined():
    """Every example errors: denominator is every non-empty-expected example,
    numerator is 0 -- accuracy 0.0, not the old 0/0 -> 0.0 by different logic.
    """
    evaluator = _DummyEvaluator()
    outputs = [None, None]
    expected = ["a", "b"]
    errors = ["boom", "boom"]

    accuracy = evaluator._compute_accuracy(outputs, expected, errors)

    assert accuracy == 0.0


class TestLocalEvaluatorAccuracyAggregated:
    """``_compute_accuracy_aggregated`` mirrors the same denominator fix."""

    def test_counts_errored_examples_against_the_denominator(self):
        evaluator = LocalEvaluator(metrics=["accuracy"])
        dataset = Dataset(
            [EvaluationExample({"i": i}, "match") for i in range(4)],
            name="mixed_errors_aggregated",
        )
        outputs = ["match", "match", None, None]
        errors = [None, None, "boom", "boom"]

        accuracy, total = evaluator._compute_accuracy_aggregated(
            outputs, dataset, errors
        )

        # Old (buggy): 2/2 = 1.0 (errored examples dropped from total too).
        # Fixed: 2/4 = 0.5.
        assert total == 4
        assert accuracy == pytest.approx(0.5)
        assert accuracy != pytest.approx(1.0)

    def test_omitted_errors_param_treats_every_example_as_successful(self):
        """Backward compatibility: existing direct callers that never passed
        ``errors`` must see unchanged behaviour.
        """
        evaluator = LocalEvaluator(metrics=["accuracy"])
        dataset = Dataset(
            [EvaluationExample({"i": i}, "match") for i in range(2)],
            name="no_errors_param",
        )
        outputs = ["match", "match"]

        accuracy, total = evaluator._compute_accuracy_aggregated(outputs, dataset)

        assert total == 2
        assert accuracy == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_local_evaluator_evaluate_reports_the_honest_accuracy_with_errors():
    """End-to-end through the real ``evaluate()`` path: the number a user
    actually sees for the default (local) evaluator.
    """
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    dataset = Dataset(
        [EvaluationExample({"idx": i}, "match") for i in range(4)],
        name="mixed_errors_end_to_end_1963",
    )

    def maybe_error(input_data: dict) -> str:
        if input_data["idx"] % 2 == 0:
            raise ValueError(f"boom {input_data['idx']}")
        return "match"

    result = await evaluator.evaluate(maybe_error, {}, dataset)

    # 2 of 4 error; the other 2 match. Old: 2/2 = 1.0. Fixed: 2/4 = 0.5.
    assert result.metrics["accuracy"] == pytest.approx(0.5)
    assert result.metrics["accuracy"] != pytest.approx(1.0)
