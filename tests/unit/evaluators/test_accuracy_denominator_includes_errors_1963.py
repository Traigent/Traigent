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

The fixture shape used above (an errored row's output is ALSO ``None``) does
not actually exercise the numerator-credit guard: a ``None`` output never
matches a real expected value regardless of error status, so those tests pass
identically whether or not the ``error is None`` guard exists. The realistic
failure mode -- and the one that actually motivates the fix -- is a call that
errors AFTER producing a matching output (a downstream failure, a
post-processing exception, a timeout after the model already responded):
``test_compute_accuracy_matching_output_with_error_earns_no_numerator_credit``
and ``TestLocalEvaluatorAccuracyAggregated
.test_matching_output_with_error_earns_no_numerator_credit`` cover that case
directly, and are the tests that actually go RED if either evaluator's
numerator guard is removed. ``TestMissingExpectedOutputSemanticsAgree`` covers
a related but separate discrepancy: the two paths must exclude the SAME set of
"missing expected output" rows from the denominator, not just agree on
error-counting.
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


def test_compute_accuracy_matching_output_with_error_earns_no_numerator_credit():
    """The realistic case the fixture above cannot catch: a call errors AFTER
    producing a matching output -- a downstream failure, a post-processing
    exception, a timeout after the model already responded. That row carries
    a MATCHING output and a non-null error simultaneously.

    Under the pre-#1963 code that row was dropped from both numerator and
    denominator (no visible effect either way). The bug this test actually
    targets is narrower and lives in the surviving guard itself: does an
    errored-but-matching row silently earn numerator credit? It must not --
    it counts in the denominator with zero credit.
    """
    evaluator = _DummyEvaluator()
    # Row 0: no error, matches -> correct.
    # Row 1: MATCHES ("a" == "a") but errored downstream -> must NOT count.
    # Row 2: no error, mismatches -> not correct.
    outputs = ["a", "a", "b"]
    expected = ["a", "a", "a"]
    errors = [None, "boom", None]

    accuracy = evaluator._compute_accuracy(outputs, expected, errors)

    # Denominator = 3 (all have real expected output). Numerator = 1 (only
    # row 0: correct output AND no error). If the guard were dropped, row 1
    # would also count, giving 2/3 (~0.667) instead of 1/3.
    assert accuracy == pytest.approx(1 / 3)
    assert accuracy != pytest.approx(2 / 3)


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

    def test_matching_output_with_error_earns_no_numerator_credit(self):
        """Same realistic case as the BaseEvaluator sibling above: a call
        errors AFTER producing a matching output. The row must count in the
        denominator but never earn numerator credit merely because its
        (pre-error) output happens to equal expected.
        """
        evaluator = LocalEvaluator(metrics=["accuracy"])
        dataset = Dataset(
            [
                EvaluationExample({"i": 0}, "match"),
                EvaluationExample({"i": 1}, "match"),
                EvaluationExample({"i": 2}, "match"),
            ],
            name="matching_output_but_errored_aggregated",
        )
        # Row 0: matches, no error -> correct.
        # Row 1: MATCHES but errored downstream -> must NOT count.
        # Row 2: mismatches, no error -> not correct.
        outputs = ["match", "match", "nope"]
        errors = [None, "boom", None]

        accuracy, total = evaluator._compute_accuracy_aggregated(
            outputs, dataset, errors
        )

        # Denominator = 3. Numerator = 1. If the `if error is not None:
        # continue` guard were dropped, row 1 would also count: 2/3 (~0.667)
        # instead of 1/3.
        assert total == 3
        assert accuracy == pytest.approx(1 / 3)
        assert accuracy != pytest.approx(2 / 3)

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


class TestMissingExpectedOutputSemanticsAgree:
    """``BaseEvaluator._compute_accuracy`` excludes a missing expected output
    via ``_is_empty_expected_output`` (None, or an empty/whitespace-only
    string). ``LocalEvaluator._compute_accuracy_aggregated`` must use the
    SAME predicate, not a narrower ``expected is None`` check -- otherwise an
    empty-string expected output is excluded on one path and counted as a
    real (near-certain) miss on the other, and the two accuracy numbers a
    user could see for the identical dataset disagree about the denominator
    itself, independent of the error-counting fix above.
    """

    def test_compute_accuracy_excludes_empty_and_whitespace_expected_output(self):
        """Base path: already covered by
        ``test_compute_accuracy_still_excludes_missing_expected_output``
        above; restated here with a whitespace-only case alongside empty for
        direct comparison with the LocalEvaluator test below.
        """
        evaluator = _DummyEvaluator()
        outputs = ["a", "b", "c"]
        expected = ["a", "", "   "]
        errors = [None, None, None]

        accuracy = evaluator._compute_accuracy(outputs, expected, errors)

        # Examples 1 (empty) and 2 (whitespace-only) are excluded; only
        # example 0 remains, and it matches: 1/1.
        assert accuracy == pytest.approx(1.0)

    def test_compute_accuracy_aggregated_excludes_empty_and_whitespace_expected_output(
        self,
    ):
        evaluator = LocalEvaluator(metrics=["accuracy"])
        dataset = Dataset(
            [
                EvaluationExample({"i": 0}, "match"),
                EvaluationExample({"i": 1}, ""),  # empty expected -> excluded
                EvaluationExample({"i": 2}, "   "),  # whitespace-only -> excluded
            ],
            name="empty_expected_output_local",
        )
        # Rows 1 and 2 would mismatch if counted at all (output != expected),
        # so a narrower `expected is None` check would drag accuracy down to
        # 1/3 by counting them as real misses instead of excluding them.
        outputs = ["match", "definitely not empty", "definitely not blank"]
        errors = [None, None, None]

        accuracy, total = evaluator._compute_accuracy_aggregated(
            outputs, dataset, errors
        )

        # Only example 0 has a real expected output; it matches: 1/1, not
        # 1/3.
        assert total == 1
        assert accuracy == pytest.approx(1.0)
        assert accuracy != pytest.approx(1 / 3)


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
