"""Regression tests for evaluator exact-match comparison semantics."""

from __future__ import annotations

import pytest

from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics import MetricsComputer
from traigent.invokers.base import InvocationResult


class _DummyBaseEvaluator(BaseEvaluator):
    async def evaluate(self, func, config, dataset, **kwargs):  # noqa: D401, ANN001
        raise NotImplementedError


def test_exact_match_coerces_string_output_for_typed_expected(caplog) -> None:
    """String outputs matching typed scalar expected values should score correctly."""
    caplog.set_level("WARNING", logger="traigent.evaluators.base")
    base = _DummyBaseEvaluator()
    local = LocalEvaluator(metrics=["accuracy"])
    dataset = Dataset(
        [
            EvaluationExample({"q": "int"}, 42),
            EvaluationExample({"q": "bool"}, True),
            EvaluationExample({"q": "float"}, 3.5),
        ]
    )
    outputs = ["42", "true", "3.5"]
    expected = [example.expected_output for example in dataset.examples]
    errors = [None, None, None]

    assert base._compute_accuracy(outputs, expected, errors) == pytest.approx(1.0)
    assert local._calculate_example_accuracy("42", 42) == pytest.approx(1.0)
    assert local._compute_accuracy_aggregated(outputs, dataset)[0] == pytest.approx(1.0)
    assert local._compute_real_accuracy("true", True) == pytest.approx(1.0)

    metrics_result = MetricsComputer(metrics=["accuracy"]).compute_metrics(
        [InvocationResult(result=value, is_successful=True) for value in outputs],
        expected,
    )
    assert metrics_result.metrics["accuracy"] == pytest.approx(1.0)
    assert "Coercing string output" in caplog.text


def test_numeric_accuracy_uses_float_tolerance_across_paths() -> None:
    """Float representation noise should not make numeric accuracy miss."""
    actual = 0.1 + 0.2
    expected = 0.3
    base = _DummyBaseEvaluator()
    local = LocalEvaluator(metrics=["accuracy"])
    dataset = Dataset([EvaluationExample({"q": "float"}, expected)])

    assert base._compute_accuracy([actual], [expected], [None]) == pytest.approx(1.0)
    assert local._calculate_example_accuracy(actual, expected) == pytest.approx(1.0)
    assert local._compute_accuracy_aggregated([actual], dataset)[0] == pytest.approx(
        1.0
    )
    assert local._compute_real_accuracy(actual, expected) == pytest.approx(1.0)

    metrics_result = MetricsComputer(metrics=["accuracy"]).compute_metrics(
        [InvocationResult(result=actual, is_successful=True)],
        [expected],
    )
    assert metrics_result.metrics["accuracy"] == pytest.approx(1.0)
