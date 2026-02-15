"""Tests for asyncio.CancelledError re-raise in SimpleScoringEvaluator.

Verifies that CancelledError is NOT swallowed by ``except Exception``
handlers in _call_metric_functions, _call_scoring_function, and evaluate().
SonarQube S7497 requires CancelledError to always propagate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 1) -> Dataset:
    """Create a minimal dataset with *n* examples."""
    examples = [
        EvaluationExample(
            input_data={"question": f"q{i}"},
            expected_output=f"a{i}",
        )
        for i in range(n)
    ]
    return Dataset(name="test_ds", examples=examples)


def _raising_metric(**kwargs: Any) -> float:
    """Metric function that always raises CancelledError."""
    raise asyncio.CancelledError


def _raising_scoring(output: Any, expected: Any) -> float:
    """Scoring function that always raises CancelledError."""
    raise asyncio.CancelledError


# ---------------------------------------------------------------------------
# _call_metric_functions
# ---------------------------------------------------------------------------


def test_call_metric_functions_propagates_cancelled_error():
    """CancelledError from a metric function must propagate."""
    evaluator = SimpleScoringEvaluator(
        metric_functions={"bad_metric": _raising_metric},
    )
    dataset = _make_dataset(1)
    example = dataset.examples[0]

    with pytest.raises(asyncio.CancelledError):
        evaluator._call_metric_functions(
            output="test_output",
            example=example,
            config={},
            dataset=dataset,
            example_index=0,
            llm_metrics=None,
        )


# ---------------------------------------------------------------------------
# _call_scoring_function
# ---------------------------------------------------------------------------


def test_call_scoring_function_propagates_cancelled_error():
    """CancelledError from the scoring function must propagate."""
    evaluator = SimpleScoringEvaluator(
        scoring_function=_raising_scoring,
    )
    dataset = _make_dataset(1)
    example = dataset.examples[0]

    with pytest.raises(asyncio.CancelledError):
        evaluator._call_scoring_function(
            output="test_output",
            example=example,
            llm_metrics=None,
        )


# ---------------------------------------------------------------------------
# evaluate() — _evaluate_single_example path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_propagates_cancelled_error():
    """CancelledError during per-example evaluation must propagate."""
    evaluator = SimpleScoringEvaluator(
        scoring_function=_raising_scoring,
    )
    dataset = _make_dataset(1)

    # The user function itself is fine — only the scoring triggers the error
    def dummy_func(question: str, **kwargs: Any) -> str:
        return "answer"

    with pytest.raises(asyncio.CancelledError):
        await evaluator.evaluate(dummy_func, {}, dataset)
