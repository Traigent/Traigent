"""Tests for asyncio.CancelledError re-raise in SimpleScoringEvaluator.

Verifies that CancelledError is NOT swallowed by ``except Exception``
handlers in _call_metric_functions, _call_scoring_function, and evaluate().
SonarQube S7497 requires CancelledError to always propagate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from traigent.core.execution_budget import ExecutionBudget
from traigent.core.sample_budget import SampleBudgetManager
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


@pytest.mark.asyncio
async def test_evaluate_cancellation_refunds_execution_budget_admission():
    """Cancellation after function admission must release the shared budget."""
    started = asyncio.Event()
    never = asyncio.Event()

    async def blocking_func(question: str, **kwargs: Any) -> str:
        started.set()
        await never.wait()
        return "answer"

    evaluator = SimpleScoringEvaluator(scoring_function=lambda output, expected: 1.0)
    budget = ExecutionBudget(max_examples=1)
    evaluation = asyncio.create_task(
        evaluator.evaluate(blocking_func, {}, _make_dataset(1), budget=budget)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 0
    assert snapshot.trials == 0


@pytest.mark.asyncio
async def test_evaluate_cancellation_refunds_external_sample_lease() -> None:
    """Simple scoring must release an orchestrator lease on task cancellation."""
    started = asyncio.Event()
    never = asyncio.Event()

    async def blocking_func(question: str, **kwargs: Any) -> str:
        started.set()
        await never.wait()
        return "answer"

    manager = SampleBudgetManager(total_budget=1)
    lease = manager.create_lease("simple-cancelled")
    evaluator = SimpleScoringEvaluator(scoring_function=lambda output, expected: 1.0)
    evaluation = asyncio.create_task(
        evaluator.evaluate(
            blocking_func,
            {},
            _make_dataset(1),
            sample_lease=lease,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    assert manager.snapshot().consumed == 0
    assert lease.completed == 0
    lease.finalize()


@pytest.mark.asyncio
async def test_simple_cancellation_retains_completed_examples() -> None:
    """A completed example remains charged when the next one is cancelled."""
    second_started = asyncio.Event()
    never = asyncio.Event()
    calls = 0

    async def function(value: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            second_started.set()
            await never.wait()
        return value

    dataset = _make_dataset(2)
    manager = SampleBudgetManager(total_budget=2)
    lease = manager.create_lease("simple-partial-cancelled")
    evaluator = SimpleScoringEvaluator(scoring_function=lambda output, expected: 1.0)
    evaluation = asyncio.create_task(
        evaluator.evaluate(function, {}, dataset, sample_lease=lease)
    )
    await asyncio.wait_for(second_started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    assert lease.completed == 1
    assert lease.consumed == 1
    assert manager.snapshot().consumed == 1
    lease.finalize()


@pytest.mark.asyncio
async def test_caught_cancellation_refunds_external_lease_before_partial_return() -> (
    None
):
    """An outer partial-result handler must not strand an external admission."""
    started = asyncio.Event()
    never = asyncio.Event()
    manager = SampleBudgetManager(total_budget=1)
    lease = manager.create_lease("simple-caught-cancelled")
    evaluator = SimpleScoringEvaluator(scoring_function=lambda output, expected: 1.0)

    async def blocking_func(question: str, **kwargs: Any) -> str:
        started.set()
        await never.wait()
        return "answer"

    async def caller_returns_partial() -> str:
        try:
            await evaluator.evaluate(
                blocking_func,
                {},
                _make_dataset(1),
                sample_lease=lease,
            )
        except asyncio.CancelledError:
            return "partial"
        raise AssertionError("evaluation unexpectedly completed")

    evaluation = asyncio.create_task(caller_returns_partial())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    assert await asyncio.wait_for(evaluation, timeout=1.0) == "partial"
    assert manager.snapshot().consumed == 0
    assert lease.completed == 0
    lease.finalize()


@pytest.mark.asyncio
async def test_caught_cancellation_refunds_execution_budget_before_partial_return() -> (
    None
):
    """A caught cancellation must refund a direct execution-budget admission."""
    started = asyncio.Event()
    never = asyncio.Event()
    budget = ExecutionBudget(max_examples=1)
    evaluator = SimpleScoringEvaluator(scoring_function=lambda output, expected: 1.0)

    async def blocking_func(question: str, **kwargs: Any) -> str:
        started.set()
        await never.wait()
        return "answer"

    async def caller_returns_partial() -> str:
        try:
            await evaluator.evaluate(
                blocking_func,
                {},
                _make_dataset(1),
                budget=budget,
            )
        except asyncio.CancelledError:
            return "partial"
        raise AssertionError("evaluation unexpectedly completed")

    evaluation = asyncio.create_task(caller_returns_partial())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    assert await asyncio.wait_for(evaluation, timeout=1.0) == "partial"
    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 0
    assert snapshot.trials == 0


@pytest.mark.asyncio
async def test_sync_scoring_cancellation_refunds_external_lease_before_partial_return() -> (
    None
):
    """A sync scorer raising CancelledError cannot strand a caller lease."""
    manager = SampleBudgetManager(total_budget=1)
    lease = manager.create_lease("simple-sync-score-cancelled")

    def cancelling_score(output: Any, expected: Any) -> float:
        raise asyncio.CancelledError

    evaluator = SimpleScoringEvaluator(scoring_function=cancelling_score)

    async def caller_returns_partial() -> str:
        try:
            await evaluator.evaluate(
                lambda value: value,
                {},
                _make_dataset(1),
                sample_lease=lease,
            )
        except asyncio.CancelledError:
            return "partial"
        raise AssertionError("evaluation unexpectedly completed")

    assert await asyncio.wait_for(caller_returns_partial(), timeout=1.0) == "partial"
    assert manager.snapshot().consumed == 0
    assert lease.completed == 0
    lease.finalize()


@pytest.mark.asyncio
async def test_sync_scoring_cancellation_refunds_execution_budget_before_partial_return() -> (
    None
):
    """A sync scorer raising CancelledError cannot strand a direct admission."""
    budget = ExecutionBudget(max_examples=1)

    def cancelling_score(output: Any, expected: Any) -> float:
        raise asyncio.CancelledError

    evaluator = SimpleScoringEvaluator(scoring_function=cancelling_score)

    async def caller_returns_partial() -> str:
        try:
            await evaluator.evaluate(
                lambda value: value,
                {},
                _make_dataset(1),
                budget=budget,
            )
        except asyncio.CancelledError:
            return "partial"
        raise AssertionError("evaluation unexpectedly completed")

    assert await asyncio.wait_for(caller_returns_partial(), timeout=1.0) == "partial"
    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 0
    assert snapshot.trials == 0
