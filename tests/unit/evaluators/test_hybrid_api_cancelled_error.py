"""Tests for asyncio.CancelledError re-raise in HybridAPIEvaluator.

Verifies that CancelledError is NOT swallowed by ``except Exception``
handlers in:
  1. The progress callback try-except inside evaluate()
  2. _evaluate_outputs()

SonarQube S7497 requires CancelledError to always propagate.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.core.execution_budget import ExecutionBudget
from traigent.core.sample_budget import SampleBudgetManager
from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult

# ---------------------------------------------------------------------------
# 1. Progress callback CancelledError in evaluate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_progress_callback_propagates_cancelled_error():
    """CancelledError from the progress callback must propagate."""
    evaluator = HybridAPIEvaluator(
        api_endpoint="http://localhost:9999",
        batch_size=1,
    )

    # Mock the transport and capabilities
    mock_transport = AsyncMock()
    mock_caps = MagicMock()
    mock_caps.supports_evaluate = False  # Use execute-only mode

    # Mock execute response
    mock_execute_response = MagicMock()
    mock_execute_response.outputs = [
        {"example_id": "ex_0", "output": "result", "cost_usd": 0.01}
    ]
    mock_execute_response.operational_metrics = {"latency_ms": 100}
    mock_execute_response.get_total_cost.return_value = 0.01
    mock_transport.execute.return_value = mock_execute_response

    evaluator._transport = mock_transport
    evaluator._capabilities = mock_caps
    evaluator._tunable_id = "test"
    evaluator._session_id = "sess_1"

    # Create a minimal dataset
    from traigent.evaluators.base import Dataset, EvaluationExample

    dataset = Dataset(
        name="test_ds",
        examples=[
            EvaluationExample(input_data={"question": "q0"}, expected_output="a0"),
        ],
    )

    # Progress callback that raises CancelledError
    def bad_progress(idx: int, info: dict[str, Any]) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluator.evaluate(
            func=lambda: None,
            config={"model": "gpt-4"},
            dataset=dataset,
            progress_callback=bad_progress,
        )


# ---------------------------------------------------------------------------
# 2. _evaluate_outputs CancelledError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_outputs_propagates_cancelled_error():
    """CancelledError in _evaluate_outputs() must propagate."""
    evaluator = HybridAPIEvaluator(
        api_endpoint="http://localhost:9999",
        batch_size=1,
    )
    evaluator._tunable_id = "test"
    evaluator._session_id = "sess_1"

    mock_transport = AsyncMock()
    # Make the transport.evaluate call raise CancelledError
    mock_transport.evaluate = AsyncMock(side_effect=asyncio.CancelledError)

    from traigent.evaluators.base import EvaluationExample

    batch = [EvaluationExample(input_data={"question": "q0"}, expected_output="a0")]
    inputs = [{"example_id": "ex_0", "data": {"question": "q0"}}]

    # Mock execute response
    mock_execute_response = MagicMock()
    mock_execute_response.outputs = [
        {"example_id": "ex_0", "output": "result", "cost_usd": 0.01}
    ]
    mock_execute_response.execution_id = "exec_1"
    mock_execute_response.operational_metrics = {"latency_ms": 100}
    mock_execute_response.get_total_cost.return_value = 0.01

    with pytest.raises(asyncio.CancelledError):
        await evaluator._evaluate_outputs(
            transport=mock_transport,
            config={"model": "gpt-4"},
            batch=batch,
            inputs=inputs,
            execute_response=mock_execute_response,
        )


@pytest.mark.asyncio
async def test_evaluate_cancellation_refunds_execution_budget_admission():
    """Cancellation during a hybrid batch must release shared budget capacity."""
    started = asyncio.Event()
    never = asyncio.Event()

    evaluator = HybridAPIEvaluator(
        api_endpoint="http://localhost:9999",
        batch_size=1,
    )

    async def blocking_batch(*args: Any, **kwargs: Any) -> list[HybridExampleResult]:
        started.set()
        await never.wait()
        return []

    evaluator._get_transport = AsyncMock(return_value=MagicMock())
    evaluator._get_capabilities = AsyncMock(return_value=MagicMock())
    evaluator._ensure_lifecycle_manager = AsyncMock()
    evaluator._execute_batch = blocking_batch

    from traigent.evaluators.base import Dataset, EvaluationExample

    dataset = Dataset(
        name="hybrid-cancellation-budget-test",
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
    )
    budget = ExecutionBudget(max_examples=1)
    evaluation = asyncio.create_task(
        evaluator.evaluate(lambda value: value, {}, dataset, budget=budget)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 0
    assert snapshot.trials == 0


@pytest.mark.asyncio
async def test_evaluate_cancellation_refunds_external_sample_lease():
    """Hybrid cancellation must release admissions without closing the caller lease."""
    started = asyncio.Event()
    never = asyncio.Event()

    evaluator = HybridAPIEvaluator(
        api_endpoint="http://localhost:9999",
        batch_size=1,
    )

    async def blocking_batch(*args: Any, **kwargs: Any) -> list[HybridExampleResult]:
        started.set()
        await never.wait()
        return []

    evaluator._get_transport = AsyncMock(return_value=MagicMock())
    evaluator._get_capabilities = AsyncMock(return_value=MagicMock())
    evaluator._ensure_lifecycle_manager = AsyncMock()
    evaluator._execute_batch = blocking_batch

    from traigent.evaluators.base import Dataset, EvaluationExample

    dataset = Dataset(
        name="hybrid-external-cancellation-budget-test",
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
    )
    manager = SampleBudgetManager(total_budget=1)
    lease = manager.create_lease("hybrid-cancelled")
    evaluation = asyncio.create_task(
        evaluator.evaluate(
            lambda value: value,
            {},
            dataset,
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
