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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        {"input_id": "ex_0", "output": "result", "cost_usd": 0.01}
    ]
    mock_execute_response.operational_metrics = {"latency_ms": 100}
    mock_execute_response.get_total_cost.return_value = 0.01
    mock_transport.execute.return_value = mock_execute_response

    evaluator._transport = mock_transport
    evaluator._capabilities = mock_caps
    evaluator._capability_id = "test"
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
    evaluator._capability_id = "test"
    evaluator._session_id = "sess_1"

    mock_transport = AsyncMock()
    # Make the transport.evaluate call raise CancelledError
    mock_transport.evaluate = AsyncMock(side_effect=asyncio.CancelledError)

    from traigent.evaluators.base import EvaluationExample

    batch = [EvaluationExample(input_data={"question": "q0"}, expected_output="a0")]
    inputs = [{"input_id": "ex_0", "data": {"question": "q0"}}]

    # Mock execute response
    mock_execute_response = MagicMock()
    mock_execute_response.outputs = [
        {"input_id": "ex_0", "output": "result", "cost_usd": 0.01}
    ]
    mock_execute_response.execution_id = "exec_1"
    mock_execute_response.operational_metrics = {"latency_ms": 100}
    mock_execute_response.get_total_cost.return_value = 0.01

    with pytest.raises(asyncio.CancelledError):
        await evaluator._evaluate_outputs(
            transport=mock_transport,
            batch=batch,
            inputs=inputs,
            execute_response=mock_execute_response,
        )
