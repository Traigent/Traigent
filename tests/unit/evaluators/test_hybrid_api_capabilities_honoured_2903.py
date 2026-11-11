"""Hybrid capabilities max_batch_size, supports_evaluation_kwargs, supports_benchmarks (#2903)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.hybrid_api import HybridAPIEvaluator
from traigent.hybrid.protocol import HybridExecuteResponse, ServiceCapabilities
from traigent.utils.exceptions import ConfigurationError


def _transport(caps: ServiceCapabilities) -> MagicMock:
    transport = MagicMock()
    transport.sent_batches = []

    async def execute(request: Any) -> HybridExecuteResponse:
        transport.sent_batches.append([ex["example_id"] for ex in request.examples])
        return HybridExecuteResponse(
            request_id=request.request_id,
            execution_id="exec",
            status="completed",
            outputs=[
                {"example_id": ex["example_id"], "output": "x", "metrics": {"a": 1.0}}
                for ex in request.examples
            ],
            operational_metrics={"cost_usd": 0.0, "latency_ms": 1.0},
            quality_metrics={"a": 1.0},
        )

    transport.execute = AsyncMock(side_effect=execute)
    transport.capabilities = AsyncMock(return_value=caps)
    transport.benchmarks = AsyncMock()
    return transport


def _dataset(n: int) -> Dataset:
    return Dataset(
        examples=[EvaluationExample(input_data={"q": str(i)}) for i in range(n)],
        name="d",
    )


@pytest.mark.asyncio
async def test_batch_size_is_clamped_to_service_max_batch_size() -> None:
    transport = _transport(ServiceCapabilities(version="1.0", max_batch_size=2))
    ev = HybridAPIEvaluator(
        transport=transport, tunable_id="t", keep_alive=False, batch_size=10
    )
    ev._benchmark_id = "b"
    await ev.evaluate(lambda: None, {}, _dataset(5))
    assert [len(b) for b in transport.sent_batches] == [2, 2, 1]


@pytest.mark.asyncio
async def test_evaluation_kwargs_rejected_when_service_does_not_support_them() -> None:
    transport = _transport(ServiceCapabilities(version="1.0", supports_evaluate=True))
    ev = HybridAPIEvaluator(
        transport=transport,
        tunable_id="t",
        keep_alive=False,
        evaluation_kwargs={"judge": "x"},
    )
    ev._benchmark_id = "b"
    with pytest.raises(ConfigurationError, match="supports_evaluation_kwargs"):
        await ev.evaluate(lambda: None, {}, _dataset(1))
    transport.execute.assert_not_called()


def _kwargs_evaluator(transport: MagicMock) -> HybridAPIEvaluator:
    ev = HybridAPIEvaluator(
        transport=transport,
        tunable_id="t",
        keep_alive=False,
        evaluation_kwargs={"judge": "x"},
    )
    ev._benchmark_id = "b"
    return ev


def _recording_budget_evaluator(transport: MagicMock):
    ev = _kwargs_evaluator(transport)
    leases: list[Any] = []
    original = ev._prepare_execution_budget_evaluation

    def spy(*args: Any, **kwargs: Any):
        lease, blocked = original(*args, **kwargs)
        leases.append(lease)
        return lease, blocked

    ev._prepare_execution_budget_evaluation = spy  # type: ignore[method-assign]
    return ev, leases


@pytest.mark.asyncio
async def test_evaluation_kwargs_sent_on_two_phase_evaluate() -> None:
    transport = _transport(
        ServiceCapabilities(
            version="1.0", supports_evaluate=True, supports_evaluation_kwargs=True
        )
    )
    transport.execute.side_effect = None
    transport.execute.return_value = HybridExecuteResponse(
        request_id="r",
        execution_id="e",
        status="completed",
        outputs=[{"example_id": "ex_0", "output": "x"}],
        operational_metrics={"cost_usd": 0.0, "latency_ms": 1.0},
    )
    transport.evaluate = AsyncMock(
        return_value=MagicMock(results=[], aggregate_metrics={})
    )
    await _kwargs_evaluator(transport).evaluate(lambda: None, {}, _dataset(1))
    assert transport.evaluate.call_args.args[0].kwargs == {"judge": "x"}


@pytest.mark.asyncio
async def test_evaluation_kwargs_rejected_when_service_has_no_evaluate() -> None:
    transport = _transport(
        ServiceCapabilities(
            version="1.0", supports_evaluate=False, supports_evaluation_kwargs=True
        )
    )
    with pytest.raises(ConfigurationError, match="evaluate"):
        await _kwargs_evaluator(transport).evaluate(lambda: None, {}, _dataset(1))
    transport.execute.assert_not_called()


@pytest.mark.asyncio
async def test_evaluation_kwargs_rejected_when_execute_returns_combined_scores() -> (
    None
):
    transport = _transport(
        ServiceCapabilities(
            version="1.0", supports_evaluate=True, supports_evaluation_kwargs=True
        )
    )  # the fake execute returns quality_metrics (combined mode)
    with pytest.raises(ConfigurationError, match="execute response"):
        await _kwargs_evaluator(transport).evaluate(lambda: None, {}, _dataset(1))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caps",
    [
        ServiceCapabilities(version="1.0"),  # kwargs unsupported
        ServiceCapabilities(  # no evaluate endpoint
            version="1.0", supports_evaluation_kwargs=True
        ),
        ServiceCapabilities(  # combined mode rejects after execute
            version="1.0", supports_evaluate=True, supports_evaluation_kwargs=True
        ),
    ],
)
async def test_rejection_releases_the_execution_budget_lease(caps) -> None:
    from traigent.core.execution_budget import ExecutionBudget

    ev, leases = _recording_budget_evaluator(_transport(caps))
    with pytest.raises(ConfigurationError):
        await ev.evaluate(
            lambda: None, {}, _dataset(2), budget=ExecutionBudget(max_examples=5)
        )
    assert len(leases) == 1 and leases[0] is not None
    assert leases[0]._closed is True


def test_supports_benchmarks_is_parsed() -> None:
    assert ServiceCapabilities.from_dict({}).supports_benchmarks is None
    caps = ServiceCapabilities.from_dict({"supports_benchmarks": False})
    assert caps.supports_benchmarks is False


@pytest.mark.asyncio
async def test_discovery_does_not_probe_when_service_opted_out() -> None:
    transport = _transport(
        ServiceCapabilities(version="1.0", supports_benchmarks=False)
    )
    ev = HybridAPIEvaluator(transport=transport, tunable_id="t", keep_alive=False)
    with pytest.raises(ConfigurationError, match="supports_benchmarks"):
        await ev.discover_example_ids()
    transport.benchmarks.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(("reported_cost", "expected_untracked"), [(0.5, 0), (None, 1)])
async def test_combined_mode_rejection_still_accounts_executed_batch(
    reported_cost, expected_untracked
) -> None:
    """The batch already ran, so its examples and cost are debited before refusing."""
    from traigent.core.execution_budget import ExecutionBudget

    transport = _transport(
        ServiceCapabilities(
            version="1.0", supports_evaluate=True, supports_evaluation_kwargs=True
        )
    )
    original = transport.execute.side_effect

    async def execute(request: Any) -> HybridExecuteResponse:
        response = await original(request)
        response.operational_metrics = (
            {"cost_usd": reported_cost, "latency_ms": 1.0}
            if reported_cost is not None
            else {"latency_ms": 1.0}
        )
        return response

    transport.execute.side_effect = execute
    budget = ExecutionBudget(max_examples=10)
    with pytest.raises(ConfigurationError, match="execute response"):
        await _kwargs_evaluator(transport).evaluate(
            lambda: None, {}, _dataset(2), budget=budget
        )
    snap = budget.snapshot()
    assert snap.consumed_examples == 1  # default batch_size=1: one batch ran
    assert snap.consumed_cost == pytest.approx(reported_cost or 0.0)
    assert snap.untracked_trials == expected_untracked
