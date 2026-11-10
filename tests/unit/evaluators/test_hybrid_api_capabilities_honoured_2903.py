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


@pytest.mark.asyncio
async def test_evaluation_kwargs_allowed_when_advertised() -> None:
    transport = _transport(
        ServiceCapabilities(version="1.0", supports_evaluation_kwargs=True)
    )
    ev = HybridAPIEvaluator(
        transport=transport,
        tunable_id="t",
        keep_alive=False,
        evaluation_kwargs={"judge": "x"},
    )
    ev._benchmark_id = "b"
    await ev.evaluate(lambda: None, {}, _dataset(1))
    transport.execute.assert_called_once()


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
