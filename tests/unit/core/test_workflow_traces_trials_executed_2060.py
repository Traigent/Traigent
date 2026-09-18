"""Regression test for #2060.

``WorkflowTraceManager.submit_traces`` warns at run-level when tracing is
enabled but zero spans were collected, on the assumption that this is
almost always a span-wiring fault (#2057/#2059). That assumption is false
for a run that legitimately executes zero trials -- e.g. a shared
``ExecutionBudget`` already exhausted before the first trial (issue #1980,
covered by
``tests/unit/core/test_smart_algo_hard_fail_1681.py::
test_no_raise_for_explicit_zero_max_trials`` and siblings).

``OptimizationOrchestrator._submit_workflow_traces`` now tells the manager
whether any trial actually executed (``bool(self._trials)``), and the
manager only escalates to WARNING when it did.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from tests.shared.mocks.optimizers import MockOptimizer
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult

_SPACE = {"temperature": [0.0, 1.0]}


class _NoopEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        func: Any,
        config: dict[str, Any],
        dataset: Dataset,
        **kwargs: Any,
    ) -> EvaluationResult:
        metrics = {"accuracy": 1.0}
        return EvaluationResult(
            config=config,
            example_results=[],
            aggregated_metrics=metrics,
            total_examples=1,
            successful_examples=1,
            duration=0.0,
            metrics=metrics,
        )


def _orchestrator() -> OptimizationOrchestrator:
    return OptimizationOrchestrator(
        optimizer=MockOptimizer(_SPACE, ["accuracy"]),
        evaluator=_NoopEvaluator(),
        config=TraigentConfig(),
    )


async def test_submit_workflow_traces_marks_zero_trials_when_none_executed(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    orchestrator = _orchestrator()
    orchestrator._trials = []
    orchestrator._workflow_trace_manager.submit_traces = AsyncMock()

    await orchestrator._submit_workflow_traces("session-1")

    orchestrator._workflow_trace_manager.submit_traces.assert_awaited_once_with(
        "session-1", trials_executed=False
    )


async def test_submit_workflow_traces_marks_trials_executed_when_present(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    orchestrator = _orchestrator()
    orchestrator._trials = [object()]  # presence is all that is checked
    orchestrator._workflow_trace_manager.submit_traces = AsyncMock()

    await orchestrator._submit_workflow_traces("session-1")

    orchestrator._workflow_trace_manager.submit_traces.assert_awaited_once_with(
        "session-1", trials_executed=True
    )
