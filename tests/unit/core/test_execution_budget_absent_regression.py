"""Absent-budget byte-identical regression tests (issue #1980, matrix item 4).

When no ``ExecutionBudget`` is attached, every seam the feature touches must be
untouched: the cost enforcer's limit, ``self.timeout``, the sample-budget
manager, and the stop-condition list are all unchanged, no ``ExecutionBudget``
stop condition is registered, and the finalize hook adds neither budget metadata
nor a new warning code. This locks in the "absent => byte-identical legacy
behavior" property the implementation advertises at every seam.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.shared.mocks.optimizers import MockOptimizer
from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.stop_conditions import ExecutionBudgetStopCondition
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)


@pytest.fixture(autouse=True)
def disable_mock_mode() -> None:
    """Deterministic cost accounting; no LLM/network is ever touched here."""
    os.environ["TRAIGENT_MOCK_LLM"] = "false"


class _CostAwareEvaluator(BaseEvaluator):
    """Offline evaluator reporting a fixed cost and examples per trial."""

    def __init__(self, cost_per_eval: float = 0.05) -> None:
        super().__init__()
        self.cost_per_eval = cost_per_eval

    async def evaluate(
        self,
        func: Any,
        config: dict[str, Any],
        dataset: Dataset,
        **_kwargs: Any,
    ) -> EvaluationResult:
        total = len(list(dataset.examples))
        metrics = {
            "accuracy": 1.0,
            "cost": self.cost_per_eval,
            "examples_attempted": total,
        }
        result = EvaluationResult(
            config=config,
            example_results=[],
            aggregated_metrics=metrics,
            total_examples=total,
            successful_examples=total,
            duration=0.0,
            metrics=metrics,
        )
        result.examples_consumed = total
        return result


def _orchestrator(**kwargs: Any) -> OptimizationOrchestrator:
    return OptimizationOrchestrator(
        optimizer=MockOptimizer({"temperature": [0.1, 0.2]}, ["accuracy"]),
        evaluator=_CostAwareEvaluator(),
        **kwargs,
    )


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"query": "q1"}, "a1"),
            EvaluationExample({"query": "q2"}, "a2"),
        ],
        name="regression",
    )


# ---------------------------------------------------------------------------
# The default budget attribute is None
# ---------------------------------------------------------------------------


def test_execution_budget_attribute_defaults_to_none() -> None:
    orchestrator = _orchestrator(cost_limit=5.0, cost_approved=True)
    assert getattr(orchestrator, "execution_budget", "missing") is None


# ---------------------------------------------------------------------------
# _apply_execution_budget is a no-op when no budget is attached
# ---------------------------------------------------------------------------


def test_absent_budget_leaves_cost_enforcer_limit_unchanged() -> None:
    orchestrator = _orchestrator(cost_limit=5.0, cost_approved=True)
    before = orchestrator.cost_enforcer.config.limit
    orchestrator._apply_execution_budget()
    assert orchestrator.cost_enforcer.config.limit == before == pytest.approx(5.0)


def test_absent_budget_leaves_timeout_unchanged() -> None:
    orchestrator = _orchestrator(timeout=30.0, cost_limit=5.0, cost_approved=True)
    orchestrator._apply_execution_budget()
    assert orchestrator.timeout == pytest.approx(30.0)


def test_absent_budget_leaves_sample_manager_identical() -> None:
    orchestrator = _orchestrator(
        max_total_examples=10, cost_limit=5.0, cost_approved=True
    )
    before = orchestrator._sample_budget_manager
    orchestrator._apply_execution_budget()
    # Same object -> not rebuilt.
    assert orchestrator._sample_budget_manager is before


def test_absent_budget_registers_no_execution_budget_condition() -> None:
    orchestrator = _orchestrator(
        max_trials=5, max_total_examples=10, cost_limit=5.0, cost_approved=True
    )
    before = list(orchestrator._stop_condition_manager._conditions)
    orchestrator._apply_execution_budget()
    after = list(orchestrator._stop_condition_manager._conditions)

    # Stop-condition list is identical (same objects, same order).
    assert after == before
    assert not any(isinstance(c, ExecutionBudgetStopCondition) for c in after)


# ---------------------------------------------------------------------------
# Finalize hook adds nothing when no budget is attached
# ---------------------------------------------------------------------------


def _empty_result() -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config=None,
        best_score=None,
        optimization_id="regression",
        duration=0.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )


def test_attach_snapshot_is_noop_without_budget() -> None:
    orchestrator = _orchestrator(cost_limit=5.0, cost_approved=True)
    assert getattr(orchestrator, "execution_budget", None) is None

    result = _empty_result()
    # Method does not use `self`; call it unbound with a dummy self.
    OptimizedFunction._attach_execution_budget_snapshot(None, result, orchestrator)

    assert "execution_budget" not in result.metadata
    assert "EXECUTION_BUDGET_UNTRACKED_COST" not in result.warning_codes


# ---------------------------------------------------------------------------
# A full offline run with no budget stops normally and stays budget-free
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_absent_budget_full_run_stops_normally() -> None:
    orchestrator = _orchestrator(
        max_trials=3, cost_limit=5.0, cost_approved=True, timeout=60.0
    )

    async def func(**_kwargs: Any) -> str:
        return "ok"

    await orchestrator.optimize(func=func, dataset=_dataset())

    # No budget was attached and none appeared.
    assert getattr(orchestrator, "execution_budget", None) is None
    # A normal (non-budget) stop reason was recorded.
    assert orchestrator._stop_reason != "execution_budget"
