from datetime import UTC, datetime
from typing import Any

import pytest

from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.optimizers.base import BaseOptimizer


class _StaticOptimizer(BaseOptimizer):
    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        return {"model": "mock"}

    def should_stop(self, history: list[TrialResult]) -> bool:
        return True


class _UnusedEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        func: Any,
        config: dict[str, Any],
        dataset: Dataset,
        **kwargs: Any,
    ) -> EvaluationResult:
        raise AssertionError("test injects completed trials directly")


def test_unmatched_objective_metric_returns_honest_failed_result() -> None:
    optimizer = _StaticOptimizer(
        {"model": ["mock"]},
        objectives=["accuarcy"],
    )
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=_UnusedEvaluator(metrics=["accuracy"]),
        max_trials=1,
    )
    orchestrator._trials = [
        TrialResult(
            trial_id="trial_1",
            config={"model": "mock"},
            metrics={"accuracy": 0.8, "cost": 0.0},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
            metadata={
                "successful_examples": 2,
                "examples_attempted": 2,
                "comparability": {
                    "schema_version": "1.0",
                    "primary_objective": "accuracy",
                    "evaluation_mode": "evaluated",
                    "total_examples": 2,
                    "examples_with_primary_metric": 2,
                    "coverage_ratio": 1.0,
                    "derivation_path": "explicit",
                    "ranking_eligible": True,
                    "warning_codes": [],
                    "per_metric_coverage": {
                        "accuracy": {"present": 2, "total": 2, "ratio": 1.0}
                    },
                    "missing_example_ids": [],
                },
            },
        )
    ]
    orchestrator._status = OptimizationStatus.COMPLETED

    result = orchestrator._create_optimization_result()

    assert result.status is OptimizationStatus.FAILED
    assert result.reason_code == "NO_RANKING_ELIGIBLE_TRIALS"
    assert result.best_config is None
    assert result.best_score is None
    assert result.success_rate == pytest.approx(0.0)
    assert result.convergence_info["success_rate"] == pytest.approx(0.0)
    assert "OBJECTIVE_UNMATCHED" in result.warning_codes
    assert result.warnings
    warning = result.warnings[0]
    assert "accuarcy" in warning
    assert "accuracy" in warning
    assert "Did you mean 'accuracy'?" in warning
    assert (
        result.metadata["session_summary"]["reason_code"]
        == "NO_RANKING_ELIGIBLE_TRIALS"
    )
    assert (
        "objective metric 'accuarcy' was never measured"
        in result.metadata["session_summary"]["reason"]
    )
