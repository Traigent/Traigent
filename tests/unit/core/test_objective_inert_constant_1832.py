"""Regression tests for issue #1832.

A declared, *weighted*, matched objective whose value is uniformly constant
across the ranking-eligible trials cannot affect ``best_config`` — its weight is
a silent no-op (e.g. cost/latency = 0 on a no-LLM-scored or free/unpriceable
model run). This must surface as the ``OBJECTIVE_INERT_CONSTANT`` warning code +
a human-readable warning, mirroring the #1691 unmatched-objective precedent.

These tests reproduce Evidence (a) from the issue: ``accuracy(max)+cost(min)``
offline where ``cost`` binds to ``total_cost`` = 0 uniformly. On the pre-fix code
no warning code is emitted (there is no such code); after the fix the code is
present. The control (cost varies across trials) must NOT emit it, proving real
priced runs are unaffected.
"""

from datetime import UTC, datetime
from typing import Any

from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.orchestrator import (
    _OBJECTIVE_INERT_CONSTANT_WARNING_CODE,
    OptimizationOrchestrator,
    _detect_inert_constant_objectives,
)
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


def _weighted_acc_cost_schema() -> ObjectiveSchema:
    # Non-uniform, meaningful weights over 2 non-banded objectives → weighted
    # selection governs ranking (issue #1682); mirrors Evidence (a) 0.1/0.9.
    return ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(name="accuracy", orientation="maximize", weight=0.1),
            ObjectiveDefinition(name="cost", orientation="minimize", weight=0.9),
        ]
    )


def _eligible_trial(
    trial_id: str, model: str, accuracy: float, cost: float
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"model": model},
        metrics={"accuracy": accuracy, "cost": cost},
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
                    "accuracy": {"present": 2, "total": 2, "ratio": 1.0},
                    "cost": {"present": 2, "total": 2, "ratio": 1.0},
                },
                "missing_example_ids": [],
            },
        },
    )


def _build_orchestrator(trials: list[TrialResult]) -> OptimizationOrchestrator:
    optimizer = _StaticOptimizer(
        {"model": ["gpt-4o", "gpt-3.5-turbo"]},
        objectives=["accuracy", "cost"],
    )
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=_UnusedEvaluator(metrics=["accuracy", "cost"]),
        max_trials=2,
        objective_schema=_weighted_acc_cost_schema(),
    )
    orchestrator._trials = trials
    orchestrator._status = OptimizationStatus.COMPLETED
    return orchestrator


def test_inert_constant_cost_emits_warning_code() -> None:
    # Evidence (a): cost binds to total_cost = 0 uniformly across both configs.
    orchestrator = _build_orchestrator(
        [
            _eligible_trial("trial_1", "gpt-4o", accuracy=0.9, cost=0.0),
            _eligible_trial("trial_2", "gpt-3.5-turbo", accuracy=0.7, cost=0.0),
        ]
    )

    result = orchestrator._create_optimization_result()

    assert _OBJECTIVE_INERT_CONSTANT_WARNING_CODE in result.warning_codes
    assert result.warnings
    warning = next(w for w in result.warnings if "uniformly constant" in w)
    assert "'cost'" in warning
    # Warning-only: selection math unaffected — a real winner is still chosen.
    assert result.best_config is not None
    assert result.status is OptimizationStatus.COMPLETED
    assert result.metadata["session_summary"]["inert_constant_objectives"] == ["cost"]


def test_varying_cost_does_not_emit_warning_code() -> None:
    # Control: cost varies across configs (a real priced run) → weight matters,
    # objective is not inert, no warning.
    orchestrator = _build_orchestrator(
        [
            _eligible_trial("trial_1", "gpt-4o", accuracy=0.9, cost=0.005),
            _eligible_trial("trial_2", "gpt-3.5-turbo", accuracy=0.7, cost=0.001),
        ]
    )

    result = orchestrator._create_optimization_result()

    assert _OBJECTIVE_INERT_CONSTANT_WARNING_CODE not in result.warning_codes


def test_single_trial_run_does_not_emit_warning_code() -> None:
    # A <2 ranking-eligible-trial run is degenerate: every objective is trivially
    # constant when there was no choice. It must NOT warn.
    orchestrator = _build_orchestrator(
        [_eligible_trial("trial_1", "gpt-4o", accuracy=0.9, cost=0.0)]
    )

    result = orchestrator._create_optimization_result()

    assert _OBJECTIVE_INERT_CONSTANT_WARNING_CODE not in result.warning_codes


def test_detect_helper_predicate() -> None:
    schema = _weighted_acc_cost_schema()

    constant = [
        _eligible_trial("t1", "a", accuracy=0.9, cost=0.0),
        _eligible_trial("t2", "b", accuracy=0.7, cost=0.0),
    ]
    assert _detect_inert_constant_objectives(constant, schema) == ["cost"]

    varying = [
        _eligible_trial("t1", "a", accuracy=0.9, cost=0.005),
        _eligible_trial("t2", "b", accuracy=0.7, cost=0.001),
    ]
    assert _detect_inert_constant_objectives(varying, schema) == []

    # Degenerate single-trial set never warns.
    assert _detect_inert_constant_objectives(constant[:1], schema) == []

    # Both objectives constant → both reported inert.
    both_constant = [
        _eligible_trial("t1", "a", accuracy=0.8, cost=0.0),
        _eligible_trial("t2", "b", accuracy=0.8, cost=0.0),
    ]
    assert _detect_inert_constant_objectives(both_constant, schema) == [
        "accuracy",
        "cost",
    ]

    # Float noise within tolerance still reads as constant (robustness).
    noisy = [
        _eligible_trial("t1", "a", accuracy=0.9, cost=1e-15),
        _eligible_trial("t2", "b", accuracy=0.7, cost=0.0),
    ]
    assert _detect_inert_constant_objectives(noisy, schema) == ["cost"]
