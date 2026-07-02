"""Semantic saturation stop-condition rebuild tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft7Validator

from tests.shared.mocks.optimizers import MockOptimizer
from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.stop_conditions import SemanticSaturationStopCondition
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.tvl.models import BandTarget

SCHEMA_FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "contracts"
    / "semantic_saturation_schema.json"
)


class NoopEvaluator(BaseEvaluator):
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
            total_examples=0,
            successful_examples=0,
            duration=0.0,
            metrics=metrics,
        )


class PrivacySentinelExample:
    def __init__(self, example_id: str, metrics: dict[str, Any]) -> None:
        self.example_id = example_id
        self.metrics = metrics

    @property
    def input_data(self) -> str:
        raise AssertionError("semantic saturation read input_data")

    @property
    def expected_output(self) -> str:
        raise AssertionError("semantic saturation read expected_output")

    @property
    def actual_output(self) -> str:
        raise AssertionError("semantic saturation read actual_output")

    @property
    def content(self) -> str:
        raise AssertionError("semantic saturation read content")


def _schema_validator() -> Draft7Validator:
    # Fixture provenance: copied verbatim from TraigentSchema origin/develop:
    # traigent_schema/schemas/optimization/semantic_saturation_schema.json.
    with SCHEMA_FIXTURE.open(encoding="utf-8") as handle:
        return Draft7Validator(json.load(handle))


def _trial(
    trial_id: str,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    example_results: list[Any] | None = None,
) -> TrialResult:
    metadata: dict[str, Any] = {}
    if example_results is not None:
        metadata["example_results"] = example_results
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics or {},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
        metadata=metadata,
    )


def _examples(metric_name: str, values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "example_id": example_id,
            "metrics": {metric_name: value},
            "input_data": "SECRET_INPUT_SHOULD_NOT_BE_READ",
            "expected_output": "SECRET_EXPECTED_SHOULD_NOT_BE_READ",
            "actual_output": "SECRET_ACTUAL_SHOULD_NOT_BE_READ",
        }
        for example_id, value in values.items()
    ]


def _condition(
    config: dict[str, Any] | None = None,
    schema: ObjectiveSchema | None = None,
) -> SemanticSaturationStopCondition:
    defaults = {
        "window": 3,
        "min_trials": 3,
    }
    if config:
        defaults.update(config)
    return SemanticSaturationStopCondition(defaults, objective_schema=schema)


def test_stable_example_vectors_stop():
    condition = _condition()
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 0.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 0.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 0.0}),
        ),
    ]

    assert condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    assert diagnostics["reason_detail"] == "all_objectives_saturated"
    assert diagnostics["objectives"]["accuracy"]["reason_detail"] == (
        "quality_saturated"
    )


def test_boolean_example_scores_saturate():
    condition = _condition()
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"passed": 1.0},
            example_results=_examples("passed", {"a": True, "b": False}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"passed": 1.0},
            example_results=_examples("passed", {"a": True, "b": False}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"passed": 1.0},
            example_results=_examples("passed", {"a": True, "b": False}),
        ),
    ]

    assert condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    assert "passed" in diagnostics["objectives"]


def test_aggregate_flat_accuracy_without_example_scores_does_not_stop():
    condition = _condition()
    trials = [
        _trial("t1", config={"temperature": 0.1}, metrics={"accuracy": 0.8}),
        _trial("t2", config={"temperature": 0.2}, metrics={"accuracy": 0.8}),
        _trial("t3", config={"temperature": 0.3}, metrics={"accuracy": 0.8}),
    ]

    assert not condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    assert diagnostics["reason_detail"] == "insufficient_example_scores"


def test_repeated_config_does_not_satisfy_warmup():
    condition = _condition()
    repeated_config = {"temperature": 0.1}
    trials = [
        _trial(
            "t1",
            config=repeated_config,
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
        _trial(
            "t2",
            config=repeated_config,
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
    ]

    assert not condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    assert diagnostics["reason_detail"] == "warmup"
    assert diagnostics["trials_considered"] == ["t2", "t3"]


def test_cost_improvement_continues_after_quality_saturates():
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("accuracy", "maximize", 0.8),
            ObjectiveDefinition("cost", "minimize", 0.2),
        ]
    )
    condition = _condition({"relative_improvement_epsilon": 0.01}, schema)
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0, "cost": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0, "cost": 0.8},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0, "cost": 0.6},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
    ]

    assert not condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    assert diagnostics["reason_detail"] == "quality_saturated_efficiency_improving"
    assert diagnostics["objectives"]["cost"]["reason_detail"] == (
        "continuous_objective_improving"
    )
    assert diagnostics["objectives"]["cost"]["direction"] == "minimize"


def test_emitted_result_metadata_diagnostics_validate_against_schema():
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition("accuracy", "maximize", 1.0)]
    )
    orchestrator = OptimizationOrchestrator(
        optimizer=MockOptimizer({"temperature": [0.1, 0.2, 0.3]}, ["accuracy"]),
        evaluator=NoopEvaluator(),
        max_trials=10,
        objective_schema=schema,
        semantic_saturation={
            "window": 3,
            "min_trials": 3,
            "continuous_objectives": [],
            "max_example_ids": 2,
        },
    )
    orchestrator._start_time = 1.0
    orchestrator._status = OptimizationStatus.COMPLETED
    orchestrator._trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
    ]

    result = orchestrator._create_optimization_result()
    diagnostics = result.metadata["semantic_saturation"]

    _schema_validator().validate(diagnostics)
    assert diagnostics["condition"] == "semantic_saturation"
    assert diagnostics["decision"] == "stop"


def test_warmup_diagnostics_validate_against_schema():
    condition = _condition()
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
    ]

    assert not condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    _schema_validator().validate(diagnostics)
    assert diagnostics["reason_detail"] == "warmup"


def test_insufficient_overlap_diagnostics_validate_against_schema():
    condition = _condition({"min_overlap": 0.8})
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"b": 1.0, "c": 1.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"b": 1.0, "d": 1.0}),
        ),
    ]

    assert not condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    _schema_validator().validate(diagnostics)
    assert diagnostics["objectives"]["accuracy"]["reason_detail"] == (
        "insufficient_overlap"
    )


def test_churning_diagnostics_validate_against_schema():
    condition = _condition()
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 0.0, "b": 1.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
    ]

    assert not condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    _schema_validator().validate(diagnostics)
    assert diagnostics["reason_detail"] == "accuracy_churning"


def test_disabled_diagnostics_validate_against_schema():
    condition = _condition({"enabled": "false"})

    assert not condition.should_stop([])
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    _schema_validator().validate(diagnostics)
    assert diagnostics["reason_detail"] == "disabled"


def test_unsupported_band_objective_diagnostics_validate_against_schema():
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("accuracy", "maximize", 1.0),
            ObjectiveDefinition(
                "consistency",
                "band",
                1.0,
                band=BandTarget(low=0.8, high=0.9),
            ),
        ]
    )
    condition = _condition({"continuous_objectives": None}, schema)
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0, "consistency": 0.85},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0, "consistency": 0.86},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0, "consistency": 0.84},
            example_results=_examples("accuracy", {"a": 1.0, "b": 1.0}),
        ),
    ]

    assert not condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    _schema_validator().validate(diagnostics)
    assert diagnostics["objectives"]["consistency"]["reason_detail"] == (
        "unsupported_band_objective"
    )


def test_semantic_saturation_config_bool_strings_are_parsed():
    disabled = SemanticSaturationStopCondition({"enabled": "false"})
    assert not disabled.should_stop([])
    assert disabled.diagnostics is not None
    assert disabled.diagnostics["reason_detail"] == "disabled"

    condition = _condition({"include_example_ids": "false"})
    trials = [
        _trial(
            "t1",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
        _trial(
            "t2",
            config={"temperature": 0.2},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
        _trial(
            "t3",
            config={"temperature": 0.3},
            metrics={"accuracy": 1.0},
            example_results=_examples("accuracy", {"a": 1.0}),
        ),
    ]

    assert condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    objective = diagnostics["objectives"]["accuracy"]
    assert "stable_example_ids" not in objective
    assert "changed_example_ids" not in objective


@pytest.mark.parametrize("option", ["enabled", "include_example_ids"])
def test_semantic_saturation_config_bool_strings_reject_unknown_literals(
    option: str,
):
    with pytest.raises(ValueError, match=f"{option} must be a boolean"):
        SemanticSaturationStopCondition({option: "not-a-bool"})


def test_privacy_boundary_reads_only_ids_and_metric_values():
    examples = [PrivacySentinelExample(f"ex{i}", {"accuracy": 1.0}) for i in range(5)]
    condition = _condition({"max_example_ids": 2})
    trials = [
        _trial(
            f"t{index}",
            config={"temperature": index},
            metrics={"accuracy": 1.0},
            example_results=examples,
        )
        for index in range(3)
    ]

    assert condition.should_stop(trials)
    diagnostics = condition.diagnostics
    assert diagnostics is not None
    payload = json.dumps(diagnostics, sort_keys=True)

    assert "SECRET" not in payload
    vector_diag = diagnostics["objectives"]["accuracy"]
    assert len(vector_diag["stable_example_ids"]) == 2
    assert vector_diag["stable_example_ids_truncated_count"] == 3
    _schema_validator().validate(diagnostics)
