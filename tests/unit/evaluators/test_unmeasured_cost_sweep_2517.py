"""Probe matrix for the unmeasured-cost contract (#2517), round 3 sweep.

Each probe drives a PUBLIC path end to end and asserts the same invariant: a
trial whose cost is unknown never carries a cost of 0 (or a partial floor),
carries the ``cost_unmeasured`` flag, is ineligible as a best config and loses
to a fully measured trial. One shared classifier
(``traigent.utils.trial_costs.trial_cost_unknown``) decides every row-based case.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.api.types import ExampleResult, TrialResult, TrialStatus
from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission,
)
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.orchestrator_helpers import extract_cost_from_results
from traigent.evaluators.base import (
    Dataset,
    EvaluationExample,
    EvaluationResult,
    SimpleScoringEvaluator,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.invokers.base import InvocationResult
from traigent.optimizers.batch_optimizers import (
    AdaptiveBatchOptimizer,
    BatchOptimizationConfig,
    ParallelBatchOptimizer,
)
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.multi_objective import ParetoFrontCalculator
from traigent.utils.trial_costs import trial_cost_unknown

COST_KEYS = ("cost", "total_cost", "input_cost", "output_cost")


def _dataset(n: int = 2) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(n)
        ],
        name="sweep",
    )


def _assert_no_cost(metrics: dict[str, Any]) -> None:
    for key in COST_KEYS:
        assert metrics.get(key) is None, f"{key}={metrics.get(key)!r} must be absent"
    assert metrics["cost_unmeasured"] == 1.0


async def _identity(value: int) -> int:
    return value


def _wrapper(rows: list[dict[str, Any]]) -> CustomEvaluatorWrapper:
    async def custom_evaluator(func, config, example):
        i = example.input_data["value"]
        return ExampleResult(
            example_id=f"row-{i}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=i,
            metrics={"accuracy": 1.0, **rows[i]},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    return CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy", "cost"])


# --------------------------------------------------------------------------- #
# The shared classifier                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rows", "successes", "unknown"),
    [
        ([], None, False),  # nothing ran
        ([{"total_cost": 0.25}, {"total_cost": 0.25}], None, False),
        ([{"total_cost": 0.0}], None, False),  # a measured zero stays measured
        ([{"total_cost": 0.25}, {}], None, True),  # successful row, no cost
        ([{}, {}], None, True),  # nothing captured
        ([{"total_cost": 0.25}, {}], [True, False], False),  # failed row skipped
        ([{}, {}], [False, False], True),  # every example errored
        ([{"total_cost": 0.25, "cost_unmeasured": 1.0}], None, True),  # flagged
        ([{"total_cost": float("nan")}], None, True),  # non-finite is not a cost
        ([{"total_cost": True}], None, True),  # bool is not a cost
        ([None, {"total_cost": 0.1}], None, True),  # empty successful row
    ],
)
def test_classifier_matrix(rows, successes, unknown) -> None:
    assert trial_cost_unknown(rows, successes) is unknown


# --------------------------------------------------------------------------- #
# Evaluator paths                                                             #
# --------------------------------------------------------------------------- #
def _boom(value: int) -> int:
    raise RuntimeError("provider failed")


@pytest.mark.asyncio
async def test_simple_scoring_all_error_trial_has_no_cost() -> None:
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda *a, **k: 1.0, metrics=["accuracy", "cost"]
    )
    result = await evaluator.evaluate(_boom, {}, _dataset())
    _assert_no_cost(result.aggregated_metrics)


@pytest.mark.asyncio
async def test_local_evaluator_error_only_trial_has_no_cost() -> None:
    result = await LocalEvaluator(metrics=["accuracy", "cost"]).evaluate(
        _boom, {}, _dataset()
    )
    _assert_no_cost(result.aggregated_metrics)


@pytest.mark.asyncio
async def test_wrapper_partial_row_cost_is_unknown_not_a_floor() -> None:
    result = await _wrapper([{"total_cost": 0.25}, {}]).evaluate(
        _identity, {}, _dataset()
    )
    _assert_no_cost(result.aggregated_metrics)


@pytest.mark.asyncio
async def test_wrapper_all_uncaptured_rows_are_unknown_not_zero() -> None:
    result = await _wrapper([{}, {}]).evaluate(_identity, {}, _dataset())
    _assert_no_cost(result.aggregated_metrics)


@pytest.mark.asyncio
async def test_wrapper_fully_measured_rows_keep_their_cost() -> None:
    result = await _wrapper([{"total_cost": 0.25}, {"total_cost": 0.25}]).evaluate(
        _identity, {}, _dataset()
    )
    assert result.aggregated_metrics["total_cost"] == pytest.approx(0.5)
    assert not result.aggregated_metrics.get("cost_unmeasured")


# --------------------------------------------------------------------------- #
# Upload: budget submission never backfills $0 for an unknown cost            #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_budget_submission_does_not_backfill_zero_for_uncaptured_wrapper_rows() -> (
    None
):
    result = await _wrapper([{}, {}]).evaluate(_identity, {}, _dataset())
    metrics = dict(result.aggregated_metrics)
    client = SimpleNamespace(_cost_budget_armed_sessions={"s1"})
    ensure_cost_metric_for_budgeted_completed_submission(
        client=client, session_id="s1", metrics=metrics, status="COMPLETED"
    )
    assert (
        metrics["cost"] == 0.0 and metrics["cost_unmeasured"] == 1.0
    )  # lower bound + flag

    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = {"s1"}
    manager._cost_budget_zero_backfill_logged = False
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics=dict(result.aggregated_metrics),
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    payload = dict(result.aggregated_metrics)
    manager._ensure_budget_cost_metric(
        session_id="s1", trial_result=trial, metrics_payload=payload, metadata={}
    )
    assert (
        payload["cost"] == 0.0 and payload["cost_unmeasured"] == 1.0
    )  # lower bound + flag


def test_orchestrator_cost_extraction_ignores_floor_and_flagged_trials() -> None:
    def row(metrics: dict[str, Any], success: bool = True) -> ExampleResult:
        return ExampleResult(
            example_id="e",
            input_data={},
            expected_output=None,
            actual_output=None,
            metrics=metrics,
            execution_time=0.0,
            success=success,
            error_message=None,
            metadata={},
        )

    partial = EvaluationResult(
        config={},
        example_results=[row({"total_cost": 0.25}), row({})],
        aggregated_metrics={"accuracy": 1.0},
        total_examples=2,
        successful_examples=2,
        duration=0.0,
    )
    assert extract_cost_from_results(partial, None, "t")[1] is None

    flagged = EvaluationResult(
        config={},
        aggregated_metrics={"cost_unmeasured": 1.0, "total_cost": 0.25},
        total_examples=1,
        successful_examples=1,
        duration=0.0,
    )
    assert extract_cost_from_results(flagged, {"total_cost": 0.25}, "t")[1] is None

    measured = EvaluationResult(
        config={},
        example_results=[row({"total_cost": 0.25}), row({"total_cost": 0.25})],
        aggregated_metrics={},
        total_examples=2,
        successful_examples=2,
        duration=0.0,
    )
    assert extract_cost_from_results(measured, None, "t")[1] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Ranking: batch optimizers and Pareto                                        #
# --------------------------------------------------------------------------- #
class _Invoker:
    def __init__(self) -> None:
        self.last_config: dict[str, Any] = {}

    async def invoke(self, func, config, input_data):
        self.last_config = dict(config)
        return InvocationResult(result="x", is_successful=True, execution_time=0.1)

    async def invoke_batch(self, func, config, input_batch):
        self.last_config = dict(config)
        return [
            InvocationResult(result="x", is_successful=True, execution_time=0.1)
            for _ in input_batch
        ]


class _Evaluator:
    """Config p=1 -> the unknown-cost wrapper result; p=2 -> measured $0.50."""

    def __init__(self, invoker: _Invoker, unknown: EvaluationResult) -> None:
        self._invoker = invoker
        self._unknown = unknown

    async def evaluate(self, invocation_results, expected_outputs, dataset):
        if self._invoker.last_config.get("p") == 1:
            return self._unknown
        return EvaluationResult(
            config={},
            total_examples=len(invocation_results),
            successful_examples=len(invocation_results),
            duration=1.0,
            aggregated_metrics={"accuracy": 0.5, "cost": 0.5, "total_cost": 0.5},
            metrics={"accuracy": 0.5, "cost": 0.5, "total_cost": 0.5},
        )


async def _unknown_wrapper_result(rows: list[dict[str, Any]]) -> EvaluationResult:
    result = await _wrapper(rows).evaluate(_identity, {}, _dataset())
    result.aggregated_metrics["accuracy"] = 0.9
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "optimizer_cls", [ParallelBatchOptimizer, AdaptiveBatchOptimizer]
)
@pytest.mark.parametrize("rows", [[{"total_cost": 0.25}, {}], [{}, {}]])
async def test_batch_optimizers_never_pick_an_unknown_cost_trial(
    optimizer_cls, rows
) -> None:
    unknown = await _unknown_wrapper_result(rows)
    invoker = _Invoker()
    optimizer = optimizer_cls(
        base_optimizer=GridSearchOptimizer({"p": [1, 2]}, ["accuracy", "cost"]),
        batch_config=BatchOptimizationConfig(max_parallel_trials=1, batch_size=2),
    )
    result = await optimizer.optimize(
        func=lambda **k: "x",
        dataset=_dataset(),
        invoker=invoker,
        evaluator=_Evaluator(invoker, unknown),
        max_trials=2,
    )
    assert result.best_config == {"p": 2}


@pytest.mark.asyncio
async def test_batch_with_only_unknown_trials_has_no_winner_and_strict_json_score() -> (
    None
):
    unknown = await _unknown_wrapper_result([{}, {}])

    class _AlwaysUnknown:
        async def evaluate(self, *a, **k):
            return unknown

    optimizer = ParallelBatchOptimizer(
        base_optimizer=GridSearchOptimizer({"p": [1, 2]}, ["accuracy", "cost"]),
        batch_config=BatchOptimizationConfig(max_parallel_trials=1, batch_size=2),
    )
    result = await optimizer.optimize(
        func=lambda **k: "x",
        dataset=_dataset(),
        invoker=_Invoker(),
        evaluator=_AlwaysUnknown(),
        max_trials=2,
    )
    assert result.best_config == {}
    json.dumps(result.best_score, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[{"total_cost": 0.25}, {}], [{}, {}]])
async def test_pareto_does_not_rank_an_unknown_trial_over_a_measured_one(rows) -> None:
    unknown = await _unknown_wrapper_result(rows)
    flagged = TrialResult(
        trial_id="unknown",
        config={},
        metrics=dict(unknown.aggregated_metrics),
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    measured = TrialResult(
        trial_id="measured",
        config={},
        metrics={"accuracy": 0.5, "cost": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    calc = ParetoFrontCalculator(maximize={"accuracy": True, "cost": False})
    front = calc.calculate_pareto_front([flagged, measured], ["accuracy", "cost"])
    assert [p.trial.trial_id for p in front] == ["measured"]


# --------------------------------------------------------------------------- #
# Progress payloads                                                           #
# --------------------------------------------------------------------------- #
def test_local_progress_payload_carries_no_zero_cost_for_unmeasured_example() -> None:
    from traigent.evaluators.metrics_tracker import CostMetrics, ExampleMetrics

    evaluator = LocalEvaluator(metrics=["accuracy", "cost"])
    payload = evaluator._build_local_progress_payload(
        ExampleMetrics(cost=CostMetrics(unmeasured=True)), None, "out"
    )
    metrics = payload["metrics"] if "metrics" in payload else payload
    assert "cost" not in metrics and "total_cost" not in metrics
    assert metrics["cost_unmeasured"] == 1.0
