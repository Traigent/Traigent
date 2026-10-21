"""Review follow-ups for the unmeasured-cost contract (#2517, #2446).

Every path that can carry a trial's cost must keep "unmeasured" distinct from
a measured zero: uploads, the simple-scoring lane, an all-unmeasured tracker,
known judge spend, the mock-mode provenance decision and Pareto eligibility.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission,
)
from traigent.cloud.sync_manager import SyncManager
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.evaluators.base import SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    RESERVED_METRIC_KEYS,
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    TokenMetrics,
    extract_llm_metrics,
)
from traigent.utils.multi_objective import ParetoFrontCalculator


# --------------------------------------------------------------------------- #
# 1. Uploads never invent a cost                                              #
# --------------------------------------------------------------------------- #
def test_budgeted_submission_does_not_backfill_zero_for_unmeasured_cost() -> None:
    client = SimpleNamespace()
    client._cost_budget_armed_sessions = {"s1"}
    metrics: dict[str, Any] = {"accuracy": 1.0, "cost_unmeasured": 1.0}

    mutated = ensure_cost_metric_for_budgeted_completed_submission(
        client=client, session_id="s1", metrics=metrics, status="COMPLETED"
    )

    assert mutated is True
    assert (
        metrics["cost"] == 0.0 and metrics["cost_unmeasured"] == 1.0
    )  # lower bound + flag


def test_budgeted_submission_flags_instead_of_backfilling_zero_when_not_flagged() -> (
    None
):
    """No cost and no flag is an unknown cost, not a free trial (round 7)."""
    client = SimpleNamespace()
    client._cost_budget_armed_sessions = {"s1"}
    metrics: dict[str, Any] = {"accuracy": 1.0}
    ensure_cost_metric_for_budgeted_completed_submission(
        client=client, session_id="s1", metrics=metrics, status="COMPLETED"
    )
    assert (
        metrics["cost"] == 0.0 and metrics["cost_unmeasured"] == 1.0
    )  # lower bound + flag
    assert metrics["cost_unmeasured"] == 1.0


def test_backend_session_manager_does_not_backfill_zero_for_unmeasured_cost() -> None:
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = {"s1"}
    manager._cost_budget_zero_backfill_logged = False
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"accuracy": 1.0, "cost_unmeasured": 1.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    payload: dict[str, Any] = {"accuracy": 1.0, "cost_unmeasured": 1.0}

    manager._ensure_budget_cost_metric(
        session_id="s1", trial_result=trial, metrics_payload=payload, metadata={}
    )

    assert (
        payload["cost"] == 0.0 and payload["cost_unmeasured"] == 1.0
    )  # lower bound + flag


def test_sync_backfill_never_substitutes_score_or_zero_for_unmeasured_cost() -> None:
    runs = [
        {
            "status": "COMPLETED",
            "measures": {"accuracy": 0.9, "score": 0.9, "cost_unmeasured": 1.0},
        }
    ]
    SyncManager._backfill_objective_measures(runs, ["accuracy", "cost"])
    assert "cost" not in runs[0]["measures"]


def test_sync_backfill_still_fills_a_missing_non_cost_objective() -> None:
    runs = [{"status": "COMPLETED", "measures": {"score": 0.4}}]
    SyncManager._backfill_objective_measures(runs, ["accuracy"])
    assert runs[0]["measures"]["accuracy"] == 0.4


# --------------------------------------------------------------------------- #
# 2. Alternate lanes                                                          #
# --------------------------------------------------------------------------- #
def test_simple_scoring_lane_aggregates_unmeasured_cost_as_unmeasured() -> None:
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda *a, **k: 1.0, metrics=["accuracy", "cost"]
    )
    metrics_obj = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=7, output_tokens=2, total_tokens=9),
        cost=CostMetrics(unmeasured=True),
    )
    llm = evaluator._build_llm_metrics_dict(metrics_obj, "gpt-4o-mini")
    row: dict[str, Any] = {}
    evaluator._add_llm_metrics_to_example(row, llm)
    aggregated = evaluator._aggregate_llm_metrics([row, dict(row)], [])

    assert "total_cost" not in aggregated
    assert "input_cost" not in aggregated
    assert aggregated["cost_unmeasured"] == 1.0


def test_simple_scoring_lane_keeps_measured_zero_distinct() -> None:
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda *a, **k: 1.0, metrics=["accuracy", "cost"]
    )
    metrics_obj = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=7, output_tokens=2, total_tokens=9),
        cost=CostMetrics(),
    )
    llm = evaluator._build_llm_metrics_dict(metrics_obj, "free-model")
    row: dict[str, Any] = {}
    evaluator._add_llm_metrics_to_example(row, llm)
    aggregated = evaluator._aggregate_llm_metrics([row], [])

    assert aggregated["total_cost"] == 0.0
    assert aggregated["cost_unmeasured"] == 0.0


def test_tracker_with_only_unextracted_rows_says_unmeasured() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(ExampleMetrics(measured=False))
    tracker.add_example_metrics(ExampleMetrics(measured=False))
    formatted = tracker.format_for_backend()

    assert formatted["cost"] is None
    assert formatted["cost_unmeasured"] == 1.0


def test_empty_tracker_keeps_the_legacy_zero_default() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    assert tracker.format_for_backend()["cost_unmeasured"] == 0.0


# --------------------------------------------------------------------------- #
# 3. Known judge spend is kept as a lower bound                               #
# --------------------------------------------------------------------------- #
class _Usage:
    prompt_tokens = 20
    completion_tokens = 10
    total_tokens = 30


class _JudgeResponse:
    model = "gpt-4o-mini"
    usage = _Usage()


def test_judge_spend_on_an_unmeasured_example_is_kept_as_a_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.evaluators import local as local_module

    monkeypatch.setattr(
        local_module,
        "drain_captured_responses",
        lambda: ([_JudgeResponse()], None, None),
    )
    row = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=7, output_tokens=2, total_tokens=9),
        cost=CostMetrics(unmeasured=True),
    )
    LocalEvaluator(metrics=["cost"])._fold_metric_function_llm_cost(row)

    assert row.cost.unmeasured is True
    assert row.cost.known_cost > 0.0
    assert row.cost.total_cost == 0.0  # the agent's own cost is not invented

    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(row)
    formatted = tracker.format_for_backend()

    assert formatted["cost"] is None
    assert formatted["cost_unmeasured"] == 1.0
    assert formatted["cost_lower_bound"] == pytest.approx(row.cost.known_cost)


def test_lower_bound_includes_measured_rows_in_a_partial_trial() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(
        ExampleMetrics(cost=CostMetrics(input_cost=0.001, output_cost=0.001))
    )
    tracker.add_example_metrics(
        ExampleMetrics(cost=CostMetrics(unmeasured=True, known_cost=0.0005))
    )
    formatted = tracker.format_for_backend()

    assert formatted["cost_unmeasured"] == 1.0
    assert formatted["cost_lower_bound"] == pytest.approx(0.0025)


def test_lower_bound_is_a_reserved_key() -> None:
    assert "cost_lower_bound" in RESERVED_METRIC_KEYS


# --------------------------------------------------------------------------- #
# 4. Mock mode is not provenance                                              #
# --------------------------------------------------------------------------- #
def test_global_mock_flag_alone_does_not_make_a_zero_cost_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    metrics = extract_llm_metrics(response="plain text", model_name="gpt-4o-mini")
    assert metrics.cost.unmeasured is True


# --------------------------------------------------------------------------- #
# Follow-ups                                                                  #
# --------------------------------------------------------------------------- #
def test_pareto_treats_a_none_objective_value_as_missing() -> None:
    calc = ParetoFrontCalculator(maximize={"accuracy": True, "cost": False})
    measured = TrialResult(
        trial_id="m",
        config={},
        metrics={"accuracy": 0.8, "cost": 0.05},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    none_cost = TrialResult(
        trial_id="n",
        config={},
        metrics={"accuracy": 0.9, "cost": None},  # type: ignore[dict-item]
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    front = calc.calculate_pareto_front([measured, none_cost], ["accuracy", "cost"])
    assert [p.trial.trial_id for p in front] == ["m"]


def test_batch_composite_unmeasured_score_is_strict_json_serializable() -> None:
    from traigent.optimizers.registry import get_optimizer

    optimizer = get_optimizer("parallel_batch", {"p": [1, 2]}, ["accuracy", "cost"])
    score = optimizer._calculate_composite_score({"accuracy": 0.9})
    assert math.isfinite(score)
    json.dumps(score, allow_nan=False)
    measured = optimizer._calculate_composite_score({"accuracy": 0.9, "cost": 0.1})
    assert score < measured


# =========================================================================== #
# Round 2 review (cost2): final aggregation, upload order, partial unknown,   #
# eligibility, remaining zero writers                                         #
# =========================================================================== #
_COST_KEYS = ("cost", "input_cost", "output_cost", "total_cost")


def _simple_dataset() -> Any:
    from traigent.evaluators.base import Dataset, EvaluationExample

    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": "a"}, expected_output="a"),
            EvaluationExample(input_data={"q": "b"}, expected_output="b"),
        ],
        name="d",
    )


@pytest.mark.asyncio
async def test_simple_scoring_final_merge_has_no_invented_zero_cost() -> None:
    """Uncaptured responses must not become cost=0.0 / total_cost=0.0."""
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda *a, **k: 1.0, metrics=["accuracy", "cost"]
    )
    result = await evaluator.evaluate(lambda q: q, {}, _simple_dataset())

    for key in _COST_KEYS:
        assert result.aggregated_metrics.get(key) is None, key
    assert result.aggregated_metrics["cost_unmeasured"] == 1.0


def test_simple_scoring_unmeasured_row_drops_partial_custom_cost() -> None:
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda *a, **k: 1.0, metrics=["accuracy", "cost"]
    )
    measured = {"accuracy": 1.0, "cost": 0.5, "total_cost": 0.5}
    unmeasured = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    llm = evaluator._aggregate_llm_metrics([measured, unmeasured], [])

    assert llm["cost_unmeasured"] == 1.0
    for key in ("input_cost", "output_cost", "total_cost"):
        assert key not in llm


@pytest.mark.asyncio
async def test_local_evaluator_error_only_result_has_no_zero_cost() -> None:
    def boom(q: str) -> str:
        raise RuntimeError("boom")

    evaluator = LocalEvaluator(metrics=["accuracy", "cost"], detailed_mode=True)
    result = await evaluator.evaluate(boom, {}, _simple_dataset())

    assert "cost" not in result.aggregated_metrics
    assert result.aggregated_metrics["cost_unmeasured"] == 1.0


def test_compute_cost_is_none_for_partial_unknown_rows() -> None:
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda *a, **k: 1.0, metrics=["cost"]
    )
    rows = [
        ExampleMetrics(cost=CostMetrics()),
        ExampleMetrics(cost=CostMetrics(unmeasured=True)),
    ]
    assert evaluator._compute_cost([], [], [], example_metrics=rows) is None


def _trial_metrics_payload(trial_flag: bool = True) -> dict[str, Any]:
    return {"accuracy": 1.0, "cost_unmeasured": 1.0} if trial_flag else {}


def test_budget_guard_ignores_telemetry_zero_when_flagged() -> None:
    client = SimpleNamespace()
    client._cost_budget_armed_sessions = {"s1"}
    metrics = _trial_metrics_payload()

    mutated = ensure_cost_metric_for_budgeted_completed_submission(
        client=client,
        session_id="s1",
        metrics=metrics,
        status="COMPLETED",
        telemetry_sources=({"total_cost": 0.0}, {"metadata": {"total_cost": 0.0}}),
    )

    assert mutated is True
    assert (
        metrics["cost"] == 0.0 and metrics["cost_unmeasured"] == 1.0
    )  # lower bound + flag


def test_backend_session_manager_ignores_metadata_zero_when_flagged() -> None:
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = {"s1"}
    manager._cost_budget_zero_backfill_logged = False
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"accuracy": 1.0, "cost_unmeasured": 1.0, "total_cost": 0.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={"total_cost": 0.0},
    )
    payload = _trial_metrics_payload()

    manager._ensure_budget_cost_metric(
        session_id="s1",
        trial_result=trial,
        metrics_payload=payload,
        metadata={"total_cost": 0.0, "cost": 0.0},
    )

    assert (
        payload["cost"] == 0.0 and payload["cost_unmeasured"] == 1.0
    )  # lower bound + flag


def test_tracker_measured_free_plus_unknown_row_claims_no_total() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(ExampleMetrics(cost=CostMetrics()))
    tracker.add_example_metrics(ExampleMetrics(cost=CostMetrics(unmeasured=True)))
    formatted = tracker.format_for_backend()

    assert formatted["cost_unmeasured"] == 1.0
    assert formatted["cost"] is None
    assert formatted["cost_per_example_mean"] is None


def test_pareto_ignores_a_cost_number_next_to_the_unmeasured_flag() -> None:
    calc = ParetoFrontCalculator(maximize={"accuracy": True, "cost": False})
    measured = TrialResult(
        trial_id="m",
        config={},
        metrics={"accuracy": 0.8, "cost": 0.05},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    flagged = TrialResult(
        trial_id="f",
        config={},
        metrics={"accuracy": 0.9, "cost": 0.0, "cost_unmeasured": 1.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    front = calc.calculate_pareto_front([measured, flagged], ["accuracy", "cost"])
    assert [p.trial.trial_id for p in front] == ["m"]


def test_batch_composite_flagged_cost_scores_below_measured() -> None:
    from traigent.optimizers.registry import get_optimizer

    optimizer = get_optimizer("parallel_batch", {"p": [1, 2]}, ["accuracy", "cost"])
    flagged = optimizer._calculate_composite_score(
        {"accuracy": 0.9, "cost": 0.0, "cost_unmeasured": 1.0}
    )
    measured = optimizer._calculate_composite_score({"accuracy": 0.5, "cost": 0.1})
    assert flagged < measured
    assert optimizer._calculate_composite_score({}) < measured


def test_multi_objective_rejects_flagged_cost_trial() -> None:
    from traigent.optimizers.batch_optimizers import MultiObjectiveBatchOptimizer

    optimizer = MultiObjectiveBatchOptimizer(
        configuration_space={"p": [1, 2]}, objectives=["accuracy", "cost"]
    )
    assert (
        optimizer._compose_trial_scores(
            {"accuracy": 0.9, "cost": 0.0, "cost_unmeasured": 1.0}
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["adaptive_batch", "parallel_batch"])
async def test_all_unmeasured_trials_yield_no_winner(name: str) -> None:
    from traigent.evaluators.base import EvaluationResult
    from traigent.invokers.base import InvocationResult
    from traigent.optimizers.registry import get_optimizer

    class _Invoker:
        async def invoke(self, func: Any, config: Any, input_data: Any) -> Any:
            return InvocationResult(result="x", is_successful=True, execution_time=0.1)

        async def invoke_batch(self, func: Any, config: Any, batch: Any) -> Any:
            return [await self.invoke(func, config, b) for b in batch]

    class _Evaluator:
        async def evaluate(self, results: Any, expected: Any, dataset: Any) -> Any:
            return EvaluationResult(
                config={},
                total_examples=len(results),
                successful_examples=len(results),
                duration=1.0,
                aggregated_metrics={"accuracy": 0.9, "cost_unmeasured": 1.0},
            )

    optimizer = get_optimizer(name, {"p": [1, 2]}, ["accuracy", "cost"])
    result = await optimizer.optimize(
        lambda **k: "x", _simple_dataset(), _Invoker(), _Evaluator(), max_trials=2
    )

    assert result.best_config == {}
    assert result.successful_trials == 0
    assert all(not t.is_successful for t in result.trials)


def test_batch_trial_flagged_unmeasured_is_not_successful() -> None:
    from traigent.optimizers.results import BatchTrial

    trial = BatchTrial(
        configuration={},
        score=-1.0,
        duration=0.1,
        metadata={"objective_unmeasured": True},
    )
    assert trial.is_successful is False


def test_evaluator_wrapper_does_not_copy_unmeasured_cost_as_zero() -> None:
    from traigent.api.types import ExampleResult
    from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper

    result = ExampleResult(
        example_id="e",
        input_data={},
        expected_output=None,
        actual_output=None,
        metrics={},
        execution_time=0.1,
        success=True,
    )
    metrics_obj = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=7, output_tokens=2, total_tokens=9),
        cost=CostMetrics(unmeasured=True),
    )
    wrapper = CustomEvaluatorWrapper.__new__(CustomEvaluatorWrapper)
    wrapper.capture_llm_metrics = True
    wrapper._metrics_available = True
    wrapper._enhance_result_with_llm_metrics(result, metrics_obj, 0.1)

    for key in ("input_cost", "output_cost", "total_cost"):
        assert key not in result.metrics
    assert result.metrics["cost_unmeasured"] == 1.0


def test_workflow_span_omits_cost_for_unmeasured_trial() -> None:
    from traigent.core.trial_lifecycle import _span_cost
    from traigent.integrations.observability.workflow_traces import SpanPayload

    cost = _span_cost({"cost_unmeasured": 1.0, "total_cost": 0.0}, False)
    assert cost is None
    assert _span_cost({"cost_unmeasured": 1.0}, True) is None
    assert _span_cost({"total_cost": 0.25}, False) == 0.25

    span = SpanPayload(
        span_id="s",
        trace_id="t",
        configuration_run_id="c",
        span_name="n",
        span_type="node",
        start_time="x",
        cost_usd=cost,
    )
    payload = span.to_dict()
    # The wire contract requires a number: the floor (0.0 here) plus the
    # reserved metadata flag, never a bare zero (owner decision A).
    assert payload["cost_usd"] == 0.0
    assert payload["metadata"]["traigent.cost_unmeasured"] is True
