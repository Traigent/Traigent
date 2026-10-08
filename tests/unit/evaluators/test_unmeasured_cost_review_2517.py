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

    assert mutated is False
    assert "cost" not in metrics


def test_budgeted_submission_still_backfills_zero_when_not_flagged() -> None:
    client = SimpleNamespace()
    client._cost_budget_armed_sessions = {"s1"}
    metrics: dict[str, Any] = {"accuracy": 1.0}
    ensure_cost_metric_for_budgeted_completed_submission(
        client=client, session_id="s1", metrics=metrics, status="COMPLETED"
    )
    assert metrics["cost"] == 0.0


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

    assert "cost" not in payload


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
        local_module, "get_all_captured_responses", lambda: [_JudgeResponse()]
    )
    monkeypatch.setattr(local_module, "clear_captured_responses", lambda: None)
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
