"""Honest cost round 10: flag + lower bound survive summary submission, the
known-spend floor never loses reported spend, and flagged costs neither rank
nor display as measured (Traigent#2517, owner decision A).
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.cloud.trial_operations import TrialOperations
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
)
from traigent.utils.trial_costs import known_spend_lower_bound

try:  # the table helper must be importable on the old code too
    from traigent.utils.results_table import print_results_table
except ImportError:  # pragma: no cover
    print_results_table = None  # type: ignore[assignment]


def _trial(trial_id: str, **metrics: float) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"id": trial_id},
        metrics=dict(metrics),
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _result(trials: list[TrialResult]) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config={"id": "measured"},
        best_score=0.5,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["cost"],
        algorithm="grid",
        timestamp=datetime.now(),
    )


# --- 1. tracker summary -> submit_summary_stats keeps the flag -----------------


async def _submit(summary_stats: dict, budget_armed: bool = False) -> dict:
    client = Mock()
    client.backend_config = Mock()
    client.backend_config.backend_base_url = "https://api.example.com"
    client.backend_config.api_base_url = "https://api.example.com"
    client.auth_manager = AsyncMock()
    client.auth_manager.augment_headers = AsyncMock(return_value={})
    client._map_to_backend_status = Mock(return_value="COMPLETED")
    client.no_egress = False
    client._cost_budget_armed_sessions = {"s1"} if budget_armed else set()
    ops = TrialOperations(client)
    response = Mock()
    response.status = 201
    post_ctx = AsyncMock()
    post_ctx.__aenter__ = AsyncMock(return_value=response)
    post_ctx.__aexit__ = AsyncMock(return_value=False)
    session = AsyncMock()
    session.post = Mock(return_value=post_ctx)
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    ops._handle_trial_success_response = Mock(return_value=True)
    with (
        patch("traigent.cloud.trial_operations.is_backend_offline", return_value=False),
        patch("traigent.cloud.client.is_backend_offline", return_value=False),
        patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.trial_operations.aiohttp") as aio,
    ):
        aio.ClientSession = Mock(return_value=session_ctx)
        aio.ClientTimeout = Mock()
        await ops.submit_summary_stats(
            session_id="s1",
            trial_id="t1",
            config={"model": "m"},
            summary_stats=summary_stats,
            status="completed",
        )
    return session.post.call_args.kwargs["json"]


@pytest.mark.asyncio
async def test_summary_submission_flags_unmeasured_cost_not_zero_means() -> None:
    tracker = MetricsTracker()
    tracker.add_example_metrics(ExampleMetrics(cost=CostMetrics(unmeasured=True)))
    payload = await _submit(tracker.format_as_summary_stats())
    metrics = payload["metrics"]
    assert metrics.get("cost_unmeasured") == 1.0
    for key in ("cost", "total_cost", "input_cost", "output_cost"):
        assert key not in metrics, (key, metrics)


@pytest.mark.asyncio
async def test_summary_submission_sends_lower_bound_with_flag_when_armed() -> None:
    tracker = MetricsTracker()
    tracker.add_example_metrics(
        ExampleMetrics(cost=CostMetrics(input_cost=0.1, output_cost=0.3))
    )
    tracker.add_example_metrics(ExampleMetrics(cost=CostMetrics(unmeasured=True)))
    payload = await _submit(tracker.format_as_summary_stats(), budget_armed=True)
    metrics = payload["metrics"]
    assert metrics["cost_unmeasured"] == 1.0
    assert metrics["cost"] == pytest.approx(0.4)
    assert metrics.get("total_cost") is None


# --- 2. known_spend_lower_bound -----------------------------------------------


def test_metadata_zero_does_not_mask_nested_reported_cost() -> None:
    sources = ({"cost": 0.0}, {"metadata": {"cost": 0.0}, "metrics": {"cost": 0.4}})
    assert known_spend_lower_bound(*sources) == pytest.approx(0.4)


def test_explicit_cost_lower_bound_is_honoured() -> None:
    assert known_spend_lower_bound(
        {"cost": 0.0, "cost_lower_bound": 0.4, "cost_unmeasured": 1.0}
    ) == pytest.approx(0.4)


def test_explicit_lower_bound_outranks_plain_cost_and_nothing_is_zero() -> None:
    assert known_spend_lower_bound({"cost": 0.1}, {"cost_lower_bound": 0.3}) == (
        pytest.approx(0.3)
    )
    assert known_spend_lower_bound({"accuracy": 1.0}) == 0.0


# --- 3. ranking and display ---------------------------------------------------


def test_flagged_zero_cost_does_not_beat_measured_cost_in_weighted_scores() -> None:
    measured = _trial("measured", cost=0.40)
    flagged = _trial("flagged", cost=0.0, cost_unmeasured=1.0)
    out = _result([flagged, measured]).calculate_weighted_scores(
        objective_weights={"cost": 1.0}, minimize_objectives=["cost"]
    )
    assert out["best_weighted_config"] == {"id": "measured"}
    assert [t.trial_id for t, _ in out["weighted_scores"]] == ["measured"]


def _table(trials: list[TrialResult]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_results_table(
            _result(trials), config_space={"id": []}, objectives=["cost"]
        )
    return buf.getvalue()


def test_table_never_renders_flagged_cost_as_dollars_or_best() -> None:
    out = _table(
        [
            _trial("flagged", cost=0.0, cost_unmeasured=1.0),
            _trial("measured", cost=0.40),
        ]
    )
    assert "$0.00000" not in out
    assert "unmeasured" in out
    assert "$0.40000" in out


def test_table_renders_lower_bound_as_floor() -> None:
    out = _table(
        [
            _trial("floor", cost=0.4, cost_unmeasured=1.0, cost_lower_bound=0.4),
            _trial("measured", cost=0.9),
        ]
    )
    assert "≥ $0.40000" in out


# --- 4. terminal best_config selection (result_selection) ----------------------

from traigent.core.result_selection import select_best_configuration  # noqa: E402


def _select(trials, primary="cost", **kwargs):
    return select_best_configuration(
        trials,
        primary_objective=primary,
        config_space_keys=["id"],
        aggregate_configs=False,
        comparability_mode="legacy",
        **kwargs,
    )


def test_terminal_selection_flagged_zero_cost_does_not_win() -> None:
    measured = _trial("measured", cost=0.40)
    flagged = _trial("flagged", cost=0.0, cost_unmeasured=1.0)
    res = _select([flagged, measured])
    assert res.best_trial_id == "measured"


def test_terminal_selection_all_flagged_cost_has_no_winner() -> None:
    res = _select([_trial("a", cost=0.0, cost_unmeasured=1.0)])
    assert res.best_config is None


def test_cost_secondary_tiebreak_ignores_flagged_zero() -> None:
    flagged = _trial("a_flagged", accuracy=0.9, cost=0.0, cost_unmeasured=1.0)
    measured = _trial("b_measured", accuracy=0.9, cost=0.4)
    res = _select(
        [flagged, measured],
        primary="accuracy",
        objective_order=["accuracy", "cost"],
        objective_orientations={"accuracy": "maximize", "cost": "minimize"},
    )
    assert res.best_trial_id == "b_measured"


def test_aggregated_mean_omits_flagged_cost() -> None:
    from traigent.core.result_selection import _aggregate_trials, _compute_mean_metrics

    agg = _aggregate_trials(
        [
            _trial("a", accuracy=0.9, cost=0.0, cost_unmeasured=1.0),
            _trial("b", accuracy=0.9, cost=0.4),
        ],
        {"id"},
    )
    # distinct configs -> two entries; the flagged one carries no cost mean
    means = [_compute_mean_metrics(e) for e in agg.values()]
    assert any("cost" not in m for m in means)
    assert all(m.get("cost") != 0.0 for m in means)


# --- 5. summary_stats cost describe carries no zero means ----------------------


def test_summary_stats_cost_describe_is_null_when_no_cost_sample() -> None:
    tracker = MetricsTracker()
    tracker.add_example_metrics(ExampleMetrics(cost=CostMetrics(unmeasured=True)))
    metrics = tracker.format_as_summary_stats()["metrics"]
    assert metrics["total_cost"] is None
    assert metrics["cost_unmeasured"] == 1.0


# --- round 11 -------------------------------------------------------------------


def test_table_all_flagged_cost_has_no_overall_best() -> None:
    out = _table(
        [
            _trial("a", cost=0.0, cost_unmeasured=1.0),
            _trial("b", cost=0.0, cost_unmeasured=1.0, cost_lower_bound=0.3),
        ]
    )
    assert "★" not in out
    assert "Overall Best" not in out
    assert "No measured winner" in out


def test_known_spend_bound_does_not_lower_larger_reported_cost() -> None:
    assert known_spend_lower_bound({"cost_lower_bound": 0.20, "cost": 0.40}) == 0.40
    assert known_spend_lower_bound({"cost_lower_bound": 0.50, "cost": 0.40}) == 0.50


def test_hybrid_mixed_batch_keeps_reported_floor_with_flag() -> None:
    from traigent.evaluators.hybrid_api import (  # noqa: PLC0415
        HybridAPIEvaluator,
        HybridExampleResult,
    )

    results = [
        HybridExampleResult(example_id="a", cost_usd=0.40, metrics={"accuracy": 1.0}),
        HybridExampleResult(
            example_id="b",
            cost_usd=0.0,
            cost_unmeasured=True,
            metrics={"accuracy": 1.0},
        ),
    ]
    ev = HybridAPIEvaluator.__new__(HybridAPIEvaluator)
    agg, _ = ev._compute_aggregated_metrics_with_comparability(results, 0.40)
    assert agg["cost_unmeasured"] == 1.0
    assert "cost" not in agg and "total_cost" not in agg
    assert agg["cost_lower_bound"] == pytest.approx(0.40)
    assert known_spend_lower_bound(agg) == pytest.approx(0.40)


# --- round 12 -------------------------------------------------------------------


def _table_for(trials, objectives, metadata=None, best_config=None, no_weighted=False):
    result = _result(trials)
    if no_weighted:
        result.calculate_weighted_scores = lambda **_kw: {"weighted_scores": []}  # type: ignore[method-assign]
    if isinstance(objectives, list):
        result.objectives = list(objectives)
    if metadata:
        result.metadata = dict(metadata)
    if best_config is not None:
        result.best_config = best_config
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_results_table(result, config_space={"id": []}, objectives=objectives)
    return buf.getvalue()


def test_multi_objective_accuracy_cost_flagged_trial_is_not_starred() -> None:
    from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("accuracy", "maximize", 1.0),
            ObjectiveDefinition("cost", "minimize", 1.0),
        ]
    )
    out = _table_for(
        [_trial("a", accuracy=0.9, cost=0.0, cost_unmeasured=1.0)],
        schema,
        best_config={"id": "none"},
    )
    assert "★" not in out
    assert "Overall Best" not in out


def test_metadata_best_trial_id_naming_flagged_trial_is_not_starred() -> None:
    out = _table_for(
        [_trial("a", cost=0.0, cost_unmeasured=1.0)],
        ["cost"],
        metadata={"best_trial_id": "a"},
        best_config={"id": "none"},
    )
    assert "★" not in out
    assert "Overall Best" not in out


def test_no_winner_for_other_reason_prints_neutral_line() -> None:
    from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
    from traigent.tvl.models import BandTarget

    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("length", "band", 1.0, band=BandTarget(low=1, high=5)),
            ObjectiveDefinition("cost", "minimize", 1.0),
        ]
    )
    out = _table_for(
        [
            _trial("a", length=3.0, cost=0.0, cost_unmeasured=1.0),
            _trial(
                "b", length=3.0, cost=0.40, cost_lower_bound=0.4, cost_unmeasured=1.0
            ),
            _trial("c", length=3.0, cost=0.40),
        ],
        schema,
        best_config={"id": "none"},
        no_weighted=True,
    )
    assert "★" not in out
    assert "every candidate's cost" not in out
    assert "No overall best could be determined" in out


def test_measured_trial_still_wins_beside_flagged_trial() -> None:
    out = _table_for(
        [
            _trial("flagged", cost=0.0, cost_unmeasured=1.0),
            _trial("measured", cost=0.40),
        ],
        ["cost"],
    )
    assert "★" in out
