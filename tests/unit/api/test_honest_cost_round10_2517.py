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
