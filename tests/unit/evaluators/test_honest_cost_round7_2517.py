"""Round 7 (#2517): five more places that wrote a zero for an unknown cost.

Each test drives the public path of one writer the round-6 review found and
asserts the unknown cost is *absent or flagged* -- never a number:

1. budget submission with neither a cost nor a flag (``cost=0.0`` invented),
2. failed-run summaries for a trial that attempted work with no telemetry,
3. PydanticAI ``usage()`` returning nothing / missing counters,
4. Langfuse costs that are empty, invalid, negative or non-finite,
5. the cost enforcer's required-tracking gate under ``TRAIGENT_MOCK_LLM``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission,
)
from traigent.cloud.trial_operations import TrialOperations
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.cost_enforcement import (
    CostEnforcer,
    CostEnforcerConfig,
    CostTrackingRequiredError,
    Permit,
)
from traigent.core.trial_result_factory import (
    build_failed_result,
    build_pruned_result,
)
from traigent.integrations.langfuse.client import LangfuseClient, LangfuseTraceMetrics
from traigent.integrations.pydantic_ai import handler as pydantic_ai_handler
from traigent.integrations.pydantic_ai.handler import PydanticAIHandler

# SDK #2033: the budget-submission test uses the connected code path.
pytestmark = pytest.mark.backend_online


# --------------------------------------------------------------------------- #
# 1. Budget submission: absent cost AND absent flag is unmeasured              #
# --------------------------------------------------------------------------- #
def _post_mock() -> tuple[Mock, Mock]:
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
    return session, session_ctx


async def _submit_through_public_path(
    metrics: dict[str, Any], *, armed: bool = True
) -> dict[str, Any]:
    """Run ``submit_trial_result_via_session`` and return the posted metrics."""
    client = Mock()
    client.backend_config = Mock()
    client.backend_config.backend_base_url = "https://api.example.com"
    client.backend_config.api_base_url = "https://api.example.com/api/v1"
    client.auth_manager = AsyncMock()
    client.auth_manager.augment_headers = AsyncMock(return_value={})
    client._map_to_backend_status = Mock(return_value="COMPLETED")
    client._sanitize_error_message = Mock(return_value="")
    client._cost_budget_armed_sessions = {"s1"} if armed else set()

    ops = TrialOperations(client)
    ops._handle_trial_success_response = AsyncMock(return_value=True)
    session, session_ctx = _post_mock()
    with (
        patch("traigent.cloud.trial_operations.is_backend_offline", return_value=False),
        patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.trial_operations.validate_configuration_run_submission"),
        patch("traigent.cloud.trial_operations.aiohttp") as aiohttp_mock,
    ):
        aiohttp_mock.ClientSession = Mock(return_value=session_ctx)
        aiohttp_mock.ClientTimeout = Mock()
        outcome = await ops.submit_trial_result_via_session(
            session_id="s1",
            trial_id="t1",
            config={"p": 1},
            metrics=dict(metrics),
            status="completed",
        )
    assert outcome is True
    return dict(session.post.call_args.kwargs["json"]["metrics"])


@pytest.mark.asyncio
async def test_public_submit_without_cost_or_flag_does_not_send_zero() -> None:
    posted = await _submit_through_public_path({"accuracy": 1.0})

    assert "cost" not in posted, posted
    assert posted["cost_unmeasured"] == 1.0


@pytest.mark.asyncio
async def test_public_submit_keeps_a_reported_cost_and_a_measured_zero() -> None:
    reported = await _submit_through_public_path({"accuracy": 1.0, "cost": 0.25})
    assert reported["cost"] == 0.25
    assert "cost_unmeasured" not in reported

    zero = await _submit_through_public_path({"accuracy": 1.0, "cost": 0.0})
    assert zero["cost"] == 0.0
    assert "cost_unmeasured" not in zero


@pytest.mark.asyncio
async def test_public_submit_on_an_unbudgeted_session_is_unchanged() -> None:
    posted = await _submit_through_public_path({"accuracy": 1.0}, armed=False)

    assert "cost" not in posted
    assert "cost_unmeasured" not in posted


def test_budget_guard_uses_telemetry_cost_before_declaring_unmeasured() -> None:
    client = SimpleNamespace(_cost_budget_armed_sessions={"s1"})
    metrics: dict[str, Any] = {"accuracy": 1.0}

    mutated = ensure_cost_metric_for_budgeted_completed_submission(
        client=client,
        session_id="s1",
        metrics=metrics,
        status="COMPLETED",
        telemetry_sources=({"total_cost": 0.4},),
    )

    assert mutated is True
    assert metrics["cost"] == pytest.approx(0.4)
    assert "cost_unmeasured" not in metrics


def test_backend_session_manager_flags_instead_of_backfilling_zero() -> None:
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = {"s1"}
    manager._cost_budget_zero_backfill_logged = False
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    payload: dict[str, Any] = {"accuracy": 1.0}

    manager._ensure_budget_cost_metric(
        session_id="s1", trial_result=trial, metrics_payload=payload, metadata={}
    )

    assert "cost" not in payload
    assert payload["cost_unmeasured"] == 1.0


# --------------------------------------------------------------------------- #
# 2. Failed-run summaries: attempted work with no telemetry is unmeasured       #
# --------------------------------------------------------------------------- #
def _result(trials: list[TrialResult]) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config=None,
        best_score=None,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.FAILED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )


def _failed(progress_state: dict[str, Any] | None, trial_id: str = "f") -> TrialResult:
    return build_failed_result(
        trial_id=trial_id,
        evaluation_config={"p": 1},
        duration=1.0,
        error=RuntimeError("agent crashed"),
        progress_state=progress_state,
        optuna_trial_id=None,
    )


def test_failed_trial_with_attempted_work_and_no_telemetry_is_unmeasured() -> None:
    trial = _failed({"evaluated": 3})
    assert trial.metrics.get("total_cost") is None  # the factory recorded nothing

    result = _result([trial])
    summary = result.get_summary()
    stats = result.experiment_stats

    assert summary["total_cost"] is None
    assert summary["cost_unmeasured_trials"] == 1
    assert stats.total_cost is None
    assert stats.cost_per_configuration is None
    assert stats.cost_unmeasured_trials == 1


def test_pruned_trial_with_attempted_work_and_no_telemetry_is_unmeasured() -> None:
    from traigent.utils.exceptions import TrialPrunedError

    pruned = build_pruned_result(
        trial_id="p",
        evaluation_config={"p": 1},
        duration=1.0,
        prune_error=TrialPrunedError("pruned", step=1),
        progress_state={"evaluated": 2, "total_examples": 4, "correct_sum": 1.0},
        optuna_trial_id=None,
    )
    assert pruned.metrics.get("total_cost") is None

    assert _result([pruned]).get_summary()["total_cost"] is None


def test_failed_trial_with_no_telemetry_is_unmeasured_but_zero_attempts_is_not() -> (
    None
):
    # Missing attempt telemetry is unknown, not zero (round 8).
    assert _result([_failed(None)]).get_summary()["total_cost"] is None

    result = _result([_failed({"evaluated": 0}, trial_id="g")])

    assert result.get_summary()["total_cost"] == 0.0
    assert result.experiment_stats.total_cost == 0.0


def test_failed_trial_with_a_recorded_cost_stays_measured() -> None:
    measured = _failed({"evaluated": 3, "total_cost": 0.5}, trial_id="m")
    zero = _failed({"evaluated": 3, "total_cost": 0.0}, trial_id="z")

    summary = _result([measured, zero]).get_summary()

    assert summary["total_cost"] == pytest.approx(0.5)
    assert "cost_unmeasured_trials" not in summary


def test_unmeasured_failed_trial_withholds_the_run_total_but_keeps_the_floor() -> None:
    measured = _failed({"evaluated": 3, "total_cost": 0.5}, trial_id="m")
    unknown = _failed({"evaluated": 3}, trial_id="u")

    result = _result([measured, unknown])
    summary = result.get_summary()

    assert summary["total_cost"] is None
    assert summary["cost_lower_bound"] == pytest.approx(0.5)
    assert result.experiment_stats.cost_lower_bound == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 3. PydanticAI: missing usage is unmeasured, not zero tokens                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _pydantic_ai_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pydantic_ai_handler, "PYDANTICAI_AVAILABLE", True)


def _handler() -> PydanticAIHandler:
    agent = MagicMock()
    agent.model = "openai:gpt-4o"
    return PydanticAIHandler(agent)


def _record(handler: PydanticAIHandler, usage: Any) -> dict[str, float | int]:
    result = MagicMock()
    result.usage.return_value = usage
    handler._record_usage(result, 0.0)
    return handler.get_measures_dict()


@pytest.mark.parametrize(
    "usage",
    [
        None,
        SimpleNamespace(),  # no counters at all
        SimpleNamespace(requests=1, total_tokens=15),  # no input/output counters
        SimpleNamespace(input_tokens=10, requests=1),  # one counter missing
        SimpleNamespace(input_tokens=None, output_tokens=5),
        SimpleNamespace(input_tokens="ten", output_tokens=5),
        SimpleNamespace(input_tokens=float("nan"), output_tokens=5),
        SimpleNamespace(input_tokens=-1, output_tokens=5),
    ],
)
def test_missing_usage_counters_export_unmeasured_not_zero(usage: Any) -> None:
    measures = _record(_handler(), usage)

    assert measures["pydantic_ai_cost_unmeasured"] == 1.0
    assert "pydantic_ai_total_cost" not in measures


def test_reported_zero_tokens_is_a_measured_zero() -> None:
    measures = _record(
        _handler(),
        SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0, requests=1),
    )

    assert measures["pydantic_ai_total_cost"] == 0.0
    assert "pydantic_ai_cost_unmeasured" not in measures


def test_reported_tokens_are_priced() -> None:
    measures = _record(
        _handler(),
        SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150),
    )

    assert measures["pydantic_ai_total_cost"] > 0.0
    assert "pydantic_ai_cost_unmeasured" not in measures


# --------------------------------------------------------------------------- #
# 4. Langfuse: only a finite non-negative number is a reported cost            #
# --------------------------------------------------------------------------- #
def _generation(**fields: Any) -> dict[str, Any]:
    return {"id": "o", "name": "gen", "type": "GENERATION", "model": "m", **fields}


@pytest.mark.parametrize(
    "fields",
    [
        {"totalCost": ""},
        {"totalCost": "   "},
        {"totalCost": "not-a-number"},
        {"totalCost": "-0.5"},
        {"calculatedTotalCost": -0.5},
        {"calculatedTotalCost": float("nan")},
        {"calculatedTotalCost": float("inf")},
        {"totalCost": "NaN"},
        {"totalCost": "Infinity"},
        {"calculatedTotalCost": True},
        {"calculatedTotalCost": None, "totalCost": ""},
    ],
)
def test_langfuse_invalid_cost_is_unmeasured(fields: dict[str, Any]) -> None:
    client = LangfuseClient.__new__(LangfuseClient)

    observation = client._dict_to_observation(_generation(**fields))
    assert observation.cost_reported is False

    metrics = client._extract_metrics_from_trace(
        {"id": "t", "observations": [_generation(**fields)]}
    )
    assert isinstance(metrics, LangfuseTraceMetrics)
    assert metrics.cost_unmeasured is True
    measures = metrics.to_measures_dict(prefix="lf_")
    assert "lf_total_cost" not in measures
    assert measures["lf_cost_unmeasured"] == 1.0


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"calculatedTotalCost": 0.0}, 0.0),
        ({"calculatedTotalCost": 0}, 0.0),
        ({"totalCost": "0"}, 0.0),
        ({"totalCost": "0.0125"}, 0.0125),
        ({"calculatedTotalCost": 0.25}, 0.25),
        ({"calculatedTotalCost": 0.0, "totalCost": "0.5"}, 0.5),
    ],
)
def test_langfuse_valid_cost_is_reported(
    fields: dict[str, Any], expected: float
) -> None:
    client = LangfuseClient.__new__(LangfuseClient)

    observation = client._dict_to_observation(_generation(**fields))

    assert observation.cost_reported is True
    assert math.isclose(observation.cost, expected)
    metrics = client._extract_metrics_from_trace(
        {"id": "t", "observations": [_generation(**fields)]}
    )
    assert metrics.cost_unmeasured is False
    assert metrics.to_measures_dict()["total_cost"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 5. Required cost tracking is not bypassed by the mock-LLM flag               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("trial_failed", [False, True])
def test_unknown_cost_raises_under_mock_llm_when_tracking_is_required(
    monkeypatch: pytest.MonkeyPatch, trial_failed: bool
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "true")
    enforcer = CostEnforcer(CostEnforcerConfig(fallback_trial_limit=5))

    with pytest.raises(CostTrackingRequiredError):
        enforcer.track_cost(
            None,
            permit=Permit(id=0, amount=0.0, active=True),
            trial_failed=trial_failed,
        )


@pytest.mark.asyncio
async def test_async_unknown_cost_raises_under_mock_llm_when_tracking_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "true")
    enforcer = CostEnforcer(CostEnforcerConfig(fallback_trial_limit=5))

    with pytest.raises(CostTrackingRequiredError):
        await enforcer.track_cost_async(
            None, permit=Permit(id=0, amount=0.0, active=True)
        )


def test_unknown_cost_without_required_tracking_still_degrades_under_mock_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "false")
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    enforcer = CostEnforcer(CostEnforcerConfig(fallback_trial_limit=5))

    enforcer.track_cost(None, permit=Permit(id=0, amount=0.0, active=True))

    assert enforcer.get_status().unknown_cost_mode is True
