"""Execution-layer tests for consolidated execution policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import traigent
from traigent.api.decorators import (
    ExternalServiceEvaluator,
    HybridAPIOptions,
    optimize,
)
from traigent.cloud.models import (
    DatasetSubsetIndices,
    NextTrialResponse,
    TrialSuggestion,
)
from traigent.core.session_types import (
    SessionCreationFailureDetail,
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.hybrid.protocol import (
    HybridEvaluateResponse,
    HybridExecuteResponse,
    ServiceCapabilities,
)
from traigent.utils.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _isolate_execution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)


def _dataset() -> Dataset:
    return Dataset([EvaluationExample({"text": "q"}, "YES")], name="policy")


def _scorer(prediction: str, expected: str) -> float:
    return 1.0 if prediction == expected else 0.0


def _make_agent(call_recorder: Mock | None = None, **optimize_kwargs):
    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1]},
        scoring_function=_scorer,
        **optimize_kwargs,
    )
    def agent(text: str) -> str:
        if call_recorder is not None:
            call_recorder(text)
        return "YES"

    return agent


class FakeBackendClient:
    def __init__(
        self,
        session_result: SessionCreationResult | None = None,
        *,
        has_api_key: bool = True,
    ) -> None:
        self.create_session = Mock(
            return_value=session_result or SessionCreationResult.connected("sess-1")
        )
        self.get_session_mapping = Mock(
            return_value=SimpleNamespace(
                experiment_id="exp-1",
                experiment_run_id="run-1",
            )
        )
        self.upload_example_features = Mock(return_value=True)
        self.submit_result = Mock()
        self.request_trial_slot = AsyncMock(return_value="slot-should-not-be-used")
        self._submit_trial_result_via_session = AsyncMock(return_value=True)
        self.update_trial_weighted_scores = AsyncMock(return_value=True)
        self.finalize_session_sync = Mock(return_value={"status": "completed"})
        self.auth_manager = SimpleNamespace(has_api_key=lambda: has_api_key)
        self.auth = self.auth_manager
        self.close = AsyncMock()


def _next_trial_response() -> NextTrialResponse:
    return NextTrialResponse(
        suggestion=TrialSuggestion(
            trial_id="cloud-trial-1",
            session_id="sess-1",
            trial_number=1,
            config={"temperature": 0.1},
            dataset_subset=DatasetSubsetIndices(
                indices=[0],
                selection_strategy="all",
                confidence_level=1.0,
                estimated_representativeness=1.0,
            ),
            exploration_type="exploration",
        ),
        should_continue=True,
    )


@pytest.mark.asyncio
async def test_offline_env_makes_zero_backend_calls_even_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    agent = _make_agent()

    with (
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client"
        ) as create_backend_client,
        patch("traigent.cloud.client.TraigentCloudClient.get_next_trial") as next_trial,
    ):
        result = await agent.optimize(max_trials=1)

    create_backend_client.assert_not_called()
    next_trial.assert_not_called()
    assert result.source == "offline"
    assert result.metadata["source"] == "offline"


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_mode", ["edge_analytics", "local"])
async def test_legacy_local_modes_make_zero_backend_calls(
    monkeypatch: pytest.MonkeyPatch,
    legacy_mode: str,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    agent = _make_agent(execution_mode=legacy_mode)

    with (
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client"
        ) as create_backend_client,
        patch("traigent.cloud.client.TraigentCloudClient.get_next_trial") as next_trial,
    ):
        result = await agent.optimize(max_trials=1)

    create_backend_client.assert_not_called()
    next_trial.assert_not_called()
    assert result.source == "offline"


@pytest.mark.asyncio
async def test_cloud_brain_default_uses_next_trial_when_backend_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    backend = FakeBackendClient()
    agent = _make_agent()

    with (
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client",
            return_value=backend,
        ),
        patch(
            "traigent.cloud.client.TraigentCloudClient.get_next_trial",
            new=AsyncMock(return_value=_next_trial_response()),
        ) as next_trial,
    ):
        result = await agent.optimize(max_trials=1)

    assert next_trial.await_count == 1
    backend.request_trial_slot.assert_not_called()
    backend._submit_trial_result_via_session.assert_awaited_once()
    assert result.source == "cloud_brain"
    assert result.metadata["source"] == "cloud_brain"


@pytest.mark.parametrize(
    ("failure_reason", "detail", "has_api_key"),
    [
        (
            SessionCreationFailureReason.NO_API_KEY,
            "No API key configured",
            False,
        ),
        (
            SessionCreationFailureReason.SESSION_FAILED,
            "Backend unavailable (connection failed)",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_cloud_brain_session_create_fallback_runs_real_local_trials(
    monkeypatch: pytest.MonkeyPatch,
    failure_reason: SessionCreationFailureReason,
    detail: str,
    has_api_key: bool,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    if has_api_key:
        monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    else:
        monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    failure = SessionCreationResult.fallback(
        session_id="local-fallback",
        reason=failure_reason,
        detail=detail,
        failure_response=SessionCreationFailureDetail(
            message=detail,
        ),
    )
    backend = FakeBackendClient(session_result=failure, has_api_key=has_api_key)
    calls = Mock()
    agent = _make_agent(call_recorder=calls)

    with (
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client",
            return_value=backend,
        ) as create_backend_client,
        patch(
            "traigent.cloud.client.TraigentCloudClient.get_next_trial",
            new=AsyncMock(return_value=_next_trial_response()),
        ) as next_trial,
    ):
        result = await agent.optimize(max_trials=1)

    assert create_backend_client.call_count == 1
    next_trial.assert_not_called()
    assert result.source == "local_fallback"
    assert result.metadata["source"] == "local_fallback"
    assert "fallback_reason" in result.metadata
    assert len(result.trials) >= 1
    assert result.best_score == pytest.approx(1.0)
    assert calls.call_count >= 1


@pytest.mark.asyncio
async def test_cloud_brain_runtime_degradation_stops_remote_next_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    backend = FakeBackendClient()
    backend._submit_trial_result_via_session = AsyncMock(return_value=False)
    calls = Mock()
    agent = _make_agent(call_recorder=calls)

    with (
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client",
            return_value=backend,
        ),
        patch(
            "traigent.cloud.client.TraigentCloudClient.get_next_trial",
            new=AsyncMock(return_value=_next_trial_response()),
        ) as next_trial,
    ):
        result = await agent.optimize(max_trials=2)

    assert next_trial.await_count == 1
    backend._submit_trial_result_via_session.assert_awaited_once()
    assert result.source == "local_fallback"
    assert result.metadata["source"] == "local_fallback"
    assert result.metadata["fallback_reason"] == "trial submission"
    assert len(result.trials) >= 1
    assert result.best_score == pytest.approx(1.0)
    assert calls.call_count >= 2


@pytest.mark.asyncio
async def test_explicit_smart_backend_down_hard_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    failure = SessionCreationResult.fallback(
        session_id="local-fallback",
        reason=SessionCreationFailureReason.SESSION_FAILED,
        detail="Backend unavailable (connection failed)",
    )
    backend = FakeBackendClient(session_result=failure)
    agent = _make_agent(algorithm="optuna")

    with patch(
        "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client",
        return_value=backend,
    ):
        with pytest.raises(ConfigurationError, match="Cloud execution is required"):
            await agent.optimize(max_trials=1)


@pytest.mark.asyncio
async def test_explicit_grid_runs_local_without_next_trial_or_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    agent = _make_agent(algorithm="grid")

    with (
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client"
        ) as create_backend_client,
        patch("traigent.cloud.client.TraigentCloudClient.get_next_trial") as next_trial,
    ):
        result = await agent.optimize(max_trials=1)

    create_backend_client.assert_not_called()
    next_trial.assert_not_called()
    assert result.source == "explicit_local"


class FakeHybridTransport:
    def __init__(self) -> None:
        self.execute = AsyncMock(side_effect=self._execute)
        self.evaluate = AsyncMock(side_effect=self._evaluate)
        self.capabilities = AsyncMock(
            return_value=ServiceCapabilities(
                version="1.0",
                supports_evaluate=True,
                supports_keep_alive=False,
            )
        )
        self.close = AsyncMock()

    async def _execute(self, request):
        return HybridExecuteResponse(
            request_id=request.request_id,
            execution_id="exec-1",
            status="completed",
            outputs=[
                {
                    "example_id": item["example_id"],
                    "output": "YES",
                }
                for item in request.examples
            ],
            operational_metrics={"total_cost_usd": 0.0},
        )

    async def _evaluate(self, request):
        return HybridEvaluateResponse(
            request_id=request.request_id,
            status="completed",
            results=[
                {"example_id": row["example_id"], "metrics": {"accuracy": 1.0}}
                for row in request.evaluations
            ],
            aggregate_metrics={"accuracy": {"mean": 1.0, "n": 1}},
            execution_id=request.execution_id,
        )


@pytest.mark.asyncio
async def test_hybrid_api_options_dispatch_through_external_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")
    transport = FakeHybridTransport()

    @optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1]},
        evaluator=ExternalServiceEvaluator(
            hybrid_api=HybridAPIOptions(transport=transport, keep_alive=False)
        ),
    )
    def external_agent(text: str) -> str:
        return "not-called"

    with (
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client"
        ) as create_backend_client,
        patch("traigent.cloud.client.TraigentCloudClient.get_next_trial") as next_trial,
    ):
        result = await external_agent.optimize(max_trials=1)

    create_backend_client.assert_not_called()
    next_trial.assert_not_called()
    transport.execute.assert_awaited_once()
    transport.evaluate.assert_awaited_once()
    assert result.source == "explicit_local"
    assert result.best_score == pytest.approx(1.0)
