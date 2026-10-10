"""Required-cloud runs must stop on an unacknowledged tracking write."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.execution_policy_runtime import CloudBrainUnavailableError
from traigent.core.objectives import create_default_objectives


@pytest.fixture
def connected_manager(monkeypatch):
    for name in ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_REQUIRE_CLOUD", "1")
    client = Mock()
    client.auth_manager = Mock(spec=["has_api_key"])
    client.auth_manager.has_api_key.return_value = True
    client.get_session_mapping.return_value = SimpleNamespace(
        experiment_id="exp-local-control", experiment_run_id="run-local-control"
    )
    client.request_trial_slot = AsyncMock(return_value="slot-local-control")
    client._submit_trial_result_via_session = AsyncMock(return_value=True)
    config = TraigentConfig()
    config.execution_mode = "hybrid"
    optimizer = Mock()
    optimizer.config_space = {"temperature": [0.1]}
    manager = BackendSessionManager(
        backend_client=client,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(objective_names=["accuracy"]),
        optimizer=optimizer,
        optimization_id="opt-local-control",
        optimization_status=OptimizationStatus.RUNNING,
    )
    manager._no_egress = False
    manager._backend_tracking_enabled = True
    assert manager._egress_disabled() is False
    return manager, client


def trial():
    result = Mock(spec=TrialResult)
    result.trial_id = "client-local-control"
    result.config = {"temperature": 0.1}
    result.metrics = {"accuracy": 1.0}
    result.status = TrialStatus.COMPLETED
    result.error_message = None
    result.metadata = {}
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, None, ConnectionError("local outage")])
async def test_required_cloud_rejects_failed_submission(connected_manager, failure):
    manager, client = connected_manager
    if isinstance(failure, Exception):
        client._submit_trial_result_via_session.side_effect = failure
    else:
        client._submit_trial_result_via_session.return_value = failure
    with pytest.raises(CloudBrainUnavailableError):
        await manager._log_trial_to_backend("session-local-control", trial(), 1.0, {})
    assert manager._runtime_degraded is False
    assert manager._acknowledged_trials == set()


@pytest.mark.asyncio
async def test_required_cloud_accepts_acknowledged_submission(connected_manager):
    manager, _ = connected_manager
    await manager._log_trial_to_backend("session-local-control", trial(), 1.0, {})
    assert manager._acknowledged_trials == {
        ("session-local-control", "slot-local-control")
    }


@pytest.mark.asyncio
async def test_optional_cloud_preserves_local_fallback(connected_manager, monkeypatch):
    manager, client = connected_manager
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD")
    client._submit_trial_result_via_session.return_value = False
    await manager._log_trial_to_backend("session-local-control", trial(), 1.0, {})
    assert manager._runtime_degraded is True


@pytest.mark.asyncio
async def test_public_required_cloud_stops_before_next_agent_call(monkeypatch):
    from unittest.mock import patch

    import traigent
    from tests.unit.core.test_execution_policy_execution import FakeBackendClient
    from traigent.evaluators.base import Dataset, EvaluationExample

    monkeypatch.setenv("TRAIGENT_REQUIRE_CLOUD", "1")
    for name in ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.delenv(name, raising=False)
    backend = FakeBackendClient()
    backend.request_trial_slot.side_effect = ["first-slot", "second-slot", "third-slot"]
    backend._submit_trial_result_via_session.side_effect = [True, False, True]
    calls = []

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")]),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.2, 0.3]},
        scoring_function=lambda prediction, expected: float(prediction == expected),
        algorithm="grid",
    )
    def agent(text):
        calls.append((text, traigent.get_config()["temperature"]))
        return "YES"

    with patch(
        "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client",
        return_value=backend,
    ):
        with pytest.raises(CloudBrainUnavailableError):
            await agent.optimize(max_trials=3)
    assert len(calls) == 2
    assert backend._submit_trial_result_via_session.await_count == 2
    backend.finalize_session_sync.assert_called_once()
    assert backend.finalize_session_sync.call_args.args[1] is False
    assert backend.finalize_session_sync.call_args.kwargs["stop_reason"] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["credential", "mapping"])
async def test_required_cloud_rejects_missing_tracking_prerequisite(
    connected_manager, missing
):
    manager, client = connected_manager
    if missing == "credential":
        client.auth_manager.has_api_key.return_value = False
    else:
        client.get_session_mapping.return_value = None
    with pytest.raises(CloudBrainUnavailableError):
        await manager._log_trial_to_backend("session-local-control", trial(), 1.0, {})
    client._submit_trial_result_via_session.assert_not_called()
    assert manager._runtime_degraded is False


def test_rejection_diagnostic_does_not_promise_local_fallback(caplog):
    from traigent.cloud.trial_operations import TrialOperations

    operations = TrialOperations(SimpleNamespace())
    with caplog.at_level("ERROR"):
        result = operations._handle_trial_error_response(
            401,
            "trial-local-control",
            "session-local-control",
            "http://127.0.0.1/local-control",
            '{"message": "invalid local control"}',
        )
    assert result.permanent_rejection is True
    assert "LOCALLY ONLY" not in caplog.text
    assert "local_fallback" not in caplog.text
    assert "PERMANENT" in caplog.text
