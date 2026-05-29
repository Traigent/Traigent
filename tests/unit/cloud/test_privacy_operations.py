"""Concurrency safeguards for PrivacyOperations."""

import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from traigent.cloud.client import (
    CloudRemoteExecutionUnavailableError,
    CloudServiceError,
)
from traigent.cloud.models import OptimizationSession, OptimizationSessionStatus
from traigent.cloud.privacy_operations import PrivacyOperations


class FakeLock:
    """Lock stub recording enter counts."""

    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyClient:
    """Minimal backend client stub for privacy operations tests."""

    def __init__(self, lock: FakeLock) -> None:
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._max_active_sessions = 5
        self._active_sessions_lock = lock
        self.session_bridge = SimpleNamespace(
            create_session_mapping=lambda **kwargs: None,
            get_session_mapping=lambda session_id: SimpleNamespace(
                experiment_run_id="run-1"
            ),
            add_trial_mapping=lambda *args, **kwargs: None,
        )

    async def _check_rate_limit(self) -> None:
        return None

    async def _create_cloud_session(self, request):
        return SimpleNamespace(
            session_id="session-1",
            metadata={"created_at": time.time()},
            optimization_strategy={"mode": "privacy"},
        )

    async def _create_traigent_session_via_api(self, request):
        return ("session-1", "experiment-1", "run-1")

    def _register_security_session(self, *args, **kwargs):
        return None

    async def _get_cloud_trial_suggestion(self, request):
        return SimpleNamespace(suggestion=SimpleNamespace(trial_id="trial-1"))

    async def _submit_cloud_trial_results(self, submission):
        return None

    async def _submit_trial_result_via_session(self, *args, **kwargs):
        return True


@pytest.mark.asyncio
async def test_create_privacy_session_uses_lock(monkeypatch):
    lock = FakeLock()
    client = DummyClient(lock)
    ops = PrivacyOperations(client)

    session_id, _, _ = await ops.create_privacy_optimization_session(
        function_name="test",
        configuration_space={"param": [1, 2]},
        objectives=["accuracy"],
        dataset_metadata={"size": 10},
        max_trials=5,
    )

    assert session_id == "session-1"
    assert lock.enter_count == 1  # Lock entered exactly once for session creation
    assert "session-1" in client._active_sessions


@pytest.mark.asyncio
async def test_get_next_privacy_trial_updates_session_with_lock(monkeypatch):
    lock = FakeLock()
    client = DummyClient(lock)
    ops = PrivacyOperations(client)

    client._active_sessions["session-1"] = OptimizationSession(
        session_id="session-1",
        function_name="test",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    suggestion = await ops.get_next_privacy_trial("session-1", None)
    assert suggestion is not None
    assert lock.enter_count == 1  # Lock entered exactly once for trial retrieval
    # Regression for #889: receiving a suggestion is NOT a completion.
    # completed_trials must NOT advance here. The counter advances only in
    # submit_privacy_trial_results when the result is accepted.
    assert client._active_sessions["session-1"].completed_trials == 0


@pytest.mark.asyncio
async def test_submit_privacy_trial_results_increments_completed_trials():
    """Regression for #889: completed_trials advances exactly once per
    accepted result, not at suggestion time.
    """
    lock = FakeLock()
    client = DummyClient(lock)
    ops = PrivacyOperations(client)

    client._active_sessions["session-1"] = OptimizationSession(
        session_id="session-1",
        function_name="test",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    success = await ops.submit_privacy_trial_results(
        session_id="session-1",
        trial_id="trial-1",
        config={"x": 1},
        metrics={"accuracy": 0.9},
        duration=1.0,
    )

    assert success is True
    assert client._active_sessions["session-1"].completed_trials == 1
    # Submit checks the session under lock, then increments under lock.
    assert lock.enter_count == 2

    # Submit a second result; counter advances to 2 (exactly once per submit).
    success = await ops.submit_privacy_trial_results(
        session_id="session-1",
        trial_id="trial-2",
        config={"x": 2},
        metrics={"accuracy": 0.8},
        duration=1.0,
    )
    assert success is True
    assert client._active_sessions["session-1"].completed_trials == 2
    assert lock.enter_count == 4


@pytest.mark.asyncio
async def test_submit_privacy_trial_results_does_not_increment_on_backend_failure():
    """If the backend submission fails, completed_trials must not advance.
    Closes the symmetric concern to #889 on the submission side.
    """
    lock = FakeLock()
    client = DummyClient(lock)

    # Make backend submission fail.
    async def failing_submit(*args, **kwargs):
        return False

    client._submit_trial_result_via_session = failing_submit  # type: ignore[method-assign]
    ops = PrivacyOperations(client)

    client._active_sessions["session-1"] = OptimizationSession(
        session_id="session-1",
        function_name="test",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    success = await ops.submit_privacy_trial_results(
        session_id="session-1",
        trial_id="trial-1",
        config={"x": 1},
        metrics={"accuracy": 0.9},
        duration=1.0,
    )
    assert success is False
    assert client._active_sessions["session-1"].completed_trials == 0
    assert lock.enter_count == 1


@pytest.mark.asyncio
async def test_get_next_privacy_trial_propagates_cloud_unavailable():
    """Regression for #888: when the cloud remote suggestion service raises
    CloudRemoteExecutionUnavailableError, the public method must re-raise it,
    not swallow into None. Callers MUST be able to distinguish capability
    unavailability from optimization completion.
    """
    lock = FakeLock()
    client = DummyClient(lock)

    async def raise_unavailable(request):
        raise CloudRemoteExecutionUnavailableError("get_cloud_trial_suggestion")

    client._get_cloud_trial_suggestion = raise_unavailable  # type: ignore[method-assign]

    ops = PrivacyOperations(client)
    client._active_sessions["session-1"] = OptimizationSession(
        session_id="session-1",
        function_name="test",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    with pytest.raises(CloudRemoteExecutionUnavailableError):
        await ops.get_next_privacy_trial("session-1", None)


@pytest.mark.asyncio
async def test_get_next_privacy_trial_propagates_cloud_service_error():
    """Companion to the #888 fix: other typed CloudServiceError subclasses
    (auth/transient) also propagate, not silently None.
    """
    lock = FakeLock()
    client = DummyClient(lock)

    async def raise_cse(request):
        raise CloudServiceError("auth failed")

    client._get_cloud_trial_suggestion = raise_cse  # type: ignore[method-assign]

    ops = PrivacyOperations(client)
    client._active_sessions["session-1"] = OptimizationSession(
        session_id="session-1",
        function_name="test",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    with pytest.raises(CloudServiceError):
        await ops.get_next_privacy_trial("session-1", None)


@pytest.mark.asyncio
async def test_get_next_privacy_trial_returns_none_on_no_suggestion():
    """Companion to the #888 fix: when the backend explicitly returns no
    suggestion, the method returns None as before — that's the
    optimization-complete sentinel and distinct from capability failure.
    """
    lock = FakeLock()
    client = DummyClient(lock)

    async def empty_suggestion(request):
        return SimpleNamespace(suggestion=None)

    client._get_cloud_trial_suggestion = empty_suggestion  # type: ignore[method-assign]

    ops = PrivacyOperations(client)
    client._active_sessions["session-1"] = OptimizationSession(
        session_id="session-1",
        function_name="test",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    result = await ops.get_next_privacy_trial("session-1", None)
    assert result is None


@pytest.mark.asyncio
async def test_get_next_privacy_trial_returns_none_on_unexpected_error():
    """Unexpected internal errors keep the legacy None fallback."""
    lock = FakeLock()
    client = DummyClient(lock)

    async def raise_unexpected(request):
        raise RuntimeError("boom")

    client._get_cloud_trial_suggestion = raise_unexpected  # type: ignore[method-assign]

    ops = PrivacyOperations(client)
    client._active_sessions["session-1"] = OptimizationSession(
        session_id="session-1",
        function_name="test",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    result = await ops.get_next_privacy_trial("session-1", None)
    assert result is None
