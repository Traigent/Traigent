"""Concurrency safeguards for PrivacyOperations."""

import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

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
    assert lock.enter_count >= 1
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
    assert lock.enter_count >= 1
    assert client._active_sessions["session-1"].completed_trials == 1
