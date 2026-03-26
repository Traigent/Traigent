"""Validation and locking tests for SessionOperations."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.models import OptimizationSession, OptimizationSessionStatus
from traigent.cloud.session_operations import SessionOperations
from traigent.utils.exceptions import ValidationError as ValidationException


class TrackingLock:
    """Lock stub that records entry count."""

    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeSessionBridge:
    def create_session_mapping(self, **kwargs) -> None:  # pragma: no cover - stub
        return None

    def get_session_mapping(self, session_id: str):
        return SimpleNamespace(experiment_run_id="run-1")

    _session_mappings: dict[str, Any] = {}


class FakeAuth:
    async def get_headers(self) -> dict[str, str]:
        return {}


class FakeAuthManager:
    def __init__(self) -> None:
        self.auth = FakeAuth()

    def has_api_key(self) -> bool:
        return True


class FakeClient:
    def __init__(self, lock: TrackingLock | None = None) -> None:
        self._active_sessions_lock = lock or TrackingLock()
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._max_active_sessions = 5
        self.session_bridge = FakeSessionBridge()
        self.backend_config = SimpleNamespace(api_base_url=None, backend_base_url=None)
        self.auth_manager = FakeAuthManager()
        self._register_security_session = MagicMock()

    async def _ensure_session(self):  # pragma: no cover - not used in validation tests
        return SimpleNamespace(post=AsyncMock(), get=AsyncMock())

    def _register_security_session(
        self, *args, **kwargs
    ) -> None:  # pragma: no cover - stub
        return None

    def _revoke_security_session(
        self, *args, **kwargs
    ) -> None:  # pragma: no cover - stub
        return None


def test_create_session_validates_function_name():
    client = FakeClient()
    ops = SessionOperations(client)

    with pytest.raises(ValidationException):
        ops.create_session("", {"param": [1, 2]})


@pytest.mark.asyncio
async def test_create_hybrid_session_validates_problem_statement():
    client = FakeClient()
    ops = SessionOperations(client)

    with pytest.raises(ValidationException):
        await ops.create_hybrid_session("", {"param": [1]}, {"max_trials": 10})


@pytest.mark.asyncio
async def test_finalize_session_uses_lock(monkeypatch):
    lock = TrackingLock()
    client = FakeClient(lock)
    now = datetime.now(UTC)
    session = OptimizationSession(
        session_id="session-1",
        function_name="demo",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=5,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    client._active_sessions["session-1"] = session

    ops = SessionOperations(client)
    monkeypatch.setattr(ops, "_finalize_session_via_api", AsyncMock(return_value=False))

    await ops.finalize_session("session-1")
    assert lock.enter_count == 1  # Lock entered exactly once for finalization


def test_create_session_tracks_connected_session_locally(monkeypatch):
    client = FakeClient()
    ops = SessionOperations(client)

    async def create_via_api(_request):
        return ("session-123", "experiment-123", "run-123")

    monkeypatch.setattr(ops.client, "_create_traigent_session_via_api", create_via_api)

    result = ops.create_session(
        "demo-function",
        {"model": ["gpt-4o"]},
        metadata={"max_trials": 3, "dataset_size": 4},
    )

    assert result.backend_connected is True
    assert result.session_id == "session-123"
    assert client._register_security_session.called is True
    tracked = client._active_sessions["session-123"]
    assert tracked.max_trials == 3
    assert tracked.metadata["experiment_id"] == "experiment-123"
    assert tracked.metadata["experiment_run_id"] == "run-123"
