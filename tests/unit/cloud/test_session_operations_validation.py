"""Validation and locking tests for SessionOperations."""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.api_operations import TraigentSessionApiResult
from traigent.cloud.auth import AuthenticationError
from traigent.cloud.models import OptimizationSession, OptimizationSessionStatus
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.session_types import SessionCreationFailureReason
from traigent.utils.exceptions import ValidationError as ValidationException


@pytest.fixture(autouse=True)
def _backend_enabled_for_session_operations(monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")


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
        # Connected runs mirror the backend session into local_storage (#1279).
        # Tests that care assign a real LocalStorageManager; default None keeps
        # the connected path a no-op for unrelated tests.
        self.local_storage = None

    async def _ensure_session(self):  # pragma: no cover - not used in validation tests
        return SimpleNamespace(post=AsyncMock(), get=AsyncMock())

    async def _create_traigent_session_via_api(
        self, _request
    ):  # pragma: no cover - overridden in tests
        return ("session-default", "experiment-default", "run-default")

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
    # _finalize_session_via_api now returns dict|None; None = endpoint
    # unavailable (formerly False under the dropped bool contract).
    monkeypatch.setattr(ops, "_finalize_session_via_api", AsyncMock(return_value=None))

    await ops.finalize_session("session-1")
    assert lock.enter_count == 1  # Lock entered exactly once for finalization


def test_create_session_tracks_connected_session_locally(monkeypatch):
    client = FakeClient()
    ops = SessionOperations(client)

    async def create_via_api(_request):
        return TraigentSessionApiResult(
            "session-123",
            "experiment-123",
            "run-123",
            project_id="project-123",
            tenant_id="tenant-123",
        )

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
    assert tracked.metadata["project_id"] == "project-123"
    assert tracked.metadata["tenant_id"] == "tenant-123"
    assert result.project_id == "project-123"
    assert result.tenant_id == "tenant-123"


def test_connected_session_is_mirrored_to_local_storage(monkeypatch, tmp_path):
    """#1279: the connected hybrid path must create a local_storage session
    keyed to the backend session_id, so _persist_trial_locally has a durable
    record on connected runs. Before the fix, local_storage never learned of the
    connected session and add_trial_result raised "session not found".
    """
    from traigent.storage.local_storage import LocalStorageManager

    client = FakeClient()
    client.local_storage = LocalStorageManager(str(tmp_path))
    ops = SessionOperations(client)

    async def create_via_api(_request):
        return ("backend-session-xyz", "experiment-xyz", "run-xyz")

    monkeypatch.setattr(ops.client, "_create_traigent_session_via_api", create_via_api)

    result = ops.create_session(
        "demo-function",
        {"model": ["gpt-4o"]},
        metadata={"max_trials": 3},
    )

    assert result.session_id == "backend-session-xyz"
    # The local mirror exists, keyed to the *backend* session id...
    mirrored = client.local_storage.load_session("backend-session-xyz")
    assert mirrored is not None
    assert mirrored.metadata["execution_mode"] == "connected"
    # ...so a trial keyed to the backend id now persists locally instead of
    # raising TraigentStorageError("Session ... not found").
    client.local_storage.add_trial_result(
        session_id="backend-session-xyz",
        config={"model": "gpt-4o"},
        score=0.9,
    )
    reloaded = client.local_storage.load_session("backend-session-xyz")
    assert reloaded.trials is not None
    assert len(reloaded.trials) == 1


def test_create_session_prunes_completed_sessions_with_aware_timestamps(monkeypatch):
    client = FakeClient()
    client._max_active_sessions = 2
    now = datetime.now(UTC)
    client._active_sessions["completed-oldest"] = OptimizationSession(
        session_id="completed-oldest",
        function_name="demo-completed",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=1,
        status=OptimizationSessionStatus.COMPLETED,
        created_at=now,
        updated_at=now,
    )
    client._active_sessions["active-newer"] = OptimizationSession(
        session_id="active-newer",
        function_name="demo-active",
        configuration_space={},
        objectives=["accuracy"],
        max_trials=1,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
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
    assert "completed-oldest" not in client._active_sessions
    assert "active-newer" in client._active_sessions
    assert "session-123" in client._active_sessions


def test_create_session_auth_failure_logs_warning_not_debug(monkeypatch, caplog):
    """Issue #1373(A): a backend AuthenticationError during the outer
    ``SessionOperations.create_session`` path must surface at WARNING.

    This drives the REAL create-session code path (not just
    ``handle_session_creation_result``): ``_create_traigent_session_via_api``
    raises ``AuthenticationError``, which is caught by the inner handler that
    logs ``"Backend auth failed for ..."`` (session_operations.py ~:573).

    Pre-fix that site was ``logger.debug``, so the failed publish was invisible
    at the default log level. After the fix it is ``logger.warning``.

    Coverage proof: reverting ONLY that line back to ``logger.debug`` makes
    this test FAIL (no WARNING record captured); restoring ``logger.warning``
    makes it PASS.
    """
    client = FakeClient()
    ops = SessionOperations(client)

    async def create_via_api(_request):
        raise AuthenticationError("403 Forbidden: missing_permissions")

    monkeypatch.setattr(ops.client, "_create_traigent_session_via_api", create_via_api)

    with caplog.at_level(logging.WARNING, logger="traigent.cloud.session_operations"):
        # Non-governed call → AuthenticationError is NOT fail-loud; it is caught,
        # logged, and downgraded to a local fallback result.
        result = ops.create_session(
            "demo-function",
            {"model": ["gpt-4o"]},
            metadata={"max_trials": 3, "dataset_size": 4},
        )

    # The call returns a local fallback with reason=AUTH (no exception raised).
    assert result.backend_connected is False
    assert result.failure_reason == SessionCreationFailureReason.AUTH

    # The auth failure MUST be visible at WARNING level (issue #1373(A)).
    auth_warnings = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "Backend auth failed" in record.getMessage()
    ]
    assert auth_warnings, (
        "Expected a WARNING-level 'Backend auth failed' record from the "
        "create_session auth-failure path; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
