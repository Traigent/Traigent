"""Validation tests for IntegrationManager identifier handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import traigent.cloud.integration_manager as integration_module
from traigent.cloud.integration_manager import IntegrationManager, IntegrationResult


class StubLifecycleManager:
    """Lifecycle manager stub that records invocations."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_session_state(self, session_id: str):
        self.calls.append(("get_session_state", session_id))
        return {}

    def complete_session(self, session_id: str, results):
        self.calls.append(("complete_session", session_id))

    def cancel_session(self, session_id: str):
        self.calls.append(("cancel_session", session_id))


@pytest.mark.asyncio
async def test_get_next_trial_rejects_blank_session(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace()
    manager._active_integrations = {}

    result = await manager.get_next_trial("", None)
    assert result is None
    assert lifecycle_stub.calls == []


@pytest.mark.asyncio
async def test_submit_trial_results_validates_identifiers(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace(submit_privacy_trial_results=AsyncMock())
    manager._active_integrations = {
        "integration-1": {
            "result": IntegrationResult(success=True, session_id="session-1"),
            "mode": "private",
        }
    }

    success = await manager.submit_trial_results(
        "session-1", "", {}, {"accuracy": 0.9}, 1.0
    )
    assert success is False
    backend_mock: AsyncMock = manager._backend_client.submit_privacy_trial_results
    assert backend_mock.await_count == 0


@pytest.mark.asyncio
async def test_finalize_session_validates_identifiers(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace(finalize_session=AsyncMock())

    success = await manager.finalize_session("", None)
    assert success is False
    assert lifecycle_stub.calls == []


@pytest.mark.asyncio
async def test_cancel_session_validates_identifiers(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace()

    success = await manager.cancel_session("")
    assert success is False
    assert lifecycle_stub.calls == []
