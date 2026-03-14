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


# Tests for RuntimeError when clients are not initialized


@pytest.mark.asyncio
async def test_get_next_trial_raises_when_backend_not_initialized(monkeypatch):
    """Test that get_next_trial raises RuntimeError when backend client is None."""
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = None  # Not initialized
    manager._active_integrations = {}

    with pytest.raises(RuntimeError, match="Backend client not initialized"):
        await manager.get_next_trial("session-123", None)


@pytest.mark.asyncio
async def test_submit_trial_results_raises_when_backend_not_initialized(monkeypatch):
    """Test that submit_trial_results raises RuntimeError when backend client is None."""
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = None  # Not initialized
    manager._active_integrations = {
        "integration-1": {
            "result": IntegrationResult(success=True, session_id="session-1"),
            "mode": "private",
        }
    }

    with pytest.raises(RuntimeError, match="Backend client not initialized"):
        await manager.submit_trial_results(
            "session-1", "trial-123", {"param": 1}, {"accuracy": 0.9}, 1.0
        )


@pytest.mark.asyncio
async def test_finalize_session_raises_when_backend_not_initialized(monkeypatch):
    """Test that finalize_session raises RuntimeError when backend client is None."""
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = None  # Not initialized

    with pytest.raises(RuntimeError, match="Backend client not initialized"):
        await manager.finalize_session("session-123", None)


# Tests for RuntimeError when MCP client is not initialized


@pytest.mark.asyncio
async def test_start_privacy_integration_raises_when_mcp_not_initialized(monkeypatch):
    """Test that _start_privacy_integration raises RuntimeError when MCP client is None."""
    from unittest.mock import Mock

    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._mcp_client = None  # Not initialized
    manager._backend_client = SimpleNamespace()  # Backend is initialized

    # Create a mock optimization request
    mock_request = Mock()
    mock_request.objectives = ["accuracy"]
    mock_request.configuration_space = {}

    with pytest.raises(RuntimeError, match="MCP client not initialized"):
        await manager._start_privacy_integration(mock_request, "test-integration-id")


@pytest.mark.asyncio
async def test_start_cloud_integration_raises_when_mcp_not_initialized(monkeypatch):
    """Test that _start_cloud_integration raises RuntimeError when MCP client is None."""
    from unittest.mock import Mock

    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._mcp_client = None  # Not initialized
    manager._backend_client = SimpleNamespace()  # Backend is initialized

    # Create a mock optimization request
    mock_request = Mock()
    mock_request.objectives = ["accuracy"]
    mock_request.configuration_space = {}

    with pytest.raises(RuntimeError, match="MCP client not initialized"):
        await manager._start_cloud_integration(mock_request, "test-integration-id")


def test_get_integration_statistics_returns_snapshot():
    """Returned stats should not expose the mutable internal dict."""
    manager = IntegrationManager()
    manager._integration_stats["total_integrations"] = 1

    stats = manager.get_integration_statistics()
    stats["total_integrations"] = 99

    assert manager._integration_stats["total_integrations"] == 1


def test_get_active_integrations_returns_copy():
    """Returned active integration mapping should be detached from internal state."""
    manager = IntegrationManager()
    manager._active_integrations = {
        "integration-1": {
            "result": IntegrationResult(success=True, session_id="session-1"),
            "mode": "privacy",
        }
    }

    snapshot = manager.get_active_integrations()
    snapshot.clear()

    assert "integration-1" in manager._active_integrations
