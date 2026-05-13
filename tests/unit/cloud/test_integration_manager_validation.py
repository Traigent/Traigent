"""Validation tests for IntegrationManager identifier handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
async def test_submit_trial_results_returns_false_without_integration(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace(
        submit_privacy_trial_results=AsyncMock(return_value=True)
    )
    manager._active_integrations = {}

    success = await manager.submit_trial_results(
        "session-1", "trial-1", {}, {"accuracy": 0.9}, 1.0
    )

    assert success is False
    manager._backend_client.submit_privacy_trial_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_trial_results_returns_false_for_unsupported_mode(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace(
        submit_privacy_trial_results=AsyncMock(return_value=True)
    )
    manager._active_integrations = {
        "integration-1": {
            "result": IntegrationResult(success=True, session_id="session-1"),
            "mode": "cloud",
        }
    }

    success = await manager.submit_trial_results(
        "session-1", "trial-1", {}, {"accuracy": 0.9}, 1.0
    )

    assert success is False
    manager._backend_client.submit_privacy_trial_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_trial_results_uses_privacy_mode_alias(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace(
        submit_privacy_trial_results=AsyncMock(return_value=True)
    )
    manager._active_integrations = {
        "integration-1": {
            "result": IntegrationResult(
                success=True,
                session_id="session-1",
                metadata={"mode": "privacy"},
            ),
            "mode": "standard",
        }
    }

    success = await manager.submit_trial_results(
        "session-1", "trial-1", {}, {"accuracy": 0.9}, 1.0
    )

    assert success is True
    manager._backend_client.submit_privacy_trial_results.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_trial_results_ignores_mock_metadata(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    result = Mock()
    result.session_id = "session-1"

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace(
        submit_privacy_trial_results=AsyncMock(return_value=True)
    )
    manager._active_integrations = {
        "integration-1": {
            "result": result,
            "mode": "private",
        }
    }

    success = await manager.submit_trial_results(
        "session-1", "trial-1", {}, {"accuracy": 0.9}, 1.0
    )

    assert success is True
    manager._backend_client.submit_privacy_trial_results.assert_awaited_once()


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


@pytest.mark.asyncio
async def test_cancel_session_returns_false_without_backend_cancel(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace()
    manager._active_integrations = {
        "integration-1": {
            "result": IntegrationResult(success=True, session_id="session-1"),
            "mode": "privacy",
        }
    }
    manager._integration_stats["active_sessions"] = 1

    success = await manager.cancel_session("session-1")

    assert success is False
    assert lifecycle_stub.calls == []
    assert "integration-1" in manager._active_integrations
    assert manager._integration_stats["active_sessions"] == 1


@pytest.mark.asyncio
async def test_cancel_session_returns_backend_cancel_result(monkeypatch):
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = SimpleNamespace(
        cancel_session=AsyncMock(return_value=True)
    )
    manager._active_integrations = {}

    success = await manager.cancel_session("session-1")

    assert success is True
    assert lifecycle_stub.calls == [("cancel_session", "session-1")]
    manager._backend_client.cancel_session.assert_awaited_once_with("session-1")


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


@pytest.mark.asyncio
async def test_cancel_session_raises_when_backend_not_initialized(monkeypatch):
    """Test that cancel_session raises RuntimeError when backend client is None."""
    lifecycle_stub = StubLifecycleManager()
    monkeypatch.setattr(integration_module, "lifecycle_manager", lifecycle_stub)

    manager = IntegrationManager()
    manager._initialized = True
    manager._backend_client = None

    with pytest.raises(RuntimeError, match="Backend client not initialized"):
        await manager.cancel_session("session-123")


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
