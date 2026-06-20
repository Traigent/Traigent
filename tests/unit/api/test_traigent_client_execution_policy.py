"""TraigentClient execution-policy compatibility tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import traigent.traigent_client as client_module
from traigent.config import backend_config
from traigent.config.types import ExecutionIntent, ExecutionMode
from traigent.traigent_client import TraigentClient


class _FakeBackendConfig:
    @staticmethod
    def get_api_key() -> str:
        return "default-key"  # pragma: allowlist secret

    @staticmethod
    def get_backend_url() -> str:
        return "https://backend.example.test"


class _FakeBackendClientConfig:
    def __init__(self, backend_base_url: str) -> None:
        self.backend_base_url = backend_base_url


@pytest.fixture
def backend_client_factory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    factory = Mock()
    monkeypatch.setattr(backend_config, "BackendConfig", _FakeBackendConfig)
    monkeypatch.setattr(client_module, "_CLOUD_AVAILABLE", True)
    monkeypatch.setattr(
        client_module, "BackendClientConfig", _FakeBackendClientConfig, raising=False
    )
    monkeypatch.setattr(
        client_module, "BackendIntegratedClient", factory, raising=False
    )
    return factory


def test_traigent_client_algorithm_auto_uses_shared_cloud_brain_policy(
    backend_client_factory: Mock,
) -> None:
    client = TraigentClient(algorithm="auto")

    assert client.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert client.execution_policy.algorithm == "auto"
    assert client.execution_policy.offline is False
    assert client.execution_mode is ExecutionMode.HYBRID
    backend_client_factory.assert_called_once()


def test_traigent_client_offline_uses_local_only_policy(
    backend_client_factory: Mock,
) -> None:
    client = TraigentClient(offline=True)

    assert client.execution_policy.intent is ExecutionIntent.LOCAL_ONLY
    assert client.execution_policy.offline is True
    assert client.execution_mode is ExecutionMode.EDGE_ANALYTICS
    assert client.backend_client is None
    backend_client_factory.assert_not_called()


def test_traigent_client_deprecated_execution_mode_warns_and_maps_local(
    backend_client_factory: Mock,
) -> None:
    with pytest.warns(DeprecationWarning, match="execution_mode"):
        client = TraigentClient(execution_mode="edge_analytics")

    assert client.execution_policy.intent is ExecutionIntent.LOCAL_ONLY
    assert client.execution_mode is ExecutionMode.EDGE_ANALYTICS
    assert client.backend_client is None
    backend_client_factory.assert_not_called()


def test_traigent_client_deprecated_cloud_warns_and_maps_edge_analytics_mode(
    backend_client_factory: Mock,
) -> None:
    with pytest.warns(DeprecationWarning, match="semantic flip"):
        client = TraigentClient(execution_mode="cloud")

    assert client.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert client.execution_policy.offline is False
    assert client.execution_mode is ExecutionMode.EDGE_ANALYTICS
    assert client.backend_client is None
    backend_client_factory.assert_not_called()


def test_traigent_client_deprecated_execution_mode_auto_maps_edge_analytics(
    backend_client_factory: Mock,
) -> None:
    client = TraigentClient(execution_mode="auto")

    assert client.execution_policy.intent is ExecutionIntent.LOCAL_ONLY
    assert client.execution_policy.offline is True
    assert client.execution_mode is ExecutionMode.EDGE_ANALYTICS
    assert client.backend_client is None
    backend_client_factory.assert_not_called()
