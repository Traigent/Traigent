"""#2997: TRAIGENT_PRIVATE_DATA / TRAIGENT_COMPLIANCE_MODE stop ALL backend egress."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from traigent.utils.env_config import (
    is_backend_offline,
    raise_if_backend_offline,
)
from traigent.utils.error_handler import OfflineModeError

_NAMES = ("TRAIGENT_PRIVATE_DATA", "TRAIGENT_COMPLIANCE_MODE")
_OFFLINE = ("TRAIGENT_OFFLINE_MODE", "TRAIGENT_OFFLINE", "TRAIGENT_DISABLE_TELEMETRY")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (*_NAMES, *_OFFLINE):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(params=_NAMES)
def private_env(monkeypatch, request):
    monkeypatch.setenv(request.param, "1")
    return request.param


def test_is_backend_offline_follows_private_mode(private_env):
    assert is_backend_offline() is True
    with pytest.raises(OfflineModeError):
        raise_if_backend_offline("x")


def test_unparseable_value_counts_as_offline(monkeypatch):
    monkeypatch.setenv("TRAIGENT_PRIVATE_DATA", "banana")
    assert is_backend_offline() is True


def test_off_value_is_not_offline(monkeypatch):
    monkeypatch.setenv("TRAIGENT_PRIVATE_DATA", "false")
    assert is_backend_offline() is False


def test_cloud_client_gate_blocks(private_env):
    from traigent.cloud.client import (
        CloudEgressBlockedError,
        cloud_backend_egress_disabled,
        raise_if_cloud_egress_disabled,
    )

    assert cloud_backend_egress_disabled() is True
    with pytest.raises(CloudEgressBlockedError):
        raise_if_cloud_egress_disabled("op")


def test_execution_policy_runtime_blocks(private_env):
    from traigent.core.execution_policy_runtime import backend_egress_disabled

    assert backend_egress_disabled(None) is True


def test_economics_client_sends_nothing_even_with_credentials(private_env, monkeypatch):
    from traigent.economics import client as client_mod

    created: list[Any] = []

    class _Fake:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _Fake)
    c = client_mod.EconomicsTelemetryClient(
        backend_url="https://api.traigent.ai", api_key="k"
    )
    with pytest.raises(OfflineModeError):
        asyncio.run(c._post_with_retry(b"{}", {}))
    assert created == []


def test_analytics_client_blocks(private_env):
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    c = BackendAnalyticsClient(backend_url="https://api.traigent.ai", api_key="k")
    with pytest.raises(OfflineModeError):
        c._get_client()


def test_langfuse_client_blocks(private_env):
    from traigent.integrations.langfuse.client import LangfuseClient

    with pytest.raises(OfflineModeError):
        LangfuseClient._raise_if_offline()


def test_label_transport_sends_nothing(private_env, monkeypatch):
    from traigent.labels.transport import LabelTransport

    async def _headers() -> dict[str, str]:
        raise AssertionError("credentials must not be looked up")

    t = LabelTransport("https://api.traigent.ai", _headers)
    with pytest.raises(OfflineModeError):
        asyncio.run(t._request("GET", "https://api.traigent.ai/labels/x", None, 1.0))


def _obs_config():
    from traigent.observability.config import ObservabilityConfig

    return ObservabilityConfig(
        backend_origin="http://localhost:5000",
        api_key="test-key",  # pragma: allowlist secret
        batch_size=10,
        max_buffer_age=999.0,
        max_queue_size=10,
        enable_atexit_flush=False,
    )


def test_observability_client_sends_nothing_even_with_credentials(monkeypatch):
    from traigent.observability import client as obs_mod
    from traigent.observability.client import ClientError, ObservabilityClient

    def _no_network(*args: Any, **kwargs: Any):
        raise AssertionError("network egress attempted in private mode")

    monkeypatch.setattr(obs_mod.request, "urlopen", _no_network)
    client = ObservabilityClient(_obs_config())
    # Env set AFTER construction: the live check must still block.
    monkeypatch.setenv("TRAIGENT_PRIVATE_DATA", "1")
    monkeypatch.setattr(client, "_open_http_request", _no_network, raising=False)
    try:
        assert client._post_batch_sync([{"id": "t"}]) is None
        with pytest.raises(ClientError):
            client.list_sessions()
        assert client._ingest_suppressed is True
    finally:
        client.close()


def test_observability_config_built_in_private_mode_is_offline(private_env):
    assert _obs_config().offline_mode is True
