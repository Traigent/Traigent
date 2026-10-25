"""optimization_only: the observability client never reaches the transport."""

from __future__ import annotations

from unittest import mock

import pytest

from traigent.observability import ObservabilityClient, ObservabilityConfig
from traigent.utils import optimization_only_egress as oo


@pytest.fixture(autouse=True)
def _env(monkeypatch, jwt_development_mode):
    del jwt_development_mode
    monkeypatch.delenv("TRAIGENT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_ENV", "development")
    public = ".".join(["93", "184", "216", "34"])
    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        lambda *a, **k: [(None, None, None, None, (public, 443))],
    )
    oo.reset_latch()
    yield
    oo.reset_latch()


def _client() -> ObservabilityClient:
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="https://auth.example.com",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=1,
            max_buffer_age=5.0,
            max_queue_size=10,
        )
    )
    client._http_opener = mock.MagicMock()
    return client


def _run_trace(client: ObservabilityClient) -> None:
    trace_id = client.start_trace("t", trace_id="trace_oo_1")
    client.end_trace(trace_id)
    client.flush()
    client.close()


def test_mode_on_makes_zero_transport_calls(monkeypatch) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    client = _client()
    _run_trace(client)
    assert client._http_opener.open.call_count == 0


def test_mode_on_blocks_post_and_request_paths_directly(monkeypatch) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    client = _client()
    assert client._post_batch_sync([{"trace_id": "x"}]) is None
    with pytest.raises(oo.OptimizationOnlyRefusal):
        client._request_json("GET", "/traces")
    assert client._http_opener.open.call_count == 0
    client.close()


def test_latched_refusal_also_stops_exporter(monkeypatch) -> None:
    client = _client()
    oo.latch_on()
    _run_trace(client)
    assert client._http_opener.open.call_count == 0


def test_mode_off_still_attempts_delivery() -> None:
    client = _client()
    _run_trace(client)
    assert client._http_opener.open.call_count >= 1


def test_custom_local_sender_keeps_working_in_mode(monkeypatch) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    seen: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(backend_origin="https://auth.example.com"),
        sender=seen.append,
    )
    trace_id = client.start_trace("t", trace_id="trace_oo_2")
    client.end_trace(trace_id)
    client.flush()
    client.close()
    assert seen
