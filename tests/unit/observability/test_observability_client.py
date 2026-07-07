from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import UTC, datetime
from urllib import error

import pytest

from traigent.cloud.async_batch_transport import BatchFlushResult
from traigent.config.context import ConfigurationContext, TrialContext
from traigent.observability import (
    FlushResult,
    ObservabilityClient,
    ObservabilityConfig,
    ObservationType,
    ObserveContext,
    PromptReferenceDTO,
    ThumbRating,
    observe,
)
from traigent.observability.client import _SyncBatchTransport
from traigent.observability.decorators import set_default_observability_client
from traigent.observability.dtos import ObservationDTO, TraceDTO
from traigent.utils.exceptions import AuthenticationError, ClientError


def _encoded_trace_batch_size(traces: list[dict]) -> int:
    return len(json.dumps({"traces": traces}).encode("utf-8"))


FAKE_TRACE_API_KEY = (
    "sk-ant-canary-DO-NOT-USE-123456789abcdef"  # pragma: allowlist secret
)


@pytest.fixture(autouse=True)
def _clear_observability_offline_mode(monkeypatch, jwt_development_mode):
    del jwt_development_mode
    monkeypatch.delenv("TRAIGENT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", raising=False)
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_CONTENT", raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_ENV", "development")


def _mock_public_backend_dns(monkeypatch):
    public_addr = ".".join(["93", "184", "216", "34"])
    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, (public_addr, 443))],
    )


def test_observability_config_uses_canonical_environment_resolution(monkeypatch):
    """Trace metadata defaults should not ignore the legacy SDK env alias."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("TRAIGENT_ENVIRONMENT", raising=False)
    monkeypatch.setenv("TRAIGENT_ENV", "staging")
    _mock_public_backend_dns(monkeypatch)

    config = ObservabilityConfig(backend_origin="https://auth.example.com")

    assert config.default_environment == "staging"


def test_observability_config_does_not_emit_unknown_environment_content(
    monkeypatch, caplog
):
    """Unknown environment aliases should not become trace metadata."""
    sentinel = "alice@example.com"
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("TRAIGENT_ENVIRONMENT", raising=False)
    monkeypatch.setenv("TRAIGENT_ENV", sentinel)
    _mock_public_backend_dns(monkeypatch)
    caplog.set_level(logging.WARNING)

    config = ObservabilityConfig(backend_origin="https://auth.example.com")

    assert config.default_environment is None
    assert sentinel not in json.dumps(config.__dict__)
    assert "Ignoring unknown environment label" in caplog.text
    assert sentinel not in caplog.text


def test_observability_config_defaults_to_metadata_content_mode(monkeypatch):
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_CONTENT", raising=False)
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", raising=False)
    _mock_public_backend_dns(monkeypatch)

    config = ObservabilityConfig(backend_origin="https://auth.example.com")

    assert config.content_mode == "metadata"


def test_observability_config_uses_disable_telemetry_as_offline_mode(monkeypatch):
    monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "true")
    _mock_public_backend_dns(monkeypatch)

    config = ObservabilityConfig(backend_origin="https://auth.example.com")

    assert config.offline_mode is True


@pytest.mark.parametrize(
    ("origin", "message"),
    [
        ("http://api.traigent.example", "https"),
        ("metadata-ip", "private or loopback"),
    ],
)
def test_observability_config_rejects_unsafe_production_origins(
    monkeypatch, origin, message
):
    monkeypatch.setenv("TRAIGENT_ENV", "production")
    if origin == "metadata-ip":
        origin = f"https://{'.'.join(['169', '254', '169', '254'])}"

    with pytest.raises(ValueError, match=message):
        ObservabilityConfig(backend_origin=origin)


def test_observability_client_uses_no_redirect_http_opener():
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
            enable_atexit_flush=False,
        )
    )

    handler_names = {type(handler).__name__ for handler in client._http_opener.handlers}
    client.close()

    assert "_NoRedirectHandler" in handler_names


def test_observability_client_flushes_trace_payloads():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=2,
            max_buffer_age=5.0,
            max_queue_size=10,
        ),
        sender=sender,
    )

    trace_id = client.start_trace(
        "customer-support",
        trace_id="trace_test_001",
        session_id="session_test_001",
        user_id="user_001",
        tags=["demo"],
        metadata={"source": "unit-test"},
        prompt_reference=PromptReferenceDTO(
            name="support/welcome",
            version=2,
            label="latest",
            variables={"customer_name": "Ada"},
        ),
    )
    root_observation_id = client.record_observation(
        trace_id,
        observation_id="obs_root",
        name="root-span",
        observation_type=ObservationType.SPAN,
        input_tokens=10,
        output_tokens=5,
    )
    client.record_observation(
        trace_id,
        observation_id="obs_child",
        parent_observation_id=root_observation_id,
        name="llm-call",
        observation_type=ObservationType.GENERATION,
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.0025,
        model_name="gpt-4.1-mini",
        metadata={"provider": "openai"},
        prompt_reference={
            "name": "support/welcome",
            "version": 1,
            "label": "production",
            "variables": {"customer_name": "Ada"},
        },
    )
    client.end_trace(trace_id, output_data={"answer": "hello"})

    result = client.flush()
    client.close()

    assert result.success is True
    assert result.items_sent >= 1
    assert sent_batches

    sent_trace = sent_batches[-1][-1]
    assert sent_trace["id"] == "trace_test_001"
    assert sent_trace["session_id"] == "session_test_001"
    assert sent_trace["prompt_reference"]["name"] == "support/welcome"
    assert sent_trace["observations"][0]["id"] == "obs_root"
    assert sent_trace["observations"][0]["children"][0]["id"] == "obs_child"
    assert sent_trace["observations"][0]["children"][0]["type"] == "generation"
    assert (
        sent_trace["observations"][0]["children"][0]["prompt_reference"]["version"] == 1
    )


def test_observability_client_offline_mode_does_not_attempt_ingest(monkeypatch, caplog):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    urlopen_calls = {"count": 0}

    def fake_urlopen(*args, **kwargs):
        urlopen_calls["count"] += 1
        raise AssertionError("network attempted")

    monkeypatch.setattr(
        "traigent.observability.client.request.urlopen",
        fake_urlopen,
    )

    caplog.set_level(logging.INFO, logger="traigent.observability.client")
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=1,
            max_buffer_age=0.1,
            max_queue_size=10,
            enable_atexit_flush=False,
        )
    )

    trace_id = client.start_trace("offline-probe", trace_id="trace_offline")
    client.record_observation(trace_id, name="offline-observation")
    client.end_trace(trace_id)

    result = client.flush()
    assert client._post_batch_sync([{"id": "trace_manual"}]) is None
    close_result = client.close()

    assert client.config.offline_mode is True
    assert result.success is True
    assert result.items_sent == 0
    assert result.items_pending == 0
    assert result.items_dropped == 0
    assert result.successful_batches == 0
    assert result.failed_batches == 0
    assert close_result.success is True
    assert close_result.items_sent == 0
    assert urlopen_calls["count"] == 0
    assert caplog.text.count("Observability transport in offline mode") == 1


def test_observability_client_offline_mode_memory_sender_has_no_backend_egress(
    monkeypatch,
):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    sentinel = "offline-memory-canary-local-only"
    sent_batches: list[list[dict]] = []
    backend_attempts: list[tuple[str, str]] = []

    def fake_urlopen(http_request, *args, **kwargs):
        body = getattr(http_request, "data", b"")
        body_text = body.decode("utf-8") if isinstance(body, bytes) else str(body)
        backend_attempts.append(("urlopen", body_text))
        raise AssertionError("network attempted")

    def memory_sender(traces):
        sent_batches.append(traces)

    def request_sender(method: str, path: str, payload: dict | None):
        backend_attempts.append(("request_sender", json.dumps([method, path, payload])))
        raise AssertionError("request sender should not be called in offline mode")

    monkeypatch.setattr(
        "traigent.observability.client.request.urlopen",
        fake_urlopen,
    )
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
            enable_atexit_flush=False,
        ),
        sender=memory_sender,
        request_sender=request_sender,
    )

    trace_id = client.start_trace(
        "offline-memory-canary",
        trace_id="trace_offline_memory_canary",
        metadata={"sentinel": sentinel},
        input_data={"sentinel": sentinel},
    )
    client.record_observation(
        trace_id,
        name="offline-memory-observation",
        input_data={"sentinel": sentinel},
    )
    client.end_trace(trace_id, output_data={"sentinel": sentinel})

    result = client.flush()
    with pytest.raises(ClientError, match="TRAIGENT_OFFLINE_MODE=true"):
        client.list_sessions()
    close_result = client.close()

    memory_payload = json.dumps(sent_batches)
    egress_payload = json.dumps(backend_attempts)
    assert client.config.offline_mode is True
    assert result.success is True
    assert result.items_sent >= 1
    assert close_result.success is True
    assert sentinel in memory_payload
    assert backend_attempts == []
    assert sentinel not in egress_payload


def test_observability_client_offline_mode_blocks_request_api():
    request_calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        request_calls.append((method, path, payload))
        raise AssertionError("request sender should not be called in offline mode")

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
            enable_atexit_flush=False,
            offline_mode=True,
        ),
        request_sender=request_sender,
    )

    with pytest.raises(ClientError, match="TRAIGENT_OFFLINE_MODE=true"):
        client.list_sessions()

    client.close()

    assert request_calls == []


def test_observability_client_tracks_dropped_payloads_when_buffer_is_full():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=1,
        ),
        sender=sender,
    )

    trace_a = client.start_trace("trace-a", trace_id="trace_a")
    trace_b = client.start_trace("trace-b", trace_id="trace_b")
    client.end_trace(trace_a)
    client.end_trace(trace_b)

    stats = client.get_stats()
    result = client.close()

    assert stats["dropped_items"] >= 1
    assert result.items_dropped >= 1


def test_observability_client_logs_trace_snapshot_submit_failure(caplog):
    """Transport rejections must be visible instead of silently dropping traces."""

    class RejectingTransport:
        def submit(self, trace_id, payload):
            assert trace_id == "trace_rejected"
            assert payload["id"] == "trace_rejected"
            return False

        def get_stats(self):
            return {"errors": ["queue full for api-secret_123456789012345"]}

        def close(self):
            return BatchFlushResult(
                success=True,
                items_sent=0,
                items_pending=0,
                items_dropped=0,
                successful_batches=0,
                failed_batches=0,
                errors=[],
                warnings=[],
            )

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            max_queue_size=1,
        )
    )
    client._transport = RejectingTransport()

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    trace_id = client.start_trace("trace-rejected", trace_id="trace_rejected")

    client._queue_trace_snapshot(trace_id)
    client.close()

    assert "trace_rejected" in caplog.text
    assert "queue full" in caplog.text
    assert "api-secret_123456789012345" not in caplog.text
    assert "[REDACTED:api_key]" in caplog.text


def test_observability_client_redacts_trace_payloads_before_submit():
    """Trace payloads must be scrubbed before they reach the transport."""
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=1,
            max_buffer_age=999.0,
            max_queue_size=10,
        ),
        sender=sender,
    )

    trace_id = client.start_trace(
        "trace-redaction",
        trace_id="trace_redaction",
        user_id="alice@example.com",
        metadata={"token": FAKE_TRACE_API_KEY},
        input_data={"ssn": "123-45-6789"},
    )
    client.record_observation(
        trace_id,
        name="llm-call",
        observation_type=ObservationType.GENERATION,
        input_data={"prompt": "card 4111111111111111"},
        output_data={"answer": "Bearer canary.jwt.header.payload.signature"},
        metadata={"email": "alice@example.com"},
    )
    client.end_trace(
        trace_id,
        output_data={"answer": FAKE_TRACE_API_KEY},
    )

    client.flush()
    client.close()

    payload_blob = str(sent_batches)
    assert "alice@example.com" not in payload_blob
    assert "123-45-6789" not in payload_blob
    assert "4111111111111111" not in payload_blob
    assert FAKE_TRACE_API_KEY not in payload_blob
    assert "canary.jwt.header.payload.signature" not in payload_blob
    assert "[REDACTED:email]" in payload_blob
    assert "[REDACTED:ssn]" in payload_blob
    assert "[REDACTED:credit_card]" in payload_blob
    assert "[REDACTED:api_key]" in payload_blob
    assert "[REDACTED:bearer_token]" in payload_blob


def test_sync_batch_transport_redacts_direct_submitted_payloads():
    """Direct transport submissions must be scrubbed before buffering and sending."""
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
    )

    accepted = transport.submit(
        "trace_direct",
        {
            "id": "trace_direct",
            "user_id": "alice@example.com",
            "metadata": {"api_key": FAKE_TRACE_API_KEY},
        },
    )

    result = transport.flush()
    transport.close()

    payload_blob = str(sent_batches)
    assert accepted is True
    assert result.success is True
    assert "alice@example.com" not in payload_blob
    assert FAKE_TRACE_API_KEY not in payload_blob
    assert "[REDACTED:email]" in payload_blob
    # The ``api_key`` metadata value is now masked by the credential KEY-NAME
    # rule (strictly stronger than the prior value-scan ``[REDACTED:api_key]``
    # tag): any value under a credential-like key is fully masked, so a
    # regex-evading low-entropy secret can no longer ride along.
    assert "'api_key': '[REDACTED]'" in payload_blob


def test_observability_client_close_flushes_active_trace_payloads_without_explicit_flush():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
        ),
        sender=sender,
    )

    trace_id = client.start_trace("close-flush", trace_id="trace_close_flush")
    client.record_observation(trace_id, name="close-observation")

    result = client.close()

    assert result.success is True
    assert result.items_sent == 1
    assert sent_batches[-1][-1]["id"] == "trace_close_flush"


def test_observability_client_close_waits_for_inflight_snapshot_submission(monkeypatch):
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
        ),
        sender=sender,
    )
    trace_id = client.start_trace("close-race", trace_id="trace_close_race")

    submit_entered = threading.Event()
    release_submit = threading.Event()
    original_submit = client._submit_trace_snapshot

    def delayed_submit(trace_id: str, payload: dict) -> None:
        submit_entered.set()
        assert release_submit.wait(timeout=2.0)
        original_submit(trace_id, payload)

    monkeypatch.setattr(client, "_submit_trace_snapshot", delayed_submit)

    record_error: list[BaseException] = []

    def record_observation() -> None:
        try:
            client.record_observation(trace_id, name="close-race-observation")
        except BaseException as exc:
            record_error.append(exc)

    record_thread = threading.Thread(target=record_observation)
    record_thread.start()
    assert submit_entered.wait(timeout=2.0)

    close_result: list[FlushResult] = []
    close_thread = threading.Thread(target=lambda: close_result.append(client.close()))
    close_thread.start()

    close_thread.join(timeout=0.05)
    assert close_thread.is_alive()
    release_submit.set()

    record_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)

    assert not record_thread.is_alive()
    assert not close_thread.is_alive()
    assert record_error == []
    assert close_result
    assert close_result[0].items_dropped == 0
    assert all(
        "transport closed; dropped payload" not in error
        for error in close_result[0].errors
    )
    assert sent_batches[-1][-1]["id"] == "trace_close_race"


def test_sync_batch_transport_records_closed_transport_drops():
    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=1024,
    )

    transport.close()
    accepted = transport.submit("trace_after_close", {"id": "trace_after_close"})

    assert accepted is False
    stats = transport.get_stats()
    assert (
        "transport closed; dropped payload for item 'trace_after_close'"
        in stats["errors"]
    )


def test_sync_batch_transport_stats_snapshot_includes_locked_diagnostics():
    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=1024,
    )

    transport._append_error("error-1")
    transport._append_warning("warning-1")

    stats = transport.get_stats()
    result = transport.close()

    assert stats["errors"] == ["error-1"]
    assert stats["warnings"] == ["warning-1"]
    assert result.errors == ["error-1"]
    assert result.warnings == ["warning-1"]


def test_sync_batch_transport_batch_size_flush_does_not_block_submit():
    send_entered = threading.Event()
    release_send = threading.Event()
    sent_batches: list[list[dict]] = []

    def sender(traces):
        send_entered.set()
        assert release_send.wait(timeout=2.0)
        sent_batches.append(traces)

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=2,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
    )

    assert transport.submit("trace_one", {"id": "trace_one"}) is True
    started = time.monotonic()
    assert transport.submit("trace_two", {"id": "trace_two"}) is True
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert send_entered.wait(timeout=2.0)
    stats = transport.get_stats()
    assert stats["send_in_progress"] is True
    assert stats["inflight_items"] == 2

    release_send.set()
    result = transport.flush()
    transport.close()

    assert result.success is True
    assert sent_batches == [[{"id": "trace_one"}, {"id": "trace_two"}]]


def test_sync_batch_transport_keeps_age_timer_armed_on_batch_size_path():
    """The batch-size path must leave an age timer armed as a backstop.

    Regression guard: previously the batch-size path cancelled the age timer and
    relied solely on the daemon flush thread. If that thread was between its
    buffer-empty check and exit when the next item arrived, ``is_alive()`` still
    read True so no new thread spawned and, with the timer cancelled, the tail
    item was stranded until the next submit/close. The timer must stay armed.
    """
    release_send = threading.Event()

    def sender(traces):
        # Hold the flush thread inside _send_available so it cannot reach
        # flush() (which would cancel the timer) before we inspect timer state.
        assert release_send.wait(timeout=2.0)

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=1,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
    )
    try:
        assert transport.submit("trace_one", {"id": "trace_one"}) is True
        # Crossing batch_size dispatched a flush thread; an age timer must still
        # be armed as the backstop rather than cancelled.
        assert transport._timer is not None
        assert transport._timer.is_alive()
    finally:
        release_send.set()
        transport.close()


def test_sync_batch_transport_emits_health_event_for_queue_full():
    events: list[tuple[str, dict]] = []
    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=10_000,
        health_callback=lambda event_type, payload: events.append(
            (event_type, payload)
        ),
    )

    assert transport.submit("trace_one", {"id": "trace_one"}) is True
    assert transport.submit("trace_two", {"id": "trace_two"}) is False
    result = transport.close()

    assert result.items_dropped == 1
    assert events == [
        (
            "queue_full",
            {
                "message": "transport queue full; dropped payload for item 'trace_two'",
                "item_id": "trace_two",
            },
        )
    ]


def test_observability_client_chunks_flushes_by_byte_limit():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    payloads = [
        {"id": f"trace_{index}", "input_data": {"prompt": "x" * 64}}
        for index in range(3)
    ]
    max_batch_bytes = _encoded_trace_batch_size(payloads[:2])
    assert _encoded_trace_batch_size(payloads[:3]) > max_batch_bytes

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=max_batch_bytes,
    )

    for payload in payloads:
        transport.submit(payload["id"], payload)

    result = transport.flush()
    transport.close()

    assert result.success is True
    assert [len(batch) for batch in sent_batches] == [2, 1]
    assert result.items_pending == 0
    assert result.successful_batches == 2


def test_observability_client_drops_single_payload_over_byte_limit():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    payload = {"id": "trace_oversized", "input_data": {"prompt": "x" * 128}}
    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=_encoded_trace_batch_size([payload]) - 1,
    )

    assert transport.submit(payload["id"], payload) is False

    result = transport.flush()
    transport.close()

    assert sent_batches == []
    assert result.items_sent == 0
    assert result.items_dropped == 1
    assert result.items_pending == 0
    assert "exceeding max_batch_bytes" in result.errors[0]


def test_observability_client_drops_non_json_payload_before_send():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
    )

    assert (
        transport.submit("trace_bad", {"id": "trace_bad", "when": datetime.now()})
        is False
    )
    assert transport.submit("trace_good", {"id": "trace_good", "name": "ok"}) is True

    result = transport.flush()
    transport.close()

    assert sent_batches == [[{"id": "trace_good", "name": "ok"}]]
    assert result.items_sent == 1
    assert result.items_dropped == 1
    assert result.items_pending == 0
    assert "not JSON serializable" in result.errors[0]


def test_observability_client_preserves_existing_usage_fields_on_update():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    trace_id = client.start_trace("usage-preservation", trace_id="trace_usage")
    client.record_observation(
        trace_id,
        observation_id="obs_usage",
        name="llm-call",
        input_tokens=12,
        output_tokens=4,
        cost_usd=0.25,
    )
    client.record_observation(
        trace_id,
        observation_id="obs_usage",
        name="llm-call",
        status="completed",
        input_tokens=None,
        output_tokens=None,
        cost_usd=None,
    )
    client.end_trace(trace_id)

    result = client.flush()
    client.close()

    assert result.success is True
    observation = sent_batches[-1][-1]["observations"][0]
    assert observation["input_tokens"] == 12
    assert observation["output_tokens"] == 4
    assert observation["cost_usd"] == 0.25


def test_observability_client_omits_unknown_generation_usage_fields(caplog):
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    trace_id = client.start_trace("usage-unknown", trace_id="trace_usage_unknown")
    with caplog.at_level(logging.WARNING):
        client.record_observation(
            trace_id,
            observation_id="obs_usage_unknown",
            name="llm-call",
            observation_type=ObservationType.GENERATION,
            status="completed",
        )
        client.record_observation(
            trace_id,
            observation_id="obs_usage_unknown_2",
            name="llm-call-2",
            observation_type=ObservationType.GENERATION,
            status="completed",
        )
    client.end_trace(trace_id)

    result = client.flush()
    client.close()

    assert result.success is True
    observation = sent_batches[-1][-1]["observations"][0]
    assert "input_tokens" not in observation
    assert "output_tokens" not in observation
    assert "total_tokens" not in observation
    assert "cost_usd" not in observation
    assert caplog.text.count("usage will be reported as unknown") == 1


def test_observe_decorator_excludes_sdk_setup_from_latency(monkeypatch):
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )
    setup_started_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    application_started_at = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
    application_ended_at = datetime(2026, 1, 1, 0, 0, 1, 7_000, tzinfo=UTC)
    utc_now_values = iter(
        [setup_started_at, application_started_at, application_ended_at]
    )
    monkeypatch.setattr(
        "traigent.observability.decorators.utc_now", lambda: next(utc_now_values)
    )

    @observe("tight-latency", client=client)
    def instrumented() -> str:
        return "ok"

    assert instrumented() == "ok"
    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    observation = trace_payload["observations"][0]
    assert trace_payload["started_at"] == application_started_at.isoformat()
    assert observation["started_at"] == application_started_at.isoformat()
    assert observation["ended_at"] == application_ended_at.isoformat()
    assert observation["latency_ms"] == 7


def test_observe_decorator_creates_nested_observations():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )
    set_default_observability_client(client)

    @observe(
        "inner-operation", client=client, observation_type=ObservationType.TOOL_CALL
    )
    def inner(value: int) -> int:
        return value + 1

    @observe("outer-operation", client=client)
    def outer(value: int) -> int:
        return inner(value) * 2

    assert outer(2) == 6

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    assert trace_payload["name"] == "outer-operation"
    root_observation = trace_payload["observations"][0]
    assert root_observation["name"] == "outer-operation"
    assert root_observation["children"][0]["name"] == "inner-operation"
    assert root_observation["children"][0]["type"] == "tool_call"


def test_observe_decorator_enriches_trace_metadata_for_trial_runs():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe("optimized-call", client=client, metadata={"custom_label": "golden-path"})
    def optimized_call() -> str:
        return "ok"

    with ConfigurationContext(
        {
            "model": "gpt-4o",
            "temperature": 0.1,
            "api_key": "top-secret",  # pragma: allowlist secret
            "_optuna_trial_id": 99,
        }
    ):
        with TrialContext(
            "trial-7",
            metadata={
                "optimization_id": "opt-1",
                "experiment_id": "exp-1",
                "experiment_run_id": "run-1",
                "config_snapshot": {
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "api_key": "top-secret",  # pragma: allowlist secret
                },
            },
        ):
            assert optimized_call() == "ok"

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    assert trace_payload["metadata"]["custom_label"] == "golden-path"
    assert trace_payload["metadata"]["traigent_active_config"] == {
        "model": "gpt-4o",
        "temperature": 0.1,
        "api_key": "[REDACTED]",
    }
    assert trace_payload["metadata"]["traigent_optimization_context"] == {
        "trial_id": "trial-7",
        "optimization_id": "opt-1",
        "experiment_id": "exp-1",
        "experiment_run_id": "run-1",
        "config_source": "trial-config",
    }
    assert trace_payload["observations"][0]["metadata"]["traigent_active_config"] == {
        "model": "gpt-4o",
        "temperature": 0.1,
        "api_key": "[REDACTED]",
    }


def test_observe_decorator_enriches_trace_metadata_for_direct_runs():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe("post-best-config-call", client=client)
    def post_best_config_call() -> str:
        return "done"

    with ConfigurationContext({"model": "gpt-4o-mini", "temperature": 0.7}):
        assert post_best_config_call() == "done"

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    assert trace_payload["metadata"]["traigent_active_config"] == {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
    }
    assert trace_payload["metadata"]["traigent_optimization_context"] == {
        "config_source": "applied-config"
    }


def test_observe_decorator_can_set_root_trace_identifiers():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe(
        "identified-call",
        client=client,
        session_id="session-demo-1",
        user_id="guided-demo-user",
        custom_trace_id="guided-demo:baseline",
    )
    def identified_call() -> str:
        return "ok"

    assert identified_call() == "ok"

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    assert trace_payload["session_id"] == "session-demo-1"
    assert trace_payload["user_id"] == "guided-demo-user"
    assert trace_payload["custom_trace_id"] == "guided-demo:baseline"


def test_observe_decorator_can_redact_inputs():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe("sensitive-operation", client=client, redact_input=True)
    def sensitive(password: str) -> str:
        return f"len={len(password)}"

    assert sensitive("super-secret-password") == "len=21"

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    assert trace_payload["input_data"] == {"redacted": True}
    assert trace_payload["observations"][0]["input_data"] == {"redacted": True}


def test_observe_decorator_omits_input_output_by_default():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe("metadata-only", client=client)
    def sensitive(secret: str) -> dict[str, str]:
        return {"answer": f"derived from {secret}"}

    assert sensitive("top-secret-value") == {"answer": "derived from top-secret-value"}

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    observation = trace_payload["observations"][0]
    assert "input_data" not in trace_payload
    assert "output_data" not in trace_payload
    assert "input_data" not in observation
    assert "output_data" not in observation


def test_observe_decorator_redacted_content_mode_omits_raw_content():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe("redacted-content-mode", client=client, content_mode="redacted")
    def sensitive(secret: str) -> dict[str, str]:
        return {"answer": f"derived from {secret}"}

    assert sensitive("top-secret-value") == {"answer": "derived from top-secret-value"}

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    observation = trace_payload["observations"][0]
    assert trace_payload["input_data"] == {"redacted": True}
    assert trace_payload["output_data"] == {"redacted": True}
    assert observation["input_data"] == {"redacted": True}
    assert observation["output_data"] == {"redacted": True}


@pytest.mark.parametrize(
    ("redact_input", "redact_output"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_observe_decorator_redacts_input_output_combinations(
    redact_input, redact_output
):
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe(
        "redaction-matrix",
        client=client,
        redact_input=redact_input,
        redact_output=redact_output,
        content_mode="record",
    )
    def sensitive(secret: str) -> dict[str, str]:
        return {"answer": f"derived from {secret}"}

    assert sensitive("top-secret-value") == {"answer": "derived from top-secret-value"}

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    observation = trace_payload["observations"][0]
    if redact_input:
        assert trace_payload["input_data"] == {"redacted": True}
        assert observation["input_data"] == {"redacted": True}
    else:
        assert trace_payload["input_data"]["args"] == ["top-secret-value"]
        assert observation["input_data"]["args"] == ["top-secret-value"]

    if redact_output:
        assert trace_payload["output_data"] == {"redacted": True}
        assert observation["output_data"] == {"redacted": True}
    else:
        assert trace_payload["output_data"] == {
            "answer": "derived from top-secret-value"
        }
        assert observation["output_data"] == {"answer": "derived from top-secret-value"}


def test_observe_context_redacts_input_and_output():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    with ObserveContext(
        name="context-redaction",
        client=client,
        input_data={"password": "top-secret-value"},  # pragma: allowlist secret
        redact_input=True,
        redact_output=True,
    ) as ctx:
        ctx._result = {"answer": "top-secret-value"}  # pragma: allowlist secret

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    observation = trace_payload["observations"][0]
    assert trace_payload["input_data"] == {"redacted": True}
    assert observation["input_data"] == {"redacted": True}
    assert trace_payload["output_data"] == {"redacted": True}
    assert observation["output_data"] == {"redacted": True}


def test_observability_client_flush_surfaces_backend_warnings():
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: {
            "warnings": [
                "Prompt reference could not be resolved for trace 'trace_warn': support/missing (label=latest)"
            ]
        },
    )

    trace_id = client.start_trace("warn-trace", trace_id="trace_warn")
    client.end_trace(trace_id)

    result = client.flush()
    client.close()

    assert result.success is True
    assert result.warnings == [
        "Prompt reference could not be resolved for trace 'trace_warn': support/missing (label=latest)"
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 10_001, "batch_size"),
        ("max_queue_size", 1_000_001, "max_queue_size"),
        ("max_buffer_age", 3601.0, "max_buffer_age"),
        ("flush_timeout", 601.0, "flush_timeout"),
        ("request_timeout", 601.0, "request_timeout"),
    ],
)
def test_observability_config_rejects_unbounded_values(field, value, message):
    kwargs = {
        "backend_origin": "http://localhost:5000",
        "api_key": "test-key",  # pragma: allowlist secret
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        ObservabilityConfig(**kwargs)


def test_observation_dto_rejects_negative_values():
    with pytest.raises(ValueError, match="input_tokens"):
        ObservationDTO(
            id="obs_bad",
            type=ObservationType.SPAN,
            name="bad-observation",
            input_tokens=-1,
        )


@pytest.mark.parametrize("status", ["", "x" * 65])
def test_observation_dto_rejects_invalid_status(status):
    with pytest.raises(ValueError, match="status"):
        ObservationDTO(
            id="obs_bad_status",
            type=ObservationType.SPAN,
            name="bad-observation",
            status=status,
        )


@pytest.mark.parametrize("status", ["", "x" * 65])
def test_trace_dto_rejects_invalid_status(status):
    with pytest.raises(ValueError, match="status"):
        TraceDTO(id="trace_bad_status", name="bad-trace", status=status)


def test_observation_dto_rejects_event_with_children():
    child = ObservationDTO(
        id="obs_child",
        type=ObservationType.SPAN,
        name="child",
    )

    with pytest.raises(ValueError, match="event observations cannot have children"):
        ObservationDTO(
            id="obs_event",
            type=ObservationType.EVENT,
            name="event-parent",
            children=[child],
        )


def test_observability_client_rejects_child_under_event_observation():
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
    )

    trace_id = client.start_trace("event-child-validation", trace_id="trace_event")
    event_id = client.record_observation(
        trace_id,
        observation_id="obs_event",
        name="event-parent",
        observation_type=ObservationType.EVENT,
    )

    with pytest.raises(ValueError, match="event observations cannot have children"):
        client.record_observation(
            trace_id,
            observation_id="obs_child",
            parent_observation_id=event_id,
            name="invalid-child",
            observation_type=ObservationType.SPAN,
        )

    client.close()


def test_observability_client_rejects_converting_parent_to_event():
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
    )

    trace_id = client.start_trace("event-parent-validation", trace_id="trace_parent")
    parent_id = client.record_observation(
        trace_id,
        observation_id="obs_parent",
        name="parent",
        observation_type=ObservationType.SPAN,
    )
    client.record_observation(
        trace_id,
        observation_id="obs_child",
        parent_observation_id=parent_id,
        name="child",
        observation_type=ObservationType.SPAN,
    )

    with pytest.raises(ValueError, match="event observations cannot have children"):
        client.record_observation(
            trace_id,
            observation_id=parent_id,
            name="parent",
            observation_type=ObservationType.EVENT,
        )

    client.close()


def test_observability_client_collaboration_helpers_follow_backend_contract():
    request_calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        request_calls.append((method, path, payload))
        if method == "GET" and path == "/traces/trace_sdk/comments":
            return {
                "data": {
                    "trace_id": "trace_sdk",
                    "count": 1,
                    "items": [
                        {
                            "id": "comment_1",
                            "trace_id": "trace_sdk",
                            "author_user_id": "sdk-user",
                            "content": "Investigate the answer format.",
                            "created_at": "2026-03-10T14:10:00+00:00",
                            "updated_at": "2026-03-10T14:10:00+00:00",
                        }
                    ],
                }
            }
        if method == "POST" and path == "/traces/trace_sdk/comments":
            return {
                "data": {
                    "id": "comment_2",
                    "trace_id": "trace_sdk",
                    "author_user_id": "sdk-user",
                    "content": payload["content"],
                    "created_at": "2026-03-10T14:11:00+00:00",
                    "updated_at": "2026-03-10T14:11:00+00:00",
                }
            }
        if method == "PUT" and path == "/traces/trace_sdk/feedback":
            return {
                "data": {
                    "feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_sdk",
                        "author_user_id": "sdk-user",
                        "rating": payload["rating"],
                        "comment": payload["comment"],
                        "correction_output": payload["correction_output"],
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                    "summary": {
                        "up_count": 1 if payload["rating"] == "up" else 0,
                        "down_count": 1 if payload["rating"] == "down" else 0,
                    },
                }
            }
        if method == "PATCH" and path == "/traces/trace_sdk/collaboration":
            return {
                "data": {
                    "is_bookmarked": bool(payload.get("is_bookmarked")),
                    "bookmarked_at": (
                        "2026-03-10T14:13:00+00:00"
                        if payload.get("is_bookmarked")
                        else None
                    ),
                    "bookmarked_by": (
                        "sdk-user" if payload.get("is_bookmarked") else None
                    ),
                    "is_published": bool(payload.get("is_published")),
                    "published_at": (
                        "2026-03-10T14:14:00+00:00"
                        if payload.get("is_published")
                        else None
                    ),
                    "published_by": "sdk-user" if payload.get("is_published") else None,
                    "comment_count": 2,
                    "feedback_summary": {"up_count": 1, "down_count": 0},
                    "current_user_feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_sdk",
                        "author_user_id": "sdk-user",
                        "rating": "up",
                        "comment": "Approved",
                        "correction_output": {"answer": "Approved answer"},
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                }
            }
        raise AssertionError(f"Unexpected SDK request: {method} {path}")

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=request_sender,
    )

    comments = client.list_comments("trace_sdk")
    created_comment = client.add_comment("trace_sdk", "Ship this after QA review.")
    feedback = client.submit_feedback(
        "trace_sdk",
        ThumbRating.UP,
        comment="Approved",
        correction_output={"answer": "Approved answer"},
    )
    bookmarked = client.set_bookmarked("trace_sdk", True)
    published = client.set_published("trace_sdk", True)
    client.close()

    assert comments.count == 1
    assert comments.items[0].content == "Investigate the answer format."
    assert created_comment.content == "Ship this after QA review."
    assert feedback.feedback.rating is ThumbRating.UP
    assert feedback.summary.up_count == 1
    assert bookmarked.is_bookmarked is True
    assert bookmarked.current_user_feedback is not None
    assert published.is_published is True
    assert request_calls == [
        ("GET", "/traces/trace_sdk/comments", None),
        (
            "POST",
            "/traces/trace_sdk/comments",
            {"content": "Ship this after QA review."},
        ),
        (
            "PUT",
            "/traces/trace_sdk/feedback",
            {
                "rating": "up",
                "comment": "Approved",
                "correction_output": {"answer": "Approved answer"},
            },
        ),
        (
            "PATCH",
            "/traces/trace_sdk/collaboration",
            {"is_bookmarked": True, "is_published": None},
        ),
        (
            "PATCH",
            "/traces/trace_sdk/collaboration",
            {"is_bookmarked": None, "is_published": True},
        ),
    ]


def test_observability_client_query_helpers_follow_backend_contract():
    request_calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        request_calls.append((method, path, payload))
        if method == "GET" and path.startswith("/traces?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "trace_sdk_query",
                            "name": "query-trace",
                            "status": "running",
                            "session_id": "session_sdk_query",
                            "user_id": "sdk-user",
                            "environment": "production",
                            "release": "2026.03.10",
                            "tags": ["demo"],
                            "observation_count": 2,
                            "root_observation_count": 1,
                            "total_input_tokens": 10,
                            "total_output_tokens": 5,
                            "total_tokens": 15,
                            "total_cost_usd": 0.002,
                            "total_latency_ms": 250,
                        }
                    ],
                    "pagination": {
                        "page": 2,
                        "per_page": 10,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": True,
                    },
                }
            }
        if method == "GET" and path == "/traces/trace_sdk_query":
            return {
                "data": {
                    "id": "trace_sdk_query",
                    "name": "query-trace",
                    "status": "completed",
                    "session_id": "session_sdk_query",
                    "user_id": "sdk-user",
                    "environment": "production",
                    "session": {
                        "id": "session_sdk_query",
                        "trace_count": 1,
                        "observation_count": 2,
                        "total_tokens": 15,
                    },
                    "collaboration": {
                        "is_bookmarked": True,
                        "comment_count": 1,
                        "feedback_summary": {"up_count": 1, "down_count": 0},
                    },
                }
            }
        if method == "GET" and path == "/traces/trace_sdk_query/observations":
            return {
                "data": {
                    "trace_id": "trace_sdk_query",
                    "observation_count": 1,
                    "items": [
                        {
                            "id": "obs_sdk_root",
                            "trace_id": "trace_sdk_query",
                            "type": "span",
                            "name": "root",
                            "status": "completed",
                            "depth": 0,
                            "sequence_number": 0,
                            "latency_ms": 250,
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "cost_usd": 0.002,
                            "children": [],
                        }
                    ],
                }
            }
        if method == "GET" and path.startswith("/sessions?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "session_sdk_query",
                            "user_id": "sdk-user",
                            "environment": "production",
                            "trace_count": 1,
                            "observation_count": 2,
                            "total_tokens": 15,
                            "total_cost_usd": 0.002,
                            "ended_at": "2026-03-10T14:00:00+00:00",
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }
        if method == "GET" and path == "/sessions/session_sdk_query":
            return {
                "data": {
                    "id": "session_sdk_query",
                    "user_id": "sdk-user",
                    "environment": "production",
                    "trace_count": 1,
                    "observation_count": 2,
                    "total_tokens": 15,
                    "traces": [
                        {
                            "id": "trace_sdk_query",
                            "name": "query-trace",
                            "status": "completed",
                            "observation_count": 2,
                            "root_observation_count": 1,
                            "total_tokens": 15,
                            "total_cost_usd": 0.002,
                            "total_latency_ms": 250,
                        }
                    ],
                }
            }
        raise AssertionError(f"Unexpected SDK request: {method} {path}")

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=request_sender,
    )

    traces = client.list_traces(
        page=2,
        per_page=10,
        environment="production",
        tags=["demo"],
        start_time_from=datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC),
    )
    trace = client.get_trace("trace_sdk_query")
    observations = client.get_trace_observations("trace_sdk_query")
    sessions = client.list_sessions(search="session_sdk_query", release="2026.03.10")
    session = client.get_session("session_sdk_query")
    client.close()

    assert traces.pagination.page == 2
    assert traces.items[0].status == "running"
    assert trace.collaboration is not None
    assert trace.collaboration.is_bookmarked is True
    assert observations.items[0].type.value == "span"
    assert sessions.items[0].id == "session_sdk_query"
    assert session.traces[0].id == "trace_sdk_query"
    assert request_calls[0][1].startswith(
        "/traces?page=2&per_page=10&environment=production&tags=demo&start_time_from="
    )


def test_observability_client_rejects_collaboration_requests_after_close():
    request_calls: list[tuple[str, str, dict | None]] = []

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=lambda method, path, payload: (
            request_calls.append((method, path, payload)) or {"data": {}}
        ),
    )
    client.close()

    with pytest.raises(ClientError, match="closed"):
        client.list_comments("trace_sdk")

    assert request_calls == []


def test_observability_client_validates_feedback_correction_output():
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=lambda method, path, payload: {"data": {}},
    )

    with pytest.raises(ClientError, match="correction_output"):
        client.submit_feedback(
            "trace_sdk", ThumbRating.UP, correction_output={"bad": {1, 2, 3}}
        )

    client.close()


def test_observability_client_surfaces_collaboration_error_paths():
    def missing_trace_request_sender(method: str, path: str, payload: dict | None):
        raise ClientError(
            "Observability request failed with status 404",
            status_code=404,
            details={"body": '{"error":"not found"}'},
        )

    missing_trace_client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=missing_trace_request_sender,
    )

    with pytest.raises(ClientError, match="404"):
        missing_trace_client.add_comment("missing_trace", "Comment")

    missing_trace_client.close()

    def forbidden_request_sender(method: str, path: str, payload: dict | None):
        raise AuthenticationError("Observability request rejected with status 403")

    forbidden_client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=forbidden_request_sender,
    )

    with pytest.raises(AuthenticationError, match="403"):
        forbidden_client.submit_feedback("trace_sdk", ThumbRating.UP)

    forbidden_client.close()


def test_observability_client_logs_ingest_warnings(monkeypatch, caplog):
    class _FakeResponse:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"data":{"warnings":["Prompt reference could not be resolved for trace '
                b"'trace_warn': support/missing (label=latest)\"]}}"
            )

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
    )

    monkeypatch.setattr(
        client, "_open_http_request", lambda http_request: _FakeResponse()
    )

    with caplog.at_level(logging.WARNING):
        client._post_batch_sync(
            [{"id": "trace_warn", "name": "warn-trace", "observations": []}]
        )

    client.close()

    assert caplog.text.count("Observability ingest warning") == 1


@pytest.mark.parametrize(
    ("method_name", "args", "message"),
    [
        (
            "_post_batch_sync",
            ([{"id": "trace_sdk"}],),
            "Observability ingest failed with status 500",
        ),
        (
            "_request_json_sync",
            ("GET", "/traces/trace_sdk", None),
            "Observability request failed with status 500",
        ),
    ],
)
def test_observability_client_closes_http_errors(
    monkeypatch, method_name, args, message
):
    http_error = error.HTTPError(
        url="http://localhost:5000",
        code=500,
        msg="boom",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"boom"}'),
    )
    close_calls = {"count": 0}
    original_close = http_error.close

    def close() -> None:
        close_calls["count"] += 1
        original_close()

    http_error.close = close

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
    )

    monkeypatch.setattr(
        client,
        "_open_http_request",
        lambda http_request: (_ for _ in ()).throw(http_error),
    )

    with pytest.raises(ClientError, match=message):
        getattr(client, method_name)(*args)

    client.close()

    assert close_calls["count"] == 1
