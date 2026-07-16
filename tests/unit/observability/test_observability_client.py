from __future__ import annotations

import importlib
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
    ExecutionContextDTO,
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

retry_module = importlib.import_module("traigent.utils.retry")


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
        # The IMDS literal is rejected by IP value as a metadata service, ahead
        # of (and independently of) the production private/loopback branch.
        ("metadata-ip", "metadata service"),
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


def test_observability_client_merges_content_free_lineage_defaults(monkeypatch):
    monkeypatch.setenv("TRAIGENT_AGENT_ID", "agent-from-env")
    monkeypatch.setenv("TRAIGENT_TOOLSET_ID", "toolset-from-env")
    sent_batches: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            enable_atexit_flush=False,
        ),
        sender=sent_batches.append,
    )

    client.start_trace(
        "lineage-defaults",
        trace_id="trace_lineage_defaults",
        execution_context={
            "release_id": "release-explicit",
            "toolset_id": None,
            "prompt_id": "prompt-7",
        },
    )
    client.flush()
    client.close()

    execution_context = sent_batches[-1][-1]["execution_context"]
    assert execution_context["schema_version"] == "1.0"
    assert execution_context["agent_id"] == "agent-from-env"
    assert execution_context["release_id"] == "release-explicit"
    assert execution_context["prompt_id"] == "prompt-7"
    assert execution_context["toolset_id"] is None


def test_execution_context_rejects_content_and_invalid_identifiers():
    with pytest.raises(ValueError, match="unsupported field.*metadata"):
        ExecutionContextDTO.from_dict({"metadata": {"prompt": "must not leak"}})
    with pytest.raises(ValueError, match="agent_id must not be empty"):
        ExecutionContextDTO(agent_id="")


def test_execution_context_dto_explicit_null_clears_client_default():
    context = ExecutionContextDTO(toolset_id=None)
    assert context.to_dict() == {"schema_version": "1.0", "toolset_id": None}

    sent_batches: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            enable_atexit_flush=False,
            default_execution_context={
                "agent_id": "agent-default",
                "toolset_id": "toolset-default",
            },
        ),
        sender=sent_batches.append,
    )
    client.start_trace(
        "lineage-explicit-null",
        trace_id="trace_lineage_explicit_null",
        execution_context=context,
    )
    client.flush()
    client.close()

    execution_context = sent_batches[-1][-1]["execution_context"]
    assert execution_context["agent_id"] == "agent-default"
    assert execution_context["toolset_id"] is None


def test_tool_observation_infers_stable_tool_name_and_failure_status():
    sent_batches: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            enable_atexit_flush=False,
        ),
        sender=sent_batches.append,
    )
    trace_id = client.start_trace("tool-failure", trace_id="trace_tool_failure")
    client.record_observation(
        trace_id,
        name="search.lookup",
        observation_type=ObservationType.TOOL_CALL,
        status="failed",
    )
    client.end_trace(trace_id, status="failed")
    client.flush()
    client.close()

    observation = sent_batches[-1][-1]["observations"][0]
    assert observation["tool_name"] == "search.lookup"
    assert observation["status"] == "failed"


def test_observe_propagates_tool_identity_and_execution_context():
    sent_batches: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            enable_atexit_flush=False,
        ),
        sender=sent_batches.append,
    )

    with observe(
        "search.lookup",
        client=client,
        observation_type=ObservationType.TOOL_CALL,
        tool_name="search.lookup.v2",
        execution_context={
            "agent_id": "agent-1",
            "release_id": "release-2",
            "prompt_id": "prompt-3",
            "toolset_id": "toolset-4",
        },
    ):
        pass
    client.flush()
    client.close()

    trace = sent_batches[-1][-1]
    assert trace["execution_context"]["agent_id"] == "agent-1"
    assert trace["execution_context"]["release_id"] == "release-2"
    assert trace["observations"][0]["tool_name"] == "search.lookup.v2"
    assert trace["observations"][0]["status"] == "completed"


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
    assert stats["dropped_by_reason"] == {"queue_full": 2}
    assert stats["queue_depth"] == 1
    assert stats["retry_attempts"] == 0
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

        def close(self, timeout=None):
            del timeout
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


def test_sync_batch_transport_retries_status_client_errors(monkeypatch):
    sent_batches: list[list[dict]] = []
    sleep_calls: list[float] = []
    call_count = 0

    monkeypatch.setattr(retry_module.time, "sleep", sleep_calls.append)

    def sender(traces):
        nonlocal call_count
        call_count += 1
        sent_batches.append(traces)
        if call_count <= 2:
            raise ClientError("rate limited", status_code=429)
        return None

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
    )

    assert (
        transport.submit("trace_retry", {"id": "trace_retry", "name": "retry"}) is True
    )

    result = transport.flush()
    transport.close()

    assert result.success is True
    assert result.items_sent == 1
    assert result.items_dropped == 0
    assert result.failed_batches == 0
    assert call_count == 3
    assert len(sent_batches) == 3
    assert len(sleep_calls) == 2
    assert transport.get_stats()["retry_attempts"] == 2


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


@pytest.mark.timeout(3)
def test_timed_out_sender_remains_single_flight_until_it_reconciles():
    send_started = threading.Event()
    first_flush_returned = threading.Event()
    release_first_send = threading.Event()
    second_flush_reached_orphan_wait = threading.Event()
    release_second_flush = threading.Event()
    second_flush_finished = threading.Event()
    active_lock = threading.Lock()
    active_senders = 0
    max_active_senders = 0
    send_count = 0

    def sender(traces):
        del traces
        nonlocal active_senders, max_active_senders, send_count
        with active_lock:
            active_senders += 1
            max_active_senders = max(max_active_senders, active_senders)
            send_count += 1
            call_number = send_count
        try:
            if call_number == 1:
                send_started.set()
                assert release_first_send.wait(timeout=2.0)
        finally:
            with active_lock:
                active_senders -= 1

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=2,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
    )
    assert transport.submit("trace_one", {"id": "trace_one"}) is True

    first_results: list[BatchFlushResult] = []

    def first_flush() -> None:
        first_results.append(transport.flush(timeout=0.05))
        first_flush_returned.set()

    first_thread = threading.Thread(target=first_flush)
    first_thread.start()
    assert send_started.wait(timeout=1.0)
    assert first_flush_returned.wait(timeout=1.0)

    assert first_results[0].success is False
    assert first_results[0].items_pending == 1
    stats_while_hung = transport.get_stats()
    assert stats_while_hung["send_in_progress"] is True
    assert stats_while_hung["inflight_items"] == 1
    assert stats_while_hung["oldest_inflight_age_seconds"] is not None

    assert transport.submit("trace_two", {"id": "trace_two"}) is True

    original_wait_for_active_send = transport._wait_for_active_send

    def wait_for_active_send(deadline: float | None) -> bool:
        with transport._lock:
            active_send_completion = transport._active_send_completion
        if active_send_completion is None:
            return original_wait_for_active_send(deadline)
        second_flush_reached_orphan_wait.set()
        assert release_second_flush.wait(timeout=2.0)
        return original_wait_for_active_send(deadline)

    transport._wait_for_active_send = wait_for_active_send  # type: ignore[method-assign]

    def second_flush() -> None:
        transport.flush(timeout=1.0)
        second_flush_finished.set()

    second_thread = threading.Thread(target=second_flush)
    second_thread.start()
    assert second_flush_reached_orphan_wait.wait(timeout=1.0)
    with active_lock:
        assert max_active_senders == 1

    release_first_send.set()
    release_second_flush.set()
    assert second_flush_finished.wait(timeout=1.0)
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    result = transport.flush()

    assert result.success is True
    assert result.items_sent == 2
    assert result.items_pending == 0
    assert transport.get_stats()["send_in_progress"] is False
    assert transport.get_stats()["inflight_items"] == 0
    with active_lock:
        assert max_active_senders == 1


def test_observability_client_default_flush_sends_on_calling_thread():
    sender_threads: list[int] = []

    def sender(traces):
        del traces
        sender_threads.append(threading.get_ident())

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
            enable_atexit_flush=False,
        ),
        sender=sender,
    )
    trace_id = client.start_trace("prompt", trace_id="trace_prompt_flush")
    client.end_trace(trace_id)

    result = client.flush()

    assert result.success is True
    assert result.items_sent == 1
    assert sender_threads == [threading.get_ident()]


def test_observability_client_timeout_zero_is_warning_free_poll():
    sender_calls: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
            enable_atexit_flush=False,
        ),
        sender=lambda traces: sender_calls.append(traces),
    )
    trace_id = client.start_trace("poll", trace_id="trace_poll_flush")
    client.end_trace(trace_id)

    result = client.flush(timeout=0)

    assert result.success is False
    assert result.items_pending == 1
    assert sender_calls == []
    assert not any("flush deadline exceeded" in warning for warning in result.warnings)


@pytest.mark.timeout(3)
def test_sync_batch_transport_bounded_flush_and_close_cover_transport_state_lock():
    lock_held = threading.Event()
    release_lock = threading.Event()
    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
    )
    assert transport.submit("locked", {"id": "locked"}) is True

    def hold_lock() -> None:
        with transport._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2.0)

    lock_holder = threading.Thread(target=hold_lock)
    lock_holder.start()
    assert lock_held.wait(timeout=1.0)
    try:
        flush_started = time.monotonic()
        flush_result = transport.flush(timeout=0.05)
        flush_elapsed = time.monotonic() - flush_started
        close_started = time.monotonic()
        close_result = transport.close(timeout=0.05)
        close_elapsed = time.monotonic() - close_started

        assert flush_elapsed < 0.15
        assert close_elapsed < 0.15
        assert flush_result.success is False
        assert close_result.success is False
        assert flush_result.items_pending == 1
        assert close_result.items_pending == 1
        assert any("state lock" in warning for warning in flush_result.warnings)
        assert any("state lock" in warning for warning in close_result.warnings)
        assert transport._closed is False
    finally:
        release_lock.set()
        lock_holder.join(timeout=1.0)
    assert not lock_holder.is_alive()
    assert transport.close().success is True


@pytest.mark.timeout(3)
def test_observability_client_bounded_lock_probes_return_truthful_results(
    monkeypatch,
):
    lock_held = threading.Event()
    release_lock = threading.Event()
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            enable_atexit_flush=False,
        ),
        sender=lambda traces: None,
    )

    def hold_client_lock() -> None:
        with client._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2.0)

    lock_holder = threading.Thread(target=hold_client_lock)
    lock_holder.start()
    assert lock_held.wait(timeout=1.0)
    try:
        flush_started = time.monotonic()
        flush_result = client.flush(timeout=0.05)
        flush_elapsed = time.monotonic() - flush_started

        assert flush_elapsed < 0.15
        assert flush_result.success is False
        assert flush_result.items_pending == 0
        assert any("client state lock" in warning for warning in flush_result.warnings)
    finally:
        release_lock.set()
        lock_holder.join(timeout=1.0)
    assert not lock_holder.is_alive()

    transport_lock_held = threading.Event()
    release_transport_lock = threading.Event()
    trace_id = client.start_trace("transport-locked", trace_id="transport_locked")
    client.end_trace(trace_id)

    def hold_transport_lock() -> None:
        with client._transport._lock:
            transport_lock_held.set()
            assert release_transport_lock.wait(timeout=2.0)

    transport_lock_holder = threading.Thread(target=hold_transport_lock)
    transport_lock_holder.start()
    assert transport_lock_held.wait(timeout=1.0)
    try:
        transport_flush_started = time.monotonic()
        transport_flush_result = client.flush(timeout=0.05)
        transport_flush_elapsed = time.monotonic() - transport_flush_started

        assert transport_flush_elapsed < 0.15
        assert transport_flush_result.success is False
        assert transport_flush_result.items_pending == 1
    finally:
        release_transport_lock.set()
        transport_lock_holder.join(timeout=1.0)
    assert not transport_lock_holder.is_alive()
    assert client.flush().success is True

    finalizer_lock_held = threading.Event()
    release_finalizer_lock = threading.Event()
    start_finalizer_holder = threading.Event()
    original_transport_close = client._transport.close

    def hold_finalizer_lock() -> None:
        assert start_finalizer_holder.wait(timeout=1.0)
        with client._lock:
            finalizer_lock_held.set()
            assert release_finalizer_lock.wait(timeout=2.0)

    finalizer_lock_holder = threading.Thread(target=hold_finalizer_lock)
    finalizer_lock_holder.start()

    def coordinated_transport_close(*args, **kwargs):
        start_finalizer_holder.set()
        assert finalizer_lock_held.wait(timeout=1.0)
        return original_transport_close(*args, **kwargs)

    monkeypatch.setattr(client._transport, "close", coordinated_transport_close)
    try:
        close_started = time.monotonic()
        close_result = client.close(timeout=0.05)
        close_elapsed = time.monotonic() - close_started

        assert close_elapsed < 0.15
        assert close_result.success is False
        assert close_result.items_pending == 0
        assert any(
            "reacquiring the client state lock" in warning
            for warning in close_result.warnings
        )
    finally:
        release_finalizer_lock.set()
        finalizer_lock_holder.join(timeout=1.0)
    assert not finalizer_lock_holder.is_alive()
    assert client._close_complete.wait(timeout=1.0)
    assert client.close().success is True


@pytest.mark.timeout(3)
def test_observability_client_bounded_flush_covers_client_state_lock():
    lock_held = threading.Event()
    release_lock = threading.Event()
    sender_calls: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
            enable_atexit_flush=False,
        ),
        sender=lambda traces: sender_calls.append(traces),
    )
    trace_id = client.start_trace("locked-flush", trace_id="trace_locked_flush")
    client.end_trace(trace_id)

    def hold_lock() -> None:
        with client._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2.0)

    lock_holder = threading.Thread(target=hold_lock)
    lock_holder.start()
    assert lock_held.wait(timeout=1.0)
    try:
        started = time.monotonic()
        result = client.flush(timeout=0.05)
        elapsed = time.monotonic() - started

        assert elapsed < 0.15
        assert result.success is False
        assert result.items_pending == 1
        assert sender_calls == []
    finally:
        release_lock.set()
        lock_holder.join(timeout=1.0)
    assert not lock_holder.is_alive()
    assert client.flush().success is True


@pytest.mark.timeout(3)
def test_observability_client_bounded_close_covers_client_state_lock():
    lock_held = threading.Event()
    release_lock = threading.Event()
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            enable_atexit_flush=False,
        )
    )

    def hold_lock() -> None:
        with client._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2.0)

    lock_holder = threading.Thread(target=hold_lock)
    lock_holder.start()
    assert lock_held.wait(timeout=1.0)
    try:
        result = client.close(timeout=0.05)

        assert result.success is False
        assert result.items_pending == client._transport.get_stats()["pending_items"]
        assert any("client state lock" in warning for warning in result.warnings)
        assert client._closed is False
        assert client._transport._closed is False
    finally:
        release_lock.set()
        lock_holder.join(timeout=1.0)
    assert not lock_holder.is_alive()
    assert client.close().success is True


@pytest.mark.timeout(3)
def test_observability_client_retries_pending_transport_close_after_lock_timeout():
    lock_held = threading.Event()
    release_lock = threading.Event()
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            enable_atexit_flush=False,
            offline_mode=True,
            health_callback=lambda event_type, payload: None,
        ),
        sender=lambda traces: None,
    )
    transport = client._transport
    dispatcher = transport._health_dispatcher_thread
    assert dispatcher is not None

    def hold_transport_lock() -> None:
        with transport._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2.0)

    lock_holder = threading.Thread(target=hold_transport_lock)
    lock_holder.start()
    assert lock_held.wait(timeout=1.0)
    try:
        started = time.monotonic()
        first = client.close(timeout=0.05)

        assert time.monotonic() - started < 0.15
        assert first.success is False
        assert client._closed is False
        assert client._close_pending is True
        assert transport._closed is False
    finally:
        release_lock.set()
        lock_holder.join(timeout=1.0)
    assert not lock_holder.is_alive()

    second = client.close(timeout=0.1)

    assert second.success is True
    assert client._closed is True
    assert transport._closed is True
    assert dispatcher not in threading.enumerate()


@pytest.mark.timeout(3)
def test_observability_client_repeated_close_reports_orphan_until_reconciled():
    send_started = threading.Event()
    release_send = threading.Event()
    send_completed = threading.Event()

    def sender(traces: list[dict]) -> None:
        del traces
        send_started.set()
        assert release_send.wait(timeout=2.0)
        send_completed.set()

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
            enable_atexit_flush=False,
        ),
        sender=sender,
    )
    trace_id = client.start_trace("orphan-close", trace_id="trace_orphan_close")
    client.end_trace(trace_id)

    first = client.close(timeout=0.03)
    assert send_started.wait(timeout=1.0)
    second = client.close(timeout=0)

    assert first.success is False
    assert first.items_pending == 1
    assert second.success is False
    assert second.items_pending == 1

    release_send.set()
    assert send_completed.wait(timeout=1.0)
    third = client.close(timeout=0)

    assert third.success is True
    assert third.items_pending == 0
    assert client.get_stats()["inflight_items"] == 0


def test_observability_client_atexit_uses_configured_flush_deadline(monkeypatch):
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            flush_timeout=0.5,
            enable_atexit_flush=False,
        )
    )
    timeouts: list[float | None] = []

    def close(*, timeout: float | None = None) -> FlushResult:
        timeouts.append(timeout)
        return FlushResult(True, 0, 0, 0, 0, 0, [], [])

    monkeypatch.setattr(client, "close", close)

    client._atexit_close()

    assert timeouts == [0.5]


@pytest.mark.timeout(3)
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
    delayed_submission_thread: list[int] = []

    def delayed_submit(
        trace_id: str, payload: dict, *, deadline: float | None = None
    ) -> None:
        if not delayed_submission_thread:
            delayed_submission_thread.append(threading.get_ident())
            submit_entered.set()
            assert release_submit.wait(timeout=2.0)
        else:
            assert threading.get_ident() != delayed_submission_thread[0]
        original_submit(trace_id, payload, deadline=deadline)

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


@pytest.mark.timeout(3)
def test_observability_client_concurrent_close_has_single_initial_closer(monkeypatch):
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=10,
            enable_atexit_flush=False,
        ),
        sender=lambda traces: None,
    )
    trace_id = client.start_trace("concurrent-close", trace_id="trace_concurrent_close")
    client.end_trace(trace_id)

    initial_releases = threading.Barrier(2)
    release_counts: dict[int, int] = {}
    original_client_lock = client._lock

    class CoordinatedInitialReleaseLock:
        def acquire(self, *args, **kwargs):
            return original_client_lock.acquire(*args, **kwargs)

        def release(self) -> None:
            original_client_lock.release()
            if threading.current_thread().name not in {
                "initial-closer",
                "competing-closer",
            }:
                return
            thread_id = threading.get_ident()
            release_count = release_counts.get(thread_id, 0)
            release_counts[thread_id] = release_count + 1
            if release_count == 0:
                initial_releases.wait(timeout=1.0)

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.release()

    client._lock = CoordinatedInitialReleaseLock()  # type: ignore[assignment]

    first_submit_entered = threading.Event()
    release_first_submit = threading.Event()
    competing_reached_transport_close = threading.Event()
    submission_claim_lock = threading.Lock()
    submission_claimed = False
    original_submit = client._submit_trace_snapshot
    original_transport_close = client._transport.close

    def delayed_first_submit(
        trace_id: str, payload: dict, *, deadline: float | None = None
    ) -> None:
        nonlocal submission_claimed
        with submission_claim_lock:
            is_first_submission = not submission_claimed
            submission_claimed = True
        if is_first_submission:
            first_submit_entered.set()
            assert release_first_submit.wait(timeout=2.0)
        original_submit(trace_id, payload, deadline=deadline)

    def observed_transport_close(*args, **kwargs):
        if not release_first_submit.is_set():
            competing_reached_transport_close.set()
        return original_transport_close(*args, **kwargs)

    monkeypatch.setattr(client, "_submit_trace_snapshot", delayed_first_submit)
    monkeypatch.setattr(client._transport, "close", observed_transport_close)

    first_results: list[FlushResult] = []
    second_results: list[FlushResult] = []
    first = threading.Thread(
        target=lambda: first_results.append(client.close()), name="initial-closer"
    )
    second = threading.Thread(
        target=lambda: second_results.append(client.close()), name="competing-closer"
    )
    first.start()
    second.start()
    assert first_submit_entered.wait(timeout=1.0)

    assert not competing_reached_transport_close.wait(timeout=0.1)
    release_first_submit.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert first_results[0].items_dropped == 0
    assert second_results[0].items_dropped == 0
    assert client.get_stats()["dropped_by_reason"].get("transport_closed", 0) == 0


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
    callback_finished = threading.Event()

    def health_callback(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))
        callback_finished.set()

    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )

    assert transport.submit("trace_one", {"id": "trace_one"}) is True
    assert transport.submit("trace_two", {"id": "trace_two"}) is False
    result = transport.close()

    assert result.items_dropped == 1
    assert callback_finished.wait(timeout=1.0)
    assert events == [
        (
            "queue_full",
            {
                "drop_reason": "queue_full",
                "dropped_items": 1,
                "queue_depth": 1,
                "message": "transport queue full; dropped payload for item 'trace_two'",
                "item_id": "trace_two",
            },
        )
    ]


@pytest.mark.timeout(3)
def test_sync_batch_transport_delivers_health_events_in_snapshot_order():
    callback_one_started = threading.Event()
    release_callback_one = threading.Event()
    second_drop_snapshotted = threading.Event()
    delivery_complete = threading.Event()
    delivered_dropped_items: list[int] = []

    def health_callback(event_type: str, payload: dict) -> None:
        assert event_type == "queue_full"
        if payload["dropped_items"] == 1:
            callback_one_started.set()
            assert release_callback_one.wait(timeout=2.0)
        delivered_dropped_items.append(payload["dropped_items"])
        if len(delivered_dropped_items) == 2:
            delivery_complete.set()

    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )
    original_record_drop_locked = transport._record_drop_locked

    def record_drop_locked(
        event_type: str, message: str, **details: object
    ) -> tuple[str, dict[str, object]]:
        event = original_record_drop_locked(event_type, message, **details)
        if event[1]["dropped_items"] == 2:
            second_drop_snapshotted.set()
        return event

    transport._record_drop_locked = record_drop_locked  # type: ignore[method-assign]

    assert transport.submit("trace_one", {"id": "trace_one"}) is True

    first_drop = threading.Thread(
        target=lambda: transport.submit("trace_two", {"id": "trace_two"})
    )
    first_drop.start()
    assert callback_one_started.wait(timeout=1.0)

    second_drop = threading.Thread(
        target=lambda: transport.submit("trace_three", {"id": "trace_three"})
    )
    second_drop.start()
    assert second_drop_snapshotted.wait(timeout=1.0)

    release_callback_one.set()
    first_drop.join(timeout=1.0)
    second_drop.join(timeout=1.0)
    assert not first_drop.is_alive()
    assert not second_drop.is_alive()
    assert delivery_complete.wait(timeout=1.0)
    assert delivered_dropped_items == [1, 2]
    assert all(
        earlier <= later
        for earlier, later in zip(
            delivered_dropped_items, delivered_dropped_items[1:], strict=False
        )
    )
    transport.close()


@pytest.mark.timeout(3)
def test_sync_batch_transport_health_queue_drops_oldest_on_overflow():
    first_callback_started = threading.Event()
    release_first_callback = threading.Event()
    delivered: list[int] = []

    def health_callback(event_type: str, payload: dict) -> None:
        assert event_type == "queue_full"
        if payload["dropped_items"] == 1:
            first_callback_started.set()
            assert release_first_callback.wait(timeout=2.0)
        delivered.append(payload["dropped_items"])

    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )
    assert transport.submit("accepted", {"id": "accepted"}) is True
    assert transport.submit("drop_1", {"id": "drop_1"}) is False
    assert first_callback_started.wait(timeout=1.0)
    for index in range(2, 259):
        assert transport.submit(f"drop_{index}", {"id": f"drop_{index}"}) is False

    assert transport.get_stats()["dropped_health_events"] == 1
    release_first_callback.set()
    transport.close()

    assert delivered == [1, *range(3, 259)]


@pytest.mark.timeout(3)
def test_sync_batch_transport_close_drains_health_events_before_dispatcher_stops():
    first_callback_started = threading.Event()
    release_first_callback = threading.Event()
    delivered: list[int] = []

    def health_callback(event_type: str, payload: dict) -> None:
        assert event_type == "queue_full"
        if payload["dropped_items"] == 1:
            first_callback_started.set()
            assert release_first_callback.wait(timeout=2.0)
        delivered.append(payload["dropped_items"])

    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )
    assert transport.submit("accepted", {"id": "accepted"}) is True
    assert transport.submit("drop_1", {"id": "drop_1"}) is False
    assert first_callback_started.wait(timeout=1.0)
    assert transport.submit("drop_2", {"id": "drop_2"}) is False

    close_thread = threading.Thread(target=transport.close)
    close_thread.start()
    close_thread.join(timeout=0.1)
    assert close_thread.is_alive()

    release_first_callback.set()
    close_thread.join(timeout=1.0)
    assert not close_thread.is_alive()
    assert delivered == [1, 2]
    assert transport._health_dispatcher_thread is not None
    assert not transport._health_dispatcher_thread.is_alive()

    assert transport.submit("after_close", {"id": "after_close"}) is False
    assert delivered == [1, 2]
    assert transport.get_stats()["dropped_health_events"] == 1


def test_observability_client_close_does_not_leak_health_dispatchers():
    thread_name = "traigent-observability-health-dispatcher"
    baseline = sum(thread.name == thread_name for thread in threading.enumerate())
    clients = [
        ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                enable_atexit_flush=False,
                health_callback=lambda event_type, payload: None,
            ),
            sender=lambda traces: None,
        )
        for _ in range(5)
    ]

    for client in clients:
        client.close()

    assert (
        sum(thread.name == thread_name for thread in threading.enumerate()) == baseline
    )


@pytest.mark.timeout(3)
def test_blocking_health_callback_does_not_extend_flush_deadline():
    callback_started = threading.Event()
    release_callback = threading.Event()
    events_delivered = threading.Event()
    delivered_dropped_items: list[int] = []

    def health_callback(event_type: str, payload: dict) -> None:
        assert event_type == "queue_full"
        if payload["dropped_items"] == 1:
            callback_started.set()
            assert release_callback.wait(timeout=2.0)
        delivered_dropped_items.append(payload["dropped_items"])
        if len(delivered_dropped_items) == 2:
            events_delivered.set()

    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )
    assert transport.submit("trace_one", {"id": "trace_one"}) is True
    assert transport.submit("trace_two", {"id": "trace_two"}) is False
    assert callback_started.wait(timeout=1.0)
    assert transport.submit("trace_three", {"id": "trace_three"}) is False

    started = time.monotonic()
    result = transport.flush(timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert result.success is True
    release_callback.set()
    assert events_delivered.wait(timeout=1.0)
    assert delivered_dropped_items == [1, 2]


def test_health_callback_get_stats_runs_after_submit_state_is_complete():
    callback_finished = threading.Event()
    callback_stats: list[dict] = []

    def health_callback(event_type: str, payload: dict) -> None:
        del event_type, payload

        def read_stats() -> None:
            callback_stats.append(transport.get_stats())
            callback_finished.set()

        reader = threading.Thread(target=read_stats)
        reader.start()
        reader.join(timeout=1.0)
        assert not reader.is_alive()

    transport = _SyncBatchTransport(
        sender=lambda traces: None,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )

    assert transport.submit("trace_one", {"id": "trace_one"}) is True
    assert transport.submit("trace_two", {"id": "trace_two"}) is False

    assert callback_finished.wait(timeout=1.0)
    assert len(callback_stats) == 1
    assert callback_stats[0]["dropped_items"] == 1
    assert callback_stats[0]["queue_depth"] == 1
    assert callback_stats[0]["pending_items"] == 1


@pytest.mark.timeout(3)
def test_health_callback_can_flush_after_batch_delivery_failure():
    callback_finished = threading.Event()
    callback_results: list[BatchFlushResult] = []

    def sender(traces):
        del traces
        raise AuthenticationError("invalid credentials")

    def health_callback(event_type: str, payload: dict) -> None:
        del payload
        if event_type == "batch_delivery_failed":
            callback_results.append(transport.flush())
            callback_finished.set()

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )
    assert transport.submit("trace_one", {"id": "trace_one"}) is True

    flush_finished = threading.Event()

    def flush() -> None:
        transport.flush()
        flush_finished.set()

    flush_thread = threading.Thread(target=flush)
    flush_thread.start()
    assert flush_finished.wait(timeout=1.0)
    assert callback_finished.wait(timeout=1.0)
    flush_thread.join(timeout=1.0)
    assert not flush_thread.is_alive()
    assert callback_results[0].items_pending == 0


def test_sync_batch_transport_reports_batch_delivery_drops_in_health_snapshot():
    events: list[tuple[str, dict]] = []
    callback_finished = threading.Event()

    def sender(traces):
        del traces
        raise AuthenticationError("invalid credentials")

    def health_callback(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))
        callback_finished.set()

    transport = _SyncBatchTransport(
        sender=sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=10,
        max_batch_bytes=10_000,
        health_callback=health_callback,
    )
    assert transport.submit("trace_one", {"id": "trace_one"}) is True
    assert transport.submit("trace_two", {"id": "trace_two"}) is True

    result = transport.flush()
    stats = transport.get_stats()

    assert result.items_dropped == 2
    assert stats["dropped_by_reason"] == {"batch_delivery_failed": 2}
    assert callback_finished.wait(timeout=1.0)
    assert events == [
        (
            "batch_delivery_failed",
            {
                "drop_reason": "batch_delivery_failed",
                "dropped_items": 2,
                "queue_depth": 0,
                "message": "invalid credentials",
                "item_count": 2,
                "trace_ids": ["trace_one", "trace_two"],
                "trace_ids_truncated": False,
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


def test_observability_client_preserves_tool_type_on_lifecycle_update():
    sent_batches: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: sent_batches.append(traces),
    )

    trace_id = client.start_trace("tool-lifecycle", trace_id="trace_tool_lifecycle")
    observation_id = client.record_observation(
        trace_id,
        observation_id="obs_tool_lifecycle",
        name="search.lookup",
        observation_type=ObservationType.TOOL_CALL,
    )
    client.record_observation(
        trace_id,
        observation_id=observation_id,
        name="search.lookup",
        status="completed",
    )
    client.end_trace(trace_id)
    client.flush()
    client.close()

    observation = sent_batches[-1][-1]["observations"][0]
    assert observation["type"] == "tool_call"
    assert observation["tool_name"] == "search.lookup"
    assert observation["status"] == "completed"


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


@pytest.mark.parametrize(
    ("content_mode", "expected_error_message", "content_should_ship"),
    [
        # Default (metadata-only): the exception string must not ship at all.
        (None, None, False),
        ("redacted", "[REDACTED]", False),
        ("record", "parse failed on: PATIENT diagnosis cancer stage 3", True),
    ],
)
def test_observe_error_message_honors_content_mode(
    content_mode, expected_error_message, content_should_ship
):
    """Exception messages must obey the same content gate as input/output.

    Regression for the error-path egress leak: `error_message` carries
    free-form content (f-strings interpolate prompts, records, PII that
    pattern redaction cannot catch), so the metadata-only default must not
    ship it. `error_type` is only a class name and stays in every mode.
    """
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

    sensitive = "PATIENT diagnosis cancer stage 3"
    with pytest.raises(ValueError):
        with observe("parse", client=client, content_mode=content_mode):
            raise ValueError(f"parse failed on: {sensitive}")

    result = client.flush()
    client.close()

    assert result.success is True
    observation = sent_batches[-1][-1]["observations"][0]
    metadata = observation["metadata"]
    assert observation["status"] == "failed"
    # error_type is a class name, never content, so it is retained everywhere.
    assert metadata["error_type"] == "ValueError"
    if expected_error_message is None:
        assert "error_message" not in metadata
    else:
        assert metadata["error_message"] == expected_error_message
    # The sensitive free-form content only ever ships in "record" mode.
    assert (sensitive in json.dumps(sent_batches)) is content_should_ship


def test_observability_client_disables_egress_when_no_credential_resolves(
    monkeypatch, caplog
):
    """A missing credential must fail fast, not silently 401-retry-storm.

    With no API key or JWT and network egress otherwise enabled, the client
    logs one actionable warning naming TRAIGENT_API_KEY and disables its own
    network lanes for the process — never attempting an (inevitably rejected)
    unauthenticated ingest, and never raising. ``config.offline_mode`` stays
    untouched: it reflects the caller's explicit setting, not the guard.
    """
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)

    http_attempts = {"count": 0}

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key=None,
            jwt_token=None,
            batch_size=1,
            max_buffer_age=0.1,
            max_queue_size=10,
            enable_atexit_flush=False,
        )
    )

    def fail_if_called(*args, **kwargs):
        http_attempts["count"] += 1
        raise AssertionError("network egress attempted without a credential")

    monkeypatch.setattr(client._http_opener, "open", fail_if_called)

    trace_id = client.start_trace("no-credential-probe", trace_id="trace_no_cred")
    client.record_observation(trace_id, name="no-credential-observation")
    client.end_trace(trace_id)

    result = client.flush()
    with pytest.raises(ClientError, match="no credential resolved"):
        client.list_sessions()
    close_result = client.close()

    assert client.config.offline_mode is False
    assert client._credential_egress_disabled is True
    assert http_attempts["count"] == 0
    assert result.success is True
    assert result.items_sent == 0
    assert close_result.success is True
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "TRAIGENT_API_KEY" in message
    assert "egress disabled" in message


def test_observability_client_keeps_egress_when_api_key_present(monkeypatch, caplog):
    """The credential-missing fail-fast must not fire when a key resolves."""
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            enable_atexit_flush=False,
        )
    )

    assert client.config.offline_mode is False
    assert "TRAIGENT_API_KEY" not in caplog.text
    client.close()


@pytest.mark.parametrize("header_name", ["Authorization", "x-api-key"])
def test_observability_client_keeps_egress_when_auth_rides_extra_headers(
    monkeypatch, caplog, header_name
):
    """Auth supplied via extra_headers (gateway/proxy setups) is a working
    credential — the missing-credential fail-fast must not force it offline."""
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key=None,
            extra_headers={header_name: "Bearer gateway-token"},
            enable_atexit_flush=False,
        )
    )

    assert client.config.offline_mode is False
    assert "TRAIGENT_API_KEY" not in caplog.text
    client.close()


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"api_key": "   ", "jwt_token": None},
        {"api_key": None, "jwt_token": "   "},
        {"api_key": None, "jwt_token": None, "extra_headers": {"Authorization": ""}},
        {"api_key": None, "jwt_token": None, "extra_headers": {"X-API-Key": "   "}},
    ],
    ids=["blank-api-key", "blank-jwt", "empty-authorization", "blank-x-api-key"],
)
def test_observability_client_blank_credentials_do_not_bypass_guard(
    monkeypatch, caplog, config_kwargs
):
    """Whitespace-only credentials are as unauthenticated as missing ones."""
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            enable_atexit_flush=False,
            **config_kwargs,
        )
    )

    assert client._credential_egress_disabled is True
    assert "TRAIGENT_API_KEY" in caplog.text
    client.close()


@pytest.mark.parametrize(
    ("config_kwargs", "header_name", "expected_value"),
    [
        (
            {"api_key": "   ", "jwt_token": None},
            "X-API-Key",
            "valid-header-key",
        ),
        (
            {"api_key": None, "jwt_token": "   "},
            "Authorization",
            "Bearer valid-header-token",
        ),
    ],
    ids=["blank-api-key-vs-header", "blank-jwt-vs-header"],
)
def test_blank_explicit_credentials_do_not_overwrite_header_auth(
    monkeypatch, caplog, config_kwargs, header_name, expected_value
):
    """A blank explicit credential must behave as absent end to end: the guard
    keeps egress enabled because extra_headers carries working auth, and
    build_headers() must NOT overwrite that auth with the blank value."""
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    config = ObservabilityConfig(
        backend_origin="http://localhost:5000",
        enable_atexit_flush=False,
        extra_headers={header_name: expected_value},
        **config_kwargs,
    )
    client = ObservabilityClient(config)

    headers = config.build_headers()
    assert headers[header_name] == expected_value
    assert client._credential_egress_disabled is False
    assert "TRAIGENT_API_KEY" not in caplog.text
    client.close()


def test_observability_client_no_credential_sender_only_keeps_ingest_lane(
    monkeypatch, caplog
):
    """A custom sender owns ingest delivery, so the missing-credential guard
    must keep it working while blocking the un-overridden control-plane lane
    from emitting unauthenticated network requests."""
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)

    sent_batches: list[list[dict]] = []
    http_attempts = {"count": 0}

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key=None,
            jwt_token=None,
            enable_atexit_flush=False,
        ),
        sender=sent_batches.append,
    )

    def fail_if_called(*args, **kwargs):
        http_attempts["count"] += 1
        raise AssertionError("network egress attempted without a credential")

    monkeypatch.setattr(client._http_opener, "open", fail_if_called)

    trace_id = client.start_trace("sender-only-probe", trace_id="trace_sender_only")
    client.end_trace(trace_id)
    result = client.flush()

    with pytest.raises(ClientError, match="no credential resolved"):
        client.list_sessions()

    client.close()
    assert client._credential_egress_disabled is True
    assert result.success is True
    assert result.items_sent >= 1
    assert any(
        trace["id"] == "trace_sender_only" for batch in sent_batches for trace in batch
    )
    assert http_attempts["count"] == 0
    assert "TRAIGENT_API_KEY" in caplog.text


def test_observability_client_no_credential_request_sender_only_keeps_control_lane(
    monkeypatch, caplog
):
    """A custom request_sender owns control-plane calls, so the guard must keep
    it working while suppressing the un-overridden network ingest lane."""
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)

    request_calls: list[tuple[str, str]] = []
    http_attempts = {"count": 0}

    canned_response = {"ok": True}

    def request_sender(method: str, path: str, payload: dict | None):
        request_calls.append((method, path))
        return canned_response

    caplog.set_level(logging.WARNING, logger="traigent.observability.client")
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key=None,
            jwt_token=None,
            enable_atexit_flush=False,
        ),
        request_sender=request_sender,
    )

    def fail_if_called(*args, **kwargs):
        http_attempts["count"] += 1
        raise AssertionError("network egress attempted without a credential")

    monkeypatch.setattr(client._http_opener, "open", fail_if_called)

    trace_id = client.start_trace("request-sender-only-probe", trace_id="trace_rs_only")
    client.end_trace(trace_id)
    result = client.flush()
    # Exercise the control-plane dispatch directly: the override must be
    # consulted (list_* wrappers add response-shape parsing that is not what
    # this test pins).
    response = client._request_json("GET", "/sessions")
    client.close()

    assert client._credential_egress_disabled is True
    assert result.success is True
    assert result.items_sent == 0
    assert response == canned_response
    assert request_calls == [("GET", "/sessions")]
    assert http_attempts["count"] == 0
    assert "TRAIGENT_API_KEY" in caplog.text


def test_observation_dto_rejects_negative_values():
    with pytest.raises(ValueError, match="input_tokens"):
        ObservationDTO(
            id="obs_bad",
            type=ObservationType.SPAN,
            name="bad-observation",
            input_tokens=-1,
        )


@pytest.mark.parametrize("status", ["", "cancelled", "x" * 65])
def test_observation_dto_rejects_invalid_status(status):
    with pytest.raises(ValueError, match="status"):
        ObservationDTO(
            id="obs_bad_status",
            type=ObservationType.SPAN,
            name="bad-observation",
            status=status,
        )


@pytest.mark.parametrize("status", ["", "cancelled", "x" * 65])
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


def test_observability_ingest_http_error_attaches_retry_after(monkeypatch):
    http_error = error.HTTPError(
        url="http://localhost:5000",
        code=429,
        msg="rate limited",
        hdrs={"Retry-After": "4.25"},
        fp=io.BytesIO(b'{"error":"rate limited"}'),
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
        client,
        "_open_http_request",
        lambda http_request: (_ for _ in ()).throw(http_error),
    )

    with pytest.raises(ClientError) as exc_info:
        client._post_batch_sync([{"id": "trace_retry_after"}])

    client.close()

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 4.25
