from __future__ import annotations

import copy
import importlib
import io
import json
import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any
from urllib import error

import pytest

from traigent.cloud.async_batch_transport import BatchFlushResult
from traigent.config.context import ConfigurationContext, TrialContext
from traigent.observability import (
    CorrelationIds,
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
from traigent.observability.config import OBSERVABILITY_CONTENT_MODES
from traigent.observability.decorators import (
    _current_client,
    set_default_observability_client,
)
from traigent.observability.dtos import ObservationDTO, TraceDTO
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    ContentDisabledError,
)

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
        def submit(self, trace_id, payload, *, deadline=None, state=None):
            del deadline, state
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
    """Trace payloads must be scrubbed before they reach the transport.

    Uses `content_mode="record"` throughout: this test is about the separate,
    pattern-based credential/PII scrubber (`redact_sensitive_data`) that runs
    on whatever content_mode lets through, not about content_mode gating
    itself (covered by `TestDirectClientCallsHonorContentMode`). Content_mode
    defaults to "metadata" (withhold), which would omit `input_data`/
    `output_data` entirely and give the pattern scrubber nothing to redact.
    """
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
        content_mode="record",
    )
    client.record_observation(
        trace_id,
        name="llm-call",
        observation_type=ObservationType.GENERATION,
        input_data={"prompt": "card 4111111111111111"},
        output_data={"answer": "Bearer canary.jwt.header.payload.signature"},
        metadata={"email": "alice@example.com"},
        content_mode="record",
    )
    client.end_trace(
        trace_id,
        output_data={"answer": FAKE_TRACE_API_KEY},
        content_mode="record",
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
        trace_id: str,
        payload: dict,
        *,
        deadline: float | None = None,
        state: Any = None,
    ) -> None:
        if not delayed_submission_thread:
            delayed_submission_thread.append(threading.get_ident())
            submit_entered.set()
            assert release_submit.wait(timeout=2.0)
        else:
            assert threading.get_ident() != delayed_submission_thread[0]
        original_submit(trace_id, payload, deadline=deadline, state=state)

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
        trace_id: str,
        payload: dict,
        *,
        deadline: float | None = None,
        state: Any = None,
    ) -> None:
        nonlocal submission_claimed
        with submission_claim_lock:
            is_first_submission = not submission_claimed
            submission_claimed = True
        if is_first_submission:
            first_submit_entered.set()
            assert release_first_submit.wait(timeout=2.0)
        original_submit(trace_id, payload, deadline=deadline, state=state)

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


def _make_recording_client(sender):
    return ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )


class TestDirectClientCallsHonorContentMode:
    """D3 regression: `content_mode` was enforced only inside the `observe`
    decorator/context manager. `ObservabilityClient.start_trace` /
    `record_observation` / `end_trace` did not gate their `input_data` /
    `output_data` arguments at all, so any caller instrumenting Traigent
    directly (without `observe`) shipped raw content regardless of the
    configured `content_mode`. These tests call the client methods directly,
    with no decorator anywhere in the call stack.
    """

    @pytest.mark.parametrize("mode", sorted(OBSERVABILITY_CONTENT_MODES))
    def test_direct_start_trace_and_end_trace_honor_content_mode(self, mode):
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive_input = {"prompt": "PATIENT diagnosis cancer stage 3"}
        sensitive_output = {"answer": "confirmed cancer stage 3, prescribe X"}

        trace_id = client.start_trace(
            "direct-trace-start",
            input_data=dict(sensitive_input),
            content_mode=mode,
        )
        client.end_trace(
            trace_id, output_data=dict(sensitive_output), content_mode=mode
        )

        result = client.flush()
        client.close()

        assert result.success is True
        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        blob = json.dumps(sent_batches)
        if mode == "record":
            assert trace_payload["input_data"] == sensitive_input
            assert trace_payload["output_data"] == sensitive_output
            assert "PATIENT diagnosis cancer stage 3" in blob
        elif mode == "redacted":
            assert trace_payload["input_data"] == {"redacted": True}
            assert trace_payload["output_data"] == {"redacted": True}
            assert "PATIENT diagnosis cancer stage 3" not in blob
        else:
            assert mode == "metadata"
            assert "input_data" not in trace_payload
            assert "output_data" not in trace_payload
            assert "PATIENT diagnosis cancer stage 3" not in blob

    @pytest.mark.parametrize("mode", sorted(OBSERVABILITY_CONTENT_MODES))
    def test_direct_record_observation_start_and_update_honor_content_mode(self, mode):
        """Covers both the initial (status=running) and the update/end call."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive_input = {"prompt": "PATIENT diagnosis cancer stage 3"}
        sensitive_output = {"answer": "confirmed cancer stage 3, prescribe X"}

        trace_id = client.start_trace("direct-observation-trace")
        observation_id = client.record_observation(
            trace_id,
            name="direct-op",
            observation_type=ObservationType.SPAN,
            status="running",
            input_data=dict(sensitive_input),
            content_mode=mode,
        )
        # Update/end call: a distinct entry point from the initial record.
        client.record_observation(
            trace_id,
            observation_id=observation_id,
            name="direct-op",
            status="completed",
            output_data=dict(sensitive_output),
            content_mode=mode,
        )

        result = client.flush()
        client.close()

        assert result.success is True
        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        blob = json.dumps(sent_batches)
        if mode == "record":
            assert observation["input_data"] == sensitive_input
            assert observation["output_data"] == sensitive_output
            assert "PATIENT diagnosis cancer stage 3" in blob
        elif mode == "redacted":
            assert observation["input_data"] == {"redacted": True}
            assert observation["output_data"] == {"redacted": True}
            assert "PATIENT diagnosis cancer stage 3" not in blob
        else:
            assert mode == "metadata"
            assert "input_data" not in observation
            assert "output_data" not in observation
            assert "PATIENT diagnosis cancer stage 3" not in blob

    @pytest.mark.parametrize("mode", sorted(OBSERVABILITY_CONTENT_MODES))
    def test_decorated_and_direct_paths_agree_on_content_mode(self, mode):
        """The `observe` decorator and a hand-written direct call must reach
        the exact same wire shape for the same `content_mode` -- proving the
        gate is one shared enforcement point, not two independent ones."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        secret = "top-secret-value"

        @observe("decorated-call", client=client, content_mode=mode)
        def decorated(value):
            return {"answer": f"derived from {value}"}

        decorated(secret)

        direct_trace_id = client.start_trace(
            "direct-call",
            input_data={"args": [secret], "kwargs": {}},
            content_mode=mode,
        )
        client.record_observation(
            direct_trace_id,
            name="direct-call",
            observation_type=ObservationType.SPAN,
            status="completed",
            input_data={"args": [secret], "kwargs": {}},
            output_data={"answer": f"derived from {secret}"},
            content_mode=mode,
        )
        client.end_trace(
            direct_trace_id,
            output_data={"answer": f"derived from {secret}"},
            content_mode=mode,
        )

        result = client.flush()
        client.close()
        assert result.success is True

        all_traces = [t for batch in sent_batches for t in batch]
        decorated_trace = next(t for t in all_traces if t["name"] == "decorated-call")
        direct_trace = next(t for t in all_traces if t["name"] == "direct-call")

        def _content_shape(trace: dict) -> tuple:
            observation = trace["observations"][0]
            return (
                trace.get("input_data"),
                trace.get("output_data"),
                observation.get("input_data"),
                observation.get("output_data"),
            )

        assert _content_shape(decorated_trace) == _content_shape(direct_trace)

    @pytest.mark.parametrize(
        ("content_mode", "expected_error_message", "content_should_ship"),
        [
            (None, None, False),
            ("redacted", "[REDACTED]", False),
            ("record", "parse failed on: PATIENT diagnosis cancer stage 3", True),
        ],
    )
    def test_direct_record_observation_error_honors_content_mode(
        self, content_mode, expected_error_message, content_should_ship
    ):
        """A direct `record_observation(..., error=...)` call -- no decorator
        anywhere -- must gate the exception message exactly like `observe`."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("direct-error-trace")
        client.record_observation(
            trace_id,
            name="direct-op",
            status="failed",
            error=ValueError(f"parse failed on: {sensitive}"),
            content_mode=content_mode,
        )
        client.end_trace(trace_id, status="failed")

        result = client.flush()
        client.close()

        assert result.success is True
        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        metadata = observation["metadata"]
        assert observation["status"] == "failed"
        assert metadata["error_type"] == "ValueError"
        if expected_error_message is None:
            assert "error_message" not in metadata
        else:
            assert metadata["error_message"] == expected_error_message
        assert (sensitive in json.dumps(sent_batches)) is content_should_ship

    def test_retries_resend_the_same_already_gated_payload(self, monkeypatch):
        """Content is gated once, at record time, into the stored DTO
        snapshot. A retry re-sends that exact snapshot rather than
        re-deriving a payload from raw caller data, so a retry cannot
        reintroduce content the first attempt withheld."""
        monkeypatch.setattr(retry_module.time, "sleep", lambda _delay: None)

        attempts: list[list[dict]] = []
        call_count = 0

        def flaky_sender(traces):
            nonlocal call_count
            call_count += 1
            attempts.append(copy.deepcopy(traces))
            if call_count < 3:
                raise ClientError("transient failure", status_code=503)

        client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=3600.0,
                max_queue_size=10,
            ),
            sender=flaky_sender,
        )
        client._transport._retry_handler = retry_module.RetryHandler(
            retry_module.RetryConfig(
                max_attempts=5,
                initial_delay=0.0,
                max_delay=0.0,
                jitter=False,
                retry_on_status={503},
            )
        )

        sensitive = "PATIENT diagnosis cancer stage 3"
        # Default content_mode ("metadata") withholds content at record time,
        # before the transport (and therefore before any retry) ever sees it.
        trace_id = client.start_trace("retry-trace", input_data={"prompt": sensitive})
        client.end_trace(trace_id, output_data={"answer": sensitive})

        with client._lock:
            payload = client._trace_states[trace_id].to_payload()

        # `_send_batch` returns queued health events, which are empty on a
        # plain success with no warnings; the retry itself is evidenced by
        # `call_count` and the retry-attempt log records above.
        client._transport._send_batch([(trace_id, payload)])

        # Drain (without sending) the snapshot `start_trace`/`end_trace`
        # already queued on the buffer, so close() below does not trigger
        # a second, uncounted send of the same already-gated payload.
        with client._transport._lock:
            client._transport._buffer.clear()
        client.close()

        assert call_count >= 3, "sender was not actually retried"
        combined = json.dumps(attempts)
        assert sensitive not in combined
        for attempt_traces in attempts:
            sent_trace = next(t for t in attempt_traces if t["id"] == trace_id)
            assert "input_data" not in sent_trace
            assert "output_data" not in sent_trace

    def test_direct_calls_do_not_mutate_caller_supplied_payloads(self):
        """The content gate must never mutate the caller's own objects."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        input_payload = {"prompt": "secret prompt", "nested": {"a": 1}}
        output_payload = {"answer": "secret answer"}
        input_snapshot = copy.deepcopy(input_payload)
        output_snapshot = copy.deepcopy(output_payload)

        trace_id = client.start_trace(
            "no-mutation-trace", input_data=input_payload, content_mode="redacted"
        )
        client.record_observation(
            trace_id,
            name="op",
            status="completed",
            input_data=input_payload,
            output_data=output_payload,
            content_mode="redacted",
        )
        client.end_trace(trace_id, output_data=output_payload, content_mode="redacted")

        result = client.flush()
        client.close()

        assert result.success is True
        assert input_payload == input_snapshot
        assert output_payload == output_snapshot

    @pytest.mark.parametrize("mode", sorted(OBSERVABILITY_CONTENT_MODES))
    def test_content_mode_withholds_content_but_preserves_structural_fields(self, mode):
        """Gating content must never touch IDs, nesting, timestamps, token
        counts, or cost."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive_input = {"prompt": "sensitive"}
        sensitive_output = {"answer": "sensitive answer"}

        trace_id = client.start_trace(
            "structural-trace",
            session_id="session-xyz",
            user_id="user-1",
            custom_trace_id="custom-42",
            input_data=dict(sensitive_input),
            content_mode=mode,
        )
        parent_id = client.record_observation(
            trace_id,
            name="parent-span",
            observation_type=ObservationType.SPAN,
            status="running",
            input_data=dict(sensitive_input),
            content_mode=mode,
        )
        child_id = client.record_observation(
            trace_id,
            name="child-generation",
            observation_type=ObservationType.GENERATION,
            parent_observation_id=parent_id,
            status="completed",
            ended_at=datetime.now(UTC),
            input_tokens=12,
            output_tokens=34,
            cost_usd=0.0042,
            input_data=dict(sensitive_input),
            output_data=dict(sensitive_output),
            content_mode=mode,
        )
        client.record_observation(
            trace_id,
            observation_id=parent_id,
            name="parent-span",
            status="completed",
            output_data=dict(sensitive_output),
            content_mode=mode,
        )
        client.end_trace(
            trace_id, output_data=dict(sensitive_output), content_mode=mode
        )

        result = client.flush()
        client.close()
        assert result.success is True

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert trace_payload["id"] == trace_id
        assert trace_payload["session_id"] == "session-xyz"
        assert trace_payload["user_id"] == "user-1"
        assert trace_payload["custom_trace_id"] == "custom-42"

        [root_obs] = trace_payload["observations"]
        assert root_obs["id"] == parent_id
        [child_obs] = root_obs["children"]
        assert child_obs["id"] == child_id
        assert child_obs["parent_observation_id"] == parent_id
        assert child_obs["input_tokens"] == 12
        assert child_obs["output_tokens"] == 34
        assert child_obs["total_tokens"] == 46
        assert child_obs["cost_usd"] == 0.0042
        assert isinstance(child_obs["started_at"], str)
        assert isinstance(child_obs["ended_at"], str)

        assert ("sensitive" in json.dumps(sent_batches)) == (mode == "record")


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
            # M3: `add_comment` raises in the default `metadata` mode (it has
            # no non-content payload to fall back to), so this happy-path
            # wire-shape test needs `record` to exercise the full
            # comment/feedback flow. Metadata-mode raising and
            # redacted/record placeholder behavior are covered by dedicated
            # tests in `TestContentModeGatesCommentsAndFeedback`.
            content_mode="record",
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


def test_observability_client_withholds_non_serializable_feedback_correction_output():
    """B2 (addendum R3, A5 completion) supersedes the pre-fix behavior this
    test used to assert (`pytest.raises(ClientError, match="correction_
    output")`): astra's second REJECT confirmed that raising here broke the
    "no direct client call may raise from a hostile value" contract
    (client.py:2121). A non-JSON-serializable `correction_output` must now
    be withheld (omitted from the wire payload) and the call must succeed,
    exactly like any other scrub/gating failure -- see
    `TestB2FailureBoundaryCoversIntakeAndSerialization`."""
    captured: dict[str, Any] = {}

    def request_sender(method, path, payload):
        captured["payload"] = payload
        return {
            "data": {
                "feedback": {
                    "id": "feedback_1",
                    "trace_id": "trace_sdk",
                    "author_user_id": "sdk-user",
                    "rating": payload["rating"],
                    "comment": payload.get("comment"),
                    "correction_output": payload.get("correction_output"),
                    "created_at": "2026-03-10T14:12:00+00:00",
                    "updated_at": "2026-03-10T14:12:00+00:00",
                },
                "summary": {"up_count": 1, "down_count": 0},
            }
        }

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

    # Must not raise.
    client.submit_feedback(
        "trace_sdk", ThumbRating.UP, correction_output={"bad": {1, 2, 3}}
    )

    assert "correction_output" not in captured["payload"]
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
            # M3: `add_comment` raises locally in the default `metadata`
            # mode, before any request is sent -- this test is about the
            # request-layer 404, so it needs `record` to actually reach
            # `missing_trace_request_sender`.
            content_mode="record",
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


def test_observability_retry_exhaustion_runs_real_retry_handler(monkeypatch):
    """A permanently failing sender is attempted exactly at the configured cap."""
    attempts = 0

    def failing_sender(_payloads):
        nonlocal attempts
        attempts += 1
        raise ClientError("observability unavailable", status_code=503)

    monkeypatch.setattr(retry_module.time, "sleep", lambda _delay: None)
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=failing_sender,
    )
    client._transport._retry_handler = retry_module.RetryHandler(
        retry_module.RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            max_delay=0.0,
            jitter=False,
            retry_on_status={503},
        )
    )

    events = client._transport._send_batch([("trace_retry_exhausted", {"id": "trace"})])
    stats = client._transport.get_stats()
    client.close()

    assert attempts == 3
    assert stats["failed_batches"] == 1
    assert stats["dropped_items"] == 1
    assert stats["errors"]
    assert events


# --------------------------------------------------------------------------
# SDK privacy parity stage 1 (spec-sdk-privacy-stage1.md, Traigent#2452)
# --------------------------------------------------------------------------


class TestM2ContentModePrecedence:
    """M2: most-restrictive resolution across client-level sources, and the
    per-call-may-only-tighten rule."""

    def test_env_beats_legacy_env_when_more_restrictive(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CONTENT", "metadata")
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", "true")
        _mock_public_backend_dns(monkeypatch)

        config = ObservabilityConfig(backend_origin="https://auth.example.com")

        assert config.content_mode == "metadata"
        assert config.content_mode_explicit is True

    def test_legacy_env_beats_env_when_more_restrictive(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CONTENT", "record")
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", "false")
        _mock_public_backend_dns(monkeypatch)

        config = ObservabilityConfig(backend_origin="https://auth.example.com")

        # Legacy false -> metadata, more restrictive than env's "record".
        assert config.content_mode == "metadata"

    def test_legacy_capture_content_true_maps_to_record(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", "true")
        _mock_public_backend_dns(monkeypatch)

        config = ObservabilityConfig(backend_origin="https://auth.example.com")

        assert config.content_mode == "record"
        assert config.content_mode_explicit is True

    def test_constructor_value_is_tightened_by_more_restrictive_env(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CONTENT", "metadata")
        _mock_public_backend_dns(monkeypatch)

        config = ObservabilityConfig(
            backend_origin="https://auth.example.com", content_mode="record"
        )

        # An explicit constructor "record" must not silently ignore a more
        # restrictive env source (the historical bug: a default_factory only
        # ran when content_mode was omitted, so an explicit value bypassed
        # env resolution entirely).
        assert config.content_mode == "metadata"

    def test_empty_string_env_raises(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CONTENT", "")
        _mock_public_backend_dns(monkeypatch)

        with pytest.raises(ValueError, match="TRAIGENT_OBSERVABILITY_CONTENT"):
            ObservabilityConfig(backend_origin="https://auth.example.com")

    def test_empty_string_constructor_value_raises(self, monkeypatch):
        _mock_public_backend_dns(monkeypatch)

        with pytest.raises(ValueError, match="content_mode"):
            ObservabilityConfig(
                backend_origin="https://auth.example.com", content_mode=""
            )

    def test_per_call_cannot_loosen_an_explicit_client_policy(self):
        """An explicit client-level `metadata` policy must never lose to a
        looser per-call `content_mode='record'` -- negative control: revert
        the `_resolve_content_mode` tightening branch and this fails."""
        sent_batches: list[list[dict]] = []
        client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=0.1,
                max_queue_size=10,
                content_mode="metadata",
            ),
            sender=lambda traces: sent_batches.append(traces),
        )
        sensitive = "PATIENT diagnosis cancer stage 3"

        client.start_trace(
            "explicit-policy-trace",
            input_data={"prompt": sensitive},
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert "input_data" not in trace_payload
        assert sensitive not in json.dumps(sent_batches)

    def test_per_call_may_pick_any_mode_when_client_has_no_explicit_policy(self):
        """When nothing was explicitly configured at the client level (the
        bare 'metadata' default), a per-call `content_mode=` is the
        documented, normal way to opt a single call into looser capture --
        this is NOT the "loosening an explicit policy" case M2 forbids, it
        is the opt-in mechanism the whole feature exists for. Ambiguity
        resolution: see the report for why this reading was chosen over a
        literal "never loosen the bare default either" reading."""
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
        assert client.config.content_mode_explicit is False
        sensitive = "PATIENT diagnosis cancer stage 3"

        client.start_trace(
            "opt-in-trace", input_data={"prompt": sensitive}, content_mode="record"
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["input_data"] == {"prompt": sensitive}

    def test_invalid_content_mode_raises_before_any_context_mutation(self):
        """M2: an invalid `observe(content_mode=...)` must raise at
        decorator creation, before `_current_client` is ever touched --
        negative control: reverting the `__init__`-time validation in
        `_ObserveFactory`/`ObserveContext` (restoring the old
        `__enter__`-time check) makes this fail because
        `_current_client.get()` would then be corrupted."""
        baseline = _current_client.get()

        with pytest.raises(ValueError, match="content_mode"):
            observe("bad-mode", content_mode="not-a-real-mode")

        assert _current_client.get() is baseline

    def test_empty_string_content_mode_raises_on_observe_context(self):
        """The historical bug: `ObserveContext` used `if content_mode else
        None` (Python truthiness), which silently turned `""` into `None`
        (falling back to the client default) instead of raising."""
        baseline = _current_client.get()

        with pytest.raises(ValueError, match="content_mode"):
            ObserveContext(name="empty-mode", content_mode="")

        assert _current_client.get() is baseline

    def test_setup_failure_restores_context_variables(self, monkeypatch):
        """M4: a setup-time failure (here: `record_observation` raising for
        an unrelated reason) must restore every context variable `__enter__`
        touched -- `__exit__` is never invoked when `__enter__` raises, so
        nothing else would reset `_current_client`."""
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
        baseline = _current_client.get()

        def _boom(*args, **kwargs):
            raise RuntimeError("setup failed")

        monkeypatch.setattr(client, "record_observation", _boom)

        with pytest.raises(RuntimeError, match="setup failed"):
            with observe("will-fail-setup", client=client):
                pass  # pragma: no cover - never reached

        assert _current_client.get() is baseline
        client.close()


class TestM3CommentsAndFeedback:
    """M3: `add_comment` / `submit_feedback` honor `content_mode`."""

    def _client(self, *, content_mode: str, request_sender):
        return ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=0.1,
                max_queue_size=10,
                content_mode=content_mode,
            ),
            sender=lambda traces: None,
            request_sender=request_sender,
        )

    def test_add_comment_raises_content_disabled_in_metadata_mode(self):
        calls: list[tuple[str, str, dict | None]] = []
        client = self._client(
            content_mode="metadata",
            request_sender=lambda m, p, payload: (
                calls.append((m, p, payload)) or {"data": {}}
            ),
        )

        with pytest.raises(ContentDisabledError, match="metadata"):
            client.add_comment("trace_1", "some review comment")

        assert calls == []
        client.close()

    def test_add_comment_sends_placeholder_in_redacted_mode(self):
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "id": "c1",
                    "trace_id": "trace_1",
                    "author_user_id": "sdk-user",
                    "content": payload["content"],
                    "created_at": "2026-03-10T14:11:00+00:00",
                    "updated_at": "2026-03-10T14:11:00+00:00",
                }
            }

        client = self._client(content_mode="redacted", request_sender=request_sender)
        client.add_comment("trace_1", "Ship this after QA review.")
        client.close()

        assert captured["payload"] == {"content": "[REDACTED]"}

    def test_add_comment_sends_scrubbed_text_in_record_mode(self):
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "id": "c1",
                    "trace_id": "trace_1",
                    "author_user_id": "sdk-user",
                    "content": payload["content"],
                    "created_at": "2026-03-10T14:11:00+00:00",
                    "updated_at": "2026-03-10T14:11:00+00:00",
                }
            }

        client = self._client(content_mode="record", request_sender=request_sender)
        client.add_comment("trace_1", "Contact alice@example.com for details.")
        client.close()

        assert captured["payload"] == {
            "content": "Contact [REDACTED:email] for details."
        }

    def test_submit_feedback_omits_comment_and_correction_in_metadata_mode(self):
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_1",
                        "author_user_id": "sdk-user",
                        "rating": payload["rating"],
                        "comment": payload.get("comment"),
                        "correction_output": payload.get("correction_output"),
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                    "summary": {"up_count": 1, "down_count": 0},
                }
            }

        client = self._client(content_mode="metadata", request_sender=request_sender)
        client.submit_feedback(
            "trace_1",
            ThumbRating.UP,
            comment="Approved",
            correction_output={"answer": "corrected"},
        )
        client.close()

        assert captured["payload"] == {"rating": "up"}

    def test_submit_feedback_sends_placeholders_in_redacted_mode(self):
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_1",
                        "author_user_id": "sdk-user",
                        "rating": payload["rating"],
                        "comment": payload.get("comment"),
                        "correction_output": payload.get("correction_output"),
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                    "summary": {"up_count": 1, "down_count": 0},
                }
            }

        client = self._client(content_mode="redacted", request_sender=request_sender)
        client.submit_feedback(
            "trace_1",
            ThumbRating.UP,
            comment="Approved",
            correction_output={"answer": "corrected"},
        )
        client.close()

        assert captured["payload"] == {
            "rating": "up",
            "comment": "[REDACTED]",
            "correction_output": {"redacted": True},
        }

    def test_submit_feedback_scrubs_secrets_in_record_mode(self):
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_1",
                        "author_user_id": "sdk-user",
                        "rating": payload["rating"],
                        "comment": payload.get("comment"),
                        "correction_output": payload.get("correction_output"),
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                    "summary": {"up_count": 1, "down_count": 0},
                }
            }

        client = self._client(content_mode="record", request_sender=request_sender)
        client.submit_feedback(
            "trace_1",
            ThumbRating.UP,
            comment="Reviewer email: bob@example.com",
            correction_output={
                "answer": "corrected",
                "api_key": "sk-shouldnotship",  # pragma: allowlist secret
            },
        )
        client.close()

        assert captured["payload"]["comment"] == "Reviewer email: [REDACTED:email]"
        assert captured["payload"]["correction_output"] == {
            "answer": "corrected",
            "api_key": "[REDACTED]",
        }


class TestA1TraceScopedContentMode:
    """ADDENDUM R2 A1: an observation/end_trace/comment/feedback call
    resolves its `content_mode` against the TRACE's own resolved mode (fixed
    at `start_trace`), not straight against the client config. Finding 8's
    confirmed probe: starting a `redacted` trace and recording an
    observation with no per-call override produced a placeholder in TS and
    an omission in Python -- these tests pin the now-matching behavior."""

    def test_unannotated_observation_inherits_the_explicit_trace_mode(self):
        """Negative control: reverting `record_observation` to call
        `_resolve_content_mode` (client-level only) instead of
        `_resolve_trace_scoped_content_mode` makes this fail -- the
        unconfigured client's default (`metadata`) would omit `input_data`
        instead of emitting the trace's own `redacted` placeholder."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace("redacted-trace", content_mode="redacted")
        client.record_observation(trace_id, name="op", input_data={"prompt": "secret"})

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        observation = trace_payload["observations"][0]
        assert observation["input_data"] == {"redacted": True}

    def test_per_call_override_may_tighten_the_explicit_trace_mode(self):
        """An explicit trace mode (`content_mode_locked`) may be tightened
        by a stricter per-call override, exactly like the client-level
        rule."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("record-trace", content_mode="record")
        client.record_observation(
            trace_id,
            name="op",
            input_data={"prompt": sensitive},
            content_mode="metadata",
        )

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        observation = trace_payload["observations"][0]
        assert "input_data" not in observation
        assert sensitive not in json.dumps(sent_batches)

    def test_per_call_override_cannot_loosen_the_explicit_trace_mode(self):
        """The trace's own explicit mode is a floor a per-call override may
        not loosen -- the same M2 rule, applied at the trace level."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("metadata-trace", content_mode="metadata")
        client.record_observation(
            trace_id,
            name="op",
            input_data={"prompt": sensitive},
            content_mode="record",
        )

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        observation = trace_payload["observations"][0]
        assert "input_data" not in observation
        assert sensitive not in json.dumps(sent_batches)

    def test_per_call_override_used_as_is_when_neither_side_is_explicit(self):
        """When neither the client nor the trace's own mode was set
        explicitly, a per-call override is the normal opt-in path and is
        used as-is -- identical to the client-level bare-default case."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("unconfigured-trace")
        client.record_observation(
            trace_id,
            name="op",
            input_data={"prompt": sensitive},
            content_mode="record",
        )

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        observation = trace_payload["observations"][0]
        assert observation["input_data"] == {"prompt": sensitive}

    def test_untracked_trace_id_falls_back_to_client_level_resolution(self):
        """A1: "Top-level ... addComment/submitFeedback without a trace use
        the client-level result" -- a trace_id this client never locally
        tracked (e.g. created by another process) must resolve exactly like
        a bare client-level call, not raise or silently pick `metadata`."""
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "id": "c1",
                    "trace_id": "untracked-trace",
                    "author_user_id": "sdk-user",
                    "content": payload["content"],
                    "created_at": "2026-03-10T14:11:00+00:00",
                    "updated_at": "2026-03-10T14:11:00+00:00",
                }
            }

        client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=0.1,
                max_queue_size=10,
                content_mode="record",
            ),
            sender=lambda traces: None,
            request_sender=request_sender,
        )

        client.add_comment("untracked-trace", "Contact alice@example.com for details.")
        client.close()

        assert captured["payload"] == {
            "content": "Contact [REDACTED:email] for details."
        }

    def test_add_comment_content_mode_parameter_tightens_per_call(self):
        """Python/TS parity (A1, finding 8): `add_comment` now accepts a
        per-call `content_mode`, resolved the same way as
        `record_observation` -- a `record`-mode trace can still be
        tightened to `metadata` (which refuses locally, M3) for one
        specific comment call."""
        client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=0.1,
                max_queue_size=10,
            ),
            sender=lambda traces: None,
            request_sender=lambda m, p, payload: {"data": {}},
        )

        trace_id = client.start_trace("record-trace-for-comment", content_mode="record")
        with pytest.raises(ContentDisabledError):
            client.add_comment(trace_id, "some review comment", content_mode="metadata")

        client.close()


class TestM4UpdateTightening:
    """M4: NOT_SUPPLIED vs. explicit-null vs. withheld; tightening on update
    always wins; mutation after enqueue never leaks; retries resend the same
    gated snapshot (covered separately by
    `test_retries_resend_the_same_already_gated_payload`)."""

    def test_metadata_update_clears_earlier_record_mode_content(self):
        """Negative control: reverting `record_observation`'s NOT_SUPPLIED
        fix (restoring `input_data if input_data is not None else
        existing.input_data`) makes this fail -- the update would silently
        keep the record-mode content.

        The wire value is the explicit `{"redacted": true}` placeholder, not
        an omitted key: the backend's `apply_updates` skips an omitted/`None`
        field on update and keeps whatever it already has stored, so merely
        omitting would not clear the record-mode content this SAME
        observation already shipped it (ingest-contract parity fix,
        `wire_value_for_tightened_update`; see traigent-js cf2b700da)."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("tighten-trace")
        observation_id = client.record_observation(
            trace_id,
            name="op",
            status="running",
            input_data={"prompt": sensitive},
            output_data={"answer": sensitive},
            content_mode="record",
        )
        # A later call, in the (default) metadata mode, supplies NEW content
        # for the same fields -- tightening must win, not retain the
        # earlier record-mode content.
        client.record_observation(
            trace_id,
            observation_id=observation_id,
            name="op",
            status="completed",
            input_data={"prompt": sensitive},
            output_data={"answer": sensitive},
        )

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        assert observation["input_data"] == {"redacted": True}
        assert observation["output_data"] == {"redacted": True}
        assert sensitive not in json.dumps(sent_batches)

    def test_end_trace_output_update_tightens_after_earlier_record_mode_content(self):
        """Same bug, at the trace level: `end_trace`'s own output_data merge
        had the identical `if output_data is not None` conflation. Same wire
        contract fix applies: the placeholder, not an omitted key -- see the
        sibling test above.

        ADDENDUM R2 A1 changed WHAT a plain, un-annotated `end_trace(...)`
        call resolves to: it now continues the TRACE's own base mode (fixed
        at `start_trace`, here explicitly `"record"`), not the CLIENT's
        config default -- so an `end_trace` call that supplies no
        `content_mode` of its own no longer tightens anything (there is
        nothing to tighten against; see
        `test_untightened_end_trace_continues_the_explicit_trace_mode`
        directly below for that case). Tightening now requires the SAME
        thing it always required at the observation level: an explicit,
        stricter per-call override -- which `content_mode_locked` (the
        trace's own mode was itself set explicitly) then permits to win over
        the trace's `"record"` base.
        """
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace(
            "tighten-end-trace",
            output_data={"answer": sensitive},
            content_mode="record",
        )
        # Explicit per-call tightening: the trace's own mode was itself set
        # explicitly (`content_mode="record"` at `start_trace`), so this
        # stricter override is allowed to win.
        client.end_trace(
            trace_id, output_data={"answer": sensitive}, content_mode="metadata"
        )

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert trace_payload["output_data"] == {"redacted": True}
        assert sensitive not in json.dumps(sent_batches)

    def test_untightened_end_trace_continues_the_explicit_trace_mode(self):
        """A1 regression guard: a plain `end_trace(...)` call with NO
        `content_mode` of its own must continue the trace's own explicitly-
        set base mode, not silently fall back to the client's config
        default -- negative control: reverting
        `_resolve_trace_scoped_content_mode` to the pre-fix
        `_resolve_content_mode` (which resolves straight against
        `self.config.content_mode`) makes this fail, because the
        unconfigured client's default is `"metadata"` and would withhold
        `output_data` here instead of sending it."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("continue-trace-mode", content_mode="record")
        client.end_trace(trace_id, output_data={"answer": sensitive})

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert trace_payload["output_data"] == {"answer": sensitive}

    def test_status_only_update_does_not_touch_existing_content(self):
        """NOT_SUPPLIED (the default): a call that never mentions
        input_data/output_data must leave previously recorded content alone."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace("status-only-trace", content_mode="record")
        observation_id = client.record_observation(
            trace_id,
            name="op",
            status="running",
            input_data={"prompt": "hello"},
            content_mode="record",
        )
        # Status-only update: input_data/output_data are not mentioned at all.
        client.record_observation(
            trace_id,
            observation_id=observation_id,
            name="op",
            status="completed",
            content_mode="record",
        )

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        assert observation["input_data"] == {"prompt": "hello"}
        assert observation["status"] == "completed"

    def test_mutation_after_enqueue_does_not_leak(self):
        """M4: gating/redaction happen at enqueue time on an immutable
        snapshot -- mutating the caller's own dict AFTER the call returns
        (but before flush actually sends it) must never change what gets
        sent."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        payload = {"prompt": "original"}
        trace_id = client.start_trace(
            "mutate-after-enqueue", input_data=payload, content_mode="record"
        )
        payload["prompt"] = "mutated-after-the-call-returned"
        payload["injected"] = "should never appear"

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert trace_payload["input_data"] == {"prompt": "original"}

    def test_error_message_tightening_never_merges_stale_text(self):
        """Finding 2/A2: a record-mode error, followed by a metadata-mode
        update to the SAME observation, must force `error_message` to
        `"[REDACTED]"` -- never let the merge (`merged_metadata.update(...)`)
        silently keep the earlier call's real exception text because the
        later call's `metadata` dict never mentioned the key at all.

        Negative control: reverting the merge-branch fix (deleting the
        `if error_supplied and gated_error_message is None and
        "error_message" in existing.metadata:` block in
        `record_observation`) makes this fail -- the sensitive exception
        text from the first call would still be present after the second.
        """
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("error-tighten-trace")
        observation_id = client.record_observation(
            trace_id,
            name="op",
            status="failed",
            content_mode="record",
            error=ValueError(sensitive),
        )
        # Same observation, metadata mode (tighter): must overwrite, not
        # merge the stale record-mode error text back in.
        client.record_observation(
            trace_id,
            observation_id=observation_id,
            name="op",
            status="failed",
            content_mode="metadata",
            error=ValueError(sensitive),
        )

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        assert observation["metadata"]["error_message"] == "[REDACTED]"
        assert observation["metadata"]["error_type"] == "ValueError"
        assert sensitive not in json.dumps(sent_batches)

    def test_first_write_metadata_mode_error_omits_error_message(self):
        """A2: a genuine FIRST write (no prior `error_message` stored at
        all) still omits cleanly in metadata mode -- there is nothing
        stored server-side yet to leak, so no placeholder is needed."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace("error-first-write-trace")
        client.record_observation(
            trace_id,
            name="op",
            status="failed",
            content_mode="metadata",
            error=ValueError("first failure"),
        )

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        observation = trace_payload["observations"][0]
        assert "error_message" not in observation["metadata"]
        assert observation["metadata"]["error_type"] == "ValueError"

    def test_nested_metadata_mutation_after_start_trace_does_not_leak(self):
        """Finding 7/A8: a shallow `dict(metadata or {})` only detaches the
        top-level dict -- a caller mutating a NESTED dict/list inside their
        own metadata after `start_trace` returns must never change what a
        later `flush()`/`close()` sends.

        Negative control: reverting `_detach_metadata` to `dict(metadata or
        {})` (a shallow copy) makes this fail -- the nested mutation below
        would still reach the sent payload.
        """
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        nested = {"level": "before"}
        client.start_trace("nested-metadata-trace", metadata={"nested": nested})
        nested["level"] = "after-the-call-returned"
        nested["injected"] = "should never appear"

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["metadata"]["nested"] == {"level": "before"}

    def test_session_metadata_mutation_after_start_trace_does_not_leak(self):
        """Finding 7/A8: `session.metadata` is a caller-owned bag too --
        mutating it after `start_trace` returns must never leak into a later
        snapshot, exactly like trace/observation metadata."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        session_metadata = {"plan": "before"}
        client.start_trace(
            "session-metadata-trace",
            session={"id": "sess_1", "metadata": session_metadata},
        )
        session_metadata["plan"] = "after-the-call-returned"

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["session"]["metadata"] == {"plan": "before"}

    def test_explicit_null_clear_in_record_mode_sends_placeholder_over_prior_content(
        self,
    ):
        """Finding 6/A2: the backend's `apply_updates` skips an
        omitted/`None` field and keeps whatever it already has stored, so a
        `record`-mode caller supplying an explicit `None` to CLEAR a field
        that previously carried real content must also force the
        placeholder -- not just a `metadata`-mode withhold. This is NOT
        privacy gating (the mode never changes); it is the same ingest-
        contract fix widened beyond `metadata`.

        Negative control: reverting `wire_value_for_tightened_update` to
        only force the placeholder when `effective_content_mode ==
        "metadata"` makes this fail -- the explicit-null clear in `record`
        mode would omit the field instead, and the backend would keep the
        stale content.
        """
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace(
            "explicit-null-clear-trace", content_mode="record"
        )
        observation_id = client.record_observation(
            trace_id,
            name="op",
            input_data={"prompt": sensitive},
            content_mode="record",
        )
        # Explicit clear: the caller supplies `None` itself (not
        # NOT_SUPPLIED), still in `record` mode -- a real "clear this field"
        # request, not a mode withholding content.
        client.record_observation(
            trace_id,
            observation_id=observation_id,
            name="op",
            input_data=None,
            content_mode="record",
        )

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        assert observation["input_data"] == {"redacted": True}
        assert sensitive not in json.dumps(sent_batches)

    def test_explicit_null_with_no_prior_content_is_omitted(self):
        """A2: "EXPLICIT_NULL with no prior content -> omit" (Python) -- a
        first write that supplies `None` has nothing stored server-side yet
        to leak, so it is fine to omit the field entirely rather than send a
        placeholder."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace("explicit-null-first-write-trace")
        client.record_observation(
            trace_id,
            name="op",
            input_data=None,
            content_mode="record",
        )

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        observation = trace_payload["observations"][0]
        assert "input_data" not in observation

    def test_observe_decorator_none_return_in_redacted_mode_emits_placeholder(self):
        """M4 fix: a decorated function that legitimately returns `None`
        under `content_mode='redacted'` must still emit the `{"redacted":
        true}` structural placeholder, not silently omit the field --
        `apply_content_mode` no longer special-cases a supplied `None`."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        @observe("returns-none", client=client, content_mode="redacted")
        def side_effect_only(value: str) -> None:
            return None

        assert side_effect_only("secret") is None

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        observation = trace_payload["observations"][0]
        assert trace_payload["output_data"] == {"redacted": True}
        assert observation["output_data"] == {"redacted": True}


class TestM5FullPipelineNumericExemption:
    """M5 through the full record/redact pipeline (not just the
    `traigent.security.redaction` unit -- see `test_text_redaction.py` for
    the exhaustive unit-level cases)."""

    def test_numeric_secret_under_metadata_key_is_masked_end_to_end(self):
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace(
            "numeric-secret-trace",
            metadata={"password": 123456, "credit_card": "4111111111111111"},
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["metadata"]["password"] == "[REDACTED]"
        assert trace_payload["metadata"]["credit_card"] == "[REDACTED]"

    def test_m5_canonical_root_counterexamples_end_to_end(self):
        """A3: the exact M5 counterexamples the addendum names (finding 5) --
        `passwd`, `jwt`, `bearer`, camelCase `privateKey`, and `authStuff`
        (a plausible non-secret-looking key that still substring-matches the
        `auth` root, by design -- see `redaction.py`'s
        `_redact_credential_key_value` docstring on the safe-side
        trade-off) -- through the full end-to-end pipeline, not just the
        `traigent.security.redaction` unit."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace(
            "m5-root-counterexamples-trace",
            metadata={
                "passwd": "hunter2",
                "jwt": "eyJhbGciOiJIUzI1NiJ9.secret.sig",
                "bearer": "some-bearer-value",
                "privateKey": "not-a-real-key-placeholder",
                "authStuff": "should also be masked",
            },
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        metadata = trace_payload["metadata"]
        assert metadata["passwd"] == "[REDACTED]"
        assert metadata["jwt"] == "[REDACTED]"
        assert metadata["bearer"] == "[REDACTED]"
        assert metadata["privateKey"] == "[REDACTED]"
        assert metadata["authStuff"] == "[REDACTED]"

    def test_usage_totaltokens_camelcase_is_exempt_like_snake_case(self):
        """A3: the counter exemption is keyed on NORMALIZED (separator-
        stripped) names, so `usage.totalTokens` (camelCase, as a caller-
        embedded provider-response blob might spell it) is exempt exactly
        like `usage.total_tokens` -- this is the specific mismatch finding 5
        called out (Python previously masked `usage.totalTokens` because
        `token` substring-matched it before the parent-scoped exemption
        applied)."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace(
            "usage-camelcase-trace",
            metadata={"usage": {"totalTokens": 70}},
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["metadata"]["usage"]["totalTokens"] == 70

    def test_sensitive_ancestor_dominates_counter_exemptions_end_to_end(self):
        """A3-dominance: `api_key.max_tokens` and
        `api_key.usage.total_tokens` are masked -- an ancestor key being
        credential-like (`api_key`) DOMINATES the usage-counter/model-
        parameter exemptions, which only apply "when no ancestor is
        sensitive".

        Negative control: reverting `_redact_credential_key_value` to check
        `_is_exempt_numeric_counter` before collapsing a credential subtree
        (i.e. re-applying the exemption INSIDE an already-flagged secret
        subtree) makes this fail -- both counters below would survive as
        plain numbers instead of "[REDACTED]".
        """
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace(
            "api-key-dominance-trace",
            metadata={
                "api_key": {
                    "max_tokens": 4096,
                    "usage": {"total_tokens": 12},
                }
            },
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        api_key_bag = trace_payload["metadata"]["api_key"]
        assert api_key_bag["max_tokens"] == "[REDACTED]"
        assert api_key_bag["usage"]["total_tokens"] == "[REDACTED]"

    def test_usage_counter_and_dto_token_fields_survive_end_to_end(self):
        """Both the approved `usage.*` exemption (for a caller-supplied
        provider-response blob embedded in metadata) and Traigent's own
        structural `input_tokens`/`output_tokens`/`total_tokens` DTO fields
        must survive the full pipeline -- negative control: reverting
        `_harden_content_bags`'s bag-scoping (applying
        `redact_credential_keys=True` to the whole payload again, as the
        original single-call-site code did) makes the DTO field assertions
        below fail with `'[REDACTED]' == 12`."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace(
            "usage-trace",
            metadata={
                "provider_response": {
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 20,
                        "total_tokens": 70,
                    }
                }
            },
            content_mode="record",
        )
        client.record_observation(
            trace_id,
            name="generation",
            observation_type=ObservationType.GENERATION,
            status="completed",
            input_tokens=12,
            output_tokens=34,
            total_tokens=46,
            cost_usd=0.0042,
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        usage = trace_payload["metadata"]["provider_response"]["usage"]
        assert usage == {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        }
        observation = trace_payload["observations"][0]
        assert observation["input_tokens"] == 12
        assert observation["output_tokens"] == 34
        assert observation["total_tokens"] == 46
        assert observation["cost_usd"] == 0.0042


class TestM6UserId:
    def test_trace_user_id_email_is_redacted(self):
        """M6: Python already scans the whole trace payload by value, so an
        email-shaped `user_id` is redacted like any other PII-shaped text --
        this is a regression guard, not a behavior change (TS is the SDK
        that needed the code fix; see the spec)."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace("user-id-trace", user_id="alice@example.com")
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["user_id"] == "[REDACTED:email]"


class TestA4KeyBasedScrubbingCoverage:
    """ADDENDUM R2 A4/finding 3: key-based scrubbing covers every arbitrary-
    data bag, not just trace/observation `metadata`/`input_data`/
    `output_data` -- session metadata and `prompt_reference.variables` are
    arbitrary caller-chosen bags too.

    Negative control: reverting `_harden_content_bags` to not inspect
    `session`/`prompt_reference` at all (the pre-fix version) makes both
    tests below fail -- a plain numeric secret like `password: 123456`
    reaches the sender unredacted because it never matches any VALUE
    pattern, only the key-name check catches it, and that check never ran
    on these bags."""

    def test_session_metadata_password_is_masked_end_to_end(self):
        """Finding 3's confirmed probe: `session.metadata.password` reached
        the custom sender unredacted."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace(
            "session-secret-trace",
            session={"id": "sess_1", "metadata": {"password": 123456}},
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["session"]["metadata"]["password"] == "[REDACTED]"

    def test_prompt_reference_variables_secret_is_masked_end_to_end(self):
        """`prompt_reference.variables` is an arbitrary `dict[str, Any]`
        (`PromptReferenceDTO.variables`), same category of bag as
        `metadata`."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace(
            "prompt-reference-secret-trace",
            prompt_reference=PromptReferenceDTO(
                name="greeting", variables={"api_key": "sk-shouldnotship"}
            ),
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["prompt_reference"]["variables"]["api_key"] == "[REDACTED]"


class TestM8FailClosed:
    def test_redaction_failure_withholds_content_and_never_raises(self, monkeypatch):
        """M8: if redaction/gating throws, withhold content and continue --
        never break the caller's application, never send unredacted."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        def _boom(*args, **kwargs):
            raise RuntimeError("redactor exploded")

        monkeypatch.setattr(
            "traigent.observability.client.redact_sensitive_data", _boom
        )

        # Must not raise out of the caller's own call.
        client.start_trace(
            "fail-closed-trace",
            input_data={"prompt": sensitive},
            metadata={"note": sensitive},
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert "input_data" not in trace_payload or trace_payload["input_data"] is None
        assert sensitive not in json.dumps(sent_batches)

    def test_redaction_failure_also_masks_name_and_user_id(self, monkeypatch):
        """A5: the safe-allowlisted fallback (not just the previous
        `input_data`/`output_data`/`metadata` bags) also masks `name` and
        `user_id` -- finding 4's confirmed probe: an injected-redactor
        failure still emitted raw email-shaped `name`/`user_id` because the
        old fallback (`_withhold_all_content_bags`) only stripped the three
        content bags and left every other field, including `name`/
        `user_id`, untouched.

        Negative control: reverting `_redact_trace_payload`'s except branch
        to call the old `_withhold_all_content_bags(payload)` instead of
        `_safe_allowlisted_snapshot_or_static_fallback(payload)` makes this
        fail -- `trace_payload["name"]` and `["user_id"]` would still be the
        raw email-shaped strings below.
        """
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        def _boom(*args, **kwargs):
            raise RuntimeError("redactor exploded")

        monkeypatch.setattr(
            "traigent.observability.client.redact_sensitive_data", _boom
        )

        client.start_trace(
            "alice@example.com's session",
            user_id="alice@example.com",
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["name"] == "[REDACTED]"
        assert trace_payload["user_id"] == "[REDACTED]"
        assert "alice@example.com" not in json.dumps(sent_batches)

    def test_cyclic_input_data_does_not_raise_and_is_withheld(self):
        """Finding 4's confirmed probe: a cyclic caller-supplied `input_data`
        structure raised `RecursionError` straight out of `start_trace`,
        because `state.to_payload()` (DTO serialization, via `to_jsonable`'s
        unbounded recursion) ran OUTSIDE `_redact_trace_payload`'s own
        try/except at every call site. Must not raise into the caller's
        application, and must not send the cyclic structure.

        Negative control: reverting `_queue_trace_snapshot`'s call from
        `_safe_trace_snapshot(state)` back to
        `_redact_trace_payload(state.to_payload())` makes this fail --
        `state.to_payload()` evaluates before `_redact_trace_payload`'s own
        try/except can ever run, so the `RecursionError` propagates straight
        out of `start_trace`.
        """
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        cyclic: dict[str, Any] = {"prompt": "hello"}
        cyclic["self"] = cyclic

        # Must not raise.
        trace_id = client.start_trace(
            "cyclic-input-trace", input_data=cyclic, content_mode="record"
        )

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert "input_data" not in trace_payload or trace_payload["input_data"] is None

    def test_add_comment_scrub_failure_withholds_and_does_not_raise(self, monkeypatch):
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "id": "c1",
                    "trace_id": "trace_1",
                    "author_user_id": "sdk-user",
                    "content": payload["content"],
                    "created_at": "2026-03-10T14:11:00+00:00",
                    "updated_at": "2026-03-10T14:11:00+00:00",
                }
            }

        client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=0.1,
                max_queue_size=10,
                content_mode="record",
            ),
            sender=lambda traces: None,
            request_sender=request_sender,
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("scrub exploded")

        monkeypatch.setattr(
            "traigent.observability.client.redact_sensitive_text", _boom
        )

        client.add_comment("trace_1", "a comment that would have been scrubbed")
        client.close()

        # A6: the backend's `TraceCommentCreateRequest.content` requires a
        # non-empty string -- `None` is not a withhold, it is an invalid
        # request the backend rejects. A scrub failure must fall back to the
        # `"[REDACTED]"` placeholder, never null/omitted.
        assert captured["payload"] == {"content": "[REDACTED]"}


class TestB3SerializationAndScrubbingFailureSendsPlaceholder:
    """ADDENDUM R3 B3 (A2 completion), astra REJECT #2 finding 1 (CONFIRMED,
    out-astra-privacy-r3.md). Distinct from
    `TestB3FailedUpdateOverPriorContentSendsPlaceholder` below (a COPY
    failure inside `apply_content_mode`, already fixed): this class covers
    the two failure modes finding 1 found still broken -- trace/observation
    SERIALIZATION (`state.to_payload()`/`to_jsonable`, a cyclic structure)
    and SCRUBBING (`redact_sensitive_data` itself raising) -- both of which
    happen at snapshot-BUILD time (flush/close), decoupled from the
    record_observation/end_trace call that supplied the value, so they
    cannot rely on `apply_content_mode`'s own try/except at all.

    When gating/serialization/scrubbing FAILS on an UPDATE and the field
    previously carried content, the SDK must send the clearing placeholder
    (`{"redacted": true}`) for that field -- never omit it, because the
    backend's `apply_updates` skips omitted fields and keeps the OLD
    content. Deep-copy failures already did this; serialization
    (`state.to_payload()` / `to_jsonable`, client.py `_safe_trace_snapshot`)
    and scrubbing (`redact_sensitive_data`, `_redact_trace_payload`) failures
    did not: replacing previously-sent output with a cyclic value produced a
    minimal snapshot WITHOUT `output_data`; a cyclic OBSERVATION update
    omitted the entire observation; an injected scrubber exception on an
    update over prior content likewise omitted the clearing field.

    Fix: `_TraceState.content_history` tracks, independently of the
    (possibly hostile) value currently held, whether a trace/observation
    content field carried content BEFORE the update that is now failing to
    build. `_safe_allowlisted_snapshot`/`_safe_allowlisted_observation_
    snapshot` consult it to emit the placeholder instead of dropping the
    field, and never drop the observation itself.

    Negative control (each test below): reverting `_safe_allowlisted_
    snapshot` to the pre-fix version (drop `input_data`/`output_data`/
    `observations` unconditionally, ignoring `state`) makes tests 1-3 below
    fail -- verified by hand for this report; see the worker report's
    control-lines table. Test 4 (first write, no prior content) continues to
    pass either way -- omission is still correct there (M1/A2): nothing is
    stored server-side yet to leak.
    """

    def test_1_cyclic_trace_output_update_over_prior_content_sends_placeholder(self):
        """(1) record-mode output sent, then an update with a cyclic output
        -> the payload carries `output_data == {"redacted": true}`."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace(
            "b3-cyclic-trace-output",
            output_data={"answer": "42"},
            content_mode="record",
        )
        client.flush()
        # Confirm the real content was actually delivered first, so the
        # later placeholder is a genuine CLEAR, not a first-write omission.
        assert sent_batches[-1][-1]["output_data"] == {"answer": "42"}

        cyclic: dict[str, Any] = {"answer": "hello"}
        cyclic["self"] = cyclic
        client.end_trace(trace_id, output_data=cyclic)
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["id"] == trace_id
        assert trace_payload["output_data"] == {"redacted": True}

    def test_2_cyclic_observation_output_update_never_omits_the_observation(self):
        """(2) same for an observation update with a cyclic value -> the
        observation is present with a placeholder, never dropped from the
        update entirely."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace(
            "b3-cyclic-observation-output", content_mode="record"
        )
        observation_id = client.record_observation(
            trace_id,
            name="step-1",
            output_data={"answer": "42"},
            content_mode="record",
        )
        client.flush()
        first_observations = sent_batches[-1][-1]["observations"]
        assert any(
            obs["id"] == observation_id and obs["output_data"] == {"answer": "42"}
            for obs in first_observations
        ), first_observations

        cyclic: dict[str, Any] = {"answer": "hello"}
        cyclic["self"] = cyclic
        client.record_observation(
            trace_id,
            name="step-1",
            observation_id=observation_id,
            output_data=cyclic,
        )
        client.flush()
        client.close()

        observations = sent_batches[-1][-1].get("observations", [])
        matching = [obs for obs in observations if obs.get("id") == observation_id]
        assert matching, (
            "the observation must never be dropped from the update entirely "
            f"-- got observations={observations!r}"
        )
        assert matching[0]["output_data"] == {"redacted": True}

    def test_3_scrubber_exception_on_update_over_prior_content_sends_placeholder(
        self, monkeypatch
    ):
        """(3) an injected scrubber exception on an update over prior
        content -> placeholder (no cyclic value involved at all -- this
        exercises the REDACTION failure path, `_redact_trace_payload`,
        distinctly from test 1's SERIALIZATION failure path,
        `_safe_trace_snapshot`'s own `state.to_payload()` call)."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace(
            "b3-scrubber-exception-trace",
            output_data={"answer": "42"},
            content_mode="record",
        )
        client.flush()
        assert sent_batches[-1][-1]["output_data"] == {"answer": "42"}

        def _boom(*args, **kwargs):
            raise RuntimeError("scrub exploded")

        monkeypatch.setattr(
            "traigent.observability.client.redact_sensitive_data", _boom
        )

        client.end_trace(trace_id, output_data={"answer": "a brand new answer"})
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["output_data"] == {"redacted": True}
        assert "a brand new answer" not in json.dumps(sent_batches)

    def test_4_first_write_failure_with_no_prior_content_may_omit(self):
        """(4) first write with a failure and no prior content -> the field
        may be omitted (regression guard for the two existing M8 tests
        covering this: `TestM8FailClosed.
        test_redaction_failure_withholds_content_and_never_raises` and
        `test_cyclic_input_data_does_not_raise_and_is_withheld`). Exercised
        here for `output_data` specifically, at both the trace and
        observation level, in one test."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        cyclic: dict[str, Any] = {"answer": "hello"}
        cyclic["self"] = cyclic

        trace_id = client.start_trace(
            "b3-first-write-trace", output_data=cyclic, content_mode="record"
        )
        observation_id = client.record_observation(
            trace_id, name="step-1", output_data=cyclic, content_mode="record"
        )
        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert (
            "output_data" not in trace_payload or trace_payload["output_data"] is None
        )
        observations = trace_payload.get("observations", [])
        matching = [obs for obs in observations if obs.get("id") == observation_id]
        if matching:
            assert (
                "output_data" not in matching[0] or matching[0]["output_data"] is None
            )


class TestB1OutboundCounterexamples:
    """ADDENDUM R3 B1 (+ R3.1, captain ruling): the exact named
    counterexamples, through the full end-to-end outbound pipeline
    (unit-level normalization/prefix-rule cases are in
    tests/unit/security/test_text_redaction.py).

    R3.1 negative control: removing `_EXACT_CREDENTIAL_KEY_NAMES`'s `pass`
    entry (or the `if normalized in _EXACT_CREDENTIAL_KEY_NAMES:` branch in
    `is_credential_key_name`, traigent/security/redaction.py) makes
    `metadata["pass"] == "[REDACTED]"` below fail -- `pass` would pass
    through unmasked (see the worker report's control-lines table).
    """

    def test_b1_named_counterexamples_end_to_end(self):
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        client.start_trace(
            "b1-counterexamples-trace",
            metadata={
                "nested_pwd": 123,
                "nested_access_key": 456,
                "oauthClient": "x",
                "auth_header": "should be masked",
                "AUTH-STUFF": "should be masked",
                "usage": {"total-tokens": 42},
                # R3.1: `pass` is an EXACT-match credential key name, not a
                # substring root -- `passenger`/`bypass` must NOT match.
                "pass": 13,
                "passenger": "x",
                "bypass": 1,
                # R3.1 (parity confirmation): `authorization` is not a
                # substring root in Python either -- only the `auth` PREFIX
                # rule applies, and neither of these normalized keys starts
                # with "auth" ("oauthauthorization" starts with "oauth";
                # "preauthorization" starts with "pre").
                "oauthAuthorization": "y",
                "pre_authorization": 12,
            },
            content_mode="record",
        )
        client.flush()
        client.close()

        metadata = sent_batches[-1][-1]["metadata"]
        assert metadata["nested_pwd"] == "[REDACTED]"
        assert metadata["nested_access_key"] == "[REDACTED]"
        # Not masked by the auth rule: "oauthClient" normalizes to
        # "oauthclient", which does not START with "auth" (B1: `auth` is a
        # PREFIX match only).
        assert metadata["oauthClient"] == "x"
        assert metadata["auth_header"] == "[REDACTED]"
        assert metadata["AUTH-STUFF"] == "[REDACTED]"
        # Counter exemption still applies with punctuation in the key:
        # normalized "totaltokens" (hyphen stripped) with immediate parent
        # normalized "usage".
        assert metadata["usage"]["total-tokens"] == 42
        # R3.1: exact match only.
        assert metadata["pass"] == "[REDACTED]"
        assert metadata["passenger"] == "x"
        assert metadata["bypass"] == 1
        # R3.1: `authorization` is not a substring root in Python.
        assert metadata["oauthAuthorization"] == "y"
        assert metadata["pre_authorization"] == 12


class TestB2FailureBoundaryCoversIntakeAndSerialization:
    """ADDENDUM R3 B2 (A5 completion): intake copying (deep copy), property
    traversal (a throwing getter/`__deepcopy__`), and serialization (incl.
    feedback) must all happen INSIDE the failure guard that produces the
    safe/withheld result -- no direct client call may raise from a hostile
    value. astra's second REJECT confirmed Python raised straight out of
    `start_trace`/`record_observation`/`submit_feedback` before any
    safe-snapshot guard ran: unguarded `copy.deepcopy` in
    `_detach_metadata`/`_coerce_session`/`_coerce_correlation_ids`/
    `_coerce_prompt_reference`, and feedback `correction_output`
    JSON-serializability validation raising `ClientError` (was
    client.py:137/2735/2744/2758/2121, pre-fix).

    Negative control (each case): reverting the corresponding `_coerce_*`/
    `_detach_metadata` helper to call `copy.deepcopy` directly (bypassing
    `_safe_deepcopy_or_default`), or reverting `submit_feedback`'s
    correction_output validation to call `_ensure_json_serializable`
    outside a try/except, makes the matching test below raise instead of
    pass -- verified by hand for this report (see the worker report's
    control-lines table).
    """

    def test_metadata_with_uncopyable_value_does_not_raise(self):
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        # Must not raise.
        trace_id = client.start_trace(
            "hostile-metadata-trace",
            metadata={"lock": threading.Lock(), "safe": "kept"},
        )
        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        # The whole metadata bag is withheld rather than raising or
        # partially leaking a sibling of the hostile value.
        assert trace_payload.get("metadata", {}) == {}

    def test_session_with_throwing_getter_does_not_raise(self):
        class _Hostile:
            def __deepcopy__(self, memo):
                raise RuntimeError("hostile getter exploded")

        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        # Must not raise.
        trace_id = client.start_trace(
            "hostile-session-trace",
            session={"id": "sess_1", "metadata": {"note": _Hostile()}},
        )
        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert trace_payload.get("session") is None

    def test_input_output_with_uncopyable_value_does_not_raise(self):
        """Regression guard: `apply_content_mode`'s own try/except already
        covered this before B2, but it belongs in the same probe set B2
        names explicitly."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        # Must not raise.
        trace_id = client.start_trace(
            "hostile-input-output-trace",
            input_data={"lock": threading.Lock()},
            output_data={"lock": threading.Lock()},
            content_mode="record",
        )
        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert trace_payload.get("input_data") is None
        assert trace_payload.get("output_data") is None

    def test_feedback_correction_output_json_serialization_failure_does_not_raise(
        self,
    ):
        captured: dict[str, Any] = {}

        def request_sender(method, path, payload):
            captured["payload"] = payload
            return {
                "data": {
                    "feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_1",
                        "author_user_id": "sdk-user",
                        "rating": payload["rating"],
                        "comment": payload.get("comment"),
                        "correction_output": payload.get("correction_output"),
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                    "summary": {"up_count": 1, "down_count": 0},
                }
            }

        client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=0.1,
                max_queue_size=10,
                content_mode="record",
            ),
            sender=lambda traces: None,
            request_sender=request_sender,
        )

        # Must not raise -- previously raised `ClientError` straight out of
        # this call (astra's confirmed finding at client.py:2121).
        client.submit_feedback(
            "trace_1",
            ThumbRating.UP,
            correction_output={"lock": threading.Lock()},
        )
        client.close()

        assert "correction_output" not in captured["payload"]
        assert captured["payload"]["rating"] == "up"


class TestB3FailedUpdateOverPriorContentSendsPlaceholder:
    """ADDENDUM R3 B3 (A2 completion): when gating/copying FAILS for a
    per-field content value on an UPDATE and the field previously carried
    real content, the wire value must be the clearing placeholder -- never
    omitted -- or the backend's `apply_updates` (which skips an omitted/
    `None` field) silently keeps the stale content. This is the same
    `wire_value_for_tightened_update` mechanism the M4/A2 explicit-null
    tests exercise (see `test_explicit_null_clear_in_record_mode_...`
    above), here triggered by a COPY failure (an uncopyable value) rather
    than an explicit `None`.

    Negative control: reverting `apply_content_mode`'s `record`-branch
    `except Exception: return None` to instead re-raise, or reverting
    `wire_value_for_tightened_update` to force the placeholder only when
    the mode is exactly `"metadata"`, makes this fail -- the second call's
    uncopyable `input_data` would either raise out of `record_observation`
    or omit the field, letting the first call's real content survive
    server-side.
    """

    def test_uncopyable_input_data_on_update_over_prior_content_sends_placeholder(
        self,
    ):
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        sensitive = "PATIENT diagnosis cancer stage 3"

        trace_id = client.start_trace("b3-failed-update-trace", content_mode="record")
        observation_id = client.record_observation(
            trace_id,
            name="op",
            input_data={"prompt": sensitive},
            content_mode="record",
        )
        # Update: copying FAILS for this call's input_data (an uncopyable
        # value), but there IS prior content for this field. Must not raise.
        client.record_observation(
            trace_id,
            observation_id=observation_id,
            name="op",
            input_data={"lock": threading.Lock()},
            content_mode="record",
        )

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        assert observation["input_data"] == {"redacted": True}
        assert sensitive not in json.dumps(sent_batches)


class TestB4KeyBasedScrubberCoverage:
    """ADDENDUM R3 B4 (A4 completion): the common key-based scrubber
    (`redact_sensitive_data(..., redact_credential_keys=True)`) now runs on
    `correlation_ids`, `tags`, and the COMPLETE `prompt_reference`
    structure (previously only `.variables`) -- never on the fixed numeric
    DTO fields. astra's ruling (a): these three bags are fixed/typed shapes
    today, so there is no way to smuggle a credential-shaped KEY into a
    typed `CorrelationIds`/`PromptReferenceDTO` through the public
    constructor to observe a VALUE difference outbound; this test verifies
    the coverage contract directly (the scrubber is actually invoked on
    these bags), matching how astra's own ruling frames the requirement --
    architecture coverage, not a value-level regression today.

    Negative control: reverting `_harden_content_bags` to the pre-fix
    version (which only hardened `metadata`/`input_data`/`output_data`/
    `session.metadata`/`prompt_reference.variables`) makes the spy
    assertions below fail -- `redact_sensitive_data` would never be called
    with `redact_credential_keys=True` for `correlation_ids`/`tags`/the
    full `prompt_reference` dict.
    """

    def test_correlation_ids_tags_and_prompt_reference_reach_the_key_based_scrubber(
        self, monkeypatch
    ):
        from traigent.observability import client as client_module

        seen_credential_calls: list[Any] = []
        original = client_module.redact_sensitive_data

        def _spy(value, *, redact_credential_keys=False):
            if redact_credential_keys:
                seen_credential_calls.append(copy.deepcopy(value))
            return original(value, redact_credential_keys=redact_credential_keys)

        monkeypatch.setattr(client_module, "redact_sensitive_data", _spy)

        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))
        client.start_trace(
            "b4-coverage-trace",
            tags=["prod", "release-1"],
            correlation_ids=CorrelationIds(otel_trace_id="trace-abc"),
            prompt_reference=PromptReferenceDTO(name="greeting", variables={}),
            content_mode="record",
        )
        client.flush()
        client.close()

        assert any(call == ["prod", "release-1"] for call in seen_credential_calls)
        assert any(
            isinstance(call, dict) and call.get("otel_trace_id") == "trace-abc"
            for call in seen_credential_calls
        )
        assert any(
            isinstance(call, dict) and call.get("name") == "greeting"
            for call in seen_credential_calls
        )

        # Fixed numeric/id DTO fields are never routed through the
        # key-based scrubber and survive untouched.
        trace_payload = sent_batches[-1][-1]
        assert trace_payload["correlation_ids"] == {"otel_trace_id": "trace-abc"}
        assert trace_payload["tags"] == ["prod", "release-1"]


class TestB5CorrelationIdsDetachedAtIntake:
    """ADDENDUM R3 B5 (A8 completion): `CorrelationIds` (a caller-supplied,
    mutable DTO) must be detached (deep copied) at intake exactly like
    session/metadata/prompt_reference -- astra's confirmed probe: Python
    returned the caller's own `CorrelationIds` instance BY REFERENCE
    (client.py:2744, pre-fix), so mutating it after `start_trace` returned
    changed a later `flush()`.

    Negative control: reverting `_coerce_correlation_ids` to `return
    correlation_ids` (no copy) for the `isinstance(correlation_ids,
    CorrelationIds)` branch makes every test below fail -- the mutation
    would reach the flushed payload.
    """

    def test_mutating_correlation_ids_after_start_trace_does_not_leak(self):
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        correlation_ids = CorrelationIds(otel_trace_id="BEFORE")
        trace_id = client.start_trace(
            "b5-correlation-ids-trace", correlation_ids=correlation_ids
        )
        correlation_ids.otel_trace_id = "AFTER"

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        assert trace_payload["correlation_ids"]["otel_trace_id"] == "BEFORE"

    def test_mutating_correlation_ids_after_update_does_not_leak(self):
        """The update path too: `record_observation`'s
        `_coerce_correlation_ids(correlation_ids) or existing.correlation_ids`
        must also detach."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        trace_id = client.start_trace("b5-update-trace")
        correlation_ids = CorrelationIds(otel_span_id="BEFORE")
        client.record_observation(
            trace_id,
            name="op",
            correlation_ids=correlation_ids,
        )
        correlation_ids.otel_span_id = "AFTER"

        client.flush()
        client.close()

        trace_payload = next(
            t for batch in sent_batches for t in batch if t["id"] == trace_id
        )
        observation = trace_payload["observations"][0]
        assert observation["correlation_ids"]["otel_span_id"] == "BEFORE"

    def test_mutating_correlation_ids_after_close_does_not_leak(self):
        """Same guarantee across `close()`."""
        sent_batches: list[list[dict]] = []
        client = _make_recording_client(lambda traces: sent_batches.append(traces))

        correlation_ids = CorrelationIds(otel_parent_span_id="BEFORE")
        client.start_trace("b5-close-trace", correlation_ids=correlation_ids)
        correlation_ids.otel_parent_span_id = "AFTER"

        client.flush()
        client.close()

        trace_payload = sent_batches[-1][-1]
        assert trace_payload["correlation_ids"]["otel_parent_span_id"] == "BEFORE"


class TestB6LegacyBooleanEnvValidation:
    """ADDENDUM R3 B6 (A1 completion): the legacy
    `TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT` env var accepts ONLY
    "true"/"false"/"1"/"0" (case-insensitive) -- never `is_truthy`'s wider
    vocabulary ("yes"/"on") and never a silent default on garbage/empty.
    astra's confirmed probe: both an empty and an invalid legacy value were
    previously accepted silently (config.py:115, pre-fix).

    Negative control: reverting `_legacy_capture_content_mode` to `"record"
    if is_truthy(raw) else "metadata"` makes the raising tests below fail --
    `""`, `"banana"`, and `"yes"` would all silently resolve to a content
    mode instead of raising.
    """

    def test_empty_legacy_env_raises_before_construction_succeeds(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", "")
        _mock_public_backend_dns(monkeypatch)

        with pytest.raises(ValueError, match="TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT"):
            ObservabilityConfig(backend_origin="https://auth.example.com")

    def test_invalid_legacy_env_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", "banana")
        _mock_public_backend_dns(monkeypatch)

        with pytest.raises(ValueError, match="TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT"):
            ObservabilityConfig(backend_origin="https://auth.example.com")

    def test_legacy_env_broader_is_truthy_vocabulary_now_rejected(self, monkeypatch):
        """`is_truthy` accepts "yes"/"on" -- the legacy var must NOT (B6
        narrows it to the exact true/false/1/0 spellings)."""
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", "yes")
        _mock_public_backend_dns(monkeypatch)

        with pytest.raises(ValueError, match="TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT"):
            ObservabilityConfig(backend_origin="https://auth.example.com")

    @pytest.mark.parametrize(
        ("raw", "expected_mode"),
        [
            ("true", "record"),
            ("TRUE", "record"),
            ("1", "record"),
            ("false", "metadata"),
            ("FALSE", "metadata"),
            ("0", "metadata"),
        ],
    )
    def test_accepted_boolean_spellings_resolve_case_insensitively(
        self, monkeypatch, raw, expected_mode
    ):
        monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT", raw)
        _mock_public_backend_dns(monkeypatch)

        config = ObservabilityConfig(backend_origin="https://auth.example.com")

        assert config.content_mode == expected_mode
