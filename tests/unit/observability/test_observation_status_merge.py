"""An update to an existing observation merges; it never resets its status (#3080)."""

from __future__ import annotations

import pytest

from traigent.observability.client import ObservabilityClient
from traigent.observability.config import ObservabilityConfig


@pytest.fixture(autouse=True)
def _online_env(monkeypatch, jwt_development_mode):
    del jwt_development_mode
    for name in ("TRAIGENT_DISABLE_TELEMETRY", "TRAIGENT_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_ENV", "development")


def _client() -> ObservabilityClient:
    return ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            enable_atexit_flush=False,
        ),
        sender=lambda _batch: None,
    )


def _stored(client, trace_id, oid):
    return client._trace_states[trace_id].observations[oid]


@pytest.mark.parametrize("terminal", ["completed", "failed", "rejected"])
def test_metadata_only_update_keeps_terminal_status(terminal):
    client = _client()
    trace_id = client.start_trace("t")
    oid = client.record_observation(
        trace_id, name="llm", status=terminal, input_tokens=5, output_tokens=7
    )
    client.record_observation(
        trace_id, name="llm", observation_id=oid, metadata={"note": "late"}
    )
    obs = _stored(client, trace_id, oid)
    assert obs.status == terminal
    assert (obs.input_tokens, obs.output_tokens) == (5, 7)
    assert obs.metadata["note"] == "late"


def test_explicit_running_does_not_regress_terminal_status():
    client = _client()
    trace_id = client.start_trace("t")
    oid = client.record_observation(trace_id, name="llm", status="completed")
    client.record_observation(
        trace_id, name="llm", observation_id=oid, status="running"
    )
    assert _stored(client, trace_id, oid).status == "completed"


def test_running_advances_to_terminal_and_new_defaults_to_running():
    client = _client()
    trace_id = client.start_trace("t")
    oid = client.record_observation(trace_id, name="llm")
    assert _stored(client, trace_id, oid).status == "running"
    client.record_observation(trace_id, name="llm", observation_id=oid, status="failed")
    assert _stored(client, trace_id, oid).status == "failed"
