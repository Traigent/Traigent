"""A rejected trace update must not let an older queued snapshot go out (#2601).

Before this fix, when an update to an already-queued trace was dropped
(e.g. ``payload_too_large``), the older partial snapshot stayed queued and was
sent: the backend accepted (202) a stale running trace, and the application's
latest write was lost behind a successful flush.
"""

from __future__ import annotations

import threading

import pytest

from traigent.observability.client import ObservabilityClient, _SyncBatchTransport
from traigent.observability.config import ObservabilityConfig
from traigent.utils.exceptions import AuthenticationError


@pytest.fixture(autouse=True)
def _online_env(monkeypatch, jwt_development_mode):
    del jwt_development_mode
    for name in ("TRAIGENT_DISABLE_TELEMETRY", "TRAIGENT_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_ENV", "development")


def _transport(sender, *, max_batch_bytes: int = 2048) -> _SyncBatchTransport:
    return _SyncBatchTransport(
        sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=100,
        max_batch_bytes=max_batch_bytes,
    )


def _sent_ids(requests: list[list[dict]]) -> list[str]:
    return [trace["id"] for request in requests for trace in request]


def test_oversized_update_evicts_the_older_queued_snapshot():
    requests: list[list[dict]] = []
    transport = _transport(requests.append)

    assert transport.submit("trace_x", {"id": "trace_x", "status": "running"})
    assert transport.submit("trace_y", {"id": "trace_y", "status": "running"})
    assert not transport.submit(
        "trace_x", {"id": "trace_x", "status": "completed", "blob": "x" * 4096}
    )
    result = transport.close()

    assert _sent_ids(requests) == ["trace_y"]
    assert result.items_dropped == 1
    assert result.items_pending == 0
    assert any("trace_x" in error for error in result.errors)


def test_non_serializable_update_evicts_the_older_queued_snapshot():
    requests: list[list[dict]] = []
    transport = _transport(requests.append)

    assert transport.submit("trace_x", {"id": "trace_x", "status": "running"})
    assert not transport.submit("trace_x", {"id": "trace_x", "bad": object()})
    transport.close()

    assert requests == []


def test_rejected_update_without_a_queued_snapshot_drops_only_itself():
    requests: list[list[dict]] = []
    transport = _transport(requests.append)

    assert not transport.submit("trace_x", {"id": "trace_x", "blob": "x" * 4096})
    assert transport.submit("trace_y", {"id": "trace_y"})
    result = transport.close()

    assert _sent_ids(requests) == ["trace_y"]
    assert result.items_dropped == 1


def test_failed_inflight_snapshot_superseded_by_a_rejected_update_is_not_retained():
    entered = threading.Event()
    release = threading.Event()
    attempts: list[list[str]] = []

    def sender(traces):
        attempts.append([trace["id"] for trace in traces])
        entered.set()
        assert release.wait(timeout=5.0)
        raise AuthenticationError("invalid credentials")  # not retried

    transport = _transport(sender)
    assert transport.submit("trace_x", {"id": "trace_x", "status": "running"})
    assert transport.submit("trace_y", {"id": "trace_y", "status": "running"})

    flush_thread = threading.Thread(target=transport.flush)
    flush_thread.start()
    assert entered.wait(timeout=5.0)
    # trace_x's older snapshot is in flight; its update is now rejected.
    assert not transport.submit(
        "trace_x", {"id": "trace_x", "status": "completed", "blob": "x" * 4096}
    )
    release.set()
    flush_thread.join(timeout=5.0)
    assert not flush_thread.is_alive()

    stats = transport.get_stats()
    # trace_y is retained for a later retry; the stale trace_x is not.
    assert stats["queue_depth"] == 1
    assert list(transport._buffer) == ["trace_y"]


def test_client_record_mode_oversized_observation_does_not_send_stale_trace():
    """The issue's repro: start, o00, oversized o01, o02..o04, end, flush."""
    requests: list[list[dict]] = []
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            content_mode="record",
            batch_size=100,
            max_buffer_age=999.0,
            max_batch_bytes=8192,
            enable_atexit_flush=False,
        ),
        sender=requests.append,
    )

    trace_id = client.start_trace("repro", trace_id="trace_repro")
    client.record_observation(trace_id, name="o00", observation_id="o00")
    client.record_observation(
        trace_id, name="o01", observation_id="o01", output_data="x" * 16384
    )
    for index in range(2, 5):
        client.record_observation(
            trace_id, name=f"o0{index}", observation_id=f"o0{index}"
        )
    client.end_trace(trace_id)
    result = client.close()

    # No snapshot of the trace may reach the backend: every one after o01 is
    # too large, and the older partial one (trace + o00) is stale.
    assert "trace_repro" not in _sent_ids(requests)
    assert result.items_dropped >= 1
    assert any("trace_repro" in error for error in result.errors)


def test_a_superseded_mark_does_not_outlive_its_in_flight_batch():
    """Rejecting X while an unrelated batch is in flight must not make a later,
    valid X unretainable once that batch has finished."""
    entered = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    def sender(traces):
        calls["count"] += 1
        if calls["count"] == 1:  # the unrelated batch [y]
            entered.set()
            assert release.wait(timeout=5.0)
            return None
        raise AuthenticationError("invalid credentials")  # not retried

    transport = _transport(sender)
    assert transport.submit("trace_y", {"id": "trace_y"})
    flush_thread = threading.Thread(target=transport.flush)
    flush_thread.start()
    assert entered.wait(timeout=5.0)
    assert not transport.submit("trace_x", {"id": "trace_x", "blob": "x" * 4096})
    release.set()
    flush_thread.join(timeout=5.0)
    assert not flush_thread.is_alive()

    assert transport.submit("trace_x", {"id": "trace_x", "status": "completed"})
    result = transport.flush()

    # The send failed, so the valid snapshot is retained for a later retry.
    assert result.success is False
    assert list(transport._buffer) == ["trace_x"]
