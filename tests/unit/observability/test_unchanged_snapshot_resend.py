"""flush()/close() must not re-send snapshots that were already delivered (#2604).

Before this fix every flush() and close() re-submitted the latest snapshot of
every trace the client had ever started, and the transport sent each one again
even when the backend already had exactly that snapshot.
"""

from __future__ import annotations

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


def _client(sender) -> ObservabilityClient:
    return ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            enable_atexit_flush=False,
        ),
        sender=sender,
    )


def _sent_ids(requests: list[list[dict]]) -> list[str]:
    return [trace["id"] for request in requests for trace in request]


def test_repeated_flushes_of_an_unchanged_ended_trace_send_it_once():
    requests: list[list[dict]] = []
    client = _client(requests.append)
    trace_id = client.start_trace("t", trace_id="trace_once")
    client.end_trace(trace_id)

    results = [client.flush() for _ in range(5)]
    close_result = client.close()

    assert _sent_ids(requests) == ["trace_once"]
    assert all(result.success for result in [*results, close_result])
    assert close_result.items_sent == 1
    assert client._transport.get_stats()["skipped_unchanged_items"] >= 5


def test_many_traces_are_not_resent_on_every_flush():
    requests: list[list[dict]] = []
    client = _client(requests.append)
    for index in range(3):
        client.end_trace(client.start_trace("t", trace_id=f"trace_{index}"))

    client.flush()
    client.flush()
    client.close()

    # The issue's probe saw 3 payloads after flush() and 6 after close().
    assert sorted(_sent_ids(requests)) == ["trace_0", "trace_1", "trace_2"]


def test_a_trace_changed_after_delivery_is_sent_again():
    requests: list[list[dict]] = []
    client = _client(requests.append)
    trace_id = client.start_trace("t", trace_id="trace_changed")
    client.flush()

    client.record_observation(trace_id, name="step", observation_id="obs_1")
    client.flush()
    client.close()

    assert _sent_ids(requests) == ["trace_changed", "trace_changed"]
    assert requests[0][0].get("observations") in (None, [])
    assert [obs["id"] for obs in requests[1][0]["observations"]] == ["obs_1"]


def test_a_failed_delivery_is_not_treated_as_delivered():
    requests: list[list[dict]] = []
    attempts = {"count": 0}

    def sender(traces):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise AuthenticationError("invalid credentials")  # not retried
        requests.append(traces)

    client = _client(sender)
    client.end_trace(client.start_trace("t", trace_id="trace_retry"))

    first = client.flush()
    second = client.flush()
    third = client.flush()

    assert first.success is False
    assert second.items_pending == 0
    assert _sent_ids(requests) == ["trace_retry"]
    assert third.items_pending == 0
    assert attempts["count"] == 2


def _transport(requests, *, max_queue_size: int = 100) -> _SyncBatchTransport:
    return _SyncBatchTransport(
        requests.append,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=max_queue_size,
        max_batch_bytes=1_000_000,
    )


def test_dedup_compares_against_the_last_delivered_snapshot_only():
    requests: list[list[dict]] = []
    transport = _transport(requests)
    v1 = {"id": "x", "status": "running"}
    v2 = {"id": "x", "status": "completed"}

    for payload in (v1, v1, v2, v2, v1):
        assert transport.submit("x", payload)
        transport.flush()
    transport.close()

    # v1 sent, v1 skipped, v2 sent, v2 skipped, v1 differs from v2 -> sent.
    assert [request[0]["status"] for request in requests] == [
        "running",
        "completed",
        "running",
    ]


def test_delivered_digest_memory_is_bounded_and_forgetting_only_resends():
    requests: list[list[dict]] = []
    transport = _transport(requests, max_queue_size=2)
    for item_id in ("a", "b", "c"):
        assert transport.submit(item_id, {"id": item_id})
        transport.flush()

    assert len(transport._delivered_digests) == 2
    # "a" was forgotten, so an unchanged "a" is (harmlessly) sent again;
    # "c" is still remembered and skipped.
    assert transport.submit("a", {"id": "a"})
    assert transport.submit("c", {"id": "c"})
    transport.close()

    assert _sent_ids(requests) == ["a", "b", "c", "a"]


def test_a_snapshot_evicted_after_a_failed_send_is_sent_when_resubmitted():
    """A failed snapshot is never remembered as delivered: if the outbox
    evicts it and the app re-submits the same content, it must still go out."""
    import threading

    entered = threading.Event()
    release = threading.Event()
    requests: list[list[dict]] = []
    attempts = {"count": 0}

    def sender(traces):
        attempts["count"] += 1
        if attempts["count"] == 1:
            entered.set()
            assert release.wait(timeout=5.0)
            raise AuthenticationError("invalid credentials")  # not retried
        requests.append(traces)

    transport = _SyncBatchTransport(
        sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=1,
        max_batch_bytes=1_000_000,
    )
    assert transport.submit("a", {"id": "a"})
    flush_thread = threading.Thread(target=transport.flush)
    flush_thread.start()
    assert entered.wait(timeout=5.0)
    assert transport.submit("b", {"id": "b"})  # queued while "a" is in flight
    release.set()
    flush_thread.join(timeout=5.0)
    assert not flush_thread.is_alive()
    # The failed "a" was evicted from the full outbox in favour of newer "b".
    assert transport.get_stats()["dropped_by_reason"] == {"outbox_full": 1}
    # Nothing was delivered, so nothing may be remembered as delivered.
    assert dict(transport._delivered_digests) == {}
    transport.flush()  # delivers "b", freeing the one-slot outbox

    assert transport.submit("a", {"id": "a"})
    transport.close()

    assert sorted(_sent_ids(requests)) == ["a", "b"]


def _blocking_sender(requests, *, fail: bool = False):
    import threading

    entered = threading.Event()
    release = threading.Event()
    state = {"block": False}

    def sender(traces):
        if state["block"]:
            state["block"] = False
            entered.set()
            assert release.wait(timeout=5.0)
            if fail:
                raise AuthenticationError("invalid credentials")  # not retried
        requests.append(traces)

    return sender, state, entered, release


def _flush_in_thread(transport):
    import threading

    thread = threading.Thread(target=transport.flush)
    thread.start()
    return thread


def test_reverting_to_the_delivered_snapshot_while_another_is_in_flight_is_sent():
    """Delivered A, B in flight, app re-submits A: A must still go out, or the
    backend ends on B while the application is back at A."""
    requests: list[list[dict]] = []
    sender, state, entered, release = _blocking_sender(requests)
    transport = _SyncBatchTransport(
        sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=100,
        max_batch_bytes=1_000_000,
    )
    a = {"id": "x", "v": "A"}
    b = {"id": "x", "v": "B"}
    assert transport.submit("x", a)
    transport.flush()

    assert transport.submit("x", b)
    state["block"] = True
    thread = _flush_in_thread(transport)
    assert entered.wait(timeout=5.0)
    assert transport.submit("x", dict(a))
    release.set()
    thread.join(timeout=5.0)
    transport.close()

    assert [request[0]["v"] for request in requests] == ["A", "B", "A"]


def test_reverting_to_the_delivered_snapshot_while_another_is_queued_is_sent():
    requests: list[list[dict]] = []
    transport = _transport(requests)
    assert transport.submit("x", {"id": "x", "v": "A"})
    transport.flush()

    assert transport.submit("x", {"id": "x", "v": "B"})  # queued, not sent
    assert transport.submit("x", {"id": "x", "v": "A"})  # replaces B
    transport.close()

    # B never left the buffer; the backend's last write is the app's last: A.
    assert [request[0]["v"] for request in requests] == ["A", "A"]


def test_a_response_with_warnings_does_not_count_as_delivered():
    requests: list[list[dict]] = []

    def sender(traces):
        requests.append(traces)
        return {"warnings": ["prompt link not resolved"]}

    transport = _SyncBatchTransport(
        sender,
        batch_size=100,
        max_buffer_age=999.0,
        max_queue_size=100,
        max_batch_bytes=1_000_000,
    )
    for _ in range(2):
        assert transport.submit("x", {"id": "x"})
        transport.flush()
    transport.close()

    assert _sent_ids(requests) == ["x", "x"]
    assert transport.get_stats()["skipped_unchanged_items"] == 0
