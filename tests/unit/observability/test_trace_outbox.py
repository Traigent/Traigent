"""Failed trace batches survive outages without an unbounded drain loop."""

import copy
import json
import threading

import pytest

from traigent.observability.client import _SyncBatchTransport
from traigent.utils.exceptions import ClientError
from traigent.utils.retry import RetryConfig, RetryHandler


@pytest.fixture
def transport_factory():
    transports = []

    def create(sender, *, max_queue_size=10, max_batch_bytes=10_000):
        transport = _SyncBatchTransport(
            sender,
            batch_size=100,
            max_buffer_age=999,
            max_queue_size=max_queue_size,
            max_batch_bytes=max_batch_bytes,
        )
        transport._retry_handler = RetryHandler(
            RetryConfig(
                max_attempts=1,
                initial_delay=0,
                jitter=False,
                retry_on_status={503},
            )
        )
        transports.append(transport)
        return transport

    yield create
    for transport in transports:
        transport._sender = lambda _payloads: None
        transport.close(timeout=1)


def unavailable():
    raise ClientError("fictional offline outage", status_code=503)


@pytest.mark.timeout(3)
def test_failed_batch_is_sent_after_connection_recovers(transport_factory):
    connected = False
    attempts = []

    def sender(payloads):
        attempts.append(payloads)
        if not connected:
            unavailable()

    transport = transport_factory(sender)
    assert transport.submit("trace", {"id": "trace", "input_data": {"text": "sample"}})
    failed = transport.flush()
    assert failed.items_pending == 1
    assert failed.items_dropped == 0
    assert len(attempts) == 1
    assert transport._timer is None
    connected = True
    recovered = transport.flush()
    assert recovered.items_sent == 1
    assert recovered.items_pending == 0
    assert recovered.items_dropped == 0
    assert attempts[1] == attempts[0]
    transport.flush()
    assert len(attempts) == 2


@pytest.mark.timeout(3)
@pytest.mark.parametrize("timeout", [None, 1])
def test_failed_old_snapshot_cannot_replace_newer_queued_snapshot(
    transport_factory, timeout
):
    entered = threading.Event()
    release = threading.Event()
    attempts = []

    def sender(payloads):
        attempts.append(payloads)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(1)
            unavailable()

    transport = transport_factory(sender)
    assert transport.submit("trace", {"id": "trace", "status": "running"})
    thread = threading.Thread(target=lambda: transport.flush(timeout=timeout))
    thread.start()
    assert entered.wait(1)
    assert transport.submit("trace", {"id": "trace", "status": "completed"})
    release.set()
    thread.join(1)
    assert not thread.is_alive()
    assert transport.get_stats()["pending_items"] == 1
    transport.flush()
    assert attempts[-1] == [{"id": "trace", "status": "completed"}]
    assert transport.get_stats()["dropped_items"] == 0


@pytest.mark.timeout(3)
def test_failed_batch_merges_with_new_items_under_bounded_oldest_eviction(
    transport_factory,
):
    entered = threading.Event()
    release = threading.Event()
    delivered = []
    connected = False

    def sender(payloads):
        if not connected:
            entered.set()
            assert release.wait(1)
            unavailable()
        delivered.extend(payloads)

    transport = transport_factory(sender, max_queue_size=2)
    assert transport.submit("oldest", {"id": "oldest"})
    assert transport.submit("older", {"id": "older"})
    thread = threading.Thread(target=transport.flush)
    thread.start()
    assert entered.wait(1)
    assert transport.submit("newest", {"id": "newest"})
    release.set()
    thread.join(1)
    stats = transport.get_stats()
    assert stats["queue_depth"] == 2
    assert stats["pending_items"] == 2
    assert stats["dropped_by_reason"] == {"outbox_full": 1}
    assert stats["warnings"]
    connected = True
    transport.flush()
    assert delivered == [{"id": "older"}, {"id": "newest"}]


@pytest.mark.timeout(3)
def test_late_failed_sender_retains_once_and_close_reports_pending(transport_factory):
    entered = threading.Event()
    release = threading.Event()
    attempts = []

    def sender(payloads):
        attempts.append(payloads)
        entered.set()
        assert release.wait(1)
        unavailable()

    transport = transport_factory(sender)
    assert transport.submit("trace", {"id": "trace"})
    timed_out = transport.flush(timeout=0.01)
    assert entered.wait(1)
    assert timed_out.items_pending == 1
    assert transport.flush(timeout=0).items_pending == 1
    assert len(attempts) == 1
    release.set()
    completion = transport._active_send_completion
    if completion is not None:
        assert completion.wait(1)
    result = transport.close(timeout=1)
    assert result.items_pending == 1
    assert result.items_dropped == 0
    assert not result.success
    assert transport.get_stats()["inflight_items"] == 0


@pytest.mark.timeout(3)
def test_retained_prepared_snapshots_keep_redaction_and_byte_batches(transport_factory):
    connected = False
    attempts = []

    def sender(payloads):
        attempts.append(copy.deepcopy(payloads))
        if not connected:
            unavailable()

    payloads = [
        {
            "id": f"trace-{index}",
            "input_data": {
                "authorization": "fictional-sensitive-value",
                "message": "x" * 100,
            },
        }
        for index in range(3)
    ]
    # Compute the byte cap from the same public serialization/redaction path
    # used for ordinary submissions, then pin every actual retry batch to it.
    sizing = transport_factory(lambda _payloads: None)
    prepared = [sizing._prepare_payload(payload).payload for payload in payloads]
    byte_cap = len(json.dumps({"traces": prepared[:2]}).encode("utf-8"))
    assert len(json.dumps({"traces": prepared}).encode("utf-8")) > byte_cap
    transport = transport_factory(sender, max_batch_bytes=byte_cap)
    for payload in payloads:
        assert transport.submit(payload["id"], payload)
    for payload in payloads:
        payload["input_data"]["message"] = "mutated after submission"
    assert transport.flush().items_pending == 3
    assert len(attempts) == 1
    connected = True
    assert transport.flush().items_sent == 3
    assert attempts[0] == attempts[1] == prepared[:2]
    assert attempts[2] == prepared[2:]
    for batch in attempts:
        serialized = json.dumps({"traces": batch})
        assert len(serialized.encode("utf-8")) <= byte_cap
        assert "fictional-sensitive-value" not in serialized
        assert "mutated after submission" not in serialized
