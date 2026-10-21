"""Ended, delivered traces are evicted from the client's state map (#2604)."""

from __future__ import annotations

import pytest

from traigent.observability.client import ObservabilityClient
from traigent.observability.config import ObservabilityConfig

GRACE = 60.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _online_env(monkeypatch, jwt_development_mode):
    del jwt_development_mode
    for name in ("TRAIGENT_DISABLE_TELEMETRY", "TRAIGENT_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_ENV", "development")


def _client(sender, clock) -> ObservabilityClient:
    return ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=1000,
            max_buffer_age=999.0,
            enable_atexit_flush=False,
        ),
        sender=sender,
        trace_state_grace_seconds=GRACE,
        clock=clock,
    )


def test_ended_confirmed_trace_is_evicted_only_after_grace():
    clock = FakeClock()
    requests: list[list[dict]] = []
    client = _client(requests.append, clock)
    trace_id = client.start_trace("t", trace_id="t1")
    client.end_trace(trace_id)
    assert client.flush().success

    clock.advance(GRACE - 1)
    client._evict_expired_trace_states()
    assert "t1" in client._trace_states

    clock.advance(2)
    assert client._evict_expired_trace_states() == 1
    assert "t1" not in client._trace_states


def test_unended_trace_is_never_evicted():
    clock = FakeClock()
    client = _client(lambda batch: None, clock)
    client.start_trace("t", trace_id="running")
    assert client.flush().success
    clock.advance(GRACE * 100)
    client._evict_expired_trace_states()
    assert "running" in client._trace_states


def test_ended_but_unflushed_trace_is_kept_until_confirmed():
    clock = FakeClock()
    requests: list[list[dict]] = []
    client = _client(requests.append, clock)
    client.end_trace(client.start_trace("t", trace_id="queued"))
    clock.advance(GRACE * 2)
    assert client._evict_expired_trace_states() == 0
    assert "queued" in client._trace_states
    assert requests == []

    assert client.flush().success
    clock.advance(GRACE * 2)
    client._evict_expired_trace_states()
    assert "queued" not in client._trace_states


def test_failed_delivery_keeps_the_trace_for_retry():
    clock = FakeClock()
    calls: list[int] = []
    healthy = {"ok": False}

    def sender(batch):
        calls.append(len(batch))
        if not healthy["ok"]:
            raise RuntimeError("backend down")

    client = _client(sender, clock)
    client.end_trace(client.start_trace("t", trace_id="failing"))
    assert not client.flush(timeout=5).success
    clock.advance(GRACE * 3)
    client._evict_expired_trace_states()
    assert "failing" in client._trace_states

    healthy["ok"] = True
    result = client.flush(timeout=5)  # failure history is cumulative
    assert result.items_pending == 0
    clock.advance(GRACE * 3)
    client._evict_expired_trace_states()
    assert "failing" not in client._trace_states


def test_restart_after_eviction_creates_a_fresh_delivered_snapshot():
    clock = FakeClock()
    requests: list[list[dict]] = []
    client = _client(requests.append, clock)
    client.end_trace(client.start_trace("first", trace_id="again"))
    client.flush()
    clock.advance(GRACE + 1)
    client._evict_expired_trace_states()
    assert "again" not in client._trace_states

    client.start_trace("second", trace_id="again")
    client.end_trace("again")
    assert client.flush().success
    sent = [
        trace for request in requests for trace in request if trace["id"] == "again"
    ]
    assert [trace["name"] for trace in sent] == ["first", "second"]
    assert "again" in client._trace_states


def test_unchanged_ended_trace_still_deduplicated_before_eviction():
    clock = FakeClock()
    requests: list[list[dict]] = []
    client = _client(requests.append, clock)
    client.end_trace(client.start_trace("t", trace_id="once"))
    for _ in range(3):
        client.flush()
    assert [t["id"] for r in requests for t in r] == ["once"]


def test_restarted_trace_is_not_evicted_by_its_old_queue_entry():
    clock = FakeClock()
    client = _client(lambda batch: None, clock)
    client.end_trace(client.start_trace("t", trace_id="r"))
    client.flush()
    client.start_trace("t2", trace_id="r")  # running again, replaces ended state
    clock.advance(GRACE * 2)
    client._evict_expired_trace_states()
    assert "r" in client._trace_states


def test_memory_stays_bounded_over_many_ended_traces():
    clock = FakeClock()
    client = _client(lambda batch: None, clock)
    n = 10_000
    peak = 0
    for i in range(n):
        client.end_trace(client.start_trace("t", trace_id=f"bulk-{i}"))
        if i % 100 == 99:
            assert client.flush().success
            clock.advance(GRACE + 1)
        peak = max(peak, len(client._trace_states))
    assert client.flush().success
    clock.advance(GRACE + 1)
    client._evict_expired_trace_states()
    assert peak <= 400
    assert len(client._trace_states) <= 100
    assert len(client._eviction_queue) <= 100


def test_eviction_pass_work_is_bounded():
    from traigent.observability import client as client_module

    clock = FakeClock()
    client = _client(lambda batch: None, clock)
    for i in range(client_module._MAX_EVICTION_STEPS_PER_PASS * 3):
        client.end_trace(client.start_trace("t", trace_id=f"b-{i}"))
    client.flush()
    clock.advance(GRACE + 1)
    assert (
        client._evict_expired_trace_states()
        <= client_module._MAX_EVICTION_STEPS_PER_PASS
    )


def _small_queue_client(sender, clock, max_queue_size: int) -> ObservabilityClient:
    return ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=1000,
            max_queue_size=max_queue_size,
            max_buffer_age=999.0,
            enable_atexit_flush=False,
        ),
        sender=sender,
        trace_state_grace_seconds=GRACE,
        clock=clock,
    )


def test_eviction_does_not_depend_on_a_bounded_delivery_lru():
    clock = FakeClock()
    cap = 8
    client = _small_queue_client(lambda batch: None, clock, cap)
    total = cap * 5
    for i in range(total):
        client.end_trace(client.start_trace("t", trace_id=f"w-{i}"))
        if i % cap == cap - 1:
            assert client.flush().success  # all within one grace window
    assert client.flush().success
    assert len(client._trace_states) == total
    clock.advance(GRACE + 1)
    assert client._evict_expired_trace_states() == total
    assert client._trace_states == {}
    assert client._transport._settled_items == set()


def test_late_update_to_evicted_trace_starts_a_fresh_delivered_snapshot():
    clock = FakeClock()
    requests: list[list[dict]] = []
    client = _client(requests.append, clock)
    client.end_trace(client.start_trace("keep-name", trace_id="late"))
    client.flush()
    clock.advance(GRACE + 1)
    client._evict_expired_trace_states()
    assert "late" not in client._trace_states

    client.record_observation("late", name="obs")
    client.end_trace("late")
    assert client.flush().success
    sent = [t for r in requests for t in r if t["id"] == "late"]
    assert len(sent) == 2
    assert sent[1]["name"] == "keep-name"
    assert len(sent[1]["observations"]) == 1

    clock.advance(GRACE + 1)
    client._evict_expired_trace_states()
    client.end_trace("late")  # end_trace on a tombstoned id also restarts it
    assert "late" in client._trace_states


def test_never_seen_trace_id_still_raises():
    client = _client(lambda batch: None, FakeClock())
    with pytest.raises(ValueError, match="Unknown trace_id"):
        client.record_observation("ghost", name="obs")
    with pytest.raises(ValueError, match="Unknown trace_id"):
        client.end_trace("ghost")


def test_tombstones_are_bounded_fifo(monkeypatch):
    from traigent.observability import client as client_module

    monkeypatch.setattr(client_module, "_MAX_EVICTED_TRACE_TOMBSTONES", 3)
    clock = FakeClock()
    client = _client(lambda batch: None, clock)
    for i in range(6):
        client.end_trace(client.start_trace("t", trace_id=f"f-{i}"))
    client.flush()
    clock.advance(GRACE + 1)
    client._evict_expired_trace_states()
    assert len(client._evicted_traces) == 3
    with pytest.raises(ValueError, match="Unknown trace_id"):
        client.end_trace("f-0")  # aged out of the bounded tombstone set
    client.end_trace("f-5")  # still tombstoned
