"""Every ingest request must fit the backend's per-request caps (#2600).

The backend rejects a request with more than 100 traces (422) or more than
1,000 observations counting nested children (413). Before this fix the SDK cut
requests only by ``batch_size`` (up to 10,000) and bytes, so such batches
failed, were retained unchanged, and failed again on every flush.
"""

from __future__ import annotations

import pytest

from traigent.observability.client import ObservabilityClient, _SyncBatchTransport
from traigent.observability.config import (
    INGEST_MAX_OBSERVATIONS_PER_REQUEST,
    INGEST_MAX_TRACES_PER_REQUEST,
    ObservabilityConfig,
)


def test_caps_match_the_backend_ingest_contract():
    # Mirrors ObservabilityIngestRequest.traces max_length and
    # MAX_OBSERVABILITY_EVENTS_PER_INGEST on the backend.
    assert INGEST_MAX_TRACES_PER_REQUEST == 100
    assert INGEST_MAX_OBSERVATIONS_PER_REQUEST == 1000


class _RecordingSender:
    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    def __call__(self, traces: list[dict]) -> None:
        self.requests.append(traces)


def _transport(sender: _RecordingSender, *, batch_size: int = 10_000):
    return _SyncBatchTransport(
        sender,
        batch_size=batch_size,
        max_buffer_age=999.0,
        max_queue_size=100_000,
        max_batch_bytes=10 * 1024 * 1024,
    )


def _observations(count: int, prefix: str) -> list[dict]:
    return [{"id": f"{prefix}_o{i}", "name": "step"} for i in range(count)]


def _trace(trace_id: str, observation_count: int = 0) -> dict:
    return {
        "id": trace_id,
        "name": "trace",
        "observations": _observations(observation_count, trace_id),
    }


def _observation_total(traces: list[dict]) -> int:
    def count(observations: list[dict]) -> int:
        return sum(1 + count(obs.get("children") or []) for obs in observations)

    return sum(count(trace.get("observations") or []) for trace in traces)


@pytest.mark.parametrize(
    ("trace_count", "expected_sizes"),
    [
        (INGEST_MAX_TRACES_PER_REQUEST - 1, [99]),
        (INGEST_MAX_TRACES_PER_REQUEST, [100]),
        (INGEST_MAX_TRACES_PER_REQUEST + 1, [100, 1]),
        (250, [100, 100, 50]),
    ],
)
def test_requests_never_exceed_the_trace_cap(trace_count, expected_sizes):
    sender = _RecordingSender()
    transport = _transport(sender)
    for index in range(trace_count):
        assert transport.submit(f"t{index}", _trace(f"t{index}"))

    result = transport.close()

    assert [len(request) for request in sender.requests] == expected_sizes
    assert result.success is True
    assert result.items_sent == trace_count
    assert result.items_dropped == 0


@pytest.mark.parametrize(
    ("observation_counts", "expected_requests"),
    [
        # Exactly at the cap: one request.
        ([500, 500], 1),
        # One over the cap: must be split.
        ([500, 501], 2),
        ([1, INGEST_MAX_OBSERVATIONS_PER_REQUEST], 2),
        ([12] * 100, 2),  # the issue's 100 traces x 12 observations case
    ],
)
def test_requests_never_exceed_the_observation_cap(
    observation_counts, expected_requests
):
    sender = _RecordingSender()
    transport = _transport(sender)
    for index, count in enumerate(observation_counts):
        assert transport.submit(f"t{index}", _trace(f"t{index}", count))

    result = transport.close()

    assert len(sender.requests) == expected_requests
    assert all(
        _observation_total(request) <= INGEST_MAX_OBSERVATIONS_PER_REQUEST
        for request in sender.requests
    )
    assert result.success is True
    assert result.items_sent == len(observation_counts)


def test_single_trace_exactly_at_the_observation_cap_is_sent():
    sender = _RecordingSender()
    transport = _transport(sender)

    assert transport.submit(
        "t_full", _trace("t_full", INGEST_MAX_OBSERVATIONS_PER_REQUEST)
    )
    result = transport.close()

    assert len(sender.requests) == 1
    assert result.items_dropped == 0


def test_trace_over_the_observation_cap_is_dropped_and_does_not_block_others():
    sender = _RecordingSender()
    transport = _transport(sender)

    assert transport.submit("t_ok", _trace("t_ok", 3))
    assert not transport.submit(
        "t_huge", _trace("t_huge", INGEST_MAX_OBSERVATIONS_PER_REQUEST + 1)
    )
    result = transport.close()
    stats = transport.get_stats()

    assert [[trace["id"] for trace in request] for request in sender.requests] == [
        ["t_ok"]
    ]
    assert result.success is True
    assert result.items_dropped == 1
    assert stats["dropped_by_reason"] == {"too_many_observations": 1}
    assert any("t_huge" in error for error in result.errors)


def test_nested_children_count_toward_the_observation_cap():
    sender = _RecordingSender()
    transport = _transport(sender)
    root = {
        "id": "root",
        "name": "root",
        "children": _observations(INGEST_MAX_OBSERVATIONS_PER_REQUEST, "child"),
    }

    accepted = transport.submit(
        "t_nested", {"id": "t_nested", "name": "trace", "observations": [root]}
    )
    transport.close()

    # 1 root + 1,000 children = 1,001 > cap, even though only 1 is top-level.
    assert accepted is False
    assert sender.requests == []


def test_client_with_large_batch_size_still_sends_capped_requests():
    sender = _RecordingSender()
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=500,  # still accepted: batch_size is the flush trigger
            max_buffer_age=999.0,
            enable_atexit_flush=False,
        ),
        sender=sender,
    )
    for index in range(200):
        trace_id = client.start_trace("trace", trace_id=f"trace_{index}")
        client.end_trace(trace_id)

    result = client.close()

    assert result.success is True
    assert sender.requests
    assert max(len(request) for request in sender.requests) <= 100
    delivered = {trace["id"] for request in sender.requests for trace in request}
    assert delivered == {f"trace_{index}" for index in range(200)}
