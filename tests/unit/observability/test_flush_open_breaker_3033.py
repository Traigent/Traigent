"""Issue #3033: flushing while the transport's circuit breaker is OPEN returns at
once, keeps the traces pending, and does not count phantom retries."""

from __future__ import annotations

import time

import pytest

from traigent.observability import ObservabilityClient, ObservabilityConfig
from traigent.utils.retry import CircuitBreakerState


@pytest.mark.timeout(20)
def test_flush_with_open_breaker_fails_fast_and_keeps_traces() -> None:
    requests: list[int] = []

    def sender(traces):
        requests.append(len(traces))
        raise ConnectionError("ingest down")

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=60.0,
            max_queue_size=100,
            enable_atexit_flush=False,
        ),
        sender=sender,
    )
    transport = client._transport
    # Keep the real retry/breaker logic, only shrink its backoff for the test.
    transport._retry_handler.config.initial_delay = 0.001
    transport._retry_handler.config.max_delay = 0.001
    try:
        for index in range(3):
            trace_id = client.start_trace("t", trace_id=f"trace_3033_{index}")
            client.end_trace(trace_id)

        first = client.flush()
        assert first.success is False
        breaker = transport._retry_handler.circuit_breaker
        assert breaker is not None
        assert breaker.state is CircuitBreakerState.OPEN
        sent_before = len(requests)
        retries_before = client.get_stats()["retry_attempts"]

        # Restore a realistic backoff: the old code would now sleep it out.
        transport._retry_handler.config.initial_delay = 2.0
        transport._retry_handler.config.max_delay = 60.0
        started = time.monotonic()
        second = client.flush()
        elapsed = time.monotonic() - started

        assert second.success is False
        assert second.items_pending == 3
        assert len(requests) == sent_before  # breaker rejected, nothing sent
        assert client.get_stats()["retry_attempts"] == retries_before
        assert elapsed < 1.5
    finally:
        client.close(timeout=0)
