"""Issue #3033: a call rejected by an OPEN circuit breaker must not be retried.

The breaker's own "Circuit breaker is OPEN" rejection used to be retried with
full exponential backoff although no request was sent, so a flush during an
outage slept ~32 s for nothing and every phantom retry was reported through
``callback_on_retry`` (the observability ``retry_attempts`` counter).
"""

from __future__ import annotations

import time

from traigent.utils.retry import (
    CircuitBreakerState,
    RetryConfig,
    RetryHandler,
    ServiceUnavailableError,
)


def _open_handler(retries: list[int]) -> RetryHandler:
    config = RetryConfig(
        max_attempts=4,
        initial_delay=0.2,
        jitter=False,
        enable_circuit_breaker=True,
        failure_threshold=1,
        recovery_timeout=60.0,
        callback_on_retry=lambda _exc, attempt: retries.append(attempt),
    )
    handler = RetryHandler(config)
    assert handler.circuit_breaker is not None
    handler.circuit_breaker.state = CircuitBreakerState.OPEN
    handler.circuit_breaker.failure_count = 1
    from datetime import UTC, datetime

    handler.circuit_breaker.last_failure_time = datetime.now(UTC)
    return handler


def test_open_breaker_rejection_returns_at_once_without_retry_callbacks() -> None:
    retries: list[int] = []
    handler = _open_handler(retries)
    calls: list[int] = []

    started = time.monotonic()
    result = handler.execute_with_result(lambda: calls.append(1))
    elapsed = time.monotonic() - started

    assert not result.success
    assert calls == []  # nothing was sent
    assert result.attempts == 1
    assert result.total_delay == 0.0
    assert retries == []  # no phantom retry reported
    assert elapsed < 0.15  # the old loop slept 0.2 + 0.4 + 0.8 s
    # Backward compatible: still the module's retryable ServiceUnavailableError.
    assert isinstance(result.error, ServiceUnavailableError)
    assert "Circuit breaker is OPEN" in str(result.error)


def test_half_open_probe_in_flight_rejection_is_still_retried() -> None:
    """A probe is really running; waiting for its outcome is not a phantom retry."""
    retries: list[int] = []
    config = RetryConfig(
        max_attempts=2,
        initial_delay=0.01,
        jitter=False,
        enable_circuit_breaker=True,
        callback_on_retry=lambda _exc, attempt: retries.append(attempt),
    )
    handler = RetryHandler(config)
    assert handler.circuit_breaker is not None
    handler.circuit_breaker.state = CircuitBreakerState.HALF_OPEN
    handler.circuit_breaker._half_open_in_flight = True

    result = handler.execute_with_result(lambda: "ok")

    assert not result.success
    assert result.attempts == 2
    assert retries == [1]
