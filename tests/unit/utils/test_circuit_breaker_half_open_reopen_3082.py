"""Issue #3082: a failing half-open probe re-opens the breaker immediately,
even when an earlier probe in the same half-open window succeeded."""

from __future__ import annotations

import time

import pytest

from traigent.utils.retry import CircuitBreaker, CircuitBreakerState


def _fail() -> None:
    raise ConnectionError("down")


def test_failed_probe_after_successful_probe_reopens() -> None:
    breaker = CircuitBreaker(
        failure_threshold=2, recovery_timeout=0.0, success_threshold=2
    )
    for _ in range(2):
        with pytest.raises(ConnectionError):
            breaker.call(_fail)
    assert breaker.state is CircuitBreakerState.OPEN

    time.sleep(0.01)
    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state is CircuitBreakerState.HALF_OPEN

    with pytest.raises(ConnectionError):
        breaker.call(_fail)

    assert breaker.state is CircuitBreakerState.OPEN
    assert breaker.last_failure_time is not None


def test_first_probe_failure_reopens() -> None:
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=0.0, success_threshold=2
    )
    with pytest.raises(ConnectionError):
        breaker.call(_fail)
    time.sleep(0.01)
    with pytest.raises(ConnectionError):
        breaker.call(_fail)
    assert breaker.state is CircuitBreakerState.OPEN


def test_closed_failures_below_threshold_stay_closed() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    with pytest.raises(ConnectionError):
        breaker.call(_fail)
    assert breaker.state is CircuitBreakerState.CLOSED
