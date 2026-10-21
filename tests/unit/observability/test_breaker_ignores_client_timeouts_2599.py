"""Client-side ingest timeouts must not trip the circuit breaker (#2599)."""

from __future__ import annotations

from urllib import error

import pytest

from traigent.observability.client import ObservabilityClient, _is_client_timeout
from traigent.observability.config import ObservabilityConfig
from traigent.utils.exceptions import TraigentConnectionError
from traigent.utils.retry import (
    CircuitBreaker,
    CircuitBreakerState,
    ServiceUnavailableError,
)


def _timeout_exc() -> Exception:
    try:
        try:
            raise error.URLError(TimeoutError("timed out"))
        except error.URLError as exc:
            raise TraigentConnectionError("Failed to connect") from exc
    except TraigentConnectionError as wrapped:
        return wrapped


def test_predicate_classifies_timeouts_only():
    assert _is_client_timeout(_timeout_exc())
    assert _is_client_timeout(TimeoutError("x"))
    assert not _is_client_timeout(TraigentConnectionError("refused"))
    assert not _is_client_timeout(ValueError("x"))


def test_breaker_ignores_predicate_failures_but_counts_others():
    breaker = CircuitBreaker(failure_threshold=2, ignore_failure=_is_client_timeout)

    def boom_timeout():
        raise _timeout_exc()

    def boom_hard():
        raise TraigentConnectionError("refused")

    for _ in range(10):
        with pytest.raises(TraigentConnectionError):
            breaker.call(boom_timeout)
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.failure_count == 0

    for _ in range(2):
        with pytest.raises(TraigentConnectionError):
            breaker.call(boom_hard)
    assert breaker.state == CircuitBreakerState.OPEN
    with pytest.raises(ServiceUnavailableError):
        breaker.call(boom_hard)


def test_default_breaker_still_counts_everything():
    breaker = CircuitBreaker(failure_threshold=2)
    for _ in range(2):
        with pytest.raises(TraigentConnectionError):
            breaker.call(lambda: (_ for _ in ()).throw(_timeout_exc()))
    assert breaker.state == CircuitBreakerState.OPEN


def test_client_breaker_stays_closed_over_repeated_slow_server_timeouts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    config = ObservabilityConfig(
        backend_origin="http://localhost:5000",
        api_key="test-key",  # pragma: allowlist secret
        enable_atexit_flush=False,
        offline_mode=False,
    )
    calls = {"n": 0}

    def slow(_payloads):
        calls["n"] += 1
        raise _timeout_exc()

    client = ObservabilityClient(config, sender=slow)
    for _ in range(12):  # more timed-out sends than the breaker threshold of 5
        client._transport._retry_handler.execute_with_result(
            client._transport._sender, [{"id": "t"}]
        )
    breaker = client._transport._retry_handler.circuit_breaker
    assert breaker.state == CircuitBreakerState.CLOSED
    assert calls["n"] >= 12  # every send was attempted, none short-circuited
