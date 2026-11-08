"""Issue #3084: RetryWithCircuitBreaker must not mutate the config it is given."""

from __future__ import annotations

from traigent.utils.retry import (
    DEFAULT_RETRY,
    RetryConfig,
    RetryHandler,
    RetryWithCircuitBreaker,
)


def test_shared_preset_is_left_untouched() -> None:
    assert DEFAULT_RETRY.enable_circuit_breaker is False

    decorator = RetryWithCircuitBreaker(DEFAULT_RETRY)

    assert decorator.handler.circuit_breaker is not None
    assert DEFAULT_RETRY.enable_circuit_breaker is False
    assert RetryHandler(DEFAULT_RETRY).circuit_breaker is None


def test_caller_config_keeps_its_other_settings() -> None:
    config = RetryConfig(max_attempts=7, initial_delay=0.5, failure_threshold=9)

    decorator = RetryWithCircuitBreaker(config)

    assert config.enable_circuit_breaker is False
    assert decorator.handler.config.max_attempts == 7
    assert decorator.handler.config.initial_delay == 0.5
    assert decorator.handler.circuit_breaker is not None
    assert decorator.handler.circuit_breaker.failure_threshold == 9
