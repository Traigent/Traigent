"""Unit tests for rate limit handling.

This module tests the rate limit exception handling, retry behavior,
circuit breaker integration, and the rate limit simulator infrastructure.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.fixtures.rate_limit_simulator import (
    ProviderRateLimitSimulator,
    RateLimitConfig,
    RateLimitSimulator,
    RateLimitType,
    SequentialRateLimiter,
)
from traigent.utils.exceptions import (
    NonRetryableError,
    RateLimitError,
    RetryableError,
)
from traigent.utils.retry import (
    CircuitBreaker,
    CircuitBreakerState,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
)


class TestRateLimitErrorException:
    """Tests for RateLimitError exception class."""

    @pytest.mark.unit
    def test_rate_limit_error_is_retryable(self) -> None:
        """RateLimitError should inherit from RetryableError."""
        error = RateLimitError("Rate limit exceeded")
        assert isinstance(error, RetryableError)
        assert isinstance(error, Exception)

    @pytest.mark.unit
    def test_rate_limit_error_default_message(self) -> None:
        """RateLimitError should have default message."""
        error = RateLimitError()
        assert "Rate limit exceeded" in str(error)

    @pytest.mark.unit
    def test_rate_limit_error_custom_message(self) -> None:
        """RateLimitError should accept custom message."""
        error = RateLimitError("OpenAI API limit reached")
        assert "OpenAI API limit reached" in str(error)

    @pytest.mark.unit
    def test_retry_after_is_preserved(self) -> None:
        """Retry-after value should be accessible from exception."""
        retry_after = 30.5
        error = RateLimitError("Rate limited", retry_after=retry_after)
        assert error.retry_after == retry_after

    @pytest.mark.unit
    def test_retry_after_none_by_default(self) -> None:
        """Retry-after should be None when not provided."""
        error = RateLimitError("Rate limited")
        assert error.retry_after is None

    @pytest.mark.unit
    def test_rate_limit_error_repr(self) -> None:
        """RateLimitError should have reasonable string representation."""
        error = RateLimitError("API limit", retry_after=10.0)
        error_str = str(error)
        assert "API limit" in error_str


class TestRetryConfigWithRateLimits:
    """Tests for RetryConfig behavior with rate limits."""

    @pytest.mark.unit
    def test_rate_limit_in_default_exceptions(self) -> None:
        """RateLimitError should be in default retryable exceptions."""
        config = RetryConfig()
        assert RateLimitError in config.retry_on_exception

    @pytest.mark.unit
    def test_calculate_delay_respects_retry_after(self) -> None:
        """RetryConfig should respect retry_after when configured."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=60.0,
            respect_retry_after=True,
        )

        # retry_after should be used
        delay = config.calculate_delay(attempt=1, retry_after=10.0)
        assert delay == 10.0

    @pytest.mark.unit
    def test_calculate_delay_caps_retry_after(self) -> None:
        """RetryConfig should cap retry_after at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=30.0,
            respect_retry_after=True,
        )

        # retry_after should be capped at max_delay
        delay = config.calculate_delay(attempt=1, retry_after=60.0)
        assert delay == 30.0

    @pytest.mark.unit
    def test_calculate_delay_ignores_retry_after_when_disabled(self) -> None:
        """RetryConfig should ignore retry_after when respect_retry_after=False."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=60.0,
            respect_retry_after=False,
            strategy=RetryStrategy.FIXED,
            jitter=False,
        )

        delay = config.calculate_delay(attempt=1, retry_after=30.0)
        # Should use initial_delay, not retry_after
        assert delay == 1.0

    @pytest.mark.unit
    def test_exponential_backoff_calculation(self) -> None:
        """Exponential backoff should double delay on each attempt."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )

        delay1 = config.calculate_delay(attempt=1)
        delay2 = config.calculate_delay(attempt=2)
        delay3 = config.calculate_delay(attempt=3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0


class TestRetryHandlerWithRateLimits:
    """Tests for RetryHandler behavior with rate limits."""

    @pytest.mark.unit
    def test_retry_on_rate_limit_error(self) -> None:
        """RetryHandler should retry on RateLimitError."""
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        call_count = 0

        def failing_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited", retry_after=0.01)
            return "success"

        result = handler.execute(failing_then_success)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.unit
    def test_retry_respects_retry_after(self) -> None:
        """RetryHandler should use retry_after from exception."""
        config = RetryConfig(
            max_attempts=2,
            initial_delay=0.1,
            respect_retry_after=True,
            jitter=False,
        )
        handler = RetryHandler(config)

        retry_after_value = 0.05
        start_time = time.monotonic()
        call_count = 0

        def failing_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("Rate limited", retry_after=retry_after_value)
            return "success"

        result = handler.execute(failing_once)
        elapsed = time.monotonic() - start_time

        assert result == "success"
        # Should have waited at least retry_after seconds
        assert elapsed >= retry_after_value

    @pytest.mark.unit
    def test_max_retries_exceeded_on_rate_limit(self) -> None:
        """RetryHandler should give up after max attempts."""
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        call_count = 0

        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Rate limited")

        result = handler.execute_with_result(always_fails)

        assert result.success is False
        assert isinstance(result.error, RateLimitError)
        assert result.attempts == 3

    @pytest.mark.unit
    def test_does_not_retry_non_retryable_error(self) -> None:
        """RetryHandler should not retry NonRetryableError."""
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
        )
        handler = RetryHandler(config)

        call_count = 0

        def non_retryable_failure() -> str:
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("Authentication failed")

        result = handler.execute_with_result(non_retryable_failure)

        assert result.success is False
        assert isinstance(result.error, NonRetryableError)
        assert call_count == 1  # Should only be called once

    @pytest.mark.unit
    def test_callback_on_retry_called(self) -> None:
        """RetryHandler should call callback on each retry."""
        retry_events: list[tuple[Exception, int]] = []

        def on_retry(exc: Exception, attempt: int) -> None:
            retry_events.append((exc, attempt))

        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            callback_on_retry=on_retry,
            jitter=False,
        )
        handler = RetryHandler(config)

        call_count = 0

        def fails_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"

        handler.execute(fails_twice)

        assert len(retry_events) == 2
        assert all(isinstance(e, RateLimitError) for e, _ in retry_events)
        assert [a for _, a in retry_events] == [1, 2]


class TestCircuitBreakerWithRateLimits:
    """Tests for CircuitBreaker behavior with rate limits."""

    @pytest.mark.unit
    def test_circuit_opens_on_repeated_failures(self) -> None:
        """Circuit breaker should open after threshold failures."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=1,
        )

        assert cb.state == CircuitBreakerState.CLOSED

        # Trigger failures
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RateLimitError("Rate limited")))
            except RateLimitError:
                pass

        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.unit
    def test_open_circuit_rejects_calls(self) -> None:
        """Open circuit should reject all calls."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=60.0,
        )

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(RateLimitError("Rate limited")))
        except RateLimitError:
            pass

        assert cb.state == CircuitBreakerState.OPEN

        # Further calls should be rejected
        from traigent.utils.retry import ServiceUnavailableError

        with pytest.raises(ServiceUnavailableError):
            cb.call(lambda: "should not run")

    @pytest.mark.unit
    def test_circuit_transitions_to_half_open(self) -> None:
        """Circuit should transition to half-open after recovery timeout."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,  # Very short for testing
            success_threshold=1,  # Only need 1 success to close
        )

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(RateLimitError("Rate limited")))
        except RateLimitError:
            pass

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.1)

        # Next call should be allowed (half-open) and succeed
        result = cb.call(lambda: "success")
        assert result == "success"
        # With success_threshold=1, one success should close the circuit
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.unit
    def test_circuit_closes_on_success_in_half_open(self) -> None:
        """Circuit should close after successful calls in half-open state."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(RateLimitError("Rate limited")))
        except RateLimitError:
            pass

        time.sleep(0.02)

        # First success in half-open
        cb.call(lambda: "success")

        # Need one more success to close (success_threshold=2)
        # Wait for half-open to allow another call
        time.sleep(0.02)
        cb.call(lambda: "success")

        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.unit
    def test_failure_count_resets_on_success(self) -> None:
        """Failure count should reset on successful call."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
        )

        # Two failures (not enough to open)
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RateLimitError("Rate limited")))
            except RateLimitError:
                pass

        assert cb.failure_count == 2

        # One success resets
        cb.call(lambda: "success")
        assert cb.failure_count == 0

        # Need 3 more failures to open
        assert cb.state == CircuitBreakerState.CLOSED


class TestRateLimitSimulator:
    """Tests for RateLimitSimulator class."""

    @pytest.mark.unit
    def test_rpm_limit_enforcement(self) -> None:
        """Simulator should enforce RPM limits."""
        config = RateLimitConfig(rpm_limit=3, retry_after=1.0)
        simulator = RateLimitSimulator(config)

        # First 3 requests should pass
        for _ in range(3):
            result = simulator.check_rate_limit()
            assert not result.limited
            simulator.record_request()

        # 4th request should be limited
        result = simulator.check_rate_limit()
        assert result.limited
        assert result.limit_type == RateLimitType.RPM

    @pytest.mark.unit
    def test_tpm_limit_enforcement(self) -> None:
        """Simulator should enforce TPM limits."""
        config = RateLimitConfig(tpm_limit=1000, retry_after=1.0)
        simulator = RateLimitSimulator(config)

        # Request within limit
        result = simulator.check_rate_limit(token_count=500)
        assert not result.limited
        simulator.record_request(token_count=500)

        # Request that would exceed limit
        result = simulator.check_rate_limit(token_count=600)
        assert result.limited
        assert result.limit_type == RateLimitType.TPM

    @pytest.mark.unit
    def test_burst_limit_enforcement(self) -> None:
        """Simulator should enforce burst limits."""
        config = RateLimitConfig(burst_limit=2, retry_after=1.0)
        simulator = RateLimitSimulator(config)

        # First 2 requests should pass
        for _ in range(2):
            result = simulator.check_rate_limit()
            assert not result.limited
            simulator.record_request()

        # 3rd request should be limited
        result = simulator.check_rate_limit()
        assert result.limited
        assert result.limit_type == RateLimitType.BURST

    @pytest.mark.unit
    def test_daily_limit_enforcement(self) -> None:
        """Simulator should enforce daily limits."""
        config = RateLimitConfig(daily_limit=5, retry_after=1.0)
        simulator = RateLimitSimulator(config)

        # First 5 requests should pass
        for _ in range(5):
            result = simulator.check_rate_limit()
            assert not result.limited
            simulator.record_request()

        # 6th request should be limited
        result = simulator.check_rate_limit()
        assert result.limited
        assert result.limit_type == RateLimitType.DAILY

    @pytest.mark.unit
    def test_chaos_mode(self) -> None:
        """Simulator should inject random failures in chaos mode."""
        config = RateLimitConfig(error_probability=1.0)  # Always fail
        simulator = RateLimitSimulator(config, random_seed=42)

        result = simulator.check_rate_limit()
        assert result.limited
        assert result.limit_type == RateLimitType.CHAOS

    @pytest.mark.unit
    def test_chaos_mode_with_seed_reproducible(self) -> None:
        """Chaos mode should be reproducible with same seed."""
        config = RateLimitConfig(error_probability=0.5)

        results1 = []
        simulator1 = RateLimitSimulator(config, random_seed=12345)
        for _ in range(10):
            results1.append(simulator1.check_rate_limit().limited)

        results2 = []
        simulator2 = RateLimitSimulator(config, random_seed=12345)
        for _ in range(10):
            results2.append(simulator2.check_rate_limit().limited)

        assert results1 == results2

    @pytest.mark.unit
    def test_window_reset(self) -> None:
        """Simulator should reset counters when window resets."""
        config = RateLimitConfig(rpm_limit=2, window_seconds=0.05)
        simulator = RateLimitSimulator(config)

        # Use up limit
        for _ in range(2):
            simulator.check_rate_limit()
            simulator.record_request()

        result = simulator.check_rate_limit()
        assert result.limited

        # Wait for window to reset
        time.sleep(0.1)

        result = simulator.check_rate_limit()
        assert not result.limited

    @pytest.mark.unit
    def test_retry_after_in_result(self) -> None:
        """Simulator should include retry_after in result."""
        config = RateLimitConfig(rpm_limit=1, retry_after=5.0)
        simulator = RateLimitSimulator(config)

        simulator.record_request()
        result = simulator.check_rate_limit()

        assert result.limited
        assert result.retry_after is not None
        assert result.retry_after >= 5.0

    @pytest.mark.unit
    def test_error_message_includes_provider(self) -> None:
        """Error message should include provider name."""
        config = RateLimitConfig(rpm_limit=1, provider="anthropic")
        simulator = RateLimitSimulator(config)

        simulator.record_request()
        result = simulator.check_rate_limit()

        assert "Anthropic" in result.message

    @pytest.mark.unit
    def test_get_stats(self) -> None:
        """Simulator should return current statistics."""
        config = RateLimitConfig(rpm_limit=10, tpm_limit=1000)
        simulator = RateLimitSimulator(config)

        simulator.record_request(token_count=100)
        simulator.record_request(token_count=200)

        stats = simulator.get_stats()
        assert stats["request_count"] == 2
        assert stats["token_count"] == 300
        assert stats["rpm_limit"] == 10
        assert stats["tpm_limit"] == 1000


class TestProviderRateLimitSimulator:
    """Tests for ProviderRateLimitSimulator class."""

    @pytest.mark.unit
    def test_openai_defaults(self) -> None:
        """ProviderRateLimitSimulator should have OpenAI defaults."""
        simulator = ProviderRateLimitSimulator("openai")
        assert simulator.config.provider == "openai"
        assert simulator.config.rpm_limit is not None
        assert simulator.config.tpm_limit is not None

    @pytest.mark.unit
    def test_anthropic_defaults(self) -> None:
        """ProviderRateLimitSimulator should have Anthropic defaults."""
        simulator = ProviderRateLimitSimulator("anthropic")
        assert simulator.config.provider == "anthropic"
        assert simulator.config.rpm_limit is not None

    @pytest.mark.unit
    def test_config_overrides(self) -> None:
        """ProviderRateLimitSimulator should accept config overrides."""
        simulator = ProviderRateLimitSimulator(
            "openai",
            config_overrides={"rpm_limit": 5, "retry_after": 10.0},
        )
        assert simulator.config.rpm_limit == 5
        assert simulator.config.retry_after == 10.0


class TestSequentialRateLimiter:
    """Tests for SequentialRateLimiter class."""

    @pytest.mark.unit
    def test_fails_on_specified_calls(self) -> None:
        """SequentialRateLimiter should fail on specified call numbers."""
        limiter = SequentialRateLimiter(fail_on_calls=[2, 3])

        # Call 1 succeeds
        result = limiter.check_rate_limit()
        assert not result.limited
        limiter.record_request()

        # Call 2 fails
        result = limiter.check_rate_limit()
        assert result.limited
        limiter.record_request()

        # Call 3 fails
        result = limiter.check_rate_limit()
        assert result.limited
        limiter.record_request()

        # Call 4 succeeds
        result = limiter.check_rate_limit()
        assert not result.limited

    @pytest.mark.unit
    def test_call_count_tracking(self) -> None:
        """SequentialRateLimiter should track call count."""
        limiter = SequentialRateLimiter()

        assert limiter.call_count == 0

        limiter.record_request()
        assert limiter.call_count == 1

        limiter.record_request()
        assert limiter.call_count == 2

    @pytest.mark.unit
    def test_reset(self) -> None:
        """SequentialRateLimiter should reset call count."""
        limiter = SequentialRateLimiter(fail_on_calls=[1])

        limiter.record_request()
        limiter.record_request()
        assert limiter.call_count == 2

        limiter.reset()
        assert limiter.call_count == 0


class TestRateLimitFixtures:
    """Tests for rate limit pytest fixtures."""

    @pytest.mark.unit
    def test_rate_limit_simulator_factory(self, rate_limit_simulator_factory) -> None:
        """rate_limit_simulator_factory should create simulators."""
        simulator = rate_limit_simulator_factory(rpm_limit=5)
        assert simulator.config.rpm_limit == 5

    @pytest.mark.unit
    def test_provider_rate_limiter_factory(self, provider_rate_limiter_factory) -> None:
        """provider_rate_limiter_factory should create provider simulators."""
        simulator = provider_rate_limiter_factory("openai")
        assert simulator.config.provider == "openai"

    @pytest.mark.unit
    def test_sequential_rate_limiter_factory(
        self, sequential_rate_limiter_factory
    ) -> None:
        """sequential_rate_limiter_factory should create sequential limiters."""
        limiter = sequential_rate_limiter_factory(fail_on_calls=[1, 2])
        assert 1 in limiter.fail_on_calls
        assert 2 in limiter.fail_on_calls

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limited_async_function(
        self, rate_limited_async_function
    ) -> None:
        """rate_limited_async_function should wrap async functions."""

        async def my_func() -> str:
            return "success"

        wrapped = rate_limited_async_function(my_func, fail_on_calls=[2])

        # First call succeeds
        result = await wrapped()
        assert result == "success"

        # Second call fails
        with pytest.raises(RateLimitError):
            await wrapped()

        # Third call succeeds
        result = await wrapped()
        assert result == "success"

    @pytest.mark.unit
    def test_rate_limited_sync_function(self, rate_limited_sync_function) -> None:
        """rate_limited_sync_function should wrap sync functions."""

        def my_func() -> str:
            return "success"

        wrapped = rate_limited_sync_function(my_func, fail_on_calls=[1])

        # First call fails
        with pytest.raises(RateLimitError):
            wrapped()

        # Second call succeeds
        result = wrapped()
        assert result == "success"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transient_rate_limiter(self, transient_rate_limiter) -> None:
        """transient_rate_limiter should fail N times then succeed."""

        async def my_func() -> str:
            return "success"

        wrapped = transient_rate_limiter(my_func, fail_count=2)

        # First 2 calls fail
        with pytest.raises(RateLimitError):
            await wrapped()
        with pytest.raises(RateLimitError):
            await wrapped()

        # Third call succeeds
        result = await wrapped()
        assert result == "success"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chaos_rate_limiter(self, chaos_rate_limiter) -> None:
        """chaos_rate_limiter should inject random failures."""

        async def my_func() -> str:
            return "success"

        # 100% failure probability for deterministic test
        wrapped = chaos_rate_limiter(my_func, fail_probability=1.0)

        with pytest.raises(RateLimitError):
            await wrapped()

        # 0% failure probability
        wrapped = chaos_rate_limiter(my_func, fail_probability=0.0)
        result = await wrapped()
        assert result == "success"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_burst_rate_limiter(self, burst_rate_limiter) -> None:
        """burst_rate_limiter should enforce burst limits."""

        async def my_func() -> str:
            return "success"

        wrapped = burst_rate_limiter(my_func, burst_size=2, cooldown=0.1)

        # First 2 succeed
        await wrapped()
        await wrapped()

        # 3rd fails (burst exceeded)
        with pytest.raises(RateLimitError):
            await wrapped()

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Now should work again
        result = await wrapped()
        assert result == "success"


class TestRateLimitIntegrationWithRetry:
    """Integration tests for rate limits with retry handler."""

    @pytest.mark.unit
    @pytest.mark.rate_limit
    def test_retry_handler_with_simulator(self, rate_limit_simulator_factory) -> None:
        """RetryHandler should work with rate limit simulator."""
        simulator = rate_limit_simulator_factory(rpm_limit=2, retry_after=0.01)
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        call_count = 0

        def rate_limited_operation() -> str:
            nonlocal call_count
            call_count += 1

            result = simulator.check_rate_limit()
            if result.limited:
                raise RateLimitError(result.message, retry_after=result.retry_after)

            simulator.record_request()
            return f"success-{call_count}"

        # First 2 calls succeed, 3rd fails, window resets, then succeeds
        # Due to short window, retries should succeed eventually
        result = handler.execute(rate_limited_operation)
        assert "success" in result

    @pytest.mark.unit
    @pytest.mark.rate_limit
    def test_sequential_limiter_with_retry(
        self, sequential_rate_limiter_factory
    ) -> None:
        """RetryHandler should handle sequential rate limiter."""
        limiter = sequential_rate_limiter_factory(fail_on_calls=[1, 2])
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        def rate_limited_operation() -> str:
            result = limiter.check_rate_limit()
            limiter.record_request()

            if result.limited:
                raise RateLimitError(result.message, retry_after=result.retry_after)

            return "success"

        result = handler.execute(rate_limited_operation)
        assert result == "success"
        assert limiter.call_count == 3  # Failed twice, succeeded on third


class TestEdgeCases:
    """Tests for edge cases in rate limit handling."""

    @pytest.mark.unit
    def test_zero_retry_after(self) -> None:
        """RateLimitError with zero retry_after should be valid."""
        error = RateLimitError("Rate limited", retry_after=0.0)
        assert error.retry_after == 0.0

    @pytest.mark.unit
    def test_negative_retry_after_stored(self) -> None:
        """Negative retry_after should be stored (validation is caller's job)."""
        error = RateLimitError("Rate limited", retry_after=-1.0)
        assert error.retry_after == -1.0

    @pytest.mark.unit
    def test_simulator_with_no_limits(self) -> None:
        """Simulator with no limits should never rate limit."""
        config = RateLimitConfig()  # All limits None
        simulator = RateLimitSimulator(config)

        for _ in range(100):
            result = simulator.check_rate_limit()
            assert not result.limited
            simulator.record_request()

    @pytest.mark.unit
    def test_simulator_with_zero_limit(self) -> None:
        """Simulator with zero limit should immediately rate limit."""
        config = RateLimitConfig(rpm_limit=0)
        simulator = RateLimitSimulator(config)

        result = simulator.check_rate_limit()
        assert result.limited

    @pytest.mark.unit
    def test_concurrent_limit_types(self) -> None:
        """Multiple limit types should all be checked."""
        config = RateLimitConfig(
            rpm_limit=10,
            tpm_limit=100,
            burst_limit=5,
        )
        simulator = RateLimitSimulator(config)

        # Hit burst limit first
        for _ in range(5):
            simulator.record_request()

        result = simulator.check_rate_limit()
        assert result.limited
        assert result.limit_type == RateLimitType.BURST
