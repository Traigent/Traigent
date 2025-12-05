"""Tests for consolidated retry utilities.

This module tests the unified retry system that combines simple decorators
with advanced retry handling for cloud operations.
"""

import threading
import time
from unittest.mock import Mock

import pytest

from traigent.utils.retry import (
    CircuitBreaker,
    CircuitBreakerState,
    NetworkError,
    NonRetryableError,
    RateLimitError,
    RetryableError,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    ServiceUnavailableError,
    retry,
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.strategy == RetryStrategy.EXPONENTIAL

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            strategy=RetryStrategy.LINEAR,
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.strategy == RetryStrategy.LINEAR


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(
            failure_threshold=3, recovery_timeout=30.0, success_threshold=2
        )
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0
        assert cb.success_threshold == 2
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_failure_tracking(self):
        """Test that circuit breaker tracks failures correctly."""
        cb = CircuitBreaker(failure_threshold=2)

        # Test failures through _on_failure method
        cb._on_failure()
        assert cb.state == CircuitBreakerState.CLOSED

        cb._on_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_success_reset(self):
        """Test that circuit breaker resets on success."""
        cb = CircuitBreaker(failure_threshold=2)

        # Cause failures
        cb._on_failure()
        cb._on_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Force half-open for testing
        cb.state = CircuitBreakerState.HALF_OPEN
        cb._on_success()
        cb._on_success()  # Need 2 successes by default
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_call_blocking(self):
        """Test that circuit breaker blocks calls when open."""
        cb = CircuitBreaker(failure_threshold=1)

        # Trigger open state
        cb._on_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Should raise exception when calling
        with pytest.raises(ServiceUnavailableError, match="Circuit breaker is OPEN"):
            cb.call(lambda: "success")

    def test_circuit_breaker_half_open_allows_single_thread(self):
        """Ensure only one thread can probe the circuit during half-open reset."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.0,
            success_threshold=1,
        )

        def fail():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            cb.call(fail)

        assert cb.state == CircuitBreakerState.OPEN

        # Ensure the reset window is eligible
        time.sleep(0.005)

        start_event = threading.Event()
        call_counter = 0
        counter_lock = threading.Lock()
        outcomes: list[str] = []

        def worker():
            nonlocal call_counter  # noqa: F824
            start_event.wait()
            try:

                def succeed():
                    nonlocal call_counter  # noqa: F824
                    with counter_lock:
                        call_counter += 1
                    time.sleep(0.01)
                    return "ok"

                cb.call(succeed)
                outcomes.append("executed")
            except ServiceUnavailableError:
                outcomes.append("rejected")

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()

        start_event.set()
        for thread in threads:
            thread.join()

        assert outcomes.count("executed") == 1
        assert outcomes.count("rejected") == 1
        assert call_counter == 1
        assert cb.state == CircuitBreakerState.CLOSED


class TestRetryHandler:
    """Test retry handler functionality."""

    def test_retry_handler_success(self):
        """Test retry handler with successful operation."""
        config = RetryConfig()
        handler = RetryHandler(config)
        mock_func = Mock(return_value="success")

        result = handler.execute_with_result(mock_func)
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1
        assert mock_func.call_count == 1

    def test_retry_handler_with_retryable_error(self):
        """Test retry handler with retryable errors."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        handler = RetryHandler(config)

        # Mock function that fails twice then succeeds
        mock_func = Mock(
            side_effect=[
                RetryableError("First failure"),
                RetryableError("Second failure"),
                "success",
            ]
        )

        result = handler.execute_with_result(mock_func)
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 3
        assert mock_func.call_count == 3

    def test_retry_handler_max_attempts_exceeded(self):
        """Test retry handler when max attempts exceeded."""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        handler = RetryHandler(config)

        mock_func = Mock(side_effect=RetryableError("Always fails"))

        result = handler.execute_with_result(mock_func)
        assert result.success is False
        assert result.attempts == 2
        assert mock_func.call_count == 2
        assert isinstance(result.error, RetryableError)

    def test_retry_handler_non_retryable_error(self):
        """Test retry handler with non-retryable error."""
        # Use NonRetryableError which should never be retried
        config = RetryConfig()
        handler = RetryHandler(config)
        mock_func = Mock(side_effect=NonRetryableError("Non-retryable"))

        result = handler.execute_with_result(mock_func)
        assert result.success is False
        assert result.attempts == 1
        assert mock_func.call_count == 1
        assert isinstance(result.error, NonRetryableError)

    @pytest.mark.asyncio
    async def test_async_retry_handler(self):
        """Test async retry handler."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        handler = RetryHandler(config)

        # Async function that fails once then succeeds
        async def async_func():
            if not hasattr(async_func, "called"):
                async_func.called = True
                raise RetryableError("First failure")
            return "async success"

        result = await handler.execute_async_with_result(async_func)
        assert result.success is True
        assert result.result == "async success"
        assert result.attempts == 2


class TestRetryDecorators:
    """Test retry decorators."""

    def test_retry_with_backoff_decorator(self):
        """Test retry_with_backoff decorator."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError(f"Failure {call_count}")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker functionality integrated with retry handler."""
        failure_count = 0

        # Circuit breaker functionality is built into RetryHandler
        config = RetryConfig(
            enable_circuit_breaker=True,
            failure_threshold=2,
            initial_delay=0.01,
            max_attempts=1,  # Only one attempt per call to test circuit breaker
        )
        handler = RetryHandler(config)

        def unreliable_function():
            nonlocal failure_count
            failure_count += 1
            raise ServiceUnavailableError("Service down")

        # First call should fail and record one failure
        result1 = handler.execute_with_result(unreliable_function)
        assert not result1.success
        assert handler.circuit_breaker.state == CircuitBreakerState.CLOSED

        # Second call should fail and open circuit
        result2 = handler.execute_with_result(unreliable_function)
        assert not result2.success
        assert handler.circuit_breaker.state == CircuitBreakerState.OPEN

        # Third call should return a failed result with circuit breaker blocking
        result3 = handler.execute_with_result(unreliable_function)
        assert not result3.success
        assert isinstance(result3.error, ServiceUnavailableError)
        assert "Circuit breaker is OPEN" in str(result3.error)

    @pytest.mark.asyncio
    async def test_async_retry_decorator(self):
        """Test async retry decorator."""
        call_count = 0

        @retry(max_attempts=2, delay=0.01)
        async def async_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NetworkError("Network issue")
            return "async success"

        result = await async_flaky_function()
        assert result == "async success"
        assert call_count == 2


class TestRetryStrategies:
    """Test different retry strategies."""

    def test_exponential_backoff_timing(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False,
            strategy=RetryStrategy.EXPONENTIAL,
        )

        # Test delay calculation using config method
        delay1 = config.calculate_delay(1)
        delay2 = config.calculate_delay(2)
        delay3 = config.calculate_delay(3)

        assert delay1 == 1.0  # initial_delay
        assert delay2 == 2.0  # initial_delay * exponential_base
        assert delay3 == 4.0  # initial_delay * exponential_base^2

    def test_linear_backoff_timing(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0, jitter=False, strategy=RetryStrategy.LINEAR
        )

        delay1 = config.calculate_delay(1)
        delay2 = config.calculate_delay(2)
        delay3 = config.calculate_delay(3)

        assert delay1 == 1.0  # initial_delay * 1
        assert delay2 == 2.0  # initial_delay * 2
        assert delay3 == 3.0  # initial_delay * 3

    def test_fixed_backoff_timing(self):
        """Test fixed backoff delay calculation."""
        config = RetryConfig(
            initial_delay=2.0, strategy=RetryStrategy.FIXED, jitter=False
        )

        # All delays should be the same
        for attempt in range(1, 5):
            delay = config.calculate_delay(attempt)
            assert delay == 2.0


class TestRetryableErrors:
    """Test retryable error types."""

    def test_retryable_error_hierarchy(self):
        """Test that error types inherit correctly."""
        assert issubclass(RateLimitError, RetryableError)
        assert issubclass(ServiceUnavailableError, RetryableError)
        assert issubclass(NetworkError, RetryableError)

    def test_error_detection(self):
        """Test that retry handler correctly identifies retryable errors."""
        config = RetryConfig()
        handler = RetryHandler(config)

        # These should be retryable (in the default retry_on_exception set)
        assert handler._should_retry(RetryableError("test"), 1)
        assert handler._should_retry(RateLimitError("rate limit"), 1)
        assert handler._should_retry(ServiceUnavailableError("service down"), 1)
        assert handler._should_retry(NetworkError("network issue"), 1)
        assert handler._should_retry(ConnectionError("connection"), 1)

        # These should NOT be retryable (not in the default set)
        assert not handler._should_retry(ValueError("test"), 1)
        assert not handler._should_retry(TypeError("test"), 1)

        # NonRetryableError should never be retryable
        assert not handler._should_retry(NonRetryableError("never retry"), 1)


class TestRetryIntegration:
    """Test retry system integration scenarios."""

    def test_retry_with_circuit_breaker_integration(self):
        """Test retry handler working with circuit breaker."""
        retry_config = RetryConfig(
            max_attempts=5,
            initial_delay=0.01,
            enable_circuit_breaker=True,
            failure_threshold=2,
            recovery_timeout=0.1,
        )

        handler = RetryHandler(retry_config)

        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ServiceUnavailableError("Always fails")

        # Should fail and eventually open circuit
        result = handler.execute_with_result(failing_function)
        assert result.success is False
        # Circuit breaker should be opened after failures
        assert handler.circuit_breaker is not None
        assert handler.circuit_breaker.state == CircuitBreakerState.OPEN

    def test_retry_performance_timing(self):
        """Test that retry timing works as expected."""
        start_time = time.time()

        config = RetryConfig(max_attempts=3, initial_delay=0.1, jitter=False)
        handler = RetryHandler(config)

        mock_func = Mock(side_effect=RetryableError("Always fails"))
        result = handler.execute_with_result(mock_func)

        elapsed = time.time() - start_time
        # Should take at least initial_delay + initial_delay*2 = 0.3 seconds
        assert elapsed >= 0.25  # Allow some tolerance
        assert result.success is False
        assert result.attempts == 3
