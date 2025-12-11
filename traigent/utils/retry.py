"""Consolidated retry utilities for TraiGent SDK.

This module combines the best features from both retry systems:
- retry.py: Simple decorator-based API for external users
- retry_handler.py: Advanced features for cloud operations

The consolidated module provides both simple and advanced retry capabilities
with a unified interface.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Performance FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Any, TypeVar, cast

from traigent.utils.exceptions import (
    NetworkError,
    NonRetryableError,
    RateLimitError,
    RetryableError,
    RetryError,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry delay strategies."""

    FIXED = "fixed"  # Constant delay
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear increase
    JITTER = "jitter"  # Random jitter added

    # Aliases for backward compatibility
    FIXED_DELAY = "fixed"
    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR_BACKOFF = "linear"
    RANDOM_JITTER = "jitter"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if service recovered


# Local retryable service unavailable for circuit breaker logic
# Note: This is different from traigent.utils.exceptions.ServiceUnavailableError
# which inherits from ServiceError. This one inherits from RetryableError
# to integrate with the retry logic in this module.
class ServiceUnavailableError(RetryableError):
    """Service temporarily unavailable (retryable variant for circuit breaker)."""


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    This combines features from both retry implementations to provide
    a comprehensive configuration for all retry scenarios.
    """

    # Basic retry settings
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL

    # Advanced settings (from retry_handler)
    retry_on_exception: set[type[Exception]] | None = None
    retry_on_status: set[int] | None = None  # HTTP status codes
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True
    respect_retry_after: bool = True  # Honor Retry-After headers

    # Circuit breaker settings
    enable_circuit_breaker: bool = False
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2

    # Monitoring
    log_attempts: bool = True
    callback_on_retry: Callable[[Exception, int], None] | None = None

    def __post_init__(self) -> None:
        """Initialize default retryable exceptions if not provided."""
        if self.retry_on_exception is None:
            self.retry_on_exception: set[Any] = {
                RetryableError,
                ConnectionError,
                TimeoutError,
                OSError,
                RateLimitError,
                ServiceUnavailableError,
                NetworkError,
            }

        # Ensure strategy is properly set (handle both old and new names)
        if isinstance(self.strategy, str):
            # Map old names to new ones
            strategy_map = {
                "fixed_delay": RetryStrategy.FIXED,
                "exponential_backoff": RetryStrategy.EXPONENTIAL,
                "linear_backoff": RetryStrategy.LINEAR,
                "random_jitter": RetryStrategy.JITTER,
            }
            self.strategy = RetryStrategy(
                strategy_map.get(self.strategy, self.strategy)
            )

    def calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate delay before next retry attempt."""
        if retry_after and self.respect_retry_after:
            return min(retry_after, self.max_delay)

        if self.strategy in (RetryStrategy.FIXED, RetryStrategy.FIXED_DELAY):
            delay = self.initial_delay
        elif self.strategy in (
            RetryStrategy.EXPONENTIAL,
            RetryStrategy.EXPONENTIAL_BACKOFF,
        ):
            delay = min(
                self.initial_delay * (self.exponential_base ** (attempt - 1)),
                self.max_delay,
            )
        elif self.strategy in (RetryStrategy.LINEAR, RetryStrategy.LINEAR_BACKOFF):
            delay = min(self.initial_delay * attempt, self.max_delay)
        else:  # JITTER / RANDOM_JITTER
            base_delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
            delay = min(base_delay, self.max_delay)

        # Add jitter if enabled
        if self.jitter and self.strategy != RetryStrategy.JITTER:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        elif self.strategy == RetryStrategy.JITTER:
            delay = random.uniform(0, delay)

        return delay


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    result: Any | None = None
    error: Exception | None = None
    attempts: int = 0
    total_delay: float = 0.0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def duration(self) -> float:
        """Total duration of the operation."""
        return (self.end_time - self.start_time).total_seconds()


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.last_state_change: datetime = datetime.now(UTC)
        self._lock = RLock()
        self._half_open_in_flight = False

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                now = datetime.now(UTC)
                if self._should_attempt_reset(now):
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    self.last_state_change = now
                    self._half_open_in_flight = True
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise ServiceUnavailableError("Circuit breaker is OPEN")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_in_flight:
                    raise ServiceUnavailableError("Circuit breaker is HALF_OPEN")
                self._half_open_in_flight = True

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self, reference_time: datetime | None = None) -> bool:
        """Check if we should try to reset the circuit."""
        if self.last_failure_time is None:
            return False
        if reference_time is None:
            reference_time = datetime.now(UTC)
        return reference_time - self.last_failure_time > timedelta(
            seconds=self.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                self._half_open_in_flight = False
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0
                    self.last_state_change = datetime.now(UTC)
                    logger.info("Circuit breaker entering CLOSED state")
            else:
                self._half_open_in_flight = False

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(UTC)
            self.success_count = 0
            self._half_open_in_flight = False

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.last_state_change = datetime.now(UTC)
                logger.warning(
                    f"Circuit breaker entering OPEN state after {self.failure_count} failures"
                )


class RetryHandler:
    """Advanced retry handler with circuit breaker support."""

    def __init__(self, config: RetryConfig) -> None:
        """Initialize retry handler."""
        self.config = config
        self.circuit_breaker: CircuitBreaker | None = None

        if config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=config.failure_threshold,
                recovery_timeout=config.recovery_timeout,
                success_threshold=config.success_threshold,
            )

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        return self.execute_with_result(func, *args, **kwargs).result  # type: ignore[return-value]

    def execute_with_result(
        self, func: Callable[..., T], *args, **kwargs
    ) -> RetryResult:
        """Execute function with retry logic and return detailed result."""
        start_time = datetime.now(UTC)
        last_exception = None
        total_delay = 0.0
        final_attempt = 0

        for attempt in range(1, self.config.max_attempts + 1):
            final_attempt = attempt
            try:
                # Use circuit breaker if enabled
                if self.circuit_breaker:
                    result = self.circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_delay=total_delay,
                    start_time=start_time,
                    end_time=datetime.now(UTC),
                )

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self._should_retry(e, attempt):
                    break

                # Calculate delay
                retry_after = getattr(e, "retry_after", None)
                delay = self.config.calculate_delay(attempt, retry_after)

                # Log retry attempt
                if self.config.log_attempts:
                    logger.warning(
                        f"Retry attempt {attempt}/{self.config.max_attempts} after {type(e).__name__}: {e}. "
                        f"Waiting {delay:.1f}s before retry."
                    )

                # Callback
                if self.config.callback_on_retry:
                    self.config.callback_on_retry(e, attempt)

                # Wait before retry
                time.sleep(delay)
                total_delay += delay

        # All retries exhausted or early exit
        return RetryResult(
            success=False,
            error=last_exception,
            attempts=final_attempt,
            total_delay=total_delay,
            start_time=start_time,
            end_time=datetime.now(UTC),
        )

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with retry logic."""
        result = await self.execute_async_with_result(func, *args, **kwargs)
        if result.success:
            return result.result  # type: ignore[return-value]
        raise result.error or RetryError(f"All {result.attempts} retry attempts failed")

    async def execute_async_with_result(
        self, func: Callable[..., T], *args, **kwargs
    ) -> RetryResult:
        """Execute async function with retry logic and return detailed result."""
        start_time = datetime.now(UTC)
        last_exception = None
        total_delay = 0.0
        final_attempt = 0

        for attempt in range(1, self.config.max_attempts + 1):
            final_attempt = attempt
            try:
                result = await func(*args, **kwargs)  # type: ignore[misc]
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_delay=total_delay,
                    start_time=start_time,
                    end_time=datetime.now(UTC),
                )

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self._should_retry(e, attempt):
                    break

                # Calculate delay
                retry_after = getattr(e, "retry_after", None)
                delay = self.config.calculate_delay(attempt, retry_after)

                # Log retry attempt
                if self.config.log_attempts:
                    logger.warning(
                        f"Retry attempt {attempt}/{self.config.max_attempts} after {type(e).__name__}: {e}. "
                        f"Waiting {delay:.1f}s before retry."
                    )

                # Callback
                if self.config.callback_on_retry:
                    self.config.callback_on_retry(e, attempt)

                # Wait before retry
                await asyncio.sleep(delay)
                total_delay += delay

        # All retries exhausted or early exit
        return RetryResult(
            success=False,
            error=last_exception,
            attempts=final_attempt,
            total_delay=total_delay,
            start_time=start_time,
            end_time=datetime.now(UTC),
        )

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry after an exception."""
        if attempt >= self.config.max_attempts:
            return False

        # Check if exception type is retryable
        if isinstance(exception, NonRetryableError):
            return False

        if self.config.retry_on_exception:
            return any(
                isinstance(exception, exc_type)
                for exc_type in self.config.retry_on_exception
            )

        return True


# Decorator functions for simple usage
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    strategy: str | RetryStrategy = RetryStrategy.EXPONENTIAL,
    exceptions: list[type[Exception]] | None = None,
) -> Callable[..., Any]:
    """Simple retry decorator.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        strategy: Retry strategy to use
        exceptions: List of exceptions to retry on

    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=delay,
        strategy=(
            strategy if isinstance(strategy, RetryStrategy) else RetryStrategy(strategy)
        ),
        retry_on_exception=set(exceptions) if exceptions else None,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(config)
            return handler.execute(func, *args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(config)
            return await handler.execute_async(func, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper

    return decorator


def retry_with_config(config: RetryConfig) -> Callable[..., Any]:
    """Retry decorator with custom configuration.

    Args:
        config: RetryConfig instance

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(config)
            return handler.execute(func, *args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            handler = RetryHandler(config)
            return await handler.execute_async(func, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper

    return decorator


# Pre-configured retry configurations
DEFAULT_RETRY = RetryConfig(max_attempts=3, initial_delay=1.0)
AGGRESSIVE_RETRY = RetryConfig(max_attempts=5, initial_delay=0.5, max_delay=30.0)
CONSERVATIVE_RETRY = RetryConfig(max_attempts=2, initial_delay=2.0, max_delay=10.0)

# HTTP-specific configurations
HTTP_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    retry_on_status={429, 500, 502, 503, 504},
    respect_retry_after=True,
)

CLOUD_API_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    max_delay=60.0,
    retry_on_status={429, 500, 502, 503, 504},
    respect_retry_after=True,
    enable_circuit_breaker=True,
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    strategy=RetryStrategy.EXPONENTIAL,
    retry_on_exception={ConnectionError, TimeoutError},
)


# Utility functions
@contextmanager
def ErrorContext(operation: str, retryable: bool = True):
    """Context manager for error handling with retry hints.

    Args:
        operation: Description of the operation
        retryable: Whether errors should be retryable

    Example:
        with ErrorContext("API call", retryable=True):
            response = requests.get(url)
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation}: {e}")
        if retryable and not isinstance(e, (RetryableError, NonRetryableError)):
            # Wrap in RetryableError to hint it should be retried
            raise RetryableError(f"Error during {operation}: {e}") from e
        raise


def retry_http_request(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator specifically for HTTP requests."""
    return cast(Callable[..., T], retry_with_config(HTTP_RETRY_CONFIG)(func))


def retry_cloud_api(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator specifically for cloud API calls."""
    return cast(Callable[..., T], retry_with_config(CLOUD_API_RETRY_CONFIG)(func))


def retry_database_operation(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator specifically for database operations."""
    return cast(Callable[..., T], retry_with_config(DATABASE_RETRY_CONFIG)(func))


class RetryWithCircuitBreaker:
    """Combined retry and circuit breaker pattern."""

    def __init__(self, retry_config: RetryConfig) -> None:
        """Initialize with retry configuration."""
        # Ensure circuit breaker is enabled
        retry_config.enable_circuit_breaker = True
        self.handler = RetryHandler(retry_config)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator implementation."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.handler.execute(func, *args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            return await self.handler.execute_async(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
