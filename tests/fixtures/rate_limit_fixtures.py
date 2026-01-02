"""pytest fixtures for rate limit testing.

This module provides pytest fixtures for simulating rate limits during tests,
including factory fixtures for creating simulators and decorators for
injecting rate limit errors into functions.
"""

from __future__ import annotations

import functools
import random
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

import pytest

from traigent.utils.exceptions import RateLimitError

from .rate_limit_simulator import (
    ProviderRateLimitSimulator,
    RateLimitConfig,
    RateLimitSimulator,
    SequentialRateLimiter,
)

P = ParamSpec("P")
T = TypeVar("T")


@pytest.fixture
def rate_limit_config() -> type[RateLimitConfig]:
    """Provide access to RateLimitConfig class for creating configs."""
    return RateLimitConfig


@pytest.fixture
def rate_limit_simulator_factory():
    """Factory fixture for creating rate limit simulators.

    Returns:
        A factory function that creates RateLimitSimulator instances

    Example:
        ```python
        def test_rate_limits(rate_limit_simulator_factory):
            simulator = rate_limit_simulator_factory(
                rpm_limit=10,
                retry_after=0.5,
            )
            # Use simulator...
        ```
    """

    def _create(
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        burst_limit: int | None = None,
        daily_limit: int | None = None,
        retry_after: float = 1.0,
        error_probability: float = 0.0,
        provider: str = "openai",
        window_seconds: float = 60.0,
        random_seed: int | None = None,
    ) -> RateLimitSimulator:
        config = RateLimitConfig(
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            burst_limit=burst_limit,
            daily_limit=daily_limit,
            retry_after=retry_after,
            error_probability=error_probability,
            provider=provider,
            window_seconds=window_seconds,
        )
        return RateLimitSimulator(config, random_seed=random_seed)

    return _create


@pytest.fixture
def provider_rate_limiter_factory():
    """Factory fixture for creating provider-specific rate limit simulators.

    Returns:
        A factory function that creates ProviderRateLimitSimulator instances

    Example:
        ```python
        def test_openai_limits(provider_rate_limiter_factory):
            simulator = provider_rate_limiter_factory("openai")
            # Uses default OpenAI limits
        ```
    """

    def _create(
        provider: str = "openai",
        config_overrides: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ) -> ProviderRateLimitSimulator:
        return ProviderRateLimitSimulator(
            provider=provider,
            config_overrides=config_overrides,
            random_seed=random_seed,
        )

    return _create


@pytest.fixture
def sequential_rate_limiter_factory():
    """Factory fixture for creating sequential rate limiters.

    Returns:
        A factory function that creates SequentialRateLimiter instances

    Example:
        ```python
        def test_retry_on_third_call(sequential_rate_limiter_factory):
            limiter = sequential_rate_limiter_factory(fail_on_calls=[3])
            # Fails only on 3rd call
        ```
    """

    def _create(
        fail_on_calls: list[int] | None = None,
        retry_after: float = 1.0,
        provider: str = "openai",
    ) -> SequentialRateLimiter:
        return SequentialRateLimiter(
            fail_on_calls=fail_on_calls,
            retry_after=retry_after,
            provider=provider,
        )

    return _create


@pytest.fixture
def rate_limited_async_function():
    """Fixture that creates rate-limited async function wrappers.

    Returns:
        A factory function that wraps async functions with rate limit injection

    Example:
        ```python
        async def test_function_with_rate_limits(rate_limited_async_function):
            async def my_func():
                return "success"

            # Fail on calls 2 and 3, then succeed
            wrapped = rate_limited_async_function(
                my_func,
                fail_on_calls=[2, 3],
            )

            result1 = await wrapped()  # Succeeds
            try:
                result2 = await wrapped()  # Fails with RateLimitError
            except RateLimitError:
                pass
        ```
    """

    def _create(
        fn: Callable[P, Awaitable[T]],
        fail_on_calls: list[int] | None = None,
        fail_probability: float = 0.0,
        retry_after: float = 1.0,
        random_seed: int | None = None,
    ) -> Callable[P, Awaitable[T]]:
        call_count = 0
        fail_set = set(fail_on_calls or [])
        rng = random.Random(random_seed)

        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal call_count
            call_count += 1

            # Check deterministic failures
            if call_count in fail_set:
                raise RateLimitError(
                    f"Rate limit exceeded (call {call_count})",
                    retry_after=retry_after,
                )

            # Check probabilistic failures
            if fail_probability > 0 and rng.random() < fail_probability:
                raise RateLimitError(
                    "Rate limit exceeded (random)",
                    retry_after=retry_after,
                )

            return await fn(*args, **kwargs)

        # Attach helper methods
        wrapped.get_call_count = lambda: call_count  # type: ignore[attr-defined]
        wrapped.reset_call_count = lambda: call_count  # type: ignore[attr-defined]

        return wrapped

    return _create


@pytest.fixture
def rate_limited_sync_function():
    """Fixture that creates rate-limited sync function wrappers.

    Returns:
        A factory function that wraps sync functions with rate limit injection

    Example:
        ```python
        def test_sync_function_with_rate_limits(rate_limited_sync_function):
            def my_func():
                return "success"

            wrapped = rate_limited_sync_function(
                my_func,
                fail_on_calls=[1],  # Fail on first call
            )

            try:
                result = wrapped()  # Fails with RateLimitError
            except RateLimitError:
                pass
        ```
    """

    def _create(
        fn: Callable[P, T],
        fail_on_calls: list[int] | None = None,
        fail_probability: float = 0.0,
        retry_after: float = 1.0,
        random_seed: int | None = None,
    ) -> Callable[P, T]:
        call_count = 0
        fail_set = set(fail_on_calls or [])
        rng = random.Random(random_seed)

        @functools.wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal call_count
            call_count += 1

            # Check deterministic failures
            if call_count in fail_set:
                raise RateLimitError(
                    f"Rate limit exceeded (call {call_count})",
                    retry_after=retry_after,
                )

            # Check probabilistic failures
            if fail_probability > 0 and rng.random() < fail_probability:
                raise RateLimitError(
                    "Rate limit exceeded (random)",
                    retry_after=retry_after,
                )

            return fn(*args, **kwargs)

        # Attach helper methods
        wrapped.get_call_count = lambda: call_count  # type: ignore[attr-defined]
        wrapped.reset_call_count = lambda: call_count  # type: ignore[attr-defined]

        return wrapped

    return _create


@pytest.fixture
def transient_rate_limiter():
    """Fixture that creates a limiter that fails N times then succeeds.

    Returns:
        A factory function that creates a transient failure wrapper

    Example:
        ```python
        async def test_transient_failure(transient_rate_limiter):
            async def my_func():
                return "success"

            # Fail first 3 calls, then always succeed
            wrapped = transient_rate_limiter(my_func, fail_count=3)

            for i in range(3):
                with pytest.raises(RateLimitError):
                    await wrapped()

            result = await wrapped()  # Succeeds
            assert result == "success"
        ```
    """

    def _create(
        fn: Callable[P, Awaitable[T]],
        fail_count: int = 1,
        retry_after: float = 1.0,
    ) -> Callable[P, Awaitable[T]]:
        call_count = 0

        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal call_count
            call_count += 1

            if call_count <= fail_count:
                raise RateLimitError(
                    f"Rate limit exceeded ({call_count}/{fail_count})",
                    retry_after=retry_after,
                )

            return await fn(*args, **kwargs)

        wrapped.get_call_count = lambda: call_count  # type: ignore[attr-defined]

        return wrapped

    return _create


@pytest.fixture
def burst_rate_limiter():
    """Fixture that creates a limiter simulating burst limits.

    The limiter allows N requests in quick succession, then rate limits
    until a cooldown period has passed.

    Returns:
        A factory function that creates a burst limiter wrapper

    Example:
        ```python
        async def test_burst_limit(burst_rate_limiter):
            async def my_func():
                return "success"

            # Allow 5 requests, then rate limit for 0.5s
            wrapped = burst_rate_limiter(my_func, burst_size=5, cooldown=0.5)

            # First 5 succeed
            for _ in range(5):
                await wrapped()

            # 6th fails
            with pytest.raises(RateLimitError):
                await wrapped()

            # After cooldown, works again
            await asyncio.sleep(0.6)
            await wrapped()  # Succeeds
        ```
    """
    import time

    def _create(
        fn: Callable[P, Awaitable[T]],
        burst_size: int = 5,
        cooldown: float = 1.0,
        retry_after: float | None = None,
    ) -> Callable[P, Awaitable[T]]:
        request_times: list[float] = []
        effective_retry_after = retry_after if retry_after is not None else cooldown

        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            now = time.monotonic()

            # Remove requests older than cooldown period
            while request_times and (now - request_times[0]) > cooldown:
                request_times.pop(0)

            # Check if at burst limit
            if len(request_times) >= burst_size:
                raise RateLimitError(
                    f"Burst limit exceeded ({burst_size} requests)",
                    retry_after=effective_retry_after,
                )

            # Record this request and execute
            request_times.append(now)
            return await fn(*args, **kwargs)

        wrapped.get_request_count = lambda: len(request_times)  # type: ignore[attr-defined]
        wrapped.reset = lambda: request_times.clear()  # type: ignore[attr-defined]

        return wrapped

    return _create


@pytest.fixture
def chaos_rate_limiter():
    """Fixture that creates a limiter with random failures.

    Returns:
        A factory function that creates a chaos limiter wrapper

    Example:
        ```python
        async def test_chaos_mode(chaos_rate_limiter):
            async def my_func():
                return "success"

            # 30% chance of rate limit on each call
            wrapped = chaos_rate_limiter(my_func, fail_probability=0.3)

            successes = 0
            failures = 0
            for _ in range(100):
                try:
                    await wrapped()
                    successes += 1
                except RateLimitError:
                    failures += 1

            # Should have ~70 successes, ~30 failures
            assert 50 < successes < 90
        ```
    """

    def _create(
        fn: Callable[P, Awaitable[T]],
        fail_probability: float = 0.1,
        retry_after: float = 1.0,
        random_seed: int | None = None,
    ) -> Callable[P, Awaitable[T]]:
        rng = random.Random(random_seed)

        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            if rng.random() < fail_probability:
                raise RateLimitError(
                    "Rate limit exceeded (chaos mode)",
                    retry_after=retry_after,
                )

            return await fn(*args, **kwargs)

        return wrapped

    return _create


# Convenience fixture that provides all rate limit test utilities
@pytest.fixture
def rate_limit_utils(
    rate_limit_simulator_factory,
    provider_rate_limiter_factory,
    sequential_rate_limiter_factory,
    rate_limited_async_function,
    rate_limited_sync_function,
    transient_rate_limiter,
    burst_rate_limiter,
    chaos_rate_limiter,
):
    """Convenience fixture providing all rate limit testing utilities.

    Returns:
        A namespace object with all rate limit fixtures

    Example:
        ```python
        def test_with_all_utils(rate_limit_utils):
            simulator = rate_limit_utils.simulator(rpm_limit=10)
            provider_sim = rate_limit_utils.provider("openai")
            # etc.
        ```
    """

    class RateLimitUtils:
        simulator = staticmethod(rate_limit_simulator_factory)
        provider = staticmethod(provider_rate_limiter_factory)
        sequential = staticmethod(sequential_rate_limiter_factory)
        async_wrapper = staticmethod(rate_limited_async_function)
        sync_wrapper = staticmethod(rate_limited_sync_function)
        transient = staticmethod(transient_rate_limiter)
        burst = staticmethod(burst_rate_limiter)
        chaos = staticmethod(chaos_rate_limiter)

    return RateLimitUtils()
