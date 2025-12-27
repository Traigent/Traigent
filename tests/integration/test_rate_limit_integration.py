"""Integration tests for rate limit handling with optimizer.

This module tests how the optimizer and evaluator handle rate limits
during optimization runs, including transient failures, recovery,
and parallel execution scenarios.
"""

from __future__ import annotations

import asyncio

import pytest

from traigent.utils.exceptions import RateLimitError
from traigent.utils.retry import RetryConfig, RetryHandler


class TestOptimizerRateLimitResilience:
    """Integration tests for optimizer rate limit handling."""

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_optimization_survives_transient_rate_limit(
        self,
        rate_limited_async_function,
    ) -> None:
        """Optimization should continue after transient rate limit.

        This test simulates a function that gets rate limited on specific
        calls but recovers, verifying that the optimization can continue.
        """

        async def mock_llm_call(config: dict) -> dict:
            """Simulate an LLM call that returns mock metrics."""
            return {
                "accuracy": 0.85 + (hash(str(config)) % 100) / 1000,
                "model": config.get("model", "unknown"),
            }

        # Fail on calls 2 and 3, then succeed
        wrapped = rate_limited_async_function(
            mock_llm_call,
            fail_on_calls=[2, 3],
            retry_after=0.01,
        )

        results = []
        configs = [
            {"model": "gpt-3.5-turbo"},
            {"model": "gpt-4"},
            {"model": "gpt-4-turbo"},
        ]

        for cfg in configs:
            # Async retry loop
            for _ in range(5):
                try:
                    result = await wrapped(cfg)
                    results.append(result)
                    break
                except RateLimitError:
                    await asyncio.sleep(0.01)

        # Should have results from all configs despite rate limits
        assert len(results) == 3
        assert all("accuracy" in r for r in results)

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_optimization_with_persistent_rate_limit(
        self,
        rate_limited_async_function,
    ) -> None:
        """Optimization should fail gracefully on persistent rate limit.

        When rate limits persist beyond retry attempts, the optimization
        should fail gracefully without crashing.
        """

        async def mock_llm_call(config: dict) -> dict:
            return {"accuracy": 0.85}

        # Always fail (100% probability)
        wrapped = rate_limited_async_function(
            mock_llm_call,
            fail_probability=1.0,
            retry_after=0.01,
        )

        max_attempts = 3
        attempts = 0
        last_error = None

        for _ in range(max_attempts):
            attempts += 1
            try:
                await wrapped({"model": "gpt-4"})
                break
            except RateLimitError as e:
                last_error = e
                await asyncio.sleep(0.01)

        # Should fail after max attempts
        assert last_error is not None
        assert isinstance(last_error, RateLimitError)
        assert attempts == max_attempts

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_parallel_trials_handle_rate_limits(
        self,
        transient_rate_limiter,
    ) -> None:
        """Parallel execution should handle rate limits without corruption.

        When running multiple trials in parallel, rate limits on one
        trial should not corrupt other trials.
        """

        async def mock_llm_call(config: dict) -> dict:
            await asyncio.sleep(0.01)  # Simulate API call
            return {"accuracy": 0.85, "config": config}

        # Create multiple wrapped functions with different failure patterns
        funcs = [
            transient_rate_limiter(mock_llm_call, fail_count=0),  # No failures
            transient_rate_limiter(mock_llm_call, fail_count=1),  # 1 failure
            transient_rate_limiter(mock_llm_call, fail_count=2),  # 2 failures
        ]

        async def run_with_retry(fn, config):
            """Run function with retry on rate limit."""
            for _ in range(5):
                try:
                    return await fn(config)
                except RateLimitError:
                    await asyncio.sleep(0.01)
            raise RateLimitError("Max retries exceeded")

        # Run in parallel
        configs = [{"model": f"model-{i}"} for i in range(3)]
        tasks = [
            run_with_retry(fn, cfg) for fn, cfg in zip(funcs, configs, strict=True)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed after retries
        assert len(results) == 3
        assert all("accuracy" in r for r in results)
        # Each should have its own config
        assert {r["config"]["model"] for r in results} == {
            "model-0",
            "model-1",
            "model-2",
        }


class TestEvaluatorRateLimitHandling:
    """Tests for evaluator-level rate limit handling."""

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_evaluator_retries_on_rate_limit(
        self,
        rate_limited_async_function,
    ) -> None:
        """Evaluator should retry when rate limited."""

        async def evaluate_example(example: dict) -> dict:
            """Simulate evaluating a single example."""
            return {
                "correct": example.get("expected") == example.get("actual"),
                "example_id": example.get("id"),
            }

        # Fail on first call only
        wrapped = rate_limited_async_function(
            evaluate_example,
            fail_on_calls=[1],
            retry_after=0.01,
        )

        # Simulate evaluator with retry
        examples = [
            {"id": 1, "expected": "yes", "actual": "yes"},
            {"id": 2, "expected": "no", "actual": "no"},
            {"id": 3, "expected": "yes", "actual": "no"},
        ]

        results = []
        for example in examples:
            for _ in range(3):
                try:
                    result = await wrapped(example)
                    results.append(result)
                    break
                except RateLimitError:
                    await asyncio.sleep(0.01)

        # Should have evaluated all examples
        assert len(results) == 3
        assert sum(r["correct"] for r in results) == 2

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_partial_evaluation_on_rate_limit(
        self,
        sequential_rate_limiter_factory,
    ) -> None:
        """Partial results should be preserved when rate limit hits mid-eval.

        If rate limit occurs during evaluation, already-completed examples
        should be preserved.
        """
        limiter = sequential_rate_limiter_factory(
            fail_on_calls=[3],  # Fail on 3rd example
            retry_after=0.01,
        )

        async def evaluate_example(example: dict) -> dict:
            result = limiter.check_rate_limit()
            limiter.record_request()

            if result.limited:
                raise RateLimitError(result.message, retry_after=result.retry_after)

            return {"id": example["id"], "score": 1.0}

        examples = [{"id": i} for i in range(5)]
        results = []
        failed_at = None

        for i, example in enumerate(examples):
            try:
                result = await evaluate_example(example)
                results.append(result)
            except RateLimitError:
                failed_at = i
                break

        # Should have preserved results before failure
        assert len(results) == 2  # Examples 0 and 1 completed
        assert failed_at == 2  # Failed on 3rd (index 2)
        assert [r["id"] for r in results] == [0, 1]


class TestRetryWithRateLimitSimulator:
    """Integration tests combining retry handler with rate limit simulator."""

    @pytest.mark.integration
    @pytest.mark.rate_limit
    def test_rpm_limit_with_retry(
        self,
        rate_limit_simulator_factory,
    ) -> None:
        """Retry handler should work with RPM-limited operations."""
        simulator = rate_limit_simulator_factory(
            rpm_limit=3,
            retry_after=0.05,
            window_seconds=0.1,  # Very short window for testing
        )

        config = RetryConfig(
            max_attempts=10,
            initial_delay=0.02,
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

        # Make multiple calls - some will be rate limited but should succeed
        results = []
        for _ in range(5):
            result = handler.execute_with_result(rate_limited_operation)
            if result.success:
                results.append(result.result)

        # Should have gotten some results despite rate limits
        assert len(results) >= 1

    @pytest.mark.integration
    @pytest.mark.rate_limit
    def test_burst_limit_with_retry(
        self,
        rate_limit_simulator_factory,
    ) -> None:
        """Retry handler should handle burst limits."""
        simulator = rate_limit_simulator_factory(
            burst_limit=2,
            retry_after=0.01,
        )

        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        def burst_limited_operation() -> str:
            result = simulator.check_rate_limit()
            if result.limited:
                simulator.reset_window()  # Simulate burst window reset
                raise RateLimitError(result.message, retry_after=result.retry_after)

            simulator.record_request()
            return "success"

        # First two should succeed, third should hit limit but retry after reset
        for i in range(3):
            result = handler.execute_with_result(burst_limited_operation)
            assert result.success, f"Call {i+1} should succeed after retry"


class TestRateLimitRecoveryPatterns:
    """Tests for various rate limit recovery patterns."""

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_exponential_backoff_recovery(
        self,
        transient_rate_limiter,
    ) -> None:
        """Exponential backoff should eventually recover from rate limits."""
        import time

        async def tracked_call(config: dict) -> dict:
            return {"result": "success"}

        # Fail first 2 calls
        wrapped = transient_rate_limiter(
            tracked_call,
            fail_count=2,
            retry_after=0.02,
        )

        result = None
        attempt_times = []
        # Manual exponential backoff
        delay = 0.01
        for _ in range(5):
            attempt_times.append(time.monotonic())
            try:
                result = await wrapped({"model": "test"})
                break
            except RateLimitError:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        assert result is not None
        assert result["result"] == "success"
        # Should have made 3 attempts total (2 failed + 1 success)
        assert len(attempt_times) == 3

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_retry_after_header_respected(
        self,
        sequential_rate_limiter_factory,
    ) -> None:
        """System should respect Retry-After header values."""
        import time

        limiter = sequential_rate_limiter_factory(
            fail_on_calls=[1],
            retry_after=0.05,  # Suggest waiting 50ms
        )

        async def api_call() -> str:
            result = limiter.check_rate_limit()
            limiter.record_request()

            if result.limited:
                raise RateLimitError(
                    result.message,
                    retry_after=result.retry_after,
                )

            return "success"

        start = time.monotonic()
        result = None

        for _ in range(3):
            try:
                result = await api_call()
                break
            except RateLimitError as e:
                if e.retry_after:
                    await asyncio.sleep(e.retry_after)

        elapsed = time.monotonic() - start

        assert result == "success"
        # Should have waited at least the retry_after time
        assert elapsed >= 0.05


class TestConcurrentRateLimiting:
    """Tests for rate limiting in concurrent scenarios."""

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_rpm(
        self,
        rate_limit_simulator_factory,
    ) -> None:
        """Concurrent requests should collectively respect RPM limits."""
        simulator = rate_limit_simulator_factory(
            rpm_limit=5,
            retry_after=0.01,
        )

        async def make_request(request_id: int) -> dict:
            result = simulator.check_rate_limit()
            if result.limited:
                raise RateLimitError(result.message)

            simulator.record_request()
            return {"id": request_id, "status": "success"}

        # Try to make 10 concurrent requests when limit is 5
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if isinstance(r, RateLimitError)]

        # First 5 should succeed, rest should fail
        assert len(successes) == 5
        assert len(failures) == 5

    @pytest.mark.integration
    @pytest.mark.rate_limit
    @pytest.mark.asyncio
    async def test_staggered_requests_avoid_rate_limit(
        self,
        rate_limit_simulator_factory,
    ) -> None:
        """Staggered requests should avoid triggering rate limits."""
        simulator = rate_limit_simulator_factory(
            rpm_limit=10,
            window_seconds=0.1,  # Short window
        )

        async def make_request(request_id: int) -> dict:
            result = simulator.check_rate_limit()
            if result.limited:
                raise RateLimitError(result.message)

            simulator.record_request()
            return {"id": request_id, "status": "success"}

        results = []
        for i in range(15):
            try:
                result = await make_request(i)
                results.append(result)
            except RateLimitError:
                # Wait for window to reset
                await asyncio.sleep(0.1)
                # Retry
                result = await make_request(i)
                results.append(result)

        # All should eventually succeed
        assert len(results) == 15
