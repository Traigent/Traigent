"""Chaos tests for rate limit handling under unpredictable conditions.

These tests inject random failures and verify the system handles them
gracefully. They are marked as slow and may occasionally be flaky due
to their randomized nature.
"""

from __future__ import annotations

import asyncio
import random
import time

import pytest

from traigent.utils.exceptions import RateLimitError


class TestRandomRateLimitChaos:
    """Chaos tests with random rate limit failures."""

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_random_rate_limits_during_batch_processing(
        self,
        chaos_rate_limiter,
    ) -> None:
        """Batch processing should handle random rate limits.

        Simulates a batch of requests where some randomly fail with
        rate limits, verifying that retry logic eventually succeeds.
        """
        success_count = 0

        async def process_item(item_id: int) -> dict:
            nonlocal success_count
            success_count += 1
            return {"id": item_id, "status": "processed"}

        # 20% chance of rate limit on each call
        wrapped = chaos_rate_limiter(
            process_item,
            fail_probability=0.2,
            retry_after=0.01,
            random_seed=42,  # Reproducible for testing
        )

        results = []
        batch_size = 20

        for item_id in range(batch_size):
            # Retry up to 5 times per item
            for _ in range(5):
                try:
                    result = await wrapped(item_id)
                    results.append(result)
                    break
                except RateLimitError:
                    await asyncio.sleep(0.01)
            else:
                # If all retries failed, record as failed
                results.append({"id": item_id, "status": "failed"})

        # With 20% failure rate and 5 retries, most should succeed
        successful = [r for r in results if r["status"] == "processed"]
        assert len(successful) >= batch_size * 0.8  # At least 80% success

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_failure_rate_still_recovers(
        self,
        chaos_rate_limiter,
    ) -> None:
        """System should eventually recover even with high failure rate."""

        async def api_call() -> str:
            return "success"

        # 50% failure rate - very aggressive
        wrapped = chaos_rate_limiter(
            api_call,
            fail_probability=0.5,
            retry_after=0.005,
            random_seed=123,
        )

        # With 50% failure rate, should still succeed with enough retries
        successes = 0
        attempts = 0
        max_attempts = 50

        while successes < 10 and attempts < max_attempts:
            attempts += 1
            try:
                await wrapped()
                successes += 1
            except RateLimitError:
                await asyncio.sleep(0.005)

        # Should have gotten at least 10 successes
        assert successes >= 10
        # But it should have taken more than 10 attempts (due to failures)
        assert attempts > 10

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_chaos_operations(
        self,
        chaos_rate_limiter,
    ) -> None:
        """Concurrent operations should handle chaos independently."""

        async def process(task_id: int) -> dict:
            await asyncio.sleep(0.005)  # Simulate work
            return {"task_id": task_id, "completed": True}

        # 30% failure rate
        wrapped = chaos_rate_limiter(
            process,
            fail_probability=0.3,
            retry_after=0.01,
            random_seed=456,
        )

        async def run_with_retries(task_id: int) -> dict:
            for _ in range(10):
                try:
                    return await wrapped(task_id)
                except RateLimitError:
                    await asyncio.sleep(0.01)
            return {"task_id": task_id, "completed": False}

        # Run 10 concurrent tasks
        tasks = [run_with_retries(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should complete (with retries)
        completed = [r for r in results if r["completed"]]
        assert len(completed) >= 8  # At least 80% should succeed


class TestBurstRateLimitChaos:
    """Chaos tests for burst rate limiting scenarios."""

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_burst_recovery_under_load(
        self,
        burst_rate_limiter,
    ) -> None:
        """System should recover from burst limits under continuous load."""

        async def api_call() -> str:
            return "success"

        # Burst of 3, cooldown of 50ms
        wrapped = burst_rate_limiter(
            api_call,
            burst_size=3,
            cooldown=0.05,
            retry_after=0.05,
        )

        results = []
        rate_limited_count = 0

        # Try to make 15 calls in quick succession
        for _ in range(15):
            try:
                result = await wrapped()
                results.append(result)
            except RateLimitError:
                rate_limited_count += 1
                # Wait for cooldown
                await asyncio.sleep(0.06)
                # Retry after cooldown
                try:
                    result = await wrapped()
                    results.append(result)
                except RateLimitError:
                    pass

        # Should have hit rate limit multiple times
        assert rate_limited_count > 0
        # But should have recovered and gotten results
        assert len(results) >= 5

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_burst_with_parallel_requests(
        self,
        burst_rate_limiter,
    ) -> None:
        """Parallel requests should compete for burst capacity."""

        async def api_call(request_id: int) -> dict:
            await asyncio.sleep(0.001)
            return {"id": request_id, "status": "success"}

        # Small burst size
        wrapped = burst_rate_limiter(
            api_call,
            burst_size=2,
            cooldown=0.03,
        )

        async def make_request(request_id: int) -> dict | None:
            for _ in range(5):
                try:
                    return await wrapped(request_id)
                except RateLimitError:
                    await asyncio.sleep(0.04)
            return None

        # Launch 5 parallel requests competing for 2 slots
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should eventually succeed
        successful = [r for r in results if r is not None]
        assert len(successful) == 5


class TestTransientFailureChaos:
    """Chaos tests for transient failure patterns."""

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_intermittent_failures_pattern(
        self,
        rate_limited_async_function,
    ) -> None:
        """System handles intermittent failure patterns."""

        async def api_call(x: int) -> int:
            return x * 2

        # Fail on calls 2, 5, 8, 11 (every 3rd call after first)
        wrapped = rate_limited_async_function(
            api_call,
            fail_on_calls=[2, 5, 8, 11],
            retry_after=0.01,
        )

        results = []
        for i in range(10):
            for _ in range(3):
                try:
                    result = await wrapped(i)
                    results.append(result)
                    break
                except RateLimitError:
                    await asyncio.sleep(0.01)

        # Should have all 10 results
        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_escalating_failures(
        self,
        transient_rate_limiter,
    ) -> None:
        """System handles escalating failure sequences."""

        async def operation() -> str:
            return "complete"

        # Fail first 5 calls
        wrapped = transient_rate_limiter(
            operation,
            fail_count=5,
            retry_after=0.01,
        )

        result = None
        attempts = 0
        delay = 0.01

        # Exponential backoff with max attempts
        for _ in range(10):
            attempts += 1
            try:
                result = await wrapped()
                break
            except RateLimitError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 0.1)  # Cap at 100ms

        assert result == "complete"
        assert attempts == 6  # 5 failures + 1 success


class TestMixedFailureChaos:
    """Chaos tests with mixed failure types."""

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_multiple_error_types(
        self,
        chaos_rate_limiter,
    ) -> None:
        """System handles multiple concurrent error sources."""
        rng = random.Random(789)
        call_count = 0

        async def flaky_operation() -> dict:
            nonlocal call_count
            call_count += 1

            # 20% chance of transient error (non-rate-limit)
            if rng.random() < 0.2:
                raise ConnectionError("Transient connection error")

            return {"call": call_count, "status": "ok"}

        # Add 20% rate limit on top
        wrapped = chaos_rate_limiter(
            flaky_operation,
            fail_probability=0.2,
            retry_after=0.01,
            random_seed=789,
        )

        results = []
        errors = {"rate_limit": 0, "connection": 0}

        for _ in range(20):
            for _ in range(5):
                try:
                    result = await wrapped()
                    results.append(result)
                    break
                except RateLimitError:
                    errors["rate_limit"] += 1
                    await asyncio.sleep(0.01)
                except ConnectionError:
                    errors["connection"] += 1
                    await asyncio.sleep(0.01)

        # Should have gotten some results despite mixed errors
        assert len(results) >= 10

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cascading_retry_backoff(
        self,
        rate_limit_simulator_factory,
    ) -> None:
        """Retry backoff should prevent cascading failures."""
        simulator = rate_limit_simulator_factory(
            rpm_limit=5,
            retry_after=0.02,
            window_seconds=0.1,
        )

        call_times: list[float] = []

        async def rate_limited_call() -> str:
            result = simulator.check_rate_limit()
            if result.limited:
                raise RateLimitError(result.message, retry_after=result.retry_after)
            simulator.record_request()
            call_times.append(time.monotonic())
            return "success"

        results = []
        for _ in range(10):
            # Manual retry with backoff
            delay = 0.02
            for _ in range(10):
                try:
                    result = await rate_limited_call()
                    results.append(result)
                    break
                except RateLimitError:
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 0.1)
            # Small delay between batches
            await asyncio.sleep(0.02)

        # Should have gotten some results
        assert len(results) >= 3


class TestLongRunningChaos:
    """Chaos tests for long-running operations."""

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_rate_limiting(
        self,
        rate_limit_simulator_factory,
    ) -> None:
        """System handles sustained rate limiting over time."""
        simulator = rate_limit_simulator_factory(
            rpm_limit=10,
            window_seconds=0.2,  # 200ms window
            retry_after=0.05,
        )

        async def api_call() -> str:
            result = simulator.check_rate_limit()
            if result.limited:
                raise RateLimitError(result.message, retry_after=result.retry_after)
            simulator.record_request()
            return "success"

        successes = 0
        rate_limits = 0
        start_time = time.monotonic()
        duration = 0.5  # Run for 500ms

        while time.monotonic() - start_time < duration:
            try:
                await api_call()
                successes += 1
            except RateLimitError:
                rate_limits += 1
                await asyncio.sleep(0.05)

        # Should have hit rate limits and recovered multiple times
        assert rate_limits > 0
        assert successes > 0
        # Success rate should be reasonable
        total = successes + rate_limits
        assert successes / total >= 0.3  # At least 30% success rate

    @pytest.mark.chaos
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_variable_load_handling(
        self,
        rate_limit_simulator_factory,
    ) -> None:
        """System handles variable load patterns."""
        simulator = rate_limit_simulator_factory(
            rpm_limit=5,
            window_seconds=0.1,
            retry_after=0.02,
        )

        async def api_call() -> str:
            result = simulator.check_rate_limit()
            if result.limited:
                raise RateLimitError(result.message, retry_after=result.retry_after)
            simulator.record_request()
            return "success"

        results_by_phase: dict[str, list] = {"low": [], "high": [], "low2": []}

        # Low load phase
        for _ in range(3):
            try:
                result = await api_call()
                results_by_phase["low"].append(result)
            except RateLimitError:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.05)  # Slow pace

        # High load phase (burst)
        for _ in range(10):
            try:
                result = await api_call()
                results_by_phase["high"].append(result)
            except RateLimitError:
                await asyncio.sleep(0.02)

        # Back to low load
        await asyncio.sleep(0.2)  # Let window reset
        for _ in range(3):
            try:
                result = await api_call()
                results_by_phase["low2"].append(result)
            except RateLimitError:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.05)

        # Low load phases should have high success rate
        assert len(results_by_phase["low"]) >= 2
        assert len(results_by_phase["low2"]) >= 2
        # High load should have some successes but likely hit limits
        assert len(results_by_phase["high"]) >= 1
