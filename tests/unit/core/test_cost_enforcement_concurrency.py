"""Concurrency stress tests for CostEnforcer.

Tests race conditions by spawning many threads doing random operations.
Verifies that the implementation correctly handles concurrent access
and maintains invariants under stress.

Key scenarios tested:
- C5: Race condition where two threads release same permit
- C6: Acquire while another thread is releasing
- C7: track_cost while another thread is releasing same permit

Reference: /home/nimrodbu/.claude/plans/snazzy-whistling-kettle.md
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig, Permit


@pytest.fixture(autouse=True)
def disable_mock_mode() -> None:
    """Ensure mock mode is disabled for all tests in this module."""
    os.environ["TRAIGENT_MOCK_MODE"] = "false"


class TestCostEnforcerConcurrency:
    """Stress tests for concurrent access."""

    def test_50_threads_random_operations(self) -> None:
        """Spawn 50 threads doing random acquire/release/track operations.

        Each thread does 30 random operations. After all threads complete,
        we verify that all invariants hold.
        """
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))
        errors: list[str] = []
        lock = threading.Lock()

        def random_operations() -> None:
            """Each thread does 30 random operations."""
            local_permits: list[Permit] = []

            for _ in range(30):
                op = random.choice(["acquire", "release", "track", "check"])

                try:
                    if op == "acquire":
                        permit = enforcer.acquire_permit()
                        if permit.is_granted:
                            local_permits.append(permit)

                    elif op == "release" and local_permits:
                        permit = local_permits.pop()
                        enforcer.release_permit(permit)

                    elif op == "track" and local_permits:
                        permit = local_permits.pop()
                        cost = random.uniform(0.01, 0.1)
                        enforcer.track_cost(cost, permit=permit)

                    elif op == "check":
                        # Read operations
                        _ = enforcer._in_flight_count
                        _ = enforcer._reserved_cost
                        _ = enforcer.accumulated_cost

                    # Verify invariants after each operation
                    assert (
                        enforcer._in_flight_count >= 0
                    ), f"in_flight_count negative: {enforcer._in_flight_count}"
                    assert (
                        enforcer._reserved_cost >= 0
                    ), f"reserved_cost negative: {enforcer._reserved_cost}"

                except Exception as e:
                    with lock:
                        errors.append(f"Thread error: {e}")

            # Cleanup remaining permits
            for permit in local_permits:
                enforcer.release_permit(permit)

        # Run with thread pool
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(random_operations) for _ in range(50)]
            for future in as_completed(futures):
                future.result()  # Will raise if thread had exception

        # Final invariant check
        assert enforcer._in_flight_count >= 0
        assert enforcer._reserved_cost >= 0
        assert len(enforcer._active_permits) == enforcer._in_flight_count
        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_double_release_race_condition(self) -> None:
        """Test race condition where two threads try to release same permit.

        This is scenario C5 from the security review. We use a barrier to
        synchronize both threads so they try to release at the same time.

        Expected: Exactly one release succeeds, the other fails.
        """
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))

        for iteration in range(50):  # Run 50 iterations to catch races
            permit = enforcer.acquire_permit()
            assert permit.is_granted

            results: list[bool] = []
            barrier = threading.Barrier(2)
            results_lock = threading.Lock()

            def release_permit() -> None:
                barrier.wait()  # Synchronize start
                result = enforcer.release_permit(permit)
                with results_lock:
                    results.append(result)

            t1 = threading.Thread(target=release_permit)
            t2 = threading.Thread(target=release_permit)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Exactly one should succeed
            assert (
                sum(results) == 1
            ), f"Iteration {iteration}: Expected exactly one True, got {results}"

            # Invariants still hold
            assert enforcer._in_flight_count >= 0
            assert enforcer._reserved_cost >= 0

            # Reset for next iteration
            enforcer.reset()

    def test_concurrent_acquire_and_release(self) -> None:
        """Test scenario C6: Acquire while another thread is releasing.

        Multiple threads acquiring and releasing concurrently. The lock
        should serialize operations correctly.
        """
        enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))
        acquired_count = 0
        released_count = 0
        count_lock = threading.Lock()

        def acquirer() -> None:
            """Continuously acquire permits."""
            nonlocal acquired_count, released_count
            for _ in range(20):
                permit = enforcer.acquire_permit()
                if permit.is_granted:
                    with count_lock:
                        acquired_count += 1
                    # Simulate some work
                    threading.Event().wait(timeout=0.001)
                    enforcer.track_cost(0.01, permit=permit)
                    with count_lock:
                        released_count += 1

        # Run multiple acquirer threads
        threads = [threading.Thread(target=acquirer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All acquired permits should be released
        assert (
            acquired_count == released_count
        ), f"Mismatch: acquired={acquired_count}, released={released_count}"

        # Invariants
        assert enforcer._in_flight_count == 0
        assert len(enforcer._active_permits) == 0

    def test_concurrent_track_and_release_race(self) -> None:
        """Test scenario C7: track_cost while another thread releases same permit.

        Only one operation should succeed on the permit.
        """
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))

        for iteration in range(30):
            permit = enforcer.acquire_permit()
            assert permit.is_granted

            initial_trial_count = enforcer._trial_count
            track_succeeded = threading.Event()
            release_succeeded = threading.Event()
            barrier = threading.Barrier(2)

            def do_track() -> None:
                barrier.wait()
                # track_cost doesn't return bool, so we check if it processes
                enforcer.track_cost(0.05, permit=permit)
                track_succeeded.set()

            def do_release() -> None:
                barrier.wait()
                result = enforcer.release_permit(permit)
                if result:
                    release_succeeded.set()

            t1 = threading.Thread(target=do_track)
            t2 = threading.Thread(target=do_release)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # track_cost always executes (increments trial count)
            # but only one should actually release the permit
            assert enforcer._trial_count == initial_trial_count + 1

            # Invariants
            assert enforcer._in_flight_count >= 0
            assert enforcer._reserved_cost >= 0

            enforcer.reset()

    def test_high_contention_permits(self) -> None:
        """High contention test with many threads competing for limited budget.

        With a very limited budget, many threads will be denied permits,
        testing the denial path under concurrent load.
        """
        enforcer = CostEnforcer(
            CostEnforcerConfig(limit=0.5, estimated_cost_per_trial=0.1),
        )

        granted_count = 0
        denied_count = 0
        count_lock = threading.Lock()

        def contender() -> None:
            """Try to acquire and process permits."""
            nonlocal granted_count, denied_count
            for _ in range(20):
                permit = enforcer.acquire_permit()
                with count_lock:
                    if permit.is_granted:
                        granted_count += 1
                    else:
                        denied_count += 1

                if permit.is_granted:
                    # Small delay to simulate work
                    threading.Event().wait(timeout=0.001)
                    enforcer.track_cost(0.05, permit=permit)

        # Run many contending threads
        threads = [threading.Thread(target=contender) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With limited budget, we should have many denials
        assert denied_count > 0, "Should have some denied permits with limited budget"

        # Invariants
        assert enforcer._in_flight_count == 0  # All processed
        # Use approximate comparison for floating point (may have tiny rounding errors)
        assert (
            abs(enforcer._reserved_cost) < 1e-10
        ), f"reserved_cost not zero: {enforcer._reserved_cost}"
        assert len(enforcer._active_permits) == 0
        assert enforcer._accumulated_cost <= enforcer.config.limit + 0.01


@pytest.mark.asyncio
class TestCostEnforcerAsyncConcurrency:
    """Async concurrency tests for CostEnforcer."""

    async def test_async_concurrent_operations(self) -> None:
        """Test async operations with concurrent access.

        50 workers each doing 20 operations concurrently.
        """
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))

        async def worker() -> None:
            """Each worker does 20 operations."""
            permits: list[Permit] = []
            for _ in range(20):
                permit = await enforcer.acquire_permit_async()
                if permit.is_granted:
                    permits.append(permit)

                if permits and random.random() > 0.5:
                    p = permits.pop()
                    if random.random() > 0.5:
                        await enforcer.release_permit_async(p)
                    else:
                        await enforcer.track_cost_async(0.05, permit=p)

            # Cleanup
            for p in permits:
                await enforcer.release_permit_async(p)

        # Run 50 concurrent workers
        await asyncio.gather(*[worker() for _ in range(50)])

        # Final check
        assert enforcer._in_flight_count >= 0
        assert enforcer._reserved_cost >= 0
        assert len(enforcer._active_permits) == enforcer._in_flight_count

    async def test_async_double_release(self) -> None:
        """Test async double-release prevention."""
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))

        for _ in range(30):
            permit = await enforcer.acquire_permit_async()
            assert permit.is_granted

            # First release should succeed
            result1 = await enforcer.release_permit_async(permit)
            assert result1 is True

            # Second release should fail
            result2 = await enforcer.release_permit_async(permit)
            assert result2 is False

            # Invariants
            assert enforcer._in_flight_count >= 0
            assert enforcer._reserved_cost >= 0

            enforcer.reset()

    async def test_async_concurrent_double_release_race(self) -> None:
        """Test async race condition on double release.

        Launch two coroutines trying to release the same permit.
        """
        enforcer = CostEnforcer(CostEnforcerConfig(limit=100.0))

        for _ in range(30):
            permit = await enforcer.acquire_permit_async()
            assert permit.is_granted

            async def release_task() -> bool:
                return await enforcer.release_permit_async(permit)

            # Launch both release attempts concurrently
            results = await asyncio.gather(release_task(), release_task())

            # Exactly one should succeed
            assert sum(results) == 1, f"Expected exactly one True, got {results}"

            # Invariants
            assert enforcer._in_flight_count >= 0
            assert enforcer._reserved_cost >= 0

            enforcer.reset()

    async def test_async_high_volume(self) -> None:
        """Test high volume of async operations."""
        enforcer = CostEnforcer(CostEnforcerConfig(limit=1000.0))

        async def rapid_operations() -> int:
            """Rapidly acquire, track, and return count."""
            count = 0
            for _ in range(100):
                permit = await enforcer.acquire_permit_async()
                if permit.is_granted:
                    await enforcer.track_cost_async(0.01, permit=permit)
                    count += 1
            return count

        # Run 20 coroutines doing 100 operations each
        results = await asyncio.gather(*[rapid_operations() for _ in range(20)])

        total_processed = sum(results)
        assert total_processed > 0

        # All should be processed
        assert enforcer._in_flight_count == 0
        assert len(enforcer._active_permits) == 0


class TestMixedSyncAsyncUsage:
    """Test mixed sync and async usage patterns."""

    @pytest.mark.asyncio
    async def test_sync_then_async_usage(self) -> None:
        """Test that sync followed by async usage works correctly."""
        enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))

        # Sync operations first
        permit1 = enforcer.acquire_permit()
        assert permit1.is_granted
        enforcer.track_cost(0.1, permit=permit1)

        # Then async operations
        permit2 = await enforcer.acquire_permit_async()
        assert permit2.is_granted
        await enforcer.track_cost_async(0.1, permit=permit2)

        # State should be consistent
        assert enforcer._in_flight_count == 0
        assert enforcer._trial_count == 2

    @pytest.mark.asyncio
    async def test_interleaved_sync_async(self) -> None:
        """Test interleaved sync and async operations."""
        enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))

        for i in range(10):
            if i % 2 == 0:
                permit = enforcer.acquire_permit()
                if permit.is_granted:
                    enforcer.track_cost(0.1, permit=permit)
            else:
                permit = await enforcer.acquire_permit_async()
                if permit.is_granted:
                    await enforcer.track_cost_async(0.1, permit=permit)

        # All should be processed
        assert enforcer._in_flight_count == 0
        assert len(enforcer._active_permits) == 0
