"""Integration tests for parallel execution with cost enforcement.

Tests parallel_execution_manager.py integration with CostEnforcer.
Verifies batch permit handling, concurrent operations, and race conditions.

Key scenarios tested:
- Batch permit acquisition for parallel trials
- Permit denial propagation in parallel context
- Exception handling releases permits correctly
- High concurrency fairness and invariant preservation
- Mixed success/failure batches

Reference: REDACTED_HOME/.claude/plans/snazzy-whistling-kettle.md
"""

from __future__ import annotations

import asyncio
import os
import random

import pytest

# Ensure mock mode is disabled for these tests
os.environ["TRAIGENT_MOCK_MODE"] = "false"

from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig, Permit

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-10


@pytest.fixture(autouse=True)
def disable_mock_mode() -> None:
    """Ensure mock mode is disabled for all tests in this module."""
    os.environ["TRAIGENT_MOCK_MODE"] = "false"


class TestParallelBatchPermits:
    """Tests for batch permit acquisition in parallel execution."""

    @pytest.fixture
    def cost_enforcer(self) -> CostEnforcer:
        """Create a cost enforcer with low limit for testing."""
        return CostEnforcer(
            CostEnforcerConfig(
                limit=0.50,
                estimated_cost_per_trial=0.10,
            )
        )

    @pytest.mark.asyncio
    async def test_batch_permits_acquired_sequentially(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify batch of permits can be acquired for parallel trials."""
        permits: list[Permit] = []

        # Acquire 5 permits (should all succeed with 0.50 limit)
        for _ in range(5):
            permit = await cost_enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 5
        assert cost_enforcer._in_flight_count == 5
        assert abs(cost_enforcer._reserved_cost - 0.50) < FLOAT_TOLERANCE

        # Track costs for all permits
        for permit in permits:
            await cost_enforcer.track_cost_async(0.08, permit=permit)

        assert cost_enforcer._in_flight_count == 0
        assert abs(cost_enforcer._reserved_cost) < FLOAT_TOLERANCE
        assert abs(cost_enforcer._accumulated_cost - 0.40) < 0.0001

    @pytest.mark.asyncio
    async def test_batch_permits_partial_denial(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify permits are denied when budget is exhausted mid-batch."""
        permits: list[Permit] = []
        denied_count = 0

        # Try to acquire 8 permits (only 5 should succeed)
        for _ in range(8):
            permit = await cost_enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)
            else:
                denied_count += 1

        assert len(permits) == 5
        assert denied_count == 3

        # Verify denied permits have correct properties
        denied_permit = await cost_enforcer.acquire_permit_async()
        assert denied_permit.id == -1
        assert abs(denied_permit.amount) < FLOAT_TOLERANCE
        assert denied_permit.active is False
        assert denied_permit.is_granted is False

        # Cleanup
        for permit in permits:
            await cost_enforcer.release_permit_async(permit)

    @pytest.mark.asyncio
    async def test_batch_mixed_outcomes(self, cost_enforcer: CostEnforcer) -> None:
        """Test batch with mixed success/failure/exception outcomes."""
        permits: list[Permit] = []

        # Acquire permits
        for _ in range(4):
            permit = await cost_enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 4
        assert cost_enforcer._in_flight_count == 4

        # Simulate mixed outcomes:
        # - Permit 0: Success with cost tracking
        # - Permit 1: Exception path (release without cost)
        # - Permit 2: Success with cost tracking
        # - Permit 3: Exception path (release without cost)

        await cost_enforcer.track_cost_async(0.08, permit=permits[0])
        await cost_enforcer.release_permit_async(permits[1])
        await cost_enforcer.track_cost_async(0.09, permit=permits[2])
        await cost_enforcer.release_permit_async(permits[3])

        # Verify final state
        assert cost_enforcer._in_flight_count == 0
        assert abs(cost_enforcer._reserved_cost) < FLOAT_TOLERANCE
        assert abs(cost_enforcer._accumulated_cost - 0.17) < 0.0001
        assert cost_enforcer._trial_count == 2  # Only track_cost increments this


class TestParallelConcurrency:
    """Tests for concurrent permit operations."""

    @pytest.fixture
    def cost_enforcer(self) -> CostEnforcer:
        """Create a cost enforcer for concurrency tests."""
        return CostEnforcer(
            CostEnforcerConfig(
                limit=1.0,
                estimated_cost_per_trial=0.10,
            )
        )

    @pytest.mark.asyncio
    async def test_concurrent_acquire_respects_limit(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify concurrent acquire calls respect the budget limit."""

        async def try_acquire() -> Permit:
            await asyncio.sleep(random.uniform(0, 0.01))  # Add jitter
            return await cost_enforcer.acquire_permit_async()

        # Launch 20 concurrent acquire attempts (only 10 should succeed)
        tasks = [try_acquire() for _ in range(20)]
        permits = await asyncio.gather(*tasks)

        granted = [p for p in permits if p.is_granted]
        denied = [p for p in permits if not p.is_granted]

        assert len(granted) == 10
        assert len(denied) == 10

        # Verify all denied permits have correct properties
        for p in denied:
            assert p.id == -1
            assert abs(p.amount) < FLOAT_TOLERANCE

        # Cleanup
        for p in granted:
            await cost_enforcer.release_permit_async(p)

        assert cost_enforcer._in_flight_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_track_cost(self, cost_enforcer: CostEnforcer) -> None:
        """Verify concurrent track_cost calls are thread-safe."""
        # Acquire permits first
        permits: list[Permit] = []
        for _ in range(10):
            permit = await cost_enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 10

        async def track_with_jitter(permit: Permit) -> None:
            await asyncio.sleep(random.uniform(0, 0.01))
            await cost_enforcer.track_cost_async(0.05, permit=permit)

        # Track costs concurrently
        await asyncio.gather(*[track_with_jitter(p) for p in permits])

        # Verify final state
        assert cost_enforcer._in_flight_count == 0
        assert abs(cost_enforcer._reserved_cost) < FLOAT_TOLERANCE
        assert abs(cost_enforcer._accumulated_cost - 0.50) < 0.0001
        assert cost_enforcer._trial_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_release_same_permit(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify only one concurrent release succeeds for same permit."""
        for _ in range(50):  # Run multiple iterations
            permit = await cost_enforcer.acquire_permit_async()
            assert permit.is_granted

            async def try_release(permit=permit) -> bool:
                return await cost_enforcer.release_permit_async(permit)

            # Launch two concurrent release attempts
            results = await asyncio.gather(try_release(), try_release())

            # Exactly one should succeed
            assert sum(results) == 1, f"Expected exactly one True, got {results}"

            # Verify invariants
            assert cost_enforcer._in_flight_count >= 0
            assert cost_enforcer._reserved_cost >= 0

            cost_enforcer.reset()

    @pytest.mark.asyncio
    async def test_concurrent_acquire_release_interleave(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Test interleaved acquire/release operations."""
        completed_trials = 0
        lock = asyncio.Lock()

        async def worker(worker_id: int) -> int:
            nonlocal completed_trials
            local_count = 0

            for _ in range(5):
                permit = await cost_enforcer.acquire_permit_async()
                if permit.is_granted:
                    await asyncio.sleep(random.uniform(0, 0.005))
                    await cost_enforcer.track_cost_async(0.05, permit=permit)
                    async with lock:
                        completed_trials += 1
                    local_count += 1

            return local_count

        # Run 8 workers concurrently
        results = await asyncio.gather(*[worker(i) for i in range(8)])

        # Verify invariants after all workers complete
        assert cost_enforcer._in_flight_count == 0
        assert cost_enforcer._reserved_cost >= 0
        assert len(cost_enforcer._active_permits) == 0
        assert completed_trials == sum(results)


class TestParallelExceptionHandling:
    """Tests for exception handling in parallel execution."""

    @pytest.fixture
    def cost_enforcer(self) -> CostEnforcer:
        """Create a cost enforcer for exception tests."""
        return CostEnforcer(
            CostEnforcerConfig(
                limit=1.0,
                estimated_cost_per_trial=0.10,
            )
        )

    @pytest.mark.asyncio
    async def test_exception_in_one_task_releases_permit(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify exception in one task doesn't affect others."""
        permits: list[Permit] = []

        # Acquire 3 permits
        for _ in range(3):
            permit = await cost_enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 3
        assert cost_enforcer._in_flight_count == 3

        async def task_success(permit: Permit) -> str:
            await asyncio.sleep(0.01)
            await cost_enforcer.track_cost_async(0.05, permit=permit)
            return "success"

        async def task_failure(permit: Permit) -> str:
            try:
                await asyncio.sleep(0.01)
                raise RuntimeError("Simulated failure")
            except RuntimeError:
                if permit.active:
                    await cost_enforcer.release_permit_async(permit)
                raise

        # Run with exception handling
        tasks = [
            task_success(permits[0]),
            task_failure(permits[1]),
            task_success(permits[2]),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        assert results[0] == "success"
        assert isinstance(results[1], RuntimeError)
        assert results[2] == "success"

        # All permits should be released
        assert cost_enforcer._in_flight_count == 0
        assert abs(cost_enforcer._reserved_cost) < FLOAT_TOLERANCE
        # Only 2 trials tracked (the exception path used release_permit)
        assert cost_enforcer._trial_count == 2

    @pytest.mark.asyncio
    async def test_all_tasks_fail_releases_all_permits(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """Verify all permits are released when all tasks fail."""
        permits: list[Permit] = []

        for _ in range(5):
            permit = await cost_enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 5

        async def failing_task(permit: Permit) -> str:
            try:
                await asyncio.sleep(random.uniform(0, 0.01))
                raise ValueError("All tasks fail")
            except ValueError:
                if permit.active:
                    await cost_enforcer.release_permit_async(permit)
                raise

        results = await asyncio.gather(
            *[failing_task(p) for p in permits], return_exceptions=True
        )

        # All should be ValueErrors
        for r in results:
            assert isinstance(r, ValueError)

        # All permits should be released
        assert cost_enforcer._in_flight_count == 0
        assert abs(cost_enforcer._reserved_cost) < FLOAT_TOLERANCE
        assert cost_enforcer._trial_count == 0  # No successful trials


class TestParallelBudgetBoundaries:
    """Tests for budget boundary conditions in parallel execution."""

    @pytest.mark.asyncio
    async def test_parallel_exactly_at_limit(self) -> None:
        """Test parallel execution with budget exactly at limit."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=0.50,
                estimated_cost_per_trial=0.10,
            )
        )

        # Acquire all 5 permits (exactly at limit)
        permits: list[Permit] = []
        for _ in range(5):
            permit = await enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 5
        assert abs(enforcer._reserved_cost - 0.50) < FLOAT_TOLERANCE

        # 6th permit should be denied
        denied = await enforcer.acquire_permit_async()
        assert not denied.is_granted

        # Track exact costs
        for permit in permits:
            await enforcer.track_cost_async(0.10, permit=permit)

        assert abs(enforcer._accumulated_cost - 0.50) < FLOAT_TOLERANCE
        assert abs(enforcer._reserved_cost) < FLOAT_TOLERANCE

    @pytest.mark.asyncio
    async def test_parallel_costs_higher_than_reserved(self) -> None:
        """Test when actual parallel costs exceed reserved amounts."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=1.0,
                estimated_cost_per_trial=0.10,  # Low estimate
            )
        )

        # Acquire 5 permits (0.50 reserved)
        permits: list[Permit] = []
        for _ in range(5):
            permit = await enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)

        assert len(permits) == 5
        assert abs(enforcer._reserved_cost - 0.50) < FLOAT_TOLERANCE

        # Track higher costs than reserved
        for permit in permits:
            await enforcer.track_cost_async(0.15, permit=permit)

        # Accumulated should reflect actual costs
        assert abs(enforcer._accumulated_cost - 0.75) < 0.0001
        assert abs(enforcer._reserved_cost) < FLOAT_TOLERANCE

    @pytest.mark.asyncio
    async def test_parallel_ema_updates_affect_future_permits(self) -> None:
        """Test that EMA updates from completed trials affect future permits."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=1.0,
                estimated_cost_per_trial=0.10,
            )
        )

        # First batch: track lower costs to update EMA
        initial_estimate = enforcer._estimated_cost
        assert abs(initial_estimate - 0.10) < FLOAT_TOLERANCE

        for _ in range(3):
            permit = await enforcer.acquire_permit_async()
            if permit.is_granted:
                await enforcer.track_cost_async(0.05, permit=permit)

        # EMA should have decreased
        assert enforcer._estimated_cost < initial_estimate

        # Second batch should reserve less per permit
        permits: list[Permit] = []
        for _ in range(3):
            permit = await enforcer.acquire_permit_async()
            if permit.is_granted:
                permits.append(permit)
                # Each permit should reserve less than original estimate
                assert permit.amount < initial_estimate

        for permit in permits:
            await enforcer.release_permit_async(permit)


class TestParallelHighConcurrency:
    """High concurrency stress tests for parallel execution."""

    @pytest.mark.asyncio
    async def test_high_concurrency_100_workers(self) -> None:
        """Stress test with 100 concurrent workers."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=10.0,
                estimated_cost_per_trial=0.05,
            )
        )

        errors: list[str] = []

        async def worker(worker_id: int) -> int:
            """Each worker attempts 10 operations."""
            completed = 0
            for _ in range(10):
                try:
                    permit = await enforcer.acquire_permit_async()
                    if permit.is_granted:
                        await asyncio.sleep(random.uniform(0, 0.001))
                        await enforcer.track_cost_async(
                            random.uniform(0.01, 0.05), permit=permit
                        )
                        completed += 1

                    # Check invariants
                    if enforcer._in_flight_count < 0:
                        errors.append(f"Worker {worker_id}: negative in_flight")
                    if enforcer._reserved_cost < 0:
                        errors.append(f"Worker {worker_id}: negative reserved")
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")

            return completed

        # Run 100 concurrent workers
        results = await asyncio.gather(*[worker(i) for i in range(100)])

        # Verify no errors
        assert len(errors) == 0, f"Errors: {errors}"

        # Verify final invariants
        assert enforcer._in_flight_count == 0
        assert enforcer._reserved_cost >= 0
        assert abs(enforcer._reserved_cost) < FLOAT_TOLERANCE
        assert len(enforcer._active_permits) == 0

        # Verify reasonable completion
        total_completed = sum(results)
        assert total_completed > 0

    @pytest.mark.asyncio
    async def test_high_contention_limited_budget(self) -> None:
        """Test high contention with very limited budget."""
        enforcer = CostEnforcer(
            CostEnforcerConfig(
                limit=0.30,  # Only 3 permits at 0.10 each
                estimated_cost_per_trial=0.10,
            )
        )

        granted_count = 0
        denied_count = 0
        lock = asyncio.Lock()

        async def contender(contender_id: int) -> None:
            nonlocal granted_count, denied_count

            for _ in range(20):
                permit = await enforcer.acquire_permit_async()
                async with lock:
                    if permit.is_granted:
                        granted_count += 1
                    else:
                        denied_count += 1

                if permit.is_granted:
                    await asyncio.sleep(random.uniform(0, 0.001))
                    await enforcer.track_cost_async(0.05, permit=permit)

        # Run 20 contending workers
        await asyncio.gather(*[contender(i) for i in range(20)])

        # Many should be denied due to limited budget
        assert denied_count > 0
        assert granted_count > 0

        # Final invariants
        assert enforcer._in_flight_count == 0
        assert abs(enforcer._reserved_cost) < FLOAT_TOLERANCE
        assert len(enforcer._active_permits) == 0


class TestParallelInvariantPreservation:
    """Tests verifying invariants are preserved in parallel execution."""

    @pytest.fixture
    def cost_enforcer(self) -> CostEnforcer:
        """Create a cost enforcer for invariant tests."""
        return CostEnforcer(
            CostEnforcerConfig(
                limit=5.0,
                estimated_cost_per_trial=0.10,
            )
        )

    @pytest.mark.asyncio
    async def test_invariant_i1_in_flight_never_negative(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """I1: in_flight_count >= 0 always."""
        violations: list[str] = []

        async def operation(op_id: int) -> None:
            for _ in range(20):
                permit = await cost_enforcer.acquire_permit_async()
                if permit.is_granted:
                    await asyncio.sleep(random.uniform(0, 0.001))
                    if random.random() > 0.5:
                        await cost_enforcer.track_cost_async(0.05, permit=permit)
                    else:
                        await cost_enforcer.release_permit_async(permit)

                if cost_enforcer._in_flight_count < 0:
                    violations.append(f"I1 violated at op {op_id}")

        await asyncio.gather(*[operation(i) for i in range(20)])

        assert len(violations) == 0, f"Violations: {violations}"
        assert cost_enforcer._in_flight_count >= 0

    @pytest.mark.asyncio
    async def test_invariant_i2_reserved_never_negative(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """I2: reserved_cost >= 0 always."""
        violations: list[str] = []

        async def operation(op_id: int) -> None:
            for _ in range(20):
                permit = await cost_enforcer.acquire_permit_async()
                if permit.is_granted:
                    await asyncio.sleep(random.uniform(0, 0.001))
                    await cost_enforcer.track_cost_async(
                        random.uniform(0.01, 0.15), permit=permit
                    )

                if cost_enforcer._reserved_cost < 0:
                    violations.append(
                        f"I2 violated at op {op_id}: {cost_enforcer._reserved_cost}"
                    )

        await asyncio.gather(*[operation(i) for i in range(20)])

        assert len(violations) == 0, f"Violations: {violations}"
        assert cost_enforcer._reserved_cost >= 0

    @pytest.mark.asyncio
    async def test_invariant_i3_active_permits_consistency(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """I3: len(active_permits) == in_flight_count always."""
        violations: list[str] = []

        async def operation(op_id: int) -> None:
            for _ in range(20):
                permit = await cost_enforcer.acquire_permit_async()
                if permit.is_granted:
                    await asyncio.sleep(random.uniform(0, 0.001))
                    await cost_enforcer.track_cost_async(0.05, permit=permit)

                # Check consistency
                active = len(cost_enforcer._active_permits)
                in_flight = cost_enforcer._in_flight_count
                if active != in_flight:
                    violations.append(
                        f"I3 violated at op {op_id}: active={active}, in_flight={in_flight}"
                    )

        await asyncio.gather(*[operation(i) for i in range(20)])

        assert len(violations) == 0, f"Violations: {violations}"
        assert len(cost_enforcer._active_permits) == cost_enforcer._in_flight_count

    @pytest.mark.asyncio
    async def test_invariant_i7_permit_ids_monotonic(
        self, cost_enforcer: CostEnforcer
    ) -> None:
        """I7: permit IDs are monotonically increasing."""
        permit_ids: list[int] = []
        lock = asyncio.Lock()

        async def acquire_and_record() -> None:
            for _ in range(10):
                permit = await cost_enforcer.acquire_permit_async()
                if permit.is_granted:
                    async with lock:
                        permit_ids.append(permit.id)
                    await cost_enforcer.track_cost_async(0.05, permit=permit)

        await asyncio.gather(*[acquire_and_record() for _ in range(5)])

        # All IDs should be unique
        assert len(permit_ids) == len(set(permit_ids)), "Duplicate permit IDs found"

        # When sorted, should match the order of monotonic generation
        sorted_ids = sorted(permit_ids)
        for i in range(1, len(sorted_ids)):
            assert sorted_ids[i] > sorted_ids[i - 1], "Permit IDs not monotonic"
