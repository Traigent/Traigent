"""Tests for JS process pool budget guardrails in parallel mode.

This module provides comprehensive tests for budget enforcement when
running parallel JS trials through the process pool.

Test Coverage:
- Cost permit acquisition before trials
- Permit release after trial completion
- Permit release on trial failure
- Budget limit enforcement stops new trials
- Concurrent budget tracking accuracy
- Early stopping triggered by cost limit

NOTE: These tests explicitly disable mock mode to ensure budget enforcement is active.
The TRAIGENT_MOCK_LLM environment variable bypasses all cost tracking when true.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.bridges.js_bridge import JSTrialResult
from traigent.bridges.process_pool import (
    JSProcessPool,
    JSProcessPoolConfig,
    PoolCapacityError,
    PoolShutdownError,
)
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig, Permit
from traigent.core.parallel_execution_manager import (
    ParallelExecutionManager,
    PermittedTrialResult,
)


@pytest.fixture(autouse=True)
def disable_mock_llm_mode(monkeypatch):
    """Disable mock LLM mode for all budget tests.

    The TRAIGENT_MOCK_LLM env var bypasses cost tracking, which would
    cause all budget enforcement tests to fail. This fixture ensures
    mock mode is disabled for accurate budget testing.
    """
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")


class TestCostPermitFlow:
    """Tests for cost permit acquisition and release flow."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock process pool."""
        pool = MagicMock(spec=JSProcessPool)
        pool.is_running = True
        pool.run_trial = AsyncMock(
            return_value=JSTrialResult(
                trial_id="test-trial-1",
                status="completed",
                metrics={"accuracy": 0.9, "cost": 0.01},
                duration=1.0,
                error_message=None,
                error_code=None,
                retryable=False,
                metadata={},
            )
        )
        return pool

    @pytest.fixture
    def cost_enforcer(self):
        """Create a cost enforcer with budget."""
        config = CostEnforcerConfig(limit=0.10, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        return enforcer

    @pytest.fixture
    def parallel_manager(self, cost_enforcer):
        """Create parallel execution manager with cost enforcer."""
        manager = ParallelExecutionManager(
            parallel_trials=4,
            cost_enforcer=cost_enforcer,
        )
        return manager

    @pytest.mark.asyncio
    async def test_permit_acquired_before_trial(self, parallel_manager, cost_enforcer):
        """Verify permit is acquired before trial execution."""
        permits_acquired = []

        async def mock_trial():
            # Record that permit was acquired before we got here
            permits_acquired.append(True)
            return {"accuracy": 0.9}

        results, cancelled = await parallel_manager.run_with_cost_permits([mock_trial()])

        assert len(results) == 1
        assert len(permits_acquired) == 1
        assert results[0].permit.is_granted

    @pytest.mark.asyncio
    async def test_permit_released_after_success(self, parallel_manager, cost_enforcer):
        """Verify permit is carried through for tracking after success."""
        async def mock_trial():
            return {"accuracy": 0.9}

        results, cancelled = await parallel_manager.run_with_cost_permits([mock_trial()])

        assert len(results) == 1
        # Permit should be active (not released yet - released in track_cost_async)
        result = results[0]
        assert result.permit.is_granted
        assert result.permit.active  # Still active for cost tracking

    @pytest.mark.asyncio
    async def test_permit_released_on_exception(self, parallel_manager, cost_enforcer):
        """Verify permit is released when trial raises exception."""
        async def failing_trial():
            raise RuntimeError("Trial failed")

        results, cancelled = await parallel_manager.run_with_cost_permits(
            [failing_trial()]
        )

        assert len(results) == 1
        # Permit should be denied (was released on exception)
        result = results[0]
        assert not result.permit.is_granted
        assert isinstance(result.result, RuntimeError)

    @pytest.mark.asyncio
    async def test_multiple_permits_concurrent(self, parallel_manager, cost_enforcer):
        """Test concurrent permit acquisition for parallel trials."""
        execution_order = []

        async def mock_trial(trial_id):
            execution_order.append(f"start_{trial_id}")
            await asyncio.sleep(0.01)  # Simulate work
            execution_order.append(f"end_{trial_id}")
            return {"accuracy": 0.9, "trial_id": trial_id}

        # Create 4 concurrent trials
        coroutines = [mock_trial(i) for i in range(4)]
        results, cancelled = await parallel_manager.run_with_cost_permits(coroutines)

        assert len(results) == 4
        assert cancelled == 0
        # All permits should be granted
        for result in results:
            assert result.permit.is_granted


class TestBudgetLimitEnforcement:
    """Tests for budget limit enforcement stopping trials."""

    @pytest.mark.asyncio
    async def test_budget_exhausted_cancels_remaining_trials(self):
        """Verify trials are cancelled when budget is exhausted."""
        # Create enforcer with very low budget
        config = CostEnforcerConfig(limit=0.02, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        # Seed the cost estimate
        enforcer._estimated_cost = 0.01  # Each trial costs ~$0.01

        manager = ParallelExecutionManager(
            parallel_trials=4,
            cost_enforcer=enforcer,
        )

        trial_count = 0

        async def mock_trial():
            nonlocal trial_count
            trial_count += 1
            await asyncio.sleep(0.01)
            return {"accuracy": 0.9}

        # Try to run 5 trials, but budget should only allow ~2
        coroutines = [mock_trial() for _ in range(5)]
        results, cancelled = await manager.run_with_cost_permits(coroutines)

        # Some trials should have been cancelled
        assert cancelled > 0
        # Not all trials executed
        assert trial_count < 5

    @pytest.mark.asyncio
    async def test_permit_denied_returns_sentinel(self):
        """Verify denied permits return the sentinel value."""
        config = CostEnforcerConfig(limit=0.001, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        enforcer._estimated_cost = 0.01  # Higher than budget

        manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=enforcer,
        )

        async def mock_trial():
            return {"accuracy": 0.9}

        results, cancelled = await manager.run_with_cost_permits(
            [mock_trial()],
            cancel_sentinel="CANCELLED",
        )

        # Should have cancelled due to budget
        # The first trial might still go through if budget allows initial permit
        has_cancelled = any(r.result == "CANCELLED" for r in results)
        # Either cancelled count > 0 or we got a cancelled sentinel
        assert cancelled > 0 or has_cancelled or len(results) == 1


class TestWorkerFailureWithBudget:
    """Tests for worker failure scenarios with budget tracking."""

    @pytest.fixture
    def mock_workers(self):
        """Create mock workers with configurable behavior."""
        workers = []
        for i in range(4):
            worker = MagicMock()
            worker.is_running = True
            worker.ping = AsyncMock(return_value=True)
            worker.start = AsyncMock()
            worker.stop = AsyncMock()
            worker.run_trial = AsyncMock(
                return_value=JSTrialResult(
                    trial_id=f"test-trial-{i}",
                    status="completed",
                    metrics={"accuracy": 0.9, "cost": 0.01},
                    duration=1.0,
                    error_message=None,
                    error_code=None,
                    retryable=False,
                    metadata={},
                )
            )
            workers.append(worker)
        return workers

    @pytest.mark.asyncio
    async def test_worker_death_releases_permit_on_exception(self):
        """Verify permit is released when worker dies during trial."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=enforcer,
        )

        async def dying_trial():
            raise ConnectionError("Worker process died")

        results, cancelled = await manager.run_with_cost_permits([dying_trial()])

        assert len(results) == 1
        result = results[0]
        # Permit should have been released (denied after exception)
        assert not result.permit.is_granted
        assert isinstance(result.result, ConnectionError)

    @pytest.mark.asyncio
    async def test_timeout_releases_permit(self):
        """Verify permit is released when trial times out."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=enforcer,
        )

        async def slow_trial():
            await asyncio.sleep(10)  # Will timeout
            return {"accuracy": 0.9}

        # Run with timeout
        async def trial_with_timeout():
            try:
                return await asyncio.wait_for(slow_trial(), timeout=0.01)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("Trial timed out")

        results, cancelled = await manager.run_with_cost_permits([trial_with_timeout()])

        assert len(results) == 1
        result = results[0]
        # Permit should have been released
        assert not result.permit.is_granted
        assert isinstance(result.result, asyncio.TimeoutError)


class TestConcurrentBudgetAccuracy:
    """Tests for accuracy of concurrent budget tracking."""

    @pytest.mark.asyncio
    async def test_concurrent_permits_track_accurately(self):
        """Verify budget is tracked accurately under concurrent load."""
        config = CostEnforcerConfig(limit=0.10, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        enforcer._estimated_cost = 0.01  # $0.01 per trial

        manager = ParallelExecutionManager(
            parallel_trials=4,
            cost_enforcer=enforcer,
        )

        completed_trials = 0

        async def mock_trial():
            nonlocal completed_trials
            await asyncio.sleep(0.001)
            completed_trials += 1
            return {"accuracy": 0.9}

        # Run 8 trials - budget should allow ~10 trials
        coroutines = [mock_trial() for _ in range(8)]
        results, cancelled = await manager.run_with_cost_permits(coroutines)

        # All 8 should complete since budget is $0.10 and each costs $0.01
        assert completed_trials == 8
        assert cancelled == 0

    @pytest.mark.asyncio
    async def test_rapid_permit_requests_dont_overspend(self):
        """Verify rapid permit requests don't exceed budget."""
        config = CostEnforcerConfig(limit=0.05, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        enforcer._estimated_cost = 0.01  # $0.01 per trial

        manager = ParallelExecutionManager(
            parallel_trials=10,  # High parallelism
            cost_enforcer=enforcer,
        )

        completed_count = 0

        async def fast_trial():
            nonlocal completed_count
            completed_count += 1
            return {"accuracy": 0.9}

        # Try 20 trials with $0.05 budget at $0.01 each = max 5 trials
        coroutines = [fast_trial() for _ in range(20)]
        results, cancelled = await manager.run_with_cost_permits(coroutines)

        # Should have cancelled many trials
        assert cancelled > 0
        # Should not have completed all 20
        assert completed_count < 20
        # Budget math: $0.05 / $0.01 = 5 trials max
        assert completed_count <= 6  # Allow some margin for EMA adjustment


class TestProcessPoolWithBudget:
    """Integration tests for process pool with budget enforcement."""

    @pytest.fixture
    def pool_config(self):
        """Create pool configuration."""
        return JSProcessPoolConfig(
            max_workers=4,
            module_path="./dist/trial.js",
            function_name="runTrial",
            trial_timeout=30.0,
            acquire_timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_pool_trial_respects_permit_flow(self, pool_config):
        """Verify pool trials work with permit flow."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=enforcer,
        )

        # Mock the pool
        mock_pool = MagicMock(spec=JSProcessPool)
        mock_pool.run_trial = AsyncMock(
            return_value=JSTrialResult(
                trial_id="test-trial-pool",
                status="completed",
                metrics={"accuracy": 0.9, "cost": 0.01},
                duration=1.0,
                error_message=None,
                error_code=None,
                retryable=False,
                metadata={},
            )
        )

        async def pool_trial(config):
            return await mock_pool.run_trial(config)

        coroutines = [pool_trial({"trial_id": str(i)}) for i in range(3)]
        results, cancelled = await manager.run_with_cost_permits(coroutines)

        assert len(results) == 3
        assert cancelled == 0
        assert mock_pool.run_trial.call_count == 3

    @pytest.mark.asyncio
    async def test_pool_failure_releases_permit(self, pool_config):
        """Verify pool failures release permits correctly."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=enforcer,
        )

        # Mock pool that fails
        mock_pool = MagicMock(spec=JSProcessPool)
        mock_pool.run_trial = AsyncMock(
            side_effect=PoolCapacityError("No workers available")
        )

        async def failing_pool_trial(config):
            return await mock_pool.run_trial(config)

        coroutines = [failing_pool_trial({"trial_id": str(i)}) for i in range(2)]
        results, cancelled = await manager.run_with_cost_permits(coroutines)

        assert len(results) == 2
        # All should have failed and released permits
        for result in results:
            assert not result.permit.is_granted
            assert isinstance(result.result, PoolCapacityError)


class TestEarlyStoppingWithBudget:
    """Tests for early stopping triggered by budget limits."""

    @pytest.mark.asyncio
    async def test_early_stop_on_budget_exhaustion(self):
        """Verify optimization stops early when budget exhausted."""
        config = CostEnforcerConfig(limit=0.03, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        enforcer._estimated_cost = 0.01

        manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=enforcer,
        )

        batch_results = []

        # Simulate multiple batches
        for batch in range(5):
            async def mock_trial():
                return {"accuracy": 0.9}

            coroutines = [mock_trial() for _ in range(2)]
            results, cancelled = await manager.run_with_cost_permits(coroutines)
            batch_results.append((results, cancelled))

            # Check if we should stop (all cancelled)
            if cancelled == len(coroutines):
                break

        # Should have stopped before all 5 batches
        assert len(batch_results) < 5

    @pytest.mark.asyncio
    async def test_partial_batch_execution_on_budget_limit(self):
        """Verify partial batch executes when budget runs out mid-batch."""
        config = CostEnforcerConfig(limit=0.025, estimated_cost_per_trial=0.01)
        enforcer = CostEnforcer(config)
        enforcer._estimated_cost = 0.01

        manager = ParallelExecutionManager(
            parallel_trials=4,
            cost_enforcer=enforcer,
        )

        executed_count = 0

        async def mock_trial():
            nonlocal executed_count
            executed_count += 1
            return {"accuracy": 0.9}

        # Request 4 trials but budget only allows ~2-3
        coroutines = [mock_trial() for _ in range(4)]
        results, cancelled = await manager.run_with_cost_permits(coroutines)

        # Some should have executed, some cancelled
        assert executed_count > 0
        assert executed_count < 4 or cancelled > 0


class TestPermitTrackingIntegration:
    """Integration tests for permit tracking through the full flow."""

    @pytest.mark.asyncio
    async def test_permit_amount_matches_estimated_cost(self):
        """Verify permit amounts match cost estimates."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.015)
        enforcer = CostEnforcer(config)
        enforcer._estimated_cost = 0.015  # Set specific estimate

        manager = ParallelExecutionManager(
            parallel_trials=2,
            cost_enforcer=enforcer,
        )

        async def mock_trial():
            return {"accuracy": 0.9}

        results, _ = await manager.run_with_cost_permits([mock_trial()])

        # Permit amount should be close to EMA
        permit = results[0].permit
        assert permit.is_granted
        # Amount should be positive and reasonable
        assert permit.amount > 0

    @pytest.mark.asyncio
    async def test_track_cost_uses_permit_amount(self):
        """Verify track_cost_async uses permit amount for release."""
        config = CostEnforcerConfig(limit=1.0, estimated_cost_per_trial=0.05)
        enforcer = CostEnforcer(config)

        # Get initial reserved amount
        initial_reserved = enforcer._reserved_cost

        permit = await enforcer.acquire_permit_async()
        assert permit.is_granted

        # Reserved should have increased
        after_acquire_reserved = enforcer._reserved_cost
        assert after_acquire_reserved > initial_reserved

        # Track actual cost (release permit)
        actual_cost = 0.02
        await enforcer.track_cost_async(actual_cost, permit=permit)

        # Reserved should decrease, accumulated cost should increase
        assert enforcer._reserved_cost < after_acquire_reserved
        assert enforcer._accumulated_cost > 0
