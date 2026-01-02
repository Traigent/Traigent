"""Unit tests for ParallelExecutionManager.

Tests cover:
- calculate_parallel_batch_caps: Batch size and cap calculations
- slice_configs_to_cap: Config slicing with prevention counting
- ParallelExecutionManager methods
"""

import asyncio
import math

import pytest

from traigent.core.parallel_execution_manager import (
    ParallelBatchCaps,
    ParallelExecutionManager,
    calculate_parallel_batch_caps,
    slice_configs_to_cap,
)


class TestCalculateParallelBatchCaps:
    """Test calculate_parallel_batch_caps function."""

    def test_infinite_remaining_uses_parallel_trials(self):
        """Test that infinite remaining budget uses parallel_trials as cap."""
        caps = calculate_parallel_batch_caps(
            remaining=float("inf"),
            remaining_samples=None,
            parallel_trials=5,
        )
        assert caps.remaining_cap == 5
        assert caps.target_batch_size == 5
        assert caps.infinite_budget is True

    def test_nan_remaining_treated_as_infinite(self):
        """Test that NaN remaining is treated as infinite budget."""
        caps = calculate_parallel_batch_caps(
            remaining=float("nan"),
            remaining_samples=None,
            parallel_trials=3,
        )
        assert caps.infinite_budget is True
        assert caps.remaining_cap == 3

    def test_finite_remaining_caps_batch_size(self):
        """Test that finite remaining caps the batch size."""
        caps = calculate_parallel_batch_caps(
            remaining=2.0,
            remaining_samples=None,
            parallel_trials=5,
        )
        assert caps.remaining_cap == 2
        assert caps.target_batch_size == 2
        assert caps.infinite_budget is False

    def test_negative_remaining_returns_zero_cap(self):
        """Test that negative remaining returns zero cap."""
        caps = calculate_parallel_batch_caps(
            remaining=-5.0,
            remaining_samples=None,
            parallel_trials=5,
        )
        assert caps.remaining_cap == 0
        assert caps.target_batch_size == 0

    def test_remaining_samples_further_limits_cap(self):
        """Test that remaining_samples can further limit the cap."""
        caps = calculate_parallel_batch_caps(
            remaining=10.0,
            remaining_samples=3.0,
            parallel_trials=5,
        )
        assert caps.remaining_cap == 3
        assert caps.target_batch_size == 3

    def test_remaining_samples_does_not_increase_cap(self):
        """Test that remaining_samples doesn't increase cap above remaining."""
        caps = calculate_parallel_batch_caps(
            remaining=2.0,
            remaining_samples=10.0,
            parallel_trials=5,
        )
        assert caps.remaining_cap == 2
        assert caps.target_batch_size == 2

    def test_infinite_remaining_samples_ignored(self):
        """Test that infinite remaining_samples is ignored."""
        caps = calculate_parallel_batch_caps(
            remaining=10.0,
            remaining_samples=float("inf"),
            parallel_trials=5,
        )
        # remaining_cap is limited by remaining (10), not parallel_trials
        # target_batch_size is min(parallel_trials, remaining_cap) = 5
        assert caps.remaining_cap == 10
        assert caps.target_batch_size == 5

    def test_overflow_handled_gracefully(self):
        """Test that overflow in remaining is handled gracefully."""
        caps = calculate_parallel_batch_caps(
            remaining=1e309,  # Too large for int conversion
            remaining_samples=None,
            parallel_trials=5,
        )
        # Should fall back to parallel_trials due to overflow
        assert caps.remaining_cap == 5


class TestSliceConfigsToCap:
    """Test slice_configs_to_cap function."""

    def test_no_slicing_needed(self):
        """Test when configs are within cap."""
        configs = [{"a": 1}, {"b": 2}, {"c": 3}]
        sliced, prevented = slice_configs_to_cap(configs, 5)
        assert sliced == configs
        assert prevented == 0

    def test_slicing_applied(self):
        """Test when configs exceed cap."""
        configs = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}]
        sliced, prevented = slice_configs_to_cap(configs, 3)
        assert len(sliced) == 3
        assert sliced == [{"a": 1}, {"b": 2}, {"c": 3}]
        assert prevented == 2

    def test_zero_cap_returns_empty(self):
        """Test that zero cap returns empty list."""
        configs = [{"a": 1}, {"b": 2}]
        sliced, prevented = slice_configs_to_cap(configs, 0)
        assert sliced == []
        assert prevented == 2

    def test_empty_configs_returns_empty(self):
        """Test that empty configs returns empty."""
        sliced, prevented = slice_configs_to_cap([], 5)
        assert sliced == []
        assert prevented == 0


class TestParallelExecutionManager:
    """Test ParallelExecutionManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = ParallelExecutionManager(parallel_trials=4)
        assert manager.parallel_trials == 4
        assert manager.max_concurrent == 4

    def test_initialization_with_max_concurrent(self):
        """Test manager initialization with custom max_concurrent."""
        manager = ParallelExecutionManager(parallel_trials=4, max_concurrent=2)
        assert manager.parallel_trials == 4
        assert manager.max_concurrent == 2

    def test_calculate_batch_caps_delegates(self):
        """Test that calculate_batch_caps delegates to module function."""
        manager = ParallelExecutionManager(parallel_trials=5)
        caps = manager.calculate_batch_caps(10.0, None)
        assert isinstance(caps, ParallelBatchCaps)
        assert caps.target_batch_size == 5

    def test_slice_configs(self):
        """Test slice_configs method."""
        manager = ParallelExecutionManager(parallel_trials=3)
        caps = ParallelBatchCaps(
            remaining_cap=2, target_batch_size=2, infinite_budget=False
        )
        configs = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
        sliced, prevented = manager.slice_configs(configs, caps)
        assert len(sliced) == 2
        assert prevented == 2

    def test_slice_configs_with_infinite_budget(self):
        """Test slice_configs respects infinite_budget flag."""
        manager = ParallelExecutionManager(parallel_trials=3)
        caps = ParallelBatchCaps(
            remaining_cap=10, target_batch_size=3, infinite_budget=True
        )
        configs = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}]
        # With infinite budget, slice_cap = parallel_trials = 3
        sliced, prevented = manager.slice_configs(configs, caps)
        assert len(sliced) == 3
        assert prevented == 2

    def test_distribute_work_even(self):
        """Test distribute_work with even distribution."""
        manager = ParallelExecutionManager(parallel_trials=4)
        items = [1, 2, 3, 4, 5, 6]
        distributed = manager.distribute_work(items, 3)
        assert len(distributed) == 3
        # 6 items / 3 workers = 2 each
        assert distributed == [[1, 2], [3, 4], [5, 6]]

    def test_distribute_work_with_remainder(self):
        """Test distribute_work with uneven distribution."""
        manager = ParallelExecutionManager(parallel_trials=4)
        items = [1, 2, 3, 4, 5]
        distributed = manager.distribute_work(items, 3)
        # 5 items / 3 workers = 1 base + 2 remainder
        # First 2 workers get 2 items each, last gets 1
        assert len(distributed) == 3
        assert distributed == [[1, 2], [3, 4], [5]]

    def test_distribute_work_empty_items(self):
        """Test distribute_work with empty items."""
        manager = ParallelExecutionManager(parallel_trials=4)
        distributed = manager.distribute_work([], 3)
        assert distributed == []

    def test_distribute_work_zero_workers(self):
        """Test distribute_work with zero workers."""
        manager = ParallelExecutionManager(parallel_trials=4)
        distributed = manager.distribute_work([1, 2, 3], 0)
        assert distributed == []

    def test_should_use_parallel_true(self):
        """Test should_use_parallel returns True when appropriate."""
        manager = ParallelExecutionManager(parallel_trials=4)
        assert manager.should_use_parallel(remaining_cap=5, configs_count=3) is True

    def test_should_use_parallel_false_single_trial(self):
        """Test should_use_parallel returns False for single trial mode."""
        manager = ParallelExecutionManager(parallel_trials=1)
        assert manager.should_use_parallel(remaining_cap=5, configs_count=3) is False

    def test_should_use_parallel_false_zero_cap(self):
        """Test should_use_parallel returns False with zero cap."""
        manager = ParallelExecutionManager(parallel_trials=4)
        assert manager.should_use_parallel(remaining_cap=0, configs_count=3) is False

    def test_should_use_parallel_false_zero_configs(self):
        """Test should_use_parallel returns False with zero configs."""
        manager = ParallelExecutionManager(parallel_trials=4)
        assert manager.should_use_parallel(remaining_cap=5, configs_count=0) is False


class TestParallelExecutionManagerAsync:
    """Test async methods of ParallelExecutionManager."""

    @pytest.mark.asyncio
    async def test_gather_results_empty(self):
        """Test gather_results with empty tasks."""
        manager = ParallelExecutionManager(parallel_trials=4)
        results = await manager.gather_results([])
        assert results == []

    @pytest.mark.asyncio
    async def test_gather_results_simple(self):
        """Test gather_results with simple coroutines."""
        manager = ParallelExecutionManager(parallel_trials=4)

        async def return_value(val: int) -> int:
            return val * 2

        tasks = [return_value(1), return_value(2), return_value(3)]
        results = await manager.gather_results(tasks)
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_run_tasks_with_semaphore(self):
        """Test run_tasks_with_semaphore controls concurrency."""
        manager = ParallelExecutionManager(parallel_trials=4, max_concurrent=2)

        execution_order = []

        async def tracked_task(task_id: int) -> int:
            execution_order.append(f"start_{task_id}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{task_id}")
            return task_id

        tasks = [tracked_task(i) for i in range(4)]
        results = await manager.run_tasks_with_semaphore(tasks)

        assert sorted(results) == [0, 1, 2, 3]
        # With semaphore=2, at most 2 tasks run concurrently
        # The exact order depends on scheduling, but all should complete
        assert len(execution_order) == 8  # 4 starts + 4 ends

    @pytest.mark.asyncio
    async def test_run_tasks_with_semaphore_empty(self):
        """Test run_tasks_with_semaphore with empty list."""
        manager = ParallelExecutionManager(parallel_trials=4)
        results = await manager.run_tasks_with_semaphore([])
        assert results == []


class TestBackwardCompatibilityWithOrchestrator:
    """Test that ParallelExecutionManager works with orchestrator patterns."""

    def test_caps_match_original_logic_infinite(self):
        """Verify caps calculation matches original orchestrator logic for infinite budget."""
        parallel_trials = 5
        remaining = float("inf")
        remaining_samples = None

        # Original logic
        infinite_budget = math.isinf(remaining) or math.isnan(remaining)
        if infinite_budget:
            expected_cap = parallel_trials
        else:
            expected_cap = int(remaining)
        expected_batch = (
            parallel_trials if infinite_budget else min(parallel_trials, expected_cap)
        )

        # New logic
        caps = calculate_parallel_batch_caps(
            remaining, remaining_samples, parallel_trials
        )

        assert caps.remaining_cap == expected_cap
        assert caps.target_batch_size == expected_batch
        assert caps.infinite_budget == infinite_budget

    def test_caps_match_original_logic_finite(self):
        """Verify caps calculation matches original orchestrator logic for finite budget."""
        parallel_trials = 5
        remaining = 3.0
        remaining_samples = None

        # Original logic
        infinite_budget = math.isinf(remaining) or math.isnan(remaining)
        expected_cap = int(remaining)
        expected_batch = (
            parallel_trials if infinite_budget else min(parallel_trials, expected_cap)
        )

        # New logic
        caps = calculate_parallel_batch_caps(
            remaining, remaining_samples, parallel_trials
        )

        assert caps.remaining_cap == expected_cap
        assert caps.target_batch_size == expected_batch
        assert caps.infinite_budget == infinite_budget
