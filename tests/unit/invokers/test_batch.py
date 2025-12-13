"""Unit tests for BatchInvoker.

Tests for batch function invocation with parallelization.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance
# CONC-Quality-Reliability FUNC-INVOKERS REQ-INV-006 REQ-INJ-002
# SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from traigent.invokers.base import InvocationResult
from traigent.invokers.batch import BatchInvoker
from traigent.utils.exceptions import InvocationError


class TestBatchInvoker:
    """Tests for BatchInvoker functionality."""

    @pytest.fixture
    def invoker(self) -> BatchInvoker:
        """Create test instance with default configuration."""
        return BatchInvoker()

    @pytest.fixture
    def custom_invoker(self) -> BatchInvoker:
        """Create test instance with custom configuration."""
        return BatchInvoker(
            timeout=30.0,
            max_retries=2,
            max_workers=8,
            batch_size=20,
            batch_timeout=600.0,
            adaptive_batching=False,
        )

    # Happy path tests

    def test_init_default_values(self) -> None:
        """Test BatchInvoker initialization with default values."""
        invoker = BatchInvoker()

        assert invoker.timeout == 60.0
        assert invoker.max_retries == 0
        assert invoker.max_workers == 4
        assert invoker.batch_size == 10
        assert invoker.batch_timeout == 300.0
        assert invoker.adaptive_batching is True
        assert invoker._optimal_batch_size == 10
        assert invoker._recent_times == []

    def test_init_custom_values(self, custom_invoker: BatchInvoker) -> None:
        """Test BatchInvoker initialization with custom values."""
        assert custom_invoker.timeout == 30.0
        assert custom_invoker.max_retries == 2
        assert custom_invoker.max_workers == 8
        assert custom_invoker.batch_size == 20
        assert custom_invoker.batch_timeout == 600.0
        assert custom_invoker.adaptive_batching is False

    @pytest.mark.asyncio
    async def test_invoke_batch_empty(self, invoker: BatchInvoker) -> None:
        """Test batch invocation with empty batch returns empty list."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch: list[dict[str, Any]] = []

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert results == []

    @pytest.mark.asyncio
    async def test_invoke_batch_single_item(self, invoker: BatchInvoker) -> None:
        """Test batch invocation with single item."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": 5}]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 1
        assert results[0].is_successful
        assert results[0].output == 10
        assert results[0].metadata["batch_index"] == 0

    @pytest.mark.asyncio
    async def test_invoke_batch_multiple_items(self, invoker: BatchInvoker) -> None:
        """Test batch invocation with multiple items."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(5)]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.is_successful
            assert result.output == i * 2
            assert result.metadata["batch_index"] == i

    @pytest.mark.asyncio
    async def test_invoke_batch_async_function(self, invoker: BatchInvoker) -> None:
        """Test batch invocation with async function."""

        async def test_func(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 3

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(3)]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.is_successful
            assert result.output == i * 3
            assert result.metadata["batch_index"] == i

    @pytest.mark.asyncio
    async def test_invoke_batch_with_partial_failures(
        self, invoker: BatchInvoker
    ) -> None:
        """Test batch invocation with some failures."""

        def test_func(value: int) -> int:
            if value == 2:
                raise ValueError("Error for value 2")
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": 1}, {"value": 2}, {"value": 3}]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 3
        assert results[0].is_successful
        assert results[0].output == 2
        assert not results[1].is_successful
        assert "Error for value 2" in results[1].error
        assert results[2].is_successful
        assert results[2].output == 6

    @pytest.mark.asyncio
    async def test_invoke_batch_concurrency(self, invoker: BatchInvoker) -> None:
        """Test that batch processing respects max_workers concurrency."""
        execution_times = []

        async def slow_func(value: int) -> int:
            start = time.time()
            await asyncio.sleep(0.1)
            execution_times.append(time.time() - start)
            return value

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(8)]

        start_time = time.time()
        results = await invoker.invoke_batch(slow_func, config, input_batch)
        total_time = time.time() - start_time

        assert len(results) == 8
        # With 4 workers and 8 items taking 0.1s each, should take ~0.2s total
        # (2 batches of 4), not 0.8s (sequential)
        assert total_time < 0.5
        assert all(r.is_successful for r in results)

    @pytest.mark.asyncio
    async def test_invoke_batch_timeout(self, invoker: BatchInvoker) -> None:
        """Test batch invocation with timeout."""
        invoker.batch_timeout = 0.1

        async def slow_func(value: int) -> int:
            await asyncio.sleep(1.0)
            return value

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(3)]

        results = await invoker.invoke_batch(slow_func, config, input_batch)

        # All items should have timeout errors
        assert len(results) == 3
        for i, result in enumerate(results):
            assert not result.is_successful
            assert "Batch timeout" in result.error
            assert result.metadata["batch_timeout"] is True
            assert result.metadata["batch_index"] == i

    @pytest.mark.asyncio
    async def test_invoke_batch_exception_handling(self, invoker: BatchInvoker) -> None:
        """Test batch invocation handles unexpected exceptions."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(3)]

        # Mock _process_batch_concurrent to raise an exception
        with patch.object(
            invoker, "_process_batch_concurrent", side_effect=RuntimeError("Test error")
        ):
            results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert not result.is_successful
            assert "Batch processing failed" in result.error
            assert "Test error" in result.error
            assert result.metadata["batch_error"] is True
            assert result.metadata["batch_index"] == i

    @pytest.mark.asyncio
    async def test_invoke_single(self, invoker: BatchInvoker) -> None:
        """Test single invocation delegates to parent LocalInvoker."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_data = {"value": 5}

        result = await invoker.invoke(test_func, config, input_data)

        assert isinstance(result, InvocationResult)
        assert result.is_successful
        assert result.output == 10
        assert result.execution_time > 0

    # Adaptive batching tests

    @pytest.mark.asyncio
    async def test_adaptive_batching_enabled(self, invoker: BatchInvoker) -> None:
        """Test that adaptive batching updates optimal batch size."""
        assert invoker.adaptive_batching is True
        initial_size = invoker._optimal_batch_size

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(10)]

        # Run first batch
        await invoker.invoke_batch(test_func, config, input_batch)

        # Should have recorded timing
        assert len(invoker._recent_times) == 1
        # Optimal batch size should still be initial
        assert invoker._optimal_batch_size == initial_size

    @pytest.mark.asyncio
    async def test_adaptive_batching_disabled(
        self, custom_invoker: BatchInvoker
    ) -> None:
        """Test that adaptive batching doesn't update when disabled."""
        assert custom_invoker.adaptive_batching is False
        initial_size = custom_invoker._optimal_batch_size

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(10)]

        await custom_invoker.invoke_batch(test_func, config, input_batch)

        # Should not have recorded timing
        assert len(custom_invoker._recent_times) == 0
        # Optimal batch size should not change
        assert custom_invoker._optimal_batch_size == initial_size

    def test_update_adaptive_batch_size_performance_drop(
        self, invoker: BatchInvoker
    ) -> None:
        """Test adaptive batch size reduction on performance drop."""
        # Simulate declining performance
        invoker._recent_times = [100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0]
        invoker._optimal_batch_size = 10

        invoker._update_adaptive_batch_size(1.0, 50)

        # Should reduce batch size due to declining performance
        assert invoker._optimal_batch_size < 10

    def test_update_adaptive_batch_size_performance_improvement(
        self, invoker: BatchInvoker
    ) -> None:
        """Test adaptive batch size increase on performance improvement."""
        # Simulate improving performance
        invoker._recent_times = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]
        invoker._optimal_batch_size = 10

        invoker._update_adaptive_batch_size(1.0, 120)

        # Should increase batch size due to improving performance
        assert invoker._optimal_batch_size > 10

    def test_update_adaptive_batch_size_minimum_cap(
        self, invoker: BatchInvoker
    ) -> None:
        """Test adaptive batch size has minimum of 1."""
        invoker._recent_times = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0]
        invoker._optimal_batch_size = 2

        # Keep reducing
        for _ in range(10):
            invoker._update_adaptive_batch_size(1.0, 10)

        # Should not go below 1
        assert invoker._optimal_batch_size >= 1

    def test_update_adaptive_batch_size_maximum_cap(
        self, invoker: BatchInvoker
    ) -> None:
        """Test adaptive batch size has maximum of 100."""
        invoker._recent_times = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        invoker._optimal_batch_size = 90

        # Keep increasing
        for _ in range(10):
            invoker._update_adaptive_batch_size(1.0, 200)

        # Should not exceed 100
        assert invoker._optimal_batch_size <= 100

    def test_get_optimal_batch_size_with_adaptive(self, invoker: BatchInvoker) -> None:
        """Test get_optimal_batch_size returns adaptive size when enabled."""
        invoker.adaptive_batching = True
        invoker._optimal_batch_size = 15

        assert invoker.get_optimal_batch_size() == 15

    def test_get_optimal_batch_size_without_adaptive(
        self, custom_invoker: BatchInvoker
    ) -> None:
        """Test get_optimal_batch_size returns configured size when disabled."""
        custom_invoker.adaptive_batching = False
        custom_invoker._optimal_batch_size = 15
        custom_invoker.batch_size = 20

        assert custom_invoker.get_optimal_batch_size() == 20

    # Capability tests

    def test_supports_streaming(self, invoker: BatchInvoker) -> None:
        """Test that BatchInvoker does not support streaming."""
        assert invoker.supports_streaming() is False

    def test_supports_batch(self, invoker: BatchInvoker) -> None:
        """Test that BatchInvoker supports batch processing."""
        assert invoker.supports_batch() is True

    # Validation tests

    def test_validate_max_workers_too_low(self) -> None:
        """Test validation fails for max_workers below minimum."""
        with pytest.raises(InvocationError, match="max_workers must be >= 1"):
            BatchInvoker(max_workers=0)

    def test_validate_max_workers_too_high(self) -> None:
        """Test validation fails for max_workers above maximum."""
        with pytest.raises(
            InvocationError, match="max_workers .* exceeds maximum allowed 256"
        ):
            BatchInvoker(max_workers=300)

    def test_validate_max_workers_not_int(self) -> None:
        """Test validation fails for non-integer max_workers."""
        with pytest.raises(InvocationError, match="max_workers must be an integer"):
            BatchInvoker(max_workers="4")

    def test_validate_batch_size_too_low(self) -> None:
        """Test validation fails for batch_size below minimum."""
        with pytest.raises(InvocationError, match="batch_size must be >= 1"):
            BatchInvoker(batch_size=0)

    def test_validate_batch_size_too_high(self) -> None:
        """Test validation fails for batch_size above maximum."""
        with pytest.raises(
            InvocationError, match="batch_size .* exceeds maximum allowed 10000"
        ):
            BatchInvoker(batch_size=20000)

    def test_validate_batch_size_not_int(self) -> None:
        """Test validation fails for non-integer batch_size."""
        with pytest.raises(InvocationError, match="batch_size must be an integer"):
            BatchInvoker(batch_size=5.5)

    def test_validate_batch_timeout_none(self) -> None:
        """Test validation allows None for batch_timeout."""
        invoker = BatchInvoker(batch_timeout=None)
        assert invoker.batch_timeout is None

    def test_validate_batch_timeout_negative(self) -> None:
        """Test validation fails for negative batch_timeout."""
        with pytest.raises(
            InvocationError, match="batch_timeout must be greater than zero"
        ):
            BatchInvoker(batch_timeout=-1.0)

    def test_validate_batch_timeout_zero(self) -> None:
        """Test validation fails for zero batch_timeout."""
        with pytest.raises(
            InvocationError, match="batch_timeout must be greater than zero"
        ):
            BatchInvoker(batch_timeout=0.0)

    def test_validate_batch_timeout_too_high(self) -> None:
        """Test validation fails for batch_timeout above maximum."""
        max_timeout = BatchInvoker.MAX_TIMEOUT_SECONDS
        with pytest.raises(
            InvocationError,
            match=f"batch_timeout .* exceeds maximum allowed {max_timeout}",
        ):
            BatchInvoker(batch_timeout=max_timeout + 1)

    def test_validate_batch_timeout_not_numeric(self) -> None:
        """Test validation fails for non-numeric batch_timeout."""
        with pytest.raises(InvocationError, match="batch_timeout must be numeric"):
            BatchInvoker(batch_timeout="300")

    # Edge cases

    @pytest.mark.asyncio
    async def test_invoke_batch_no_timeout(self, invoker: BatchInvoker) -> None:
        """Test batch invocation with no timeout (None)."""
        invoker.batch_timeout = None

        async def test_func(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(3)]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 3
        assert all(r.is_successful for r in results)

    @pytest.mark.asyncio
    async def test_process_batch_concurrent_with_task_exception(
        self, invoker: BatchInvoker
    ) -> None:
        """Test _process_batch_concurrent handles task-level exceptions."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(3)]

        # Mock invoke to raise exception for some items
        original_invoke = invoker.invoke

        async def mock_invoke(func, config, input_data):
            if input_data["value"] == 1:
                raise RuntimeError("Task exception")
            return await original_invoke(func, config, input_data)

        with patch.object(invoker, "invoke", side_effect=mock_invoke):
            results = await invoker._process_batch_concurrent(
                test_func, config, input_batch
            )

        assert len(results) == 3
        assert results[0].is_successful
        assert not results[1].is_successful
        assert "Task exception" in results[1].error
        assert results[2].is_successful

    @pytest.mark.asyncio
    async def test_process_batch_concurrent_fills_none_results(
        self, invoker: BatchInvoker
    ) -> None:
        """Test that None results are filled with error results."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(3)]

        # Mock gather to return tasks with exceptions mixed in
        async def mock_gather(*tasks, return_exceptions=True):
            # Return a mix of successful results and exceptions
            return [
                (0, InvocationResult(result=0, is_successful=True)),
                RuntimeError("Gather exception"),
                (2, InvocationResult(result=4, is_successful=True)),
            ]

        with patch("asyncio.gather", side_effect=mock_gather):
            results = await invoker._process_batch_concurrent(
                test_func, config, input_batch
            )

        assert len(results) == 3
        assert results[0].is_successful
        # Middle result should be error due to exception
        assert not results[1].is_successful
        assert "Task failed to complete" in results[1].error
        assert results[2].is_successful

    @pytest.mark.asyncio
    async def test_batch_metadata_includes_batch_index(
        self, invoker: BatchInvoker
    ) -> None:
        """Test that all results include batch_index in metadata."""

        def test_func(value: int) -> int:
            if value == 1:
                raise ValueError("Test error")
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": 0}, {"value": 1}, {"value": 2}]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        for i, result in enumerate(results):
            assert "batch_index" in result.metadata
            assert result.metadata["batch_index"] == i

    @pytest.mark.asyncio
    async def test_adaptive_batching_zero_duration(self, invoker: BatchInvoker) -> None:
        """Test adaptive batching handles zero duration gracefully."""
        # This edge case could occur with very fast operations
        invoker._update_adaptive_batch_size(0.0, 10)

        # Should not crash, timing is recorded
        assert len(invoker._recent_times) == 1

    def test_adaptive_batching_history_limited_to_10(
        self, invoker: BatchInvoker
    ) -> None:
        """Test that adaptive batching only keeps last 10 timing records."""
        # Add 15 timing records
        for _ in range(15):
            invoker._update_adaptive_batch_size(1.0, 10)

        # Should only keep last 10
        assert len(invoker._recent_times) <= 10

    @pytest.mark.asyncio
    async def test_large_batch_processing(self, invoker: BatchInvoker) -> None:
        """Test processing a large batch completes successfully."""

        def test_func(value: int) -> int:
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(100)]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 100
        assert all(r.is_successful for r in results)
        assert all(r.metadata["batch_index"] == i for i, r in enumerate(results))

    @pytest.mark.asyncio
    async def test_batch_statistics_logging(self, invoker: BatchInvoker) -> None:
        """Test that batch completion logs statistics correctly."""

        def test_func(value: int) -> int:
            if value == 1:
                raise ValueError("Error")
            return value * 2

        config: dict[str, Any] = {}
        input_batch = [{"value": i} for i in range(5)]

        with patch("traigent.invokers.batch.logger") as mock_logger:
            results = await invoker.invoke_batch(test_func, config, input_batch)

        # Verify logger was called with batch statistics
        assert len(results) == 5
        assert mock_logger.info.called
        # Check that log includes successful/total counts
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        batch_complete_logged = any(
            "Batch completed" in str(call) for call in log_calls
        )
        assert batch_complete_logged
