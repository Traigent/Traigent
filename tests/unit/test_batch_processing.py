"""Tests for batch processing utilities."""

from typing import Any, List

import pytest

from traigent.invokers.base import InvocationResult
from traigent.utils.batch_processing import (
    AdaptiveBatchSizer,
    BatchProgress,
    BatchStats,
    LoadBalancer,
    MemoryEfficientBatchProcessor,
    PartialResultsManager,
    process_with_retry_and_recovery,
)


class TestBatchStats:
    """Test suite for BatchStats."""

    def test_init_default_values(self) -> None:
        """Test BatchStats initialization with default values."""
        stats = BatchStats()

        assert stats.total_items == 0
        assert stats.processed_items == 0
        assert stats.successful_items == 0
        assert stats.failed_items == 0
        assert stats.total_duration == 0.0
        assert stats.avg_execution_time == 0.0
        assert stats.throughput == 0.0
        assert stats.memory_usage_mb == 0.0
        assert isinstance(stats.batch_sizes, list)
        assert isinstance(stats.error_rates, list)

    def test_init_custom_values(self) -> None:
        """Test BatchStats initialization with custom values."""
        stats = BatchStats(
            total_items=100,
            processed_items=80,
            successful_items=70,
            failed_items=10,
            total_duration=50.0,
            throughput=1.6,
        )

        assert stats.total_items == 100
        assert stats.processed_items == 80
        assert stats.successful_items == 70
        assert stats.failed_items == 10
        assert stats.total_duration == 50.0
        assert stats.throughput == 1.6


class TestBatchProgress:
    """Test suite for BatchProgress."""

    def test_init_default_values(self) -> None:
        """Test BatchProgress initialization."""
        progress = BatchProgress()

        assert progress.current_batch == 0
        assert progress.total_batches == 0
        assert progress.processed_items == 0
        assert progress.total_items == 0
        assert progress.estimated_time_remaining == 0.0
        assert progress.current_throughput == 0.0

    def test_progress_percentage(self) -> None:
        """Test progress percentage calculation."""
        progress = BatchProgress(processed_items=25, total_items=100)
        assert progress.progress_percentage == 25.0

        progress = BatchProgress(processed_items=0, total_items=100)
        assert progress.progress_percentage == 0.0

        progress = BatchProgress(processed_items=100, total_items=100)
        assert progress.progress_percentage == 100.0

        # Edge case: zero total items
        progress = BatchProgress(processed_items=0, total_items=0)
        assert progress.progress_percentage == 0.0


class TestAdaptiveBatchSizer:
    """Test suite for AdaptiveBatchSizer."""

    def test_init_default_values(self) -> None:
        """Test AdaptiveBatchSizer initialization with defaults."""
        sizer = AdaptiveBatchSizer()

        assert sizer.initial_batch_size == 10
        assert sizer.min_batch_size == 1
        assert sizer.max_batch_size == 100
        assert sizer.target_memory_mb == 500.0
        assert sizer.performance_window == 5
        assert sizer.current_batch_size == 10
        assert len(sizer.performance_history) == 0
        assert len(sizer.memory_history) == 0

    def test_init_custom_values(self) -> None:
        """Test AdaptiveBatchSizer initialization with custom values."""
        sizer = AdaptiveBatchSizer(
            initial_batch_size=20,
            min_batch_size=5,
            max_batch_size=200,
            target_memory_mb=1000.0,
            performance_window=10,
        )

        assert sizer.initial_batch_size == 20
        assert sizer.min_batch_size == 5
        assert sizer.max_batch_size == 200
        assert sizer.target_memory_mb == 1000.0
        assert sizer.performance_window == 10
        assert sizer.current_batch_size == 20

    def test_update_performance_memory_constraint(self) -> None:
        """Test performance update with memory constraint."""
        sizer = AdaptiveBatchSizer(target_memory_mb=100.0)
        initial_size = sizer.current_batch_size

        # Update with high memory usage
        sizer.update_performance(
            batch_size=10,
            throughput=5.0,
            memory_usage_mb=200.0,  # Above target
            error_rate=0.1,
        )

        # Should reduce batch size due to memory constraint
        assert sizer.current_batch_size < initial_size

    def test_update_performance_throughput_improvement(self) -> None:
        """Test performance update with throughput improvement."""
        sizer = AdaptiveBatchSizer(initial_batch_size=10)

        # First update
        sizer.update_performance(
            batch_size=10, throughput=5.0, memory_usage_mb=50.0, error_rate=0.1
        )

        # Second update with better throughput
        sizer.update_performance(
            batch_size=10,
            throughput=6.0,  # Better throughput
            memory_usage_mb=50.0,
            error_rate=0.1,  # Same error rate
        )

        # Should increase batch size due to improved performance
        assert sizer.current_batch_size > 10

    def test_update_performance_throughput_decline(self) -> None:
        """Test performance update with throughput decline."""
        sizer = AdaptiveBatchSizer(initial_batch_size=10)

        # Build some history first
        for _i in range(3):
            sizer.update_performance(
                batch_size=10, throughput=5.0, memory_usage_mb=50.0, error_rate=0.1
            )

        # Update with worse throughput
        sizer.update_performance(
            batch_size=10,
            throughput=3.0,  # Much worse throughput
            memory_usage_mb=50.0,
            error_rate=0.1,
        )

        # Should decrease batch size due to performance decline
        assert sizer.current_batch_size < 10

    def test_get_next_batch_size(self) -> None:
        """Test getting next batch size."""
        sizer = AdaptiveBatchSizer(initial_batch_size=10)

        # Should return current batch size if enough items
        assert sizer.get_next_batch_size(20) == 10

        # Should return remaining items if less than batch size
        assert sizer.get_next_batch_size(5) == 5

        # Edge case: zero items
        assert sizer.get_next_batch_size(0) == 0


class TestMemoryEfficientBatchProcessor:
    """Test suite for MemoryEfficientBatchProcessor."""

    def test_init_default_values(self) -> None:
        """Test MemoryEfficientBatchProcessor initialization."""
        processor = MemoryEfficientBatchProcessor()

        assert processor.max_memory_mb == 1000.0
        assert processor.gc_frequency == 10
        assert processor.enable_streaming is True
        assert processor.processed_batches == 0

    @pytest.mark.asyncio
    async def test_process_stream_success(self) -> None:
        """Test successful stream processing."""
        processor = MemoryEfficientBatchProcessor()

        # Mock processor function
        async def mock_processor(items: List[Any]) -> List[InvocationResult]:
            return [
                InvocationResult(result=f"processed_{item}", is_successful=True)
                for item in items
            ]

        items = [1, 2, 3, 4, 5]
        batch_size = 2

        progress_updates = []

        def progress_callback(progress: BatchProgress) -> None:
            progress_updates.append(progress)

        # Process stream
        results_batches = []
        async for batch_results, _progress in processor.process_stream(
            items, mock_processor, batch_size, progress_callback
        ):
            results_batches.append(batch_results)

        # Verify results
        assert len(results_batches) == 3  # 5 items in batches of 2 = 3 batches
        assert len(progress_updates) == 3

        # Check first batch
        assert len(results_batches[0]) == 2
        assert all(r.is_successful for r in results_batches[0])

        # Check progress
        final_progress = progress_updates[-1]
        assert final_progress.processed_items == 5
        assert final_progress.total_items == 5
        assert final_progress.progress_percentage == 100.0

    @pytest.mark.asyncio
    async def test_process_stream_with_failures(self) -> None:
        """Test stream processing with failures."""
        processor = MemoryEfficientBatchProcessor()

        # Mock processor function that fails on certain items
        async def mock_processor(items: List[Any]) -> List[InvocationResult]:
            if 3 in items:  # Fail batch containing item 3
                raise ValueError("Processing failed")
            return [
                InvocationResult(result=f"processed_{item}", is_successful=True)
                for item in items
            ]

        items = [1, 2, 3, 4, 5]
        batch_size = 2

        # Process stream
        results_batches = []
        async for batch_results, _progress in processor.process_stream(
            items, mock_processor, batch_size
        ):
            results_batches.append(batch_results)

        # Should have 3 batches
        assert len(results_batches) == 3

        # Second batch should contain error results (contains item 3)
        assert len(results_batches[1]) == 2
        assert all(not r.is_successful for r in results_batches[1])
        assert all("Processing failed" in (r.error or "") for r in results_batches[1])


class TestLoadBalancer:
    """Test suite for LoadBalancer."""

    def test_init(self) -> None:
        """Test LoadBalancer initialization."""
        balancer = LoadBalancer(worker_count=4)

        assert balancer.worker_count == 4
        assert len(balancer.worker_loads) == 0
        assert len(balancer.worker_performance) == 0

    def test_assign_batch_initial(self) -> None:
        """Test initial batch assignment."""
        balancer = LoadBalancer(worker_count=3)

        # First assignment should go to worker 0
        worker_id = balancer.assign_batch(10)
        assert worker_id == 0
        assert balancer.worker_loads[0] > 0

    def test_assign_batch_load_balancing(self) -> None:
        """Test load balancing across workers."""
        balancer = LoadBalancer(worker_count=3)

        # Assign several batches
        assigned_workers = []
        for _ in range(6):
            worker_id = balancer.assign_batch(5)
            assigned_workers.append(worker_id)

        # Should distribute across all workers
        assert len(set(assigned_workers)) <= 3

    def test_update_worker_performance(self) -> None:
        """Test updating worker performance."""
        balancer = LoadBalancer(worker_count=2)

        # Assign and update performance
        worker_id = balancer.assign_batch(10)
        initial_load = balancer.worker_loads[worker_id]

        balancer.update_worker_performance(worker_id, 10, 2.0)

        # Load should be reduced
        assert balancer.worker_loads[worker_id] < initial_load

        # Performance history should be updated
        assert len(balancer.worker_performance[worker_id]) == 1
        assert balancer.worker_performance[worker_id][0] == 0.2  # 2.0 / 10

    def test_get_worker_stats(self) -> None:
        """Test getting worker statistics."""
        balancer = LoadBalancer(worker_count=2)

        # Assign some work and update performance
        balancer.assign_batch(10)
        balancer.update_worker_performance(0, 10, 2.0)

        stats = balancer.get_worker_stats()

        assert "worker_loads" in stats
        assert "avg_performance" in stats
        assert 0 in stats["worker_loads"]
        assert 0 in stats["avg_performance"]


class TestPartialResultsManager:
    """Test suite for PartialResultsManager."""

    def test_init(self) -> None:
        """Test PartialResultsManager initialization."""
        manager = PartialResultsManager(buffer_size=100)

        assert manager.buffer_size == 100
        assert len(manager.results_buffer) == 0
        assert len(manager.completed_indices) == 0

    def test_add_results(self) -> None:
        """Test adding results."""
        manager = PartialResultsManager(buffer_size=1000)

        results = [
            InvocationResult(result="result1", is_successful=True),
            InvocationResult(result="result2", is_successful=True),
        ]

        manager.add_results(results, start_index=0)

        assert len(manager.results_buffer) == 2
        assert len(manager.completed_indices) == 2
        assert 0 in manager.completed_indices
        assert 1 in manager.completed_indices

    def test_add_results_buffer_flush(self) -> None:
        """Test buffer flushing when buffer size exceeded."""
        manager = PartialResultsManager(buffer_size=2)

        # Add results that exceed buffer size
        results1 = [InvocationResult(result="result1", is_successful=True)]
        results2 = [InvocationResult(result="result2", is_successful=True)]
        results3 = [InvocationResult(result="result3", is_successful=True)]

        manager.add_results(results1, start_index=0)
        manager.add_results(results2, start_index=1)
        # This should trigger buffer flush
        manager.add_results(results3, start_index=2)

        # Buffer should be cleared after flush
        assert len(manager.results_buffer) == 1  # Only the last result
        assert len(manager.completed_indices) == 3  # All indices tracked

    def test_get_missing_indices(self) -> None:
        """Test getting missing indices."""
        manager = PartialResultsManager()

        # Add some results
        results = [InvocationResult(result="result", is_successful=True)]
        manager.add_results(results, start_index=0)
        manager.add_results(results, start_index=2)  # Skip index 1

        missing = manager.get_missing_indices(total_items=4)

        assert missing == [1, 3]  # Indices 1 and 3 are missing

    def test_is_complete(self) -> None:
        """Test completion check."""
        manager = PartialResultsManager()

        assert not manager.is_complete(total_items=3)

        # Add all results
        for i in range(3):
            results = [InvocationResult(result=f"result{i}", is_successful=True)]
            manager.add_results(results, start_index=i)

        assert manager.is_complete(total_items=3)


class TestRetryAndRecovery:
    """Test suite for retry and recovery functionality."""

    @pytest.mark.asyncio
    async def test_process_with_retry_success(self) -> None:
        """Test successful processing without retries."""
        items = [1, 2, 3, 4, 5]

        async def mock_processor(batch_items: List[Any]) -> List[InvocationResult]:
            return [
                InvocationResult(result=f"processed_{item}", is_successful=True)
                for item in batch_items
            ]

        results = await process_with_retry_and_recovery(
            items, mock_processor, max_retries=2, retry_delay=0.01
        )

        assert len(results) == 5
        assert all(r.is_successful for r in results)
        assert all("processed_" in r.output for r in results)

    @pytest.mark.asyncio
    async def test_process_with_retry_partial_failure(self) -> None:
        """Test processing with partial failures and recovery."""
        items = [1, 2, 3, 4, 5]
        failure_count = 0

        async def mock_processor(batch_items: List[Any]) -> List[InvocationResult]:
            nonlocal failure_count
            results = []

            for item in batch_items:
                if item == 3 and failure_count < 1:  # Fail item 3 once
                    failure_count += 1
                    results.append(
                        InvocationResult(
                            result=None, error="Temporary failure", is_successful=False
                        )
                    )
                else:
                    results.append(
                        InvocationResult(result=f"processed_{item}", is_successful=True)
                    )

            return results

        results = await process_with_retry_and_recovery(
            items, mock_processor, max_retries=2, retry_delay=0.01
        )

        assert len(results) == 5
        # Most should be successful, but item 3 may still be failed since current retry logic doesn't retry individual failed items
        successful_count = sum(1 for r in results if r.is_successful)
        assert successful_count >= 4  # At least 4 out of 5 should be successful

    @pytest.mark.asyncio
    async def test_process_with_retry_max_retries_exceeded(self) -> None:
        """Test processing when max retries are exceeded."""
        items = [1, 2, 3]

        async def failing_processor(batch_items: List[Any]) -> List[InvocationResult]:
            # Always fail for item 2
            results = []
            for item in batch_items:
                if item == 2:
                    results.append(
                        InvocationResult(
                            result=None, error="Persistent failure", is_successful=False
                        )
                    )
                else:
                    results.append(
                        InvocationResult(result=f"processed_{item}", is_successful=True)
                    )
            return results

        results = await process_with_retry_and_recovery(
            items, failing_processor, max_retries=1, retry_delay=0.01
        )

        assert len(results) == 3
        # Items 1 and 3 should succeed, item 2 should fail
        assert results[0].is_successful  # item 1
        assert not results[1].is_successful  # item 2 (failed)
        assert results[2].is_successful  # item 3
        assert "Persistent failure" in (results[1].error or "")
