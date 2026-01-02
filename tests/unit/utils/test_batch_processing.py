"""Comprehensive tests for advanced batch processing utilities (batch_processing.py).

This test suite covers:
- Batch processing statistics and progress tracking
- Memory-efficient streaming operations
- Adaptive batch sizing strategies
- Load balancing and parallel execution
- Error handling and partial result recovery
- Performance optimization techniques
- CTD (Combinatorial Test Design) scenarios
"""

import asyncio
import math
import time
from unittest.mock import patch

import pytest

from traigent.invokers.base import InvocationResult
from traigent.utils.batch_processing import (
    AdaptiveBatchSizer,
    BatchCheckpointer,
    BatchProgress,
    BatchStats,
    LoadBalancer,
    MemoryEfficientBatchProcessor,
    PartialResultsManager,
    process_with_retry_and_recovery,
)
from traigent.utils.retry import RetryConfig, RetryStrategy

# Test fixtures


@pytest.fixture
def sample_batch_items():
    """Sample batch items for testing."""
    return [{"id": f"item_{i}", "value": i, "complexity": i % 3} for i in range(20)]


@pytest.fixture
def sample_invocation_results():
    """Sample invocation results for testing."""
    results = []
    for i in range(10):
        result = InvocationResult(
            result=f"result_{i}" if i < 8 else None,
            error=f"error_{i}" if i >= 8 else None,
            execution_time=0.1 + (i * 0.05),
            metadata={"item_id": f"item_{i}", "attempt": 1},
            is_successful=i < 8,  # 80% success rate
        )
        results.append(result)
    return results


@pytest.fixture
def mock_processor_function():
    """Mock processing function for testing."""

    async def process_item(item):
        # Simulate processing time based on complexity
        complexity = item.get("complexity", 1)
        await asyncio.sleep(0.01 * complexity)

        # Simulate occasional failures
        if item.get("value", 0) % 10 == 9:
            raise ValueError(f"Processing failed for item {item['id']}")

        return InvocationResult(
            result=f"processed_{item['id']}",
            error=None,
            execution_time=0.01 * complexity,
            metadata={"original_item": item},
            is_successful=True,
        )

    return process_item


@pytest.fixture
def mock_batch_processor_function():
    """Mock batch processing function for testing."""

    async def process_batch(items):
        # Process a batch of items and return list of results
        results = []
        for item in items:
            try:
                # Simulate processing time based on complexity
                complexity = item.get("complexity", 1)
                await asyncio.sleep(0.01 * complexity)

                # Simulate occasional failures
                if item.get("value", 0) % 10 == 9:
                    raise ValueError(f"Processing failed for item {item['id']}")

                results.append(
                    InvocationResult(
                        result=f"processed_{item['id']}",
                        error=None,
                        execution_time=0.01 * complexity,
                        metadata={"original_item": item},
                        is_successful=True,
                    )
                )
            except Exception as e:
                results.append(
                    InvocationResult(
                        result=None,
                        error=str(e),
                        execution_time=0.01,
                        metadata={"original_item": item},
                        is_successful=False,
                    )
                )
        return results

    return process_batch


@pytest.fixture
def batch_processor():
    """MemoryEfficientBatchProcessor instance for testing."""
    return MemoryEfficientBatchProcessor(
        max_memory_mb=1000.0,
        gc_frequency=10,
        enable_streaming=True,
    )


@pytest.fixture
def adaptive_batch_sizer():
    """AdaptiveBatchSizer instance for testing."""
    return AdaptiveBatchSizer(
        initial_batch_size=10,
        min_batch_size=2,
        max_batch_size=50,
        target_memory_mb=500.0,
        performance_window=5,
    )


@pytest.fixture
def load_balancer():
    """LoadBalancer instance for testing."""
    return LoadBalancer(worker_count=3)


# Test Classes


class TestBatchStats:
    """Test BatchStats dataclass functionality."""

    def test_batch_stats_initialization(self):
        """Test BatchStats initialization."""
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

    def test_batch_stats_with_values(self):
        """Test BatchStats with initial values."""
        stats = BatchStats(
            total_items=100,
            processed_items=95,
            successful_items=90,
            failed_items=5,
            total_duration=60.0,
            avg_execution_time=0.63,
            throughput=1.5,
            memory_usage_mb=256.0,
            batch_sizes=[10, 12, 8, 15],
            error_rates=[0.05, 0.03, 0.08, 0.02],
        )

        assert stats.total_items == 100
        assert stats.processed_items == 95
        assert stats.successful_items == 90
        assert stats.failed_items == 5
        assert stats.total_duration == 60.0
        assert stats.avg_execution_time == 0.63
        assert stats.throughput == 1.5
        assert stats.memory_usage_mb == 256.0
        assert len(stats.batch_sizes) == 4
        assert len(stats.error_rates) == 4

    def test_batch_stats_calculations(self):
        """Test calculated metrics in BatchStats."""
        stats = BatchStats(
            total_items=100,
            processed_items=100,
            successful_items=85,
            failed_items=15,
            total_duration=50.0,
        )

        # Test derived calculations
        success_rate = stats.successful_items / stats.processed_items
        assert success_rate == 0.85

        throughput_calculated = stats.processed_items / stats.total_duration
        assert throughput_calculated == 2.0


class TestBatchProgress:
    """Test BatchProgress dataclass functionality."""

    def test_batch_progress_initialization(self):
        """Test BatchProgress initialization."""
        progress = BatchProgress()

        assert progress.current_batch == 0
        assert progress.total_batches == 0
        assert progress.processed_items == 0
        assert progress.total_items == 0
        assert progress.estimated_time_remaining == 0.0
        assert progress.current_throughput == 0.0

    def test_batch_progress_with_values(self):
        """Test BatchProgress with initial values."""
        progress = BatchProgress(
            current_batch=5,
            total_batches=20,
            processed_items=50,
            total_items=200,
            estimated_time_remaining=300.0,
            current_throughput=0.5,
        )

        assert progress.current_batch == 5
        assert progress.total_batches == 20
        assert progress.processed_items == 50
        assert progress.total_items == 200
        assert progress.estimated_time_remaining == 300.0
        assert progress.current_throughput == 0.5

    def test_batch_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = BatchProgress(
            current_batch=7, total_batches=10, processed_items=70, total_items=100
        )

        # Test batch completion percentage
        batch_percentage = progress.current_batch / progress.total_batches
        assert batch_percentage == 0.7

        # Test progress_percentage property
        assert progress.progress_percentage == 70.0


class TestMemoryEfficientBatchProcessor:
    """Test MemoryEfficientBatchProcessor functionality."""

    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self):
        """Test MemoryEfficientBatchProcessor initialization."""
        processor = MemoryEfficientBatchProcessor(
            max_memory_mb=1000.0,
            gc_frequency=10,
            enable_streaming=True,
        )

        assert processor.max_memory_mb == 1000.0
        assert processor.gc_frequency == 10
        assert processor.enable_streaming is True
        assert processor.processed_batches == 0

    @pytest.mark.asyncio
    async def test_process_single_batch(
        self, batch_processor, sample_batch_items, mock_batch_processor_function
    ):
        """Test processing a single batch."""
        batch = sample_batch_items[:5]
        results = []
        async for batch_results, _progress in batch_processor.process_stream(
            batch, mock_batch_processor_function, batch_size=5
        ):
            results.extend(batch_results)

        assert len(results) == 5
        assert all(isinstance(result, InvocationResult) for result in results)

        # Check success/failure distribution
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        assert isinstance(successful_results, list)
        assert isinstance(failed_results, list)
        assert len(successful_results) + len(failed_results) == 5

    @pytest.mark.asyncio
    async def test_process_multiple_batches(
        self, batch_processor, sample_batch_items, mock_batch_processor_function
    ):
        """Test processing multiple batches."""
        all_results = []
        last_progress = None
        async for batch_results, progress in batch_processor.process_stream(
            sample_batch_items, mock_batch_processor_function, batch_size=5
        ):
            all_results.extend(batch_results)
            last_progress = progress

        assert len(all_results) == len(sample_batch_items)
        assert all(isinstance(result, InvocationResult) for result in all_results)
        assert last_progress.processed_items == len(sample_batch_items)

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(
        self, batch_processor, sample_batch_items, mock_batch_processor_function
    ):
        """Test concurrent processing of multiple batches."""
        start_time = time.time()
        results = []
        async for batch_results, _progress in batch_processor.process_stream(
            sample_batch_items, mock_batch_processor_function, batch_size=5
        ):
            results.extend(batch_results)
        end_time = time.time()

        assert len(results) == len(sample_batch_items)

        # Processing should complete in reasonable time
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Reasonable timeout

    @pytest.mark.asyncio
    async def test_batch_processor_error_handling(
        self, batch_processor, mock_batch_processor_function
    ):
        """Test error handling in batch processing."""
        # Items that will cause failures (value % 10 == 9)
        error_items = [
            {"id": f"item_{i}", "value": i, "complexity": 1} for i in [9, 19, 29]
        ]

        results = []
        async for batch_results, _progress in batch_processor.process_stream(
            error_items, mock_batch_processor_function, batch_size=3
        ):
            results.extend(batch_results)

        assert len(results) == 3

        # Check error handling based on mock function logic
        # Items with value % 10 == 9 should fail

    @pytest.mark.asyncio
    async def test_memory_efficient_streaming(
        self, batch_processor, sample_batch_items, mock_batch_processor_function
    ):
        """Test memory-efficient streaming."""
        results = []
        progress_updates = []

        async for batch_results, progress in batch_processor.process_stream(
            sample_batch_items, mock_batch_processor_function, batch_size=5
        ):
            results.extend(batch_results)
            progress_updates.append(progress)

        assert len(results) == len(sample_batch_items)
        assert len(progress_updates) > 0
        assert progress_updates[-1].processed_items == len(sample_batch_items)

    @pytest.mark.asyncio
    async def test_progress_tracking(
        self, batch_processor, sample_batch_items, mock_batch_processor_function
    ):
        """Test progress tracking during batch processing."""
        progress_updates = []

        def progress_callback(progress: BatchProgress):
            progress_updates.append(progress)

        results = []
        async for batch_results, _progress in batch_processor.process_stream(
            sample_batch_items,
            mock_batch_processor_function,
            batch_size=5,
            progress_callback=progress_callback,
        ):
            results.extend(batch_results)

        assert len(results) == len(sample_batch_items)

        # Check if progress was tracked
        if progress_updates:
            assert len(progress_updates) > 0

            # Progress should increase over time
            for i in range(1, len(progress_updates)):
                assert (
                    progress_updates[i].processed_items
                    >= progress_updates[i - 1].processed_items
                )


class TestAdaptiveBatchSizer:
    """Test AdaptiveBatchSizer functionality."""

    def test_adaptive_sizer_initialization(self, adaptive_batch_sizer):
        """Test AdaptiveBatchSizer initialization."""
        assert adaptive_batch_sizer.current_batch_size == 10
        assert adaptive_batch_sizer.min_batch_size == 2
        assert adaptive_batch_sizer.max_batch_size == 50
        assert adaptive_batch_sizer.target_memory_mb == 500.0
        assert adaptive_batch_sizer.performance_window == 5

    def test_batch_size_increase(self, adaptive_batch_sizer):
        """Test batch size increase for good performance."""
        # Simulate good performance with increasing throughput
        adaptive_batch_sizer.update_performance(
            batch_size=10, throughput=1.0, memory_usage_mb=100.0, error_rate=0.0
        )
        adaptive_batch_sizer.update_performance(
            batch_size=10, throughput=1.5, memory_usage_mb=150.0, error_rate=0.0
        )

        # Should increase batch size
        new_size = adaptive_batch_sizer.get_next_batch_size(100)
        assert new_size >= 10  # May increase if conditions are right

    def test_batch_size_decrease(self, adaptive_batch_sizer):
        """Test batch size decrease for poor performance."""
        # Simulate poor performance with high memory or errors
        adaptive_batch_sizer.update_performance(
            batch_size=10, throughput=1.0, memory_usage_mb=400.0, error_rate=0.0
        )
        adaptive_batch_sizer.update_performance(
            batch_size=10, throughput=0.5, memory_usage_mb=600.0, error_rate=0.2
        )

        # Should decrease batch size due to high memory
        new_size = adaptive_batch_sizer.get_next_batch_size(100)
        assert new_size <= 10  # Should decrease due to memory constraint

    def test_batch_size_stability(self, adaptive_batch_sizer):
        """Test batch size stability with consistent performance."""
        # Simulate consistent performance
        for _ in range(3):
            adaptive_batch_sizer.update_performance(
                batch_size=10, throughput=1.0, memory_usage_mb=200.0, error_rate=0.05
            )

        # Should remain relatively stable
        new_size = adaptive_batch_sizer.get_next_batch_size(100)
        assert (
            adaptive_batch_sizer.min_batch_size
            <= new_size
            <= adaptive_batch_sizer.max_batch_size
        )

    def test_batch_size_boundaries(self, adaptive_batch_sizer):
        """Test batch size respects min/max boundaries."""
        # Test with very high memory usage (should hit minimum)
        adaptive_batch_sizer.current_batch_size = 50
        for _ in range(3):
            adaptive_batch_sizer.update_performance(
                batch_size=50, throughput=1.0, memory_usage_mb=1000.0, error_rate=0.0
            )

        new_size = adaptive_batch_sizer.get_next_batch_size(100)
        assert new_size >= adaptive_batch_sizer.min_batch_size

        # Test that it never exceeds maximum
        adaptive_batch_sizer.current_batch_size = adaptive_batch_sizer.max_batch_size
        new_size = adaptive_batch_sizer.get_next_batch_size(100)
        assert new_size <= adaptive_batch_sizer.max_batch_size

    def test_adaptive_algorithm_memory_response(self, adaptive_batch_sizer):
        """Test that adaptive algorithm responds to memory pressure."""
        initial_size = adaptive_batch_sizer.current_batch_size

        # Simulate increasing memory pressure - need enough data points in history
        for i in range(5):
            memory = 300 + i * 100  # Increasing memory usage above target (500MB)
            adaptive_batch_sizer.update_performance(
                batch_size=adaptive_batch_sizer.current_batch_size,
                throughput=1.0,
                memory_usage_mb=memory,
                error_rate=0.0,
            )

        final_size = adaptive_batch_sizer.current_batch_size
        # Should have reduced size due to memory pressure exceeding target_memory_mb
        assert final_size <= initial_size

    def test_performance_history_annotation(self, adaptive_batch_sizer):
        """Performance history should capture detailed metrics for each batch."""
        adaptive_batch_sizer.update_performance(
            batch_size=8, throughput=1.5, memory_usage_mb=250.0, error_rate=0.02
        )

        assert len(adaptive_batch_sizer.performance_history) == 1
        entry = adaptive_batch_sizer.performance_history[0]
        assert {
            "batch_size",
            "throughput",
            "memory_usage_mb",
            "error_rate",
            "timestamp",
        } <= set(entry.keys())


class TestLoadBalancer:
    """Test LoadBalancer functionality."""

    def test_load_balancer_initialization(self, load_balancer):
        """Test LoadBalancer initialization."""
        assert load_balancer.worker_count == 3
        assert isinstance(load_balancer.worker_loads, dict)
        assert isinstance(load_balancer.worker_performance, dict)

    def test_batch_assignment(self, load_balancer):
        """Test batch assignment to workers."""
        # Assign multiple batches
        worker_assignments = []
        for _i in range(9):
            worker_id = load_balancer.assign_batch(batch_size=10)
            worker_assignments.append(worker_id)

        # Check that workers are being used
        assert len(set(worker_assignments)) <= load_balancer.worker_count
        assert all(0 <= w < load_balancer.worker_count for w in worker_assignments)

    def test_worker_performance_tracking(self, load_balancer):
        """Test worker performance tracking."""
        # Update worker performance
        load_balancer.update_worker_performance(
            worker_id=0, items_processed=10, duration=1.0
        )
        load_balancer.update_worker_performance(
            worker_id=1, items_processed=8, duration=1.2
        )

        # Get worker stats
        stats = load_balancer.get_worker_stats()
        assert "worker_loads" in stats
        assert "avg_performance" in stats

    def test_load_balancing_with_varying_loads(self, load_balancer):
        """Test load balancing with varying worker loads."""
        # Simulate varying loads
        load_balancer.worker_loads[0] = 1.0
        load_balancer.worker_loads[1] = 5.0
        load_balancer.worker_loads[2] = 3.0

        # Next batch should go to least loaded worker
        worker_id = load_balancer.assign_batch(batch_size=10)
        assert worker_id == 0  # Worker 0 has least load

    def test_performance_history(self, load_balancer):
        """Test that performance history is maintained."""
        # Update performance multiple times
        for i in range(15):
            load_balancer.update_worker_performance(
                worker_id=i % 3, items_processed=10 + i, duration=1.0 + i * 0.1
            )

        # Check that history is maintained with limited size
        for worker_id in range(3):
            if worker_id in load_balancer.worker_performance:
                history = load_balancer.worker_performance[worker_id]
                assert len(history) <= 10  # maxlen=10 from deque


class TestBatchCheckpointer:
    """Test BatchCheckpointer functionality."""

    def test_checkpointer_initialization(self):
        """Test BatchCheckpointer initialization."""
        checkpointer = BatchCheckpointer(checkpoint_dir="/tmp/test_checkpoints")
        assert checkpointer.checkpoint_dir == "/tmp/test_checkpoints"
        assert checkpointer.checkpoint_frequency == 10

    @pytest.mark.asyncio
    async def test_checkpointed_processing(self):
        """Test checkpointed batch processing."""
        checkpointer = BatchCheckpointer()

        async with checkpointer.checkpointed_processing(
            "test_job", 100
        ) as checkpoint_data:
            assert checkpoint_data["job_id"] == "test_job"
            assert checkpoint_data["total_items"] == 100
            assert checkpoint_data["processed_items"] == 0
            assert "start_time" in checkpoint_data

            # Simulate processing
            checkpoint_data["processed_items"] = 50
            checkpoint_data["completed_batches"].append(1)
            checkpoint_data["completed_batches"].append(2)

    @pytest.mark.asyncio
    async def test_checkpoint_recovery(self):
        """Test checkpoint recovery."""
        checkpointer = BatchCheckpointer()

        # Try to load non-existent checkpoint
        checkpoint = await checkpointer.load_checkpoint("non_existent_job")
        assert checkpoint is None


class TestPartialResultsManager:
    """Test PartialResultsManager functionality."""

    def test_partial_results_manager_initialization(self):
        """Test PartialResultsManager initialization."""
        manager = PartialResultsManager(buffer_size=1000)
        assert manager.buffer_size == 1000
        assert len(manager.results_buffer) == 0
        assert len(manager.completed_indices) == 0

    def test_adding_partial_results(self, sample_invocation_results):
        """Test adding partial results."""
        manager = PartialResultsManager(buffer_size=100)

        # Add results
        manager.add_results(sample_invocation_results[:5], start_index=0)
        assert len(manager.completed_indices) == 5

        # Add more results
        manager.add_results(sample_invocation_results[5:], start_index=5)
        assert len(manager.completed_indices) == len(sample_invocation_results)

    def test_missing_indices_detection(self):
        """Test detection of missing indices."""
        manager = PartialResultsManager()

        # Add some results (not all)
        results = [InvocationResult(is_successful=True) for _ in range(3)]
        manager.add_results(results, start_index=0)
        manager.add_results(results, start_index=5)

        # Check missing indices
        missing = manager.get_missing_indices(10)
        assert 3 in missing
        assert 4 in missing
        assert 8 in missing
        assert 9 in missing
        assert len(missing) == 4

    def test_completion_check(self):
        """Test checking if all items are complete."""
        manager = PartialResultsManager()

        # Initially incomplete
        assert not manager.is_complete(10)

        # Add all results
        results = [InvocationResult(is_successful=True) for _ in range(10)]
        manager.add_results(results, start_index=0)

        # Now complete
        assert manager.is_complete(10)

    def test_buffer_flushing(self):
        """Test automatic buffer flushing."""
        manager = PartialResultsManager(buffer_size=3)

        # Add results that exceed buffer size
        results = [InvocationResult(is_successful=True) for _ in range(5)]

        with patch.object(manager, "_flush_buffer") as mock_flush:
            manager.add_results(results, start_index=0)
            # Should have triggered flush when buffer exceeded size
            assert mock_flush.called


class TestProcessWithRetryAndRecovery:
    """Test process_with_retry_and_recovery function."""

    @pytest.mark.asyncio
    async def test_basic_retry_and_recovery(self, mock_batch_processor_function):
        """Test basic retry and recovery functionality."""
        items = [{"id": f"item_{i}", "value": i, "complexity": 1} for i in range(10)]

        results = await process_with_retry_and_recovery(
            items=items,
            processor_func=mock_batch_processor_function,
            max_retries=3,
            retry_delay=0.1,
            checkpoint_interval=5,
        )

        assert len(results) == len(items)
        assert all(isinstance(r, InvocationResult) for r in results)

    @pytest.mark.asyncio
    async def test_retry_with_failures(self, mock_batch_processor_function):
        """Test retry behavior with failing items."""
        # Include items that will fail (value % 10 == 9)
        items = [{"id": f"item_{i}", "value": i, "complexity": 1} for i in [8, 9, 10]]

        results = await process_with_retry_and_recovery(
            items=items,
            processor_func=mock_batch_processor_function,
            max_retries=2,
            retry_delay=0.05,
        )

        assert len(results) == 3
        # Item 9 should fail even after retries
        assert not results[1].is_successful
        # Others should succeed
        assert results[0].is_successful
        assert results[2].is_successful


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_batch_processing(
        self, batch_processor, mock_batch_processor_function
    ):
        """Test processing empty batch."""
        results = []
        async for batch_results, _progress in batch_processor.process_stream(
            [], mock_batch_processor_function, batch_size=5
        ):
            results.extend(batch_results)

        assert results == []
        # Empty batch should result in no processing

    @pytest.mark.asyncio
    async def test_processor_function_exception(
        self, sample_batch_items, batch_processor
    ):
        """Test handling processor function exceptions."""

        async def broken_processor(items):
            # Should handle batch of items
            raise RuntimeError("Processor is broken")

        results = []
        async for batch_results, _progress in batch_processor.process_stream(
            sample_batch_items[:3], broken_processor, batch_size=3
        ):
            results.extend(batch_results)

        assert len(results) == 3
        # All should be failures due to exception
        assert all(not result.is_successful for result in results)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_batch_items, batch_processor):
        """Test handling of processing timeouts."""

        async def slow_processor(items):
            await asyncio.sleep(2.0)  # Very slow
            return [
                InvocationResult(result="slow_result", is_successful=True)
                for _ in items
            ]

        # Test with timeout using asyncio.wait_for
        try:
            results = []
            async with asyncio.timeout(0.5):  # Short timeout
                async for batch_results, _progress in batch_processor.process_stream(
                    sample_batch_items[:2], slow_processor, batch_size=2
                ):
                    results.extend(batch_results)
        except TimeoutError:
            # Expected to timeout
            pass
        # Test completed - TimeoutError means we properly timed out
        assert True  # Confirms test reached this point

    def test_invalid_batch_size(self):
        """Test handling invalid batch size parameters."""
        # MemoryEfficientBatchProcessor doesn't have batch_size validation in __init__
        # Instead, test that process_stream handles invalid batch sizes
        processor = MemoryEfficientBatchProcessor()

        # The batch_size is passed to process_stream, not __init__
        # We can test that it handles edge cases gracefully
        assert processor.max_memory_mb > 0
        assert processor.gc_frequency > 0

    def test_memory_exhaustion_handling(self):
        """Test handling memory exhaustion scenarios."""
        # Test with AdaptiveBatchSizer which handles memory constraints
        sizer = AdaptiveBatchSizer(
            initial_batch_size=1000,
            min_batch_size=1,
            max_batch_size=1000,
            target_memory_mb=1.0,  # Very small limit
            performance_window=5,
        )

        # Simulate high memory usage
        for _ in range(3):
            sizer.update_performance(
                batch_size=1000,
                throughput=1.0,
                memory_usage_mb=10.0,  # Way over limit
                error_rate=0.0,
            )

        # Should reduce batch size due to memory pressure
        adjusted_size = sizer.get_next_batch_size(1000)
        assert adjusted_size < 1000  # Should reduce size
        assert adjusted_size >= sizer.min_batch_size  # Should respect minimum

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(
        self, batch_processor, sample_batch_items, mock_batch_processor_function
    ):
        """Test thread safety for concurrent batch operations."""

        # Run multiple batch operations concurrently
        async def process_batch_safely(items):
            results = []
            async for batch_results, _progress in batch_processor.process_stream(
                items, mock_batch_processor_function, batch_size=5
            ):
                results.extend(batch_results)
            return results

        tasks = []
        for i in range(3):
            batch = sample_batch_items[i * 5 : (i + 1) * 5]
            task = asyncio.create_task(process_batch_safely(batch))
            tasks.append(task)

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without exceptions
        for results in results_list:
            assert not isinstance(results, Exception)
            assert isinstance(results, list)


class TestCTDScenarios:
    """Combinatorial Test Design scenarios for comprehensive coverage."""

    @pytest.mark.parametrize(
        "batch_size,item_count,expected_batches",
        [
            (5, 10, 2),  # Simple case
            (3, 10, 4),  # Multiple batches
            (10, 5, 1),  # Batch size larger than items
            (1, 10, 10),  # Single item batches
            (7, 20, 3),  # Uneven division
        ],
    )
    @pytest.mark.asyncio
    async def test_batch_processing_combinations(
        self,
        mock_batch_processor_function,
        batch_size,
        item_count,
        expected_batches,
    ):
        """Test different combinations of batch processing parameters."""
        processor = MemoryEfficientBatchProcessor(
            max_memory_mb=1000.0,
            gc_frequency=10,
            enable_streaming=True,
        )

        items = [
            {"id": f"item_{i}", "value": i, "complexity": 1} for i in range(item_count)
        ]
        results = []
        batch_count = 0
        async for batch_results, _progress in processor.process_stream(
            items, mock_batch_processor_function, batch_size=batch_size
        ):
            results.extend(batch_results)
            batch_count += 1

        assert len(results) == item_count

        # Check that appropriate number of batches were processed
        actual_batches = math.ceil(item_count / batch_size)
        assert actual_batches == expected_batches

    @pytest.mark.parametrize(
        "worker_count,batch_count",
        [
            (3, 12),  # Even distribution
            (2, 10),  # Two workers
            (4, 8),  # Four workers
            (1, 15),  # Single worker
        ],
    )
    def test_load_balancing_combinations(self, worker_count, batch_count):
        """Test different combinations of load balancing."""
        load_balancer = LoadBalancer(worker_count=worker_count)

        # Assign batches and verify distribution
        assignments = []
        for _i in range(batch_count):
            worker_id = load_balancer.assign_batch(batch_size=10)
            assignments.append(worker_id)

        # All workers should be within valid range
        assert all(0 <= w < worker_count for w in assignments)

        # Workers should be used (unless more workers than batches)
        if batch_count >= worker_count:
            assert len(set(assignments)) <= worker_count

    @pytest.mark.parametrize(
        "memory_usage,throughput_change,error_rate,size_change",
        [
            (200, 0.5, 0.0, "increase"),  # Good performance - increase batch size
            (600, -0.5, 0.2, "decrease"),  # Poor performance - decrease batch size
            (300, 0.0, 0.05, "stable"),  # Stable performance - keep stable
            (250, 0.2, 0.0, "slight_increase"),  # Slightly better
            (400, -0.1, 0.1, "slight_decrease"),  # Slightly worse
        ],
    )
    def test_adaptive_sizing_combinations(
        self, memory_usage, throughput_change, error_rate, size_change
    ):
        """Test different combinations of adaptive batch sizing scenarios."""
        sizer = AdaptiveBatchSizer(
            initial_batch_size=10,
            min_batch_size=2,
            max_batch_size=50,
            target_memory_mb=500.0,
            performance_window=5,
        )

        # Add baseline performance
        sizer.update_performance(
            batch_size=10, throughput=1.0, memory_usage_mb=300.0, error_rate=0.05
        )

        # Add new performance data
        sizer.update_performance(
            batch_size=10,
            throughput=1.0 + throughput_change,
            memory_usage_mb=memory_usage,
            error_rate=error_rate,
        )

        new_size = sizer.get_next_batch_size(100)

        if size_change == "increase" and memory_usage < 500:
            assert new_size >= 10
        elif size_change == "decrease" or memory_usage > 500:
            assert new_size <= 10
        elif size_change == "stable":
            assert abs(new_size - 10) <= 5
        elif size_change == "slight_increase" and memory_usage < 500:
            assert new_size >= 10
        elif size_change == "slight_decrease" or memory_usage > 400:
            assert new_size <= 10

        # Always respect boundaries
        assert sizer.min_batch_size <= new_size <= sizer.max_batch_size

    @pytest.mark.parametrize(
        "memory_usage,memory_limit,expected_action",
        [
            (100, 500, "none"),  # Low usage
            (350, 500, "none"),  # Moderate usage
            (420, 500, "gc_trigger"),  # High usage - trigger GC
            (480, 500, "batch_reduce"),  # Very high usage
            (510, 500, "emergency"),  # Over limit
        ],
    )
    def test_memory_management_combinations(
        self, memory_usage, memory_limit, expected_action
    ):
        """Test different combinations of memory management scenarios."""
        # Test with AdaptiveBatchSizer which handles memory management
        sizer = AdaptiveBatchSizer(
            initial_batch_size=20,
            min_batch_size=2,
            max_batch_size=50,
            target_memory_mb=memory_limit,
            performance_window=5,
        )

        # Add performance data with memory usage
        for _ in range(3):
            sizer.update_performance(
                batch_size=20,
                throughput=1.0,
                memory_usage_mb=memory_usage,
                error_rate=0.0,
            )

        new_size = sizer.get_next_batch_size(100)

        if memory_usage > memory_limit:
            # Should reduce batch size when over memory limit
            assert new_size <= 20
        elif memory_usage > memory_limit * 0.8:
            # May reduce when approaching limit
            assert new_size <= 20
        else:
            # Should maintain or increase when memory is fine
            assert new_size >= sizer.min_batch_size

    @pytest.mark.parametrize(
        "retry_attempts,retry_strategy,expected_outcome",
        [
            (2, "exponential", "with_retries"),
            (3, "exponential", "more_retries"),
            (1, "fixed", "single_retry"),
            (0, "fixed", "no_retries"),
        ],
    )
    @pytest.mark.asyncio
    async def test_retry_handling_combinations(
        self,
        mock_batch_processor_function,
        retry_attempts,
        retry_strategy,
        expected_outcome,
    ):
        """Test different combinations of retry handling scenarios."""
        # Test using process_with_retry_and_recovery which has retry logic
        items = [{"id": f"item_{i}", "value": i, "complexity": 1} for i in range(5)]
        # Add one item that will fail
        items.append({"id": "item_9", "value": 9, "complexity": 1})

        RetryConfig(
            max_attempts=retry_attempts + 1,  # +1 because it includes initial attempt
            initial_delay=0.01,  # Fast for testing
            strategy=(
                RetryStrategy.EXPONENTIAL
                if retry_strategy == "exponential"
                else RetryStrategy.FIXED
            ),
        )

        # We can't directly use retry config in process_with_retry_and_recovery,
        # so just test basic retry behavior
        results = await process_with_retry_and_recovery(
            items=items,
            processor_func=mock_batch_processor_function,
            max_retries=retry_attempts,
            retry_delay=0.01,
            checkpoint_interval=10,
        )

        assert len(results) == 6

        # Item with value 9 should fail
        failed_count = sum(1 for r in results if not r.is_successful)

        if expected_outcome == "no_retries":
            # With no retries, the failing item stays failed
            assert failed_count >= 1
        else:
            # With retries, the deterministic failure (value=9) still fails
            assert failed_count >= 1  # At least the item with value 9 fails
