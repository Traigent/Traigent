"""Advanced batch processing utilities for TraiGent SDK.

This module provides enhanced batch processing capabilities including:
- Memory-efficient streaming
- Progress tracking
- Partial result handling
- Adaptive batch sizing
- Load balancing strategies
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import gc
import math
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, AsyncGenerator, Callable, cast

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from traigent.invokers.base import InvocationResult
from traigent.utils.logging import get_logger
from traigent.utils.retry import RetryConfig, RetryHandler, RetryStrategy

logger = get_logger(__name__)


@dataclass
class BatchStats:
    """Statistics for batch processing performance."""

    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    total_duration: float = 0.0
    avg_execution_time: float = 0.0
    throughput: float = 0.0  # items per second
    memory_usage_mb: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)
    error_rates: list[float] = field(default_factory=list)


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""

    current_batch: int = 0
    total_batches: int = 0
    processed_items: int = 0
    total_items: int = 0
    estimated_time_remaining: float = 0.0
    current_throughput: float = 0.0

    @property
    def progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100


class AdaptiveBatchSizer:
    """Adaptive batch sizing based on performance metrics."""

    def __init__(
        self,
        initial_batch_size: int = 10,
        min_batch_size: int = 1,
        max_batch_size: int = 100,
        target_memory_mb: float = 500.0,
        performance_window: int = 5,
    ) -> None:
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_mb = target_memory_mb
        self.performance_window = performance_window

        self.current_batch_size = initial_batch_size
        self.performance_history: deque[dict[str, Any]] = deque(
            maxlen=performance_window
        )
        self.memory_history: deque[float] = deque(maxlen=performance_window)

    def update_performance(
        self,
        batch_size: int,
        throughput: float,
        memory_usage_mb: float,
        error_rate: float,
    ) -> None:
        """Update performance metrics and adjust batch size."""
        self.performance_history.append(
            {
                "batch_size": batch_size,
                "throughput": throughput,
                "memory_usage_mb": memory_usage_mb,
                "error_rate": error_rate,
                "timestamp": time.time(),
            }
        )

        self.memory_history.append(memory_usage_mb)

        # Adjust batch size based on recent performance
        self._adjust_batch_size()

    def _adjust_batch_size(self) -> None:
        """Adjust batch size based on performance trends."""
        # Memory constraint check (always applies)
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        if avg_memory > self.target_memory_mb:
            # Reduce batch size if memory usage is too high
            new_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
            logger.debug(
                f"Reducing batch size due to memory: {self.current_batch_size} -> {new_size}"
            )
            self.current_batch_size = new_size
            return

        # Performance-based adjustment requires at least 2 history entries
        if len(self.performance_history) < 2:
            return

        recent = list(self.performance_history)[-2:]
        prev_perf, curr_perf = recent[0], recent[1]

        # Performance-based adjustment
        throughput_change = curr_perf["throughput"] - prev_perf["throughput"]
        error_rate_change = curr_perf["error_rate"] - prev_perf["error_rate"]

        # Increase batch size if throughput improved and errors didn't increase significantly
        if throughput_change > 0 and error_rate_change <= 0.05:
            new_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            logger.debug(
                f"Increasing batch size due to good performance: {self.current_batch_size} -> {new_size}"
            )
            self.current_batch_size = new_size

        # Decrease batch size if throughput declined or error rate increased
        elif throughput_change < -0.1 or error_rate_change > 0.1:
            new_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
            logger.debug(
                f"Decreasing batch size due to poor performance: {self.current_batch_size} -> {new_size}"
            )
            self.current_batch_size = new_size

    def get_next_batch_size(self, remaining_items: int) -> int:
        """Get optimal batch size for next batch."""
        return min(self.current_batch_size, remaining_items)


class MemoryEfficientBatchProcessor:
    """Memory-efficient batch processor with streaming capabilities."""

    def __init__(
        self,
        max_memory_mb: float = 1000.0,
        gc_frequency: int = 10,
        enable_streaming: bool = True,
    ) -> None:
        self.max_memory_mb = max_memory_mb
        self.gc_frequency = gc_frequency
        self.enable_streaming = enable_streaming
        self.processed_batches = 0

    async def process_stream(
        self,
        items: list[Any],
        processor_func: Callable[..., Any],
        batch_size: int = 10,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> AsyncGenerator[tuple[list[InvocationResult], BatchProgress], None]:
        """Process items in memory-efficient batches with progress tracking."""
        total_items = len(items)
        total_batches = math.ceil(total_items / batch_size)

        stats = BatchStats(total_items=total_items)
        processed_items = 0
        start_time = time.time()

        for batch_idx in range(total_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min(batch_start_idx + batch_size, total_items)
            batch_items = items[batch_start_idx:batch_end_idx]

            batch_start_time = time.time()

            # Process batch
            try:
                batch_results = await processor_func(batch_items)
                processed_items += len(batch_results)

                # Update statistics
                time.time() - batch_start_time
                stats.processed_items = processed_items
                stats.successful_items += sum(
                    1 for r in batch_results if r.is_successful
                )
                stats.failed_items += sum(
                    1 for r in batch_results if not r.is_successful
                )

                # Calculate progress
                elapsed_time = time.time() - start_time
                current_throughput = (
                    processed_items / elapsed_time if elapsed_time > 0 else 0
                )
                estimated_remaining = (
                    (total_items - processed_items) / current_throughput
                    if current_throughput > 0
                    else 0
                )

                progress = BatchProgress(
                    current_batch=batch_idx + 1,
                    total_batches=total_batches,
                    processed_items=processed_items,
                    total_items=total_items,
                    estimated_time_remaining=estimated_remaining,
                    current_throughput=current_throughput,
                )

                # Trigger progress callback
                if progress_callback:
                    progress_callback(progress)

                # Memory management with monitoring
                if (batch_idx + 1) % self.gc_frequency == 0:
                    # Check memory usage before cleanup
                    current_memory_mb = self._get_memory_usage_mb()

                    # Force garbage collection if memory is high
                    if current_memory_mb > self.max_memory_mb * 0.8:
                        logger.warning(
                            f"High memory usage: {current_memory_mb:.1f}MB, forcing GC"
                        )
                        gc.collect()

                        # Check if memory cleanup was effective
                        after_cleanup_mb = self._get_memory_usage_mb()
                        if after_cleanup_mb > self.max_memory_mb:
                            logger.error(
                                f"Memory limit exceeded: {after_cleanup_mb:.1f}MB > {self.max_memory_mb}MB"
                            )

                    await asyncio.sleep(0)  # Allow other tasks to run

                yield batch_results, progress

            except Exception as e:
                logger.error(f"Batch {batch_idx} processing failed: {e}")
                # Create error results for the batch
                error_results = [
                    InvocationResult(
                        error=f"Batch processing failed: {e}",
                        is_successful=False,
                        metadata={"batch_index": batch_idx, "item_index": i},
                    )
                    for i in range(len(batch_items))
                ]

                processed_items += len(error_results)
                stats.failed_items += len(error_results)

                progress = BatchProgress(
                    current_batch=batch_idx + 1,
                    total_batches=total_batches,
                    processed_items=processed_items,
                    total_items=total_items,
                    current_throughput=0.0,
                )

                yield error_results, progress

        # Final statistics
        stats.total_duration = time.time() - start_time
        stats.throughput = (
            stats.processed_items / stats.total_duration
            if stats.total_duration > 0
            else 0
        )
        stats.memory_usage_mb = self._get_memory_usage_mb()
        logger.info(f"Batch processing completed: {stats}")

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return cast(float, memory_info.rss / (1024 * 1024))  # Convert bytes to MB
        except (psutil.NoSuchProcess, AttributeError):
            # Fallback if process monitoring fails
            return 0.0


class LoadBalancer:
    """Load balancer for distributing work across multiple workers."""

    def __init__(self, worker_count: int = 4) -> None:
        self.worker_count = worker_count
        self.worker_loads: dict[int, float] = defaultdict(float)
        self.worker_performance: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=10)
        )

    def assign_batch(self, batch_size: int) -> int:
        """Assign batch to worker with lowest load."""
        # Find worker with minimum load
        worker_id = min(
            self.worker_loads.keys() or range(self.worker_count),
            key=lambda w: self.worker_loads[w],
        )

        # Estimate processing time based on historical performance
        if self.worker_performance[worker_id]:
            avg_time_per_item = sum(self.worker_performance[worker_id]) / len(
                self.worker_performance[worker_id]
            )
            estimated_time = batch_size * avg_time_per_item
        else:
            estimated_time = batch_size * 0.1  # Default estimate

        # Update worker load
        self.worker_loads[worker_id] += estimated_time

        return worker_id

    def update_worker_performance(
        self, worker_id: int, items_processed: int, duration: float
    ) -> None:
        """Update worker performance metrics."""
        time_per_item = duration / max(1, items_processed)
        self.worker_performance[worker_id].append(time_per_item)

        # Reduce worker load
        self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - duration)

    def get_worker_stats(self) -> dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "worker_loads": dict(self.worker_loads),
            "avg_performance": {
                worker_id: sum(perf) / len(perf) if perf else 0
                for worker_id, perf in self.worker_performance.items()
            },
        }


class BatchCheckpointer:
    """Checkpoint manager for batch processing with recovery capabilities."""

    def __init__(self, checkpoint_dir: str | None = None) -> None:
        self._tempdir: TemporaryDirectory[str] | None = None
        if checkpoint_dir is None:
            # Create a secure temporary directory and ensure restrictive permissions
            self._tempdir = TemporaryDirectory(prefix="traigent_checkpoints_")
            self.checkpoint_dir = self._tempdir.name
            os.chmod(self.checkpoint_dir, 0o700)
        else:
            self.checkpoint_dir = checkpoint_dir
            # Create directory if it doesn't exist
            os.makedirs(self.checkpoint_dir, mode=0o700, exist_ok=True)

        self.checkpoint_frequency = 10  # Save every N batches

    @asynccontextmanager
    async def checkpointed_processing(self, job_id: str, total_items: int):
        """Context manager for checkpointed batch processing."""
        checkpoint_data = {
            "job_id": job_id,
            "total_items": total_items,
            "processed_items": 0,
            "completed_batches": [],
            "failed_batches": [],
            "start_time": time.time(),
        }

        try:
            yield checkpoint_data
        finally:
            # Final checkpoint
            await self._save_checkpoint(job_id, checkpoint_data)
            self.cleanup()

    async def _save_checkpoint(self, job_id: str, data: dict[str, Any]) -> None:
        """Save checkpoint data."""
        # In a real implementation, this would save to persistent storage
        logger.debug(
            f"Checkpoint saved for job {job_id}: {data['processed_items']}/{data['total_items']} items"
        )

    async def load_checkpoint(self, job_id: str) -> dict[str, Any] | None:
        """Load checkpoint data for job recovery."""
        # In a real implementation, this would load from persistent storage
        return None

    def cleanup(self) -> None:
        """Remove temporary checkpoint directory if this manager created it."""
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.cleanup()
        except Exception:
            # Never raise during interpreter shutdown
            pass


class PartialResultsManager:
    """Manager for handling partial results and resumable processing."""

    def __init__(self, buffer_size: int = 1000) -> None:
        self.buffer_size = buffer_size
        self.results_buffer: list[InvocationResult] = []
        self.completed_indices: set[int] = set()

    def add_results(self, results: list[InvocationResult], start_index: int) -> None:
        """Add batch results to the manager."""
        for i, result in enumerate(results):
            actual_index = start_index + i
            if actual_index not in self.completed_indices:
                self.results_buffer.append(result)
                self.completed_indices.add(actual_index)

                # Trigger buffer flush if needed
                if len(self.results_buffer) >= self.buffer_size:
                    self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush results buffer (could save to disk or database)."""
        logger.debug(f"Flushing {len(self.results_buffer)} results from buffer")
        self.results_buffer.clear()

    def get_missing_indices(self, total_items: int) -> list[int]:
        """Get list of indices that haven't been processed yet."""
        all_indices = set(range(total_items))
        return sorted(all_indices - self.completed_indices)

    def is_complete(self, total_items: int) -> bool:
        """Check if all items have been processed."""
        return len(self.completed_indices) >= total_items


async def process_with_retry_and_recovery(
    items: list[Any],
    processor_func: Callable[..., Any],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    checkpoint_interval: int = 10,
) -> list[InvocationResult]:
    """Process items with automatic retry and recovery capabilities using unified retry handler."""

    results_manager = PartialResultsManager()
    adaptive_sizer = AdaptiveBatchSizer()

    total_items = len(items)
    all_results: list[InvocationResult | None] = [None] * total_items

    # Configure retry handler with exponential backoff
    retry_config = RetryConfig(
        max_attempts=max_retries + 1,  # +1 because attempts includes initial try
        initial_delay=retry_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        max_delay=60.0,
        jitter=True,
        exponential_base=1.5,  # Same as original retry_delay *= 1.5
    )
    retry_handler = RetryHandler(retry_config)

    async def process_missing_items():
        """Process missing items with batch recovery logic."""
        missing_indices = results_manager.get_missing_indices(total_items)

        if not missing_indices:
            return True  # No missing items, success

        logger.info(f"Processing {len(missing_indices)} missing items")

        # Process missing items in batches
        batch_size = adaptive_sizer.get_next_batch_size(len(missing_indices))

        for i in range(0, len(missing_indices), batch_size):
            batch_indices = missing_indices[i : i + batch_size]
            batch_items = [items[idx] for idx in batch_indices]

            try:
                batch_start = time.time()
                batch_results = await processor_func(batch_items)
                batch_duration = time.time() - batch_start

                # Update results
                for j, result in enumerate(batch_results):
                    original_index = batch_indices[j]
                    all_results[original_index] = result

                results_manager.add_results(batch_results, batch_indices[0])

                # Update adaptive sizing
                successful_count = sum(1 for r in batch_results if r.is_successful)
                error_rate = 1.0 - (successful_count / len(batch_results))
                throughput = (
                    len(batch_results) / batch_duration if batch_duration > 0 else 0
                )

                adaptive_sizer.update_performance(
                    batch_size=len(batch_results),
                    throughput=throughput,
                    memory_usage_mb=0.0,  # Would measure actual memory
                    error_rate=error_rate,
                )

            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                # Re-raise to trigger retry handler
                raise

        # Check if we achieved completion after this round
        return results_manager.is_complete(total_items)

    # Use retry handler for the entire processing operation
    try:
        result = await retry_handler.execute_async(process_missing_items)

        if result.success and result.value:
            logger.info("Batch processing completed successfully")
        else:
            logger.warning(
                f"Batch processing completed with partial results after {result.attempts} attempts"
            )

    except Exception as e:
        logger.error(f"Batch processing failed after {max_retries} retries: {e}")
        # Continue to fill remaining None results with errors

    # Fill any remaining None results with error results
    for i, result in enumerate(all_results):
        if result is None:
            all_results[i] = InvocationResult(
                error="Failed to process after all retries",
                is_successful=False,
                metadata={"max_retries_exceeded": True, "item_index": i},
            )

    # All None values are now replaced with InvocationResult
    return cast(list[InvocationResult], all_results)
