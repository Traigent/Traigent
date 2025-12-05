"""Batch function invocation strategy with parallelization."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-INVOKERS REQ-INV-006 REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, cast

from traigent.invokers.base import InvocationResult
from traigent.invokers.local import LocalInvoker
from traigent.utils.exceptions import InvocationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class BatchInvoker(LocalInvoker):
    """Batch function invocation with parallel execution.

    Extends LocalInvoker to support efficient batch processing with
    configurable concurrency and adaptive batching strategies.

    Example:
        >>> invoker = BatchInvoker(max_workers=4, batch_size=10)
        >>> results = await invoker.invoke_batch(func, config, input_batch)
    """

    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 0,
        max_workers: int = 4,
        batch_size: int = 10,
        batch_timeout: float = 300.0,
        adaptive_batching: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize batch invoker.

        Args:
            timeout: Timeout for individual invocations (seconds)
            max_retries: Maximum number of retries for failed invocations
            max_workers: Maximum number of concurrent workers
            batch_size: Preferred batch size for processing
            batch_timeout: Timeout for entire batch processing
            adaptive_batching: Whether to adapt batch size based on performance
            **kwargs: Additional configuration passed to LocalInvoker
        """
        super().__init__(timeout, max_retries, **kwargs)
        self.max_workers = self._validate_positive_int(
            "max_workers", max_workers, minimum=1, maximum=256
        )
        self.batch_size = self._validate_positive_int(
            "batch_size", batch_size, minimum=1, maximum=10_000
        )
        self.batch_timeout = self._validate_batch_timeout(batch_timeout)
        self.adaptive_batching = bool(adaptive_batching)

        # Adaptive batching state
        self._recent_times: list[float] = []
        self._optimal_batch_size = self.batch_size

        logger.debug(
            f"BatchInvoker configured: max_workers={max_workers}, "
            f"batch_size={batch_size}, adaptive={adaptive_batching}"
        )

    async def invoke_batch(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_batch: list[dict[str, Any]],
    ) -> list[InvocationResult]:
        """Invoke function on multiple inputs with parallel execution.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_batch: List of input data dictionaries

        Returns:
            List of InvocationResult objects in same order as input
        """
        if not input_batch:
            return []

        batch_start = time.time()
        total_items = len(input_batch)

        logger.info(
            f"Starting batch processing: {total_items} items, {self.max_workers} workers"
        )

        try:
            # Process batch with concurrency control
            results = await self._process_batch_concurrent(func, config, input_batch)

            batch_end = time.time()
            batch_duration = batch_end - batch_start

            # Update adaptive batching
            if self.adaptive_batching:
                self._update_adaptive_batch_size(batch_duration, total_items)

            # Log batch statistics
            successful = sum(1 for r in results if r.is_successful)
            avg_time = sum(
                r.execution_time for r in results if r.execution_time > 0
            ) / max(1, len(results))

            logger.info(
                f"Batch completed: {successful}/{total_items} successful, "
                f"avg_time={avg_time:.3f}s, total_time={batch_duration:.3f}s"
            )

            return results

        except asyncio.TimeoutError:
            logger.error(f"Batch processing timed out after {self.batch_timeout}s")
            # Return timeout results for all items
            return [
                InvocationResult(
                    error=f"Batch timeout after {self.batch_timeout}s",
                    is_successful=False,
                    metadata={"batch_timeout": True, "batch_index": i},
                )
                for i in range(len(input_batch))
            ]

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return error results for all items
            return [
                InvocationResult(
                    error=f"Batch processing failed: {e}",
                    is_successful=False,
                    metadata={"batch_error": True, "batch_index": i},
                )
                for i in range(len(input_batch))
            ]

    async def _process_batch_concurrent(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_batch: list[dict[str, Any]],
    ) -> list[InvocationResult]:
        """Process batch with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_single_with_semaphore(index: int, input_data: dict[str, Any]):
            async with semaphore:
                try:
                    result = await self.invoke(func, config, input_data)
                    # Add batch metadata
                    result.metadata["batch_index"] = index
                    return index, result
                except Exception as e:
                    logger.warning(f"Batch item {index} failed: {e}")
                    return index, InvocationResult(
                        error=str(e),
                        is_successful=False,
                        metadata={
                            "batch_index": index,
                            "batch_exception": type(e).__name__,
                        },
                    )

        # Create tasks for all items
        tasks = [
            process_single_with_semaphore(i, input_data)
            for i, input_data in enumerate(input_batch)
        ]

        # Execute with batch timeout
        if self.batch_timeout:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.batch_timeout,
            )
        else:
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort results by original order
        results: list[InvocationResult | None] = [None] * len(input_batch)
        for task_result in completed_tasks:
            if isinstance(task_result, BaseException):
                # Handle task-level exceptions
                logger.error(f"Task failed with exception: {task_result}")
                continue

            index, result = task_result
            results[index] = result

        # Fill any None results with error results
        for i, result in enumerate(results):
            if result is None:
                results[i] = InvocationResult(
                    error="Task failed to complete",
                    is_successful=False,
                    metadata={"batch_index": i, "task_incomplete": True},
                )

        # All None values are now replaced with InvocationResult
        return cast(list[InvocationResult], results)

    def _update_adaptive_batch_size(
        self, batch_duration: float, batch_size: int
    ) -> None:
        """Update optimal batch size based on recent performance."""
        items_per_second = batch_size / batch_duration if batch_duration > 0 else 0

        # Keep recent performance history
        self._recent_times.append(items_per_second)
        if len(self._recent_times) > 10:
            self._recent_times.pop(0)

        # Simple adaptive strategy: if performance is declining, reduce batch size
        if len(self._recent_times) >= 3:
            recent_avg = sum(self._recent_times[-3:]) / 3
            older_avg = sum(self._recent_times[:-3]) / max(
                1, len(self._recent_times) - 3
            )

            if recent_avg < older_avg * 0.9:  # 10% performance drop
                self._optimal_batch_size = max(1, int(self._optimal_batch_size * 0.8))
                logger.debug(
                    f"Reduced optimal batch size to {self._optimal_batch_size}"
                )
            elif recent_avg > older_avg * 1.1:  # 10% performance improvement
                self._optimal_batch_size = min(100, int(self._optimal_batch_size * 1.2))
                logger.debug(
                    f"Increased optimal batch size to {self._optimal_batch_size}"
                )

    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self._optimal_batch_size if self.adaptive_batching else self.batch_size

    def supports_streaming(self) -> bool:
        """Batch invoker does not support streaming."""
        return False

    def supports_batch(self) -> bool:
        """Batch invoker supports batch processing."""
        return True

    async def invoke(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> InvocationResult:
        """Invoke single function (delegates to parent LocalInvoker)."""
        return await super().invoke(func, config, input_data)

    @staticmethod
    def _validate_positive_int(
        field_name: str, value: int, *, minimum: int, maximum: int
    ) -> int:
        """Validate that a field is an integer within bounds."""
        if not isinstance(value, int):
            raise InvocationError(f"{field_name} must be an integer")
        if value < minimum:
            raise InvocationError(f"{field_name} must be >= {minimum}")
        if value > maximum:
            raise InvocationError(
                f"{field_name} {value} exceeds maximum allowed {maximum}"
            )
        return value

    def _validate_batch_timeout(self, timeout: float | None) -> float | None:
        """Validate the batch timeout setting."""
        if timeout is None:
            return None
        if not isinstance(timeout, (int, float)):
            raise InvocationError("batch_timeout must be numeric or None")
        timeout_value = float(timeout)
        if timeout_value <= 0:
            raise InvocationError("batch_timeout must be greater than zero seconds")
        if timeout_value > self.MAX_TIMEOUT_SECONDS:
            raise InvocationError(
                f"batch_timeout {timeout_value}s exceeds maximum allowed "
                f"{self.MAX_TIMEOUT_SECONDS}s"
            )
        return timeout_value
