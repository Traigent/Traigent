"""Batch processing utilities for optimization workflows."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.evaluators.metrics import MetricsEvaluationResult
from traigent.invokers.base import BaseInvoker
from traigent.utils.batch_processing import AdaptiveBatchSizer
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchOptimizationStats:
    """Statistics for batch optimization runs."""

    total_configurations: int = 0
    processed_configurations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    total_duration: float = 0.0
    avg_batch_size: float = 0.0
    throughput: float = 0.0  # configs per second


class BatchOptimizationHelper:
    """Helper class for batch-optimized evaluation of configurations."""

    def __init__(
        self,
        adaptive_batching: bool = True,
        initial_batch_size: int = 10,
        max_batch_size: int = 50,
        target_memory_mb: float = 500.0,
    ) -> None:
        self.adaptive_batching = adaptive_batching
        self.adaptive_sizer = (
            AdaptiveBatchSizer(
                initial_batch_size=initial_batch_size,
                max_batch_size=max_batch_size,
                target_memory_mb=target_memory_mb,
            )
            if adaptive_batching
            else None
        )
        self.batch_size = initial_batch_size
        self.stats = BatchOptimizationStats()

    async def evaluate_configurations_batch(
        self,
        configurations: list[dict[str, Any]],
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[MetricsEvaluationResult]:
        """Evaluate multiple configurations using batch processing.

        Args:
            configurations: List of configurations to evaluate
            func: Function to evaluate
            dataset: Dataset for evaluation
            invoker: Function invoker
            evaluator: Result evaluator
            progress_callback: Optional callback for progress updates

        Returns:
            List of evaluation results for each configuration
        """
        self._validate_batch_inputs(
            configurations, func, dataset, invoker, evaluator, progress_callback
        )

        if not configurations:
            logger.info("No configurations supplied for batch evaluation; returning")
            return []

        start_time = time.time()
        self.stats.total_configurations = len(configurations)
        self.stats.processed_configurations = 0

        results = []
        batch_sizes = []

        logger.info(
            f"Starting batch evaluation of {len(configurations)} configurations"
        )

        for i, config in enumerate(configurations):
            try:
                # Get current batch size
                current_batch_size = self._get_current_batch_size(len(dataset.examples))
                batch_sizes.append(current_batch_size)

                # Evaluate single configuration with batch processing
                eval_result = await self._evaluate_single_config_batch(
                    config, func, dataset, invoker, evaluator, current_batch_size
                )

                results.append(eval_result)

                # Count success/failure based on actual result
                if eval_result is not None:
                    self.stats.successful_evaluations += 1
                    # Update adaptive sizing if enabled
                    if self.adaptive_batching:
                        self._update_adaptive_sizing(eval_result, current_batch_size)
                else:
                    self.stats.failed_evaluations += 1

                # Progress callback
                self.stats.processed_configurations += 1
                if progress_callback:
                    progress_callback(
                        self.stats.processed_configurations,
                        self.stats.total_configurations,
                    )

                logger.debug(f"Evaluated config {i + 1}/{len(configurations)}")

            except Exception as e:
                logger.error(f"Failed to evaluate configuration {i}: {e}")
                self.stats.failed_evaluations += 1
                # Add empty result to maintain list alignment
                results.append(None)

        # Update final statistics
        self.stats.total_duration = time.time() - start_time
        self.stats.avg_batch_size = (
            sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        )
        self.stats.throughput = (
            self.stats.processed_configurations / self.stats.total_duration
            if self.stats.total_duration > 0
            else 0
        )

        logger.info(f"Batch evaluation completed: {self.stats}")

        return results

    def _validate_batch_inputs(
        self,
        configurations: list[dict[str, Any]] | Any,
        func: Callable[..., Any] | Any,
        dataset: Dataset | Any,
        invoker: BaseInvoker | Any,
        evaluator: BaseEvaluator | Any,
        progress_callback: Callable[[int, int], None] | None,
    ) -> None:
        """Validate inputs for batch evaluation."""
        if not isinstance(configurations, list):
            raise TypeError("configurations must be provided as a list")

        for index, config in enumerate(configurations):
            if not isinstance(config, dict):
                raise TypeError(f"Configuration at index {index} must be a dictionary")

        if not callable(func):
            raise TypeError("func must be callable")

        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an instance of Dataset")

        if not getattr(dataset, "examples", None):
            raise ValueError("dataset must contain at least one evaluation example")

        if not callable(getattr(invoker, "invoke", None)):
            raise TypeError("invoker must expose an async 'invoke' coroutine")

        if not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must expose an async 'evaluate' coroutine")

        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable when provided")

    async def _evaluate_single_config_batch(
        self,
        config: dict[str, Any],
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
        batch_size: int,
    ) -> MetricsEvaluationResult | None:
        """Evaluate a single configuration using batch processing."""
        try:
            # Process dataset in batches
            all_invocation_results = []

            for i in range(0, len(dataset.examples), batch_size):
                batch_examples = dataset.examples[i : i + batch_size]
                batch_inputs = [ex.input_data for ex in batch_examples]

                # Use batch invocation if available, otherwise sequential
                if hasattr(invoker, "invoke_batch"):
                    batch_results = await invoker.invoke_batch(
                        func, config, batch_inputs
                    )
                else:
                    batch_results = []
                    for input_data in batch_inputs:
                        result = await invoker.invoke(func, config, input_data)
                        batch_results.append(result)

                all_invocation_results.extend(batch_results)

            # Evaluate all results
            expected_outputs = [ex.expected_output for ex in dataset.examples]
            evaluation_result = await evaluator.evaluate(
                all_invocation_results, expected_outputs, dataset  # type: ignore[arg-type]
            )

            return evaluation_result

        except Exception as e:
            logger.error(f"Configuration evaluation failed: {e}")
            return None

    def _get_current_batch_size(self, total_items: int) -> int:
        """Get current batch size for processing."""
        if self.adaptive_batching and self.adaptive_sizer:
            return self.adaptive_sizer.get_next_batch_size(total_items)
        else:
            return min(self.batch_size, total_items)

    def _update_adaptive_sizing(
        self, eval_result: MetricsEvaluationResult, batch_size: int
    ) -> None:
        """Update adaptive batch sizing based on evaluation result."""
        if not self.adaptive_sizer:
            return

        # Extract performance metrics
        throughput = (
            eval_result.total_invocations / eval_result.duration
            if eval_result.duration > 0
            else 0
        )
        error_rate = 1.0 - (
            eval_result.successful_invocations / max(1, eval_result.total_invocations)
        )

        # Update adaptive sizer
        self.adaptive_sizer.update_performance(
            batch_size=batch_size,
            throughput=throughput,
            memory_usage_mb=0.0,  # Would measure actual memory in production
            error_rate=error_rate,
        )

    def get_optimization_stats(self) -> BatchOptimizationStats:
        """Get current optimization statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset optimization statistics."""
        self.stats = BatchOptimizationStats()


async def parallel_config_evaluation(
    configurations: list[dict[str, Any]],
    func: Callable[..., Any],
    dataset: Dataset,
    invoker: BaseInvoker,
    evaluator: BaseEvaluator,
    max_parallel: int = 4,
    batch_size: int = 10,
) -> list[MetricsEvaluationResult | None]:
    """Evaluate configurations in parallel batches.

    Args:
        configurations: List of configurations to evaluate
        func: Function to evaluate
        dataset: Dataset for evaluation
        invoker: Function invoker
        evaluator: Result evaluator
        max_parallel: Maximum number of parallel evaluations
        batch_size: Batch size for each evaluation

    Returns:
        List of evaluation results (None for failed evaluations)
    """
    semaphore = asyncio.Semaphore(max_parallel)

    async def evaluate_single_with_semaphore(
        config: dict[str, Any],
    ) -> MetricsEvaluationResult | None:
        async with semaphore:
            helper = BatchOptimizationHelper(adaptive_batching=False)
            helper.batch_size = batch_size

            results = await helper.evaluate_configurations_batch(
                [config], func, dataset, invoker, evaluator
            )

            return results[0] if results else None

    # Create tasks for all configurations
    tasks = [evaluate_single_with_semaphore(config) for config in configurations]

    # Execute all tasks
    logger.info(
        f"Starting parallel evaluation of {len(configurations)} configs with {max_parallel} workers"
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    processed_results: list[MetricsEvaluationResult | None] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.error(f"Parallel evaluation failed: {result}")
            processed_results.append(None)
        else:
            # Type narrowed: result is MetricsEvaluationResult | None
            processed_results.append(result)  # type: ignore[arg-type]

    return processed_results


def create_batch_progress_callback(
    log_interval: int = 10,
) -> Callable[[int, int], None]:
    """Create a progress callback that logs batch evaluation progress.

    Args:
        log_interval: Log progress every N configurations

    Returns:
        Progress callback function
    """

    def progress_callback(processed: int, total: int) -> None:
        if processed % log_interval == 0 or processed == total:
            percentage = (processed / total) * 100 if total > 0 else 0
            logger.info(
                f"Batch evaluation progress: {processed}/{total} ({percentage:.1f}%)"
            )

    return progress_callback
