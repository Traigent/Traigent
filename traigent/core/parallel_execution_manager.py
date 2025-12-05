"""Parallel execution management for optimization trials.

This module provides the ParallelExecutionManager class that handles
parallel trial execution with resource control and async-first design.

Extracted from OptimizationOrchestrator to reduce class complexity
and improve testability.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class ParallelBatchCaps:
    """Calculated caps for parallel batch execution."""

    remaining_cap: int
    target_batch_size: int
    infinite_budget: bool


@dataclass
class TrialDescriptor:
    """Descriptor for a single trial in parallel execution."""

    original_config: dict[str, Any]
    eval_config: dict[str, Any]
    dataset: Any  # Dataset
    optuna_id: int | None
    sample_ceiling: int | None = None


def calculate_parallel_batch_caps(
    remaining: float,
    remaining_samples: float | None,
    parallel_trials: int,
) -> ParallelBatchCaps:
    """Calculate caps for parallel batch execution.

    Args:
        remaining: Remaining trial budget (can be inf/nan for unlimited)
        remaining_samples: Remaining sample budget (None for unlimited)
        parallel_trials: Number of parallel trials configured

    Returns:
        ParallelBatchCaps with computed values
    """
    infinite_budget = math.isinf(remaining) or math.isnan(remaining)
    if infinite_budget:
        remaining_cap = parallel_trials
    else:
        try:
            remaining_cap = int(remaining)
        except OverflowError:
            remaining_cap = parallel_trials

    if remaining_cap < 0:
        remaining_cap = 0

    if remaining_samples is not None and not math.isinf(remaining_samples):
        sample_cap = max(int(remaining_samples), 0)
        if sample_cap < remaining_cap:
            remaining_cap = sample_cap

    target_batch_size = (
        parallel_trials if infinite_budget else min(parallel_trials, remaining_cap)
    )

    return ParallelBatchCaps(
        remaining_cap=remaining_cap,
        target_batch_size=target_batch_size,
        infinite_budget=infinite_budget,
    )


def slice_configs_to_cap(
    configs: list[dict[str, Any]],
    slice_cap: int,
) -> tuple[list[dict[str, Any]], int]:
    """Slice configs to respect cap and count prevented trials.

    Args:
        configs: List of configuration dictionaries
        slice_cap: Maximum number of configs to keep

    Returns:
        Tuple of (sliced_configs, trials_prevented_count)
    """
    original_count = len(configs)
    sliced = configs[:slice_cap]
    prevented = max(0, original_count - len(sliced))
    return sliced, prevented


class ParallelExecutionManager:
    """Manages parallel trial execution with resource control.

    This class coordinates parallel trial execution, handling:
    - Batch size calculation and caps
    - Task scheduling and gathering
    - Result collection

    Designed for async-first execution to align with cloud/backend patterns.
    """

    def __init__(
        self,
        parallel_trials: int,
        *,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialize parallel execution manager.

        Args:
            parallel_trials: Number of trials to run in parallel
            max_concurrent: Maximum concurrent tasks (defaults to parallel_trials)
        """
        self.parallel_trials = parallel_trials
        self.max_concurrent = max_concurrent or parallel_trials
        self._semaphore: asyncio.Semaphore | None = None
        # Lock to protect lazy semaphore initialization (P5 fix)
        self._semaphore_init_lock = asyncio.Lock()

    def calculate_batch_caps(
        self,
        remaining: float,
        remaining_samples: float | None = None,
    ) -> ParallelBatchCaps:
        """Calculate caps for parallel batch execution.

        Args:
            remaining: Remaining trial budget
            remaining_samples: Remaining sample budget (optional)

        Returns:
            ParallelBatchCaps with computed values
        """
        return calculate_parallel_batch_caps(
            remaining, remaining_samples, self.parallel_trials
        )

    def slice_configs(
        self,
        configs: list[dict[str, Any]],
        caps: ParallelBatchCaps,
    ) -> tuple[list[dict[str, Any]], int]:
        """Slice configs to respect caps.

        Args:
            configs: List of configuration dictionaries
            caps: Calculated batch caps

        Returns:
            Tuple of (sliced_configs, trials_prevented_count)
        """
        slice_cap = self.parallel_trials if caps.infinite_budget else caps.remaining_cap
        return slice_configs_to_cap(configs, slice_cap)

    async def run_tasks_with_semaphore(
        self,
        coroutines: list[Any],
    ) -> list[Any]:
        """Run coroutines with concurrency control via semaphore.

        Args:
            coroutines: List of coroutines to execute

        Returns:
            List of results from coroutines
        """
        if not coroutines:
            return []

        # Initialize semaphore with double-check locking pattern (P5 fix)
        if self._semaphore is None:
            async with self._semaphore_init_lock:
                # Double-check after acquiring lock
                if self._semaphore is None:
                    self._semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_task(coro: Any) -> Any:
            async with self._semaphore:  # type: ignore[union-attr]
                return await coro

        bounded_tasks = [bounded_task(coro) for coro in coroutines]
        return await asyncio.gather(*bounded_tasks, return_exceptions=False)

    async def gather_results(
        self,
        tasks: list[Any],
        *,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Gather results from async tasks.

        Simple wrapper around asyncio.gather for consistency.

        Args:
            tasks: List of coroutines/tasks to gather
            return_exceptions: Whether to return exceptions as results

        Returns:
            List of results
        """
        if not tasks:
            return []
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    def distribute_work(
        self,
        items: list[T],
        num_workers: int,
    ) -> list[list[T]]:
        """Distribute items across workers evenly.

        Args:
            items: Items to distribute
            num_workers: Number of workers

        Returns:
            List of item lists, one per worker
        """
        if not items or num_workers <= 0:
            return []

        # Distribute evenly with remainder going to first workers
        base_size = len(items) // num_workers
        remainder = len(items) % num_workers

        result: list[list[T]] = []
        start = 0
        for i in range(num_workers):
            # First 'remainder' workers get one extra item
            worker_size = base_size + (1 if i < remainder else 0)
            if worker_size > 0:
                result.append(items[start : start + worker_size])
                start += worker_size

        return result

    def should_use_parallel(
        self,
        remaining_cap: int,
        configs_count: int,
    ) -> bool:
        """Determine if parallel execution should be used.

        Args:
            remaining_cap: Remaining trial budget
            configs_count: Number of configs to execute

        Returns:
            True if parallel execution is beneficial
        """
        # Use parallel if we have more than 1 trial to run and capacity
        return self.parallel_trials > 1 and remaining_cap > 0 and configs_count > 0
