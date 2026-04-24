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
    from traigent.core.cost_enforcement import CostEnforcer

from traigent.core.cost_enforcement import Permit

logger = get_logger(__name__)

T = TypeVar("T")


def _close_unstarted_coroutine(coro: Any) -> None:
    """Close a coroutine object that will not be executed.

    ``run_with_cost_permits`` receives already-created coroutine objects. When a
    permit is denied, that coroutine is intentionally skipped; closing it avoids
    "coroutine was never awaited" warnings and releases any coroutine-local
    resources held before first execution.
    """

    close = getattr(coro, "close", None)
    if callable(close):
        close()


@dataclass
class ParallelBatchCaps:
    """Calculated caps for parallel batch execution."""

    remaining_cap: int
    target_batch_size: int
    infinite_budget: bool


@dataclass
class CostPermitResult:
    """Result of a cost permit check for parallel execution."""

    permitted: bool
    cancelled_count: int = 0


@dataclass
class PermittedTrialResult:
    """Result from a trial execution with associated cost permit info.

    This wrapper carries the Permit object through parallel execution
    so that track_cost_async can use the exact reserved amount and
    single-release semantics, even if the EMA estimate has changed
    since permit acquisition.
    """

    result: Any  # TrialResult, Exception, or None (cancelled)
    permit: Permit  # Permit object with single-release semantics


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
        cost_enforcer: CostEnforcer | None = None,
    ) -> None:
        """Initialize parallel execution manager.

        Args:
            parallel_trials: Number of trials to run in parallel
            max_concurrent: Maximum concurrent tasks (defaults to parallel_trials)
            cost_enforcer: Optional CostEnforcer for permit-based cost control
        """
        self.parallel_trials = parallel_trials
        self.max_concurrent = max_concurrent or parallel_trials
        self.cost_enforcer = cost_enforcer
        self._semaphore: asyncio.Semaphore | None = None
        # Lock to protect lazy semaphore initialization (P5 fix)
        self._semaphore_init_lock = asyncio.Lock()

    def set_cost_enforcer(self, cost_enforcer: CostEnforcer) -> None:
        """Set or update the cost enforcer for permit-based cost control.

        Args:
            cost_enforcer: CostEnforcer instance for cost tracking
        """
        self.cost_enforcer = cost_enforcer

    async def acquire_cost_permit(self) -> Permit:
        """Acquire a cost permit before executing a trial.

        Returns:
            Permit object if granted (with amount > 0 and is_granted=True),
            or a denied Permit (with amount=0 and is_granted=False) if limit reached.
        """
        if self.cost_enforcer is None:
            # No enforcer, return a mock permit for API compatibility
            return Permit(id=0, amount=0.05, active=True)
        return await self.cost_enforcer.acquire_permit_async()

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

    async def run_with_cost_permits(
        self,
        coroutines: list[Any],
        *,
        cancel_sentinel: Any = None,
    ) -> tuple[list[PermittedTrialResult], int]:
        """Run coroutines with cost permit checking before each execution.

        Each coroutine will only execute if a cost permit is acquired.
        Coroutines that fail to acquire a permit are cancelled and
        wrapped in PermittedTrialResult with a denied Permit.

        If a coroutine raises an exception after a permit was acquired,
        the permit is released to avoid stranding reserved budget.

        Results are wrapped in PermittedTrialResult to carry the Permit object
        through to track_cost_async, ensuring exact budget release with
        single-release semantics even if the cost estimate (EMA) changes.

        Args:
            coroutines: List of coroutines to execute with permit checking
            cancel_sentinel: Value to return for cancelled coroutines (default: None)

        Returns:
            Tuple of (permitted_results, cancelled_count) where each result is a
            PermittedTrialResult containing (result, permit).
        """
        if not coroutines:
            return [], 0

        cancelled_count = 0
        results: list[PermittedTrialResult] = []

        # Initialize semaphore with double-check locking pattern
        if self._semaphore is None:
            async with self._semaphore_init_lock:
                if self._semaphore is None:
                    self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create a denied permit for cancelled/failed trials
        denied_permit = Permit(id=-1, amount=0.0, active=False)

        async def execute_with_permit(coro: Any, index: int) -> tuple[int, Any, Permit]:
            """Execute coroutine if permit acquired, return (index, result, permit).

            Always returns a tuple (index, result_or_exception, permit)
            to preserve order and carry the permit through.
            If an exception occurs after permit acquisition, the permit is released
            and a denied permit is returned (already released).
            """
            nonlocal cancelled_count

            permit: Permit | None = None
            coroutine_started = False
            try:
                # Check permit before acquiring semaphore to fail fast
                permit = await self.acquire_cost_permit()
                if not permit.is_granted:
                    # Increment cancelled count, don't execute - no permit was acquired
                    cancelled_count += 1
                    _close_unstarted_coroutine(coro)
                    logger.info(
                        "Trial at index %d cancelled due to cost limit reached",
                        index,
                    )
                    # Return with denied permit (no permit was acquired)
                    return (index, cancel_sentinel, permit)

                # Permit acquired, execute with concurrency control
                # On exception, release the permit
                async with self._semaphore:  # type: ignore[union-attr]
                    coroutine_started = True
                    result = await coro
                    # Success: return with the permit for track_cost_async
                    return (index, result, permit)
            except asyncio.CancelledError:
                if not coroutine_started:
                    _close_unstarted_coroutine(coro)
                if (
                    permit is not None
                    and permit.is_granted
                    and self.cost_enforcer is not None
                ):
                    await self.cost_enforcer.release_permit_async(permit)
                    logger.debug(
                        "Released permit %d for trial %d after cancellation",
                        permit.id,
                        index,
                    )
                raise
            except BaseException as e:
                # Release the permit on exception - the trial failed after permit
                # was granted, so we need to release the reservation
                if (
                    permit is not None
                    and permit.is_granted
                    and self.cost_enforcer is not None
                ):
                    await self.cost_enforcer.release_permit_async(permit)
                    logger.debug(
                        "Released permit %d for trial %d after exception: %s",
                        permit.id,
                        index,
                        type(e).__name__,
                    )
                # Return exception with denied permit (already released)
                return (index, e, denied_permit)

        # Create bounded tasks preserving order
        tasks = [execute_with_permit(coro, idx) for idx, coro in enumerate(coroutines)]

        # Run all tasks and collect results
        indexed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort by index to preserve original order
        indexed_results = sorted(
            indexed_results,
            key=lambda x: x[0] if isinstance(x, tuple) and len(x) >= 1 else -1,
        )

        # Extract results and wrap in PermittedTrialResult
        for item in indexed_results:
            if isinstance(item, tuple) and len(item) == 3:
                _index, result, permit = item
                results.append(PermittedTrialResult(result=result, permit=permit))
            elif isinstance(item, tuple) and len(item) == 2:
                # Legacy format (should not happen, but handle gracefully)
                _index, result = item
                results.append(
                    PermittedTrialResult(result=result, permit=denied_permit)
                )
            else:
                # Exception from gather itself - wrap with denied permit
                results.append(PermittedTrialResult(result=item, permit=denied_permit))

        if cancelled_count > 0:
            logger.info(
                "Cancelled %d trial(s) due to cost limit",
                cancelled_count,
            )

        return results, cancelled_count
