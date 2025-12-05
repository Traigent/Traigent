"""Sample budget coordination for intra-trial enforcement."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import threading
from dataclasses import dataclass

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LeaseClosure:
    """Summary emitted when a lease completes."""

    trial_id: str
    consumed: int
    exhausted: bool
    global_remaining: float
    wasted: int


@dataclass(slots=True)
class BudgetMetrics:
    """Aggregate metrics capturing sample budget utilisation."""

    total_budget: int | None
    consumed: int = 0
    wasted: int = 0

    @property
    def remaining(self) -> float:
        if self.total_budget is None:
            return float("inf")
        return max(self.total_budget - self.consumed, 0)

    @property
    def efficiency(self) -> float:
        if self.consumed <= 0:
            return 0.0
        used = max(self.consumed - self.wasted, 0)
        return max(0.0, min(1.0, used / self.consumed))


class SampleBudgetLease:
    """Per-trial lease that mediates access to the shared budget."""

    def __init__(
        self,
        manager: SampleBudgetManager,
        trial_id: str,
        *,
        ceiling: int | None,
    ) -> None:
        self._manager = manager
        self.trial_id = trial_id
        self._ceiling = ceiling if ceiling is None or ceiling >= 0 else 0
        self._consumed = 0
        self._closed = False
        self._exhausted = False
        self._wasted = 0
        self._lock = threading.RLock()

    def remaining(self) -> float:
        """Return remaining allowance for this lease."""
        with self._lock:
            if self._closed:
                return 0
            return self._manager._remaining_for_lease(self)  # noqa: SLF001

    def try_take(self, count: int = 1) -> bool:
        """Attempt to consume ``count`` examples from the global budget."""
        if count <= 0:
            raise ValueError("count must be a positive integer")

        with self._lock:
            if self._closed:
                logger.debug("Lease %s already closed", self.trial_id)
                return False

            acquired = self._manager._acquire(self, count)  # noqa: SLF001
            if acquired:
                self._consumed += count
            else:
                self._exhausted = True
            return acquired

    def rollback(self, count: int) -> None:
        """Return unused samples to the manager."""
        if count <= 0:
            return

        with self._lock:
            if count > self._consumed:
                count = self._consumed
            if count <= 0:
                return
            self._consumed -= count
            self._manager._release(self, count)  # noqa: SLF001
            self._wasted += count
            self._exhausted = False

    def finalize(self) -> LeaseClosure:
        """Close the lease and report utilisation."""
        with self._lock:
            if self._closed:
                global_remaining = self._manager.remaining()
                return LeaseClosure(
                    trial_id=self.trial_id,
                    consumed=self._consumed,
                    exhausted=self._exhausted or global_remaining == 0,
                    global_remaining=global_remaining,
                    wasted=self._wasted,
                )

            self._closed = True
            closure = self._manager._finalize(self)  # noqa: SLF001
            self._exhausted = self._exhausted or closure.exhausted
            return closure

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._exhausted


class SampleBudgetManager:
    """Thread-safe coordination for intra-trial sample budgets."""

    def __init__(
        self,
        total_budget: int | None,
        *,
        include_pruned: bool = True,
    ) -> None:
        if total_budget is not None and total_budget <= 0:
            raise ValueError("total_budget must be positive or None")

        self._total_budget = total_budget
        self._include_pruned = include_pruned
        self._lock = threading.RLock()
        self._consumed = 0
        self._wasted = 0
        self._leases: dict[str, SampleBudgetLease] = {}

    @property
    def include_pruned(self) -> bool:
        return self._include_pruned

    def create_lease(
        self, trial_id: str, *, ceiling: int | None = None
    ) -> SampleBudgetLease:
        """Create a new lease for ``trial_id``."""
        with self._lock:
            lease = SampleBudgetLease(self, trial_id, ceiling=ceiling)
            self._leases[trial_id] = lease
            logger.debug(
                "Created sample budget lease for trial %s (ceiling=%s, remaining=%s)",
                trial_id,
                ceiling,
                self.remaining(),
            )
            return lease

    def remaining(self) -> float:
        """Return global remaining budget."""
        with self._lock:
            if self._total_budget is None:
                return float("inf")
            return max(self._total_budget - self._consumed, 0)

    def consumed(self) -> int:
        """Return total consumed examples."""
        with self._lock:
            return self._consumed

    def snapshot(self) -> BudgetMetrics:
        """Return current aggregate metrics."""
        with self._lock:
            return BudgetMetrics(
                total_budget=self._total_budget,
                consumed=self._consumed,
                wasted=self._wasted,
            )

    # --- Internal helpers (used by SampleBudgetLease) ---
    def _remaining_for_lease(self, lease: SampleBudgetLease) -> float:
        ceiling_remaining = float("inf")
        if lease._ceiling is not None:  # noqa: SLF001
            ceiling_remaining = max(lease._ceiling - lease._consumed, 0)  # noqa: SLF001

        with self._lock:
            global_remaining = self.remaining()
        if self._total_budget is None:
            return ceiling_remaining
        return min(ceiling_remaining, global_remaining)

    def _acquire(self, lease: SampleBudgetLease, count: int) -> bool:
        with self._lock:
            if self._total_budget is not None and self._consumed >= self._total_budget:
                return False

            if (
                lease._ceiling is not None and lease._consumed + count > lease._ceiling
            ):  # noqa: SLF001
                return False

            if self._total_budget is not None:
                remaining = self._total_budget - self._consumed
                if count > remaining:
                    return False

            self._consumed += count
            return True

    def _release(self, lease: SampleBudgetLease, count: int) -> None:
        with self._lock:
            if self._total_budget is not None:
                self._consumed = max(self._consumed - count, 0)
            self._wasted += count

    def _finalize(self, lease: SampleBudgetLease) -> LeaseClosure:
        with self._lock:
            exhausted = False
            if self._total_budget is not None and self._consumed >= self._total_budget:
                exhausted = True

            self._leases.pop(lease.trial_id, None)

            remaining = self.remaining()
            logger.debug(
                "Finalised lease %s: consumed=%s, remaining=%s, exhausted=%s",
                lease.trial_id,
                lease._consumed,  # noqa: SLF001
                remaining,
                exhausted,
            )
            return LeaseClosure(
                trial_id=lease.trial_id,
                consumed=lease._consumed,  # noqa: SLF001
                exhausted=exhausted,
                global_remaining=remaining,
                wasted=lease._wasted,  # noqa: SLF001
            )
