"""Cumulative execution budget shared across optimize() phases.

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

``ExecutionBudget`` is an **experimental** primitive (issue #1980): a single cap on
cost / examples / wall-clock time that can be shared across direct ``evaluate()``
and ``optimize()`` calls (e.g. baseline -> search -> holdout phases), so the
*total* spend across the whole workflow is bounded rather than each call being
independently capped.

Honesty contract (mirrors the SDK's cost-accounting discipline):

- **Examples are a hard limit** — observable on every execution path, so exhaustion
  is enforced deterministically.
- **The wall-clock deadline is enforced at trial boundaries.** Between trials it is a
  hard stop; a single *hung* trial (a deadlocked sampler or stuck provider call) may
  overrun the deadline by the orchestrator's watchdog grace (25% of the remaining
  budget, floored at ``1s``, capped at ``5min``) before the watchdog aborts it — so
  the deadline is hard at trial granularity, not a mid-trial guillotine.
- **The monetary cap is enforced at batch/trial boundaries, not per token.** The
  orchestrator checks the shared remaining in its pre-batch admission gate before
  each batch and in the stop condition between trials. In *parallel* mode a batch of
  up to ``parallel_trials`` trials is admitted together (the gate only admits a batch
  the remaining can fund in full), so the cap is a tight bound of **± one batch**
  (± one trial in sequential mode) — not a per-trial-exact hard ceiling. Any overshoot
  is bounded to one batch's newly-admitted work and only arises when trials cost more
  than their pre-batch estimate.
- **The monetary cap is a hard limit only when cost is fully observable.** On raw
  provider paths, self-hosted models, or unpriced models the SDK cannot see the true
  cost, so a trial may debit ``$0`` it did not actually cost. When that happens the
  budget reports ``cost_tracking != "complete"`` and treats the consumed cost as a
  **lower bound** — it never claims a hard monetary guarantee it cannot honor.
- **``enforce_untracked_cost=True`` fails closed mid-run only for trials that report
  NO cost** (``cost is None`` — e.g. a raw-provider trial with no usage data): the
  next admission after such a trial stops with ``stop_reason == "execution_budget"``.
  A model that is merely *unpriced* but reports a concrete ``$0`` is NOT caught
  mid-run (that debit is indistinguishable from a genuinely free ``$0`` trial); the
  silent-``$0`` path is reconciled at **finalization** — folded into
  ``cost_tracking`` via the ``CostEnforcer`` ``unknown_cost_mode`` state and the
  ``UNPRICED_MODEL_RUNTIME`` warning (see ``mark_cost_untracked``) and surfaced as a
  lower-bound cost, not stopped while running.

The budget is thread-safe (``threading.RLock``); ``debit_trial`` is a plain sync
method called from inside the orchestrator's async state-lock and **never raises into
the hot path** — exhaustion is a graceful stop, the same contract as ``CostEnforcer``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

from traigent.utils.exceptions import ConfigurationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

CostTracking = Literal["complete", "partial", "untracked"]

# Dimension identifiers returned by ``exhausted_dimension`` / snapshots.
_DIM_COST = "cost"
_DIM_EXAMPLES = "examples"
_DIM_DEADLINE = "deadline"
_DIM_UNTRACKED_COST = "untracked_cost"


@dataclass(frozen=True, slots=True)
class ExecutionBudgetSnapshot:
    """Immutable point-in-time view of an :class:`ExecutionBudget`.

    ``remaining_*`` fields are ``float("inf")`` for an unbounded dimension (the
    ``BudgetMetrics.remaining`` convention). ``as_dict`` renders unbounded remaining
    as ``None`` so the snapshot is JSON-serializable when written to
    ``OptimizationResult.metadata``.
    """

    max_cost_usd: float | None
    max_examples: int | None
    deadline_seconds: float | None
    consumed_cost: float
    consumed_examples: int
    elapsed_seconds: float
    remaining_cost: float
    remaining_examples: float
    remaining_seconds: float
    runs: int
    trials: int
    untracked_trials: int
    cost_tracking: CostTracking
    exhausted_dimension: str | None

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return None if value == float("inf") else value

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict (unbounded remaining -> ``None``)."""
        return {
            "max_cost_usd": self.max_cost_usd,
            "max_examples": self.max_examples,
            "deadline_seconds": self.deadline_seconds,
            "consumed_cost": self.consumed_cost,
            "consumed_examples": self.consumed_examples,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_cost": self._finite_or_none(self.remaining_cost),
            "remaining_examples": self._finite_or_none(self.remaining_examples),
            "remaining_seconds": self._finite_or_none(self.remaining_seconds),
            "runs": self.runs,
            "trials": self.trials,
            "untracked_trials": self.untracked_trials,
            "cost_tracking": self.cost_tracking,
            "exhausted_dimension": self.exhausted_dimension,
        }


def _validate_positive(value: object, name: str) -> float | None:
    """Reject the ``<= 0`` footgun for a budget dimension (mirrors #1684)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"{name} must be a positive number, got {value!r} ({type(value).__name__})."
        )
    numeric = float(value)
    if numeric != numeric or numeric <= 0:  # NaN or non-positive
        raise ConfigurationError(
            f"{name} must be > 0, got {numeric!r}. A nonpositive limit would "
            "block every trial before it starts (silent 0-trial run). Leave it "
            "unset to leave that dimension unbounded."
        )
    return numeric


class ExecutionBudget:
    """A cumulative cost/examples/time cap shared across ``optimize()`` calls.

    Experimental (issue #1980). At least one of ``max_cost_usd`` / ``max_examples`` /
    ``deadline_seconds`` must be set; any value ``<= 0`` raises
    :class:`ConfigurationError` at construction.

    Attach it explicitly per call — ``evaluator.evaluate(..., budget=b)`` /
    ``my_fn.optimize(budget=b)`` / ``my_fn.optimize_sync(budget=b)`` — passing the
    *same* instance to each phase so the shared remaining is spent down across
    baseline, search, and holdout. Using it as a context manager starts the
    wall-clock deadline on ``__enter__`` and freezes a final snapshot on
    ``__exit__``; it never registers an ambient/implicit budget.
    """

    def __init__(
        self,
        *,
        max_cost_usd: float | None = None,
        max_examples: int | None = None,
        deadline_seconds: float | None = None,
        enforce_untracked_cost: bool = False,
    ) -> None:
        self.max_cost_usd = _validate_positive(max_cost_usd, "max_cost_usd")
        validated_examples = _validate_positive(max_examples, "max_examples")
        self.max_examples: int | None = (
            None if validated_examples is None else int(validated_examples)
        )
        self.deadline_seconds = _validate_positive(deadline_seconds, "deadline_seconds")
        self.enforce_untracked_cost = bool(enforce_untracked_cost)

        if (
            self.max_cost_usd is None
            and self.max_examples is None
            and self.deadline_seconds is None
        ):
            raise ConfigurationError(
                "ExecutionBudget requires at least one of max_cost_usd, "
                "max_examples, or deadline_seconds."
            )

        self._lock = threading.RLock()
        self._start_monotonic: float | None = None
        self._consumed_cost: float = 0.0
        self._consumed_examples: int = 0
        self._runs: int = 0
        self._trials: int = 0
        self._untracked_trials: int = 0
        self._external_untracked: bool = False
        self._final_snapshot: ExecutionBudgetSnapshot | None = None

    # -- clock / run lifecycle -------------------------------------------------

    def _start_clock_locked(self) -> None:
        if self._start_monotonic is None:
            self._start_monotonic = time.monotonic()

    def start_clock(self) -> None:
        """Start the wall-clock deadline (idempotent)."""
        with self._lock:
            self._start_clock_locked()

    def begin_run(self) -> None:
        """Mark the start of an attached optimization run.

        Starts the deadline clock (idempotent) and increments the run counter.
        Called by the orchestrator at run start when a budget is attached; a
        no-budget run never calls this, keeping the absent path byte-identical.
        """
        with self._lock:
            self._start_clock_locked()
            self._runs += 1

    # -- debit / record --------------------------------------------------------

    def debit_trial(
        self,
        *,
        cost: float | None,
        examples: int = 0,
        untracked: bool = False,
        trial_id: str | None = None,
    ) -> None:
        """Debit one completed trial from the shared budget.

        Sync, called from inside the orchestrator's async state-lock. **Never
        raises into the hot path** — exhaustion is a graceful stop, evaluated by
        the stop condition, not by this method.

        Args:
            cost: Observed trial cost in USD, or ``None`` when the cost was not
                observable (raw provider / unpriced path). ``None`` counts the
                trial as untracked.
            examples: Number of examples this trial attempted.
            untracked: Force-mark the trial as untracked even if a cost is given.
            trial_id: Trial identifier, for debug logging only.
        """
        try:
            with self._lock:
                self._start_clock_locked()
                self._trials += 1
                if cost is not None:
                    self._consumed_cost += float(cost)
                if untracked or cost is None:
                    self._untracked_trials += 1
                if examples:
                    self._consumed_examples += int(examples)
        except Exception:  # pragma: no cover - defensive; never break a run
            logger.debug(
                "ExecutionBudget.debit_trial swallowed an error for trial %s",
                trial_id,
                exc_info=True,
            )

    def record_external(
        self,
        *,
        cost_usd: float | None = None,
        examples: int | None = None,
    ) -> None:
        """Record consumption from a non-optimize path (``.run()`` / ``__call__``).

        The escape hatch for production invocations and raw provider calls whose
        cost/usage the optimizer never sees (fed e.g. from ``with_usage()`` data).
        A ``cost_usd`` of ``None`` marks the budget's cost tracking as incomplete.
        """
        with self._lock:
            self._start_clock_locked()
            if cost_usd is not None:
                self._consumed_cost += float(cost_usd)
            else:
                self._external_untracked = True
            if examples is not None:
                self._consumed_examples += int(examples)

    def mark_cost_untracked(self) -> None:
        """Flag that some cost on this run was unobservable (fold-in at finalize).

        Called by the orchestrator/finalizer to fold in the ``CostEnforcer``
        ``unknown_cost_mode`` state and the ``UNPRICED_MODEL_RUNTIME`` warning, so
        ``cost_tracking`` reflects every silent-``$0`` path, not just per-trial ones.
        """
        with self._lock:
            self._external_untracked = True

    # -- locked computations ---------------------------------------------------

    def _elapsed_seconds_locked(self) -> float:
        if self._start_monotonic is None:
            return 0.0
        return max(time.monotonic() - self._start_monotonic, 0.0)

    def _remaining_cost_locked(self) -> float:
        if self.max_cost_usd is None:
            return float("inf")
        return max(self.max_cost_usd - self._consumed_cost, 0.0)

    def _remaining_examples_locked(self) -> float:
        if self.max_examples is None:
            return float("inf")
        return float(max(self.max_examples - self._consumed_examples, 0))

    def _remaining_seconds_locked(self) -> float:
        if self.deadline_seconds is None:
            return float("inf")
        return max(self.deadline_seconds - self._elapsed_seconds_locked(), 0.0)

    def _cost_tracking_locked(self) -> CostTracking:
        if self._trials == 0:
            return "untracked" if self._external_untracked else "complete"
        if self._untracked_trials == 0 and not self._external_untracked:
            return "complete"
        if self._untracked_trials >= self._trials:
            return "untracked"
        return "partial"

    def _exhausted_dimension_locked(self) -> str | None:
        if self.enforce_untracked_cost and self._cost_tracking_locked() != "complete":
            return _DIM_UNTRACKED_COST
        if self._remaining_cost_locked() <= 0.0:
            return _DIM_COST
        if self._remaining_examples_locked() <= 0.0:
            return _DIM_EXAMPLES
        if self._remaining_seconds_locked() <= 0.0:
            return _DIM_DEADLINE
        return None

    # -- public read surface ---------------------------------------------------

    @property
    def consumed_cost(self) -> float:
        with self._lock:
            return self._consumed_cost

    @property
    def consumed_examples(self) -> int:
        with self._lock:
            return self._consumed_examples

    @property
    def elapsed_seconds(self) -> float:
        with self._lock:
            return self._elapsed_seconds_locked()

    @property
    def remaining_cost(self) -> float:
        with self._lock:
            return self._remaining_cost_locked()

    @property
    def remaining_examples(self) -> float:
        with self._lock:
            return self._remaining_examples_locked()

    @property
    def remaining_seconds(self) -> float:
        with self._lock:
            return self._remaining_seconds_locked()

    @property
    def exhausted_dimension(self) -> str | None:
        """The first exhausted dimension, or ``None`` when budget remains."""
        with self._lock:
            return self._exhausted_dimension_locked()

    @property
    def cost_tracking(self) -> CostTracking:
        """Tri-state honesty flag for the monetary cap.

        ``"complete"`` when every trial's cost was observed; ``"partial"`` when some
        were unobservable; ``"untracked"`` when none were. The cap is a hard
        guarantee only when this is ``"complete"``.
        """
        with self._lock:
            return self._cost_tracking_locked()

    @property
    def was_any_cost_untracked(self) -> bool:
        """True when the monetary cap is a lower bound, not a hard guarantee."""
        with self._lock:
            return self._cost_tracking_locked() != "complete"

    def snapshot(self) -> ExecutionBudgetSnapshot:
        """Return an immutable snapshot of the current budget state."""
        with self._lock:
            return ExecutionBudgetSnapshot(
                max_cost_usd=self.max_cost_usd,
                max_examples=self.max_examples,
                deadline_seconds=self.deadline_seconds,
                consumed_cost=self._consumed_cost,
                consumed_examples=self._consumed_examples,
                elapsed_seconds=self._elapsed_seconds_locked(),
                remaining_cost=self._remaining_cost_locked(),
                remaining_examples=self._remaining_examples_locked(),
                remaining_seconds=self._remaining_seconds_locked(),
                runs=self._runs,
                trials=self._trials,
                untracked_trials=self._untracked_trials,
                cost_tracking=self._cost_tracking_locked(),
                exhausted_dimension=self._exhausted_dimension_locked(),
            )

    # -- context manager (deadline clock + final snapshot) ---------------------

    def __enter__(self) -> ExecutionBudget:
        self.start_clock()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        # Freeze a final snapshot; never suppress exceptions.
        self._final_snapshot = self.snapshot()
        return False

    @property
    def final_snapshot(self) -> ExecutionBudgetSnapshot | None:
        """The snapshot frozen on ``__exit__`` (``None`` if never used as a CM)."""
        return self._final_snapshot

    def __repr__(self) -> str:
        snap = self.snapshot()
        tracking = snap.cost_tracking
        caveat = (
            ""
            if tracking == "complete"
            else " cost=LOWER_BOUND(not a hard guarantee: some cost was unobservable)"
        )
        return (
            "ExecutionBudget("
            f"max_cost_usd={self.max_cost_usd}, max_examples={self.max_examples}, "
            f"deadline_seconds={self.deadline_seconds}, "
            f"consumed_cost={snap.consumed_cost:.6g}, "
            f"consumed_examples={snap.consumed_examples}, "
            f"elapsed_seconds={snap.elapsed_seconds:.3g}, "
            f"trials={snap.trials}, cost_tracking={tracking!r}{caveat})"
        )


__all__ = ["ExecutionBudget", "ExecutionBudgetSnapshot"]
