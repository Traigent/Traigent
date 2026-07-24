"""Unit tests for the cumulative :class:`ExecutionBudget` primitive (issue #1980).

These tests are offline/pure — they exercise the budget object in isolation with
no orchestrator, no evaluator, no LLM, and no network. Behaviour is asserted
against the real implementation in ``traigent.core.execution_budget``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import FrozenInstanceError

import pytest

from traigent import ExecutionBudget as PublicExecutionBudget
from traigent.core.execution_budget import (
    ExecutionBudget,
    ExecutionBudgetSnapshot,
)
from traigent.utils.exceptions import ConfigurationError

INF = float("inf")


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------


def test_public_export_is_the_same_class() -> None:
    """``traigent.ExecutionBudget`` resolves to the core class (lazy export)."""
    assert PublicExecutionBudget is ExecutionBudget


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_all_none_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="at least one of"):
        ExecutionBudget()


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
def test_nonpositive_cost_raises(bad: float) -> None:
    with pytest.raises(ConfigurationError, match="max_cost_usd"):
        ExecutionBudget(max_cost_usd=bad)


@pytest.mark.parametrize("bad", [0, -3])
def test_nonpositive_examples_raises(bad: int) -> None:
    with pytest.raises(ConfigurationError, match="max_examples"):
        ExecutionBudget(max_examples=bad)


@pytest.mark.parametrize("bad", [0, 0.0, -2.0])
def test_nonpositive_deadline_raises(bad: float) -> None:
    with pytest.raises(ConfigurationError, match="deadline_seconds"):
        ExecutionBudget(deadline_seconds=bad)


def test_nan_cost_raises() -> None:
    with pytest.raises(ConfigurationError):
        ExecutionBudget(max_cost_usd=float("nan"))


def test_bool_is_rejected_as_non_numeric() -> None:
    # A bool is not a valid numeric budget dimension (mirrors #1684 guard).
    with pytest.raises(ConfigurationError):
        ExecutionBudget(max_examples=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cost_usd": 1.5},
        {"max_examples": 10},
        {"deadline_seconds": 30.0},
    ],
)
def test_single_dimension_is_enough(kwargs: dict[str, float]) -> None:
    budget = ExecutionBudget(**kwargs)
    assert isinstance(budget, ExecutionBudget)


def test_max_examples_coerced_to_int() -> None:
    budget = ExecutionBudget(max_examples=5.0)
    assert budget.max_examples == 5
    assert isinstance(budget.max_examples, int)


def test_enforce_untracked_cost_defaults_false() -> None:
    assert ExecutionBudget(max_cost_usd=1.0).enforce_untracked_cost is False
    assert (
        ExecutionBudget(
            max_cost_usd=1.0, enforce_untracked_cost=True
        ).enforce_untracked_cost
        is True
    )


# ---------------------------------------------------------------------------
# debit_trial accounting
# ---------------------------------------------------------------------------


def test_debit_trial_accumulates_cost_and_examples() -> None:
    budget = ExecutionBudget(max_cost_usd=10.0, max_examples=100)
    budget.debit_trial(cost=0.25, examples=3)
    budget.debit_trial(cost=0.75, examples=7)

    snap = budget.snapshot()
    assert snap.consumed_cost == pytest.approx(1.0)
    assert snap.consumed_examples == 10
    assert snap.trials == 2
    assert snap.untracked_trials == 0
    assert budget.consumed_cost == pytest.approx(1.0)
    assert budget.consumed_examples == 10


def test_debit_trial_none_cost_marks_untracked() -> None:
    budget = ExecutionBudget(max_cost_usd=10.0)
    budget.debit_trial(cost=None, examples=2)

    snap = budget.snapshot()
    assert snap.trials == 1
    assert snap.untracked_trials == 1
    # None cost contributes nothing to consumed_cost (cost is a lower bound).
    assert snap.consumed_cost == pytest.approx(0.0)
    assert snap.consumed_examples == 2


def test_debit_trial_untracked_flag_forces_untracked_even_with_cost() -> None:
    budget = ExecutionBudget(max_cost_usd=10.0)
    budget.debit_trial(cost=0.5, untracked=True)

    snap = budget.snapshot()
    assert snap.trials == 1
    assert snap.untracked_trials == 1
    # The observed cost is still added, but the trial counts as untracked.
    assert snap.consumed_cost == pytest.approx(0.5)


def test_debit_trial_default_examples_is_zero() -> None:
    budget = ExecutionBudget(max_examples=10)
    budget.debit_trial(cost=None)
    assert budget.consumed_examples == 0
    assert budget.snapshot().trials == 1


def test_debit_trial_never_raises_on_bad_examples() -> None:
    # Even a pathological examples value must not raise into the hot path.
    budget = ExecutionBudget(max_examples=10)
    budget.debit_trial(cost=1.0, examples="not-an-int")  # type: ignore[arg-type]
    # The trial counter still advanced before the swallowed error, or the whole
    # debit was swallowed — either way, no exception escaped.


# ---------------------------------------------------------------------------
# remaining_* — inf when unbounded, floors at zero
# ---------------------------------------------------------------------------


def test_remaining_is_inf_for_unbounded_dimensions() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    assert budget.remaining_examples == INF
    assert budget.remaining_seconds == INF
    assert budget.remaining_cost == pytest.approx(1.0)


def test_remaining_cost_decreases_and_floors_at_zero() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=0.4)
    assert budget.remaining_cost == pytest.approx(0.6)
    budget.debit_trial(cost=5.0)  # overspend
    assert budget.remaining_cost == pytest.approx(0.0)


def test_remaining_examples_decreases_and_floors_at_zero() -> None:
    budget = ExecutionBudget(max_examples=5)
    budget.debit_trial(cost=0.0, examples=2)
    assert budget.remaining_examples == pytest.approx(3.0)
    budget.debit_trial(cost=0.0, examples=10)
    assert budget.remaining_examples == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# snapshot() / as_dict()
# ---------------------------------------------------------------------------


def test_snapshot_is_immutable_point_in_time() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0, max_examples=10)
    snap = budget.snapshot()
    assert isinstance(snap, ExecutionBudgetSnapshot)
    # Mutating the budget after snapshotting does not change the frozen view.
    budget.debit_trial(cost=0.5, examples=2)
    assert snap.consumed_cost == pytest.approx(0.0)
    assert snap.consumed_examples == 0
    with pytest.raises(FrozenInstanceError):
        snap.consumed_cost = 9.9  # type: ignore[misc]


def test_as_dict_renders_unbounded_remaining_as_none() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    data = budget.snapshot().as_dict()
    assert data["remaining_cost"] == pytest.approx(1.0)
    assert data["remaining_examples"] is None
    assert data["remaining_seconds"] is None
    assert data["cost_tracking"] == "complete"
    assert data["exhausted_dimension"] is None
    assert data["max_examples"] is None


def test_as_dict_is_json_serializable() -> None:
    import json

    budget = ExecutionBudget(max_cost_usd=1.0, max_examples=10, deadline_seconds=5.0)
    budget.debit_trial(cost=0.1, examples=1)
    # Round-trips without raising -> safe for OptimizationResult.metadata.
    round_tripped = json.loads(json.dumps(budget.snapshot().as_dict()))
    assert round_tripped["trials"] == 1


# ---------------------------------------------------------------------------
# record_external
# ---------------------------------------------------------------------------


def test_record_external_accumulates_cost_and_examples() -> None:
    budget = ExecutionBudget(max_cost_usd=10.0, max_examples=100)
    budget.record_external(cost_usd=1.5, examples=4)
    snap = budget.snapshot()
    assert snap.consumed_cost == pytest.approx(1.5)
    assert snap.consumed_examples == 4
    # record_external is not a "trial" — it does not bump the trial counter.
    assert snap.trials == 0


def test_record_external_none_cost_marks_tracking_incomplete() -> None:
    budget = ExecutionBudget(max_cost_usd=10.0)
    budget.record_external(cost_usd=None, examples=2)
    # No trials, but external untracked flag flips tracking away from complete.
    assert budget.cost_tracking == "untracked"
    assert budget.was_any_cost_untracked is True
    assert budget.consumed_examples == 2


# ---------------------------------------------------------------------------
# cost_tracking synthesis + was_any_cost_untracked
# ---------------------------------------------------------------------------


def test_cost_tracking_complete_when_no_activity() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    assert budget.cost_tracking == "complete"
    assert budget.was_any_cost_untracked is False


def test_cost_tracking_complete_when_all_trials_tracked() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=0.1)
    budget.debit_trial(cost=0.2)
    assert budget.cost_tracking == "complete"
    assert budget.was_any_cost_untracked is False


def test_cost_tracking_partial_when_some_untracked() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=0.1)
    budget.debit_trial(cost=None)
    assert budget.cost_tracking == "partial"
    assert budget.was_any_cost_untracked is True


def test_cost_tracking_untracked_when_all_untracked() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=None)
    budget.debit_trial(cost=None)
    assert budget.cost_tracking == "untracked"
    assert budget.was_any_cost_untracked is True


def test_mark_cost_untracked_folds_into_tracking() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=0.1)
    assert budget.cost_tracking == "complete"
    budget.mark_cost_untracked()  # e.g. UNPRICED_MODEL_RUNTIME fold-in at finalize
    assert budget.cost_tracking == "partial"
    assert budget.was_any_cost_untracked is True


# ---------------------------------------------------------------------------
# exhausted_dimension
# ---------------------------------------------------------------------------


def test_exhausted_dimension_none_when_budget_remains() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0, max_examples=10)
    assert budget.exhausted_dimension is None


def test_exhausted_dimension_cost() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=1.0)
    assert budget.exhausted_dimension == "cost"


def test_exhausted_dimension_examples() -> None:
    budget = ExecutionBudget(max_examples=3)
    budget.debit_trial(cost=0.0, examples=3)
    assert budget.exhausted_dimension == "examples"


def test_exhausted_dimension_deadline_via_clock() -> None:
    budget = ExecutionBudget(deadline_seconds=0.01)
    budget.start_clock()
    # Force the monotonic start far into the past for a deterministic deadline.
    budget._start_monotonic = time.monotonic() - 100.0
    assert budget.remaining_seconds == pytest.approx(0.0)
    assert budget.exhausted_dimension == "deadline"


def test_enforce_untracked_cost_exhausts_on_untracked_trial() -> None:
    budget = ExecutionBudget(max_examples=100, enforce_untracked_cost=True)
    assert budget.exhausted_dimension is None
    budget.debit_trial(cost=None)  # cost became unobservable
    assert budget.exhausted_dimension == "untracked_cost"


def test_enforce_untracked_cost_stays_open_when_tracked() -> None:
    budget = ExecutionBudget(max_cost_usd=10.0, enforce_untracked_cost=True)
    budget.debit_trial(cost=0.1)
    assert budget.exhausted_dimension is None


def test_untracked_cost_takes_precedence_over_cost_dimension() -> None:
    # When enforce_untracked_cost is set and cost is unobservable, the reported
    # dimension is untracked_cost even if the (observable) cost is also spent.
    budget = ExecutionBudget(max_cost_usd=1.0, enforce_untracked_cost=True)
    budget.debit_trial(cost=None)
    assert budget.exhausted_dimension == "untracked_cost"


# ---------------------------------------------------------------------------
# Deadline clock + context manager
# ---------------------------------------------------------------------------


def test_start_clock_is_idempotent() -> None:
    budget = ExecutionBudget(deadline_seconds=10.0)
    budget.start_clock()
    first = budget._start_monotonic
    budget.start_clock()
    assert budget._start_monotonic == first


def test_elapsed_zero_before_clock_started() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    assert budget.elapsed_seconds == pytest.approx(0.0)


def test_begin_run_starts_clock_and_counts_runs() -> None:
    budget = ExecutionBudget(deadline_seconds=10.0)
    assert budget.snapshot().runs == 0
    budget.begin_run()
    budget.begin_run()
    snap = budget.snapshot()
    assert snap.runs == 2
    assert budget._start_monotonic is not None


def test_context_manager_starts_clock_and_freezes_snapshot() -> None:
    budget = ExecutionBudget(deadline_seconds=10.0)
    assert budget.final_snapshot is None
    with budget as entered:
        assert entered is budget
        assert budget._start_monotonic is not None
        budget.debit_trial(cost=None, examples=1)
    frozen = budget.final_snapshot
    assert frozen is not None
    assert frozen.consumed_examples == 1


def test_context_manager_does_not_suppress_exceptions() -> None:
    budget = ExecutionBudget(deadline_seconds=10.0)
    with pytest.raises(ValueError, match="boom"):
        with budget:
            raise ValueError("boom")
    # Snapshot still frozen on the way out.
    assert budget.final_snapshot is not None


def test_repr_flags_lower_bound_when_untracked() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=None)
    text = repr(budget)
    assert "cost_tracking='untracked'" in text
    assert "LOWER_BOUND" in text


# ---------------------------------------------------------------------------
# Thread-safety / concurrency (model on test_cost_enforcement_concurrency.py)
# ---------------------------------------------------------------------------


class TestExecutionBudgetConcurrency:
    """Many threads debiting a single shared budget must keep invariants."""

    def test_concurrent_debits_are_exactly_accounted(self) -> None:
        budget = ExecutionBudget(max_cost_usd=10_000.0, max_examples=10_000_000)

        threads = 40
        per_thread = 50
        cost_each = 0.01
        examples_each = 2

        def worker() -> None:
            for _ in range(per_thread):
                budget.debit_trial(cost=cost_each, examples=examples_each)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(worker) for _ in range(threads)]
            for future in as_completed(futures):
                future.result()  # re-raise any thread error

        expected_trials = threads * per_thread
        snap = budget.snapshot()
        assert snap.trials == expected_trials
        assert snap.consumed_examples == expected_trials * examples_each
        assert snap.consumed_cost == pytest.approx(expected_trials * cost_each)
        # Nothing was lost or double-counted -> remaining is exactly consistent.
        assert snap.remaining_cost == pytest.approx(
            10_000.0 - expected_trials * cost_each
        )

    def test_concurrent_mixed_tracked_and_untracked(self) -> None:
        budget = ExecutionBudget(max_cost_usd=10_000.0)
        threads = 30
        per_thread = 40

        def worker(untracked: bool) -> None:
            for _ in range(per_thread):
                budget.debit_trial(cost=None if untracked else 0.01)

        barrier = threading.Barrier(threads)

        def run(idx: int) -> None:
            barrier.wait()
            worker(untracked=(idx % 2 == 0))

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(run, i) for i in range(threads)]
            for future in as_completed(futures):
                future.result()

        snap = budget.snapshot()
        assert snap.trials == threads * per_thread
        # Half the threads emitted untracked trials.
        assert snap.untracked_trials == (threads // 2) * per_thread
        # With both tracked and untracked present, tracking is "partial".
        assert snap.cost_tracking == "partial"

    def test_concurrent_readers_never_raise(self) -> None:
        budget = ExecutionBudget(
            max_cost_usd=100.0, max_examples=1000, deadline_seconds=60.0
        )
        stop = threading.Event()
        errors: list[str] = []
        lock = threading.Lock()

        def reader() -> None:
            while not stop.is_set():
                try:
                    _ = budget.remaining_cost
                    _ = budget.remaining_examples
                    _ = budget.remaining_seconds
                    _ = budget.cost_tracking
                    _ = budget.exhausted_dimension
                    _ = budget.snapshot().as_dict()
                except Exception as exc:  # pragma: no cover - failure path
                    with lock:
                        errors.append(str(exc))
                    return

        def writer() -> None:
            for _ in range(500):
                budget.debit_trial(cost=0.001, examples=1)

        readers = [threading.Thread(target=reader) for _ in range(8)]
        for thread in readers:
            thread.start()
        writer_thread = threading.Thread(target=writer)
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for thread in readers:
            thread.join()

        assert errors == []
