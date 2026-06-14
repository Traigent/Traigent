"""Thread-safety regression tests for the Optuna-backed optimizer.

Regression for issue #1266: the optimizer shares a single Optuna study across
the orchestrator's worker threads. Before the per-study lock, concurrent
``suggest_next_trial`` / ``report_trial_result`` calls mutated the study's
in-memory storage (and the optimizer's own ``_active_trials`` /
``_pending_configs`` dicts) while another thread iterated them, which could
abort a worker with a native (pyo3) ``"dictionary changed size during
iteration"`` panic and leave the run deadlocked.

These tests hammer the optimizer from many threads and assert that no thread
raises and that the bookkeeping stays internally consistent.
"""

from __future__ import annotations

import threading

import pytest

from traigent.optimizers.optuna_optimizer import OptunaRandomOptimizer
from traigent.utils.exceptions import OptimizationError

# A wide-but-finite categorical space so many distinct trials are possible.
CONFIG_SPACE = {
    "model": ["a", "b", "c", "d"],
    "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
    "top_p": [0.1, 0.5, 0.9],
}


def test_concurrent_ask_tell_does_not_raise_or_corrupt_state():
    """Concurrent suggest/report from many threads must not raise.

    Without the per-study lock this reliably trips Optuna's non-thread-safe
    in-memory storage and/or the optimizer's unguarded dict mutations.
    """
    optimizer = OptunaRandomOptimizer(
        CONFIG_SPACE,
        ["accuracy"],
        max_trials=500,
    )

    errors: list[BaseException] = []
    completed = 0
    completed_lock = threading.Lock()

    def worker() -> None:
        nonlocal completed
        for _ in range(40):
            try:
                config = optimizer.suggest_next_trial([])
            except OptimizationError:
                # max_trials reached — an expected, ordered stop, not a race.
                return
            except BaseException as exc:  # noqa: BLE001 - capture pyo3 panics too
                errors.append(exc)
                return

            trial_id = config["_optuna_trial_id"]
            try:
                optimizer.report_trial_result(trial_id, 0.5)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return
            with completed_lock:
                completed += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(t.is_alive() for t in threads), "a worker thread deadlocked"
    assert not errors, f"concurrent ask/tell raised: {errors!r}"

    # Every trial that was suggested was also reported, so nothing is left
    # dangling in the active-trial bookkeeping.
    assert optimizer._active_trials == {}
    assert len(optimizer._synced_trials) == completed


def test_study_lock_is_reentrant():
    """``report_trial_result`` dispatches to ``report_trial_pruned`` while
    holding the study lock; a non-reentrant lock would self-deadlock."""
    optimizer = OptunaRandomOptimizer(CONFIG_SPACE, ["accuracy"], max_trials=4)

    config = optimizer.suggest_next_trial([])
    trial_id = config["_optuna_trial_id"]

    # objectives=None + state="pruned" forces the nested, lock-reentrant path.
    optimizer.report_trial_result(
        trial_id, None, metadata={"state": "pruned", "step": 0}
    )

    assert trial_id not in optimizer._active_trials


def test_optimizer_has_a_reentrant_study_lock():
    """Lock the contract in place so the guard isn't silently removed."""
    optimizer = OptunaRandomOptimizer(CONFIG_SPACE, ["accuracy"], max_trials=1)
    # threading.RLock() returns an instance of a private type exposing acquire.
    assert hasattr(optimizer, "_study_lock")
    assert optimizer._study_lock.acquire(blocking=False)
    # Reentrant: a second acquire on the same thread must also succeed.
    assert optimizer._study_lock.acquire(blocking=False)
    optimizer._study_lock.release()
    optimizer._study_lock.release()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
