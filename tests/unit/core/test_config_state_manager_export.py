"""Tests for ConfigStateManager._create_full_export — contract field direct access.

Regression (#1492): _create_full_export must use t.status directly (a
TrialStatus StrEnum — no hasattr/"completed" fallback) and iterate
self._optimization_results.trials directly (no dead `or []` guard).
"""
# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

from datetime import UTC, datetime

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.config_state_manager import ConfigStateManager


def _minimal_manager(opt_result: OptimizationResult) -> ConfigStateManager:
    """Construct a bare ConfigStateManager without calling __init__.

    Only the attributes required by _create_full_export are set.
    """
    mgr: ConfigStateManager = ConfigStateManager.__new__(ConfigStateManager)
    mgr._optimization_results = opt_result
    mgr._best_config = opt_result.best_config
    mgr.func = lambda x: x
    mgr.func.__name__ = "test_fn"  # type: ignore[method-assign]
    mgr.configuration_space = None
    return mgr


def _make_trial(
    tid: str,
    status: TrialStatus,
    score: float,
) -> TrialResult:
    return TrialResult(
        trial_id=tid,
        config={"model": "gpt-4o"},
        metrics={"accuracy": score},
        status=status,
        duration=1.5,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_opt_result(*trials: TrialResult) -> OptimizationResult:
    return OptimizationResult(
        trials=list(trials),
        best_config={"model": "gpt-4o"},
        best_score=0.9,
        optimization_id="opt-test",
        duration=5.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="random",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )


class TestCreateFullExportContractFields:
    """_create_full_export reads contract-guaranteed TrialResult fields directly."""

    def test_export_uses_real_status_not_completed_fallback(self) -> None:
        """t.status is used directly — a FAILED trial must export TrialStatus.FAILED,
        not the removed hasattr/"completed" fallback value."""
        trials = [
            _make_trial("t1", TrialStatus.COMPLETED, 0.9),
            _make_trial("t2", TrialStatus.FAILED, 0.0),
            _make_trial("t3", TrialStatus.CANCELLED, 0.5),
        ]
        mgr = _minimal_manager(_make_opt_result(*trials))

        export = mgr._create_full_export(include_metadata=False)

        statuses = [t["status"] for t in export["trials"]]
        # Direct t.status access: TrialStatus is a StrEnum so it compares equal
        # to its string value, but must NOT be the hard-coded "completed" string.
        assert statuses[0] == TrialStatus.COMPLETED  # "completed"
        assert statuses[1] == TrialStatus.FAILED  # "failed" — key regression
        assert statuses[2] == TrialStatus.CANCELLED  # "cancelled"
        # The former bug: without direct access, any status that failed hasattr
        # would have been silently emitted as "completed".
        assert statuses[1] != "completed", (
            "FAILED trial must not be exported as 'completed' (bug-masking fallback removed)"
        )

    def test_export_includes_all_trials_from_contract_list(self) -> None:
        """Every trial in OptimizationResult.trials is exported; the removed
        `or []` guard cannot silently truncate the list."""
        trials = [
            _make_trial(f"t{i}", TrialStatus.COMPLETED, float(i) * 0.1)
            for i in range(5)
        ]
        mgr = _minimal_manager(_make_opt_result(*trials))

        export = mgr._create_full_export(include_metadata=False)

        assert len(export["trials"]) == 5
        exported_ids = [t["trial_id"] for t in export["trials"]]
        assert exported_ids == [f"t{i}" for i in range(5)]

    def test_export_total_trials_consistent_with_trial_list_length(self) -> None:
        """optimization.total_trials must equal len(export["trials"]).

        The former `or []` guard at :1002 was inconsistent with the sibling
        unguarded len() at :1006; both must see the same list.
        """
        trials = [_make_trial(f"t{i}", TrialStatus.COMPLETED, 0.8) for i in range(3)]
        mgr = _minimal_manager(_make_opt_result(*trials))

        export = mgr._create_full_export(include_metadata=False)

        assert export["optimization"]["total_trials"] == 3
        assert export["optimization"]["total_trials"] == len(export["trials"])

    def test_export_with_empty_trials_list_produces_empty_trials_key(self) -> None:
        """An OptimizationResult with zero trials produces an empty list — not a
        crash (the `or []` removal is safe since trials is always a list)."""
        mgr = _minimal_manager(_make_opt_result())  # empty trials

        export = mgr._create_full_export(include_metadata=False)

        assert export["trials"] == []
        assert export["optimization"]["total_trials"] == 0
