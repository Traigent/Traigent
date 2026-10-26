"""Characterization tests for ``plan_resume`` classification and bookkeeping.

Pin how each prior trial is classified (reusable, unfinished, partial, other),
the counters on the plan, the order of reused trials, the ``max_trials`` cap and
the reuse annotations, so the planner can be restructured safely.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.resume import REUSED_TRIAL_KEY, plan_resume

_FPS = {
    "dataset": "fp1:d",
    "evaluator": "fp1:e",
    "agent": "fp1:a",
    "config_space": "fp1:c",
}


def _trial(
    tid: str,
    status: TrialStatus = TrialStatus.COMPLETED,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> TrialResult:
    trial = TrialResult(
        trial_id=tid,
        config=config if config is not None else {"v": tid},
        metrics=(
            metrics if metrics is not None else {"accuracy": 1.0, "total_cost": 0.5}
        ),
        status=status,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )
    if rows is not None:
        trial.metadata = {"example_results": rows}
    return trial


def _row(example_id: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "example_id": example_id,
        "success": True,
        "input_data": {"q": example_id},
        "expected_output": example_id,
        "actual_output": example_id,
        "metrics": {"accuracy": 1.0, "total_cost": 0.1},
    }
    row.update(overrides)
    return row


def _prior(trials: list[TrialResult], fps: dict[str, Any] | None = None):
    return OptimizationResult(
        trials=trials,
        best_config=None,
        best_score=None,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        metadata={"artifact_fingerprints": dict(_FPS if fps is None else fps)},
    )


def test_classification_counters_and_blocked_configs():
    trials = [
        _trial("ok1"),
        _trial("run", TrialStatus.RUNNING, config={"v": "run"}),
        _trial("pend", TrialStatus.PENDING, config={"v": "pend"}),
        _trial("notstarted", TrialStatus.NOT_STARTED),
        _trial("failed", TrialStatus.FAILED),
        _trial("cancelled", TrialStatus.CANCELLED),
        _trial("pruned", TrialStatus.PRUNED),
        _trial("nocost", metrics={"accuracy": 1.0}),
        _trial("ok2", metrics={"accuracy": 0.5, "cost": 0.25}),
    ]
    plan = plan_resume(_prior(trials), _FPS)

    assert [t.trial_id for t in plan.reusable] == ["resumed-ok1", "resumed-ok2"]
    assert plan.not_reusable_unfinished == 3
    assert plan.not_reusable_other == 4  # failed, cancelled, pruned, no cost
    assert plan.blocked_configs == ({"v": "run"}, {"v": "pend"}, {"v": "notstarted"})
    assert plan.partial_trials == 0
    assert plan.prefill is None
    assert plan.dropped_over_max_trials == 0
    assert plan.mismatched_components == ()
    assert plan.identity_unavailable is False


def test_reused_clone_annotations_and_cost_normalisation():
    original = _trial("ok2", metrics={"accuracy": 0.5, "cost": 0.25})
    original.metadata = {"keep": 1}
    plan = plan_resume(_prior([original]), _FPS)

    (clone,) = plan.reusable
    assert clone is not original
    assert clone.trial_id == "resumed-ok2"
    assert clone.metadata == {
        "keep": 1,
        REUSED_TRIAL_KEY: True,
        "resumed_original_trial_id": "ok2",
    }
    assert clone.metrics == {"accuracy": 0.5, "cost": 0.25, "total_cost": 0.25}
    # the source is untouched
    assert original.trial_id == "ok2"
    assert original.metadata == {"keep": 1}
    assert original.metrics == {"accuracy": 0.5, "cost": 0.25}


def test_unmeasured_cost_trial_is_reused_without_a_normalised_cost():
    trial = _trial("u", metrics={"accuracy": 1.0, "cost_unmeasured": 1.0})
    plan = plan_resume(_prior([trial]), _FPS)
    (clone,) = plan.reusable
    assert clone.metrics == {"accuracy": 1.0, "cost_unmeasured": 1.0}


def test_blocked_configs_are_deep_copies():
    running = _trial("run", TrialStatus.RUNNING, config={"nested": {"x": 1}})
    plan = plan_resume(_prior([running]), _FPS)
    plan.blocked_configs[0]["nested"]["x"] = 99
    assert running.config == {"nested": {"x": 1}}


def test_max_trials_cap_keeps_the_first_eligible_and_counts_dropped():
    trials = [_trial(f"t{i}") for i in range(5)]
    plan = plan_resume(_prior(trials), _FPS, max_trials=2)
    assert [t.trial_id for t in plan.reusable] == ["resumed-t0", "resumed-t1"]
    assert plan.dropped_over_max_trials == 3

    plan_all = plan_resume(_prior(trials), _FPS, max_trials=5)
    assert len(plan_all.reusable) == 5
    assert plan_all.dropped_over_max_trials == 0

    plan_zero = plan_resume(_prior(trials), _FPS, max_trials=0)
    assert plan_zero.reusable == ()
    assert plan_zero.dropped_over_max_trials == 5


def test_mismatch_returns_counters_but_no_reuse_and_no_blocked_or_partial():
    trials = [
        _trial("ok"),
        _trial("run", TrialStatus.RUNNING),
        _trial("failed", TrialStatus.FAILED),
        _trial("partial", rows=[_row("a")], config={"v": "partial"}),
    ]
    current = {**_FPS, "agent": "fp1:other"}
    plan = plan_resume(
        _prior(trials), current, max_trials=1, dataset_example_ids=["a", "b"]
    )

    assert plan.reusable == ()
    assert plan.blocked_configs == ()
    assert plan.partial_trials == 0
    assert plan.prefill is None
    assert plan.dropped_over_max_trials == 0
    assert plan.not_reusable_unfinished == 1
    assert plan.not_reusable_other == 1
    assert plan.mismatched_components == ("agent",)


def test_missing_current_identity_is_unavailable_and_reuses_nothing():
    plan = plan_resume(_prior([_trial("ok")]), None)
    assert plan.reusable == ()
    assert plan.identity_unavailable is True
    assert plan.mismatched_components != ()


def test_partial_trial_feeds_prefill_pool_and_is_not_counted_or_reused():
    partial = _trial(
        "p", config={"v": "p"}, rows=[_row("a"), _row("b", success=False), _row("z")]
    )
    full = _trial("f", config={"v": "f"}, rows=[_row("a"), _row("b")])
    plan = plan_resume(_prior([partial, full]), _FPS, dataset_example_ids=["a", "b"])

    assert plan.partial_trials == 1
    assert [t.trial_id for t in plan.reusable] == ["resumed-f"]
    assert plan.not_reusable_other == 0
    assert plan.prefill is not None
    assert plan.prefill.configs() == 1
    hit = plan.prefill.lookup({"v": "p"}, "a", {"q": "a"}, "a")
    assert hit is not None and hit["example_id"] == "a"
    # failed example and out-of-dataset example are not pooled
    assert plan.prefill.lookup({"v": "p"}, "b", {"q": "b"}, "b") is None
    assert plan.prefill.lookup({"v": "p"}, "z", {"q": "z"}, "z") is None
    # input/expected mismatch is a miss
    assert plan.prefill.lookup({"v": "p"}, "a", {"q": "other"}, "a") is None


def test_first_pooled_row_wins_for_the_same_config_and_example():
    first = _trial("p1", config={"v": "same"}, rows=[_row("a", actual_output="one")])
    second = _trial("p2", config={"v": "same"}, rows=[_row("a", actual_output="two")])
    plan = plan_resume(_prior([first, second]), _FPS, dataset_example_ids=["a", "b"])

    assert plan.partial_trials == 2
    assert plan.prefill is not None and plan.prefill.configs() == 1
    hit = plan.prefill.lookup({"v": "same"}, "a", {"q": "a"}, "a")
    assert hit is not None and hit["actual_output"] == "one"


def test_unmeasured_cost_partial_trial_flags_reused_rows():
    partial = _trial(
        "p",
        config={"v": "p"},
        metrics={"accuracy": 1.0, "cost_unmeasured": 1.0},
        rows=[_row("a")],
    )
    plan = plan_resume(_prior([partial]), _FPS, dataset_example_ids=["a", "b"])
    assert plan.prefill is not None
    hit = plan.prefill.lookup({"v": "p"}, "a", {"q": "a"}, "a")
    assert hit is not None
    assert hit["metrics"].get("cost_unmeasured") in (1, 1.0, True)
    assert hit["metrics"].get("total_cost") is None


def test_trial_without_rows_falls_back_to_trial_level_reuse():
    plan = plan_resume(_prior([_trial("ok")]), _FPS, dataset_example_ids=["a"])
    assert [t.trial_id for t in plan.reusable] == ["resumed-ok"]
    assert plan.partial_trials == 0


def test_no_dataset_ids_means_rows_are_not_inspected():
    partial = _trial("p", rows=[_row("a")])
    plan = plan_resume(_prior([partial]), _FPS)
    assert [t.trial_id for t in plan.reusable] == ["resumed-p"]
    assert plan.partial_trials == 0


def test_empty_prior_trials_gives_an_empty_plan():
    plan = plan_resume(_prior([]), _FPS)
    assert plan.reusable == ()
    assert plan.blocked_configs == ()
    assert plan.prefill is None


@pytest.mark.parametrize("status", [TrialStatus.FAILED, TrialStatus.CANCELLED])
def test_non_completed_finished_trials_count_as_other(status):
    plan = plan_resume(_prior([_trial("x", status)]), _FPS)
    assert plan.not_reusable_other == 1
    assert plan.not_reusable_unfinished == 0
