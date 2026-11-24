"""#3219: plateau scores trials with observed-range normalization (multi-objective)."""

from __future__ import annotations

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.stop_conditions import PlateauAfterNStopCondition


def _trial(i: int, metrics: dict[str, float]) -> TrialResult:
    return TrialResult(
        trial_id=f"t{i}",
        config={"p": i},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=None,
    )


def _two_objective_schema() -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("score", "maximize", 1.0),
            ObjectiveDefinition("latency", "minimize", 1.0),
        ]
    )


def _decisions(cond: PlateauAfterNStopCondition, rows: list[dict[str, float]]):
    out, history = [], []
    for i, metrics in enumerate(rows):
        history.append(_trial(i, metrics))
        out.append(cond.should_stop(history))
    return out


def test_negative_maximize_values_do_not_plateau_while_improving():
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.0, objective_schema=_two_objective_schema()
    )
    rows = [{"score": v, "latency": 1.0} for v in (-3.0, -2.0, -1.0, -0.5)]
    # Before the fix: [False, True, True, True] (every value clipped to 0).
    assert _decisions(cond, rows) == [False, False, False, False]


def test_seconds_and_milliseconds_give_the_same_decisions():
    def run(scale: float):
        cond = PlateauAfterNStopCondition(
            window_size=2, epsilon=0.01, objective_schema=_two_objective_schema()
        )
        rows = [{"score": 0.5, "latency": lat * scale} for lat in (3.0, 1.0, 0.1, 0.01)]
        return _decisions(cond, rows)

    assert run(1.0) == run(1000.0)


def test_true_plateau_still_stops():
    cond = PlateauAfterNStopCondition(
        window_size=3, epsilon=1e-9, objective_schema=_two_objective_schema()
    )
    rows = [
        {"score": 0.2, "latency": 3.0},
        {"score": 0.9, "latency": 1.0},
        {"score": 0.5, "latency": 2.0},
        {"score": 0.4, "latency": 2.5},
        {"score": 0.3, "latency": 2.9},
    ]
    assert _decisions(cond, rows) == [False, False, False, True, True]


def test_ranges_widening_rescores_earlier_trials():
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.0, objective_schema=_two_objective_schema()
    )
    rows = [
        {"score": 0.5, "latency": 1.0},
        {"score": 0.5, "latency": 1.0},  # zero-width ranges: tied, plateau
        {"score": 0.9, "latency": 1.0},  # range widens; new best
    ]
    assert _decisions(cond, rows) == [False, True, False]


def test_zero_width_ranges_do_not_crash():
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.0, objective_schema=_two_objective_schema()
    )
    rows = [{"score": 1.0, "latency": 1.0}] * 3
    assert _decisions(cond, rows) == [False, True, True]


def test_incremental_equals_full_recompute():
    rows = [
        {"score": s, "latency": lat}
        for s, lat in [(-1, 5), (0, 4), (-0.5, 4), (2, 1), (2, 1), (2, 1), (1.5, 3)]
    ]
    inc = PlateauAfterNStopCondition(
        window_size=3, epsilon=0.0, objective_schema=_two_objective_schema()
    )
    got = _decisions(inc, rows)
    want = []
    for n in range(1, len(rows) + 1):
        fresh = PlateauAfterNStopCondition(
            window_size=3, epsilon=0.0, objective_schema=_two_objective_schema()
        )
        want.append(fresh.should_stop([_trial(i, m) for i, m in enumerate(rows[:n])]))
    assert got == want


def test_reset_clears_ranged_state():
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.0, objective_schema=_two_objective_schema()
    )
    rows = [{"score": 1.0, "latency": 1.0}] * 2
    assert _decisions(cond, rows) == [False, True]
    cond.reset()
    assert _decisions(cond, rows) == [False, True]


def test_single_objective_control_is_unchanged():
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition("accuracy", "maximize", 1.0)]
    )
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.01, objective_schema=schema
    )
    improving = [{"accuracy": 0.1 * (i + 1)} for i in range(4)]
    assert _decisions(cond, improving) == [False, False, False, False]
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.01, objective_schema=schema
    )
    flat = [{"accuracy": 0.5}] * 3
    assert _decisions(cond, flat) == [False, True, True]
    # range-free path keeps raw units: epsilon is compared to the raw score gap
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.5, objective_schema=schema
    )
    assert _decisions(cond, [{"accuracy": 0.1}, {"accuracy": 0.4}]) == [False, True]


def test_unscorable_trial_does_not_cause_premature_stop():
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("a", "maximize", 1.0),
            ObjectiveDefinition("b", "maximize", 1.0),
        ]
    )
    rows = [{}, {"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}]
    inc = PlateauAfterNStopCondition(
        window_size=3, epsilon=0.0, objective_schema=schema
    )
    got = _decisions(inc, rows)
    fresh = PlateauAfterNStopCondition(
        window_size=3, epsilon=0.0, objective_schema=schema
    )
    assert (
        got[-1]
        == fresh.should_stop([_trial(i, m) for i, m in enumerate(rows)])
        is False
    )


def test_non_finite_metric_is_skipped_not_scored_as_best():
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("a", "maximize", 1.0),
            ObjectiveDefinition("b", "maximize", 1.0),
        ]
    )
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.0, objective_schema=schema
    )
    rows = [{"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 2.0}, {"a": float("nan"), "b": 1.5}]
    # NaN is skipped (not clipped to a best score), so trial 3 is no new best
    assert _decisions(cond, rows) == [False, False, True]
    assert max(cond._ranged_scores) == cond._ranged_scores[-1]


def test_unmeasured_cost_is_not_scored_as_cheap():
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("score", "maximize", 1.0),
            ObjectiveDefinition("cost", "minimize", 1.0),
        ]
    )
    cond = PlateauAfterNStopCondition(
        window_size=2, epsilon=0.0, objective_schema=schema
    )
    rows = [
        {"score": 0.5, "cost": 1.0},
        {"score": 0.5, "cost": 2.0},
        {"score": 0.5, "cost": 0.0, "cost_unmeasured": True},
    ]
    # the flagged zero cost is dropped, so the third trial is not a cheap new best
    assert _decisions(cond, rows) == [False, True, True]
