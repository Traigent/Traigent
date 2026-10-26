"""Characterization tests pinning ParetoFrontCalculator.calculate_pareto_front."""

from __future__ import annotations

from types import SimpleNamespace

from traigent.utils.multi_objective import ParetoFrontCalculator


def _trial(metrics, status="completed", config=None):
    return SimpleNamespace(
        status=status, metrics=metrics, config=config or {"id": id(metrics)}
    )


def _front(trials, objectives, maximize=None):
    calc = ParetoFrontCalculator(maximize=maximize or {})
    return calc.calculate_pareto_front(trials, objectives)


def test_no_trials_returns_empty():
    assert _front([], ["accuracy"]) == []


def test_non_completed_and_empty_metrics_are_skipped():
    trials = [
        _trial({"accuracy": 0.9}, status="failed"),
        _trial({}, status="completed"),
        _trial(None, status="completed"),
        SimpleNamespace(metrics={"accuracy": 0.8}, config={}),  # no status attr
    ]
    assert _front(trials, ["accuracy"]) == []


def test_single_objective_picks_max_and_keeps_ties():
    a, b, c = _trial({"accuracy": 0.5}), _trial({"accuracy": 0.9}), _trial({"accuracy": 0.9})
    front = _front([a, b, c], ["accuracy"], {"accuracy": True})
    assert [p.trial for p in front] == [b, c]
    assert front[0].objectives == {"accuracy": 0.9}
    assert front[0].config is b.config


def test_two_objectives_non_dominated_in_input_order():
    a = _trial({"accuracy": 0.9, "cost": 5.0})
    b = _trial({"accuracy": 0.7, "cost": 1.0})
    c = _trial({"accuracy": 0.6, "cost": 6.0})  # dominated by a and b
    front = _front([a, b, c], ["accuracy", "cost"], {"accuracy": True, "cost": False})
    assert [p.trial for p in front] == [a, b]


def test_trial_missing_a_configured_objective_is_excluded():
    a = _trial({"accuracy": 0.9})
    b = _trial({"accuracy": 0.1, "cost": 1.0})
    front = _front([a, b], ["accuracy", "cost"], {"accuracy": True, "cost": False})
    assert [p.trial for p in front] == [b]


def test_none_metric_value_counts_as_missing():
    a = _trial({"accuracy": None, "cost": 1.0})
    assert _front([a], ["accuracy", "cost"]) == []


def test_cost_flag_drops_cost_objectives_so_trial_excluded():
    flagged = _trial({"accuracy": 0.99, "cost": 0.0, "cost_unmeasured": 1.0})
    ok = _trial({"accuracy": 0.5, "cost": 2.0})
    front = _front([flagged, ok], ["accuracy", "cost"], {"accuracy": True, "cost": False})
    assert [p.trial for p in front] == [ok]


def test_cost_flag_irrelevant_for_non_cost_objectives():
    flagged = _trial({"accuracy": 0.99, "cost_unpriced": 1.0})
    front = _front([flagged], ["accuracy"])
    assert [p.trial for p in front] == [flagged]


def test_duplicate_objective_names_use_unique_count():
    a = _trial({"accuracy": 0.9})
    front = _front([a], ["accuracy", "accuracy"])
    assert [p.trial for p in front] == [a]


def test_objectives_subset_ignores_other_metrics():
    a = _trial({"accuracy": 0.9, "latency": 100})
    front = _front([a], ["accuracy"])
    assert front[0].objectives == {"accuracy": 0.9}


def test_enum_status_is_recognised():
    from traigent.api.types import TrialStatus

    a = _trial({"accuracy": 0.9}, status=TrialStatus.COMPLETED)
    b = _trial({"accuracy": 0.99}, status=TrialStatus.FAILED)
    assert [p.trial for p in _front([a, b], ["accuracy"])] == [a]
