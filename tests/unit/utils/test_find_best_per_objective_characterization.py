"""Characterization tests pinning results_table._find_best_per_objective."""

from __future__ import annotations

from types import SimpleNamespace

from traigent.utils.results_table import _find_best_per_objective


def _t(metrics, ok=True, metadata=None):
    return SimpleNamespace(metrics=metrics, is_successful=ok, metadata=metadata or {})


def test_empty_inputs():
    assert _find_best_per_objective([], []) == {}
    assert _find_best_per_objective([], [("accuracy", "maximize")]) == {
        "accuracy": set()
    }


def test_band_objective_always_empty():
    trials = [_t({"lat": 1.0})]
    assert _find_best_per_objective(trials, [("lat", "band")]) == {"lat": set()}


def test_maximize_and_minimize():
    trials = [
        _t({"accuracy": 0.5, "cost": 3.0}),
        _t({"accuracy": 0.9, "cost": 1.0}),
        _t({"accuracy": 0.7, "cost": 2.0}),
    ]
    out = _find_best_per_objective(
        trials, [("accuracy", "maximize"), ("cost", "minimize")]
    )
    assert out == {"accuracy": {1}, "cost": {1}}


def test_ties_within_tolerance_are_all_kept():
    trials = [
        _t({"accuracy": 0.9}),
        _t({"accuracy": 0.9 + 1e-13}),
        _t({"accuracy": 0.1}),
        _t({"accuracy": 0.9}),
    ]
    assert _find_best_per_objective(trials, [("accuracy", "maximize")]) == {
        "accuracy": {0, 1, 3}
    }


def test_later_better_value_resets_set():
    trials = [_t({"a": 1.0}), _t({"a": 1.0}), _t({"a": 2.0})]
    assert _find_best_per_objective(trials, [("a", "maximize")]) == {"a": {2}}
    assert _find_best_per_objective(trials, [("a", "minimize")]) == {"a": {0, 1}}


def test_unsuccessful_and_zero_success_trials_skipped():
    trials = [
        _t({"a": 100.0}, ok=False),
        _t({"a": 50.0}, metadata={"successful_examples": 0, "examples_attempted": 3}),
        _t({"a": 1.0}),
    ]
    assert _find_best_per_objective(trials, [("a", "maximize")]) == {"a": {2}}


def test_missing_none_bool_and_non_numeric_values_skipped():
    trials = [
        _t({}),
        _t({"a": None}),
        _t({"a": True}),
        _t({"a": "abc"}),
        _t({"a": "2.5"}),
    ]
    assert _find_best_per_objective(trials, [("a", "maximize")]) == {"a": {4}}


def test_no_candidates_gives_empty_set():
    assert _find_best_per_objective([_t({"a": 1}, ok=False)], [("a", "maximize")]) == {
        "a": set()
    }


def test_unmeasured_cost_trial_skipped_for_cost_metric_only():
    flagged = _t({"cost": 0.0, "accuracy": 0.99, "cost_unmeasured": 1.0})
    ok = _t({"cost": 5.0, "accuracy": 0.1})
    out = _find_best_per_objective(
        [flagged, ok], [("cost", "minimize"), ("accuracy", "maximize")]
    )
    assert out == {"cost": {1}, "accuracy": {0}}


def test_metric_overrides_take_precedence_when_index_in_range():
    trials = [_t({"a": 1.0}), _t({"a": 2.0}), _t({"a": 3.0})]
    out = _find_best_per_objective(
        trials, [("a", "maximize")], metric_overrides={"a": [10.0, 5.0]}
    )
    # index 2 beyond override -> falls back to trial metric 3.0
    assert out == {"a": {0}}


def test_override_for_other_metric_ignored():
    trials = [_t({"a": 1.0}), _t({"a": 2.0})]
    out = _find_best_per_objective(
        trials, [("a", "maximize")], metric_overrides={"b": [9.0, 0.0]}
    )
    assert out == {"a": {1}}
