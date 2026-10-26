"""Characterization tests pinning comparison.aggregate_config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.core.comparison import aggregate_config


def _stats(total=1.0, unmeasured=0, lower=None):
    return SimpleNamespace(
        total_cost=total, cost_unmeasured_trials=unmeasured, cost_lower_bound=lower
    )


def _trial(score=0.5, metrics=None, ok=True, use_attr=True):
    t = SimpleNamespace(metrics=metrics, is_successful=ok)
    if use_attr:
        t.score = score
    return t


def _run(trial, stats=None):
    return SimpleNamespace(
        trials=[trial] if trial is not None else [],
        experiment_stats=stats or _stats(),
    )


def _agg(runs, primary="accuracy", cfg=None):
    return aggregate_config(cfg or {"m": 1}, runs, primary, ("e1",))


def test_empty_runs():
    row = _agg([])
    assert row.repeats == 0 and row.scores == []
    assert row.mean_score is None and row.score_stdev is None
    assert row.score_min is None and row.score_max is None
    assert row.metrics_mean == {} and row.trials == []
    assert row.total_cost == 0.0 and row.cost_lower_bound is None
    assert row.cost_unmeasured_repeats == 0 and row.failed_repeats == 0
    assert row.example_ids == ("e1",)


def test_basic_aggregation_and_config_copied():
    cfg = {"m": 1}
    runs = [
        _run(_trial(0.4, {"accuracy": 0.4, "lat": 10}), _stats(1.0)),
        _run(_trial(0.6, {"accuracy": 0.6, "lat": 20, "tag": "x"}), _stats(2.0)),
    ]
    row = _agg(runs, cfg=cfg)
    assert row.config == cfg and row.config is not cfg
    assert row.scores == [0.4, 0.6]
    assert row.mean_score == pytest.approx(0.5)
    assert row.score_stdev == pytest.approx(0.1414213562)
    assert (row.score_min, row.score_max) == (0.4, 0.6)
    assert row.metrics_mean == {
        "accuracy": pytest.approx(0.5),
        "lat": pytest.approx(15.0),
    }
    assert list(row.metrics_mean) == ["accuracy", "lat"]  # sorted
    assert row.total_cost == 3.0 and row.cost_lower_bound is None
    assert row.failed_repeats == 0 and row.cost_unmeasured_repeats == 0
    assert row.trials == [r.trials[0] for r in runs]


def test_missing_trial_counts_failed_and_unmeasured_without_stats_read():
    class _NoStats:
        trials: list = []

        @property
        def experiment_stats(self):
            raise AssertionError("stats must not be read for a missing trial")

    row = _agg([_NoStats(), _run(_trial(0.5, {}), _stats(1.5))])
    assert row.scores == [None, 0.5]
    assert row.failed_repeats == 1
    assert row.cost_unmeasured_repeats == 1
    assert row.total_cost is None
    assert row.cost_lower_bound == pytest.approx(1.5)
    assert len(row.trials) == 1


def test_score_falls_back_to_primary_metric_when_attr_none():
    row = _agg([_run(_trial(None, {"accuracy": 0.7}))])
    assert row.scores == [0.7]


def test_score_missing_attribute_uses_metric():
    row = _agg([_run(_trial(metrics={"accuracy": 0.8}, use_attr=False))])
    assert row.scores == [0.8]


@pytest.mark.parametrize("score", ["0.5", float("nan"), float("inf"), True], ids=str)
def test_non_numeric_score_counts_failed(score):
    row = _agg([_run(_trial(score, {}))])
    assert row.scores == [None] and row.failed_repeats == 1
    assert row.mean_score is None


def test_unsuccessful_trial_counts_failed_but_metrics_still_collected():
    row = _agg([_run(_trial(0.9, {"lat": 5.0}, ok=False))])
    assert row.scores == [None] and row.failed_repeats == 1
    assert row.metrics_mean == {"lat": 5.0}
    assert len(row.trials) == 1


def test_none_metrics_tolerated():
    row = _agg([_run(_trial(0.3, None))])
    assert row.scores == [0.3] and row.metrics_mean == {}


def test_non_finite_and_bool_metric_values_ignored():
    row = _agg([_run(_trial(0.3, {"a": float("nan"), "b": True, "c": 2, "d": "x"}))])
    assert row.metrics_mean == {"c": 2.0}


def test_cost_unmeasured_by_none_total_with_lower_bound_floor():
    runs = [
        _run(_trial(0.5, {}), _stats(2.0)),
        _run(_trial(0.5, {}), _stats(None, lower=0.75)),
    ]
    row = _agg(runs)
    assert row.total_cost is None
    assert row.cost_lower_bound == pytest.approx(2.75)
    assert row.cost_unmeasured_repeats == 1


def test_cost_unmeasured_by_flag_ignores_total_and_zero_floor():
    runs = [_run(_trial(0.5, {}), _stats(9.0, unmeasured=1, lower=None))]
    row = _agg(runs)
    assert row.total_cost is None
    assert row.cost_lower_bound == 0.0
    assert row.cost_unmeasured_repeats == 1


def test_single_valid_score_has_no_stdev():
    row = _agg([_run(_trial(0.5, {}))])
    assert row.score_stdev is None and row.mean_score == 0.5


def test_result_with_trials_attribute_none():
    r = SimpleNamespace(trials=None, experiment_stats=_stats())
    row = _agg([r])
    assert row.scores == [None] and row.failed_repeats == 1
