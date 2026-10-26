"""Characterization tests pinning OptimizationResult._aggregated_mean_metrics."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)

CFG = {"m": 1}


def _trial(tid, metrics, config=None, ok=True):
    return TrialResult(
        trial_id=tid,
        config=config if config is not None else CFG,
        metrics=metrics,
        status=TrialStatus.COMPLETED if ok else TrialStatus.FAILED,
        duration=0.1,
        timestamp=datetime.now(),
        metadata={"successful_examples": 5 if ok else 0, "examples_attempted": 5},
    )


def _result(trials, summary=None, best_config=CFG, best_score=None, objectives=("acc",)):
    metadata: Any = {"session_summary": summary} if summary is not None else {}
    return OptimizationResult(
        trials=trials,
        best_config=best_config,
        best_score=best_score,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=list(objectives),
        algorithm="random",
        timestamp=datetime.now(),
        metadata=metadata,
    )


AGG = {"selection_mode": "aggregated_mean"}


def test_not_aggregated_returns_none():
    r = _result([_trial("a", {"acc": 1.0})], summary={})
    assert r._aggregated_mean_metrics() is None


def test_no_best_config_returns_none():
    r = _result([_trial("a", {"acc": 1.0})], summary=AGG, best_config={})
    assert r._aggregated_mean_metrics() is None


def test_means_by_config_fallback_with_primary_pinned_to_best_score():
    ts = [
        _trial("a", {"acc": 0.8, "cost": 1.0}),
        _trial("b", {"acc": 0.9, "cost": 3.0}),
        _trial("c", {"acc": 0.1, "cost": 100.0}, config={"m": 2}),
    ]
    r = _result(ts, summary=AGG, best_score=0.123)
    got = r._aggregated_mean_metrics()
    assert got == {"acc": 0.123, "cost": pytest.approx(2.0)}


def test_primary_not_pinned_without_best_score_or_objectives():
    ts = [_trial("a", {"acc": 0.8}), _trial("b", {"acc": 0.9})]
    assert _result(ts, summary=AGG)._aggregated_mean_metrics() == {
        "acc": pytest.approx(0.85)
    }
    assert _result(
        ts, summary=AGG, best_score=0.5, objectives=()
    )._aggregated_mean_metrics() == {"acc": pytest.approx(0.85)}
    assert _result(
        ts, summary=AGG, best_score=0.5, objectives=("other",)
    )._aggregated_mean_metrics() == {"acc": pytest.approx(0.85)}


def test_excludes_bool_examples_attempted_and_non_numeric():
    ts = [_trial("a", {"acc": 1.0, "flag": True, "examples_attempted": 9, "s": "x"})]
    assert _result(ts, summary=AGG)._aggregated_mean_metrics() == {"acc": 1.0}


def test_only_non_numeric_metrics_returns_none():
    ts = [_trial("a", {"flag": True, "s": "x"})]
    assert _result(ts, summary=AGG)._aggregated_mean_metrics() is None


def test_unsuccessful_and_metricless_trials_excluded_and_empty_returns_none():
    ts = [_trial("a", {"acc": 1.0}, ok=False), _trial("b", {})]
    assert _result(ts, summary=AGG)._aggregated_mean_metrics() is None


def test_stamped_winning_ids_take_precedence_over_config_equality():
    ts = [
        _trial("a", {"acc": 0.2}, config={"m": 1, "aux": 1}),
        _trial("b", {"acc": 0.4}, config={"m": 1, "aux": 2}),
        _trial("c", {"acc": 0.9}, config=CFG),
    ]
    summary = {**AGG, "winning_trial_ids": ["a", "b"]}
    got = _result(ts, summary=summary)._aggregated_mean_metrics()
    assert got == {"acc": pytest.approx(0.3)}


def test_stamped_ids_matching_nothing_returns_none_without_config_fallback():
    ts = [_trial("a", {"acc": 0.2})]
    summary = {**AGG, "winning_trial_ids": ["zzz"]}
    assert _result(ts, summary=summary)._aggregated_mean_metrics() is None


@pytest.mark.parametrize("stamped", [[], None, "ab", {"a": 1}])
def test_empty_or_invalid_stamp_falls_back_to_config_equality(stamped):
    ts = [_trial("a", {"acc": 0.2}), _trial("b", {"acc": 0.4})]
    summary = {**AGG, "winning_trial_ids": stamped}
    got = _result(ts, summary=summary)._aggregated_mean_metrics()
    assert got == {"acc": pytest.approx(0.3)}


def test_numeric_trial_ids_are_stringified_for_match():
    ts = [_trial(7, {"acc": 0.5}), _trial(8, {"acc": 0.9})]
    summary = {**AGG, "winning_trial_ids": [7]}
    assert _result(ts, summary=summary)._aggregated_mean_metrics() == {"acc": 0.5}


def test_samples_per_config_alone_marks_aggregated():
    r = _result([_trial("a", {"acc": 0.5})], summary={"samples_per_config": {"x": 1}})
    assert r._aggregated_mean_metrics() == {"acc": 0.5}


def test_metric_missing_in_some_replicates_averaged_over_its_own_count():
    ts = [_trial("a", {"acc": 0.2, "x": 4.0}), _trial("b", {"acc": 0.4})]
    got = _result(ts, summary=AGG)._aggregated_mean_metrics()
    assert got == {"acc": pytest.approx(0.3), "x": 4.0}


def test_non_dict_metadata_is_not_aggregated():
    r = _result([_trial("a", {"acc": 0.5})], summary=AGG)
    r.metadata = "junk"
    assert r._aggregated_mean_metrics() is None
