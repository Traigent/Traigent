"""Characterization tests pinning OptimizationResult.best_metrics."""

from __future__ import annotations

from datetime import datetime

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)

CFG = {"m": 1}
OTHER = {"m": 2}


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


def _result(
    trials, best_config=CFG, best_score=0.5, objectives=("acc",), metadata=None
):
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
        metadata=metadata if metadata is not None else {},
    )


def test_no_winner_returns_empty():
    r = _result([_trial("a", {"acc": 1.0})], best_config={}, best_score=None)
    assert r.best_metrics == {}


def test_no_trials_returns_empty():
    assert _result([]).best_metrics == {}


def test_no_objectives_prefers_successful_config_match():
    r = _result(
        [_trial("a", {"acc": 0.1}, OTHER), _trial("b", {"acc": 0.9}, CFG)],
        objectives=(),
    )
    assert r.best_metrics == {"acc": 0.9}


def test_no_objectives_skips_failed_config_match_then_first_with_metrics():
    r = _result(
        [
            _trial("a", {}, OTHER),
            _trial("b", {"acc": 0.2}, OTHER),
            _trial("c", {"acc": 0.9}, CFG, ok=False),
        ],
        objectives=(),
    )
    assert r.best_metrics == {"acc": 0.2}


def test_no_objectives_no_metrics_anywhere_returns_empty():
    r = _result([_trial("a", {}, OTHER)], objectives=())
    assert r.best_metrics == {}


def test_no_objectives_empty_config_valid_winner_takes_first_with_metrics():
    r = _result(
        [_trial("a", {"acc": 0.3}, OTHER)],
        best_config={},
        best_score=0.0,
        objectives=(),
    )
    assert r.best_metrics == {"acc": 0.3}


def test_objectives_config_match_ignores_success_flag_and_returns_copy():
    t = _trial("a", {"acc": 0.9}, CFG, ok=False)
    r = _result([_trial("z", {"acc": 0.1}, OTHER), t])
    out = r.best_metrics
    assert out == {"acc": 0.9}
    assert out is not t.metrics


def test_objectives_falls_back_to_best_trial_id():
    r = _result(
        [_trial("a", {"acc": 0.1}, OTHER), _trial("b", {"acc": 0.7}, {"m": 3})],
        metadata={"best_trial_id": "b"},
    )
    assert r.best_metrics == {"acc": 0.7}


def test_objectives_last_resort_first_successful_with_metrics():
    r = _result(
        [
            _trial("a", {"acc": 0.1}, OTHER, ok=False),
            _trial("b", {"acc": 0.4}, {"m": 3}),
        ],
    )
    assert r.best_metrics == {"acc": 0.4}


def test_objectives_nothing_matches_returns_empty():
    r = _result([_trial("a", {"acc": 0.1}, OTHER, ok=False)])
    assert r.best_metrics == {}


def test_odd_none_best_config_with_score_and_none_metadata():
    r = _result([_trial("a", {"acc": 0.4}, OTHER)], best_config=None, best_score=0.0)
    r.metadata = None
    assert r.best_metrics == {"acc": 0.4}
    r2 = _result([_trial("a", {"acc": 0.4}, OTHER)], best_config={}, best_score=0.0)
    r2.metadata = None
    assert r2.best_metrics == {"acc": 0.4}


def test_odd_none_trial_metrics_is_skipped():
    r = _result([_trial("a", None, CFG), _trial("b", {"acc": 0.2}, CFG)])
    assert r.best_metrics == {"acc": 0.2}
