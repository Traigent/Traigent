"""Regression tests for issue #2952: no dataframe column may silently overwrite another.

Tuned variables named like an identity column, a metric or a derived column are
exported as ``config.<name>``; metrics named like an identity column are exported
as ``metric.<name>``. Non-colliding output is unchanged.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)

TS = datetime(2026, 1, 2, 3, 4, 5)


def _trial(trial_id: str, config: dict, metrics: dict, metadata=None) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=1.5,
        timestamp=TS,
        metadata=metadata or {},
    )


def _result(trials: list[TrialResult]) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config={},
        best_score=0.0,
        optimization_id="opt",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["score"],
        algorithm="grid",
        timestamp=TS,
    )


def test_config_named_like_identity_and_metric_columns_is_preserved() -> None:
    df = _result(
        [
            _trial("t1", {"status": "a", "score": 0.3, "cost": 10}, {"score": 1.0}),
            _trial("t2", {"status": "b", "score": 0.4, "cost": 20}, {"score": 1.0}),
        ]
    ).to_dataframe()
    assert list(df["status"]) == ["completed", "completed"]
    assert list(df["config.status"]) == ["a", "b"]
    assert list(df["score"]) == [1.0, 1.0]
    assert list(df["config.score"]) == [0.3, 0.4]
    assert list(df["cost"]) == [10, 20]  # non-colliding name unchanged


def test_config_named_like_derived_column_is_preserved() -> None:
    df = _result(
        [_trial("t1", {"total_cost": 7}, {"acc": 0.5}, {"total_example_cost": 0.01})]
    ).to_dataframe()
    assert df["config.total_cost"].iloc[0] == 7
    assert df["total_cost"].iloc[0] == 0.01


def test_metric_named_like_identity_column_is_preserved() -> None:
    df = _result([_trial("t1", {"m": 1}, {"duration": 99.0})]).to_dataframe()
    assert df["duration"].iloc[0] == 1.5
    assert df["metric.duration"].iloc[0] == 99.0


def test_prefixed_name_is_unique_when_already_taken() -> None:
    df = _result(
        [_trial("t1", {"status": "a", "config.status": "x"}, {"acc": 1.0})]
    ).to_dataframe()
    assert df["config.status"].iloc[0] == "x"
    assert df["config.status_"].iloc[0] == "a"
    assert df["status"].iloc[0] == "completed"


def test_collision_decision_is_consistent_across_rows() -> None:
    # "score" is a metric only in trial 2; the config column must still be
    # renamed in trial 1 so a column never changes meaning between rows.
    df = _result(
        [_trial("t1", {"score": 0.1}, {}), _trial("t2", {"score": 0.2}, {"score": 5})]
    ).to_dataframe()
    assert list(df["config.score"]) == [0.1, 0.2]


def test_non_colliding_output_is_unchanged() -> None:
    df = _result(
        [_trial("t1", {"model": "m", "temp": 0.1}, {"accuracy": 0.5})]
    ).to_dataframe()
    assert list(df.columns) == [
        "trial_id",
        "status",
        "duration",
        "timestamp",
        "model",
        "temp",
        "accuracy",
        "avg_response_time",
        "avg_response_time_ms",
        "total_cost",
    ]
    assert df.iloc[0]["model"] == "m"
    assert df.iloc[0]["accuracy"] == 0.5


def test_aggregated_dataframe_does_not_overwrite_columns() -> None:
    df = _result(
        [
            _trial("t1", {"samples_count": 9, "accuracy": "cfg"}, {"accuracy": 0.5}),
            _trial("t2", {"samples_count": 9, "accuracy": "cfg"}, {"accuracy": 0.7}),
        ]
    ).to_aggregated_dataframe()
    assert df["samples_count"].iloc[0] == 2
    assert df["config.samples_count"].iloc[0] == 9
    assert df["accuracy"].iloc[0] == 0.6
    assert df["config.accuracy"].iloc[0] == "cfg"


def test_aggregated_non_colliding_columns_unchanged() -> None:
    df = _result(
        [_trial("t1", {"model": "m"}, {"accuracy": 0.5})]
    ).to_aggregated_dataframe()
    assert list(df.columns) == [
        "config_hash",
        "samples_count",
        "model",
        "accuracy",
        "duration",
    ]
    assert isinstance(df, pd.DataFrame)


def test_renamed_metric_never_lands_on_an_existing_config_column() -> None:
    trials = [_trial("t1", {"metric.duration": "tuned"}, {"duration": 42})]
    for df in (
        _result(trials).to_dataframe(),
        _result(trials).to_aggregated_dataframe(),
    ):
        assert df["metric.duration"].iloc[0] == "tuned"
        assert df["metric.duration_"].iloc[0] == 42


def test_aggregated_sort_uses_renamed_metric_column() -> None:
    # Metric "duration" is exported as metric.duration; sorting by the
    # objective "duration" must use it, not the trial-duration column.
    trials = [
        TrialResult(
            trial_id=f"t{i}",
            config={"m": i},
            metrics={"duration": float(i)},
            status=TrialStatus.COMPLETED,
            duration=d,
            timestamp=TS,
            metadata={},
        )
        for i, d in ((1, 10.0), (2, 1.0))
    ]
    df = _result(trials).to_aggregated_dataframe(primary_objective="duration")
    # "duration" is a minimize metric by name -> metric 1 first.
    assert list(df["metric.duration"]) == [1.0, 2.0]
