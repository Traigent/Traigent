"""Characterization tests for ``HistoricalAnalyticsEngine.get_insights``.

The expected values were generated from the implementation before it was
split into helpers, so they pin exact keys, key order, value types, exception
behaviour and the order in which record attributes are read.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from traigent.analytics.meta_learning import (
    AlgorithmType,
    HistoricalAnalyticsEngine,
    OptimizationRecord,
    OptimizationStatus,
)

BASE = datetime(2024, 1, 1, tzinfo=UTC)

GRID = AlgorithmType.GRID
RANDOM = AlgorithmType.RANDOM
BAYESIAN = AlgorithmType.BAYESIAN
GENETIC = AlgorithmType.GENETIC

COMPLETED = OptimizationStatus.COMPLETED
FAILED = OptimizationStatus.FAILED
TIMEOUT = OptimizationStatus.TIMEOUT
CANCELLED = OptimizationStatus.CANCELLED

_FIELDS = frozenset(OptimizationRecord.__dataclass_fields__)


class _TracedRecord(OptimizationRecord):
    """Record that logs every dataclass-field read to a shared list."""

    def __getattribute__(self, name: str) -> Any:
        if name in _FIELDS:
            log = object.__getattribute__(self, "_log")
            log.append(f"{object.__getattribute__(self, '_tag')}.{name}")
        return object.__getattribute__(self, name)


def _rec(
    idx: int,
    *,
    fn: str = "f",
    alg: AlgorithmType = GRID,
    status: OptimizationStatus = COMPLETED,
    score: Any = 0.5,
    dur: Any = 10.0,
    trials: Any = 5,
    day: float | None = None,
    ts: datetime | None = None,
    cls: type[OptimizationRecord] = OptimizationRecord,
) -> OptimizationRecord:
    if ts is None:
        ts = BASE + timedelta(days=idx if day is None else day)
    return cls(
        optimization_id=f"opt_{idx}",
        function_name=fn,
        algorithm=alg,
        configuration_space={"p": [1, 2]},
        objectives=["accuracy"],
        dataset_size=10,
        best_score=score,
        total_trials=trials,
        duration_seconds=dur,
        status=status,
        timestamp=ts,
    )


def _engine(tmp_path, records: list[OptimizationRecord]) -> HistoricalAnalyticsEngine:
    engine = HistoricalAnalyticsEngine(storage_path=str(tmp_path / "history.jsonl"))
    engine.history.records.extend(records)
    return engine


def _mixed() -> list[OptimizationRecord]:
    return [
        _rec(0, fn="a", alg=GRID, status=COMPLETED, score=0.6, dur=30.0, day=3),
        _rec(1, fn="a", alg=RANDOM, status=FAILED, score=0.1, dur=5.0, day=1),
        _rec(2, fn="b", alg=GRID, status=COMPLETED, score=0.8, dur=50.0, day=0),
        _rec(3, fn="a", alg=BAYESIAN, status=TIMEOUT, score=0.0, dur=99.5, day=2),
        _rec(4, fn="b", alg=RANDOM, status=COMPLETED, score=0.7, dur=20, day=5),
        _rec(5, fn="a", alg=GRID, status=CANCELLED, score=0.3, dur=7.25, day=4),
        _rec(6, fn="b", alg=BAYESIAN, status=FAILED, score=0.2, dur=1.0, day=6),
        _rec(7, fn="a", alg=RANDOM, status=COMPLETED, score=0.9, dur=40.0, day=7),
    ]


def _as_text(result: dict[str, Any]) -> str:
    return json.dumps(result)


GOLDEN_MIXED_ALL = '{"total_optimizations": 8, "successful_optimizations": 4, "algorithm_performance": {"grid": {"average_score": 0.7, "average_duration": 40.0, "success_rate": 0.6666666666666666, "total_runs": 2}, "random": {"average_score": 0.8, "average_duration": 30.0, "success_rate": 0.6666666666666666, "total_runs": 2}}, "recent_trend": "improving", "most_common_algorithm": ["grid", 3], "average_optimization_time": 31.59375, "functions_optimized": 2}'
GOLDEN_MIXED_A = '{"total_optimizations": 5, "successful_optimizations": 2, "algorithm_performance": {"grid": {"average_score": 0.6, "average_duration": 30.0, "success_rate": 0.5, "total_runs": 1}, "random": {"average_score": 0.9, "average_duration": 40.0, "success_rate": 0.5, "total_runs": 1}}, "recent_trend": "improving", "most_common_algorithm": ["grid", 2], "average_optimization_time": 36.35, "functions_optimized": 1}'
GOLDEN_MIXED_B = '{"total_optimizations": 3, "successful_optimizations": 2, "algorithm_performance": {"grid": {"average_score": 0.8, "average_duration": 50.0, "success_rate": 1.0, "total_runs": 1}, "random": {"average_score": 0.7, "average_duration": 20, "success_rate": 1.0, "total_runs": 1}}, "recent_trend": "declining", "most_common_algorithm": ["grid", 1], "average_optimization_time": 23.666666666666668, "functions_optimized": 1}'
GOLDEN_TRACE_ALL = [
    "r0.status",
    "r0.algorithm",
    "r0.best_score",
    "r0.duration_seconds",
    "r0.total_trials",
    "r1.status",
    "r2.status",
    "r2.algorithm",
    "r2.best_score",
    "r2.duration_seconds",
    "r2.total_trials",
    "r3.status",
    "r0.algorithm",
    "r1.algorithm",
    "r2.algorithm",
    "r3.algorithm",
    "r0.timestamp",
    "r1.timestamp",
    "r2.timestamp",
    "r3.timestamp",
    "r2.status",
    "r2.best_score",
    "r1.status",
    "r3.status",
    "r0.status",
    "r0.best_score",
    "r0.status",
    "r1.status",
    "r2.status",
    "r3.status",
    "r0.algorithm",
    "r1.algorithm",
    "r2.algorithm",
    "r3.algorithm",
    "r0.duration_seconds",
    "r1.duration_seconds",
    "r2.duration_seconds",
    "r3.duration_seconds",
    "r0.function_name",
    "r1.function_name",
    "r2.function_name",
    "r3.function_name",
]
GOLDEN_TRACE_FILTERED = [
    "r0.function_name",
    "r1.function_name",
    "r2.function_name",
    "r3.function_name",
    "r0.status",
    "r0.algorithm",
    "r0.best_score",
    "r0.duration_seconds",
    "r0.total_trials",
    "r2.status",
    "r2.algorithm",
    "r2.best_score",
    "r2.duration_seconds",
    "r2.total_trials",
    "r3.status",
    "r0.algorithm",
    "r2.algorithm",
    "r3.algorithm",
    "r0.timestamp",
    "r2.timestamp",
    "r3.timestamp",
    "r2.status",
    "r2.best_score",
    "r3.status",
    "r0.status",
    "r0.best_score",
    "r0.status",
    "r2.status",
    "r3.status",
    "r0.algorithm",
    "r2.algorithm",
    "r3.algorithm",
    "r0.duration_seconds",
    "r2.duration_seconds",
    "r3.duration_seconds",
    "r0.function_name",
    "r2.function_name",
    "r3.function_name",
]


def test_empty_history_returns_message_only(tmp_path):
    result = _engine(tmp_path, []).get_insights()
    assert result == {"message": "No optimization history available"}
    assert list(result) == ["message"]


def test_unknown_function_filter_returns_message_only(tmp_path):
    engine = _engine(tmp_path, _mixed())
    assert engine.get_insights("missing") == {
        "message": "No optimization history available"
    }


def test_empty_function_name_does_not_filter(tmp_path):
    engine = _engine(tmp_path, _mixed())
    assert _as_text(engine.get_insights("")) == _as_text(engine.get_insights())


def test_mixed_history_golden_text(tmp_path):
    engine = _engine(tmp_path, _mixed())
    assert _as_text(engine.get_insights()) == GOLDEN_MIXED_ALL
    assert _as_text(engine.get_insights(function_name="a")) == GOLDEN_MIXED_A
    assert _as_text(engine.get_insights(function_name="b")) == GOLDEN_MIXED_B


def test_mixed_history_value_types_and_denominators(tmp_path):
    result = _engine(tmp_path, _mixed()).get_insights()

    assert list(result) == [
        "total_optimizations",
        "successful_optimizations",
        "algorithm_performance",
        "recent_trend",
        "most_common_algorithm",
        "average_optimization_time",
        "functions_optimized",
    ]
    # Three runs each of grid and random: the first-seen algorithm wins the tie.
    assert result["most_common_algorithm"] == ("grid", 3)
    assert isinstance(result["most_common_algorithm"], tuple)
    performance = result["algorithm_performance"]
    # Keys follow the first completed record per algorithm; bayesian never
    # completed so it is absent even though it appears in the counts.
    assert list(performance) == ["grid", "random"]
    assert performance["grid"]["success_rate"] == pytest.approx(2 / 3)
    assert performance["random"]["success_rate"] == pytest.approx(2 / 3)
    assert performance["grid"]["total_runs"] == 2
    assert list(performance["grid"]) == [
        "average_score",
        "average_duration",
        "success_rate",
        "total_runs",
    ]
    # Integer duration stays an int when it is the only value.
    single = _engine(tmp_path, [_rec(0, dur=60)]).get_insights()
    assert single["average_optimization_time"] == 60
    assert isinstance(single["average_optimization_time"], int)


def test_all_failed_history(tmp_path):
    records = [
        _rec(0, status=FAILED, score=None, trials=None, dur=3.0),
        _rec(1, status=TIMEOUT, score=None, trials=None, dur=4.0, alg=RANDOM),
        _rec(2, status=CANCELLED, score=None, trials=None, dur=5.0, alg=RANDOM),
    ]
    result = _engine(tmp_path, records).get_insights()
    assert _as_text(result) == (
        '{"total_optimizations": 3, "successful_optimizations": 0, '
        '"algorithm_performance": {}, "recent_trend": "insufficient_data", '
        '"most_common_algorithm": ["random", 2], '
        '"average_optimization_time": 4.0, "functions_optimized": 1}'
    )


def test_single_record_is_insufficient_for_trend(tmp_path):
    result = _engine(tmp_path, [_rec(0)]).get_insights()
    assert result["recent_trend"] == "insufficient_data"
    assert result["algorithm_performance"]["grid"]["success_rate"] == 1.0


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.1, 0.9], "improving"),
        ([0.9, 0.1], "declining"),
        ([0.5, 0.5], "declining"),
        ([0.1, 0.99, 0.2], "improving"),
        ([0.2, 0.01, 0.1], "declining"),
        ([0.5, 0.1, 0.5], "declining"),
    ],
)
def test_trend_compares_first_and_last_completed_scores(tmp_path, scores, expected):
    records = [_rec(i, score=s) for i, s in enumerate(scores)]
    assert _engine(tmp_path, records).get_insights()["recent_trend"] == expected


def test_trend_uses_timestamp_order_not_insertion_order(tmp_path):
    records = [
        _rec(0, score=0.9, day=2),
        _rec(1, score=0.1, day=0),
        _rec(2, score=0.5, day=1),
    ]
    # Time order is 0.1, 0.5, 0.9 -> improving; insertion order would decline.
    assert _engine(tmp_path, records).get_insights()["recent_trend"] == "improving"


def test_trend_window_is_last_ten_by_timestamp(tmp_path):
    scores = [100.0, 90.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    records = [_rec(i, score=s) for i, s in enumerate(scores)]
    # Insert out of order; first two (largest scores) fall outside the window.
    shuffled = [records[i] for i in (7, 0, 11, 3, 1, 9, 5, 2, 10, 4, 8, 6)]
    result = _engine(tmp_path, shuffled).get_insights()
    assert result["recent_trend"] == "improving"
    assert result["total_optimizations"] == 12


def test_trend_window_counts_non_completed_records(tmp_path):
    # Ten newest records: nine failures and one completed -> too few scores,
    # even though two older completed records exist outside the window.
    records = [
        _rec(0, score=0.1),
        _rec(1, score=0.9),
        *[_rec(i, status=FAILED, score=0.5) for i in range(2, 11)],
        _rec(11, score=0.3),
    ]
    result = _engine(tmp_path, records).get_insights()
    assert result["recent_trend"] == "insufficient_data"
    assert result["successful_optimizations"] == 3


def test_two_records_with_one_completed_is_insufficient(tmp_path):
    records = [_rec(0, score=0.1), _rec(1, status=FAILED, score=0.9)]
    assert _engine(tmp_path, records).get_insights()["recent_trend"] == (
        "insufficient_data"
    )


def test_equal_timestamps_keep_insertion_order(tmp_path):
    first = [_rec(0, score=0.2, day=1), _rec(1, score=0.8, day=1)]
    second = [_rec(0, score=0.8, day=1), _rec(1, score=0.2, day=1)]
    assert _engine(tmp_path, first).get_insights()["recent_trend"] == "improving"
    assert _engine(tmp_path, second).get_insights()["recent_trend"] == "declining"


def test_nan_scores_compare_false_and_propagate_to_mean(tmp_path):
    nan = float("nan")
    for scores in ([nan, 1.0], [1.0, nan]):
        records = [_rec(i, score=s) for i, s in enumerate(scores)]
        result = _engine(tmp_path, records).get_insights()
        assert result["recent_trend"] == "declining"
        assert math.isnan(result["algorithm_performance"]["grid"]["average_score"])


def test_float_means_use_exact_statistics_mean(tmp_path):
    records = [
        _rec(0, score=0.1, dur=0.1),
        _rec(1, score=0.2, dur=0.2),
        _rec(2, score=0.3, dur=0.3),
    ]
    result = _engine(tmp_path, records).get_insights()
    grid = result["algorithm_performance"]["grid"]
    # A naive sum / 3 would give 0.20000000000000004.
    assert repr(grid["average_score"]) == "0.2"
    assert repr(grid["average_duration"]) == "0.2"
    assert repr(result["average_optimization_time"]) == "0.2"


def test_most_common_algorithm_counts_every_status(tmp_path):
    records = [
        _rec(0, alg=RANDOM, status=FAILED),
        _rec(1, alg=RANDOM, status=TIMEOUT),
        _rec(2, alg=GRID),
        _rec(3, alg=GENETIC, status=CANCELLED),
    ]
    result = _engine(tmp_path, records).get_insights()
    assert result["most_common_algorithm"] == ("random", 2)
    assert list(result["algorithm_performance"]) == ["grid"]
    assert result["functions_optimized"] == 1


def test_functions_optimized_counts_distinct_names(tmp_path):
    records = [_rec(0, fn="x"), _rec(1, fn="y"), _rec(2, fn="x")]
    assert _engine(tmp_path, records).get_insights()["functions_optimized"] == 2


def test_naive_and_aware_timestamps_raise_on_ordering(tmp_path):
    records = [_rec(0, ts=datetime(2024, 1, 1)), _rec(1, ts=BASE)]
    with pytest.raises(TypeError, match="offset-naive and offset-aware"):
        _engine(tmp_path, records).get_insights()


def test_single_naive_timestamp_is_never_compared(tmp_path):
    result = _engine(tmp_path, [_rec(0, ts=datetime(2024, 1, 1))]).get_insights()
    assert result["recent_trend"] == "insufficient_data"


def test_score_error_surfaces_before_timestamp_error(tmp_path):
    records = [
        _rec(0, score="bad", ts=datetime(2024, 1, 1)),
        _rec(1, score=0.5, ts=BASE),
    ]
    with pytest.raises(TypeError) as excinfo:
        _engine(tmp_path, records).get_insights()
    assert "offset" not in str(excinfo.value)


def test_duration_error_in_failed_record_surfaces_last(tmp_path):
    records = [_rec(0), _rec(1, status=FAILED, dur="slow")]
    with pytest.raises(TypeError):
        _engine(tmp_path, records).get_insights()


def test_timestamp_error_surfaces_before_final_duration_error(tmp_path):
    records = [
        _rec(0, ts=datetime(2024, 1, 1)),
        _rec(1, status=FAILED, dur="slow", ts=BASE),
    ]
    with pytest.raises(TypeError, match="offset-naive and offset-aware"):
        _engine(tmp_path, records).get_insights()


def test_score_mean_error_precedes_duration_mean_error(tmp_path):
    records = [_rec(0, score="bad", dur="slow")]
    with pytest.raises(TypeError) as excinfo:
        _engine(tmp_path, records).get_insights()
    assert "bad" in str(excinfo.value) or "str" in str(excinfo.value)


def test_does_not_mutate_history(tmp_path):
    engine = _engine(tmp_path, _mixed())
    before = list(engine.history.records)
    engine.get_insights()
    engine.get_insights(function_name="a")
    assert engine.history.records == before
    assert [r.optimization_id for r in engine.history.records] == [
        f"opt_{i}" for i in range(8)
    ]


def _traced(tmp_path, log: list[str]) -> HistoricalAnalyticsEngine:
    specs = [
        {
            "fn": "a",
            "alg": GRID,
            "status": COMPLETED,
            "score": 0.6,
            "dur": 30.0,
            "day": 3,
        },
        {
            "fn": "b",
            "alg": RANDOM,
            "status": FAILED,
            "score": None,
            "dur": 5.0,
            "trials": None,
            "day": 1,
        },
        {
            "fn": "a",
            "alg": GRID,
            "status": COMPLETED,
            "score": 0.8,
            "dur": 50.0,
            "day": 0,
        },
        {
            "fn": "a",
            "alg": BAYESIAN,
            "status": TIMEOUT,
            "score": None,
            "dur": 9.5,
            "day": 2,
        },
    ]
    records = []
    for idx, spec in enumerate(specs):
        record = _rec(idx, cls=_TracedRecord, **spec)
        object.__setattr__(record, "_log", log)
        object.__setattr__(record, "_tag", f"r{idx}")
        records.append(record)
    log.clear()
    return _engine(tmp_path, records)


def test_attribute_read_order_unfiltered(tmp_path):
    log: list[str] = []
    engine = _traced(tmp_path, log)
    engine.get_insights()
    assert log == GOLDEN_TRACE_ALL


def test_attribute_read_order_filtered(tmp_path):
    log: list[str] = []
    engine = _traced(tmp_path, log)
    engine.get_insights(function_name="a")
    assert log == GOLDEN_TRACE_FILTERED
