"""Golden characterization for ``TrendAnalyzer._resolve_trend_input``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from traigent.analytics.predictive import TrendAnalyzer

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _ta_with_history(name: str, ages_days: list[float], values: list[float]):
    ta = TrendAnalyzer()
    now = datetime.now(UTC)
    ta.metrics_history[name] = [
        (now - timedelta(days=a), v) for a, v in zip(ages_days, values, strict=True)
    ]
    return ta


def test_metric_name_missing():
    out = TrendAnalyzer()._resolve_trend_input("m", None, 30)
    assert out == {"error": "No data available for metric: m"}


def test_metric_name_filters_by_period_and_keeps_order():
    ta = _ta_with_history("m", [40, 5, 3, 1], [9, 1, 2, 3])
    out = ta._resolve_trend_input("m", 10, 30)
    assert out == {"values": [1, 2, 3], "metric_name": "m", "period_days": 10}


def test_metric_name_non_int_period_keeps_default():
    ta = _ta_with_history("m", [5, 3, 1], [1, 2, 3])
    out = ta._resolve_trend_input("m", "10", 30)
    assert out["period_days"] == 30
    assert out["values"] == [1, 2, 3]
    # bool is an int: True -> period of 1 day
    out = ta._resolve_trend_input("m", True, 30)
    assert out == {"error": "Insufficient data for trend analysis"}


def test_metric_name_needs_three_recent():
    ta = _ta_with_history("m", [5, 3], [1, 2])
    assert ta._resolve_trend_input("m", None, 30) == {
        "error": "Insufficient data for trend analysis"
    }


@pytest.mark.parametrize("empty", [None, [], ()])
def test_no_data_points(empty):
    assert TrendAnalyzer()._resolve_trend_input(empty, None, 30) == {
        "error": "No data points provided"
    }


def test_data_points_dicts_sorted_by_timestamp():
    pts = [
        {"timestamp": _T0 + timedelta(hours=2), "m": 3},
        {"timestamp": _T0, "m": "1.5"},
        {"timestamp": _T0 + timedelta(hours=1), "value": 2},
    ]
    out = TrendAnalyzer()._resolve_trend_input(pts, "m", 7)
    assert out == {"values": [1.5, 2.0, 3.0], "metric_name": "m", "period_days": 7}


def test_data_points_default_metric_name_and_missing_value_is_zero():
    pts = [
        {"timestamp": _T0, "other": 5},
        {"timestamp": _T0 + timedelta(hours=1), "value": 4},
    ]
    out = TrendAnalyzer()._resolve_trend_input(pts, "", 7)
    assert out == {"values": [0.0, 4.0], "metric_name": "value", "period_days": 7}
    out = TrendAnalyzer()._resolve_trend_input(pts, None, 7)
    assert out["metric_name"] == "value"


def test_data_points_metric_key_beats_value_key():
    pts = [
        {"timestamp": _T0, "m": 1, "value": 100},
        {"timestamp": _T0 + timedelta(hours=1), "m": 2, "value": 100},
    ]
    assert TrendAnalyzer()._resolve_trend_input(pts, "m", 7)["values"] == [1.0, 2.0]


def test_data_points_scalars_and_single_point_insufficient():
    out = TrendAnalyzer()._resolve_trend_input([1, "2", 3.5], None, 7)
    assert out["values"] == [1.0, 2.0, 3.5]
    assert TrendAnalyzer()._resolve_trend_input([4], None, 7) == {
        "error": "Insufficient data for trend analysis"
    }


def test_data_points_bad_value_raises():
    with pytest.raises(ValueError):
        TrendAnalyzer()._resolve_trend_input([{"m": "abc"}, 1], "m", 7)
    with pytest.raises(TypeError):
        TrendAnalyzer()._resolve_trend_input([None, 1], None, 7)


def test_data_points_missing_timestamp_uses_now_and_keeps_input_order():
    out = TrendAnalyzer()._resolve_trend_input([{"m": 1}, {"m": 2}, 3], "m", 7)
    assert out["values"] == [1.0, 2.0, 3.0]
