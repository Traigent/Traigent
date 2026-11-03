"""Characterization tests for TrendAnalyzer.detect_anomalies."""

from datetime import UTC, datetime

import pytest

from traigent.analytics.predictive import TrendAnalyzer

TS = datetime(2026, 1, 1, tzinfo=UTC)


def _points(values, key="cost"):
    return [{key: v, "timestamp": TS} for v in values]


def _run(points, metric="cost"):
    return TrendAnalyzer().detect_anomalies(points, metric)


@pytest.mark.parametrize("points", [None, [], [{"cost": 1}, {"cost": 2}]])
def test_too_few_points_returns_empty(points):
    assert _run(points) == []


def test_non_dict_points_are_skipped_and_can_leave_too_few():
    assert _run([1, 2, {"cost": 1}, {"cost": 2}, "x"]) == []


def test_points_without_metric_or_value_are_skipped():
    points = [{"other": 1}, {"other": 2}, {"cost": 1}, {"cost": 2}]
    assert _run(points) == []


def test_constant_values_have_zero_std_and_no_anomalies():
    assert _run(_points([5, 5, 5, 5])) == []


def test_value_key_is_fallback_when_metric_missing():
    values = [10] * 20 + [1000]
    points = [{"value": v, "timestamp": TS} for v in values]
    out = _run(points)
    assert [a["value"] for a in out] == [1000]


def test_metric_key_preferred_over_value_key():
    points = [{"cost": 10, "value": -999, "timestamp": TS} for _ in range(20)]
    points.append({"cost": 1000, "value": -999, "timestamp": TS})
    out = _run(points)
    assert [a["value"] for a in out] == [1000]


def test_high_medium_low_severities():
    base = [0.0] * 200
    # High: z > 4 on a single large outlier among many zeros.
    out_high = _run(_points(base + [100.0]))
    assert [a["severity"] for a in out_high] == ["high"]
    assert out_high[0]["type"] == "statistical_outlier"
    assert out_high[0]["timestamp"] == TS
    assert out_high[0]["z_score"] > 4

    # Medium: 3 < z <= 4 (z of a single outlier among n points is bounded by
    # (n-1)/sqrt(n); n=13 gives ~3.33).
    out_med = _run(_points([0.0] * 12 + [1.0]))
    assert [a["severity"] for a in out_med] == ["medium"]
    assert 3 < out_med[0]["z_score"] <= 4

    # Low: 2 < z <= 3 (n=6 gives ~2.04).
    out_low = _run(_points([0.0] * 5 + [1.0]))
    assert [a["severity"] for a in out_low] == ["low"]
    assert 2 < out_low[0]["z_score"] <= 3


def test_missing_timestamp_defaults_to_now_datetime():
    points = [{"cost": 0.0} for _ in range(200)] + [{"cost": 100.0}]
    out = _run(points)
    assert len(out) == 1
    assert isinstance(out[0]["timestamp"], datetime)


def test_anomalies_keep_input_order():
    values = [0.0] * 200
    values[5] = 100.0
    values[150] = -100.0
    out = _run(_points(values))
    assert [a["value"] for a in out] == [100.0, -100.0]
