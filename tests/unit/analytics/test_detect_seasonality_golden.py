"""Golden characterization for ``TrendAnalyzer.detect_seasonality``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from traigent.analytics.predictive import TrendAnalyzer

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _pt(hours: int, **kw: Any) -> dict[str, Any]:
    return {"timestamp": _T0 + timedelta(hours=hours), **kw}


def _call(points: Any, metric: str = "m") -> dict[str, Any]:
    return TrendAnalyzer().detect_seasonality(points, metric)


def test_insufficient_points() -> None:
    expected = {
        "has_seasonality": False,
        "period": None,
        "pattern_strength": 0.0,
        "note": "Insufficient data for seasonality detection",
    }
    assert _call(None) == expected
    assert _call([]) == expected
    assert _call([_pt(i, m=1) for i in range(23)]) == expected


def test_length_gate_counts_non_dict_points_but_they_are_skipped() -> None:
    points: list[Any] = [_pt(i, m=1.0) for i in range(11)] + ["x"] * 13
    out = _call(points)
    assert out["note"] == "Insufficient hourly data"


def test_fewer_than_twelve_distinct_hours() -> None:
    out = _call([_pt(i % 11, m=float(i)) for i in range(48)])
    assert out == {
        "has_seasonality": False,
        "period": None,
        "pattern_strength": 0.0,
        "note": "Insufficient hourly data",
    }


def test_flat_signal_has_no_seasonality() -> None:
    out = _call([_pt(i, m=5.0) for i in range(48)])
    assert out == {"has_seasonality": False, "period": None, "pattern_strength": 0.0}


def test_daily_pattern_detected_with_exact_strength() -> None:
    out = _call([_pt(i, m=float(10 + (i % 24))) for i in range(48)])
    means = [10.0 + h for h in range(24)]
    mean = sum(means) / 24
    var = sum((x - mean) ** 2 for x in means) / 23
    assert out["has_seasonality"] is True
    assert out["period"] == "daily"
    assert out["pattern_strength"] == pytest.approx(var / mean**2)
    assert "note" not in out


def test_weak_pattern_below_threshold() -> None:
    out = _call([_pt(i, m=1000.0 + (i % 24) * 0.01) for i in range(48)])
    assert out["has_seasonality"] is False
    assert out["period"] is None
    assert 0 < out["pattern_strength"] <= 0.01


def test_value_key_resolution_order_and_missing_default_zero() -> None:
    # metric key wins over "value"; "value" used when metric absent; else 0
    pts = [_pt(h, m=3.0, value=100.0) for h in range(12)]
    pts += [_pt(h + 12, value=3.0) for h in range(12)]
    out = _call(pts)
    assert out["pattern_strength"] == 0.0 and out["has_seasonality"] is False
    pts = [_pt(h, other=9) for h in range(24)]  # all values default to 0
    out = _call(pts)
    assert out == {"has_seasonality": False, "period": None, "pattern_strength": 0}


def test_falsey_metric_value_is_kept_not_replaced() -> None:
    pts = [_pt(h, m=0, value=50.0) for h in range(24)]
    out = _call(pts)
    assert out["pattern_strength"] == 0 and out["has_seasonality"] is False


def test_unsorted_input_is_accepted() -> None:
    pts = [_pt(i, m=float(10 + (i % 24))) for i in range(48)]
    assert _call(list(reversed(pts))) == _call(pts)


def test_non_dict_points_ignored_in_aggregation() -> None:
    base = [_pt(i, m=float(10 + (i % 24))) for i in range(48)]
    assert _call([*base, 7, None, "s"]) == _call(base)
