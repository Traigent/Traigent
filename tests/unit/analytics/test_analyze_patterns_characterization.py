"""Characterization tests for PredictiveAnalytics.analyze_patterns."""

import pytest

from traigent.analytics.predictive import PredictiveAnalytics


def _hist(n, metrics_fn):
    return [{"metrics": metrics_fn(i)} for i in range(n)]


def test_empty_history_is_insufficient():
    assert PredictiveAnalytics().analyze_patterns([]) == {
        "summary": {"status": "insufficient_data"},
        "trends": [],
        "predictions": {},
        "recommendations": [],
        "confidence": 0.0,
    }


def test_points_without_metrics_yield_no_analyzable_metrics():
    out = PredictiveAnalytics().analyze_patterns([{"x": 1}, {"y": 2}])
    assert out["summary"] == {
        "status": "analyzed",
        "data_points": 2,
        "metrics_analyzed": 0,
    }
    assert out["trends"] == {}
    assert out["warnings"] == ["insufficient_data", "no_analyzable_metrics"]
    assert out["confidence"] == pytest.approx(0.1)


def test_metric_with_fewer_than_three_values_is_not_analyzed():
    data = [{"metrics": {"a": 1.0}}, {"metrics": {"a": 2.0}}]
    out = PredictiveAnalytics().analyze_patterns(data)
    assert out["trends"] == {}
    assert "no_analyzable_metrics" in out["warnings"]


def test_increasing_trend_fields():
    out = PredictiveAnalytics().analyze_patterns(_hist(5, lambda i: {"acc": 1.0 + i}))
    trend = out["trends"]["acc"]
    assert trend["direction"] == "increasing"
    assert trend["strength"] == pytest.approx(4.0)
    assert trend["volatility"] == pytest.approx(1.5811388, rel=1e-6)
    assert out["summary"] == {
        "status": "analyzed",
        "data_points": 5,
        "metrics_analyzed": 1,
    }
    assert "warnings" not in out
    assert out["confidence"] == pytest.approx(0.25)


def test_flat_and_decreasing_directions_and_zero_base_floor():
    data = _hist(4, lambda i: {"flat": 2.0, "down": 4.0 - i, "zero": 0.0 + i})
    out = PredictiveAnalytics().analyze_patterns(data)
    assert out["trends"]["flat"]["direction"] == "decreasing"  # not strictly greater
    assert out["trends"]["flat"]["strength"] == 0
    assert out["trends"]["down"]["direction"] == "decreasing"
    # zero first value is floored at 0.001 in the denominator
    assert out["trends"]["zero"]["strength"] == pytest.approx(3.0 / 0.001)


def test_warning_for_short_history_with_analyzable_metric():
    out = PredictiveAnalytics().analyze_patterns(_hist(3, lambda i: {"acc": 1.0 + i}))
    assert out["warnings"] == ["insufficient_data"]


def test_confidence_caps_at_one():
    out = PredictiveAnalytics().analyze_patterns(_hist(30, lambda i: {"acc": 1.0 + i}))
    assert out["confidence"] == 1.0
    assert "warnings" not in out


def test_sparse_metrics_collected_per_name():
    data = [
        {"metrics": {"a": 1.0}},
        {"other": 1},
        {"metrics": {"a": 2.0, "b": 5.0}},
        {"metrics": {"a": 3.0, "b": 6.0}},
        {"metrics": {"b": 7.0}},
        {"metrics": {}},
    ]
    out = PredictiveAnalytics().analyze_patterns(data)
    assert set(out["trends"]) == {"a", "b"}
    assert out["trends"]["a"]["strength"] == pytest.approx(2.0)
    assert out["trends"]["b"]["strength"] == pytest.approx(0.4)
    assert out["summary"]["data_points"] == 6


def test_predictions_and_recommendations_receive_collected_inputs(monkeypatch):
    seen = {}
    pa = PredictiveAnalytics()

    def fake_predictions(trends):
        seen["trends"] = trends
        return {"p": 1}

    def fake_recs(analysis):
        seen["analysis"] = analysis
        return ["r"]

    monkeypatch.setattr(pa, "_generate_predictions", fake_predictions)
    monkeypatch.setattr(pa, "_generate_recommendations", fake_recs)
    out = pa.analyze_patterns(_hist(3, lambda i: {"acc": float(i + 1)}))
    assert seen["trends"] == {"acc": [1.0, 2.0, 3.0]}
    assert set(seen["analysis"]) == {"acc"}
    assert out["predictions"] == {"p": 1}
    assert out["recommendations"] == ["r"]


def test_non_dict_metrics_raises_attribute_error():
    with pytest.raises(AttributeError):
        PredictiveAnalytics().analyze_patterns([{"metrics": [1, 2]}])
