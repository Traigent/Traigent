"""#2847: regression and trend detectors handle zero and negative baselines.

A 0 -> non-zero error rate was never reported, and a negative baseline flipped
the sign (an improvement flagged CRITICAL, a regression ignored). The plugin
mirrors must behave identically.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from traigent.analytics import anomaly as core_anomaly
from traigent.analytics import predictive as core_predictive
from traigent.analytics.predictive import TrendDirection
from traigent.utils.relative_change import signed_relative_delta

PLUGIN_DIR = (
    Path(__file__).resolve().parents[3]
    / "plugins"
    / "traigent-analytics"
    / "traigent_analytics"
)


def _load_plugin(name: str) -> ModuleType:
    module_name = f"_plugin_analytics_2847_{name}"
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["core", "plugin"])
def modules(request: pytest.FixtureRequest) -> tuple[ModuleType, ModuleType]:
    if request.param == "core":
        return core_anomaly, core_predictive
    return _load_plugin("anomaly"), _load_plugin("predictive")


def _series(metric: str, baseline: float, recent: float):
    start = datetime.now(UTC) - timedelta(days=1)
    rows = [
        (start + timedelta(minutes=i), {metric: baseline if i < 6 else recent})
        for i in range(10)
    ]
    return rows[:6], rows[6:]


def _regression(anomaly: ModuleType, metric: str, baseline: float, recent: float):
    detector = anomaly.RegressionDetector(sensitivity=0.1)
    base, rec = _series(metric, baseline, recent)
    return detector._analyze_metric_regression(metric, base, rec, "fn", "grid")


def test_zero_baseline_error_rate_is_reported(modules) -> None:
    anomaly, _ = modules
    event = _regression(anomaly, "error_rate", 0.0, 0.4)
    assert event is not None
    assert event.severity == anomaly.AlertSeverity.CRITICAL


def test_small_baseline_error_rate_still_critical(modules) -> None:
    anomaly, _ = modules
    event = _regression(anomaly, "error_rate", 0.01, 0.4)
    assert event is not None
    assert event.severity == anomaly.AlertSeverity.CRITICAL


def test_zero_to_zero_is_not_a_regression(modules) -> None:
    anomaly, _ = modules
    assert _regression(anomaly, "error_rate", 0.0, 0.0) is None


def test_negative_baseline_worse_accuracy_is_reported(modules) -> None:
    anomaly, _ = modules
    assert _regression(anomaly, "accuracy", -0.2, -0.6) is not None


def test_negative_baseline_better_accuracy_is_not_a_regression(modules) -> None:
    anomaly, _ = modules
    assert _regression(anomaly, "accuracy", -0.6, -0.2) is None


def test_trend_from_zero_baseline_is_increasing(modules) -> None:
    _, predictive = modules
    direction, change = predictive.TrendAnalyzer._classify_trend(0.0, 0.5)
    assert direction.value == TrendDirection.INCREASING.value
    assert change > 0


def test_trend_with_negative_baseline_uses_the_real_sign(modules) -> None:
    _, predictive = modules
    direction, change = predictive.TrendAnalyzer._classify_trend(-0.2, -0.4)
    assert direction.value == TrendDirection.DECREASING.value
    assert change == pytest.approx(-1.0)


def test_trend_zero_to_zero_is_stable(modules) -> None:
    _, predictive = modules
    direction, change = predictive.TrendAnalyzer._classify_trend(0.0, 0.0)
    assert direction.value == TrendDirection.STABLE.value
    assert change == 0.0


def test_performance_monitor_zero_baseline_duration_regression(modules) -> None:
    anomaly, _ = modules
    monitor = anomaly.PerformanceMonitor()
    now = datetime.now(UTC)
    history = [(now - timedelta(hours=48, minutes=-i), 0.0) for i in range(20)] + [
        (now - timedelta(minutes=5 - i), 5.0) for i in range(5)
    ]
    monitor.performance_history["d"] = history
    events = monitor._analyze_regression("d", "duration", 24, higher_is_better=False)
    assert events and events[0].severity == anomaly.AlertSeverity.CRITICAL


@pytest.mark.parametrize(
    ("baseline", "current", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.4, 40.0),
        (-0.2, -0.6, pytest.approx(-2.0)),
        (-0.6, -0.2, pytest.approx(2 / 3)),
        (0.5, 0.25, -0.5),
    ],
)
def test_signed_relative_delta(baseline, current, expected) -> None:
    assert signed_relative_delta(baseline, current) == expected
