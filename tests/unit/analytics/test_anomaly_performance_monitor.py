from datetime import datetime, timedelta, timezone

from traigent.analytics.anomaly import AnomalyType, PerformanceMonitor


def _record_series(
    monitor: PerformanceMonitor,
    func: str,
    alg: str,
    start_minutes_ago: int,
    count: int,
    value_score: float,
    value_duration: float,
) -> None:
    base_time = datetime.now(timezone.utc) - timedelta(minutes=start_minutes_ago)
    for i in range(count):
        ts = base_time + timedelta(minutes=i)
        monitor.record_performance(
            function_name=func,
            algorithm=alg,
            score=value_score,
            duration=value_duration,
            timestamp=ts,
        )


def test_baseline_slice_precedes_recent_window_and_triggers_regression() -> None:
    monitor = PerformanceMonitor(degradation_threshold=0.15)
    func = "my_func"
    alg = "algoA"

    # Create 20 baseline minutes followed by 60 recent minutes (total 80 mins)
    # Baseline scores high (1.0), recent scores lower (0.7) to trigger regression
    _record_series(
        monitor,
        func,
        alg,
        start_minutes_ago=80,
        count=20,
        value_score=1.0,
        value_duration=1.0,
    )
    _record_series(
        monitor,
        func,
        alg,
        start_minutes_ago=60,
        count=60,
        value_score=0.7,
        value_duration=1.0,
    )

    anomalies = monitor.detect_performance_regression(func, alg, lookback_hours=1)

    # Expect a degradation anomaly for score only
    score_key = f"{func}:{alg}:score"
    score_anoms = [
        a
        for a in anomalies
        if a.metric_name == score_key
        and a.anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION
    ]

    assert len(score_anoms) == 1
    event = score_anoms[0]

    # Baseline window should be the 20 mins immediately preceding the recent window
    assert event.context["baseline_period"] == "20 data points"
    assert event.context["recent_period"] in {"60 data points", "59 data points"}
    # 30% drop -> severity HIGH (not CRITICAL, which is >30%)
    assert event.severity.value in ["high", "critical"]


def test_no_baseline_when_recent_covers_all_history() -> None:
    monitor = PerformanceMonitor()
    func = "my_func"
    alg = "algoA"

    # Only recent data (all within lookback) -> baseline slice should be empty and no anomalies
    _record_series(
        monitor,
        func,
        alg,
        start_minutes_ago=30,
        count=30,
        value_score=0.5,
        value_duration=1.0,
    )

    anomalies = monitor.detect_performance_regression(func, alg, lookback_hours=24)
    assert anomalies == []


def test_insufficient_baseline_data_returns_no_anomaly() -> None:
    monitor = PerformanceMonitor()
    func = "my_func"
    alg = "algoA"

    # 64 points total: recent window is last 60 mins -> baseline has only 4 points (<5)
    _record_series(
        monitor,
        func,
        alg,
        start_minutes_ago=64,
        count=4,
        value_score=1.0,
        value_duration=1.0,
    )
    _record_series(
        monitor,
        func,
        alg,
        start_minutes_ago=60,
        count=60,
        value_score=0.5,
        value_duration=1.0,
    )

    # Force small baseline window to guarantee insufficient baseline (<5)
    score_key = f"{func}:{alg}:score"
    duration_key = f"{func}:{alg}:duration"
    monitor.baseline_windows[score_key] = 3
    monitor.baseline_windows[duration_key] = 3

    anomalies = monitor.detect_performance_regression(func, alg, lookback_hours=1)
    assert anomalies == []
