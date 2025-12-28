"""Anomaly detection and performance monitoring for Traigent optimizations."""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Reliability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import statistics
import threading
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from ..core.constants import (
    HISTORY_PRUNE_RATIO,
    MAX_ALERT_HISTORY_SIZE,
    MAX_PERFORMANCE_HISTORY_SIZE,
    MAX_REGRESSION_HISTORY_SIZE,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


def _ensure_utc(timestamp: datetime | None) -> datetime:
    """Coerce timestamps to timezone-aware UTC."""
    if timestamp is None:
        return _utc_now()
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    COST_SPIKE = "cost_spike"
    DURATION_ANOMALY = "duration_anomaly"
    SUCCESS_RATE_DROP = "success_rate_drop"
    RESOURCE_USAGE_SPIKE = "resource_usage_spike"
    PATTERN_DEVIATION = "pattern_deviation"
    THRESHOLD_VIOLATION = "threshold_violation"


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AnomalyEvent:
    """Detected anomaly event."""

    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    metric_name: str
    actual_value: float
    expected_value: float
    deviation_score: float  # How far from normal (z-score like)
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class MonitoringRule:
    """Monitoring rule configuration."""

    rule_id: str
    metric_name: str
    rule_type: str  # threshold, statistical, pattern
    parameters: dict[str, Any]
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.MEDIUM


@dataclass
class PerformanceBaseline:
    """Performance baseline for anomaly detection."""

    metric_name: str
    mean_value: float
    std_deviation: float
    min_value: float
    max_value: float
    sample_count: int
    last_updated: datetime
    percentiles: dict[str, float] = field(default_factory=dict)


class StatisticalDetector:
    """Statistical anomaly detection using z-score and percentile methods."""

    def __init__(self, window_size: int = 100, z_threshold: float = 2.5) -> None:
        """Initialize statistical detector."""
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.baselines: dict[str, PerformanceBaseline] = {}
        self.data_windows: dict[str, deque[tuple[datetime, float]]] = {}

    def add_data_point(
        self, metric_name: str, value: float, timestamp: datetime | None = None
    ) -> None:
        """Add data point and update baseline."""
        timestamp = _ensure_utc(timestamp)

        # Initialize data window if needed
        if metric_name not in self.data_windows:
            self.data_windows[metric_name] = deque(maxlen=self.window_size)

        self.data_windows[metric_name].append((timestamp, value))

        # Update baseline if we have enough data
        if len(self.data_windows[metric_name]) >= 10:
            self._update_baseline(metric_name)

    def detect_anomalies(self, metric_name: str, value: float) -> list[AnomalyEvent]:
        """Detect anomalies in a metric value."""
        anomalies: list[AnomalyEvent] = []

        baseline = self.baselines.get(metric_name)
        if not baseline:
            return anomalies  # No baseline established yet

        # Z-score detection
        if baseline.std_deviation > 0:
            z_score = abs(value - baseline.mean_value) / baseline.std_deviation

            if z_score > self.z_threshold:
                severity = self._calculate_severity(z_score)

                anomaly = AnomalyEvent(
                    anomaly_id=f"stat_{metric_name}_{int(datetime.now(UTC).timestamp())}",
                    anomaly_type=AnomalyType.PATTERN_DEVIATION,
                    severity=severity,
                    metric_name=metric_name,
                    actual_value=value,
                    expected_value=baseline.mean_value,
                    deviation_score=z_score,
                    timestamp=datetime.now(UTC),
                    description=(
                        f"Statistical anomaly: {metric_name} value {value:.3f} deviates "
                        f"{z_score:.2f} standard deviations from baseline {baseline.mean_value:.3f}"
                    ),
                    recommendations=self._get_statistical_recommendations(
                        metric_name, z_score
                    ),
                )

                anomalies.append(anomaly)

        # Percentile-based detection
        if baseline.percentiles:
            p99 = baseline.percentiles.get("p99", float("inf"))
            p1 = baseline.percentiles.get("p1", float("-inf"))

            if value > p99:
                anomaly = AnomalyEvent(
                    anomaly_id=f"perc_{metric_name}_{int(datetime.now(UTC).timestamp())}",
                    anomaly_type=AnomalyType.THRESHOLD_VIOLATION,
                    severity=AlertSeverity.HIGH,
                    metric_name=metric_name,
                    actual_value=value,
                    expected_value=baseline.percentiles.get("p50", baseline.mean_value),
                    deviation_score=(value - p99) / max(baseline.std_deviation, 1),
                    timestamp=datetime.now(UTC),
                    description=f"Value {value:.3f} exceeds 99th percentile ({p99:.3f}) for {metric_name}",
                    recommendations=self._get_percentile_recommendations(
                        metric_name, "high"
                    ),
                )
                anomalies.append(anomaly)

            elif value < p1:
                anomaly = AnomalyEvent(
                    anomaly_id=f"perc_{metric_name}_{int(datetime.now(UTC).timestamp())}",
                    anomaly_type=AnomalyType.THRESHOLD_VIOLATION,
                    severity=AlertSeverity.MEDIUM,
                    metric_name=metric_name,
                    actual_value=value,
                    expected_value=baseline.percentiles.get("p50", baseline.mean_value),
                    deviation_score=(p1 - value) / max(baseline.std_deviation, 1),
                    timestamp=datetime.now(UTC),
                    description=f"Value {value:.3f} below 1st percentile ({p1:.3f}) for {metric_name}",
                    recommendations=self._get_percentile_recommendations(
                        metric_name, "low"
                    ),
                )
                anomalies.append(anomaly)

        return anomalies

    def _update_baseline(self, metric_name: str) -> None:
        """Update statistical baseline for a metric."""
        data_points = [value for _, value in self.data_windows[metric_name]]

        if len(data_points) < 2:
            return

        mean_val = statistics.mean(data_points)
        std_val = statistics.stdev(data_points) if len(data_points) > 1 else 0.0
        min_val = min(data_points)
        max_val = max(data_points)

        # Calculate percentiles
        sorted_data = sorted(data_points)
        n = len(sorted_data)
        percentiles = {}

        for p in [1, 5, 25, 50, 75, 95, 99]:
            idx = int((p / 100) * (n - 1))
            percentiles[f"p{p}"] = sorted_data[idx]

        self.baselines[metric_name] = PerformanceBaseline(
            metric_name=metric_name,
            mean_value=mean_val,
            std_deviation=std_val,
            min_value=min_val,
            max_value=max_val,
            sample_count=len(data_points),
            last_updated=datetime.now(UTC),
            percentiles=percentiles,
        )

        logger.debug(
            f"Updated baseline for {metric_name}: mean={mean_val:.3f}, std={std_val:.3f}"
        )

    def _calculate_severity(self, z_score: float) -> AlertSeverity:
        """Calculate alert severity based on z-score."""
        if z_score >= 4.0:
            return AlertSeverity.CRITICAL
        elif z_score >= 3.0:
            return AlertSeverity.HIGH
        elif z_score >= 2.5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _get_statistical_recommendations(
        self, metric_name: str, z_score: float
    ) -> list[str]:
        """Get recommendations for statistical anomalies."""
        recommendations = []

        if "cost" in metric_name.lower():
            if z_score > 3.0:
                recommendations.append(
                    "Investigate sudden cost increase - check for resource usage spikes"
                )
                recommendations.append(
                    "Review recent optimization configurations for inefficiencies"
                )
            else:
                recommendations.append("Monitor cost trends closely")

        elif "duration" in metric_name.lower() or "time" in metric_name.lower():
            recommendations.append("Check system performance and resource availability")
            recommendations.append("Review evaluation function complexity")

        elif "score" in metric_name.lower() or "accuracy" in metric_name.lower():
            recommendations.append("Investigate data quality and model configuration")
            recommendations.append("Check for changes in evaluation criteria")

        else:
            recommendations.append(f"Investigate unusual behavior in {metric_name}")
            recommendations.append("Check for recent system or configuration changes")

        return recommendations

    def _get_percentile_recommendations(
        self, metric_name: str, direction: str
    ) -> list[str]:
        """Get recommendations for percentile-based anomalies."""
        recommendations = []

        if direction == "high":
            recommendations.append(f"Unusually high {metric_name} detected")
            recommendations.append("Check for system overload or configuration issues")
        else:
            recommendations.append(f"Unusually low {metric_name} detected")
            recommendations.append("Verify system is functioning correctly")

        return recommendations


class ThresholdDetector:
    """Simple threshold-based anomaly detection."""

    def __init__(self) -> None:
        """Initialize threshold detector."""
        self.thresholds: dict[str, dict[str, float]] = {}

    def set_threshold(
        self,
        metric_name: str,
        min_threshold: float | None = None,
        max_threshold: float | None = None,
    ) -> None:
        """Set thresholds for a metric."""
        self.thresholds[metric_name] = {}

        if min_threshold is not None:
            self.thresholds[metric_name]["min"] = min_threshold

        if max_threshold is not None:
            self.thresholds[metric_name]["max"] = max_threshold

    def detect_anomalies(self, metric_name: str, value: float) -> list[AnomalyEvent]:
        """Detect threshold violations."""
        anomalies = []

        thresholds = self.thresholds.get(metric_name, {})

        # Check minimum threshold
        if "min" in thresholds and value < thresholds["min"]:
            anomaly = AnomalyEvent(
                anomaly_id=f"thresh_min_{metric_name}_{int(_utc_now().timestamp())}",
                anomaly_type=AnomalyType.THRESHOLD_VIOLATION,
                severity=AlertSeverity.HIGH,
                metric_name=metric_name,
                actual_value=value,
                expected_value=thresholds["min"],
                deviation_score=self._normalized_deviation(
                    expected=thresholds["min"], actual=value
                ),
                timestamp=_utc_now(),
                description=f"{metric_name} value {value:.3f} below minimum threshold {thresholds['min']:.3f}",
                recommendations=[
                    f"Investigate why {metric_name} dropped below acceptable levels"
                ],
            )
            anomalies.append(anomaly)

        # Check maximum threshold
        if "max" in thresholds and value > thresholds["max"]:
            anomaly = AnomalyEvent(
                anomaly_id=f"thresh_max_{metric_name}_{int(_utc_now().timestamp())}",
                anomaly_type=AnomalyType.THRESHOLD_VIOLATION,
                severity=AlertSeverity.HIGH,
                metric_name=metric_name,
                actual_value=value,
                expected_value=thresholds["max"],
                deviation_score=self._normalized_deviation(
                    expected=thresholds["max"], actual=value
                ),
                timestamp=_utc_now(),
                description=f"{metric_name} value {value:.3f} exceeds maximum threshold {thresholds['max']:.3f}",
                recommendations=[
                    f"Investigate why {metric_name} exceeded acceptable levels"
                ],
            )
            anomalies.append(anomaly)

        return anomalies

    @staticmethod
    def _normalized_deviation(*, expected: float, actual: float) -> float:
        """Return normalized deviation while guarding against zero thresholds."""
        denominator = abs(expected)
        if denominator == 0:
            return abs(actual - expected)
        return abs(actual - expected) / denominator


class PerformanceMonitor:
    """Monitors optimization performance for degradation and anomalies.

    Thread-safe: All mutable state access is protected by a lock.
    """

    def __init__(self, degradation_threshold: float = 0.15) -> None:
        """Initialize performance monitor."""
        self.degradation_threshold = degradation_threshold
        self._lock = threading.Lock()
        self.performance_history: dict[str, list[tuple[datetime, float]]] = {}
        self.baseline_windows: dict[str, int] = {}

    def record_performance(
        self,
        function_name: str,
        algorithm: str,
        score: float,
        duration: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record performance metrics."""
        timestamp = _ensure_utc(timestamp)

        # Create composite keys for different metric types
        score_key = f"{function_name}:{algorithm}:score"
        duration_key = f"{function_name}:{algorithm}:duration"

        with self._lock:
            # Initialize history if needed
            for key in [score_key, duration_key]:
                if key not in self.performance_history:
                    self.performance_history[key] = []
                    self.baseline_windows[key] = 20  # Default baseline window

            # Record metrics
            self.performance_history[score_key].append((timestamp, score))
            self.performance_history[duration_key].append((timestamp, duration))

            # Keep only recent history (enforce memory limit)
            for key in [score_key, duration_key]:
                if len(self.performance_history[key]) > MAX_PERFORMANCE_HISTORY_SIZE:
                    items_to_keep = int(
                        MAX_PERFORMANCE_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO)
                    )
                    self.performance_history[key] = self.performance_history[key][
                        -items_to_keep:
                    ]

    def detect_performance_regression(
        self, function_name: str, algorithm: str, lookback_hours: int = 24
    ) -> list[AnomalyEvent]:
        """Detect performance regression."""
        anomalies = []

        score_key = f"{function_name}:{algorithm}:score"
        duration_key = f"{function_name}:{algorithm}:duration"

        # Analyze score regression
        score_anomalies = self._analyze_regression(
            score_key, "score", lookback_hours, higher_is_better=True
        )
        anomalies.extend(score_anomalies)

        # Analyze duration regression (performance slowdown)
        duration_anomalies = self._analyze_regression(
            duration_key, "duration", lookback_hours, higher_is_better=False
        )
        anomalies.extend(duration_anomalies)

        return anomalies

    def _analyze_regression(
        self,
        metric_key: str,
        metric_type: str,
        lookback_hours: int,
        higher_is_better: bool,
    ) -> list[AnomalyEvent]:
        """Analyze performance regression for a specific metric."""
        anomalies: list[AnomalyEvent] = []

        if metric_key not in self.performance_history:
            return anomalies

        history = self.performance_history[metric_key]
        if len(history) < 10:
            return anomalies  # Not enough data

        cutoff_time = datetime.now(UTC) - timedelta(hours=lookback_hours)
        recent_data = [(t, v) for t, v in history if t >= cutoff_time]

        if len(recent_data) < 5:
            return anomalies  # Not enough recent data

        # Split into baseline and recent periods
        baseline_window = self.baseline_windows.get(metric_key, 20)
        # Take a window immediately preceding recent_data; ensure slice bounds are valid
        if len(recent_data) >= len(history):
            baseline_data = []
        else:
            end_idx = len(history) - len(recent_data)
            start_idx = max(0, end_idx - baseline_window)
            baseline_data = history[start_idx:end_idx]

        if len(baseline_data) < 5:
            return anomalies  # Not enough baseline data

        # Calculate performance change
        baseline_values = [v for _, v in baseline_data]
        recent_values = [v for _, v in recent_data]

        baseline_avg = statistics.mean(baseline_values)
        recent_avg = statistics.mean(recent_values)

        if baseline_avg == 0:
            return anomalies

        change_ratio = (recent_avg - baseline_avg) / baseline_avg

        # Determine if this is a regression
        is_regression = False
        if higher_is_better and change_ratio < -self.degradation_threshold:
            is_regression = True
        elif not higher_is_better and change_ratio > self.degradation_threshold:
            is_regression = True

        if is_regression:
            severity = (
                AlertSeverity.CRITICAL
                if abs(change_ratio) > 0.3
                else AlertSeverity.HIGH
            )

            anomaly = AnomalyEvent(
                anomaly_id=f"regression_{metric_key}_{int(datetime.now(UTC).timestamp())}",
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=severity,
                metric_name=metric_key,
                actual_value=recent_avg,
                expected_value=baseline_avg,
                deviation_score=abs(change_ratio),
                timestamp=datetime.now(UTC),
                description=f"Performance regression detected: {metric_type} changed by {change_ratio:.1%} from baseline",
                recommendations=self._get_regression_recommendations(
                    metric_type, change_ratio
                ),
                context={
                    "baseline_period": f"{len(baseline_data)} data points",
                    "recent_period": f"{len(recent_data)} data points",
                    "change_percentage": change_ratio * 100,
                },
            )

            anomalies.append(anomaly)

        return anomalies

    def _get_regression_recommendations(
        self, metric_type: str, change_ratio: float
    ) -> list[str]:
        """Get recommendations for performance regression."""
        recommendations = []

        if metric_type == "score":
            recommendations.append(
                "Score degradation detected - investigate data quality and model configuration"
            )
            recommendations.append(
                "Check for recent changes to evaluation criteria or algorithms"
            )
            if change_ratio < -0.3:
                recommendations.append(
                    "Critical score drop - immediate investigation required"
                )

        elif metric_type == "duration":
            recommendations.append(
                "Performance slowdown detected - check system resources"
            )
            recommendations.append(
                "Review evaluation function complexity and optimization algorithms"
            )
            if change_ratio > 0.5:
                recommendations.append(
                    "Severe slowdown - check for resource constraints or inefficient algorithms"
                )

        recommendations.append("Compare recent optimizations with historical baselines")
        recommendations.append(
            "Consider rolling back recent changes if regression persists"
        )

        return recommendations


class RegressionDetector:
    """Specialized detector for performance regressions."""

    def __init__(self, sensitivity: float = 0.1) -> None:
        """Initialize regression detector."""
        self.sensitivity = sensitivity
        self.metric_histories: dict[
            str, list[tuple[datetime, dict[str, float | object]]]
        ] = {}

    def add_optimization_result(
        self,
        optimization_id: str,
        function_name: str,
        algorithm: str,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """Add optimization result for regression analysis."""
        timestamp = _ensure_utc(timestamp)

        key = f"{function_name}:{algorithm}"
        if key not in self.metric_histories:
            self.metric_histories[key] = []

        # Add metadata to metrics
        enhanced_metrics = {**metrics, "_optimization_id": optimization_id}
        self.metric_histories[key].append((timestamp, enhanced_metrics))

        # Keep only recent history (enforce memory limit)
        if len(self.metric_histories[key]) > MAX_REGRESSION_HISTORY_SIZE:
            items_to_keep = int(MAX_REGRESSION_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO))
            self.metric_histories[key] = self.metric_histories[key][-items_to_keep:]

    def detect_regressions(
        self, function_name: str, algorithm: str, analysis_window_hours: int = 48
    ) -> list[AnomalyEvent]:
        """Detect performance regressions across multiple metrics."""
        key = f"{function_name}:{algorithm}"

        if key not in self.metric_histories:
            return []

        history = self.metric_histories[key]
        if len(history) < 20:
            return []  # Need sufficient history

        cutoff_time = _utc_now() - timedelta(hours=analysis_window_hours)
        recent_data = [(t, m) for t, m in history if t >= cutoff_time]

        if len(recent_data) < 5:
            return []

        # Get baseline data (before the analysis window)
        baseline_data = [(t, m) for t, m in history if t < cutoff_time][
            -20:
        ]  # Last 20 baseline points

        if len(baseline_data) < 10:
            return []

        anomalies = []

        # Get all metric names (excluding metadata)
        all_metrics: set[str] = set()
        for _, metrics in baseline_data + recent_data:
            all_metrics.update(k for k in metrics.keys() if not k.startswith("_"))

        for metric_name in all_metrics:
            regression = self._analyze_metric_regression(
                metric_name, baseline_data, recent_data, function_name, algorithm
            )
            if regression:
                anomalies.append(regression)

        return anomalies

    def _analyze_metric_regression(
        self,
        metric_name: str,
        baseline_data: Sequence[tuple[datetime, Mapping[str, float | object]]],
        recent_data: Sequence[tuple[datetime, Mapping[str, float | object]]],
        function_name: str,
        algorithm: str,
    ) -> AnomalyEvent | None:
        """Analyze regression for a specific metric."""

        # Extract metric values
        baseline_values = [
            m.get(metric_name) for _, m in baseline_data if metric_name in m
        ]
        recent_values = [m.get(metric_name) for _, m in recent_data if metric_name in m]

        # Filter out None values and ensure float type
        baseline_float_values: list[float] = [
            float(v)
            for v in baseline_values
            if v is not None and isinstance(v, (int, float))
        ]
        recent_float_values: list[float] = [
            float(v)
            for v in recent_values
            if v is not None and isinstance(v, (int, float))
        ]

        if len(baseline_float_values) < 5 or len(recent_float_values) < 3:
            return None

        baseline_avg = statistics.mean(baseline_float_values)
        recent_avg = statistics.mean(recent_float_values)

        if baseline_avg == 0:
            return None

        change_ratio = (recent_avg - baseline_avg) / baseline_avg

        # Determine if this is significant regression based on metric type
        is_regression = self._is_significant_regression(metric_name, change_ratio)

        if is_regression:
            severity = self._calculate_regression_severity(metric_name, change_ratio)

            return AnomalyEvent(
                anomaly_id=f"metric_regression_{metric_name}_{int(_utc_now().timestamp())}",
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=severity,
                metric_name=f"{function_name}:{algorithm}:{metric_name}",
                actual_value=recent_avg,
                expected_value=baseline_avg,
                deviation_score=abs(change_ratio),
                timestamp=_utc_now(),
                description=f"Regression in {metric_name}: {change_ratio:.1%} change from baseline",
                recommendations=self._get_metric_regression_recommendations(
                    metric_name, change_ratio
                ),
                context={
                    "function_name": function_name,
                    "algorithm": algorithm,
                    "baseline_samples": len(baseline_float_values),
                    "recent_samples": len(recent_float_values),
                    "baseline_std": (
                        statistics.stdev(baseline_float_values)
                        if len(baseline_float_values) > 1
                        else 0
                    ),
                    "recent_std": (
                        statistics.stdev(recent_float_values)
                        if len(recent_float_values) > 1
                        else 0
                    ),
                },
            )

        return None

    def _is_significant_regression(self, metric_name: str, change_ratio: float) -> bool:
        """Determine if change represents significant regression."""

        # Different thresholds for different metric types
        if any(
            keyword in metric_name.lower()
            for keyword in ["accuracy", "score", "precision", "recall", "f1"]
        ):
            # Performance metrics - degradation is negative change
            return change_ratio < -self.sensitivity

        elif any(
            keyword in metric_name.lower()
            for keyword in ["error", "loss", "cost", "duration", "time"]
        ):
            # Cost/error metrics - degradation is positive change
            return change_ratio > self.sensitivity

        else:
            # Unknown metric - consider both directions
            return abs(change_ratio) > self.sensitivity

    def _calculate_regression_severity(
        self, metric_name: str, change_ratio: float
    ) -> AlertSeverity:
        """Calculate severity of regression."""

        abs_change = abs(change_ratio)

        if abs_change > 0.5:  # 50% change
            return AlertSeverity.CRITICAL
        elif abs_change > 0.3:  # 30% change
            return AlertSeverity.HIGH
        elif abs_change > 0.15:  # 15% change
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _get_metric_regression_recommendations(
        self, metric_name: str, change_ratio: float
    ) -> list[str]:
        """Get recommendations for metric regression."""

        recommendations = []

        if "accuracy" in metric_name.lower() or "score" in metric_name.lower():
            recommendations.append(
                "Accuracy regression detected - check data quality and model configuration"
            )
            recommendations.append(
                "Review evaluation dataset for changes or corruption"
            )

        elif "cost" in metric_name.lower():
            recommendations.append(
                "Cost increase detected - review resource usage and optimization efficiency"
            )
            recommendations.append(
                "Check for algorithm configuration changes that increase computational cost"
            )

        elif "duration" in metric_name.lower() or "time" in metric_name.lower():
            recommendations.append(
                "Performance slowdown detected - check system resources and algorithm efficiency"
            )
            recommendations.append(
                "Review recent changes to evaluation functions or optimization parameters"
            )

        else:
            recommendations.append(
                f"Regression detected in {metric_name} - investigate recent changes"
            )

        # Add severity-based recommendations
        if abs(change_ratio) > 0.3:
            recommendations.append(
                "Severe regression - consider immediate rollback of recent changes"
            )

        return recommendations


class AlertManager:
    """Manages anomaly alerts and notifications.

    Thread-safe: All mutable state access is protected by a lock.
    """

    def __init__(self, max_alerts_per_hour: int = 50) -> None:
        """Initialize alert manager."""
        self.max_alerts_per_hour = max_alerts_per_hour
        self._lock = threading.Lock()
        self.alert_history: list[AnomalyEvent] = []
        self.alert_rules: list[MonitoringRule] = []
        self.suppression_rules: dict[str, datetime] = {}
        self.notification_callbacks: list[Callable[[AnomalyEvent], None]] = []

    def add_monitoring_rule(self, rule: MonitoringRule) -> None:
        """Add monitoring rule."""
        with self._lock:
            self.alert_rules.append(rule)
        logger.info(f"Added monitoring rule: {rule.rule_id}")

    def add_notification_callback(
        self, callback: Callable[[AnomalyEvent], None]
    ) -> None:
        """Add notification callback function."""
        with self._lock:
            self.notification_callbacks.append(callback)

    def process_anomaly(self, anomaly: AnomalyEvent) -> bool:
        """Process and potentially send alert for anomaly."""
        anomaly.timestamp = _ensure_utc(anomaly.timestamp)

        with self._lock:
            # Check if alert should be suppressed
            if self._is_suppressed(anomaly):
                logger.debug(f"Alert suppressed: {anomaly.anomaly_id}")
                return False

            # Check rate limiting
            if self._is_rate_limited():
                logger.warning("Alert rate limit exceeded")
                return False

            # Add to history
            self.alert_history.append(anomaly)

            # Enforce memory limit on alert history
            if len(self.alert_history) > MAX_ALERT_HISTORY_SIZE:
                items_to_keep = int(MAX_ALERT_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO))
                self.alert_history = self.alert_history[-items_to_keep:]
                logger.debug(
                    f"Pruned alert history to {items_to_keep} items to stay within "
                    f"memory limit of {MAX_ALERT_HISTORY_SIZE}"
                )

            # Get a copy of callbacks to call outside the lock
            callbacks = list(self.notification_callbacks)

        # Send notifications outside the lock to avoid deadlock
        for callback in callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

        logger.info(
            f"Alert processed: {anomaly.anomaly_type.value} for {anomaly.metric_name}"
        )
        return True

    def get_recent_alerts(
        self, hours: int = 24, severity: AlertSeverity | None = None
    ) -> list[AnomalyEvent]:
        """Get recent alerts."""
        cutoff_time = _utc_now() - timedelta(hours=hours)

        with self._lock:
            filtered_alerts = [
                alert for alert in self.alert_history if alert.timestamp >= cutoff_time
            ]

            if severity:
                filtered_alerts = [
                    alert for alert in filtered_alerts if alert.severity == severity
                ]

        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get summary of alerts."""
        recent_alerts = self.get_recent_alerts(hours)

        if not recent_alerts:
            return {
                "total_alerts": 0,
                "period_hours": hours,
                "summary": "No alerts in the specified period",
            }

        # Count by severity
        severity_counts: dict[str, int] = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1

        # Count by type
        type_counts: dict[str, int] = defaultdict(int)
        for alert in recent_alerts:
            type_counts[alert.anomaly_type.value] += 1

        # Find most common metrics with issues
        metric_counts: dict[str, int] = defaultdict(int)
        for alert in recent_alerts:
            metric_counts[alert.metric_name] += 1

        most_problematic_metrics = sorted(
            metric_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_alerts": len(recent_alerts),
            "period_hours": hours,
            "severity_breakdown": dict(severity_counts),
            "type_breakdown": dict(type_counts),
            "most_problematic_metrics": most_problematic_metrics,
            "latest_alert": (
                recent_alerts[0].timestamp.isoformat() if recent_alerts else None
            ),
            "critical_alerts": len(
                [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
            ),
        }

    def _is_suppressed(self, anomaly: AnomalyEvent) -> bool:
        """Check if anomaly should be suppressed."""

        # Check metric-specific suppression
        metric_key = f"metric:{anomaly.metric_name}"
        if metric_key in self.suppression_rules:
            if _utc_now() < self.suppression_rules[metric_key]:
                return True

        # Check type-specific suppression
        type_key = f"type:{anomaly.anomaly_type.value}"
        if type_key in self.suppression_rules:
            if _utc_now() < self.suppression_rules[type_key]:
                return True

        return False

    def _is_rate_limited(self) -> bool:
        """Check if alerts are rate limited."""
        one_hour_ago = _utc_now() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history if alert.timestamp >= one_hour_ago
        ]

        return len(recent_alerts) >= self.max_alerts_per_hour

    def suppress_alerts(
        self,
        metric_name: str | None = None,
        anomaly_type: AnomalyType | None = None,
        duration_minutes: int = 60,
    ) -> None:
        """Suppress alerts for specified criteria."""

        suppression_end = _utc_now() + timedelta(minutes=duration_minutes)

        if metric_name:
            key = f"metric:{metric_name}"
            self.suppression_rules[key] = suppression_end
            logger.info(
                f"Suppressed alerts for metric {metric_name} until {suppression_end}"
            )

        if anomaly_type:
            key = f"type:{anomaly_type.value}"
            self.suppression_rules[key] = suppression_end
            logger.info(
                f"Suppressed alerts for type {anomaly_type.value} until {suppression_end}"
            )


class AnomalyDetector:
    """Main anomaly detection engine coordinating all detection methods."""

    def __init__(self) -> None:
        """Initialize anomaly detector."""
        self.statistical_detector = StatisticalDetector()
        self.threshold_detector = ThresholdDetector()
        self.performance_monitor = PerformanceMonitor()
        self.regression_detector = RegressionDetector()
        self.alert_manager = AlertManager()

        # Setup default monitoring rules
        self._setup_default_rules()

        logger.info("AnomalyDetector initialized with all detection engines")

    def monitor_metric(
        self, metric_name: str, value: float, timestamp: datetime | None = None
    ) -> list[AnomalyEvent]:
        """Monitor a metric value for anomalies."""
        timestamp = _ensure_utc(timestamp)

        all_anomalies = []

        # Statistical detection
        self.statistical_detector.add_data_point(metric_name, value, timestamp)
        stat_anomalies = self.statistical_detector.detect_anomalies(metric_name, value)
        all_anomalies.extend(stat_anomalies)

        # Threshold detection
        threshold_anomalies = self.threshold_detector.detect_anomalies(
            metric_name, value
        )
        all_anomalies.extend(threshold_anomalies)

        # Process alerts
        for anomaly in all_anomalies:
            self.alert_manager.process_anomaly(anomaly)

        return all_anomalies

    def monitor_optimization(
        self,
        optimization_id: str,
        function_name: str,
        algorithm: str,
        score: float,
        duration: float,
        cost: float | None = None,
        additional_metrics: dict[str, float] | None = None,
        timestamp: datetime | None = None,
    ) -> list[AnomalyEvent]:
        """Monitor optimization performance for anomalies."""
        timestamp = _ensure_utc(timestamp)

        all_anomalies = []

        # Record in performance monitor
        self.performance_monitor.record_performance(
            function_name, algorithm, score, duration, timestamp
        )

        # Check for performance regression
        regression_anomalies = self.performance_monitor.detect_performance_regression(
            function_name, algorithm
        )
        all_anomalies.extend(regression_anomalies)

        # Record in regression detector
        metrics = {"score": score, "duration": duration}
        if cost is not None:
            metrics["cost"] = cost
        if additional_metrics:
            metrics.update(additional_metrics)

        self.regression_detector.add_optimization_result(
            optimization_id, function_name, algorithm, metrics, timestamp
        )

        # Check for metric regressions
        metric_regressions = self.regression_detector.detect_regressions(
            function_name, algorithm
        )
        all_anomalies.extend(metric_regressions)

        # Monitor individual metrics
        for metric_name, metric_value in metrics.items():
            if metric_name != "_optimization_id":
                metric_key = f"{function_name}:{algorithm}:{metric_name}"
                metric_anomalies = self.monitor_metric(
                    metric_key, metric_value, timestamp
                )
                all_anomalies.extend(metric_anomalies)

        return all_anomalies

    def set_metric_threshold(
        self,
        metric_name: str,
        min_threshold: float | None = None,
        max_threshold: float | None = None,
    ) -> None:
        """Set thresholds for metric monitoring."""
        self.threshold_detector.set_threshold(metric_name, min_threshold, max_threshold)
        logger.info(
            f"Set thresholds for {metric_name}: min={min_threshold}, max={max_threshold}"
        )

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health based on recent anomalies."""

        recent_alerts = self.alert_manager.get_recent_alerts(hours=1)
        alert_summary = self.alert_manager.get_alert_summary(hours=24)

        # Determine health status
        critical_alerts = len(
            [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
        )
        high_alerts = len(
            [a for a in recent_alerts if a.severity == AlertSeverity.HIGH]
        )

        if critical_alerts > 0:
            health_status = "critical"
        elif high_alerts > 3:
            health_status = "degraded"
        elif len(recent_alerts) > 10:
            health_status = "warning"
        else:
            health_status = "healthy"

        return {
            "health_status": health_status,
            "recent_alerts_1h": len(recent_alerts),
            "critical_alerts_24h": alert_summary.get("critical_alerts", 0),
            "alert_summary_24h": alert_summary,
            "monitoring_active": True,
            "last_check": _utc_now().isoformat(),
        }

    def _setup_default_rules(self) -> None:
        """Setup default monitoring rules."""

        # Default rules for common metrics
        default_rules = [
            MonitoringRule(
                rule_id="cost_spike",
                metric_name="cost",
                rule_type="statistical",
                parameters={"z_threshold": 3.0},
                severity=AlertSeverity.HIGH,
            ),
            MonitoringRule(
                rule_id="duration_anomaly",
                metric_name="duration",
                rule_type="statistical",
                parameters={"z_threshold": 2.5},
                severity=AlertSeverity.MEDIUM,
            ),
            MonitoringRule(
                rule_id="score_degradation",
                metric_name="score",
                rule_type="regression",
                parameters={"degradation_threshold": 0.1},
                severity=AlertSeverity.HIGH,
            ),
        ]

        for rule in default_rules:
            self.alert_manager.add_monitoring_rule(rule)
