"""Backward-compatible analytics API.

.. deprecated:: 0.9.0
    The embedded implementation is deprecated. It remains available through
    ``traigent.analytics`` in the ``traigent`` package.

When an optional ``traigent_analytics`` module is already available, this
module prefers its implementations. Otherwise, it uses the embedded
implementation. No separately published replacement package is required for
the embedded path.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Maintainability

from __future__ import annotations

import warnings as _warnings

_PLUGIN_AVAILABLE = False

# Try to import from the plugin first
try:
    from traigent_analytics import (  # Anomaly detection; Cost optimization; Historical analytics; Predictive analytics; Scheduling
        AlertManager,
        AlertSeverity,
        AlgorithmSelector,
        AnomalyDetector,
        AnomalyType,
        BudgetAllocation,
        BudgetAllocator,
        CostForecaster,
        CostOptimizationAction,
        CostOptimizationAI,
        HistoricalAnalyticsEngine,
        JobPriority,
        OptimizationHistory,
        OptimizationStrategy,
        PerformanceForecaster,
        PerformanceMonitor,
        PerformancePredictor,
        PredictiveAnalytics,
        Priority,
        RegressionDetector,
        ResourceOptimizer,
        ResourceType,
        ResourceUsage,
        ScheduledJob,
        ScheduleType,
        ScheduleWindow,
        SchedulingPolicy,
        SmartScheduler,
        TrendAnalyzer,
        UsageMetric,
    )

    _PLUGIN_AVAILABLE = True

except ImportError:
    # Plugin not installed, use embedded implementation (deprecated)
    # Example insights
    # Anomaly detection
    from .anomaly import (
        AlertManager,
        AlertSeverity,
        AnomalyDetector,
        AnomalyType,
        PerformanceMonitor,
        RegressionDetector,
    )

    # Cost optimization
    from .cost_optimization import (
        BudgetAllocation,
        BudgetAllocator,
        CostOptimizationAction,
        OptimizationStrategy,
        Priority,
        ResourceOptimizer,
        ResourceType,
        ResourceUsage,
    )
    from .example_insights import ExampleInsightsClient

    # Main AI intelligence
    from .intelligence import CostOptimizationAI

    # Historical analytics
    from .meta_learning import (
        AlgorithmSelector,
        HistoricalAnalyticsEngine,
        OptimizationHistory,
        PerformancePredictor,
    )
    from .optimization_plan import OptimizationPlanClient

    # Predictive analytics
    from .predictive import (
        CostForecaster,
        PerformanceForecaster,
        PredictiveAnalytics,
        TrendAnalyzer,
        UsageMetric,
    )

    # Scheduling
    from .scheduling import (
        JobPriority,
        ScheduledJob,
        ScheduleType,
        ScheduleWindow,
        SchedulingPolicy,
        SmartScheduler,
    )

    _warnings.warn(
        "traigent.analytics embedded implementation is deprecated. "
        "It remains available through traigent.analytics in the traigent package.",
        DeprecationWarning,
        stacklevel=2,
    )


def is_plugin_installed() -> bool:
    """Check if the analytics plugin is installed."""
    return _PLUGIN_AVAILABLE


__all__ = [
    # Example Insights
    "ExampleInsightsClient",
    "OptimizationPlanClient",
    # Historical analytics
    "HistoricalAnalyticsEngine",
    "OptimizationHistory",
    "AlgorithmSelector",
    "PerformancePredictor",
    # Predictive Analytics
    "PredictiveAnalytics",
    "CostForecaster",
    "PerformanceForecaster",
    "TrendAnalyzer",
    # Anomaly Detection
    "AnomalyDetector",
    "PerformanceMonitor",
    "RegressionDetector",
    "AlertManager",
    # Cost Optimization
    "CostOptimizationAI",
    "BudgetAllocation",
    "BudgetAllocator",
    "ResourceOptimizer",
    "ResourceType",
    "ResourceUsage",
    "OptimizationStrategy",
    "CostOptimizationAction",
    "Priority",
    # Scheduling
    "SmartScheduler",
    "ScheduledJob",
    "ScheduleType",
    "ScheduleWindow",
    "SchedulingPolicy",
    "JobPriority",
    # Types and Enums
    "UsageMetric",
    "AlertSeverity",
    "AnomalyType",
    # Helpers
    "is_plugin_installed",
]
