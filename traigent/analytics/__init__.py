"""Backward compatibility shim for traigent.analytics.

.. deprecated:: 0.9.0
    This module is deprecated. Install and use ``traigent-analytics`` plugin instead:
    ``pip install traigent-analytics`` or ``pip install traigent[analytics]``

The analytics functionality has been moved to the traigent-analytics plugin
for better modularity. This shim provides backward compatibility by:
1. Trying to import from traigent-analytics plugin (preferred)
2. Falling back to embedded implementation (deprecated)
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Maintainability

from __future__ import annotations

import warnings as _warnings

_PLUGIN_AVAILABLE = False

# Try to import from the plugin first
try:
    from traigent_analytics import (  # Anomaly detection; Cost optimization; Meta-learning; Predictive analytics; Scheduling
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
        JobPriority,
        MetaLearningEngine,
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

    # Meta-learning
    from .meta_learning import (
        AlgorithmSelector,
        MetaLearningEngine,
        OptimizationHistory,
        PerformancePredictor,
    )

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
        "Install traigent-analytics plugin for better support: "
        "pip install traigent-analytics",
        DeprecationWarning,
        stacklevel=2,
    )


def is_analytics_available() -> bool:
    """Check if analytics is available (either via plugin or embedded)."""
    return True  # Always available through embedded fallback


def is_plugin_installed() -> bool:
    """Check if the analytics plugin is installed."""
    return _PLUGIN_AVAILABLE


__all__ = [
    # Example Insights
    "ExampleInsightsClient",
    # Meta-learning
    "MetaLearningEngine",
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
    "is_analytics_available",
    "is_plugin_installed",
]
