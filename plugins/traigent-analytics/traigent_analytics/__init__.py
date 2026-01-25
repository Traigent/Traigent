"""Advanced analytics and intelligence plugin for Traigent SDK.

This plugin provides advanced AI analytics capabilities including:
- Meta-learning for algorithm selection
- Predictive analytics for cost and performance forecasting
- Anomaly detection for performance regression detection
- Cost optimization and budget allocation
- Smart scheduling for job management
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Maintainability

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traigent.plugins.registry import PluginInterface

# Anomaly detection
from .anomaly import (
    AlertManager,
    AlertSeverity,
    AnomalyDetector,
    AnomalyType,
    PerformanceMonitor,
    RegressionDetector,
)

# Cost optimization (from split module)
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

# Main AI intelligence (refactored)
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

# Scheduling (from split module)
from .scheduling import (
    JobPriority,
    ScheduledJob,
    ScheduleType,
    ScheduleWindow,
    SchedulingPolicy,
    SmartScheduler,
)

__all__ = [
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
    # Plugin
    "AnalyticsPlugin",
]


class AnalyticsPlugin:
    """Traigent Analytics Plugin.

    Provides advanced AI analytics capabilities for optimization workflows.
    """

    name = "analytics"
    version = "0.1.0"
    min_traigent_version = "0.9.0"
    features = [
        "meta_learning",
        "predictive_analytics",
        "anomaly_detection",
        "cost_optimization",
        "scheduling",
    ]

    @classmethod
    def initialize(cls) -> None:
        """Initialize the analytics plugin."""
        # Analytics is standalone, no initialization needed
        pass

    @classmethod
    def cleanup(cls) -> None:
        """Cleanup analytics plugin resources."""
        pass

    @classmethod
    def get_capabilities(cls) -> dict:
        """Return plugin capabilities."""
        return {
            "meta_learning": True,
            "predictive_analytics": True,
            "anomaly_detection": True,
            "cost_optimization": True,
            "scheduling": True,
        }
