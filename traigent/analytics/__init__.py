"""Advanced analytics and intelligence module for Traigent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Maintainability

from __future__ import annotations

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
]
