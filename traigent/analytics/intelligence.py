"""AI-powered cost optimization and intelligent resource management.

This is the refactored version of intelligence.py with classes split into
separate modules for better maintainability.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Performance FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, cast

from ..core.constants import (
    HISTORY_PRUNE_RATIO,
    MAX_OPTIMIZATION_HISTORY_SIZE,
    MAX_USAGE_HISTORY_SIZE,
)
from ..utils.logging import get_logger

# Import from split modules
from .cost_optimization import (
    BudgetAllocator,
    CostOptimizationAction,
    OptimizationStrategy,
    ResourceOptimizer,
    ResourceType,
    ResourceUsage,
)
from .meta_learning import OptimizationHistory
from .scheduling import (
    ScheduleType,
    SmartScheduler,
)

logger = get_logger(__name__)


class CostAnalysisEngine:
    """Handles cost analysis, trend detection, and anomaly identification.

    Thread-safe: All mutable state access is protected by a lock.
    """

    def __init__(self, anomaly_threshold: float = 2.0) -> None:
        self.anomaly_threshold = anomaly_threshold
        self._lock = threading.Lock()
        self.usage_history: dict[ResourceType, list[ResourceUsage]] = defaultdict(list)
        self.cost_history: list[float] = []
        self._max_history_items = MAX_USAGE_HISTORY_SIZE

    def analyze_current_state(self) -> dict[str, Any]:
        """Analyze current cost and usage state."""
        state = {
            "resource_analysis": {},
            "cost_summary": {},
            "trends": {},
            "anomalies": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        with self._lock:
            # Analyze each resource type
            for resource_type, usage_list in self.usage_history.items():
                if not usage_list:
                    continue

                recent_usage = usage_list[-30:]  # Last 30 entries
                costs = [u.total_cost for u in recent_usage]
                utilizations = [u.utilization_percent for u in recent_usage]

                state["resource_analysis"][resource_type.value] = {
                    "avg_cost": sum(costs) / len(costs) if costs else 0,
                    "avg_utilization": (
                        sum(utilizations) / len(utilizations) if utilizations else 0
                    ),
                    "trend": self._calculate_trend(costs),
                    "total_usage_points": len(recent_usage),
                    "cost_variance": self._calculate_variance(costs),
                }

            # Overall cost summary
            if self.cost_history:
                recent_costs = self.cost_history[-30:]
                state["cost_summary"] = {
                    "total_cost": sum(recent_costs),
                    "avg_daily_cost": sum(recent_costs) / len(recent_costs),
                    "trend": self._calculate_trend(recent_costs),
                    "anomalies": self._detect_anomalies(recent_costs),
                }

        return state

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple trend calculation
        recent_avg = sum(values[-5:]) / min(5, len(values))
        older_avg = sum(values[:5]) / min(5, len(values))

        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _detect_anomalies(self, values: list[float]) -> list[int]:
        """Detect anomalies in cost values using statistical method."""
        if len(values) < 3:
            return []

        mean = sum(values) / len(values)
        variance = self._calculate_variance(values)
        std_dev = variance**0.5 if variance > 0 else 0

        anomalies = []
        for i, value in enumerate(values):
            if std_dev > 0 and abs(value - mean) > (self.anomaly_threshold * std_dev):
                anomalies.append(i)

        return anomalies


class OptimizationPlanManager:
    """Manages optimization plan creation and execution."""

    def __init__(
        self, strategy: OptimizationStrategy, min_confidence: float = 0.7
    ) -> None:
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.budget_allocator = BudgetAllocator()
        self.resource_optimizer = ResourceOptimizer()

    def _create_implementation_timeline(
        self, actions: list[CostOptimizationAction]
    ) -> list[dict[str, Any]]:
        """Create implementation timeline for optimization actions."""
        if not actions:
            return []

        timeline = []
        current_date = datetime.utcnow()

        # Sort actions by priority and timeline
        sorted_actions = sorted(
            actions, key=lambda a: (a.priority.value, a.timeline_days)
        )

        for action in sorted_actions:
            start_date = current_date
            end_date = current_date + timedelta(days=action.timeline_days)

            timeline.append(
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_days": action.timeline_days,
                    "priority": action.priority.value,
                    "dependencies": [],  # Could be enhanced with actual dependencies
                }
            )

            # Update current date for sequential planning
            current_date = end_date + timedelta(days=1)  # Add buffer day

        return timeline

    def _assess_implementation_risks(
        self, actions: list[CostOptimizationAction]
    ) -> dict[str, Any]:
        """Assess risks associated with implementing optimization actions."""
        if not actions:
            return {"overall_risk": "low", "risk_factors": []}

        risk_factors = []
        high_risk_count = 0
        medium_risk_count = 0

        for action in actions:
            if action.risk_level == "high":
                high_risk_count += 1
                risk_factors.append(
                    {
                        "action_id": action.action_id,
                        "risk_type": "implementation",
                        "risk_level": "high",
                        "description": f"High-risk implementation: {action.description}",
                        "mitigation": "Thorough testing and gradual rollout recommended",
                    }
                )
            elif action.risk_level == "medium":
                medium_risk_count += 1

        # Calculate overall risk
        total_actions = len(actions)
        if high_risk_count > total_actions * 0.3:  # > 30% high risk
            overall_risk = "high"
        elif (
            high_risk_count + medium_risk_count
        ) > total_actions * 0.5:  # > 50% medium+
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "high_risk_actions": high_risk_count,
            "medium_risk_actions": medium_risk_count,
            "low_risk_actions": total_actions - high_risk_count - medium_risk_count,
        }


class CostOptimizationAI:
    """AI-powered cost optimization engine.

    This is the main orchestrator that combines budget allocation,
    resource optimization, and smart scheduling capabilities.
    """

    def __init__(
        self,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        min_confidence: float = 0.7,
        anomaly_threshold: float = 2.0,
        enable_ml: bool = False,
    ) -> None:
        """Initialize cost optimization AI."""
        # Initialize extracted components
        self.analysis_engine = CostAnalysisEngine(anomaly_threshold)
        self.plan_manager = OptimizationPlanManager(
            optimization_strategy, min_confidence
        )
        self.budget_allocator = self.plan_manager.budget_allocator
        self.resource_optimizer = self.plan_manager.resource_optimizer

        # Store configuration
        self.strategy = optimization_strategy
        self.min_confidence = min_confidence
        self.anomaly_threshold = anomaly_threshold
        self.enable_ml = enable_ml

        # Initialize remaining components
        self.optimization_history = OptimizationHistory()
        self.scheduler = SmartScheduler()
        self.optimization_results: list[dict[str, Any]] = []

        # Memory bounds to prevent unbounded growth
        self._max_optimization_results = MAX_OPTIMIZATION_HISTORY_SIZE

        # ML models (if enabled)
        self.cost_predictor = None
        self.usage_predictor = None
        self.tracked_results: dict[str, Any] = {}
        if enable_ml:
            self._initialize_ml_models()

    def _initialize_ml_models(self) -> None:
        """Initialize ML models for prediction."""
        try:
            # Placeholder for actual ML model initialization
            logger.info("ML models initialized for cost optimization")
        except Exception as e:
            logger.warning(f"Failed to initialize ML models: {e}")
            self.enable_ml = False

    @property
    def usage_history(self):
        """Access to usage history through analysis engine."""
        return self.analysis_engine.usage_history

    @property
    def cost_history(self):
        """Access to cost history through analysis engine."""
        return self.analysis_engine.cost_history

    def analyze_current_state(self) -> dict[str, Any]:
        """Analyze current resource usage and cost state."""
        # Delegate to analysis engine
        return self.analysis_engine.analyze_current_state()

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend from historical values."""
        if len(values) < 3:
            return "insufficient_data"

        # Simple trend detection using linear regression
        n = len(values)
        if n == 0:
            return "no_data"

        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Categorize trend
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _detect_anomalies(self, values: list[float]) -> list[int]:
        """Detect anomalies in values using statistical methods."""
        if len(values) < 10:
            return []

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
            if z_score > self.anomaly_threshold:
                anomalies.append(i)

        return anomalies

    def generate_optimization_plan(
        self,
        current_state: dict[str, Any],
        budget_constraint: float | None = None,
        time_constraint: timedelta | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive optimization plan."""
        plan: dict[str, Any] = {
            "strategy": self.strategy.value,
            "actions": [],
            "estimated_savings": 0,
            "implementation_timeline": [],
            "risk_assessment": {},
            "budget_reallocation": {},
            "scheduling_recommendations": {},
        }

        # Generate and prioritize optimization actions
        plan["actions"] = self._generate_optimization_actions(current_state)

        # Apply constraints
        plan["actions"] = self._apply_plan_constraints(
            plan["actions"], budget_constraint, time_constraint
        )

        # Calculate savings and generate additional plan components
        plan["estimated_savings"] = sum(
            a.estimated_savings_absolute for a in plan["actions"]
        )

        plan["implementation_timeline"] = self._create_implementation_timeline(
            plan["actions"]
        )
        plan["risk_assessment"] = self._assess_implementation_risks(plan["actions"])
        plan["budget_reallocation"] = self._generate_budget_reallocation(current_state)
        plan["scheduling_recommendations"] = self._generate_scheduling_recommendations()

        return plan

    def _generate_optimization_actions(
        self, current_state: dict[str, Any]
    ) -> list[CostOptimizationAction]:
        """Generate optimization actions for each resource type."""
        actions = []

        # Generate optimization actions for each resource
        for resource_type in ResourceType:
            if resource_type.value in current_state.get("resource_analysis", {}):
                patterns = self.resource_optimizer.analyze_usage_patterns(resource_type)
                if patterns:
                    resource_actions = (
                        self.resource_optimizer.generate_optimization_actions(
                            resource_type, patterns, self.strategy
                        )
                    )
                    actions.extend(resource_actions)

        # Sort actions by priority and ROI
        actions.sort(key=lambda a: (a.priority.value, -a.estimated_savings_percent))

        return actions

    def _apply_plan_constraints(
        self,
        actions: list[CostOptimizationAction],
        budget_constraint: float | None,
        time_constraint: timedelta | None,
    ) -> list[CostOptimizationAction]:
        """Apply budget and time constraints to optimization actions."""
        filtered_actions = actions

        # Apply budget constraint
        if budget_constraint:
            affordable_actions = []
            remaining_budget = budget_constraint

            for action in filtered_actions:
                implementation_cost = (
                    action.estimated_savings_absolute * 0.1
                )  # Assume 10% implementation cost
                if implementation_cost <= remaining_budget:
                    affordable_actions.append(action)
                    remaining_budget -= implementation_cost

            filtered_actions = affordable_actions

        # Apply time constraint
        if time_constraint:
            total_days = time_constraint.days
            filtered_actions = [
                a for a in filtered_actions if a.timeline_days <= total_days
            ]

        return filtered_actions

    def _generate_budget_reallocation(
        self, current_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate budget reallocation recommendations."""
        if not current_state.get("resource_analysis"):
            return {}

        priorities = {
            ResourceType[k.upper()]: (1.0 / v["avg_cost"] if v["avg_cost"] > 0 else 1.0)
            for k, v in current_state["resource_analysis"].items()
            if k.upper() in ResourceType.__members__
        }

        historical_usage = {
            ResourceType[k.upper()]: [
                u.total_cost
                for u in self.usage_history.get(ResourceType[k.upper()], [])
            ]
            for k in current_state["resource_analysis"].keys()
            if k.upper() in ResourceType.__members__
        }

        total_budget = sum(
            v["avg_cost"] * 30  # Monthly budget estimate
            for v in current_state["resource_analysis"].values()
        )

        return self.budget_allocator.allocate_budget(
            total_budget, priorities, historical_usage
        )

    def _create_implementation_timeline(
        self, actions: list[CostOptimizationAction]
    ) -> list[dict[str, Any]]:
        """Create implementation timeline for optimization actions."""
        return self.plan_manager._create_implementation_timeline(actions)

    def _assess_implementation_risks(
        self, actions: list[CostOptimizationAction]
    ) -> dict[str, Any]:
        """Assess risks associated with implementing optimization actions."""
        return self.plan_manager._assess_implementation_risks(actions)

    def _generate_scheduling_recommendations(self) -> dict[str, Any]:
        """Generate job scheduling recommendations."""
        recommendations = {
            "optimal_schedule_type": ScheduleType.ADAPTIVE.value,
            "policy_adjustments": {},
            "cost_saving_windows": [],
            "batch_opportunities": [],
        }

        # Analyze historical patterns if available
        if self.scheduler.completed_jobs:
            schedule_optimization = self.scheduler.optimize_future_schedules()

            if "optimal_hours" in schedule_optimization:
                recommendations["cost_saving_windows"] = [
                    {
                        "hours": schedule_optimization["optimal_hours"],
                        "reason": "historically lower costs",
                    }
                ]

            if "duration_adjustment_factor" in schedule_optimization:
                recommendations["policy_adjustments"]["duration_estimates"] = {
                    "current_accuracy": schedule_optimization[
                        "duration_adjustment_factor"
                    ],
                    "recommendation": (
                        "adjust estimates by factor"
                        if schedule_optimization["duration_adjustment_factor"] != 1.0
                        else "estimates are accurate"
                    ),
                }

        # Identify batching opportunities
        pending_jobs = self.scheduler.job_queue
        job_types = defaultdict(list)
        for job in pending_jobs:
            job_types[job.job_type].append(job)

        for job_type, jobs in job_types.items():
            if len(jobs) > 3:
                recommendations["batch_opportunities"].append(
                    {
                        "job_type": job_type,
                        "count": len(jobs),
                        "potential_savings": len(jobs)
                        * 0.1,  # 10% savings per job when batched
                    }
                )

        return recommendations

    def _analyze_cost_trends(self, usage_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze cost trends over time."""
        if not usage_data:
            return {"overall": "no_data", "recent": "no_data"}

        # Sort by timestamp
        sorted_data = sorted(usage_data, key=lambda x: x.get("timestamp", 0))
        costs = [item.get("cost", 0) for item in sorted_data]

        # Overall trend
        overall_trend = self._calculate_trend(costs)

        # Recent trend (last 20% of data)
        recent_count = max(1, len(costs) // 5)
        recent_trend = self._calculate_trend(costs[-recent_count:])

        return {
            "overall": overall_trend,
            "recent": recent_trend,
            "change_rate": (
                (costs[-1] - costs[0]) / costs[0] if costs and costs[0] > 0 else 0
            ),
        }

    def execute_optimization(
        self, action: CostOptimizationAction, dry_run: bool = False
    ) -> dict[str, Any]:
        """Execute a specific optimization action."""
        notes: list[str] = []
        result: dict[str, Any] = {
            "action_id": action.action_id,
            "status": "pending",
            "actual_savings": 0.0,
            "execution_time": None,
            "notes": notes,
        }

        if dry_run:
            result["status"] = "dry_run"
            notes.append("Dry run mode - no actual changes made")
            return result

        # Log execution start
        start_time = datetime.utcnow()
        logger.info(f"Executing optimization action: {action.action_id}")

        try:
            # Simulate execution based on action type
            if action.action_type == "rightsize":
                notes.append(f"Rightsizing {action.resource_type.value} resources")
                result["actual_savings"] = (
                    action.estimated_savings_absolute * 0.8
                )  # 80% of estimate

            elif action.action_type == "cost_control":
                notes.append(
                    f"Implementing cost controls for {action.resource_type.value}"
                )
                result["actual_savings"] = (
                    action.estimated_savings_absolute * 0.9
                )  # 90% of estimate

            elif action.action_type == "load_balance":
                notes.append(
                    f"Setting up load balancing for {action.resource_type.value}"
                )
                result["actual_savings"] = (
                    action.estimated_savings_absolute * 0.7
                )  # 70% of estimate

            result["status"] = "completed"
            result["execution_time"] = (datetime.utcnow() - start_time).total_seconds()

            # Track savings
            self.resource_optimizer.track_savings(
                action, cast(float, result["actual_savings"])
            )

        except Exception as e:
            result["status"] = "failed"
            notes.append(f"Error: {str(e)}")
            logger.error(f"Failed to execute action {action.action_id}: {e}")

        # Store result with bounds checking
        if len(self.optimization_results) >= self._max_optimization_results:
            # Remove oldest results using prune ratio
            items_to_keep = int(self._max_optimization_results * (1 - HISTORY_PRUNE_RATIO))
            items_to_remove = len(self.optimization_results) - items_to_keep
            self.optimization_results = self.optimization_results[-items_to_keep:]
            logger.debug(
                f"Pruned {items_to_remove} old optimization results to stay within memory limit"
            )

        self.optimization_results.append(result)

        return result

    def monitor_optimization_progress(self) -> dict[str, Any]:
        """Monitor progress of ongoing optimizations."""
        progress = {
            "total_actions": len(self.optimization_results),
            "completed_actions": sum(
                1 for r in self.optimization_results if r["status"] == "completed"
            ),
            "failed_actions": sum(
                1 for r in self.optimization_results if r["status"] == "failed"
            ),
            "total_savings_achieved": sum(
                r.get("actual_savings", 0) for r in self.optimization_results
            ),
            "average_execution_time": 0.0,
            "success_rate": 0,
            "savings_by_type": defaultdict(float),
        }

        if progress["total_actions"] > 0:
            progress["success_rate"] = (
                progress["completed_actions"] / progress["total_actions"] * 100
            )

            execution_times = [
                r["execution_time"]
                for r in self.optimization_results
                if r.get("execution_time") is not None
            ]
            if execution_times:
                progress["average_execution_time"] = statistics.mean(execution_times)

        # Group savings by action type
        for result in self.optimization_results:
            if result["status"] == "completed" and "actual_savings" in result:
                # Extract action type from action_id
                action_type = (
                    result["action_id"].split("_")[1]
                    if "_" in result["action_id"]
                    else "unknown"
                )
                progress["savings_by_type"][action_type] += result["actual_savings"]

        return progress

    def generate_report(self) -> str:
        """Generate comprehensive optimization report."""
        current_state = self.analyze_current_state()
        progress = self.monitor_optimization_progress()

        report = f"""
# Cost Optimization Report
Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Executive Summary
- Optimization Strategy: {self.strategy.value}
- Total Resources Monitored: {current_state["total_resources"]}
- Total Cost Last Period: ${current_state["total_cost_last_period"]:,.2f}
- Total Savings Achieved: ${progress["total_savings_achieved"]:,.2f}
- Success Rate: {progress["success_rate"]:.1f}%

## Resource Analysis
"""

        for resource, analysis in current_state.get("resource_analysis", {}).items():
            report += f"""
### {resource.upper()}
- Average Usage: {analysis["avg_usage"]:.2f}
- Average Cost: ${analysis["avg_cost"]:.2f}
- Utilization: {analysis["utilization"]:.1f}%
- Cost Trend: {analysis["cost_trend"]}
- Anomalies Detected: {len(analysis["anomalies"])}
"""

        report += f"""
## Optimization Progress
- Total Actions: {progress["total_actions"]}
- Completed: {progress["completed_actions"]}
- Failed: {progress["failed_actions"]}
- Average Execution Time: {progress["average_execution_time"]:.1f}s

## Savings by Type
"""

        for action_type, savings in progress["savings_by_type"].items():
            report += f"- {action_type}: ${savings:,.2f}\n"

        report += """
## Recommendations
"""

        for opportunity in current_state.get("optimization_opportunities", []):
            report += f"- {opportunity['resource']}: {opportunity['type']} (potential savings: ${opportunity['potential_savings']:,.2f})\n"

        return report

    def get_ml_predictions(
        self, resource_type: ResourceType, horizon_days: int = 7
    ) -> dict[str, Any] | None:
        """Get ML-based predictions for resource usage and costs."""
        if not self.enable_ml:
            return None

        predictions: dict[str, Any] = {
            "resource_type": resource_type.value,
            "horizon_days": horizon_days,
            "usage_forecast": [],
            "cost_forecast": [],
            "confidence": 0.0,
            "recommendations": [],
        }

        # Placeholder for actual ML predictions
        # In a real implementation, this would use trained models

        return predictions

    # Implemented methods for backward compatibility
    def analyze_cost_patterns(
        self, usage_data=None, days_back=30, safe_mode=False, **kwargs
    ):
        """Analyze cost patterns and trends from LLM usage data.

        Performs comprehensive analysis of cost patterns including trend detection,
        anomaly identification, and usage optimization opportunities.

        Args:
            usage_data: Optional usage data dict. If None, uses stored data
            days_back: Number of days to analyze (default: 30)
            safe_mode: Enable additional validation and error handling (default: False)
            **kwargs: Additional analysis parameters

        Returns:
            dict: Analysis results containing:
                - total_cost: Total cost for period
                - avg_cost: Average cost per request
                - trends: Cost trend analysis
                - anomalies: Detected anomalous patterns
                - recommendations: Optimization suggestions

        Example:
            >>> analyzer = CostAnalysisEngine()
            >>> results = analyzer.analyze_cost_patterns(days_back=7, safe_mode=True)
            >>> print(f"Total cost: ${results['total_cost']:.2f}")
        """
        usage_data = self._prepare_usage_data(usage_data)

        validation_result = self._run_safe_mode_validation(usage_data, safe_mode)
        if validation_result is not None:
            return validation_result

        if not usage_data:
            return self._empty_analysis_result()

        summary, error = self._calculate_cost_summary(usage_data, safe_mode)
        if error is not None:
            return error

        insights = self._collect_cost_insights(usage_data, summary["patterns"])

        return {
            "patterns": insights["patterns"],
            "anomalies": insights["anomalies"],
            "trends": insights["trends"],
            "recommendations": self._generate_basic_recommendations(
                insights["patterns"]
            ),
            "summary": {
                "total_cost": summary["total_cost"],
                "avg_cost": summary["avg_cost"],
                "model_distribution": summary["model_costs"],
            },
        }

    def _run_safe_mode_validation(self, usage_data, safe_mode: bool):
        """Execute safe-mode validation and return early result if needed."""

        # Prepare and validate usage data
        if safe_mode:
            validation_result = self._validate_usage_data(usage_data)
            if validation_result:
                return validation_result
        return None

    def _prepare_usage_data(self, usage_data):
        """Prepare usage data from optimization history if not provided."""
        if usage_data is None:
            # Use internal optimization history
            usage_data = [
                self._build_usage_entry(run)
                for run in self._iter_history_runs(limit=100)
            ]

        # Check if usage_data is explicitly empty list (not None)
        if usage_data is not None and not usage_data:
            raise ValueError("Cannot analyze empty usage data")

        return usage_data

    def _calculate_cost_summary(
        self, usage_data, safe_mode: bool
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Calculate cost metrics and model patterns, returning errors when needed."""

        total_cost, avg_cost = self._calculate_cost_metrics(usage_data, safe_mode)
        if safe_mode and total_cost is None:
            return {}, self._error_analysis_result("Invalid cost data types detected")

        patterns, model_costs = self._analyze_model_patterns(usage_data)
        return (
            {
                "total_cost": total_cost,
                "avg_cost": avg_cost,
                "model_costs": model_costs,
                "patterns": patterns,
            },
            None,
        )

    def _collect_cost_insights(
        self, usage_data, base_patterns: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate temporal patterns, trends, and anomalies."""

        patterns = list(base_patterns)
        patterns.extend(self._analyze_temporal_patterns(usage_data))

        return {
            "patterns": patterns,
            "trends": self._analyze_cost_trends(usage_data),
            "anomalies": self._detect_cost_anomalies(usage_data),
        }

    def _validate_usage_data(self, usage_data):
        """Validate usage data in safe mode."""
        try:
            # Validate data
            for item in usage_data:
                cost = item.get("cost", 0)
                if not isinstance(cost, (int, float)):
                    raise TypeError(f"Invalid cost value: {cost}")
        except (TypeError, ValueError) as e:
            return self._error_analysis_result(f"Invalid data detected: {str(e)}")
        return None

    def _iter_history_runs(self, limit: int | None = None) -> list[Any]:
        """Return recent optimization history entries."""
        if not getattr(self, "optimization_history", None):
            return []

        history = self.optimization_history
        runs_attr = getattr(history, "runs", None)
        records: list[Any] = []

        if isinstance(runs_attr, list):
            records = runs_attr
        else:
            records_attr = getattr(history, "records", None)
            if isinstance(records_attr, list):
                records = records_attr
            elif hasattr(history, "get_records"):
                try:
                    fetched = history.get_records()
                    if isinstance(fetched, list):
                        records = fetched
                except Exception as e:
                    logger.debug(f"Could not fetch records from history: {e}")
                    records = []

        if limit is not None and limit > 0:
            return records[-limit:]
        return records

    @staticmethod
    def _build_usage_entry(run: Any) -> dict[str, Any]:
        """Convert a history run/record to a usage entry dict."""
        metadata = getattr(run, "metadata", None)

        if isinstance(run, dict):
            best_config = run.get("best_config") or {}
            timestamp = run.get("start_time") or run.get("timestamp") or time.time()
            total_cost = run.get("total_cost", 0)
            total_tokens = run.get("total_tokens", 0)
            duration = run.get("duration", 0)
        else:
            best_config = getattr(run, "best_config", None)
            if best_config is None and isinstance(metadata, dict):
                best_config = metadata.get("best_config")
            if not isinstance(best_config, dict):
                best_config = {}

            timestamp = (
                getattr(run, "start_time", None)
                or getattr(run, "timestamp", None)
                or time.time()
            )
            total_cost = getattr(run, "total_cost", None)
            if total_cost is None and isinstance(metadata, dict):
                total_cost = metadata.get("total_cost")
            total_tokens = getattr(run, "total_tokens", None)
            if total_tokens is None and isinstance(metadata, dict):
                total_tokens = metadata.get("total_tokens", 0)
            duration = getattr(run, "duration", None)
            if duration is None and isinstance(metadata, dict):
                duration = metadata.get("duration", 0)

        cost_value = float(total_cost) if isinstance(total_cost, (int, float)) else 0.0
        tokens_value = (
            int(total_tokens) if isinstance(total_tokens, (int, float)) else 0
        )
        duration_value = float(duration) if isinstance(duration, (int, float)) else 0.0
        model_name = (
            str(best_config.get("model", "unknown"))
            if isinstance(best_config, dict)
            else "unknown"
        )

        return {
            "timestamp": timestamp,
            "model": model_name,
            "cost": cost_value,
            "tokens": tokens_value,
            "duration": duration_value,
        }

    def _empty_analysis_result(self):
        """Return empty analysis result."""
        return {
            "patterns": [],
            "anomalies": [],
            "recommendations": [],
            "summary": {"total_cost": 0, "avg_cost": 0},
        }

    def _error_analysis_result(self, error_message):
        """Return error analysis result."""
        return {
            "error": error_message,
            "patterns": [],
            "anomalies": [],
            "trends": {"overall": "error", "recent": "error"},
            "recommendations": [],
            "summary": {
                "total_cost": 0,
                "avg_cost": 0,
                "model_distribution": {},
            },
        }

    def _calculate_cost_metrics(self, usage_data, safe_mode):
        """Calculate total and average cost metrics."""
        try:
            total_cost = sum(item.get("cost", 0) for item in usage_data)
            avg_cost = total_cost / len(usage_data) if usage_data else 0
            return total_cost, avg_cost
        except TypeError:
            # Handle invalid cost data
            if safe_mode:
                return None, None
            else:
                raise

    def _analyze_model_patterns(self, usage_data):
        """Analyze patterns by model."""
        patterns: list[dict[str, Any]] = []
        model_costs: dict[str, list[float]] = {}

        # Group by model
        for item in usage_data:
            model = item.get("model", "unknown")
            cost = item.get("cost", 0)
            if model not in model_costs:
                model_costs[model] = []
            model_costs[model].append(cost)

        # Generate model patterns
        for model, costs in model_costs.items():
            patterns.append(
                {
                    "type": "model_usage",
                    "model": model,
                    "total_cost": sum(costs),
                    "avg_cost": sum(costs) / len(costs),
                    "usage_count": len(costs),
                    "cost_trend": (
                        "increasing"
                        if len(costs) > 1 and costs[-1] > costs[0]
                        else "stable"
                    ),
                }
            )

        return patterns, model_costs

    def _detect_cost_anomalies(self, usage_data):
        """Detect cost anomalies in usage data."""
        anomalies: list[dict[str, Any]] = []

        if not usage_data:
            return anomalies

        costs = [
            item.get("cost", 0)
            for item in usage_data
            if isinstance(item.get("cost", 0), (int, float))
        ]

        if not costs:
            return anomalies

        # Calculate threshold for anomaly detection
        if len(costs) == 1:
            # Single data point - check if it's extremely high
            threshold = 1000000  # 1M threshold for single points
        else:
            avg_cost_calc = sum(costs) / len(costs)
            # Consider values 10x above average as anomalies
            threshold = max(avg_cost_calc * 10, 1000)

        # Detect anomalies
        for i, item in enumerate(usage_data):
            cost = item.get("cost", 0)
            if isinstance(cost, (int, float)) and cost > threshold:
                anomalies.append(
                    {
                        "index": i,
                        "cost": cost,
                        "threshold": threshold,
                        "severity": ("high" if cost > threshold * 10 else "medium"),
                        "deviation_score": (
                            cost / self.anomaly_threshold
                            if hasattr(self, "anomaly_threshold")
                            else cost / 100
                        ),
                        "timestamp": item.get("timestamp"),
                        "resource_type": item.get("resource_type", "unknown"),
                    }
                )

        return anomalies

    def get_optimization_recommendations(
        self,
        current_usage=None,
        objectives=None,
        constraints=None,
        current_config=None,
        usage_history=None,
        **kwargs,
    ):
        """Get optimization recommendations based on usage."""
        recommendations = []

        # Analyze current configuration
        if current_config:
            model = current_config.get("model")
            if model and "gpt-4" in model:
                recommendations.append(
                    {
                        "resource_type": "compute",
                        "type": "model_downgrade",
                        "priority": "high",
                        "action": "Consider using gpt-3.5-turbo or claude-3-5-haiku-latest",
                        "potential_savings": "50-90%",
                        "impact": "Minimal accuracy loss for most tasks",
                        "confidence": 0.85,
                        "estimated_savings": 5000,
                    }
                )

        # Add general recommendations
        recommendations.extend(
            [
                {
                    "resource_type": "compute",
                    "type": "caching",
                    "priority": "medium",
                    "action": "Implement response caching",
                    "potential_savings": "30-50%",
                    "impact": "Reduced API calls",
                    "confidence": 0.75,
                    "estimated_savings": 2000,
                },
                {
                    "resource_type": "compute",
                    "type": "batching",
                    "priority": "medium",
                    "action": "Batch similar requests",
                    "potential_savings": "20-30%",
                    "impact": "Improved efficiency",
                    "confidence": 0.70,
                    "estimated_savings": 1500,
                },
            ]
        )

        return recommendations

    def _analyze_temporal_patterns(
        self, usage_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze temporal patterns in usage data."""
        patterns: list[dict[str, Any]] = []

        if not usage_data:
            return patterns

        # Group data by time periods
        temporal_groups = self._group_data_by_time_periods(usage_data)

        # Analyze each temporal pattern
        patterns.extend(self._analyze_weekly_patterns(temporal_groups["weekly"]))
        patterns.extend(self._analyze_daily_patterns(temporal_groups["daily"]))
        patterns.extend(self._analyze_seasonal_patterns(temporal_groups["monthly"]))

        return patterns

    def _group_data_by_time_periods(
        self, usage_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Group usage data by different time periods."""
        weekly_costs = defaultdict(list)
        daily_costs = defaultdict(list)
        monthly_costs = defaultdict(list)

        for item in usage_data:
            timestamp = item.get("timestamp", time.time())
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = timestamp

            day_of_week = dt.weekday()
            hour = dt.hour
            month = dt.month
            cost = item.get("cost", 0)

            weekly_costs[day_of_week].append(cost)
            daily_costs[hour].append(cost)
            monthly_costs[month].append(cost)

        return {"weekly": weekly_costs, "daily": daily_costs, "monthly": monthly_costs}

    def _analyze_weekly_patterns(
        self, weekly_costs: dict[int, list[float]]
    ) -> list[dict[str, Any]]:
        """Analyze weekly usage patterns."""
        patterns: list[dict[str, Any]] = []

        if not weekly_costs:
            return patterns

        avg_by_day = {
            day: sum(costs) / len(costs) for day, costs in weekly_costs.items()
        }

        patterns.append(
            {
                "type": "weekly",
                "pattern": (
                    "workweek_heavy"
                    if sum(avg_by_day.get(i, 0) for i in range(5))
                    > sum(avg_by_day.get(i, 0) for i in [5, 6]) * 2
                    else "even"
                ),
                "peak_days": [
                    day
                    for day, avg in avg_by_day.items()
                    if avg > sum(avg_by_day.values()) / len(avg_by_day)
                ],
                "avg_by_day": avg_by_day,
            }
        )

        return patterns

    def _analyze_daily_patterns(
        self, daily_costs: dict[int, list[float]]
    ) -> list[dict[str, Any]]:
        """Analyze daily usage patterns."""
        patterns: list[dict[str, Any]] = []

        if not daily_costs:
            return patterns

        avg_by_hour = {
            hour: sum(costs) / len(costs) for hour, costs in daily_costs.items()
        }

        patterns.append(
            {
                "type": "daily",
                "pattern": (
                    "business_hours"
                    if sum(avg_by_hour.get(i, 0) for i in range(9, 18))
                    > sum(avg_by_hour.get(i, 0) for i in range(24)) * 0.5
                    else "distributed"
                ),
                "peak_hours": [
                    hour
                    for hour, avg in avg_by_hour.items()
                    if avg > sum(avg_by_hour.values()) / len(avg_by_hour)
                ],
                "avg_by_hour": avg_by_hour,
            }
        )

        return patterns

    def _analyze_seasonal_patterns(
        self, monthly_costs: dict[int, list[float]]
    ) -> list[dict[str, Any]]:
        """Analyze seasonal usage patterns."""
        patterns: list[dict[str, Any]] = []

        # Need at least 3 months of data for seasonal analysis
        if not monthly_costs or len(monthly_costs) < 3:
            return patterns

        avg_by_month = {
            month: sum(costs) / len(costs) for month, costs in monthly_costs.items()
        }
        overall_avg = sum(avg_by_month.values()) / len(avg_by_month)

        # Find peak months (above 1.5x average)
        peak_months = [
            month for month, avg in avg_by_month.items() if avg > overall_avg * 1.5
        ]

        if peak_months:
            patterns.append(
                {
                    "type": "seasonal",
                    "pattern": (
                        "holiday_peaks"
                        if any(month in [11, 12] for month in peak_months)
                        else "seasonal_variation"
                    ),
                    "peak_months": peak_months,
                    "avg_by_month": avg_by_month,
                }
            )

        return patterns

    def simulate_optimization_impact(
        self,
        actions,
        simulation_days=30,
        duration_days=None,
        current_metrics=None,
        **kwargs,
    ):
        """Simulate the impact of optimization actions."""
        days = self._resolve_simulation_days(simulation_days, duration_days)
        current_daily_cost = self._determine_current_daily_cost(current_metrics)
        projected_savings = self._estimate_projected_savings(
            actions, current_daily_cost
        )
        projected_daily_cost = max(current_daily_cost - projected_savings, 0)

        return {
            "current_cost": current_daily_cost * days,
            "projected_cost": projected_daily_cost * days,
            "estimated_savings": projected_savings * days,
            "cost_savings": {
                "total": projected_savings * days,
                "percentage": (
                    (projected_savings / current_daily_cost) * 100
                    if current_daily_cost > 0
                    else 0
                ),
            },
            "performance_impact": {
                "latency_change": 0.02,  # 2% improvement
                "accuracy_change": 0.01,  # 1% improvement
            },
            "risk_assessment": {
                "implementation_risk": "low",
                "performance_risk": "low",
            },
            "timeline": {"implementation_days": 5, "full_impact_days": 14},
            "savings_percentage": (
                (projected_savings / current_daily_cost) * 100
                if current_daily_cost > 0
                else 0
            ),
            "break_even_days": 0,  # Immediate savings
            "confidence_level": 0.75,
        }

    @staticmethod
    def _resolve_simulation_days(
        simulation_days: int, duration_days: int | None
    ) -> int:
        """Resolve the simulation window using duration override when provided."""

        return duration_days or simulation_days

    @staticmethod
    def _determine_current_daily_cost(current_metrics: dict[str, Any] | None) -> float:
        """Derive the baseline daily cost from current metrics."""

        if not current_metrics:
            return 10.0
        if "daily_cost" in current_metrics:
            return float(current_metrics.get("daily_cost", 1000)) / 30
        if "monthly_cost" in current_metrics:
            return float(current_metrics["monthly_cost"]) / 30
        return float(current_metrics.get("baseline_cost", 300)) / 30

    def _estimate_projected_savings(
        self, actions: Any, current_daily_cost: float
    ) -> float:
        """Estimate daily savings produced by the provided actions."""

        if current_daily_cost <= 0:
            return 0.0

        if isinstance(actions, dict):
            return self._estimate_dict_action_savings(actions, current_daily_cost)

        return self._estimate_iterable_action_savings(actions, current_daily_cost)

    @staticmethod
    def _estimate_dict_action_savings(
        actions: dict[str, Any], current_daily_cost: float
    ) -> float:
        """Estimate savings for dictionary-based optimization parameters."""

        savings = 0.0
        if "compute_reduction" in actions:
            savings += current_daily_cost * actions.get("compute_reduction", 0.0)

        if actions.get("storage_tiering"):
            savings += current_daily_cost * 0.2

        if "spot_instances" in actions:
            savings += current_daily_cost * actions.get("spot_instances", 0.0) * 0.5

        return savings

    @staticmethod
    def _estimate_iterable_action_savings(
        actions: Any, current_daily_cost: float
    ) -> float:
        """Estimate savings for list/string action descriptions."""

        if not actions:
            return 0.0

        impact_map = {
            "switch_to_cheaper_model": 0.5,
            "implement_caching": 0.3,
            "batch_processing": 0.2,
            "reduce_token_usage": 0.15,
        }

        savings = 0.0
        for action in actions:
            action_type = action if isinstance(action, str) else action.get("type", "")
            savings += current_daily_cost * impact_map.get(action_type, 0.0)

        return savings

    def identify_usage_anomalies(self, usage_data, threshold=2.0, **kwargs):
        """Identify anomalies in usage patterns."""
        if not usage_data:
            return []

        # Calculate statistics
        costs = [item.get("cost", 0) for item in usage_data]
        if not costs:
            return []

        mean_cost = sum(costs) / len(costs)
        std_dev = (sum((x - mean_cost) ** 2 for x in costs) / len(costs)) ** 0.5

        anomalies = []
        for i, item in enumerate(usage_data):
            cost = item.get("cost", 0)
            z_score = (cost - mean_cost) / std_dev if std_dev > 0 else 0

            if abs(z_score) > threshold:
                anomalies.append(
                    {
                        "index": i,
                        "timestamp": item.get("timestamp", time.time()),
                        "cost": cost,
                        "z_score": z_score,
                        "deviation_score": abs(z_score),
                        "severity": "high" if abs(z_score) > 3 else "medium",
                        "description": f"Cost {'spike' if z_score > 0 else 'drop'} detected",
                    }
                )

        return anomalies

    def predict_future_costs(self, historical_data=None, prediction_days=30, **kwargs):
        """Predict future costs based on historical data."""
        if not historical_data:
            historical_data = [
                {
                    "date": usage_entry["timestamp"],
                    "cost": usage_entry["cost"],
                }
                for usage_entry in (
                    self._build_usage_entry(run)
                    for run in self._iter_history_runs(limit=30)
                )
            ]

        if not historical_data:
            return {
                "predictions": [],
                "total_predicted_cost": 0,
                "confidence_interval": (0, 0),
                "trend": "insufficient_data",
            }

        # Simple linear projection
        daily_costs = [item.get("cost", 0) for item in historical_data]
        avg_daily_cost = sum(daily_costs) / len(daily_costs) if daily_costs else 0

        predictions = []
        for day in range(prediction_days):
            predictions.append(
                {
                    "day": day + 1,
                    "predicted_cost": avg_daily_cost,
                    "confidence": 0.8 - (day * 0.01),  # Confidence decreases over time
                }
            )

        total_predicted = avg_daily_cost * prediction_days

        # Calculate growth rate
        growth_rate = 0
        if len(daily_costs) > 1:
            growth_rate = (
                (daily_costs[-1] - daily_costs[0]) / daily_costs[0]
                if daily_costs[0] > 0
                else 0
            )

        # Simple seasonality detection (weekly pattern)
        seasonality_factors = {}
        if len(daily_costs) >= 7:
            for i in range(7):
                day_costs = [daily_costs[j] for j in range(i, len(daily_costs), 7)]
                if day_costs and avg_daily_cost > 0:
                    seasonality_factors[i] = (
                        sum(day_costs) / len(day_costs) / avg_daily_cost
                    )
                else:
                    seasonality_factors[i] = 1.0

        return {
            "predictions": predictions,
            "total_predicted_cost": total_predicted,
            "confidence_interval": (total_predicted * 0.8, total_predicted * 1.2),
            "confidence_intervals": (
                total_predicted * 0.8,
                total_predicted * 1.2,
            ),  # Add plural for backward compatibility
            "trend": "stable",
            "avg_daily_cost": avg_daily_cost,
            "growth_rate": growth_rate,
            "seasonality_factors": seasonality_factors,
        }

    def optimize_resource_allocation(self, resources, constraints=None, **kwargs):
        """Optimize resource allocation across models and tasks."""
        if not resources:
            return {"allocations": [], "expected_cost": 0}

        # Simple allocation strategy
        allocations = []
        total_budget = resources.get("budget", 1000)
        tasks = resources.get("tasks", [])

        for task in tasks:
            priority = task.get("priority", 1)
            estimated_cost = task.get("estimated_cost", 10)

            # Allocate based on priority
            allocation = min(estimated_cost * priority, total_budget * 0.3)

            allocations.append(
                {
                    "task": task.get("name", "unknown"),
                    "allocated_budget": allocation,
                    "recommended_model": (
                        "claude-3-5-haiku-latest"
                        if allocation < 50
                        else "gpt-3.5-turbo"
                    ),
                    "expected_performance": 0.85,
                }
            )

        # Calculate recommended allocation
        current_compute = resources.get("current_allocation", {}).get("compute", 100)
        current_storage = resources.get("current_allocation", {}).get("storage", 100)

        # Simple optimization: reduce underutilized resources
        recommended_compute = (
            current_compute * 0.8 if current_compute > 50 else current_compute
        )
        recommended_storage = (
            current_storage * 0.9 if current_storage > 50 else current_storage
        )

        total_expected_cost = sum(a["allocated_budget"] for a in allocations)

        return {
            "allocations": allocations,
            "expected_cost": total_expected_cost,
            "optimization_score": 0.75,
            "recommended_allocation": {
                "compute": recommended_compute,
                "storage": recommended_storage,
                "memory": resources.get("current_allocation", {}).get("memory", 50)
                * 0.85,
            },
            "estimated_savings": total_expected_cost * 0.2,  # 20% savings estimate
            "utilization_improvement": 0.15,  # 15% improvement
        }

    def optimize_multi_cloud(
        self,
        cloud_usage=None,
        constraints=None,
        providers=None,
        workload=None,
        **kwargs,
    ):
        """Optimize workload distribution across cloud providers."""
        cloud_usage = cloud_usage or {}
        # Extract providers from cloud_usage if not explicitly provided
        if not providers and cloud_usage:
            providers = list(cloud_usage.keys())

        if not providers:
            return {
                "workload_distribution": {},
                "estimated_savings": 0,
                "migration_plan": [],
            }

        # Analyze current usage to calculate total costs
        total_cost = 0
        workload_distribution = {}

        for provider in providers:
            if provider in cloud_usage:
                provider_data = cloud_usage[provider]
                compute_cost = provider_data.get("compute", {}).get("cost", 0)
                storage_cost = provider_data.get("storage", {}).get("cost", 0)
                provider_total = compute_cost + storage_cost
                total_cost += provider_total

                # Calculate current distribution
                workload_distribution[provider] = {
                    "current_cost": provider_total,
                    "compute_instances": provider_data.get("compute", {}).get(
                        "instances", 0
                    ),
                    "storage_gb": provider_data.get("storage", {}).get("gb", 0),
                }

        # Apply constraints and optimize
        if constraints:
            # Apply data sovereignty constraints
            sovereignty = constraints.get("data_sovereignty", {})
            for region, required_provider in sovereignty.items():
                if required_provider in workload_distribution:
                    workload_distribution[required_provider]["regions"] = (
                        workload_distribution[required_provider].get("regions", [])
                        + [region]
                    )

        # Calculate estimated savings (mock 15% savings)
        estimated_savings = total_cost * 0.15

        # Generate migration plan
        migration_plan = []
        for provider in providers:
            if provider in workload_distribution:
                current_cost = workload_distribution[provider]["current_cost"]
                if current_cost > 5000:  # Only migrate high-cost workloads
                    migration_plan.append(
                        {
                            "source_provider": provider,
                            "target_provider": min(
                                providers,
                                key=lambda p: cloud_usage.get(p, {})
                                .get("compute", {})
                                .get("cost", float("inf")),
                            ),
                            "workload_type": "compute",
                            "estimated_effort": "medium",
                            "potential_savings": current_cost * 0.2,
                        }
                    )

        return {
            "workload_distribution": workload_distribution,
            "estimated_savings": estimated_savings,
            "migration_plan": migration_plan,
            "optimization_strategy": "cost_based_distribution",
        }

    def prioritize_recommendations(self, recommendations, criteria=None, **kwargs):
        """Prioritize recommendations based on impact and effort."""
        if not recommendations:
            return []

        # Handle different criteria formats
        if criteria is None:
            criteria = {"cost_weight": 0.4, "effort_weight": 0.3, "risk_weight": 0.3}
        elif isinstance(criteria, list):
            # Convert list format to dict format
            default_weights = {"savings": 0.4, "effort": 0.3, "risk": 0.3}
            criteria = {
                "cost_weight": default_weights.get("savings", 0.4),
                "effort_weight": default_weights.get("effort", 0.3),
                "risk_weight": default_weights.get("risk", 0.3),
            }

        prioritized = []
        for rec in recommendations:
            # Calculate priority score
            cost_impact = rec.get("potential_savings_pct", 0) / 100

            # Handle implementation_effort as either number or string
            impl_effort = rec.get("implementation_effort", 5)
            if isinstance(impl_effort, str):
                # Map string efforts to numeric values
                effort_map = {"low": 2, "medium": 5, "high": 8}
                impl_effort = effort_map.get(impl_effort.lower(), 5)
            effort = 1 - (impl_effort / 10)

            # Handle risk_level similarly
            risk_val = rec.get("risk_level", 5)
            if isinstance(risk_val, str):
                risk_map = {"low": 2, "medium": 5, "high": 8}
                risk_val = risk_map.get(risk_val.lower(), 5)
            risk = 1 - (risk_val / 10)

            score = (
                cost_impact * criteria["cost_weight"]
                + effort * criteria["effort_weight"]
                + risk * criteria["risk_weight"]
            )

            prioritized.append(
                {
                    **rec,
                    "priority_score": score,
                    "priority_rank": (
                        "high" if score > 0.7 else "medium" if score > 0.4 else "low"
                    ),
                }
            )

        # Sort by score
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)

        return prioritized

    def analyze_cost_allocation(self, cost_data, **kwargs):
        """Analyze cost allocation across different dimensions."""
        if not cost_data:
            return {"allocations": {}, "insights": []}

        # Handle complex cost data structure
        by_department = {}
        by_resource_type = {"compute": 0, "storage": 0}
        inefficiencies = []
        reallocation_opportunities = []

        # Process departments data
        if "departments" in cost_data:
            for dept, resources in cost_data["departments"].items():
                dept_total = 0
                for resource_type, cost in resources.items():
                    dept_total += cost
                    by_resource_type[resource_type] = (
                        by_resource_type.get(resource_type, 0) + cost
                    )
                by_department[dept] = dept_total

                # Detect inefficiencies
                if dept_total > 10000:
                    inefficiencies.append(f"{dept} has high costs: ${dept_total}")

        # Process projects data
        if "projects" in cost_data:
            for project, details in cost_data["projects"].items():
                project_cost = details.get("cost", 0)
                dept = details.get("department", "unknown")

                # Check for reallocation opportunities
                if project_cost > 5000:
                    reallocation_opportunities.append(
                        {
                            "project": project,
                            "current_cost": project_cost,
                            "potential_savings": project_cost * 0.2,
                            "recommendation": f"Consider optimizing {project}",
                        }
                    )

        return {
            "by_department": by_department,
            "by_resource_type": by_resource_type,
            "inefficiencies": inefficiencies,
            "reallocation_opportunities": reallocation_opportunities,
            "total_cost": sum(by_department.values()),
        }

    def fetch_current_pricing(self, providers=None, **kwargs):
        """Fetch current pricing information using tokencost library with fallback."""
        if providers is None:
            providers = ["openai", "anthropic"]

        pricing: dict[str, Any] = {}

        try:
            from tokencost import calculate_completion_cost, calculate_prompt_cost

            # Use tokencost to get real-time pricing for common models
            test_prompt = [{"role": "user", "content": "test"}]
            test_completion = "test"

            # Common models to check pricing for
            model_mappings = {
                "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo"],
                "anthropic": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-latest",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                ],
            }

            for provider in providers:
                if provider in model_mappings:
                    pricing[provider] = {}
                    for model in model_mappings[provider]:
                        try:
                            input_cost = calculate_prompt_cost(test_prompt, model)
                            output_cost = calculate_completion_cost(
                                test_completion, model
                            )
                            pricing[provider][model] = {
                                "input": input_cost,
                                "output": output_cost,
                            }
                        except Exception as e:
                            # Skip models that tokencost doesn't recognize
                            logger.debug(
                                f"Skipping model {model}: tokencost error: {e}"
                            )
                            continue

        except ImportError:
            # Fallback to static pricing data if tokencost not available
            pricing_data = {
                "openai": {
                    "gpt-4o": {"input": 0.01, "output": 0.03},
                    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                },
                "anthropic": {
                    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
                    "claude-3-5-haiku-latest": {"input": 0.00025, "output": 0.00125},
                },
            }

            for provider in providers:
                if provider in pricing_data:
                    pricing[provider] = pricing_data[provider]

        # Add non-LLM pricing data (these aren't handled by tokencost)
        if "aws" in providers:
            pricing["aws"] = {
                "compute": {"on_demand": 0.1, "spot": 0.03},
                "storage": {"standard": 0.02, "archive": 0.004},
                "bedrock-claude": {"input": 0.008, "output": 0.024},
            }

        if "azure" in providers:
            pricing["azure"] = {
                "compute": {"on_demand": 0.1, "spot": 0.03},
                "storage": {"standard": 0.02, "archive": 0.004},
                "gpt-4": {"input": 0.01, "output": 0.03},
            }

        return pricing

    def generate_optimization_recommendations(self, **kwargs):
        """Generate optimization recommendations based on current usage."""
        return self.get_optimization_recommendations(**kwargs)

    def track_optimization_results(self, optimization_id, results, **kwargs):
        """Track optimization results for analysis."""
        # Store in optimization history
        self.tracked_results[optimization_id] = {
            "timestamp": time.time(),
            "results": results,
            "metrics": kwargs.get("metrics", {}),
        }

        return {
            "status": "tracked",
            "optimization_id": optimization_id,
            "timestamp": time.time(),
        }

    def _generate_basic_recommendations(self, patterns):
        """Generate basic recommendations from patterns."""
        recommendations = []

        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")

            if pattern_type == "weekly":
                if pattern.get("pattern") == "workweek_heavy":
                    recommendations.append(
                        "Consider implementing weekend scaling to reduce workweek costs"
                    )

            elif pattern_type == "daily":
                if pattern.get("pattern") == "business_hours":
                    recommendations.append(
                        "Peak usage during business hours - consider off-hours scheduling for batch jobs"
                    )

            elif pattern_type == "seasonal":
                if pattern.get("pattern") == "holiday_peaks":
                    recommendations.append(
                        "Holiday season cost spikes detected - plan capacity scaling for peak periods"
                    )

            # Legacy support for old pattern format
            if "avg_cost" in pattern and pattern["avg_cost"] > 0.1:
                model = pattern.get("model", "unknown")
                recommendations.append(
                    f"Consider optimizing {model} usage - avg cost ${pattern['avg_cost']:.3f}"
                )

            if pattern.get("cost_trend") == "increasing":
                model = pattern.get("model", "service")
                recommendations.append(
                    f"Cost trend increasing for {model} - review usage patterns"
                )

        return recommendations
