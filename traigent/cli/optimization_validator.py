"""Optimization validator using existing Pareto efficiency system."""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import os
from datetime import UTC
from typing import Any

from rich.console import Console

from traigent.api.types import TrialResult, TrialStatus
from traigent.cli.validation_types import OptimizedFunction, ValidationResult
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.logging import get_logger
from traigent.utils.multi_objective import ParetoFrontCalculator, ParetoPoint

logger = get_logger(__name__)
console = Console()


class OptimizationValidator:
    """Validates that TraiGent optimization improves over default parameters using Pareto efficiency."""

    def __init__(self, threshold_pct: float = 10.0) -> None:
        """Initialize the optimization validator.

        Args:
            threshold_pct: Minimum improvement threshold percentage (default: 10%)
        """
        self.threshold = threshold_pct / 100.0  # Convert to decimal (0.1 for 10%)
        self.pareto_calculator = ParetoFrontCalculator()
        logger.info(
            f"Initialized OptimizationValidator with {threshold_pct}% threshold"
        )

    async def validate_optimization(
        self, func_info: OptimizedFunction
    ) -> ValidationResult:
        """Validate that optimization improves over default parameters.

        Args:
            func_info: Information about the function to validate

        Returns:
            ValidationResult with comparison details and verdict
        """
        logger.info(f"Starting validation of {func_info.name}")

        # Pre-validation checks
        blocking_issues = self._check_prerequisites(func_info)
        if blocking_issues:
            return ValidationResult(
                function_name=func_info.name,
                baseline_metrics={},
                optimized_metrics={},
                is_superior=False,
                improvement_details={},
                blocking_issues=blocking_issues,
                threshold_used=self.threshold,
            )

        try:
            # Step 1: Run function with default parameters (baseline)
            console.print("🔍 Running baseline with default parameters...")
            baseline_metrics, baseline_config = await self._run_baseline(func_info)

            # Step 2: Run TraiGent optimization
            console.print("🚀 Running TraiGent optimization...")
            optimized_metrics, optimized_config = await self._run_optimization(
                func_info
            )

            # Step 3: Compare using Pareto efficiency
            console.print("⚖️  Comparing results using Pareto efficiency...")
            is_superior, improvement_details = self._compare_results(
                baseline_metrics, optimized_metrics, func_info.objectives
            )

            return ValidationResult(
                function_name=func_info.name,
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                is_superior=is_superior,
                improvement_details=improvement_details,
                blocking_issues=[],
                baseline_config=baseline_config,
                optimized_config=optimized_config,
                threshold_used=self.threshold,
            )

        except Exception as e:
            logger.error(f"Validation failed for {func_info.name}: {e}")
            return ValidationResult(
                function_name=func_info.name,
                baseline_metrics={},
                optimized_metrics={},
                is_superior=False,
                improvement_details={},
                blocking_issues=[f"Validation error: {str(e)}"],
                threshold_used=self.threshold,
            )

    def _check_prerequisites(self, func_info: OptimizedFunction) -> list[str]:
        """Check if function meets prerequisites for validation.

        Args:
            func_info: Function information to check

        Returns:
            List of blocking issues (empty if all checks pass)
        """
        issues = []

        if not func_info.has_dataset:
            issues.append("Missing evaluation dataset")

        if not func_info.objectives:
            issues.append("No objectives specified")

        config_space = func_info.decorator_config.get("configuration_space", {})
        if not config_space:
            issues.append("No configuration space specified")

        return issues

    async def _run_baseline(
        self, func_info: OptimizedFunction
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run function with default parameters to get baseline metrics.

        Args:
            func_info: Function information

        Returns:
            Tuple of (baseline_metrics, baseline_config)
        """
        # Use default parameters as baseline configuration
        baseline_config = func_info.default_params.copy()

        # If no defaults available, use empty config (function should handle this)
        if not baseline_config:
            baseline_config = {}
            logger.warning(
                f"No default parameters found for {func_info.name}, using empty config"
            )

        baseline_metrics = await self._evaluate_configuration(
            func_info, baseline_config
        )

        logger.info(f"Baseline metrics for {func_info.name}: {baseline_metrics}")
        return baseline_metrics, baseline_config

    async def _run_optimization(
        self, func_info: OptimizedFunction
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run TraiGent optimization to get optimized metrics.

        Args:
            func_info: Function information

        Returns:
            Tuple of (optimized_metrics, optimized_config)
        """
        if os.environ.get("TRAIGENT_MOCK_MODE", "").lower() == "true":
            try:
                metrics = func_info.func({})
            except TypeError:
                metrics = func_info.func()
            if not isinstance(metrics, dict):
                metrics = {}
            return metrics, {}

        try:
            # Run the optimization using the decorated function's optimize method
            result = await func_info.func.optimize(
                algorithm="random",  # Use fast algorithm for validation
                max_trials=5,  # Minimal trials for speed
                timeout=30.0,  # 30 second timeout
            )

            # Extract metrics from the best trial
            if not result.successful_trials:
                raise RuntimeError("No successful optimization trials") from None

            # Use best_metrics property instead of best_trial
            optimized_metrics = result.best_metrics
            optimized_config = result.best_config

            logger.info(f"Optimized metrics for {func_info.name}: {optimized_metrics}")
            return optimized_metrics, optimized_config

        except Exception as e:
            logger.error(f"Optimization failed for {func_info.name}: {e}")
            raise RuntimeError(f"Optimization failed: {e}") from e

    async def _evaluate_configuration(
        self,
        func_info: OptimizedFunction,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate a function with a specific configuration on its dataset."""

        # Fast-path mock mode: call the provided function directly without
        # requiring decorator infrastructure.
        if os.environ.get("TRAIGENT_MOCK_MODE", "").lower() == "true":
            try:
                metrics = func_info.func(config)
            except TypeError:
                metrics = func_info.func()

            if not isinstance(metrics, dict):
                metrics = {}

            if func_info.objectives:
                return {
                    objective: metrics.get(objective, 0.0)
                    for objective in func_info.objectives
                    if objective in metrics
                }

            return metrics

        try:
            dataset = func_info.func._load_dataset()
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Failed to load evaluation dataset for {func_info.name}: {exc}"
            ) from exc

        # Ensure we always operate on a dict (copy to avoid mutating defaults)
        effective_config = (config or {}).copy()

        provider = func_info.func._provider
        configured_callable = provider.inject_config(
            func_info.func.func, effective_config, func_info.func.config_param
        )

        evaluator = LocalEvaluator(
            metrics=func_info.objectives,
            execution_mode=getattr(func_info.func, "execution_mode", "cloud"),
            privacy_enabled=getattr(func_info.func, "privacy_enabled", False),
            mock_mode_config=getattr(func_info.func, "mock_mode_config", None),
        )

        evaluation_result = await evaluator.evaluate(
            configured_callable, effective_config, dataset
        )

        metrics = evaluation_result.metrics or {}

        # Filter to requested objectives if provided
        if func_info.objectives:
            filtered_metrics: dict[str, float] = {}
            for objective in func_info.objectives:
                if objective in metrics:
                    filtered_metrics[objective] = metrics[objective]
                else:
                    logger.warning(
                        "Objective '%s' not present in evaluation metrics for %s",
                        objective,
                        func_info.name,
                    )
            if filtered_metrics:
                return filtered_metrics

        return metrics

    def _compare_results(
        self,
        baseline_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
        objectives: list[str],
    ) -> tuple[bool, dict[str, float]]:
        """Compare baseline vs optimized results using Pareto efficiency.

        Args:
            baseline_metrics: Metrics from baseline run
            optimized_metrics: Metrics from optimized run
            objectives: List of objective names

        Returns:
            Tuple of (is_superior, improvement_details)
        """
        # Create ParetoPoint objects for comparison
        baseline_point = ParetoPoint(
            config={"type": "baseline"},
            objectives=baseline_metrics,
            trial=self._create_mock_trial(baseline_metrics, "baseline"),
        )

        optimized_point = ParetoPoint(
            config={"type": "optimized"},
            objectives=optimized_metrics,
            trial=self._create_mock_trial(optimized_metrics, "optimized"),
        )

        # Define maximize behavior (assume we want to maximize all metrics by default)
        # This should be configurable based on objective types (accuracy=maximize, cost=minimize)
        maximize_config = self._get_maximize_config(objectives)

        # Check if optimized dominates baseline
        optimized_dominates_baseline = optimized_point.dominates(
            baseline_point, maximize_config
        )

        # Calculate improvement percentages
        improvement_details = {}
        for metric in objectives:
            if metric in baseline_metrics and metric in optimized_metrics:
                baseline_val = baseline_metrics[metric]
                optimized_val = optimized_metrics[metric]

                if baseline_val != 0:
                    improvement_pct = (
                        (optimized_val - baseline_val) / abs(baseline_val)
                    ) * 100
                    # Adjust sign based on maximize/minimize preference
                    if not maximize_config.get(
                        metric, True
                    ):  # If minimizing, flip sign
                        improvement_pct = -improvement_pct
                    improvement_details[metric] = improvement_pct
                else:
                    improvement_details[metric] = 0.0

        # Check threshold requirement: must be better in ALL metrics AND >threshold in ≥1 metric
        is_superior = self._check_superior_criteria(
            optimized_dominates_baseline, improvement_details
        )

        logger.info(f"Pareto dominance: {optimized_dominates_baseline}")
        logger.info(f"Improvement details: {improvement_details}")
        logger.info(f"Is superior: {is_superior}")

        return is_superior, improvement_details

    def _get_maximize_config(self, objectives: list[str]) -> dict[str, bool]:
        """Get maximize configuration for objectives.

        Args:
            objectives: List of objective names

        Returns:
            Dictionary mapping objective names to maximize boolean
        """
        # Default maximize behavior based on common metric types
        maximize_defaults = {
            "accuracy": True,
            "precision": True,
            "recall": True,
            "f1": True,
            "f1_score": True,
            "score": True,
            "success_rate": True,
            "throughput": True,
            "speed": True,
            "cost": False,
            "latency": False,
            "response_time": False,
            "error_rate": False,
            "loss": False,
            "duration": False,
        }

        maximize_config = {}
        for obj in objectives:
            obj_lower = obj.lower()
            # Check for exact matches or substrings
            maximize = True  # Default to maximize
            for key, should_maximize in maximize_defaults.items():
                if key in obj_lower or obj_lower in key:
                    maximize = should_maximize
                    break
            maximize_config[obj] = maximize

        return maximize_config

    def _check_superior_criteria(
        self, pareto_dominates: bool, improvement_details: dict[str, float]
    ) -> bool:
        """Check if optimization meets superior criteria.

        Args:
            pareto_dominates: Whether optimized Pareto dominates baseline
            improvement_details: Per-metric improvement percentages

        Returns:
            True if optimization is superior to baseline
        """
        if not pareto_dominates:
            return False

        # Check if at least one metric improves by more than threshold
        threshold_pct = self.threshold * 100  # Convert to percentage
        significant_improvement = any(
            abs(improvement) > threshold_pct
            for improvement in improvement_details.values()
            if improvement > 0  # Only count positive improvements
        )

        return significant_improvement

    def _create_mock_trial(
        self, metrics: dict[str, float], config_type: str
    ) -> TrialResult:
        """Create a mock TrialResult for Pareto comparison.

        Args:
            metrics: Metrics dictionary
            config_type: Type of configuration ("baseline" or "optimized")

        Returns:
            Mock TrialResult object
        """
        from datetime import datetime

        return TrialResult(
            trial_id=f"mock_{config_type}",
            config={config_type: True},
            metrics=metrics,
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=datetime.now(UTC),
            error_message=None,
        )
