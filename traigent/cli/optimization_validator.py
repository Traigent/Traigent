"""Optimization validator using existing Pareto efficiency system."""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import math
import time
from datetime import UTC
from typing import Any

from rich.console import Console

from traigent.api.types import TrialResult, TrialStatus
from traigent.cli.validation_types import OptimizedFunction, ValidationResult
from traigent.config.context import TrialContext
from traigent.config.types import resolve_execution_policy
from traigent.core.objective_directions import resolve_objective_orientation
from traigent.core.orchestrator_helpers import extract_cost_from_results
from traigent.core.trial_result_factory import build_success_result
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.env_config import is_mock_llm
from traigent.utils.langchain_interceptor import capture_scope
from traigent.utils.logging import get_logger
from traigent.utils.multi_objective import ParetoFrontCalculator, ParetoPoint

logger = get_logger(__name__)
console = Console()


_THRESHOLD_TOLERANCE = 1e-9


def _declared_orientations(func_info: OptimizedFunction) -> dict[str, str]:
    schema = getattr(func_info.func, "objective_schema", None)
    definitions = getattr(schema, "objectives", [])
    return {definition.name: str(definition.orientation) for definition in definitions}


def check_prerequisites(func_info: OptimizedFunction) -> list[str]:
    """Return the blocking issues that stop ``traigent check`` from running.

    Shared by the real run and ``check --dry-run`` so both agree on what blocks.
    """
    issues = []

    if not func_info.has_dataset:
        issues.append("Missing evaluation dataset")

    if not func_info.objectives:
        issues.append("No objectives specified")

    config_space = func_info.decorator_config.get("configuration_space", {})
    if not config_space:
        issues.append("No configuration space specified")

    banded = [
        name
        for name, orientation in _declared_orientations(func_info).items()
        if orientation == "band"
    ]
    if banded:
        issues.append(
            "CLI validation does not support target-band objectives: "
            + ", ".join(banded)
        )

    return issues


class OptimizationValidator:
    """Validates that Traigent optimization improves over default parameters using Pareto efficiency."""

    def __init__(self, threshold_pct: float = 10.0) -> None:
        """Initialize the optimization validator.

        Args:
            threshold_pct: Minimum improvement threshold percentage (default: 10%)
        """
        if not math.isfinite(threshold_pct) or threshold_pct < 0:
            raise ValueError(
                f"threshold_pct must be a finite number >= 0, got {threshold_pct!r}"
            )
        self.threshold_pct = float(threshold_pct)
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
            console.print(
                f"   Baseline configuration ({func_info.baseline_source}): "
                f"{baseline_config or '{}'}"
            )

            # Step 2: Run Traigent optimization
            console.print("🚀 Running Traigent optimization...")
            optimized_metrics, optimized_config = await self._run_optimization(
                func_info
            )

            # Step 3: Compare using Pareto efficiency
            console.print("⚖️  Comparing results using Pareto efficiency...")
            incomparable = self._incomparable_objectives(
                baseline_metrics, optimized_metrics, func_info.objectives
            )
            if incomparable:
                return ValidationResult(
                    function_name=func_info.name,
                    baseline_metrics=baseline_metrics,
                    optimized_metrics=optimized_metrics,
                    is_superior=False,
                    improvement_details={},
                    blocking_issues=incomparable,
                    baseline_config=baseline_config,
                    optimized_config=optimized_config,
                    threshold_used=self.threshold,
                )
            is_superior, improvement_details = self._compare_results(
                baseline_metrics,
                optimized_metrics,
                func_info.objectives,
                objective_orientations=self._objective_orientations(func_info),
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
            logger.exception(f"Validation failed for {func_info.name}: {e}")
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
        return check_prerequisites(func_info)

    async def _run_baseline(
        self, func_info: OptimizedFunction
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run function with default parameters to get baseline metrics.

        Args:
            func_info: Function information

        Returns:
            Tuple of (baseline_metrics, baseline_config)
        """
        # The decorator's default_config is the declared default; signature
        # defaults are the fallback, and an empty config the last resort.
        baseline_config = func_info.baseline_config
        if not baseline_config:
            logger.warning(
                f"No default_config or signature defaults found for "
                f"{func_info.name}, using empty config"
            )

        baseline_metrics = await self._evaluate_configuration(
            func_info, baseline_config
        )

        logger.info(f"Baseline metrics for {func_info.name}: {baseline_metrics}")
        return baseline_metrics, baseline_config

    async def _run_optimization(
        self, func_info: OptimizedFunction
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run Traigent optimization to get optimized metrics.

        Args:
            func_info: Function information

        Returns:
            Tuple of (optimized_metrics, optimized_config)
        """
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
            optimized_metrics = self._filter_metrics_to_objectives(
                dict(result.best_metrics or {}), func_info, is_mock_llm()
            )
            optimized_config = result.best_config

            logger.info(f"Optimized metrics for {func_info.name}: {optimized_metrics}")
            return optimized_metrics, optimized_config

        except Exception as e:
            logger.exception(f"Optimization failed for {func_info.name}: {e}")
            raise RuntimeError(f"Optimization failed: {e}") from e

    def _create_evaluator(
        self, func_info: OptimizedFunction, is_mock_mode: bool
    ) -> LocalEvaluator:
        """Create a LocalEvaluator with appropriate settings for mock or real mode."""
        execution_mode = self._local_evaluator_mode(func_info, is_mock_mode)
        objectives = func_info.objectives or ["accuracy"]
        # The baseline must be scored by the same scorers the optimized side
        # uses (#2654); otherwise custom objectives are missing from it.
        from traigent.core.optimization_pipeline import build_metric_functions

        raw_metric_fns = getattr(func_info.func, "metric_functions", None)
        raw_scoring_fn = getattr(func_info.func, "scoring_function", None)
        metric_fns = build_metric_functions(
            raw_metric_fns if isinstance(raw_metric_fns, dict) else None,
            raw_scoring_fn if callable(raw_scoring_fn) else None,
            objectives,
        )
        return LocalEvaluator(
            metrics=objectives,
            # Per-example rows are needed to total the baseline's cost, as an
            # optimized trial does (#2824).
            detailed=True,
            execution_mode=execution_mode,
            metric_functions=metric_fns or None,
        )

    def _local_evaluator_mode(
        self, func_info: OptimizedFunction, is_mock_mode: bool
    ) -> str:
        """Resolve the evaluator's internal compatibility mode from public knobs."""
        policy = getattr(func_info.func, "execution_policy", None)
        legacy_mode = getattr(policy, "legacy_execution_mode_value", None)
        if isinstance(legacy_mode, str):
            return legacy_mode

        if is_mock_mode:
            policy = resolve_execution_policy(
                algorithm="random",
                offline=True,
                source_hint="cli_check_mock",
            )
            return policy.legacy_execution_mode_value

        algorithm = getattr(func_info.func, "algorithm", None) or "auto"
        offline = bool(getattr(func_info.func, "offline", False))
        try:
            policy = resolve_execution_policy(
                algorithm=algorithm,
                offline=offline,
                source_hint="cli_check",
            )
        except (TypeError, ValueError):
            policy = resolve_execution_policy(
                algorithm="auto",
                offline=offline,
                source_hint="cli_check",
            )
        return policy.legacy_execution_mode_value

    def _filter_metrics_to_objectives(
        self,
        metrics: dict[str, float],
        func_info: OptimizedFunction,
        is_mock_mode: bool,
    ) -> dict[str, float]:
        """Filter metrics to only include requested objectives."""
        if not func_info.objectives:
            return metrics

        if is_mock_mode:
            # In mock mode, silently return available objectives
            return {
                obj: metrics.get(obj, 0.0)
                for obj in func_info.objectives
                if obj in metrics
            }

        # In real mode, warn about missing objectives
        filtered: dict[str, float] = {}
        for objective in func_info.objectives:
            if objective in metrics:
                filtered[objective] = metrics[objective]
            else:
                logger.warning(
                    "Objective '%s' not present in evaluation metrics for %s",
                    objective,
                    func_info.name,
                )
        return filtered

    async def _evaluate_configuration(
        self,
        func_info: OptimizedFunction,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate a function with a specific configuration on its dataset."""
        # Load dataset first - needed for both mock mode and real evaluation
        try:
            dataset = func_info.func._load_dataset()
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Failed to load evaluation dataset for {func_info.name}: {exc}"
            ) from exc

        is_mock_mode = is_mock_llm()
        effective_config = (config or {}).copy()

        provider = func_info.func._provider
        configured_callable = provider.inject_config(
            func_info.func.func, effective_config, func_info.func.config_param
        )

        evaluator = self._create_evaluator(func_info, is_mock_mode)
        # Run inside the same trial scopes as an optimized trial: ``with_usage``
        # only attaches cost/latency metadata when a trial context is active.
        started = time.time()
        async with (
            capture_scope(ledger_owner=True),
            TrialContext(trial_id="baseline", metadata={"config": effective_config}),
        ):
            evaluation_result = await evaluator.evaluate(
                configured_callable, effective_config, dataset
            )

        # Assemble the metrics the way an optimized trial does, so cost and
        # latency objectives are measured on the baseline too (#2824).
        examples_attempted, total_cost = extract_cost_from_results(
            evaluation_result, None, "baseline"
        )
        trial = build_success_result(
            trial_id="baseline",
            evaluation_config=effective_config,
            eval_result=evaluation_result,
            duration=time.time() - started,
            examples_attempted=examples_attempted,
            total_cost=total_cost,
            optuna_trial_id=None,
            primary_objective=func_info.objectives[0] if func_info.objectives else None,
        )
        return self._filter_metrics_to_objectives(
            dict(trial.metrics or {}), func_info, is_mock_mode
        )

    @staticmethod
    def _non_finite_objectives(
        baseline_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
        objectives: list[str],
    ) -> list[str]:
        """Names of objectives whose value is NaN or infinite on either side."""
        bad = []
        for name in objectives:
            for metrics in (baseline_metrics, optimized_metrics):
                value = metrics.get(name)
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    bad.append(name)
                    break
        return bad

    def _incomparable_objectives(
        self,
        baseline_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
        objectives: list[str],
    ) -> list[str]:
        """Blocking reasons for objectives that cannot be compared (#2824)."""
        reasons = []
        for name in objectives:
            missing = [
                side
                for side, metrics in (
                    ("baseline", baseline_metrics),
                    ("optimized", optimized_metrics),
                )
                if name not in metrics
            ]
            if missing:
                reasons.append(
                    f"Objective '{name}' could not be compared: it was not "
                    f"measured on the {' or '.join(missing)} run"
                )
        for name in self._non_finite_objectives(
            baseline_metrics, optimized_metrics, objectives
        ):
            reasons.append(
                f"Objective '{name}' could not be compared: its value is not a "
                "finite number (NaN or infinity)"
            )
        return reasons

    def _compare_results(
        self,
        baseline_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
        objectives: list[str],
        objective_orientations: dict[str, str] | None = None,
    ) -> tuple[bool, dict[str, float]]:
        """Compare baseline vs optimized results using Pareto efficiency.

        Args:
            baseline_metrics: Metrics from baseline run
            optimized_metrics: Metrics from optimized run
            objectives: List of objective names

        Returns:
            Tuple of (is_superior, improvement_details)
        """
        if self._non_finite_objectives(baseline_metrics, optimized_metrics, objectives):
            return False, {}

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
        maximize_config = self._get_maximize_config(
            objectives, objective_orientations=objective_orientations
        )

        # Check if optimized dominates baseline over the CONFIGURED objective
        # set (not just whichever metrics both points happen to report), so a
        # universally-missing objective makes the pair non-comparable (#1941).
        optimized_dominates_baseline = optimized_point.dominates(
            baseline_point, maximize_config, objectives=objectives
        )

        # Calculate improvement percentages
        improvement_details = {}
        for metric in objectives:
            if metric in baseline_metrics and metric in optimized_metrics:
                baseline_val = baseline_metrics[metric]
                optimized_val = optimized_metrics[metric]

                maximize = maximize_config.get(metric, True)
                if baseline_val != 0:
                    improvement_pct = (
                        (optimized_val - baseline_val) / abs(baseline_val)
                    ) * 100
                    # Adjust sign based on maximize/minimize preference
                    if not maximize:  # If minimizing, flip sign
                        improvement_pct = -improvement_pct
                    improvement_details[metric] = improvement_pct
                else:
                    # A relative change from zero is unbounded: any strictly
                    # better value is an unbounded improvement and any worse
                    # value an unbounded regression. Never report it as 0.0%
                    # (#2803); the report shows the absolute delta instead.
                    delta = optimized_val - baseline_val
                    if not maximize:
                        delta = -delta
                    improvement_details[metric] = (
                        math.copysign(math.inf, delta) if delta != 0 else 0.0
                    )

        # Check threshold requirement: must be better in ALL metrics AND >threshold in ≥1 metric
        is_superior = self._check_superior_criteria(
            optimized_dominates_baseline, improvement_details
        )

        logger.info(f"Pareto dominance: {optimized_dominates_baseline}")
        logger.info(f"Improvement details: {improvement_details}")
        logger.info(f"Is superior: {is_superior}")

        return is_superior, improvement_details

    @staticmethod
    def _objective_orientations(func_info: OptimizedFunction) -> dict[str, str]:
        return _declared_orientations(func_info)

    def _get_maximize_config(
        self,
        objectives: list[str],
        objective_orientations: dict[str, str] | None = None,
    ) -> dict[str, bool]:
        """Get maximize configuration for objectives.

        Args:
            objectives: List of objective names

        Returns:
            Dictionary mapping objective names to maximize boolean
        """
        maximize_config: dict[str, bool] = {}
        for objective in objectives:
            explicit = (objective_orientations or {}).get(objective)
            if explicit == "band":
                raise ValueError(
                    "CLI validation does not support target-band objective "
                    f"'{objective}'"
                )
            maximize_config[objective] = (
                resolve_objective_orientation(objective, explicit) == "maximize"
            )
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
        # Strictly more than the threshold, in percent; the tolerance keeps
        # float noise (0.3 -> 0.33 is 10.000000000000009%) from flipping it.
        significant_improvement = any(
            improvement - self.threshold_pct > _THRESHOLD_TOLERANCE
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
