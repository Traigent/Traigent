"""Parameter importance analysis for TraiGent optimization."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from ..api.types import OptimizationResult, TrialResult, TrialStatus


@dataclass
class ImportanceResult:
    """Result of parameter importance analysis."""

    parameter_name: str
    importance_score: float
    confidence_interval: tuple[float, float]
    method: str
    sample_size: int

    def __str__(self) -> str:
        """String representation of importance result."""
        return (
            f"{self.parameter_name}: {self.importance_score:.3f} "
            f"({self.method}, n={self.sample_size})"
        )


class ParameterImportanceAnalyzer:
    """Analyzes parameter importance using various methods."""

    def __init__(self, objective: str = "accuracy") -> None:
        """Initialize parameter importance analyzer.

        Args:
            objective: Primary objective to analyze
        """
        self.objective = objective

    def analyze_variance_based(
        self, trials: list[TrialResult]
    ) -> dict[str, ImportanceResult]:
        """Analyze parameter importance using variance-based method.

        This method calculates how much variance in the objective can be
        explained by each parameter.

        Args:
            trials: List of completed trials

        Returns:
            Dictionary mapping parameter names to importance results
        """
        if not trials:
            return {}

        # Filter successful trials
        successful_trials = [
            t
            for t in trials
            if t.status == TrialStatus.COMPLETED.value
            and t.metrics
            and self.objective in t.metrics
        ]

        if len(successful_trials) < 10:  # Need sufficient data
            return {}

        results = {}

        # Get all parameters
        all_params: set[str] = set()
        for trial in successful_trials:
            all_params.update(trial.config.keys())

        # Calculate total variance
        objective_values = [t.metrics[self.objective] for t in successful_trials]
        total_variance = (
            statistics.variance(objective_values) if len(objective_values) > 1 else 0.0
        )

        if total_variance == 0:
            return {}

        for param in all_params:
            importance = self._calculate_parameter_variance_importance(
                successful_trials, param, objective_values, total_variance
            )

            if importance is not None:
                results[param] = importance

        return results

    def _calculate_parameter_variance_importance(
        self,
        trials: list[TrialResult],
        parameter: str,
        objective_values: list[float],
        total_variance: float,
    ) -> ImportanceResult | None:
        """Calculate variance-based importance for a single parameter."""
        # Group trials by parameter value
        param_groups = defaultdict(list)

        for i, trial in enumerate(trials):
            if parameter in trial.config:
                param_value = trial.config[parameter]
                param_groups[param_value].append(objective_values[i])

        if len(param_groups) < 2:  # Need at least 2 different values
            return None

        # Calculate between-group variance
        group_means = []
        group_sizes = []

        for group_values in param_groups.values():
            if len(group_values) > 0:
                group_means.append(statistics.mean(group_values))
                group_sizes.append(len(group_values))

        if len(group_means) < 2:
            return None

        # Calculate overall mean
        overall_mean = statistics.mean(objective_values)

        # Calculate between-group sum of squares
        between_ss = sum(
            size * (mean - overall_mean) ** 2
            for mean, size in zip(group_means, group_sizes)
        )

        # Calculate within-group sum of squares
        within_ss = 0.0
        for group_values in param_groups.values():
            if len(group_values) > 1:
                group_mean = statistics.mean(group_values)
                within_ss += sum((value - group_mean) ** 2 for value in group_values)

        # Calculate importance as explained variance ratio
        total_ss = between_ss + within_ss
        if total_ss > 0:
            importance_score = between_ss / total_ss
        else:
            importance_score = 0.0

        # Simple confidence interval (could be improved with proper statistical methods)
        len(param_groups)
        total_samples = len(trials)
        confidence_interval = (
            max(0.0, importance_score - 0.1 / math.sqrt(total_samples)),
            min(1.0, importance_score + 0.1 / math.sqrt(total_samples)),
        )

        return ImportanceResult(
            parameter_name=parameter,
            importance_score=importance_score,
            confidence_interval=confidence_interval,
            method="variance_based",
            sample_size=total_samples,
        )

    def analyze_correlation_based(
        self, trials: list[TrialResult]
    ) -> dict[str, ImportanceResult]:
        """Analyze parameter importance using correlation-based method.

        Args:
            trials: List of completed trials

        Returns:
            Dictionary mapping parameter names to importance results
        """
        successful_trials = [
            t
            for t in trials
            if t.status == TrialStatus.COMPLETED.value
            and t.metrics
            and self.objective in t.metrics
        ]

        if len(successful_trials) < 10:
            return {}

        results = {}
        objective_values = [t.metrics[self.objective] for t in successful_trials]

        # Get all parameters
        all_params: set[str] = set()
        for trial in successful_trials:
            all_params.update(trial.config.keys())

        for param in all_params:
            importance = self._calculate_parameter_correlation_importance(
                successful_trials, param, objective_values
            )

            if importance is not None:
                results[param] = importance

        return results

    def _calculate_parameter_correlation_importance(
        self, trials: list[TrialResult], parameter: str, objective_values: list[float]
    ) -> ImportanceResult | None:
        """Calculate correlation-based importance for a single parameter."""
        # Extract parameter values (handle categorical by encoding)
        param_values = []
        valid_indices = []

        # Create mapping for categorical values
        categorical_mapping: dict[str, int] = {}

        for i, trial in enumerate(trials):
            if parameter in trial.config:
                param_value = trial.config[parameter]

                # Handle different parameter types
                if isinstance(param_value, (int, float)):
                    param_values.append(float(param_value))
                    valid_indices.append(i)
                elif isinstance(param_value, str):
                    # Encode categorical values
                    if param_value not in categorical_mapping:
                        categorical_mapping[param_value] = len(categorical_mapping)
                    param_values.append(float(categorical_mapping[param_value]))
                    valid_indices.append(i)
                elif isinstance(param_value, bool):
                    param_values.append(float(param_value))
                    valid_indices.append(i)

        if len(param_values) < 10 or len(set(param_values)) < 2:
            return None

        # Calculate correlation
        filtered_objectives = [objective_values[i] for i in valid_indices]
        correlation = self._calculate_correlation(param_values, filtered_objectives)

        # Use absolute correlation as importance score
        importance_score = abs(correlation)

        # Simple confidence interval
        n = len(param_values)
        std_error = 1.0 / math.sqrt(n - 3) if n > 3 else 0.5
        confidence_interval = (
            max(0.0, importance_score - 1.96 * std_error),
            min(1.0, importance_score + 1.96 * std_error),
        )

        return ImportanceResult(
            parameter_name=parameter,
            importance_score=importance_score,
            confidence_interval=confidence_interval,
            method="correlation_based",
            sample_size=n,
        )

    def _calculate_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt(
            (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def analyze_permutation_based(
        self, trials: list[TrialResult], n_permutations: int = 100
    ) -> dict[str, ImportanceResult]:
        """Analyze parameter importance using permutation-based method.

        This method shuffles parameter values and measures the decrease in
        model performance to estimate importance.

        Args:
            trials: List of completed trials
            n_permutations: Number of permutations for each parameter

        Returns:
            Dictionary mapping parameter names to importance results
        """
        successful_trials = [
            t
            for t in trials
            if t.status == TrialStatus.COMPLETED.value
            and t.metrics
            and self.objective in t.metrics
        ]

        if len(successful_trials) < 20:  # Need sufficient data for permutations
            return {}

        results = {}
        baseline_score = self._calculate_baseline_score(successful_trials)

        # Get all parameters
        all_params: set[str] = set()
        for trial in successful_trials:
            all_params.update(trial.config.keys())

        for param in all_params:
            importance_scores = []

            for _ in range(n_permutations):
                # Create permuted trials
                permuted_trials = self._permute_parameter(successful_trials, param)
                permuted_score = self._calculate_baseline_score(permuted_trials)

                # Calculate importance as performance drop
                importance = baseline_score - permuted_score
                importance_scores.append(max(0.0, importance))  # Ensure non-negative

            if importance_scores:
                mean_importance = statistics.mean(importance_scores)
                std_importance = (
                    statistics.stdev(importance_scores)
                    if len(importance_scores) > 1
                    else 0.0
                )

                # Confidence interval
                confidence_interval = (
                    max(0.0, mean_importance - 1.96 * std_importance),
                    mean_importance + 1.96 * std_importance,
                )

                results[param] = ImportanceResult(
                    parameter_name=param,
                    importance_score=mean_importance,
                    confidence_interval=confidence_interval,
                    method="permutation_based",
                    sample_size=len(successful_trials),
                )

        return results

    def _calculate_baseline_score(self, trials: list[TrialResult]) -> float:
        """Calculate baseline score for permutation importance."""
        objective_values = [t.metrics[self.objective] for t in trials]
        return statistics.mean(objective_values) if objective_values else 0.0

    def _permute_parameter(
        self, trials: list[TrialResult], parameter: str
    ) -> list[TrialResult]:
        """Create trials with permuted parameter values."""
        import copy
        import random

        # Extract parameter values
        param_values = []
        for trial in trials:
            if parameter in trial.config:
                param_values.append(trial.config[parameter])

        if not param_values:
            return trials

        # Shuffle parameter values
        shuffled_values = param_values.copy()
        random.shuffle(shuffled_values)

        # Create permuted trials
        permuted_trials = []
        shuffle_index = 0

        for trial in trials:
            new_trial = copy.deepcopy(trial)
            if parameter in trial.config and shuffle_index < len(shuffled_values):
                new_trial.config[parameter] = shuffled_values[shuffle_index]
                shuffle_index += 1
            permuted_trials.append(new_trial)

        return permuted_trials

    def get_top_parameters(
        self, analysis_results: dict[str, ImportanceResult], top_k: int = 5
    ) -> list[ImportanceResult]:
        """Get top K most important parameters.

        Args:
            analysis_results: Results from importance analysis
            top_k: Number of top parameters to return

        Returns:
            List of top importance results, sorted by importance score
        """
        sorted_results = sorted(
            analysis_results.values(), key=lambda x: x.importance_score, reverse=True
        )
        return sorted_results[:top_k]

    def generate_importance_report(
        self, optimization_result: OptimizationResult
    ) -> str:
        """Generate a comprehensive importance analysis report.

        Args:
            optimization_result: Optimization result to analyze

        Returns:
            Formatted importance analysis report
        """
        report_lines = []
        report_lines.append("🔍 Parameter Importance Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Optimization ID: {optimization_result.optimization_id}")
        report_lines.append(f"Objective: {self.objective}")
        report_lines.append(f"Total trials: {len(optimization_result.trials)}")
        report_lines.append(
            f"Successful trials: {len(optimization_result.successful_trials)}"
        )
        report_lines.append("")

        # Run different analysis methods
        methods = [
            ("Variance-based", self.analyze_variance_based),
            ("Correlation-based", self.analyze_correlation_based),
        ]

        # Only run permutation-based if we have enough data
        if len(optimization_result.successful_trials) >= 20:
            methods.append(("Permutation-based", self.analyze_permutation_based))

        all_results = {}

        for method_name, method_func in methods:
            report_lines.append(f"📊 {method_name} Analysis:")
            report_lines.append("-" * 30)

            try:
                results = method_func(optimization_result.trials)
                all_results[method_name] = results

                if results:
                    top_params = self.get_top_parameters(results, top_k=5)
                    for result in top_params:
                        ci_low, ci_high = result.confidence_interval
                        report_lines.append(
                            f"  {result.parameter_name:15s}: {result.importance_score:.3f} "
                            f"[{ci_low:.3f}, {ci_high:.3f}]"
                        )
                else:
                    report_lines.append("  No significant importance detected")

            except Exception as e:
                report_lines.append(f"  Error in analysis: {e}")

            report_lines.append("")

        # Summary and recommendations
        report_lines.append("💡 Summary & Recommendations:")
        report_lines.append("-" * 30)

        if all_results:
            # Find parameters that are consistently important across methods
            param_scores = defaultdict(list)
            for method_results in all_results.values():
                for param, result in method_results.items():
                    param_scores[param].append(result.importance_score)

            # Calculate average importance across methods
            avg_importance = {}
            for param, scores in param_scores.items():
                avg_importance[param] = statistics.mean(scores)

            # Sort by average importance
            sorted_params = sorted(
                avg_importance.items(), key=lambda x: x[1], reverse=True
            )

            if sorted_params:
                report_lines.append("  Top parameters to focus on:")
                for param, avg_score in sorted_params[:3]:
                    report_lines.append(
                        f"    • {param}: avg importance = {avg_score:.3f}"
                    )

                report_lines.append("")
                report_lines.append("  Consider:")
                report_lines.append(
                    "    • Fine-tuning the top parameters with more granular values"
                )
                report_lines.append(
                    "    • Removing or fixing low-importance parameters"
                )
                report_lines.append(
                    "    • Using these insights for future optimization runs"
                )
            else:
                report_lines.append("  No clear parameter importance patterns detected")
        else:
            report_lines.append("  Insufficient data for reliable importance analysis")
            report_lines.append("  Recommendation: Run more trials for better insights")

        return "\n".join(report_lines)
