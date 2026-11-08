"""Parameter importance analysis for Traigent optimization."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from ..api.types import OptimizationResult, TrialResult, TrialStatus


@dataclass
class ImportanceResult:
    """Result of parameter importance analysis.

    ``importance_score`` and ``confidence_interval`` are Optional: both are
    None, and ``not_measurable_reason`` is set, when the data cannot support an
    importance value for the parameter (for example no replicated values,
    insufficient replication coverage, or a constant objective). None means
    "unknown", never "zero importance".
    """

    parameter_name: str
    importance_score: float | None
    confidence_interval: tuple[float, float] | None
    method: str
    sample_size: int
    # Set (and the score/interval are None) when the data cannot support an
    # importance value for this parameter; "not measurable" is not "zero".
    not_measurable_reason: str | None = None

    @property
    def is_measurable(self) -> bool:
        """True when an importance score could be computed."""
        return self.importance_score is not None

    def __str__(self) -> str:
        """String representation of importance result."""
        if self.importance_score is None:
            return (
                f"{self.parameter_name}: not measurable "
                f"({self.not_measurable_reason}; {self.method}, "
                f"n={self.sample_size})"
            )
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

    @staticmethod
    def _hashable_key(value: object) -> object:
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @staticmethod
    def _not_measurable(
        parameter: str, method: str, reason: str, n: int
    ) -> ImportanceResult:
        return ImportanceResult(
            parameter_name=parameter,
            importance_score=None,
            confidence_interval=None,
            method=method,
            sample_size=n,
            not_measurable_reason=reason,
        )

    @classmethod
    def _grouped_effect(
        cls, keys: list[object], ys: list[float]
    ) -> tuple[float | None, int, str | None]:
        """Share of objective variance explained by grouping on ``keys``.

        Only values seen at least twice are kept (a value seen once has no
        within-group spread, so keeping it would score every unique-valued
        knob 1.0), and at least half of the trials must survive. Returns
        ``(eta_squared, n_kept, None)`` or ``(None, n_kept, reason)``.
        """
        counts: dict[object, int] = defaultdict(int)
        for key in keys:
            counts[key] += 1
        kept = [i for i, key in enumerate(keys) if counts[key] >= 2]
        if len({keys[i] for i in kept}) < 2:
            return None, len(kept), "fewer than 2 parameter values with replication"
        if len(kept) * 2 < len(ys):
            return None, len(kept), "insufficient_replication_coverage"
        kept_y = [ys[i] for i in kept]
        mean = statistics.fmean(kept_y)
        total_ss = sum((v - mean) ** 2 for v in kept_y)
        if total_ss <= 0.0:
            return None, len(kept), "objective is constant"
        groups: dict[object, list[float]] = defaultdict(list)
        for i in kept:
            groups[keys[i]].append(ys[i])
        between_ss = sum(
            len(g) * (statistics.fmean(g) - mean) ** 2 for g in groups.values()
        )
        return min(1.0, max(0.0, between_ss / total_ss)), len(kept), None

    def _calculate_parameter_variance_importance(
        self,
        trials: list[TrialResult],
        parameter: str,
        objective_values: list[float],
        total_variance: float,
    ) -> ImportanceResult | None:
        """Calculate variance-based importance (eta-squared) for one parameter.

        Uses only replicated parameter values, like the permutation method, so
        a continuous knob whose values are all unique is "not measurable"
        rather than 1.0. No confidence interval is reported: none has been
        derived, and an invented one is worse than none.
        """
        keys: list[object] = []
        ys: list[float] = []
        for i, trial in enumerate(trials):
            if parameter in trial.config:
                keys.append(self._hashable_key(trial.config[parameter]))
                ys.append(float(objective_values[i]))

        if len(set(keys)) < 2:  # Need at least 2 different values
            return None

        eta2, n_kept, reason = self._grouped_effect(keys, ys)
        if eta2 is None:
            return self._not_measurable(
                parameter, "variance_based", reason or "not measurable", n_kept
            )
        return ImportanceResult(
            parameter_name=parameter,
            importance_score=eta2,
            confidence_interval=None,
            method="variance_based",
            sample_size=n_kept,
        )

    def analyze_correlation_based(
        self, trials: list[TrialResult]
    ) -> dict[str, ImportanceResult]:
        """Analyze parameter importance using an association-based method.

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

    @staticmethod
    def _encode_parameter_values(
        trials: list[TrialResult], parameter: str
    ) -> tuple[list[float], list[int], list[object]]:
        """Encode a parameter's values as floats (strings by first-seen order).

        The string codes are arbitrary and only used to detect distinct values;
        scores for string categories use a grouped effect, not these codes.

        Returns the encoded values, the indices of the trials they came
        from and the raw (unencoded) values; trials without the parameter or with other value types are
        skipped.
        """
        param_values: list[float] = []
        valid_indices: list[int] = []
        raw_values: list[object] = []
        categorical_mapping: dict[str, int] = {}

        for i, trial in enumerate(trials):
            if parameter not in trial.config:
                continue
            param_value = trial.config[parameter]

            # bool is an int subclass, so it is encoded by the numeric branch
            if isinstance(param_value, (int, float)):
                encoded = float(param_value)
            elif isinstance(param_value, str):
                if param_value not in categorical_mapping:
                    categorical_mapping[param_value] = len(categorical_mapping)
                encoded = float(categorical_mapping[param_value])
            else:
                continue
            param_values.append(encoded)
            valid_indices.append(i)
            raw_values.append(param_value)
        return param_values, valid_indices, raw_values

    def _calculate_parameter_correlation_importance(
        self, trials: list[TrialResult], parameter: str, objective_values: list[float]
    ) -> ImportanceResult | None:
        """Calculate association-based importance for a single parameter.

        Numeric values use |Pearson r|; unordered string categories use the
        correlation ratio. No confidence interval is reported.
        """
        param_values, valid_indices, raw_values = self._encode_parameter_values(
            trials, parameter
        )

        if len(param_values) < 10:
            return None

        filtered_objectives = [objective_values[i] for i in valid_indices]
        categorical = any(isinstance(v, str) for v in raw_values)

        # Distinct values are judged on raw values for categoricals: the
        # arbitrary string codes can collide with real numbers (0 vs "x").
        distinct = (
            {self._hashable_key(v) for v in raw_values}
            if categorical
            else set(param_values)
        )
        if len(distinct) < 2:
            return None

        if categorical:
            # A string category has no order, so a Pearson r on an arbitrary
            # integer encoding depends on first-seen order. Use the
            # correlation ratio (sqrt of eta-squared over replicated groups).
            keys = [self._hashable_key(v) for v in raw_values]
            eta2, n_kept, reason = self._grouped_effect(keys, filtered_objectives)
            if eta2 is None:
                return self._not_measurable(
                    parameter, "correlation_based", reason or "not measurable", n_kept
                )
            return ImportanceResult(
                parameter_name=parameter,
                importance_score=math.sqrt(eta2),
                confidence_interval=None,
                method="correlation_based",
                sample_size=n_kept,
            )

        correlation = self._calculate_correlation(param_values, filtered_objectives)
        return ImportanceResult(
            parameter_name=parameter,
            importance_score=abs(correlation),
            confidence_interval=None,
            method="correlation_based",
            sample_size=len(param_values),
        )

    def _calculate_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate a Pearson association coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
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
        """Analyze parameter importance using permutation of a grouped-mean model.

        For each parameter a grouped-mean predictor (the mean objective of the
        other trials sharing the same parameter value) is scored with
        leave-one-out squared error. The parameter's values are then shuffled
        across trials and the same predictor is re-scored; the increase in
        error, relative to the objective's variance, is the importance.

        Only trials that set the parameter are scored, and a parameter whose
        values are never repeated (e.g. a continuous value unique to each
        trial) cannot be evaluated this way. Such parameters are returned with
        ``importance_score=None`` and a ``not_measurable_reason`` instead of a
        misleading 0.0.

        The ``confidence_interval`` is mean +/- 1.96 standard deviations of the
        importance across shuffles of the observed trials. It describes shuffle
        spread only, not sampling uncertainty over trials.

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

        all_params: set[str] = set()
        for trial in successful_trials:
            all_params.update(trial.config.keys())

        n_permutations = max(1, n_permutations)
        results: dict[str, ImportanceResult] = {}
        for param in sorted(all_params):
            results[param] = self._permutation_importance_for_parameter(
                successful_trials, param, n_permutations
            )
        return results

    def _permutation_importance_for_parameter(
        self, trials: list[TrialResult], parameter: str, n_permutations: int
    ) -> ImportanceResult:
        import random

        def not_measurable(reason: str, n: int) -> ImportanceResult:
            return ImportanceResult(
                parameter_name=parameter,
                importance_score=None,
                confidence_interval=None,
                method="permutation_based",
                sample_size=n,
                not_measurable_reason=reason,
            )

        # Only runs that set the parameter carry information about it.
        keys: list[object] = []
        ys: list[float] = []
        for trial in trials:
            if parameter in trial.config:
                keys.append(self._hashable_key(trial.config[parameter]))
                ys.append(float(trial.metrics[self.objective]))
        if len(ys) < 2:
            return not_measurable("parameter set in fewer than 2 trials", len(ys))

        # Keep only replicated values: a value seen once cannot be predicted
        # from other trials, and keeping it would reward unique values.
        counts: dict[object, int] = defaultdict(int)
        for key in keys:
            counts[key] += 1
        kept = [i for i, key in enumerate(keys) if counts[key] >= 2]
        kept_groups = {keys[i] for i in kept}
        if len(kept_groups) < 2:
            return not_measurable(
                "fewer than 2 parameter values with replication", len(kept)
            )
        if len(kept) * 2 < len(ys):
            return not_measurable("insufficient_replication_coverage", len(kept))

        ks = [keys[i] for i in kept]
        y = [ys[i] for i in kept]
        n = len(y)
        y_mean = statistics.fmean(y)
        total_var = sum((v - y_mean) ** 2 for v in y) / n
        if total_var <= 0.0:
            return not_measurable("objective is constant", n)

        sums: dict[object, float] = defaultdict(float)
        sizes: dict[object, int] = defaultdict(int)
        for key, val in zip(ks, y, strict=True):
            sums[key] += val
            sizes[key] += 1

        def loo_error(assigned: list[object]) -> float:
            """Mean squared error of the grouped-mean predictor, leaving each
            trial out of the group it was originally measured in."""
            total = 0.0
            for own_key, key, val in zip(ks, assigned, y, strict=True):
                g_sum, g_n = sums[key], sizes[key]
                if key == own_key:
                    g_sum -= val
                    g_n -= 1
                total += (val - g_sum / g_n) ** 2
            return total / n

        baseline_error = loo_error(ks)
        rng = random.Random(0)  # NOSONAR - statistical permutation, no crypto
        scores: list[float] = []
        for _ in range(n_permutations):
            shuffled = ks.copy()
            rng.shuffle(shuffled)
            scores.append((loo_error(shuffled) - baseline_error) / total_var)

        mean_importance = min(1.0, max(0.0, statistics.fmean(scores)))
        std_importance = statistics.stdev(scores) if len(scores) > 1 else 0.0
        return ImportanceResult(
            parameter_name=parameter,
            importance_score=mean_importance,
            confidence_interval=(
                max(0.0, mean_importance - 1.96 * std_importance),
                min(1.0, mean_importance + 1.96 * std_importance),
            ),
            method="permutation_based",
            sample_size=n,
        )

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
        measurable = [
            r for r in analysis_results.values() if r.importance_score is not None
        ]
        sorted_results = sorted(
            measurable,
            key=lambda x: x.importance_score,  # type: ignore[arg-type,return-value]
            reverse=True,
        )
        return sorted_results[:top_k]

    def _format_method_results(
        self, results: dict[str, ImportanceResult], lines: list[str]
    ) -> None:
        """Append result lines to ``lines`` one at a time.

        Appending incrementally (not buffering) keeps already-formatted lines
        in the report if a later result is malformed and raises.
        """
        if not results:
            lines.append("  No significant importance detected")
            return
        for result in self.get_top_parameters(results, top_k=5):
            interval = (
                f" [{result.confidence_interval[0]:.3f}, "
                f"{result.confidence_interval[1]:.3f}]"
                if result.confidence_interval is not None
                else " (no interval)"
            )
            lines.append(
                f"  {result.parameter_name:15s}: {result.importance_score:.3f}"
                f"{interval}"
            )
        for result in results.values():
            if result.importance_score is None:
                lines.append(
                    f"  {result.parameter_name:15s}: not measurable "
                    f"({result.not_measurable_reason})"
                )

    @staticmethod
    def _format_report_summary(
        all_results: dict[str, dict[str, ImportanceResult]],
    ) -> list[str]:
        if not all_results:
            return [
                "  Insufficient data for reliable importance analysis",
                "  Recommendation: Run more trials for better insights",
            ]

        # Find parameters that are consistently important across methods
        param_scores = defaultdict(list)
        for method_results in all_results.values():
            for param, result in method_results.items():
                if result.importance_score is not None:
                    param_scores[param].append(result.importance_score)

        # Calculate average importance across methods
        avg_importance = {}
        for param, scores in param_scores.items():
            avg_importance[param] = statistics.mean(scores)

        # Sort by average importance
        sorted_params = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

        if not sorted_params:
            return ["  No clear parameter importance patterns detected"]

        lines = ["  Top parameters to focus on:"]
        for param, avg_score in sorted_params[:3]:
            lines.append(f"    • {param}: avg importance = {avg_score:.3f}")
        lines.extend(
            [
                "",
                "  Consider:",
                "    • Fine-tuning the top parameters with more granular values",
                "    • Removing or fixing low-importance parameters",
                "    • Using these insights for future optimization runs",
            ]
        )
        return lines

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
            ("Association-based", self.analyze_correlation_based),
        ]

        # Only run permutation-based if we have enough data
        if len(optimization_result.successful_trials) >= 20:
            methods.append(("Permutation-based", self.analyze_permutation_based))

        all_results: dict[str, dict[str, ImportanceResult]] = {}

        for method_name, method_func in methods:
            report_lines.append(f"📊 {method_name} Analysis:")
            report_lines.append("-" * 30)

            try:
                results = method_func(optimization_result.trials)
                all_results[method_name] = results
                self._format_method_results(results, report_lines)
            except Exception as e:
                report_lines.append(f"  Error in analysis: {e}")

            report_lines.append("")

        # Summary and recommendations
        report_lines.append("💡 Summary & Recommendations:")
        report_lines.append("-" * 30)
        report_lines.extend(self._format_report_summary(all_results))

        return "\n".join(report_lines)
