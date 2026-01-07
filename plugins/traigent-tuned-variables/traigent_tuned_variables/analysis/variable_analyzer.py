"""Centralized variable analysis for optimization results.

Provides post-optimization analysis including:
- Variable importance scoring
- Value-level performance rankings
- Dominated value detection
- Elimination suggestions
- Configuration space refinement
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

# numpy is optional - only needed for percentile calculation
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import ParameterRange
    from traigent.core.result_types import OptimizationResult

logger = logging.getLogger(__name__)


def _mean(values: list[float]) -> float:
    """Calculate mean using statistics stdlib."""
    if not values:
        return 0.0
    return statistics.mean(values)


def _stdev(values: list[float]) -> float:
    """Calculate sample standard deviation using statistics stdlib."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _variance(values: list[float]) -> float:
    """Calculate sample variance using statistics stdlib."""
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def _percentile(values: list[float], p: float) -> float:
    """Calculate percentile.

    Uses numpy if available for exact calculation,
    falls back to linear interpolation otherwise.
    """
    if not values:
        return 0.0

    if HAS_NUMPY:
        return float(np.percentile(values, p))

    # Fallback: simple linear interpolation
    sorted_values = sorted(values)
    n = len(sorted_values)
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


class EliminationAction(str, Enum):
    """Actions for variable/value elimination."""

    KEEP = "keep"
    ELIMINATE = "eliminate"
    PRUNE_VALUES = "prune_values"
    NARROW_RANGE = "narrow_range"


@dataclass
class EliminationSuggestion:
    """Suggestion for variable or value elimination."""

    variable: str
    action: EliminationAction
    reason: str
    importance_score: float
    dominated_values: list[Any] | None = None
    suggested_range: tuple[float, float] | None = None
    suggested_values: list[Any] | None = None


@dataclass
class ValueRanking:
    """Performance ranking for a categorical value."""

    value: Any
    mean_score: float
    std_score: float
    trial_count: int
    is_dominated: bool = False
    confidence_interval: tuple[float, float] | None = None


@dataclass
class VariableAnalysis:
    """Complete analysis for a single variable."""

    name: str
    var_type: Literal["numeric", "categorical"]
    importance: float
    correlation: float | None = None
    p_value: float | None = None
    suggestion: EliminationSuggestion | None = None
    value_rankings: list[ValueRanking] | None = None


@dataclass
class OptimizationAnalysis:
    """Post-optimization analysis with elimination suggestions."""

    variables: dict[str, VariableAnalysis] = field(default_factory=dict)
    elimination_suggestions: list[EliminationSuggestion] = field(default_factory=list)
    interactions: dict[tuple[str, str], float] = field(default_factory=dict)
    refined_space: dict[str, Any] = field(default_factory=dict)


class VariableAnalyzer:
    """Centralized analysis of optimization results.

    Analyzes trial history to provide:
    - Variable importance scores
    - Value-level performance rankings
    - Dominated value detection
    - Elimination suggestions
    - Refined configuration space
    """

    def __init__(
        self,
        result: OptimizationResult,
        *,
        importance_method: Literal[
            "variance", "correlation", "permutation"
        ] = "variance",
        significance_threshold: float = 0.05,
        min_trials_per_value: int = 3,
        elimination_threshold: float = 0.05,
    ):
        """Initialize the analyzer.

        Args:
            result: OptimizationResult containing trials and configuration space
            importance_method: Method for calculating variable importance
            significance_threshold: P-value threshold for significance tests
            min_trials_per_value: Minimum trials needed per value for analysis
            elimination_threshold: Importance below this suggests elimination
        """
        self.result = result
        self.importance_method = importance_method
        self.significance_threshold = significance_threshold
        self.min_trials_per_value = min_trials_per_value
        self.elimination_threshold = elimination_threshold

        # Cache computed values
        self._importance_cache: dict[str, dict[str, float]] = {}
        self._value_stats_cache: dict[str, dict[Any, list[float]]] = {}

    def analyze(self, objective: str) -> OptimizationAnalysis:
        """Run full analysis for given objective.

        Args:
            objective: Name of the objective to analyze

        Returns:
            OptimizationAnalysis with all results
        """
        analysis = OptimizationAnalysis()

        # Get configuration space from result
        config_space = self._get_config_space()
        if not config_space:
            logger.warning("No configuration space found in result")
            return analysis

        # Analyze each variable
        importance_scores = self.get_variable_importance(objective)

        for var_name in config_space:
            var_type = self._get_variable_type(var_name)
            importance = importance_scores.get(var_name, 0.0)

            var_analysis = VariableAnalysis(
                name=var_name,
                var_type=var_type,
                importance=importance,
            )

            # Categorical-specific analysis
            if var_type == "categorical":
                var_analysis.value_rankings = self.get_value_rankings(
                    var_name, objective
                )

            # Generate elimination suggestion
            suggestion = self._generate_suggestion(var_name, objective, var_analysis)
            var_analysis.suggestion = suggestion

            analysis.variables[var_name] = var_analysis

            if suggestion.action != EliminationAction.KEEP:
                analysis.elimination_suggestions.append(suggestion)

        # Sort suggestions by importance (least important first for elimination)
        analysis.elimination_suggestions.sort(key=lambda s: s.importance_score)

        # Generate refined space
        analysis.refined_space = self.get_refined_space([objective])

        return analysis

    def get_variable_importance(self, objective: str) -> dict[str, float]:
        """Get importance scores for all variables.

        Args:
            objective: Name of the objective

        Returns:
            Dictionary mapping variable names to importance scores
        """
        if objective in self._importance_cache:
            return self._importance_cache[objective]

        # Try to use existing importance analyzer from traigent
        try:
            from traigent.utils.importance import ParameterImportanceAnalyzer

            analyzer = ParameterImportanceAnalyzer(self._get_trials_as_dicts())
            if self.importance_method == "variance":
                results = analyzer.analyze_variance_based(objective)
            elif self.importance_method == "correlation":
                results = analyzer.analyze_correlation_based(objective)
            else:
                results = analyzer.analyze_permutation_based(objective)

            importance = {r.parameter: r.importance for r in results}
        except (ImportError, AttributeError, TypeError):
            # Fallback to simple variance-based importance if core analyzer fails
            # This can happen if trials don't have expected format (e.g., mock trials)
            importance = self._compute_variance_importance(objective)

        self._importance_cache[objective] = importance
        return importance

    def get_value_rankings(
        self,
        variable: str,
        objective: str,
    ) -> list[ValueRanking]:
        """Rank categorical values by performance.

        Args:
            variable: Name of the categorical variable
            objective: Name of the objective

        Returns:
            List of ValueRanking sorted by mean score (descending)
        """
        value_stats = self._get_value_stats(variable, objective)
        rankings = []

        for value, scores in value_stats.items():
            if len(scores) < self.min_trials_per_value:
                continue

            mean_score = _mean(scores)
            std_score = _stdev(scores)

            # Calculate confidence interval
            n = len(scores)
            se = std_score / math.sqrt(n) if n > 1 else 0.0
            ci = (mean_score - 1.96 * se, mean_score + 1.96 * se)

            rankings.append(
                ValueRanking(
                    value=value,
                    mean_score=mean_score,
                    std_score=std_score,
                    trial_count=len(scores),
                    confidence_interval=ci,
                )
            )

        # Sort by mean score (descending)
        rankings.sort(key=lambda r: r.mean_score, reverse=True)

        # Mark dominated values
        if rankings:
            best_lower_bound = (
                rankings[0].confidence_interval[0]
                if rankings[0].confidence_interval
                else rankings[0].mean_score
            )
            for ranking in rankings[1:]:
                if ranking.confidence_interval:
                    # Value is dominated if its upper CI bound < best's lower CI bound
                    if ranking.confidence_interval[1] < best_lower_bound:
                        ranking.is_dominated = True

        return rankings

    def get_dominated_values(
        self,
        variable: str,
        objectives: list[str],
    ) -> list[Any]:
        """Find values that are Pareto-dominated across objectives.

        Args:
            variable: Name of the categorical variable
            objectives: List of objectives to consider

        Returns:
            List of dominated values
        """
        # For single objective, use value rankings
        if len(objectives) == 1:
            rankings = self.get_value_rankings(variable, objectives[0])
            return [r.value for r in rankings if r.is_dominated]

        # For multiple objectives, compute Pareto dominance
        value_stats = {}
        for obj in objectives:
            stats = self._get_value_stats(variable, obj)
            for value, scores in stats.items():
                if value not in value_stats:
                    value_stats[value] = {}
                value_stats[value][obj] = _mean(scores) if scores else 0.0

        # Check Pareto dominance
        dominated = []
        values = list(value_stats.keys())

        for i, v1 in enumerate(values):
            is_dominated = False
            for j, v2 in enumerate(values):
                if i == j:
                    continue
                # Check if v2 dominates v1 (v2 >= v1 in all objectives, > in at least one)
                dominates = True
                strictly_better = False
                for obj in objectives:
                    s1 = value_stats[v1].get(obj, 0.0)
                    s2 = value_stats[v2].get(obj, 0.0)
                    if s2 < s1:
                        dominates = False
                        break
                    if s2 > s1:
                        strictly_better = True
                if dominates and strictly_better:
                    is_dominated = True
                    break
            if is_dominated:
                dominated.append(v1)

        return dominated

    def suggest_range_adjustment(
        self,
        variable: str,
        objective: str,
    ) -> tuple[float, float] | None:
        """Suggest narrowed range based on best trials.

        Args:
            variable: Name of the numeric variable
            objective: Name of the objective

        Returns:
            Suggested (low, high) range or None if no adjustment needed
        """
        value_stats = self._get_value_stats(variable, objective)
        if not value_stats:
            return None

        # Get values from top quartile of trials
        all_values = []
        all_scores = []
        for value, scores in value_stats.items():
            for score in scores:
                all_values.append(float(value))
                all_scores.append(score)

        if len(all_values) < 4:
            return None

        # Find values in top 25% of scores
        threshold = _percentile(all_scores, 75)
        top_values = [
            v for v, s in zip(all_values, all_scores, strict=True) if s >= threshold
        ]

        if not top_values:
            return None

        # Suggest range covering top values with some margin
        margin = 0.1 * (max(top_values) - min(top_values))
        return (min(top_values) - margin, max(top_values) + margin)

    def get_refined_space(
        self,
        objectives: list[str],
        *,
        prune_low_importance: bool = True,
        prune_dominated_values: bool = True,
        narrow_ranges: bool = True,
        return_typed: bool = True,
    ) -> dict[str, ParameterRange] | dict[str, Any]:
        """Return pruned configuration space for next optimization.

        Args:
            objectives: List of objectives to consider
            prune_low_importance: Remove variables with low importance
            prune_dominated_values: Remove dominated categorical values
            narrow_ranges: Narrow numeric ranges to promising regions
            return_typed: If True, return typed ParameterRange objects;
                         if False, return raw values (lists, tuples)

        Returns:
            Refined configuration space with ParameterRange objects or raw values
        """
        config_space = self._get_config_space()
        if not config_space:
            return {}

        refined: dict[str, Any] = {}
        primary_objective = objectives[0] if objectives else None

        for var_name, var_config in config_space.items():
            var_type = self._get_variable_type(var_name)

            # Check importance
            if prune_low_importance and primary_objective:
                importance = self.get_variable_importance(primary_objective).get(
                    var_name, 0.0
                )
                if importance < self.elimination_threshold:
                    logger.info(
                        f"Eliminating {var_name}: low importance ({importance:.3f})"
                    )
                    continue

            if var_type == "categorical":
                if prune_dominated_values:
                    dominated = self.get_dominated_values(var_name, objectives)
                    if dominated and isinstance(var_config, (list, tuple)):
                        remaining = [v for v in var_config if v not in dominated]
                        if remaining:
                            refined[var_name] = (
                                self._to_parameter_range(
                                    var_name, remaining, "categorical"
                                )
                                if return_typed
                                else remaining
                            )
                            logger.info(
                                f"Pruned {var_name}: removed {dominated}, keeping {remaining}"
                            )
                            continue
                refined[var_name] = (
                    self._to_parameter_range(var_name, var_config, "categorical")
                    if return_typed
                    else var_config
                )

            elif var_type == "numeric":
                if narrow_ranges and primary_objective:
                    suggested = self.suggest_range_adjustment(
                        var_name, primary_objective
                    )
                    if suggested:
                        refined[var_name] = (
                            self._to_parameter_range(var_name, suggested, "numeric")
                            if return_typed
                            else suggested
                        )
                        logger.info(f"Narrowed {var_name}: {suggested}")
                        continue
                refined[var_name] = (
                    self._to_parameter_range(var_name, var_config, "numeric")
                    if return_typed
                    else var_config
                )

            else:
                refined[var_name] = (
                    self._to_parameter_range(var_name, var_config, "categorical")
                    if return_typed
                    else var_config
                )

        return refined

    def _to_parameter_range(
        self,
        name: str,
        value: Any,
        var_type: Literal["numeric", "categorical"],
    ) -> ParameterRange:
        """Convert raw value to typed ParameterRange.

        Args:
            name: Variable name
            value: Raw value (list, tuple, etc.)
            var_type: Variable type

        Returns:
            Typed ParameterRange object
        """
        from traigent.api.parameter_ranges import Choices, IntRange, Range

        if var_type == "categorical":
            if isinstance(value, (list, tuple)):
                return Choices(list(value), name=name)
            return Choices([value], name=name)

        elif var_type == "numeric":
            if isinstance(value, tuple) and len(value) == 2:
                low, high = value
                # Check if integer or float
                if isinstance(low, int) and isinstance(high, int):
                    return IntRange(low, high, name=name)
                return Range(float(low), float(high), name=name)
            elif isinstance(value, (int, float)):
                # Single value - return as tight range
                return Range(float(value), float(value), name=name)

        # Fallback: treat as categorical
        if isinstance(value, (list, tuple)):
            return Choices(list(value), name=name)
        return Choices([value], name=name)

    # === Private helpers ===

    def _get_config_space(self) -> dict[str, Any]:
        """Extract configuration space from result."""
        if hasattr(self.result, "configuration_space"):
            return self.result.configuration_space or {}
        if hasattr(self.result, "config_space"):
            return self.result.config_space or {}
        return {}

    def _get_trials_as_dicts(self) -> list[dict]:
        """Convert trials to list of dicts for analysis."""
        trials = []
        if hasattr(self.result, "trials"):
            for trial in self.result.trials:
                if hasattr(trial, "config") and hasattr(trial, "metrics"):
                    trial_dict = {**trial.config, **trial.metrics}
                    trials.append(trial_dict)
        return trials

    def _get_variable_type(self, var_name: str) -> Literal["numeric", "categorical"]:
        """Determine if variable is numeric or categorical."""
        config_space = self._get_config_space()
        var_config = config_space.get(var_name)

        # Tuple of exactly 2 numbers = numeric range (low, high)
        if (
            isinstance(var_config, tuple)
            and len(var_config) == 2
            and all(isinstance(v, (int, float)) for v in var_config)
        ):
            return "numeric"

        # List of numbers - could be categorical or discrete numeric
        if isinstance(var_config, (list, tuple)) and all(
            isinstance(v, (int, float)) for v in var_config
        ):
            if len(var_config) <= 5:
                return "categorical"
            return "numeric"

        # Plain list = categorical
        if isinstance(var_config, list):
            return "categorical"

        return "categorical"

    def _get_value_stats(self, variable: str, objective: str) -> dict[Any, list[float]]:
        """Get per-value statistics from trials."""
        cache_key = f"{variable}:{objective}"
        if cache_key in self._value_stats_cache:
            return self._value_stats_cache[cache_key]

        stats: dict[Any, list[float]] = {}
        trials = self._get_trials_as_dicts()

        for trial in trials:
            if variable in trial and objective in trial:
                value = trial[variable]
                score = trial[objective]
                if value not in stats:
                    stats[value] = []
                stats[value].append(float(score))

        self._value_stats_cache[cache_key] = stats
        return stats

    def _compute_variance_importance(self, objective: str) -> dict[str, float]:
        """Fallback variance-based importance calculation."""
        trials = self._get_trials_as_dicts()
        if not trials:
            return {}

        config_space = self._get_config_space()
        importance = {}

        for var_name in config_space:
            value_stats = self._get_value_stats(var_name, objective)
            if not value_stats:
                importance[var_name] = 0.0
                continue

            # Calculate between-group variance
            group_means = [_mean(scores) for scores in value_stats.values() if scores]
            if len(group_means) < 2:
                importance[var_name] = 0.0
                continue

            between_var = _variance(group_means) if len(group_means) > 1 else 0.0
            all_scores = [s for scores in value_stats.values() for s in scores]
            total_var = _variance(all_scores) if len(all_scores) > 1 else 1.0

            importance[var_name] = (
                float(between_var / total_var) if total_var > 0 else 0.0
            )

        return importance

    def _generate_suggestion(
        self,
        var_name: str,
        objective: str,
        var_analysis: VariableAnalysis,
    ) -> EliminationSuggestion:
        """Generate elimination suggestion for a variable."""
        importance = var_analysis.importance

        # Low importance -> suggest elimination
        if importance < self.elimination_threshold:
            return EliminationSuggestion(
                variable=var_name,
                action=EliminationAction.ELIMINATE,
                reason=f"Low importance ({importance:.3f} < {self.elimination_threshold})",
                importance_score=importance,
            )

        # Categorical with dominated values -> suggest pruning
        if var_analysis.var_type == "categorical" and var_analysis.value_rankings:
            dominated = [r.value for r in var_analysis.value_rankings if r.is_dominated]
            if dominated:
                remaining = [
                    r.value for r in var_analysis.value_rankings if not r.is_dominated
                ]
                return EliminationSuggestion(
                    variable=var_name,
                    action=EliminationAction.PRUNE_VALUES,
                    reason=f"Values {dominated} are dominated",
                    importance_score=importance,
                    dominated_values=dominated,
                    suggested_values=remaining,
                )

        # Numeric with suggested range -> suggest narrowing
        if var_analysis.var_type == "numeric":
            suggested_range = self.suggest_range_adjustment(var_name, objective)
            if suggested_range:
                return EliminationSuggestion(
                    variable=var_name,
                    action=EliminationAction.NARROW_RANGE,
                    reason=f"Best trials concentrated in {suggested_range}",
                    importance_score=importance,
                    suggested_range=suggested_range,
                )

        # Default: keep
        return EliminationSuggestion(
            variable=var_name,
            action=EliminationAction.KEEP,
            reason=f"Important variable ({importance:.3f})",
            importance_score=importance,
        )
