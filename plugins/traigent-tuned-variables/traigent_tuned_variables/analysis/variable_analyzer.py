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
    from traigent.api.types import OptimizationResult

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


@dataclass
class MultiObjectiveVariableAnalysis:
    """Analysis for a single variable across multiple objectives."""

    name: str
    var_type: Literal["numeric", "categorical"]
    importance_by_objective: dict[str, float]
    aggregate_importance: float
    pareto_dominated_values: list[Any] | None = None
    suggestion: EliminationSuggestion | None = None


@dataclass
class MultiObjectiveAnalysis:
    """Post-optimization analysis across multiple objectives.

    Provides unified analysis considering all objectives simultaneously,
    including Pareto dominance and aggregated importance scores.
    """

    objectives: list[str] = field(default_factory=list)
    variables: dict[str, MultiObjectiveVariableAnalysis] = field(default_factory=dict)
    elimination_suggestions: list[EliminationSuggestion] = field(default_factory=list)
    pareto_frontier_values: dict[str, list[Any]] = field(default_factory=dict)
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
        configuration_space: dict[str, Any] | None = None,
        objectives: list[str] | None = None,
        directions: dict[str, Literal["maximize", "minimize"]] | None = None,
        importance_method: Literal[
            "variance", "correlation", "permutation"
        ] = "variance",
        significance_threshold: float = 0.05,
        min_trials_per_value: int = 3,
        elimination_threshold: float = 0.05,
    ):
        """Initialize the analyzer.

        Args:
            result: OptimizationResult containing trials (and optionally config space)
            configuration_space: Explicit configuration space. If not provided,
                extracted from result.configuration_space or result.config_space.
                Use this when analyzing results that don't include their config space.
            objectives: List of objective names. If not provided, extracted from result.
            directions: Dictionary mapping objective names to "maximize" or "minimize".
                If not provided, defaults to "maximize" for all objectives.
                Used for ranking values and detecting dominance.
            importance_method: Method for calculating variable importance
            significance_threshold: P-value threshold for significance tests
            min_trials_per_value: Minimum trials needed per value for analysis
            elimination_threshold: Importance below this suggests elimination
        """
        self.result = result
        self._explicit_config_space = configuration_space
        self._explicit_objectives = objectives
        self._directions = directions or {}
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

    def analyze_multi_objective(
        self,
        objectives: list[str],
        *,
        aggregation: Literal["mean", "max", "min"] = "mean",
    ) -> MultiObjectiveAnalysis:
        """Run full analysis across multiple objectives.

        Analyzes variable importance and value performance across all objectives
        simultaneously, using Pareto dominance for value elimination.

        Args:
            objectives: List of objective names to analyze
            aggregation: Method to aggregate importance scores across objectives.
                - "mean": Average importance (default, balanced view)
                - "max": Maximum importance (keep if important for any objective)
                - "min": Minimum importance (eliminate only if unimportant for all)

        Returns:
            MultiObjectiveAnalysis with unified results across all objectives
        """
        if not objectives:
            return MultiObjectiveAnalysis()

        config_space = self._get_config_space()
        if not config_space:
            logger.warning("No configuration space found in result")
            return MultiObjectiveAnalysis(objectives=objectives)

        # Build analysis
        analysis = MultiObjectiveAnalysis(objectives=objectives)
        importance_by_var = self._compute_importance_by_variable(
            config_space, objectives
        )

        # Analyze each variable
        for var_name in config_space:
            var_analysis = self._analyze_variable_multi_objective(
                var_name, objectives, importance_by_var[var_name], aggregation
            )
            analysis.variables[var_name] = var_analysis

            if var_analysis.suggestion and var_analysis.suggestion.action != EliminationAction.KEEP:
                analysis.elimination_suggestions.append(var_analysis.suggestion)

        # Sort and finalize
        analysis.elimination_suggestions.sort(key=lambda s: s.importance_score)
        analysis.pareto_frontier_values = self._compute_pareto_frontiers(
            analysis.variables
        )
        analysis.refined_space = self.get_refined_space(
            objectives,
            prune_low_importance=True,
            prune_dominated_values=True,
            narrow_ranges=True,
        )

        return analysis

    def _compute_importance_by_variable(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute importance for each variable across all objectives."""
        importance_by_var: dict[str, dict[str, float]] = {}
        for var_name in config_space:
            importance_by_var[var_name] = {
                obj: self.get_variable_importance(obj).get(var_name, 0.0)
                for obj in objectives
            }
        return importance_by_var

    def _analyze_variable_multi_objective(
        self,
        var_name: str,
        objectives: list[str],
        obj_importance: dict[str, float],
        aggregation: Literal["mean", "max", "min"],
    ) -> MultiObjectiveVariableAnalysis:
        """Analyze a single variable across multiple objectives."""
        var_type = self._get_variable_type(var_name)
        aggregate_imp = self._aggregate_importance(obj_importance, aggregation)

        pareto_dominated = None
        if var_type == "categorical":
            pareto_dominated = self.get_dominated_values(var_name, objectives)

        var_analysis = MultiObjectiveVariableAnalysis(
            name=var_name,
            var_type=var_type,
            importance_by_objective=obj_importance,
            aggregate_importance=aggregate_imp,
            pareto_dominated_values=pareto_dominated,
        )

        var_analysis.suggestion = self._generate_multi_objective_suggestion(
            var_name, objectives, var_analysis
        )
        return var_analysis

    def _aggregate_importance(
        self,
        obj_importance: dict[str, float],
        aggregation: Literal["mean", "max", "min"],
    ) -> float:
        """Aggregate importance values using specified method."""
        values = list(obj_importance.values())
        if not values:
            return 0.0
        if aggregation == "mean":
            return _mean(values)
        if aggregation == "max":
            return max(values)
        return min(values)  # aggregation == "min"

    def _compute_pareto_frontiers(
        self,
        variables: dict[str, MultiObjectiveVariableAnalysis],
    ) -> dict[str, list[Any]]:
        """Compute Pareto frontier values for each categorical variable."""
        frontiers: dict[str, list[Any]] = {}
        for var_name, var_analysis in variables.items():
            if var_analysis.var_type == "categorical":
                all_values = self._get_all_values(var_name)
                dominated = var_analysis.pareto_dominated_values or []
                frontiers[var_name] = [v for v in all_values if v not in dominated]
        return frontiers

    def _generate_multi_objective_suggestion(
        self,
        var_name: str,
        objectives: list[str],
        var_analysis: MultiObjectiveVariableAnalysis,
    ) -> EliminationSuggestion:
        """Generate elimination suggestion for multi-objective analysis."""
        importance = var_analysis.aggregate_importance

        # Low aggregate importance -> suggest elimination
        if importance < self.elimination_threshold:
            return EliminationSuggestion(
                variable=var_name,
                action=EliminationAction.ELIMINATE,
                reason=(
                    f"Low aggregate importance ({importance:.3f} < "
                    f"{self.elimination_threshold}) across {len(objectives)} objectives"
                ),
                importance_score=importance,
            )

        # Categorical with Pareto-dominated values -> suggest pruning
        if (
            var_analysis.var_type == "categorical"
            and var_analysis.pareto_dominated_values
        ):
            dominated = var_analysis.pareto_dominated_values
            all_values = self._get_all_values(var_name)
            remaining = [v for v in all_values if v not in dominated]

            return EliminationSuggestion(
                variable=var_name,
                action=EliminationAction.PRUNE_VALUES,
                reason=(
                    f"Values {dominated} are Pareto-dominated across "
                    f"{len(objectives)} objectives"
                ),
                importance_score=importance,
                dominated_values=dominated,
                suggested_values=remaining,
            )

        # Default: keep
        return EliminationSuggestion(
            variable=var_name,
            action=EliminationAction.KEEP,
            reason=(
                f"Important variable (aggregate: {importance:.3f}) "
                f"across {len(objectives)} objectives"
            ),
            importance_score=importance,
        )

    def _get_all_values(self, variable: str) -> list[Any]:
        """Get all possible values for a categorical variable."""
        config_space = self._get_config_space()
        var_config = config_space.get(variable)

        # Handle Choices type
        if hasattr(var_config, "values"):
            return list(getattr(var_config, "values", []))

        # Handle raw list/tuple
        if isinstance(var_config, (list, tuple)):
            return list(var_config)

        return []

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

            # Get trials as TrialResult objects (required by core analyzer)
            trials = self._get_trial_results()
            if not trials:
                importance = self._compute_variance_importance(objective)
            else:
                # Core API: constructor takes objective name, method takes TrialResult list
                analyzer = ParameterImportanceAnalyzer(objective)
                if self.importance_method == "variance":
                    results = analyzer.analyze_variance_based(trials)
                elif self.importance_method == "correlation":
                    results = analyzer.analyze_correlation_based(trials)
                else:
                    results = analyzer.analyze_permutation_based(trials)

                # Results is dict[str, ImportanceResult], extract importance_score
                importance = {
                    name: result.importance_score for name, result in results.items()
                }

                # If core analyzer returned no results (insufficient data), use fallback
                if not importance:
                    importance = self._compute_variance_importance(objective)
        except (ImportError, AttributeError, TypeError) as e:
            # Fallback to simple variance-based importance if core analyzer fails
            # This can happen if trials don't have expected format (e.g., mock trials)
            logger.debug(f"Core analyzer failed, using fallback: {e}")
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
            List of ValueRanking sorted by performance (best first).
            For "maximize" objectives, higher scores rank first.
            For "minimize" objectives, lower scores rank first.
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

        # Sort by mean score (direction-aware)
        direction = self._get_direction(objective)
        is_maximize = direction == "maximize"
        rankings.sort(key=lambda r: r.mean_score, reverse=is_maximize)

        # Mark dominated values (direction-aware)
        if rankings:
            best_ci = rankings[0].confidence_interval
            if best_ci:
                # For maximize: best's lower bound is the threshold
                # For minimize: best's upper bound is the threshold
                best_threshold = best_ci[0] if is_maximize else best_ci[1]

                for ranking in rankings[1:]:
                    if ranking.confidence_interval:
                        # For maximize: dominated if upper CI < best's lower CI
                        # For minimize: dominated if lower CI > best's upper CI
                        if is_maximize:
                            if ranking.confidence_interval[1] < best_threshold:
                                ranking.is_dominated = True
                        else:
                            if ranking.confidence_interval[0] > best_threshold:
                                ranking.is_dominated = True

        return rankings

    def get_dominated_values(
        self,
        variable: str,
        objectives: list[str],
    ) -> list[Any]:
        """Find values that are Pareto-dominated across objectives.

        Respects objective directions: for "maximize" objectives, higher is better;
        for "minimize" objectives, lower is better.

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
        value_stats: dict[Any, dict[str, float]] = {}
        for obj in objectives:
            stats = self._get_value_stats(variable, obj)
            for value, scores in stats.items():
                if value not in value_stats:
                    value_stats[value] = {}
                value_stats[value][obj] = _mean(scores) if scores else 0.0

        # Check Pareto dominance (direction-aware)
        dominated = []
        values = list(value_stats.keys())

        for i, v1 in enumerate(values):
            is_dominated = False
            for j, v2 in enumerate(values):
                if i == j:
                    continue
                # Check if v2 dominates v1 (v2 is "better or equal" in all, "better" in at least one)
                dominates = True
                strictly_better = False
                for obj in objectives:
                    s1 = value_stats[v1].get(obj, 0.0)
                    s2 = value_stats[v2].get(obj, 0.0)
                    direction = self._get_direction(obj)

                    # Compare based on direction
                    if direction == "maximize":
                        # v2 dominates if v2 >= v1
                        if s2 < s1:
                            dominates = False
                            break
                        if s2 > s1:
                            strictly_better = True
                    else:  # minimize
                        # v2 dominates if v2 <= v1
                        if s2 > s1:
                            dominates = False
                            break
                        if s2 < s1:
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

        Respects objective direction: for "maximize" objectives, finds values
        in the top 25% of scores; for "minimize" objectives, finds values
        in the bottom 25% of scores.

        Args:
            variable: Name of the numeric variable
            objective: Name of the objective

        Returns:
            Suggested (low, high) range or None if no adjustment needed
        """
        value_stats = self._get_value_stats(variable, objective)
        if not value_stats:
            return None

        # Get values from best quartile of trials (direction-aware)
        all_values = []
        all_scores = []
        for value, scores in value_stats.items():
            for score in scores:
                all_values.append(float(value))
                all_scores.append(score)

        if len(all_values) < 4:
            return None

        # Find values in best 25% of scores based on direction
        direction = self._get_direction(objective)
        if direction == "maximize":
            # Top 25% (high scores are better)
            threshold = _percentile(all_scores, 75)
            top_values = [
                v for v, s in zip(all_values, all_scores, strict=True) if s >= threshold
            ]
        else:
            # Bottom 25% (low scores are better)
            threshold = _percentile(all_scores, 25)
            top_values = [
                v for v, s in zip(all_values, all_scores, strict=True) if s <= threshold
            ]

        if not top_values:
            return None

        # Suggest range covering top values with some margin
        margin = 0.1 * (max(top_values) - min(top_values))
        # Ensure margin is non-zero if all top values are the same
        if margin == 0:
            margin = abs(top_values[0]) * 0.1 if top_values[0] != 0 else 0.1
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
                # Normalize config to list for value manipulation
                normalized_config = self._normalize_config_value(var_config)

                if prune_dominated_values:
                    dominated = self.get_dominated_values(var_name, objectives)
                    if dominated and isinstance(normalized_config, (list, tuple)):
                        remaining = [v for v in normalized_config if v not in dominated]
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
                    self._to_parameter_range(var_name, normalized_config, "categorical")
                    if return_typed
                    else normalized_config
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
                # Normalize numeric config for consistent handling
                normalized_numeric = self._normalize_config_value(var_config)
                refined[var_name] = (
                    self._to_parameter_range(var_name, normalized_numeric, "numeric")
                    if return_typed
                    else normalized_numeric
                )

            else:
                # Normalize other config types for consistent handling
                normalized_other = self._normalize_config_value(var_config)
                refined[var_name] = (
                    self._to_parameter_range(var_name, normalized_other, "categorical")
                    if return_typed
                    else normalized_other
                )

        return refined

    def _to_parameter_range(
        self,
        name: str,
        value: Any,
        var_type: Literal["numeric", "categorical"],
    ) -> ParameterRange:
        """Convert raw value to typed ParameterRange.

        Handles edge cases like collapsed ranges (low == high) by adding
        a small epsilon to create a valid range.

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

                # Handle collapsed range (low == high)
                if low == high:
                    # Add small epsilon to create valid range
                    epsilon = abs(low) * 0.01 if low != 0 else 0.01
                    if isinstance(low, int) and isinstance(high, int):
                        # For integers, expand by 1 in each direction
                        return IntRange(low - 1, high + 1, name=name)
                    return Range(float(low) - epsilon, float(high) + epsilon, name=name)

                # Check if integer or float
                if isinstance(low, int) and isinstance(high, int):
                    return IntRange(low, high, name=name)
                return Range(float(low), float(high), name=name)
            elif isinstance(value, (int, float)):
                # Single value - expand to small range around it
                epsilon = abs(value) * 0.01 if value != 0 else 0.01
                if isinstance(value, int):
                    return IntRange(value - 1, value + 1, name=name)
                return Range(float(value) - epsilon, float(value) + epsilon, name=name)

        # Fallback: treat as categorical
        if isinstance(value, (list, tuple)):
            return Choices(list(value), name=name)
        return Choices([value], name=name)

    # === Private helpers ===

    def _get_config_space(self) -> dict[str, Any]:
        """Extract configuration space from explicit arg or result."""
        # Prefer explicit config space if provided
        if self._explicit_config_space is not None:
            return self._explicit_config_space

        # Try to extract from result
        if hasattr(self.result, "configuration_space"):
            return self.result.configuration_space or {}
        if hasattr(self.result, "config_space"):
            return self.result.config_space or {}
        return {}

    def _get_direction(self, objective: str) -> Literal["maximize", "minimize"]:
        """Get optimization direction for an objective.

        Args:
            objective: Name of the objective

        Returns:
            "maximize" or "minimize"
        """
        # Check explicit directions first
        if objective in self._directions:
            return self._directions[objective]

        # Try to get from result's objectives metadata
        if hasattr(self.result, "objectives") and self.result.objectives:
            for obj in self.result.objectives:
                if hasattr(obj, "name") and obj.name == objective:
                    if hasattr(obj, "direction"):
                        direction = obj.direction
                        # Validate and return if valid
                        if direction in ("maximize", "minimize"):
                            return direction  # type: ignore[return-value]

        # Default to maximize
        return "maximize"

    def _get_trial_results(self) -> list[Any]:
        """Get trials as TrialResult objects for use with core analyzer.

        Returns the raw trial objects if they appear to be TrialResult-like
        (have config, metrics, and status attributes).
        """
        if not hasattr(self.result, "trials"):
            return []

        trials = []
        for trial in self.result.trials:
            # Check if trial looks like a TrialResult
            if (
                hasattr(trial, "config")
                and hasattr(trial, "metrics")
                and hasattr(trial, "status")
            ):
                trials.append(trial)
        return trials

    def _get_trials_as_dicts(self) -> list[dict]:
        """Convert trials to list of dicts for analysis.

        Handles potential name collisions between config and metrics
        by prefixing metrics with 'metric_' if there's a collision.
        """
        trials = []
        if hasattr(self.result, "trials"):
            for trial in self.result.trials:
                if hasattr(trial, "config") and hasattr(trial, "metrics"):
                    trial_dict = dict(trial.config)

                    # Add metrics, handling potential collisions
                    for metric_name, metric_value in trial.metrics.items():
                        if metric_name in trial_dict:
                            # Collision detected - prefix metric with 'metric_'
                            logger.debug(
                                f"Name collision for '{metric_name}' between "
                                f"config and metrics. Using 'metric_{metric_name}'."
                            )
                            trial_dict[f"metric_{metric_name}"] = metric_value
                        else:
                            trial_dict[metric_name] = metric_value

                    trials.append(trial_dict)
        return trials

    def _get_variable_type(self, var_name: str) -> Literal["numeric", "categorical"]:
        """Determine if variable is numeric or categorical.

        Handles both raw config values (lists, tuples) and typed ParameterRange
        objects (Range, IntRange, Choices, etc.).
        """
        config_space = self._get_config_space()
        var_config = config_space.get(var_name)

        # Handle typed ParameterRange objects
        # Check for Range/IntRange/LogRange (numeric types)
        if hasattr(var_config, "low") and hasattr(var_config, "high"):
            return "numeric"

        # Check for Choices (categorical type)
        if hasattr(var_config, "values") and isinstance(
            getattr(var_config, "values", None), (list, tuple)
        ):
            return "categorical"

        # Normalize ParameterRange to raw value for further checks
        var_config = self._normalize_config_value(var_config)

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

    def _normalize_config_value(self, config_value: Any) -> Any:
        """Normalize a config value from ParameterRange to raw type.

        Converts Range -> (low, high) tuple, Choices -> list of values, etc.
        """
        if config_value is None:
            return None

        # Handle Range/IntRange/LogRange - extract (low, high) tuple
        if hasattr(config_value, "low") and hasattr(config_value, "high"):
            return (config_value.low, config_value.high)

        # Handle Choices - extract values list
        if hasattr(config_value, "values"):
            values = getattr(config_value, "values", None)
            if isinstance(values, (list, tuple)):
                return list(values)

        # Already a raw value
        return config_value

    def _get_value_stats(self, variable: str, objective: str) -> dict[Any, list[float]]:
        """Get per-value statistics from trials.

        Handles metric/config name collisions: if the objective was renamed
        to 'metric_{objective}' due to collision, looks for that key as fallback.
        """
        cache_key = f"{variable}:{objective}"
        if cache_key in self._value_stats_cache:
            return self._value_stats_cache[cache_key]

        stats: dict[Any, list[float]] = {}
        trials = self._get_trials_as_dicts()

        for trial in trials:
            if variable not in trial:
                continue

            # Handle potential metric/config name collision
            # If objective was renamed to 'metric_{objective}', use that
            score = None
            if objective in trial:
                score = trial[objective]
            elif f"metric_{objective}" in trial:
                # Collision case: metric was renamed
                score = trial[f"metric_{objective}"]

            if score is not None:
                value = trial[variable]
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
