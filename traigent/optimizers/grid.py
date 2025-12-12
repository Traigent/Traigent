"""Grid search optimization algorithm."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

from traigent.api.types import TrialResult
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimization algorithm.

    Systematically evaluates all combinations of parameters in the configuration
    space. Best for small parameter spaces where exhaustive search is feasible.

    Example:
        >>> config_space = {
        ...     "model": ["gpt-4o-mini", "GPT-4o"],
        ...     "temperature": [0.0, 0.5, 1.0]
        ... }
        >>> optimizer = GridSearchOptimizer(config_space, ["accuracy"])
        >>> config = optimizer.suggest_next_trial([])
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        context=None,
        objective_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize grid search optimizer.

        Args:
            config_space: Dictionary defining parameter search space
            objectives: List of objective names to optimize
            context: Optional TraigentConfig for accessing global configuration
            objective_weights: Optional weights for each objective
            **kwargs: Additional configuration (unused for grid search)
        """
        order_spec = kwargs.get("parameter_order")
        if order_spec is None:
            order_spec = kwargs.get("order")

        super().__init__(config_space, objectives, context, objective_weights, **kwargs)

        self._parameter_order = self._normalize_parameter_order(order_spec)
        self._ordered_param_names = self._resolve_parameter_names()

        # Generate all parameter combinations
        self._grid_points = self._generate_grid()
        self._current_index = 0

        logger.info(f"Generated grid with {len(self._grid_points)} combinations")

    def _generate_grid(self) -> list[dict[str, Any]]:
        """Generate all possible parameter combinations.

        Returns:
            List of dictionaries, each representing a parameter combination

        Raises:
            OptimizationError: If configuration space is invalid for grid search
        """
        if not self.config_space:
            raise OptimizationError("No valid parameter combinations found")

        param_names = self._ordered_param_names.copy()
        param_values = []
        for param_name in param_names:
            param_def = self.config_space[param_name]

            if isinstance(param_def, list):
                # Categorical parameter
                param_values.append(param_def)
            elif isinstance(param_def, tuple) and len(param_def) == 2:
                # Continuous parameter - not directly supported in basic grid search
                raise OptimizationError(
                    f"Grid search does not support continuous parameter '{param_name}'. "
                    f"Use a list of discrete values instead of range {param_def}."
                )
            else:
                # Single value (fixed parameter)
                param_values.append([param_def])

        # Generate cartesian product
        combinations = list(itertools.product(*param_values))

        # Convert to list of dictionaries
        grid_points = []
        for combination in combinations:
            config = dict(zip(param_names, combination, strict=False))
            grid_points.append(config)

        if not grid_points:
            raise OptimizationError("No valid parameter combinations found")

        return grid_points

    def _normalize_parameter_order(
        self, order_spec: Mapping[str, Any] | None
    ) -> dict[str, float]:
        """Validate and normalize optional parameter order specification."""
        if order_spec is None:
            return {}

        if not isinstance(order_spec, Mapping):
            raise OptimizationError(
                "parameter_order must be a mapping of parameter names to numeric priorities"
            )

        normalized: dict[str, float] = {}
        for name, priority in order_spec.items():
            if not isinstance(name, str):
                raise OptimizationError("parameter_order keys must be strings")
            if not isinstance(priority, (int, float)):
                raise OptimizationError(
                    f"parameter_order for '{name}' must be numeric, got {type(priority).__name__}"
                )
            normalized[name] = float(priority)

        if not normalized:
            return {}

        known_params = set(self.config_space.keys())
        unknown = sorted(set(normalized) - known_params)
        if unknown:
            logger.warning(
                "Ignoring parameter_order entries for unknown parameters: %s",
                ", ".join(unknown),
            )
            for name in unknown:
                normalized.pop(name, None)

        return normalized

    def _resolve_parameter_names(self) -> list[str]:
        """Determine parameter iteration order respecting optional overrides.

        Note: In itertools.product, the rightmost parameter varies fastest.
        Putting 'model' last means it will cycle through all models early,
        allowing quick comparison of different models with the same other parameters.
        """
        sorted_names = sorted(self.config_space.keys())
        if not sorted_names:
            return []

        if self._parameter_order:
            position_lookup = {name: idx for idx, name in enumerate(sorted_names)}

            def sort_key(param_name: str) -> tuple[float, int]:
                priority = self._parameter_order.get(param_name, float("inf"))
                fallback = position_lookup[param_name]
                return (priority, fallback)

            return sorted(sorted_names, key=sort_key)

        if "model" in sorted_names:
            remaining = [name for name in sorted_names if name != "model"]
            return [*remaining, "model"]

        return sorted_names

    def _generate_configurations(self) -> list[dict[str, Any]]:
        """Generate all configurations for testing compatibility.

        Returns:
            List of all possible parameter combinations
        """
        return self._grid_points.copy()

    def _calculate_composite_score(self, metrics: dict[str, float]) -> float:
        """Calculate composite score from multiple metrics using weighted scalarization."""
        if not metrics:
            return 0.0

        # Use scalarize_objectives for weighted scoring
        from traigent.utils.multi_objective import scalarize_objectives

        return scalarize_objectives(metrics, self.objective_weights)

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate.

        Args:
            history: List of previous trials (used to track progress)

        Returns:
            Next configuration to evaluate

        Raises:
            OptimizationError: If all combinations have been evaluated
        """
        if self._current_index >= len(self._grid_points):
            raise OptimizationError("All grid combinations have been evaluated")

        config = self._grid_points[self._current_index]
        self._current_index += 1
        self._trial_count += 1

        logger.debug(
            f"Suggesting trial {self._trial_count}/{len(self._grid_points)}: {config}"
        )

        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop.

        Grid search stops when all combinations have been evaluated.

        Args:
            history: List of previous trials

        Returns:
            True if all combinations evaluated, False otherwise
        """
        return self._current_index >= len(self._grid_points)

    def reset(self) -> None:
        """Reset optimizer state for new optimization run."""
        super().reset()
        self._current_index = 0

    @property
    def total_combinations(self) -> int:
        """Get total number of parameter combinations."""
        return len(self._grid_points)

    @property
    def progress(self) -> float:
        """Get optimization progress as a fraction (0.0 to 1.0)."""
        if not self._grid_points:
            return 0.0
        return min(self._current_index / len(self._grid_points), 1.0)

    def get_remaining_combinations(self) -> list[dict[str, Any]]:
        """Get list of remaining parameter combinations.

        Returns:
            List of configurations not yet evaluated
        """
        return self._grid_points[self._current_index :]

    def get_algorithm_info(self) -> dict[str, Any]:
        """Get information about this optimization algorithm."""
        info = super().get_algorithm_info()
        info.update(
            {
                "total_combinations": self.total_combinations,
                "progress": self.progress,
                "supports_continuous": False,
                "supports_categorical": True,
                "deterministic": True,
            }
        )
        return info
