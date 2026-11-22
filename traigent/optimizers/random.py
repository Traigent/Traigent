"""Random search optimization algorithm."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import random
from itertools import product
from typing import Any

from traigent.api.types import TrialResult
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.discrete_domains import (
    discrete_values_for_config_param,
    sample_stepped_float,
    sample_stepped_int,
    uniform_within,
)
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger
from traigent.utils.validation import Validators

logger = get_logger(__name__)
MAX_COMPLEMENT_SCAN_CONFIGS = 100_000


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization algorithm.

    Randomly samples parameters from the configuration space. Good baseline
    algorithm that works well for both categorical and continuous parameters.

    For discrete-only config spaces, automatically detects when all unique
    configurations have been tried and stops early.

    Example:
        >>> config_space = {
        ...     "model": ["gpt-4o-mini", "gpt-4o"],
        ...     "temperature": (0.0, 1.0)
        ... }
        >>> optimizer = RandomSearchOptimizer(config_space, ["accuracy"])
        >>> config = optimizer.suggest_next_trial([])

        With reproducible seed:

        >>> optimizer = RandomSearchOptimizer(
        ...     config_space,
        ...     ["accuracy"],
        ...     random_seed=42,
        ...     max_trials=50,
        ... )
    """

    _honours_seed = True

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 100,
        random_seed: int | None = None,
        context=None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize random search optimizer.

        Args:
            config_space: Dictionary defining parameter search space.
                Supports lists (categorical) and tuples (continuous ranges).
            objectives: List of objective names to optimize.
            max_trials: Maximum number of trials to run. Default 100.
            random_seed: Random seed for reproducibility. If provided, the same
                sequence of configurations will be generated across runs.
            context: Optional TraigentConfig for accessing global configuration.
            seed: Alias for ``random_seed`` (the name ``.optimize(seed=...)``
                users reach for). Supplying both with different values raises
                ``ValueError``.
            **kwargs: Additional arguments passed to BaseOptimizer
                (e.g., ``objective_weights``).
        """
        if seed is not None and random_seed is not None and seed != random_seed:
            raise ValueError(
                "seed and random_seed are aliases but were given different "
                f"values ({seed!r} vs {random_seed!r}); pass only one."
            )
        if random_seed is None:
            random_seed = seed
        # Keep ``seed`` in algorithm_config so run metadata records it.
        if seed is not None:
            kwargs["seed"] = seed
        super().__init__(config_space, objectives, context, **kwargs)

        self.max_trials = max_trials
        self.random_seed = random_seed

        # Create separate random instance for reproducibility
        self._random = random.Random()
        if random_seed is not None:
            self._random.seed(random_seed)

        logger.info(f"Initialized random search with max_trials={max_trials}")

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate.

        Args:
            history: List of previous trials (used for trial count)

        Returns:
            Randomly sampled configuration

        Raises:
            OptimizationError: If maximum trials reached or space exhausted
        """
        if self._trial_count >= self.max_trials:
            raise OptimizationError(f"Maximum trials ({self.max_trials}) reached")

        # Check if discrete space is exhausted
        if self.is_config_space_exhausted():
            raise OptimizationError(
                f"Config space exhausted: all {self.config_space_cardinality} "
                "unique configurations have been tried"
            )

        # For discrete spaces, avoid duplicates by retrying, then fall back to a
        # lazy complement scan so collision streaks do not masquerade as exhaustion.
        cardinality = self.config_space_cardinality
        max_attempts = self._max_unique_sampling_attempts(cardinality)

        config = RandomSearchOptimizer._sample_unique_config(
            self, cardinality, max_attempts
        )
        self._trial_count += 1

        logger.debug(f"Suggesting random trial {self._trial_count}: {config}")

        return config

    def _sample_unique_config(
        self, cardinality: int | None, max_attempts: int
    ) -> dict[str, Any]:
        for _attempt in range(max_attempts):
            config = {}
            for param_name, param_def in self.config_space.items():
                config[param_name] = self._sample_parameter(param_name, param_def)

            if RandomSearchOptimizer._accept_sample(self, config, cardinality):
                break
        else:
            fallback_config = self._find_untried_discrete_config()
            if fallback_config is None:
                # Exhausted retry attempts (shouldn't happen with proper cardinality)
                raise OptimizationError(
                    f"Failed to find unique config after {max_attempts} attempts"
                )
            config = fallback_config
        return config

    def _accept_sample(self, config: dict[str, Any], cardinality: int | None) -> bool:
        if cardinality is not None:
            is_new = self.register_tried_config(config)
            if is_new:
                return True
            if self.is_config_space_exhausted():
                raise OptimizationError(
                    f"Config space exhausted: all {cardinality} "
                    "unique configurations have been tried"
                )
            return False
        # A resumed continuous run may replay previously evaluated configs.
        self.register_tried_config(config)
        return self._hash_config(config) not in getattr(
            self, "_resume_marked_hashes", ()
        )

    def _max_unique_sampling_attempts(self, cardinality: int | None) -> int:
        """Scale retry effort with finite search-space size."""

        if cardinality is None:
            # Continuous space: a resumed run (#1697) registers prior configs,
            # and a same-seed replay yields them first; allow skipping all.
            return 100 + len(self._tried_config_hashes)
        return min(max(cardinality * 2, 100), 10_000)

    def _find_untried_discrete_config(self) -> dict[str, Any] | None:
        """Lazily scan a finite discrete space for an untried configuration."""

        cardinality = self.config_space_cardinality
        if cardinality is None or cardinality > MAX_COMPLEMENT_SCAN_CONFIGS:
            return None

        value_domains: list[tuple[Any, ...]] = []
        param_names = list(self.config_space)
        for param_name in param_names:
            values = discrete_values_for_config_param(self.config_space[param_name])
            if values is None:
                return None
            if not values:
                raise OptimizationError("Config space exhausted: no values to sample")
            value_domains.append(values)

        for combination in product(*value_domains):
            config = dict(zip(param_names, combination, strict=True))
            if self.register_tried_config(config):
                return config

        if cardinality is not None:
            raise OptimizationError(
                f"Config space exhausted: all {cardinality} "
                "unique configurations have been tried"
            )
        return None

    def _sample_range_int(self, param_name: str, param_def: dict) -> int:
        """Sample an integer value from a range-dict definition."""
        try:
            low_f, high_f = float(param_def["low"]), float(param_def["high"])
        except (TypeError, ValueError) as exc:
            raise OptimizationError(
                f"Invalid integer range for parameter '{param_name}': "
                "low/high must be numeric"
            ) from exc
        if not low_f.is_integer() or not high_f.is_integer():
            raise OptimizationError(
                f"Integer parameter '{param_name}' requires integral low/high bounds"
            )
        # Keep exact integers as-is: a float round trip loses precision above 2**53.
        raw_low, raw_high = param_def["low"], param_def["high"]
        exact_ints = (
            isinstance(raw_low, int)
            and isinstance(raw_high, int)
            and not isinstance(raw_low, bool)
            and not isinstance(raw_high, bool)
        )

        step_raw = param_def.get("step", 1)
        if step_raw is None:
            step_raw = 1
        try:
            step_f = float(step_raw)
        except (TypeError, ValueError) as exc:
            raise OptimizationError(
                f"Invalid step for integer parameter '{param_name}': {step_raw}"
            ) from exc
        if step_f <= 0 or not step_f.is_integer():
            raise OptimizationError(
                f"Invalid step for integer parameter '{param_name}': {step_raw}"
            )

        low_i: int
        high_i: int
        if exact_ints:
            low_i, high_i = int(raw_low), int(raw_high)
        else:
            low_i, high_i = int(low_f), int(high_f)
        int_step = max(int(step_f), 1)
        if low_i > high_i:
            raise OptimizationError(
                f"Invalid integer range for parameter '{param_name}': "
                f"low ({low_i}) must be <= high ({high_i})"
            )

        # Pick a grid index arithmetically instead of materializing the grid, so
        # a wide step-less IntRange costs O(1) per sample. The grid is low,
        # low+step, ... plus the inclusive high endpoint when it is off-grid.
        return sample_stepped_int(self._random, low_i, high_i, int_step)

    def _sample_range_float(self, param_name: str, param_def: dict) -> float:
        """Sample a float value from a range-dict definition."""
        try:
            low_f, high_f = float(param_def["low"]), float(param_def["high"])
        except (TypeError, ValueError) as exc:
            raise OptimizationError(
                f"Invalid float range for parameter '{param_name}': "
                "low/high must be numeric"
            ) from exc
        if low_f > high_f:
            raise OptimizationError(
                f"Invalid float range for parameter '{param_name}': "
                f"low ({low_f}) must be <= high ({high_f})"
            )
        step_raw = param_def.get("step")
        if step_raw is not None:
            try:
                step = float(step_raw)
            except (TypeError, ValueError) as exc:
                raise OptimizationError(
                    f"Invalid step for float parameter '{param_name}': {step_raw}"
                ) from exc
            if step <= 0:
                raise OptimizationError(
                    f"Invalid step for float parameter '{param_name}': {step}"
                )
            try:
                return sample_stepped_float(self._random, low_f, high_f, step)
            except ValueError as exc:
                raise OptimizationError(
                    f"Invalid float range for parameter '{param_name}': {exc}"
                ) from exc
        return uniform_within(self._random, low_f, high_f)

    def _sample_parameter(self, param_name: str, param_def: Any) -> Any:
        """Sample a single parameter value.

        Args:
            param_name: Name of the parameter
            param_def: Parameter definition (list, tuple, dict, or single value)

        Returns:
            Sampled parameter value

        Raises:
            OptimizationError: If parameter definition is invalid
        """
        if isinstance(param_def, list):
            return self._random.choice(param_def)

        if isinstance(param_def, tuple) and len(param_def) == 2:
            low, high = param_def
            if isinstance(low, int) and isinstance(high, int):
                return self._random.randint(low, high)
            return self._random.uniform(low, high)

        if isinstance(param_def, dict) and "low" in param_def and "high" in param_def:
            param_type = str(param_def.get("type", "")).lower()
            if param_type in {"int", "integer"}:
                return self._sample_range_int(param_name, param_def)
            return self._sample_range_float(param_name, param_def)

        # Fixed parameter
        return param_def

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop.

        Random search stops when maximum trials is reached or when a discrete
        configuration space has been exhausted.

        Args:
            history: List of previous trials

        Returns:
            True if max trials reached or space exhausted, False otherwise
        """
        if self._trial_count >= self.max_trials:
            return True

        # Early termination for exhausted discrete config spaces
        if self.is_config_space_exhausted():
            cardinality = self.config_space_cardinality
            logger.info(
                "Config space exhausted after %d unique configurations. "
                "Stopping early (requested %d trials, but only %d possible).",
                self.unique_configs_tried,
                self.max_trials,
                cardinality,
            )
            return True

        return False

    def reset(self) -> None:
        """Reset optimizer state for new optimization run."""
        super().reset()

        # Reset random seed if specified
        if self.random_seed is not None:
            self._random.seed(self.random_seed)

    @property
    def progress(self) -> float:
        """Get optimization progress as a fraction (0.0 to 1.0)."""
        if self.max_trials <= 0:
            # A zero-trial run is a no-op that is immediately complete.
            return 1.0
        return float(min(self._trial_count / self.max_trials, 1.0))

    def set_max_trials(self, max_trials: int) -> None:
        """Update maximum number of trials.

        Args:
            max_trials: New maximum number of trials

        Raises:
            ValueError: If max_trials is not positive
        """
        result = Validators.validate_positive_int(max_trials, "max_trials")
        if not result.is_valid:
            raise ValueError(f"Invalid max_trials: {result.errors}")
        self.max_trials = max_trials
        logger.info(f"Updated max_trials to {self.max_trials}")

    def get_algorithm_info(self) -> dict[str, Any]:
        """Get information about this optimization algorithm."""
        info: dict[str, Any] = super().get_algorithm_info()
        info.update(
            {
                "max_trials": self.max_trials,
                "progress": self.progress,
                "supports_continuous": True,
                "supports_categorical": True,
                "deterministic": self.random_seed is not None,
                "random_seed": self.random_seed,
            }
        )
        return info
