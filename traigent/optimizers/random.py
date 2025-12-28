"""Random search optimization algorithm."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import random
from typing import Any

from traigent.api.types import TrialResult
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger
from traigent.utils.validation import Validators

logger = get_logger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization algorithm.

    Randomly samples parameters from the configuration space. Good baseline
    algorithm that works well for both categorical and continuous parameters.

    Example:
        >>> config_space = {
        ...     "model": ["gpt-4o-mini", "GPT-4o"],
        ...     "temperature": (0.0, 1.0)
        ... }
        >>> optimizer = RandomSearchOptimizer(config_space, ["accuracy"])
        >>> config = optimizer.suggest_next_trial([])
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 100,
        random_seed: int | None = None,
        context=None,
        **kwargs: Any,
    ) -> None:
        """Initialize random search optimizer.

        Args:
            config_space: Dictionary defining parameter search space
            objectives: List of objective names to optimize
            max_trials: Maximum number of trials to run
            random_seed: Random seed for reproducibility
            context: Optional TraigentConfig for accessing global configuration
            **kwargs: Additional configuration
        """
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

        # For discrete spaces, avoid duplicates by retrying
        max_attempts = 100  # Prevent infinite loops
        cardinality = self.config_space_cardinality

        for _attempt in range(max_attempts):
            config = {}
            for param_name, param_def in self.config_space.items():
                config[param_name] = self._sample_parameter(param_name, param_def)

            # For discrete spaces, check if this config is new
            if cardinality is not None:
                is_new = self.register_tried_config(config)
                if is_new:
                    break
                # Duplicate - try again if space not exhausted
                if self.is_config_space_exhausted():
                    raise OptimizationError(
                        f"Config space exhausted: all {cardinality} "
                        "unique configurations have been tried"
                    )
            else:
                # Continuous space - always accept (duplicates very unlikely)
                self.register_tried_config(config)
                break
        else:
            # Exhausted retry attempts (shouldn't happen with proper cardinality)
            raise OptimizationError(
                f"Failed to find unique config after {max_attempts} attempts"
            )

        self._trial_count += 1

        logger.debug(f"Suggesting random trial {self._trial_count}: {config}")

        return config

    def _sample_parameter(self, param_name: str, param_def: Any) -> Any:
        """Sample a single parameter value.

        Args:
            param_name: Name of the parameter
            param_def: Parameter definition (list, tuple, or single value)

        Returns:
            Sampled parameter value

        Raises:
            OptimizationError: If parameter definition is invalid
        """
        if isinstance(param_def, list):
            # Categorical parameter - random choice
            return self._random.choice(param_def)

        elif isinstance(param_def, tuple) and len(param_def) == 2:
            # Continuous parameter - uniform random sample
            low, high = param_def
            if isinstance(low, int) and isinstance(high, int):
                # Integer range
                return self._random.randint(low, high)
            else:
                # Float range
                return self._random.uniform(low, high)

        else:
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
        return min(self._trial_count / self.max_trials, 1.0)

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
        info = super().get_algorithm_info()
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
