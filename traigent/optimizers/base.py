"""Base classes for optimization algorithms."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Performance FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from traigent.api.types import TrialResult
from traigent.config.types import TraigentConfig
from traigent.utils.logging import get_logger
from traigent.utils.validation import validate_objectives

logger = get_logger(__name__)


class BaseOptimizer(ABC):
    """Base class for all optimization algorithms.

    This class defines the interface that all optimization algorithms must implement.
    It follows the Strategy pattern to allow different algorithms to be used
    interchangeably.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        context: TraigentConfig | None = None,
        objective_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize optimizer with configuration space and objectives.

        Args:
            config_space: Dictionary defining parameter search space
            objectives: List of objective names to optimize
            context: Optional TraigentConfig for accessing global configuration
            objective_weights: Optional weights for each objective
            **kwargs: Algorithm-specific configuration
        """
        self.config_space = config_space
        if isinstance(objectives, (list, tuple, set)):
            normalized_objectives = list(objectives)
        elif objectives is None:
            normalized_objectives = []
        else:
            normalized_objectives = [objectives]
        if normalized_objectives:
            validate_objectives(normalized_objectives)
            self.objectives = list(normalized_objectives)
        else:
            get_logger(__name__).debug(
                "Initializing %s without objectives; weighted scoring disabled",
                self.__class__.__name__,
            )
            self.objectives = []
        self.context = context
        self.algorithm_config = kwargs

        # Handle objective weights - support both parameter and kwargs
        weights = objective_weights or kwargs.get("objective_weights", {})

        # Initialize objective weights with proper handling
        if not self.objectives:
            self.objective_weights = {}
        elif not weights:
            # Default to equal weights for all objectives
            self.objective_weights = dict.fromkeys(self.objectives, 1.0)
        else:
            # Start with provided weights
            self.objective_weights = {}
            for obj in self.objectives:
                if obj in weights:
                    self.objective_weights[obj] = weights[obj]
                else:
                    # Use default weight for missing objectives
                    self.objective_weights[obj] = 1.0

            # Only keep weights for objectives that are specified
            self.objective_weights = {
                obj: weight
                for obj, weight in self.objective_weights.items()
                if obj in self.objectives
            }

        # Initialize internal state
        self._trial_count = 0
        self._best_score: float | None = None
        self._best_config: dict[str, Any] | None = None

    @abstractmethod
    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate.

        Args:
            history: List of previous trials with results

        Returns:
            Dictionary containing suggested parameter values

        Raises:
            OptimizationError: If unable to suggest next trial
        """
        pass

    async def suggest_next_trial_async(
        self, history: list[TrialResult], remote_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Async version of suggest_next_trial for remote service support.

        Args:
            history: List of previous trials with results
            remote_context: Optional context from remote optimization service

        Returns:
            Dictionary containing suggested parameter values

        Raises:
            OptimizationError: If unable to suggest next trial
        """
        # Default implementation calls synchronous version
        # Remote optimizers can override this for true async behavior
        return self.suggest_next_trial(history)

    @abstractmethod
    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop.

        Args:
            history: List of previous trials with results

        Returns:
            True if optimization should stop, False otherwise
        """
        pass

    async def should_stop_async(
        self, history: list[TrialResult], remote_context: dict[str, Any] | None = None
    ) -> bool:
        """Async version of should_stop for remote service support.

        Args:
            history: List of previous trials with results
            remote_context: Optional context from remote optimization service

        Returns:
            True if optimization should stop, False otherwise
        """
        # Default implementation calls synchronous version
        # Remote optimizers can override this for true async behavior
        return self.should_stop(history)

    def generate_candidates(self, max_candidates: int) -> list[dict[str, Any]]:
        """Generate multiple candidate configurations.

        Args:
            max_candidates: Maximum number of candidates to generate

        Returns:
            List of candidate configurations
        """
        candidates = []
        history: list[TrialResult] = []  # Empty history for generation

        for _ in range(max_candidates):
            if self.should_stop(history):
                break
            try:
                candidate = self.suggest_next_trial(history)
                candidates.append(candidate)
            except Exception as e:
                # If we can't generate more candidates, break
                logger.debug(
                    f"Candidate generation stopped after {len(candidates)} candidates: {e}"
                )
                break

        return candidates

    async def generate_candidates_async(
        self, max_candidates: int, remote_context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Async version of generate_candidates for remote service support.

        Args:
            max_candidates: Maximum number of candidates to generate
            remote_context: Optional context from remote optimization service

        Returns:
            List of candidate configurations
        """
        candidates = []
        history: list[TrialResult] = []  # Empty history for generation

        for _ in range(max_candidates):
            if await self.should_stop_async(history, remote_context):
                break
            try:
                candidate = await self.suggest_next_trial_async(history, remote_context)
                candidates.append(candidate)
            except Exception as e:
                # If we can't generate more candidates, break
                logger.debug(
                    f"Async candidate generation stopped after {len(candidates)} candidates: {e}"
                )
                break

        return candidates

    def update_best(self, trial: TrialResult) -> None:
        """Update best trial information.

        Args:
            trial: Latest trial result
        """
        if not trial.is_successful:
            return

        # Get primary objective score
        primary_objective = self.objectives[0]
        score = trial.get_metric(primary_objective)

        if score is not None and (self._best_score is None or score > self._best_score):
            self._best_score = score
            self._best_config = trial.config.copy()

    @property
    def best_score(self) -> float | None:
        """Get best score achieved so far."""
        return self._best_score

    @property
    def best_config(self) -> dict[str, Any] | None:
        """Get best configuration found so far."""
        return self._best_config

    @property
    def trial_count(self) -> int:
        """Get number of trials suggested so far."""
        return self._trial_count

    def reset(self) -> None:
        """Reset optimizer state for new optimization run."""
        self._trial_count = 0
        self._best_score = None
        self._best_config = None

    def get_algorithm_info(self) -> dict[str, Any]:
        """Get information about this optimization algorithm.

        Returns:
            Dictionary with algorithm name, description, and parameters
        """
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "No description available",
            "config": self.algorithm_config,
            "objectives": self.objectives,
            "config_space": self.config_space,
            "supports_async": True,
            "supports_remote": hasattr(self, "remote_service"),
            "context_aware": self.context is not None,
        }
