"""Mock optimizer implementations for testing.

This module provides mock implementations of optimization algorithms
for testing optimization workflows without running actual optimization.
"""

import random
from typing import Any

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.base import BaseOptimizer


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing orchestration workflows."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self.suggested_configs = []
        self.trial_results = []
        self._suggest_count = 0
        self._max_suggestions = 5
        self._should_stop = False

    def set_max_suggestions(self, max_suggestions: int):
        """Set maximum suggestions for testing."""
        self._max_suggestions = max_suggestions
        # If max is 0, stop immediately
        if max_suggestions == 0:
            self._should_stop = True

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate."""
        # Create realistic configs from the config space
        config = {}
        for param, values in self.config_space.items():
            if isinstance(values, list):
                config[param] = values[self._suggest_count % len(values)]
            else:
                config[param] = values

        # Add some variation
        config["_trial_id"] = self._suggest_count

        self.suggested_configs.append(config.copy())
        self._suggest_count += 1

        # Set stop flag after suggesting the last config
        if self._suggest_count >= self._max_suggestions:
            self._should_stop = True

        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        return self._should_stop

    def suggest(self) -> dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.suggest_next_trial([])

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        """Record trial result."""
        self.trial_results.append((config, result))

    def is_finished(self) -> bool:
        """Check if optimization should stop."""
        return self._should_stop

    def force_stop(self):
        """Force optimizer to stop."""
        self._should_stop = True


class MockAsyncOptimizer(MockOptimizer):
    """Mock async optimizer for testing async workflows."""

    async def suggest_next_trial_async(
        self, history: list[TrialResult]
    ) -> dict[str, Any]:
        """Async version of suggest_next_trial."""
        return self.suggest_next_trial(history)

    async def tell_async(self, config: dict[str, Any], result: TrialResult) -> None:
        """Async version of tell."""
        self.tell(config, result)


class MockBayesianOptimizer(MockOptimizer):
    """Mock Bayesian optimizer with realistic behavior."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self.acquisition_function = kwargs.get(
            "acquisition_function", "expected_improvement"
        )
        self.initial_random_samples = kwargs.get("initial_random_samples", 3)
        self._exploration_phase = True

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next trial with Bayesian-like behavior."""
        if self._suggest_count < self.initial_random_samples:
            # Random exploration phase
            config = self._random_config()
        else:
            # Exploitation phase - pick from promising regions
            config = self._exploitative_config(history)

        config["_acquisition_value"] = random.uniform(0.1, 0.9)
        config["_trial_id"] = self._suggest_count

        self.suggested_configs.append(config.copy())
        self._suggest_count += 1

        if self._suggest_count >= self._max_suggestions:
            self._should_stop = True

        return config

    def _random_config(self) -> dict[str, Any]:
        """Generate random configuration."""
        config = {}
        for param, values in self.config_space.items():
            if isinstance(values, list):
                config[param] = random.choice(values)
            else:
                config[param] = values
        return config

    def _exploitative_config(self, history: list[TrialResult]) -> dict[str, Any]:
        """Generate exploitative configuration based on history."""
        if not history:
            return self._random_config()

        # Find best performing trial
        best_trial = max(history, key=lambda t: t.metrics.get("accuracy", 0))
        config = best_trial.config.copy()

        # Add small perturbation
        for param, values in self.config_space.items():
            if isinstance(values, list) and param in config:
                current_idx = values.index(config[param])
                # Small chance to explore nearby values
                if random.random() < 0.3 and len(values) > 1:
                    new_idx = (current_idx + random.choice([-1, 1])) % len(values)
                    config[param] = values[new_idx]

        return config


class MockGridOptimizer(MockOptimizer):
    """Mock grid search optimizer."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self._grid_configs = self._generate_grid()
        self._max_suggestions = len(self._grid_configs)

    def _generate_grid(self) -> list[dict[str, Any]]:
        """Generate all combinations in the grid."""
        import itertools

        params = []
        values = []

        for param, param_values in self.config_space.items():
            if isinstance(param_values, list):
                params.append(param)
                values.append(param_values)

        # Generate cartesian product
        grid_configs = []
        for combination in itertools.product(*values):
            config = dict(zip(params, combination))
            grid_configs.append(config)

        return grid_configs

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next grid point."""
        if self._suggest_count >= len(self._grid_configs):
            self._should_stop = True
            return {}

        config = self._grid_configs[self._suggest_count].copy()
        config["_grid_index"] = self._suggest_count
        config["_trial_id"] = self._suggest_count

        self.suggested_configs.append(config.copy())
        self._suggest_count += 1

        if self._suggest_count >= len(self._grid_configs):
            self._should_stop = True

        return config


class MockRandomOptimizer(MockOptimizer):
    """Mock random search optimizer."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self.random_seed = kwargs.get("random_seed", 42)
        random.seed(self.random_seed)

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest random configuration."""
        config = {}
        for param, values in self.config_space.items():
            if isinstance(values, list):
                config[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Assume it's a range (min, max)
                config[param] = random.uniform(values[0], values[1])
            else:
                config[param] = values

        config["_random_seed"] = self.random_seed + self._suggest_count
        config["_trial_id"] = self._suggest_count

        self.suggested_configs.append(config.copy())
        self._suggest_count += 1

        if self._suggest_count >= self._max_suggestions:
            self._should_stop = True

        return config


def create_mock_optimizer(
    optimizer_type: str, config_space: dict[str, Any], objectives: list[str], **kwargs
) -> MockOptimizer:
    """Factory function to create mock optimizers."""
    optimizers = {
        "base": MockOptimizer,
        "async": MockAsyncOptimizer,
        "bayesian": MockBayesianOptimizer,
        "grid": MockGridOptimizer,
        "random": MockRandomOptimizer,
    }

    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizers[optimizer_type](config_space, objectives, **kwargs)


def create_realistic_optimizer_results(
    optimizer: MockOptimizer, num_trials: int = 5
) -> list[TrialResult]:
    """Create realistic trial results for a mock optimizer."""
    results = []

    for i in range(num_trials):
        config = optimizer.suggest_next_trial(results)

        # Generate realistic metrics based on config
        base_accuracy = 0.7
        if config.get("model") == "gpt-4":
            base_accuracy += 0.1
        if config.get("temperature", 0.5) < 0.3:
            base_accuracy += 0.05

        # Add some noise
        accuracy = base_accuracy + random.uniform(-0.1, 0.1)
        accuracy = max(0.0, min(1.0, accuracy))

        cost = 0.01 + (0.005 if config.get("model") == "gpt-4" else 0)
        latency = 1.0 + random.uniform(0, 0.5)

        result = TrialResult(
            trial_id=f"trial_{i+1}",
            config=config,
            metrics={
                "accuracy": round(accuracy, 3),
                "cost": round(cost, 4),
                "latency": round(latency, 2),
            },
            status=TrialStatus.COMPLETED,
            duration=latency,
            error_message=None,
        )

        results.append(result)
        optimizer.tell(config, result)

    return results
