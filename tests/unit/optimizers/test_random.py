"""Comprehensive tests for random search optimizer."""

from datetime import datetime
from unittest.mock import patch

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.exceptions import OptimizationError


class TestRandomSearchOptimizer:
    """Test suite for RandomSearchOptimizer."""

    def test_initialization_default(self):
        """Test default initialization."""
        config_space = {"x": [0, 1, 2], "y": (0.0, 1.0)}
        objectives = ["accuracy"]

        optimizer = RandomSearchOptimizer(config_space, objectives)

        assert optimizer.config_space == config_space
        assert optimizer.objectives == objectives
        assert optimizer.max_trials == 100  # default
        assert optimizer.random_seed is None
        assert optimizer._trial_count == 0

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        config_space = {"x": [0, 1, 2]}
        objectives = ["accuracy", "latency"]

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            max_trials=50,
            random_seed=42,
            custom_param="value",
        )

        assert optimizer.max_trials == 50
        assert optimizer.random_seed == 42
        assert optimizer.algorithm_config["custom_param"] == "value"

    def test_suggest_next_trial_categorical(self):
        """Test suggesting configurations with categorical parameters."""
        config_space = {"model": ["a", "b", "c"], "size": ["small", "medium", "large"]}
        optimizer = RandomSearchOptimizer(config_space, ["accuracy"], random_seed=42)

        # Config space has 9 unique combinations (3x3), so we can only generate 9 configs
        # Generate all 9 unique configs
        configs = []
        for _ in range(9):
            config = optimizer.suggest_next_trial([])
            configs.append(config)

            # Check all parameters are present and valid
            assert "model" in config
            assert "size" in config
            assert config["model"] in ["a", "b", "c"]
            assert config["size"] in ["small", "medium", "large"]

        # All 9 should be unique
        config_strs = [str(sorted(c.items())) for c in configs]
        assert len(set(config_strs)) == 9

        # With fixed seed, should be deterministic
        optimizer2 = RandomSearchOptimizer(config_space, ["accuracy"], random_seed=42)
        configs2 = [optimizer2.suggest_next_trial([]) for _ in range(9)]

        assert configs == configs2

    def test_suggest_next_trial_continuous(self):
        """Test suggesting configurations with continuous parameters."""
        config_space = {"learning_rate": (0.001, 0.1), "dropout": (0.0, 0.5)}
        optimizer = RandomSearchOptimizer(config_space, ["accuracy"])

        for _ in range(20):
            config = optimizer.suggest_next_trial([])

            # Check parameters are in valid ranges
            assert 0.001 <= config["learning_rate"] <= 0.1
            assert 0.0 <= config["dropout"] <= 0.5

    def test_suggest_next_trial_integer_range(self):
        """Test suggesting configurations with integer range parameters."""
        config_space = {"batch_size": (16, 128), "num_layers": (1, 10)}
        optimizer = RandomSearchOptimizer(config_space, ["accuracy"])

        for _ in range(20):
            config = optimizer.suggest_next_trial([])

            # Check parameters are integers in valid ranges
            assert isinstance(config["batch_size"], int)
            assert isinstance(config["num_layers"], int)
            assert 16 <= config["batch_size"] <= 128
            assert 1 <= config["num_layers"] <= 10

    def test_suggest_next_trial_mixed_types(self):
        """Test suggesting configurations with mixed parameter types."""
        config_space = {
            "model": ["a", "b", "c"],
            "learning_rate": (0.001, 0.1),
            "batch_size": (16, 128),
            "use_dropout": [True, False],
            "fixed_param": "constant",
        }
        optimizer = RandomSearchOptimizer(config_space, ["accuracy"])

        for _ in range(10):
            config = optimizer.suggest_next_trial([])

            assert config["model"] in ["a", "b", "c"]
            assert 0.001 <= config["learning_rate"] <= 0.1
            assert 16 <= config["batch_size"] <= 128
            assert config["use_dropout"] in [True, False]
            assert config["fixed_param"] == "constant"

    def test_suggest_next_trial_max_trials_reached(self):
        """Test that suggestion fails when max trials is reached."""
        # Use continuous space so exhaustion doesn't trigger before max_trials
        optimizer = RandomSearchOptimizer({"x": (0.0, 1.0)}, ["accuracy"], max_trials=3)

        # Suggest 3 trials
        for _ in range(3):
            optimizer.suggest_next_trial([])

        # 4th trial should raise error
        with pytest.raises(OptimizationError, match="Maximum trials .* reached"):
            optimizer.suggest_next_trial([])

    def test_suggest_next_trial_increments_count(self):
        """Test that trial count is incremented properly."""
        optimizer = RandomSearchOptimizer({"x": [0, 1]}, ["accuracy"])

        assert optimizer._trial_count == 0

        optimizer.suggest_next_trial([])
        assert optimizer._trial_count == 1

        optimizer.suggest_next_trial([])
        assert optimizer._trial_count == 2

    def test_sample_parameter_list(self):
        """Test sampling from list parameter."""
        optimizer = RandomSearchOptimizer(
            {"x": [1, 2, 3]}, ["accuracy"], random_seed=42
        )

        samples = []
        for _ in range(100):
            value = optimizer._sample_parameter("x", [1, 2, 3])
            samples.append(value)
            assert value in [1, 2, 3]

        # Should have all values represented
        assert set(samples) == {1, 2, 3}

    def test_sample_parameter_continuous_tuple(self):
        """Test sampling from continuous range."""
        optimizer = RandomSearchOptimizer({"x": [(0.0, 1.0)]}, ["accuracy"])

        samples = []
        for _ in range(100):
            value = optimizer._sample_parameter("x", (0.0, 1.0))
            samples.append(value)
            assert 0.0 <= value <= 1.0

        # Should have good spread
        assert min(samples) < 0.1
        assert max(samples) > 0.9

    def test_sample_parameter_integer_tuple(self):
        """Test sampling from integer range."""
        optimizer = RandomSearchOptimizer({"x": [(1, 10)]}, ["accuracy"])

        samples = []
        for _ in range(100):
            value = optimizer._sample_parameter("x", (1, 10))
            samples.append(value)
            assert isinstance(value, int)
            assert 1 <= value <= 10

        # Should have all values represented
        assert len(set(samples)) > 5

    def test_sample_parameter_fixed_value(self):
        """Test sampling fixed parameter value."""
        optimizer = RandomSearchOptimizer({"x": [0]}, ["accuracy"])

        # Single value list
        value = optimizer._sample_parameter("x", [42])
        assert value == 42

        # Non-list fixed value
        value = optimizer._sample_parameter("x", "fixed")
        assert value == "fixed"

    def test_should_stop(self):
        """Test stop condition checking."""
        optimizer = RandomSearchOptimizer({"x": [0, 1]}, ["accuracy"], max_trials=5)

        history = []

        # Initially should not stop
        assert optimizer.should_stop(history) is False

        # Simulate trials
        for i in range(5):
            optimizer._trial_count = i
            assert optimizer.should_stop(history) == (i >= 5)

        # At max trials, should stop
        optimizer._trial_count = 5
        assert optimizer.should_stop(history) is True

    def test_reset(self):
        """Test resetting optimizer state."""
        optimizer = RandomSearchOptimizer(
            {"x": [0, 1, 2]}, ["accuracy"], random_seed=42
        )

        # Generate some trials
        configs1 = []
        for _ in range(3):
            configs1.append(optimizer.suggest_next_trial([]))

        # Reset
        optimizer.reset()

        assert optimizer._trial_count == 0
        assert optimizer._best_score is None
        assert optimizer._best_config is None

        # Should generate same sequence after reset with seed
        configs2 = []
        for _ in range(3):
            configs2.append(optimizer.suggest_next_trial([]))

        assert configs1 == configs2

    def test_reset_without_seed(self):
        """Test reset without random seed."""
        optimizer = RandomSearchOptimizer({"x": [0, 1, 2]}, ["accuracy"])

        # Generate trial
        optimizer.suggest_next_trial([])
        assert optimizer._trial_count == 1

        # Reset
        optimizer.reset()
        assert optimizer._trial_count == 0

    def test_progress(self):
        """Test progress calculation."""
        optimizer = RandomSearchOptimizer({"x": [0, 1]}, ["accuracy"], max_trials=10)

        assert optimizer.progress == 0.0

        optimizer._trial_count = 5
        assert optimizer.progress == 0.5

        optimizer._trial_count = 10
        assert optimizer.progress == 1.0

        # Should cap at 1.0
        optimizer._trial_count = 15
        assert optimizer.progress == 1.0

    def test_set_max_trials_valid(self):
        """Test setting max trials with valid value."""
        optimizer = RandomSearchOptimizer({"x": [0, 1]}, ["accuracy"], max_trials=10)

        optimizer.set_max_trials(20)
        assert optimizer.max_trials == 20

        # Progress should update
        optimizer._trial_count = 10
        assert optimizer.progress == 0.5

    def test_set_max_trials_invalid(self):
        """Test setting max trials with invalid values."""
        optimizer = RandomSearchOptimizer({"x": [0, 1]}, ["accuracy"])

        with pytest.raises(ValueError):
            optimizer.set_max_trials(0)

        with pytest.raises(ValueError):
            optimizer.set_max_trials(-1)

    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        config_space = {"x": [0, 1], "y": (0.0, 1.0)}
        objectives = ["accuracy", "latency"]

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            max_trials=50,
            random_seed=42,
        )

        info = optimizer.get_algorithm_info()

        assert info["name"] == "RandomSearchOptimizer"
        assert "Random search optimization algorithm" in info["description"]
        assert info["config_space"] == config_space
        assert info["objectives"] == objectives
        assert info["max_trials"] == 50
        assert info["progress"] == 0.0
        assert info["supports_continuous"] is True
        assert info["supports_categorical"] is True
        assert info["deterministic"] is True
        assert info["random_seed"] == 42

    def test_deterministic_behavior(self):
        """Test deterministic behavior with random seed."""
        config_space = {"x": [0, 1, 2, 3, 4], "y": (0.0, 1.0), "z": ["a", "b", "c"]}

        # Create two optimizers with same seed
        opt1 = RandomSearchOptimizer(config_space, ["accuracy"], random_seed=123)
        opt2 = RandomSearchOptimizer(config_space, ["accuracy"], random_seed=123)

        # Generate configs from both
        configs1 = [opt1.suggest_next_trial([]) for _ in range(10)]
        configs2 = [opt2.suggest_next_trial([]) for _ in range(10)]

        # Should be identical
        assert configs1 == configs2

    def test_non_deterministic_behavior(self):
        """Test non-deterministic behavior without seed."""
        config_space = {"x": list(range(100))}

        # Create two optimizers without seed
        opt1 = RandomSearchOptimizer(config_space, ["accuracy"])
        opt2 = RandomSearchOptimizer(config_space, ["accuracy"])

        # Generate configs from both
        configs1 = [opt1.suggest_next_trial([]) for _ in range(10)]
        configs2 = [opt2.suggest_next_trial([]) for _ in range(10)]

        # Should be different (with very high probability)
        assert configs1 != configs2

    def test_empty_config_space(self):
        """Test behavior with empty config space."""
        optimizer = RandomSearchOptimizer({}, ["accuracy"])

        # Empty config space has cardinality 0, so it's immediately exhausted
        assert optimizer.config_space_cardinality == 0
        assert optimizer.is_config_space_exhausted()

        with pytest.raises(OptimizationError, match="Config space exhausted"):
            optimizer.suggest_next_trial([])

    def test_single_value_parameters(self):
        """Test parameters with single possible value."""
        config_space = {
            "fixed_list": [42],
            "fixed_value": "constant",
            "empty_range": (5, 5),
        }
        optimizer = RandomSearchOptimizer(config_space, ["accuracy"])

        for _ in range(5):
            config = optimizer.suggest_next_trial([])
            assert config["fixed_list"] == 42
            assert config["fixed_value"] == "constant"
            assert config["empty_range"] == 5

    def test_large_config_space(self):
        """Test performance with large configuration space."""
        config_space = {f"param_{i}": list(range(100)) for i in range(20)}
        optimizer = RandomSearchOptimizer(config_space, ["accuracy"])

        # Should handle large spaces efficiently
        import time

        start = time.time()

        for _ in range(100):
            config = optimizer.suggest_next_trial([])
            assert len(config) == 20

        duration = time.time() - start
        assert duration < 1.0  # Should be fast

    def test_logging_behavior(self):
        """Test logging behavior."""
        with patch("traigent.optimizers.random.logger") as mock_logger:
            optimizer = RandomSearchOptimizer(
                {"x": [0, 1]}, ["accuracy"], max_trials=10
            )

            # Check initialization logging
            mock_logger.info.assert_called_with(
                "Initialized random search with max_trials=10"
            )

            # Check trial suggestion logging
            optimizer.suggest_next_trial([])
            assert mock_logger.debug.called

            # Check max trials update logging
            optimizer.set_max_trials(20)
            assert any(
                "Updated max_trials to 20" in str(call)
                for call in mock_logger.info.call_args_list
            )

    def test_update_best_integration(self):
        """Test integration with base class update_best method."""
        optimizer = RandomSearchOptimizer({"x": [0, 1, 2]}, ["accuracy"])

        # Generate config
        config = optimizer.suggest_next_trial([])

        # Create trial result
        trial = TrialResult(
            trial_id="trial-1",
            config=config,
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        # Update best
        optimizer.update_best(trial)

        assert optimizer.best_score == 0.9
        assert optimizer.best_config == config
