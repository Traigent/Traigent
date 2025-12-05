"""Comprehensive tests for grid search optimizer."""

from datetime import datetime
from unittest.mock import patch

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.exceptions import OptimizationError


class TestGridSearchOptimizer:
    """Test suite for GridSearchOptimizer."""

    def test_initialization_simple(self):
        """Test initialization with simple config space."""
        config_space = {"x": [0, 1, 2], "y": ["a", "b"]}
        objectives = ["accuracy"]

        optimizer = GridSearchOptimizer(config_space, objectives)

        assert optimizer.config_space == config_space
        assert optimizer.objectives == objectives
        assert optimizer.total_combinations == 6  # 3 * 2
        assert optimizer._current_index == 0
        assert optimizer._trial_count == 0

    def test_initialization_with_fixed_params(self):
        """Test initialization with fixed parameters."""
        config_space = {
            "x": [0, 1],
            "y": "fixed",  # Single fixed value
            "z": [True],  # List with single value
        }

        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer.total_combinations == 2  # Only x varies

        # Check that fixed params are included
        grid_point = optimizer._grid_points[0]
        assert grid_point["y"] == "fixed"
        assert grid_point["z"]

    def test_initialization_continuous_parameter_error(self):
        """Test that continuous parameters raise error."""
        config_space = {"x": [0, 1, 2], "y": (0.0, 1.0)}  # Continuous range

        with pytest.raises(OptimizationError, match="does not support continuous"):
            GridSearchOptimizer(config_space, ["accuracy"])

    def test_initialization_empty_config_space(self):
        """Test initialization with empty config space."""
        with pytest.raises(OptimizationError, match="No valid parameter combinations"):
            GridSearchOptimizer({}, ["accuracy"])

    def test_generate_grid_simple(self):
        """Test grid generation with simple parameters."""
        config_space = {"x": [1, 2], "y": ["a", "b", "c"]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        expected_grid = [
            {"x": 1, "y": "a"},
            {"x": 1, "y": "b"},
            {"x": 1, "y": "c"},
            {"x": 2, "y": "a"},
            {"x": 2, "y": "b"},
            {"x": 2, "y": "c"},
        ]

        assert optimizer._grid_points == expected_grid

    def test_generate_grid_with_many_params(self):
        """Test grid generation with many parameters."""
        config_space = {
            "a": [1, 2],
            "b": ["x", "y"],
            "c": [True, False],
            "d": [0.1, 0.2, 0.3],
        }
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer.total_combinations == 2 * 2 * 2 * 3  # 24

        # Check all combinations are unique
        unique_configs = {str(config) for config in optimizer._grid_points}
        assert len(unique_configs) == 24

    def test_suggest_next_trial_sequential(self):
        """Test suggesting trials in sequence."""
        config_space = {"x": [0, 1], "y": ["a", "b"]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        # Get all suggestions
        suggestions = []
        for i in range(4):
            config = optimizer.suggest_next_trial([])
            suggestions.append(config)
            assert optimizer._current_index == i + 1
            assert optimizer._trial_count == i + 1

        # Check we got all combinations
        expected = [
            {"x": 0, "y": "a"},
            {"x": 0, "y": "b"},
            {"x": 1, "y": "a"},
            {"x": 1, "y": "b"},
        ]
        assert suggestions == expected

    def test_suggest_next_trial_exhausted(self):
        """Test error when all combinations exhausted."""
        config_space = {"x": [0, 1]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        # Exhaust all combinations
        optimizer.suggest_next_trial([])
        optimizer.suggest_next_trial([])

        # Next should raise error
        with pytest.raises(OptimizationError, match="All grid combinations"):
            optimizer.suggest_next_trial([])

    def test_should_stop(self):
        """Test stop condition checking."""
        config_space = {"x": [0, 1], "y": ["a", "b"]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        history = []  # History not used by grid search

        # Initially should not stop
        assert optimizer.should_stop(history) is False

        # Suggest some trials
        optimizer.suggest_next_trial(history)
        assert optimizer.should_stop(history) is False

        optimizer.suggest_next_trial(history)
        assert optimizer.should_stop(history) is False

        # Exhaust all
        optimizer.suggest_next_trial(history)
        optimizer.suggest_next_trial(history)

        # Now should stop
        assert optimizer.should_stop(history) is True

    def test_reset(self):
        """Test resetting optimizer state."""
        config_space = {"x": [0, 1], "y": ["a", "b"]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        # Suggest some trials
        optimizer.suggest_next_trial([])
        optimizer.suggest_next_trial([])

        assert optimizer._current_index == 2
        assert optimizer._trial_count == 2

        # Reset
        optimizer.reset()

        assert optimizer._current_index == 0
        assert optimizer._trial_count == 0
        assert optimizer._best_score is None
        assert optimizer._best_config is None

        # Should be able to suggest same sequence again
        config1 = optimizer.suggest_next_trial([])
        assert config1 == {"x": 0, "y": "a"}

    def test_total_combinations_property(self):
        """Test total_combinations property."""
        config_space = {"a": [1, 2, 3], "b": ["x", "y"], "c": [True, False, None]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer.total_combinations == 3 * 2 * 3  # 18

    def test_progress_property(self):
        """Test progress calculation."""
        config_space = {"x": [0, 1, 2, 3]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer.progress == 0.0

        optimizer.suggest_next_trial([])
        assert optimizer.progress == 0.25

        optimizer.suggest_next_trial([])
        assert optimizer.progress == 0.5

        optimizer.suggest_next_trial([])
        assert optimizer.progress == 0.75

        optimizer.suggest_next_trial([])
        assert optimizer.progress == 1.0

        # Should cap at 1.0 even if index goes beyond
        optimizer._current_index = 10
        assert optimizer.progress == 1.0

    def test_progress_empty_grid(self):
        """Test progress with empty grid."""
        optimizer = GridSearchOptimizer({"x": [1]}, ["accuracy"])
        optimizer._grid_points = []  # Simulate empty grid

        assert optimizer.progress == 0.0

    def test_get_remaining_combinations(self):
        """Test getting remaining combinations."""
        config_space = {"x": [0, 1, 2], "y": ["a", "b"]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        # Initially all should be remaining
        remaining = optimizer.get_remaining_combinations()
        assert len(remaining) == 6

        # Suggest some
        optimizer.suggest_next_trial([])
        optimizer.suggest_next_trial([])

        remaining = optimizer.get_remaining_combinations()
        assert len(remaining) == 4
        assert remaining[0] == {"x": 1, "y": "a"}  # Next in sequence (sorted order)

        # Exhaust all
        for _ in range(4):
            optimizer.suggest_next_trial([])

        remaining = optimizer.get_remaining_combinations()
        assert len(remaining) == 0

    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        config_space = {"x": [0, 1], "y": ["a", "b", "c"]}
        objectives = ["accuracy", "latency"]

        optimizer = GridSearchOptimizer(config_space, objectives)

        info = optimizer.get_algorithm_info()

        assert info["name"] == "GridSearchOptimizer"
        assert "Grid search optimization algorithm" in info["description"]
        assert info["config_space"] == config_space
        assert info["objectives"] == objectives
        assert info["total_combinations"] == 6
        assert info["progress"] == 0.0
        assert info["supports_continuous"] is False
        assert info["supports_categorical"] is True
        assert info["deterministic"] is True

    def test_deterministic_behavior(self):
        """Test that grid search is deterministic."""
        config_space = {"x": [0, 1, 2], "y": ["a", "b"], "z": [True, False]}

        # Create two optimizers
        opt1 = GridSearchOptimizer(config_space, ["accuracy"])
        opt2 = GridSearchOptimizer(config_space, ["accuracy"])

        # Generate all configs from both
        configs1 = []
        configs2 = []

        while not opt1.should_stop([]):
            configs1.append(opt1.suggest_next_trial([]))

        while not opt2.should_stop([]):
            configs2.append(opt2.suggest_next_trial([]))

        # Should be identical
        assert configs1 == configs2

    def test_large_grid_space(self):
        """Test handling of large grid spaces."""
        # Create a moderately large space
        config_space = {f"param_{i}": list(range(5)) for i in range(5)}

        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        # Should have 5^5 = 3125 combinations
        assert optimizer.total_combinations == 3125

        # Should handle efficiently
        import time

        start = time.time()

        # Get first 100 suggestions
        for _ in range(100):
            optimizer.suggest_next_trial([])

        duration = time.time() - start
        assert duration < 0.1  # Should be fast

    def test_single_parameter_grid(self):
        """Test grid with single parameter."""
        config_space = {"x": [1, 2, 3, 4, 5]}
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer.total_combinations == 5

        configs = []
        while not optimizer.should_stop([]):
            configs.append(optimizer.suggest_next_trial([]))

        expected = [{"x": i} for i in [1, 2, 3, 4, 5]]
        assert configs == expected

    def test_boolean_parameters(self):
        """Test grid with boolean parameters."""
        config_space = {
            "use_feature_a": [True, False],
            "use_feature_b": [True, False],
            "use_feature_c": [True, False],
        }
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer.total_combinations == 8  # 2^3

        # Check all combinations are generated
        configs = []
        while not optimizer.should_stop([]):
            configs.append(optimizer.suggest_next_trial([]))

        # Verify we have all boolean combinations
        assert len(configs) == 8
        assert all(isinstance(c["use_feature_a"], bool) for c in configs)

    def test_mixed_type_parameters(self):
        """Test grid with mixed parameter types."""
        config_space = {
            "int_param": [1, 2, 3],
            "str_param": ["a", "b"],
            "bool_param": [True, False],
            "float_param": [0.1, 0.2],
            "none_param": [None, "not_none"],
        }
        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer.total_combinations == 3 * 2 * 2 * 2 * 2  # 48

        # Get a few configs and check types
        for _ in range(5):
            config = optimizer.suggest_next_trial([])
            assert isinstance(config["int_param"], int)
            assert isinstance(config["str_param"], str)
            assert isinstance(config["bool_param"], bool)
            assert isinstance(config["float_param"], float)

    def test_logging_behavior(self):
        """Test logging behavior."""
        with patch("traigent.optimizers.grid.logger") as mock_logger:
            config_space = {"x": [0, 1], "y": ["a", "b", "c"]}
            optimizer = GridSearchOptimizer(config_space, ["accuracy"])

            # Check initialization logging
            mock_logger.info.assert_called_with("Generated grid with 6 combinations")

            # Check trial suggestion logging
            optimizer.suggest_next_trial([])
            assert mock_logger.debug.called
            debug_call = mock_logger.debug.call_args[0][0]
            assert "Suggesting trial 1/6" in debug_call

    def test_update_best_integration(self):
        """Test integration with base class update_best method."""
        optimizer = GridSearchOptimizer({"x": [0, 1], "y": ["a", "b"]}, ["accuracy"])

        # Generate and evaluate trials
        history = []
        scores = [0.7, 0.9, 0.8, 0.6]

        for i in range(4):
            config = optimizer.suggest_next_trial(history)

            trial = TrialResult(
                trial_id=f"trial-{i}",
                config=config,
                metrics={"accuracy": scores[i]},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )

            optimizer.update_best(trial)
            history.append(trial)

        # Best should be second trial with score 0.9
        assert optimizer.best_score == 0.9
        assert optimizer.best_config == {"x": 0, "y": "b"}

    def test_parameter_order_consistency(self):
        """Test that parameter order is consistent."""
        config_space = {"z": [1, 2], "a": ["x", "y"], "m": [True, False]}

        optimizer = GridSearchOptimizer(config_space, ["accuracy"])
        assert optimizer._ordered_param_names == ["a", "m", "z"]

        # Get all configs
        configs = []
        while not optimizer.should_stop([]):
            configs.append(optimizer.suggest_next_trial([]))

        # Check that all configs have same key order
        keys_list = [list(config.keys()) for config in configs]
        assert all(keys == keys_list[0] for keys in keys_list)

    def test_model_prioritized_when_no_order(self):
        """Ensure model parameter appears last when no order is provided.

        By placing model last in itertools.product, it varies fastest,
        allowing quick comparison of different models with the same other parameters.
        """
        config_space = {
            "temperature": [0.0, 0.5],
            "embedding_model": ["embed-a"],
            "model": ["gpt-4o"],
            "prompt": ["cot", "zero_shot"],
        }

        optimizer = GridSearchOptimizer(config_space, ["accuracy"])

        assert optimizer._ordered_param_names == [
            "embedding_model",
            "prompt",
            "temperature",
            "model",
        ]

        first_config = optimizer._grid_points[0]
        assert list(first_config.keys()) == [
            "embedding_model",
            "prompt",
            "temperature",
            "model",
        ]

    def test_explicit_parameter_order_overrides_default(self):
        """Ensure explicit numeric ordering takes precedence."""
        config_space = {
            "temperature": [0.0, 0.5],
            "embedding_model": ["embed-a"],
            "model": ["gpt-4o"],
            "prompt": ["cot", "zero_shot"],
        }

        optimizer = GridSearchOptimizer(
            config_space,
            ["accuracy"],
            parameter_order={"temperature": 0, "model": 2},
        )

        assert optimizer._ordered_param_names == [
            "temperature",
            "model",
            "embedding_model",
            "prompt",
        ]

        first_config = optimizer._grid_points[0]
        assert list(first_config.keys()) == [
            "temperature",
            "model",
            "embedding_model",
            "prompt",
        ]

    def test_parameter_order_invalid_type_raises(self):
        """Reject non-mapping parameter order specifications."""
        config_space = {"model": ["gpt-4o"], "temperature": [0.0, 0.5]}

        with pytest.raises(
            OptimizationError, match="parameter_order must be a mapping"
        ):
            GridSearchOptimizer(
                config_space,
                ["accuracy"],
                parameter_order=["model", "temperature"],  # type: ignore[arg-type]
            )

    def test_parameter_order_non_numeric_priority(self):
        """Raise error when priorities are not numeric values."""
        config_space = {"model": ["gpt-4o"], "temperature": [0.0, 0.5]}

        with pytest.raises(
            OptimizationError, match="parameter_order for 'model' must be numeric"
        ):
            GridSearchOptimizer(
                config_space,
                ["accuracy"],
                parameter_order={"model": "first"},  # type: ignore[arg-type]
            )
