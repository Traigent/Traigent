"""Tests for OptimizedFunction initialization and configuration.

Tests basic creation, validation, and configuration of OptimizedFunction instances.
"""

from unittest.mock import Mock

import pytest

from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ValidationError


class TestOptimizedFunctionInitialization:
    """Test OptimizedFunction initialization."""

    def test_basic_initialization(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test basic OptimizedFunction creation."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
        )

        assert opt_func is not None
        assert opt_func.func == simple_function
        assert opt_func.configuration_space == sample_config_space
        assert opt_func.objectives == sample_objectives
        assert opt_func.__name__ == "mock_function"

    def test_initialization_with_all_parameters(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test initialization with all optional parameters."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            max_trials=50,  # renamed from num_trials
            timeout=300,
            execution_mode="cloud",
            injection_mode="context",  # Use context injection instead
            # These are passed via kwargs and stored for later use
            parallel_config={"trial_concurrency": 2},
            cache_results=True,
            verbose=True,
        )

        assert opt_func.max_trials == 50
        assert opt_func.timeout == 300
        assert opt_func.execution_mode == "cloud"
        assert opt_func.injection_mode == "context"
        # Check kwargs for other parameters
        parallel_cfg = opt_func.kwargs.get("parallel_config")
        assert parallel_cfg is not None
        assert getattr(parallel_cfg, "trial_concurrency", None) == 2
        assert opt_func.kwargs.get("cache_results") is True

    def test_cost_limit_stored_in_decorator_runtime_overrides(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that cost_limit passed at decoration time is stored for later use.

        This verifies that cost_limit (and cost_approved) can be specified in the
        @traigent.optimize() decorator and will be properly forwarded to the
        orchestrator when .optimize() is called.
        """
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            cost_limit=2.5,
            cost_approved=True,
        )

        # Verify values are stored in _decorator_runtime_overrides
        assert opt_func._decorator_runtime_overrides.get("cost_limit") == pytest.approx(
            2.5
        )
        assert opt_func._decorator_runtime_overrides.get("cost_approved") is True

    def test_initialization_with_custom_evaluator(
        self, simple_function, sample_config_space
    ):
        """Test initialization with custom evaluator."""

        def custom_evaluator(func, config, example):
            # Simple custom evaluator
            return Mock(metrics={"custom": 1.0})

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            custom_evaluator=custom_evaluator,
        )

        assert opt_func.custom_evaluator == custom_evaluator

    def test_invalid_function_type(self, sample_config_space, sample_objectives):
        """Test error when function is not callable."""
        with pytest.raises(TypeError, match="func must be callable"):
            OptimizedFunction(
                func="not a function",
                configuration_space=sample_config_space,
                objectives=sample_objectives,
            )

    def test_invalid_configuration_space(self, simple_function, sample_objectives):
        """Test error with invalid configuration space."""

        with pytest.raises((TypeError, ValidationError)):
            OptimizedFunction(
                func=simple_function,
                configuration_space="invalid",  # Should be dict
                objectives=sample_objectives,
            )

    def test_empty_configuration_space(self, simple_function, sample_objectives):
        """Test error with empty configuration space."""
        with pytest.raises(ValueError, match="Configuration space cannot be empty"):
            OptimizedFunction(
                func=simple_function,
                configuration_space={},
                objectives=sample_objectives,
            )

    def test_invalid_objectives_type(self, simple_function, sample_config_space):
        """Test error with invalid objectives type."""
        # Now raises ValidationError from validation module

        with pytest.raises((TypeError, ValidationError)):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives="accuracy",  # Should be list
            )

    def test_empty_objectives_uses_default(self, simple_function, sample_config_space):
        """Test that empty objectives uses default accuracy."""
        # Passing empty list raises error, but None uses default
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=None,  # Will use default
        )

        # Should use default objectives
        assert opt_func.objectives == ["accuracy"]

    def test_negative_num_trials(self, simple_function, sample_config_space):
        """Test error with negative max_trials."""
        with pytest.raises(ValueError, match="max_trials must be non-negative"):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                max_trials=-1,
            )

    def test_negative_timeout(self, simple_function, sample_config_space):
        """Test error with negative timeout."""
        with pytest.raises(ValueError, match="timeout must be non-negative"):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                timeout=-1,
            )

    def test_invalid_injection_mode(self, simple_function, sample_config_space):
        """Test error with invalid injection mode."""
        # Will raise ValueError (wrapping ConfigurationError)

        with pytest.raises(ValueError):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                injection_mode="invalid",
            )

    def test_initialization_with_default_config(
        self, simple_function, sample_config_space
    ):
        """Test initialization with default configuration."""
        default_config = {"temperature": 0.5, "max_tokens": 100, "model": "gpt-3.5"}

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            default_config=default_config,
        )

        # Should accept valid default config
        assert opt_func.default_config == default_config

    def test_initialization_preserves_function_attributes(self, sample_config_space):
        """Test that OptimizedFunction preserves original function attributes."""

        def custom_function(x: int) -> int:
            """Custom function with docstring."""
            return x * 2

        custom_function.custom_attr = "test_value"

        opt_func = OptimizedFunction(
            func=custom_function, configuration_space=sample_config_space
        )

        assert opt_func.__name__ == "custom_function"
        assert opt_func.__doc__ == "Custom function with docstring."
        # OptimizedFunction doesn't copy custom attributes from the original function
        # It wraps the function, so we check the wrapped function
        assert hasattr(opt_func.func, "custom_attr")
        assert opt_func.func.custom_attr == "test_value"
