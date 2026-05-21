"""Contract tests for reserved OptimizedFunction cloud mode."""

import pytest

from traigent.config.types import ExecutionMode
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError


class TestCloudIntegration:
    """Cloud is reserved; supported modes continue to initialize."""

    def test_cloud_mode_fails_closed_at_initialization(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Public cloud mode must not construct a locally completing wrapper."""
        with pytest.raises(ConfigurationError, match="not available yet"):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                execution_mode="cloud",
            )

    def test_cloud_mode_fails_closed_even_with_auto_fallback(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Fallback policy cannot make the reserved cloud mode available."""
        with pytest.raises(ConfigurationError, match="not available yet"):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                execution_mode="cloud",
                cloud_fallback_policy="auto",
            )

    def test_hybrid_mode_remains_supported(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Portal-tracked local optimization uses hybrid mode today."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            execution_mode="hybrid",
        )

        assert opt_func.execution_mode == ExecutionMode.HYBRID.value
        assert opt_func._effective_execution_mode is ExecutionMode.HYBRID

    def test_invalid_cloud_fallback_policy_still_rejected(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Fallback policy values are validated even while cloud mode is reserved."""
        with pytest.raises(ValueError, match="cloud_fallback_policy"):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                execution_mode="hybrid",
                cloud_fallback_policy="sometimes",
            )

    def test_edge_mode_remains_supported(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Default local execution still initializes normally."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="edge_analytics",
        )

        assert opt_func.execution_mode == ExecutionMode.EDGE_ANALYTICS.value
        assert opt_func._effective_execution_mode is ExecutionMode.EDGE_ANALYTICS
