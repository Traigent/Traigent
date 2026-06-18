"""Contract tests for deprecated OptimizedFunction cloud mode aliases."""

import warnings


from traigent.config.types import ExecutionMode
from traigent.core.optimized_function import OptimizedFunction


class TestCloudIntegration:
    """Cloud mode is deprecated; it resolves to edge_analytics with a DeprecationWarning."""

    def test_cloud_mode_deprecated_resolves_to_edge_analytics(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Deprecated cloud mode resolves to edge_analytics with DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt_func = OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                execution_mode="cloud",
            )
        assert opt_func.execution_mode == ExecutionMode.EDGE_ANALYTICS.value
        assert opt_func._effective_execution_mode is ExecutionMode.EDGE_ANALYTICS
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_cloud_mode_with_fallback_policy_emits_deprecation(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """cloud_fallback_policy param is deprecated and emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt_func = OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                execution_mode="edge_analytics",
                cloud_fallback_policy="auto",
            )
        assert opt_func.execution_mode == ExecutionMode.EDGE_ANALYTICS.value
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

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
        """Unknown cloud_fallback_policy values are warned (no longer raise ValueError)."""
        # cloud_fallback_policy is fully deprecated; any value just emits DeprecationWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt_func = OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                execution_mode="hybrid",
                cloud_fallback_policy="sometimes",
            )
        assert opt_func.execution_mode == ExecutionMode.HYBRID.value
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

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
