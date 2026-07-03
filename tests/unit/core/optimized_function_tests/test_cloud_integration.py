"""Contract tests for deprecated OptimizedFunction cloud mode aliases."""

import warnings

import pytest

from traigent.config.types import ExecutionMode
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError


class TestCloudIntegration:
    """Cloud mode is deprecated and now fails closed."""

    def test_cloud_mode_deprecated_fails_closed(
        self,
        simple_function,
        sample_config_space,
        sample_objectives,
        sample_dataset,
        monkeypatch,
    ):
        """Deprecated cloud mode raises even with the old env override."""
        monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

        with pytest.raises(ConfigurationError, match="fails closed"):
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=sample_dataset,
                execution_mode="cloud",
            )

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
                execution_mode="local",
                cloud_fallback_policy="auto",
            )
        assert opt_func.execution_mode == ExecutionMode.LOCAL.value
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

    def test_local_mode_remains_supported(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Default local execution still initializes normally."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="local",
        )

        assert opt_func.execution_mode == ExecutionMode.LOCAL.value
        assert opt_func._effective_execution_mode is ExecutionMode.LOCAL

    def test_removed_edge_analytics_mode_raises_migration_error(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """The removed edge_analytics selector hard-fails at construction."""
        from traigent.utils.exceptions import ConfigurationError

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ConfigurationError, match="has been removed"):
                OptimizedFunction(
                    func=simple_function,
                    configuration_space=sample_config_space,
                    objectives=sample_objectives,
                    execution_mode="edge_analytics",
                )
