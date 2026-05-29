"""Comprehensive tests for traigent.api.functions module."""

import warnings
from unittest.mock import Mock, patch

import pytest

from traigent.api.functions import (
    _GLOBAL_CONFIG,
    configure,
    configure_for_budget,
    get_available_strategies,
    get_config,
    get_current_config,
    get_global_config,
    get_trial_config,
    get_version_info,
    override_config,
)
from traigent.config.types import TraigentConfig
from traigent.utils.exceptions import ConfigAccessWarning, OptimizationStateError


def test_batch_optimizers_are_publicly_exported() -> None:
    """Batch optimizer classes should be importable from traigent.optimizers."""
    from traigent.optimizers import (
        AdaptiveBatchOptimizer,
        BatchOptimizationConfig,
        MultiObjectiveBatchOptimizer,
        ParallelBatchOptimizer,
    )

    assert BatchOptimizationConfig.__name__ == "BatchOptimizationConfig"
    assert ParallelBatchOptimizer.__name__ == "ParallelBatchOptimizer"
    assert MultiObjectiveBatchOptimizer.__name__ == "MultiObjectiveBatchOptimizer"
    assert AdaptiveBatchOptimizer.__name__ == "AdaptiveBatchOptimizer"


class TestConfigure:
    """Test the configure function."""

    def setup_method(self):
        """Reset global config before each test."""
        # Store original config
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    def test_configure_default_storage_backend(self):
        """Test configuring default storage backend."""
        result = configure(default_storage_backend="s3")
        assert result is True
        assert get_global_config()["default_storage_backend"] == "s3"

    def test_configure_parallel_workers(self):
        """Test configuring parallel workers."""
        result = configure(parallel_workers=8)
        assert result is True
        assert get_global_config()["parallel_workers"] == 8

    def test_configure_invalid_parallel_workers(self):
        """Test configuring with invalid parallel workers."""
        with pytest.raises(ValueError, match="Must be positive"):
            configure(parallel_workers=0)

        with pytest.raises(ValueError, match="Must be positive"):
            configure(parallel_workers=-5)

    def test_configure_cache_policy(self):
        """Test configuring cache policy."""
        result = configure(cache_policy="disk")
        assert result is True
        assert get_global_config()["cache_policy"] == "disk"

    def test_configure_invalid_cache_policy(self):
        """Test configuring with invalid cache policy."""
        with pytest.raises(ValueError, match="Must be one of"):
            configure(cache_policy="invalid_policy")

    @patch("traigent.api.functions.setup_logging")
    def test_configure_logging_level(self, mock_setup_logging):
        """Test configuring logging level."""
        result = configure(logging_level="DEBUG")
        assert result is True
        assert get_global_config()["logging_level"] == "DEBUG"
        mock_setup_logging.assert_called_once_with(level="DEBUG")

    def test_configure_invalid_logging_level(self):
        """Test configuring with invalid logging level."""
        with pytest.raises(ValueError, match="Must be one of"):
            configure(logging_level="INVALID")

    def test_configure_api_keys(self):
        """Test configuring API keys."""
        api_keys = {"openai": "example-key-123", "anthropic": "example-key-456"}

        # Should trigger security warning
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = configure(api_keys=api_keys)

            # Should warn about API keys in code
            assert any(
                "API keys detected in code" in str(warning.message) for warning in w
            )

        assert result is True

        config = get_global_config()
        assert config["api_keys"]["openai"] == "example-key-123"
        assert config["api_keys"]["anthropic"] == "example-key-456"

    def test_configure_api_keys_update(self):
        """Test that API keys are updated, not replaced."""
        # First configure
        configure(api_keys={"openai": "example-key-old"})

        # Update with new keys
        configure(api_keys={"anthropic": "example-key-new"})

        config = get_global_config()
        assert config["api_keys"]["openai"] == "example-key-old"  # Preserved
        assert config["api_keys"]["anthropic"] == "example-key-new"  # Added

    def test_configure_invalid_api_keys(self):
        """Test configuring with invalid API keys."""
        with pytest.raises(ValueError, match="Expected dict"):
            configure(api_keys="not_a_dict")  # pragma: allowlist secret

    @patch("traigent.api.functions.logger")
    def test_configure_logging(self, mock_logger):
        """Test that configuration is logged."""
        configure(parallel_workers=4)
        mock_logger.info.assert_called_with("Updated global configuration")

    def test_configure_multiple_settings(self):
        """Test configuring multiple settings at once."""
        result = configure(
            default_storage_backend="gcs",
            parallel_workers=6,
            cache_policy="distributed",
            logging_level="WARNING",
            api_keys={"test": "key"},
        )

        assert result is True
        config = get_global_config()
        assert config["default_storage_backend"] == "gcs"
        assert config["parallel_workers"] == 6
        assert config["cache_policy"] == "distributed"
        assert config["logging_level"] == "WARNING"
        assert config["api_keys"]["test"] == "key"

    def test_configure_partial_update(self):
        """Test that unconfigured values remain unchanged."""
        original_workers = get_global_config()["parallel_workers"]

        configure(cache_policy="disk")

        config = get_global_config()
        assert config["cache_policy"] == "disk"
        assert config["parallel_workers"] == original_workers  # Unchanged


class TestGetGlobalConfig:
    """Test the get_global_config function."""

    def test_get_global_config_returns_copy(self):
        """Test that get_global_config returns a copy."""
        config1 = get_global_config()
        config2 = get_global_config()

        # Should be equal but not the same object
        assert config1 == config2
        assert config1 is not config2

        # Modifying returned config shouldn't affect global
        config1["test_key"] = "test_value"
        assert "test_key" not in get_global_config()

    def test_get_global_config_content(self):
        """Test that get_global_config returns expected content."""
        config = get_global_config()

        # Check expected keys exist
        assert "default_storage_backend" in config
        assert "parallel_workers" in config
        assert "cache_policy" in config
        assert "logging_level" in config
        assert "api_keys" in config


class TestGetCurrentConfig:
    """Test the get_current_config function."""

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_from_traigent_config(self, mock_get_config):
        """Test getting config from TraigentConfig object."""
        mock_config = Mock(spec=TraigentConfig)
        mock_config.to_dict.return_value = {"model": "gpt-4", "temperature": 0.7}
        mock_get_config.return_value = mock_config

        result = get_current_config()

        assert result == {"model": "gpt-4", "temperature": 0.7}
        mock_config.to_dict.assert_called_once()

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_from_dict(self, mock_get_config):
        """Test getting config from dict."""
        mock_get_config.return_value = {"model": "gpt-3.5", "max_tokens": 100}

        result = get_current_config()

        assert result == {"model": "gpt-3.5", "max_tokens": 100}
        assert result is not mock_get_config.return_value  # Should be a copy

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_fallback(self, mock_get_config):
        """Test fallback when config is neither TraigentConfig nor dict."""
        mock_get_config.return_value = "not_a_config"

        result = get_current_config()

        assert result == {}

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_none(self, mock_get_config):
        """Test when context config returns None."""
        mock_get_config.return_value = None

        result = get_current_config()

        assert result == {}


class TestOverrideConfig:
    """Test the override_config function."""

    def test_override_config_objectives(self):
        """Test overriding objectives."""
        result = override_config(objectives=["accuracy", "cost"])
        assert result == {"objectives": ["accuracy", "cost"]}

    def test_override_config_configuration_space(self):
        """Test overriding configuration space."""
        config_space = {"model": ["gpt-3.5", "gpt-4"], "temperature": [0.5, 0.7, 0.9]}
        result = override_config(configuration_space=config_space)
        assert result == {"configuration_space": config_space}

    def test_override_config_constraints(self):
        """Test overriding constraints."""

        def constraint1(x):
            return x > 0

        def constraint2(x):
            return x < 100

        result = override_config(constraints=[constraint1, constraint2])
        assert result["constraints"] == [constraint1, constraint2]

    def test_override_config_max_trials(self):
        """Test overriding max trials."""
        result = override_config(max_trials=50)
        assert result == {"max_trials": 50}

    def test_override_config_invalid_max_trials(self):
        """Test invalid max trials."""
        result = override_config(max_trials=0)
        assert result == {"max_trials": 0}

        with pytest.raises(ValueError, match="max_trials must be non-negative"):
            override_config(max_trials=-10)

    def test_override_config_timeout(self):
        """Test overriding timeout."""
        result = override_config(timeout=3600)
        assert result == {"timeout": 3600}

    def test_override_config_invalid_timeout(self):
        """Test invalid timeout."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            override_config(timeout=0)

        with pytest.raises(ValueError, match="timeout must be > 0"):
            override_config(timeout=-100)

    def test_override_config_max_total_examples(self):
        """Test overriding global sample budget and pruned toggle."""
        result = override_config(max_total_examples=250, samples_include_pruned=False)
        assert result["max_total_examples"] == 250
        assert result["samples_include_pruned"] is False

    def test_override_config_invalid_max_total_examples(self):
        """Test invalid sample budget."""
        with pytest.raises(ValueError, match="max_total_examples must be > 0"):
            override_config(max_total_examples=0)

    def test_override_config_multiple(self):
        """Test overriding multiple settings."""
        result = override_config(
            objectives=["cost"],
            configuration_space={"model": ["gpt-3.5"]},
            max_trials=10,
            timeout=600,
        )

        assert result == {
            "objectives": ["cost"],
            "configuration_space": {"model": ["gpt-3.5"]},
            "max_trials": 10,
            "timeout": 600,
        }

    def test_override_config_empty(self):
        """Test override with no arguments."""
        result = override_config()
        assert result == {}


class TestConfigureForBudget:
    """Tests for budget-aware configuration helper."""

    def test_filters_models_and_derives_limits(self):
        """Affordable models should be selected and optimize-safe overrides returned."""
        recommendation = configure_for_budget(
            budget_usd=10.0,
            model_pricing={
                "cheap-model": 0.5,
                "mid-model": 1.0,
                "expensive-model": 3.0,
            },
            min_instances=5,
            reserve_ratio=0.0,
            max_parallel_workers=3,
        )

        assert recommendation["configuration_space"]["model"] == [
            "cheap-model",
            "mid-model",
        ]
        assert recommendation["max_trials"] == 20
        assert recommendation["cost_limit"] == 10.0
        assert "parallel_config" in recommendation
        assert "max_instances" not in recommendation
        assert "parallel_workers" not in recommendation

    def test_raises_when_no_affordable_model(self):
        """No affordable model should fail with clear message."""
        with pytest.raises(ValueError, match="No models can satisfy"):
            configure_for_budget(
                budget_usd=2.0,
                model_pricing={"model-a": 1.0},
                min_instances=5,
            )

    def test_validates_inputs(self):
        """Invalid helper inputs should raise ValueError."""
        with pytest.raises(ValueError, match="budget_usd must be > 0"):
            configure_for_budget(budget_usd=0.0, model_pricing={"m": 1.0})
        with pytest.raises(ValueError, match="model_pricing must not be empty"):
            configure_for_budget(budget_usd=10.0, model_pricing={})
        with pytest.raises(ValueError, match="reserve_ratio must be in"):
            configure_for_budget(
                budget_usd=10.0, model_pricing={"m": 1.0}, reserve_ratio=1.0
            )

    def test_returns_diagnostics_when_requested(self):
        """Diagnostics should be returned separately when requested."""
        overrides, diagnostics = configure_for_budget(
            budget_usd=10.0,
            model_pricing={"cheap-model": 0.5, "mid-model": 1.0},
            min_instances=2,
            max_parallel_workers=4,
            return_diagnostics=True,
        )

        assert set(overrides.keys()) == {
            "configuration_space",
            "max_trials",
            "parallel_config",
            "cost_limit",
        }
        assert diagnostics["max_instances"] == 18
        assert diagnostics["parallel_workers"] == 4
        assert diagnostics["budget_usd"] == 10.0
        assert diagnostics["effective_budget_usd"] == 9.0


class TestGetAvailableStrategies:
    """Test the get_available_strategies function."""

    @patch("traigent.api.functions.list_optimizers")
    def test_get_available_strategies_grid(self, mock_list_optimizers):
        """Test grid search strategy info."""
        mock_list_optimizers.return_value = ["grid"]

        strategies = get_available_strategies()

        assert "grid" in strategies
        grid_info = strategies["grid"]
        assert grid_info["name"] == "Grid Search"
        assert grid_info["supports_continuous"] is False
        assert grid_info["supports_categorical"] is True
        assert grid_info["deterministic"] is True
        assert "description" in grid_info
        assert "parameters" in grid_info
        assert "best_for" in grid_info

    @patch("traigent.api.functions.list_optimizers")
    def test_get_available_strategies_random(self, mock_list_optimizers):
        """Test random search strategy info."""
        mock_list_optimizers.return_value = ["random"]

        strategies = get_available_strategies()

        assert "random" in strategies
        random_info = strategies["random"]
        assert random_info["name"] == "Random Search"
        assert random_info["supports_continuous"] is True
        assert random_info["supports_categorical"] is True
        assert random_info["deterministic"] is False
        assert "max_trials" in random_info["parameters"]
        assert "random_seed" in random_info["parameters"]

    @patch("traigent.api.functions.list_optimizers")
    def test_get_available_strategies_bayesian(self, mock_list_optimizers):
        """Test Bayesian optimization strategy info."""
        mock_list_optimizers.return_value = ["bayesian"]

        strategies = get_available_strategies()

        assert "bayesian" in strategies
        bayesian_info = strategies["bayesian"]
        assert bayesian_info["name"] == "Bayesian Optimization"
        assert bayesian_info["supports_continuous"] is True
        assert bayesian_info["supports_categorical"] is True
        assert bayesian_info["deterministic"] is False
        assert "acquisition_function" in bayesian_info["parameters"]
        assert "initial_random_samples" in bayesian_info["parameters"]

    @patch("traigent.api.functions.list_optimizers")
    def test_get_available_strategies_optuna_metadata(self, mock_list_optimizers):
        """Test registered Optuna strategies have concrete metadata."""
        mock_list_optimizers.return_value = ["optuna_tpe", "nsga2", "optuna_grid"]

        strategies = get_available_strategies()

        assert strategies["optuna_tpe"]["name"] == "Optuna TPE Optimization"
        assert strategies["nsga2"]["name"] == "Optuna NSGA-II Optimization"
        assert strategies["optuna_grid"]["description"] != "Custom optimization algorithm"
        assert "max_trials" in strategies["optuna_tpe"]["parameters"]

    @patch("traigent.api.functions.list_optimizers")
    def test_get_available_strategies_custom(self, mock_list_optimizers):
        """Test custom strategy info."""
        mock_list_optimizers.return_value = ["custom_algo"]

        strategies = get_available_strategies()

        assert "custom_algo" in strategies
        custom_info = strategies["custom_algo"]
        assert custom_info["name"] == "Custom_Algo"
        assert custom_info["description"] == "Custom optimization algorithm"

    @patch("traigent.api.functions.list_optimizers")
    def test_get_available_strategies_multiple(self, mock_list_optimizers):
        """Test multiple strategies."""
        mock_list_optimizers.return_value = ["grid", "random", "bayesian"]

        strategies = get_available_strategies()

        assert len(strategies) == 3
        assert all(algo in strategies for algo in ["grid", "random", "bayesian"])


class TestGetVersionInfo:
    """Test the get_version_info function."""

    @patch("traigent.api.functions.list_optimizers")
    @patch("platform.platform")
    @patch("sys.version", "3.9.0")
    def test_get_version_info(self, mock_platform, mock_list_optimizers):
        """Test version info retrieval."""
        mock_platform.return_value = "Linux-5.10.0"
        mock_list_optimizers.return_value = ["grid", "random", "bayesian"]

        info = get_version_info()

        # Check structure
        assert "version" in info
        assert "python_version" in info
        assert "platform" in info
        assert "algorithms" in info
        assert "features" in info
        assert "integrations" in info
        assert "global_config" in info

        # Check algorithms
        assert info["algorithms"] == ["grid", "random", "bayesian"]

        # Check features
        features = info["features"]
        assert features["grid_search"] is True
        assert features["random_search"] is True
        assert features["bayesian_optimization"] is True
        assert features["multi_objective"] is True
        assert features["constraint_handling"] is True
        assert features["async_evaluation"] is True
        assert features["parallel_evaluation"] is True
        assert features["seamless_injection"] is True

        # Check integrations
        integrations = info["integrations"]
        assert "langchain" in integrations
        assert "openai" in integrations
        assert "mlflow" in integrations

        # Check global config is a copy
        assert info["global_config"] == _GLOBAL_CONFIG
        assert info["global_config"] is not _GLOBAL_CONFIG


class TestIntegration:
    """Integration tests for API functions."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    def test_configure_and_get_config_integration(self):
        """Test configuration and retrieval integration."""
        # Configure with specific settings
        result = configure(
            parallel_workers=4, logging_level="INFO", cache_policy="disk"
        )

        # Verify configuration was successful
        assert result is True

        # Check global config was updated
        global_config = get_global_config()
        assert global_config["parallel_workers"] == 4
        assert global_config["logging_level"] == "INFO"
        assert global_config["cache_policy"] == "disk"


class TestGetTrialConfig:
    """Test the get_trial_config function."""

    def test_get_trial_config_raises_when_no_context(self):
        """Test that get_trial_config raises when called outside optimization."""
        # No mocking needed - default context is None
        with pytest.raises(OptimizationStateError) as exc_info:
            get_trial_config()

        assert "can only be called during an active optimization trial" in str(
            exc_info.value
        )

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions._get_context_config")
    def test_get_trial_config_from_traigent_config(
        self, mock_get_config, mock_get_trial_context
    ):
        """Test getting trial config from TraigentConfig object."""
        # Mock that we're in an active trial
        mock_get_trial_context.return_value = {"trial_id": "test_trial"}

        mock_config = Mock(spec=TraigentConfig)
        mock_config.to_dict.return_value = {"model": "gpt-4", "temperature": 0.7}
        mock_get_config.return_value = mock_config

        result = get_trial_config()

        assert result == {"model": "gpt-4", "temperature": 0.7}
        mock_config.to_dict.assert_called_once()

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions._get_context_config")
    def test_get_trial_config_from_dict(self, mock_get_config, mock_get_trial_context):
        """Test getting trial config from dict."""
        # Mock that we're in an active trial
        mock_get_trial_context.return_value = {"trial_id": "test_trial"}

        mock_get_config.return_value = {"model": "gpt-3.5", "max_tokens": 100}

        result = get_trial_config()

        assert result == {"model": "gpt-3.5", "max_tokens": 100}

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions._get_context_config")
    def test_get_trial_config_returns_copy(
        self, mock_get_config, mock_get_trial_context
    ):
        """Test that get_trial_config returns a copy, not the original."""
        # Mock that we're in an active trial
        mock_get_trial_context.return_value = {"trial_id": "test_trial"}

        original_dict = {"model": "gpt-4"}
        mock_get_config.return_value = original_dict

        result = get_trial_config()

        # Modify the result
        result["model"] = "claude"

        # Original should be unchanged
        assert original_dict["model"] == "gpt-4"

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions._get_context_config")
    def test_get_trial_config_raises_on_invalid_config_type(
        self, mock_get_config, mock_get_trial_context
    ):
        """Test that invalid config type raises OptimizationStateError.

        Previously returned {} silently, but this masked bugs. Now we raise
        an error to surface corrupted context state immediately.
        """
        # Mock that we're in an active trial
        mock_get_trial_context.return_value = {"trial_id": "test_trial"}

        # A truthy non-dict, non-TraigentConfig value
        mock_get_config.return_value = "some_string"

        with pytest.raises(OptimizationStateError) as exc_info:
            get_trial_config()

        assert "invalid type: str" in str(exc_info.value)


class TestGetConfig:
    """Tests for the unified get_config helper."""

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions._get_context_config")
    @patch("traigent.api.functions.get_applied_config")
    def test_get_config_prefers_trial(
        self, mock_get_applied_config, mock_get_context_config, mock_get_trial_context
    ):
        """Trial context should win over applied config."""
        mock_get_trial_context.return_value = {"trial_id": "trial-1"}
        mock_get_context_config.return_value = {"model": "gpt-4"}
        mock_get_applied_config.return_value = {"model": "should_not_use"}

        result = get_config()

        assert result == {"model": "gpt-4"}
        assert result is not mock_get_context_config.return_value
        mock_get_applied_config.assert_not_called()

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions.get_applied_config")
    def test_get_config_uses_applied_config(
        self, mock_get_applied_config, mock_get_trial_context
    ):
        """When no trial is active, fall back to applied config."""
        mock_get_trial_context.return_value = None
        mock_get_applied_config.return_value = {"temperature": 0.2}

        result = get_config()

        assert result == {"temperature": 0.2}
        assert result is not mock_get_applied_config.return_value

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions.get_applied_config")
    def test_get_config_raises_without_any_config(
        self, mock_get_applied_config, mock_get_trial_context
    ):
        """No trial and no applied config should raise."""
        mock_get_trial_context.return_value = None
        mock_get_applied_config.return_value = None

        with pytest.raises(OptimizationStateError) as exc_info:
            get_config()

        assert "No config available" in str(exc_info.value)

    @patch("traigent.api.functions.get_trial_context")
    @patch("traigent.api.functions.get_applied_config")
    def test_get_config_raises_on_invalid_type(
        self, mock_get_applied_config, mock_get_trial_context
    ):
        """Invalid applied config types should surface as errors."""
        mock_get_trial_context.return_value = None
        mock_get_applied_config.return_value = "invalid"

        with pytest.raises(OptimizationStateError) as exc_info:
            get_config()

        assert "invalid type" in str(exc_info.value)


class TestGetCurrentConfigDeprecation:
    """Test that get_current_config shows deprecation warning."""

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_shows_deprecation_warning(self, mock_get_config):
        """Test that get_current_config emits ConfigAccessWarning."""
        mock_get_config.return_value = {"model": "gpt-4"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_current_config()

            # Check that warning was raised
            assert len(w) >= 1
            config_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, ConfigAccessWarning)
            ]
            assert len(config_warnings) == 1
            assert "deprecated" in str(config_warnings[0].message).lower()
            assert "get_config" in str(config_warnings[0].message)

        assert result == {"model": "gpt-4"}
