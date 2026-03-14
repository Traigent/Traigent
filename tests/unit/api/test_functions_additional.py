"""Additional comprehensive tests for traigent.api.functions module - targeting uncovered code."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from traigent.api.functions import (
    _GLOBAL_CONFIG,
    _apply_additional_overrides,
    _apply_config_settings,
    _configure_api_keys,
    _configure_backend_url,
    _configure_logging_settings,
    configure,
    get_api_key,
    get_current_config,
    get_global_parallel_config,
    get_optimization_insights,
    initialize,
    override_config,
    set_strategy,
)
from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.config.parallel import ParallelConfig
from traigent.config.types import TraigentConfig


class TestConfigureParallelConfig:
    """Test configure() with parallel_config parameter."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    def test_configure_parallel_config_dict(self):
        """Test configuring with parallel_config as dict."""
        result = configure(
            parallel_config={"thread_workers": 8, "trial_concurrency": 2}
        )

        assert result is True
        config = _GLOBAL_CONFIG.get("parallel_config")
        assert isinstance(config, ParallelConfig)
        assert config.thread_workers == 8
        assert config.trial_concurrency == 2

    def test_configure_parallel_config_object(self):
        """Test configuring with parallel_config as ParallelConfig object."""
        parallel_cfg = ParallelConfig(thread_workers=4, trial_concurrency=1)
        result = configure(parallel_config=parallel_cfg)

        assert result is True
        config = _GLOBAL_CONFIG.get("parallel_config")
        assert config.thread_workers == 4

    def test_configure_parallel_config_none(self):
        """Test configuring with parallel_config=None."""
        # Set initial config
        _GLOBAL_CONFIG["parallel_config"] = ParallelConfig(thread_workers=5)

        result = configure(parallel_config=None)

        assert result is True
        # Should remain unchanged
        assert _GLOBAL_CONFIG["parallel_config"].thread_workers == 5

    def test_configure_parallel_config_syncs_parallel_workers(self):
        """Test that parallel_config.thread_workers syncs to parallel_workers."""
        result = configure(parallel_config={"thread_workers": 10})

        assert result is True
        assert _GLOBAL_CONFIG["parallel_workers"] == 10

    def test_configure_parallel_config_merges_with_existing(self):
        """Test that parallel_config merges with existing config."""
        # Set initial config
        _GLOBAL_CONFIG["parallel_config"] = ParallelConfig(thread_workers=5)

        # Update with new values
        result = configure(parallel_config={"trial_concurrency": 2})

        assert result is True
        config = _GLOBAL_CONFIG.get("parallel_config")
        assert config.trial_concurrency == 2


class TestConfigureObjectives:
    """Test configure() with objectives parameter."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    def test_configure_objectives_list(self):
        """Test configuring with objectives as list of strings."""
        result = configure(objectives=["accuracy", "cost", "latency"])

        assert result is True
        assert "objective_schema" in _GLOBAL_CONFIG
        assert "objectives" in _GLOBAL_CONFIG
        assert _GLOBAL_CONFIG["objectives"] == ["accuracy", "cost", "latency"]

    def test_configure_objectives_schema(self):
        """Test configuring with ObjectiveSchema."""
        with (
            patch("traigent.core.objectives.normalize_objectives") as mock_normalize,
            patch(
                "traigent.core.objectives.schema_to_objective_names"
            ) as mock_schema_to_names,
        ):

            mock_schema = Mock()
            mock_normalize.return_value = mock_schema
            mock_schema_to_names.return_value = ["obj1", "obj2"]

            result = configure(objectives=mock_schema)

            assert result is True
            assert _GLOBAL_CONFIG["objective_schema"] == mock_schema
            assert _GLOBAL_CONFIG["objectives"] == ["obj1", "obj2"]


class TestConfigureFeatureFlags:
    """Test configure() with feature_flags parameter."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    @patch("traigent.api.functions.flag_registry")
    def test_configure_feature_flags(self, mock_registry):
        """Test configuring with feature_flags."""
        flags = {"experimental_feature": True, "beta_optimizer": False}
        result = configure(feature_flags=flags)

        assert result is True
        mock_registry.apply_config.assert_called_once_with(flags)

    def test_configure_feature_flags_invalid_type(self):
        """Test configuring with invalid feature_flags type."""
        with pytest.raises(ValueError, match="Expected dict"):
            configure(feature_flags="not_a_dict")


class TestInitializeFunction:
    """Test initialize() function."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    @patch("traigent.api.functions._configure_api_keys")
    @patch("traigent.api.functions._configure_backend_url")
    @patch("traigent.api.functions._apply_config_settings")
    @patch("traigent.api.functions._apply_additional_overrides")
    @patch("traigent.api.functions._configure_logging_settings")
    def test_initialize_with_explicit_api_key(
        self,
        mock_logging,
        mock_overrides,
        mock_config_settings,
        mock_backend_url,
        mock_api_keys,
    ):
        """Test initialize with explicit API key."""
        result = initialize(api_key="test-key-123", api_url="http://localhost:5000")

        assert result is True
        mock_api_keys.assert_called_once()
        mock_backend_url.assert_called_once()

    @patch("traigent.api.functions._configure_api_keys")
    @patch("traigent.api.functions._configure_backend_url")
    @patch("traigent.api.functions._apply_config_settings")
    @patch("traigent.api.functions._apply_additional_overrides")
    @patch("traigent.api.functions._configure_logging_settings")
    def test_initialize_with_config(
        self,
        mock_logging,
        mock_overrides,
        mock_config_settings,
        mock_backend_url,
        mock_api_keys,
    ):
        """Test initialize with TraigentConfig."""
        config = TraigentConfig.edge_analytics_mode()
        result = initialize(config=config)

        assert result is True
        mock_config_settings.assert_called_once_with(config)

    @patch("traigent.api.functions._configure_api_keys")
    @patch("traigent.api.functions._configure_backend_url")
    @patch("traigent.api.functions._apply_config_settings")
    @patch("traigent.api.functions._apply_additional_overrides")
    @patch("traigent.api.functions._configure_logging_settings")
    def test_initialize_with_kwargs(
        self,
        mock_logging,
        mock_overrides,
        mock_config_settings,
        mock_backend_url,
        mock_api_keys,
    ):
        """Test initialize with additional kwargs."""
        result = initialize(custom_param="value", another_setting=123)

        assert result is True
        mock_overrides.assert_called_once()
        call_kwargs = mock_overrides.call_args[0][0]
        assert call_kwargs["custom_param"] == "value"
        assert call_kwargs["another_setting"] == 123


class TestConfigureApiKeys:
    """Test _configure_api_keys helper function."""

    @patch("traigent.api.functions._API_KEY_MANAGER")
    @patch("traigent.api.functions.logger")
    def test_configure_api_keys_with_explicit_key(self, mock_logger, mock_manager):
        """Test _configure_api_keys with explicit API key."""
        mock_backend_config = Mock()

        _configure_api_keys("explicit-key", mock_backend_config)

        mock_manager.set_api_key.assert_called_once_with(
            "traigent", "explicit-key", source="initialization"
        )
        mock_logger.info.assert_called_with("Traigent API key configured")

    @patch("traigent.api.functions._API_KEY_MANAGER")
    @patch("traigent.api.functions.logger")
    def test_configure_api_keys_from_environment(self, mock_logger, mock_manager):
        """Test _configure_api_keys from environment."""
        mock_backend_config = Mock()
        mock_backend_config.get_api_key.return_value = "env-key"

        _configure_api_keys(None, mock_backend_config)

        mock_manager.set_api_key.assert_called_once_with(
            "traigent", "env-key", source="environment"
        )
        mock_logger.info.assert_called_with(
            "Traigent API key configured from environment"
        )

    @patch("traigent.api.functions._API_KEY_MANAGER")
    def test_configure_api_keys_no_key_available(self, mock_manager):
        """Test _configure_api_keys when no key is available."""
        mock_backend_config = Mock()
        mock_backend_config.get_api_key.return_value = None

        # Should not raise, just not set any key
        _configure_api_keys(None, mock_backend_config)

        mock_manager.set_api_key.assert_not_called()


class TestConfigureBackendUrl:
    """Test _configure_backend_url helper function."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    @patch("traigent.api.functions.logger")
    def test_configure_backend_url_with_explicit_url(self, mock_logger):
        """Test _configure_backend_url with explicit URL."""
        mock_backend_config = Mock()
        mock_backend_config.split_api_url.return_value = (
            "http://localhost:5000",
            "/api/v1",
        )
        mock_backend_config.get_default_api_path.return_value = "/api"

        _configure_backend_url("http://localhost:5000/api/v1", mock_backend_config)

        assert _GLOBAL_CONFIG["traigent_api_url"] == "http://localhost:5000/api/v1"
        mock_logger.info.assert_called_with(
            "Traigent backend API set to: http://localhost:5000/api/v1"
        )

    @patch("traigent.api.functions.logger")
    def test_configure_backend_url_with_origin_only(self, mock_logger):
        """Test _configure_backend_url with origin but no path."""
        mock_backend_config = Mock()
        mock_backend_config.split_api_url.return_value = ("http://localhost:5000", None)
        mock_backend_config.get_default_api_path.return_value = "/api"

        _configure_backend_url("http://localhost:5000", mock_backend_config)

        assert _GLOBAL_CONFIG["traigent_api_url"] == "http://localhost:5000/api"

    @patch("traigent.api.functions.logger")
    def test_configure_backend_url_no_origin(self, mock_logger):
        """Test _configure_backend_url when split returns no origin."""
        mock_backend_config = Mock()
        mock_backend_config.split_api_url.return_value = (None, None)

        _configure_backend_url("relative/path", mock_backend_config)

        assert _GLOBAL_CONFIG["traigent_api_url"] == "relative/path"

    @patch("traigent.api.functions.logger")
    def test_configure_backend_url_from_environment(self, mock_logger):
        """Test _configure_backend_url from environment."""
        mock_backend_config = Mock()
        mock_backend_config.get_backend_api_url.return_value = "http://env-backend:8000"

        _configure_backend_url(None, mock_backend_config)

        assert _GLOBAL_CONFIG["traigent_api_url"] == "http://env-backend:8000"
        mock_logger.info.assert_called_with(
            "Traigent backend API configured: http://env-backend:8000"
        )


class TestApplyConfigSettings:
    """Test _apply_config_settings helper function."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    @patch("traigent.api.functions.logger")
    def test_apply_config_settings_execution_mode(self, mock_logger):
        """Test _apply_config_settings with execution_mode."""
        config = TraigentConfig(execution_mode="cloud")

        _apply_config_settings(config)

        assert _GLOBAL_CONFIG["execution_mode"] == "cloud"
        assert _GLOBAL_CONFIG["default_storage_backend"] == "cloud"

    @patch("traigent.api.functions.logger")
    def test_apply_config_settings_edge_analytics(self, mock_logger):
        """Test _apply_config_settings with edge_analytics mode."""
        config = TraigentConfig.edge_analytics_mode()

        _apply_config_settings(config)

        assert _GLOBAL_CONFIG["default_storage_backend"] == "edge_analytics"

    @patch("traigent.api.functions.logger")
    def test_apply_config_settings_local_storage_path(self, mock_logger):
        """Test _apply_config_settings with local_storage_path."""
        config = TraigentConfig(local_storage_path="/custom/path")

        _apply_config_settings(config)

        assert _GLOBAL_CONFIG["local_storage_path"] == "/custom/path"

    @patch("traigent.api.functions.logger")
    def test_apply_config_settings_minimal_logging(self, mock_logger):
        """Test _apply_config_settings with minimal_logging."""
        config = TraigentConfig(minimal_logging=True)

        _apply_config_settings(config)

        assert _GLOBAL_CONFIG["minimal_logging"] is True

    @patch("traigent.api.functions.logger")
    def test_apply_config_settings_auto_sync(self, mock_logger):
        """Test _apply_config_settings with auto_sync."""
        config = TraigentConfig(auto_sync=False)

        _apply_config_settings(config)

        assert _GLOBAL_CONFIG["auto_sync"] is False


class TestApplyAdditionalOverrides:
    """Test _apply_additional_overrides helper function."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    def test_apply_additional_overrides_empty(self):
        """Test _apply_additional_overrides with empty dict."""
        _apply_additional_overrides({})

        # Should not change anything
        assert _GLOBAL_CONFIG == self.original_config

    def test_apply_additional_overrides_single_value(self):
        """Test _apply_additional_overrides with single override."""
        _apply_additional_overrides({"custom_setting": "value123"})

        assert _GLOBAL_CONFIG["custom_setting"] == "value123"

    def test_apply_additional_overrides_multiple_values(self):
        """Test _apply_additional_overrides with multiple overrides."""
        _apply_additional_overrides(
            {"setting1": 100, "setting2": "test", "setting3": True}
        )

        assert _GLOBAL_CONFIG["setting1"] == 100
        assert _GLOBAL_CONFIG["setting2"] == "test"
        assert _GLOBAL_CONFIG["setting3"] is True


class TestConfigureLoggingSettings:
    """Test _configure_logging_settings helper function."""

    @patch("traigent.api.functions.setup_logging")
    def test_configure_logging_settings_minimal(self, mock_setup_logging):
        """Test _configure_logging_settings with minimal_logging=True."""
        config = TraigentConfig(minimal_logging=True)

        _configure_logging_settings(config)

        mock_setup_logging.assert_called_once_with(level="WARNING")

    @patch("traigent.api.functions.setup_logging")
    def test_configure_logging_settings_normal(self, mock_setup_logging):
        """Test _configure_logging_settings with minimal_logging=False."""
        config = TraigentConfig(minimal_logging=False)
        _GLOBAL_CONFIG["logging_level"] = "DEBUG"

        _configure_logging_settings(config)

        mock_setup_logging.assert_called_once_with(level="DEBUG")

    @patch("traigent.api.functions.setup_logging")
    def test_configure_logging_settings_no_config(self, mock_setup_logging):
        """Test _configure_logging_settings with no config."""
        _GLOBAL_CONFIG["logging_level"] = "ERROR"

        _configure_logging_settings(None)

        mock_setup_logging.assert_called_once_with(level="ERROR")

    @patch("traigent.api.functions.setup_logging")
    def test_configure_logging_settings_default_level(self, mock_setup_logging):
        """Test _configure_logging_settings with default level."""
        # Remove logging_level from config
        if "logging_level" in _GLOBAL_CONFIG:
            del _GLOBAL_CONFIG["logging_level"]

        _configure_logging_settings(None)

        mock_setup_logging.assert_called_once_with(level="INFO")


class TestGetApiKey:
    """Test get_api_key function."""

    @patch("traigent.api.functions._API_KEY_MANAGER")
    def test_get_api_key_exists(self, mock_manager):
        """Test get_api_key when key exists."""
        mock_manager.get_api_key.return_value = "test-key-123"

        result = get_api_key("openai")

        assert result == "test-key-123"
        mock_manager.get_api_key.assert_called_once_with("openai")

    @patch("traigent.api.functions._API_KEY_MANAGER")
    def test_get_api_key_not_found(self, mock_manager):
        """Test get_api_key when key doesn't exist."""
        mock_manager.get_api_key.return_value = None

        result = get_api_key("nonexistent")

        assert result is None


class TestGetCurrentConfig:
    """Test get_current_config function."""

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_traigent_config(self, mock_get_config):
        """Test get_current_config with TraigentConfig object."""
        config = TraigentConfig(execution_mode="cloud")
        mock_get_config.return_value = config

        result = get_current_config()

        assert isinstance(result, dict)
        # TraigentConfig.to_dict() returns a dict

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_dict(self, mock_get_config):
        """Test get_current_config with dict."""
        mock_get_config.return_value = {"model": "gpt-4", "temperature": 0.7}

        result = get_current_config()

        assert result == {"model": "gpt-4", "temperature": 0.7}

    @patch("traigent.api.functions._get_context_config")
    def test_get_current_config_unexpected_type(self, mock_get_config):
        """Test get_current_config with unexpected type."""
        mock_get_config.return_value = "unexpected_string"

        result = get_current_config()

        assert result == {}


class TestOverrideConfig:
    """Test override_config function."""

    def test_override_config_objectives(self):
        """Test override_config with objectives."""
        result = override_config(objectives=["accuracy", "cost"])

        assert result["objectives"] == ["accuracy", "cost"]

    def test_override_config_configuration_space(self):
        """Test override_config with configuration_space."""
        result = override_config(configuration_space={"model": ["gpt-4"]})

        assert result["configuration_space"] == {"model": ["gpt-4"]}

    def test_override_config_constraints(self):
        """Test override_config with constraints."""

        def constraint(x):
            return x["temperature"] < 1.0

        result = override_config(constraints=[constraint])

        assert result["constraints"] == [constraint]

    def test_override_config_max_trials_valid(self):
        """Test override_config with valid max_trials."""
        result = override_config(max_trials=50)

        assert result["max_trials"] == 50

    def test_override_config_max_trials_invalid(self):
        """Test override_config with invalid max_trials."""
        result = override_config(max_trials=0)
        assert result["max_trials"] == 0

        with pytest.raises(ValueError, match="max_trials must be non-negative"):
            override_config(max_trials=-5)

    def test_override_config_timeout_valid(self):
        """Test override_config with valid timeout."""
        result = override_config(timeout=300)

        assert result["timeout"] == 300

    def test_override_config_max_total_examples_valid(self):
        """Test override_config with valid max_total_examples."""
        result = override_config(max_total_examples=500, samples_include_pruned=True)

        assert result["max_total_examples"] == 500
        assert result["samples_include_pruned"] is True

    def test_override_config_timeout_invalid(self):
        """Test override_config with invalid timeout."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            override_config(timeout=0)

        with pytest.raises(ValueError, match="timeout must be > 0"):
            override_config(timeout=-10)

    def test_override_config_max_total_examples_invalid(self):
        """Test override_config with invalid max_total_examples."""
        with pytest.raises(ValueError):
            override_config(max_total_examples=0)

    def test_override_config_multiple_params(self):
        """Test override_config with multiple parameters."""
        result = override_config(
            objectives=["latency"],
            configuration_space={"model": ["gpt-3.5"]},
            max_trials=20,
            timeout=120,
        )

        assert result["objectives"] == ["latency"]
        assert result["configuration_space"] == {"model": ["gpt-3.5"]}
        assert result["max_trials"] == 20
        assert result["timeout"] == 120

    def test_override_config_empty(self):
        """Test override_config with no parameters."""
        result = override_config()

        assert result == {}


class TestSetStrategy:
    """Test set_strategy function."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    def test_set_strategy_bayesian(self):
        """Test set_strategy with bayesian algorithm."""
        strategy = set_strategy(algorithm="bayesian")

        assert strategy.algorithm == "bayesian"
        assert isinstance(strategy.algorithm_config, dict)
        assert strategy.parallel_workers == 1  # Default from global config

    def test_set_strategy_with_config(self):
        """Test set_strategy with algorithm config."""
        config = {"acquisition_function": "ei", "initial_random_samples": 10}
        strategy = set_strategy(algorithm="bayesian", algorithm_config=config)

        assert strategy.algorithm_config == config

    def test_set_strategy_with_parallel_workers(self):
        """Test set_strategy with custom parallel workers."""
        strategy = set_strategy(algorithm="random", parallel_workers=8)

        assert strategy.parallel_workers == 8

    def test_set_strategy_with_resource_limits(self):
        """Test set_strategy with resource limits."""
        limits = {"max_memory": "4GB", "max_time": 3600}
        strategy = set_strategy(algorithm="grid", resource_limits=limits)

        assert strategy.resource_limits == limits

    def test_set_strategy_invalid_algorithm(self):
        """Test set_strategy with invalid algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            set_strategy(algorithm="nonexistent_algo")

    def test_set_strategy_uses_global_workers(self):
        """Test set_strategy uses global parallel_workers when not specified."""
        _GLOBAL_CONFIG["parallel_workers"] = 6

        strategy = set_strategy(algorithm="random")

        assert strategy.parallel_workers == 6


class TestGetOptimizationInsights:
    """Test get_optimization_insights function."""

    @patch("traigent.api.functions._get_optimization_insights")
    def test_get_optimization_insights(self, mock_insights):
        """Test get_optimization_insights delegates to utils function."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "gpt-4"},
                metrics={"accuracy": 0.9},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={"model": "gpt-4"},
            best_score=0.9,
            optimization_id="opt_001",
            duration=5.0,
            convergence_info={},
            status="completed",
            objectives=["accuracy"],
            algorithm="bayesian",
            timestamp=datetime.now(),
        )

        mock_insights.return_value = {"top_configurations": [], "recommendations": []}

        insights = get_optimization_insights(result)

        mock_insights.assert_called_once_with(result)
        assert "top_configurations" in insights


class TestGetGlobalParallelConfig:
    """Test get_global_parallel_config function."""

    def setup_method(self):
        """Reset global config before each test."""
        self.original_config = _GLOBAL_CONFIG.copy()

    def teardown_method(self):
        """Restore original config after each test."""
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(self.original_config)

    def test_get_global_parallel_config_when_set(self):
        """Test get_global_parallel_config when already set."""
        parallel_cfg = ParallelConfig(thread_workers=8)
        _GLOBAL_CONFIG["parallel_config"] = parallel_cfg

        result = get_global_parallel_config()

        assert result.thread_workers == 8

    def test_get_global_parallel_config_coercion(self):
        """Test get_global_parallel_config coerces dict to ParallelConfig."""
        _GLOBAL_CONFIG["parallel_config"] = {
            "thread_workers": 6,
            "trial_concurrency": 2,
        }

        result = get_global_parallel_config()

        assert isinstance(result, ParallelConfig)
        assert result.thread_workers == 6
        assert result.trial_concurrency == 2

    def test_get_global_parallel_config_default(self):
        """Test get_global_parallel_config returns default when None."""
        _GLOBAL_CONFIG["parallel_config"] = None

        result = get_global_parallel_config()

        assert isinstance(result, ParallelConfig)
        # Default values
        assert result.thread_workers is None or result.thread_workers == 1

    def test_get_global_parallel_config_stores_coerced(self):
        """Test get_global_parallel_config stores coerced value."""
        _GLOBAL_CONFIG["parallel_config"] = {"thread_workers": 4}

        get_global_parallel_config()

        # Should store the coerced ParallelConfig
        assert isinstance(_GLOBAL_CONFIG["parallel_config"], ParallelConfig)
