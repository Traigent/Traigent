"""
Unit tests for configuration builder module.
"""

from unittest.mock import patch

import pytest

from traigent.api.config_builder import (
    ConfigurationBuilder,
    build_optimize_configuration,
    clear_global_config,
    get_global_config,
    update_global_config,
)
from traigent.api.parameter_validator import OptimizeParameters
from traigent.config.types import InjectionMode
from traigent.core.objectives import ObjectiveSchema
from traigent.utils.exceptions import ConfigurationError


class TestConfigurationBuilder:
    """Test ConfigurationBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_global_config()
        self.builder = ConfigurationBuilder()

    def test_resolve_execution_mode_default(self):
        """Test execution mode resolution with default values."""
        # Mock global config with execution mode
        self.builder.global_config = {"execution_mode": "privacy"}

        with patch(
            "traigent.api.config_builder.get_optimize_default", return_value="cloud"
        ):
            result = self.builder._resolve_execution_mode("cloud", "cloud")
            assert result == "privacy"  # Should use global config

    def test_resolve_execution_mode_explicit(self):
        """Test execution mode resolution with explicit values."""
        self.builder.global_config = {"execution_mode": "privacy"}

        result = self.builder._resolve_execution_mode(
            "edge_analytics", "edge_analytics"
        )
        assert result == "privacy"  # Matches global config when value equals default

    def test_resolve_injection_mode_parameter_valid(self):
        """Test injection mode resolution for parameter mode."""
        result = self.builder._resolve_injection_mode(InjectionMode.PARAMETER, "config")
        assert result == InjectionMode.PARAMETER

    def test_resolve_injection_mode_parameter_missing(self):
        """Test injection mode resolution with missing config_param."""
        with pytest.raises(ConfigurationError) as exc_info:
            self.builder._resolve_injection_mode(InjectionMode.PARAMETER, None)
        assert "config_param must be specified" in str(exc_info.value)

    @patch("traigent.api.config_builder.logger")
    def test_resolve_injection_mode_warning(self, mock_logger):
        """Test injection mode resolution with unnecessary config_param."""
        result = self.builder._resolve_injection_mode(
            InjectionMode.CONTEXT, "unused_param"
        )
        assert result == InjectionMode.CONTEXT
        mock_logger.warning.assert_called()

    def test_build_configuration_invalid_injection_mode_type(self):
        """Builder should raise when injection mode has unexpected type."""
        params = OptimizeParameters(injection_mode=123)  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError) as exc_info:
            self.builder.build_configuration(params, "cloud")

        assert "injection_mode must be a str or InjectionMode" in str(exc_info.value)

    def test_resolve_privacy_settings_explicit(self):
        """Test privacy settings with explicit value."""
        result = self.builder._resolve_privacy_settings(True, "edge_analytics")
        assert result is True

        result = self.builder._resolve_privacy_settings(False, "privacy")
        assert result is False

    def test_resolve_privacy_settings_auto_privacy(self):
        """Test automatic privacy enabling for privacy mode."""
        result = self.builder._resolve_privacy_settings(None, "privacy")
        assert result is True

    def test_resolve_privacy_settings_no_auto(self):
        """Test privacy settings without auto-enabling."""
        result = self.builder._resolve_privacy_settings(None, "edge_analytics")
        assert result is None

    def test_build_configuration_complete(self):
        """Test complete configuration building."""
        params = OptimizeParameters(
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            execution_mode="edge_analytics",
            injection_mode=InjectionMode.CONTEXT,
            privacy_enabled=True,
            kwargs={"parallel_config": {"example_concurrency": 5}},
        )

        config = self.builder.build_configuration(params, "edge_analytics")

        assert config["eval_dataset"] == "test.jsonl"
        assert isinstance(config["objectives"], ObjectiveSchema)
        assert [obj.name for obj in config["objectives"].objectives] == ["accuracy"]
        assert config["execution_mode"] == "edge_analytics"
        assert config["injection_mode"] == InjectionMode.CONTEXT
        assert config["parallel_config"].example_concurrency == 5
        assert config["privacy_enabled"] is True
        assert "func" in config

    def test_update_global_config(self):
        """Test global configuration updates."""
        self.builder.update_global_config(execution_mode="privacy", debug=True)

        config = self.builder.get_global_config()
        assert config["execution_mode"] == "privacy"
        assert config["debug"] is True

    def test_clear_global_config(self):
        """Test global configuration clearing."""
        self.builder.update_global_config(test="value")
        assert len(self.builder.get_global_config()) > 0

        self.builder.clear_global_config()
        assert len(self.builder.get_global_config()) == 0


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_global_config()

    def test_build_optimize_configuration(self):
        """Test configuration building function."""
        params = OptimizeParameters(eval_dataset="test.jsonl", execution_mode="cloud")

        config = build_optimize_configuration(params, "cloud")

        assert config["eval_dataset"] == "test.jsonl"
        assert config["execution_mode"] == "cloud"
        assert "func" in config
        assert config["objectives"] is None

    def test_global_config_functions(self):
        """Test global configuration management functions."""
        # Test update
        update_global_config(test_param="value", debug=True)

        config = get_global_config()
        assert config["test_param"] == "value"
        assert config["debug"] is True

        # Test clear
        clear_global_config()
        config = get_global_config()
        assert len(config) == 0


@pytest.mark.integration
class TestConfigurationBuilderIntegration:
    """Integration tests for configuration builder."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_global_config()

    def test_real_world_configuration(self):
        """Test configuration building with realistic parameters."""
        # Set up global configuration
        update_global_config(execution_mode="privacy", debug=True)

        # Create parameters
        params = OptimizeParameters(
            eval_dataset=["test1.jsonl", "test2.jsonl"],
            objectives=["accuracy", "cost"],
            configuration_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": (0.1, 1.0),
            },
            constraints=[lambda config: config.get("temperature", 0) < 0.9],
            injection_mode="parameter",
            config_param="llm_config",
            kwargs={
                "parallel_config": {
                    "example_concurrency": 10,
                    "trial_concurrency": 3,
                }
            },
        )

        # Build configuration
        config = build_optimize_configuration(params, "cloud")

        # Verify configuration
        assert config["eval_dataset"] == ["test1.jsonl", "test2.jsonl"]
        assert isinstance(config["objectives"], ObjectiveSchema)
        assert [obj.name for obj in config["objectives"].objectives] == [
            "accuracy",
            "cost",
        ]
        assert "model" in config["configuration_space"]
        assert len(config["constraints"]) == 1
        assert config["injection_mode"] == InjectionMode.PARAMETER
        assert config["config_param"] == "llm_config"
        assert config["parallel_config"].example_concurrency == 10
        assert config["parallel_config"].trial_concurrency == 3
        assert config["execution_mode"] == "privacy"  # From global config
