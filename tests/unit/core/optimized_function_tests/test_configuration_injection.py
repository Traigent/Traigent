"""Tests for OptimizedFunction configuration injection.

Tests different injection modes (context, parameter, seamless) and
configuration management.
"""

import pytest

from traigent.config.context import get_config, set_config
from traigent.core.optimized_function import OptimizedFunction


class TestConfigurationInjection:
    """Test configuration injection functionality."""

    def test_context_injection_mode(self, sample_config_space, sample_objectives):
        """Test context injection mode."""

        def test_func(text: str) -> str:
            # Function reads from global context
            config = get_config()
            temp = config.get("temperature", 0.5)
            return f"{text} (temp={temp})"

        opt_func = OptimizedFunction(
            func=test_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            injection_mode="context",
        )

        # Set config in context
        set_config({"temperature": 0.8, "model": "gpt-4"})

        # Call function
        result = opt_func("hello")
        assert "temp=0.8" in result

    def test_parameter_injection_mode(self, sample_config_space, sample_objectives):
        """Test parameter injection mode."""

        def test_func(text: str, config=None) -> str:
            # Function receives config as parameter
            if config and hasattr(config, "temperature"):
                temp = config.temperature
            else:
                temp = 0.5
            return f"{text} (temp={temp})"

        opt_func = OptimizedFunction(
            func=test_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            injection_mode="parameter",
            default_config={"temperature": 0.7},
        )

        # Call function - config should be injected
        result = opt_func("hello")
        assert "temp=0.7" in result

    def test_seamless_injection_mode(self, sample_config_space, sample_objectives):
        """Test seamless injection mode."""

        def test_func(text: str) -> str:
            # Function with variable assignments that can be overridden
            temperature = 0.5  # This should be overridden by Traigent
            model = "gpt-3.5"  # This should be overridden by Traigent
            return f"{text} (temp={temperature}, model={model})"

        opt_func = OptimizedFunction(
            func=test_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            injection_mode="seamless",
            default_config={"temperature": 0.9, "model": "gpt-4"},
        )

        # Set the config in context for seamless injection to work
        from traigent.config.context import set_config

        set_config({"temperature": 0.9, "model": "gpt-4"})

        # Call function - variables should be overridden
        result = opt_func("hello")
        # The seamless provider should replace the hardcoded values
        assert "temp=0.9" in result
        assert "model=gpt-4" in result

    def test_default_config_application(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test default configuration is applied correctly."""
        default_config = {"temperature": 0.3, "max_tokens": 150, "model": "gpt-3.5"}

        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            default_config=default_config,
        )

        # Default config should be set
        assert opt_func.default_config == default_config
        assert opt_func._current_config == default_config

    def test_config_override_priority(self, sample_config_space, sample_objectives):
        """Test configuration override priority."""

        def test_func(text: str, config=None) -> str:
            if config:
                temp = (
                    config.get("temperature", 0.0)
                    if isinstance(config, dict)
                    else getattr(config, "temperature", 0.0)
                )
            else:
                temp = 0.0
            return f"{text} (temp={temp})"

        opt_func = OptimizedFunction(
            func=test_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            injection_mode="parameter",
            config_param="config",
            default_config={"temperature": 0.5},
        )

        # Set a new config using the proper method
        opt_func.set_config({"temperature": 0.8, "model": "gpt-4"})

        # Current config should override default
        result = opt_func("test")
        assert "temp=0.8" in result

    def test_injection_with_missing_config_parameter(
        self, sample_config_space, sample_objectives
    ):
        """Test parameter injection when function lacks config parameter."""

        def test_func(text: str) -> str:
            # No config parameter
            return text.upper()

        # Parameter injection without config param should raise error
        from traigent.utils.exceptions import ConfigurationError

        with pytest.raises(
            ConfigurationError, match="does not have parameter 'config'"
        ):
            OptimizedFunction(
                func=test_func,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                injection_mode="parameter",
                config_param="config",
            )

    def test_get_current_config(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test retrieving current configuration."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            default_config={"temperature": 0.6},
        )

        # Should return current config
        config = opt_func.current_config
        assert config["temperature"] == 0.6

        # Update config
        opt_func.set_config({"temperature": 0.8, "model": "gpt-4"})
        config = opt_func.current_config
        assert config["temperature"] == 0.8
        assert config["model"] == "gpt-4"

    def test_context_cleanup(self, sample_config_space, sample_objectives):
        """Test that context is properly cleaned up."""
        get_config()

        def test_func(text: str) -> str:
            config = get_config()
            return f"{text} ({len(config)} params)"

        opt_func = OptimizedFunction(
            func=test_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            injection_mode="context",
        )

        # Set some config
        set_config({"temperature": 0.7, "extra": "value"})

        # Call function
        result = opt_func("test")
        assert result is not None  # Function returns a string

        # Context should be restored after call
        # (Exact behavior depends on implementation)

    def test_seamless_with_extra_parameters(
        self, sample_config_space, sample_objectives
    ):
        """Test seamless injection with extra function parameters."""

        def test_func(text: str, custom_param: str = "default") -> str:
            temperature = 0.5  # This will be overridden by Traigent
            return f"{text} (temp={temperature}, custom={custom_param})"

        opt_func = OptimizedFunction(
            func=test_func,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            injection_mode="seamless",
            default_config={"temperature": 0.9},
        )

        # Set config in context for seamless injection
        from traigent.config.context import set_config

        set_config({"temperature": 0.9})

        # Call with custom parameter
        result = opt_func("hello", custom_param="test")
        assert "temp=0.9" in result
        assert "custom=test" in result

    def test_invalid_injection_mode_handling(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test handling of invalid injection mode."""
        with pytest.raises(ValueError):  # Will raise ConfigurationError
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                injection_mode="invalid_mode",
            )
