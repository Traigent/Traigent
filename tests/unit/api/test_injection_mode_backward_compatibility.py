"""Test backward compatibility for removed injection modes.

Both 'decorator' and 'attribute' injection modes have been removed in v2.x.
This module verifies that using these modes raises ConfigurationError with
migration guidance directing users to 'context' or 'seamless' modes.
"""

import pytest

from traigent.api.decorators import optimize
from traigent.config.types import InjectionMode
from traigent.utils.exceptions import ConfigurationError


class TestRemovedInjectionModes:
    """Test that removed injection modes raise ConfigurationError with migration guide."""

    def test_decorator_mode_raises_configuration_error(self):
        """Test that using 'decorator' injection mode raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="decorator",
            )
            def test_func(text: str) -> str:
                return text

        error_message = str(exc_info.value)
        assert "decorator" in error_message
        assert "removed" in error_message.lower()
        assert "context" in error_message  # Migration suggestion
        assert "seamless" in error_message  # Alternative suggestion

    def test_attribute_mode_raises_configuration_error(self):
        """Test that using 'attribute' injection mode raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="attribute",
            )
            def test_func(text: str) -> str:
                return text

        error_message = str(exc_info.value)
        assert "attribute" in error_message
        assert "removed" in error_message.lower()
        assert "context" in error_message  # Migration suggestion
        assert "seamless" in error_message  # Alternative suggestion


class TestInjectionModeEnum:
    """Test that InjectionMode enum has correct values (without removed modes)."""

    def test_injection_mode_enum_values(self):
        """Test that InjectionMode enum has only supported values."""
        assert InjectionMode.CONTEXT.value == "context"
        assert InjectionMode.PARAMETER.value == "parameter"
        assert InjectionMode.SEAMLESS.value == "seamless"

        # Removed modes should not be in the enum
        with pytest.raises(ValueError):
            InjectionMode("attribute")
        with pytest.raises(ValueError):
            InjectionMode("decorator")

    def test_enum_values_work_directly(self):
        """Test that enum values can be used directly in decorator."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode=InjectionMode.CONTEXT,
        )
        def test_func(text: str) -> str:
            return text

        assert hasattr(test_func, "optimize")
        assert hasattr(test_func, "injection_mode")


class TestSupportedInjectionModes:
    """Test that supported injection modes still work correctly."""

    def test_context_mode_works(self):
        """Test that context injection mode works correctly."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="context",
        )
        def test_func(text: str) -> str:
            return text

        assert hasattr(test_func, "optimize")

    def test_seamless_mode_works(self):
        """Test that seamless injection mode works correctly."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="seamless",
        )
        def test_func(text: str) -> str:
            model = "default"
            return f"{model}: {text}"

        assert hasattr(test_func, "optimize")

    def test_parameter_mode_works(self):
        """Test that parameter injection mode works correctly."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
        )
        def test_func(text: str, config) -> str:
            return text

        assert hasattr(test_func, "optimize")

    def test_all_supported_injection_modes_via_enum(self):
        """Test that all supported injection modes can be specified via enum."""

        # Context mode
        @optimize(
            configuration_space={"temp": [0.1, 0.5]},
            injection_mode=InjectionMode.CONTEXT,
        )
        def context_func(x: str) -> str:
            return x

        # Parameter mode
        @optimize(
            configuration_space={"temp": [0.1, 0.5]},
            injection_mode=InjectionMode.PARAMETER,
            config_param="config",
        )
        def param_func(x: str, config) -> str:
            return x

        # Seamless mode
        @optimize(
            configuration_space={"temp": [0.1, 0.5]},
            injection_mode=InjectionMode.SEAMLESS,
        )
        def seamless_func(x: str) -> str:
            return x

        # All should have optimize method
        assert all(
            hasattr(f, "optimize") for f in [context_func, param_func, seamless_func]
        )
