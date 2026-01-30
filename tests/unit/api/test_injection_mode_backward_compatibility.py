"""Test backward compatibility for injection modes after ATTRIBUTE removal.

Note: The ATTRIBUTE injection mode was removed in v2.x due to thread-safety issues
with parallel trials. This test file validates that the remaining modes work correctly
and that attempting to use removed modes raises appropriate errors.
"""

import warnings

import pytest

from traigent.api.decorators import optimize
from traigent.config.types import InjectionMode
from traigent.core.optimized_function import OptimizedFunction


class TestInjectionModeBackwardCompatibility:
    """Test that current injection modes work correctly after ATTRIBUTE removal."""

    def test_context_mode_works(self):
        """Test that context injection mode works correctly."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="context",
            )
            def test_func(text: str) -> str:
                return f"response: {text}"

            # Should have no warnings for valid mode
            deprecation_warnings = [
                warning for warning in w if "deprecated" in str(warning.message).lower()
            ]
            assert len(deprecation_warnings) == 0

            # Function should work
            assert isinstance(test_func, OptimizedFunction)

    def test_parameter_mode_works(self):
        """Test that parameter injection mode works correctly."""

        @optimize(
            configuration_space={"temperature": [0.1, 0.5, 0.9]},
            injection_mode="parameter",
            config_param="config",
        )
        def test_func(query: str, config=None) -> str:
            temp = config.get("temperature", 0.7) if config else 0.7
            return f"Temperature: {temp}"

        assert isinstance(test_func, OptimizedFunction)

    def test_seamless_mode_works(self):
        """Test that seamless injection mode works correctly."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="seamless",
            )
            def test_func(text: str) -> str:
                return f"response: {text}"

            # Should have no warnings for valid mode
            deprecation_warnings = [
                warning for warning in w if "deprecated" in str(warning.message).lower()
            ]
            assert len(deprecation_warnings) == 0

            # Function should work
            assert isinstance(test_func, OptimizedFunction)

    def test_injection_mode_enum(self):
        """Test that InjectionMode enum has correct values."""
        assert InjectionMode.CONTEXT.value == "context"
        assert InjectionMode.PARAMETER.value == "parameter"
        assert InjectionMode.SEAMLESS.value == "seamless"

        # ATTRIBUTE was removed - ensure it doesn't exist
        assert not hasattr(InjectionMode, "ATTRIBUTE")

        # Ensure old names are not in the enum
        with pytest.raises(ValueError):
            InjectionMode("attribute")

        with pytest.raises(ValueError):
            InjectionMode("decorator")

    def test_enum_values_work_directly(self):
        """Test that enum values can be used directly in decorator."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode=InjectionMode.SEAMLESS,
        )
        def test_func(text: str) -> str:
            return f"response: {text}"

        assert isinstance(test_func, OptimizedFunction)
        # Check that the internal injection mode was properly set
        assert hasattr(test_func, "injection_mode")

    def test_all_injection_modes_via_enum(self):
        """Test that all injection modes can be specified via enum."""

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
        def param_func(x: str, config=None) -> str:
            return x

        # Seamless mode
        @optimize(
            configuration_space={"temp": [0.1, 0.5]},
            injection_mode=InjectionMode.SEAMLESS,
        )
        def seamless_func(x: str) -> str:
            return x

        # All should be OptimizedFunction instances
        assert all(
            isinstance(f, OptimizedFunction)
            for f in [context_func, param_func, seamless_func]
        )

    def test_invalid_injection_mode_raises_error(self):
        """Test that invalid injection modes raise appropriate errors."""
        from traigent.utils.exceptions import ConfigurationError

        with pytest.raises((ValueError, KeyError, ConfigurationError)):

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="attribute",  # Removed mode
            )
            def test_func(text: str) -> str:
                return text
