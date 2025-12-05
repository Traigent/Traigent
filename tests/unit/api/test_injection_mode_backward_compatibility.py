"""Test backward compatibility for injection mode renaming from 'decorator' to 'attribute'."""

import warnings

import pytest

from traigent.api.decorators import optimize
from traigent.config.types import InjectionMode
from traigent.core.optimized_function import OptimizedFunction


class TestInjectionModeBackwardCompatibility:
    """Test that 'decorator' injection mode still works but shows deprecation warning."""

    def test_decorator_mode_shows_deprecation_warning(self):
        """Test that using 'decorator' injection mode shows deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="decorator",  # Old name
            )
            def test_func(text: str) -> str:
                # Access config via function attribute
                config = getattr(test_func, "current_config", {})
                return f"{config.get('model', 'default')} response: {text}"

            # Should have gotten a deprecation warning
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()
            assert "attribute" in str(w[0].message)

            # Function should still work
            assert isinstance(test_func, OptimizedFunction)

    def test_decorator_mode_still_functions(self):
        """Test that 'decorator' mode still works correctly."""

        @optimize(
            configuration_space={"temperature": [0.1, 0.5, 0.9]},
            injection_mode="decorator",  # Old name
        )
        def test_func(query: str) -> str:
            config = getattr(test_func, "current_config", {})
            temp = config.get("temperature", 0.7)
            return f"Temperature: {temp}"

        # Set a config
        test_func.set_config({"temperature": 0.3})

        # Should work correctly
        result = test_func("test")
        assert "Temperature: 0.3" in result

    def test_attribute_mode_works_without_warning(self):
        """Test that using 'attribute' injection mode works without warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="attribute",  # New name
            )
            def test_func(text: str) -> str:
                config = getattr(test_func, "current_config", {})
                return f"{config.get('model', 'default')} response: {text}"

            # Should have no warnings
            assert len(w) == 0

            # Function should work
            assert isinstance(test_func, OptimizedFunction)

    def test_injection_mode_enum(self):
        """Test that InjectionMode enum has correct values."""
        assert InjectionMode.CONTEXT.value == "context"
        assert InjectionMode.PARAMETER.value == "parameter"
        assert InjectionMode.ATTRIBUTE.value == "attribute"
        assert InjectionMode.SEAMLESS.value == "seamless"

        # Ensure 'decorator' is not in the enum
        with pytest.raises(ValueError):
            InjectionMode("decorator")

    def test_enum_values_work_directly(self):
        """Test that enum values can be used directly in decorator."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode=InjectionMode.ATTRIBUTE,
        )
        def test_func(text: str) -> str:
            config = getattr(test_func, "current_config", {})
            return f"{config.get('model', 'default')} response: {text}"

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
        def param_func(x: str, config) -> str:
            return x

        # Attribute mode
        @optimize(
            configuration_space={"temp": [0.1, 0.5]},
            injection_mode=InjectionMode.ATTRIBUTE,
        )
        def attr_func(x: str) -> str:
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
            for f in [context_func, param_func, attr_func, seamless_func]
        )
