"""Test backward compatibility - injection mode 'attribute' was removed in v2.x.

Note: The 'decorator' alias and 'attribute' injection mode were both removed in v2.x.
These tests verify that using removed modes raises appropriate errors.
"""

import pytest

from traigent.api.decorators import optimize
from traigent.config.types import InjectionMode
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError


class TestInjectionModeBackwardCompatibility:
    """Test that removed injection modes raise appropriate errors."""

    def test_decorator_mode_raises_error(self):
        """Test that using 'decorator' injection mode raises error (removed in v2.x)."""
        with pytest.raises(ConfigurationError, match="removed"):

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="decorator",  # Removed - was alias for 'attribute'
            )
            def test_func(text: str) -> str:
                return text

    def test_attribute_mode_raises_error(self):
        """Test that using 'attribute' injection mode raises error (removed in v2.x)."""
        with pytest.raises(ConfigurationError, match="removed"):

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="attribute",  # Removed in v2.x
            )
            def test_func(text: str) -> str:
                return text

    def test_injection_mode_enum_has_valid_modes(self):
        """Test that InjectionMode enum has only valid modes (no attribute)."""
        assert InjectionMode.CONTEXT.value == "context"
        assert InjectionMode.PARAMETER.value == "parameter"
        assert InjectionMode.SEAMLESS.value == "seamless"

        # Ensure 'attribute' is not in the enum
        with pytest.raises(ValueError):
            InjectionMode("attribute")

        # Ensure 'decorator' is not in the enum
        with pytest.raises(ValueError):
            InjectionMode("decorator")

    def test_enum_values_work_directly(self):
        """Test that valid enum values can be used directly in decorator."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode=InjectionMode.CONTEXT,
        )
        def test_func(text: str) -> str:
            return text

        assert isinstance(test_func, OptimizedFunction)

    def test_all_valid_injection_modes_via_enum(self):
        """Test that all valid injection modes can be specified via enum."""

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

        # All should be OptimizedFunction instances
        assert all(
            isinstance(f, OptimizedFunction)
            for f in [context_func, param_func, seamless_func]
        )
