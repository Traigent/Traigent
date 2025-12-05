"""Tests for wrapper factory functions.

Tests Protocol-based interface and helper functions for creating
constructor and method wrappers.

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.integrations.wrappers import (
    OverrideContext,
    apply_parameter_overrides,
    create_method_wrapper,
    create_resilient_wrapper,
    create_wrapper,
)


class TestApplyParameterOverrides:
    """Tests for apply_parameter_overrides function."""

    def test_basic_override(self):
        """Test basic parameter override."""
        kwargs = {"existing": "value"}
        config_dict = {"temperature": 0.7, "max_tokens": 100}
        parameter_mapping = {"temperature": "temp", "max_tokens": "max_tokens"}

        result = apply_parameter_overrides(kwargs, config_dict, parameter_mapping)

        assert result["existing"] == "value"
        assert result["temp"] == 0.7
        assert result["max_tokens"] == 100

    def test_no_override_for_missing_config(self):
        """Test that missing config values don't create overrides."""
        kwargs = {"existing": "value"}
        config_dict = {"temperature": 0.7}
        parameter_mapping = {"temperature": "temp", "max_tokens": "max_tokens"}

        result = apply_parameter_overrides(kwargs, config_dict, parameter_mapping)

        assert "temp" in result
        assert "max_tokens" not in result

    def test_original_kwargs_unchanged(self):
        """Test that original kwargs are not modified."""
        kwargs = {"existing": "value"}
        config_dict = {"temperature": 0.7}
        parameter_mapping = {"temperature": "temp"}

        result = apply_parameter_overrides(kwargs, config_dict, parameter_mapping)

        assert "temp" not in kwargs
        assert "temp" in result

    def test_supported_params_filter(self):
        """Test filtering by supported params list."""
        kwargs = {}
        config_dict = {"temperature": 0.7, "max_tokens": 100}
        parameter_mapping = {"temperature": "temp", "max_tokens": "max_tokens"}
        supported_params = ["temperature"]

        result = apply_parameter_overrides(
            kwargs, config_dict, parameter_mapping, supported_params=supported_params
        )

        assert "temp" in result
        assert "max_tokens" not in result

    def test_config_space_filter(self):
        """Test filtering by config space."""
        kwargs = {}
        config_dict = {"temperature": 0.7, "max_tokens": 100}
        parameter_mapping = {"temperature": "temp", "max_tokens": "max_tokens"}
        config_space = {"temperature": [0.1, 0.7, 0.9]}

        result = apply_parameter_overrides(
            kwargs, config_dict, parameter_mapping, config_space=config_space
        )

        assert "temp" in result
        assert "max_tokens" not in result

    def test_both_filters_combined(self):
        """Test both supported_params and config_space filters."""
        kwargs = {}
        config_dict = {"temperature": 0.7, "max_tokens": 100, "model": "gpt-4"}
        parameter_mapping = {
            "temperature": "temp",
            "max_tokens": "max_tokens",
            "model": "model",
        }
        supported_params = ["temperature", "model"]
        config_space = {"temperature": [0.1, 0.7], "max_tokens": [50, 100]}

        result = apply_parameter_overrides(
            kwargs,
            config_dict,
            parameter_mapping,
            supported_params=supported_params,
            config_space=config_space,
        )

        # Only temperature passes both filters
        assert "temp" in result
        assert "max_tokens" not in result  # Not in supported_params
        assert "model" not in result  # Not in config_space


class TestCreateWrapper:
    """Tests for create_wrapper function."""

    def test_sync_wrapper_inactive(self):
        """Test sync wrapper returns original when inactive."""
        original = MagicMock(return_value="original_result")
        is_active = MagicMock(return_value=False)
        get_config = MagicMock()
        extract_config = MagicMock()
        apply_overrides = MagicMock()

        wrapper = create_wrapper(
            original, is_active, get_config, extract_config, apply_overrides
        )
        result = wrapper("arg1", kwarg1="value1")

        assert result == "original_result"
        original.assert_called_once_with("arg1", kwarg1="value1")
        get_config.assert_not_called()

    def test_sync_wrapper_active_no_config(self):
        """Test sync wrapper returns original when config is None."""
        original = MagicMock(return_value="original_result")
        is_active = MagicMock(return_value=True)
        get_config = MagicMock(return_value=None)
        extract_config = MagicMock(return_value=None)
        apply_overrides = MagicMock()

        wrapper = create_wrapper(
            original, is_active, get_config, extract_config, apply_overrides
        )
        result = wrapper("arg1")

        assert result == "original_result"
        original.assert_called_once_with("arg1")
        apply_overrides.assert_not_called()

    def test_sync_wrapper_applies_overrides(self):
        """Test sync wrapper applies overrides when active with config."""
        original = MagicMock(return_value="result")
        is_active = MagicMock(return_value=True)
        get_config = MagicMock(return_value={"temp": 0.5})
        extract_config = MagicMock(return_value={"temp": 0.5})
        apply_overrides = MagicMock(return_value={"kwarg1": "overridden"})

        wrapper = create_wrapper(
            original, is_active, get_config, extract_config, apply_overrides
        )
        result = wrapper("arg1", kwarg1="original")

        assert result == "result"
        apply_overrides.assert_called_once()
        original.assert_called_once_with("arg1", kwarg1="overridden")

    @pytest.mark.asyncio
    async def test_async_wrapper_inactive(self):
        """Test async wrapper returns original when inactive."""

        async def async_original(*args, **kwargs):
            return "async_result"

        is_active = MagicMock(return_value=False)
        get_config = MagicMock()
        extract_config = MagicMock()
        apply_overrides = MagicMock()

        wrapper = create_wrapper(
            async_original, is_active, get_config, extract_config, apply_overrides
        )
        result = await wrapper("arg1")

        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_async_wrapper_applies_overrides(self):
        """Test async wrapper applies overrides when active."""

        async def async_original(*args, **kwargs):
            return f"result with {kwargs.get('param', 'none')}"

        is_active = MagicMock(return_value=True)
        get_config = MagicMock(return_value={"param": "overridden"})
        extract_config = MagicMock(return_value={"param": "overridden"})
        apply_overrides = MagicMock(return_value={"param": "overridden"})

        wrapper = create_wrapper(
            async_original, is_active, get_config, extract_config, apply_overrides
        )
        result = await wrapper()

        assert result == "result with overridden"

    def test_preserves_functools_wraps(self):
        """Test that wrapper preserves function metadata."""

        def original_function():
            """Original docstring."""
            pass

        wrapper = create_wrapper(
            original_function,
            lambda: False,
            lambda: None,
            lambda x: None,
            lambda x: x,
        )

        assert wrapper.__name__ == "original_function"
        assert wrapper.__doc__ == "Original docstring."


class TestCreateMethodWrapper:
    """Tests for create_method_wrapper function."""

    def test_method_wrapper_inactive(self):
        """Test method wrapper returns original when inactive."""

        def original_method(self, *args, **kwargs):
            return f"original {args} {kwargs}"

        is_active = MagicMock(return_value=False)
        get_config = MagicMock()
        extract_config = MagicMock()
        apply_overrides = MagicMock()

        wrapper = create_method_wrapper(
            original_method, is_active, get_config, extract_config, apply_overrides
        )

        instance = MagicMock()
        result = wrapper(instance, "arg1", kwarg1="value1")

        assert "original" in result

    def test_method_wrapper_applies_overrides(self):
        """Test method wrapper applies overrides when active."""

        def original_method(self, *args, **kwargs):
            return kwargs.get("param", "default")

        is_active = MagicMock(return_value=True)
        get_config = MagicMock(return_value={"param": "configured"})
        extract_config = MagicMock(return_value={"param": "configured"})
        apply_overrides = MagicMock(return_value={"param": "overridden"})

        wrapper = create_method_wrapper(
            original_method, is_active, get_config, extract_config, apply_overrides
        )

        instance = MagicMock()
        result = wrapper(instance)

        assert result == "overridden"

    @pytest.mark.asyncio
    async def test_async_method_wrapper(self):
        """Test async method wrapper."""

        async def async_method(self, *args, **kwargs):
            return f"async {kwargs.get('param', 'default')}"

        is_active = MagicMock(return_value=True)
        get_config = MagicMock(return_value={"param": "configured"})
        extract_config = MagicMock(return_value={"param": "configured"})
        apply_overrides = MagicMock(return_value={"param": "overridden"})

        wrapper = create_method_wrapper(
            async_method, is_active, get_config, extract_config, apply_overrides
        )

        instance = MagicMock()
        result = await wrapper(instance)

        assert result == "async overridden"


class TestCreateResilientWrapper:
    """Tests for create_resilient_wrapper function."""

    def test_resilient_wrapper_success(self):
        """Test resilient wrapper passes through on success."""

        def original():
            return "original"

        def wrapper_func():
            return "wrapped"

        resilient = create_resilient_wrapper(original, wrapper_func)
        result = resilient()

        assert result == "wrapped"

    def test_resilient_wrapper_fallback_on_error(self):
        """Test resilient wrapper falls back to original on error."""

        def original():
            return "original"

        def wrapper_func():
            raise ValueError("Wrapper failed")

        resilient = create_resilient_wrapper(original, wrapper_func)
        result = resilient()

        assert result == "original"

    def test_resilient_wrapper_no_fallback(self):
        """Test resilient wrapper raises error when fallback disabled."""

        def original():
            return "original"

        def wrapper_func():
            raise ValueError("Wrapper failed")

        resilient = create_resilient_wrapper(
            original, wrapper_func, fallback_on_error=False
        )

        with pytest.raises(ValueError, match="Wrapper failed"):
            resilient()

    @pytest.mark.asyncio
    async def test_async_resilient_wrapper_success(self):
        """Test async resilient wrapper passes through on success."""

        async def original():
            return "original"

        async def wrapper_func():
            return "wrapped"

        resilient = create_resilient_wrapper(original, wrapper_func)
        result = await resilient()

        assert result == "wrapped"

    @pytest.mark.asyncio
    async def test_async_resilient_wrapper_fallback(self):
        """Test async resilient wrapper falls back on error."""

        async def original():
            return "original"

        async def wrapper_func():
            raise ValueError("Async wrapper failed")

        resilient = create_resilient_wrapper(original, wrapper_func)
        result = await resilient()

        assert result == "original"


class TestOverrideContextProtocol:
    """Tests for OverrideContext Protocol compliance."""

    def test_protocol_implementation(self):
        """Test that a class can implement OverrideContext Protocol."""

        class MockContext:
            def is_override_active(self) -> bool:
                return True

            def get_parameter_mapping(self, class_name: str) -> dict[str, str]:
                return {"param": "mapped_param"}

            def get_method_params(self, class_name: str, method_name: str) -> list[str]:
                return ["param1", "param2"]

            def extract_config_dict(self, config: Any) -> dict[str, Any] | None:
                return {"key": "value"}

        context = MockContext()

        # Verify Protocol compliance through duck typing
        assert context.is_override_active() is True
        assert context.get_parameter_mapping("test.Class") == {"param": "mapped_param"}
        assert context.get_method_params("test.Class", "method") == ["param1", "param2"]
        assert context.extract_config_dict(None) == {"key": "value"}

    def test_protocol_type_checking(self):
        """Test Protocol can be used for type hints."""

        def use_context(ctx: OverrideContext) -> bool:
            return ctx.is_override_active()

        class ValidContext:
            def is_override_active(self) -> bool:
                return False

            def get_parameter_mapping(self, class_name: str) -> dict[str, str]:
                return {}

            def get_method_params(self, class_name: str, method_name: str) -> list[str]:
                return []

            def extract_config_dict(self, config: Any) -> dict[str, Any] | None:
                return None

        # This should work at runtime (duck typing)
        result = use_context(ValidContext())
        assert result is False
