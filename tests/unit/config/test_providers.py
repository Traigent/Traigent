"""Tests for configuration providers."""

import pytest

from traigent.config.context import ConfigurationContext, get_config
from traigent.config.providers import (
    AttributeBasedProvider,
    ContextBasedProvider,
    ParameterBasedProvider,
    get_provider,
)
from traigent.config.types import TraigentConfig
from traigent.utils.exceptions import ConfigurationError


class TestContextBasedProvider:
    """Test suite for ContextBasedProvider."""

    def test_inject_config_sync_function(self):
        """Test context injection with synchronous function."""
        provider = ContextBasedProvider()
        config = {"model": "GPT-4o", "temperature": 0.7}

        def test_func(query: str) -> str:
            current_config = get_config()
            if isinstance(current_config, dict):
                return f"Using {current_config.get('model')}"
            else:
                return f"Using {getattr(current_config, 'model', None)}"

        wrapped_func = provider.inject_config(test_func, config)
        result = wrapped_func("test query")

        assert result == "Using GPT-4o"

    @pytest.mark.asyncio
    async def test_inject_config_async_function(self):
        """Test context injection with asynchronous function."""
        provider = ContextBasedProvider()
        config = {"model": "GPT-4o", "temperature": 0.7}

        async def test_func(query: str) -> str:
            current_config = get_config()
            return f"Using {current_config.get('model')}"

        wrapped_func = provider.inject_config(test_func, config)
        result = await wrapped_func("test query")

        assert result == "Using GPT-4o"

    def test_inject_config_merges_with_context(self):
        """Test that injection merges with existing context."""
        provider = ContextBasedProvider()
        context_config = {"model": "o4-mini", "max_tokens": 1000}
        injection_config = {"temperature": 0.8}

        def test_func() -> dict:
            return get_config()

        with ConfigurationContext(context_config):
            wrapped_func = provider.inject_config(test_func, injection_config)
            result = wrapped_func()

            assert result["model"] == "o4-mini"  # From context
            assert result["temperature"] == 0.8  # From injection
            assert result["max_tokens"] == 1000  # From context

    def test_extract_config(self):
        """Test extracting configuration from context."""
        provider = ContextBasedProvider()
        config = {"model": "GPT-4o"}

        with ConfigurationContext(config):
            extracted = provider.extract_config(lambda: None)
            assert extracted == config

    def test_supports_function(self):
        """Test that context provider supports any function."""
        provider = ContextBasedProvider()

        def sync_func():
            pass

        async def async_func():
            pass

        assert provider.supports_function(sync_func) is True
        assert provider.supports_function(async_func) is True


class TestParameterBasedProvider:
    """Test suite for ParameterBasedProvider."""

    def test_inject_config_with_default_param_name(self):
        """Test parameter injection with default parameter name."""
        provider = ParameterBasedProvider()
        config = {"model": "GPT-4o", "temperature": 0.7}

        def test_func(query: str, config: TraigentConfig) -> str:
            return f"Using {config.model}"

        wrapped_func = provider.inject_config(test_func, config)
        result = wrapped_func("test query")

        assert result == "Using GPT-4o"

    def test_inject_config_with_custom_param_name(self):
        """Test parameter injection with custom parameter name."""
        provider = ParameterBasedProvider(default_param_name="my_config")
        config = {"model": "GPT-4o", "temperature": 0.7}

        def test_func(query: str, my_config: TraigentConfig) -> str:
            return f"Using {my_config.model}"

        wrapped_func = provider.inject_config(test_func, config, "my_config")
        result = wrapped_func("test query")

        assert result == "Using GPT-4o"

    @pytest.mark.asyncio
    async def test_inject_config_async_function(self):
        """Test parameter injection with async function."""
        provider = ParameterBasedProvider()
        config = {"model": "GPT-4o", "temperature": 0.7}

        async def test_func(query: str, config: TraigentConfig) -> str:
            return f"Using {config.model}"

        wrapped_func = provider.inject_config(test_func, config)
        result = await wrapped_func("test query")

        assert result == "Using GPT-4o"

    def test_inject_config_missing_parameter(self):
        """Test error when function doesn't have config parameter."""
        provider = ParameterBasedProvider()
        config = {"model": "GPT-4o"}

        def test_func(query: str) -> str:
            return query

        with pytest.raises(
            ConfigurationError, match="does not have parameter 'config'"
        ):
            provider.inject_config(test_func, config)

    def test_inject_config_creates_traigent_config(self):
        """Test that dict config is converted to TraigentConfig."""
        provider = ParameterBasedProvider()
        config = {"model": "GPT-4o", "temperature": 0.7}

        def test_func(query: str, config: TraigentConfig) -> TraigentConfig:
            return config

        wrapped_func = provider.inject_config(test_func, config)
        result = wrapped_func("test")

        assert isinstance(result, TraigentConfig)
        assert result.model == "GPT-4o"
        assert result.temperature == 0.7

    def test_supports_function_with_config_param(self):
        """Test supports_function returns True for functions with config param."""
        provider = ParameterBasedProvider()

        def func_with_config(query: str, config: TraigentConfig):
            pass

        def func_without_config(query: str):
            pass

        assert provider.supports_function(func_with_config) is True
        assert provider.supports_function(func_without_config) is False

    def test_extract_config_returns_none(self):
        """Test that parameter provider cannot extract config."""
        provider = ParameterBasedProvider()

        def test_func(config: TraigentConfig):
            pass

        assert provider.extract_config(test_func) is None


class TestAttributeBasedProvider:
    """Test suite for AttributeBasedProvider."""

    def test_inject_config_sync_function(self):
        """Test decorator injection with sync function."""
        provider = AttributeBasedProvider()
        config = {"model": "GPT-4o", "temperature": 0.7}

        wrapped_func = provider.inject_config(lambda query: None, config)

        # Test that config is accessible on wrapper
        assert hasattr(wrapped_func, "current_config")
        assert wrapped_func.current_config == config

        # Test the actual functionality in a more realistic way
        def test_func(query: str) -> str:
            return f"Using {query}"

        wrapped = provider.inject_config(test_func, config)
        result = wrapped("GPT-4o")
        assert result == "Using GPT-4o"
        assert wrapped.current_config == config

    @pytest.mark.asyncio
    async def test_inject_config_async_function(self):
        """Test decorator injection with async function."""
        provider = AttributeBasedProvider()
        config = {"model": "GPT-4o", "temperature": 0.7}

        async def test_func(query: str) -> str:
            current_config = test_func.current_config
            return f"Using {current_config['model']}"

        # Rebind test_func to the wrapper so the closure sees the wrapper
        test_func = provider.inject_config(test_func, config)
        result = await test_func("test query")

        assert result == "Using GPT-4o"

    def test_inject_config_custom_attribute_name(self):
        """Test decorator injection with custom attribute name."""
        provider = AttributeBasedProvider(attribute_name="my_config")
        config = {"model": "GPT-4o"}

        def test_func() -> str:
            return "test"

        wrapped_func = provider.inject_config(test_func, config)
        result = wrapped_func()

        assert result == "test"
        assert hasattr(wrapped_func, "my_config")
        assert wrapped_func.my_config == config

    def test_extract_config(self):
        """Test extracting config from function attribute."""
        provider = AttributeBasedProvider()
        config = {"model": "GPT-4o"}

        def test_func():
            pass

        wrapped_func = provider.inject_config(test_func, config)
        extracted = provider.extract_config(wrapped_func)

        assert extracted == config

    def test_supports_function(self):
        """Test that decorator provider supports any function."""
        provider = AttributeBasedProvider()

        def test_func():
            pass

        assert provider.supports_function(test_func) is True


class TestGetProvider:
    """Test suite for get_provider function."""

    def test_get_context_provider(self):
        """Test getting context provider."""
        provider = get_provider("context")
        assert isinstance(provider, ContextBasedProvider)

    def test_get_parameter_provider(self):
        """Test getting parameter provider."""
        provider = get_provider("parameter")
        assert isinstance(provider, ParameterBasedProvider)
        assert provider.default_param_name == "config"

    def test_get_parameter_provider_with_custom_param(self):
        """Test getting parameter provider with custom param name."""
        provider = get_provider("parameter", config_param="my_config")
        assert isinstance(provider, ParameterBasedProvider)
        assert provider.default_param_name == "my_config"

    def test_get_decorator_provider(self):
        """Test getting decorator provider."""
        provider = get_provider("attribute")
        assert isinstance(provider, AttributeBasedProvider)
        assert provider.attribute_name == "current_config"

    def test_get_decorator_provider_with_custom_attribute(self):
        """Test getting decorator provider with custom attribute name."""
        provider = get_provider("attribute", attribute_name="my_config")
        assert isinstance(provider, AttributeBasedProvider)
        assert provider.attribute_name == "my_config"

    def test_get_unknown_provider(self):
        """Test error for unknown provider."""
        with pytest.raises(ConfigurationError, match="Unknown injection mode: unknown"):
            get_provider("unknown")

    def test_decorator_backward_compatibility(self):
        """Test that 'decorator' mode still works for backward compatibility."""
        # Should not raise an error
        provider = get_provider("decorator")
        assert isinstance(provider, AttributeBasedProvider)
