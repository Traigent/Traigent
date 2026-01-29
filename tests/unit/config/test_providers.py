"""Tests for configuration providers."""

import pytest

from traigent.config.context import ConfigurationContext, get_config
from traigent.config.providers import (
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

    def test_get_unknown_provider(self):
        """Test error for unknown provider."""
        with pytest.raises(ConfigurationError, match="Unknown injection mode: unknown"):
            get_provider("unknown")

    def test_attribute_mode_raises_configuration_error(self):
        """Test that attribute mode raises ConfigurationError with migration guide."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_provider("attribute")

        error_message = str(exc_info.value)
        assert "attribute" in error_message
        assert "removed" in error_message.lower()
        assert "context" in error_message  # Migration suggestion
        assert "seamless" in error_message  # Alternative

    def test_decorator_mode_raises_configuration_error(self):
        """Test that decorator mode (legacy alias) raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_provider("decorator")

        error_message = str(exc_info.value)
        assert "decorator" in error_message
        assert "removed" in error_message.lower()
