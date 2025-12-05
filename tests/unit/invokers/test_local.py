"""Tests for LocalInvoker."""

import asyncio
import time

import pytest

from traigent.invokers.base import InvocationResult
from traigent.invokers.local import LocalInvoker
from traigent.utils.exceptions import InvocationError


class TestLocalInvoker:
    """Test suite for LocalInvoker."""

    def test_init_default_values(self):
        """Test LocalInvoker initialization with default values."""
        invoker = LocalInvoker()

        assert invoker.timeout == 60.0
        assert invoker.max_retries == 0
        assert invoker.injection_mode == "context"
        assert invoker.config_param == "config"
        assert invoker._provider is not None

    def test_init_custom_values(self):
        """Test LocalInvoker initialization with custom values."""
        invoker = LocalInvoker(
            timeout=30.0,
            max_retries=3,
            injection_mode="parameter",
            config_param="my_config",
        )

        assert invoker.timeout == 30.0
        assert invoker.max_retries == 3
        assert invoker.injection_mode == "parameter"
        assert invoker.config_param == "my_config"

    @pytest.mark.asyncio
    async def test_invoke_sync_function_context_mode(self):
        """Test invoking sync function with context injection."""
        invoker = LocalInvoker(injection_mode="context")

        def test_func(value: int) -> int:
            # In real usage, this would get config from context
            # For test, just return doubled value
            return value * 2

        config = {"multiplier": 2}
        input_data = {"value": 5}

        result = await invoker.invoke(test_func, config, input_data)

        assert isinstance(result, InvocationResult)
        assert result.is_successful
        assert result.output == 10
        assert result.execution_time > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_invoke_sync_function_parameter_mode(self):
        """Test invoking sync function with parameter injection."""
        invoker = LocalInvoker(injection_mode="parameter", config_param="config")

        def test_func(value: int, config) -> int:
            multiplier = config.custom_params.get("multiplier", 1)
            return value * multiplier

        config = {"multiplier": 3}
        input_data = {"value": 4}

        result = await invoker.invoke(test_func, config, input_data)

        assert isinstance(result, InvocationResult)
        assert result.is_successful
        assert result.output == 12
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_invoke_async_function(self):
        """Test invoking async function."""
        invoker = LocalInvoker(injection_mode="context")

        async def test_func(value: int) -> int:
            await asyncio.sleep(0.01)  # Simulate async work
            return value * 2

        config = {}
        input_data = {"value": 7}

        result = await invoker.invoke(test_func, config, input_data)

        assert isinstance(result, InvocationResult)
        assert result.is_successful
        assert result.output == 14
        assert result.execution_time >= 0.01

    @pytest.mark.asyncio
    async def test_invoke_with_timeout(self):
        """Test invoking function that times out."""
        invoker = LocalInvoker(timeout=0.1)  # Very short timeout

        def slow_func(value: int) -> int:
            time.sleep(0.2)  # Sleep longer than timeout
            return value

        config = {}
        input_data = {"value": 1}

        result = await invoker.invoke(slow_func, config, input_data)

        assert isinstance(result, InvocationResult)
        assert not result.is_successful
        assert result.output is None
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_invoke_function_raises_exception(self):
        """Test invoking function that raises exception."""
        invoker = LocalInvoker()

        def error_func(value: int) -> int:
            raise ValueError("Test error")

        config = {}
        input_data = {"value": 1}

        result = await invoker.invoke(error_func, config, input_data)

        assert isinstance(result, InvocationResult)
        assert not result.is_successful
        assert result.output is None
        assert "Test error" in result.error

    @pytest.mark.asyncio
    async def test_invoke_batch_sequential(self):
        """Test batch invocation."""
        invoker = LocalInvoker()

        def test_func(value: int) -> int:
            return value * 2

        config = {}
        input_batch = [{"value": 1}, {"value": 2}, {"value": 3}]

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 3
        assert all(isinstance(r, InvocationResult) for r in results)
        assert all(r.is_successful for r in results)
        assert [r.output for r in results] == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_invoke_batch_with_failures(self):
        """Test batch invocation with some failures."""
        invoker = LocalInvoker()

        def test_func(value: int) -> int:
            if value == 2:
                raise ValueError("Error for value 2")
            return value * 2

        config = {}
        input_batch = [{"value": 1}, {"value": 2}, {"value": 3}]  # This will fail

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 3
        assert results[0].is_successful
        assert not results[1].is_successful  # Failed
        assert results[2].is_successful
        assert results[0].output == 2
        assert results[1].output is None
        assert results[2].output == 6

    @pytest.mark.asyncio
    async def test_invoke_batch_empty(self):
        """Test batch invocation with empty batch."""
        invoker = LocalInvoker()

        def test_func(value: int) -> int:
            return value * 2

        config = {}
        input_batch = []

        results = await invoker.invoke_batch(test_func, config, input_batch)

        assert len(results) == 0

    def test_supports_streaming(self):
        """Test supports_streaming method."""
        invoker = LocalInvoker()
        assert invoker.supports_streaming() is False

    def test_supports_batch(self):
        """Test supports_batch method."""
        invoker = LocalInvoker()
        assert invoker.supports_batch() is True

    def test_validate_function_invalid(self):
        """Test function validation with invalid function."""
        invoker = LocalInvoker()

        with pytest.raises(InvocationError, match="Function must be callable"):
            invoker.validate_function("not_a_function")

    def test_validate_function_parameter_injection_missing_param(self):
        """Test function validation for parameter injection without required param."""
        invoker = LocalInvoker(injection_mode="parameter", config_param="config")

        def func_without_config(value: int) -> int:
            return value

        with pytest.raises(InvocationError, match="does not have parameter 'config'"):
            invoker.validate_function(func_without_config)

    def test_validate_function_parameter_injection_with_param(self):
        """Test function validation for parameter injection with required param."""
        invoker = LocalInvoker(injection_mode="parameter", config_param="config")

        def func_with_config(value: int, config) -> int:
            return value

        # Should not raise
        invoker.validate_function(func_with_config)

    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        invoker = LocalInvoker()

        with pytest.raises(InvocationError, match="Configuration must be a dictionary"):
            invoker.validate_config("not_a_dict")

    def test_validate_input_invalid(self):
        """Test input validation with invalid input."""
        invoker = LocalInvoker()

        with pytest.raises(InvocationError, match="Input data must be a dictionary"):
            invoker.validate_input("not_a_dict")
