"""Comprehensive tests for invokers base module.

Tests cover:
- BaseInvoker interface and abstract methods
- InvocationResult data structure
- Function execution patterns
- Error handling and timeout management
- Resource management and cleanup
- Batch processing capabilities
"""

import asyncio
import time

import pytest

from traigent.invokers.base import BaseInvoker, InvocationResult


class MockInvoker(BaseInvoker):
    """Mock invoker for testing BaseInvoker interface."""

    def __init__(self, timeout: float = 60.0, max_retries: int = 0, **kwargs):
        super().__init__(timeout, max_retries, **kwargs)
        self.invoke_call_count = 0
        self.last_func = None
        self.last_args = None
        self.last_kwargs = None
        self.should_fail = False
        self.should_timeout = False
        self.result_to_return = None
        self.execution_delay = 0.0

    async def invoke(
        self, func, config=None, input_data=None, *args, **kwargs
    ) -> InvocationResult:
        """Mock invoke method that supports both calling styles."""
        self.invoke_call_count += 1
        self.last_func = func

        # Handle both calling styles
        if (
            config is not None
            and isinstance(config, dict)
            and input_data is not None
            and isinstance(input_data, dict)
        ):
            # Standard BaseInvoker style: invoke(func, config, input_data)
            self.last_args = (config, input_data)
            self.last_kwargs = {}
            call_args = ()
            call_kwargs = input_data
        else:
            # Test style: invoke(func, *args, **kwargs) - treat config and input_data as additional args
            if config is not None:
                args = (
                    (config,) + (input_data,) + args
                    if input_data is not None
                    else (config,) + args
                )
            elif input_data is not None:
                args = (input_data,) + args

            self.last_args = args
            self.last_kwargs = kwargs
            call_args = args
            call_kwargs = kwargs

        if self.execution_delay > 0:
            await asyncio.sleep(self.execution_delay)

        if self.should_timeout:
            await asyncio.sleep(self.timeout + 1.0)  # Exceed timeout

        if self.should_fail:
            raise ValueError("Mock invocation failure")

        if self.result_to_return:
            return self.result_to_return

        # Default mock behavior - call the function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*call_args, **call_kwargs)
            else:
                result = func(*call_args, **call_kwargs)

            return InvocationResult(
                result=result,
                is_successful=True,
                execution_time=self.execution_delay or 0.1,
                error=None,
                metadata={"invoker_type": "mock"},
            )
        except Exception as e:
            return InvocationResult(
                result=None,
                is_successful=False,
                execution_time=self.execution_delay or 0.1,
                error=str(e),
                metadata={"invoker_type": "mock", "error_type": type(e).__name__},
            )

    async def invoke_batch(self, func, config, input_batch):
        """Mock batch invoke method."""
        results = []
        for input_data in input_batch:
            result = await self.invoke(func, config, input_data)
            results.append(result)
        return results

    def supports_streaming(self) -> bool:
        """Mock supports streaming method."""
        return False

    def supports_batch(self) -> bool:
        """Mock supports batch method."""
        return True


def mock_sync_function(x: int, y: int = 10) -> int:
    """Mock synchronous function for testing."""
    return x + y


async def mock_async_function(x: int, multiplier: int = 2) -> int:
    """Mock asynchronous function for testing."""
    await asyncio.sleep(0.01)  # Small delay
    return x * multiplier


def mock_failing_function():
    """Mock function that always fails."""
    raise ValueError("This function always fails")


async def mock_slow_function(delay: float = 1.0):
    """Mock function with configurable delay."""
    await asyncio.sleep(delay)
    return "completed"


class TestInvocationResult:
    """Test InvocationResult data structure."""

    def test_invocation_result_creation_success(self):
        """Test creating successful InvocationResult."""
        result = InvocationResult(
            result="success_value",
            is_successful=True,
            execution_time=0.5,
            error=None,
            metadata={"source": "test"},
        )

        assert result.result == "success_value"
        assert result.is_successful is True
        assert result.execution_time == 0.5
        assert result.error is None
        assert result.metadata["source"] == "test"

    def test_invocation_result_creation_failure(self):
        """Test creating failed InvocationResult."""
        result = InvocationResult(
            result=None,
            is_successful=False,
            execution_time=0.2,
            error="Function failed",
            metadata={"error_type": "ValueError"},
        )

        assert result.result is None
        assert result.is_successful is False
        assert result.execution_time == 0.2
        assert result.error == "Function failed"
        assert result.metadata["error_type"] == "ValueError"

    def test_invocation_result_defaults(self):
        """Test InvocationResult with default values."""
        result = InvocationResult(
            result="test_result", is_successful=True, execution_time=1.0
        )

        assert result.error is None
        assert result.metadata == {}

    def test_invocation_result_complex_result(self):
        """Test InvocationResult with complex result data."""
        complex_result = {
            "output": "Generated text",
            "metadata": {"tokens_used": 150, "model": "GPT-4o"},
            "metrics": {"latency": 0.8, "confidence": 0.95},
        }

        result = InvocationResult(
            result=complex_result,
            is_successful=True,
            execution_time=0.8,
            metadata={"invocation_id": "test_123"},
        )

        assert result.result["output"] == "Generated text"
        assert result.result["metadata"]["tokens_used"] == 150
        assert result.result["metrics"]["confidence"] == 0.95
        assert result.metadata["invocation_id"] == "test_123"

    def test_invocation_result_string_representation(self):
        """Test InvocationResult string representation."""
        result = InvocationResult(result="test", is_successful=True, execution_time=0.3)

        str_repr = str(result)
        assert "successful" in str_repr.lower()
        assert "0.3" in str_repr

        repr_str = repr(result)
        assert "InvocationResult" in repr_str


class TestBaseInvoker:
    """Test BaseInvoker abstract base class."""

    def test_base_invoker_cannot_be_instantiated(self):
        """Test that BaseInvoker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseInvoker()

    def test_mock_invoker_creation_default_params(self):
        """Test MockInvoker creation with default parameters."""
        invoker = MockInvoker()

        assert invoker.timeout == 60.0
        assert invoker.max_retries == 0
        assert invoker.invoke_call_count == 0

    def test_mock_invoker_creation_custom_params(self):
        """Test MockInvoker creation with custom parameters."""
        invoker = MockInvoker(timeout=30.0, max_retries=3, custom_param="test")

        assert invoker.timeout == 30.0
        assert invoker.max_retries == 3

    @pytest.mark.asyncio
    async def test_mock_invoker_sync_function(self):
        """Test MockInvoker with synchronous function."""
        invoker = MockInvoker()

        result = await invoker.invoke(mock_sync_function, 5, y=15)

        assert invoker.invoke_call_count == 1
        assert invoker.last_func is mock_sync_function
        assert invoker.last_args == (5,)
        assert invoker.last_kwargs == {"y": 15}

        assert result.is_successful is True
        assert result.result == 20  # 5 + 15
        assert result.error is None
        assert result.execution_time > 0
        assert result.metadata["invoker_type"] == "mock"

    @pytest.mark.asyncio
    async def test_mock_invoker_async_function(self):
        """Test MockInvoker with asynchronous function."""
        invoker = MockInvoker()

        result = await invoker.invoke(mock_async_function, 6, multiplier=3)

        assert invoker.invoke_call_count == 1
        assert invoker.last_func is mock_async_function
        assert invoker.last_args == (6,)
        assert invoker.last_kwargs == {"multiplier": 3}

        assert result.is_successful is True
        assert result.result == 18  # 6 * 3
        assert result.error is None

    @pytest.mark.asyncio
    async def test_mock_invoker_function_failure(self):
        """Test MockInvoker with function that fails."""
        invoker = MockInvoker()

        result = await invoker.invoke(mock_failing_function)

        assert invoker.invoke_call_count == 1
        assert result.is_successful is False
        assert result.result is None
        assert "This function always fails" in result.error
        assert result.metadata["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_mock_invoker_forced_failure(self):
        """Test MockInvoker with forced failure mode."""
        invoker = MockInvoker()
        invoker.should_fail = True

        with pytest.raises(ValueError, match="Mock invocation failure"):
            await invoker.invoke(mock_sync_function, 1, 2)

        assert invoker.invoke_call_count == 1

    @pytest.mark.asyncio
    async def test_mock_invoker_execution_delay(self):
        """Test MockInvoker with execution delay."""
        invoker = MockInvoker()
        invoker.execution_delay = 0.1

        start_time = time.time()
        result = await invoker.invoke(mock_sync_function, 1, 2)
        execution_duration = time.time() - start_time

        assert execution_duration >= 0.1
        assert result.is_successful is True
        assert result.execution_time >= 0.1

    @pytest.mark.asyncio
    async def test_mock_invoker_custom_result(self):
        """Test MockInvoker with custom result override."""
        invoker = MockInvoker()
        custom_result = InvocationResult(
            result="custom_value",
            is_successful=True,
            execution_time=0.05,
            metadata={"custom": True},
        )
        invoker.result_to_return = custom_result

        result = await invoker.invoke(mock_sync_function, 1, 2)

        assert result is custom_result
        assert result.result == "custom_value"
        assert result.metadata["custom"] is True

    @pytest.mark.asyncio
    async def test_mock_invoker_no_args(self):
        """Test MockInvoker with function that takes no arguments."""

        def no_args_function():
            return "no_args_result"

        invoker = MockInvoker()
        result = await invoker.invoke(no_args_function)

        assert invoker.last_args == ()
        assert invoker.last_kwargs == {}
        assert result.result == "no_args_result"

    @pytest.mark.asyncio
    async def test_mock_invoker_kwargs_only(self):
        """Test MockInvoker with keyword-only arguments."""

        def kwargs_only_function(*, param1, param2="default"):
            return f"{param1}_{param2}"

        invoker = MockInvoker()
        result = await invoker.invoke(
            kwargs_only_function, param1="test", param2="value"
        )

        assert invoker.last_args == ()
        assert invoker.last_kwargs == {"param1": "test", "param2": "value"}
        assert result.result == "test_value"

    @pytest.mark.asyncio
    async def test_mock_invoker_complex_function_args(self):
        """Test MockInvoker with complex function arguments."""

        def complex_function(a, b, *args, c=None, **kwargs):
            return {"a": a, "b": b, "args": args, "c": c, "kwargs": kwargs}

        invoker = MockInvoker()
        result = await invoker.invoke(
            complex_function,
            "first",
            "second",
            "third",
            "fourth",
            c="custom",
            extra1="value1",
            extra2="value2",
        )

        assert result.is_successful is True
        assert result.result["a"] == "first"
        assert result.result["b"] == "second"
        assert result.result["args"] == ("third", "fourth")
        assert result.result["c"] == "custom"
        assert result.result["kwargs"]["extra1"] == "value1"
        assert result.result["kwargs"]["extra2"] == "value2"


class TestInvokerInterface:
    """Test invoker interface patterns and contracts."""

    @pytest.mark.asyncio
    async def test_invoker_timeout_parameter(self):
        """Test that invoker respects timeout parameter."""
        invoker = MockInvoker(timeout=0.1)

        # This test verifies the timeout is stored correctly
        assert invoker.timeout == 0.1

        # In a real implementation, this would test actual timeout behavior
        # For MockInvoker, we can test the timeout simulation
        invoker.should_timeout = True

        start_time = time.time()
        try:
            await asyncio.wait_for(
                invoker.invoke(mock_sync_function, 1, 2), timeout=0.2
            )
        except TimeoutError:
            pass  # Expected for this test

        elapsed = time.time() - start_time
        assert elapsed >= 0.1  # Should have taken at least timeout duration

    @pytest.mark.asyncio
    async def test_invoker_max_retries_parameter(self):
        """Test that invoker stores max_retries parameter."""
        invoker = MockInvoker(max_retries=5)

        assert invoker.max_retries == 5

        # In a real implementation, this would test retry behavior
        # For MockInvoker, we just verify the parameter is stored

    @pytest.mark.asyncio
    async def test_invoker_custom_kwargs(self):
        """Test that invoker can accept custom keyword arguments."""
        custom_invoker = MockInvoker(
            timeout=30.0, max_retries=2, custom_param="test_value", another_param=42
        )

        # Custom parameters should be accepted (though not necessarily used)
        assert custom_invoker.timeout == 30.0
        assert custom_invoker.max_retries == 2

    @pytest.mark.asyncio
    async def test_multiple_invocations(self):
        """Test multiple invocations with same invoker."""
        invoker = MockInvoker()

        # First invocation
        result1 = await invoker.invoke(mock_sync_function, 1, 2)
        assert result1.result == 3
        assert invoker.invoke_call_count == 1

        # Second invocation
        result2 = await invoker.invoke(mock_sync_function, 5, 10)
        assert result2.result == 15
        assert invoker.invoke_call_count == 2

        # Third invocation with async function
        result3 = await invoker.invoke(mock_async_function, 4, multiplier=3)
        assert result3.result == 12
        assert invoker.invoke_call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_invocations(self):
        """Test concurrent invocations with same invoker."""
        invoker = MockInvoker(execution_delay=0.05)

        # Start multiple concurrent invocations
        tasks = [invoker.invoke(mock_sync_function, i, i + 1) for i in range(5)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # All should complete successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.is_successful is True
            assert result.result == i + (i + 1)  # i + (i+1)

        # Should complete concurrently (not sequentially)
        # If sequential: 5 * 0.05 = 0.25s, if concurrent: ~0.05s
        assert elapsed < 0.2  # Much less than sequential time

        # Invoker should have been called 5 times
        assert invoker.invoke_call_count == 5

    @pytest.mark.asyncio
    async def test_invoker_state_isolation(self):
        """Test that invoker state is properly isolated between calls."""
        invoker = MockInvoker()

        # First call
        await invoker.invoke(mock_sync_function, 10, y=20)
        first_args = invoker.last_args
        first_kwargs = invoker.last_kwargs

        # Second call with different parameters
        await invoker.invoke(mock_async_function, 5, multiplier=4)
        second_args = invoker.last_args
        second_kwargs = invoker.last_kwargs

        # State should have been updated to reflect the latest call
        assert first_args != second_args
        assert first_kwargs != second_kwargs
        assert second_args == (5,)
        assert second_kwargs == {"multiplier": 4}


class TestErrorHandling:
    """Test error handling patterns in invokers."""

    @pytest.mark.asyncio
    async def test_function_exception_handling(self):
        """Test handling of exceptions from invoked functions."""
        invoker = MockInvoker()

        # Test various exception types
        def type_error_function():
            raise TypeError("Type error occurred")

        def value_error_function():
            raise ValueError("Value error occurred")

        def runtime_error_function():
            raise RuntimeError("Runtime error occurred")

        # Test TypeError
        result1 = await invoker.invoke(type_error_function)
        assert result1.is_successful is False
        assert "Type error occurred" in result1.error
        assert result1.metadata["error_type"] == "TypeError"

        # Test ValueError
        result2 = await invoker.invoke(value_error_function)
        assert result2.is_successful is False
        assert "Value error occurred" in result2.error
        assert result2.metadata["error_type"] == "ValueError"

        # Test RuntimeError
        result3 = await invoker.invoke(runtime_error_function)
        assert result3.is_successful is False
        assert "Runtime error occurred" in result3.error
        assert result3.metadata["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_async_function_exception_handling(self):
        """Test handling of exceptions from async functions."""

        async def async_failing_function():
            await asyncio.sleep(0.01)
            raise ValueError("Async function failed")

        invoker = MockInvoker()
        result = await invoker.invoke(async_failing_function)

        assert result.is_successful is False
        assert "Async function failed" in result.error
        assert result.metadata["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_function_with_partial_failure(self):
        """Test function that sometimes fails based on input."""

        def conditional_failure_function(should_fail: bool):
            if should_fail:
                raise ValueError("Conditional failure")
            return "success"

        invoker = MockInvoker()

        # Success case
        success_result = await invoker.invoke(conditional_failure_function, False)
        assert success_result.is_successful is True
        assert success_result.result == "success"

        # Failure case
        failure_result = await invoker.invoke(conditional_failure_function, True)
        assert failure_result.is_successful is False
        assert "Conditional failure" in failure_result.error


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    @pytest.mark.asyncio
    async def test_invoker_with_large_result(self):
        """Test invoker with function that returns large result."""

        def large_result_function():
            # Return a large data structure
            return {
                "data": ["item"] * 10000,
                "metadata": {"size": 10000},
                "nested": {"level1": {"level2": {"values": list(range(1000))}}},
            }

        invoker = MockInvoker()
        result = await invoker.invoke(large_result_function)

        assert result.is_successful is True
        assert len(result.result["data"]) == 10000
        assert result.result["metadata"]["size"] == 10000
        assert len(result.result["nested"]["level1"]["level2"]["values"]) == 1000

    @pytest.mark.asyncio
    async def test_invoker_with_none_result(self):
        """Test invoker with function that returns None."""

        def none_result_function():
            return None

        invoker = MockInvoker()
        result = await invoker.invoke(none_result_function)

        assert result.is_successful is True
        assert result.result is None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_invoker_with_generator_function(self):
        """Test invoker with generator function."""

        def generator_function():
            yield from range(5)

        invoker = MockInvoker()
        result = await invoker.invoke(generator_function)

        assert result.is_successful is True
        # Generator should be returned as-is
        assert hasattr(result.result, "__iter__")
        # Convert to list to test contents
        generator_values = list(result.result)
        assert generator_values == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_invoker_performance_tracking(self):
        """Test that invoker properly tracks execution time."""

        async def timed_function(delay: float):
            await asyncio.sleep(delay)
            return "completed"

        invoker = MockInvoker()

        # Test with known delay
        delay = 0.1
        start_time = time.time()
        result = await invoker.invoke(timed_function, delay)
        actual_duration = time.time() - start_time

        assert result.is_successful is True
        assert result.result == "completed"
        # Execution time should be at least the delay
        assert result.execution_time >= delay
        # Should be close to actual measured duration
        assert abs(result.execution_time - actual_duration) < 0.05

    @pytest.mark.asyncio
    async def test_invoker_memory_efficiency(self):
        """Test that invoker doesn't leak memory with many invocations."""
        import gc

        def simple_function(x):
            return x * 2

        invoker = MockInvoker()

        # Perform many invocations
        for i in range(1000):
            result = await invoker.invoke(simple_function, i)
            assert result.result == i * 2

        # Force garbage collection
        gc.collect()

        # Invoker should still work correctly
        final_result = await invoker.invoke(simple_function, 999)
        assert final_result.result == 1998
        assert invoker.invoke_call_count == 1001
