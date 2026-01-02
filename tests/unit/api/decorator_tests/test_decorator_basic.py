"""Basic decorator functionality tests.

Tests core decorator behavior including:
- Basic decoration and function wrapping
- Configuration space validation
- Default configuration handling
- Function signature preservation
"""

import pytest

from traigent.api.decorators import optimize
from traigent.core.optimized_function import OptimizedFunction

from .mock_infrastructure import create_simple_dataset
from .test_base import DecoratorTestBase


class TestDecoratorBasics(DecoratorTestBase):
    """Test basic decorator functionality."""

    def test_decorator_creates_optimized_function(self):
        """Test that decorator creates an OptimizedFunction instance."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]}, objectives=["accuracy"]
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)
        assert test_func.__name__ == "test_func"

    def test_decorator_preserves_function_signature(self):
        """Test that decorator preserves original function signature."""

        @optimize(configuration_space={"temperature": [0.0, 0.5, 1.0]})
        def complex_func(arg1: str, arg2: int = 5, *args, **kwargs) -> str:
            return f"{arg1}-{arg2}-{len(args)}-{len(kwargs)}"

        # Function should still be callable with original signature
        result = complex_func("test", 10, "extra", key="value")
        assert "test-10-1-1" in result

    def test_decorator_with_default_config(self):
        """Test decorator with default configuration."""

        @optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4"],
                "temperature": [0.0, 0.5, 1.0],
            },
            default_config={"model": "gpt-3.5", "temperature": 0.5},
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)
        # Default config should be accessible through the optimized function

    def test_decorator_without_configuration_space_raises_error(self):
        """Test that decorator without configuration space raises error."""
        # The decorator now raises ValueError for empty config space
        with pytest.raises(ValueError, match="Configuration space cannot be empty"):

            @optimize()  # Missing required configuration_space
            def test_func(text: str) -> str:
                return text

    def test_decorator_with_empty_configuration_space(self):
        """Test decorator with empty configuration space."""
        # Empty config space now raises ValueError
        with pytest.raises(ValueError, match="Configuration space cannot be empty"):

            @optimize(configuration_space={})
            def test_func(text: str) -> str:
                return text

    def test_decorator_with_invalid_configuration_space(self):
        """Test decorator with invalid configuration space."""
        from traigent.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Expected dictionary"):

            @optimize(configuration_space="invalid")  # Should be dict
            def test_func(text: str) -> str:
                return text

    def test_decorated_function_execution(self):
        """Test that decorated function can be executed normally."""

        @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
        def test_func(text: str, multiplier: int = 1) -> str:
            return text * multiplier

        result = test_func("hello", multiplier=3)
        assert result == "hellohellohello"

    def test_decorator_with_all_parameters(self):
        """Test decorator with all supported parameters."""
        dataset = create_simple_dataset(3)

        @optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4"],
                "temperature": [0.0, 0.5, 1.0],
            },
            objectives=["accuracy", "cost"],
            default_config={"model": "gpt-3.5", "temperature": 0.5},
            eval_dataset=dataset,
            injection_mode="parameter",
            execution_mode="edge_analytics",
        )
        def test_func(text: str, config=None) -> str:
            # Parameter mode requires config parameter
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)

    def test_async_function_decoration(self):
        """Test decoration of async functions."""

        @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
        async def async_func(text: str) -> str:
            return f"Async response: {text}"

        assert isinstance(async_func, OptimizedFunction)

        # Test execution
        import asyncio

        result = asyncio.run(async_func("test"))
        assert "Async response: test" in result

    def test_decorator_with_complex_configuration_space(self):
        """Test decorator with flattened configuration space."""

        # Traigent expects flat configuration spaces, not nested
        @optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4"],
                "temperature": [0.0, 0.5, 1.0],
                "max_tokens": [100, 200, 500],
                "top_p": [0.9, 0.95, 1.0],
                "max_retries": [1, 3, 5],
                "backoff_factor": [1.0, 2.0],
            }
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)
