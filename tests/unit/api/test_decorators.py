"""Comprehensive tests for traigent.api.decorators module."""

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from traigent.api.decorators import optimize
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset


class TestOptimizeDecorator:
    """Test suite for the @optimize decorator."""

    def test_decorator_with_config_space(self):
        """Test decorator with configuration space."""

        @optimize(configuration_space={"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]})
        def sample_function(x: int, y: float) -> float:
            """Sample function for testing."""
            return x * y

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.__name__ == "sample_function"
        assert sample_function.__doc__ == "Sample function for testing."

    def test_decorator_with_objectives(self):
        """Test decorator with objectives."""

        @optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4"],
                "temperature": [0.0, 0.5, 1.0],
            },
            objectives=["accuracy", "cost"],
        )
        def llm_function(model: str, temperature: float) -> str:
            return f"Result from {model} with temp {temperature}"

        assert isinstance(llm_function, OptimizedFunction)

    def test_decorator_with_default_config(self):
        """Test decorator with default configuration."""

        @optimize(
            configuration_space={"batch_size": [16, 32, 64], "lr": [0.001, 0.01, 0.1]},
            default_config={"batch_size": 32, "lr": 0.01},
        )
        def training_function(batch_size: int, lr: float) -> float:
            return batch_size * lr

        assert isinstance(training_function, OptimizedFunction)

    def test_decorated_function_execution(self):
        """Test that decorated function can still be called normally."""

        @optimize(configuration_space={"multiplier": [1, 2, 3, 4, 5]})
        def multiply_function(value: int, multiplier: int = 2) -> int:
            return value * multiplier

        # Function should still work normally
        result = multiply_function(10, multiplier=3)
        assert result == 30

    def test_decorator_preserves_signature(self):
        """Test that decorator creates callable optimized function."""

        @optimize(configuration_space={"param": ["a", "b", "c"]})
        def complex_function(arg1: str, arg2: int = 5, *args, **kwargs) -> str:
            return f"{arg1}-{arg2}-{len(args)}-{len(kwargs)}"

        # Should be callable and maintain name/doc
        assert callable(complex_function)
        assert complex_function.__name__ == "complex_function"
        assert isinstance(complex_function, OptimizedFunction)

        # Should still work when called
        result = complex_function("test", 10, "extra", key="value")
        assert "test-10-1-1" in result

    def test_decorator_with_cloud_execution_mode(self):
        """Test decorator when execution_mode='cloud'."""

        @optimize(
            configuration_space={"model": ["claude", "gpt-4"]},
            execution_mode="cloud",
        )
        def ai_function(model: str) -> str:
            return f"Using {model}"

        assert isinstance(ai_function, OptimizedFunction)

    def test_decorator_with_auto_optimize(self):
        """Test decorator with auto optimization enabled."""

        with pytest.raises(TypeError, match="auto_optimize"):

            @optimize(
                configuration_space={"threshold": [0.1, 0.5, 0.9]},
                auto_optimize=True,
            )
            def threshold_function(value: float, threshold: float = 0.5) -> bool:
                return value > threshold

    def test_decorator_without_config_space(self):
        """Test decorator without configuration space raises ValueError."""

        with pytest.raises(ValueError, match="Configuration space cannot be empty"):

            @optimize()
            def no_config_function():
                return "test"

    def test_multiple_decorated_functions(self):
        """Test that multiple functions can be decorated independently."""

        @optimize(configuration_space={"param1": [1, 2, 3]})
        def function1(param1: int) -> int:
            return param1 * 2

        @optimize(configuration_space={"param2": ["a", "b", "c"]})
        def function2(param2: str) -> str:
            return param2.upper()

        assert isinstance(function1, OptimizedFunction)
        assert isinstance(function2, OptimizedFunction)
        assert function1 != function2

    def test_decorator_with_constraints(self):
        """Test decorator with constraint functions."""

        def constraint_func(config):
            return config["x"] < config["y"]

        @optimize(
            configuration_space={"x": [1, 2, 3], "y": [2, 3, 4]},
            constraints=[constraint_func],
        )
        def constrained_function(x: int, y: int) -> int:
            return x + y

        assert isinstance(constrained_function, OptimizedFunction)

    def test_decorator_loads_tvl_spec(self):
        """Supplying tvl_spec hydrates the configuration automatically."""

        spec_path = Path(
            "docs/tvl/tvl-website/client/public/examples/ch2_hello_tvl.tvl.yml"
        )

        @optimize(tvl_spec=spec_path)
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert "model" in tvl_wrapped.configuration_space
        assert tvl_wrapped.configuration_space["max_tokens"] == [256, 384, 512]

    def test_decorator_wires_evaluation_set_dataset(self, tmp_path):
        """TVL 0.9 evaluation_set.dataset populates eval_dataset when omitted."""
        spec_path = tmp_path / "evalset.tvl.yml"
        spec_path.write_text(
            """tvl:
  module: test.evalset
tvl_version: "0.9"
evaluation_set:
  dataset: test.jsonl
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]
objectives:
  - name: accuracy
    direction: maximize
"""
        )

        @optimize(tvl_spec=spec_path)
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert tvl_wrapped.eval_dataset == "test.jsonl"

    def test_decorator_does_not_override_explicit_eval_dataset(self, tmp_path):
        """Explicit eval_dataset beats evaluation_set in a TVL spec."""
        spec_path = tmp_path / "evalset_override.tvl.yml"
        spec_path.write_text(
            """tvl:
  module: test.evalset.override
tvl_version: "0.9"
evaluation_set:
  dataset: spec.jsonl
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]
objectives:
  - name: accuracy
    direction: maximize
"""
        )

        @optimize(tvl_spec=spec_path, eval_dataset="user.jsonl")
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert tvl_wrapped.eval_dataset == "user.jsonl"


class TestOptimizedFunctionIntegration:
    """Integration tests for OptimizedFunction wrapper."""

    def test_optimized_function_has_optimization_methods(self):
        """Test that OptimizedFunction has required optimization methods."""

        @optimize(configuration_space={"param": [1, 2, 3]})
        def test_function(param: int) -> int:
            return param**2

        # Should have optimization-related methods
        assert hasattr(test_function, "optimize")
        assert hasattr(test_function, "run")

        # Methods should be callable
        assert callable(test_function.optimize)
        assert callable(test_function.run)

    def test_function_call_vs_run_method(self):
        """Test difference between direct call and run method."""

        @optimize(configuration_space={"multiplier": [2, 3, 4]})
        def calc_function(value: int, multiplier: int = 2) -> int:
            return value * multiplier

        # Direct call should work
        direct_result = calc_function(5, multiplier=3)
        assert direct_result == 15

        # Run method should also work
        run_result = calc_function.run(5, multiplier=4)
        assert run_result == 20

    def test_optimization_context_preservation(self):
        """Test that optimization context is preserved across calls."""

        @optimize(
            configuration_space={"mode": ["fast", "accurate"]},
            objectives=["speed", "quality"],
        )
        def contextual_function(data: str, mode: str = "fast") -> dict:
            return {"data": data, "mode": mode, "result": len(data)}

        # Should preserve configuration across multiple calls
        result1 = contextual_function("test1", mode="fast")
        result2 = contextual_function("test2", mode="accurate")

        assert result1["mode"] == "fast"
        assert result2["mode"] == "accurate"

    def test_async_function_decoration(self):
        """Test decoration of async functions."""

        @optimize(configuration_space={"delay": [0.01, 0.02, 0.05]})
        async def async_func(value: int, delay: float = 0.01) -> int:
            await asyncio.sleep(delay)
            return value * 2

        assert isinstance(async_func, OptimizedFunction)
        # Test execution
        result = asyncio.run(async_func(5))
        assert result == 10

    def test_decorator_with_all_parameters(self):
        """Test decorator with all available parameters."""
        mock_dataset = Mock(spec=Dataset)
        mock_constraint = Mock()

        @optimize(
            eval_dataset=mock_dataset,
            objectives=["accuracy", "cost"],
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            default_config={"model": "gpt-3.5"},
            constraints=[mock_constraint],
            injection_mode="parameter",
            config_param="llm_config",
            execution_mode="cloud",
            auto_override_frameworks=False,
            framework_targets=["openai.OpenAI"],
            custom_param="test_value",
        )
        def full_function(query: str, llm_config: dict = None) -> str:
            return f"Processing: {query}"

        assert isinstance(full_function, OptimizedFunction)

    def test_dataset_parameter_variations(self):
        """Test different types of dataset parameters."""

        # String dataset
        @optimize(eval_dataset="data.jsonl", configuration_space={"x": [1, 2, 3]})
        def func1(x):
            return x

        assert isinstance(func1, OptimizedFunction)

        # List of strings
        @optimize(
            eval_dataset=["data1.jsonl", "data2.jsonl"],
            configuration_space={"x": [1, 2, 3]},
        )
        def func2(x):
            return x

        assert isinstance(func2, OptimizedFunction)

        # Dataset object
        mock_dataset = Mock(spec=Dataset)

        @optimize(eval_dataset=mock_dataset, configuration_space={"x": [1, 2, 3]})
        def func3(x):
            return x

        assert isinstance(func3, OptimizedFunction)

    def test_injection_modes(self):
        """Test different injection modes."""

        # Context injection (default)
        @optimize(injection_mode="context", configuration_space={"x": [1, 2, 3]})
        def func_context(x):
            return x

        assert isinstance(func_context, OptimizedFunction)

        # Parameter injection
        @optimize(
            injection_mode="parameter",
            config_param="config",
            configuration_space={"x": [1, 2, 3]},
        )
        def func_param(x, config=None):
            return x

        assert isinstance(func_param, OptimizedFunction)

        # Decorator injection
        @optimize(injection_mode="attribute", configuration_space={"x": [1, 2, 3]})
        def func_decorator(x):
            return x

        assert isinstance(func_decorator, OptimizedFunction)

    def test_framework_override_configuration(self):
        """Test framework override configuration."""

        # Default auto_override_frameworks=True
        @optimize(configuration_space={"x": [1, 2, 3]})
        def func1(x):
            return x

        assert isinstance(func1, OptimizedFunction)

        # Disable framework override
        @optimize(auto_override_frameworks=False, configuration_space={"x": [1, 2, 3]})
        def func2(x):
            return x

        assert isinstance(func2, OptimizedFunction)

        # Specific framework targets
        @optimize(
            framework_targets=["openai.OpenAI", "langchain.ChatOpenAI"],
            configuration_space={"x": [1, 2, 3]},
        )
        def func3(x):
            return x

        assert isinstance(func3, OptimizedFunction)

    @patch("traigent.api.decorators.logger")
    def test_logging(self, mock_logger):
        """Test that appropriate logging occurs."""

        @optimize(configuration_space={"x": [1, 2, 3]})
        def test_func(x):
            return x

        # Check debug log for decoration
        mock_logger.debug.assert_any_call(
            "Decorating function test_func with @traigent.optimize"
        )

        # Check info log for creation
        mock_logger.info.assert_called_with("Created optimizable function: test_func")

    def test_decorator_factory_pattern(self):
        """Test that optimize returns a decorator function."""
        decorator = optimize(
            objectives=["accuracy"], configuration_space={"x": [1, 2, 3]}
        )
        assert callable(decorator)

        # Apply the decorator
        def my_func(x):
            return x

        decorated = decorator(my_func)
        assert isinstance(decorated, OptimizedFunction)

    def test_class_method_decoration(self):
        """Test decoration of class methods."""

        class MyClass:
            def __init__(self):
                self.value = 2

            @optimize(configuration_space={"x": [2, 3, 4]})
            def method(self, x: int) -> int:
                return x * self.value

        # Static and classmethod don't work well with the decorator
        # because they need to be applied after @optimize

        @optimize(configuration_space={"x": [3, 4, 5]})
        def static_func(x: int) -> int:
            return x * 3

        # Create instance and test
        obj = MyClass()

        # The method is bound, so we test it exists
        assert hasattr(obj, "method")
        assert callable(obj.method)

        # Test the standalone function
        assert isinstance(static_func, OptimizedFunction)
        assert static_func(5) == 15
