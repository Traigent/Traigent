"""Tests for the public cloud-mode contract."""

import pytest

import traigent
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ConfigurationError


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"text": "Test input 1"}, expected_output="output1"
        ),
        EvaluationExample(
            input_data={"text": "Test input 2"}, expected_output="output2"
        ),
    ]
    return Dataset(examples=examples, name="test_dataset")


class TestDeprecatedCloudMode:
    """Public cloud mode is deprecated and fails closed."""

    def test_decorator_with_cloud_execution_fails_closed(self):
        """The deprecated cloud mode raises before decorator construction."""
        with pytest.raises(ConfigurationError, match="fails closed"):

            @traigent.optimize(
                eval_dataset=None,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="cloud",
            )
            def test_function(input_text: str) -> str:
                return f"processed: {input_text}"

    def test_decorator_cloud_execution_with_fallback_policy_deprecated(self):
        """cloud_fallback_policy is deprecated but accepted with DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @traigent.optimize(
                eval_dataset=None,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="edge_analytics",
                cloud_fallback_policy="auto",
            )
            def test_function(input_text: str) -> str:
                return f"processed: {input_text}"

        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_optimized_function_cloud_deprecated_fails_closed(self, sample_dataset):
        """Direct OptimizedFunction with cloud mode raises before normalization."""

        def test_func(x: str) -> str:
            return x.upper()

        with pytest.raises(ConfigurationError, match="fails closed"):
            OptimizedFunction(
                func=test_func,
                eval_dataset=sample_dataset,
                objectives=["accuracy"],
                configuration_space={"param": [1, 2, 3]},
                execution_mode="cloud",
            )


class TestSupportedExecutionModes:
    """Supported decorated execution modes still behave normally."""

    def test_decorator_with_edge_execution(self):
        """Default auto execution creates a cloud-brain OptimizedFunction."""

        @traigent.optimize(
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
        )
        def test_function(input_text: str) -> str:
            return f"processed: {input_text}"

        assert isinstance(test_function, OptimizedFunction)
        assert test_function.execution_policy.intent.value == "cloud_brain"
        assert test_function.execution_mode == "hybrid"

    def test_decorator_with_hybrid_execution(self):
        """Legacy hybrid resolves to cloud-brain policy with hybrid runtime."""

        @traigent.optimize(
            eval_dataset=None,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3]},
            execution_mode="hybrid",
            injection_mode="context",
        )
        def test_function(input_text: str) -> str:
            return f"processed: {input_text}"

        assert isinstance(test_function, OptimizedFunction)
        assert test_function.execution_policy.intent.value == "cloud_brain"
        assert test_function.execution_mode == "hybrid"
        assert test_function.injection_mode == "context"
        assert test_function.objectives == ["accuracy"]

    def test_hybrid_execution_function_behavior(self, sample_dataset):
        """Supported execution modes keep normal call behavior."""

        @traigent.optimize(
            eval_dataset=sample_dataset,
            objectives=["accuracy"],
            configuration_space={"strategy": ["upper", "lower"]},
            execution_mode="hybrid",
            default_config={"strategy": "upper"},
        )
        def text_processor(input_data: dict) -> str:
            config = traigent.get_config()
            text = input_data["text"]
            if config and config.get("strategy") == "upper":
                return text.upper()
            return text.lower()

        result = text_processor({"text": "hello world"})
        assert result == "HELLO WORLD"
