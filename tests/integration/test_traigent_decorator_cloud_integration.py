"""Integration-level execution-mode contract tests for the decorator."""

import warnings

import pytest

from traigent.api.decorators import ExecutionOptions, optimize
from traigent.config.types import ExecutionMode
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ConfigurationError


def _dataset() -> Dataset:
    return Dataset(
        name="math_qa",
        examples=[
            EvaluationExample(
                input_data={"question": "What is 2+2?"},
                expected_output="4",
            )
        ],
    )


class TestTraigentDecoratorCloudIntegration:
    """Deprecated cloud fails closed; standard still resolves to hybrid."""

    def test_basic_decorator_with_cloud_mode_fails_closed(self):
        """A cloud request raises before decorator construction."""
        with pytest.raises(ConfigurationError, match="fails closed"):

            @optimize(
                eval_dataset=_dataset(),
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o-mini"]},
                execution_mode="cloud",
            )
            def answer(question: str) -> str:
                return question

    def test_execution_bundle_with_cloud_mode_fails_closed(self):
        """ExecutionOptions with cloud raises before decorator construction."""
        with pytest.raises(ConfigurationError, match="fails closed"):

            @optimize(
                eval_dataset=_dataset(),
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o-mini"]},
                execution=ExecutionOptions(
                    execution_mode="cloud",
                    cloud_fallback_policy="auto",
                ),
            )
            def answer(question: str) -> str:
                return question

    def test_standard_mode_deprecated_resolves_to_hybrid(self):
        """The deprecated standard mode resolves to hybrid with DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @optimize(
                eval_dataset=_dataset(),
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o-mini"]},
                execution_mode="standard",
            )
            def answer(question: str) -> str:
                return question

        assert isinstance(answer, OptimizedFunction)
        assert answer.execution_mode == ExecutionMode.HYBRID.value
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_privacy_alias_fails_closed(self):
        """Privacy raises before decorator construction."""
        with pytest.raises(ConfigurationError, match="fails closed"):

            @optimize(
                eval_dataset=_dataset(),
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o-mini"]},
                execution_mode="privacy",
            )
            def answer(question: str) -> str:
                return question

    def test_hybrid_mode_still_constructs(self):
        """Hybrid is the supported portal-tracked local mode."""

        @optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space={"model": ["gpt-4o-mini"]},
            execution_mode="hybrid",
        )
        def answer(question: str) -> str:
            return question

        assert isinstance(answer, OptimizedFunction)
        assert answer.execution_mode == ExecutionMode.HYBRID.value
