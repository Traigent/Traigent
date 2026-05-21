"""Integration-level execution-mode contract tests for the decorator."""

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
    """Reserved cloud mode fails closed on the decorator surface."""

    def test_basic_decorator_with_cloud_mode_fails_closed(self):
        """A cloud request should not produce a local OptimizedFunction."""
        with pytest.raises(ConfigurationError, match="not available yet"):

            @optimize(
                eval_dataset=_dataset(),
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o-mini"]},
                execution_mode="cloud",
            )
            def answer(question: str) -> str:
                return question

    def test_execution_bundle_with_cloud_mode_fails_closed(self):
        """ExecutionOptions follows the same cloud contract."""
        with pytest.raises(ConfigurationError, match="not available yet"):

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

    def test_standard_mode_fails_closed(self):
        """The removed standard mode is rejected consistently."""
        with pytest.raises(ConfigurationError, match="No such mode 'standard'"):

            @optimize(
                eval_dataset=_dataset(),
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o-mini"]},
                execution_mode="standard",
            )
            def answer(question: str) -> str:
                return question

    def test_privacy_alias_maps_to_hybrid_with_privacy_enabled(self):
        """Privacy remains a compatibility alias for hybrid."""

        @optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space={"model": ["gpt-4o-mini"]},
            execution_mode="privacy",
        )
        def answer(question: str) -> str:
            return question

        assert isinstance(answer, OptimizedFunction)
        assert answer.execution_mode == ExecutionMode.HYBRID.value
        assert answer._effective_execution_mode is ExecutionMode.HYBRID
        assert answer.privacy_enabled is True

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
