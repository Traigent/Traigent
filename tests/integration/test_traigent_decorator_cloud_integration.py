"""Integration-level execution-mode contract tests for the decorator."""

import warnings

from traigent.api.decorators import ExecutionOptions, optimize
from traigent.config.types import ExecutionMode
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample


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
    """Deprecated cloud mode resolves to edge_analytics; standard mode resolves to hybrid."""

    def test_basic_decorator_with_cloud_mode_deprecated(self):
        """A cloud request emits DeprecationWarning and resolves to edge_analytics."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @optimize(
                eval_dataset=_dataset(),
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o-mini"]},
                execution_mode="cloud",
            )
            def answer(question: str) -> str:
                return question

        assert isinstance(answer, OptimizedFunction)
        assert answer.execution_mode == ExecutionMode.EDGE_ANALYTICS.value
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_execution_bundle_with_cloud_mode_deprecated(self):
        """ExecutionOptions with cloud emits DeprecationWarning and resolves to edge_analytics."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

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

        assert isinstance(answer, OptimizedFunction)
        assert answer.execution_mode == ExecutionMode.EDGE_ANALYTICS.value
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

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
