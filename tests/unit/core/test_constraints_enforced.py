import pytest

from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.types import TrialResult
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationExample
from traigent.optimizers.base import BaseOptimizer


class _DummyOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__(
            config_space={"x": [2], "y": [1]},
            objectives=["accuracy"],
        )
        self._trial_count = 0
        self._suggestions_made = 0

    def suggest_next_trial(self, history: list[TrialResult]):
        self._trial_count += 1
        self._suggestions_made += 1
        return {"x": 2, "y": 1}

    def should_stop(self, history: list[TrialResult]) -> bool:
        # Constraint rejection intentionally gives back the optimizer's private
        # `_trial_count` slot. Use a separate suggestion counter here so the test
        # still verifies that the orchestrator respects optimizer stop signals.
        return self._suggestions_made >= 1


class _FailIfCalledEvaluator(BaseEvaluator):
    async def evaluate(  # type: ignore[override]
        self,
        func,
        config,
        dataset,
        **kwargs,
    ):
        raise AssertionError("Evaluator should not be called when pre-constraints fail")


@pytest.mark.asyncio
async def test_pre_constraints_prevent_evaluation() -> None:
    """Test that pre-constraints prevent evaluation without consuming trial slots.

    After fix for issue #27, constraint-rejected configs do NOT consume trial slots,
    meaning they don't create trial records. The evaluator should never be called,
    and the result should have no trials (since all configs were rejected).
    """
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"text": "hi"}, expected_output="hi")],
        name="t",
    )

    def always_false(_config):
        return False

    orchestrator = OptimizationOrchestrator(
        optimizer=_DummyOptimizer(),
        evaluator=_FailIfCalledEvaluator(),
        max_trials=1,
        callbacks=[],
        config=None,
        objectives=["accuracy"],
        constraints=[always_false],
    )

    def func(_: str) -> str:
        return "hi"

    result = await orchestrator.optimize(func=func, dataset=dataset)
    # With fix for issue #27: constraint failures don't consume trial slots,
    # so no trials are recorded (evaluator is never called)
    assert result.trials == []
    # Verify the optimization completed (optimizer's should_stop was respected)
    assert result.status.value == "completed"
