import asyncio
from typing import Any

import pytest

from traigent.api.types import ExampleResult
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import TrialPrunedError


@pytest.mark.asyncio
async def test_local_evaluator_parallel_propagates_trial_pruned_error_from_progress_callback():
    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=4, detailed=True)
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(10)
        ],
        name="pruning-parallel",
        description="Pruning propagation in parallel evaluation",
    )

    async def identity(value: int) -> int:
        await asyncio.sleep(0.01)
        return value

    did_prune = False

    def progress_callback(_step: int, _payload: dict[str, Any]) -> None:
        nonlocal did_prune
        if not did_prune:
            did_prune = True
            raise TrialPrunedError(step=0)

    with pytest.raises(TrialPrunedError):
        await evaluator.evaluate(
            identity, {}, dataset, progress_callback=progress_callback
        )


@pytest.mark.asyncio
async def test_custom_evaluator_propagates_trial_pruned_error_from_progress_callback():
    async def custom_eval(
        _func: Any, _config: dict[str, Any], example: EvaluationExample
    ) -> ExampleResult:
        return ExampleResult(
            example_id="example",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=example.expected_output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
        )

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        max_workers=1,
        detailed=True,
        custom_eval_func=custom_eval,
    )
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
        name="pruning-custom-eval",
        description="Pruning propagation in custom evaluator",
    )

    def progress_callback(_step: int, _payload: dict[str, Any]) -> None:
        raise TrialPrunedError(step=0)

    with pytest.raises(TrialPrunedError):
        await evaluator.evaluate(
            lambda value: value, {}, dataset, progress_callback=progress_callback
        )
