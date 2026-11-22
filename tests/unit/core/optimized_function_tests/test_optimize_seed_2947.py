"""Traigent#2947: ``.optimize(seed=...)`` must seed local random search."""

import pytest

from traigent.api.types import ExampleResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample

SPACE = {
    "a": list(range(30)),
    "m": ["w", "x", "y", "z"],
}


def _evaluator(func, config, example):
    return ExampleResult(
        example_id="e1",
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output="ok",
        metrics={"accuracy": 1.0},
        execution_time=0.0,
        success=True,
        error_message=None,
    )


def _build() -> OptimizedFunction:
    def target(text: str, **_config) -> str:
        return "ok"

    return OptimizedFunction(
        func=target,
        configuration_space=SPACE,
        objectives=["accuracy"],
        eval_dataset=Dataset([EvaluationExample({"text": "hi"}, "ok")]),
        custom_evaluator=_evaluator,
        max_trials=5,
    )


async def _configs(**kwargs):
    result = await _build().optimize(
        algorithm="random", max_trials=5, progress_bar=False, **kwargs
    )
    return [dict(t.config) for t in result.trials]


@pytest.mark.asyncio
async def test_same_seed_same_config_sequence():
    first = await _configs(seed=7)
    assert len(first) == 5
    assert first == await _configs(seed=7)


@pytest.mark.asyncio
async def test_different_seed_different_config_sequence():
    assert await _configs(seed=7) != await _configs(seed=8)
