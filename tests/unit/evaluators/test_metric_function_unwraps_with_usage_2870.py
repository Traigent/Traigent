"""Custom metric functions see the function's real output, not the with_usage() wrapper (Traigent#2870)."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.api.functions import with_usage
from traigent.config.context import trial_context
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def _dataset() -> Dataset:
    return Dataset([EvaluationExample({"q": "a"}, ["t1", "t2"])], name="usage")


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_metric_function_receives_unwrapped_output(detailed: bool) -> None:
    seen: list[Any] = []

    def recall_at_budget(output: Any, expected: Any) -> float:
        seen.append(output)
        return len(set(output) & set(expected)) / len(expected)

    evaluator = LocalEvaluator(
        metrics=["recall_at_budget"],
        metric_functions={"recall_at_budget": recall_at_budget},
        detailed=detailed,
    )

    raw: list[Any] = []

    def agent(_x: Any) -> Any:
        out = with_usage(["t1", "t2"], total_cost=0.002)
        raw.append(out)
        return out

    token = trial_context.set({"trial_id": "t-2870"})
    try:
        await evaluator.evaluate(agent, {}, _dataset())
    finally:
        trial_context.reset(token)

    assert "__traigent_meta__" in raw[0]  # the wrapper really was in play
    assert seen == [["t1", "t2"]]


@pytest.mark.asyncio
async def test_plain_dict_with_text_key_is_not_unwrapped() -> None:
    seen: list[Any] = []

    def metric(output: Any, expected: Any) -> float:
        seen.append(output)
        return 1.0

    evaluator = LocalEvaluator(metrics=["m"], metric_functions={"m": metric})
    plain = {"text": "x", "other": 1}
    await evaluator.evaluate(lambda _x: dict(plain), {}, _dataset())

    assert seen == [plain]
