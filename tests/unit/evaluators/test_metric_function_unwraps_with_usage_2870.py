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

    ctx_reset = trial_context.set({"trial_id": "t-2870"})
    try:
        await evaluator.evaluate(agent, {}, _dataset())
    finally:
        trial_context.reset(ctx_reset)

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


@pytest.mark.asyncio
async def test_progress_payload_metric_sees_unwrapped_output() -> None:
    """The running score sent with progress updates uses the unwrapped output too.

    Cloud smart pruning acts on these progress scores, so a wrapper-shaped
    input would score 0.0 here while the final aggregate scores 1.0.
    """

    def recall_at_budget(output: Any, expected: Any) -> float:
        return len(set(output) & set(expected)) / len(expected)

    evaluator = LocalEvaluator(
        metrics=["recall_at_budget"],
        metric_functions={"recall_at_budget": recall_at_budget},
        detailed=True,
    )

    payloads: list[dict[str, Any]] = []

    def progress_callback(_index: int, payload: dict[str, Any]) -> None:
        payloads.append(payload)

    def agent(_x: Any) -> Any:
        return with_usage(["t1", "t2"], total_cost=0.002)

    ctx_reset = trial_context.set({"trial_id": "t-2870-progress"})
    try:
        await evaluator.evaluate(
            agent, {}, _dataset(), progress_callback=progress_callback
        )
    finally:
        trial_context.reset(ctx_reset)

    metric_values = [
        payload["metrics"]["recall_at_budget"]
        for payload in payloads
        if "recall_at_budget" in payload.get("metrics", {})
    ]
    assert metric_values, "progress callback never reported the metric"
    assert metric_values == [1.0] * len(metric_values)


@pytest.mark.asyncio
async def test_simple_scoring_evaluator_metric_function_receives_unwrapped_output() -> (
    None
):
    """SimpleScoringEvaluator's metric functions get the unwrapped output too."""
    from traigent.evaluators.base import SimpleScoringEvaluator

    seen: list[Any] = []

    def recall_at_budget(output: Any, expected: Any) -> float:
        seen.append(output)
        return len(set(output) & set(expected)) / len(expected)

    evaluator = SimpleScoringEvaluator(
        metric_functions={"recall_at_budget": recall_at_budget},
        metrics=["recall_at_budget"],
    )

    def agent(*_args: Any, **_kwargs: Any) -> Any:
        return with_usage(["t1", "t2"], total_cost=0.002)

    ctx_reset = trial_context.set({"trial_id": "t-2870-simple"})
    try:
        await evaluator.evaluate(agent, {}, _dataset())
    finally:
        trial_context.reset(ctx_reset)

    assert seen == [["t1", "t2"]]


@pytest.mark.asyncio
async def test_simple_scoring_evaluator_scoring_function_receives_unwrapped_output() -> (
    None
):
    """SimpleScoringEvaluator's single scoring_function gets the unwrapped value."""
    from traigent.evaluators.base import SimpleScoringEvaluator

    seen: list[Any] = []

    def score(output: Any, expected: Any) -> float:
        seen.append(output)
        return 1.0

    evaluator = SimpleScoringEvaluator(scoring_function=score)

    def agent(*_args: Any, **_kwargs: Any) -> Any:
        return with_usage(["t1", "t2"], total_cost=0.002)

    ctx_reset = trial_context.set({"trial_id": "t-2870-scoring"})
    try:
        await evaluator.evaluate(agent, {}, _dataset())
    finally:
        trial_context.reset(ctx_reset)

    assert seen == [["t1", "t2"]]
