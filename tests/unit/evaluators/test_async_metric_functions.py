from __future__ import annotations

import asyncio

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def _disable_backend_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setattr(
        "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client",
        staticmethod(lambda _config: None),
    )


@pytest.mark.asyncio
async def test_local_evaluator_awaits_async_metric_function() -> None:
    calls: list[tuple[str, str]] = []

    async def async_score(output: str, expected: str) -> float:
        await asyncio.sleep(0)
        calls.append((output, expected))
        return 0.875

    def agent(text: str) -> str:
        return text

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"text": "YES"}, expected_output="YES")],
        name="async_metric_functions",
    )
    evaluator = LocalEvaluator(
        metrics=["async_score"],
        metric_functions={"async_score": async_score},
        detailed=True,
        execution_mode="local",
    )

    result = await evaluator.evaluate(agent, {}, dataset)

    assert calls == [("YES", "YES")]
    assert result.example_results[0].metrics["async_score"] == pytest.approx(0.875)
    assert result.metrics["async_score"] == pytest.approx(0.875)


@pytest.mark.asyncio
async def test_optimize_awaits_async_scoring_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_backend_tracking(monkeypatch)
    calls: list[tuple[str, str]] = []

    async def scorer(prediction: str, expected: str) -> float:
        await asyncio.sleep(0)
        calls.append((prediction, expected))
        return 0.625

    @traigent.optimize(
        eval_dataset=Dataset(
            [EvaluationExample(input_data={"text": "q"}, expected_output="YES")],
            name="async_scoring_function",
        ),
        objectives=["quality"],
        configuration_space={"style": ["plain"]},
        scoring_function=scorer,
    )
    def agent(text: str) -> str:
        return "YES"

    result = await agent.optimize(algorithm="grid", max_trials=1)

    assert calls == [("YES", "YES")]
    assert result.best_score == pytest.approx(0.625)
    assert result.trials[0].metrics["quality"] == pytest.approx(0.625)


@pytest.mark.asyncio
async def test_mapping_metric_awaits_awaitable_subvalue() -> None:
    """A metric function may return a Mapping whose sub-VALUE is awaitable
    (e.g. {"quality": async_score(...)}); it must be awaited, not passed to
    float() as a raw coroutine."""

    async def _quality() -> float:
        await asyncio.sleep(0)
        return 0.5

    def combo(output: str, expected: str) -> dict:
        return {"quality": _quality(), "plain": 0.25}

    def agent(text: str) -> str:
        return text

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"text": "YES"}, expected_output="YES")],
        name="async_mapping_subvalue",
    )
    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        metric_functions={"combo": combo},
        detailed=True,
        execution_mode="local",
    )

    result = await evaluator.evaluate(agent, {}, dataset)

    assert result.example_results[0].metrics["quality"] == pytest.approx(0.5)
    assert result.example_results[0].metrics["plain"] == pytest.approx(0.25)
