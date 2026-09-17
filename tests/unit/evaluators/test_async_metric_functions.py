from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import EvaluationError


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


@pytest.mark.asyncio
async def test_mapping_metric_closes_later_coroutines_on_objective_failure(
    recwarn,
) -> None:
    """If an objective mapping sub-value's coroutine raises, later coroutine
    sub-values in the same mapping must still be closed, not left unawaited.

    Regression for #1801 (PR #1794 follow-up): ``_apply_custom_metric_functions``
    used to exit its ``value.items()`` loop immediately on the first raise,
    stranding any later awaitable sub-values -- a
    "coroutine ... was never awaited" leak.
    """
    later_awaited = False

    async def _bad() -> float:
        await asyncio.sleep(0)
        raise ValueError("boom")

    async def _later() -> float:
        nonlocal later_awaited
        await asyncio.sleep(0)
        later_awaited = True
        return 0.25

    def combo(output: str, expected: str) -> dict:
        return {"quality": _bad(), "other": _later()}

    def agent(text: str) -> str:
        return text

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"text": "YES"}, expected_output="YES")],
        name="mapping_failure_coroutine_leak",
    )
    evaluator = LocalEvaluator(
        metrics=["quality"],
        metric_functions={"combo": combo},
        detailed=True,
        execution_mode="local",
    )

    with pytest.raises(EvaluationError):
        await evaluator.evaluate(agent, {}, dataset)

    await asyncio.sleep(0)
    gc.collect()

    unawaited_warnings = [
        warning for warning in recwarn if "was never awaited" in str(warning.message)
    ]
    assert unawaited_warnings == []
    # ``_later`` was never actually awaited by application code (the
    # objective failure raises fail-closed before any value it would have
    # produced is used) -- it is closed, not silently run to completion.
    assert later_awaited is False


@pytest.mark.asyncio
async def test_progress_callback_skips_and_closes_awaitable_metric(recwarn) -> None:
    """The best-effort progress payload (built via
    ``BaseEvaluator._build_progress_payload`` -> ``_call_progress_metric_functions``,
    fired once per example from ``_evaluate_single_detailed`` before
    ``LocalEvaluator``'s own second, fully-async metrics pass runs -- via
    ``_invoke_metric_function``, since ``LocalEvaluator`` has no
    ``_call_metric_functions`` of its own) must skip an awaitable metric
    value -- never coerce/``float()`` a raw coroutine -- and close that
    coroutine so it never leaks an unawaited-coroutine warning. The final
    detailed result (LocalEvaluator's own second pass) still carries the
    fully-resolved value.

    Regression for #1801 (PR #1794 follow-up): ``base.py``'s
    ``_call_progress_metric_functions`` skip/close handling for an awaitable
    progress metric value (``inspect.isawaitable(value): ... value.close()``)
    was not exercised by any test supplying a ``progress_callback`` --
    reverting it alone left the suite green.
    """

    async def _quality(output: str, expected: str) -> float:
        await asyncio.sleep(0)
        return 0.5

    def agent(text: str) -> str:
        return text

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"text": "YES"}, expected_output="YES")],
        name="progress_callback_awaitable_metric",
    )
    evaluator = LocalEvaluator(
        metrics=["quality"],
        metric_functions={"quality": _quality},
        detailed=True,
        execution_mode="local",
    )

    progress_payloads: list[dict[str, Any]] = []

    def progress_callback(index: int, payload: dict[str, Any]) -> None:
        progress_payloads.append(payload)

    result = await evaluator.evaluate(
        agent, {}, dataset, progress_callback=progress_callback
    )

    await asyncio.sleep(0)
    gc.collect()

    unawaited_warnings = [
        warning for warning in recwarn if "was never awaited" in str(warning.message)
    ]
    assert unawaited_warnings == []

    # LocalEvaluator fires progress_callback twice per example: first from
    # the shared base-class per-example executor's best-effort payload
    # (the one under test here), then again from LocalEvaluator's own
    # second pass once the metric function has actually been awaited.
    assert len(progress_payloads) >= 1, "progress_callback was never invoked"
    best_effort_payload = progress_payloads[0]
    # The awaitable metric is skipped for the best-effort progress payload
    # (never coerced/float()'d as a raw coroutine) -- it produces no
    # "quality" entry at all (an empty metrics_payload never sets the
    # "metrics" key).
    assert "quality" not in best_effort_payload.get("metrics", {})

    # The final (fully async-resolved) result still has the real value.
    assert result.example_results[0].metrics["quality"] == pytest.approx(0.5)
