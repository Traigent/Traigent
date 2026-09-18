"""Regression test for Traigent#2387.

Concurrent trials must each be charged their OWN judge spend.

``LangChainMetadataCapture`` kept every captured LLM response in one
process-global list with no trial ownership, while ``optimize(...)`` gathers
trial coroutines on a single event loop
(``core/parallel_execution_manager.py``, ``asyncio.gather`` under a semaphore of
``parallel_trials``).  A judge call made by trial A could therefore be drained
-- and charged -- by trial B at
``LocalEvaluator._fold_metric_function_llm_cost``.

That matters beyond reporting: ``cost`` is a selectable minimize-objective
(``core/objectives.py``), so contaminated per-trial cost can decide which
configuration the optimizer returns.  Run-level totals were never wrong -- each
response was counted exactly once, just possibly against the wrong trial.

``threading.local()`` is no defence here: concurrent trials are coroutines
sharing one thread, so they share the thread-local too.  Ownership has to come
from a ``ContextVar`` scope entered per trial.
"""

from __future__ import annotations

import asyncio

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.langchain_interceptor import capture_scope


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _DummyRawResp:
    """A minimal raw_response-shaped object (mirrors an OpenAI ChatCompletion)."""

    def __init__(self) -> None:
        self.model = "gpt-4o-mini"
        self.usage = _Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)


def _make_dataset(name: str) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": "Q0"},
                expected_output="ok",
                metadata={"example_id": f"{name}_example_0"},
            )
        ],
        name=name,
    )


def _agent(text: str, **config: object) -> dict:
    """An agent whose own spend arrives on the output, not via the interceptor."""
    return {"text": "ok", "raw_response": _DummyRawResp()}


def _one_judge_call() -> None:
    import litellm

    litellm.completion(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "judge this"}],
    )


def _make_interleaving_judge(mine: asyncio.Event, theirs: asyncio.Event):
    """A judge that parks with its response in the buffer while the other runs.

    This is the interleaving the event loop can produce on its own whenever a
    judge awaits anything (an async LLM client, a rate-limit sleep); the events
    only make it deterministic instead of timing-dependent.
    """

    async def judge(output: object, expected: object, input_data: object) -> float:
        _one_judge_call()
        mine.set()
        await asyncio.wait_for(theirs.wait(), timeout=10)
        return 1.0

    return judge


def _evaluator(judge) -> LocalEvaluator:
    return LocalEvaluator(
        metrics=["cost"],
        metric_functions={"judge": judge},
        detailed=True,
    )


async def _solo_reference_cost() -> tuple[float, float]:
    """Cost and judge share for one trial running entirely alone."""

    async def judge(output: object, expected: object, input_data: object) -> float:
        _one_judge_call()
        return 1.0

    result = await _evaluator(judge).evaluate(_agent, {}, _make_dataset("solo"))

    judge_share = sum(
        ex.metrics.get("evaluation_cost", 0.0) for ex in result.example_results
    )
    return result.metrics["cost"], judge_share


@pytest.mark.asyncio
async def test_concurrent_trials_are_not_charged_each_others_judge_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two interleaved trials must each pay for exactly one judge call (#2387)."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    solo_cost, solo_judge_share = await _solo_reference_cost()
    assert solo_judge_share > 0.0, "the judge call must cost something to attribute"

    a_captured, b_captured = asyncio.Event(), asyncio.Event()

    async def run_trial(name: str, mine: asyncio.Event, theirs: asyncio.Event):
        # NOTE: no capture_scope() here on purpose. The product must isolate
        # these itself -- if this test opened the scope it would be testing its
        # own setup rather than the fix.
        return await _evaluator(_make_interleaving_judge(mine, theirs)).evaluate(
            _agent, {}, _make_dataset(name)
        )

    result_a, result_b = await asyncio.gather(
        run_trial("trial_a", a_captured, b_captured),
        run_trial("trial_b", b_captured, a_captured),
    )

    for name, result in (("trial_a", result_a), ("trial_b", result_b)):
        judge_share = sum(
            ex.metrics.get("evaluation_cost", 0.0) for ex in result.example_results
        )
        assert judge_share == pytest.approx(solo_judge_share), (
            f"{name} was charged {judge_share} of judge spend but made exactly one "
            f"judge call worth {solo_judge_share} -- a concurrently running trial's "
            "judge call leaked into its ledger (Traigent#2387)"
        )
        assert result.metrics["cost"] == pytest.approx(solo_cost), (
            f"{name} total cost {result.metrics['cost']} != solo-run cost "
            f"{solo_cost} (Traigent#2387)"
        )


@pytest.mark.asyncio
async def test_capture_scope_isolates_concurrent_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scope itself must not leak responses between concurrent tasks."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    from traigent.utils.langchain_interceptor import (
        capture_langchain_response,
        get_all_captured_responses,
    )

    first_captured = asyncio.Event()
    drained: dict[str, list[str]] = {}

    async def task(name: str, first: bool) -> None:
        async with capture_scope():
            capture_langchain_response({"owner": name})
            if first:
                first_captured.set()
            else:
                await asyncio.wait_for(first_captured.wait(), timeout=10)
            await asyncio.sleep(0)
            drained[name] = [r["owner"] for r in get_all_captured_responses()]

    await asyncio.gather(task("A", True), task("B", False))

    assert drained["A"] == ["A"], f"trial A drained {drained['A']}"
    assert drained["B"] == ["B"], f"trial B drained {drained['B']}"
