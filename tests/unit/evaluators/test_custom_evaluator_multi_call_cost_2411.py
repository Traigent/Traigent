"""Traigent#2411: custom_evaluator rows meter EVERY LLM call, not the first.

Keyless and offline. LiteLLM calls use ``mock_response`` (a real
``ModelResponse`` with usage 10/20 tokens, priced from litellm's bundled
table); the LangChain/OpenAI pair uses a real client on ``httpx.MockTransport``.
Each test compares the trial's reported cost with the sum of
``litellm.completion_cost`` over every call the evaluator actually made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

litellm = pytest.importorskip("litellm")

from traigent.api.types import ExampleResult  # noqa: E402
from traigent.core.optimized_function import OptimizedFunction  # noqa: E402
from traigent.evaluators.base import Dataset, EvaluationExample  # noqa: E402

ROWS = 3


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.delenv("TRAIGENT_RUN_COST_LIMIT", raising=False)


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"question": f"q{i}"}, expected_output="paris")
            for i in range(ROWS)
        ],
        name="multi-call-cost",
    )


class _Calls:
    """Every completion the test makes, with litellm's own price for it."""

    def __init__(self) -> None:
        self.costs: list[tuple[str, float]] = []

    def call(self, model: str) -> str:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "q"}],
            mock_response="paris",
        )
        self.costs.append(
            (model, litellm.completion_cost(completion_response=response))
        )
        return response.choices[0].message.content

    @property
    def total(self) -> float:
        return sum(cost for _, cost in self.costs)


def _run(
    calls: _Calls,
    *,
    agent_model: str,
    n_agent: int,
    judge_model: str | None = None,
    max_trials: int = 1,
    temperatures: int = 1,
):
    def agent(question: str) -> str:
        return calls.call(agent_model)

    def evaluator(func, config, example):
        outputs = [func(**example.input_data) for _ in range(n_agent)]
        if judge_model:
            calls.call(judge_model)
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=outputs[0],
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt = OptimizedFunction(
        func=agent,
        configuration_space={"temperature": [i / 10 for i in range(temperatures)]},
        objectives=["accuracy", "cost"],
        eval_dataset=_dataset(),
        custom_evaluator=evaluator,
    )
    return opt.optimize_sync(
        algorithm="grid", max_trials=max_trials, progress_bar=False
    )


@pytest.mark.parametrize("n_agent", [1, 3, 5])
def test_trial_cost_sums_every_agent_call_in_the_row(n_agent: int) -> None:
    calls = _Calls()
    result = _run(calls, agent_model="gpt-4o-mini", n_agent=n_agent)
    assert len(calls.costs) == ROWS * n_agent
    assert calls.total > 0
    (trial,) = result.trials
    assert trial.metrics["cost"] == pytest.approx(calls.total)


def test_agent_and_judge_are_each_priced_at_their_own_model() -> None:
    calls = _Calls()
    result = _run(calls, agent_model="gpt-4o-mini", n_agent=1, judge_model="gpt-4o")
    # 3 rows x (gpt-4o-mini + gpt-4o) at 10/20 tokens, litellm bundled prices.
    assert calls.total == pytest.approx(0.0007155)
    (trial,) = result.trials
    assert trial.metrics["cost"] == pytest.approx(calls.total)


def test_run_cost_limit_stops_on_true_multi_call_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = 0.06
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", str(cap))
    calls = _Calls()
    # gpt-4, 5 calls per row: $0.0225 per trial of true spend.
    result = _run(calls, agent_model="gpt-4", n_agent=5, max_trials=6, temperatures=6)
    reported = sum(t.metrics["cost"] for t in result.trials)
    assert result.stop_reason == "cost_limit"
    assert len(result.trials) < 6
    assert reported == pytest.approx(calls.total)
    per_trial = calls.total / len(result.trials)
    assert calls.total <= cap + per_trial


def _completion_json(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def test_call_seen_by_provider_hook_and_framework_wrapper_counts_once() -> None:
    """An outer wrapper (ChatOpenAI.invoke / litellm.completion) records its own
    response and marks the provider call underneath; the openai SDK hook must
    not record it again, so one call is one charge in the row's cost."""
    openai = pytest.importorskip("openai")

    from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
    from traigent.integrations.framework_override import FrameworkOverrideManager
    from traigent.utils.langchain_interceptor import (
        capture_langchain_response,
        clear_captured_responses,
        instrumented_provider_call,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json=_completion_json(json.loads(request.content)["model"])
        )
    )
    client = openai.OpenAI(
        api_key="test-key",
        base_url="https://gateway.invalid/v1",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )

    def agent(question: str) -> str:
        # What an instrumented wrapper does: mark the call, make it, record it.
        with instrumented_provider_call():
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
            )
        capture_langchain_response(response)
        return response.choices[0].message.content

    async def evaluator(func, config, example):
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(**example.input_data),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    import asyncio

    manager = FrameworkOverrideManager()
    manager.activate_overrides(["openai.OpenAI"])
    clear_captured_responses()
    try:
        wrapper = CustomEvaluatorWrapper(evaluator, metrics=["accuracy"])
        result = asyncio.run(
            wrapper.evaluate(agent, {"model": "gpt-4o-mini"}, _dataset())
        )
    finally:
        manager.deactivate_overrides()
        clear_captured_responses()

    for row in result.example_results:
        assert row.metrics["input_tokens"] == 11
        assert row.metrics["output_tokens"] == 7
