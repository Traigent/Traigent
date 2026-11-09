"""Traigent#3251: a captured response without usable usage is unknown, not measured.

In the custom_evaluator lane the wrapper hands the prompt and answer text to
``extract_llm_metrics`` for the first captured call. When the provider returned
no usage (or null token counts), that text must never be turned into a
length-based cost that is then reported as a measurement with 0 tokens.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from traigent.api.types import ExampleResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.metrics_tracker import extract_llm_metrics

PROMPT = "What is the capital of France? " * 10
ANSWER = "Paris is the capital of France. " * 10


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "false")
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.delenv("TRAIGENT_RUN_COST_LIMIT", raising=False)


def _openai_no_usage() -> Any:
    return SimpleNamespace(model="gpt-4o-mini", usage=None)


def _openai_null_tokens() -> Any:
    return SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=None, completion_tokens=None, total_tokens=None
        ),
    )


def _langchain_no_usage() -> Any:
    return SimpleNamespace(content="ok", usage_metadata=None, response_metadata={})


def _litellm_no_usage() -> Any:
    litellm = pytest.importorskip("litellm")
    response = litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[{"message": {"role": "assistant", "content": "ok"}}],
    )
    if hasattr(response, "usage"):
        try:
            delattr(response, "usage")
        except AttributeError:
            pass
    return response


NO_USAGE = [
    pytest.param(_openai_no_usage, id="openai-usage-none"),
    pytest.param(_openai_null_tokens, id="openai-null-tokens"),
    pytest.param(_langchain_no_usage, id="langchain-no-usage"),
    pytest.param(_litellm_no_usage, id="litellm-no-usage"),
]


@pytest.mark.parametrize("build", NO_USAGE)
def test_captured_response_without_usage_is_unmeasured_even_with_text(
    build: Any,
) -> None:
    """The exact wrapper call: captured response plus prompt and answer text."""
    metrics = extract_llm_metrics(
        build(),
        model_name="gpt-4o-mini",
        original_prompt=PROMPT,
        response_text=ANSWER,
        estimate_from_text=False,
    )
    assert metrics.tokens.input_tokens == 0
    assert metrics.tokens.output_tokens == 0
    assert metrics.cost.total_cost == 0
    assert metrics.cost.unmeasured is True
    assert metrics.cost_measured is False


def test_captured_response_with_valid_usage_still_priced_as_measured() -> None:
    response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )
    metrics = extract_llm_metrics(
        response,
        model_name="gpt-4o-mini",
        original_prompt=PROMPT,
        response_text=ANSWER,
        estimate_from_text=False,
    )
    assert (metrics.tokens.input_tokens, metrics.tokens.output_tokens) == (11, 7)
    assert metrics.cost.total_cost > 0
    assert metrics.cost.unmeasured is False
    assert metrics.cost_measured is True


def _completion_json(usage: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ANSWER},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is not ...:
        body["usage"] = usage
    return body


USAGES = {
    "missing": ...,
    "null-tokens": {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    },
    "valid": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}


def _run(usage: Any, *, objectives: list[str]):
    openai = pytest.importorskip("openai")
    from traigent.utils.langchain_interceptor import (
        capture_langchain_response,
        instrumented_provider_call,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_completion_json(usage))
    )
    client = openai.OpenAI(
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://gateway.invalid/v1",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )

    def agent(question: str) -> str:
        with instrumented_provider_call():
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": PROMPT + question}],
            )
        capture_langchain_response(response)
        return response.choices[0].message.content

    async def evaluator(func, config, example):
        output = func(**example.input_data)
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt = OptimizedFunction(
        func=agent,
        configuration_space={"temperature": [0.0]},
        objectives=objectives,
        eval_dataset=Dataset(
            examples=[
                EvaluationExample(
                    input_data={"question": f"q{i}"}, expected_output="paris"
                )
                for i in range(2)
            ],
            name="no-usage-3251",
        ),
        custom_evaluator=evaluator,
    )
    return opt.optimize_sync(algorithm="grid", max_trials=1, progress_bar=False)


@pytest.mark.parametrize("kind", ["missing", "null-tokens"])
def test_no_usage_aborts_under_a_cost_objective(kind: str) -> None:
    with pytest.raises(Exception, match="Cost extraction failed|unpriced"):
        _run(USAGES[kind], objectives=["accuracy", "cost"])


def test_valid_usage_completes_under_a_cost_objective() -> None:
    result = _run(USAGES["valid"], objectives=["accuracy", "cost"])
    (trial,) = result.trials
    assert trial.metrics["input_tokens"] == 22
    assert trial.metrics["output_tokens"] == 14
    assert trial.metrics["cost"] > 0
    assert not trial.metrics.get("cost_unmeasured")


@pytest.mark.parametrize("kind", ["missing", "null-tokens"])
def test_no_usage_aborts_under_strict_accounting_without_cost_objective(
    kind: str,
) -> None:
    """Strict accounting (the default) refuses unknown spend even when only
    accuracy is optimised: it is never recorded as a measured $0."""
    with pytest.raises(Exception, match="Cost extraction failed|unpriced"):
        _run(USAGES[kind], objectives=["accuracy"])


def test_no_text_estimate_flag_grants_no_provider_billed_trust() -> None:
    """``estimate_from_text=False`` is not ``response_captured``: a lookalike with
    a self-reported zero charge must stay self-reported (#2505)."""
    from traigent.evaluators.metrics_tracker import COST_SOURCE_PROVIDER_BILLED

    lookalike = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=1000, completion_tokens=1000, total_tokens=2000, cost=0.0
        ),
        _hidden_params={"custom_llm_provider": "openrouter"},
    )
    metrics = extract_llm_metrics(
        lookalike,
        model_name="gpt-4o-mini",
        original_prompt=PROMPT,
        response_text=ANSWER,
        estimate_from_text=False,
    )
    assert metrics.cost.cost_source != COST_SOURCE_PROVIDER_BILLED
