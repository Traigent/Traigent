"""Raw ``anthropic`` SDK calls must reach trial cost/token metrics (#3250).

Only the raw OpenAI clients were captured; ``anthropic.Anthropic().messages
.create`` / ``.stream`` were never recorded, so a cost-objective run aborted
at trial 1 and an accuracy-only run showed token counts estimated from text
length. These tests drive the REAL ``anthropic`` client against an
``httpx.MockTransport`` (no network, no spend).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")

# Newer anthropic releases ship their own ``httpx2`` and reject ``httpx``
# clients; older ones need ``httpx``. Use whichever the installed SDK uses.
try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover - depends on the installed SDK
    import httpx

from traigent.api.types import ExampleResult  # noqa: E402
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper  # noqa: E402
from traigent.evaluators.base import Dataset, EvaluationExample  # noqa: E402
from traigent.integrations.framework_override import (  # noqa: E402
    FrameworkOverrideManager,
)
from traigent.utils.cost_calculator import cost_from_tokens  # noqa: E402
from traigent.utils.langchain_interceptor import (  # noqa: E402
    clear_captured_responses,
    get_all_captured_responses,
)

INPUT_TOKENS = 11
OUTPUT_TOKENS = 7
MODEL = "claude-3-5-haiku-20241022"
LLM_KEYS = ("input_tokens", "output_tokens", "total_tokens", "total_cost")


def _message_body(model: str, *, usage: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }
    if usage:
        body["usage"] = {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS}
    return body


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_text(model: str, *, finish: bool) -> str:
    start = _message_body(model, usage=True)
    start["content"] = []
    start["stop_reason"] = None
    start["usage"] = {"input_tokens": INPUT_TOKENS, "output_tokens": 1}
    parts = [
        _sse("message_start", {"type": "message_start", "message": start}),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]
    if finish:
        parts.append(
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": OUTPUT_TOKENS},
                },
            )
        )
        parts.append(_sse("message_stop", {"type": "message_stop"}))
    return "".join(parts)


def _handler(*, usage: bool = True, finish: bool = True):
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            return httpx.Response(
                200,
                text=_stream_text(payload["model"], finish=finish),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=_message_body(payload["model"], usage=usage))

    return handle


def _sync_client(**kw: Any) -> Any:
    return anthropic.Anthropic(
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://gateway.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(**kw))),
        max_retries=0,
    )


def _async_client(**kw: Any) -> Any:
    return anthropic.AsyncAnthropic(
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://gateway.invalid",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(**kw))),
        max_retries=0,
    )


def _dataset(n: int = 2) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"question": f"q{i}"}, expected_output="ok")
            for i in range(n)
        ],
        name="raw-anthropic",
    )


def _evaluator(metrics: list[str] | None = None) -> CustomEvaluatorWrapper:
    async def custom_evaluator(func, config, example):
        output = func(**example.input_data)
        if hasattr(output, "__await__"):
            output = await output
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0 if output == "ok" else 0.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    return CustomEvaluatorWrapper(custom_evaluator, metrics=metrics or ["accuracy"])


@pytest.fixture
def anthropic_overrides():
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["anthropic.Anthropic", "anthropic.AsyncAnthropic"])
    clear_captured_responses()
    try:
        yield manager
    finally:
        manager.deactivate_overrides()
        clear_captured_responses()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.delenv("TRAIGENT_STRICT_METRICS_NULLS", raising=False)


def _create_agent(client: Any):
    def agent(question: str) -> str:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text

    return agent


def _stream_agent(client: Any):
    def agent(question: str) -> str:
        with client.messages.stream(
            model=MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            return "".join(stream.text_stream)

    return agent


def _assert_measured(result, n: int = 2) -> None:
    in_cost, out_cost = cost_from_tokens(INPUT_TOKENS, OUTPUT_TOKENS, MODEL)
    assert in_cost + out_cost > 0, "test premise: model must be priced"
    for row in result.example_results:
        assert row.metrics["input_tokens"] == INPUT_TOKENS
        assert row.metrics["output_tokens"] == OUTPUT_TOKENS
        assert row.metadata["llm_usage_measured"] is True
    agg = result.aggregated_metrics
    assert agg["total_tokens"] == (INPUT_TOKENS + OUTPUT_TOKENS) * n
    assert agg["total_cost"] == pytest.approx((in_cost + out_cost) * n)


def _assert_unmeasured(result) -> None:
    for row in result.example_results:
        assert row.metadata["llm_usage_measured"] is False
        assert not any(key in row.metrics for key in LLM_KEYS)
    assert not any(key in result.aggregated_metrics for key in LLM_KEYS)


@pytest.mark.asyncio
async def test_raw_sync_anthropic_create_usage_is_measured(anthropic_overrides):
    result = await _evaluator().evaluate(
        _create_agent(_sync_client()), {"model": MODEL}, _dataset()
    )
    _assert_measured(result)


@pytest.mark.asyncio
async def test_raw_async_anthropic_create_usage_is_measured(anthropic_overrides):
    client = _async_client()

    async def agent(question: str) -> str:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text

    result = await _evaluator().evaluate(agent, {"model": MODEL}, _dataset())
    _assert_measured(result)


@pytest.mark.asyncio
async def test_cost_objective_completes_with_measured_cost(anthropic_overrides):
    result = await _evaluator(["accuracy", "cost"]).evaluate(
        _create_agent(_sync_client()), {"model": MODEL}, _dataset()
    )
    in_cost, out_cost = cost_from_tokens(INPUT_TOKENS, OUTPUT_TOKENS, MODEL)
    assert result.aggregated_metrics["cost"] == pytest.approx(2 * (in_cost + out_cost))


@pytest.mark.asyncio
async def test_stream_usage_is_measured(anthropic_overrides):
    result = await _evaluator().evaluate(
        _stream_agent(_sync_client()), {"model": MODEL}, _dataset()
    )
    assert [r.actual_output for r in result.example_results] == ["ok", "ok"]
    _assert_measured(result)


@pytest.mark.asyncio
async def test_async_stream_usage_is_measured(anthropic_overrides):
    client = _async_client()

    async def agent(question: str) -> str:
        async with client.messages.stream(
            model=MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            return "".join([t async for t in stream.text_stream])

    result = await _evaluator().evaluate(agent, {"model": MODEL}, _dataset())
    _assert_measured(result)


@pytest.mark.asyncio
async def test_missing_usage_is_unknown_not_a_text_estimate(anthropic_overrides):
    result = await _evaluator().evaluate(
        _create_agent(_sync_client(usage=False)), {"model": MODEL}, _dataset()
    )
    _assert_unmeasured(result)


@pytest.mark.asyncio
async def test_unfinished_stream_is_unknown_not_partial(anthropic_overrides):
    result = await _evaluator().evaluate(
        _stream_agent(_sync_client(finish=False)), {"model": MODEL}, _dataset()
    )
    _assert_unmeasured(result)


def test_capture_does_not_fire_when_overrides_are_inactive():
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["anthropic.Anthropic"])
    manager.deactivate_overrides()
    clear_captured_responses()
    _create_agent(_sync_client())("q")
    assert get_all_captured_responses() == []


@pytest.mark.asyncio
async def test_unmeasured_call_beside_a_measured_one_keeps_cost_unknown(
    anthropic_overrides,
):
    # Terra review F1: a usage-less call must not vanish from an example that
    # also made a measured call, or the measured part is published as the total.
    measured, unmeasured = _sync_client(), _sync_client(usage=False)

    def agent(question: str) -> str:
        _create_agent(unmeasured)(question)
        return _create_agent(measured)(question)

    result = await _evaluator(["accuracy", "cost"]).evaluate(
        agent, {"model": MODEL}, _dataset()
    )
    assert "cost" not in result.aggregated_metrics


@pytest.mark.asyncio
async def test_unfinished_stream_beside_a_measured_call_keeps_cost_unknown(
    anthropic_overrides,
):
    measured, partial = _sync_client(), _sync_client(finish=False)

    def agent(question: str) -> str:
        _stream_agent(partial)(question)
        return _create_agent(measured)(question)

    result = await _evaluator(["accuracy", "cost"]).evaluate(
        agent, {"model": MODEL}, _dataset()
    )
    assert "cost" not in result.aggregated_metrics


@pytest.mark.asyncio
async def test_create_stream_true_beside_a_measured_call_keeps_cost_unknown(
    anthropic_overrides,
):
    measured, streaming = _sync_client(), _sync_client()

    def agent(question: str) -> str:
        for _ in streaming.messages.create(
            model=MODEL,
            max_tokens=16,
            stream=True,
            messages=[{"role": "user", "content": question}],
        ):
            pass
        return _create_agent(measured)(question)

    result = await _evaluator(["accuracy", "cost"]).evaluate(
        agent, {"model": MODEL}, _dataset()
    )
    assert "cost" not in result.aggregated_metrics


def test_unmeasured_marker_does_not_claim_a_provider_reported_model(
    anthropic_overrides,
):
    # Terra round 2: a usage-less marker must not present the caller's
    # requested model as the model the provider served.
    _create_agent(_sync_client(usage=False))("q")
    (marker,) = get_all_captured_responses()
    assert getattr(marker, "model", None) is None
    assert getattr(marker, "usage", "absent") is None
