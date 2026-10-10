"""Uneven and concurrent example calls retain their own usage (#2444)."""

import asyncio

import pytest
from openai.types.chat import ChatCompletion

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.cost_calculator import cost_from_tokens
from traigent.utils.langchain_interceptor import capture_langchain_response


def _capture(model):
    capture_langchain_response(
        ChatCompletion.model_validate(
            {
                "id": "test",
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
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )
    )


@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("async_agent", [False, True])
async def test_uneven_calls_are_owned_by_their_example(detailed, workers, async_agent):
    models = {"a": ["gpt-4o-mini"], "b": ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini"]}

    def sync_agent(q):
        for model in models[q]:
            _capture(model)
        return "ok"

    async def asynchronous_agent(q):
        for model in models[q]:
            _capture(model)
            await asyncio.sleep(0)
        return "ok"

    result = await LocalEvaluator(
        metrics=["accuracy", "cost"], detailed=detailed, max_workers=workers
    ).evaluate(
        asynchronous_agent if async_agent else sync_agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"q": q}, "ok") for q in models]),
    )
    expected_cost = sum(
        sum(cost_from_tokens(10, 20, model))
        for values in models.values()
        for model in values
    )
    assert result.aggregated_metrics["total_tokens"] == 60
    assert result.aggregated_metrics["cost"] == pytest.approx(expected_cost)
    if detailed:
        assert [r.metrics["total_tokens"] for r in result.example_results] == [30, 90]
        assert [
            r.metrics["total_cost"] for r in result.example_results
        ] == pytest.approx(
            [
                sum(sum(cost_from_tokens(10, 20, model)) for model in values)
                for values in models.values()
            ]
        )


async def test_an_example_without_calls_does_not_borrow_its_neighbours_usage():
    def agent(q):
        if q == "b":
            _capture("gpt-4o-mini")
        return "ok"

    result = await LocalEvaluator(metrics=["accuracy", "cost"], detailed=True).evaluate(
        agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"q": q}, "ok") for q in ("a", "b")]),
    )
    assert result.example_results[0].metrics["total_tokens"] < 30
    assert result.example_results[0].metrics["total_cost"] is None
    assert result.example_results[1].metrics["total_tokens"] == 30


async def test_a_failed_example_keeps_calls_it_already_made():
    def agent(q):
        _capture("gpt-4o-mini")
        if q == "a":
            _capture("gpt-4o")
            raise RuntimeError("agent failed after provider calls")
        return "ok"

    result = await LocalEvaluator(metrics=["accuracy", "cost"], detailed=True).evaluate(
        agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"q": q}, "ok") for q in ("a", "b")]),
    )
    assert result.example_results[0].success is False
    assert [row.metrics["total_tokens"] for row in result.example_results] == [60, 30]


async def test_unowned_calls_fail_instead_of_guessing_an_example():
    from traigent.utils.exceptions import EvaluationError
    from traigent.utils.langchain_interceptor import capture_key

    def agent(q):
        # A caller that deliberately drops correlation still incurs spend.
        with capture_key("unowned"):
            _capture("gpt-4o-mini")
        return "ok"

    with pytest.raises(EvaluationError, match="example correlation"):
        await LocalEvaluator(metrics=["accuracy", "cost"], detailed=True).evaluate(
            agent,
            {"model": "gpt-4o-mini"},
            Dataset([EvaluationExample({"q": "a"}, "ok")]),
        )


@pytest.mark.parametrize("second_metadata", [{"example_id": "example_1"}, {}])
def test_duplicate_explicit_or_automatic_ids_are_rejected(second_metadata):
    from traigent.utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="Duplicate example_id"):
        Dataset(
            [
                EvaluationExample(
                    {"q": "a"}, "ok", metadata={"example_id": "example_1"}
                ),
                EvaluationExample({"q": "b"}, "ok", metadata=second_metadata),
            ]
        )


async def test_explicit_null_id_and_string_id_do_not_collide_with_default_ids():
    def agent(q):
        _capture("gpt-4o-mini")
        return "ok"

    rows = [
        EvaluationExample({"q": "a"}, "ok", metadata={"example_id": None}),
        EvaluationExample({"q": "b"}, "ok", metadata={"example_id": "1"}),
        EvaluationExample({"q": "c"}, "ok"),
    ]
    result = await LocalEvaluator(metrics=["accuracy", "cost"], detailed=True).evaluate(
        agent,
        {"model": "gpt-4o-mini"},
        Dataset(rows),
    )
    assert [row.metrics["total_tokens"] for row in result.example_results] == [
        30,
        30,
        30,
    ]


def test_multicall_totals_keep_measurement_and_price_provenance(monkeypatch):
    from traigent.evaluators.metrics_tracker import (
        CostMetrics,
        ExampleMetrics,
        ResponseMetrics,
        TokenMetrics,
    )

    # One reported charge plus one estimated charge is a partly estimated
    # total; an unsupported third response must not erase the known spend.
    calls = [
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=10, output_tokens=20),
            cost=CostMetrics(input_cost=1, output_cost=2, cost_explicit=True),
            response=ResponseMetrics(response_time_ms=100, tokens_per_second=300),
        ),
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=20, output_tokens=40, estimated=True),
            cost=CostMetrics(input_cost=2, output_cost=4, cost_estimated=True),
            response=ResponseMetrics(response_time_ms=100, tokens_per_second=600),
        ),
        ExampleMetrics(measured=False),
    ]
    monkeypatch.setattr(
        "traigent.evaluators.local.extract_llm_metrics",
        lambda **kwargs: calls.pop(0),
    )
    dataset = Dataset([EvaluationExample({"q": "a"}, "ok")])
    metrics = LocalEvaluator()._extract_llm_metrics_for_output(
        "ok",
        0,
        {"model": "gpt-4o-mini"},
        dataset,
        [],
        responses_by_key={"example_0": [object(), object(), object()]},
    )
    assert metrics.cost.total_cost == 9
    assert metrics.tokens.total_tokens == 90
    assert metrics.measured is True
    assert metrics.cost.cost_estimated is True
    assert metrics.cost.cost_explicit is False
    assert metrics.tokens.estimated is True
    assert metrics.response.tokens_per_second == 450
