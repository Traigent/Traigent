"""Supported LangChain responses keep per-call prices in both evaluator lanes."""

import pytest
from langchain_core.outputs import LLMResult

from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.cost_calculator import cost_from_tokens
from traigent.utils.langchain_interceptor import capture_langchain_response


@pytest.mark.parametrize("evaluator_type", [LocalEvaluator, SimpleScoringEvaluator])
@pytest.mark.parametrize(
    ("response_models", "priced_models"),
    [
        (["gpt-4o-mini", "gpt-4o"], ["gpt-4o-mini", "gpt-4o"]),
        (["gpt-4o-mini", "internal-gateway-alias"], ["gpt-4o-mini"] * 2),
        (["gpt-4o-mini", None], ["gpt-4o-mini"] * 2),
    ],
    ids=["mixed-models", "unknown-alias-fallback", "missing-model-fallback"],
)
async def test_supported_langchain_calls_keep_response_prices_and_config_fallback(
    evaluator_type, response_models, priced_models
):
    def agent(question):
        for model in response_models:
            llm_output = {
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            }
            if model is not None:
                llm_output["model_name"] = model
            capture_langchain_response(LLMResult(generations=[], llm_output=llm_output))
        return "ok"

    evaluator = (
        LocalEvaluator(metrics=["accuracy", "cost"], detailed=True)
        if evaluator_type is LocalEvaluator
        else SimpleScoringEvaluator(metrics=["accuracy", "cost"])
    )
    result = await evaluator.evaluate(
        agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"question": "q"}, "ok")]),
    )
    expected = sum(sum(cost_from_tokens(10, 20, model)) for model in priced_models)
    assert result.example_results[0].metrics["total_tokens"] == 60
    assert result.example_results[0].metrics["total_cost"] == pytest.approx(expected)
    if evaluator_type is LocalEvaluator:
        assert result.aggregated_metrics["cost"] == pytest.approx(expected)


def test_existing_response_metadata_precedence_is_preserved():
    from types import SimpleNamespace

    from traigent.evaluators.metrics_tracker import pricing_model_for_response

    response = SimpleNamespace(
        model="gpt-4o-mini", response_metadata={"model_name": "gpt-4o"}
    )
    assert pricing_model_for_response(response, "gpt-4o-mini") == "gpt-4o"
