"""Requested default cost uses measured captures and preserves custom scores."""

import pytest
from langchain_core.outputs import LLMResult

from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.utils.cost_calculator import cost_from_tokens
from traigent.utils.langchain_interceptor import capture_langchain_response


def _agent(calls):
    for _ in range(calls):
        capture_langchain_response(
            LLMResult(
                generations=[],
                llm_output={
                    "model_name": "gpt-4o-mini",
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
            )
        )
    return "ok"


async def _evaluate(evaluator, counts=(1, 3)):
    return await evaluator.evaluate(
        _agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"calls": count}, "ok") for count in counts]),
    )


async def test_default_cost_is_measured_row_mean_while_total_cost_is_trial_sum():
    result = await _evaluate(SimpleScoringEvaluator(metrics=["cost"]))
    call_cost = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))
    assert [row.metrics["cost"] for row in result.example_results] == pytest.approx(
        [call_cost, 3 * call_cost]
    )
    assert result.aggregated_metrics["cost"] == pytest.approx(2 * call_cost)
    assert result.aggregated_metrics["total_cost"] == pytest.approx(4 * call_cost)


@pytest.mark.parametrize("custom_cost", [0.0, 0.5])
@pytest.mark.parametrize("custom_kind", ["metric-function", "scoring-function"])
async def test_explicit_custom_cost_remains_authoritative(custom_cost, custom_kind):
    if custom_kind == "metric-function":
        evaluator = SimpleScoringEvaluator(
            metric_functions={"cost": lambda output: custom_cost}
        )
    else:
        evaluator = SimpleScoringEvaluator(
            scoring_function=lambda output: {"cost": custom_cost}, metrics=["cost"]
        )
    result = await _evaluate(evaluator)
    assert [row.metrics["cost"] for row in result.example_results] == [custom_cost] * 2
    assert result.aggregated_metrics["cost"] == custom_cost
    call_cost = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))
    assert result.aggregated_metrics["total_cost"] == pytest.approx(4 * call_cost)


async def test_no_call_row_does_not_fabricate_cost_or_dilute_measured_mean():
    result = await _evaluate(SimpleScoringEvaluator(metrics=["cost"]), counts=(1, 0))
    call_cost = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))
    assert result.example_results[0].metrics["cost"] == pytest.approx(call_cost)
    assert "cost" not in result.example_results[1].metrics
    assert result.aggregated_metrics["cost"] == pytest.approx(call_cost)
    assert result.aggregated_metrics["total_cost"] == pytest.approx(call_cost)
