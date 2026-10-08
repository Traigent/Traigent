"""Direct budgets consume trial spend without changing cost objective scores."""

import pytest
from langchain_core.outputs import LLMResult
from openai.types.chat import ChatCompletion

from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.utils.cost_calculator import cost_from_tokens
from traigent.utils.langchain_interceptor import capture_langchain_response


def _capture(tokens=10):
    capture_langchain_response(
        LLMResult(
            generations=[],
            llm_output={
                "model_name": "gpt-4o-mini",
                "token_usage": {
                    "prompt_tokens": tokens,
                    "completion_tokens": tokens * 2,
                    "total_tokens": tokens * 3,
                },
            },
        )
    )


@pytest.mark.parametrize("custom_cost", [None, 0.0, 0.5])
async def test_direct_budget_debits_trial_total_and_preserves_objective(custom_cost):
    def agent(calls):
        for _ in range(calls):
            _capture()
        return "ok"

    evaluator = (
        SimpleScoringEvaluator(metrics=["cost"])
        if custom_cost is None
        else SimpleScoringEvaluator(
            metric_functions={"cost": lambda output: custom_cost}
        )
    )
    call_cost = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))
    budget = ExecutionBudget(max_cost_usd=3 * call_cost)
    result = await evaluator.evaluate(
        agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"calls": count}, "ok") for count in (1, 3)]),
        budget=budget,
    )
    expected_objective = 2 * call_cost if custom_cost is None else custom_cost
    assert result.aggregated_metrics["cost"] == pytest.approx(expected_objective)
    assert result.aggregated_metrics["total_cost"] == pytest.approx(4 * call_cost)
    assert budget.consumed_cost == pytest.approx(4 * call_cost)
    assert result.execution_budget_exhausted is True
    assert result.execution_budget["cost_tracking"] == "complete"


async def test_zero_measured_total_wins_over_custom_cost_score():
    def agent(question):
        capture_langchain_response(
            ChatCompletion.model_validate(
                {
                    "id": "test-free",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4o-mini",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                        "cost": 0.0,
                    },
                }
            )
        )
        return "ok"

    budget = ExecutionBudget(max_cost_usd=1)
    result = await SimpleScoringEvaluator(
        metric_functions={"cost": lambda output: 0.5}
    ).evaluate(
        agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"question": "q"}, "ok")]),
        budget=budget,
    )
    assert result.aggregated_metrics["cost"] == 0.5
    assert result.aggregated_metrics["total_cost"] == 0
    assert budget.consumed_cost == 0
    assert result.execution_budget["cost_tracking"] == "complete"


async def test_legacy_cost_only_evaluator_keeps_budget_fallback():
    budget = ExecutionBudget(max_cost_usd=1)
    result = await SimpleScoringEvaluator(
        metric_functions={"cost": lambda output: 0.25}, capture_llm_metrics=False
    ).evaluate(
        lambda question: "ok",
        {},
        Dataset([EvaluationExample({"question": "q"}, "ok")]),
        budget=budget,
    )
    assert "total_cost" not in result.aggregated_metrics
    assert budget.consumed_cost == 0.25
    assert result.execution_budget["cost_tracking"] == "complete"


async def test_no_cost_measurement_stays_untracked_and_stops_next_admission():
    calls = []

    def agent(question):
        calls.append(question)
        return "ok"

    evaluator = SimpleScoringEvaluator(
        metric_functions={"quality": lambda output: 1.0}, capture_llm_metrics=False
    )
    budget = ExecutionBudget(max_cost_usd=1, enforce_untracked_cost=True)
    dataset = Dataset([EvaluationExample({"question": "q"}, "ok")])
    first = await evaluator.evaluate(agent, {}, dataset, budget=budget)
    assert first.execution_budget["cost_tracking"] == "untracked"
    assert first.execution_budget["untracked_trials"] == 1
    assert budget.consumed_cost == 0
    second = await evaluator.evaluate(agent, {}, dataset, budget=budget)
    assert second.execution_budget_exhausted is True
    assert calls == ["q"]


async def test_shared_helper_preserves_local_evaluator_total_spend():
    from traigent.evaluators.local import LocalEvaluator

    def agent(calls):
        for _ in range(calls):
            _capture()
        return "ok"

    call_cost = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))
    budget = ExecutionBudget(max_cost_usd=3 * call_cost)
    result = await LocalEvaluator(metrics=["cost"], detailed=True).evaluate(
        agent,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"calls": count}, "ok") for count in (1, 3)]),
        budget=budget,
    )
    assert result.aggregated_metrics["cost"] == pytest.approx(4 * call_cost)
    assert budget.consumed_cost == pytest.approx(4 * call_cost)
    assert result.execution_budget_exhausted is True
