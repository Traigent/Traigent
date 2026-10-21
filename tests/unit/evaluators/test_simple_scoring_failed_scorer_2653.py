"""SimpleScoringEvaluator must not turn a failed scorer into a measured 0.0 (#2653)."""

from __future__ import annotations

import pytest

import traigent.evaluators.base as base_mod
from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.utils.exceptions import EvaluationError


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": f"q{i}"}, expected_output="a")
            for i in range(3)
        ],
        name="t",
    )


def agent(q: str) -> str:
    return "a"


def judge(output, expected):  # stands in for an LLM judge that times out
    raise TimeoutError("judge timed out")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs", [{"metric_functions": {"accuracy": judge}}, {"scoring_function": judge}]
)
async def test_failing_scorer_marks_examples_failed(kwargs) -> None:
    result = await SimpleScoringEvaluator(**kwargs).evaluate(
        agent, {"model": "x"}, _dataset()
    )
    assert [r.success for r in result.example_results] == [False] * 3
    assert all(
        "judge timed out" in (r.error_message or "") for r in result.example_results
    )
    assert result.successful_examples == 0


@pytest.mark.asyncio
async def test_ragas_failure_does_not_substitute_zero(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(base_mod, "compute_ragas_metrics", boom)
    evaluator = SimpleScoringEvaluator(metrics=["faithfulness"])
    evaluator._ragas_metric_names = {"faithfulness"}
    with pytest.raises(EvaluationError, match="faithfulness"):
        await evaluator.evaluate(agent, {}, _dataset())
