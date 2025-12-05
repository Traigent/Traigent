import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


@pytest.mark.asyncio
async def test_local_evaluator_accuracy_exact_match() -> None:
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)

    examples = [
        EvaluationExample({"query": "1+1"}, "2"),
        EvaluationExample({"query": "2+2"}, "4"),
    ]
    dataset = Dataset(examples, name="math", description="basic math")

    def func(input_data):
        expression = input_data["query"]
        return str(eval(expression))

    result = await evaluator.evaluate(func, {}, dataset)

    assert result.metrics is not None
    assert result.metrics.get("accuracy") == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_local_evaluator_accuracy_mismatch() -> None:
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)

    examples = [
        EvaluationExample({"query": "1+1"}, "2"),
        EvaluationExample({"query": "2+2"}, "4"),
    ]
    dataset = Dataset(examples, name="math", description="basic math")

    def func(input_data):
        if input_data["query"] == "1+1":
            return "2"
        return "5"  # wrong on purpose

    result = await evaluator.evaluate(func, {}, dataset)

    assert result.metrics is not None
    assert result.metrics.get("accuracy") == pytest.approx(0.5)
