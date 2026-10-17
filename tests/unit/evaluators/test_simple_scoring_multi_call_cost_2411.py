"""#2411: SimpleScoringEvaluator meters every LLM call an example makes.

``SimpleScoringEvaluator._capture_llm_metrics_for_example`` kept only
``captured_responses[0]``, so a function making N calls per example reported
1/N of its spend. The custom-evaluator lane was fixed in #2442; this pins the
public ``SimpleScoringEvaluator`` lane the same way. Keyless: every call is a
litellm ``mock_response`` (10 prompt / 20 completion tokens).
"""

from __future__ import annotations

import os

import pytest

litellm = pytest.importorskip("litellm")

from traigent.evaluators.base import (  # noqa: E402
    Dataset,
    EvaluationExample,
    SimpleScoringEvaluator,
)


@pytest.fixture(autouse=True)
def _local_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _call(model: str, spent: list[float]) -> str:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "q"}],
        mock_response="paris",
    )
    spent.append(float(litellm.completion_cost(response)))
    return response.choices[0].message.content


@pytest.mark.parametrize(
    "models",
    [
        ["gpt-4o-mini"],
        ["gpt-4o-mini"] * 3,
        ["gpt-4o-mini", "gpt-4o"],
    ],
    ids=["one-call", "three-calls", "agent-plus-judge"],
)
async def test_every_call_in_an_example_is_metered(models: list[str]) -> None:
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    spent: list[float] = []

    def func(question: str) -> str:
        outputs = [_call(model, spent) for model in models]
        return outputs[0]

    evaluator = SimpleScoringEvaluator(metrics=["accuracy"])
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"question": q}, expected_output="paris")
            for q in ("a", "b")
        ]
    )

    result = await evaluator.evaluate(func, {}, dataset)

    assert len(spent) == 2 * len(models)
    per_example = [er.metrics.get("total_cost") for er in result.example_results]
    expected_each = sum(spent) / 2
    assert per_example == [
        pytest.approx(expected_each),
        pytest.approx(expected_each),
    ]
