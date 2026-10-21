"""Traigent#2411 item 5 (and part of #2444): SimpleScoringEvaluator meters every
call in a row, each at its own model, instead of only ``captured_responses[0]``.

Keyless: LiteLLM ``mock_response`` calls, priced by litellm's bundled table.
"""

from __future__ import annotations

import pytest

litellm = pytest.importorskip("litellm")

from traigent.evaluators.base import (  # noqa: E402
    Dataset,
    EvaluationExample,
    SimpleScoringEvaluator,
)
from traigent.utils.langchain_interceptor import (  # noqa: E402
    capture_langchain_response,
)

ROWS = 3


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.delenv("TRAIGENT_STRICT_METRICS_NULLS", raising=False)


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": f"q{i}"}, expected_output="paris")
            for i in range(ROWS)
        ],
        name="simple-scoring-multi-call",
    )


def _call(model: str) -> tuple[str, float]:
    response = litellm.completion(
        model=model, messages=[{"role": "user", "content": "q"}], mock_response="paris"
    )
    cost = litellm.completion_cost(completion_response=response)
    return response.choices[0].message.content, cost


def _score(output, expected) -> float:
    return 1.0 if output == expected else 0.0


@pytest.mark.asyncio
async def test_row_cost_sums_all_calls_each_at_its_own_model() -> None:
    spent: list[float] = []

    def agent(q: str) -> str:
        answer, cost = _call("gpt-4o-mini")
        spent.append(cost)
        _, judge_cost = _call("gpt-4o")  # second call on a different model
        spent.append(judge_cost)
        return answer

    # The configured model is the agent's; the judge must not be priced at it.
    result = await SimpleScoringEvaluator(scoring_function=_score).evaluate(
        agent, {"model": "gpt-4o-mini"}, _dataset()
    )
    assert sum(spent) == pytest.approx(0.0007155)
    assert result.aggregated_metrics["total_cost"] == pytest.approx(sum(spent))
    for row in result.example_results:
        assert row.metrics["total_tokens"] == 2 * 30


@pytest.mark.asyncio
async def test_unmeasured_call_stays_unmeasured_not_free() -> None:
    class _NoUsage:
        model = "gpt-4o"
        usage = None

    def agent(q: str) -> str:
        answer, _ = _call("gpt-4o-mini")
        capture_langchain_response(_NoUsage())
        return answer

    result = await SimpleScoringEvaluator(scoring_function=_score).evaluate(
        agent, {"model": "gpt-4o-mini"}, _dataset()
    )
    for row in result.example_results:
        assert row.metrics.get("cost_unmeasured") == 1.0
        assert "total_cost" not in row.metrics
        assert row.metrics["known_cost"] > 0
