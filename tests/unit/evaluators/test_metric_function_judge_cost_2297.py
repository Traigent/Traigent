"""Regression test for Traigent#2297.

LLM calls made *inside* a metric function (an LLM-as-judge is the common
case) must be attributed to the SDK's cost ledger: the trial's ``total_cost``
and the permit-based cost enforcement in ``core/cost_enforcement.py`` must
see judge spend, not just the agent's own call.

``LocalEvaluator._extract_llm_metrics_for_output`` settles an example's cost
from the agent's captured response BEFORE
``LocalEvaluator._apply_custom_metric_functions`` runs the metric functions,
so a ``litellm.completion`` call placed inside a metric function was, before
this fix, captured by the interceptor but never folded back into that
example's cost.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.cost_estimator import CostEstimator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _DummyRawResp:
    """A minimal raw_response-shaped object (mirrors an OpenAI ChatCompletion)."""

    def __init__(self) -> None:
        self.model = "gpt-4o-mini"
        self.usage = _Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)


def _make_dataset(n: int = 2) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": f"Q{i}"},
                expected_output="ok",
                metadata={"example_id": f"example_{i}"},
            )
            for i in range(n)
        ],
        name="judge_cost_2297",
    )


def _agent(text: str, **config: object) -> dict:
    return {"text": "ok", "raw_response": _DummyRawResp()}


def judge_metric(output: object, expected: object, input_data: object) -> float:
    """An LLM-as-judge metric function that calls litellm directly."""
    import litellm

    litellm.completion(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "judge this"}],
    )
    return 1.0


@pytest.mark.asyncio
async def test_metric_function_llm_call_counted_in_total_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge call inside a metric function must raise total_cost (Traigent#2297)."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    dataset = _make_dataset()

    no_judge_evaluator = LocalEvaluator(metrics=["cost"], detailed=True)
    no_judge_result = await no_judge_evaluator.evaluate(_agent, {}, dataset)

    with_judge_evaluator = LocalEvaluator(
        metrics=["cost"],
        metric_functions={"judge": judge_metric},
        detailed=True,
    )
    with_judge_result = await with_judge_evaluator.evaluate(_agent, {}, dataset)

    no_judge_cost = no_judge_result.metrics["cost"]
    with_judge_cost = with_judge_result.metrics["cost"]

    assert no_judge_cost > 0.0, "agent-side cost should be measured at all"
    assert with_judge_cost > no_judge_cost, (
        "LLM calls inside metric_functions must be attributed to total_cost "
        f"(Traigent#2297): no_judge={no_judge_cost} with_judge={with_judge_cost}"
    )

    # The judge's own share is surfaced separately (a metadata entry, not a
    # new public field), never silently merged away, and the run-level total
    # is exactly the agent's cost plus every example's judge share -- not
    # merely "some positive amount" (Traigent#2297 review, minor #7).
    total_evaluation_cost = 0.0
    for example_result in with_judge_result.example_results:
        evaluation_cost = example_result.metrics.get("evaluation_cost", 0.0)
        assert evaluation_cost > 0.0
        total_evaluation_cost += evaluation_cost

        # Exact deterministic-mock token breakdown: the agent's own dummy
        # response reports 20 prompt / 10 completion tokens
        # (`_DummyRawResp`), and the judge call is served by
        # `MockAdapter`'s fixed defaults (10 prompt / 20 completion). Both
        # input_tokens/output_tokens (not just total_tokens) must reflect
        # the judge's share (Traigent#2297 review, important #2).
        assert example_result.metrics["input_tokens"] == pytest.approx(30)
        assert example_result.metrics["output_tokens"] == pytest.approx(30)
        assert example_result.metrics["total_tokens"] == pytest.approx(60)

    assert with_judge_cost == pytest.approx(no_judge_cost + total_evaluation_cost)

    # The value the permit-based cost enforcer actually reads
    # (`CostEstimator.extract_trial_cost`) must agree with `metrics["cost"]`
    # (Traigent#2297 review, minor #7).
    trial_result = TrialResult(
        trial_id="t0",
        config={},
        metrics=with_judge_result.metrics,
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )
    extracted_cost = CostEstimator.extract_trial_cost(trial_result)
    assert extracted_cost is not None
    assert extracted_cost == pytest.approx(with_judge_cost)
    assert extracted_cost > 0.0


@pytest.mark.asyncio
async def test_judge_cost_counted_even_when_every_example_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge call is real spend even when the agent never produces output.

    ``_extract_llm_metrics_for_output`` marks a row ``measured=False`` when
    the agent raises before producing any output at all, which excludes
    that row from measured-only aggregation
    (``MetricsTracker.aggregate_metrics``/``format_for_backend``). Before
    this fix, that exclusion silently dropped the judge's real spend too:
    the metric function still runs (on ``output=None``) and its LLM call is
    captured and folded into the example's cost/tokens, but the row stayed
    ``measured=False`` so the trial's aggregated ``cost`` stayed 0.0 even
    though the per-example ``evaluation_cost`` was positive
    (Traigent#2297 review, important #1).
    """
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    dataset = _make_dataset()

    def _raising_agent(text: str, **config: object) -> dict:
        raise ValueError("boom - agent never produces output")

    evaluator = LocalEvaluator(
        metrics=["cost"],
        metric_functions={"judge": judge_metric},
        detailed=True,
    )
    result = await evaluator.evaluate(_raising_agent, {}, dataset)

    assert result.metrics["cost"] > 0.0, (
        "judge spend must be counted in the trial's aggregated cost even "
        f"when every example errors before producing output: "
        f"cost={result.metrics['cost']}"
    )
    for example_result in result.example_results:
        assert example_result.metrics.get("evaluation_cost", 0.0) > 0.0
