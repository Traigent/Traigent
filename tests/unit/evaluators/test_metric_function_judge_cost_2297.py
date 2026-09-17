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
from traigent.evaluators.metrics_tracker import ExampleMetrics, MetricsTracker


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


class _EmptyCapturedResponse:
    """A captured response that prices to zero cost AND zero tokens.

    No ``usage``, no ``cost``, no ``model``: ``extract_llm_metrics`` extracts
    nothing real from it, which is exactly the "the judge call was abandoned /
    produced no usable measurement" shape the fold's zero-cost/zero-token
    guard exists to detect.
    """


def _fold_with_captured(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[object],
    example_metric: ExampleMetrics,
) -> None:
    """Run ``_fold_metric_function_llm_cost`` against a seeded capture buffer."""
    from traigent.evaluators import local as local_module

    monkeypatch.setattr(
        local_module, "get_all_captured_responses", lambda: list(responses)
    )
    monkeypatch.setattr(local_module, "clear_captured_responses", lambda: None)

    evaluator = LocalEvaluator(metrics=["cost"])
    evaluator._fold_metric_function_llm_cost(example_metric)


def test_abandoned_judge_capture_leaves_the_row_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-cost, zero-token capture must NOT flip ``measured`` to True.

    ``example_metric.measured = True`` used to be assigned ABOVE the
    zero-cost/zero-token abandon-guard, so a captured judge response that
    priced to nothing promoted a genuinely unmeasured row to ``measured=True``
    while every cost/token field stayed 0.0 -- the exact "zeros that are the
    ABSENCE of a measurement" case ``ExampleMetrics.measured``'s docblock says
    must never enter measured-only aggregation (Traigent#2297 review round 2).
    """
    example_metric = ExampleMetrics(measured=False)

    _fold_with_captured(monkeypatch, [_EmptyCapturedResponse()], example_metric)

    assert example_metric.measured is False, (
        "an abandoned (zero-cost, zero-token) judge capture must leave the row "
        "unmeasured; marking it measured feeds all-zero metrics into the "
        "measured-only MEAN denominators"
    )
    # The guard abandons the whole fold, so nothing else moved either.
    assert example_metric.cost.total_cost == 0.0
    assert example_metric.tokens.total_tokens == 0
    assert "evaluation_cost" not in example_metric.custom_metrics


def test_real_judge_capture_still_marks_the_row_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONTROL for the test above: a capture with REAL usage must still set
    ``measured=True``, which is what Traigent#2297's first review round fixed.
    Without this, moving the assignment below the guard could silently undo it.
    """
    example_metric = ExampleMetrics(measured=False)

    _fold_with_captured(monkeypatch, [_DummyRawResp()], example_metric)

    assert example_metric.measured is True
    assert example_metric.tokens.total_tokens == 30
    assert example_metric.custom_metrics["evaluation_cost"] > 0.0


def test_abandoned_capture_stays_out_of_the_measured_mean_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence that makes the ordering matter.

    ``MetricsTracker.aggregate_metrics`` means over ``measured`` rows only. A
    wrongly-promoted all-zero row halves the reported mean cost.
    """
    measured_row = ExampleMetrics(measured=True)
    measured_row.cost.total_cost = 0.02
    measured_row.tokens.total_tokens = 100

    abandoned_row = ExampleMetrics(measured=False)
    _fold_with_captured(monkeypatch, [_EmptyCapturedResponse()], abandoned_row)

    tracker = MetricsTracker()
    tracker.add_example_metrics(measured_row)
    tracker.add_example_metrics(abandoned_row)
    aggregated = tracker.aggregate_metrics()

    assert aggregated["total_cost"]["mean"] == pytest.approx(0.02), (
        "the abandoned row entered the MEAN denominator and halved the "
        f"reported cost: {aggregated['total_cost']['mean']}"
    )
    assert aggregated["total_tokens"]["mean"] == pytest.approx(100)


def test_evaluation_cost_is_droppable_under_the_measures_ceiling() -> None:
    """Pin the docstring caveat: ``evaluation_cost`` is NOT reserved.

    ``_fold_metric_function_llm_cost``'s docstring promises the judge's share
    is *reported alongside* the agent's cost, not that it is unconditionally
    present. It rides the USER metric channel, and
    ``enforce_user_metric_ceiling`` drops only non-reserved keys — so on a run
    that exceeds ``TOTAL_MEASURES_CEILING`` it can be dropped. If someone later
    reserves the key, this test fails and the docstring caveat must be removed
    with it.
    """
    from traigent.evaluators.metrics_tracker import (
        RESERVED_METRIC_KEYS,
        TOTAL_MEASURES_CEILING,
        enforce_user_metric_ceiling,
        is_reserved_metric_key,
    )

    assert "evaluation_cost" not in RESERVED_METRIC_KEYS
    assert is_reserved_metric_key("evaluation_cost") is False

    # One over the ceiling, with `evaluation_cost` sorting last among the
    # user keys so it is the one dropped.
    target: dict[str, object] = {"total_cost": 1.0, "evaluation_cost": 0.5}
    filler = TOTAL_MEASURES_CEILING + 1 - len(target)
    for i in range(filler):
        target[f"aaa_user_metric_{i:03d}"] = float(i)
    assert len(target) == TOTAL_MEASURES_CEILING + 1

    enforce_user_metric_ceiling(target, context="test_2297_receipt_caveat")

    assert len(target) == TOTAL_MEASURES_CEILING
    assert "evaluation_cost" not in target, (
        "evaluation_cost survived the ceiling — if it is now reserved, drop "
        "the CAVEAT paragraph from _fold_metric_function_llm_cost's docstring"
    )
    assert "total_cost" in target, "reserved keys must never be dropped"
