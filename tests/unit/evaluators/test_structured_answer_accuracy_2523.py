"""Per-example built-in accuracy for structured answers (Traigent#2523).

The per-example paths treated every dict output as a ``{"text": ...}``
wrapper, so a correct structured answer scored ``None`` (tracker path) or
``0.0`` (detailed path) while the aggregate scored it ``1.0``. The regression
case is a ``{category, urgency}`` triage answer.
"""

from __future__ import annotations

from typing import Any

import pytest

import traigent
from traigent.config.context import trial_context
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import MetricsTracker

TRIAGE = {"category": "billing", "urgency": "high"}
WRONG_TRIAGE = {"category": "billing", "urgency": "low"}


def _triage_dataset() -> Dataset:
    return Dataset(
        [EvaluationExample({"ticket": "charged twice"}, dict(TRIAGE))],
        name="triage",
    )


def _spy_tracker_accuracy(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the per-example accuracy each example hands the tracker."""
    seen: list[Any] = []
    original = MetricsTracker.add_example_metrics

    def spy(self: MetricsTracker, metrics: Any) -> None:
        seen.append(metrics.custom_metrics.get("accuracy"))
        original(self, metrics)

    monkeypatch.setattr(MetricsTracker, "add_example_metrics", spy)
    return seen


def _detailed_accuracy(result: Any) -> list[Any]:
    return [(r.metrics or {}).get("accuracy") for r in result.example_results or []]


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_correct_structured_answer_scores_one_per_example(
    monkeypatch: pytest.MonkeyPatch, detailed: bool
) -> None:
    seen = _spy_tracker_accuracy(monkeypatch)
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=detailed)

    result = await evaluator.evaluate(lambda _x: dict(TRIAGE), {}, _triage_dataset())

    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert seen == [1.0]
    if detailed:
        assert _detailed_accuracy(result) == [1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_wrong_structured_answer_scores_zero_not_none(
    monkeypatch: pytest.MonkeyPatch, detailed: bool
) -> None:
    # A measured miss must be 0.0 -- None let the tracker's success fallback
    # count it as correct.
    seen = _spy_tracker_accuracy(monkeypatch)
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=detailed)

    result = await evaluator.evaluate(
        lambda _x: dict(WRONG_TRIAGE), {}, _triage_dataset()
    )

    assert result.metrics["accuracy"] == pytest.approx(0.0)
    assert seen == [0.0]
    if detailed:
        assert _detailed_accuracy(result) == [0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_with_usage_wrapped_structured_answer_is_compared_as_itself(
    monkeypatch: pytest.MonkeyPatch, detailed: bool
) -> None:
    # #2522 lets with_usage() carry the triage dict; the wrapper is recognised
    # by its reserved key, so the dict inside is what gets compared.
    seen = _spy_tracker_accuracy(monkeypatch)
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=detailed)

    wrapped: list[Any] = []

    def agent(_x: Any) -> Any:
        out = traigent.with_usage(dict(TRIAGE), total_cost=0.002, input_tokens=10)
        wrapped.append(out)
        return out

    ctx_reset = trial_context.set({"trial_id": "t-2523"})
    try:
        result = await evaluator.evaluate(agent, {}, _triage_dataset())
    finally:
        trial_context.reset(ctx_reset)

    assert "__traigent_meta__" in wrapped[0]  # the wrapper really was in play

    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert seen == [1.0]
    if detailed:
        assert _detailed_accuracy(result) == [1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [False, True])
async def test_text_wrapper_dict_against_string_gold_still_unwraps(
    detailed: bool,
) -> None:
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=detailed)
    dataset = Dataset([EvaluationExample({"q": "1+1"}, "2")], name="math")

    result = await evaluator.evaluate(
        lambda _x: {"text": "2", "raw_response": object()}, {}, dataset
    )

    assert result.metrics["accuracy"] == pytest.approx(1.0)
    if detailed:
        assert _detailed_accuracy(result) == [1.0]


@pytest.mark.asyncio
async def test_structured_answer_with_text_field_is_not_reduced_to_text() -> None:
    # The measured #1771/#1772 hazard must stay closed on the per-example path:
    # a mapping gold is compared as a mapping, so the wrong "id" is not
    # discarded in favour of a re-parsed "text" field.
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    dataset = Dataset([EvaluationExample({"q": "x"}, {"id": "7"})], name="ids")

    result = await evaluator.evaluate(
        lambda _x: {"text": '{"id":"7"}', "id": "007"}, {}, dataset
    )

    assert result.metrics["accuracy"] == pytest.approx(0.0)
    assert _detailed_accuracy(result) == [0.0]
