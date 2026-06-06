"""The 2-tuple ``(output, metrics_dict)`` return rides the measures channel.

These tests exercise the REAL local/hybrid evaluation lane (no mocking of the
unpack itself): a decorated-style function returns ``(output, user_metrics)``
and we assert (a) accuracy is computed from ``output`` (tuple[0]), not the raw
tuple, and (b) the numeric ``user_metrics`` keys mean-aggregate into the trial
metrics so they ride the measures wire channel.

Strict unpack rule (see ``BaseEvaluator._unpack_user_metrics``): ONLY a length-2
tuple whose [1] is a Mapping with all-str keys and all-numeric (non-bool)
values is unpacked. Every other shape is left untouched (absolute back-compat).
"""

from __future__ import annotations

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def _two_example_dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="YES"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="YES"),
        ],
        name="tuple_metrics_channel",
    )


def _evaluator() -> LocalEvaluator:
    # edge_analytics keeps everything offline (no backend / no LLM).
    return LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        execution_mode="edge_analytics",
    )


@pytest.mark.asyncio
async def test_tuple_return_rides_composite_metrics_and_accuracy_from_output() -> None:
    """5a: composite_* keys mean-aggregate AND accuracy comes from tuple[0]."""

    async def func(text: str) -> tuple[str, dict[str, float]]:
        # Both examples return ("YES", {...}) so accuracy is 1.0 from tuple[0].
        # vote_margin differs per example so the MEAN is asserted (not a sum).
        margin = 1.0 if text == "q1" else 0.5
        return "YES", {
            "composite_vote_margin": margin,
            "composite_candidates_evaluated": 3,
        }

    result = await _evaluator().evaluate(func, {}, _two_example_dataset())

    # Accuracy is computed from tuple[0] == "YES" == expected "YES" -> 1.0.
    # (The RED here: today the raw tuple is compared to "YES" -> 0.0.)
    assert result.metrics["accuracy"] == 1.0

    # The composite_* keys rode the channel, mean-aggregated across the 2
    # examples: vote_margin mean = (1.0 + 0.5) / 2 = 0.75; candidates = 3.
    assert result.metrics["composite_vote_margin"] == pytest.approx(0.75)
    assert result.metrics["composite_candidates_evaluated"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_plain_str_return_unchanged() -> None:
    """5c: a plain str return is unaffected (no composite keys, accuracy ok)."""

    async def func(text: str) -> str:
        return "YES"

    result = await _evaluator().evaluate(func, {}, _two_example_dataset())
    assert result.metrics["accuracy"] == 1.0
    assert not any(k.startswith("composite_") for k in result.metrics)


@pytest.mark.asyncio
async def test_dict_with_text_return_unchanged() -> None:
    """5c: a dict-with-'text' return keeps its existing accuracy behavior."""

    async def func(text: str) -> dict[str, str]:
        return {"text": "YES", "other": "ignored"}

    result = await _evaluator().evaluate(func, {}, _two_example_dataset())
    # accuracy still derived from output["text"] == "YES".
    assert result.metrics["accuracy"] == 1.0
    assert not any(k.startswith("composite_") for k in result.metrics)


@pytest.mark.asyncio
async def test_three_tuple_not_unpacked() -> None:
    """5c: a length-3 tuple is NOT unpacked (left as raw output)."""

    async def func(text: str) -> tuple:
        return ("YES", {"composite_x": 1.0}, "extra")

    result = await _evaluator().evaluate(func, {}, _two_example_dataset())
    # Raw 3-tuple != "YES" so accuracy is 0.0, and no composite key leaks.
    assert result.metrics["accuracy"] == 0.0
    assert not any(k.startswith("composite_") for k in result.metrics)


@pytest.mark.asyncio
async def test_tuple_with_non_numeric_dict_values_not_unpacked() -> None:
    """5c: a 2-tuple whose [1] has a non-numeric value is NOT unpacked."""

    async def func(text: str) -> tuple[str, dict[str, object]]:
        return ("YES", {"composite_x": "not-a-number"})

    result = await _evaluator().evaluate(func, {}, _two_example_dataset())
    assert result.metrics["accuracy"] == 0.0
    assert not any(k.startswith("composite_") for k in result.metrics)


@pytest.mark.asyncio
async def test_tuple_with_list_second_element_not_unpacked() -> None:
    """5c: a 2-tuple whose [1] is a list (not a Mapping) is NOT unpacked."""

    async def func(text: str) -> tuple[str, list[int]]:
        return ("YES", [1, 2, 3])

    result = await _evaluator().evaluate(func, {}, _two_example_dataset())
    assert result.metrics["accuracy"] == 0.0
    assert not any(k.startswith("composite_") for k in result.metrics)


@pytest.mark.asyncio
async def test_user_metric_named_accuracy_does_not_override_evaluator() -> None:
    """5d: a user 'accuracy' key never overrides the evaluator's computed one."""

    async def func(text: str) -> tuple[str, dict[str, float]]:
        # User tries to inject accuracy=0.0; evaluator computes 1.0 from "YES".
        return "YES", {"accuracy": 0.0, "composite_ok": 1.0}

    result = await _evaluator().evaluate(func, {}, _two_example_dataset())
    # Evaluator's computed accuracy (1.0) wins; the user value is ignored.
    assert result.metrics["accuracy"] == 1.0
    # The non-colliding composite key still rides.
    assert result.metrics["composite_ok"] == pytest.approx(1.0)
