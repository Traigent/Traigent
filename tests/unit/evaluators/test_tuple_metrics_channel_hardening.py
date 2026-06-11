"""Hardening probes for the tuple-return metrics channel (codex review round).

These pin the four codex-verified findings on the ``(output, metrics_dict)``
return channel:

* F1 — SimpleScoringEvaluator's trial aggregation surfaces user keys on the
  result.metrics (it previously aggregated only ``self.metrics``).
* F2a — ``_unpack_user_metrics`` requires EVERY key to be a Python identifier;
  a non-identifier key (e.g. ``"bad-key"``) means the shape does not match and
  the raw tuple rides through untouched.
* F2b — user keys cannot push the aggregated trial metrics past the
  ``TOTAL_MEASURES_CEILING`` (50) total; user keys are truncated
  deterministically (evaluator keys never dropped) with a warning.
* F3 — user keys can never overwrite a reserved evaluator/tracker key
  (``success_rate``, ``latency``, ``examples_attempted``, …) at ANY merge or
  aggregation site, regardless of ordering or setdefault.
* F4 — a nested matching tuple is unpacked exactly once: the carrier already
  attached upstream is preferred, so ``_process_single_output`` does not
  re-unpack an already-unpacked output.
"""

from __future__ import annotations

import logging

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.knobs.telemetry import TOTAL_MEASURES_CEILING


def _two_example_dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="YES"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="YES"),
        ],
        name="tuple_metrics_hardening",
    )


def _local_evaluator() -> LocalEvaluator:
    return LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        execution_mode="edge_analytics",
    )


# ---------------------------------------------------------------------------
# F1: SimpleScoringEvaluator surfaces tuple-supplied user keys on result.metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_scoring_tuple_user_keys_reach_result_metrics() -> None:
    """F1: the public SimpleScoringEvaluator path mean-aggregates user keys."""

    def score(output: object, expected: object) -> float:
        return 1.0 if output == expected else 0.0

    def func(text: str) -> tuple[str, dict[str, float]]:
        margin = 1.0 if text == "q1" else 0.5
        return "YES", {"composite_vote_margin": margin}

    evaluator = SimpleScoringEvaluator(
        scoring_function=score,
        metrics=["accuracy"],
        capture_llm_metrics=False,
    )
    result = await evaluator.evaluate(func, {}, _two_example_dataset())

    # RED before F1: composite_vote_margin reaches example_results[*].metrics but
    # never result.metrics, so the channel dies on this evaluator.
    assert result.metrics["composite_vote_margin"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# F2a: non-identifier keys make the shape NOT match -> raw passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_identifier_key_is_not_unpacked() -> None:
    """F2a: a metrics dict with a non-identifier key is left as the raw tuple."""

    async def func(text: str) -> tuple[str, dict[str, float]]:
        return "ok", {"bad-key": 1.0}

    result = await _local_evaluator().evaluate(func, {}, _two_example_dataset())

    # RED before F2a: "bad-key" is unpacked and rides; the tuple is split.
    assert "bad-key" not in result.metrics
    # The raw tuple poisons accuracy (output != "YES"), proving no unpack.
    assert result.metrics["accuracy"] == 0.0


def test_unpack_helper_rejects_non_identifier_keys_directly() -> None:
    """F2a (unit): _unpack_user_metrics passes through on any non-identifier."""
    raw = ("ok", {"bad-key": 1.0, "fine_key": 2.0})
    output, user_metrics = SimpleScoringEvaluator._unpack_user_metrics(raw)
    assert output is raw
    assert user_metrics is None


# ---------------------------------------------------------------------------
# F2b: user keys cannot push aggregated trial metrics past the ceiling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_keys_truncated_to_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F2b: >50 distinct user keys are truncated to the total ceiling."""
    # 60 distinct identifier user keys; the evaluator also contributes its own
    # keys (accuracy, etc.), so the total must be clamped to the ceiling.
    user_keys = {f"composite_metric_{i}": float(i) for i in range(60)}

    async def func(text: str) -> tuple[str, dict[str, float]]:
        return "YES", dict(user_keys)

    with caplog.at_level(logging.WARNING):
        result = await _local_evaluator().evaluate(func, {}, _two_example_dataset())

    # RED before F2b: nothing clamps the total, so it exceeds the ceiling.
    assert len(result.metrics) <= TOTAL_MEASURES_CEILING
    # Evaluator keys are never dropped.
    assert "accuracy" in result.metrics
    # A truncation warning was logged.
    assert any("ceiling" in record.getMessage().lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# F3: user keys never overwrite reserved evaluator/tracker keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_keys_never_overwrite_reserved_builtins(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F3: success_rate / latency / examples_attempted stay evaluator-computed."""

    async def func(text: str) -> tuple[str, dict[str, float]]:
        return "YES", {
            "success_rate": 0.01,
            "latency": 0.0,
            "examples_attempted": 999.0,
            "composite_ok": 1.0,
        }

    evaluator = LocalEvaluator(
        metrics=["accuracy", "success_rate", "latency"],
        detailed=True,
        execution_mode="edge_analytics",
    )
    with caplog.at_level(logging.WARNING):
        result = await evaluator.evaluate(func, {}, _two_example_dataset())

    # examples_attempted is the count (2), never the user's 999.0.
    assert result.metrics["examples_attempted"] == 2
    # success_rate is the evaluator's computed value (both succeeded -> 1.0),
    # never the user's 0.01.
    assert result.metrics["success_rate"] != pytest.approx(0.01)
    # latency is evaluator-computed, never forced to the user's 0.0-by-injection;
    # the key exists and is the computed one (it is allowed to equal 0.0 if the
    # evaluator computed 0.0, but it must not be sourced from the user dict).
    assert "latency" in result.metrics
    # The non-reserved composite key still rides.
    assert result.metrics["composite_ok"] == pytest.approx(1.0)
    # Reserved-key skips are warning-logged (upgraded from debug).
    assert any(
        "reserved" in record.getMessage().lower() and record.levelno >= logging.WARNING
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# F4: nested matching tuple is unpacked exactly once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_tuple_unpacked_once() -> None:
    """F4: ((inner, {x}), {outer}) keeps output == (inner, {x}); rides {outer}."""

    captured_outputs: list[object] = []

    def _capture(index: int, payload: dict[str, object]) -> None:
        if "output" in payload:
            captured_outputs.append(payload["output"])

    async def func(text: str) -> tuple[tuple[str, dict[str, float]], dict[str, float]]:
        return (("inner", {"x": 1.0}), {"outer": 2.0})

    evaluator = _local_evaluator()
    result = await evaluator.evaluate(
        func, {}, _two_example_dataset(), progress_callback=_capture
    )

    # Only the OUTER user metric rides; the inner mapping is part of the output
    # and is NOT re-unpacked into trial metrics.
    assert result.metrics["outer"] == pytest.approx(2.0)
    assert "x" not in result.metrics
    # The resolved per-example actual_output is the inner tuple, unpacked once.
    assert result.example_results[0].actual_output == ("inner", {"x": 1.0})
    # RED before F4: _process_single_output re-unpacks the already-unpacked
    # output, so the resolved per-example output becomes "inner" (the inner
    # tuple[0]) instead of the inner tuple. The local-lane progress payload
    # carries that resolved output, so it must stay the inner tuple.
    assert captured_outputs, "expected per-example progress payloads"
    assert all(out == ("inner", {"x": 1.0}) for out in captured_outputs)
