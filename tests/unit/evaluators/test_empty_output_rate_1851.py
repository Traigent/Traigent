"""Regression tests for issue #1851 — high empty-output-rate guard.

Before the fix, an evaluator run where a large fraction of a config's outputs
were empty strings (truncation, output-parsing failure, or refusals) proceeded
silently: metrics were computed over the empties, a best_config was selected,
and the run finished with no signal — so a directional "optimization gain" that
was entirely an artifact looked like a real win.

These tests assert the decided behavior:

* every trial records an ``empty_output_rate`` metric
  (``mean(output is None or not str(output).strip())`` over the outputs);
* above the threshold (default 10%) a SINGLE run-level warning fires, naming the
  offending config — never one per example/trial;
* a clean run emits no warning and reports a rate of 0.0;
* whitespace-only and ``None`` outputs count as empty.

Each behavioral assertion is written so it FAILS on the old behavior (no metric,
no warning) and PASSES on the new. This is the metadata-free complement to the
finish_reason guard (#1809): it works for every calling pattern
(decorator/wrapper/managed) because it reads only the returned outputs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from typing import Any

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    EMPTY_OUTPUT_RATE_WARNING_THRESHOLD,
    RESERVED_METRIC_KEYS,
    compute_empty_output_rate,
    enforce_user_metric_ceiling,
    output_is_empty,
)

_LOCAL_LOGGER = "traigent.evaluators.local"
_WARNING_MARKER = "empty or whitespace-only"


def _make_func(
    outputs_by_text: dict[str, Any],
) -> Callable[..., Any]:
    """Build an async agent that returns a fixed output per input ``text``."""

    async def _func(text: str) -> Any:
        return outputs_by_text[text]

    return _func


def _dataset(texts: Collection[str]) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": t}, expected_output="ref")
            for t in texts
        ],
        name="empty-output-fixture",
    )


def _warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _WARNING_MARKER in r.getMessage()]


# --------------------------------------------------------------------------- #
# Pure helper unit tests                                                       #
# --------------------------------------------------------------------------- #
def test_output_is_empty_classification() -> None:
    assert output_is_empty(None) is True
    assert output_is_empty("") is True
    assert output_is_empty("   ") is True  # whitespace-only counts as empty
    assert output_is_empty("\n\t ") is True
    assert output_is_empty("answer") is False
    assert output_is_empty(" answer ") is False
    # A non-string, non-blank value stringifies to something non-empty.
    assert output_is_empty(0) is False
    assert output_is_empty(["x"]) is False


def test_compute_empty_output_rate_values() -> None:
    assert compute_empty_output_rate([]) == 0.0  # no outputs -> no empties
    assert compute_empty_output_rate(["a", "b", "c", "d"]) == 0.0
    assert compute_empty_output_rate(["", "b", "  ", None]) == pytest.approx(0.75)
    assert compute_empty_output_rate(["", "b"]) == pytest.approx(0.5)


def test_empty_output_rate_is_reserved_and_never_dropped() -> None:
    """The metric must be reserved so a user tuple key cannot overwrite it and it
    is never sacrificed to the MeasuresDict ceiling under user-key pressure."""
    assert "empty_output_rate" in RESERVED_METRIC_KEYS

    # Flood the trial metrics with user keys past the ceiling; the reserved
    # empty_output_rate must survive while user keys are dropped.
    metrics: dict[str, Any] = {"empty_output_rate": 0.5}
    for i in range(80):
        metrics[f"user_metric_{i:03d}"] = float(i)
    enforce_user_metric_ceiling(metrics, context="test-1851")
    assert metrics["empty_output_rate"] == 0.5


# --------------------------------------------------------------------------- #
# (a) high empty rate -> rate in metrics + warning emitted                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_high_empty_output_rate_recorded_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 4 examples, 2 return "" -> empty_output_rate == 0.5 (> 10% threshold).
    outputs = {"q0": "", "q1": "", "q2": "answer-2", "q3": "answer-3"}
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "gpt-oss-120b"}, _dataset(outputs)
        )

    # NEW: the per-config empty-output rate is exposed on the trial metrics.
    assert result.metrics["empty_output_rate"] == pytest.approx(0.5)

    # NEW: exactly one run-level warning fired, and it names the offending config.
    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "gpt-oss-120b" in message
    assert "50.0%" in message


# --------------------------------------------------------------------------- #
# (b) clean run -> no warning, rate 0.0                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_clean_run_no_warning_rate_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    outputs = {"q0": "answer-0", "q1": "answer-1", "q2": "answer-2"}
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "clean"}, _dataset(outputs)
        )

    assert result.metrics["empty_output_rate"] == 0.0
    assert _warning_records(caplog) == []


# --------------------------------------------------------------------------- #
# (c) whitespace-only counts as empty                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_whitespace_only_counts_as_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 2 whitespace outputs of 4 -> rate 0.5; whitespace must count as empty.
    outputs = {"q0": "   ", "q1": "\n\t", "q2": "answer-2", "q3": "answer-3"}
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "ws"}, _dataset(outputs)
        )

    assert result.metrics["empty_output_rate"] == pytest.approx(0.5)
    assert len(_warning_records(caplog)) == 1


@pytest.mark.asyncio
async def test_none_output_counts_as_empty() -> None:
    # A None return (e.g. a swallowed failure in the user's function) is empty.
    outputs = {"q0": None, "q1": "answer-1"}
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    result = await evaluator.evaluate(
        _make_func(outputs), {"model": "none"}, _dataset(outputs)
    )
    assert result.metrics["empty_output_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Threshold boundary + once-per-run semantics                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_at_threshold_does_not_warn_but_records_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 2 empty of 20 -> rate == 0.10 == threshold. Warning fires strictly ABOVE.
    outputs = {f"q{i}": ("" if i < 2 else f"answer-{i}") for i in range(20)}
    assert EMPTY_OUTPUT_RATE_WARNING_THRESHOLD == 0.10
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "boundary"}, _dataset(outputs)
        )

    assert result.metrics["empty_output_rate"] == pytest.approx(0.10)
    assert _warning_records(caplog) == []


@pytest.mark.asyncio
async def test_warning_fires_once_per_run_across_trials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Same evaluator instance (== one run) evaluated twice, both above threshold:
    # exactly one warning, mirroring the dual-scorer once-per-run notice.
    outputs = {"q0": "", "q1": "", "q2": "answer-2", "q3": "answer-3"}
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        r1 = await evaluator.evaluate(
            _make_func(outputs), {"config": "A"}, _dataset(outputs)
        )
        r2 = await evaluator.evaluate(
            _make_func(outputs), {"config": "B"}, _dataset(outputs)
        )

    # Both trials still record their own rate...
    assert r1.metrics["empty_output_rate"] == pytest.approx(0.5)
    assert r2.metrics["empty_output_rate"] == pytest.approx(0.5)
    # ...but the run-level warning fires only once.
    assert len(_warning_records(caplog)) == 1
