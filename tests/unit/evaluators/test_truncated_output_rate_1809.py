"""Regression tests for issue #1809 — truncated-output (finish_reason) guard.

Before the fix, ``finish_reason == "length"`` (or the Anthropic/Gemini
equivalents) on an eval call was silently ignored: the SDK proceeded, computed
metrics over the truncated output, and reported a score with no signal that
the answer was cut off. For reasoning models (gemini-2.5/3.x, gpt-5, o-series),
hidden reasoning tokens count against ``max_tokens`` before any answer text, so
a ``max_tokens`` sized for a normal model truncates the answer mid-output and
a capable model silently scores far below a cheap non-reasoning one.

These tests assert the decided behavior:

* every trial records a ``truncated_output_rate`` metric
  (mean of examples whose provider response carried a truncated finish/stop
  reason, over the examples that carried a finish/stop reason at all);
* any nonzero rate fires a SINGLE run-level warning, naming the offending
  config — never one per example/trial;
* a clean run (no truncation, or no finish_reason signal at all) emits no
  warning and reports a rate of 0.0;
* OpenAI-style ``choices[0].finish_reason == "length"`` and Anthropic-style
  ``stop_reason == "max_tokens"`` are both recognized.

Each behavioral assertion is written so it FAILS on the old behavior (no
metric, no warning, no ``finish_reason`` on ``ExampleMetrics``) and PASSES on
the new. Complement to the metadata-free empty-output-rate guard (#1851): this
fires even when the truncated text is non-empty.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    RESERVED_METRIC_KEYS,
    ExampleMetrics,
    OpenAIResponseHandler,
    compute_truncated_output_rate,
    enforce_user_metric_ceiling,
    finish_reason_is_truncated,
)

_LOCAL_LOGGER = "traigent.evaluators.local"
_WARNING_MARKER = "were truncated"


class _FakeOpenAIResponse:
    """Minimal stand-in for an OpenAI-shaped ChatCompletion (or a
    LiteLLM/OpenAI-compatible response for a reasoning model such as
    gemini-2.5-pro), just enough to satisfy ``OpenAIResponseHandler``."""

    def __init__(self, finish_reason: str, text: str = "cut off answ") -> None:
        self.choices = [
            SimpleNamespace(
                finish_reason=finish_reason, message=SimpleNamespace(content=text)
            )
        ]
        self.usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=256, total_tokens=356
        )


class _FakeAnthropicResponse:
    """Minimal stand-in for an Anthropic Message response."""

    def __init__(self, stop_reason: str, text: str = "cut off answ") -> None:
        self.model = "claude-3-5-sonnet"
        self.content = [SimpleNamespace(text=text)]
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(input_tokens=100, output_tokens=256)


def _make_func(outputs_by_text: dict[str, Any]) -> Callable[..., Any]:
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
        name="truncated-output-fixture",
    )


def _warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _WARNING_MARKER in r.getMessage()]


# --------------------------------------------------------------------------- #
# Pure helper unit tests                                                       #
# --------------------------------------------------------------------------- #
def test_finish_reason_is_truncated_classification() -> None:
    assert finish_reason_is_truncated("length") is True
    assert finish_reason_is_truncated("max_tokens") is True
    assert finish_reason_is_truncated("MAX_TOKENS") is True  # Gemini casing
    assert finish_reason_is_truncated("LENGTH") is True
    assert finish_reason_is_truncated("stop") is False
    assert finish_reason_is_truncated("tool_calls") is False
    assert finish_reason_is_truncated(None) is False
    assert finish_reason_is_truncated("") is False


def test_compute_truncated_output_rate_values() -> None:
    assert compute_truncated_output_rate([]) == 0.0  # no examples -> no rate

    # No example carries a finish_reason at all -> 0.0, not a false "clean".
    no_signal = [ExampleMetrics() for _ in range(3)]
    assert compute_truncated_output_rate(no_signal) == 0.0

    mixed = [
        ExampleMetrics(finish_reason="length"),
        ExampleMetrics(finish_reason="stop"),
        ExampleMetrics(finish_reason="stop"),
        ExampleMetrics(finish_reason="max_tokens"),
    ]
    assert compute_truncated_output_rate(mixed) == pytest.approx(0.5)

    all_clean = [ExampleMetrics(finish_reason="stop") for _ in range(4)]
    assert compute_truncated_output_rate(all_clean) == 0.0


def test_the_denominator_is_examples_that_CARRIED_a_signal() -> None:
    """The docstring's whole argument, which nothing tested.

    ``compute_truncated_output_rate`` deliberately divides by the examples that
    carried a recognizable finish/stop reason, NOT by every example in the
    trial: privacy mode, a plain string return, or an unrecognized response
    shape leaves ``finish_reason`` unset, and counting those as "not truncated"
    understates the rate using a signal that was never available.

    Every other case in this module gives every example a finish_reason, so the
    two denominators agree and the choice is invisible. Measured: dropping the
    ``if metric.finish_reason`` filter left all 11 tests green. This case is
    the one where they disagree.
    """
    rows = [
        ExampleMetrics(finish_reason="length"),  # truncated, has a signal
        ExampleMetrics(finish_reason="stop"),  # clean, has a signal
        ExampleMetrics(),  # no signal -- privacy mode / plain string
        ExampleMetrics(),  # no signal
    ]

    assert compute_truncated_output_rate(rows) == pytest.approx(0.5), (
        "1 of the 2 examples that reported a finish_reason was truncated; "
        "dividing by all 4 would report 0.25 and understate the problem using "
        "rows that never carried the signal"
    )


def test_openai_handler_extracts_truncated_finish_reason() -> None:
    handler = OpenAIResponseHandler()
    response = _FakeOpenAIResponse(finish_reason="length")
    metrics = handler.handle(response)
    assert metrics is not None
    assert metrics.finish_reason == "length"
    assert finish_reason_is_truncated(metrics.finish_reason) is True


def test_openai_handler_extracts_natural_stop() -> None:
    handler = OpenAIResponseHandler()
    response = _FakeOpenAIResponse(finish_reason="stop")
    metrics = handler.handle(response)
    assert metrics is not None
    assert metrics.finish_reason == "stop"
    assert finish_reason_is_truncated(metrics.finish_reason) is False


def test_generic_handler_extracts_anthropic_stop_reason() -> None:
    """Anthropic responses fall through to the base ``extract_finish_reason``
    via whichever handler in the chain accepts them; the generic extraction
    must still find ``stop_reason``."""
    from traigent.evaluators.metrics_tracker import ResponseHandlerFactory

    chain = ResponseHandlerFactory.create_handler_chain()
    response = _FakeAnthropicResponse(stop_reason="max_tokens")
    metrics = chain.handle(response)
    assert metrics is not None
    assert metrics.finish_reason == "max_tokens"
    assert finish_reason_is_truncated(metrics.finish_reason) is True


def test_truncated_output_rate_is_reserved_and_never_dropped() -> None:
    assert "truncated_output_rate" in RESERVED_METRIC_KEYS

    metrics: dict[str, Any] = {"truncated_output_rate": 0.25}
    for i in range(80):
        metrics[f"user_metric_{i:03d}"] = float(i)
    enforce_user_metric_ceiling(metrics, context="test-1809")
    assert metrics["truncated_output_rate"] == 0.25


# --------------------------------------------------------------------------- #
# Full-evaluator integration: rate recorded + run-level warning                 #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_truncated_outputs_recorded_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 4 examples, 2 truncated (finish_reason="length") -> rate == 0.5.
    outputs = {
        "q0": _FakeOpenAIResponse("length"),
        "q1": _FakeOpenAIResponse("length"),
        "q2": _FakeOpenAIResponse("stop"),
        "q3": _FakeOpenAIResponse("stop"),
    }
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "gemini-2.5-pro"}, _dataset(outputs)
        )

    assert result.metrics["truncated_output_rate"] == pytest.approx(0.5)

    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "gemini-2.5-pro" in message
    assert "50.0%" in message


@pytest.mark.asyncio
async def test_clean_run_no_warning_rate_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    outputs = {
        "q0": _FakeOpenAIResponse("stop"),
        "q1": _FakeOpenAIResponse("stop"),
        "q2": _FakeOpenAIResponse("stop"),
    }
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "clean"}, _dataset(outputs)
        )

    assert result.metrics["truncated_output_rate"] == 0.0
    assert _warning_records(caplog) == []


@pytest.mark.asyncio
async def test_no_finish_reason_signal_no_warning_rate_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run where the user's function returns plain strings (no captured
    provider response at all) has no finish_reason signal anywhere -- must
    report 0.0, not crash and not warn."""
    outputs = {"q0": "plain string output", "q1": "another plain string"}
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "no-signal"}, _dataset(outputs)
        )

    assert result.metrics["truncated_output_rate"] == 0.0
    assert _warning_records(caplog) == []


@pytest.mark.asyncio
async def test_single_truncated_example_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unlike the empty-output-rate guard's 10% threshold, ANY truncation
    warns -- a single cut-off example is already an unreliable comparison."""
    outputs = {
        "q0": _FakeOpenAIResponse("length"),
        "q1": _FakeOpenAIResponse("stop"),
        "q2": _FakeOpenAIResponse("stop"),
        "q3": _FakeOpenAIResponse("stop"),
        "q4": _FakeOpenAIResponse("stop"),
        "q5": _FakeOpenAIResponse("stop"),
        "q6": _FakeOpenAIResponse("stop"),
        "q7": _FakeOpenAIResponse("stop"),
        "q8": _FakeOpenAIResponse("stop"),
        "q9": _FakeOpenAIResponse("stop"),
    }
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        result = await evaluator.evaluate(
            _make_func(outputs), {"model": "one-bad"}, _dataset(outputs)
        )

    assert result.metrics["truncated_output_rate"] == pytest.approx(0.1)
    assert len(_warning_records(caplog)) == 1


@pytest.mark.asyncio
async def test_warning_fires_once_per_run_across_trials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    outputs = {
        "q0": _FakeOpenAIResponse("length"),
        "q1": _FakeOpenAIResponse("stop"),
    }
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)

    with caplog.at_level(logging.WARNING, logger=_LOCAL_LOGGER):
        r1 = await evaluator.evaluate(
            _make_func(outputs), {"config": "A"}, _dataset(outputs)
        )
        r2 = await evaluator.evaluate(
            _make_func(outputs), {"config": "B"}, _dataset(outputs)
        )

    assert r1.metrics["truncated_output_rate"] == pytest.approx(0.5)
    assert r2.metrics["truncated_output_rate"] == pytest.approx(0.5)
    assert len(_warning_records(caplog)) == 1


@pytest.mark.parametrize(
    "response,expected",
    [
        ({"metadata": {"finish_reason": "length"}}, "length"),
        ({"metadata": {"stop_reason": "max_tokens"}}, "max_tokens"),
        ({"response_metadata": {"finish_reason": "length"}}, "length"),
    ],
)
def test_a_dict_response_carrying_metadata_is_read(response, expected) -> None:
    """The metadata branch was the only one that did not handle a dict.

    ``_extract_choices_finish_reason`` and ``_extract_toplevel_finish_reason``
    both fall back to ``response.get(...)`` when the response IS a mapping.
    ``_extract_metadata_finish_reason`` read only ``getattr``, so a dict-shaped
    response carrying its reason under ``metadata`` reported "no signal" --
    including the internal wrapper shape its own docstring names
    (``integrations/utils/response_wrapper.py``).

    Measured before the fix: ``{"metadata": {"stop_reason": "max_tokens"}}``
    -> ``None``. Splitting the three branches into separate methods is what
    made the inconsistency visible; they had been interleaved in one function.
    """
    assert OpenAIResponseHandler().extract_finish_reason(response) == expected


def test_a_non_dict_metadata_value_is_still_no_signal() -> None:
    """Control: the fallback must not invent a reason out of any metadata.

    A fix that reached for ``response["metadata"]`` without checking it is a
    mapping would pass the test above and then raise, or worse, stringify
    something arbitrary into a provider finish reason.
    """
    assert (
        OpenAIResponseHandler().extract_finish_reason({"metadata": "not-a-dict"})
        is None
    )
    assert OpenAIResponseHandler().extract_finish_reason({"metadata": None}) is None
