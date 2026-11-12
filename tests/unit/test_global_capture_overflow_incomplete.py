"""Global-capture overflow must never let spend read as complete (#3098)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.api.types import ExampleResult
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.evaluators.base import EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.metrics_tracker import extract_llm_metrics
from traigent.utils import langchain_interceptor as li


def _response():
    return SimpleNamespace(
        model="gpt-4o",
        usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        ),
    )


@pytest.fixture
def overflowed(monkeypatch):
    monkeypatch.setattr(li, "_MAX_GLOBAL_RESPONSES", 2)
    li.clear_captured_responses()
    yield
    li.clear_captured_responses()


def _fill(n):
    for _ in range(n):
        li._metadata_capture.set_last_response(_response())


def test_flag_set_on_overflow_and_reset_by_clear(overflowed):
    _fill(2)
    assert li.captured_responses_incomplete() is False
    _fill(1)
    assert li.captured_responses_incomplete() is True
    li.clear_captured_responses()
    assert li.captured_responses_incomplete() is False


def test_drain_resets_flag(overflowed):
    _fill(3)
    li._metadata_capture.drain()
    assert li.captured_responses_incomplete() is False


def _custom_metrics():
    async def evaluator(func, config, example):
        raise AssertionError("not called")

    wrapper = CustomEvaluatorWrapper(evaluator, metrics=["accuracy"])
    wrapper._metrics_available = True
    wrapper.capture_llm_metrics = True
    wrapper._get_all_captured_responses = li.get_all_captured_responses
    example = EvaluationExample(input_data={"q": "x"}, expected_output="y")
    row = ExampleResult(
        example_id="e0",
        input_data=example.input_data,
        expected_output="y",
        actual_output="y",
        metrics={},
        execution_time=0.0,
        success=True,
        error_message=None,
        metadata={},
    )
    return wrapper._capture_llm_metrics_for_example(
        {"model": "gpt-4o"}, example, row, 0
    )


def _simple_metrics():
    ev = object.__new__(SimpleScoringEvaluator)
    ev.capture_llm_metrics = True
    ev._metrics_available = True
    ev._get_all_captured_responses = li.get_all_captured_responses
    ev._extract_llm_metrics = extract_llm_metrics
    return ev._capture_llm_metrics_for_example("y", {"model": "gpt-4o"}, 0)


def test_custom_wrapper_complete_without_overflow(overflowed):
    _fill(2)
    assert _custom_metrics().cost.unmeasured is False


def test_custom_wrapper_unmeasured_after_overflow(overflowed):
    _fill(5)
    assert _custom_metrics().cost.unmeasured is True


def test_simple_evaluator_complete_without_overflow(overflowed):
    _fill(2)
    assert not _simple_metrics().get("cost_unmeasured")


def test_simple_evaluator_unmeasured_after_overflow(overflowed):
    _fill(5)
    assert _simple_metrics().get("cost_unmeasured")
