"""Tests for the self_consistency cardinality strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from traigent.config.context import ConfigurationContext
from traigent.effectuation import get_strategy


def _compile_effect(*, aggregator=None):
    knob = {
        "name": "candidate_count",
        "kind": "cardinality",
        "value_set": (1, 3),
    }
    if aggregator is not None:
        knob["aggregator"] = aggregator
    return get_strategy("self_consistency").compile(knob)


def test_n3_majority_vote_returns_modal_output() -> None:
    effect = _compile_effect()
    responses = iter(["private answer", "other answer", " private answer "])
    calls: list[int] = []

    def target() -> str:
        calls.append(1)
        return next(responses)

    wrapped = effect.wrap_callable(target, plan={})

    with ConfigurationContext({"candidate_count": 3}):
        assert wrapped() == "private answer"

    assert len(calls) == 3
    events = effect.emit_events()
    assert events == [
        {
            "strategy": "self_consistency",
            "knob": "candidate_count",
            "n": 3,
            "calls": 3,
            "unique_outputs": 2,
            "aggregator": "majority_vote",
            "passthrough": False,
        }
    ]
    assert "private answer" not in repr(events)
    assert "other answer" not in repr(events)


def test_n1_is_single_call_passthrough_without_aggregator() -> None:
    def fail_if_called(outputs: Sequence[Any]) -> Any:
        raise AssertionError("aggregator should not run for N<=1")

    effect = _compile_effect(aggregator=fail_if_called)
    calls: list[int] = []

    def target() -> str:
        calls.append(1)
        return "single"

    wrapped = effect.wrap_callable(target, plan={})

    with ConfigurationContext({"candidate_count": 1}):
        assert wrapped() == "single"

    assert len(calls) == 1
    assert effect.emit_events()[0]["passthrough"] is True
    assert effect.emit_events()[0]["calls"] == 1


def test_aggregator_is_pluggable() -> None:
    def choose_last(outputs: Sequence[Any]) -> Any:
        return outputs[-1]

    effect = _compile_effect(aggregator=choose_last)
    responses = iter(["first", "second"])

    def target() -> str:
        return next(responses)

    wrapped = effect.wrap_callable(target, plan={})

    with ConfigurationContext({"candidate_count": 2}):
        assert wrapped() == "second"

    assert effect.emit_events()[0]["aggregator"] == "choose_last"


def test_wrap_callable_guards_against_double_execution() -> None:
    effect = _compile_effect()
    calls: list[int] = []

    def target() -> str:
        calls.append(1)
        return "same"

    wrapped_once = effect.wrap_callable(target, plan={})
    wrapped_twice = effect.wrap_callable(wrapped_once, plan={})

    assert wrapped_twice is wrapped_once
    with ConfigurationContext({"candidate_count": 2}):
        assert wrapped_twice() == "same"

    assert len(calls) == 2
