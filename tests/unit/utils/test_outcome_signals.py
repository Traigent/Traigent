"""Tests for the per-example signals the SDK derives locally.

These signals exist so evaluator quality can be assessed WITHOUT shipping prompts,
completions or gold labels off the client. The first test is therefore the one that
matters most: content must not survive into the payload.
"""

from __future__ import annotations

import json

import pytest

from traigent.api.types import ExampleResult
from traigent.utils.outcome_signals import (
    build_example_signals,
    example_digest,
    output_digest,
    verified_match,
)


def _result(**overrides):
    payload = {
        "example_id": "ex-1",
        "input_data": {"q": "2+2?"},
        "expected_output": "4",
        "actual_output": "4",
        "metrics": {"accuracy": 1.0},
        "execution_time": 0.1,
        "success": True,
    }
    payload.update(overrides)
    return ExampleResult(**payload)


# --- the property the whole design rests on -------------------------------


@pytest.mark.parametrize("field", ["input_data", "expected_output", "actual_output"])
def test_no_content_reaches_the_payload(field: str) -> None:
    secret = "CANARY-7f3a-CONFIDENTIAL-VALUE"
    value = {"q": secret} if field == "input_data" else secret
    payload = json.dumps(build_example_signals(_result(**{field: value})))
    assert secret not in payload
    for fragment in ("CANARY", "CONFIDENTIAL", "7f3a"):
        assert fragment not in payload


def test_digests_are_fixed_width_hex() -> None:
    signals = build_example_signals(_result())
    for key in ("example_digest", "output_digest"):
        assert len(signals[key]) == 64
        assert all(c in "0123456789abcdef" for c in signals[key])


# --- identity behaviour ---------------------------------------------------


def test_same_example_same_digest_across_calls_and_objects() -> None:
    assert (
        build_example_signals(_result())["example_digest"]
        == (build_example_signals(_result())["example_digest"])
    )


def test_dict_key_order_does_not_change_the_digest() -> None:
    a = example_digest({"a": 1, "b": 2}, "x")
    b = example_digest({"b": 2, "a": 1}, "x")
    assert a == b


def test_a_different_output_changes_only_the_output_digest() -> None:
    same = build_example_signals(_result())
    other = build_example_signals(_result(actual_output="5"))
    assert same["example_digest"] == other["example_digest"]
    assert same["output_digest"] != other["output_digest"]


def test_a_different_expected_answer_changes_the_example_digest() -> None:
    assert example_digest({"q": "x"}, "A") != example_digest({"q": "x"}, "B")


def test_digest_domains_are_separated() -> None:
    """Identical bytes under different purposes must not collide."""
    assert example_digest("v", None) != output_digest("v")


# --- the match bit --------------------------------------------------------


def test_match_and_mismatch() -> None:
    assert build_example_signals(_result())["verified_match"] == 1.0
    assert build_example_signals(_result(actual_output="5"))["verified_match"] == 0.0


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_no_usable_expected_answer_omits_the_key_entirely(empty) -> None:
    """Absent is NOT zero.

    Coercing 'cannot be checked' to 'checked and failed' would understate quality on
    exactly the datasets that lack gold labels.
    """
    signals = build_example_signals(_result(expected_output=empty))
    assert "verified_match" not in signals
    assert verified_match("anything", empty) is None


def test_an_errored_example_counts_as_a_non_match_not_as_uncheckable() -> None:
    signals = build_example_signals(_result(error_message="provider timeout"))
    assert signals["verified_match"] == 0.0


def test_match_uses_the_same_predicate_as_the_builtin_scorer() -> None:
    """Not a second implementation -- a second one would drift from the first."""
    from traigent.evaluators.base import _accuracy_values_match

    for actual, expected in (("4", "4"), (" 4 ", "4"), ("4", "5"), (4, "4")):
        expected_bit = 1.0 if _accuracy_values_match(actual, expected) else 0.0
        assert verified_match(actual, expected) == expected_bit


# --- robustness -----------------------------------------------------------


def test_unserialisable_content_still_yields_signals() -> None:
    """A weird output is still a real output; it must not break the run."""

    class Odd:
        def __repr__(self) -> str:
            return "<odd>"

    signals = build_example_signals(_result(actual_output=Odd()))
    assert len(signals["output_digest"]) == 64


def test_a_broken_example_result_yields_no_signals_rather_than_raising() -> None:
    class Exploding:
        @property
        def input_data(self):
            raise RuntimeError("boom")

    assert build_example_signals(Exploding()) == {}
