"""The keyless quickstart is a plumbing check, never a fake winner (#1690)."""

from __future__ import annotations

from traigent.examples.quickstart._messages import completion_message


def test_completion_message_is_labelled_plumbing_check() -> None:
    msg = completion_message(6).lower()
    assert "plumbing check" in msg
    assert "simulated" in msg
    assert "not a model comparison" in msg
    assert "api key" in msg


def test_completion_message_makes_no_winner_claim() -> None:
    msg = completion_message(6).lower()
    assert "best config " not in msg
    assert "scored" not in msg
    assert "winner" not in msg
    assert "0.8" not in msg
    assert "no model or configuration was found to be best" in msg
