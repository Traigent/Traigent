"""Manual validation tests for pause-on-error prompts."""

from __future__ import annotations

import builtins

from traigent.core.exception_handler import (
    handle_budget_limit_reached,
    handle_vendor_exception,
)
from traigent.utils.exceptions import RateLimitError


def _mock_input(responses: list[str]):
    iterator = iter(responses)

    def _input(_prompt: str = "") -> str:
        return next(iterator)

    return _input


def test_budget_limit_raise_with_overshoot(monkeypatch) -> None:
    monkeypatch.setattr(
        builtins,
        "input",
        _mock_input(["1", "0.05", "0.2"]),
    )

    should_continue, new_limit = handle_budget_limit_reached(2.0, 2.10)

    assert should_continue is True
    assert new_limit == 2.2


def test_budget_limit_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        builtins,
        "input",
        _mock_input(["2"]),
    )

    should_continue, new_limit = handle_budget_limit_reached(2.0, 1.95)

    assert should_continue is False
    assert new_limit == 2.0


def test_budget_limit_invalid_input_then_continue(monkeypatch) -> None:
    monkeypatch.setattr(
        builtins,
        "input",
        _mock_input(["1", "abc", "0.2"]),
    )

    should_continue, new_limit = handle_budget_limit_reached(2.0, 2.0)

    assert should_continue is True
    assert new_limit == 2.2


def test_vendor_exception_resume(monkeypatch) -> None:
    monkeypatch.setattr(
        builtins,
        "input",
        _mock_input(["3", "1"]),
    )

    should_continue = handle_vendor_exception(RateLimitError("Rate limit exceeded"))

    assert should_continue is True


def test_vendor_exception_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        builtins,
        "input",
        _mock_input(["2"]),
    )

    should_continue = handle_vendor_exception(RateLimitError("Rate limit exceeded"))

    assert should_continue is False
