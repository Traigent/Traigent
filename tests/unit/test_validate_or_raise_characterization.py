"""Golden characterization for validation.validate_or_raise."""

from __future__ import annotations

import pytest

from traigent.utils.validation import ValidationResult, validate_or_raise


def _from_api(result):
    code = compile(
        "def call(f, r):\n    return f(r)\n", "/x/traigent/api/functions.py", "exec"
    )
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - test-only frame filename simulation
    return ns["call"](validate_or_raise, result)


def test_valid_is_noop():
    assert validate_or_raise(ValidationResult()) is None
    assert _from_api(ValidationResult()) is None


def test_normal_caller_uses_raise_if_invalid():
    r = ValidationResult()
    r.add_error("f", "must be positive")
    with pytest.raises(Exception) as ei:
        validate_or_raise(r)
    assert not isinstance(ei.value, ValueError) or "Validation failed" in str(ei.value)
    assert "f: must be positive" in str(ei.value)


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("Value must be Positive", "Must be positive"),
        ("INVALID CHOICE here", "Must be one of"),
        ("Expected dict, got list", "Expected dict, got list"),
        ("expected dict but got int", "Expected dict, got int"),
        ("something else", "something else"),
        ("positive and invalid choice", "Must be positive"),
    ],
)
def test_api_functions_caller_maps_first_error(msg, expected):
    r = ValidationResult()
    r.add_error("a", msg)
    r.add_error("b", "second ignored positive")
    with pytest.raises(ValueError) as ei:
        _from_api(r)
    assert str(ei.value) == expected


def test_api_functions_caller_invalid_without_errors(monkeypatch):
    class Fake:
        is_valid = False
        errors: list = []

    with pytest.raises(ValueError) as ei:
        _from_api(Fake())
    assert str(ei.value) == "Validation failed"


def test_positive_branch_suppresses_context():
    r = ValidationResult()
    r.add_error("a", "positive")
    with pytest.raises(ValueError) as ei:
        _from_api(r)
    assert ei.value.__suppress_context__ is True
    assert ei.value.__cause__ is None
