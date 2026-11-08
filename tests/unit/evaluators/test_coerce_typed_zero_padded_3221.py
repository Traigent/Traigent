"""Typed int/float gold values keep the identifier guard the string path has."""

import pytest

from traigent.evaluators.base import _accuracy_matches_after_unwrap as match


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("007", 7),
        ("00501", 501),
        ("-007", -7),
        ("+007", 7),
        ("00.5", 0.5),
        ("1_000", 1000),
        ("１２", 12),
        ("1٣", 13),
    ],
)
def test_identifier_shaped_or_non_ascii_numeric_string_does_not_match_typed(
    actual: str, expected: float
) -> None:
    assert match(actual, expected) is False


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("7", 7),
        (" 7 ", 7),
        ("1.0", 1),
        ("0", 0),
        ("0.5", 0.5),
        (".5", 0.5),
        ("-0.5", -0.5),
        ("1e3", 1000),
        ("-7", -7),
        ("0", 0.0),
    ],
)
def test_plain_numeric_string_still_matches_typed(actual: str, expected: float) -> None:
    assert match(actual, expected) is True


def test_string_string_control_unchanged() -> None:
    assert match("007", "7") is False
    assert match("7", "007") is False
