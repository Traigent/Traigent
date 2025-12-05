"""Tests for integration validation helpers."""

from __future__ import annotations

from typing import Any, Literal

import pytest

from traigent.integrations.utils.validation import ParameterValidator


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        ([1, 2, 3], list[int], True),
        ([1, "a"], list[int], False),
        ({"score": 0.92}, dict[str, float], True),
        ({"score": "bad"}, dict[str, float], False),
        ((1, 2), tuple[int, int], True),
        ((1, "two"), tuple[int, int], False),
        ((1, 2, 3), tuple[int, ...], True),
        ("abc", Literal["abc"], True),
        ("abc", Literal["xyz"], False),
        (42, int | str, True),
        ("answer", int | str, True),
        ({"a", "b"}, set[str], True),
        ({"a", 1}, set[str], False),
        (["token"], list[Any], True),
    ],
)
def test_check_type_compatibility(value: Any, annotation: Any, expected: bool) -> None:
    """Ensure type compatibility checks respect generic arguments."""
    assert ParameterValidator._check_type_compatibility(value, annotation) is expected


def test_check_type_compatibility_optional_sequences() -> None:
    """Optional and sequence annotations should validate each element."""
    assert ParameterValidator._check_type_compatibility(
        ["alpha", "beta"], list[str] | None
    )
    assert not ParameterValidator._check_type_compatibility(
        ["alpha", 3], list[str] | None
    )
