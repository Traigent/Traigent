"""Ordering contract for the reversed-string tie-break wrapper."""

from __future__ import annotations

from traigent.knobs.runtime import _ReverseStr


def test_reversed_string_ordering_is_complete() -> None:
    a, b = _ReverseStr("a"), _ReverseStr("b")
    assert b < a
    assert a > b
    assert b <= a
    assert a >= b
    assert a <= _ReverseStr("a")
    assert a >= _ReverseStr("a")
    assert not a < _ReverseStr("a")
    assert not a > _ReverseStr("a")


def test_max_tie_break_picks_smallest_string() -> None:
    items = [(1.0, _ReverseStr("b")), (1.0, _ReverseStr("a")), (1.0, _ReverseStr("c"))]
    assert max(items)[1].value == "a"
