"""Characterization tests for ``_normalize_metric_selector``.

Pin the accepted selector shapes, the exact error messages, the comparison and
truth-evaluation forms used on user-supplied values, de-duplication order, and
how malformed or hostile inputs behave.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from traigent.core.stop_conditions import _normalize_metric_selector

ENTRIES = "metrics entries must be non-empty strings"
SHAPE = "metrics must be 'auto', a string, or a list of strings"


def _call(value: Any, name: str = "metrics") -> Any:
    return _normalize_metric_selector(value, name)


def _fails(value: Any, message: str, name: str = "metrics") -> None:
    with pytest.raises(ValueError) as info:
        _normalize_metric_selector(value, name)
    assert type(info.value) is ValueError
    assert str(info.value) == message


class _AlwaysEqual(str):
    """Non-empty str whose equality (and inequality) is scripted."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = str.__hash__


class _NeverEqual(str):
    def __eq__(self, other: object) -> bool:
        return False

    def __ne__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


class _FalseTruth(str):
    """Non-empty str whose truth value is False."""

    def __bool__(self) -> bool:
        return False


class _TrueTruth(str):
    """Empty str whose truth value is True."""

    def __bool__(self) -> bool:
        return True


def test_none_and_auto_select_nothing() -> None:
    assert _call(None) is None
    assert _call("auto") is None


def test_auto_comparison_is_exact() -> None:
    assert _call("auto") is None
    assert _call(("auto",)) == ("auto",)
    assert _call(["auto"]) == ("auto",)
    assert _call("Auto") == ("Auto",)
    assert _call("auto ") == ("auto ",)
    assert _call(" auto") == (" auto",)


def test_single_string_becomes_a_one_tuple() -> None:
    assert _call("accuracy") == ("accuracy",)
    result = _call("f1")
    assert type(result) is tuple


def test_empty_string_is_rejected() -> None:
    _fails("", ENTRIES)


def test_sequence_preserves_order_and_removes_duplicates() -> None:
    assert _call(["b", "a", "b", "c", "a"]) == ("b", "a", "c")
    assert _call(("x", "x", "x")) == ("x",)
    assert _call([]) == ()
    assert _call(()) == ()


def test_sequence_with_auto_entry_is_not_the_auto_selector() -> None:
    assert _call(["auto", "a"]) == ("auto", "a")


def test_returned_items_keep_their_identity() -> None:
    class _Sub(str):
        pass

    first = _Sub("keep")
    result = _call([first, "other"])
    assert result[0] is first
    assert type(result[0]) is _Sub


@pytest.mark.parametrize(
    "value",
    [
        [""],
        ["a", ""],
        ["", "a"],
        [None],
        [1],
        ["a", 1],
        [b"a"],
        [["a"]],
        [("a",)],
        [None, "a"],
        ("a", None),
    ],
)
def test_invalid_entries_are_rejected(value: Any) -> None:
    _fails(value, ENTRIES)


@pytest.mark.parametrize(
    "value",
    [
        5,
        2.5,
        True,
        False,
        {"a"},
        {"a": 1},
        frozenset({"a"}),
        b"a",
        bytearray(b"a"),
        b"",
    ],
)
def test_other_shapes_are_rejected(value: Any) -> None:
    _fails(value, SHAPE)


def test_generator_and_iterators_are_not_sequences() -> None:
    _fails((x for x in ["a"]), SHAPE)
    _fails(iter(["a"]), SHAPE)


def test_range_is_a_sequence_of_non_strings() -> None:
    _fails(range(2), ENTRIES)
    assert _call(range(0)) == ()


def test_error_messages_use_the_given_name() -> None:
    _fails("", "custom_name entries must be non-empty strings", name="custom_name")
    _fails([1], "custom_name entries must be non-empty strings", name="custom_name")
    _fails(
        5,
        "custom_name must be 'auto', a string, or a list of strings",
        name="custom_name",
    )


def test_name_is_formatted_once_per_failure_and_never_on_success() -> None:
    calls: list[str] = []

    class _Name:
        def __str__(self) -> str:
            calls.append("str")
            return "n"

    name: Any = _Name()
    assert _normalize_metric_selector(["a"], name) == ("a",)
    assert _normalize_metric_selector("auto", name) is None
    assert calls == []
    with pytest.raises(ValueError, match="n must be 'auto'"):
        _normalize_metric_selector(5, name)
    with pytest.raises(ValueError, match="n entries must"):
        _normalize_metric_selector([""], name)
    assert calls == ["str", "str"]


def test_raising_name_format_replaces_the_value_error() -> None:
    class _BadName:
        def __str__(self) -> str:
            raise RuntimeError("name")

    name: Any = _BadName()
    with pytest.raises(RuntimeError, match="name"):
        _normalize_metric_selector(5, name)
    with pytest.raises(RuntimeError, match="name"):
        _normalize_metric_selector([""], name)
    assert _normalize_metric_selector(["a"], name) == ("a",)


def test_scalar_str_equality_is_the_left_operand_form_and_called_once() -> None:
    events: list[tuple[str, object]] = []

    class _Recording(str):
        def __eq__(self, other: object) -> bool:
            events.append(("eq", other))
            return False

        def __ne__(self, other: object) -> bool:
            events.append(("ne", other))
            return True

        __hash__ = str.__hash__

    value = _Recording("x")
    result = _call(value)
    assert events == [("eq", "auto")]
    assert result[0] is value


def test_str_subclass_equal_to_everything_selects_nothing() -> None:
    assert _call(_AlwaysEqual("anything")) is None


def test_str_subclass_unequal_to_everything_is_a_plain_single_string() -> None:
    value = _NeverEqual("auto")
    result = _call(value)
    assert len(result) == 1
    assert result[0] is value


def test_non_str_with_scripted_equality_selects_nothing() -> None:
    class _Seq(list[Any]):
        def __eq__(self, other: object) -> bool:
            return other == "auto"

    assert _call(_Seq(["a"])) is None


def test_equality_is_not_consulted_for_none() -> None:
    assert _call(None) is None


def test_raising_equality_propagates() -> None:
    class _Boom(list[Any]):
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("eq")

    with pytest.raises(RuntimeError, match="eq"):
        _call(_Boom())


def test_raise_once_equality_is_not_retried() -> None:
    state = {"calls": 0}

    class _Once(list[Any]):
        def __eq__(self, other: object) -> bool:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("first")
            return False

    value = _Once(["a"])
    with pytest.raises(RuntimeError, match="first"):
        _call(value)
    assert state["calls"] == 1
    assert _call(value) == ("a",)
    assert state["calls"] == 2


def test_scalar_truth_is_evaluated_once() -> None:
    counter = {"n": 0}

    class _Counted(str):
        def __bool__(self) -> bool:
            counter["n"] += 1
            return True

    assert _call(_Counted("m")) == ("m",)
    assert counter["n"] == 1


def test_non_empty_str_with_false_truth_is_rejected_as_empty() -> None:
    _fails(_FalseTruth("metric"), ENTRIES)


def test_empty_str_with_true_truth_is_accepted() -> None:
    result = _call(_TrueTruth(""))
    assert result == ("",)
    assert type(result[0]) is _TrueTruth


def test_item_with_false_truth_is_rejected() -> None:
    _fails([_FalseTruth("metric")], ENTRIES)
    _fails(["a", _FalseTruth("metric")], ENTRIES)


def test_item_with_true_truth_but_empty_is_accepted() -> None:
    result = _call([_TrueTruth("")])
    assert result == ("",)


def test_item_truth_is_evaluated_once_each() -> None:
    counter = {"n": 0}

    class _Counted(str):
        def __bool__(self) -> bool:
            counter["n"] += 1
            return True

    assert _call([_Counted("a"), _Counted("b")]) == ("a", "b")
    assert counter["n"] == 2


def test_non_str_items_never_have_their_truth_evaluated() -> None:
    class _Hostile:
        def __bool__(self) -> bool:
            raise RuntimeError("truth")

    _fails([_Hostile()], ENTRIES)


def test_duplicate_detection_uses_equality_in_list_membership() -> None:
    result = _call([_AlwaysEqual("a"), _AlwaysEqual("b")])
    assert result == ("a",)
    assert type(result[0]) is _AlwaysEqual


def test_unequal_items_are_all_kept() -> None:
    result = _call([_NeverEqual("a"), _NeverEqual("a")])
    assert len(result) == 2


def test_duplicate_detection_equality_trace() -> None:
    events: list[str] = []

    class _Tracked(str):
        def __eq__(self, other: object) -> bool:
            events.append(f"eq:{self}:{other}")
            return str.__eq__(self, other)

        __hash__ = str.__hash__

    result = _call([_Tracked("a"), _Tracked("b"), _Tracked("a")])
    assert events == ["eq:a:b", "eq:a:a"]
    assert [str(x) for x in result] == ["a", "b"]


def test_iteration_stops_at_first_invalid_item() -> None:
    seen: list[int] = []

    class _Seq(Sequence[Any]):
        def __init__(self, items: list[Any]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: Any) -> Any:
            return self._items[index]

        def __iter__(self) -> Iterator[Any]:
            for i, item in enumerate(self._items):
                seen.append(i)
                yield item

    _fails(_Seq(["a", "", "c"]), ENTRIES)
    assert seen == [0, 1]


def test_custom_sequence_is_iterated_exactly_once() -> None:
    calls = {"iter": 0}

    class _Seq(Sequence[Any]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: Any) -> Any:
            return ["a", "b"][index]

        def __iter__(self) -> Iterator[Any]:
            calls["iter"] += 1
            return iter(["a", "b"])

    assert _call(_Seq()) == ("a", "b")
    assert calls["iter"] == 1


def test_raising_iteration_propagates() -> None:
    class _Seq(Sequence[Any]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: Any) -> Any:
            return "a"

        def __iter__(self) -> Iterator[Any]:
            raise RuntimeError("iter")

    with pytest.raises(RuntimeError, match="iter"):
        _call(_Seq())


def test_iteration_error_after_valid_items_propagates_not_value_error() -> None:
    class _Seq(Sequence[Any]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: Any) -> Any:
            return "a"

        def __iter__(self) -> Iterator[Any]:
            yield "a"
            raise RuntimeError("mid")

    with pytest.raises(RuntimeError, match="mid"):
        _call(_Seq())


def test_raise_once_iteration_is_not_retried() -> None:
    state = {"calls": 0}

    class _Seq(Sequence[Any]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: Any) -> Any:
            return "a"

        def __iter__(self) -> Iterator[Any]:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("first")
            return iter(["a"])

    seq = _Seq()
    with pytest.raises(RuntimeError, match="first"):
        _call(seq)
    assert state["calls"] == 1
    assert _call(seq) == ("a",)


def test_base_exception_during_iteration_propagates() -> None:
    class _Seq(Sequence[Any]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: Any) -> Any:
            return "a"

        def __iter__(self) -> Iterator[Any]:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _call(_Seq())


def test_str_check_precedes_sequence_check() -> None:
    # A str is a Sequence too; it must be treated as a single selector, not split.
    assert _call("abc") == ("abc",)
