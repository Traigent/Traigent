"""Characterization of the per-variable value-hint collection, pinned before extraction.

Full-result assertions for row/item filtering, score coercion, de-duplication and
last-row-wins semantics, plus the exact read order and truth-evaluation counts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from traigent.config_generator.subsystems.tvar_recommendations import (
    _gated_value_recommendations_by_tvar,
)


def _collect(rows: Any) -> dict[str, tuple[tuple[Any, float], ...]]:
    result: dict[str, tuple[tuple[Any, float], ...]] = (
        _gated_value_recommendations_by_tvar(rows)
    )
    return result


def _row(name: Any, hints: Any) -> dict[str, Any]:
    return {"tvar_name": name, "value_recommendations": hints}


# ------------------------------------------------------------------------ results


def test_empty_rows_give_an_empty_result() -> None:
    assert _collect([]) == {}


def test_hints_keep_order_and_float_scores() -> None:
    rows = [
        _row(
            "model",
            [
                {"value": "a", "score": 0.5},
                {"value": "b", "score": 2},
                {"value": "c", "score": "1.25"},
            ],
        )
    ]
    out = _collect(rows)
    assert out == {"model": (("a", 0.5), ("b", 2.0), ("c", 1.25))}
    assert [type(score) for _value, score in out["model"]] == [float, float, float]


def test_name_is_stringified_and_stripped() -> None:
    rows = [
        _row("  model \n", [{"value": 1, "score": 1}]),
        _row(7, [{"value": 2, "score": 2}]),
    ]
    assert _collect(rows) == {"model": ((1, 1.0),), "7": ((2, 2.0),)}


@pytest.mark.parametrize("name", [None, "", "   ", "\t\n", 0, False, [], {}])
def test_unusable_names_skip_the_row(name: Any) -> None:
    assert _collect([_row(name, [{"value": 1, "score": 1}])]) == {}


def test_name_that_is_truthy_non_string_is_stringified() -> None:
    assert _collect([_row(1.5, [{"value": 1, "score": 1}])]) == {"1.5": ((1, 1.0),)}


def test_missing_name_or_missing_hints_skip_the_row() -> None:
    assert _collect([{"value_recommendations": [{"value": 1, "score": 1}]}]) == {}
    assert _collect([{"tvar_name": "m"}]) == {}


@pytest.mark.parametrize(
    "hints", [None, 5, {"value": 1, "score": 1}, {1, 2}, 1.5, object()]
)
def test_non_sequence_hints_skip_the_row(hints: Any) -> None:
    assert _collect([_row("m", hints)]) == {}


def test_tuple_hints_are_accepted() -> None:
    assert _collect([_row("m", ({"value": 1, "score": 1},))]) == {"m": ((1, 1.0),)}


def test_string_hints_are_a_sequence_of_unusable_items() -> None:
    assert _collect([_row("m", "abc")]) == {}


def test_bytes_hints_are_a_sequence_of_unusable_items() -> None:
    assert _collect([_row("m", b"abc")]) == {}


@pytest.mark.parametrize(
    "item",
    [
        None,
        "value",
        5,
        ["value", 1],
        ("value", 1),
        {"score": 1},
        {"Value": 1, "score": 1},
    ],
)
def test_malformed_items_are_skipped(item: Any) -> None:
    good = {"value": "ok", "score": 3}
    assert _collect([_row("m", [item, good])]) == {"m": (("ok", 3.0),)}


def test_item_without_score_raises_key_error() -> None:
    with pytest.raises(KeyError):
        _collect([_row("m", [{"value": 1}])])


@pytest.mark.parametrize("score", [None, "abc", "", [1], {"a": 1}, object(), (1,)])
def test_unconvertible_scores_skip_the_item(score: Any) -> None:
    good = {"value": "ok", "score": 1}
    assert _collect([_row("m", [{"value": "bad", "score": score}, good])]) == {
        "m": (("ok", 1.0),)
    }


@pytest.mark.parametrize(
    "score,expected",
    [
        (True, 1.0),
        (False, 0.0),
        (0, 0.0),
        (-3, -3.0),
        ("  2.5 ", 2.5),
        ("1e3", 1000.0),
        ("inf", float("inf")),
        (b"4", 4.0),
    ],
)
def test_score_coercion_matches_float(score: Any, expected: float) -> None:
    assert _collect([_row("m", [{"value": 1, "score": score}])]) == {
        "m": ((1, expected),)
    }


def test_nan_score_is_kept() -> None:
    out = _collect([_row("m", [{"value": 1, "score": float("nan")}])])
    ((value, score),) = out["m"]
    assert value == 1
    assert score != score


def test_score_error_other_than_type_or_value_error_propagates() -> None:
    class Boom:
        def __float__(self) -> float:
            raise RuntimeError("float failed")

    with pytest.raises(RuntimeError, match="float failed"):
        _collect([_row("m", [{"value": 1, "score": Boom()}])])


def test_score_type_error_and_value_error_from_float_are_swallowed() -> None:
    class TypeBad:
        def __float__(self) -> float:
            raise TypeError("no")

    class ValueBad:
        def __float__(self) -> float:
            raise ValueError("no")

    rows = [
        _row("m", [{"value": 1, "score": TypeBad()}, {"value": 2, "score": ValueBad()}])
    ]
    assert _collect(rows) == {}


def test_overflowing_score_raises_overflow_error() -> None:
    with pytest.raises(OverflowError):
        _collect([_row("m", [{"value": 1, "score": 10**400}])])


def test_values_are_returned_as_given_whatever_their_type() -> None:
    marker = object()
    out = _collect(
        [
            _row(
                "m",
                [
                    {"value": None, "score": 1},
                    {"value": marker, "score": 2},
                    {"value": [1], "score": 3},
                ],
            )
        ]
    )
    assert out["m"][0] == (None, 1.0)
    assert out["m"][1][0] is marker
    assert out["m"][2] == ([1], 3.0)


def test_duplicates_are_removed_by_type_name_and_repr_keeping_the_first_score() -> None:
    rows = [
        _row(
            "m",
            [
                {"value": 1, "score": 1},
                {"value": 1.0, "score": 2},
                {"value": True, "score": 3},
                {"value": "1", "score": 4},
                {"value": 1, "score": 5},
                {"value": "1", "score": 6},
                {"value": 1.0, "score": 7},
            ],
        )
    ]
    assert _collect(rows) == {"m": ((1, 1.0), (1.0, 2.0), (True, 3.0), ("1", 4.0))}


def test_all_items_unusable_leaves_no_entry() -> None:
    assert _collect([_row("m", [None, "x", {"value": 1, "score": "bad"}])]) == {}


def test_empty_hint_list_leaves_no_entry() -> None:
    assert _collect([_row("m", [])]) == {}


def test_later_row_with_the_same_name_replaces_the_earlier() -> None:
    rows = [
        _row("m", [{"value": "a", "score": 1}]),
        _row(" m ", [{"value": "b", "score": 2}]),
    ]
    assert _collect(rows) == {"m": (("b", 2.0),)}


def test_later_row_without_usable_hints_keeps_the_earlier() -> None:
    rows = [
        _row("m", [{"value": "a", "score": 1}]),
        _row("m", [{"value": "b", "score": "bad"}]),
    ]
    assert _collect(rows) == {"m": (("a", 1.0),)}


def test_result_keeps_first_seen_name_order() -> None:
    rows = [
        _row("b", [{"value": 1, "score": 1}]),
        _row("a", [{"value": 1, "score": 1}]),
        _row("b", [{"value": 2, "score": 2}]),
    ]
    assert list(_collect(rows)) == ["b", "a"]


def test_result_values_are_tuples() -> None:
    out = _collect([_row("m", [{"value": 1, "score": 1}])])
    assert type(out["m"]) is tuple


# --------------------------------------------------------- reads and truth counts


class _Row(Mapping[str, Any]):
    def __init__(self, data: dict[str, Any], log: list[str]) -> None:
        self._data = data
        self._log = log

    def get(self, key: str, default: Any = None) -> Any:
        self._log.append(f"get:{key}")
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        self._log.append(f"getitem:{key}")
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class _Item(Mapping[str, Any]):
    def __init__(self, data: dict[str, Any], log: list[str]) -> None:
        self._data = data
        self._log = log

    def __contains__(self, key: object) -> bool:
        self._log.append(f"contains:{key}")
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        self._log.append(f"getitem:{key}")
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_row_and_item_read_sequence_for_a_good_item() -> None:
    log: list[str] = []
    item = _Item({"value": "v", "score": 2}, log)
    row = _Row({"tvar_name": "m", "value_recommendations": [item]}, log)
    assert _collect([row]) == {"m": (("v", 2.0),)}
    assert log == [
        "get:tvar_name",
        "get:value_recommendations",
        "contains:value",
        "getitem:score",
        "getitem:value",
    ]


def test_row_reads_stop_after_the_name_when_it_is_unusable_but_hints_are_read_first() -> (
    None
):
    log: list[str] = []
    row = _Row(
        {"tvar_name": "  ", "value_recommendations": [{"value": 1, "score": 1}]}, log
    )
    assert _collect([row]) == {}
    assert log == ["get:tvar_name", "get:value_recommendations"]


def test_item_without_value_key_is_never_scored() -> None:
    log: list[str] = []
    item = _Item({"score": 1}, log)
    row = _Row({"tvar_name": "m", "value_recommendations": [item]}, log)
    assert _collect([row]) == {}
    assert log == ["get:tvar_name", "get:value_recommendations", "contains:value"]


def test_unscorable_item_does_not_read_its_value() -> None:
    log: list[str] = []
    item = _Item({"value": "v", "score": "bad"}, log)
    row = _Row({"tvar_name": "m", "value_recommendations": [item]}, log)
    assert _collect([row]) == {}
    assert log == [
        "get:tvar_name",
        "get:value_recommendations",
        "contains:value",
        "getitem:score",
    ]


def test_raising_score_read_that_is_a_key_error_propagates() -> None:
    class Missing(_Item):
        def __getitem__(self, key: str) -> Any:
            raise KeyError(key)

    log: list[str] = []
    with pytest.raises(KeyError):
        _collect([_row("m", [Missing({"value": 1, "score": 1}, log)])])


def test_raising_score_read_that_is_a_type_error_is_swallowed() -> None:
    class Odd(_Item):
        def __getitem__(self, key: str) -> Any:
            if key == "score":
                raise TypeError("odd")
            return super().__getitem__(key)

    log: list[str] = []
    assert _collect([_row("m", [Odd({"value": 1, "score": 1}, log)])]) == {}


def test_raising_value_read_propagates_after_a_good_score() -> None:
    class Odd(_Item):
        def __getitem__(self, key: str) -> Any:
            if key == "value":
                raise RuntimeError("value failed")
            return super().__getitem__(key)

    log: list[str] = []
    with pytest.raises(RuntimeError, match="value failed"):
        _collect([_row("m", [Odd({"value": 1, "score": 1}, log)])])


def test_raise_once_value_read_is_not_retried() -> None:
    class Once(_Item):
        armed = True

        def __getitem__(self, key: str) -> Any:
            if key == "value" and self.armed:
                self.armed = False
                raise RuntimeError("first value read failed")
            return super().__getitem__(key)

    log: list[str] = []
    item = Once({"value": 1, "score": 1}, log)
    with pytest.raises(RuntimeError, match="first value read failed"):
        _collect([_row("m", [item])])
    assert _collect([_row("m", [item])]) == {"m": ((1, 1.0),)}


def test_raising_contains_propagates() -> None:
    class Odd(_Item):
        def __contains__(self, key: object) -> bool:
            raise RuntimeError("contains failed")

    with pytest.raises(RuntimeError, match="contains failed"):
        _collect([_row("m", [Odd({"value": 1, "score": 1}, [])])])


def test_raising_row_accessors_propagate() -> None:
    class Odd(_Row):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "value_recommendations":
                raise RuntimeError("hints failed")
            return super().get(key, default)

    with pytest.raises(RuntimeError, match="hints failed"):
        _collect([Odd({"tvar_name": "m", "value_recommendations": []}, [])])


def test_hint_sequence_is_iterated_once_and_raising_iteration_propagates() -> None:
    class Hints(list[Any]):
        iterations = 0

        def __iter__(self) -> Iterator[Any]:
            self.iterations += 1
            return super().__iter__()

    hints = Hints([{"value": 1, "score": 1}])
    assert _collect([_row("m", hints)]) == {"m": ((1, 1.0),)}
    assert hints.iterations == 1

    class Broken(list[Any]):
        def __iter__(self) -> Iterator[Any]:
            raise RuntimeError("iteration failed")

    with pytest.raises(RuntimeError, match="iteration failed"):
        _collect([_row("m", Broken([{"value": 1, "score": 1}]))])


def test_name_truth_is_evaluated_once_and_a_falsy_nonempty_name_is_replaced() -> None:
    class FalsyName(str):
        calls = 0

        def __bool__(self) -> bool:
            FalsyName.calls += 1
            return False

    FalsyName.calls = 0
    assert _collect([_row(FalsyName("model"), [{"value": 1, "score": 1}])]) == {}
    assert FalsyName.calls == 1


def test_name_with_raising_truth_propagates() -> None:
    class Odd(str):
        def __bool__(self) -> bool:
            raise RuntimeError("truth failed")

    with pytest.raises(RuntimeError, match="truth failed"):
        _collect([_row(Odd("m"), [{"value": 1, "score": 1}])])


def test_stringification_of_a_name_object_happens_once() -> None:
    class Name:
        calls = 0

        def __str__(self) -> str:
            Name.calls += 1
            return " nm "

    Name.calls = 0
    assert _collect([_row(Name(), [{"value": 1, "score": 1}])]) == {"nm": ((1, 1.0),)}
    assert Name.calls == 1


def test_a_failing_row_stops_processing_of_later_rows() -> None:
    log: list[str] = []

    class Odd(_Row):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("row failed")

    rows = [Odd({}, log), _Row({"tvar_name": "m", "value_recommendations": []}, log)]
    with pytest.raises(RuntimeError, match="row failed"):
        _collect(rows)
    assert log == []


def test_rows_are_consumed_in_order_and_hint_names_with_failures_leave_earlier_results() -> (
    None
):
    log: list[str] = []
    good = _Row(
        {"tvar_name": "a", "value_recommendations": [{"value": 1, "score": 1}]}, log
    )
    later = _Row({"tvar_name": "b", "value_recommendations": [{"value": 2}]}, log)
    with pytest.raises(KeyError):
        _collect([good, later])
    assert log == [
        "get:tvar_name",
        "get:value_recommendations",
        "get:tvar_name",
        "get:value_recommendations",
    ]


def test_dedupe_failure_on_repr_propagates() -> None:
    class BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

    with pytest.raises(RuntimeError, match="repr failed"):
        _collect([_row("m", [{"value": BadRepr(), "score": 1}])])


def test_unhashable_and_equal_but_distinct_values_are_deduplicated_by_repr_only() -> (
    None
):
    rows = [
        _row(
            "m",
            [
                {"value": [1], "score": 1},
                {"value": [1], "score": 2},
                {"value": {"a": 1}, "score": 3},
                {"value": {"a": 1}, "score": 4},
            ],
        )
    ]
    assert _collect(rows) == {"m": (([1], 1.0), ({"a": 1}, 3.0))}
