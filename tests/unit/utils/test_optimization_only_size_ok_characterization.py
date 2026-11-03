"""Characterization tests for ``_size_ok`` in the optimization-only projection.

Pin the size bound for a projected value, the separate cap for a content
identity object (top level and under ``metadata``), the order in which parts
are measured, input immutability, and how malformed values behave.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as mod
from traigent.utils.optimization_only_egress import MetadataContractViolation

IDENTITY_CAP = 30


@pytest.fixture(autouse=True)
def _small_identity_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "MAX_IDENTITY_BYTES", IDENTITY_CAP)


def _ok(value: Any, limit: int, *, identity: bool = False) -> None:
    assert mod._size_ok(value, limit, identity=identity) is None


def _violates(value: Any, limit: int, *, identity: bool = False) -> None:
    with pytest.raises(MetadataContractViolation) as info:
        mod._size_ok(value, limit, identity=identity)
    assert type(info.value) is MetadataContractViolation


class _Str:
    """Serialized via ``str()`` (json ``default=str``) with a recorded trace."""

    def __init__(self, trace: list[str], tag: str, text: str = "s") -> None:
        self._trace = trace
        self._tag = tag
        self._text = text

    def __str__(self) -> str:
        self._trace.append(self._tag)
        return self._text


def test_production_caps_are_the_documented_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    assert mod.MAX_METADATA_BYTES == 32768
    assert mod.MAX_IDENTITY_BYTES == 1056768


def test_identity_flag_uses_identity_cap_and_ignores_limit() -> None:
    _ok("x" * 28, 1, identity=True)  # json size 30 == cap
    _violates("x" * 29, 10**9, identity=True)  # json size 31 > cap


def test_identity_flag_truthiness_is_evaluated_once_and_used() -> None:
    calls: list[str] = []

    class _Flag:
        def __bool__(self) -> bool:
            calls.append("bool")
            return True

    flag: Any = _Flag()
    mod._size_ok("x" * 28, 1, identity=flag)
    assert calls == ["bool"]
    _violates("x" * 40, 10**9, identity=True)


def test_falsy_identity_flag_uses_limit() -> None:
    flag: Any = 0
    mod._size_ok("x" * 8, 10, identity=flag)
    with pytest.raises(MetadataContractViolation):
        mod._size_ok("x" * 9, 10, identity=flag)


@pytest.mark.parametrize(
    ("value", "limit", "ok"),
    [
        ("x" * 8, 10, True),
        ("x" * 9, 10, False),
        ([1, 2, 3], 7, True),
        ([1, 2, 3], 6, False),
        (None, 4, True),
        (None, 3, False),
        ({"a": "xx"}, 10, True),
        ({"a": "xx"}, 9, False),
        ({}, 2, True),
        ({}, 1, False),
        (12345, 5, True),
        (12345, 4, False),
    ],
)
def test_plain_values_are_bounded_by_limit(value: Any, limit: int, ok: bool) -> None:
    if ok:
        _ok(value, limit)
    else:
        _violates(value, limit)


def test_limit_is_independent_of_identity_cap() -> None:
    _ok("x" * 40, 42)  # larger than the identity cap, within the limit
    _violates("x" * 40, 41)


def test_top_level_identity_is_measured_against_its_own_cap() -> None:
    base = {"a": 1}
    for text_len, ok in ((28, True), (29, False)):
        value = {**base, "content_identity": "x" * text_len}
        if ok:
            _ok(value, 7)
        else:
            _violates(value, 10**9)


def test_remainder_excludes_identity_and_uses_limit() -> None:
    value = {"a": 1, "content_identity": "x" * 28}
    _ok(value, 7)  # {"a":1} is 7 bytes
    _violates(value, 6)


def test_metadata_identity_is_measured_against_its_own_cap() -> None:
    for text_len, ok in ((28, True), (29, False)):
        value = {"metadata": {"b": 1, "content_identity": "y" * text_len}}
        if ok:
            _ok(value, 20)
        else:
            _violates(value, 10**9)


def test_metadata_remainder_excludes_identity_and_uses_limit() -> None:
    value = {"metadata": {"b": 1, "content_identity": "y" * 28}}
    _ok(value, 20)  # {"metadata":{"b":1}} is 20 bytes
    _violates(value, 19)


def test_both_identity_slots_are_stripped() -> None:
    value = {
        "a": 1,
        "content_identity": "x" * 28,
        "metadata": {"b": 1, "content_identity": "y" * 28},
    }
    _ok(value, 26)  # {"a":1,"metadata":{"b":1}} is 26 bytes
    _violates(value, 25)


def test_identity_that_is_none_is_removed_from_the_measured_remainder() -> None:
    value = {"a": 1, "content_identity": None}
    _ok(value, 7)
    _violates(value, 6)


def test_identity_inside_a_non_dict_metadata_is_not_stripped() -> None:
    value = {"metadata": [{"content_identity": "z" * 100}]}
    _violates(value, 50)
    _ok(value, 200)


def test_deeper_identity_keys_are_not_stripped() -> None:
    value = {"outer": {"content_identity": "z" * 100}}
    _violates(value, 50)


def test_non_dict_metadata_value_is_left_alone() -> None:
    value = {"metadata": "m", "a": 1}
    _ok(value, 22)  # {"metadata":"m","a":1}
    _violates(value, 21)


def test_identity_flag_does_not_strip_content_identity() -> None:
    value = {"content_identity": "x" * 28}
    _violates(value, 10**9, identity=True)  # whole object is 47 bytes > 30


def test_input_is_not_mutated() -> None:
    value = {
        "a": 1,
        "content_identity": "x" * 5,
        "metadata": {"b": 1, "content_identity": "y" * 5},
    }
    snapshot = copy.deepcopy(value)
    metadata_ref = value["metadata"]
    _ok(value, 100)
    assert value == snapshot
    assert value["metadata"] is metadata_ref
    assert metadata_ref == snapshot["metadata"]


def test_input_is_not_mutated_when_the_cap_is_exceeded() -> None:
    value = {"metadata": {"content_identity": "y" * 100}}
    snapshot = copy.deepcopy(value)
    _violates(value, 10**9)
    assert value == snapshot


def test_measurement_order_top_identity_metadata_identity_remainder() -> None:
    trace: list[str] = []
    value = {
        "content_identity": _Str(trace, "top"),
        "z": _Str(trace, "rest"),
        "metadata": {"content_identity": _Str(trace, "meta")},
    }
    _ok(value, 10**6)
    assert trace == ["top", "meta", "rest"]


def test_oversize_top_identity_stops_before_later_measurements() -> None:
    trace: list[str] = []
    value = {
        "content_identity": _Str(trace, "top", "t" * 40),
        "z": _Str(trace, "rest"),
        "metadata": {"content_identity": _Str(trace, "meta")},
    }
    _violates(value, 10**6)
    assert trace == ["top"]


def test_oversize_metadata_identity_stops_before_remainder() -> None:
    trace: list[str] = []
    value = {
        "content_identity": _Str(trace, "top"),
        "z": _Str(trace, "rest"),
        "metadata": {"content_identity": _Str(trace, "meta", "m" * 40)},
    }
    _violates(value, 10**6)
    assert trace == ["top", "meta"]


def test_identity_flag_measures_the_whole_value_once() -> None:
    trace: list[str] = []
    _ok({"k": _Str(trace, "only")}, 1, identity=True)
    assert trace == ["only"]


def test_dict_subclass_get_is_not_consulted() -> None:
    class _Hostile(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get")

    _ok(_Hostile({"a": 1}), 7)
    _violates(_Hostile({"a": 1}), 6)


def test_dict_subclass_metadata_is_copied_before_stripping() -> None:
    class _Meta(dict[str, Any]):
        pass

    meta = _Meta({"b": 1, "content_identity": "y" * 5})
    _ok({"metadata": meta}, 20)
    assert dict(meta) == {"b": 1, "content_identity": "y" * 5}


def test_circular_remainder_is_a_violation() -> None:
    loop: dict[str, Any] = {}
    loop["self"] = loop
    _violates({"a": loop}, 10**6)
    _violates(loop, 10**6, identity=True)


def test_unserializable_key_is_a_violation() -> None:
    _violates({(1, 2): 1}, 10**6)


def test_circular_identity_is_a_violation() -> None:
    loop: dict[str, Any] = {}
    loop["self"] = loop
    _violates({"content_identity": loop}, 10**6)
    _violates({"metadata": {"content_identity": loop}}, 10**6)


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_other_errors_from_serialization_propagate(exc: BaseException) -> None:
    class _Bad:
        def __str__(self) -> str:
            raise exc

    for value in (
        {"x": _Bad()},
        {"content_identity": _Bad()},
        {"metadata": {"content_identity": _Bad()}},
    ):
        with pytest.raises(type(exc)):
            mod._size_ok(value, 10**6)
    with pytest.raises(type(exc)):
        mod._size_ok({"x": _Bad()}, 10**6, identity=True)


def test_raise_once_serializer_is_not_retried() -> None:
    state = {"calls": 0}

    class _Once:
        def __str__(self) -> str:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("first")
            return "ok"

    value = {"content_identity": _Once()}
    with pytest.raises(RuntimeError, match="first"):
        mod._size_ok(value, 10**6)
    assert state["calls"] == 1


def test_metadata_get_is_read_from_a_copy_and_missing_key_is_fine() -> None:
    _ok({"a": 1}, 7)
    _ok({"metadata": None}, 17)  # {"metadata":null}
    _violates({"metadata": None}, 16)
