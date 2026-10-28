"""Characterization tests for BaseOverrideManager.restore_originals.

They pin the order of state reads, the restore sequence, the shape checks on
stored entries, the exception boundaries and the exact debug log output.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from traigent.integrations.base import BaseOverrideManager


class _Trace:
    """Shared event recorder."""

    def __init__(self) -> None:
        self.events: list[Any] = []


class _Target:
    """Records attribute writes; optionally refuses selected names."""

    def __init__(self, trace: _Trace, label: str, refuse: tuple[str, ...] = ()):
        object.__setattr__(self, "_trace", trace)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_refuse", refuse)

    def __setattr__(self, name: str, value: Any) -> None:
        self._trace.events.append(("set", self._label, name, value))
        if name in self._refuse:
            raise RuntimeError(f"refused {name}")


class _State:
    """Stand-in for the activation state that records every access."""

    def __init__(
        self,
        trace: _Trace,
        constructors: dict[str, Any],
        methods: dict[str, Any],
    ) -> None:
        self._trace = trace
        self._constructors = constructors
        self._methods = methods

    @property
    def original_constructors(self) -> dict[str, Any]:
        self._trace.events.append("read_constructors")
        return self._constructors

    @property
    def original_methods(self) -> dict[str, Any]:
        self._trace.events.append("read_methods")
        return self._methods

    def clear_all(self) -> None:
        self._trace.events.append("clear_all")


def _manager(
    trace: _Trace, constructors: dict[str, Any], methods: dict[str, Any]
) -> BaseOverrideManager:
    manager = BaseOverrideManager()
    setattr(manager, "_state", _State(trace, constructors, methods))
    return manager


def _run(manager: BaseOverrideManager) -> list[Any]:
    with patch("traigent.integrations.base.logger") as log:
        manager.restore_originals()
    assert log.method_calls == [] or all(c[0] == "debug" for c in log.method_calls)
    return [c.args for c in log.debug.call_args_list]


def test_empty_state_reads_each_store_once_then_clears() -> None:
    trace = _Trace()
    manager = _manager(trace, {}, {})
    assert _run(manager) == []
    assert trace.events == ["read_constructors", "read_methods", "clear_all"]


def test_constructors_restored_before_methods_in_insertion_order() -> None:
    trace = _Trace()
    c1, c2 = _Target(trace, "c1"), _Target(trace, "c2")
    m1, m2 = _Target(trace, "m1"), _Target(trace, "m2")
    manager = _manager(
        trace,
        {"a": (c1, "__init__", "orig-c1"), "b": (c2, "build", "orig-c2")},
        {"x": (m1, "run", "orig-m1"), "y": (m2, "call", "orig-m2")},
    )
    assert _run(manager) == []
    assert trace.events == [
        "read_constructors",
        ("set", "c1", "__init__", "orig-c1"),
        ("set", "c2", "build", "orig-c2"),
        "read_methods",
        ("set", "m1", "run", "orig-m1"),
        ("set", "m2", "call", "orig-m2"),
        "clear_all",
    ]


@pytest.mark.parametrize(
    "stored",
    [
        "not-a-tuple",
        ("only", "two"),
        ("one", "two", "three", "four"),
        ["list", "of", "three"],
        None,
        (),
    ],
)
def test_constructor_entries_of_wrong_shape_are_skipped_silently(stored: Any) -> None:
    trace = _Trace()
    manager = _manager(trace, {"a": stored}, {})
    assert _run(manager) == []
    assert trace.events == ["read_constructors", "read_methods", "clear_all"]


def test_constructor_failure_is_logged_and_next_entry_still_restored() -> None:
    trace = _Trace()
    bad = _Target(trace, "bad", refuse=("slot",))
    good = _Target(trace, "good")
    stored_bad = (bad, "slot", "orig-bad")
    manager = _manager(
        trace,
        {"a": stored_bad, "b": (good, "slot", "orig-good")},
        {},
    )
    calls = _run(manager)
    assert [c[0] for c in calls] == [f"Could not restore {stored_bad}: refused slot"]
    assert trace.events == [
        "read_constructors",
        ("set", "bad", "slot", "orig-bad"),
        ("set", "good", "slot", "orig-good"),
        "read_methods",
        "clear_all",
    ]


def test_constructor_base_exception_propagates_without_clearing() -> None:
    trace = _Trace()

    class _Stop(BaseException):
        pass

    class _Raiser:
        def __setattr__(self, name: str, value: Any) -> None:
            raise _Stop

    manager = _manager(trace, {"a": (_Raiser(), "slot", 1)}, {})
    with patch("traigent.integrations.base.logger"):
        with pytest.raises(_Stop):
            manager.restore_originals()
    assert trace.events == ["read_constructors"]


def test_constructor_tuple_subclass_length_consulted_once() -> None:
    trace = _Trace()
    target = _Target(trace, "t")

    class _Tuple(tuple[Any, ...]):
        lengths = 0

        def __len__(self) -> int:
            type(self).lengths += 1
            return 3

    stored = _Tuple((target, "slot", "orig"))
    manager = _manager(trace, {"a": stored}, {})
    assert _run(manager) == []
    assert _Tuple.lengths == 1
    assert ("set", "t", "slot", "orig") in trace.events


def test_constructor_raising_len_is_inside_the_handler() -> None:
    trace = _Trace()

    class _Tuple(tuple[Any, ...]):
        def __len__(self) -> int:
            raise ValueError("len failed")

    stored = _Tuple((1, 2, 3))
    manager = _manager(trace, {"a": stored}, {})
    calls = _run(manager)
    assert [c[0] for c in calls] == ["Could not restore (1, 2, 3): len failed"]
    assert trace.events == ["read_constructors", "read_methods", "clear_all"]


@pytest.mark.parametrize(
    "stored",
    [
        "plain",
        ("two", "items"),
        ("a", "b", "c", "d"),
        (None, "name", "orig"),
        ("target", "", "orig"),
        ("target", None, "orig"),
        ["list", "of", "three"],
    ],
)
def test_method_entries_not_restorable_are_skipped_silently(stored: Any) -> None:
    trace = _Trace()
    manager = _manager(trace, {}, {"x": stored})
    assert _run(manager) == []
    assert trace.events == ["read_constructors", "read_methods", "clear_all"]


def test_method_target_with_false_truth_is_still_restored() -> None:
    trace = _Trace()

    class _Falsy(_Target):
        def __bool__(self) -> bool:
            raise AssertionError("target truth must not be consulted")

        def __len__(self) -> int:
            return 0

    target = _Falsy(trace, "falsy")
    manager = _manager(trace, {}, {"x": (target, "run", "orig")})
    assert _run(manager) == []
    assert ("set", "falsy", "run", "orig") in trace.events


def test_method_name_truth_is_consulted() -> None:
    trace = _Trace()

    class _Name(str):
        def __bool__(self) -> bool:
            return False

    target = _Target(trace, "t")
    manager = _manager(trace, {}, {"x": (target, _Name("run"), "orig")})
    assert _run(manager) == []
    assert trace.events == ["read_constructors", "read_methods", "clear_all"]


def test_method_failure_logs_the_method_name_and_continues() -> None:
    trace = _Trace()
    bad = _Target(trace, "bad", refuse=("run",))
    good = _Target(trace, "good")
    manager = _manager(
        trace,
        {},
        {"x": (bad, "run", "orig-bad"), "y": (good, "run", "orig-good")},
    )
    calls = _run(manager)
    assert [c[0] for c in calls] == ["Could not restore run: refused run"]
    assert trace.events == [
        "read_constructors",
        "read_methods",
        ("set", "bad", "run", "orig-bad"),
        ("set", "good", "run", "orig-good"),
        "clear_all",
    ]


def test_method_entry_raising_len_propagates_outside_the_handler() -> None:
    trace = _Trace()

    class _Tuple(tuple[Any, ...]):
        def __len__(self) -> int:
            raise ValueError("len failed")

    manager = _manager(trace, {}, {"x": _Tuple((1, 2, 3))})
    with patch("traigent.integrations.base.logger"):
        with pytest.raises(ValueError, match="len failed"):
            manager.restore_originals()
    assert trace.events == ["read_constructors", "read_methods"]


def test_method_name_raising_truth_propagates_outside_the_handler() -> None:
    trace = _Trace()

    class _Name(str):
        def __bool__(self) -> bool:
            raise ValueError("truth failed")

    manager = _manager(trace, {}, {"x": (_Target(trace, "t"), _Name("run"), 1)})
    with patch("traigent.integrations.base.logger"):
        with pytest.raises(ValueError, match="truth failed"):
            manager.restore_originals()
    assert trace.events == ["read_constructors", "read_methods"]


def test_method_base_exception_propagates_without_clearing() -> None:
    trace = _Trace()

    class _Stop(BaseException):
        pass

    class _Raiser:
        def __setattr__(self, name: str, value: Any) -> None:
            raise _Stop

    manager = _manager(trace, {}, {"x": (_Raiser(), "run", 1)})
    with patch("traigent.integrations.base.logger"):
        with pytest.raises(_Stop):
            manager.restore_originals()
    assert trace.events == ["read_constructors", "read_methods"]


def test_methods_store_read_after_all_constructor_writes() -> None:
    trace = _Trace()
    target = _Target(trace, "c")
    manager = _manager(trace, {"a": (target, "slot", 1)}, {})
    _run(manager)
    assert trace.events.index(("set", "c", "slot", 1)) < trace.events.index(
        "read_methods"
    )


def test_real_state_end_to_end_restores_and_clears() -> None:
    class _Plain:
        pass

    manager = BaseOverrideManager()
    ctor_target = _Plain()
    meth_target = _Plain()
    state = manager._state
    state._original_constructors["k"] = (ctor_target, "attr", "ctor-original")
    state._original_methods["m"] = (meth_target, "attr2", "meth-original")
    manager.restore_originals()
    assert vars(ctor_target) == {"attr": "ctor-original"}
    assert vars(meth_target) == {"attr2": "meth-original"}
    assert state._original_constructors == {}
    assert state._original_methods == {}
