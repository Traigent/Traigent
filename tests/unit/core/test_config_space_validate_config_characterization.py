"""Characterization of ConfigurationSpace.validate_config: order, reads, boundaries."""

from __future__ import annotations

from _collections_abc import dict_keys
from collections.abc import Callable
from typing import Any, cast

import pytest

from traigent.core import types as types_module
from traigent.core.types import ConfigurationSpace, Parameter, ParameterType

Trace = list[tuple[Any, ...]]


class RecordingLogger:
    """Stands in for the module logger and records debug calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def debug(self, *args: Any) -> None:
        self.calls.append(("debug", args))

    def __getattr__(self, name: str) -> Callable[..., None]:
        def other(*args: Any) -> None:
            self.calls.append((name, args))

        return other


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> RecordingLogger:
    recorder = RecordingLogger()
    monkeypatch.setattr(types_module, "logger", recorder)
    return recorder


def _make_probe(
    name: str,
    trace: Trace,
    result: Any = True,
    raises: Exception | None = None,
    name_raises_on_read: int | None = None,
) -> Parameter:
    state = {"reads": 0}

    class Probe(Parameter):
        def __getattribute__(self, attr: str) -> Any:
            if attr == "name":
                state["reads"] += 1
                trace.append(("name", name, state["reads"]))
                if name_raises_on_read == state["reads"]:
                    raise RuntimeError("name accessor failed")
            return super().__getattribute__(attr)

        def validate_value(self, value: Any) -> bool:
            trace.append(("validate", name, value))
            if raises is not None:
                raise raises
            return cast(bool, result)

    return Probe(name=name, type=ParameterType.CATEGORICAL, bounds=[1])


class RecordingConfig(dict[str, Any]):
    def __init__(self, trace: Trace, data: dict[str, Any]) -> None:
        super().__init__(data)
        self._trace = trace

    def keys(self) -> dict_keys[str, Any]:
        self._trace.append(("keys",))
        return super().keys()

    def __contains__(self, key: object) -> bool:
        self._trace.append(("in", key))
        return super().__contains__(key)

    def __getitem__(self, key: str) -> Any:
        self._trace.append(("item", key))
        return super().__getitem__(key)


def _constraint_probe(trace: Trace) -> Callable[[Any], bool]:
    def probe(config: Any) -> bool:
        trace.append(("constraint",))
        return True

    return probe


def _space(params: list[Parameter], constraints: Any = None) -> ConfigurationSpace:
    return ConfigurationSpace(parameters=params, constraints=constraints)


def _p(name: str, kind: ParameterType, bounds: Any) -> Parameter:
    return Parameter(name=name, type=kind, bounds=bounds)


def test_results_are_exact_booleans_for_real_parameters() -> None:
    space = _space(
        [
            _p("t", ParameterType.FLOAT, (0.0, 1.0)),
            _p("n", ParameterType.INTEGER, (1, 3)),
            _p("c", ParameterType.CATEGORICAL, ["a", "b"]),
            _p("b", ParameterType.BOOLEAN, [True, False]),
        ]
    )
    good = {"t": 1.0, "n": 3, "c": "a", "b": True, "extra": object()}
    assert space.validate_config(good) is True
    for bad in (
        {**good, "t": 1.0000001},
        {**good, "n": 0},
        {**good, "c": "z"},
        {**good, "b": 1},
    ):
        assert space.validate_config(bad) is False
    missing = dict(good)
    del missing["c"]
    assert space.validate_config(missing) is False


def test_empty_space_accepts_any_config() -> None:
    assert _space([]).validate_config({}) is True
    assert _space([]).validate_config({"x": 1}) is True


def test_valid_trace_reads_and_order() -> None:
    trace: Trace = []
    space = _space([_make_probe("a", trace), _make_probe("b", trace)])
    # Construction reads each name once (duplicate check); start tracing after.
    trace.clear()
    config = RecordingConfig(trace, {"a": 1, "b": 2})
    assert space.validate_config(config) is True
    assert trace == [
        ("name", "a", 2),
        ("name", "b", 2),
        ("keys",),
        ("name", "a", 3),
        ("in", "a"),
        ("name", "a", 4),
        ("item", "a"),
        ("validate", "a", 1),
        ("name", "b", 3),
        ("in", "b"),
        ("name", "b", 4),
        ("item", "b"),
        ("validate", "b", 2),
    ]


def test_missing_parameter_short_circuits_before_value_checks() -> None:
    trace: Trace = []
    space = _space(
        [_make_probe("a", trace), _make_probe("b", trace)],
        constraints=[_constraint_probe(trace)],
    )
    trace.clear()
    config = RecordingConfig(trace, {"a": 1})
    assert space.validate_config(config) is False
    assert trace == [("name", "a", 2), ("name", "b", 2), ("keys",)]


def test_invalid_value_stops_before_later_parameters_and_constraints() -> None:
    trace: Trace = []
    space = _space(
        [_make_probe("a", trace, result=False), _make_probe("b", trace)],
        constraints=[_constraint_probe(trace)],
    )
    config = RecordingConfig(trace, {"a": 1, "b": 2})
    assert space.validate_config(config) is False
    assert ("validate", "b", 2) not in trace
    assert ("constraint",) not in trace
    assert trace[-1] == ("validate", "a", 1)


@pytest.mark.parametrize("falsy", [False, 0, "", None, []])
def test_falsy_value_validation_result_is_invalid(falsy: Any) -> None:
    space = _space([_make_probe("a", [], result=falsy)])
    assert space.validate_config({"a": 1}) is False


@pytest.mark.parametrize("truthy", [True, 1, "yes", [0]])
def test_truthy_value_validation_result_is_valid(truthy: Any) -> None:
    space = _space([_make_probe("a", [], result=truthy)])
    assert space.validate_config({"a": 1}) is True


def test_value_validation_error_propagates_and_is_not_logged(
    log: RecordingLogger,
) -> None:
    trace: Trace = []
    space = _space(
        [_make_probe("a", trace, raises=RuntimeError("bad value"))],
        constraints=[lambda c: True],
    )
    with pytest.raises(RuntimeError, match="bad value"):
        space.validate_config({"a": 1})
    assert log.calls == []


@pytest.mark.parametrize("read", [2, 3, 4])
def test_raising_name_accessor_propagates_at_each_read(read: int) -> None:
    trace: Trace = []
    space = _space([_make_probe("a", trace, name_raises_on_read=read)])
    with pytest.raises(RuntimeError, match="name accessor failed"):
        space.validate_config({"a": 1})


def test_raising_keys_accessor_propagates_after_name_set() -> None:
    trace: Trace = []

    class Boom(dict[str, Any]):
        def keys(self) -> dict_keys[str, Any]:
            trace.append(("keys",))
            raise RuntimeError("keys failed")

    space = _space([_make_probe("a", trace)])
    with pytest.raises(RuntimeError, match="keys failed"):
        space.validate_config(Boom(a=1))
    assert trace[-2:] == [("name", "a", 2), ("keys",)]


def test_constraints_none_and_empty_are_skipped_without_logging(
    log: RecordingLogger,
) -> None:
    assert _space([], constraints=None).validate_config({}) is True
    assert _space([], constraints=[]).validate_config({}) is True
    assert log.calls == []


def test_constraints_called_in_order_with_same_config_object() -> None:
    seen: list[tuple[str, Any]] = []
    config: dict[str, Any] = {}

    def first(c: Any) -> bool:
        seen.append(("first", c))
        return True

    def second(c: Any) -> bool:
        seen.append(("second", c))
        return True

    assert _space([], constraints=[first, second]).validate_config(config) is True
    assert [name for name, _ in seen] == ["first", "second"]
    assert all(c is config for _, c in seen)


def test_false_constraint_stops_later_constraints() -> None:
    calls: list[str] = []

    def a(c: Any) -> bool:
        calls.append("a")
        return False

    def b(c: Any) -> bool:
        calls.append("b")
        return True

    assert _space([], constraints=[a, b]).validate_config({}) is False
    assert calls == ["a"]


@pytest.mark.parametrize("falsy", [0, "", None, []])
def test_falsy_constraint_results_deny(falsy: Any) -> None:
    assert _space([], constraints=[lambda c: falsy]).validate_config({}) is False


@pytest.mark.parametrize("truthy", [1, "x", [0]])
def test_truthy_constraint_results_allow(truthy: Any) -> None:
    assert _space([], constraints=[lambda c: truthy]).validate_config({}) is True


def test_constraint_exception_denies_and_logs_exact_message(
    log: RecordingLogger,
) -> None:
    calls: list[str] = []

    def bad(c: Any) -> bool:
        calls.append("bad")
        raise ValueError("boom")

    def after(c: Any) -> bool:
        calls.append("after")
        return True

    assert _space([], constraints=[bad, after]).validate_config({}) is False
    assert calls == ["bad"]
    assert log.calls == [
        ("debug", ("Constraint check raised exception (treating as invalid): boom",))
    ]


def test_constraint_exception_is_stringified_once() -> None:
    class Counted(Exception):
        n = 0

        def __str__(self) -> str:
            type(self).n += 1
            return "counted"

    def bad(c: Any) -> bool:
        raise Counted()

    assert _space([], constraints=[bad]).validate_config({}) is False
    assert Counted.n == 1


def test_constraint_raising_once_then_succeeding_uses_try_per_call(
    log: RecordingLogger,
) -> None:
    state = {"n": 0}

    def flaky(c: Any) -> bool:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("first only")
        return True

    space = _space([], constraints=[flaky])
    assert space.validate_config({}) is False
    assert space.validate_config({}) is True
    assert len(log.calls) == 1


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_base_exceptions_from_constraints_are_not_swallowed(
    exc: type[BaseException], log: RecordingLogger
) -> None:
    def bad(c: Any) -> bool:
        raise exc()

    with pytest.raises(exc):
        _space([], constraints=[bad]).validate_config({})
    assert log.calls == []


def test_raising_truth_of_constraint_result_is_inside_the_guard(
    log: RecordingLogger,
) -> None:
    class Untruthy:
        def __bool__(self) -> bool:
            raise RuntimeError("truth failed")

    assert _space([], constraints=[lambda c: Untruthy()]).validate_config({}) is False
    assert log.calls == [
        (
            "debug",
            ("Constraint check raised exception (treating as invalid): truth failed",),
        )
    ]


def test_truth_of_constraint_result_evaluated_once() -> None:
    class Once:
        n = 0

        def __bool__(self) -> bool:
            type(self).n += 1
            return True

    assert _space([], constraints=[lambda c: Once()]).validate_config({}) is True
    assert Once.n == 1


def _counting_space(
    reads: dict[str, int], constraints: Any, params: list[Parameter]
) -> ConfigurationSpace:
    class Counted(ConfigurationSpace):
        def __getattribute__(self, attr: str) -> Any:
            if attr in reads:
                reads[attr] += 1
            return super().__getattribute__(attr)

    space = Counted(parameters=params, constraints=constraints)
    for key in reads:
        reads[key] = 0
    return space


def test_space_attribute_read_counts() -> None:
    reads = {"parameters": 0, "constraints": 0}
    space = _counting_space(reads, [lambda c: True], [])
    assert space.validate_config({}) is True
    assert reads == {"parameters": 2, "constraints": 2}


def test_space_constraints_read_once_when_falsy() -> None:
    reads = {"parameters": 0, "constraints": 0}
    space = _counting_space(reads, None, [])
    assert space.validate_config({}) is True
    assert reads == {"parameters": 2, "constraints": 1}


def test_truth_of_constraint_collection_evaluated_once() -> None:
    class Constraints(list[Any]):
        n = 0

        def __bool__(self) -> bool:
            type(self).n += 1
            return True

    assert (
        _space([], constraints=Constraints([lambda c: True])).validate_config({})
        is True
    )
    assert Constraints.n == 1


def test_config_is_not_mutated() -> None:
    config = {"a": 1, "z": 2}
    _space([_p("a", ParameterType.CATEGORICAL, [1])]).validate_config(config)
    assert config == {"a": 1, "z": 2}
