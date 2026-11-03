"""Characterization tests for ``OptimizationOrchestrator._extract_objective_values``.

Pin how each objective's reported value is resolved from trial metrics (aliases,
the cost fallback, defaults), the exact metric reads and their order, truth
evaluation counts, and how malformed values and metrics behave.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.core.orchestrator import OptimizationOrchestrator


class _TestOrchestrator(OptimizationOrchestrator):
    optimizer: Any


def _orch(objectives: Any) -> _TestOrchestrator:
    orch = _TestOrchestrator.__new__(_TestOrchestrator)
    orch.optimizer = SimpleNamespace(objectives=objectives)
    return orch


class _Metrics(dict[str, Any]):
    def __init__(self, trace: list[Any], data: dict[str, Any]) -> None:
        super().__init__(data)
        self._trace = trace

    def get(self, key: Any, *default: Any) -> Any:
        self._trace.append((key, *default))
        return super().get(key, *default)


def _values(objectives: list[Any], metrics: Any) -> list[float]:
    result: list[float] = _orch(objectives)._extract_objective_values(metrics)
    return result


@pytest.mark.parametrize(
    ("objective", "metrics", "expected"),
    [
        ("score", {"score": 0.7}, 0.7),
        ("score", {}, 0.0),
        ("score", {"score": None}, 0.0),
        ("score", {"score": 0}, 0.0),
        ("score", {"score": "0.5"}, 0.5),
        ("score", {"score": True}, 1.0),
        ("score", {"score": 3}, 3.0),
        ("score", {"accuracy": 0.9}, 0.0),
        ("accuracy", {"accuracy": 0.7}, 0.7),
        ("accuracy", {"accuracy": 0.0, "success_rate": 0.5}, 0.0),
        ("accuracy", {"success_rate": 0.5}, 0.5),
        ("accuracy", {"accuracy": None, "success_rate": 0.5}, 0.5),
        ("accuracy", {}, 0.0),
        ("accuracy", {"success_rate": 0}, 0.0),
        ("success_rate", {"success_rate": 0.4}, 0.4),
        ("success_rate", {"success_rate": 0, "accuracy": 0.8}, 0.0),
        ("success_rate", {"accuracy": 0.8}, 0.8),
        ("success_rate", {"success_rate": None, "accuracy": 0.8}, 0.8),
        ("success_rate", {"success_rate": None, "accuracy": 0.0}, 0.0),
        ("success_rate", {"success_rate": None, "accuracy": 0, "x": 1}, 0.0),
        ("cost", {"cost": 0.25, "total_cost": 9.0}, 0.25),
        ("cost", {"cost": 0.0, "total_cost": 9.0}, 9.0),
        ("cost", {"cost": 0, "total_cost": 2}, 2.0),
        ("cost", {"cost": "", "total_cost": 2}, 2.0),
        ("cost", {"cost": 0.0, "total_cost": 0.0}, 0.0),
        ("cost", {"cost": 0.0}, 0.0),
        ("cost", {"cost": None, "total_cost": 4.0}, 4.0),
        ("cost", {"total_cost": 4.0}, 4.0),
        ("cost", {}, 0.0),
        ("cost", {"cost": None}, 0.0),
        ("cost", {"cost": [], "total_cost": 1.5}, 1.5),
        ("latency", {"latency": 1.5}, 1.5),
        ("latency", {}, 0.0),
        ("latency", {"latency": None}, 0.0),
        ("error", {"error": 0.1}, 0.1),
        ("error", {}, 0.0),
        ("total_cost", {"total_cost": 3.0}, 3.0),
        ("total_cost", {}, 0.0),
    ],
)
def test_value_matrix(objective: str, metrics: dict[str, Any], expected: float) -> None:
    result = _values([objective], metrics)
    assert result == [expected]
    assert type(result[0]) is float


def test_nan_and_inf_pass_through() -> None:
    result = _values(["a", "b"], {"a": float("nan"), "b": float("inf")})
    assert math.isnan(result[0])
    assert result[1] == math.inf


def test_multiple_objectives_keep_declared_order_and_duplicates() -> None:
    metrics = {"a": 1, "b": 2}
    assert _values(["b", "a", "b"], metrics) == [2.0, 1.0, 2.0]


def test_no_objectives_returns_a_fresh_empty_list_without_reading_metrics() -> None:
    first = _values([], None)
    second = _values([], None)
    assert first == [] and second == []
    assert first is not second


def test_non_mapping_metrics_raise_when_an_objective_exists() -> None:
    with pytest.raises(AttributeError):
        _values(["a"], None)
    with pytest.raises(AttributeError):
        _values(["a"], [1])


def test_non_string_objective_reads_nothing_but_defaults() -> None:
    assert _values([5], {"5": 1}) == [0.0]
    assert _values([5], {5: 2}) == [2.0]


def test_unhashable_objective_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _values([["a"]], {"a": 1})


@pytest.mark.parametrize(
    ("objective", "data", "expected_reads"),
    [
        ("score", {"score": 1}, [("score",)]),
        ("score", {}, [("score",)]),
        ("accuracy", {"accuracy": 0.5}, [("accuracy",)]),
        ("accuracy", {}, [("accuracy",), ("accuracy",), ("success_rate",)]),
        (
            "accuracy",
            {"success_rate": 0.5},
            [("accuracy",), ("accuracy",), ("success_rate",)],
        ),
        (
            "accuracy",
            {"accuracy": None, "success_rate": 0.5},
            [("accuracy",), ("accuracy",), ("success_rate",)],
        ),
        ("success_rate", {"success_rate": 0.5}, [("success_rate",)]),
        (
            "success_rate",
            {},
            [("success_rate",), ("accuracy",), ("success_rate",)],
        ),
        ("cost", {"cost": 1.0}, [("cost",)]),
        ("cost", {"cost": 0.0, "total_cost": 1.0}, [("cost",), ("total_cost",)]),
        (
            "cost",
            {"cost": 0.0, "total_cost": 0.0},
            [("cost",), ("total_cost",)],
        ),
        ("cost", {"cost": None}, [("cost",), ("total_cost",), ("cost", 0.0)]),
        ("cost", {}, [("cost",), ("total_cost",), ("cost", 0.0)]),
        ("cost", {"total_cost": 2.0}, [("cost",), ("total_cost",)]),
        ("latency", {}, [("latency",), ("latency", 0.0)]),
        ("latency", {"latency": 2.0}, [("latency",)]),
        ("error", {}, [("error",), ("error", 0.0)]),
    ],
)
def test_exact_metric_reads_and_order(
    objective: str, data: dict[str, Any], expected_reads: list[tuple[Any, ...]]
) -> None:
    trace: list[Any] = []
    _values([objective], _Metrics(trace, data))
    assert trace == expected_reads


def test_objectives_are_processed_one_at_a_time_in_order() -> None:
    trace: list[Any] = []

    def declared() -> Iterator[str]:
        for name in ("latency", "score"):
            trace.append(("yield", name))
            yield name

    metrics = _Metrics(trace, {"latency": 1.0, "score": 2.0})
    result = _orch(declared())._extract_objective_values(metrics)
    assert result == [1.0, 2.0]
    assert trace == [
        ("yield", "latency"),
        ("latency",),
        ("yield", "score"),
        ("score",),
    ]


def test_optimizer_and_objectives_are_each_read_once() -> None:
    reads: list[str] = []

    class _Optimizer:
        @property
        def objectives(self) -> list[str]:
            reads.append("objectives")
            return ["a", "b"]

    class _Orch(_TestOrchestrator):
        def __getattribute__(self, name: str) -> Any:
            if name == "optimizer":
                reads.append("optimizer")
            return super().__getattribute__(name)

    orch = _Orch.__new__(_Orch)
    orch.optimizer = _Optimizer()
    reads.clear()
    assert orch._extract_objective_values({"a": 1, "b": 2}) == [1.0, 2.0]
    assert reads == ["optimizer", "objectives"]


def test_truth_of_cost_and_total_cost_is_evaluated_once_each() -> None:
    counts = {"cost": 0, "total": 0}

    class _Falsy:
        def __bool__(self) -> bool:
            counts["cost"] += 1
            return False

        def __float__(self) -> float:
            return 7.0

    class _Truthy:
        def __bool__(self) -> bool:
            counts["total"] += 1
            return True

        def __float__(self) -> float:
            return 5.0

    assert _values(["cost"], {"cost": _Falsy(), "total_cost": _Truthy()}) == [5.0]
    assert counts == {"cost": 1, "total": 1}


def test_truthy_cost_value_skips_total_cost() -> None:
    trace: list[Any] = []

    class _Truthy:
        def __bool__(self) -> bool:
            return True

        def __float__(self) -> float:
            return 3.0

    assert _values(["cost"], _Metrics(trace, {"cost": _Truthy()})) == [3.0]
    assert trace == [("cost",)]


def test_accuracy_alias_truth_is_evaluated_once() -> None:
    counts = {"n": 0}

    class _Falsy:
        def __bool__(self) -> bool:
            counts["n"] += 1
            return False

        def __float__(self) -> float:
            return 1.0

    # success_rate missing: the alias expression falls through the falsy accuracy value
    metrics = {"accuracy": _Falsy(), "success_rate": None}
    assert _values(["success_rate"], metrics) == [0.0]
    assert counts["n"] == 1


def test_stateful_truth_object_is_not_retested() -> None:
    state = {"calls": 0}

    class _Flip:
        def __bool__(self) -> bool:
            state["calls"] += 1
            return state["calls"] > 1

        def __float__(self) -> float:
            return 8.0

    assert _values(["cost"], {"cost": _Flip(), "total_cost": 4.0}) == [4.0]
    assert state["calls"] == 1


def test_not_none_check_is_identity_based() -> None:
    class _EqNone:
        def __eq__(self, other: object) -> bool:
            return other is None

        def __hash__(self) -> int:
            return 0

        def __bool__(self) -> bool:
            return False

        def __float__(self) -> float:
            return 6.0

    assert _values(["latency"], {"latency": _EqNone()}) == [6.0]
    assert _values(["accuracy"], {"accuracy": _EqNone(), "success_rate": 9.0}) == [6.0]


def test_objective_comparisons_use_equality_with_the_objective_string() -> None:
    class _Name(str):
        pass

    assert _values([_Name("cost")], {"cost": 0.0, "total_cost": 2.0}) == [2.0]
    assert _values([_Name("accuracy")], {"success_rate": 0.25}) == [0.25]


@pytest.mark.parametrize("bad", ["abc", [1], {"a": 1}, object()])
def test_unconvertible_values_raise(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        _values(["a"], {"a": bad})


def test_conversion_error_stops_processing_later_objectives() -> None:
    trace: list[Any] = []
    metrics = _Metrics(trace, {"a": "bad", "b": 1})
    with pytest.raises(ValueError):
        _values(["a", "b"], metrics)
    assert trace == [("a",)]


def test_raising_float_conversion_propagates_once() -> None:
    state = {"calls": 0}

    class _Once:
        def __float__(self) -> float:
            state["calls"] += 1
            raise RuntimeError("float")

    with pytest.raises(RuntimeError, match="float"):
        _values(["a"], {"a": _Once()})
    assert state["calls"] == 1


def test_raising_get_propagates_without_partial_result() -> None:
    class _Bad(dict[str, Any]):
        def get(self, key: Any, *default: Any) -> Any:
            raise RuntimeError("get")

    with pytest.raises(RuntimeError, match="get"):
        _values(["a"], _Bad())


def test_raise_once_objectives_accessor_is_not_retried() -> None:
    state = {"reads": 0}

    class _Optimizer:
        @property
        def objectives(self) -> list[str]:
            state["reads"] += 1
            raise RuntimeError("objectives")

    orch = _TestOrchestrator.__new__(_TestOrchestrator)
    orch.optimizer = _Optimizer()
    with pytest.raises(RuntimeError, match="objectives"):
        orch._extract_objective_values({})
    assert state["reads"] == 1


def test_base_exception_from_get_propagates() -> None:
    class _Interrupt(dict[str, Any]):
        def get(self, key: Any, *default: Any) -> Any:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _values(["a"], _Interrupt())


def test_metrics_mapping_is_not_mutated() -> None:
    metrics = {"cost": 0.0, "total_cost": 3.0, "accuracy": 0.1}
    snapshot = dict(metrics)
    _values(["cost", "accuracy", "latency"], metrics)
    assert metrics == snapshot
