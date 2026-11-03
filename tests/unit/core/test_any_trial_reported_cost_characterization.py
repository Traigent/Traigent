"""Characterization tests for ``_any_trial_reported_cost``.

Pin which trial metrics count as "reported spend", the order and number of
attribute and key reads, and how malformed inputs behave.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.core.optimized_function import _any_trial_reported_cost


class _Trial:
    def __init__(self, metrics: Any) -> None:
        self.metrics = metrics


class _Result:
    def __init__(self, trials: Any) -> None:
        self.trials = trials


class _TracingDict(dict[str, Any]):
    def __init__(self, trace: list[str], tag: str, data: dict[str, Any]) -> None:
        super().__init__(data)
        self._trace = trace
        self._tag = tag

    def get(self, key: str, default: Any = None) -> Any:
        self._trace.append(f"{self._tag}.get:{key}")
        return super().get(key, default)


class _Value:
    def __init__(self, trace: list[str], tag: str, number: float) -> None:
        self._trace = trace
        self._tag = tag
        self._number = number

    def __float__(self) -> float:
        self._trace.append(f"float:{self._tag}")
        return self._number


class _RaisingValue:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __float__(self) -> float:
        raise self._exc


def _result(*metrics: Any) -> _Result:
    return _Result([_Trial(m) for m in metrics])


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"cost": 0.5}, True),
        ({"total_cost": 0.25}, True),
        ({"cost": 0.0, "total_cost": 0.25}, True),
        ({"cost": 0.25, "total_cost": 0.0}, True),
        ({"cost": 0.0}, False),
        ({"total_cost": 0.0}, False),
        ({"cost": -1.0, "total_cost": -2.0}, False),
        ({"cost": 1e-12}, True),
        ({"cost": "3"}, True),
        ({"cost": "0"}, False),
        ({"cost": "abc", "total_cost": 2}, True),
        ({"cost": "abc"}, False),
        ({"cost": None, "total_cost": 2}, True),
        ({"cost": None}, False),
        ({"cost": True}, False),
        ({"cost": True, "total_cost": 1}, True),
        ({"total_cost": False}, False),
        ({"cost": float("nan")}, False),
        ({"cost": float("inf")}, True),
        ({"cost": [1]}, False),
        ({"cost": {"a": 1}, "total_cost": 7}, True),
        ({"other_cost": 5.0}, False),
        ({}, False),
    ],
)
def test_single_trial_metric_matrix(metrics: dict[str, Any], expected: bool) -> None:
    assert _any_trial_reported_cost(_result(metrics)) is expected


def test_boundary_zero_is_not_spend_but_smallest_positive_is() -> None:
    assert _any_trial_reported_cost(_result({"cost": 0.0})) is False
    assert _any_trial_reported_cost(_result({"cost": 5e-324})) is True
    assert _any_trial_reported_cost(_result({"cost": -0.0})) is False


@pytest.mark.parametrize(
    "result",
    [
        _Result(None),
        _Result([]),
        _Result(()),
        _Result(0),
        _Result(""),
        object(),
    ],
)
def test_missing_or_empty_trials_report_no_cost(result: Any) -> None:
    assert _any_trial_reported_cost(result) is False


@pytest.mark.parametrize(
    "metrics",
    [None, [], [("cost", 1.0)], "cost", 5, (("cost", 1.0),), object()],
)
def test_non_dict_metrics_are_skipped_and_scan_continues(metrics: Any) -> None:
    assert _any_trial_reported_cost(_result(metrics)) is False
    assert _any_trial_reported_cost(_result(metrics, {"cost": 1.0})) is True


def test_trial_without_metrics_attribute_is_skipped() -> None:
    class _Bare:
        pass

    result = _Result([_Bare(), _Trial({"total_cost": 1})])
    assert _any_trial_reported_cost(result) is True
    assert _any_trial_reported_cost(_Result([_Bare()])) is False


def test_dict_subclass_metrics_are_accepted() -> None:
    trace: list[str] = []
    metrics = _TracingDict(trace, "m", {"cost": 2.0})
    assert _any_trial_reported_cost(_result(metrics)) is True
    assert trace == ["m.get:cost"]


def test_first_positive_trial_short_circuits_remaining_trials() -> None:
    trace: list[str] = []
    first = _TracingDict(trace, "a", {"cost": 0.0, "total_cost": 0.0})
    second = _TracingDict(trace, "b", {"cost": 1.0, "total_cost": 1.0})
    third = _TracingDict(trace, "c", {"cost": 1.0})
    assert _any_trial_reported_cost(_result(first, second, third)) is True
    assert trace == ["a.get:cost", "a.get:total_cost", "b.get:cost"]


def test_no_spend_reads_both_keys_of_every_trial_in_order() -> None:
    trace: list[str] = []
    first = _TracingDict(trace, "a", {})
    second = _TracingDict(trace, "b", {"cost": None})
    assert _any_trial_reported_cost(_result(first, second)) is False
    assert trace == [
        "a.get:cost",
        "a.get:total_cost",
        "b.get:cost",
        "b.get:total_cost",
    ]


def test_value_is_converted_once_per_key_and_only_when_needed() -> None:
    trace: list[str] = []
    metrics = _TracingDict(
        trace,
        "m",
        {
            "cost": _Value(trace, "cost", 0.0),
            "total_cost": _Value(trace, "total", 3.0),
        },
    )
    assert _any_trial_reported_cost(_result(metrics)) is True
    assert trace == ["m.get:cost", "float:cost", "m.get:total_cost", "float:total"]


def test_bool_and_none_values_are_never_converted() -> None:
    trace: list[str] = []
    metrics = _TracingDict(trace, "m", {"cost": True, "total_cost": None})
    assert _any_trial_reported_cost(_result(metrics)) is False
    assert trace == ["m.get:cost", "m.get:total_cost"]


@pytest.mark.parametrize("exc", [TypeError("t"), ValueError("v")])
def test_conversion_type_and_value_errors_are_swallowed_per_key(
    exc: BaseException,
) -> None:
    metrics = {"cost": _RaisingValue(exc), "total_cost": 4.0}
    assert _any_trial_reported_cost(_result(metrics)) is True
    assert _any_trial_reported_cost(_result({"cost": _RaisingValue(exc)})) is False


@pytest.mark.parametrize(
    "exc", [RuntimeError("r"), ZeroDivisionError("z"), KeyboardInterrupt()]
)
def test_other_conversion_errors_propagate(exc: BaseException) -> None:
    with pytest.raises(type(exc)):
        _any_trial_reported_cost(
            _result({"cost": _RaisingValue(exc), "total_cost": 4.0})
        )


def test_raise_once_conversion_is_not_retried() -> None:
    state = {"calls": 0}

    class _Once:
        def __float__(self) -> float:
            state["calls"] += 1
            if state["calls"] == 1:
                raise ValueError("first")
            return 9.0

    assert _any_trial_reported_cost(_result({"cost": _Once()})) is False
    assert state["calls"] == 1


def test_get_raising_propagates() -> None:
    class _BadGet(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get")

    with pytest.raises(RuntimeError, match="get"):
        _any_trial_reported_cost(_result(_BadGet({"cost": 1})))


def test_trials_container_is_truth_tested_and_iterated_once() -> None:
    events: list[str] = []

    class _Trials:
        def __bool__(self) -> bool:
            events.append("bool")
            return True

        def __iter__(self) -> Any:
            events.append("iter")
            return iter([_Trial({"cost": 0.0}), _Trial({"cost": 2.0})])

    assert _any_trial_reported_cost(_Result(_Trials())) is True
    assert events == ["bool", "iter"]


def test_falsy_trials_container_is_not_iterated() -> None:
    events: list[str] = []

    class _Falsy:
        def __bool__(self) -> bool:
            events.append("bool")
            return False

        def __iter__(self) -> Any:
            events.append("iter")
            return iter([_Trial({"cost": 5.0})])

    assert _any_trial_reported_cost(_Result(_Falsy())) is False
    assert events == ["bool"]


def test_trials_and_metrics_attributes_are_read_once_each_in_order() -> None:
    trace: list[str] = []

    class _TrackedTrial:
        @property
        def metrics(self) -> Any:
            trace.append("trial.metrics")
            return {"cost": 0.0}

    class _TrackedResult:
        @property
        def trials(self) -> Any:
            trace.append("result.trials")
            return [_TrackedTrial(), _TrackedTrial()]

    assert _any_trial_reported_cost(_TrackedResult()) is False
    assert trace == ["result.trials", "trial.metrics", "trial.metrics"]


def test_attribute_error_from_accessors_is_treated_as_missing() -> None:
    class _NoTrials:
        @property
        def trials(self) -> Any:
            raise AttributeError("gone")

    class _NoMetrics:
        @property
        def metrics(self) -> Any:
            raise AttributeError("gone")

    assert _any_trial_reported_cost(_NoTrials()) is False
    assert (
        _any_trial_reported_cost(_Result([_NoMetrics(), _Trial({"cost": 1})])) is True
    )


def test_other_accessor_errors_propagate() -> None:
    class _BadTrials:
        @property
        def trials(self) -> Any:
            raise RuntimeError("trials")

    class _BadMetrics:
        @property
        def metrics(self) -> Any:
            raise RuntimeError("metrics")

    with pytest.raises(RuntimeError, match="trials"):
        _any_trial_reported_cost(_BadTrials())
    with pytest.raises(RuntimeError, match="metrics"):
        _any_trial_reported_cost(_Result([_BadMetrics()]))


def test_raise_once_trials_accessor_is_read_once() -> None:
    state = {"reads": 0}

    class _Flaky:
        @property
        def trials(self) -> Any:
            state["reads"] += 1
            if state["reads"] == 1:
                raise RuntimeError("first")
            return [_Trial({"cost": 1})]

    flaky = _Flaky()
    with pytest.raises(RuntimeError, match="first"):
        _any_trial_reported_cost(flaky)
    assert state["reads"] == 1
    assert _any_trial_reported_cost(flaky) is True
    assert state["reads"] == 2


def test_non_iterable_truthy_trials_raise_type_error() -> None:
    with pytest.raises(TypeError):
        _any_trial_reported_cost(_Result(5))


def test_return_values_are_exact_booleans() -> None:
    assert _any_trial_reported_cost(_result({"cost": 1.0})) is True
    assert _any_trial_reported_cost(_result({"cost": 0.0})) is False
