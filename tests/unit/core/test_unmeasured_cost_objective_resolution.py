"""Characterization of CustomEvaluatorWrapper._resolve_unmeasured_cost_objectives.

A requested cost metric that no row measured must stay unknown (left out, or
None under strict nulls) instead of being reported as a free 0.0.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from traigent.core import evaluator_wrapper as ew
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper


def _wrapper(metrics: list[str]) -> CustomEvaluatorWrapper:
    return CustomEvaluatorWrapper(
        lambda *a, **k: None, metrics=metrics, capture_llm_metrics=False
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_STRICT_METRICS_NULLS", raising=False)


def _set_strict(monkeypatch: pytest.MonkeyPatch, strict: bool) -> None:
    if strict:
        monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", "true")
    else:
        monkeypatch.delenv("TRAIGENT_STRICT_METRICS_NULLS", raising=False)


def _strict_probe(events: list[str], value: bool) -> Callable[[], bool]:
    def probe() -> bool:
        events.append("strict")
        return value

    return probe


STRICT = pytest.mark.parametrize("strict", [False, True])


class _TraceDict(dict[str, Any]):
    def __init__(self, events: list[str], data: dict[str, Any]) -> None:
        super().__init__(data)
        self._events = events

    def get(self, key: str, default: Any = None) -> Any:
        self._events.append(f"agg.get:{key}")
        return super().get(key, default)


class _TraceRow(dict[str, Any]):
    def __init__(self, events: list[str], name: str, data: dict[str, Any]) -> None:
        super().__init__(data)
        self._events = events
        self._name = name

    def get(self, key: str, default: Any = None) -> Any:
        self._events.append(f"{self._name}.get:{key}")
        return super().get(key, default)


@STRICT
def test_unmeasured_flag_removes_or_nulls_metric_even_when_rows_measured(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 0.5, "total_cost": 0.5, "cost_unmeasured": 1.0}
    _wrapper(["cost", "accuracy"])._resolve_unmeasured_cost_objectives(
        agg, [{"cost": 0.5}]
    )
    expected: dict[str, Any] = {"total_cost": 0.5, "cost_unmeasured": 1.0}
    if strict:
        expected["cost"] = None
    assert agg == expected


@pytest.mark.parametrize("flag", [True, 1, 2.5])
@STRICT
def test_truthy_flags_all_count_as_unmeasured(
    monkeypatch: pytest.MonkeyPatch, strict: bool, flag: Any
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 0.5, "cost_unmeasured": flag}
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [{"cost": 1.0}])
    if strict:
        assert agg == {"cost": None, "cost_unmeasured": flag}
    else:
        assert agg == {"cost_unmeasured": flag}


@pytest.mark.parametrize(
    "flag", [False, 0, 0.0, -1.0, math.nan, math.inf, "1", "true", None]
)
@STRICT
def test_non_flag_values_do_not_count_as_unmeasured(
    monkeypatch: pytest.MonkeyPatch, strict: bool, flag: Any
) -> None:
    # A measured row keeps the aggregate untouched.
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 0.5, "cost_unmeasured": flag}
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [{"cost": 1.0}])
    assert list(agg) == ["cost", "cost_unmeasured"]
    assert agg["cost"] == 0.5


@STRICT
def test_flag_unmeasured_applies_to_absent_metric(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost_unmeasured": 1.0}
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [])
    if strict:
        assert agg == {"cost_unmeasured": 1.0, "cost": None}
    else:
        assert agg == {"cost_unmeasured": 1.0}


@STRICT
def test_measured_row_keeps_value(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 0.0, "total_cost": 9.0}
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(
        agg, [{"cost": 0.0}, {"cost": 3}]
    )
    assert agg == {"cost": 0.0, "total_cost": 9.0}


@pytest.mark.parametrize(
    "row_value", [None, True, False, "1.0", math.nan, math.inf, -math.inf, [1.0]]
)
@STRICT
def test_unmeasured_row_values_do_not_count(
    monkeypatch: pytest.MonkeyPatch, strict: bool, row_value: Any
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 7.0}
    rows: list[Any] = [{"cost": row_value}, {}, None]
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, rows)
    assert agg == ({"cost": None} if strict else {})


@STRICT
def test_falsey_rows_are_skipped_without_get(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 1.0}
    rows: list[Any] = [None, {}]
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, rows)
    assert agg == ({"cost": None} if strict else {})


@pytest.mark.parametrize("metric", ["input_cost", "output_cost", "total_cost"])
@STRICT
def test_captured_value_for_non_cost_metric_uses_own_key(
    monkeypatch: pytest.MonkeyPatch, strict: bool, metric: str
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {metric: 4, "cost": 99.0}
    _wrapper([metric])._resolve_unmeasured_cost_objectives(agg, [{"other": 1}])
    assert agg == {metric: 4.0, "cost": 99.0}
    assert type(agg[metric]) is float


@STRICT
def test_cost_metric_uses_total_cost_source(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 123.0, "total_cost": 2}
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [{}])
    assert agg == {"cost": 2.0, "total_cost": 2}
    assert type(agg["cost"]) is float


@pytest.mark.parametrize("captured", [None, True, "2.0", math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("have_key", [True, False])
@STRICT
def test_unusable_captured_value_leaves_metric_unknown(
    monkeypatch: pytest.MonkeyPatch, strict: bool, captured: Any, have_key: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 5.0}
    if have_key:
        agg["total_cost"] = captured
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [{"x": 1}])
    expected: dict[str, Any] = {"cost": None} if strict else {}
    if have_key:
        expected["total_cost"] = captured
    assert set(agg) == set(expected)
    assert agg.get("cost", "ABSENT") == expected.get("cost", "ABSENT")


@STRICT
def test_only_requested_measured_only_metrics_are_touched(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {
        "cost": 1.0,
        "input_cost": 1.0,
        "output_cost": 1.0,
        "total_cost": 1.0,
        "accuracy": 0.9,
    }
    _wrapper(["accuracy", "input_cost", "unknown"])._resolve_unmeasured_cost_objectives(
        agg, [{}]
    )
    # input_cost has no captured source other than itself -> 1.0 kept (as float)
    assert agg == {
        "cost": 1.0,
        "input_cost": 1.0,
        "output_cost": 1.0,
        "total_cost": 1.0,
        "accuracy": 0.9,
    }


@STRICT
def test_no_requested_cost_metric_is_noop(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"accuracy": 0.9, "cost_unmeasured": 1.0}
    _wrapper(["accuracy"])._resolve_unmeasured_cost_objectives(agg, [{}])
    assert agg == {"accuracy": 0.9, "cost_unmeasured": 1.0}


@STRICT
def test_multiple_metrics_resolved_independently(
    monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    _set_strict(monkeypatch, strict)
    agg: dict[str, Any] = {"cost": 1.0, "total_cost": 3.0, "input_cost": 2.0}
    rows: list[dict[str, Any]] = [{"input_cost": 0.5}]
    _wrapper(["cost", "total_cost", "input_cost"])._resolve_unmeasured_cost_objectives(
        agg, rows
    )
    # cost -> captured total_cost 3.0; total_cost -> own value 3.0; input measured
    assert agg == {"cost": 3.0, "total_cost": 3.0, "input_cost": 2.0}


def test_event_trace_flag_then_rows_then_source_then_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "_strict_metrics_nulls", _strict_probe(events, False))
    agg = _TraceDict(events, {"cost": 1.0})
    rows: list[Any] = [
        _TraceRow(events, "r0", {"other": 1}),
        _TraceRow(events, "r1", {"other": 2}),
    ]
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, rows)
    assert events == [
        "agg.get:cost_unmeasured",
        "r0.get:cost",
        "r1.get:cost",
        "agg.get:total_cost",
        "strict",
    ]
    assert dict(agg) == {}


def test_event_trace_flagged_trial_never_reads_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "_strict_metrics_nulls", _strict_probe(events, True))
    agg = _TraceDict(events, {"cost": 1.0, "cost_unmeasured": 1.0})
    rows: list[Any] = [_TraceRow(events, "r0", {"cost": 1.0})]
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, rows)
    assert events == ["agg.get:cost_unmeasured", "strict"]
    assert dict(agg) == {"cost": None, "cost_unmeasured": 1.0}


def test_event_trace_measured_row_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "_strict_metrics_nulls", _strict_probe(events, False))
    agg = _TraceDict(events, {"cost": 1.0})
    rows: list[Any] = [
        _TraceRow(events, "r0", {"cost": 2.0}),
        _TraceRow(events, "r1", {"cost": 3.0}),
    ]
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, rows)
    assert events == ["agg.get:cost_unmeasured", "r0.get:cost"]
    assert dict(agg) == {"cost": 1.0}


def test_event_trace_captured_value_never_consults_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "_strict_metrics_nulls", _strict_probe(events, False))
    agg = _TraceDict(events, {"total_cost": 4.0})
    _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [])
    assert events == ["agg.get:cost_unmeasured", "agg.get:total_cost"]
    assert dict(agg) == {"total_cost": 4.0, "cost": 4.0}


class _Boom(Exception):
    pass


def test_raising_flag_accessor_propagates_before_any_mutation() -> None:
    class _Agg(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise _Boom(key)

    agg = _Agg({"cost": 1.0})
    with pytest.raises(_Boom, match="cost_unmeasured"):
        _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [{"cost": 1}])
    assert dict(agg) == {"cost": 1.0}


def test_raising_row_accessor_propagates_unchanged() -> None:
    class _Row(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise _Boom(key)

    agg: dict[str, Any] = {"cost": 1.0}
    with pytest.raises(_Boom, match="cost"):
        _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [_Row(a=1)])
    assert agg == {"cost": 1.0}


def test_raising_source_accessor_propagates() -> None:
    class _Agg(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "total_cost":
                raise _Boom(key)
            return super().get(key, default)

    agg = _Agg({"cost": 1.0})
    with pytest.raises(_Boom, match="total_cost"):
        _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [])
    assert dict(agg) == {"cost": 1.0}


def test_env_value_variants_for_strict_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    for value, strict in [
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("0", False),
        ("", False),
    ]:
        monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", value)
        agg: dict[str, Any] = {"cost": 1.0}
        _wrapper(["cost"])._resolve_unmeasured_cost_objectives(agg, [{}])
        assert agg == ({"cost": None} if strict else {}), value
