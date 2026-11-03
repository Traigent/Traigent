"""Characterization tests for LocalEvaluator._log_cost_override.

The method only logs, so the tests pin the exact log records, the order and
count of reads on both metric mappings and on ``self.metrics``, the
tolerance boundaries and the failure behaviour for malformed cost values.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import traigent.evaluators.local as local_module
from traigent.evaluators.local import LocalEvaluator

_DEBUG_HEAD = "LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = "


def _evaluator(metrics: Any) -> LocalEvaluator:
    evaluator = object.__new__(LocalEvaluator)
    evaluator.metrics = metrics
    return evaluator


def _run(
    evaluator: LocalEvaluator, aggregated: Any, comprehensive: Any
) -> list[tuple[str, str]]:
    """Run the method and return the (level, message) log records in order."""
    records: list[tuple[str, str]] = []
    with patch.object(local_module, "logger") as log:
        log.debug.side_effect = lambda m, *a, **k: records.append(("debug", m))
        log.info.side_effect = lambda m, *a, **k: records.append(("info", m))
        evaluator._log_cost_override(aggregated, comprehensive)
        assert log.debug.call_count + log.info.call_count == len(records)
        for call in log.method_calls:
            assert call[0] in {"debug", "info"}
            assert call.args and len(call.args) == 1 and call.kwargs == {}
    return records


class _Recording(dict[str, Any]):
    """dict that records the read operations performed on it."""

    def __init__(self, label: str, ops: list[Any], data: dict[str, Any]) -> None:
        super().__init__(data)
        self._label = label
        self._ops = ops

    def get(self, key: str, default: Any = None) -> Any:
        self._ops.append((self._label, "get", key, default))
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self._ops.append((self._label, "getitem", key))
        return super().__getitem__(key)


class _Contains:
    def __init__(self, ops: list[Any], answer: bool) -> None:
        self._ops = ops
        self._answer = answer

    def __contains__(self, item: object) -> bool:
        self._ops.append(("metrics", "contains", item))
        return self._answer


def test_cost_not_an_objective_logs_only_the_first_debug_line() -> None:
    records = _run(_evaluator(["accuracy"]), {"cost": 0.0}, {"cost": 3.0})
    assert records == [("debug", _DEBUG_HEAD + "3.0")]


def test_missing_comprehensive_cost_reports_missing_marker() -> None:
    records = _run(_evaluator(["cost"]), {"cost": 0.0}, {})
    assert records == [("debug", _DEBUG_HEAD + "MISSING")]


def test_none_comprehensive_cost_stops_after_first_debug_line() -> None:
    records = _run(_evaluator(["cost"]), {"cost": 0.0}, {"cost": None})
    assert records == [("debug", _DEBUG_HEAD + "None")]


def test_zero_aggregated_and_positive_comprehensive_logs_override() -> None:
    records = _run(_evaluator(["cost"]), {"cost": 0.0}, {"cost": 2.5})
    assert records == [
        ("debug", _DEBUG_HEAD + "2.5"),
        ("debug", "LOCAL EVALUATOR DEBUG: aggregated cost=0.0, comprehensive cost=2.5"),
        ("info", "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 2.5"),
    ]


def test_missing_aggregated_cost_counts_as_zero() -> None:
    records = _run(_evaluator(["cost"]), {}, {"cost": 2.5})
    assert records == [
        ("debug", _DEBUG_HEAD + "2.5"),
        (
            "debug",
            "LOCAL EVALUATOR DEBUG: aggregated cost=MISSING, comprehensive cost=2.5",
        ),
        ("info", "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 2.5"),
    ]


@pytest.mark.parametrize("falsy", [None, 0, 0.0, False])
def test_falsy_aggregated_cost_is_replaced_by_zero_float(falsy: Any) -> None:
    records = _run(_evaluator(["cost"]), {"cost": falsy}, {"cost": 1.5})
    assert records[-1] == (
        "info",
        "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 1.5",
    )
    assert records[1] == (
        "debug",
        f"LOCAL EVALUATOR DEBUG: aggregated cost={falsy}, comprehensive cost=1.5",
    )


@pytest.mark.parametrize(
    ("aggregated", "comprehensive", "logged"),
    [
        (1e-9, 5.0, True),
        (-1e-9, 5.0, True),
        (1.1e-9, 5.0, False),
        (-1.1e-9, 5.0, False),
        (0.0, 1e-9, False),
        (0.0, -1e-9, False),
        (0.0, 1.1e-9, True),
        (0.0, -1.1e-9, True),
        (0.0, 0.0, False),
        (4.0, 5.0, False),
        (4.0, 0.0, False),
        (float("nan"), 5.0, False),
        (0.0, float("nan"), True),
        (0.0, float("inf"), True),
    ],
)
def test_override_is_logged_exactly_at_the_tolerance_boundaries(
    aggregated: float, comprehensive: float, logged: bool
) -> None:
    records = _run(_evaluator(["cost"]), {"cost": aggregated}, {"cost": comprehensive})
    assert len(records) == (3 if logged else 2)
    info = [r for r in records if r[0] == "info"]
    if logged:
        assert info == [
            (
                "info",
                "🔍 LOCAL EVALUATOR: Overriding cost metric: "
                f"{float(aggregated or 0.0)} -> {float(comprehensive)}",
            )
        ]
    else:
        assert info == []


def test_numeric_strings_and_bool_are_converted_by_float() -> None:
    records = _run(_evaluator(["cost"]), {"cost": "0"}, {"cost": "2"})
    assert records[-1] == (
        "info",
        "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 2.0",
    )
    records = _run(_evaluator(["cost"]), {"cost": 0.0}, {"cost": True})
    assert records[-1] == (
        "info",
        "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 1.0",
    )


def test_non_numeric_aggregated_cost_raises_after_two_debug_lines() -> None:
    records: list[tuple[str, str]] = []
    with patch.object(local_module, "logger") as log:
        log.debug.side_effect = lambda m, *a, **k: records.append(("debug", m))
        with pytest.raises(ValueError, match="could not convert string to float"):
            _evaluator(["cost"])._log_cost_override({"cost": "abc"}, {"cost": 2.0})
        assert log.info.call_count == 0
    assert [r[1] for r in records] == [
        _DEBUG_HEAD + "2.0",
        "LOCAL EVALUATOR DEBUG: aggregated cost=abc, comprehensive cost=2.0",
    ]


def test_non_numeric_comprehensive_cost_raises_after_two_debug_lines() -> None:
    records: list[tuple[str, str]] = []
    with patch.object(local_module, "logger") as log:
        log.debug.side_effect = lambda m, *a, **k: records.append(("debug", m))
        with pytest.raises(ValueError, match="could not convert string to float"):
            _evaluator(["cost"])._log_cost_override({"cost": 0.0}, {"cost": "abc"})
        assert log.info.call_count == 0
    assert len(records) == 2


def test_aggregated_conversion_happens_before_comprehensive_conversion() -> None:
    class _BadAgg:
        def __bool__(self) -> bool:
            return True

        def __float__(self) -> float:
            raise ArithmeticError("aggregated first")

    class _BadComp:
        def __float__(self) -> float:
            raise OverflowError("comprehensive second")

    with patch.object(local_module, "logger"):
        with pytest.raises(ArithmeticError, match="aggregated first"):
            _evaluator(["cost"])._log_cost_override(
                {"cost": _BadAgg()}, {"cost": _BadComp()}
            )


def test_read_operations_on_both_mappings_are_exact_when_cost_is_an_objective() -> None:
    ops: list[Any] = []
    aggregated = _Recording("agg", ops, {"cost": 0.0})
    comprehensive = _Recording("comp", ops, {"cost": 2.5})
    metrics = _Contains(ops, True)
    _run(_evaluator(metrics), aggregated, comprehensive)
    assert ops == [
        ("comp", "get", "cost", "MISSING"),
        ("metrics", "contains", "cost"),
        ("comp", "get", "cost", None),
        ("agg", "get", "cost", "MISSING"),
        ("comp", "getitem", "cost"),
        ("agg", "get", "cost", 0.0),
        ("comp", "getitem", "cost"),
    ]


def test_read_operations_stop_early_when_cost_is_not_an_objective() -> None:
    ops: list[Any] = []
    aggregated = _Recording("agg", ops, {"cost": 0.0})
    comprehensive = _Recording("comp", ops, {"cost": 2.5})
    _run(_evaluator(_Contains(ops, False)), aggregated, comprehensive)
    assert ops == [
        ("comp", "get", "cost", "MISSING"),
        ("metrics", "contains", "cost"),
    ]


def test_comprehensive_none_skips_aggregated_reads() -> None:
    ops: list[Any] = []
    aggregated = _Recording("agg", ops, {"cost": 0.0})
    comprehensive = _Recording("comp", ops, {"cost": None})
    _run(_evaluator(_Contains(ops, True)), aggregated, comprehensive)
    assert ops == [
        ("comp", "get", "cost", "MISSING"),
        ("metrics", "contains", "cost"),
        ("comp", "get", "cost", None),
    ]


def test_raising_comprehensive_get_propagates_before_any_log() -> None:
    class _Raising(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get failed")

    with patch.object(local_module, "logger") as log:
        with pytest.raises(RuntimeError, match="get failed"):
            _evaluator(["cost"])._log_cost_override({"cost": 0.0}, _Raising())
        assert log.method_calls == []


def test_raising_metrics_membership_propagates_after_first_debug_line() -> None:
    class _Bad:
        def __contains__(self, item: object) -> bool:
            raise RuntimeError("contains failed")

    records: list[str] = []
    with patch.object(local_module, "logger") as log:
        log.debug.side_effect = lambda m: records.append(m)
        with pytest.raises(RuntimeError, match="contains failed"):
            _evaluator(_Bad())._log_cost_override({}, {"cost": 1.0})
    assert records == [_DEBUG_HEAD + "1.0"]


def test_raise_once_comprehensive_getitem_fails_in_second_debug_line() -> None:
    class _RaiseOnce(dict[str, Any]):
        calls = 0

        def __getitem__(self, key: str) -> Any:
            type(self).calls += 1
            if type(self).calls == 1:
                raise KeyError("first read")
            return super().__getitem__(key)

    records: list[str] = []
    with patch.object(local_module, "logger") as log:
        log.debug.side_effect = lambda m: records.append(m)
        with pytest.raises(KeyError, match="first read"):
            _evaluator(["cost"])._log_cost_override(
                {"cost": 0.0}, _RaiseOnce({"cost": 2.0})
            )
    assert records == [_DEBUG_HEAD + "2.0"]


def test_aggregated_truth_is_evaluated_once_and_float_once() -> None:
    counts = {"bool": 0, "float": 0}

    class _Value:
        def __bool__(self) -> bool:
            counts["bool"] += 1
            return True

        def __float__(self) -> float:
            counts["float"] += 1
            return 0.0

    _run(_evaluator(["cost"]), {"cost": _Value()}, {"cost": 2.0})
    assert counts == {"bool": 1, "float": 1}


def test_comprehensive_float_conversion_runs_once() -> None:
    calls: list[int] = []

    class _Comp:
        def __float__(self) -> float:
            calls.append(1)
            return 2.0

        def __str__(self) -> str:
            return "comp"

    _run(_evaluator(["cost"]), {"cost": 0.0}, {"cost": _Comp()})
    assert calls == [1]


def test_metrics_membership_uses_in_operator_semantics() -> None:
    records = _run(_evaluator({"cost": 1}), {"cost": 0.0}, {"cost": 2.0})
    assert len(records) == 3
    records = _run(_evaluator("cost"), {"cost": 0.0}, {"cost": 2.0})
    assert len(records) == 3
    records = _run(_evaluator(()), {"cost": 0.0}, {"cost": 2.0})
    assert len(records) == 1
