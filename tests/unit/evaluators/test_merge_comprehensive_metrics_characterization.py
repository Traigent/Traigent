"""Characterization of LocalEvaluator._merge_comprehensive_metrics."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.evaluators import local as local_module
from traigent.evaluators.local import LocalEvaluator


class _Recorder:
    """Stand-in for the module logger capturing (level, rendered message)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def debug(self, msg: str, *args: Any) -> None:
        self.records.append(("debug", msg % args if args else msg))

    def info(self, msg: str, *args: Any) -> None:
        self.records.append(("info", msg % args if args else msg))


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(local_module, "logger", rec)
    return rec


def _evaluator(metrics: list[str]) -> LocalEvaluator:
    return LocalEvaluator(metrics=metrics)


def _merge(
    metrics: list[str],
    aggregated: Any,
    comprehensive: Any,
    preserve: bool,
) -> None:
    _evaluator(metrics)._merge_comprehensive_metrics(
        aggregated, comprehensive, preserve_authoritative_accuracy=preserve
    )


PRESERVE = pytest.mark.parametrize("preserve", [False, True])


@PRESERVE
def test_plain_keys_are_copied_and_none_values_skipped(preserve: bool) -> None:
    agg: dict[str, Any] = {"existing": 1.0}
    comp: dict[str, Any] = {"a": 1.0, "b": None, "existing": 5.0, "c": 0.0}
    _merge(["x"], agg, comp, preserve)
    assert agg == {"existing": 5.0, "a": 1.0, "c": 0.0}
    assert list(agg) == ["existing", "a", "c"]


@PRESERVE
def test_none_accuracy_value_is_skipped_even_when_absent(preserve: bool) -> None:
    agg: dict[str, Any] = {}
    _merge(["accuracy"], agg, {"accuracy": None, "score": None}, preserve)
    assert agg == {}


def test_preserve_aliases_missing_score_to_existing_accuracy() -> None:
    agg: dict[str, Any] = {"accuracy": 0.0}
    _merge(["accuracy"], agg, {"score": 0.75, "other": 1.0}, True)
    assert agg == {"accuracy": 0.0, "score": 0.0, "other": 1.0}


def test_no_preserve_takes_tracker_score_when_absent() -> None:
    agg: dict[str, Any] = {"accuracy": 0.0}
    _merge(["accuracy"], agg, {"score": 0.75}, False)
    assert agg == {"accuracy": 0.0, "score": 0.75}


def test_preserve_without_accuracy_takes_tracker_score() -> None:
    agg: dict[str, Any] = {}
    _merge(["accuracy"], agg, {"score": 0.75}, True)
    assert agg == {"score": 0.75}


def test_preserve_alias_not_applied_when_score_present() -> None:
    agg: dict[str, Any] = {"accuracy": 0.4, "score": 0.9}
    _merge(["accuracy"], agg, {"score": 0.1}, True)
    # present score falls to the authoritative keep rule
    assert agg == {"accuracy": 0.4, "score": 0.9}


@pytest.mark.parametrize("key", ["accuracy", "score"])
def test_preserve_keeps_existing_even_none_or_zero(key: str) -> None:
    for existing in (None, 0.0, 0.5):
        agg: dict[str, Any] = {key: existing}
        if key == "score":
            agg["accuracy"] = 0.3
        _merge(["accuracy"], agg, {key: 0.9}, True)
        assert agg[key] == existing


@pytest.mark.parametrize("key", ["accuracy", "score"])
def test_no_preserve_replaces_none_and_zero_but_keeps_nonzero(key: str) -> None:
    for existing, expected in ((None, 0.9), (0.0, 0.9), (0.5, 0.5), (1, 1)):
        agg: dict[str, Any] = {key: existing}
        _merge(["accuracy"], agg, {key: 0.9}, False)
        assert agg[key] == expected, existing


def test_no_preserve_zero_int_and_false_count_as_zero() -> None:
    for existing in (0, False):
        agg: dict[str, Any] = {"accuracy": existing}
        _merge(["accuracy"], agg, {"accuracy": 0.9}, False)
        assert agg["accuracy"] == 0.9


def test_no_preserve_nan_existing_is_kept() -> None:
    nan = float("nan")
    agg: dict[str, Any] = {"accuracy": nan}
    _merge(["accuracy"], agg, {"accuracy": 0.9}, False)
    assert agg["accuracy"] is nan


@PRESERVE
def test_other_keys_always_overwrite_even_accuracy_like_names(preserve: bool) -> None:
    agg: dict[str, Any] = {"accuracy_x": 0.5, "f1": 0.0}
    _merge(["accuracy"], agg, {"accuracy_x": 0.1, "f1": 0.7}, preserve)
    assert agg == {"accuracy_x": 0.1, "f1": 0.7}


@PRESERVE
def test_missing_cost_logs_only_the_missing_debug_line(
    recorder: _Recorder, preserve: bool
) -> None:
    _merge(["cost"], {}, {"a": 1.0}, preserve)
    assert recorder.records == [
        ("debug", "LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = MISSING")
    ]


def test_cost_not_in_objectives_logs_only_first_line(recorder: _Recorder) -> None:
    agg: dict[str, Any] = {}
    _merge(["accuracy"], agg, {"cost": 2.5}, False)
    assert recorder.records == [
        ("debug", "LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = 2.5")
    ]
    assert agg == {"cost": 2.5}


def test_none_cost_logs_first_line_and_skips_second(recorder: _Recorder) -> None:
    agg: dict[str, Any] = {}
    _merge(["cost"], agg, {"cost": None}, False)
    assert recorder.records == [
        ("debug", "LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = None")
    ]
    assert agg == {}


def test_zero_aggregated_cost_override_logs_info(recorder: _Recorder) -> None:
    agg: dict[str, Any] = {"cost": 0.0}
    _merge(["cost"], agg, {"cost": 1.5}, False)
    assert recorder.records == [
        ("debug", "LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = 1.5"),
        (
            "debug",
            "LOCAL EVALUATOR DEBUG: aggregated cost=0.0, comprehensive cost=1.5",
        ),
        ("info", "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 1.5"),
    ]
    assert agg == {"cost": 1.5}


def test_absent_aggregated_cost_counts_as_zero(recorder: _Recorder) -> None:
    agg: dict[str, Any] = {}
    _merge(["cost"], agg, {"cost": 2}, False)
    assert recorder.records == [
        ("debug", "LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = 2"),
        (
            "debug",
            "LOCAL EVALUATOR DEBUG: aggregated cost=MISSING, comprehensive cost=2",
        ),
        ("info", "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 2.0"),
    ]


def test_none_aggregated_cost_counts_as_zero(recorder: _Recorder) -> None:
    _merge(["cost"], {"cost": None}, {"cost": 2.0}, False)
    assert recorder.records[-1] == (
        "info",
        "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 2.0",
    )


def test_tiny_aggregated_cost_counts_as_zero_boundary(recorder: _Recorder) -> None:
    _merge(["cost"], {"cost": 5e-10}, {"cost": 1.0}, False)
    assert [lvl for lvl, _ in recorder.records] == ["debug", "debug", "info"]
    recorder.records.clear()
    _merge(["cost"], {"cost": 2e-9}, {"cost": 1.0}, False)
    assert [lvl for lvl, _ in recorder.records] == ["debug", "debug"]


def test_tiny_comprehensive_cost_does_not_log_override(recorder: _Recorder) -> None:
    _merge(["cost"], {"cost": 0.0}, {"cost": 5e-10}, False)
    assert [lvl for lvl, _ in recorder.records] == ["debug", "debug"]


def test_nonzero_aggregated_cost_does_not_log_override(recorder: _Recorder) -> None:
    agg: dict[str, Any] = {"cost": 3.0}
    _merge(["cost"], agg, {"cost": 1.0}, False)
    assert [lvl for lvl, _ in recorder.records] == ["debug", "debug"]
    assert agg == {"cost": 1.0}


def test_string_cost_value_converts_in_override_check(recorder: _Recorder) -> None:
    _merge(["cost"], {"cost": "0"}, {"cost": "3"}, False)
    assert recorder.records[-1] == (
        "info",
        "🔍 LOCAL EVALUATOR: Overriding cost metric: 0.0 -> 3.0",
    )


def test_unconvertible_cost_raises_before_merging() -> None:
    agg: dict[str, Any] = {"x": 1}
    with pytest.raises(ValueError):
        _merge(["cost"], agg, {"cost": "abc", "y": 2}, False)
    assert agg == {"x": 1}
    with pytest.raises(TypeError):
        _merge(["cost"], agg, {"cost": object(), "y": 2}, False)
    assert agg == {"x": 1}


def test_unconvertible_aggregated_cost_raises() -> None:
    agg: dict[str, Any] = {"cost": "abc"}
    with pytest.raises(ValueError):
        _merge(["cost"], agg, {"cost": 1.0}, False)
    assert agg == {"cost": "abc"}


def test_override_check_is_skipped_when_cost_not_requested() -> None:
    # an unconvertible cost is harmless when cost is not an objective
    agg: dict[str, Any] = {}
    _merge(["accuracy"], agg, {"cost": "abc"}, False)
    assert agg == {"cost": "abc"}


class _TraceComp(dict[str, Any]):
    def __init__(self, events: list[str], data: dict[str, Any]) -> None:
        super().__init__(data)
        self._events = events

    def get(self, key: str, default: Any = None) -> Any:
        self._events.append(f"comp.get:{key}")
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self._events.append(f"comp[{key}]")
        return super().__getitem__(key)

    def items(self) -> Any:
        self._events.append("comp.items")
        return super().items()


class _TraceAgg(dict[str, Any]):
    def __init__(self, events: list[str], data: dict[str, Any]) -> None:
        super().__init__(data)
        self._events = events

    def get(self, key: str, default: Any = None) -> Any:
        self._events.append(f"agg.get:{key}")
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self._events.append(f"agg[{key}]")
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._events.append(f"agg.in:{key}")
        return super().__contains__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._events.append(f"agg.set:{key}")
        super().__setitem__(key, value)


def test_access_trace_cost_override_then_items(recorder: _Recorder) -> None:
    events: list[str] = []
    agg = _TraceAgg(events, {"cost": 0.0})
    comp = _TraceComp(events, {"cost": 1.0})
    _merge(["cost"], agg, comp, False)
    assert events == [
        "comp.get:cost",
        "comp.get:cost",
        "agg.get:cost",
        "comp[cost]",
        "agg.get:cost",
        "comp[cost]",
        "comp.items",
        "agg.set:cost",
    ]


def test_access_trace_authoritative_keep_order() -> None:
    events: list[str] = []
    agg = _TraceAgg(events, {"accuracy": 0.5, "score": 0.5})
    comp = _TraceComp(events, {"accuracy": 0.9})
    _merge(["accuracy"], agg, comp, False)
    assert events == [
        "comp.get:cost",
        "comp.items",
        "agg.in:accuracy",
        "agg[accuracy]",
    ]
    assert dict(agg) == {"accuracy": 0.5, "score": 0.5}


def test_access_trace_preserve_keep_skips_value_lookup() -> None:
    events: list[str] = []
    agg = _TraceAgg(events, {"accuracy": 0.5})
    comp = _TraceComp(events, {"accuracy": 0.9})
    _merge(["accuracy"], agg, comp, True)
    assert events == ["comp.get:cost", "comp.items", "agg.in:accuracy"]


def test_access_trace_score_alias_reads_accuracy() -> None:
    events: list[str] = []
    agg = _TraceAgg(events, {"accuracy": 0.25})
    comp = _TraceComp(events, {"score": 0.9})
    _merge(["accuracy"], agg, comp, True)
    assert events == [
        "comp.get:cost",
        "comp.items",
        "agg.in:accuracy",
        "agg.in:score",
        "agg[accuracy]",
        "agg.set:score",
    ]


def test_access_trace_non_accuracy_key_is_a_plain_set() -> None:
    events: list[str] = []
    agg = _TraceAgg(events, {})
    comp = _TraceComp(events, {"f1": 0.9})
    _merge(["accuracy"], agg, comp, True)
    assert events == ["comp.get:cost", "comp.items", "agg.set:f1"]


def test_raising_aggregated_accessor_propagates() -> None:
    class _Agg(dict[str, Any]):
        def __contains__(self, key: object) -> bool:
            raise RuntimeError("probe")

    with pytest.raises(RuntimeError, match="probe"):
        _merge(["accuracy"], _Agg(), {"accuracy": 1.0}, False)


def test_raising_comprehensive_get_propagates_before_merge() -> None:
    class _Comp(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get")

    agg: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="get"):
        _merge(["cost"], agg, _Comp(a=1), False)
    assert agg == {}
