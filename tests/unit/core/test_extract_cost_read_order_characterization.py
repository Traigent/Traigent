"""Characterization tests for the read order and edge cases of
``extract_cost_from_results``.

Complements the value-matrix tests: these pin the number and order of attribute
reads, collaborator calls, identity of pass-through values and how malformed
inputs behave.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.core import orchestrator_helpers as helpers
from traigent.core.orchestrator_helpers import extract_cost_from_results


class _IntSub(int):
    """int subclass used to observe whether a value is passed through or replaced."""


class _FloatSub(float):
    pass


class _TracingMap(dict[str, Any]):
    def __init__(self, trace: list[str], data: dict[str, Any]) -> None:
        super().__init__(data)
        self._trace = trace

    def get(self, key: str, default: Any = None) -> Any:
        self._trace.append(f"agg.get:{key}")
        return super().get(key, default)


class _TracedResult:
    """Evaluation result whose attributes record every read."""

    def __init__(self, trace: list[str], **values: Any) -> None:
        self._trace = trace
        self._values = values

    def _read(self, name: str) -> Any:
        self._trace.append(f"read:{name}")
        if name not in self._values:
            raise AttributeError(name)
        return self._values[name]

    @property
    def example_results(self) -> Any:
        return self._read("example_results")

    @property
    def aggregated_metrics(self) -> Any:
        return self._read("aggregated_metrics")

    @property
    def total_examples(self) -> Any:
        return self._read("total_examples")


class _Row:
    def __init__(self, metrics: Any = None, success: bool = True) -> None:
        self.metrics = metrics
        self.success = success


@pytest.fixture
def collab_trace(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    trace: list[str] = []
    real_total = helpers._total_cost_from_examples
    real_unmeasured = helpers.is_cost_unmeasured

    def total_spy(examples: Any, trial_id: str, current: Any) -> Any:
        trace.append(f"total_from_examples:{trial_id}:{current!r}")
        return real_total(examples, trial_id, current)

    def unmeasured_spy(metrics: Any) -> bool:
        trace.append("is_cost_unmeasured")
        return bool(real_unmeasured(metrics))

    monkeypatch.setattr(helpers, "_total_cost_from_examples", total_spy)
    monkeypatch.setattr(helpers, "is_cost_unmeasured", unmeasured_spy)
    return trace


def test_progress_state_truth_is_evaluated_once_and_get_order(
    collab_trace: list[str],
) -> None:
    events: list[str] = []

    class _Progress(dict[str, Any]):
        def __bool__(self) -> bool:
            events.append("bool")
            return True

        def get(self, key: str, default: Any = None) -> Any:
            events.append(f"get:{key}")
            return super().get(key, default)

    progress = _Progress({"evaluated": 4, "total_cost": 0.5})
    out = extract_cost_from_results(_TracedResult([]), progress, "t")
    assert out == (4, 0.5)
    assert events == ["bool", "get:evaluated", "get:total_cost"]


def test_falsy_progress_state_is_never_read() -> None:
    events: list[str] = []

    class _Falsy(dict[str, Any]):
        def __bool__(self) -> bool:
            events.append("bool")
            return False

        def get(self, key: str, default: Any = None) -> Any:
            events.append(f"get:{key}")
            return super().get(key, default)

    out = extract_cost_from_results(
        _TracedResult([]), _Falsy({"evaluated": 4, "total_cost": 0.5}), "t"
    )
    assert out == (None, None)
    assert events == ["bool"]


def test_progress_values_are_passed_through_by_identity() -> None:
    evaluated = _IntSub(4)
    cost = _FloatSub(0.5)
    out = extract_cost_from_results(
        _TracedResult([]), {"evaluated": evaluated, "total_cost": cost}, "t"
    )
    assert out[0] is evaluated
    assert out[1] is cost


def test_raising_progress_get_propagates() -> None:
    class _Bad(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get")

    with pytest.raises(RuntimeError, match="get"):
        extract_cost_from_results(_TracedResult([]), _Bad({"evaluated": 1}), "t")


def test_attribute_reads_with_no_progress_cost_and_aggregated_fallback(
    collab_trace: list[str],
) -> None:
    trace: list[str] = []
    agg = _TracingMap(trace, {"total_cost": 0.4, "cost": 0.9})
    result = _TracedResult(trace, aggregated_metrics=agg)
    assert extract_cost_from_results(result, None, "t") == (None, 0.4)
    assert trace == [
        "read:example_results",
        "read:aggregated_metrics",
        "read:aggregated_metrics",
        "agg.get:total_cost",
        "read:aggregated_metrics",
        "agg.get:cost_unmeasured",
        "read:total_examples",
    ]
    assert collab_trace == ["is_cost_unmeasured"]


def test_cost_key_is_read_only_when_total_cost_is_none() -> None:
    trace: list[str] = []
    agg = _TracingMap(trace, {"total_cost": None, "cost": 0.9})
    result = _TracedResult(trace, aggregated_metrics=agg)
    assert extract_cost_from_results(result, None, "t") == (None, 0.9)
    assert [e for e in trace if e.startswith("agg.get")] == [
        "agg.get:total_cost",
        "agg.get:cost",
        "agg.get:cost_unmeasured",
    ]


def test_zero_total_cost_is_kept_and_cost_key_not_read() -> None:
    trace: list[str] = []
    agg = _TracingMap(trace, {"total_cost": 0, "cost": 5})
    result = _TracedResult(trace, aggregated_metrics=agg)
    out = extract_cost_from_results(result, None, "t")
    assert out == (None, 0)
    assert out[1] == 0 and out[1] is not None
    assert [e for e in trace if e.startswith("agg.get")] == [
        "agg.get:total_cost",
        "agg.get:cost_unmeasured",
    ]


def test_progress_cost_skips_the_aggregated_fallback_read() -> None:
    trace: list[str] = []
    agg = _TracingMap(trace, {"total_cost": 7.0})
    result = _TracedResult(trace, aggregated_metrics=agg)
    out = extract_cost_from_results(result, {"total_cost": 1.5}, "t")
    assert out == (None, 1.5)
    assert trace == [
        "read:example_results",
        "read:aggregated_metrics",
        "agg.get:cost_unmeasured",
        "read:total_examples",
    ]


def test_truthy_but_empty_looking_aggregated_metrics_is_replaced_by_dict() -> None:
    events: list[str] = []

    class _FalsyMap(dict[str, Any]):
        def __bool__(self) -> bool:
            events.append("bool")
            return False

        def get(self, key: str, default: Any = None) -> Any:
            events.append(f"get:{key}")
            return super().get(key, default)

    result = _TracedResult([], aggregated_metrics=_FalsyMap({"total_cost": 9.0}))
    assert extract_cost_from_results(result, None, "t") == (None, None)
    assert events[0] == "bool"
    assert "get:total_cost" not in events


def test_aggregated_metrics_none_and_falsy_values_fall_to_empty() -> None:
    aggs: list[Any] = [None, {}, [], 0, ""]
    for agg in aggs:
        result = _TracedResult([], aggregated_metrics=agg)
        assert extract_cost_from_results(result, None, "t") == (None, None)


def test_non_mapping_truthy_aggregated_metrics_raise_attribute_error() -> None:
    result = _TracedResult([], aggregated_metrics=[1, 2])
    with pytest.raises(AttributeError):
        extract_cost_from_results(result, None, "t")


def test_raising_aggregated_get_propagates() -> None:
    class _Bad(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("agg")

    result = _TracedResult([], aggregated_metrics=_Bad({"a": 1}))
    with pytest.raises(RuntimeError, match="agg"):
        extract_cost_from_results(result, None, "t")


def test_raise_once_aggregated_accessor_is_not_retried() -> None:
    state = {"reads": 0}

    class _Flaky:
        @property
        def aggregated_metrics(self) -> Any:
            state["reads"] += 1
            raise RuntimeError("first")

    with pytest.raises(RuntimeError, match="first"):
        extract_cost_from_results(_Flaky(), None, "t")
    assert state["reads"] == 1


def test_attribute_error_from_aggregated_accessor_counts_as_absent() -> None:
    class _Absent:
        @property
        def aggregated_metrics(self) -> Any:
            raise AttributeError("gone")

    assert extract_cost_from_results(_Absent(), None, "t") == (None, None)


def test_unmeasured_flag_clears_cost_after_all_other_sources(
    collab_trace: list[str],
) -> None:
    trace: list[str] = []
    agg = _TracingMap(trace, {"cost_unmeasured": 1.0, "total_cost": 5.0})
    result = _TracedResult(trace, aggregated_metrics=agg)
    out = extract_cost_from_results(result, {"evaluated": 2, "total_cost": 2.0}, "t")
    assert out == (2, None)
    assert collab_trace == ["is_cost_unmeasured"]


def test_unmeasured_check_receives_none_when_attribute_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def recorder(metrics: Any) -> bool:
        seen.append(metrics)
        return False

    monkeypatch.setattr(helpers, "is_cost_unmeasured", recorder)
    assert extract_cost_from_results(_TracedResult([]), {"total_cost": 1.0}, "t") == (
        None,
        1.0,
    )
    assert seen == [None]


def test_example_results_trace_and_collaborator_order(
    collab_trace: list[str],
) -> None:
    trace: list[str] = []
    rows = [_Row({"cost": 0.25}), _Row({"cost": 0.5})]
    result = _TracedResult(trace, example_results=rows, total_examples=2)
    assert extract_cost_from_results(result, None, "trial-1") == (2, 0.75)
    assert trace == [
        "read:example_results",
        "read:example_results",
        "read:example_results",
        "read:example_results",
        "read:example_results",
        "read:aggregated_metrics",
        "read:total_examples",
    ]
    assert collab_trace == ["total_from_examples:trial-1:None", "is_cost_unmeasured"]


def test_example_results_progress_attempted_skips_len_read() -> None:
    trace: list[str] = []
    result = _TracedResult(trace, example_results=[_Row({"cost": 1.0})])
    out = extract_cost_from_results(result, {"evaluated": 5}, "t")
    assert out == (5, 1.0)
    assert trace.count("read:example_results") == 4


def test_example_results_truthiness_evaluated_once_per_read() -> None:
    events: list[str] = []

    class _Rows(list[Any]):
        def __bool__(self) -> bool:
            events.append("bool")
            return True

    result = _TracedResult([], example_results=_Rows([_Row({"cost": 1.0})]))
    assert extract_cost_from_results(result, None, "t") == (1, 1.0)
    assert events == ["bool"]


def test_empty_example_results_fall_through_to_aggregated(
    collab_trace: list[str],
) -> None:
    result = _TracedResult([], example_results=[], aggregated_metrics={"cost": 2.5})
    assert extract_cost_from_results(result, {"evaluated": 3}, "t") == (3, 2.5)
    assert collab_trace == ["is_cost_unmeasured"]


def test_current_total_cost_is_forwarded_to_the_example_sum(
    collab_trace: list[str],
) -> None:
    result = _TracedResult([], example_results=[_Row({"x": 1})])
    out = extract_cost_from_results(result, {"evaluated": 1, "total_cost": 0.9}, "t9")
    assert out == (1, 0.9)
    assert collab_trace == ["total_from_examples:t9:0.9", "is_cost_unmeasured"]


def test_total_examples_boundaries_and_identity() -> None:
    five = _IntSub(5)
    # equal: the existing object is kept
    out = extract_cost_from_results(
        _TracedResult([], total_examples=5), {"evaluated": five}, "t"
    )
    assert out[0] is five
    # larger attempted: replaced by the reported total (exact value 5)
    out = extract_cost_from_results(
        _TracedResult([], total_examples=5), {"evaluated": _IntSub(6)}, "t"
    )
    assert out[0] == 5 and type(out[0]) is int
    # smaller attempted: kept
    out = extract_cost_from_results(
        _TracedResult([], total_examples=5), {"evaluated": _IntSub(4)}, "t"
    )
    assert out[0] == 4 and type(out[0]) is _IntSub
    # missing attempted: taken from the reported total
    out = extract_cost_from_results(_TracedResult([], total_examples=5), None, "t")
    assert out[0] == 5


@pytest.mark.parametrize("total", [0, -1, 2.0, "4", None, [4], True * 0])
def test_non_positive_or_non_int_total_examples_are_ignored(total: Any) -> None:
    out = extract_cost_from_results(
        _TracedResult([], total_examples=total), {"evaluated": 8}, "t"
    )
    assert out[0] == 8


def test_bool_true_total_examples_counts_as_the_integer_one() -> None:
    out = extract_cost_from_results(_TracedResult([], total_examples=True), None, "t")
    assert out[0] is True
    out = extract_cost_from_results(
        _TracedResult([], total_examples=True), {"evaluated": 3}, "t"
    )
    assert out[0] is True


def test_incomparable_attempted_value_raises_type_error() -> None:
    with pytest.raises(TypeError):
        extract_cost_from_results(
            _TracedResult([], total_examples=3), {"evaluated": "5"}, "t"
        )


def test_total_examples_is_read_once_and_after_cost_resolution() -> None:
    trace: list[str] = []
    result = _TracedResult(trace, total_examples=3)
    extract_cost_from_results(result, {"evaluated": 1}, "t")
    assert trace.count("read:total_examples") == 1
    assert trace[-1] == "read:total_examples"


def test_raising_total_examples_accessor_propagates() -> None:
    class _Bad:
        @property
        def total_examples(self) -> Any:
            raise RuntimeError("total")

    with pytest.raises(RuntimeError, match="total"):
        extract_cost_from_results(_Bad(), None, "t")


def test_base_exception_from_accessor_propagates() -> None:
    class _Interrupt:
        @property
        def aggregated_metrics(self) -> Any:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        extract_cost_from_results(_Interrupt(), None, "t")


def test_return_is_a_two_tuple() -> None:
    out = extract_cost_from_results(_TracedResult([]), None, "t")
    assert type(out) is tuple and len(out) == 2
