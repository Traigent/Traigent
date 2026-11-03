"""Characterization tests pinning SimpleScoringEvaluator._aggregate_custom_metrics."""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Any

import pytest

from traigent.evaluators import base
from traigent.evaluators.base import SimpleScoringEvaluator

STRICT_ENV = "TRAIGENT_STRICT_METRICS_NULLS"


def _evaluator(metrics: list[str]) -> SimpleScoringEvaluator:
    return SimpleScoringEvaluator(
        scoring_function=lambda output, expected: 1.0, metrics=metrics
    )


@pytest.fixture(autouse=True)
def _clear_strict_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STRICT_ENV, raising=False)


class _Row(dict[str, Any]):
    """A row recording truth tests and reads into a shared trace."""

    def __init__(self, name: str, trace: list[Any], data: dict[str, Any]) -> None:
        super().__init__(data)
        self.name = name
        self.trace = trace

    def __bool__(self) -> bool:
        self.trace.append(("bool", self.name))
        return len(self) > 0

    def get(self, key: str, default: Any = None) -> Any:
        self.trace.append(("get", self.name, key, default))
        return super().get(key, default)


class _Env:
    def __init__(self, trace: list[Any], value: str) -> None:
        self.trace = trace
        self.value = value

    def get(self, key: str, default: str = "") -> str:
        self.trace.append(("env", key, default))
        return self.value


# ---------------------------------------------------------------- plain metrics


def test_plain_metric_is_the_mean_in_metric_order() -> None:
    evaluator = _evaluator(["accuracy", "score"])
    rows = [{"accuracy": 1.0, "score": 0.5}, {"accuracy": 0.0, "score": 1.5}]
    result = evaluator._aggregate_custom_metrics(rows)
    assert result == {"accuracy": 0.5, "score": 1.0}
    assert list(result) == ["accuracy", "score"]


def test_plain_metric_missing_in_a_truthy_row_counts_as_zero() -> None:
    evaluator = _evaluator(["accuracy"])
    result = evaluator._aggregate_custom_metrics([{"accuracy": 3.0}, {"other": 9.0}])
    assert result == {"accuracy": 1.5}


def test_falsy_rows_are_excluded_from_plain_metrics() -> None:
    evaluator = _evaluator(["accuracy"])
    rows: list[Any] = [{"accuracy": 2.0}, {}, None, {"accuracy": 4.0}]
    assert evaluator._aggregate_custom_metrics(rows) == {"accuracy": 3.0}


@pytest.mark.parametrize("rows", [[], [{}], [None], [{}, None]])
def test_plain_metric_without_usable_rows_is_exactly_zero(rows: list[Any]) -> None:
    result = _evaluator(["accuracy"])._aggregate_custom_metrics(rows)
    assert result == {"accuracy": 0.0}
    assert isinstance(result["accuracy"], float)


def test_no_configured_metrics_gives_empty_result() -> None:
    evaluator = _evaluator(["accuracy"])
    evaluator.metrics = []
    assert evaluator._aggregate_custom_metrics([{"accuracy": 1.0}]) == {}


@pytest.mark.parametrize(
    "name", ["latency_p95", "P95_latency", "my_P95_ms", "p95", "Latency_P95"]
)
def test_p95_metrics_use_the_percentile(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls: list[tuple[list[float], float]] = []

    def fake(values: list[float], percentile: float) -> float:
        calls.append((values, percentile))
        return 42.5

    monkeypatch.setattr(
        SimpleScoringEvaluator, "_compute_percentile", staticmethod(fake)
    )
    evaluator = _evaluator([name])
    result = evaluator._aggregate_custom_metrics(
        [{name: 1.0}, {}, {name: 5.0}, {"x": 1.0}]
    )
    assert result == {name: 42.5}
    assert calls == [([1.0, 5.0, 0.0], 0.95)]


def test_p95_real_percentile_value() -> None:
    evaluator = _evaluator(["latency_p95"])
    rows = [{"latency_p95": float(v)} for v in (1, 2, 3, 4, 5)]
    # ordered [1..5], index 0.95 * 4 = 3.8 -> 4 + 0.8 * (5 - 4)
    result = evaluator._aggregate_custom_metrics(rows)
    assert result["latency_p95"] == pytest.approx(4.8)


def test_p95_without_rows_is_zero_not_percentile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(values: list[float], percentile: float) -> float:
        raise AssertionError("must not be called")

    monkeypatch.setattr(
        SimpleScoringEvaluator, "_compute_percentile", staticmethod(boom)
    )
    assert _evaluator(["latency_p95"])._aggregate_custom_metrics([{}]) == {
        "latency_p95": 0.0
    }


def test_non_p95_name_uses_mean_even_when_p95_in_other_case_missing() -> None:
    evaluator = _evaluator(["p9_latency"])
    rows = [{"p9_latency": 1.0}, {"p9_latency": 3.0}]
    assert evaluator._aggregate_custom_metrics(rows) == {"p9_latency": 2.0}


# --------------------------------------------------------------- measured-only


@pytest.mark.parametrize("name", ["cost", "input_cost", "output_cost", "total_cost"])
def test_measured_only_metric_averages_measured_rows(name: str) -> None:
    evaluator = _evaluator([name])
    rows: list[Any] = [{name: 0.25}, {}, None, {name: 0.75}]
    assert evaluator._aggregate_custom_metrics(rows) == {name: 0.5}


def test_measured_only_metric_delegates_with_the_same_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, Any]] = []

    def fake(metric: str, rows: Any) -> float:
        seen.append((metric, rows))
        return 7.25

    monkeypatch.setattr(base, "aggregate_measured_metric", fake)
    rows = [{"cost": 1.0}, {"cost": None}]
    result = _evaluator(["cost"])._aggregate_custom_metrics(rows)
    assert result == {"cost": 7.25}
    assert len(seen) == 1 and seen[0][0] == "cost" and seen[0][1] is rows


@pytest.mark.parametrize(
    "cell", [None, True, False, "0.5", float("nan"), float("inf"), -float("inf"), [1]]
)
def test_unmeasured_cell_values_do_not_count_as_measured(cell: Any) -> None:
    evaluator = _evaluator(["cost"])
    assert evaluator._aggregate_custom_metrics([{"cost": cell}, {}]) == {}


def test_int_and_zero_count_as_measured() -> None:
    evaluator = _evaluator(["cost"])
    assert evaluator._aggregate_custom_metrics([{"cost": 0}]) == {"cost": 0.0}
    assert evaluator._aggregate_custom_metrics([{"cost": 2}, {"cost": 4}]) == {
        "cost": 3.0
    }


@pytest.mark.parametrize("rows", [[], [{}], [None], [{"accuracy": 1.0}]])
def test_no_measured_rows_omit_the_key_by_default(rows: list[Any]) -> None:
    result = _evaluator(["cost"])._aggregate_custom_metrics(rows)
    assert result == {}
    assert "cost" not in result


@pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "YES", "Yes"])
def test_strict_nulls_records_none_when_nothing_measured(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(STRICT_ENV, value)
    result = _evaluator(["cost", "accuracy"])._aggregate_custom_metrics(
        [{"accuracy": 1.0}]
    )
    assert result == {"cost": None, "accuracy": 1.0}
    assert list(result) == ["cost", "accuracy"]


@pytest.mark.parametrize(
    "value", ["", "false", "0", "no", "on", " true", "true ", "y", "2", "tru"]
)
def test_other_env_values_leave_the_key_out(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(STRICT_ENV, value)
    result = _evaluator(["cost", "accuracy"])._aggregate_custom_metrics(
        [{"accuracy": 1.0}]
    )
    assert result == {"accuracy": 1.0}


def test_strict_nulls_does_not_affect_measured_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_ENV, "true")
    result = _evaluator(["cost"])._aggregate_custom_metrics([{"cost": 0.5}])
    assert result == {"cost": 0.5}


def test_strict_nulls_does_not_affect_plain_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_ENV, "true")
    result = _evaluator(["accuracy"])._aggregate_custom_metrics([{}])
    assert result == {"accuracy": 0.0}


def test_mixed_metrics_keep_configuration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_ENV, "1")
    evaluator = _evaluator(["total_cost", "accuracy", "cost", "latency_p95"])
    result = evaluator._aggregate_custom_metrics(
        [{"accuracy": 1.0, "latency_p95": 3.0, "total_cost": 2.0}]
    )
    assert list(result.items()) == [
        ("total_cost", 2.0),
        ("accuracy", 1.0),
        ("cost", None),
        ("latency_p95", 3.0),
    ]


# ------------------------------------------------------- traces and accessors


def test_event_trace_for_measured_rows_has_no_env_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    monkeypatch.setattr(base, "os", types.SimpleNamespace(environ=_Env(trace, "true")))
    real_measured = base._measured_value

    def spy(value: Any) -> Any:
        trace.append(("measured", value))
        return real_measured(value)

    monkeypatch.setattr(base, "_measured_value", spy)

    def agg(metric: str, rows: Any) -> float:
        trace.append(("agg", metric))
        return 1.0

    monkeypatch.setattr(base, "aggregate_measured_metric", agg)
    rows = [
        _Row("r0", trace, {}),
        _Row("r1", trace, {"cost": None}),
        _Row("r2", trace, {"cost": 3.0}),
        _Row("r3", trace, {"cost": 9.0}),
    ]
    result = _evaluator(["cost"])._aggregate_custom_metrics(rows)
    assert result == {"cost": 1.0}
    # ``row and ...`` truth-tests a falsy row, then ``any`` truth-tests the
    # value the expression returned: two evaluations for r0.
    assert trace == [
        ("bool", "r0"),
        ("bool", "r0"),
        ("bool", "r1"),
        ("get", "r1", "cost", None),
        ("measured", None),
        ("bool", "r2"),
        ("get", "r2", "cost", None),
        ("measured", 3.0),
        ("agg", "cost"),
    ]


def test_event_trace_for_unmeasured_rows_reads_env_once_after_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    monkeypatch.setattr(base, "os", types.SimpleNamespace(environ=_Env(trace, "yes")))
    rows = [_Row("r0", trace, {}), _Row("r1", trace, {"cost": True})]
    result = _evaluator(["cost"])._aggregate_custom_metrics(rows)
    assert result == {"cost": None}
    assert trace == [
        ("bool", "r0"),
        ("bool", "r0"),
        ("bool", "r1"),
        ("get", "r1", "cost", None),
        ("env", STRICT_ENV, ""),
    ]


def test_event_trace_for_plain_metric_reads_each_row_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[Any] = []
    monkeypatch.setattr(base, "os", types.SimpleNamespace(environ=_Env(trace, "true")))
    rows = [_Row("r0", trace, {"accuracy": 1.0}), _Row("r1", trace, {})]
    result = _evaluator(["accuracy"])._aggregate_custom_metrics(rows)
    assert result == {"accuracy": 1.0}
    assert trace == [
        ("bool", "r0"),
        ("get", "r0", "accuracy", 0.0),
        ("bool", "r1"),
    ]


def test_metrics_sequence_is_iterated_exactly_once() -> None:
    iterations: list[str] = []

    class Metrics(list[str]):
        def __iter__(self) -> Iterator[str]:
            iterations.append("iter")
            return super().__iter__()

    evaluator = _evaluator(["accuracy"])
    evaluator.metrics = Metrics(["accuracy", "cost"])
    result = evaluator._aggregate_custom_metrics([{"accuracy": 2.0, "cost": 1.0}])
    assert result == {"accuracy": 2.0, "cost": 1.0}
    assert iterations == ["iter"]


def test_row_get_error_propagates_for_plain_metric() -> None:
    class Bad(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("accessor failed")

    with pytest.raises(RuntimeError, match="accessor failed"):
        _evaluator(["accuracy"])._aggregate_custom_metrics([Bad({"accuracy": 1.0})])


def test_row_get_error_propagates_for_measured_metric() -> None:
    class Bad(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("accessor failed")

    with pytest.raises(RuntimeError, match="accessor failed"):
        _evaluator(["cost"])._aggregate_custom_metrics([Bad({"cost": 1.0})])


def test_row_bool_error_propagates_before_any_other_read() -> None:
    trace: list[Any] = []

    class Bad(dict[str, Any]):
        def __bool__(self) -> bool:
            trace.append("bool")
            raise RuntimeError("truth failed")

    with pytest.raises(RuntimeError, match="truth failed"):
        _evaluator(["cost"])._aggregate_custom_metrics([Bad()])
    assert trace == ["bool"]


def test_earlier_metrics_are_not_returned_when_a_later_one_raises() -> None:
    class Bad(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "cost":
                raise RuntimeError("accessor failed")
            return super().get(key, default)

    with pytest.raises(RuntimeError, match="accessor failed"):
        _evaluator(["accuracy", "cost"])._aggregate_custom_metrics(
            [Bad({"accuracy": 1.0})]
        )


def test_percentile_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(values: list[float], percentile: float) -> float:
        raise ValueError("percentile failed")

    monkeypatch.setattr(
        SimpleScoringEvaluator, "_compute_percentile", staticmethod(boom)
    )
    with pytest.raises(ValueError, match="percentile failed"):
        _evaluator(["latency_p95"])._aggregate_custom_metrics([{"latency_p95": 1.0}])


def test_non_numeric_plain_values_raise_type_error_from_the_sum() -> None:
    with pytest.raises(TypeError):
        _evaluator(["accuracy"])._aggregate_custom_metrics([{"accuracy": "x"}])


def test_measured_aggregation_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(metric: str, rows: Any) -> float:
        raise RuntimeError("aggregation failed")

    monkeypatch.setattr(base, "aggregate_measured_metric", boom)
    with pytest.raises(RuntimeError, match="aggregation failed"):
        _evaluator(["cost"])._aggregate_custom_metrics([{"cost": 1.0}])


def test_percentile_is_resolved_on_the_instance() -> None:
    class Overridden(SimpleScoringEvaluator):
        @staticmethod
        def _compute_percentile(values: list[float], percentile: float) -> float:
            return 99.0

    evaluator = Overridden(
        scoring_function=lambda output, expected: 1.0, metrics=["latency_p95"]
    )
    assert evaluator._aggregate_custom_metrics([{"latency_p95": 1.0}]) == {
        "latency_p95": 99.0
    }
