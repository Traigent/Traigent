"""Characterization of CustomEvaluatorWrapper._enhance_result_with_llm_metrics."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from traigent.api.types import ExampleResult
from traigent.core import evaluator_wrapper as ew
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper


def _wrapper(*, capture: bool = True, available: bool = True) -> CustomEvaluatorWrapper:
    wrapper = CustomEvaluatorWrapper(lambda *a, **k: None, capture_llm_metrics=capture)
    wrapper._metrics_available = available
    return wrapper


def _result(
    *,
    metrics: Any = None,
    metadata: Any = None,
    execution_time: Any = 0.0,
) -> ExampleResult:
    return ExampleResult(
        example_id="x",
        input_data={},
        expected_output=None,
        actual_output=None,
        metrics={} if metrics is None else metrics,
        execution_time=execution_time,
        success=True,
        metadata={} if metadata is None else metadata,
    )


def _usage(
    *,
    unmeasured: Any = False,
    known: Any = 0.0,
    response_ms: Any = 0,
    total_cost: Any = 0.003,
) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        cost=SimpleNamespace(
            input_cost=0.001,
            output_cost=0.002,
            total_cost=total_cost,
            unmeasured=unmeasured,
            known_cost=known,
        ),
        response=SimpleNamespace(response_time_ms=response_ms),
    )


def _enhance(
    result: ExampleResult,
    usage: Any,
    duration: Any,
    *,
    capture: bool = True,
    available: bool = True,
) -> None:
    _wrapper(capture=capture, available=available)._enhance_result_with_llm_metrics(
        result, usage, duration
    )


def test_measured_usage_full_result_without_response_time() -> None:
    result = _result()
    _enhance(result, _usage(), 0.25)
    assert result.metrics == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003,
        "execution_time_ms": 250.0,
        "function_duration": 0.25,
    }
    assert list(result.metrics) == [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "execution_time_ms",
        "function_duration",
    ]
    assert result.metadata == {
        "llm_usage_measured": True,
        "execution_time_ms": 250.0,
        "function_duration": 0.25,
    }
    assert result.execution_time == 0.25


def test_measured_usage_with_response_time_full_result() -> None:
    result = _result()
    _enhance(result, _usage(response_ms=1500), 2.0)
    assert result.metrics == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003,
        "response_time_ms": 1500.0,
        "model_response_time": 1.5,
        "execution_time_ms": 2000.0,
        "function_duration": 2.0,
    }
    assert type(result.metrics["response_time_ms"]) is float
    assert result.metadata == {
        "llm_usage_measured": True,
        "response_time_ms": 1500.0,
        "model_response_time": 1.5,
        "execution_time_ms": 2000.0,
        "function_duration": 2.0,
    }


@pytest.mark.parametrize("response_ms", [0, -5, 0.0, -0.1])
def test_non_positive_response_time_is_omitted(response_ms: Any) -> None:
    result = _result()
    _enhance(result, _usage(response_ms=response_ms), 1.0)
    assert "response_time_ms" not in result.metrics
    assert "model_response_time" not in result.metrics
    assert "response_time_ms" not in result.metadata


def test_missing_nested_sections_use_defaults() -> None:
    result = _result()
    _enhance(result, SimpleNamespace(marker=1), 1.0)
    assert result.metrics == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "execution_time_ms": 1000.0,
        "function_duration": 1.0,
    }


def test_non_numeric_response_time_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _enhance(_result(), _usage(response_ms="12"), 1.0)


@pytest.mark.parametrize(
    ("total", "known", "lower_bound"),
    [(0.5, 0.25, 0.75), (0.0, 0.5, 0.5), (0.25, 0.0, 0.25)],
)
def test_unmeasured_cost_flags_and_keeps_floor(
    total: float, known: float, lower_bound: float
) -> None:
    result = _result()
    _enhance(result, _usage(unmeasured=True, known=known, total_cost=total), 1.0)
    assert result.metrics == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "cost_unmeasured": 1.0,
        "cost_lower_bound": lower_bound,
        "execution_time_ms": 1000.0,
        "function_duration": 1.0,
    }


@pytest.mark.parametrize("total", [0.0, -1.0])
def test_unmeasured_cost_without_positive_floor_has_no_lower_bound(
    total: float,
) -> None:
    result = _result()
    _enhance(result, _usage(unmeasured=True, known=0.0, total_cost=total), 1.0)
    assert result.metrics == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "cost_unmeasured": 1.0,
        "execution_time_ms": 1000.0,
        "function_duration": 1.0,
    }


@pytest.mark.parametrize("unmeasured", [1, "yes", [1]])
def test_truthy_unmeasured_values_take_the_unmeasured_branch(unmeasured: Any) -> None:
    result = _result()
    _enhance(result, _usage(unmeasured=unmeasured, known=0.0, total_cost=0.0), 1.0)
    assert result.metrics["cost_unmeasured"] == 1.0
    assert "input_cost" not in result.metrics


@pytest.mark.parametrize("unmeasured", [0, "", None, False])
def test_falsey_unmeasured_values_take_the_measured_branch(unmeasured: Any) -> None:
    result = _result()
    _enhance(result, _usage(unmeasured=unmeasured), 1.0)
    assert "cost_unmeasured" not in result.metrics
    assert result.metrics["total_cost"] == 0.003


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("usage_truthy", [True, False])
def test_measured_flag_requires_capture_and_availability(
    capture: bool, available: bool, usage_truthy: bool
) -> None:
    result = _result()
    usage = _usage() if usage_truthy else None
    _enhance(result, usage, 1.0, capture=capture, available=available)
    if capture and available:
        assert result.metadata["llm_usage_measured"] is usage_truthy
    else:
        assert "llm_usage_measured" not in result.metadata


@pytest.mark.parametrize("usage", [0, "", {}, [], None])
def test_falsey_usage_skips_usage_metrics(usage: Any) -> None:
    result = _result()
    _enhance(result, usage, 0.5)
    assert result.metrics == {"execution_time_ms": 500.0, "function_duration": 0.5}
    assert result.metadata["llm_usage_measured"] is False


def test_metrics_container_is_created_for_truthy_usage() -> None:
    result = _result()
    cast(Any, result).metrics = None
    _enhance(result, _usage(), None)
    assert result.metrics["input_tokens"] == 11
    assert "execution_time_ms" not in result.metrics


def test_missing_metrics_with_duration_and_falsey_usage_raises() -> None:
    result = _result()
    cast(Any, result).metrics = None
    with pytest.raises(TypeError):
        _enhance(result, None, 1.0)


def test_metadata_none_is_replaced_with_dict() -> None:
    result = _result()
    cast(Any, result).metadata = None
    _enhance(result, _usage(), 1.0, capture=False)
    assert result.metadata == {"execution_time_ms": 1000.0, "function_duration": 1.0}


def test_no_duration_and_no_execution_time_assigns_none() -> None:
    result = _result(execution_time=0.0)
    _enhance(result, None, None)
    assert result.execution_time is None
    assert result.metrics == {}
    assert result.metadata == {"llm_usage_measured": False}


def test_existing_execution_time_is_kept_and_backfills_with_duration_missing() -> None:
    result = _result(execution_time=0.5)
    _enhance(result, None, None)
    assert result.execution_time == 0.5
    assert result.metrics == {"execution_time_ms": 500.0}
    assert result.metadata == {
        "llm_usage_measured": False,
        "execution_time_ms": 500.0,
        "function_duration": 0.5,
    }


def test_existing_execution_time_does_not_override_fresh_duration_keys() -> None:
    result = _result(execution_time=0.5)
    _enhance(result, None, 2.0)
    assert result.execution_time == 0.5
    assert result.metrics == {"execution_time_ms": 2000.0, "function_duration": 2.0}
    assert result.metadata == {
        "llm_usage_measured": False,
        "execution_time_ms": 2000.0,
        "function_duration": 2.0,
    }


def test_existing_execution_time_keeps_prefilled_values() -> None:
    result = _result(
        execution_time=0.5,
        metrics={"execution_time_ms": 1.0},
        metadata={"execution_time_ms": 2.0, "function_duration": 3.0},
    )
    _enhance(result, None, None)
    assert result.metrics == {"execution_time_ms": 1.0}
    assert result.metadata == {
        "execution_time_ms": 2.0,
        "function_duration": 3.0,
        "llm_usage_measured": False,
    }


def test_zero_duration_is_recorded_and_replaces_zero_execution_time() -> None:
    result = _result(execution_time=0.0)
    _enhance(result, None, 0.0)
    assert result.metrics == {"execution_time_ms": 0.0, "function_duration": 0.0}
    assert result.execution_time == 0.0


def test_missing_metrics_with_existing_execution_time_and_no_duration_raises() -> None:
    result = _result(execution_time=0.5)
    cast(Any, result).metrics = None
    with pytest.raises(AttributeError):
        _enhance(result, None, None)


def _recording_getter(
    events: list[str],
    raise_on: str | None = None,
) -> Callable[..., Any]:
    real = ew.safe_get_nested_attr

    def getter(obj: Any, path: str, default: Any = None) -> Any:
        events.append(path)
        if path == raise_on:
            raise RuntimeError(path)
        return real(obj, path, default)

    return getter


def test_nested_read_order_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "safe_get_nested_attr", _recording_getter(events))
    _enhance(_result(), _usage(response_ms=3), 1.0)
    assert events == [
        "tokens.input_tokens",
        "tokens.output_tokens",
        "tokens.total_tokens",
        "cost.unmeasured",
        "cost.input_cost",
        "cost.output_cost",
        "cost.total_cost",
        "response.response_time_ms",
    ]


def test_nested_read_order_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "safe_get_nested_attr", _recording_getter(events))
    _enhance(_result(), _usage(unmeasured=True, known=1.0), 1.0)
    assert events == [
        "tokens.input_tokens",
        "tokens.output_tokens",
        "tokens.total_tokens",
        "cost.unmeasured",
        "cost.total_cost",
        "cost.known_cost",
        "response.response_time_ms",
    ]


def test_nested_reads_skipped_for_falsey_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "safe_get_nested_attr", _recording_getter(events))
    _enhance(_result(), None, 1.0)
    assert events == []


@pytest.mark.parametrize(
    ("raise_on", "expected_keys"),
    [
        ("tokens.output_tokens", ["input_tokens"]),
        ("cost.unmeasured", ["input_tokens", "output_tokens", "total_tokens"]),
        (
            "cost.output_cost",
            ["input_tokens", "output_tokens", "total_tokens", "input_cost"],
        ),
        (
            "response.response_time_ms",
            [
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "input_cost",
                "output_cost",
                "total_cost",
            ],
        ),
    ],
)
def test_accessor_failure_propagates_with_partial_writes(
    monkeypatch: pytest.MonkeyPatch, raise_on: str, expected_keys: list[str]
) -> None:
    events: list[str] = []
    monkeypatch.setattr(ew, "safe_get_nested_attr", _recording_getter(events, raise_on))
    result = _result()
    with pytest.raises(RuntimeError, match=raise_on):
        _enhance(result, _usage(response_ms=3), 1.0)
    assert list(result.metrics) == expected_keys
    assert "execution_time_ms" not in result.metrics
    assert result.metadata == {"llm_usage_measured": True}
