"""Characterization tests for the embedded ``record_example_result``.

They pin the span attribute order, the scrub and serialisation steps for the
actual output, the truncation boundary, the metric and timing rules, and the
failure behaviour of each step.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from traigent.core import tracing

pytestmark = pytest.mark.skipif(
    tracing._PLUGIN_AVAILABLE, reason="embedded implementation only"
)


class _Span:
    """Span double recording attribute writes in one shared event list."""

    def __init__(self, events: list[Any], fail_on: str | None = None) -> None:
        self.events = events
        self.fail_on = fail_on

    def set_attribute(self, key: str, value: Any) -> None:
        self.events.append(("attr", key, value))
        if key == self.fail_on:
            raise RuntimeError(f"cannot set {key}")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    log: list[Any] = []

    def _error_status(span: Any, error: Any, attribute_name: Any = None) -> None:
        log.append(("error_status", span is not None, error, attribute_name))

    monkeypatch.setattr(tracing, "_set_error_status", _error_status)
    return log


@pytest.fixture
def scrubbed(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace both scrub helpers with traced identity-style doubles."""
    calls: list[Any] = []

    def _export(value: Any) -> Any:
        calls.append(("export", value))
        return value

    def _pii(text: Any) -> Any:
        calls.append(("pii", text))
        return f"<{text}>"

    monkeypatch.setattr(tracing, "_scrub_for_export", _export)
    monkeypatch.setattr(tracing, "_scrub_pii_text", _pii)
    return calls


def _record(events: list[Any], **kwargs: Any) -> None:
    kwargs.setdefault("success", True)
    tracing.record_example_result(_Span(events), **kwargs)


def test_none_span_does_nothing_and_touches_no_argument(
    events: list[Any], scrubbed: list[Any]
) -> None:
    class _Boom:
        def __bool__(self) -> bool:
            raise AssertionError("must not be read")

        def items(self) -> Any:
            raise AssertionError("must not be read")

    result = tracing.record_example_result(
        None,
        success=True,
        actual_output="x",
        metrics=_Boom(),
        error="err",
        execution_time=1.0,
    )
    assert result is None
    assert events == []
    assert scrubbed == []


def test_success_is_recorded_first_and_unchanged(events: list[Any]) -> None:
    _record(events, success=False)
    assert events == [("attr", "example.success", False)]
    del events[:]
    sentinel = object()
    _record(events, success=sentinel)
    assert events == [("attr", "example.success", sentinel)]


def test_all_sections_are_written_in_order(
    events: list[Any], scrubbed: list[Any]
) -> None:
    _record(
        events,
        actual_output="out",
        metrics={"acc": 0.5, "n": 3},
        error="boom",
        execution_time=2.0,
    )
    assert events == [
        ("attr", "example.success", True),
        ("attr", "example.actual_output", "out"),
        ("attr", "example.metric.acc", 0.5),
        ("attr", "example.metric.n", 3),
        ("attr", "example.execution_time_ms", 2000.0),
        ("error_status", True, "boom", "example.error"),
    ]
    assert scrubbed == [("export", "out")]


def test_none_output_is_skipped_but_falsy_outputs_are_recorded(
    events: list[Any], scrubbed: list[Any]
) -> None:
    _record(events, actual_output=None)
    assert events == [("attr", "example.success", True)]
    assert scrubbed == []
    for value, text in [("", ""), (0, "0"), ([], "[]"), ({}, "{}"), (False, "false")]:
        del events[:]
        _record(events, actual_output=value)
        assert events[1] == ("attr", "example.actual_output", text)


def test_string_output_is_used_raw_and_other_values_are_json_encoded(
    events: list[Any], scrubbed: list[Any]
) -> None:
    _record(events, actual_output="plain text")
    assert events[1] == ("attr", "example.actual_output", "plain text")
    del events[:]
    _record(events, actual_output={"k": [1, "two"]})
    assert events[1] == ("attr", "example.actual_output", '{"k": [1, "two"]}')
    assert scrubbed == [("export", "plain text"), ("export", {"k": [1, "two"]})]


def test_scrub_result_not_input_is_what_gets_serialised(
    events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracing, "_scrub_for_export", lambda value: {"scrubbed": 1})
    _record(events, actual_output={"secret": 2})
    assert events[1] == ("attr", "example.actual_output", '{"scrubbed": 1}')


def test_str_subclass_result_is_passed_through_unconverted(
    events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Text(str):
        pass

    value = _Text("subclass")
    monkeypatch.setattr(tracing, "_scrub_for_export", lambda v: value)
    _record(events, actual_output="ignored")
    assert events[1][2] is value


def test_unserialisable_output_falls_back_to_scrubbed_str(
    events: list[Any], scrubbed: list[Any]
) -> None:
    class _Opaque:
        def __str__(self) -> str:
            return "opaque-text"

    _record(events, actual_output=_Opaque())
    assert events[1] == ("attr", "example.actual_output", "<opaque-text>")
    assert [c[0] for c in scrubbed] == ["export", "pii"]
    assert scrubbed[1] == ("pii", "opaque-text")


def test_circular_output_falls_back_to_scrubbed_str(
    events: list[Any], scrubbed: list[Any]
) -> None:
    loop: list[Any] = []
    loop.append(loop)
    _record(events, actual_output=loop)
    assert events[1] == ("attr", "example.actual_output", "<[[...]]>")


def test_str_of_unserialisable_output_is_taken_once(
    events: list[Any], scrubbed: list[Any]
) -> None:
    counts = {"str": 0}

    class _Opaque:
        def __str__(self) -> str:
            counts["str"] += 1
            return "once"

    _record(events, actual_output=_Opaque())
    assert counts["str"] == 1


def test_other_serialisation_errors_propagate_after_success_is_written(
    events: list[Any], scrubbed: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(value: Any) -> str:
        raise RecursionError("deep")

    monkeypatch.setattr(json, "dumps", _boom)
    with pytest.raises(RecursionError):
        _record(events, actual_output={"a": 1})
    assert events == [("attr", "example.success", True)]


def test_scrub_failure_propagates_before_any_output_write(
    events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(value: Any) -> Any:
        raise RuntimeError("scrub failed")

    monkeypatch.setattr(tracing, "_scrub_for_export", _boom)
    with pytest.raises(RuntimeError, match="scrub failed"):
        _record(events, actual_output="x", metrics={"a": 1})
    assert events == [("attr", "example.success", True)]


@pytest.mark.parametrize(
    ("length", "expected_length"),
    [(499, 499), (500, 500), (501, 503), (900, 503)],
)
def test_output_is_truncated_just_above_five_hundred_characters(
    events: list[Any], scrubbed: list[Any], length: int, expected_length: int
) -> None:
    _record(events, actual_output="a" * length)
    written = events[1][2]
    assert len(written) == expected_length
    if length > 500:
        assert written == "a" * 500 + "..."
    else:
        assert written == "a" * length


def test_truncation_applies_to_serialised_text_and_to_fallback_text(
    events: list[Any], scrubbed: list[Any]
) -> None:
    _record(events, actual_output=["b" * 600])
    assert events[1][2] == json.dumps(["b" * 600])[:500] + "..."
    del events[:]

    class _Long:
        def __str__(self) -> str:
            return "c" * 600

    _record(events, actual_output=_Long())
    assert events[1][2] == ("<" + "c" * 600 + ">")[:500] + "..."


@pytest.mark.parametrize("empty", [None, {}, []])
def test_empty_metrics_are_skipped(events: list[Any], empty: Any) -> None:
    _record(events, metrics=empty)
    assert events == [("attr", "example.success", True)]


def test_metrics_only_numeric_values_are_recorded_in_order(events: list[Any]) -> None:
    _record(
        events,
        metrics={
            "b": 2,
            "a": 1.5,
            "flag": True,
            "text": "3",
            "none": None,
            "lst": [1],
            "nan": float("nan"),
            7: 9,
        },
    )
    written = [e for e in events if e[1].startswith("example.metric.")]
    assert [e[1] for e in written] == [
        "example.metric.b",
        "example.metric.a",
        "example.metric.flag",
        "example.metric.nan",
        "example.metric.7",
    ]
    assert written[2][2] is True
    assert written[0][2] == 2


def test_numeric_subclass_values_are_recorded_by_identity(events: list[Any]) -> None:
    class _Float(float):
        pass

    value = _Float(0.25)
    _record(events, metrics={"m": value})
    assert events[1][2] is value


def test_metrics_truth_and_items_are_read_once_each(events: list[Any]) -> None:
    counts = {"bool": 0, "items": 0}

    class _Metrics(dict[str, float]):
        def __bool__(self) -> bool:
            counts["bool"] += 1
            return True

        def items(self) -> Any:
            counts["items"] += 1
            return super().items()

    _record(events, metrics=_Metrics(a=1.0))
    assert counts == {"bool": 1, "items": 1}


def test_failing_metrics_items_propagates_after_output(
    events: list[Any], scrubbed: list[Any]
) -> None:
    class _Metrics(dict[str, float]):
        def items(self) -> Any:
            raise RuntimeError("items failed")

    with pytest.raises(RuntimeError, match="items failed"):
        _record(events, actual_output="o", metrics=_Metrics(a=1.0), error="e")
    assert events == [
        ("attr", "example.success", True),
        ("attr", "example.actual_output", "o"),
    ]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, 0), (0.0, 0.0), (1.5, 1500.0), (-2, -2000), (True, 1000)],
)
def test_execution_time_is_scaled_and_zero_is_recorded(
    events: list[Any], seconds: Any, expected: Any
) -> None:
    _record(events, execution_time=seconds)
    assert events[1] == ("attr", "example.execution_time_ms", expected)


def test_execution_time_none_is_skipped_and_multiplied_once(events: list[Any]) -> None:
    _record(events, execution_time=None)
    assert events == [("attr", "example.success", True)]
    calls: list[Any] = []

    class _Seconds:
        def __mul__(self, other: Any) -> str:
            calls.append(other)
            return "scaled"

    del events[:]
    _record(events, execution_time=_Seconds())
    assert calls == [1000]
    assert events[1] == ("attr", "example.execution_time_ms", "scaled")


def test_error_is_delegated_only_when_truthy(events: list[Any]) -> None:
    falsy_values: list[Any] = [None, "", 0, []]
    for falsy in falsy_values:
        del events[:]
        _record(events, error=falsy)
        assert events == [("attr", "example.success", True)]
    del events[:]
    _record(events, error="failure")
    assert events[-1] == ("error_status", True, "failure", "example.error")


def test_error_truth_is_evaluated_once_and_object_is_passed_unchanged(
    events: list[Any],
) -> None:
    counts = {"bool": 0}

    class _Err(str):
        def __bool__(self) -> bool:
            counts["bool"] += 1
            return counts["bool"] == 1

    err = _Err("x")
    _record(events, error=err)
    assert counts["bool"] == 1
    assert events[-1][2] is err


def test_false_truth_error_string_subclass_is_skipped(events: list[Any]) -> None:
    class _Err(str):
        def __bool__(self) -> bool:
            return False

    _record(events, error=_Err("nonempty"))
    assert events == [("attr", "example.success", True)]


def test_set_attribute_failure_stops_later_steps(
    events: list[Any], scrubbed: list[Any]
) -> None:
    span = _Span(events, fail_on="example.actual_output")
    with pytest.raises(RuntimeError, match="cannot set example.actual_output"):
        tracing.record_example_result(
            span, True, actual_output="o", metrics={"a": 1}, error="e"
        )
    assert events == [
        ("attr", "example.success", True),
        ("attr", "example.actual_output", "o"),
    ]


def test_positional_arguments_map_in_declared_order(
    events: list[Any], scrubbed: list[Any]
) -> None:
    tracing.record_example_result(_Span(events), False, "o", {"m": 1}, "e", 0.5)
    assert events == [
        ("attr", "example.success", False),
        ("attr", "example.actual_output", "o"),
        ("attr", "example.metric.m", 1),
        ("attr", "example.execution_time_ms", 500.0),
        ("error_status", True, "e", "example.error"),
    ]


def test_real_scrubbers_redact_email_in_output_and_fallback(events: list[Any]) -> None:
    _record(events, actual_output="mail me at a.b@example.com now")
    assert events[1][2] == "mail me at ***EMAIL*** now"
    del events[:]
    _record(events, actual_output={"k": "a.b@example.com"})
    assert events[1][2] == '{"k": "***EMAIL***"}'
    del events[:]

    class _Opaque:
        def __str__(self) -> str:
            return "x@y.org"

    _record(events, actual_output=_Opaque())
    assert events[1][2] == "***EMAIL***"


def test_span_attribute_lookup_happens_after_the_output_is_prepared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []

    class _LazySpan:
        def __getattr__(self, name: str) -> Any:
            log.append(f"getattr {name}")
            return lambda *args, **kwargs: log.append(f"call {name} {args[0]}")

    def _scrub(value: Any) -> Any:
        log.append("scrub")
        return value

    monkeypatch.setattr(tracing, "_scrub_for_export", _scrub)
    tracing.record_example_result(_LazySpan(), True, actual_output="o")
    assert log == [
        "getattr set_attribute",
        "call set_attribute example.success",
        "scrub",
        "getattr set_attribute",
        "call set_attribute example.actual_output",
    ]
