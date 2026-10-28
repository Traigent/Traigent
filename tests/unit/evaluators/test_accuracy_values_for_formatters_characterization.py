"""Characterization of MetricsTracker._accuracy_values_for_formatters.

Expectations changed for #2767 (no invented accuracy): a row with no
measurement is ABSENT from the average, never defaulted to 1.0 or 0.0 from
success. The old ``user_supplied_accuracy`` argument is gone; provenance is
per row (``accuracy_supplied`` for a user metric, ``accuracy_builtin`` for
built-in exact match). Scenarios are unchanged; see the per-test notes.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.evaluators.metrics_tracker import ExampleMetrics, MetricsTracker


def _row(
    raw: Any = None,
    *,
    has_key: bool = True,
    success: bool = True,
    eligible: bool = True,
    unavailable: bool = False,
) -> ExampleMetrics:
    custom: dict[str, Any] = {"accuracy": raw} if has_key else {}
    return ExampleMetrics(
        success=success,
        custom_metrics=custom,
        accuracy_eligible=eligible,
        accuracy_unavailable=unavailable,
    )


def _values(rows: list[Any], *, user: bool) -> list[float]:
    tracker = MetricsTracker()
    for row in rows:
        if isinstance(row, ExampleMetrics):
            row.accuracy_supplied = user
            row.accuracy_builtin = not user
    tracker.example_metrics = rows
    result: list[float] = tracker._accuracy_values_for_formatters()
    return result


def test_no_rows_gives_empty_list() -> None:
    assert _values([], user=True) == []
    assert _values([], user=False) == []


def test_user_supplied_counts_rows_with_a_value_and_omits_the_rest() -> None:
    # Old: rows with no value (None / missing key) were zero-filled.
    # New: they are unmeasured and absent.
    rows = [
        _row(0.25),
        _row(None),
        _row(has_key=False),
        _row(1, success=False),
        _row(0.5, eligible=False),
        _row(None, eligible=False, unavailable=True),
        _row("0.75"),
        _row(True),
    ]
    result = _values(rows, user=True)
    assert result == [0.25, 1.0, 0.5, 0.75, 1.0]
    assert all(type(v) is float for v in result)


def test_user_supplied_bad_value_raises() -> None:
    with pytest.raises(ValueError):
        _values([_row("abc")], user=True)
    with pytest.raises(TypeError):
        _values([_row([1])], user=True)


def test_builtin_ineligible_rows_keep_a_stored_value_but_never_a_zero() -> None:
    # Old: ineligible rows were skipped even when they carried a value.
    # New: a stored measured value is kept (resumed rows); a row with no value
    # stays absent, and an ineligible failure is not a miss.
    rows = [
        _row(1.0, eligible=False),
        _row(None, eligible=False, success=False),
        _row(0.5),
    ]
    assert _values(rows, user=False) == [1.0, 0.5]


def test_builtin_skips_unmeasured_rows_without_value() -> None:
    rows = [
        _row(None, unavailable=True),
        _row(has_key=False, unavailable=True, success=False),
        _row(None),
    ]
    # Old: the plain successful row without a value was invented as 1.0.
    # New: it is absent; the eligible failed row is a miss (0.0) even when
    # flagged unavailable.
    assert _values(rows, user=False) == [0.0]


def test_builtin_unavailable_flag_ignored_when_value_present() -> None:
    rows = [
        _row(0.4, unavailable=True),
        _row(0.9, unavailable=True, success=False),
    ]
    assert _values(rows, user=False) == [0.4, 0.0]


def test_builtin_success_uses_value_or_is_absent_failure_always_zero() -> None:
    rows = [
        _row(0.3),
        _row(None),
        _row(0.0),
        _row(1.0, success=False),
        _row(None, success=False),
        _row(0.7, success=False),
        _row("abc", success=False),
    ]
    result = _values(rows, user=False)
    assert result == [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert all(type(v) is float for v in result)


def test_builtin_success_with_bad_value_raises_but_failure_does_not() -> None:
    with pytest.raises(ValueError):
        _values([_row("abc")], user=False)
    assert _values([_row("abc", success=False)], user=False) == [0.0]


def test_builtin_mixed_order_is_preserved() -> None:
    rows = [
        _row(0.1),
        _row(0.2, eligible=False),
        _row(None, success=False),
        _row(None, unavailable=True),
        _row(None),
        _row(0.6),
    ]
    assert _values(rows, user=False) == [0.1, 0.2, 0.0, 0.6]


class _TraceRow:
    def __init__(
        self,
        events: list[str],
        name: str,
        *,
        raw: Any,
        success: bool = True,
        eligible: bool = True,
        unavailable: bool = False,
        user: bool = False,
    ) -> None:
        self._user = user
        self._events = events
        self._name = name
        self._raw = raw
        self._success = success
        self._eligible = eligible
        self._unavailable = unavailable

    @property
    def custom_metrics(self) -> dict[str, Any]:
        self._events.append(f"{self._name}.custom_metrics")
        return {"accuracy": self._raw}

    @property
    def accuracy_supplied(self) -> bool:
        self._events.append(f"{self._name}.supplied")
        return self._user

    @property
    def accuracy_builtin(self) -> bool:
        self._events.append(f"{self._name}.builtin")
        return not self._user

    @property
    def accuracy_eligible(self) -> bool:
        self._events.append(f"{self._name}.eligible")
        return self._eligible

    @property
    def accuracy_unavailable(self) -> bool:
        self._events.append(f"{self._name}.unavailable")
        return self._unavailable

    @property
    def success(self) -> bool:
        self._events.append(f"{self._name}.success")
        return self._success


def test_read_trace_user_supplied_reads_only_custom_metrics() -> None:
    events: list[str] = []
    rows = [
        _TraceRow(events, "r", raw=None, user=True),
        _TraceRow(events, "q", raw=1, user=True),
    ]
    # Old: the row without a value was zero-filled; new: it is absent.
    assert _values(rows, user=True) == [1.0]
    assert events == [
        "r.custom_metrics",
        "r.supplied",
        "r.builtin",
        "q.custom_metrics",
        "q.supplied",
    ]


def test_read_trace_builtin_order_eligible_unavailable_success() -> None:
    events: list[str] = []
    rows = [
        _TraceRow(events, "a", raw=None),
        _TraceRow(events, "b", raw=0.5),
        _TraceRow(events, "c", raw=None, unavailable=True),
        _TraceRow(events, "d", raw=1.0, eligible=False),
        _TraceRow(events, "e", raw=None, success=False),
    ]
    assert _values(rows, user=False) == [0.5, 1.0, 0.0]
    # The unavailable flag is no longer read: an unmeasured row is simply one
    # with no value, and a failed eligible attempt is a miss.
    assert events == [
        "a.custom_metrics",
        "a.supplied",
        "a.builtin",
        "a.eligible",
        "a.success",
        "b.custom_metrics",
        "b.supplied",
        "b.builtin",
        "b.eligible",
        "b.success",
        "c.custom_metrics",
        "c.supplied",
        "c.builtin",
        "c.eligible",
        "c.success",
        "d.custom_metrics",
        "d.supplied",
        "d.builtin",
        "d.eligible",
        "e.custom_metrics",
        "e.supplied",
        "e.builtin",
        "e.eligible",
        "e.success",
    ]


def test_raising_custom_metrics_accessor_propagates() -> None:
    class _Bad:
        accuracy_supplied = False
        accuracy_builtin = True

        @property
        def custom_metrics(self) -> dict[str, Any]:
            raise RuntimeError("cm")

    with pytest.raises(RuntimeError, match="cm"):
        _values([_Bad()], user=True)
    with pytest.raises(RuntimeError, match="cm"):
        _values([_Bad()], user=False)


def test_raising_eligibility_accessor_propagates() -> None:
    class _Bad:
        custom_metrics: dict[str, Any] = {}
        accuracy_supplied = False
        accuracy_builtin = True

        @property
        def accuracy_eligible(self) -> bool:
            raise RuntimeError("el")

    with pytest.raises(RuntimeError, match="el"):
        _values([_Bad()], user=False)
    # the user-supplied path never reads eligibility

    class _BadUser(_Bad):
        accuracy_supplied = True
        accuracy_builtin = False

    # Old: [0.0] (zero-filled); new: no value means absent.
    assert _values([_BadUser()], user=True) == []


def test_raising_success_accessor_propagates() -> None:
    class _Bad:
        custom_metrics: dict[str, Any] = {"accuracy": 1.0}
        accuracy_supplied = False
        accuracy_builtin = True
        accuracy_eligible = True
        accuracy_unavailable = False

        @property
        def success(self) -> bool:
            raise RuntimeError("su")

    with pytest.raises(RuntimeError, match="su"):
        _values([_Bad()], user=False)

    class _BadUser(_Bad):
        accuracy_supplied = True
        accuracy_builtin = False

    assert _values([_BadUser()], user=True) == [1.0]
