"""Uneven-step warnings point at the caller, and the IntRange presets stay quiet.

Regression tests for the review of #2890 / #2891: the warning was attributed to
the dataclass-generated ``<string>`` ``__init__`` (stacklevel too low), it was
emitted before the log/step validity errors, and the ``max_tokens`` preset
silenced it by mutating the global warning filters.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from traigent.api.parameter_ranges import IntRange, Range


def _uneven_warnings(records: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in records if issubclass(w.category, UserWarning)
            and "does not evenly divide span" in str(w.message)]


def test_int_range_uneven_step_warning_points_at_caller() -> None:
    with pytest.warns(UserWarning, match="does not evenly divide span") as record:
        IntRange(0, 10, step=3)  # this line is the caller
    assert record[0].filename == __file__


def test_range_uneven_step_warning_points_at_caller() -> None:
    with pytest.warns(UserWarning, match="does not evenly divide span") as record:
        Range(0.0, 1.0, step=0.3)  # this line is the caller
    assert record[0].filename == __file__


def test_int_range_log_with_step_raises_without_warning() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Optuna limitation"):
            IntRange(1, 10, step=4, log=True)  # 9 % 4 != 0: would warn
    assert _uneven_warnings(record) == []


def test_int_range_max_tokens_preset_emits_no_warning() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for task in ("short", "medium", "long"):
            IntRange.max_tokens(task=task)  # type: ignore[arg-type]
    assert _uneven_warnings(record) == []


def test_user_built_uneven_range_still_warns_under_always_filter() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        IntRange(50, 256, step=64)
    assert len(_uneven_warnings(record)) == 1


def test_private_preset_flag_is_not_a_dataclass_field() -> None:
    names = {f.name for f in dataclasses.fields(IntRange)}
    assert not any("suppress" in name for name in names)
