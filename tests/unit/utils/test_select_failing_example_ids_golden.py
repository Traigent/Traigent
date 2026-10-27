"""Golden characterization for ``select_failing_example_ids`` row handling."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.utils.outcome_matrix import select_failing_example_ids

_FAIL = {"success": False}
_PASS = {"score": 1.0}


def _m(rows: list[Any], trials: tuple[str, ...] = ("a", "b")) -> dict[str, Any]:
    return {
        "trials": [
            {"index": i, "trial_id": t, "config": {}} for i, t in enumerate(trials)
        ],
        "examples": rows,
    }


def _row(eid: Any, **cells: Any) -> dict[str, Any]:
    return {"example_id": eid, "cells": cells}


def test_all_configs_requires_every_column_failing() -> None:
    rows = [
        _row("x", a=_FAIL, b=_FAIL),
        _row("y", a=_FAIL, b=_PASS),
        _row("z", a=_FAIL),  # missing cell -> unknown -> excluded
        _row("w", a={"error": "boom"}, b={"score": 0.1}),
    ]
    assert select_failing_example_ids(_m(rows)) == ["x", "w"]


def test_duplicate_ids_reported_once_in_row_order() -> None:
    rows = [
        _row("x", a=_FAIL, b=_FAIL),
        _row("q", a=_FAIL, b=_FAIL),
        _row("x", a=_FAIL, b=_FAIL),
    ]
    assert select_failing_example_ids(_m(rows)) == ["x", "q"]


def test_cell_keys_are_stringified() -> None:
    rows = [{"example_id": "x", "cells": {1: _FAIL}}]
    assert select_failing_example_ids(_m(rows, ("1",))) == ["x"]


def test_threshold_boundary_equal_passes() -> None:
    rows = [_row("lo", a={"score": 0.49}), _row("eq", a={"score": 0.5})]
    assert select_failing_example_ids(_m(rows, ("a",))) == ["lo"]
    assert select_failing_example_ids(_m(rows, ("a",)), threshold=0.5) == ["lo"]


def test_row_validation_is_sequential_and_checks_id_before_cells() -> None:
    bad_id = {"example_id": 5, "cells": None}
    with pytest.raises(
        ValueError, match=r"examples\[1\]\.example_id must be a string, got 5"
    ):
        select_failing_example_ids(_m([_row("x", a=_FAIL, b=_FAIL), bad_id]))
    with pytest.raises(ValueError, match=r"examples\[0\] must be an object"):
        select_failing_example_ids(_m(["nope", bad_id]))
    with pytest.raises(ValueError, match=r"examples\[0\]\.cells must be an object"):
        select_failing_example_ids(_m([{"example_id": "x", "cells": []}]))
    with pytest.raises(ValueError, match=r"examples\[0\]\.cells must be an object"):
        select_failing_example_ids(_m([{"example_id": "x"}]))


def test_malformed_row_after_no_signal_still_reports_row_error_first() -> None:
    rows = [_row("x"), {"example_id": 3, "cells": {}}]
    with pytest.raises(ValueError, match="example_id must be a string"):
        select_failing_example_ids(_m(rows))


def test_no_signal_raises_after_scanning_everything() -> None:
    rows = [_row("x", a={"success": True}), _row("y", a={"score": float("nan")})]
    with pytest.raises(ValueError, match="No cell in the selected columns"):
        select_failing_example_ids(_m(rows, ("a",)))


def test_signal_with_no_failures_returns_empty() -> None:
    assert select_failing_example_ids(_m([_row("x", a=_PASS)], ("a",))) == []


def test_header_validation_order() -> None:
    with pytest.raises(ValueError, match="threshold"):
        select_failing_example_ids({}, threshold=True, mode="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode must be one of"):
        select_failing_example_ids({}, mode="bad")
    with pytest.raises(ValueError, match="expected a JSON object"):
        select_failing_example_ids([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'examples' must be a list"):
        select_failing_example_ids({"examples": None})
    with pytest.raises(ValueError, match="no per-example detail"):
        select_failing_example_ids({"examples": []})
    with pytest.raises(ValueError, match="'trials' must be a list"):
        select_failing_example_ids({"examples": [_row("x")]})
    with pytest.raises(ValueError, match="no trial columns"):
        select_failing_example_ids({"examples": [_row("x")], "trials": []})
