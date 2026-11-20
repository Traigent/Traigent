"""Unit tests for the regression set file (merge, atomic write, load)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.regression_set import (
    IdConflictError,
    RegressionSetError,
    content_digest,
    load_regression_dataset,
    merge_failures,
)


def _ds(*pairs: tuple[str, str], start: int = 0) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"q": q},
                expected_output=a,
                metadata={"example_id": start + i},
            )
            for i, (q, a) in enumerate(pairs)
        ]
    )


def _merge(d: Path, ds: Dataset, run: str = "r1", **kw):
    return merge_failures(
        d, ds, run=run, date="2026-01-01", config={"m": 1}, metric="accuracy", **kw
    )


def test_same_example_twice_appends_provenance_not_a_row(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0")), "r1")
    report = _merge(tmp_path, _ds(("q0", "a0")), "r2")
    assert (report.rows_added, report.provenance_added) == (0, 1)
    rows = [
        json.loads(x) for x in (tmp_path / "regression.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert [p["run"] for p in rows[0]["metadata"]["regression"]["provenance"]] == [
        "r1",
        "r2",
    ]


def test_identical_provenance_is_not_repeated(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0")))
    report = _merge(tmp_path, _ds(("q0", "a0")))
    assert report.provenance_added == 0
    assert report.rows_added == 0


def test_conflict_raises_and_names_ids_not_content(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("secret-q", "a0")))
    before = (tmp_path / "regression.jsonl").read_bytes()
    ds = Dataset(
        examples=[
            EvaluationExample(
                input_data={"q": "other-secret"},
                expected_output="x",
                metadata={"example_id": "0"},
            )
        ]
    )
    with pytest.raises(IdConflictError) as info:
        _merge(tmp_path, ds, "r2")
    assert "other-secret" not in str(info.value) and "secret-q" not in str(info.value)
    assert info.value.example_ids == ["0"]
    assert (tmp_path / "regression.jsonl").read_bytes() == before


def test_failed_write_leaves_original_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _merge(tmp_path, _ds(("q0", "a0")))
    path = tmp_path / "regression.jsonl"
    before = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        _merge(tmp_path, _ds(("q1", "a1"), start=5), "r2")
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_digest_is_stable_and_content_sensitive() -> None:
    assert content_digest({"a": 1, "b": 2}, "x") == content_digest(
        {"b": 2, "a": 1}, "x"
    )
    assert content_digest({"a": 1}, "x") != content_digest({"a": 1}, "y")


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RegressionSetError, match="does not exist"):
        load_regression_dataset(tmp_path)


@pytest.mark.parametrize(
    "content",
    ["", "{bad\n", "[1]\n", '{"input": {}, "output": 1}\n'],
)
def test_load_corrupt_file_raises(tmp_path: Path, content: str) -> None:
    (tmp_path / "regression.jsonl").write_text(content, encoding="utf-8")
    with pytest.raises(RegressionSetError):
        load_regression_dataset(tmp_path)


def test_load_with_no_matching_label_raises_not_empty(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0")))
    with pytest.raises(RegressionSetError, match="no rows with label"):
        load_regression_dataset(tmp_path, labels=("reliable",))
