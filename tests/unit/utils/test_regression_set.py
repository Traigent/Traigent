"""Unit tests for the regression set file (merge, atomic write, load)."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.file_lock import exclusive_lock
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


# ------------------------------------------------------------- concurrency


def _slow_add(directory: str, start: int, go: Any, done: Any) -> None:
    """Child process: add three rows, widening the read-to-replace window."""
    import time

    from traigent.utils import regression_set

    real_write = regression_set._atomic_write

    def slow_write(path: Path, rows: Any) -> None:
        time.sleep(0.05)
        real_write(path, rows)

    regression_set._atomic_write = slow_write  # type: ignore[assignment]
    go.wait()
    try:
        _merge(
            Path(directory),
            _ds(*[(f"q{start + i}", f"a{start + i}") for i in range(3)], start=start),
            f"r{start}",
        )
        done.put(None)
    except Exception as exc:  # reported to the parent, never swallowed
        done.put(f"{type(exc).__name__}: {exc}")


@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
)
def test_two_processes_adding_different_failures_keep_both_sets(
    tmp_path: Path,
) -> None:
    ctx = multiprocessing.get_context("fork")
    for round_no in range(20):
        directory = tmp_path / f"round{round_no}"
        go = ctx.Event()
        done = ctx.Queue()
        procs = [
            ctx.Process(target=_slow_add, args=(str(directory), start, go, done))
            for start in (0, 100)
        ]
        for proc in procs:
            proc.start()
        go.set()
        errors = [done.get(timeout=60) for _ in procs]
        for proc in procs:
            proc.join(timeout=60)
        assert errors == [None, None], errors
        stored = {
            json.loads(line)["metadata"]["example_id"]
            for line in (directory / "regression.jsonl").read_text().splitlines()
        }
        assert stored == {"0", "1", "2", "100", "101", "102"}, round_no


def test_lock_timeout_errors_and_writes_nothing(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0")))
    path = tmp_path / "regression.jsonl"
    before = path.read_bytes()
    with exclusive_lock(path):
        with pytest.raises(RegressionSetError, match="Timed out"):
            _merge(tmp_path, _ds(("q9", "a9"), start=9), lock_timeout=0.2)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
    # the lock is released afterwards, so the same add now succeeds
    assert _merge(tmp_path, _ds(("q9", "a9"), start=9)).rows_added == 1


def test_lock_file_is_not_the_data_file(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0")))
    assert (tmp_path / ".regression.jsonl.lock").exists()
    assert len((tmp_path / "regression.jsonl").read_text().splitlines()) == 1
