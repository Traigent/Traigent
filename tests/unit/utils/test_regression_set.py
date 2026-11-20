"""Unit tests for the regression set file (merge, atomic write, load)."""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from traigent.evaluators.base import (
    Dataset,
    EvaluationExample,
    _example_correlation_key,
)
from traigent.utils.file_lock import (
    FileLockUnsupportedError,
    exclusive_lock,
    lock_path_for,
)
from traigent.utils.regression_set import (
    IdConflictError,
    RegressionSetError,
    RunEvidence,
    StaleDigestError,
    config_digest_for,
    content_digest,
    load_regression_dataset,
    merge_failures,
    read_regression_rows,
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


def _evidence(
    ds: Dataset,
    run: str = "r1",
    imported: str = "2026-01-01",
    mode: str = "best_config",
    threshold: float = 0.5,
) -> RunEvidence:
    """Evidence for run ``run`` (its name and its id are the same string)."""
    basis = {
        str(_example_correlation_key(ex, i)): ("score", "accuracy")
        for i, ex in enumerate(ds.examples)
    }
    return RunEvidence(
        run_id=run,
        run_name=run,
        imported=imported,
        mode=mode,
        threshold=threshold,
        config_digest=config_digest_for(mode, {"m": 1}),
        basis=basis,
    )


def _merge(d: Path, ds: Dataset, run: str = "r1", imported: str = "2026-01-01", **kw):
    mode = kw.pop("mode", "best_config")
    threshold = kw.pop("threshold", 0.5)
    return merge_failures(
        d, ds, evidence=_evidence(ds, run, imported, mode, threshold), **kw
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


def test_valid_file_with_no_matching_label_returns_an_empty_dataset(
    tmp_path: Path,
) -> None:
    _merge(tmp_path, _ds(("q0", "a0")))
    loaded = load_regression_dataset(tmp_path, labels=("reliable",))
    assert isinstance(loaded, Dataset)
    assert len(loaded) == 0


def test_missing_corrupt_and_empty_files_still_raise_not_empty(
    tmp_path: Path,
) -> None:
    with pytest.raises(RegressionSetError, match="does not exist"):
        load_regression_dataset(tmp_path / "nowhere", labels=("reliable",))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "regression.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(RegressionSetError, match="not valid"):
        load_regression_dataset(bad, labels=("reliable",))
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "regression.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RegressionSetError, match="holds no rows"):
        load_regression_dataset(empty, labels=("reliable",))


def test_load_reads_the_file_once_and_survives_a_replace_after_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is replaced right after its only read: the rows still match."""
    from traigent.utils import regression_set

    _merge(tmp_path, _ds(("q0", "a0"), ("q1", "a1")))
    path = tmp_path / "regression.jsonl"
    other = tmp_path / "other"
    _merge(other, _ds(("OTHER0", "b0"), ("OTHER1", "b1")))
    replacement = (other / "regression.jsonl").read_bytes()

    real_read = regression_set._read_numbered_rows
    reads: list[Path] = []

    def read_then_replace(p: Path) -> Any:
        result = real_read(p)
        reads.append(p)
        path.write_bytes(replacement)  # the writer lands between the two reads
        return result

    def no_second_read(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the file must not be opened a second time")

    monkeypatch.setattr(regression_set, "_read_numbered_rows", read_then_replace)
    monkeypatch.setattr(Dataset, "from_jsonl", no_second_read)
    loaded = load_regression_dataset(tmp_path)
    assert len(reads) == 1
    assert [e.input_data for e in loaded.examples] == [{"q": "q0"}, {"q": "q1"}]
    assert [e.expected_output for e in loaded.examples] == ["a0", "a1"]


def test_load_rejects_duplicate_ids_like_the_file_loader(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0")))
    path = tmp_path / "regression.jsonl"
    line = path.read_text(encoding="utf-8")
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(RegressionSetError, match="repeats"):
        load_regression_dataset(tmp_path)


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
        with pytest.raises(RegressionSetError, match="Timed out") as err:
            _merge(tmp_path, _ds(("q9", "a9"), start=9), lock_timeout=0.2)
        message = str(err.value)
        assert "another command is still running" in message
        assert "retry" in message
        # deleting a lock another process holds would let a second writer in
        assert "remove the lock" not in message
        assert "Do not delete the lock file" in message
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
    # the lock is released afterwards, so the same add now succeeds
    assert _merge(tmp_path, _ds(("q9", "a9"), start=9)).rows_added == 1


@pytest.fixture
def lock_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private system temp dir, so lock files can be inspected."""
    lock_root = tmp_path / "systmp"
    lock_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(lock_root))
    return lock_root


def test_lock_file_is_not_in_the_repository_directory(
    tmp_path: Path, lock_tmp: Path
) -> None:
    data = tmp_path / "repo"
    _merge(data, _ds(("q0", "a0")))
    assert sorted(p.name for p in data.iterdir()) == ["regression.jsonl"]
    assert len((data / "regression.jsonl").read_text().splitlines()) == 1
    locks = list(lock_tmp.rglob("*.lock"))
    assert len(locks) == 1
    assert locks[0].parent.parent == lock_tmp
    assert oct(locks[0].parent.stat().st_mode & 0o777) == "0o700"


def test_lock_path_is_keyed_by_the_data_files_real_path(
    tmp_path: Path, lock_tmp: Path
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "a")
    one = lock_path_for(tmp_path / "a" / "regression.jsonl")
    assert lock_path_for(link / "regression.jsonl") == one  # same file, same lock
    assert lock_path_for(tmp_path / "b" / "regression.jsonl") != one


def test_a_lock_directory_owned_by_someone_else_is_refused(
    tmp_path: Path, lock_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path_for(tmp_path / "regression.jsonl")  # creates the directory
    real_getuid = os.getuid
    monkeypatch.setattr(os, "getuid", lambda: real_getuid() + 1)
    # a different "user" gets a different directory name, so plant a clash
    (lock_tmp / f"traigent-locks-{real_getuid() + 1}").mkdir()
    with pytest.raises(FileLockUnsupportedError, match="another user"):
        lock_path_for(tmp_path / "regression.jsonl")


@pytest.mark.parametrize("absent", ["", None, True])
def test_absent_ids_fall_back_to_position_like_the_dataset_loader(
    tmp_path: Path, absent: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    ds = Dataset(
        examples=[
            EvaluationExample(
                input_data={"q": f"q{i}"},
                expected_output=f"a{i}",
                metadata={"example_id": absent},
            )
            for i in range(2)
        ]
    )
    _merge(tmp_path, ds)
    reloaded = Dataset.from_jsonl(tmp_path / "regression.jsonl")
    assert [e.metadata["example_id"] for e in reloaded.examples] == [
        "example_0",
        "example_1",
    ]


# ------------------------------------------------------------ stale digest


def _hand_edit_first_row(path: Path) -> None:
    """Change a stored row's content but keep its content_digest."""
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["output"] = "zzz-sentinel-edit"
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_add_refuses_row_edited_by_hand_and_writes_nothing(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0"), ("q1", "a1")))
    path = tmp_path / "regression.jsonl"
    _hand_edit_first_row(path)
    before = path.read_bytes()
    with pytest.raises(StaleDigestError) as err:
        _merge(tmp_path, _ds(("q0", "a0")), "r2")
    message = str(err.value)
    assert "'0'" in message and "line 1" in message
    assert "zzz-sentinel-edit" not in message  # names the row, never the content
    # each resolution is described as what it does to the row
    assert "DISCARDS the edit" in message
    assert "overwriting" in message
    assert "earlier provenance entries are kept" in message
    assert "accept the edited content" not in message
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_replace_accepts_edited_row_that_is_failing_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    _merge(tmp_path, _ds(("q0", "a0"), ("q1", "a1")))
    path = tmp_path / "regression.jsonl"
    _hand_edit_first_row(path)
    report = _merge(tmp_path, _ds(("q0", "a0")), "r2", on_id_conflict="replace")
    assert report.rows_replaced == 1
    first = json.loads(path.read_text().splitlines()[0])
    assert first["output"] == "a0"
    load_regression_dataset(tmp_path)  # digests consistent again
    # the earlier entry is kept, marked with the digest of the content it described
    entries = first["metadata"]["regression"]["provenance"]
    assert [e["run"] for e in entries] == ["r1", "r2"]
    assert entries[0]["superseded_content_digest"] == content_digest({"q": "q0"}, "a0")
    assert "superseded_content_digest" not in entries[1]


def test_replace_of_a_changed_row_keeps_its_old_provenance_marked(
    tmp_path: Path,
) -> None:
    _merge(tmp_path, _ds(("q0", "a0")), "r1")
    old_digest = content_digest({"q": "q0"}, "a0")
    report = _merge(tmp_path, _ds(("q0", "CHANGED")), "r2", on_id_conflict="replace")
    assert report.rows_replaced == 1
    row = json.loads((tmp_path / "regression.jsonl").read_text().splitlines()[0])
    entries = row["metadata"]["regression"]["provenance"]
    assert [e["run"] for e in entries] == ["r1", "r2"]
    assert entries[0]["superseded_content_digest"] == old_digest
    assert row["metadata"]["content_digest"] == content_digest({"q": "q0"}, "CHANGED")


def test_replace_does_not_waive_edited_rows_it_does_not_replace(
    tmp_path: Path,
) -> None:
    _merge(tmp_path, _ds(("q0", "a0"), ("q1", "a1")))
    path = tmp_path / "regression.jsonl"
    _hand_edit_first_row(path)
    before = path.read_bytes()
    with pytest.raises(StaleDigestError):
        _merge(tmp_path, _ds(("q1", "a1"), start=1), on_id_conflict="replace")
    assert path.read_bytes() == before


def test_load_refuses_row_edited_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    _merge(tmp_path, _ds(("q0", "a0")))
    _hand_edit_first_row(tmp_path / "regression.jsonl")
    with pytest.raises(StaleDigestError, match=r"line 1"):
        load_regression_dataset(tmp_path)


# ------------------------------------------------- unicode line separators

LINE_BREAKS = "a\u2028b\u2029c\u0085d\re\x0bf\x0cg\x1ch\x1di\x1ej\nk"


def test_unicode_line_breaks_round_trip_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    ds = Dataset(
        examples=[
            EvaluationExample(
                input_data={"q": LINE_BREAKS},
                expected_output=LINE_BREAKS,
                metadata={"example_id": "x"},
            ),
            EvaluationExample(
                input_data={"q": "plain"},
                expected_output="plain",
                metadata={"example_id": "y"},
            ),
        ]
    )
    _merge(tmp_path, ds)
    # a second add reads the file back under the lock
    assert _merge(tmp_path, ds, "r2").provenance_added == 2
    path = tmp_path / "regression.jsonl"
    raw = path.read_text(encoding="utf-8")
    # one record per "\n"-terminated line, and no raw Unicode line break left
    assert raw.count("\n") == 2
    for char in ("\u2028", "\u2029", "\u0085", "\r", "\x0b", "\x0c"):
        assert char not in raw
    assert len(raw.split("\n")[0].splitlines()) == 1
    for rows in (
        read_regression_rows(path),
        [
            {"input": e.input_data, "output": e.expected_output}
            for e in load_regression_dataset(tmp_path).examples
        ],
        [
            {"input": e.input_data, "output": e.expected_output}
            for e in Dataset.from_jsonl(path).examples
        ],
    ):
        assert rows[0]["input"] == {"q": LINE_BREAKS}
        assert rows[0]["output"] == LINE_BREAKS
        assert len(rows) == 2


def test_reader_splits_on_newline_only_for_a_raw_separator_in_a_string(
    tmp_path: Path,
) -> None:
    """A file written by another tool may hold a raw U+2028 inside a string."""
    row = {
        "input": {"q": "a\u2028b"},
        "output": "o\u0085p",
        "metadata": {
            "example_id": "x",
            "content_digest": content_digest({"q": "a\u2028b"}, "o\u0085p"),
            "regression": {
                "schema_version": 1,
                "label": "seen_once",
                "provenance": [],
            },
        },
    }
    path = tmp_path / "regression.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_regression_rows(path)[0]["input"] == {"q": "a\u2028b"}


# ------------------------------------------------------- schema and identity


def _written_row(tmp_path: Path) -> dict[str, Any]:
    _merge(tmp_path, _ds(("q0", "a0")))
    return json.loads((tmp_path / "regression.jsonl").read_text().splitlines()[0])


def test_rows_carry_schema_version_one(tmp_path: Path) -> None:
    row = _written_row(tmp_path)
    assert row["metadata"]["regression"]["schema_version"] == 1


@pytest.mark.parametrize(
    ("version", "message"),
    [
        (2, "newer than this SDK"),
        (None, "schema_version.* missing"),
        ("1", "not a version"),
    ],
)
def test_reader_refuses_a_row_with_an_unreadable_schema_version(
    tmp_path: Path, version: Any, message: str
) -> None:
    row = _written_row(tmp_path)
    regression = row["metadata"]["regression"]
    if version is None:
        del regression["schema_version"]
    else:
        regression["schema_version"] = version
    path = tmp_path / "regression.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RegressionSetError, match=message) as err:
        read_regression_rows(path)
    assert "line 1" in str(err.value)
    with pytest.raises(RegressionSetError):
        load_regression_dataset(tmp_path)
    with pytest.raises(RegressionSetError):  # and a merge refuses it, untouched
        _merge(tmp_path, _ds(("q9", "a9"), start=9))


def test_same_run_on_a_later_date_or_name_is_the_same_evidence(tmp_path: Path) -> None:
    _merge(tmp_path, _ds(("q0", "a0")), "r1", imported="2026-01-01")
    before = (tmp_path / "regression.jsonl").read_bytes()
    ds = _ds(("q0", "a0"))
    renamed = RunEvidence(
        run_id="r1",
        run_name="renamed",
        imported="2027-05-05",
        mode="best_config",
        threshold=0.5,
        config_digest=config_digest_for("best_config", {"m": 1}),
        basis={"0": ("score", "accuracy")},
    )
    report = merge_failures(tmp_path, ds, evidence=renamed)
    assert report.provenance_added == 0
    assert (tmp_path / "regression.jsonl").read_bytes() == before


def test_same_run_with_another_mode_or_threshold_is_new_evidence(
    tmp_path: Path,
) -> None:
    _merge(tmp_path, _ds(("q0", "a0")), "r1")
    assert (
        _merge(tmp_path, _ds(("q0", "a0")), "r1", threshold=0.9).provenance_added == 1
    )
    assert (
        _merge(tmp_path, _ds(("q0", "a0")), "r1", mode="all_configs").provenance_added
        == 1
    )


def test_config_digest_is_the_sdk_config_hash_and_null_for_all_configs() -> None:
    from traigent.utils.hashing import generate_config_hash

    cfg = {"model": "m", "t": 0.2}
    assert config_digest_for("best_config", cfg) == generate_config_hash(cfg)
    assert config_digest_for("all_configs", cfg) is None
