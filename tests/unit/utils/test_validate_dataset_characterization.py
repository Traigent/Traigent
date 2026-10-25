"""Characterization tests pinning the behaviour of ``Validators.validate_dataset``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.validation import (
    MAX_REPORTED_DATASET_ROW_ERRORS,
    ValidationResult,
    Validators,
)


def _errs(result: ValidationResult) -> list[tuple[str, str]]:
    return [(e.field, e.error_code) for e in result.errors]


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _good(n: int = 1) -> list[str]:
    return [json.dumps({"input": {"q": i}, "expected_output": "x"}) for i in range(n)]


class TestNonPathInputs:
    def test_dataset_instance_is_valid_without_checks(self) -> None:
        result = Validators.validate_dataset(Dataset(examples=[]))
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}

    @pytest.mark.parametrize("bad", [5, None, 1.5, {"a": 1}])
    def test_unsupported_type(self, bad: object) -> None:
        result = Validators.validate_dataset(bad)
        assert _errs(result) == [("dataset", "TYPE_ERROR")]
        assert result.errors[0].message == (
            "Expected dataset path, inline example list, or Dataset object, "
            f"got {type(bad).__name__}"
        )


class TestInlineLists:
    def test_valid_inline_dicts(self) -> None:
        result = Validators.validate_dataset(
            [{"input": {"q": 1}, "output": "a"}, {"input": {"q": 2}}]
        )
        assert result.is_valid
        assert result.errors == []

    def test_valid_inline_evaluation_example(self) -> None:
        ex = EvaluationExample(input_data={"q": 1}, expected_output="a")
        assert Validators.validate_dataset([ex]).is_valid

    def test_invalid_inline_dict_reports_invalid_format(self) -> None:
        result = Validators.validate_dataset([{"nothing": 1}])
        assert _errs(result) == [("dataset", "INVALID_FORMAT")]
        assert result.errors[0].message  # str(exception) is non-empty

    def test_mixed_list_is_type_error(self) -> None:
        result = Validators.validate_dataset([{"input": 1}, "a.jsonl"])
        assert _errs(result) == [("dataset", "TYPE_ERROR")]
        assert result.errors[0].message == (
            "Dataset list entries must all be file paths or inline example objects"
        )

    def test_list_of_ints_is_type_error(self) -> None:
        result = Validators.validate_dataset([1, 2])
        assert _errs(result) == [("dataset", "TYPE_ERROR")]

    def test_empty_list_goes_through_inline_branch(self) -> None:
        with patch("traigent.evaluators.base.load_inline_dataset") as loader:
            result = Validators.validate_dataset([])
        loader.assert_called_once_with([])
        assert result.is_valid


class TestPathLists:
    def test_relative_paths_resolved_in_base_dir(self, tmp_path: Path) -> None:
        _write_lines(tmp_path / "a.jsonl", _good())
        _write_lines(tmp_path / "b.json", _good())
        result = Validators.validate_dataset(["a.jsonl", "b.json"], base_dir=tmp_path)
        assert result.is_valid
        # content is not inspected for list inputs
        assert result.metadata == {}

    def test_absolute_paths_need_no_base_dir(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "a.jsonl", _good())
        missing_base = tmp_path / "does-not-exist"
        result = Validators.validate_dataset([f], base_dir=missing_base)
        assert result.is_valid

    def test_path_objects_accepted(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "a.jsonl", _good())
        assert Validators.validate_dataset([f]).is_valid

    def test_error_field_is_indexed(self, tmp_path: Path) -> None:
        _write_lines(tmp_path / "a.jsonl", _good())
        (tmp_path / "b.txt").write_text("x")
        result = Validators.validate_dataset(["a.jsonl", "b.txt"], base_dir=tmp_path)
        assert not result.is_valid
        assert {e.field for e in result.errors} == {"dataset[1]"}

    def test_missing_file_not_found(self, tmp_path: Path) -> None:
        result = Validators.validate_dataset(["nope.jsonl"], base_dir=tmp_path)
        assert _errs(result) == [("dataset[0]", "NOT_FOUND")]

    def test_errors_accumulate_across_entries(self, tmp_path: Path) -> None:
        result = Validators.validate_dataset(
            ["nope1.jsonl", "nope2.jsonl"], base_dir=tmp_path
        )
        assert _errs(result) == [
            ("dataset[0]", "NOT_FOUND"),
            ("dataset[1]", "NOT_FOUND"),
        ]

    def test_base_dir_missing(self, tmp_path: Path) -> None:
        result = Validators.validate_dataset(["a.jsonl"], base_dir=tmp_path / "nope")
        assert _errs(result) == [("dataset", "SECURITY_ERROR")]
        assert result.errors[0].message.startswith(
            "Dataset base directory cannot be resolved: "
        )

    def test_base_dir_is_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = Validators.validate_dataset(["a.jsonl"], base_dir=f)
        assert _errs(result) == [("dataset", "SECURITY_ERROR")]
        assert result.errors[0].message == (
            f"Dataset base path is not a directory: {f.resolve()}"
        )

    def test_default_base_is_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_lines(tmp_path / "a.jsonl", _good())
        monkeypatch.chdir(tmp_path)
        assert Validators.validate_dataset(["a.jsonl"]).is_valid

    def test_traversal_outside_base_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        _write_lines(tmp_path / "outside.jsonl", _good())
        result = Validators.validate_dataset(["../outside.jsonl"], base_dir=base)
        assert not result.is_valid
        assert {e.field for e in result.errors} == {"dataset[0]"}

    def test_unresolvable_entry_stops_processing(self, tmp_path: Path) -> None:
        loop_a = tmp_path / "loop_a"
        loop_b = tmp_path / "loop_b"
        os.symlink(loop_b, loop_a)
        os.symlink(loop_a, loop_b)
        result = Validators.validate_dataset(
            ["loop_a", "nope.jsonl"], base_dir=tmp_path
        )
        # Only the resolution error; the second entry is never examined.
        assert _errs(result) == [("dataset", "SECURITY_ERROR")]
        assert result.errors[0].message.startswith("Dataset path cannot be resolved: ")

    def test_mixed_relative_and_absolute(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        _write_lines(base / "a.jsonl", _good())
        other = _write_lines(tmp_path / "b.jsonl", _good())
        assert Validators.validate_dataset(["a.jsonl", other], base_dir=base).is_valid


class TestSinglePathStructure:
    def test_relative_valid(self, tmp_path: Path) -> None:
        _write_lines(tmp_path / "d.jsonl", _good(3))
        result = Validators.validate_dataset("d.jsonl", base_dir=tmp_path)
        assert result.is_valid
        assert result.metadata == {
            "dataset_rows_inspected": 3,
            "dataset_invalid_rows": 0,
        }

    def test_absolute_str_and_path(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "d.jsonl", _good())
        assert Validators.validate_dataset(str(f)).is_valid
        assert Validators.validate_dataset(f).is_valid

    def test_nested_relative_path(self, tmp_path: Path) -> None:
        (tmp_path / "sub" / "deep").mkdir(parents=True)
        _write_lines(tmp_path / "sub" / "deep" / "d.jsonl", _good(2))
        result = Validators.validate_dataset("sub/deep/d.jsonl", base_dir=tmp_path)
        assert result.is_valid
        assert result.metadata["dataset_rows_inspected"] == 2

    def test_default_base_is_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_lines(tmp_path / "d.jsonl", _good())
        monkeypatch.chdir(tmp_path)
        assert Validators.validate_dataset("d.jsonl").is_valid

    def test_str_base_dir_with_user_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_lines(tmp_path / "d.jsonl", _good())
        monkeypatch.setenv("HOME", str(tmp_path))
        assert Validators.validate_dataset("d.jsonl", base_dir="~").is_valid

    def test_base_dir_missing(self, tmp_path: Path) -> None:
        result = Validators.validate_dataset("d.jsonl", base_dir=tmp_path / "nope")
        assert _errs(result) == [("dataset", "SECURITY_ERROR")]
        assert result.errors[0].message.startswith(
            "Dataset base directory cannot be resolved: "
        )
        assert result.metadata == {}

    def test_base_dir_is_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = Validators.validate_dataset("d.jsonl", base_dir=f)
        assert _errs(result) == [("dataset", "SECURITY_ERROR")]
        assert result.errors[0].message == (
            f"Dataset base path is not a directory: {f.resolve()}"
        )

    def test_absolute_ignores_bad_base_dir(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "d.jsonl", _good())
        assert Validators.validate_dataset(f, base_dir=tmp_path / "nope").is_valid

    def test_missing_file(self, tmp_path: Path) -> None:
        result = Validators.validate_dataset("nope.jsonl", base_dir=tmp_path)
        assert _errs(result) == [("dataset", "NOT_FOUND")]
        assert result.metadata == {}

    def test_missing_absolute_file(self, tmp_path: Path) -> None:
        result = Validators.validate_dataset(tmp_path / "nope.jsonl")
        assert _errs(result) == [("dataset", "NOT_FOUND")]

    def test_wrong_extension_skips_content(self, tmp_path: Path) -> None:
        f = tmp_path / "d.txt"
        f.write_text("not json\n")
        result = Validators.validate_dataset(f)
        assert not result.is_valid
        assert {e.field for e in result.errors} == {"dataset"}
        assert result.metadata == {}
        assert all(e.error_code != "JSON_ERROR" for e in result.errors)

    def test_directory_is_rejected(self, tmp_path: Path) -> None:
        d = tmp_path / "dir.jsonl"
        d.mkdir()
        result = Validators.validate_dataset(d)
        assert not result.is_valid
        assert result.metadata == {}

    def test_traversal_outside_base_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        _write_lines(tmp_path / "outside.jsonl", _good())
        result = Validators.validate_dataset("../outside.jsonl", base_dir=base)
        assert not result.is_valid
        assert result.metadata == {}

    def test_unresolvable_path(self, tmp_path: Path) -> None:
        a = tmp_path / "loop_a"
        b = tmp_path / "loop_b"
        os.symlink(b, a)
        os.symlink(a, b)
        result = Validators.validate_dataset("loop_a", base_dir=tmp_path)
        assert _errs(result) == [("dataset", "SECURITY_ERROR")]
        assert result.errors[0].message.startswith("Dataset path cannot be resolved: ")

    def test_relative_symlink_resolved_to_target(self, tmp_path: Path) -> None:
        real = _write_lines(tmp_path / "real.jsonl", _good(2))
        os.symlink(real, tmp_path / "link.jsonl")
        result = Validators.validate_dataset("link.jsonl", base_dir=tmp_path)
        assert result.is_valid
        assert result.metadata["dataset_rows_inspected"] == 2


class TestSinglePathContent:
    def test_invalid_json_line(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "d.jsonl", [_good()[0], "{bad", _good()[0]])
        result = Validators.validate_dataset(f)
        assert _errs(result) == [("dataset:line2", "JSON_ERROR")]
        assert result.errors[0].message == "Invalid JSON"
        assert result.metadata == {
            "dataset_rows_inspected": 3,
            "dataset_invalid_rows": 1,
        }

    @pytest.mark.parametrize(
        "text,type_name",
        [("[1, 2]", "list"), ("5", "int"), ('"s"', "str"), ("null", "NoneType")],
    )
    def test_non_object_row(self, tmp_path: Path, text: str, type_name: str) -> None:
        f = _write_lines(tmp_path / "d.jsonl", [text])
        result = Validators.validate_dataset(f)
        assert _errs(result) == [("dataset:line1", "INVALID_FORMAT")]
        assert result.errors[0].message == (f"Expected a JSON object, got {type_name}")

    def test_missing_input_field(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "d.jsonl", ['{"foo": 1}'])
        result = Validators.validate_dataset(f)
        assert _errs(result) == [("dataset:line1", "INVALID_FORMAT")]
        assert result.errors[0].message == "Missing 'input' or 'input_data' field"
        assert result.metadata["dataset_invalid_rows"] == 1

    def test_input_data_alias_accepted(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "d.jsonl", ['{"input_data": 1}'])
        assert Validators.validate_dataset(f).is_valid

    def test_blank_lines_skipped_but_numbered(self, tmp_path: Path) -> None:
        f = tmp_path / "d.jsonl"
        f.write_text('\n   \n{"input": 1}\n\n{bad\n', encoding="utf-8")
        result = Validators.validate_dataset(f)
        assert _errs(result) == [("dataset:line5", "JSON_ERROR")]
        assert result.metadata == {
            "dataset_rows_inspected": 2,
            "dataset_invalid_rows": 1,
        }

    def test_error_cap_and_summary(self, tmp_path: Path) -> None:
        total = MAX_REPORTED_DATASET_ROW_ERRORS + 5
        f = _write_lines(tmp_path / "d.jsonl", ["{bad"] * total)
        result = Validators.validate_dataset(f)
        listed = [e for e in result.errors if e.field.startswith("dataset:line")]
        assert len(listed) == MAX_REPORTED_DATASET_ROW_ERRORS
        assert [e.field for e in listed][-1] == (
            f"dataset:line{MAX_REPORTED_DATASET_ROW_ERRORS}"
        )
        summary = result.errors[-1]
        assert (summary.field, summary.error_code) == ("dataset", "INVALID_FORMAT")
        assert summary.message == (
            f"{total} of {total} rows are invalid "
            "(all rows were inspected; only the first "
            f"{MAX_REPORTED_DATASET_ROW_ERRORS} are listed above)"
        )
        assert len(result.errors) == MAX_REPORTED_DATASET_ROW_ERRORS + 1
        assert result.metadata["dataset_invalid_rows"] == total

    def test_exactly_cap_has_no_summary(self, tmp_path: Path) -> None:
        f = _write_lines(
            tmp_path / "d.jsonl", ["{bad"] * MAX_REPORTED_DATASET_ROW_ERRORS
        )
        result = Validators.validate_dataset(f)
        assert len(result.errors) == MAX_REPORTED_DATASET_ROW_ERRORS
        assert all(e.field.startswith("dataset:line") for e in result.errors)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "d.jsonl"
        f.write_text("", encoding="utf-8")
        result = Validators.validate_dataset(f)
        assert _errs(result) == [("dataset", "EMPTY_FILE")]
        assert result.errors[0].message == "Dataset file is empty"
        assert result.metadata == {
            "dataset_rows_inspected": 0,
            "dataset_invalid_rows": 0,
        }

    def test_only_blank_lines_is_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "d.jsonl"
        f.write_text("\n\n  \n", encoding="utf-8")
        assert _errs(Validators.validate_dataset(f)) == [("dataset", "EMPTY_FILE")]

    def test_json_extension_is_line_oriented(self, tmp_path: Path) -> None:
        f = tmp_path / "d.json"
        f.write_text('[{"input": 1}]', encoding="utf-8")
        result = Validators.validate_dataset(f)
        assert _errs(result) == [("dataset:line1", "INVALID_FORMAT")]

    def test_undecodable_bytes_read_error(self, tmp_path: Path) -> None:
        f = tmp_path / "d.jsonl"
        f.write_bytes(b'{"input": 1}\n\xff\xfe\n')
        result = Validators.validate_dataset(f)
        assert [(e.field, e.error_code) for e in result.errors] == [
            ("dataset", "READ_ERROR")
        ]
        assert result.errors[0].message.startswith("Could not read dataset: ")
        # metadata is assigned only after a complete read
        assert result.metadata == {}

    def test_open_failure_read_error(self, tmp_path: Path) -> None:
        f = _write_lines(tmp_path / "d.jsonl", _good())
        with patch(
            "traigent.utils.validation.safe_open", side_effect=OSError("boom")
        ) as opener:
            result = Validators.validate_dataset(f)
        assert [(e.field, e.error_code, e.message) for e in result.errors] == [
            ("dataset", "READ_ERROR", "Could not read dataset: boom")
        ]
        args, kwargs = opener.call_args
        assert args == (f.resolve(), f.resolve().parent)
        assert kwargs == {"mode": "r", "encoding": "utf-8"}


class _LatePath(str):
    """str subclass whose startswith() raises, to detect which paths get inspected."""

    def startswith(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("late path inspected")


def test_list_inspects_every_path_even_after_a_relative_one(tmp_path: Path) -> None:
    # Every entry is examined eagerly, so a later entry that misbehaves
    # surfaces even though an earlier one is already relative. The base
    # directory does not exist, so only the up-front scan can raise.
    with pytest.raises(RuntimeError, match="late path inspected"):
        Validators.validate_dataset(
            ["relative.jsonl", _LatePath("/absolute.jsonl")],
            base_dir=tmp_path / "missing",
        )
