"""Tests for validation utilities."""

import json
import warnings
from pathlib import Path

import pytest

from traigent.utils.secure_path import PathTraversalError, safe_open
from traigent.utils.validation import Validators, validate_config_space


def _write_jsonl(path: Path, records: list[dict] | None = None) -> None:
    """Write a small valid JSONL dataset (each line carries an ``input`` field)."""
    records = records or [
        {"input": {"question": "What is 2+2?"}, "expected_output": "4"}
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class TestValidatePath:
    """Tests for Validators.validate_path security handling."""

    def test_validate_path_within_allowed_base(self, tmp_path: Path) -> None:
        allowed_base = tmp_path / "workspace"
        allowed_base.mkdir()
        file_path = allowed_base / "data.txt"
        file_path.write_text("content")

        result = Validators.validate_path(
            file_path,
            "path",
            must_exist=True,
            must_be_file=True,
            allowed_base_dirs=[allowed_base],
        )

        assert result.is_valid

    def test_validate_path_outside_allowed_base(self, tmp_path: Path) -> None:
        allowed_base = tmp_path / "workspace"
        allowed_base.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        result = Validators.validate_path(
            outside_dir,
            "path",
            must_exist=True,
            must_be_dir=True,
            allowed_base_dirs=[allowed_base],
        )

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)


class TestValidateDataset:
    """Tests for Validators.validate_dataset."""

    def test_validate_dataset_accepts_jsonl_input_data_alias(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        dataset_path = tmp_path / "inline_alias.jsonl"
        dataset_path.write_text(
            json.dumps(
                {"input_data": {"question": "What is 2+2?"}, "expected_output": "4"}
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))

        result = Validators.validate_dataset(str(dataset_path))

        assert result.is_valid

    def test_validate_path_symlink_escaping_base(self, tmp_path: Path) -> None:
        allowed_base = tmp_path / "workspace"
        allowed_base.mkdir()

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        symlink_path = allowed_base / "escape"
        try:
            symlink_path.symlink_to(target_dir, target_is_directory=True)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")

        result = Validators.validate_path(
            symlink_path,
            "path",
            must_exist=True,
            must_be_dir=True,
            allowed_base_dirs=[allowed_base],
        )

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    # ----- Regression: issue #1983 (relative dataset paths) -----
    # Relative dataset paths are anchored to the invoking process's current working
    # directory. They must remain inside that base after resolving traversal and
    # symlinks, while the resolved absolute path is passed to ``safe_open`` so nested
    # relative paths are not joined a second time during the content read.

    def test_validate_dataset_accepts_nested_relative_str_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(a) A nested relative STR path is accepted (issue #1983)."""
        nested = tmp_path / "traigent-runs"
        nested.mkdir()
        _write_jsonl(nested / "dataset.jsonl")
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("traigent-runs/dataset.jsonl")

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_accepts_nested_relative_path_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(b) A nested relative Path OBJECT is accepted (issue #1983)."""
        nested = tmp_path / "traigent-runs"
        nested.mkdir()
        _write_jsonl(nested / "dataset.jsonl")
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset(Path("traigent-runs/dataset.jsonl"))

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_accepts_absolute_path(self, tmp_path: Path) -> None:
        """(c) An absolute path is still accepted (unchanged by the fix)."""
        nested = tmp_path / "traigent-runs"
        nested.mkdir()
        dataset_path = nested / "dataset.jsonl"
        _write_jsonl(dataset_path)

        result = Validators.validate_dataset(str(dataset_path.resolve()))

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_accepts_path_with_spaces(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(d) A nested relative path whose dir and file contain spaces is accepted."""
        spaced_dir = tmp_path / "my data dir"
        spaced_dir.mkdir()
        _write_jsonl(spaced_dir / "my dataset.jsonl")
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("my data dir/my dataset.jsonl")

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_uses_explicit_relative_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller can explicitly choose the allowed base for relative paths."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        nested = workspace / "datasets"
        nested.mkdir()
        _write_jsonl(nested / "dataset.jsonl")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        result = Validators.validate_dataset(
            "datasets/dataset.jsonl", base_dir=workspace
        )

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_rejects_existing_traversal_escape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing ``..`` target outside the CWD boundary is rejected."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _write_jsonl(outside / "dataset.jsonl")
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("../outside/dataset.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    def test_validate_dataset_rejects_symlink_escape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A symlink inside the CWD boundary cannot resolve outside it."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "dataset.jsonl"
        _write_jsonl(target)
        link = workspace / "dataset.jsonl"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("dataset.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    def test_validate_dataset_rejects_symlink_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A symlink loop inside the base returns a security error."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        loop = workspace / "loop.jsonl"
        try:
            loop.symlink_to(loop.name)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("loop.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    def test_validate_dataset_reports_missing_in_base_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing path inside the base retains the normal NOT_FOUND result."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("missing.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "NOT_FOUND" for error in result.errors)


class TestValidateDatasetReadsEveryRow:
    """Regression: issue #2022 (content validation stopped after 5 lines).

    ``Validators.validate_dataset`` used to ``break`` once it had seen five
    lines, so a malformed row past line 5 validated clean while the runtime
    loader (``Dataset.from_jsonl``) parsed the same file and raised. The two
    paths must agree, and the reported row count must not overstate what was
    actually inspected.
    """

    def test_malformed_row_past_line_five_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The issue's exact repro: 8 good rows, then unparseable line 9."""
        dataset_path = tmp_path / "bad.jsonl"
        lines = [json.dumps({"input": f"q{i}", "output": "a"}) for i in range(8)]
        lines.append("THIS IS NOT JSON AT ALL")
        dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("bad.jsonl")

        assert not result.is_valid
        assert any(
            error.error_code == "JSON_ERROR" and error.field == "dataset:line9"
            for error in result.errors
        ), [(e.field, e.error_code, e.message) for e in result.errors]

    def test_missing_input_field_past_line_five_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parseable-but-shapeless row past line 5 is reported too."""
        dataset_path = tmp_path / "shapeless.jsonl"
        records = [{"input": f"q{i}", "expected_output": "a"} for i in range(8)]
        records.append({"question": "no input field"})
        _write_jsonl(dataset_path, records)
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("shapeless.jsonl")

        assert not result.is_valid
        assert any(
            error.error_code == "INVALID_FORMAT" and error.field == "dataset:line9"
            for error in result.errors
        ), [(e.field, e.error_code, e.message) for e in result.errors]

    def test_valid_dataset_longer_than_five_rows_stays_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full scanning must not introduce false positives on a good file."""
        dataset_path = tmp_path / "good.jsonl"
        _write_jsonl(
            dataset_path,
            [{"input": f"q{i}", "expected_output": "a"} for i in range(50)],
        )
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("good.jsonl")

        assert result.is_valid, [(e.field, e.message) for e in result.errors]
        assert result.metadata["dataset_rows_inspected"] == 50
        assert result.metadata["dataset_invalid_rows"] == 0

    def test_blank_lines_are_skipped_like_the_runtime_loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``Dataset.from_jsonl`` skips blank lines, so validation must too."""
        dataset_path = tmp_path / "blanks.jsonl"
        dataset_path.write_text(
            "\n".join(
                [json.dumps({"input": "q1", "expected_output": "a"})]
                + ["", "   "]
                + [json.dumps({"input": "q2", "expected_output": "b"})]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("blanks.jsonl")

        assert result.is_valid, [(e.field, e.message) for e in result.errors]
        assert result.metadata["dataset_rows_inspected"] == 2

    def test_non_object_row_does_not_abort_the_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scalar row is reported per-row, not raised as an opaque READ_ERROR.

        Without this, ``"input" not in 5`` raises TypeError, the whole read is
        reported as READ_ERROR and every later row goes uninspected - which
        would re-create the partial-scan problem this fix removes.
        """
        dataset_path = tmp_path / "scalar.jsonl"
        dataset_path.write_text(
            "\n".join(
                [json.dumps({"input": "q1", "expected_output": "a"}), "5", "not json"]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("scalar.jsonl")

        assert not result.is_valid
        codes = {(e.field, e.error_code) for e in result.errors}
        assert ("dataset:line2", "INVALID_FORMAT") in codes, codes
        assert ("dataset:line3", "JSON_ERROR") in codes, codes
        assert not any(e.error_code == "READ_ERROR" for e in result.errors), codes

    def test_many_bad_rows_report_the_true_total(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Listing is bounded, but the count of inspected/invalid rows is not."""
        dataset_path = tmp_path / "all_bad.jsonl"
        dataset_path.write_text(
            "\n".join("nope" for _ in range(60)) + "\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("all_bad.jsonl")

        assert not result.is_valid
        assert result.metadata["dataset_rows_inspected"] == 60
        assert result.metadata["dataset_invalid_rows"] == 60
        assert any("60 of 60 rows are invalid" in e.message for e in result.errors), [
            e.message for e in result.errors
        ]

    def test_agrees_with_runtime_loader_on_the_same_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of the issue: validator and loader must not disagree."""
        from traigent.evaluators.base import Dataset
        from traigent.utils.exceptions import ValidationError as LoaderValidationError

        dataset_path = tmp_path / "disagree.jsonl"
        lines = [json.dumps({"input": f"q{i}", "output": "a"}) for i in range(8)]
        lines.append("THIS IS NOT JSON AT ALL")
        dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))

        with pytest.raises(LoaderValidationError):
            Dataset.from_jsonl(str(dataset_path))

        assert not Validators.validate_dataset("disagree.jsonl").is_valid


class TestSafeOpenContainmentGuard:
    """The containment guard the issue #1983 fix relies on stays intact.

    ``validate_dataset`` derives its ``safe_open`` base from the dataset path
    itself, so it cannot, on its own, reject a ``../`` escape or an escaping
    symlink (empirically verified: an existing ``../outside/x.jsonl`` is accepted).
    The real defense is ``safe_open`` -> ``_resolve_path_in_base``: given a FIXED
    base directory, an escaping target still raises ``PathTraversalError``. The fix
    keeps passing an absolute path plus its own parent, so this guard is unweakened.
    """

    def test_safe_open_rejects_relative_traversal_escape(self, tmp_path: Path) -> None:
        """(e) A relative ``../`` target escaping a fixed base raises."""
        base = tmp_path / "workspace"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _write_jsonl(outside / "secret.jsonl")

        with pytest.raises(PathTraversalError):
            with safe_open("../outside/secret.jsonl", base, mode="r"):
                pass

    def test_safe_open_rejects_absolute_path_outside_base(self, tmp_path: Path) -> None:
        """An absolute target outside a fixed base raises."""
        base = tmp_path / "workspace"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.jsonl"
        _write_jsonl(secret)

        with pytest.raises(PathTraversalError):
            with safe_open(str(secret.resolve()), base, mode="r"):
                pass

    def test_safe_open_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """(f) A symlink inside a fixed base that points outside it raises."""
        base = tmp_path / "workspace"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.jsonl"
        _write_jsonl(secret)

        link = base / "link.jsonl"
        try:
            link.symlink_to(secret)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")

        with pytest.raises(PathTraversalError):
            with safe_open("link.jsonl", base, mode="r"):
                pass


class TestValidateConfigSpaceIsSideEffectFree:
    """``validate_config_space`` must stay a pure validator (issue #2021).

    The #2021 warning is emitted by the decorator, which is the only layer that
    knows the whole resolved space and the wrapped function's name. Surfacing
    per-parameter warnings from here instead nagged on the supported "pin one
    knob, sweep another" pattern, so this validator deliberately emits nothing.
    """

    def test_single_value_space_is_valid_and_silent(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_config_space({"temperature": [0.7]})

        assert record == []

    def test_pinned_knob_alongside_varying_knob_is_silent(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_config_space({"temperature": [0.0], "model": ["a", "b"]})

        assert record == []
