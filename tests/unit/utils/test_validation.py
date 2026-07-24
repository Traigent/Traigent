"""Tests for validation utilities."""

import json
from pathlib import Path

import pytest

from traigent.utils.secure_path import PathTraversalError, safe_open
from traigent.utils.validation import Validators


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
