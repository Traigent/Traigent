"""Tests for validation utilities."""

from pathlib import Path

import pytest

from traigent.utils.validation import Validators


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
