"""A dataset path with an embedded NUL byte is a validation error, not a crash."""

from pathlib import Path

import pytest

from traigent.utils.validation import OptimizationValidator, Validators


@pytest.mark.parametrize("path", ["a\x00b", "/tmp/a\x00b.jsonl", Path("x") / "y\x00z"])
def test_validate_returns_error_for_nul_dataset_path(path) -> None:
    result = OptimizationValidator().validate(dataset=path)
    assert not result.is_valid
    assert any("dataset" in e.field for e in result.errors)


def test_validate_dataset_list_with_nul_entry() -> None:
    result = Validators.validate_dataset(["a\x00b"])
    assert not result.is_valid


def test_validate_dataset_nul_base_dir() -> None:
    result = Validators.validate_dataset("data.jsonl", base_dir="bad\x00dir")
    assert not result.is_valid


def test_missing_path_control_still_reports_not_found() -> None:
    result = OptimizationValidator().validate(dataset="/nonexistent/x.jsonl")
    assert not result.is_valid
