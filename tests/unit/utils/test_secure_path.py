from __future__ import annotations

from pathlib import Path

import pytest

from traigent.utils.secure_path import PathTraversalError, SafePath, validate_path


def test_safe_path_resolves_relative_path_inside_base(tmp_path: Path) -> None:
    resolver = SafePath(tmp_path)

    assert resolver.resolve("nested/file.txt") == tmp_path / "nested" / "file.txt"


def test_safe_path_rejects_relative_escape(tmp_path: Path) -> None:
    resolver = SafePath(tmp_path)

    with pytest.raises(PathTraversalError):
        resolver.resolve("../outside.txt")


def test_safe_path_allows_absolute_path_inside_base(tmp_path: Path) -> None:
    child = tmp_path / "child.txt"

    assert SafePath(tmp_path).resolve(child) == child


def test_validate_path_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(PathTraversalError):
        validate_path(link / "secret.txt", tmp_path)
