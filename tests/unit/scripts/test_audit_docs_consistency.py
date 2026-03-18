"""Tests for scripts.documentation.audit_docs_consistency helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.documentation import audit_docs_consistency


def test_build_output_path_returns_child_path(tmp_path: Path) -> None:
    path = audit_docs_consistency.build_output_path(tmp_path, "inventory_summary.md")
    assert path == (tmp_path / "inventory_summary.md").resolve()


def test_build_output_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid output path"):
        audit_docs_consistency.build_output_path(tmp_path, "../outside.md")
