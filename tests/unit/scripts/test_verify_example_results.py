"""Tests for scripts.verify_example_results helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import verify_example_results


def test_build_backend_url_rejects_absolute_endpoint() -> None:
    with pytest.raises(ValueError, match="relative path"):
        verify_example_results.build_backend_url(
            "http://localhost:5000/api/v1",
            "https://evil.example/api/v1/experiments",
        )


def test_build_backend_url_stays_on_configured_origin() -> None:
    url = verify_example_results.build_backend_url(
        "http://localhost:5000/api/v1/",
        "experiments",
    )
    assert url == "http://localhost:5000/api/v1/experiments"


def test_snapshot_path_rejects_traversal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify_example_results, "SNAPSHOTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Snapshot labels"):
        verify_example_results.snapshot_path("../outside")


def test_snapshot_path_accepts_safe_labels(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify_example_results, "SNAPSHOTS_DIR", tmp_path)
    path = verify_example_results.snapshot_path("before-run")
    assert path == (tmp_path / "before-run.json").resolve()
