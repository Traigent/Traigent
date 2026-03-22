"""Tests for scripts.verify_example_results helpers."""

from __future__ import annotations

import urllib.error
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


def test_write_snapshot_json_rejects_paths_outside_snapshot_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(verify_example_results, "SNAPSHOTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Invalid snapshot write path"):
        verify_example_results.write_snapshot_json(tmp_path.parent / "outside.json", {})


def test_query_backend_disallows_redirects(monkeypatch, capsys) -> None:
    class FakeOpener:
        def open(self, req, timeout):
            raise urllib.error.HTTPError(
                req.full_url,
                302,
                "Found",
                hdrs=None,
                fp=None,
            )

    monkeypatch.setattr(
        verify_example_results.urllib.request,
        "build_opener",
        lambda handler: (
            isinstance(handler, verify_example_results._NoRedirectHandler)
            and FakeOpener()
        ),
    )

    result = verify_example_results.query_backend(
        "experiments",
        "http://localhost:5000/api/v1",
        "test-api-key",
    )

    assert result is None
    assert (
        "Backend query failed (experiments): HTTP Error 302: Found"
        in capsys.readouterr().err
    )
