"""#3131: one wrong-shaped result must not make list_results raise.

Valid JSON of the wrong type, non-UTF-8 bytes and a non-string created_at raised
out of the loop, so one damaged result hid every other result.
Offline: real PersistenceManager over tmp_path.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.utils.persistence import PersistenceManager

_LOGGER = "traigent.utils.persistence"


def _write(base, name, raw: bytes) -> None:
    d = base / name
    d.mkdir()
    (d / "metadata.json").write_bytes(raw)


def _good(base, name, created_at="2026-01-01T00:00:00+00:00") -> None:
    _write(
        base,
        name,
        json.dumps(
            {
                "created_at": created_at,
                "function_name": "f",
                "algorithm": "grid",
                "total_trials": 1,
            }
        ).encode(),
    )


# --- #3131: wrong-typed / undecodable entries are skipped, not raised ------


@pytest.mark.parametrize(
    "raw",
    [b"[]", b"null", b"5", b'"text"', b"\xff\xfe{"],
    ids=["list", "null", "number", "string", "non-utf8"],
)
def test_wrong_shaped_metadata_is_skipped(tmp_path, raw):
    _good(tmp_path, "a", "2026-01-01T00:00:00+00:00")
    _good(tmp_path, "b", "2026-01-02T00:00:00+00:00")
    _write(tmp_path, "bad", raw)
    pm = PersistenceManager(tmp_path)
    assert [r["name"] for r in pm.list_results()] == ["b", "a"]
    assert pm.last_skipped_results == 1


@pytest.mark.parametrize("created_at", [None, 1700000000], ids=["null", "number"])
def test_non_string_created_at_is_listed_last_not_raised(tmp_path, created_at):
    _good(tmp_path, "a", "2026-01-01T00:00:00+00:00")
    _write(
        tmp_path,
        "odd",
        json.dumps(
            {
                "created_at": created_at,
                "function_name": "f",
                "algorithm": "grid",
                "total_trials": 1,
            }
        ).encode(),
    )
    pm = PersistenceManager(tmp_path)
    assert [r["name"] for r in pm.list_results()] == ["a", "odd"]
    assert pm.last_skipped_results == 0


def test_cli_results_list_survives_non_string_created_at(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    _good(store, "a")
    _write(
        store,
        "odd",
        json.dumps(
            {
                "created_at": None,
                "function_name": "f",
                "algorithm": "grid",
                "total_trials": 1,
            }
        ).encode(),
    )
    result = CliRunner().invoke(cli, ["results", "list", "--storage-dir", str(store)])
    assert result.exit_code == 0, result.output
    assert "odd" in result.output


@pytest.fixture
def wide_console(monkeypatch):
    """Render the CLI table wide enough that no cell is cut to an ellipsis."""
    from rich.console import Console

    import traigent.cli.main as cli_main

    monkeypatch.setattr(cli_main, "console", Console(width=200))


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "created_at": "2026-01-01T00:00:00+00:00",
            "function_name": "f",
            "algorithm": "grid",
            "total_trials": 1,
            "success_rate": None,
        },
    ],
    ids=["empty-object", "null-success-rate"],
)
def test_cli_lists_results_with_missing_or_null_fields(
    tmp_path, metadata, wide_console
):
    """An incomplete metadata object shows N/A cells; it does not abort the list."""
    store = tmp_path / "store"
    store.mkdir()
    _good(store, "a")
    _write(store, "partial", json.dumps(metadata).encode())

    result = CliRunner().invoke(cli, ["results", "list", "--storage-dir", str(store)])

    assert result.exit_code == 0, result.output
    assert "partial" in result.output
    assert "N/A" in result.output


def test_cli_does_not_invent_a_success_rate(tmp_path, wide_console):
    """A result without success_rate shows N/A, not a made-up 0.0%."""
    store = tmp_path / "store"
    store.mkdir()
    _good(store, "a")

    result = CliRunner().invoke(cli, ["results", "list", "--storage-dir", str(store)])

    assert result.exit_code == 0, result.output
    assert "0.0%" not in result.output
