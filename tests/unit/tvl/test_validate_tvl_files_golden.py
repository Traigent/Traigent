"""Golden characterization for ``traigent.tvl.__main__.validate_tvl_files``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from traigent.tvl import __main__ as cli
from traigent.tvl.spec_loader import TVLValidationError


def _w(tmp_path: Path, name: str, text: str | bytes) -> Path:
    p = tmp_path / name
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding="utf-8")
    return p


def _run(
    files: list[Path], capsys: pytest.CaptureFixture[str], **kw: Any
) -> tuple[tuple[int, int], str, str]:
    result = cli.validate_tvl_files(files, **kw)
    cap = capsys.readouterr()
    return result, cap.out, cap.err


def test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.tvl.yml"
    res, out, err = _run([missing], capsys)
    assert res == (0, 1)
    assert out == ""
    assert err == f"ERROR: File not found: {missing}\n"


def test_empty_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = _w(tmp_path, "e.tvl.yml", "")
    res, _, err = _run([f], capsys)
    assert res == (0, 1)
    assert err == f"ERROR: Empty TVL spec: {f}\n"


def test_invalid_yaml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = _w(tmp_path, "y.tvl.yml", "a: [unclosed")
    res, _, err = _run([f], capsys)
    assert res == (0, 1)
    assert err.startswith(f"FAILED: {f}\n  - Invalid YAML: ")


@pytest.mark.parametrize("payload", [b"\xff\xfe\x00bad", None])
def test_unexpected_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: bytes | None
) -> None:
    f = tmp_path if payload is None else _w(tmp_path, "b.tvl.yml", payload)
    res, _, err = _run([f], capsys)
    assert res == (0, 1)
    assert err.startswith(f"FAILED: {f}\n  - Unexpected error: ")


def test_schema_issues_listed_and_strict_flag_forwarded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = _w(tmp_path, "s.tvl.yml", "x: 1\n")
    calls: list[Any] = []

    def fake(data: Any, include_warnings: bool = False) -> list[str]:
        calls.append((data, include_warnings))
        return ["first", "second"]

    with patch.object(cli, "validate_tvl_schema", fake):
        res, _, err = _run([f], capsys, strict=True)
        res2, _, _ = _run([f], capsys)
    assert res == (0, 1) and res2 == (0, 1)
    assert err == f"FAILED: {f}\n  - first\n  - second\n"
    assert calls == [({"x": 1}, True), ({"x": 1}, False)]


def test_load_errors_distinguish_validation_from_generic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = _w(tmp_path, "l.tvl.yml", "x: 1\n")
    with (
        patch.object(cli, "validate_tvl_schema", lambda *a, **k: []),
        patch.object(cli, "load_tvl_spec", side_effect=TVLValidationError("bad")),
    ):
        res, _, err = _run([f], capsys)
    assert res == (0, 1) and err == f"FAILED: {f}\n  - bad\n"
    with (
        patch.object(cli, "validate_tvl_schema", lambda *a, **k: []),
        patch.object(cli, "load_tvl_spec", side_effect=RuntimeError("kaput")),
    ):
        res, _, err = _run([f], capsys)
    assert res == (0, 1) and err == f"FAILED: {f}\n  - Load error: kaput\n"


def test_success_verbose_and_quiet_and_load_receives_unresolved_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = _w(tmp_path, "ok.tvl.yml", "x: 1\n")
    with (
        patch.object(cli, "validate_tvl_schema", lambda *a, **k: []),
        patch.object(cli, "load_tvl_spec") as load,
    ):
        res, out, err = _run([f], capsys, verbose=True)
        assert (res, out, err) == ((1, 0), f"OK: {f}\n", "")
        res, out, err = _run([f], capsys)
        assert (res, out, err) == ((1, 0), "", "")
    assert load.call_args.kwargs == {"spec_path": str(f)}


def test_mixed_batch_counts_and_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = _w(tmp_path, "ok.tvl.yml", "x: 1\n")
    empty = _w(tmp_path, "e.tvl.yml", "")
    missing = tmp_path / "m.tvl.yml"
    with (
        patch.object(cli, "validate_tvl_schema", lambda *a, **k: []),
        patch.object(cli, "load_tvl_spec"),
    ):
        res, out, err = _run([missing, ok, empty, ok], capsys, verbose=True)
    assert res == (2, 2)
    assert out == f"OK: {ok}\nOK: {ok}\n"
    assert err.index("File not found") < err.index("Empty TVL spec")
    assert cli.validate_tvl_files([], strict=True) == (0, 0)
