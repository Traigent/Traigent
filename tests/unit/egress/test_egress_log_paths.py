"""The relay log path is operator-chosen; it must be a plain file, not a redirect."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from traigent.egress.log import RelayLog, RelayLogError, iter_lines, operator_log_path


def test_a_plain_path_opens_and_resolves(tmp_path: Path) -> None:
    log = RelayLog(tmp_path / "sub" / "relay.jsonl")
    try:
        assert log.path == (tmp_path / "sub" / "relay.jsonl").resolve()
    finally:
        log.close()


def test_a_dotdot_path_is_normalised_to_the_real_file(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    assert operator_log_path(tmp_path / "a" / ".." / "x.jsonl") == (
        tmp_path / "x.jsonl"
    ).resolve()


def test_a_nul_byte_is_refused() -> None:
    with pytest.raises(RelayLogError) as exc:
        operator_log_path("canary\0.jsonl")
    assert "canary" not in str(exc.value)


def test_a_symlinked_log_is_refused_for_write_and_read(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    link = tmp_path / "relay.jsonl"
    os.symlink(target, link)
    with pytest.raises(RelayLogError):
        RelayLog(link)
    assert list(iter_lines(link)) == []


def test_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RelayLogError):
        RelayLog(tmp_path)
