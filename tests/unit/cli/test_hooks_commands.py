"""Truthful output and exit status for `traigent hooks install|uninstall|status`.

Every test drives the real Click command tree against a real temporary
repository and checks the printed state against the files on disk. A refusal
or I/O error exits 1. A hook that is only skipped keeps exit 0, and the
summary must not claim success.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from traigent.cli.main import cli
from traigent.hooks.installer import (
    PRE_COMMIT_HOOK_SCRIPT,
    PRE_PUSH_HOOK_SCRIPT,
    HooksInstaller,
)

TEMPLATES = {"pre-push": PRE_PUSH_HOOK_SCRIPT, "pre-commit": PRE_COMMIT_HOOK_SCRIPT}
FOREIGN = b"#!/bin/sh\n# customer hook F1\necho 'customer check'\n"
ORIGINAL = b"#!/bin/sh\n# customer original F0\nexit 0\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    return tmp_path


def _hooks(repo: Path) -> Path:
    return repo / ".git" / "hooks"


def _write(path: Path, data: bytes, mode: int) -> os.stat_result:
    path.write_bytes(data)
    os.chmod(path, mode)
    return os.lstat(path)


def _invoke(repo: Path, *args: str) -> Result:
    return CliRunner().invoke(cli, ["hooks", *args, "--path", str(repo)])


def _state_rows(output: str) -> dict[str, str]:
    """Hook -> status cell of the last state table printed."""
    rows: dict[str, str] = {}
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.replace("|", "│").split("│")]
        cells = [cell for cell in cells if cell]
        if len(cells) == 2 and cells[0] in TEMPLATES:
            rows[cells[0]] = " ".join(cells[1].lower().split())
    return rows


def _unchanged(path: Path, before: os.stat_result, data: bytes) -> bool:
    after = os.lstat(path)
    return (path.read_bytes(), after.st_mode, after.st_ino) == (
        data,
        before.st_mode,
        before.st_ino,
    )


def test_install_reports_each_hook_and_the_resulting_state(repo: Path) -> None:
    result = _invoke(repo, "install")

    assert result.exit_code == 0, result.output
    assert "Installed: .git/hooks/pre-push" in result.output
    assert "Installed: .git/hooks/pre-commit" in result.output
    assert "Hooks installed successfully!" in result.output
    assert _state_rows(result.output) == dict.fromkeys(
        TEMPLATES, "installed (traigent)"
    )


def test_skipped_foreign_hook_exits_zero_without_claiming_success(
    repo: Path,
) -> None:
    hook = _hooks(repo) / "pre-push"
    before = _write(hook, FOREIGN, 0o700)

    result = _invoke(repo, "install")

    assert result.exit_code == 0, result.output
    assert "Skipped: .git/hooks/pre-push" in result.output
    assert "Installed: .git/hooks/pre-commit" in result.output
    assert "Hooks installed successfully!" not in result.output
    assert "Some hooks were skipped" in result.output
    assert _state_rows(result.output) == {
        "pre-push": "installed (other)",
        "pre-commit": "installed (traigent)",
    }
    assert _unchanged(hook, before, FOREIGN)


def test_force_conflict_after_first_hook_exits_one_and_shows_actual_state(
    repo: Path,
) -> None:
    hooks = _hooks(repo)
    active = _write(hooks / "pre-commit", FOREIGN, 0o700)
    backup = _write(hooks / "pre-commit.backup", ORIGINAL, 0o750)

    result = _invoke(repo, "install", "--force")

    assert result.exit_code == 1, result.output
    assert "Error" in result.output
    assert "pre-commit.backup" in result.output
    assert "--force" not in result.output
    assert "Hooks installed successfully!" not in result.output
    assert _state_rows(result.output) == {
        "pre-push": "installed (traigent)",
        "pre-commit": "installed (other)",
    }
    assert _unchanged(hooks / "pre-commit", active, FOREIGN)
    assert _unchanged(hooks / "pre-commit.backup", backup, ORIGINAL)


def test_force_conflict_on_first_hook_leaves_the_second_unprocessed(
    repo: Path,
) -> None:
    hooks = _hooks(repo)
    _write(hooks / "pre-push", FOREIGN, 0o700)
    _write(hooks / "pre-push.backup", ORIGINAL, 0o750)

    result = _invoke(repo, "install", "--force")

    assert result.exit_code == 1, result.output
    assert "Hooks installed successfully!" not in result.output
    assert _state_rows(result.output) == {
        "pre-push": "installed (other)",
        "pre-commit": "not installed",
    }
    assert not (hooks / "pre-commit").exists()


def test_install_refuses_orphan_backup_and_says_it_was_left(repo: Path) -> None:
    hooks = _hooks(repo)
    backup = _write(hooks / "pre-push.backup", ORIGINAL, 0o750)

    result = _invoke(repo, "install")

    assert result.exit_code == 1, result.output
    assert "Hooks installed successfully!" not in result.output
    assert _state_rows(result.output)["pre-push"] == "not installed (backup present)"
    assert ".git/hooks/pre-push.backup was left in place" in result.output
    assert not (hooks / "pre-push").exists()
    assert _unchanged(hooks / "pre-push.backup", backup, ORIGINAL)


def test_install_error_on_unsafe_hook_still_prints_state(repo: Path) -> None:
    (_hooks(repo) / "pre-push").mkdir()

    result = _invoke(repo, "install")

    assert result.exit_code == 1, result.output
    assert "Error" in result.output
    assert "Hooks installed successfully!" not in result.output
    assert _state_rows(result.output) == {
        "pre-push": "unsafe (not a regular file)",
        "pre-commit": "not installed",
    }


def test_uninstall_with_nothing_installed_does_not_claim_removal(
    repo: Path,
) -> None:
    result = _invoke(repo, "uninstall")

    assert result.exit_code == 0, result.output
    assert "Removed" not in result.output
    assert "Hooks uninstalled!" in result.output
    assert _state_rows(result.output) == dict.fromkeys(TEMPLATES, "not installed")


def test_uninstall_restores_the_saved_original(repo: Path) -> None:
    hooks = _hooks(repo)
    before = _write(hooks / "pre-push", FOREIGN, 0o700)
    assert _invoke(repo, "install", "--force").exit_code == 0

    result = _invoke(repo, "uninstall")

    assert result.exit_code == 0, result.output
    assert "Hooks uninstalled!" in result.output
    assert _state_rows(result.output) == {
        "pre-push": "installed (other)",
        "pre-commit": "not installed",
    }
    assert _unchanged(hooks / "pre-push", before, FOREIGN)
    assert not (hooks / "pre-push.backup").exists()


def test_uninstall_leaves_orphan_backup_with_a_notice(repo: Path) -> None:
    hooks = _hooks(repo)
    backup = _write(hooks / "pre-push.backup", ORIGINAL, 0o750)

    result = _invoke(repo, "uninstall")

    assert result.exit_code == 0, result.output
    assert ".git/hooks/pre-push.backup was left in place" in result.output
    assert _state_rows(result.output)["pre-push"] == "not installed (backup present)"
    assert _unchanged(hooks / "pre-push.backup", backup, ORIGINAL)


def test_uninstall_skip_does_not_claim_success(repo: Path) -> None:
    hooks = _hooks(repo)
    before = _write(hooks / "pre-push", FOREIGN, 0o700)

    result = _invoke(repo, "uninstall")

    assert result.exit_code == 0, result.output
    assert "Skipped: .git/hooks/pre-push" in result.output
    assert "Hooks uninstalled!" not in result.output
    assert "Some hooks were skipped" in result.output
    assert _unchanged(hooks / "pre-push", before, FOREIGN)


def test_uninstall_refusal_exits_one_with_state(repo: Path) -> None:
    hooks = _hooks(repo)
    _write(hooks / "pre-push", PRE_PUSH_HOOK_SCRIPT.encode(), 0o755)
    os.symlink("elsewhere", hooks / "pre-push.backup")

    result = _invoke(repo, "uninstall")

    assert result.exit_code == 1, result.output
    assert "Error" in result.output
    assert "Hooks uninstalled!" not in result.output
    assert _state_rows(result.output)["pre-push"] == "installed (traigent)"
    assert os.readlink(hooks / "pre-push.backup") == "elsewhere"


def _make_state(hooks: Path, state: str) -> None:
    hook = hooks / "pre-push"
    if state == "not installed":
        return
    if state == "not installed (backup present)":
        _write(hooks / "pre-push.backup", ORIGINAL, 0o750)
    elif state == "installed (traigent)":
        _write(hook, PRE_PUSH_HOOK_SCRIPT.encode(), 0o755)
    elif state == "installed (traigent, outdated)":
        _write(hook, PRE_PUSH_HOOK_SCRIPT.encode(), 0o644)
    elif state == "installed (other)":
        _write(hook, b"\x7fELF\x02\x01 binary, not UTF-8 \xff\xfe\n", 0o755)
    elif state == "unsafe (not a regular file)":
        os.symlink("elsewhere", hook)
    elif state == "unreadable":
        _write(hook, FOREIGN, 0o000)
        try:
            hook.read_bytes()
        except PermissionError:
            return
        pytest.skip("permission bits do not stop this process from reading")


@pytest.mark.parametrize(
    "state",
    [
        "not installed",
        "not installed (backup present)",
        "installed (traigent)",
        "installed (traigent, outdated)",
        "installed (other)",
        "unsafe (not a regular file)",
        "unreadable",
    ],
)
def test_status_command_shows_each_state_exactly(repo: Path, state: str) -> None:
    hooks = _hooks(repo)
    _make_state(hooks, state)
    before = {path.name: os.lstat(path) for path in hooks.iterdir()}

    result = _invoke(repo, "status")

    assert result.exit_code == 0, result.output
    assert _state_rows(result.output) == {
        "pre-push": state,
        "pre-commit": "not installed",
    }
    after = {path.name: os.lstat(path) for path in hooks.iterdir()}
    assert {
        name: (st.st_ino, stat.S_IMODE(st.st_mode), st.st_size)
        for name, st in after.items()
    } == {
        name: (st.st_ino, stat.S_IMODE(st.st_mode), st.st_size)
        for name, st in before.items()
    }


def test_status_command_shows_an_unknown_status_raw(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        HooksInstaller,
        "status",
        lambda self: {"pre-push": "mystery state", "pre-commit": "not installed"},
    )

    result = _invoke(repo, "status")

    assert result.exit_code == 0, result.output
    assert _state_rows(result.output)["pre-push"] == "mystery state"


def test_status_outside_a_repository_reports_the_error(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "status")

    assert "Not a Git repository" in result.output
    assert "not installed" not in result.output.lower()
