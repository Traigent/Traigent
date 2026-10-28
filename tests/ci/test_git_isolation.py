"""Tmp-repo git helpers must not follow a ``GIT_DIR`` inherited from a git hook."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._git_isolation import REPO_REDIRECT_VARS, scrubbed_git_env
from tests.ci import test_check_required_gate as required_gate_tests

_GIT_ENV_SETUP = {
    "GIT_WORK_TREE": "work",
    "GIT_INDEX_FILE": "index",
    "GIT_OBJECT_DIRECTORY": "objects",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES": "alt",
    "GIT_COMMON_DIR": "common",
}


def _plain_git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=scrubbed_git_env(),
    ).stdout


def test_scrubbed_env_drops_every_redirecting_variable() -> None:
    base = dict.fromkeys(REPO_REDIRECT_VARS, "x") | {"KEEP": "1"}
    assert scrubbed_git_env(base) == {"KEEP": "1"}


def test_helper_ignores_an_inherited_git_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = tmp_path / "decoy.git"
    _plain_git(tmp_path, "init", "--bare", "-q", str(decoy))
    decoy_config = decoy / "config"
    before = decoy_config.read_text()
    monkeypatch.setenv("GIT_DIR", str(decoy))
    for name, value in _GIT_ENV_SETUP.items():
        monkeypatch.setenv(name, str(tmp_path / value))

    repo = tmp_path / "repo"
    repo.mkdir()
    helper = required_gate_tests.TestMergeQueueTopologyReplay._git
    helper(repo, "init", "-q")
    helper(repo, "config", "user.email", "ci@example.test")
    helper(repo, "config", "user.name", "CI")

    assert decoy_config.read_text() == before
    assert "ci@example.test" not in before
    assert (repo / ".git" / "config").read_text().count("ci@example.test") == 1
