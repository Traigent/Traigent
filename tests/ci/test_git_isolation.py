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


def test_helper_ignores_git_config_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = tmp_path / "decoy.config"
    decoy.write_text("[core]\n\tbare = false\n")
    before = decoy.read_text()
    monkeypatch.setenv("GIT_CONFIG", str(decoy))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(decoy))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.email")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "injected@example.test")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'user.name=injected'")

    repo = tmp_path / "repo"
    repo.mkdir()
    helper = required_gate_tests.TestMergeQueueTopologyReplay._git
    helper(repo, "init", "-q")
    helper(repo, "config", "user.email", "ci@example.test")
    assert decoy.read_text() == before
    assert helper(repo, "config", "--get", "user.email") == "ci@example.test"


def test_scrubbed_env_drops_indexed_config_variables() -> None:
    base = {
        "GIT_CONFIG_KEY_3": "a",
        "GIT_CONFIG_VALUE_3": "b",
        "GIT_CONFIG": "f",
        "K": "1",
    }
    assert scrubbed_git_env(base) == {"K": "1"}
