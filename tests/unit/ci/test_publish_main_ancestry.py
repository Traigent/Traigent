"""Every PyPI publish path must verify main ancestry before any upload step.

Regression for TraigentBackend#4402: the ancestry check was gated on
refs/tags/v*, so workflow_dispatch skipped it and could publish a commit that
is not on main.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "publish.yml"
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "assert_sha_on_main.sh"
_UPLOAD_ACTION = "pypa/gh-action-pypi-publish"


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    return workflow["jobs"]["publish"]["steps"]


def _is_upload(step: dict[str, Any]) -> bool:
    return _UPLOAD_ACTION in str(step.get("uses", ""))


def test_publish_job_has_upload_steps() -> None:
    assert sum(_is_upload(s) for s in _steps()) >= 2


def test_ancestry_check_is_unconditional_and_precedes_every_upload() -> None:
    steps = _steps()
    idxs = [i for i, s in enumerate(steps) if "assert_sha_on_main.sh" in str(s.get("run", ""))]
    assert len(idxs) == 1, "exactly one ancestry step expected"
    check = steps[idxs[0]]
    # No `if:` means it runs on tag push AND workflow_dispatch.
    assert "if" not in check
    assert not check.get("continue-on-error")
    uploads = [i for i, s in enumerate(steps) if _is_upload(s)]
    assert all(idxs[0] < u for u in uploads)


def test_ancestry_check_uses_full_history_checkout_and_passes_sha() -> None:
    steps = _steps()
    checkout = next(s for s in steps if "actions/checkout" in str(s.get("uses", "")))
    assert checkout["with"]["fetch-depth"] == 0
    check = next(s for s in steps if "assert_sha_on_main.sh" in str(s.get("run", "")))
    assert check["env"]["GIT_SHA"] == "${{ github.sha }}"
    assert '"$GIT_SHA"' in check["run"]


def test_actions_stay_pinned_and_permissions_not_widened() -> None:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["publish"]["permissions"] == {
        "id-token": "write",
        "contents": "read",
    }
    for s in _steps():
        uses = s.get("uses")
        if uses:
            assert len(uses.split("@")[1]) == 40, uses


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        cwd=cwd,
        env={**os.environ, "SKIP_FETCH": "1"},
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "a")
    on_main = _git(tmp_path, "rev-parse", "HEAD")
    # Emulate origin/main with a remote-tracking ref.
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", on_main)
    _git(tmp_path, "checkout", "-q", "-b", "feature")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "b")
    off_main = _git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, on_main, off_main


def test_script_accepts_commit_on_main(tmp_path: Path) -> None:
    repo, on_main, _ = _repo(tmp_path)
    assert _run(repo, on_main).returncode == 0


def test_script_rejects_commit_off_main(tmp_path: Path) -> None:
    repo, _, off_main = _repo(tmp_path)
    result = _run(repo, off_main)
    assert result.returncode != 0
    assert "not reachable" in result.stdout


def test_script_rejects_tag_pointing_elsewhere(tmp_path: Path) -> None:
    repo, on_main, off_main = _repo(tmp_path)
    _git(repo, "tag", "v1.0.0", off_main)
    assert _run(repo, on_main, "v1.0.0").returncode != 0
    _git(repo, "tag", "v1.0.1", on_main)
    assert _run(repo, on_main, "v1.0.1").returncode == 0
