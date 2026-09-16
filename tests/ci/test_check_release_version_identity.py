"""Regression tests for scripts/ci/check_release_version_identity.py (#2265).

Reproduces the bug directly: build a throwaway git repo, tag it as a
release, then add a public-API file on top without bumping the version --
exactly what PR #2226 did to v0.27.0. The check must fail in that state and
pass once the version is bumped (or once the tag itself is HEAD, or once no
public-API path changed).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_release_version_identity.py"

sys.path.insert(0, str(SCRIPT.parent))
import check_release_version_identity as identity_check  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "ci@example.com")
    _git(root, "config", "user.name", "CI Test")


def _write_pyproject(root: Path, version: str) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "traigent"\nversion = "{version}"\n'
    )


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _init_dir_layout(root: Path) -> None:
    (root / "traigent" / "api").mkdir(parents=True)
    (root / "traigent" / "__init__.py").write_text("# public surface\n")
    (root / "traigent" / "api" / "decorators.py").write_text("# decorators\n")
    (root / "traigent" / "core").mkdir(parents=True)
    (root / "traigent" / "core" / "internal.py").write_text("# not public\n")


class TestPublicApiChangedAfterRelease:
    """The exact #2265 shape: tag a release, then extend the public API
    without bumping the version."""

    def test_fails_when_public_api_changes_but_version_is_reused(self, tmp_path):
        root = tmp_path
        _init_repo(root)
        _init_dir_layout(root)
        _write_pyproject(root, "0.27.0")
        _commit(root, "release 0.27.0")
        _git(root, "tag", "v0.27.0")

        # PR #2226 equivalent: add a public API surface, version untouched.
        (root / "traigent" / "api" / "decorators.py").write_text(
            "# decorators\ntask_type: str | None = None\n"
        )
        _commit(root, "feat: add EvaluationOptions.task_type")

        ok, message = identity_check.check(root)

        assert ok is False
        assert "0.27.0" in message
        assert "v0.27.0" in message
        assert "traigent/api/decorators.py" in message

    def test_cli_exits_nonzero_on_the_same_repro(self, tmp_path):
        root = tmp_path
        _init_repo(root)
        _init_dir_layout(root)
        _write_pyproject(root, "0.27.0")
        _commit(root, "release 0.27.0")
        _git(root, "tag", "v0.27.0")
        (root / "traigent" / "__init__.py").write_text("# public surface\nNEW = 1\n")
        _commit(root, "feat: new public export")

        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "::error::" in result.stderr

    def test_passes_once_version_is_bumped_past_the_tag(self, tmp_path):
        root = tmp_path
        _init_repo(root)
        _init_dir_layout(root)
        _write_pyproject(root, "0.27.0")
        _commit(root, "release 0.27.0")
        _git(root, "tag", "v0.27.0")

        (root / "traigent" / "api" / "decorators.py").write_text(
            "# decorators\ntask_type: str | None = None\n"
        )
        _write_pyproject(root, "0.28.0.dev0")
        _commit(root, "feat: add task_type; bump dev version")

        ok, message = identity_check.check(root)

        assert ok is True
        assert "0.28.0.dev0" in message


class TestBenignCases:
    def test_passes_when_head_is_exactly_the_release_tag(self, tmp_path):
        root = tmp_path
        _init_repo(root)
        _init_dir_layout(root)
        _write_pyproject(root, "0.27.0")
        _commit(root, "release 0.27.0")
        _git(root, "tag", "v0.27.0")

        ok, message = identity_check.check(root)

        assert ok is True
        assert "correctly identifies" in message

    def test_passes_when_only_internal_files_change_after_release(self, tmp_path):
        root = tmp_path
        _init_repo(root)
        _init_dir_layout(root)
        _write_pyproject(root, "0.27.0")
        _commit(root, "release 0.27.0")
        _git(root, "tag", "v0.27.0")

        # Internal-only change: no public-API path touched.
        (root / "traigent" / "core" / "internal.py").write_text(
            "# not public\nFIX = 1\n"
        )
        _commit(root, "fix: internal-only bugfix")

        ok, message = identity_check.check(root)

        assert ok is True
        assert "still accurate" in message

    def test_passes_when_version_has_no_matching_tag_yet(self, tmp_path):
        root = tmp_path
        _init_repo(root)
        _init_dir_layout(root)
        _write_pyproject(root, "0.28.0.dev0")
        _commit(root, "start next dev cycle")

        ok, message = identity_check.check(root)

        assert ok is True
        assert "no matching tag" in message
