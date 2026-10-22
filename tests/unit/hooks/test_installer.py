"""Unit tests for traigent.hooks.installer.

Tests for Git hooks installation and management, including pre-push and
pre-commit hook installation, uninstallation, and status checking.

The transition tests use an independent oracle: a literal copy of the adopted
per-hook transition table, expectations written in fixture ids, and raw
filesystem observations (bytes, mode, inode, presence). They never ask the
installer's own classifier what a file is.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability
# CONC-Quality-Reliability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import traigent.hooks.installer as installer_module
from traigent.cli.main import cli
from traigent.hooks.installer import (
    HOOK_MARKER,
    PRE_COMMIT_HOOK_SCRIPT,
    PRE_PUSH_HOOK_SCRIPT,
    HooksInstaller,
    find_git_root,
    install_hooks,
    uninstall_hooks,
)


class TestHooksInstaller:
    """Tests for HooksInstaller class."""

    @pytest.fixture
    def temp_git_repo(self) -> Generator[Path, None, None]:
        """Create temporary Git repository structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            git_dir = repo_path / ".git"
            git_dir.mkdir()
            hooks_dir = git_dir / "hooks"
            hooks_dir.mkdir()
            yield repo_path

    @pytest.fixture
    def temp_non_git_repo(self) -> Generator[Path, None, None]:
        """Create temporary non-Git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def installer(self, temp_git_repo: Path) -> HooksInstaller:
        """Create HooksInstaller instance with temp Git repo."""
        return HooksInstaller(temp_git_repo)

    # Initialization tests
    def test_initialization_with_path(self, temp_git_repo: Path) -> None:
        """Test HooksInstaller initializes with provided path."""
        installer = HooksInstaller(temp_git_repo)
        assert installer.repo_path == temp_git_repo.resolve()
        assert installer.hooks_dir == temp_git_repo / ".git" / "hooks"

    def test_initialization_with_string_path(self, temp_git_repo: Path) -> None:
        """Test HooksInstaller initializes with string path."""
        installer = HooksInstaller(str(temp_git_repo))
        assert installer.repo_path == temp_git_repo.resolve()

    @patch("traigent.hooks.installer.Path.cwd")
    def test_initialization_with_none_uses_cwd(self, mock_cwd: MagicMock) -> None:
        """Test HooksInstaller uses cwd when repo_path is None."""
        mock_path = MagicMock()
        mock_cwd.return_value = mock_path
        HooksInstaller(None)
        mock_cwd.assert_called_once()

    # is_git_repo tests
    def test_is_git_repo_returns_true_for_git_repo(
        self, installer: HooksInstaller
    ) -> None:
        """Test is_git_repo returns True for valid Git repository."""
        assert installer.is_git_repo() is True

    def test_is_git_repo_returns_false_for_non_git_repo(
        self, temp_non_git_repo: Path
    ) -> None:
        """Test is_git_repo returns False for non-Git directory."""
        installer = HooksInstaller(temp_non_git_repo)
        assert installer.is_git_repo() is False

    def test_is_git_repo_returns_false_when_git_is_file(
        self, temp_non_git_repo: Path
    ) -> None:
        """Test is_git_repo returns False when .git exists but is a file."""
        git_file = temp_non_git_repo / ".git"
        git_file.touch()
        installer = HooksInstaller(temp_non_git_repo)
        assert installer.is_git_repo() is False

    # get_hooks_dir tests
    def test_get_hooks_dir_returns_path(self, installer: HooksInstaller) -> None:
        """Test get_hooks_dir returns hooks directory path."""
        hooks_dir = installer.get_hooks_dir()
        assert hooks_dir == installer.hooks_dir
        assert hooks_dir.exists()

    def test_get_hooks_dir_creates_directory_if_missing(
        self, temp_git_repo: Path
    ) -> None:
        """Test get_hooks_dir creates hooks directory if it doesn't exist."""
        # Remove hooks directory
        hooks_dir = temp_git_repo / ".git" / "hooks"
        if hooks_dir.exists():
            hooks_dir.rmdir()

        installer = HooksInstaller(temp_git_repo)
        result = installer.get_hooks_dir()

        assert result.exists()
        assert result.is_dir()

    def test_get_hooks_dir_raises_for_non_git_repo(
        self, temp_non_git_repo: Path
    ) -> None:
        """Test get_hooks_dir raises RuntimeError for non-Git repository."""
        installer = HooksInstaller(temp_non_git_repo)
        with pytest.raises(RuntimeError, match="Not a Git repository"):
            installer.get_hooks_dir()

    # install tests
    def test_install_creates_both_hooks(self, installer: HooksInstaller) -> None:
        """Test install creates both pre-push and pre-commit hooks."""
        results = installer.install()

        assert results["pre-push"] is True
        assert results["pre-commit"] is True
        assert (installer.hooks_dir / "pre-push").exists()
        assert (installer.hooks_dir / "pre-commit").exists()

    def test_install_makes_hooks_executable(self, installer: HooksInstaller) -> None:
        """Test install sets executable permissions on hook files."""
        installer.install()

        pre_push_mode = (installer.hooks_dir / "pre-push").stat().st_mode
        pre_commit_mode = (installer.hooks_dir / "pre-commit").stat().st_mode

        assert pre_push_mode & stat.S_IXUSR
        assert pre_commit_mode & stat.S_IXUSR

    def test_install_writes_correct_content(self, installer: HooksInstaller) -> None:
        """Test install writes correct script content to hooks."""
        installer.install()

        pre_push_content = (installer.hooks_dir / "pre-push").read_text()
        pre_commit_content = (installer.hooks_dir / "pre-commit").read_text()

        assert pre_push_content == PRE_PUSH_HOOK_SCRIPT
        assert pre_commit_content == PRE_COMMIT_HOOK_SCRIPT
        assert HOOK_MARKER in pre_push_content
        assert "# Traigent pre-commit hook - quick config check" in pre_commit_content

    def test_install_without_force_skips_existing_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test install without force skips existing Traigent hooks."""
        # Install once
        installer.install()

        # Install again without force
        results = installer.install(force=False)

        assert results["pre-push"] is True
        assert results["pre-commit"] is True

    def test_install_without_force_preserves_non_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test install without force doesn't overwrite non-Traigent hooks."""
        # Create non-Traigent hook
        pre_push = installer.hooks_dir / "pre-push"
        pre_push.write_text("#!/bin/bash\necho 'custom hook'")

        results = installer.install(force=False)

        assert results["pre-push"] is False
        assert "custom hook" in pre_push.read_text()

    def test_install_with_force_overwrites_existing_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test install with force overwrites existing hooks."""
        # Create existing hook
        pre_push = installer.hooks_dir / "pre-push"
        pre_push.write_text("#!/bin/bash\necho 'old hook'")

        results = installer.install(force=True)

        assert results["pre-push"] is True
        content = pre_push.read_text()
        assert content == PRE_PUSH_HOOK_SCRIPT
        assert "old hook" not in content

    def test_install_with_force_creates_backup(self, installer: HooksInstaller) -> None:
        """Test install with force creates backup of existing hook."""
        # Create existing hook
        pre_push = installer.hooks_dir / "pre-push"
        original_content = "#!/bin/bash\necho 'original hook'"
        pre_push.write_text(original_content)

        installer.install(force=True)

        backup_path = installer.hooks_dir / "pre-push.backup"
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

    # _install_hook tests

    def test_install_publishes_exact_template_bytes_with_mode_0755(
        self, installer: HooksInstaller
    ) -> None:
        """Install writes the raw LF template bytes and an executable 0755 mode."""
        installer.install()

        for name, script in (
            ("pre-push", PRE_PUSH_HOOK_SCRIPT),
            ("pre-commit", PRE_COMMIT_HOOK_SCRIPT),
        ):
            hook_path = installer.hooks_dir / name
            assert hook_path.read_bytes() == script.encode()
            assert stat.S_IMODE(hook_path.stat().st_mode) == 0o755

    # uninstall tests
    def test_uninstall_removes_traigent_hooks(self, installer: HooksInstaller) -> None:
        """Test uninstall removes both generated Traigent hooks."""
        installer.install()
        results = installer.uninstall()

        assert results["pre-push"] is True
        assert results["pre-commit"] is True
        assert not (installer.hooks_dir / "pre-push").exists()
        assert not (installer.hooks_dir / "pre-commit").exists()

    def test_uninstall_preserves_non_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test uninstall doesn't remove non-Traigent hooks."""
        # Create non-Traigent hook
        pre_push = installer.hooks_dir / "pre-push"
        custom_content = "#!/bin/bash\necho 'custom'"
        pre_push.write_text(custom_content)

        results = installer.uninstall()

        assert results["pre-push"] is False
        assert pre_push.exists()
        assert pre_push.read_text() == custom_content

    def test_uninstall_returns_true_for_missing_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test uninstall returns True when hooks don't exist."""
        results = installer.uninstall()

        assert results["pre-push"] is True
        assert results["pre-commit"] is True

    def test_uninstall_restores_backup_if_exists(
        self, installer: HooksInstaller
    ) -> None:
        """Test uninstall restores backup file if present."""
        # Create original hook and install with force
        pre_push = installer.hooks_dir / "pre-push"
        original_content = "#!/bin/bash\necho 'original'"
        pre_push.write_text(original_content)
        installer.install(force=True)

        # Uninstall
        installer.uninstall()

        # Check backup was restored
        assert pre_push.exists()
        assert pre_push.read_text() == original_content
        assert not (installer.hooks_dir / "pre-push.backup").exists()

    # status tests
    def test_status_returns_not_installed_for_missing_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test status returns 'not installed' for missing hooks."""
        status = installer.status()

        assert status["pre-push"] == "not installed"
        assert status["pre-commit"] == "not installed"

    def test_status_returns_installed_traigent_for_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test status returns 'installed (traigent)' for hooks."""
        installer.install()
        status = installer.status()

        assert status["pre-push"] == "installed (traigent)"
        assert status["pre-commit"] == "installed (traigent)"

    def test_status_returns_installed_other_for_non_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test status returns 'installed (other)' for non-Traigent hooks."""
        pre_push = installer.hooks_dir / "pre-push"
        pre_push.write_text("#!/bin/bash\necho 'custom'")

        status = installer.status()

        assert status["pre-push"] == "installed (other)"

    def test_status_returns_error_for_non_git_repo(
        self, temp_non_git_repo: Path
    ) -> None:
        """Test status returns error for non-Git repository."""
        installer = HooksInstaller(temp_non_git_repo)
        status = installer.status()

        assert "error" in status
        assert "Not a Git repository" in status["error"]


class TestFindGitRoot:
    """Tests for find_git_root helper function."""

    @pytest.fixture
    def git_repo_structure(self) -> Generator[Path, None, None]:
        """Create nested Git repository structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            repo_root.mkdir()
            (repo_root / ".git").mkdir()

            # Create nested directories
            nested = repo_root / "src" / "deep" / "nested"
            nested.mkdir(parents=True)

            yield repo_root

    def test_find_git_root_from_repo_root(self, git_repo_structure: Path) -> None:
        """Test find_git_root finds root from repository root."""
        result = find_git_root(git_repo_structure)
        assert result == git_repo_structure

    def test_find_git_root_from_nested_directory(
        self, git_repo_structure: Path
    ) -> None:
        """Test find_git_root finds root from nested directory."""
        nested = git_repo_structure / "src" / "deep" / "nested"
        result = find_git_root(nested)
        assert result == git_repo_structure

    def test_find_git_root_returns_none_for_non_git_directory(self) -> None:
        """Test find_git_root returns None when not in Git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir).resolve()
            real_exists = Path.exists

            def exists_without_external_git_roots(path: Path) -> bool:
                if path.name == ".git":
                    try:
                        path.parent.resolve().relative_to(tmp_root)
                    except ValueError:
                        return False
                return real_exists(path)

            with patch.object(Path, "exists", exists_without_external_git_roots):
                result = find_git_root(Path(tmpdir))
                assert result is None

    @patch("traigent.hooks.installer.Path.cwd")
    def test_find_git_root_uses_cwd_when_none(
        self, mock_cwd: MagicMock, git_repo_structure: Path
    ) -> None:
        """Test find_git_root uses current directory when start_path is None."""
        mock_cwd.return_value = git_repo_structure
        result = find_git_root(None)
        assert result == git_repo_structure

    def test_find_git_root_stops_at_filesystem_root(self) -> None:
        """Test find_git_root stops at filesystem root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir).resolve()
            real_exists = Path.exists

            def exists_without_external_git_roots(path: Path) -> bool:
                if path.name == ".git":
                    try:
                        path.parent.resolve().relative_to(tmp_root)
                    except ValueError:
                        return False
                return real_exists(path)

            deep_path = tmp_root / "a" / "b" / "c" / "d" / "e"
            deep_path.mkdir(parents=True)
            with patch.object(Path, "exists", exists_without_external_git_roots):
                result = find_git_root(deep_path)
                assert result is None


class TestInstallHooks:
    """Tests for install_hooks convenience function."""

    @pytest.fixture
    def temp_git_repo(self) -> Generator[Path, None, None]:
        """Create temporary Git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            git_dir = repo_path / ".git"
            git_dir.mkdir()
            (git_dir / "hooks").mkdir()
            yield repo_path

    def test_install_hooks_with_explicit_path(self, temp_git_repo: Path) -> None:
        """Test install_hooks succeeds with explicit repository path."""
        result = install_hooks(temp_git_repo)

        assert result is True
        assert (temp_git_repo / ".git" / "hooks" / "pre-push").exists()

    def test_install_hooks_with_string_path(self, temp_git_repo: Path) -> None:
        """Test install_hooks accepts string path."""
        result = install_hooks(str(temp_git_repo))

        assert result is True

    def test_install_hooks_with_force_flag(self, temp_git_repo: Path) -> None:
        """Test install_hooks respects force flag."""
        # Create existing hook
        pre_push = temp_git_repo / ".git" / "hooks" / "pre-push"
        pre_push.write_text("#!/bin/bash\necho 'old'")

        result = install_hooks(temp_git_repo, force=True)

        assert result is True
        assert HOOK_MARKER in pre_push.read_text()

    @patch("traigent.hooks.installer.find_git_root")
    def test_install_hooks_auto_detects_repo(
        self, mock_find: MagicMock, temp_git_repo: Path
    ) -> None:
        """Test install_hooks auto-detects repository when repo_path is None."""
        mock_find.return_value = temp_git_repo

        result = install_hooks(None)

        assert result is True
        mock_find.assert_called_once()

    @patch("traigent.hooks.installer.find_git_root")
    def test_install_hooks_returns_false_when_no_repo_found(
        self, mock_find: MagicMock
    ) -> None:
        """Test install_hooks returns False when Git repository not found."""
        mock_find.return_value = None

        result = install_hooks(None)

        assert result is False

    @patch("traigent.hooks.installer.HooksInstaller.install")
    def test_install_hooks_returns_false_on_exception(
        self, mock_install: MagicMock, temp_git_repo: Path
    ) -> None:
        """Test install_hooks returns False when exception occurs."""
        mock_install.side_effect = Exception("Installation failed")

        result = install_hooks(temp_git_repo)

        assert result is False

    def test_install_hooks_returns_false_when_partial_failure(
        self, temp_git_repo: Path
    ) -> None:
        """Test install_hooks returns False when some hooks fail to install."""
        # Create non-Traigent pre-push hook
        pre_push = temp_git_repo / ".git" / "hooks" / "pre-push"
        pre_push.write_text("#!/bin/bash\necho 'custom'")

        # Install without force (pre-push will fail, pre-commit will succeed)
        result = install_hooks(temp_git_repo, force=False)

        assert result is False


class TestUninstallHooks:
    """Tests for uninstall_hooks convenience function."""

    @pytest.fixture
    def temp_git_repo_with_hooks(self) -> Generator[Path, None, None]:
        """Create temp Git repository with Traigent hooks installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            git_dir = repo_path / ".git"
            git_dir.mkdir()
            hooks_dir = git_dir / "hooks"
            hooks_dir.mkdir()

            # Install hooks
            installer = HooksInstaller(repo_path)
            installer.install()

            yield repo_path

    def test_uninstall_hooks_with_explicit_path(
        self, temp_git_repo_with_hooks: Path
    ) -> None:
        """Test uninstall_hooks with explicit path."""
        result = uninstall_hooks(temp_git_repo_with_hooks)

        assert result is True
        assert not (temp_git_repo_with_hooks / ".git" / "hooks" / "pre-push").exists()
        assert not (temp_git_repo_with_hooks / ".git" / "hooks" / "pre-commit").exists()

    def test_uninstall_hooks_with_string_path(
        self, temp_git_repo_with_hooks: Path
    ) -> None:
        """Test uninstall_hooks accepts string path."""
        result = uninstall_hooks(str(temp_git_repo_with_hooks))

        assert result is True

    @patch("traigent.hooks.installer.find_git_root")
    def test_uninstall_hooks_auto_detects_repo(
        self, mock_find: MagicMock, temp_git_repo_with_hooks: Path
    ) -> None:
        """Test uninstall_hooks auto-detects repository."""
        mock_find.return_value = temp_git_repo_with_hooks

        result = uninstall_hooks(None)

        assert result is True
        mock_find.assert_called_once()

    @patch("traigent.hooks.installer.find_git_root")
    def test_uninstall_hooks_returns_false_when_no_repo_found(
        self, mock_find: MagicMock
    ) -> None:
        """Test uninstall_hooks returns False when Git repository not found."""
        mock_find.return_value = None

        result = uninstall_hooks(None)

        assert result is False

    @patch("traigent.hooks.installer.HooksInstaller.uninstall")
    def test_uninstall_hooks_returns_false_on_exception(
        self, mock_uninstall: MagicMock, temp_git_repo_with_hooks: Path
    ) -> None:
        """Test uninstall_hooks returns False when exception occurs."""
        mock_uninstall.side_effect = Exception("Uninstallation failed")

        result = uninstall_hooks(temp_git_repo_with_hooks)

        assert result is False


class TestHookScripts:
    """Tests for hook script constants."""

    def test_pre_push_hook_script_contains_marker(self) -> None:
        """Test PRE_PUSH_HOOK_SCRIPT contains Traigent marker."""
        assert HOOK_MARKER in PRE_PUSH_HOOK_SCRIPT

    def test_pre_push_hook_script_has_shebang(self) -> None:
        """Test PRE_PUSH_HOOK_SCRIPT starts with bash shebang."""
        assert PRE_PUSH_HOOK_SCRIPT.startswith("#!/bin/bash")

    def test_pre_push_hook_script_validates_config(self) -> None:
        """Test PRE_PUSH_HOOK_SCRIPT contains validation command."""
        assert "hooks validate" in PRE_PUSH_HOOK_SCRIPT

    def test_pre_commit_hook_script_contains_traigent(self) -> None:
        """Test PRE_COMMIT_HOOK_SCRIPT contains Traigent identifier."""
        # Note: pre-commit doesn't have HOOK_MARKER
        assert "Traigent" in PRE_COMMIT_HOOK_SCRIPT

    def test_pre_commit_hook_script_has_shebang(self) -> None:
        """Test PRE_COMMIT_HOOK_SCRIPT starts with bash shebang."""
        assert PRE_COMMIT_HOOK_SCRIPT.startswith("#!/bin/bash")

    def test_pre_commit_hook_script_uses_quick_check(self) -> None:
        """Test PRE_COMMIT_HOOK_SCRIPT uses quick check flag."""
        assert "--quick" in PRE_COMMIT_HOOK_SCRIPT

    def test_hook_marker_is_descriptive(self) -> None:
        """Test HOOK_MARKER contains descriptive text."""
        assert "Traigent" in HOOK_MARKER
        assert "pre-push" in HOOK_MARKER


@pytest.mark.parametrize("hook_name", ["pre-push", "pre-commit"])
@pytest.mark.parametrize(
    "lookalike",
    [
        "unmarked",
        "body_marker",
        "quoted_marker",
        "wrong_kind",
        "missing_installer",
        "missing_shebang",
        "wrong_installer",
        "unknown_header",
    ],
)
def test_foreign_hook_bytes_survive_all_ownership_readers(
    tmp_path: Path, hook_name: str, lookalike: str
) -> None:
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    script = PRE_PUSH_HOOK_SCRIPT if hook_name == "pre-push" else PRE_COMMIT_HOOK_SCRIPT
    header = script.splitlines()[1]
    installer_line = "# Installed by: traigent hooks install"
    foreign = {
        "unmarked": "#!/bin/bash\necho 'custom hook'\n",
        "body_marker": f"#!/bin/bash\necho 'custom hook'\n{header}\n{installer_line}\n",
        "quoted_marker": f"#!/bin/bash\nprintf '%s\\n' {shlex.quote(header)}\n",
        "wrong_kind": (
            PRE_COMMIT_HOOK_SCRIPT if hook_name == "pre-push" else PRE_PUSH_HOOK_SCRIPT
        ),
        "missing_installer": f"#!/bin/bash\n{header}\nexit 0\n",
        "missing_shebang": f"{header}\n{installer_line}\nexit 0\n",
        "wrong_installer": f"#!/bin/bash\n{header}\n# Installed by: custom tool\n",
        "unknown_header": f"#!/bin/bash\n# Traigent unknown hook\n{installer_line}\n",
    }[lookalike].encode()
    hook_path = hooks_dir / hook_name
    hook_path.write_bytes(foreign)
    hook_path.chmod(0o750)
    before = hook_path.stat()
    installer = HooksInstaller(tmp_path)

    assert installer.status()[hook_name] == "installed (other)"
    assert installer.install(force=False)[hook_name] is False
    assert hook_path.read_bytes() == foreign
    assert installer.uninstall()[hook_name] is False
    after = hook_path.stat()
    assert hook_path.read_bytes() == foreign
    assert (after.st_mode, after.st_ino) == (before.st_mode, before.st_ino)


@pytest.mark.parametrize("hook_name", ["pre-push", "pre-commit"])
def test_force_install_then_uninstall_restores_custom_hook_bytes(
    tmp_path: Path, hook_name: str
) -> None:
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    hook_path = hooks_dir / hook_name
    foreign = b"#!/bin/bash\r\nprintf 'custom hook\\n'\r\n"
    hook_path.write_bytes(foreign)
    hook_path.chmod(0o750)
    before = hook_path.stat()
    installer = HooksInstaller(tmp_path)

    assert all(installer.install(force=True).values())
    assert hook_path.with_suffix(".backup").read_bytes() == foreign
    assert installer.status()[hook_name] == "installed (traigent)"
    assert all(installer.uninstall().values())
    after = hook_path.stat()
    assert hook_path.read_bytes() == foreign
    assert (after.st_mode, after.st_ino) == (before.st_mode, before.st_ino)
    assert not hook_path.with_suffix(".backup").exists()
    assert installer.status()[hook_name] == "installed (other)"


# ---------------------------------------------------------------------------
# Exact-artifact ownership: fixtures and raw filesystem oracle
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "hooks"
KINDS = ("pre-push", "pre-commit")

# Captured at import, before any test wraps the functions the installer calls,
# so the oracle and the injectors never observe or trigger their own wrappers.
_lstat = os.lstat
_link = os.link
_replace = os.replace
_unlink = os.unlink
_chmod = os.chmod
_symlink = os.symlink
_readlink = os.readlink

# Digests recorded in tests/fixtures/hooks/PROVENANCE.md (AST extraction of
# every template literal in the installer's history, plus exact CRLF forms).
FIXTURE_SHA256 = {
    "pre-push.18bfcfdfe7e3.lf.hook": "18bfcfdfe7e36238286e4aeeb65bdef3e04660c58b477f0856a40e640e26095c",
    "pre-push.18bfcfdfe7e3.crlf.hook": "b9d8398357c53b8b128ba0ca6d275a804cd0f674f94037a84333f673605d5b72",
    "pre-commit.b7b26bf4e881.lf.hook": "b7b26bf4e881ce5923c3d2d506488e6e0deec538afde892b58333f3d28319ee0",
    "pre-commit.b7b26bf4e881.crlf.hook": "9ddbae8ca0016f02820662bb5a18af416c938db7fab74614f5b98c515ab4ef18",
    "pre-push.bdf3be4f4ae4.lf.hook": "bdf3be4f4ae44aa15100ecc6a3409b448ea442b1a048bb5816c0fc0d5a8a069c",
    "pre-push.bdf3be4f4ae4.crlf.hook": "5ea26895d95b571ca10ee0625d0e7b44f1d312f378d92cd0c80255cd4dd2b092",
    "pre-commit.105f68f2146a.lf.hook": "105f68f2146a23e5a0cfae9ad5459d9721f55e1cd2127a66c26e7fb5e07c96e9",
    "pre-commit.105f68f2146a.crlf.hook": "da32c293794c1a030111120ab5a85950c8268bf61bd5ff65673f123a9db96c82",
}
CURRENT_TEMPLATE_SHA256 = {
    "pre-push": "18bfcfdfe7e36238286e4aeeb65bdef3e04660c58b477f0856a40e640e26095c",
    "pre-commit": "b7b26bf4e881ce5923c3d2d506488e6e0deec538afde892b58333f3d28319ee0",
}
# "released" shipped in v0.9.0-v0.30.0; "initial" is the earlier pair (v0.8.0-rc1).
LEGACY_FIXTURE = {
    ("pre-push", "released"): "pre-push.18bfcfdfe7e3",
    ("pre-push", "initial"): "pre-push.bdf3be4f4ae4",
    ("pre-commit", "released"): "pre-commit.b7b26bf4e881",
    ("pre-commit", "initial"): "pre-commit.105f68f2146a",
}

FOREIGN_ACTIVE = b"#!/bin/sh\n# customer hook F1\necho 'customer check'\n"
FOREIGN_BACKUP = b"#!/bin/sh\n# customer original F0\nexit 0\n"
BINARY_HOOK = b"\x7fELF\x02\x01\x01\x00\xff\xfe binary, not UTF-8 \x80\x81\n"

# Fixture id -> class in the adopted table. Assigned by hand from how each
# fixture is built below, never by asking the installer.
#   A absent | O exact generated bytes for this kind (current or historical)
#   F any other regular file | X not a regular file, or unreadable
ACTIVE_CLASS = {
    "absent": "A",
    "current": "O",
    "current-not-executable": "O",
    "legacy-released-lf": "O",
    "legacy-released-crlf": "O",
    "legacy-initial-lf": "O",
    "legacy-initial-crlf": "O",
    "foreign": "F",
    "header-edited": "F",
    # Historical text-mode writers emitted this on Windows, so it is one of
    # ours; see tests/fixtures/hooks/PROVENANCE.md.
    "current-crlf": "O",
    "binary": "F",
    "empty": "F",
    "other-kind": "F",
    "symlink-to-current": "X",
    "dangling-symlink": "X",
    "directory": "X",
    "unreadable": "X",
}
#   F= a backup that is the same inode as a foreign active hook (interrupted takeover)
BACKUP_CLASS = {
    "absent": "A",
    "current": "O",
    "legacy-released-lf": "O",
    "legacy-initial-crlf": "O",
    "foreign": "F",
    "copy-of-active": "F",
    "same-inode": "F=",
    "symlink-to-current": "X",
}
BACKUPS_OF_FOREIGN_ACTIVE = {"copy-of-active", "same-inode"}

# Literal copy of the adopted per-hook transition table.
# Each value is (result, active slot afterwards, backup slot afterwards).
#   result: True/False is the hook's value in the returned dict;
#           "refuse-backup"/"refuse-active" is a RuntimeError naming that slot.
#   slot:   "same" the identical object (bytes, mode, inode) | "absent"
#           "new" the current template with mode 0755
#           "backup" the object that was in the backup slot
#           "active" the object that was in the active slot
_REFUSE_BACKUP = ("refuse-backup", "same", "same")
_REFUSE_ACTIVE = ("refuse-active", "same", "same")
TRANSITIONS: dict[tuple[str, str], dict[str, tuple[Any, str, str]]] = {
    ("A", "A"): {
        "install": (True, "new", "absent"),
        "force": (True, "new", "absent"),
        "uninstall": (True, "absent", "absent"),
    },
    ("A", "O"): {
        "install": (True, "new", "absent"),
        "force": (True, "new", "absent"),
        "uninstall": (True, "absent", "absent"),
    },
    ("A", "F"): {
        "install": _REFUSE_BACKUP,
        "force": _REFUSE_BACKUP,
        "uninstall": (True, "absent", "same"),
    },
    ("A", "X"): {
        "install": _REFUSE_BACKUP,
        "force": _REFUSE_BACKUP,
        "uninstall": (True, "absent", "same"),
    },
    ("O", "A"): {
        "install": (True, "new", "absent"),
        "force": (True, "new", "absent"),
        "uninstall": (True, "absent", "absent"),
    },
    ("O", "O"): {
        "install": (True, "new", "absent"),
        "force": (True, "new", "absent"),
        "uninstall": (True, "absent", "absent"),
    },
    # uninstall leaves a backup it has no record of exactly where it is, and
    # removes only the generated hook -- the same answer as ("A", "F") above.
    # Restoring is reserved for a backup the installer recorded as the
    # customer's; bytes alone cannot tell their hook from an artifact some
    # unrecognised release generated.
    ("O", "F"): {
        "install": (True, "new", "same"),
        "force": (True, "new", "same"),
        "uninstall": (True, "absent", "same"),
    },
    ("O", "X"): {
        "install": (True, "new", "same"),
        "force": (True, "new", "same"),
        "uninstall": _REFUSE_BACKUP,
    },
    ("F", "A"): {
        "install": (False, "same", "absent"),
        "force": (True, "new", "active"),
        "uninstall": (False, "same", "absent"),
    },
    ("F", "O"): {
        "install": (False, "same", "same"),
        "force": (True, "new", "active"),
        "uninstall": (False, "same", "absent"),
    },
    ("F", "F"): {
        "install": (False, "same", "same"),
        "force": _REFUSE_BACKUP,
        "uninstall": (False, "same", "same"),
    },
    ("F", "F="): {
        "install": (False, "same", "same"),
        "force": (True, "new", "same"),
        "uninstall": (False, "same", "same"),
    },
    ("F", "X"): {
        "install": (False, "same", "same"),
        "force": _REFUSE_BACKUP,
        "uninstall": (False, "same", "same"),
    },
    ("X", "A"): {
        "install": _REFUSE_ACTIVE,
        "force": _REFUSE_ACTIVE,
        "uninstall": _REFUSE_ACTIVE,
    },
    ("X", "O"): {
        "install": _REFUSE_ACTIVE,
        "force": _REFUSE_ACTIVE,
        "uninstall": _REFUSE_ACTIVE,
    },
    ("X", "F"): {
        "install": _REFUSE_ACTIVE,
        "force": _REFUSE_ACTIVE,
        "uninstall": _REFUSE_ACTIVE,
    },
    ("X", "X"): {
        "install": _REFUSE_ACTIVE,
        "force": _REFUSE_ACTIVE,
        "uninstall": _REFUSE_ACTIVE,
    },
}

# Exact status() strings by active fixture; an absent active hook reports
# "not installed (backup present)" whenever anything occupies the backup name.
EXPECTED_STATUS = {
    "absent": "not installed",
    "current": "installed (traigent)",
    "current-not-executable": "installed (traigent, outdated)",
    # Byte-identical to the current template, so not "outdated".
    "legacy-released-lf": "installed (traigent)",
    "legacy-released-crlf": "installed (traigent, outdated)",
    "legacy-initial-lf": "installed (traigent, outdated)",
    "legacy-initial-crlf": "installed (traigent, outdated)",
    "foreign": "installed (other)",
    "header-edited": "installed (other)",
    "current-crlf": "installed (traigent, outdated)",
    "binary": "installed (other)",
    "empty": "installed (other)",
    "other-kind": "installed (other)",
    "symlink-to-current": "unsafe (not a regular file)",
    "dangling-symlink": "unsafe (not a regular file)",
    "directory": "unsafe (not a regular file)",
    "unreadable": "unreadable",
}


@dataclass(frozen=True)
class Obj:
    """One directory entry as the raw filesystem reports it."""

    kind: str  # absent | file | symlink | dir | other
    data: bytes = b""  # file bytes, symlink target, or directory listing
    mode: int = 0
    ino: int = 0
    dev: int = 0
    nlink: int = 0

    def same_object(self, other: Obj) -> bool:
        # nlink is excluded: a hard link made or removed by the installer
        # changes the count without changing the object.
        return (self.kind, self.data, self.mode, self.ino, self.dev) == (
            other.kind,
            other.data,
            other.mode,
            other.ino,
            other.dev,
        )

    def describe(self) -> str:
        digest = hashlib.sha256(self.data).hexdigest()[:12]
        return (
            f"{self.kind}(sha256={digest}, len={len(self.data)}, "
            f"mode={self.mode:o}, ino={self.ino}, nlink={self.nlink})"
        )


ABSENT = Obj("absent")


def _template(kind: str) -> bytes:
    return {"pre-push": PRE_PUSH_HOOK_SCRIPT, "pre-commit": PRE_COMMIT_HOOK_SCRIPT}[
        kind
    ].encode()


def _other(kind: str) -> str:
    return KINDS[1 - KINDS.index(kind)]


def _legacy_bytes(kind: str, which: str, newline: str) -> bytes:
    return (FIXTURES / f"{LEGACY_FIXTURE[kind, which]}.{newline}.hook").read_bytes()


def _read_raw(path: Path, mode: int) -> bytes:
    if mode & stat.S_IRUSR:
        return path.read_bytes()
    _chmod(path, mode | stat.S_IRUSR)
    try:
        return path.read_bytes()
    finally:
        _chmod(path, mode)


def _obj(path: Path) -> Obj:
    try:
        st = _lstat(path)
    except FileNotFoundError:
        return ABSENT
    mode = stat.S_IMODE(st.st_mode)
    if stat.S_ISLNK(st.st_mode):
        kind, data = "symlink", os.fsencode(_readlink(path))
    elif stat.S_ISDIR(st.st_mode):
        kind, data = "dir", "\n".join(sorted(os.listdir(path))).encode()
    elif stat.S_ISREG(st.st_mode):
        kind, data = "file", _read_raw(path, mode)
    else:
        kind, data = "other", b""
    return Obj(kind, data, mode, st.st_ino, st.st_dev, st.st_nlink)


def _tree(root: Path) -> dict[str, Obj]:
    """Every entry under root without following symlinks."""
    entries = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            entries[str(path.relative_to(root))] = _obj(path)
    return entries


def _is_new(obj: Obj, kind: str) -> bool:
    return (obj.kind, obj.data, obj.mode, obj.nlink) == (
        "file",
        _template(kind),
        0o755,
        1,
    )


def _hooks_dir(repo: Path) -> Path:
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    return hooks_dir


def _write(path: Path, data: bytes, mode: int) -> None:
    path.write_bytes(data)
    _chmod(path, mode)


def _require_unreadable(path: Path) -> None:
    try:
        path.read_bytes()
    except PermissionError:
        return
    pytest.skip(
        "File permission bits do not stop this process from reading (for "
        "example when running as root), so an unreadable hook cannot be built"
    )


def _make_active(hooks_dir: Path, kind: str, fixture_id: str) -> None:
    path = hooks_dir / kind
    template = _template(kind)
    regular = {
        "current": (template, 0o755),
        "current-not-executable": (template, 0o644),
        "foreign": (FOREIGN_ACTIVE, 0o700),
        "header-edited": (template + b"echo 'customer addition'\n", 0o755),
        "current-crlf": (template.replace(b"\n", b"\r\n"), 0o755),
        "binary": (BINARY_HOOK, 0o755),
        "empty": (b"", 0o644),
        "other-kind": (_template(_other(kind)), 0o755),
    }
    if fixture_id == "absent":
        return
    if fixture_id.startswith("legacy-"):
        _, which, newline = fixture_id.split("-")
        _write(path, _legacy_bytes(kind, which, newline), 0o755)
    elif fixture_id in regular:
        _write(path, *regular[fixture_id])
    elif fixture_id == "symlink-to-current":
        _write(hooks_dir / "decoy-target", template, 0o755)
        _symlink("decoy-target", path)
    elif fixture_id == "dangling-symlink":
        _symlink("decoy-missing", path)
    elif fixture_id == "directory":
        path.mkdir()
        _write(path / "kept", template, 0o755)
    elif fixture_id == "unreadable":
        _write(path, FOREIGN_ACTIVE, 0o000)
        _require_unreadable(path)
    else:
        raise AssertionError(f"unknown active fixture {fixture_id}")


def _make_backup(hooks_dir: Path, kind: str, fixture_id: str) -> None:
    path = hooks_dir / f"{kind}.backup"
    active = hooks_dir / kind
    if fixture_id == "absent":
        return
    if fixture_id == "current":
        _write(path, _template(kind), 0o755)
    elif fixture_id == "legacy-released-lf":
        _write(path, _legacy_bytes(kind, "released", "lf"), 0o755)
    elif fixture_id == "legacy-initial-crlf":
        _write(path, _legacy_bytes(kind, "initial", "crlf"), 0o755)
    elif fixture_id == "foreign":
        _write(path, FOREIGN_BACKUP, 0o750)
    elif fixture_id == "copy-of-active":
        _write(path, active.read_bytes(), stat.S_IMODE(_lstat(active).st_mode))
    elif fixture_id == "same-inode":
        _link(active, path)
    elif fixture_id == "symlink-to-current":
        _write(hooks_dir / "decoy-backup-target", _template(kind), 0o755)
        _symlink("decoy-backup-target", path)
    else:
        raise AssertionError(f"unknown backup fixture {fixture_id}")


def _run(installer: HooksInstaller, action: str) -> tuple[Any, BaseException | None]:
    try:
        if action == "uninstall":
            return installer.uninstall(), None
        return installer.install(force=action == "force"), None
    except Exception as error:  # the table decides which errors are expected
        return None, error


def _assert_refusal(cell: str, error: BaseException | None, slot: str) -> None:
    assert isinstance(error, RuntimeError), f"{cell}: expected a refusal, got {error!r}"
    message = str(error)
    assert re.search(rf"(?<![\w-]){re.escape(slot)}(?!\.backup)(?![\w-])", message), (
        f"{cell}: refusal does not name {slot}: {message}"
    )
    assert "--force" not in message, f"{cell}: refusal suggests --force: {message}"


def _assert_slot(
    label: str,
    got: Obj,
    token: str,
    kind: str,
    was_self: Obj,
    was_active: Obj,
    was_backup: Obj,
) -> None:
    expected = {
        "same": was_self,
        "absent": ABSENT,
        "backup": was_backup,
        "active": was_active,
    }
    if token == "new":
        assert _is_new(got, kind), (
            f"{label}: expected the current template, got {got.describe()}"
        )
        return
    assert got.same_object(expected[token]), (
        f"{label}: expected {token} {expected[token].describe()}, got {got.describe()}"
    )


def _assert_unmanaged_untouched(
    cell: str, kind: str, before: dict[str, Obj], after: dict[str, Obj]
) -> None:
    managed = {kind, f"{kind}.backup", _other(kind), f"{_other(kind)}.backup"}
    kept_before = {name: obj for name, obj in before.items() if name not in managed}
    kept_after = {name: obj for name, obj in after.items() if name not in managed}
    assert kept_after.keys() == kept_before.keys(), (
        f"{cell}: other names appeared or disappeared: "
        f"{sorted(kept_after.keys() ^ kept_before.keys())}"
    )
    for name, obj in kept_before.items():
        assert obj.same_object(kept_after[name]), f"{cell}: {name} was modified"


def _assert_other_kind(
    cell: str, kind: str, after: dict[str, Obj], *, installed: bool
) -> None:
    other = _other(kind)
    assert f"{other}.backup" not in after, f"{cell}: {other}.backup was created"
    if installed:
        assert _is_new(after.get(other, ABSENT), other), (
            f"{cell}: {other} not installed"
        )
    else:
        assert other not in after, f"{cell}: {other} was touched"


def _transition_cells() -> Iterator[Any]:
    for kind in KINDS:
        for active_id, active_class in ACTIVE_CLASS.items():
            for backup_id in BACKUP_CLASS:
                if backup_id in BACKUPS_OF_FOREIGN_ACTIVE and active_class != "F":
                    continue
                for action in ("install", "force", "uninstall"):
                    yield pytest.param(
                        kind,
                        active_id,
                        backup_id,
                        action,
                        id=f"{kind}:{active_id}:{backup_id}:{action}",
                    )


@pytest.mark.parametrize(
    ("kind", "active_id", "backup_id", "action"), list(_transition_cells())
)
def test_transition_table(
    tmp_path: Path, kind: str, active_id: str, backup_id: str, action: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    _make_active(hooks_dir, kind, active_id)
    _make_backup(hooks_dir, kind, backup_id)
    expected_result, expected_active, expected_backup = TRANSITIONS[
        ACTIVE_CLASS[active_id], BACKUP_CLASS[backup_id]
    ][action]
    cell = f"{kind} {active_id}/{backup_id} {action}"
    before = _tree(hooks_dir)

    result, error = _run(HooksInstaller(tmp_path), action)

    after = _tree(hooks_dir)
    if isinstance(expected_result, str):
        slot = kind if expected_result == "refuse-active" else f"{kind}.backup"
        _assert_refusal(cell, error, slot)
    else:
        assert error is None, f"{cell}: unexpected {error!r}"
        assert set(result) == set(KINDS), f"{cell}: {result!r}"
        assert result[kind] is expected_result, f"{cell}: {result!r}"
        assert result[_other(kind)] is True, f"{cell}: {result!r}"
    was_active = before.get(kind, ABSENT)
    was_backup = before.get(f"{kind}.backup", ABSENT)
    got_active = after.get(kind, ABSENT)
    _assert_slot(
        f"{cell} active",
        got_active,
        expected_active,
        kind,
        was_active,
        was_active,
        was_backup,
    )
    _assert_slot(
        f"{cell} backup",
        after.get(f"{kind}.backup", ABSENT),
        expected_backup,
        kind,
        was_backup,
        was_active,
        was_backup,
    )
    # Keyed on the bytes, not on the fixture's name: the current template is
    # also a historical one, so `legacy-released-lf` holds exactly the bytes
    # the installer ships today and must be left alone like `current`.
    was_up_to_date = was_active.data == _template(kind) and was_active.mode == 0o755
    if expected_active == "new" and was_up_to_date:
        assert got_active.ino == was_active.ino, f"{cell}: rewrote an up-to-date hook"
    if (
        expected_active == "new"
        and active_id.startswith("legacy-")
        and not was_up_to_date
    ):
        # A different inode proves the old hook was replaced, not truncated in place.
        assert got_active.ino != was_active.ino, (
            f"{cell}: rewrote the live hook in place"
        )
    _assert_unmanaged_untouched(cell, kind, before, after)
    refused = isinstance(expected_result, str)
    _assert_other_kind(
        cell,
        kind,
        after,
        installed=action != "uninstall" and not (refused and kind == "pre-push"),
    )


def _status_cells() -> Iterator[Any]:
    for kind in KINDS:
        for active_id, active_class in ACTIVE_CLASS.items():
            for backup_id in BACKUP_CLASS:
                if backup_id in BACKUPS_OF_FOREIGN_ACTIVE and active_class != "F":
                    continue
                yield pytest.param(
                    kind, active_id, backup_id, id=f"{kind}:{active_id}:{backup_id}"
                )


@pytest.mark.parametrize(("kind", "active_id", "backup_id"), list(_status_cells()))
def test_status_reports_exact_state_without_touching_files(
    tmp_path: Path, kind: str, active_id: str, backup_id: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    _make_active(hooks_dir, kind, active_id)
    _make_backup(hooks_dir, kind, backup_id)
    expected = EXPECTED_STATUS[active_id]
    if active_id == "absent" and backup_id != "absent":
        expected = "not installed (backup present)"
    before = _tree(hooks_dir)

    status = HooksInstaller(tmp_path).status()

    assert status == {kind: expected, _other(kind): "not installed"}
    assert _tree(hooks_dir) == before


def test_historical_fixture_bytes_are_pinned_and_all_exercised() -> None:
    on_disk = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FIXTURES.glob("*.hook")
    }
    assert on_disk == FIXTURE_SHA256
    for name in FIXTURE_SHA256:
        if ".lf." in name:
            lf = (FIXTURES / name).read_bytes()
            crlf = (FIXTURES / name.replace(".lf.", ".crlf.")).read_bytes()
            assert b"\r" not in lf
            assert crlf == lf.replace(b"\n", b"\r\n")
    exercised = {
        f"{LEGACY_FIXTURE[kind, which]}.{newline}.hook"
        for kind in KINDS
        for which in ("released", "initial")
        for newline in ("lf", "crlf")
        if f"legacy-{which}-{newline}" in ACTIVE_CLASS
    }
    assert exercised == set(FIXTURE_SHA256)


def test_current_templates_are_pinned_and_not_also_listed_as_superseded() -> None:
    """Pin the shipped bytes, and keep "current" and "superseded" disjoint.

    The current pair is also a *historical* pair -- these exact bytes shipped
    in v0.9.0 through v0.30.0 -- so the fixtures legitimately contain them and
    this test does not ask the two sets to be disjoint. What must stay
    disjoint is the installer's own split: a digest in `_CURRENT_TEMPLATES`
    must not also sit in `_LEGACY_DIGESTS`, or "installed (traigent)" and
    "installed (traigent, outdated)" would describe the same bytes.
    """
    for kind in KINDS:
        assert (
            hashlib.sha256(_template(kind)).hexdigest() == CURRENT_TEMPLATE_SHA256[kind]
        ), (
            "A template change must add the previous bytes to the installer's "
            "known set and to tests/fixtures/hooks (see PROVENANCE.md)."
        )
        assert (
            CURRENT_TEMPLATE_SHA256[kind] not in installer_module._LEGACY_DIGESTS[kind]
        ), f"the current {kind} template is also listed as superseded"


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

BOTH_TRUE = {"pre-push": True, "pre-commit": True}


def test_repeated_force_keeps_no_generated_backup_and_uninstall_leaves_nothing(
    tmp_path: Path,
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    installer = HooksInstaller(tmp_path)

    assert installer.install() == BOTH_TRUE
    for attempt in (1, 2):
        assert installer.install(force=True) == BOTH_TRUE
        for kind in KINDS:
            assert _is_new(_obj(hooks_dir / kind), kind), f"force #{attempt}"
            assert _obj(hooks_dir / f"{kind}.backup") == ABSENT, f"force #{attempt}"
    assert installer.uninstall() == BOTH_TRUE
    assert installer.uninstall() == BOTH_TRUE
    assert _tree(hooks_dir) == {}


@pytest.mark.parametrize("kind", KINDS)
def test_repeated_force_keeps_one_original_and_uninstall_restores_it(
    tmp_path: Path, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    installer = HooksInstaller(tmp_path)

    for attempt in (1, 2, 3):
        assert installer.install(force=True) == BOTH_TRUE
        assert _obj(backup).same_object(original), f"force #{attempt} changed it"
        assert _is_new(_obj(active), kind), f"force #{attempt}"
    assert installer.uninstall() == BOTH_TRUE
    assert _obj(active).same_object(original)
    assert _obj(backup) == ABSENT
    assert installer.uninstall()[kind] is False
    assert _obj(active).same_object(original)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("action", ["install", "uninstall"])
def test_old_force_pair_of_generated_hooks_is_cleaned_never_revived(
    tmp_path: Path, kind: str, action: str
) -> None:
    """Earlier SDKs saved their own hook as .backup on --force (O / O on disk)."""
    hooks_dir = _hooks_dir(tmp_path)
    # The initial pair, not the released one: the released pair is byte-identical
    # to the current template, so it would not be an *earlier* SDK's hook.
    _make_active(hooks_dir, kind, "legacy-initial-lf")
    _write(hooks_dir / f"{kind}.backup", _legacy_bytes(kind, "initial", "lf"), 0o755)

    result, error = _run(HooksInstaller(tmp_path), action)

    assert error is None and result == BOTH_TRUE, (result, error)
    assert _obj(hooks_dir / f"{kind}.backup") == ABSENT
    if action == "install":
        assert _is_new(_obj(hooks_dir / kind), kind)
    else:
        assert _tree(hooks_dir) == {}


# ---------------------------------------------------------------------------
# Interference and failure injection (adopted T-I1 to T-I10)
# ---------------------------------------------------------------------------


@contextmanager
def _intercept(
    monkeypatch: pytest.MonkeyPatch,
    wrappers: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]] | None],
) -> Iterator[None]:
    """Wrap os/tempfile functions as the installer sees them, for one block.

    ``None`` removes the function. A same-named alias the installer imported
    directly (``from os import link``) is wrapped too.
    """
    modules = {"os": os, "tempfile": tempfile}
    with monkeypatch.context() as patched:
        for dotted, make in wrappers.items():
            module_name, name = dotted.split(".")
            module = modules[module_name]
            real = getattr(module, name)
            aliased = getattr(installer_module, name, None) is real
            if make is None:
                patched.delattr(module, name)
                if aliased:
                    patched.delattr(installer_module, name)
                continue
            wrapper = make(real)
            patched.setattr(module, name, wrapper)
            if aliased:
                patched.setattr(installer_module, name, wrapper)
        yield


def _named(path: Any, name: str) -> bool:
    return os.path.basename(os.fsdecode(path)) == name


def _injected() -> PermissionError:
    return PermissionError(errno.EACCES, "injected by test")


@pytest.mark.parametrize("kind", KINDS)
def test_ti1_active_swapped_after_takeover_link_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    replacement = tmp_path / "customer-f2"
    _write(replacement, b"#!/bin/sh\n# customer hook F2\n", 0o740)
    f2 = _obj(replacement)
    swapped: list[bool] = []

    def make_link(real: Callable[..., Any]) -> Callable[..., Any]:
        def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            real(src, dst, *args, **kwargs)
            if _named(dst, f"{kind}.backup") and not swapped:
                _replace(replacement, active)
                swapped.append(True)

        return link

    with _intercept(monkeypatch, {"os.link": make_link}):
        result, error = _run(HooksInstaller(tmp_path), "force")

    assert swapped, "intervention not reached: no hard link to the backup name"
    assert isinstance(error, RuntimeError), (result, error)
    assert _obj(active).same_object(f2)
    assert _obj(backup).same_object(original)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("edit", ["append", "chmod"])
def test_ti2_active_edited_after_takeover_link_stops_and_unlinks_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, edit: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    edited_bytes = FOREIGN_ACTIVE + (b"echo edited\n" if edit == "append" else b"")
    edited_mode = 0o750 if edit == "chmod" else original.mode
    edited: list[bool] = []

    def make_link(real: Callable[..., Any]) -> Callable[..., Any]:
        def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            real(src, dst, *args, **kwargs)
            if _named(dst, f"{kind}.backup") and not edited:
                if edit == "append":
                    with open(active, "ab") as handle:
                        handle.write(b"echo edited\n")
                else:
                    _chmod(active, edited_mode)
                edited.append(True)

        return link

    with _intercept(monkeypatch, {"os.link": make_link}):
        result, error = _run(HooksInstaller(tmp_path), "force")

    assert edited, "intervention not reached: no hard link to the backup name"
    assert isinstance(error, RuntimeError), (result, error)
    assert _obj(backup) == ABSENT
    after = _obj(active)
    assert (after.kind, after.data, after.mode, after.ino, after.nlink) == (
        "file",
        edited_bytes,
        edited_mode,
        original.ino,
        1,
    )


def _seed_recorded_takeover(tmp_path: Path, hooks_dir: Path, kind: str) -> Obj:
    """Our hook active, the customer's at <hook>.backup, AND recorded as theirs.

    Built through `install(force=True)` rather than by writing the two files,
    because the restore path is now reached only when the installer recorded
    the backup as the customer's. Writing the files directly produces the same
    bytes with no record, which is the "leave it alone" case instead.
    """
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(hooks_dir / kind)
    assert HooksInstaller(tmp_path).install(force=True)[kind] is True
    assert _obj(hooks_dir / f"{kind}.backup").same_object(original)
    # install() does both kinds; the callers here reason about one slot pair,
    # so drop the sibling artifact it also wrote.
    sibling = hooks_dir / _other(kind)
    if sibling.exists():
        _unlink(sibling)
    return original


@pytest.mark.parametrize("kind", KINDS)
def test_ti3_backup_swapped_before_restore_is_not_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _seed_recorded_takeover(tmp_path, hooks_dir, kind)
    generated = _obj(active)
    replacement = tmp_path / "customer-f2"
    _write(replacement, b"#!/bin/sh\n# swapped original F2\n", 0o710)
    f2 = _obj(replacement)
    backup_reads: list[Any] = []
    swapped: list[bool] = []

    def make_lstat(real: Callable[..., Any]) -> Callable[..., Any]:
        def lstat(path: Any, *args: Any, **kwargs: Any) -> Any:
            if _named(path, f"{kind}.backup"):
                backup_reads.append(path)
                # The first examination classified the backup; swap it before
                # the next one, which is the check before the restore.
                if len(backup_reads) == 2 and not swapped:
                    _replace(replacement, backup)
                    swapped.append(True)
            return real(path, *args, **kwargs)

        return lstat

    with _intercept(monkeypatch, {"os.lstat": make_lstat}):
        result, error = _run(HooksInstaller(tmp_path), "uninstall")

    assert swapped, (
        "intervention not reached: the backup was examined "
        f"{len(backup_reads)} time(s) before the restore, not re-checked"
    )
    assert isinstance(error, RuntimeError), (result, error)
    assert _obj(active).same_object(generated)
    assert _obj(backup).same_object(f2)


@pytest.mark.parametrize("kind", KINDS)
def test_ti4_hook_created_before_empty_slot_publish_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active = hooks_dir / kind
    created: list[Obj] = []

    def make_link(real: Callable[..., Any]) -> Callable[..., Any]:
        def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            if _named(dst, kind) and not created:
                _write(active, b"#!/bin/sh\n# created by another tool\n", 0o700)
                created.append(_obj(active))
            return real(src, dst, *args, **kwargs)

        return link

    with _intercept(monkeypatch, {"os.link": make_link}):
        result, error = _run(HooksInstaller(tmp_path), "install")

    assert created, "intervention not reached: no hard link into the hook name"
    assert isinstance(error, RuntimeError), (result, error)
    after = _tree(hooks_dir)
    assert after[kind].same_object(created[0])
    assert f"{kind}.backup" not in after
    _assert_other_kind("T-I4", kind, after, installed=kind == "pre-commit")
    assert set(after) <= {kind, _other(kind)}, "a temporary file was left behind"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("active_after_failure", ["intact", "removed"])
def test_ti5_ti6_failed_takeover_publish_keeps_the_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    active_after_failure: str,
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active = hooks_dir / kind
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    injected = _injected()
    calls: list[Any] = []

    def make_replace(real: Callable[..., Any]) -> Callable[..., Any]:
        def replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            if _named(dst, kind):
                calls.append(dst)
                if active_after_failure == "removed":
                    _unlink(active)
                raise injected
            return real(src, dst, *args, **kwargs)

        return replace

    with _intercept(monkeypatch, {"os.replace": make_replace}):
        result, error = _run(HooksInstaller(tmp_path), "force")

    assert calls, "intervention not reached: no replace into the hook name"
    assert error is injected, (result, error)
    after = _tree(hooks_dir)
    if active_after_failure == "intact":
        # T-I5: the active name still holds the original, so the new link goes.
        assert after[kind].same_object(original)
        assert after[kind].nlink == 1
        assert f"{kind}.backup" not in after
    else:
        # T-I6: the backup is now the only name of the original; it must stay.
        assert kind not in after
        assert after[f"{kind}.backup"].same_object(original)
    leftovers = set(after) - {kind, f"{kind}.backup", _other(kind)}
    assert not leftovers, f"temporary files left behind: {sorted(leftovers)}"


def _link_failure(name: str) -> BaseException:
    return {
        "permission": PermissionError(errno.EPERM, "injected by test"),
        "unsupported": OSError(errno.EOPNOTSUPP, "injected by test"),
        "not-implemented": NotImplementedError("injected by test"),
    }[name]


@pytest.mark.parametrize("scenario", ["takeover", "empty-slot"])
@pytest.mark.parametrize(
    "failure", ["permission", "unsupported", "not-implemented", "missing"]
)
def test_ti7_unusable_hard_links_refuse_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, failure: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    if scenario == "takeover":
        _make_active(hooks_dir, "pre-push", "foreign")
    before = _tree(hooks_dir)
    link_calls: list[Any] = []
    fallback_calls: list[Any] = []

    def make_link(real: Callable[..., Any]) -> Callable[..., Any]:
        def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            link_calls.append(dst)
            raise _link_failure(failure)

        return link

    def make_recording(real: Callable[..., Any]) -> Callable[..., Any]:
        def move(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            if _named(dst, "pre-push") or _named(dst, "pre-push.backup"):
                fallback_calls.append((src, dst))
            return real(src, dst, *args, **kwargs)

        return move

    wrappers = {
        "os.link": None if failure == "missing" else make_link,
        "os.replace": make_recording,
        "os.rename": make_recording,
    }
    with _intercept(monkeypatch, wrappers):
        result, error = _run(
            HooksInstaller(tmp_path), "force" if scenario == "takeover" else "install"
        )

    if failure != "missing":
        assert link_calls, "intervention not reached: os.link was never called"
    assert isinstance(error, RuntimeError), (result, error)
    assert re.search(r"hard[ -]?link", str(error), re.IGNORECASE), str(error)
    assert fallback_calls == []
    after = _tree(hooks_dir)
    assert after.keys() == before.keys()
    for name, obj in before.items():
        assert after[name].same_object(obj), name


@pytest.mark.parametrize("kind", KINDS)
def test_ti8_staged_name_swapped_before_cleanup_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    decoy_source = tmp_path / "decoy"
    _write(decoy_source, b"not Traigent's staged file\n", 0o600)
    decoy = _obj(decoy_source)
    staged: list[Path] = []

    def make_link(real: Callable[..., Any]) -> Callable[..., Any]:
        def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            real(src, dst, *args, **kwargs)
            if _named(dst, kind) and not staged:
                staged.append(Path(os.fsdecode(src)))
                _replace(decoy_source, src)

        return link

    with _intercept(monkeypatch, {"os.link": make_link}):
        result, error = _run(HooksInstaller(tmp_path), "install")

    assert staged, "intervention not reached: no hard link into the hook name"
    assert error is None or isinstance(error, RuntimeError), error
    if error is None:
        assert result[kind] is True
    assert _is_new(_obj(hooks_dir / kind), kind)
    assert _obj(staged[0]).same_object(decoy)


@pytest.mark.parametrize("kind", KINDS)
def test_ti9_hook_replaced_right_after_refresh_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active = hooks_dir / kind
    _make_active(hooks_dir, kind, "legacy-initial-lf")
    foreign_source = tmp_path / "customer"
    _write(foreign_source, b"#!/bin/sh\n# saved by an editor\n", 0o711)
    foreign = _obj(foreign_source)
    swapped: list[bool] = []

    def make_replace(real: Callable[..., Any]) -> Callable[..., Any]:
        def replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            outcome = real(src, dst, *args, **kwargs)
            if _named(dst, kind) and not swapped:
                _replace(foreign_source, active)
                swapped.append(True)
            return outcome

        return replace

    with _intercept(monkeypatch, {"os.replace": make_replace}):
        result, error = _run(HooksInstaller(tmp_path), "install")

    assert swapped, "intervention not reached: no replace into the hook name"
    assert isinstance(error, RuntimeError), (result, error)
    after = _tree(hooks_dir)
    assert after[kind].same_object(foreign)
    assert f"{kind}.backup" not in after
    leftovers = set(after) - {kind, _other(kind)}
    assert not leftovers, f"temporary files left behind: {sorted(leftovers)}"


@pytest.mark.parametrize("kind", KINDS)
def test_ti10_interrupted_takeover_resumes_and_keeps_the_original_inode(
    tmp_path: Path, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    _make_active(hooks_dir, kind, "foreign")
    _make_backup(hooks_dir, kind, "same-inode")
    original = _obj(hooks_dir / kind)

    assert HooksInstaller(tmp_path).install(force=True) == BOTH_TRUE

    assert _is_new(_obj(hooks_dir / kind), kind)
    backup = _obj(hooks_dir / f"{kind}.backup")
    assert backup.same_object(original)
    assert backup.nlink == 1


def _staging_wrappers(
    failure: str, fired: list[Any], injected: BaseException
) -> dict[str, Callable[[Callable[..., Any]], Callable[..., Any]] | None]:
    if failure == "write":

        def make_mkstemp(real: Callable[..., Any]) -> Callable[..., Any]:
            def mkstemp(*args: Any, **kwargs: Any) -> Any:
                fd, name = real(*args, **kwargs)
                os.close(fd)
                fired.append(name)
                # A read-only descriptor makes every write to the staged file fail.
                return os.open(name, os.O_RDONLY), name

            return mkstemp

        return {"tempfile.mkstemp": make_mkstemp}

    def make_raiser(real: Callable[..., Any]) -> Callable[..., Any]:
        def fail(*args: Any, **kwargs: Any) -> Any:
            fired.append(args)
            raise injected

        return fail

    if failure == "fsync":
        return {"os.fsync": make_raiser}
    return {"os.fchmod": make_raiser, "os.chmod": make_raiser}


@pytest.mark.parametrize("scenario", ["takeover", "empty-slot", "refresh"])
@pytest.mark.parametrize("failure", ["write", "fsync", "chmod"])
def test_staging_failure_changes_nothing_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, failure: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active_fixture = {
        "takeover": "foreign",
        "empty-slot": "absent",
        "refresh": "legacy-initial-lf",
    }[scenario]
    _make_active(hooks_dir, "pre-push", active_fixture)
    before = _tree(hooks_dir)
    injected = OSError(errno.EIO, "injected by test")
    fired: list[Any] = []

    with _intercept(monkeypatch, _staging_wrappers(failure, fired, injected)):
        result, error = _run(
            HooksInstaller(tmp_path), "force" if scenario == "takeover" else "install"
        )

    assert fired, f"intervention not reached: the {failure} step was never called"
    if failure == "write":
        assert isinstance(error, OSError), (result, error)
    else:
        assert error is injected, (result, error)
    after = _tree(hooks_dir)
    assert after.keys() == before.keys(), sorted(after.keys() ^ before.keys())
    for name, obj in before.items():
        assert after[name].same_object(obj), name


def _failing_on(
    name: str, calls: list[Any], injected: BaseException
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def make(real: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            target = args[1] if real is _replace else args[0]
            if _named(target, name):
                calls.append(target)
                raise injected
            return real(*args, **kwargs)

        return call

    return make


# case -> (active fixture, backup fixture, action, failing call, failing name,
#          expected active afterwards, expected backup afterwards, retry)
MUTATION_FAILURES = {
    "refresh-replace": (
        "legacy-initial-lf",
        "absent",
        "install",
        "replace",
        "",
        "same",
        "absent",
        None,
    ),
    "restore-replace": (
        "current",
        "foreign",
        "uninstall",
        "replace",
        "",
        "same",
        "same",
        None,
    ),
    "remove-active": (
        "current",
        "absent",
        "uninstall",
        "unlink",
        "",
        "same",
        "absent",
        None,
    ),
    "remove-generated-backup-on-uninstall": (
        "current",
        "current",
        "uninstall",
        "unlink",
        ".backup",
        "absent",
        "same",
        "uninstall",
    ),
    "remove-generated-backup-on-install": (
        "absent",
        "current",
        "install",
        "unlink",
        ".backup",
        "new",
        "same",
        "install",
    ),
}


@pytest.mark.parametrize("case", list(MUTATION_FAILURES))
def test_mutation_failure_propagates_and_reports_true_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    (
        active_fixture,
        backup_fixture,
        action,
        call,
        suffix,
        expected_active,
        expected_backup,
        retry,
    ) = MUTATION_FAILURES[case]
    kind = "pre-push"
    hooks_dir = _hooks_dir(tmp_path)
    if (active_fixture, backup_fixture) == ("current", "foreign"):
        # The restore path is reached only for a backup the installer recorded
        # as the customer's, so build that state through install(force=True)
        # instead of writing the two files.
        _seed_recorded_takeover(tmp_path, hooks_dir, kind)
    else:
        _make_active(hooks_dir, kind, active_fixture)
        _make_backup(hooks_dir, kind, backup_fixture)
    before = _tree(hooks_dir)
    injected = _injected()
    calls: list[Any] = []
    failing = _failing_on(kind + suffix, calls, injected)
    wrappers = (
        {"os.replace": failing}
        if call == "replace"
        else {"os.unlink": failing, "os.remove": failing}
    )

    with _intercept(monkeypatch, wrappers):
        result, error = _run(HooksInstaller(tmp_path), action)

    assert calls, f"intervention not reached: no {call} of {kind + suffix}"
    assert error is injected, (result, error)
    after = _tree(hooks_dir)
    was_active = before.get(kind, ABSENT)
    was_backup = before.get(f"{kind}.backup", ABSENT)
    _assert_slot(
        f"{case} active",
        after.get(kind, ABSENT),
        expected_active,
        kind,
        was_active,
        was_active,
        was_backup,
    )
    _assert_slot(
        f"{case} backup",
        after.get(f"{kind}.backup", ABSENT),
        expected_backup,
        kind,
        was_backup,
        was_active,
        was_backup,
    )
    assert "pre-commit" not in after, "processing continued after the first failure"
    leftovers = set(after) - {kind, f"{kind}.backup"}
    assert not leftovers, f"temporary files left behind: {sorted(leftovers)}"
    if retry is not None:
        # A retry converges from the partial state.
        assert _run(HooksInstaller(tmp_path), retry) == (BOTH_TRUE, None)
        assert f"{kind}.backup" not in _tree(hooks_dir)


@pytest.fixture
def installed_hook_repo(tmp_path: Path) -> Path:
    """Install the real templates into an otherwise empty repository."""
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    assert all(HooksInstaller(tmp_path).install().values())
    return tmp_path


def _hook_environment(repo: Path) -> dict[str, str]:
    bin_dir = repo / "bin"
    bin_dir.mkdir(exist_ok=True)
    home = repo / "home"
    home.mkdir(exist_ok=True)
    return {
        "PATH": str(bin_dir),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "PYTHON_DOTENV_DISABLED": "1",
        "TRAIGENT_MOCK_LLM": "true",
        "TRAIGENT_OFFLINE_MODE": "true",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }


def _run_installed_hook(
    repo: Path, hook_name: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(repo / ".git" / "hooks" / hook_name)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_hook_lifecycle_allows_ordinary_git_commit_after_sdk_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uninstall must remove both owned hooks before the SDK becomes unavailable."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _hook_environment(repo)
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_AUTHOR_NAME": "Hook lifecycle test",
            "GIT_AUTHOR_EMAIL": "hook-test@example.invalid",
            "GIT_COMMITTER_NAME": "Hook lifecycle test",
            "GIT_COMMITTER_EMAIL": "hook-test@example.invalid",
        }
    )
    transcript: list[str] = []

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["/usr/bin/git", *args],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        transcript.append(
            f"git {shlex.join(args)} => {result.returncode}\n"
            + result.stdout
            + result.stderr
        )
        return result

    template_dir = tmp_path / "empty-git-template"
    template_dir.mkdir()
    assert (
        git(
            "init",
            "--initial-branch=traigent-hook-lifecycle",
            f"--template={template_dir}",
        ).returncode
        == 0
    )
    monkeypatch.chdir(repo)
    runner = CliRunner()
    problems = []
    installed = runner.invoke(cli, ["hooks", "install"])
    transcript.append(f"CLI install => {installed.exit_code}\n{installed.output}")
    assert installed.exit_code == 0, "\n".join(transcript)
    hooks = {
        name: repo / ".git" / "hooks" / name for name in ("pre-push", "pre-commit")
    }
    original = {name: path.read_bytes() for name, path in hooks.items()}
    if HooksInstaller(repo).status() != dict.fromkeys(hooks, "installed (traigent)"):
        problems.append("Both generated hooks must have Traigent ownership status")
    status = runner.invoke(cli, ["hooks", "status"])
    transcript.append(f"CLI status => {status.exit_code}\n{status.output}")
    if status.exit_code != 0 or status.output.count("Installed (Traigent)") != 2:
        problems.append("CLI status must recognize both generated hooks")
    reinstalled = runner.invoke(cli, ["hooks", "install"])
    transcript.append(
        f"CLI nonforce reinstall => {reinstalled.exit_code}\n{reinstalled.output}"
    )
    if "Skipped" in reinstalled.output or reinstalled.exit_code != 0:
        problems.append("Nonforce reinstall must recognize both owned hooks")
    assert {name: path.read_bytes() for name, path in hooks.items()} == original

    # Isolate the interpreter from site packages/PYTHONPATH so the SDK is absent.
    python = repo / "bin" / "python"
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -I -S "$@"\n')
    python.chmod(0o755)
    absent_console = subprocess.run(
        ["/bin/bash", "-c", "command -v traigent"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    transcript.append(
        f"Absent console lookup => {absent_console.returncode}\n"
        + absent_console.stdout
        + absent_console.stderr
    )
    assert absent_console.returncode != 0
    absent_module = subprocess.run(
        [str(python), "-m", "traigent", "--version"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    transcript.append(
        f"Actual Python module lookup without SDK => {absent_module.returncode}\n"
        + absent_module.stdout
        + absent_module.stderr
    )
    assert absent_module.returncode != 0
    assert "No module named traigent" in absent_module.stderr
    # With the SDK absent the installed artifacts skip rather than block --
    # that is the point of failing open, and it is what keeps a customer who
    # removed the package (and with it the uninstall command) able to commit.
    for name in hooks:
        skipped = _run_installed_hook(repo, name, env)
        transcript.append(
            f"Installed {name} with SDK absent => {skipped.returncode}\n"
            + skipped.stdout
            + skipped.stderr
        )
        assert skipped.returncode == 0, skipped.stdout + skipped.stderr
        assert "All Traigent hooks passed" not in skipped.stdout

    uninstalled = runner.invoke(cli, ["hooks", "uninstall"])
    transcript.append(f"CLI uninstall => {uninstalled.exit_code}\n{uninstalled.output}")
    if uninstalled.exit_code != 0 or "Skipped" in uninstalled.output:
        problems.append("CLI uninstall must remove both generated hooks")
    remaining = [name for name, path in hooks.items() if path.exists()]
    if remaining:
        problems.append(f"Uninstall left owned hooks behind: {remaining}")
    (repo / "committed.txt").write_text("SDK removed; ordinary commit must work.\n")
    assert git("add", "committed.txt").returncode == 0
    committed = git("commit", "-m", "Verify hook uninstall lifecycle")
    assert committed.returncode == 0, "\n".join([*problems, *transcript])
    assert not problems, "\n".join([*problems, *transcript])
    print("\n".join(transcript))


@pytest.mark.parametrize("hook_name", ["pre-push", "pre-commit"])
@pytest.mark.parametrize("python_present", [False, True])
def test_installed_hook_skips_when_the_checker_is_unavailable(
    installed_hook_repo: Path, hook_name: str, python_present: bool
) -> None:
    """With no reachable CLI the hook skips and exits 0 -- it does not block.

    This is deliberate and is the behaviour the SDK already ships. A hook that
    refused the commit instead would lock out anyone who removed the SDK
    without running `traigent hooks uninstall` first: the hook stays on disk,
    every commit fails, and the command that would remove it is gone with the
    package. There is no in-product escape from that, so the hook fails open
    and validation is skipped loudly rather than enforced silently.
    """
    repo = installed_hook_repo
    env = _hook_environment(repo)
    if python_present:
        launcher = repo / "bin" / "python"
        launcher.write_text("#!/bin/bash\nexit 127\n")
        launcher.chmod(0o755)

    result = _run_installed_hook(repo, hook_name, env)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    # The two shipped templates differ here and this pins the difference
    # rather than hiding it: pre-push says it is skipping, pre-commit exits
    # silently. Only pre-push's message is asserted, because only pre-push
    # has one.
    if hook_name == "pre-push":
        assert "skipping validation" in output.lower(), output
    # Whichever it is, it must not claim the checks ran.
    assert "All Traigent hooks passed" not in result.stdout


@pytest.mark.parametrize(
    ("hook_name", "expected_args"),
    [
        ("pre-push", "hooks validate --exit-code"),
        ("pre-commit", "hooks check --quick"),
    ],
)
@pytest.mark.parametrize("module_fallback", [False, True])
@pytest.mark.parametrize("checker_status", [0, 7])
def test_installed_hook_dispatches_checker_and_propagates_failure(
    installed_hook_repo: Path,
    hook_name: str,
    expected_args: str,
    module_fallback: bool,
    checker_status: int,
) -> None:
    repo = installed_hook_repo
    env = _hook_environment(repo)
    launcher = repo / "bin" / ("python" if module_fallback else "traigent")
    launcher.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> calls.txt\n'
        'if [[ "$*" == "-m traigent --version" ]]; then exit 0; fi\n'
        f"exit {checker_status}\n"
    )
    launcher.chmod(0o755)

    result = _run_installed_hook(repo, hook_name, env)

    prefix = "-m traigent " if module_fallback else ""
    assert (repo / "calls.txt").read_text().splitlines() == (
        ["-m traigent --version", prefix + expected_args]
        if module_fallback
        else [expected_args]
    )
    assert (result.returncode == 0) == (checker_status == 0)
    if checker_status:
        assert "All Traigent hooks passed" not in result.stdout


@pytest.mark.parametrize("hook_name", ["pre-push", "pre-commit"])
def test_installed_hook_real_module_fallback_rejects_invalid_config(
    installed_hook_repo: Path, hook_name: str
) -> None:
    repo = installed_hook_repo
    env = _hook_environment(repo)
    # Apply the no-network guard inside each actual module invocation too.
    guard_dir = repo / "guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        "import sys\n"
        "def deny_network(event, args):\n"
        "    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:\n"
        "        raise RuntimeError('Hook regression prohibits network')\n"
        "sys.addaudithook(deny_network)\n"
    )
    env["PYTHONPATH"] = str(guard_dir) + ":" + env["PYTHONPATH"]
    launcher = repo / "bin" / "python"
    launcher.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    launcher.chmod(0o755)
    (repo / "traigent.yml").write_text("hooks: [unterminated\n")

    version = subprocess.run(
        [str(launcher), "-m", "traigent", "--version"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert version.returncode == 0, version.stdout + version.stderr
    assert "traigent" in version.stdout.lower()
    result = _run_installed_hook(repo, hook_name, env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "configuration" in result.stdout.lower()
    assert "All Traigent hooks passed" not in result.stdout


# ---------------------------------------------------------------------------
# Real Git: force, restore and run (no network, no SDK on the hook's PATH)
# ---------------------------------------------------------------------------


def _real_git_repo(
    tmp_path: Path,
) -> tuple[Path, Callable[..., subprocess.CompletedProcess[str]], list[str]]:
    """Template-less repository whose hooks see a Python that cannot import the SDK."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _hook_environment(repo)
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_AUTHOR_NAME": "Hook lifecycle test",
            "GIT_AUTHOR_EMAIL": "hook-test@example.invalid",
            "GIT_COMMITTER_NAME": "Hook lifecycle test",
            "GIT_COMMITTER_EMAIL": "hook-test@example.invalid",
        }
    )
    transcript: list[str] = []

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["/usr/bin/git", *args],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        transcript.append(
            f"git {shlex.join(args)} => {result.returncode}\n"
            + result.stdout
            + result.stderr
        )
        return result

    template_dir = tmp_path / "empty-git-template"
    template_dir.mkdir()
    initialized = git(
        "init", "--initial-branch=traigent-hook-lifecycle", f"--template={template_dir}"
    )
    assert initialized.returncode == 0, "\n".join(transcript)
    python = repo / "bin" / "python"
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -I -S "$@"\n')
    python.chmod(0o755)
    return repo, git, transcript


def _cli_hooks(
    repo: Path, transcript: list[str], *args: str
) -> subprocess.CompletedProcess[str] | Any:
    result = CliRunner().invoke(cli, ["hooks", *args, "--path", str(repo)])
    transcript.append(
        f"CLI hooks {' '.join(args)} => {result.exit_code}\n{result.output}"
    )
    return result


def test_real_git_force_repeat_then_uninstall_allows_commit_without_sdk(
    tmp_path: Path,
) -> None:
    repo, git, transcript = _real_git_repo(tmp_path)
    hooks_dir = repo / ".git" / "hooks"
    problems = []
    for args in (("install",), ("install", "--force"), ("install", "--force")):
        assert _cli_hooks(repo, transcript, *args).exit_code == 0, "\n".join(transcript)
    names = sorted(path.name for path in hooks_dir.iterdir())
    if names != ["pre-commit", "pre-push"]:
        problems.append(f"Repeated --force left {names}")
    (repo / "committed.txt").write_text("Commit attempt with the SDK absent.\n")
    assert git("add", "committed.txt").returncode == 0

    # The generated hook is installed and the SDK is unreachable: it skips,
    # so the commit succeeds. Failing open here is deliberate.
    with_hook = git("commit", "-m", "Commit while the generated hook is installed")
    assert with_hook.returncode == 0, "\n".join(transcript)

    uninstalled = _cli_hooks(repo, transcript, "uninstall")
    assert uninstalled.exit_code == 0, "\n".join(transcript)
    remaining = sorted(path.name for path in hooks_dir.iterdir())
    if remaining:
        problems.append(f"Uninstall left {remaining}")
    # The stated effect: with the SDK gone, an ordinary commit must succeed.
    # Fresh content, because the commit above consumed the first file.
    (repo / "after-uninstall.txt").write_text("Commit after the hooks are gone.\n")
    assert git("add", "after-uninstall.txt").returncode == 0
    committed = git("commit", "-m", "Ordinary commit after uninstall")
    assert committed.returncode == 0, "\n".join([*problems, *transcript])
    assert not problems, "\n".join([*problems, *transcript])


def test_real_git_foreign_original_is_restored_and_runs_after_uninstall(
    tmp_path: Path,
) -> None:
    repo, git, transcript = _real_git_repo(tmp_path)
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir()  # the empty template creates no hooks directory
    customer_hook = hooks_dir / "pre-commit"
    _write(
        customer_hook,
        b"#!/bin/sh\nprintf 'customer hook ran\\n' >> customer-hook-ran.txt\n",
        0o700,
    )
    original = _obj(customer_hook)
    problems = []

    for attempt in (1, 2):
        forced = _cli_hooks(repo, transcript, "install", "--force")
        assert forced.exit_code == 0, "\n".join(transcript)
        saved = _obj(hooks_dir / "pre-commit.backup")
        if not saved.same_object(original):
            problems.append(f"force #{attempt} saved {saved.describe()}")
    uninstalled = _cli_hooks(repo, transcript, "uninstall")
    assert uninstalled.exit_code == 0, "\n".join(transcript)
    restored = _obj(customer_hook)
    if not restored.same_object(original):
        problems.append(f"uninstall left {restored.describe()} at pre-commit")
    names = sorted(path.name for path in hooks_dir.iterdir())
    if names != ["pre-commit"]:
        problems.append(f"uninstall left {names}")

    # The stated effect: the next commit runs the customer's own hook again.
    (repo / "committed.txt").write_text("The customer's own hook runs again.\n")
    assert git("add", "committed.txt").returncode == 0
    committed = git("commit", "-m", "Commit runs the restored customer hook")
    assert committed.returncode == 0, "\n".join([*problems, *transcript])
    marker = repo / "customer-hook-ran.txt"
    assert marker.exists(), "\n".join([*problems, *transcript])
    assert not problems, "\n".join([*problems, *transcript])
    assert (repo / "customer-hook-ran.txt").read_text() == "customer hook ran\n"


@pytest.mark.parametrize("checker", ["healthy", "rejecting", "absent"])
def test_emitted_pre_push_handles_git_style_arguments_and_stdin(
    installed_hook_repo: Path, checker: str
) -> None:
    repo = installed_hook_repo
    env = _hook_environment(repo)
    if checker != "absent":
        launcher = repo / "bin" / "traigent"
        launcher.write_text(
            "#!/bin/bash\n"
            'printf "%s\\n" "$*" >> calls.txt\n'
            f"exit {0 if checker == 'healthy' else 3}\n"
        )
        launcher.chmod(0o755)
    refs = f"refs/heads/main {'1' * 40} refs/heads/main {'0' * 40}\n"

    # Git runs pre-push as `<hook> <remote-name> <remote-url>` with refs on stdin.
    result = subprocess.run(
        [
            "/bin/bash",
            str(repo / ".git" / "hooks" / "pre-push"),
            "origin",
            str(repo / "no-such-remote.git"),
        ],
        cwd=repo,
        env=env,
        input=refs,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    output = result.stdout + result.stderr
    if checker == "healthy":
        assert result.returncode == 0, output
        assert (repo / "calls.txt").read_text() == "hooks validate --exit-code\n"
    elif checker == "rejecting":
        assert result.returncode != 0, output
        assert "PUSH REJECTED" in result.stdout
    else:
        # No reachable checker: the hook skips and exits 0 (fails open).
        assert result.returncode == 0, output
        assert "skipping validation" in output.lower(), output


# ---------------------------------------------------------------------------
# Generated-artifact ownership (the ownership ledger)
#
# Content alone cannot answer "did Traigent write this file?". These tests pin
# the two halves of the invariant the ledger exists to hold:
#   * the SDK removes exactly what it generated, even after its own template
#     has changed -- it never files its own hook away as the customer's and
#     then resurrects it on uninstall;
#   * the SDK never deletes or resurrects a hook it did not generate, even if
#     that hook's bytes later start matching a known Traigent artifact.
# ---------------------------------------------------------------------------

LEDGER_NAME = "traigent-hooks.json"


def _read_ledger(repo: Path) -> dict[str, Any]:
    path = repo / ".git" / LEDGER_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _template_changed(kind: str, marker: str) -> Iterator[bytes]:
    """Ship a different template for `kind`, as the next SDK release would.

    Nothing on disk changes; only the installer's idea of its current artifact.
    The previous digest is deliberately NOT added to the known legacy set --
    that omission is the drift the ownership ledger has to survive.
    """
    attribute = (
        "PRE_PUSH_HOOK_SCRIPT" if kind == "pre-push" else "PRE_COMMIT_HOOK_SCRIPT"
    )
    previous_text: str = getattr(installer_module, attribute)
    upgraded_text = previous_text.replace(
        "# Installed by: traigent hooks install",
        f"# Installed by: traigent hooks install\n# {marker}",
    )
    assert upgraded_text != previous_text
    upgraded = upgraded_text.encode("utf-8")
    saved_templates = dict(installer_module._CURRENT_TEMPLATES)
    try:
        setattr(installer_module, attribute, upgraded_text)
        installer_module._CURRENT_TEMPLATES[kind] = upgraded
        yield upgraded
    finally:
        setattr(installer_module, attribute, previous_text)
        installer_module._CURRENT_TEMPLATES.clear()
        installer_module._CURRENT_TEMPLATES.update(saved_templates)


@pytest.mark.parametrize("kind", KINDS)
def test_force_refresh_across_a_template_change_keeps_its_own_artifact(
    tmp_path: Path, kind: str
) -> None:
    """The SDK must recognise the hook it installed itself one release ago.

    Without the ledger the old artifact classifies as foreign: --force saves it
    as the customer's original and uninstall puts it back, leaving a hook that
    invokes a CLI the customer has removed.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    installer = HooksInstaller(tmp_path)
    assert installer.install() == BOTH_TRUE
    shipped = active.read_bytes()

    with _template_changed(kind, "ownership regression marker") as upgraded:
        upgraded_installer = HooksInstaller(tmp_path)
        assert upgraded_installer.install(force=True) == BOTH_TRUE
        assert active.read_bytes() == upgraded
        assert _obj(backup) == ABSENT, "the SDK filed its own hook away as foreign"
        assert upgraded_installer.status()[kind] == "installed (traigent)"
        assert upgraded_installer.uninstall() == BOTH_TRUE

    assert not active.exists(), "uninstall resurrected the SDK's own old hook"
    assert _obj(backup) == ABSENT
    assert shipped not in {
        path.read_bytes() for path in hooks_dir.iterdir() if path.is_file()
    }


@pytest.mark.parametrize("kind", KINDS)
def test_install_records_what_it_generated_and_uninstall_clears_it(
    tmp_path: Path, kind: str
) -> None:
    _hooks_dir(tmp_path)
    installer = HooksInstaller(tmp_path)
    assert installer.install() == BOTH_TRUE

    ledger = _read_ledger(tmp_path)
    assert ledger["version"] == 1
    assert ledger["hooks"][kind]["generated"] == [CURRENT_TEMPLATE_SHA256[kind]]
    assert "backup" not in ledger["hooks"][kind]

    with _template_changed(kind, "second release") as upgraded:
        assert HooksInstaller(tmp_path).install(force=True) == BOTH_TRUE
        recorded = _read_ledger(tmp_path)["hooks"][kind]["generated"]
        assert recorded == [
            CURRENT_TEMPLATE_SHA256[kind],
            hashlib.sha256(upgraded).hexdigest(),
        ]

    assert HooksInstaller(tmp_path).uninstall() == BOTH_TRUE
    assert not (tmp_path / ".git" / LEDGER_NAME).exists(), (
        "uninstall left its ownership record behind"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_install_records_the_original_it_preserved(tmp_path: Path, kind: str) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    _make_active(hooks_dir, kind, "foreign")
    original = (hooks_dir / kind).read_bytes()

    assert HooksInstaller(tmp_path).install(force=True) == BOTH_TRUE

    entry = _read_ledger(tmp_path)["hooks"][kind]
    assert entry["backup"] == {
        "sha256": hashlib.sha256(original).hexdigest(),
        "origin": "foreign",
    }


@pytest.mark.parametrize("kind", KINDS)
def test_a_preserved_original_is_never_deleted_once_its_bytes_become_known(
    tmp_path: Path, kind: str
) -> None:
    """The customer's hook stays the customer's, whatever it later looks like.

    A hook the customer copied from a newer SDK is foreign when install saves
    it. If a later release adds those bytes to the known set, content-only
    classification calls the saved file Traigent's own and deletes it. The
    ledger remembers whose it was.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    # A hook the SDK cannot claim: the current template with a line the
    # customer added. The CRLF form is no longer usable here -- historical
    # writers emitted it, so it is a recognised artifact again.
    _write(active, _template(kind) + b"echo 'customer addition'\n", 0o755)
    original = _obj(active)

    assert HooksInstaller(tmp_path).install(force=True) == BOTH_TRUE
    assert _obj(backup).same_object(original)

    # A later release recognises those exact bytes as one of its own artifacts.
    widened = dict(installer_module._LEGACY_DIGESTS)
    widened[kind] = frozenset(
        {
            *installer_module._LEGACY_DIGESTS[kind],
            hashlib.sha256(original.data).hexdigest(),
        }
    )
    with patch.object(installer_module, "_LEGACY_DIGESTS", widened):
        # Our own hook is still in place: re-installing must leave the saved
        # original alone instead of collecting it as a stale artifact.
        assert HooksInstaller(tmp_path).install(force=True) == BOTH_TRUE
        assert _obj(backup).same_object(original), "the customer's hook was deleted"

        # Something else then takes the active slot. Overwriting the saved
        # original to make room for that is refused, not done silently.
        _write(active, b"#!/bin/sh\n# another manager took over\nexit 0\n", 0o755)
        with pytest.raises(RuntimeError, match="saved for you"):
            HooksInstaller(tmp_path).install(force=True)
        assert _obj(backup).same_object(original), "the customer's hook was deleted"

        # And uninstall hands it back rather than deleting it.
        assert HooksInstaller(tmp_path).uninstall()[kind] is False
        assert _obj(backup).same_object(original)
        active.unlink()
        assert HooksInstaller(tmp_path).uninstall()[kind] is True

    assert _obj(active).same_object(original)
    assert _obj(backup) == ABSENT


@pytest.mark.parametrize("kind", KINDS)
def test_uninstall_returns_a_preserved_original_when_our_hook_is_gone(
    tmp_path: Path, kind: str
) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    assert HooksInstaller(tmp_path).install(force=True) == BOTH_TRUE

    # Something else removed the generated hook; ours is gone, theirs is not.
    active.unlink()

    assert HooksInstaller(tmp_path).uninstall()[kind] is True
    assert _obj(active).same_object(original)
    assert _obj(backup) == ABSENT


@pytest.mark.parametrize("kind", KINDS)
def test_an_unclaimable_hook_round_trips_byte_exact(tmp_path: Path, kind: str) -> None:
    """A hook the SDK cannot claim is preserved, not adopted and not destroyed.

    The CRLF form of the current template is deliberately absent from the known
    set (tests/fixtures/hooks/PROVENANCE.md: no Traigent writer ever emitted
    it), so it is indistinguishable from a customer-edited hook. force-install
    then uninstall must be a byte-exact round trip.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    # A hook the SDK cannot claim: the current template with a line the
    # customer added. The CRLF form is no longer usable here -- historical
    # writers emitted it, so it is a recognised artifact again.
    _write(active, _template(kind) + b"echo 'customer addition'\n", 0o755)
    original = _obj(active)

    installer = HooksInstaller(tmp_path)
    assert installer.install(force=True) == BOTH_TRUE
    assert _obj(backup).same_object(original)
    assert installer.status()[kind] == "installed (traigent)"
    assert installer.uninstall() == BOTH_TRUE

    assert _obj(active).same_object(original)
    assert _obj(backup) == ABSENT


@pytest.mark.parametrize(
    "corruption",
    ["{ not json", '{"version": 99, "hooks": {}}', '["not", "an", "object"]', ""],
)
def test_an_untrustworthy_ledger_degrades_to_content_classification(
    tmp_path: Path, corruption: str
) -> None:
    """A ledger that cannot be trusted is ignored loudly, never obeyed blindly."""
    hooks_dir = _hooks_dir(tmp_path)
    installer = HooksInstaller(tmp_path)
    assert installer.install() == BOTH_TRUE
    (tmp_path / ".git" / LEDGER_NAME).write_text(corruption, encoding="utf-8")

    assert installer.status() == {
        "pre-push": "installed (traigent)",
        "pre-commit": "installed (traigent)",
    }
    assert installer.uninstall() == BOTH_TRUE
    assert _tree(hooks_dir) == {}


def test_a_ledger_claiming_hooks_it_never_generated_is_ignored(
    tmp_path: Path,
) -> None:
    """Junk digests must not let the installer adopt a customer's hook."""
    hooks_dir = _hooks_dir(tmp_path)
    _make_active(hooks_dir, "pre-commit", "foreign")
    original = _obj(hooks_dir / "pre-commit")
    (tmp_path / ".git" / LEDGER_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "pre-commit": {"generated": ["not-a-digest", 17, None]},
                    "../escape": {
                        "generated": [hashlib.sha256(original.data).hexdigest()]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    installer = HooksInstaller(tmp_path)
    assert installer.status()["pre-commit"] == "installed (other)"
    assert installer.install()["pre-commit"] is False
    assert _obj(hooks_dir / "pre-commit").same_object(original)


def test_real_git_template_change_force_then_uninstall_allows_commit(
    tmp_path: Path,
) -> None:
    """The user-visible control, end to end, with a real `git commit`.

    Install, ship a new template, force-refresh, uninstall, then commit with
    the SDK unreachable. Before the ownership ledger the force-refresh saved
    the old generated hook as the customer's original, uninstall restored it,
    and this commit failed with "validation could not run".
    """
    repo, git, transcript = _real_git_repo(tmp_path)
    hooks_dir = repo / ".git" / "hooks"
    problems = []

    assert _cli_hooks(repo, transcript, "install").exit_code == 0, "\n".join(transcript)

    with (
        _template_changed("pre-commit", "release N+1"),
        _template_changed("pre-push", "release N+1"),
    ):
        forced = _cli_hooks(repo, transcript, "install", "--force")
        assert forced.exit_code == 0, "\n".join(transcript)
        saved = sorted(path.name for path in hooks_dir.iterdir())
        if saved != ["pre-commit", "pre-push"]:
            problems.append(f"force across a template change left {saved}")

        (repo / "committed.txt").write_text("Commit with the SDK absent.\n")
        assert git("add", "committed.txt").returncode == 0
        with_hook = git("commit", "-m", "Commit while the refreshed hook is installed")
        assert with_hook.returncode == 0, "\n".join(transcript)

        uninstalled = _cli_hooks(repo, transcript, "uninstall")
        assert uninstalled.exit_code == 0, "\n".join(transcript)

    remaining = sorted(path.name for path in hooks_dir.iterdir())
    if remaining:
        problems.append(f"Uninstall left {remaining}")
    (repo / "after.txt").write_text("Commit after the hooks are gone.\n")
    assert git("add", "after.txt").returncode == 0
    committed = git("commit", "-m", "Ordinary commit after uninstall")
    assert committed.returncode == 0, "\n".join([*problems, *transcript])
    assert not problems, "\n".join([*problems, *transcript])


@pytest.mark.parametrize("kind", KINDS)
def test_a_refused_install_records_no_hook_it_did_not_write(
    tmp_path: Path, kind: str
) -> None:
    """Declining to touch a customer hook must leave no ownership claim."""
    hooks_dir = _hooks_dir(tmp_path)
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(hooks_dir / kind)

    assert HooksInstaller(tmp_path).install()[kind] is False

    recorded = _read_ledger(tmp_path).get("hooks", {})
    assert kind not in recorded, f"a refused install claimed {kind}"
    assert _obj(hooks_dir / kind).same_object(original)


def test_a_wholly_refused_install_writes_no_ledger_at_all(tmp_path: Path) -> None:
    hooks_dir = _hooks_dir(tmp_path)
    for kind in KINDS:
        _make_active(hooks_dir, kind, "foreign")

    assert HooksInstaller(tmp_path).install() == {
        "pre-push": False,
        "pre-commit": False,
    }

    assert not (tmp_path / ".git" / LEDGER_NAME).exists()


def test_a_symlinked_ownership_record_is_refused_not_followed(tmp_path: Path) -> None:
    """The record is read with no more trust than a hook: no symlink following.

    The planted record names the customer's own hook as one of ours. Followed,
    it would make `uninstall` delete that hook; refused, the installer falls
    back to content classification and leaves it alone.
    """
    hooks_dir = _hooks_dir(tmp_path)
    _make_active(hooks_dir, "pre-commit", "foreign")
    victim = _obj(hooks_dir / "pre-commit")
    elsewhere = tmp_path / "planted.json"
    elsewhere.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "pre-commit": {
                        "generated": [hashlib.sha256(victim.data).hexdigest()]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _symlink(elsewhere, tmp_path / ".git" / LEDGER_NAME)

    installer = HooksInstaller(tmp_path)
    assert installer.status()["pre-commit"] == "installed (other)"
    assert installer.uninstall()["pre-commit"] is False
    assert _obj(hooks_dir / "pre-commit").same_object(victim), (
        "a planted record was followed and the customer's hook deleted"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_a_failed_install_keeps_the_record_of_a_backup_it_could_not_undo(
    tmp_path: Path, kind: str
) -> None:
    """Rollback that leaves the customer's hook must leave its record too.

    If the record is dropped while `<hook>.backup` survives, the customer's own
    hook is on disk with nothing saying it is theirs -- and a later release that
    recognises its bytes would delete it as a stale Traigent artifact.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    real_replace = _replace

    def _swap_then_fail(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        """Publish fails, and another writer has taken the active slot.

        That is the state in which _rollback_original cannot prove the inode,
        so it leaves <hook>.backup -- holding the customer's hook -- in place.
        """
        if _named(dst, kind):
            intruder = hooks_dir / f".intruder-{kind}"
            _write(intruder, b"#!/bin/sh\n# another hook manager\nexit 0\n", 0o755)
            real_replace(intruder, dst)
            raise _injected()
        real_replace(src, dst, *args, **kwargs)

    with patch.object(os, "replace", _swap_then_fail):
        with pytest.raises(PermissionError):
            HooksInstaller(tmp_path).install(force=True)

    assert _obj(backup).same_object(original), "rollback removed the customer's hook"
    entry = _read_ledger(tmp_path)["hooks"][kind]
    assert entry["backup"] == {
        "sha256": hashlib.sha256(original.data).hexdigest(),
        "origin": "foreign",
    }, "the record was dropped while the customer's hook was still on disk"

    # And the consequence the record exists to prevent: a later release that
    # recognises those bytes must still not delete the customer's hook.
    widened = dict(installer_module._LEGACY_DIGESTS)
    widened[kind] = frozenset(
        {
            *installer_module._LEGACY_DIGESTS[kind],
            hashlib.sha256(original.data).hexdigest(),
        }
    )
    with patch.object(installer_module, "_LEGACY_DIGESTS", widened):
        HooksInstaller(tmp_path).uninstall()
    assert _obj(backup).same_object(original) or _obj(active).same_object(original), (
        "the customer's hook was deleted"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_install_never_tells_the_customer_to_delete_their_own_hook(
    tmp_path: Path, kind: str
) -> None:
    """The refusal must name whose hook it is, not call it an orphan.

    With the generated hook gone and the customer's original preserved, the
    generic "restore or remove it" advice, followed literally, deletes the very
    file the installer promised to hand back.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    assert HooksInstaller(tmp_path).install(force=True) == BOTH_TRUE

    # Something removes the generated hook; the customer's is still preserved.
    active.unlink()

    with pytest.raises(RuntimeError) as refusal:
        HooksInstaller(tmp_path).install(force=True)
    message = str(refusal.value)
    assert "holds the hook Traigent saved for you" in message, message
    assert "uninstall" in message, message
    assert "remove it" not in message, message
    assert _obj(backup).same_object(original)

    # And the advice is true: uninstall really does hand it back.
    assert HooksInstaller(tmp_path).uninstall()[kind] is True
    assert _obj(active).same_object(original)
    assert _obj(backup) == ABSENT


@pytest.mark.parametrize("kind", KINDS)
def test_rollback_drops_the_record_when_the_backup_is_already_gone(
    tmp_path: Path, kind: str
) -> None:
    """ "Someone else removed it" is the gone answer, not an unreadable one.

    Returning False there would keep a record describing a file that no longer
    exists. That residue is inert, because `holds_preserved_original` requires
    a real file with a matching digest, but it is still a record of something
    Traigent is not holding.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    real_replace = _replace

    def _remove_backup_then_fail(src: Any, dst: Any, *a: Any, **k: Any) -> None:
        if _named(dst, kind):
            _unlink(backup)
            raise _injected()
        real_replace(src, dst, *a, **k)

    with patch.object(os, "replace", _remove_backup_then_fail):
        with pytest.raises(PermissionError):
            HooksInstaller(tmp_path).install(force=True)

    assert _obj(backup) == ABSENT
    assert "backup" not in _read_ledger(tmp_path).get("hooks", {}).get(kind, {}), (
        "a record was kept for a backup that no longer exists"
    )
    # The customer's hook is still in the active slot, untouched.
    assert _obj(active).kind == "file"


@pytest.mark.parametrize("kind", KINDS)
def test_full_backup_lifecycle_keeps_the_customers_bytes(
    tmp_path: Path, kind: str
) -> None:
    """The exact sequence from the data-loss report, asserted at every step.

    install -> force -> force again -> uninstall -> uninstall with no backup.
    On the unfixed installer the second `--force` overwrote `<hook>.backup`
    with Traigent's own hook, so `uninstall` handed back the wrong file and
    the customer's bytes existed nowhere.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "foreign")
    original = _obj(active)
    installer = HooksInstaller(tmp_path)

    def customer_bytes_exist() -> bool:
        return any(
            p.is_file() and p.read_bytes() == original.data for p in hooks_dir.iterdir()
        )

    # install (no --force) must refuse to touch a hook it does not own.
    assert installer.install()[kind] is False
    assert _obj(active).same_object(original)

    # force, then force again -- the step that used to destroy the backup.
    for attempt in (1, 2):
        assert installer.install(force=True)[kind] is True, f"force #{attempt}"
        assert _is_new(_obj(active), kind), f"force #{attempt}"
        assert _obj(backup).same_object(original), (
            f"force #{attempt} overwrote the customer's backup"
        )
        assert customer_bytes_exist(), f"force #{attempt} lost the customer's bytes"

    # uninstall hands the original back, byte-for-byte and same inode.
    assert installer.uninstall()[kind] is True
    assert _obj(active).same_object(original)
    assert _obj(backup) == ABSENT
    assert customer_bytes_exist()

    # uninstall again, now with no backup present: the customer's hook is not
    # ours, so it is left exactly as it is and the call reports that.
    assert installer.uninstall()[kind] is False
    assert _obj(active).same_object(original)
    assert _obj(backup) == ABSENT
    assert customer_bytes_exist()


@pytest.mark.parametrize("kind", KINDS)
def test_uninstall_leaves_an_unrecorded_backup_exactly_where_it_is(
    tmp_path: Path, kind: str
) -> None:
    """With no record of whose backup it is, do not hand it back -- say so.

    Bytes cannot settle this: an artifact generated by a release the installer
    does not recognise looks just as foreign as a customer's own hook, which
    is the ambiguity the record exists to remove. So uninstall removes only
    the generated hook and leaves the unrecorded file untouched.
    """
    hooks_dir = _hooks_dir(tmp_path)
    active, backup = hooks_dir / kind, hooks_dir / f"{kind}.backup"
    _make_active(hooks_dir, kind, "current")
    _make_backup(hooks_dir, kind, "foreign")
    stranger = _obj(backup)
    assert not (tmp_path / ".git" / LEDGER_NAME).exists()

    assert HooksInstaller(tmp_path).uninstall()[kind] is True

    assert _obj(active) == ABSENT, "the generated hook should still be removed"
    assert _obj(backup).same_object(stranger), "an unrecorded backup was touched"
    # And the state is reported, not silent.
    assert HooksInstaller(tmp_path).status()[kind] == "not installed (backup present)"
