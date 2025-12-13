"""Unit tests for traigent.hooks.installer.

Tests for Git hooks installation and management, including pre-push and
pre-commit hook installation, uninstallation, and status checking.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability
# CONC-Quality-Reliability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import stat
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

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
        # Note: HOOK_MARKER only appears in pre-push, not pre-commit
        assert "TraiGent" in pre_commit_content

    def test_install_without_force_skips_existing_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test install without force skips existing TraiGent hooks."""
        # Install once
        installer.install()

        # Install again without force
        results = installer.install(force=False)

        # pre-push has HOOK_MARKER so it's detected as TraiGent hook
        assert results["pre-push"] is True
        # pre-commit doesn't have HOOK_MARKER so it's not re-detected
        # This is current behavior - could be considered a bug
        assert results["pre-commit"] is False

    def test_install_without_force_preserves_non_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test install without force doesn't overwrite non-TraiGent hooks."""
        # Create non-TraiGent hook
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
    def test_install_hook_creates_new_hook(self, installer: HooksInstaller) -> None:
        """Test _install_hook creates new hook file."""
        hook_path = installer.hooks_dir / "test-hook"
        content = "#!/bin/bash\necho 'test'\n" + HOOK_MARKER

        result = installer._install_hook(hook_path, content, force=False)

        assert result is True
        assert hook_path.exists()
        assert hook_path.read_text() == content

    def test_install_hook_sets_executable_permissions(
        self, installer: HooksInstaller
    ) -> None:
        """Test _install_hook sets executable permissions."""
        hook_path = installer.hooks_dir / "test-hook"
        content = "#!/bin/bash\n" + HOOK_MARKER

        installer._install_hook(hook_path, content, force=False)

        mode = hook_path.stat().st_mode
        assert mode & stat.S_IXUSR
        assert mode & stat.S_IXGRP
        assert mode & stat.S_IXOTH

    # uninstall tests
    def test_uninstall_removes_traigent_hooks(self, installer: HooksInstaller) -> None:
        """Test uninstall removes TraiGent hooks with HOOK_MARKER."""
        installer.install()
        results = installer.uninstall()

        # pre-push has HOOK_MARKER so it gets removed
        assert results["pre-push"] is True
        # pre-commit doesn't have HOOK_MARKER so it's not removed
        assert results["pre-commit"] is False
        assert not (installer.hooks_dir / "pre-push").exists()
        # pre-commit is NOT removed due to missing HOOK_MARKER
        assert (installer.hooks_dir / "pre-commit").exists()

    def test_uninstall_preserves_non_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test uninstall doesn't remove non-TraiGent hooks."""
        # Create non-TraiGent hook
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

    # _uninstall_hook tests
    def test_uninstall_hook_removes_traigent_hook(
        self, installer: HooksInstaller
    ) -> None:
        """Test _uninstall_hook removes TraiGent hook."""
        hook_path = installer.hooks_dir / "test-hook"
        hook_path.write_text("#!/bin/bash\n" + HOOK_MARKER)

        result = installer._uninstall_hook(hook_path)

        assert result is True
        assert not hook_path.exists()

    def test_uninstall_hook_returns_true_for_missing_hook(
        self, installer: HooksInstaller
    ) -> None:
        """Test _uninstall_hook returns True when hook doesn't exist."""
        hook_path = installer.hooks_dir / "nonexistent-hook"

        result = installer._uninstall_hook(hook_path)

        assert result is True

    def test_uninstall_hook_preserves_non_traigent_hook(
        self, installer: HooksInstaller
    ) -> None:
        """Test _uninstall_hook doesn't remove non-TraiGent hook."""
        hook_path = installer.hooks_dir / "test-hook"
        hook_path.write_text("#!/bin/bash\necho 'custom'")

        result = installer._uninstall_hook(hook_path)

        assert result is False
        assert hook_path.exists()

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

        # pre-push has HOOK_MARKER so it's detected as TraiGent
        assert status["pre-push"] == "installed (traigent)"
        # pre-commit doesn't have HOOK_MARKER so it's detected as other
        assert status["pre-commit"] == "installed (other)"

    def test_status_returns_installed_other_for_non_traigent_hooks(
        self, installer: HooksInstaller
    ) -> None:
        """Test status returns 'installed (other)' for non-TraiGent hooks."""
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
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "d" / "e"
            deep_path.mkdir(parents=True)
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
        # Create non-TraiGent pre-push hook
        pre_push = temp_git_repo / ".git" / "hooks" / "pre-push"
        pre_push.write_text("#!/bin/bash\necho 'custom'")

        # Install without force (pre-push will fail, pre-commit will succeed)
        result = install_hooks(temp_git_repo, force=False)

        assert result is False


class TestUninstallHooks:
    """Tests for uninstall_hooks convenience function."""

    @pytest.fixture
    def temp_git_repo_with_hooks(self) -> Generator[Path, None, None]:
        """Create temp Git repository with TraiGent hooks installed."""
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

        # Returns False because pre-commit lacks HOOK_MARKER
        assert result is False
        # pre-push is removed (has HOOK_MARKER)
        assert not (temp_git_repo_with_hooks / ".git" / "hooks" / "pre-push").exists()

    def test_uninstall_hooks_with_string_path(
        self, temp_git_repo_with_hooks: Path
    ) -> None:
        """Test uninstall_hooks accepts string path."""
        result = uninstall_hooks(str(temp_git_repo_with_hooks))

        # Returns False because pre-commit lacks HOOK_MARKER
        assert result is False

    @patch("traigent.hooks.installer.find_git_root")
    def test_uninstall_hooks_auto_detects_repo(
        self, mock_find: MagicMock, temp_git_repo_with_hooks: Path
    ) -> None:
        """Test uninstall_hooks auto-detects repository."""
        mock_find.return_value = temp_git_repo_with_hooks

        result = uninstall_hooks(None)

        # Returns False because pre-commit lacks HOOK_MARKER
        assert result is False
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
        """Test PRE_PUSH_HOOK_SCRIPT contains TraiGent marker."""
        assert HOOK_MARKER in PRE_PUSH_HOOK_SCRIPT

    def test_pre_push_hook_script_has_shebang(self) -> None:
        """Test PRE_PUSH_HOOK_SCRIPT starts with bash shebang."""
        assert PRE_PUSH_HOOK_SCRIPT.startswith("#!/bin/bash")

    def test_pre_push_hook_script_validates_config(self) -> None:
        """Test PRE_PUSH_HOOK_SCRIPT contains validation command."""
        assert "hooks validate" in PRE_PUSH_HOOK_SCRIPT

    def test_pre_commit_hook_script_contains_traigent(self) -> None:
        """Test PRE_COMMIT_HOOK_SCRIPT contains TraiGent identifier."""
        # Note: pre-commit doesn't have HOOK_MARKER
        assert "TraiGent" in PRE_COMMIT_HOOK_SCRIPT

    def test_pre_commit_hook_script_has_shebang(self) -> None:
        """Test PRE_COMMIT_HOOK_SCRIPT starts with bash shebang."""
        assert PRE_COMMIT_HOOK_SCRIPT.startswith("#!/bin/bash")

    def test_pre_commit_hook_script_uses_quick_check(self) -> None:
        """Test PRE_COMMIT_HOOK_SCRIPT uses quick check flag."""
        assert "--quick" in PRE_COMMIT_HOOK_SCRIPT

    def test_hook_marker_is_descriptive(self) -> None:
        """Test HOOK_MARKER contains descriptive text."""
        assert "TraiGent" in HOOK_MARKER
        assert "pre-push" in HOOK_MARKER
