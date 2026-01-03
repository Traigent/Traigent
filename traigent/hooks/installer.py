"""Git hooks installer for Traigent.

Installs and manages Git hooks that validate agent configurations
before pushes.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Reliability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import stat
from pathlib import Path

from traigent.utils.logging import get_logger
from traigent.utils.secure_path import (
    safe_read_text,
    safe_write_text,
    validate_path,
)

logger = get_logger(__name__)

# Pre-push hook script content
PRE_PUSH_HOOK_SCRIPT = """#!/bin/bash
# Traigent pre-push hook - validates agent configurations
# Installed by: traigent hooks install

set -e

# Check if traigent is available
if ! command -v traigent &> /dev/null; then
    # Try Python module invocation
    if ! python -m traigent --version &> /dev/null; then
        echo "Warning: traigent CLI not found, skipping validation"
        exit 0
    fi
    TRAIGENT_CMD="python -m traigent"
else
    TRAIGENT_CMD="traigent"
fi

echo "[traigent-validate] Checking agent configurations..."

# Run Traigent validation
if ! $TRAIGENT_CMD hooks validate --exit-code; then
    echo ""
    echo "PUSH REJECTED: Agent configuration violates constraints"
    echo ""
    echo "To bypass (not recommended):"
    echo "  git push --no-verify"
    echo ""
    echo "To fix:"
    echo "  Review the errors above and update your agent configuration"
    exit 1
fi

echo ""
echo "All Traigent hooks passed!"
exit 0
"""

# Pre-commit hook script content (lighter validation)
PRE_COMMIT_HOOK_SCRIPT = """#!/bin/bash
# Traigent pre-commit hook - quick config check
# Installed by: traigent hooks install

set -e

# Check if traigent is available
if ! command -v traigent &> /dev/null; then
    if ! python -m traigent --version &> /dev/null; then
        exit 0  # Skip if traigent not available
    fi
    TRAIGENT_CMD="python -m traigent"
else
    TRAIGENT_CMD="traigent"
fi

# Quick syntax check only (no full validation)
$TRAIGENT_CMD hooks check --quick
"""

# Hook marker to identify Traigent-installed hooks
HOOK_MARKER = "# Traigent pre-push hook - validates agent configurations"


class HooksInstaller:
    """Installs and manages Git hooks for Traigent."""

    def __init__(self, repo_path: Path | str | None = None) -> None:
        """Initialize hooks installer.

        Args:
            repo_path: Path to Git repository (defaults to current directory)
        """
        if repo_path is None:
            repo_path = Path.cwd()
        self.repo_path = Path(repo_path).resolve()
        self.hooks_dir = self.repo_path / ".git" / "hooks"

    def is_git_repo(self) -> bool:
        """Check if the path is a Git repository.

        Returns:
            True if .git directory exists
        """
        git_dir = self.repo_path / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def get_hooks_dir(self) -> Path:
        """Get the Git hooks directory, creating if necessary.

        Returns:
            Path to hooks directory

        Raises:
            RuntimeError: If not in a Git repository
        """
        if not self.is_git_repo():
            raise RuntimeError(f"Not a Git repository: {self.repo_path}")

        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        return self.hooks_dir

    def install(self, force: bool = False) -> dict[str, bool]:
        """Install Traigent Git hooks.

        Args:
            force: Overwrite existing hooks if present

        Returns:
            Dictionary mapping hook names to success status
        """
        hooks_dir = self.get_hooks_dir()
        results = {}

        # Install pre-push hook
        results["pre-push"] = self._install_hook(
            hooks_dir / "pre-push", PRE_PUSH_HOOK_SCRIPT, force
        )

        # Install pre-commit hook (optional, lighter check)
        results["pre-commit"] = self._install_hook(
            hooks_dir / "pre-commit", PRE_COMMIT_HOOK_SCRIPT, force
        )

        return results

    def _install_hook(self, hook_path: Path, content: str, force: bool) -> bool:
        """Install a single hook script.

        Args:
            hook_path: Path to hook file
            content: Hook script content
            force: Overwrite if exists

        Returns:
            True if hook was installed successfully
        """
        hooks_dir = self.get_hooks_dir()
        hook_path = validate_path(hook_path, hooks_dir, must_exist=False)

        if hook_path.exists():
            if not force:
                # Check if it's already a Traigent hook
                existing_content = safe_read_text(hook_path, hooks_dir)
                if HOOK_MARKER in existing_content:
                    logger.info(f"Traigent hook already installed at {hook_path}")
                    return True
                else:
                    logger.warning(
                        f"Existing hook at {hook_path} - use --force to overwrite"
                    )
                    return False

            # Backup existing hook
            backup_path = hook_path.with_suffix(".backup")
            hook_path.rename(backup_path)
            logger.info(f"Backed up existing hook to {backup_path}")

        # Write hook script
        safe_write_text(hook_path, content, hooks_dir)

        # Make executable
        current_mode = hook_path.stat().st_mode
        hook_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        logger.info(f"Installed hook at {hook_path}")
        return True

    def uninstall(self) -> dict[str, bool]:
        """Uninstall Traigent Git hooks.

        Returns:
            Dictionary mapping hook names to success status
        """
        hooks_dir = self.get_hooks_dir()
        results = {}

        for hook_name in ["pre-push", "pre-commit"]:
            hook_path = hooks_dir / hook_name
            results[hook_name] = self._uninstall_hook(hook_path)

        return results

    def _uninstall_hook(self, hook_path: Path) -> bool:
        """Uninstall a single hook.

        Args:
            hook_path: Path to hook file

        Returns:
            True if hook was uninstalled (or didn't exist)
        """
        if not hook_path.exists():
            return True

        # Only remove if it's a Traigent hook
        content = safe_read_text(hook_path, self.hooks_dir)
        if HOOK_MARKER not in content:
            logger.warning(
                f"Hook at {hook_path} was not installed by Traigent - skipping"
            )
            return False

        hook_path.unlink()
        logger.info(f"Removed hook at {hook_path}")

        # Restore backup if exists
        backup_path = hook_path.with_suffix(".backup")
        if backup_path.exists():
            backup_path.rename(hook_path)
            logger.info(f"Restored backup from {backup_path}")

        return True

    def status(self) -> dict[str, str]:
        """Check status of Traigent hooks.

        Returns:
            Dictionary mapping hook names to status strings
        """
        if not self.is_git_repo():
            return {"error": "Not a Git repository"}

        hooks_dir = self.hooks_dir
        status = {}

        for hook_name in ["pre-push", "pre-commit"]:
            hook_path = hooks_dir / hook_name

            if not hook_path.exists():
                status[hook_name] = "not installed"
            else:
                content = safe_read_text(hook_path, hooks_dir)
                if HOOK_MARKER in content:
                    status[hook_name] = "installed (traigent)"
                else:
                    status[hook_name] = "installed (other)"

        return status


def find_git_root(start_path: Path | None = None) -> Path | None:
    """Find the root of the Git repository.

    Args:
        start_path: Starting directory (defaults to cwd)

    Returns:
        Path to Git repository root, or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    while True:
        if (current / ".git").exists():
            return current

        parent = current.parent
        if parent == current:
            return None
        current = parent


def install_hooks(repo_path: Path | str | None = None, force: bool = False) -> bool:
    """Convenience function to install Traigent hooks.

    Args:
        repo_path: Path to Git repository (auto-detects if None)
        force: Overwrite existing hooks

    Returns:
        True if all hooks installed successfully
    """
    if repo_path is None:
        repo_path = find_git_root()
        if repo_path is None:
            logger.error("Not in a Git repository")
            return False

    installer = HooksInstaller(repo_path)

    try:
        results = installer.install(force=force)
        return all(results.values())
    except Exception as e:
        logger.error(f"Failed to install hooks: {e}")
        return False


def uninstall_hooks(repo_path: Path | str | None = None) -> bool:
    """Convenience function to uninstall Traigent hooks.

    Args:
        repo_path: Path to Git repository (auto-detects if None)

    Returns:
        True if all hooks uninstalled successfully
    """
    if repo_path is None:
        repo_path = find_git_root()
        if repo_path is None:
            logger.error("Not in a Git repository")
            return False

    installer = HooksInstaller(repo_path)

    try:
        results = installer.uninstall()
        return all(results.values())
    except Exception as e:
        logger.error(f"Failed to uninstall hooks: {e}")
        return False
