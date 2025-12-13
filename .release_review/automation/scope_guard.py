#!/usr/bin/env python3
"""Scope Guard for Release Review Protocol.

Validates that agent changes are within their assigned scope.
Used by captain before approving any agent work.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Any


class ScopeGuard:
    """Validate agent changes stay within assigned scope."""

    def __init__(self, repo_path: str | Path | None = None) -> None:
        """Initialize scope guard.

        Args:
            repo_path: Path to git repository. Defaults to current directory.
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    def get_modified_files(
        self,
        branch: str,
        base_branch: str = "main",
    ) -> list[str]:
        """Get list of files modified in branch compared to base.

        Args:
            branch: Branch to check
            base_branch: Base branch to compare against

        Returns:
            List of modified file paths
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_branch}...{branch}"],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                check=True,
            )
            files = result.stdout.strip().split("\n")
            return [f for f in files if f]  # Filter empty strings
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git diff failed: {e.stderr}") from e

    def validate_changes(
        self,
        branch: str,
        allowed_paths: list[str],
        base_branch: str = "main",
    ) -> dict[str, Any]:
        """Check if all changes are within assigned scope.

        Args:
            branch: Branch to validate
            allowed_paths: List of allowed path patterns (supports glob)
            base_branch: Base branch to compare against

        Returns:
            Validation result with violations list
        """
        modified_files = self.get_modified_files(branch, base_branch)

        if not modified_files:
            return {
                "valid": True,
                "violations": [],
                "total_changes": 0,
                "message": "No changes detected",
            }

        violations = []
        for file in modified_files:
            if not self._is_allowed(file, allowed_paths):
                violations.append(file)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "total_changes": len(modified_files),
            "allowed_changes": len(modified_files) - len(violations),
            "message": (
                "All changes within scope"
                if not violations
                else f"SCOPE VIOLATION: {len(violations)} files outside assigned scope"
            ),
        }

    def _is_allowed(self, file: str, allowed_paths: list[str]) -> bool:
        """Check if a file path matches any allowed pattern.

        Args:
            file: File path to check
            allowed_paths: List of allowed patterns

        Returns:
            True if file is allowed
        """
        for pattern in allowed_paths:
            # Handle directory patterns (e.g., "traigent/core/")
            if pattern.endswith("/"):
                if file.startswith(pattern) or file.startswith(pattern.rstrip("/")):
                    return True
            # Handle glob patterns (e.g., "traigent/**/*.py")
            elif "*" in pattern:
                if fnmatch.fnmatch(file, pattern):
                    return True
            # Handle exact matches
            elif file == pattern or file.startswith(pattern + "/"):
                return True

        return False

    def validate_staged_changes(
        self,
        allowed_paths: list[str],
    ) -> dict[str, Any]:
        """Validate currently staged changes.

        Args:
            allowed_paths: List of allowed path patterns

        Returns:
            Validation result
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                check=True,
            )
            staged_files = result.stdout.strip().split("\n")
            staged_files = [f for f in staged_files if f]
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git diff failed: {e.stderr}") from e

        if not staged_files:
            return {
                "valid": True,
                "violations": [],
                "total_staged": 0,
                "message": "No staged changes",
            }

        violations = []
        for file in staged_files:
            if not self._is_allowed(file, allowed_paths):
                violations.append(file)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "total_staged": len(staged_files),
            "message": (
                "All staged changes within scope"
                if not violations
                else f"SCOPE VIOLATION: {violations}"
            ),
        }

    def generate_scope_declaration(
        self,
        component: str,
        allowed_paths: list[str],
    ) -> str:
        """Generate scope declaration for agent pre-flight.

        Args:
            component: Component name
            allowed_paths: Assigned paths

        Returns:
            Markdown scope declaration
        """
        paths_list = "\n".join(f"  - `{p}`" for p in allowed_paths)
        return f"""## Agent Scope Declaration

**Component**: {component}
**Assigned Paths**:
{paths_list}

### Pre-flight Checklist

- [ ] I have read my assigned component list
- [ ] I will NOT modify files outside my scope
- [ ] I will NOT merge branches (Captain-only)
- [ ] I will flag cross-component issues to Captain instead of fixing directly
"""


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: scope_guard.py <branch> <allowed_path1> [allowed_path2] ...")
        print("Example: scope_guard.py review/core/claude/20251213 traigent/core/")
        sys.exit(1)

    branch = sys.argv[1]
    allowed_paths = sys.argv[2:]

    guard = ScopeGuard()

    try:
        result = guard.validate_changes(branch, allowed_paths)

        if result["valid"]:
            print(f"✅ {result['message']}")
            print(f"   Total changes: {result['total_changes']}")
        else:
            print(f"❌ {result['message']}")
            print("   Violations:")
            for v in result["violations"]:
                print(f"     - {v}")
            sys.exit(1)

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
