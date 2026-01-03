#!/usr/bin/env python3
"""Git Helper for Post-Release Recommendation Fixes.

Utilities for branch management, merging, and rollback.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BranchInfo:
    """Information about a fix branch."""

    name: str
    fix_id: str
    description: str
    exists: bool = False
    ahead: int = 0
    behind: int = 0


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    message: str
    commit_sha: str = ""
    conflicts: list[str] | None = None


class GitHelper:
    """Git utilities for fix workflow."""

    BRANCH_PREFIX = "fix"

    def __init__(self, repo_path: str | Path = ".") -> None:
        """Initialize git helper.

        Args:
            repo_path: Path to git repository
        """
        self.repo_path = Path(repo_path)

    def _run_git(
        self,
        *args: str,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command.

        Args:
            *args: Git command arguments
            check: Raise exception on non-zero exit
            capture: Capture stdout/stderr

        Returns:
            CompletedProcess result
        """
        cmd = ["git"] + list(args)
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=capture,
            text=True,
            check=check,
            timeout=30,
        )

    def create_fix_branch(
        self,
        fix_id: str,
        description: str,
        base: str | None = None,
    ) -> BranchInfo:
        """Create a fix branch.

        Args:
            fix_id: Fix ID (e.g., "001")
            description: Short description (will be slugified)
            base: Base branch to create from

        Returns:
            BranchInfo for created branch
        """
        # Slugify description
        slug = re.sub(r"[^a-z0-9]+", "-", description.lower())[:30].strip("-")
        branch_name = f"{self.BRANCH_PREFIX}/{fix_id}/{slug}"

        # Check if branch exists
        result = self._run_git("branch", "--list", branch_name, check=False)
        if result.stdout.strip():
            # Branch exists, just checkout
            self._run_git("checkout", branch_name)
            return BranchInfo(
                name=branch_name,
                fix_id=fix_id,
                description=description,
                exists=True,
            )

        # Create new branch
        base_branch = base or os.environ.get("RR_BASE_BRANCH", "main")
        self._run_git("checkout", "-b", branch_name, base_branch)

        return BranchInfo(
            name=branch_name,
            fix_id=fix_id,
            description=description,
            exists=True,
        )

    def get_current_branch(self) -> str:
        """Get current branch name.

        Returns:
            Branch name
        """
        result = self._run_git("branch", "--show-current")
        return result.stdout.strip()

    def get_fix_branches(self) -> list[BranchInfo]:
        """Get all fix branches.

        Returns:
            List of BranchInfo objects
        """
        result = self._run_git("branch", "--list", f"{self.BRANCH_PREFIX}/*")
        branches = []

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            name = line.strip().lstrip("* ")

            # Parse fix ID and description from branch name
            match = re.match(rf"{self.BRANCH_PREFIX}/(\d+)/(.+)", name)
            if match:
                branches.append(BranchInfo(
                    name=name,
                    fix_id=match.group(1),
                    description=match.group(2).replace("-", " "),
                    exists=True,
                ))

        return branches

    def merge_fix_branch(
        self,
        branch: str,
        target: str | None = None,
        squash: bool = True,
    ) -> MergeResult:
        """Merge a fix branch to target.

        Args:
            branch: Branch to merge
            target: Target branch
            squash: Use squash merge

        Returns:
            MergeResult
        """
        # Checkout target
        target_branch = target or os.environ.get("RR_BASE_BRANCH", "main")
        self._run_git("checkout", target_branch)

        # Attempt merge
        merge_args = ["merge"]
        if squash:
            merge_args.append("--squash")
        merge_args.append(branch)

        result = self._run_git(*merge_args, check=False)

        if result.returncode != 0:
            # Check for conflicts
            conflicts = self._get_conflicts()
            if conflicts:
                return MergeResult(
                    success=False,
                    message="Merge conflicts detected",
                    conflicts=conflicts,
                )
            return MergeResult(
                success=False,
                message=result.stderr.strip(),
            )

        # Get commit SHA if squash merge, need to commit
        if squash:
            return MergeResult(
                success=True,
                message="Squash merge staged. Commit needed.",
            )

        # Get merge commit SHA
        sha_result = self._run_git("rev-parse", "HEAD")
        return MergeResult(
            success=True,
            message="Merge successful",
            commit_sha=sha_result.stdout.strip()[:7],
        )

    def _get_conflicts(self) -> list[str]:
        """Get list of conflicting files.

        Returns:
            List of conflicting file paths
        """
        result = self._run_git("diff", "--name-only", "--diff-filter=U", check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")
        return []

    def abort_merge(self) -> bool:
        """Abort an in-progress merge.

        Returns:
            True if successful
        """
        result = self._run_git("merge", "--abort", check=False)
        return result.returncode == 0

    def delete_branch(self, branch: str, force: bool = False) -> bool:
        """Delete a branch.

        Args:
            branch: Branch name to delete
            force: Force delete even if not merged

        Returns:
            True if successful
        """
        flag = "-D" if force else "-d"
        result = self._run_git("branch", flag, branch, check=False)
        return result.returncode == 0

    def get_commit_sha(self, ref: str = "HEAD") -> str:
        """Get commit SHA for a reference.

        Args:
            ref: Git reference (branch, tag, HEAD, etc.)

        Returns:
            Short SHA
        """
        result = self._run_git("rev-parse", "--short", ref)
        return result.stdout.strip()

    def revert_commit(self, commit: str, no_commit: bool = False) -> MergeResult:
        """Revert a commit.

        Args:
            commit: Commit SHA to revert
            no_commit: Stage revert without committing

        Returns:
            MergeResult
        """
        args = ["revert"]
        if no_commit:
            args.append("--no-commit")
        args.append(commit)

        result = self._run_git(*args, check=False)

        if result.returncode != 0:
            conflicts = self._get_conflicts()
            if conflicts:
                return MergeResult(
                    success=False,
                    message="Revert conflicts detected",
                    conflicts=conflicts,
                )
            return MergeResult(
                success=False,
                message=result.stderr.strip(),
            )

        if no_commit:
            return MergeResult(
                success=True,
                message="Revert staged. Commit needed.",
            )

        sha_result = self._run_git("rev-parse", "--short", "HEAD")
        return MergeResult(
            success=True,
            message="Revert successful",
            commit_sha=sha_result.stdout.strip(),
        )

    def get_branch_status(self, branch: str, target: str | None = None) -> BranchInfo:
        """Get ahead/behind status for a branch.

        Args:
            branch: Branch to check
            target: Target branch for comparison

        Returns:
            BranchInfo with ahead/behind counts
        """
        # Get ahead/behind counts
        target_branch = target or os.environ.get("RR_BASE_BRANCH", "main")
        result = self._run_git(
            "rev-list",
            "--left-right",
            "--count",
            f"{target_branch}...{branch}",
            check=False,
        )

        ahead, behind = 0, 0
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                behind, ahead = int(parts[0]), int(parts[1])

        # Parse fix ID from branch name
        match = re.match(rf"{self.BRANCH_PREFIX}/(\d+)/(.+)", branch)
        fix_id = match.group(1) if match else ""
        description = match.group(2).replace("-", " ") if match else ""

        return BranchInfo(
            name=branch,
            fix_id=fix_id,
            description=description,
            exists=True,
            ahead=ahead,
            behind=behind,
        )

    def run_tests(self, test_path: str = "tests/") -> tuple[bool, str]:
        """Run tests with TRAIGENT_MOCK_LLM.

        Args:
            test_path: Path to tests

        Returns:
            Tuple of (passed, output summary)
        """
        import os

        env = os.environ.copy()
        env["TRAIGENT_MOCK_LLM"] = "true"
        test_command = env.get("RR_TEST_COMMAND")

        if test_command:
            cmd = shlex.split(test_command)
        else:
            python_cmd = ".venv/bin/python"
            if not Path(python_cmd).exists():
                python_cmd = sys.executable
            cmd = [python_cmd, "-m", "pytest", test_path, "-q", "--tb=no"]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
            )
            summary = self._extract_pytest_summary(result.stdout, result.stderr)
            return result.returncode == 0, summary

        except subprocess.TimeoutExpired:
            return False, "Test timeout (5 min)"
        except FileNotFoundError:
            return False, "pytest not found"

    def _extract_pytest_summary(self, stdout: str, stderr: str) -> str:
        """Extract a stable pytest summary line from output."""
        combined = []
        if stdout:
            combined.extend(stdout.splitlines())
        if stderr:
            combined.extend(stderr.splitlines())

        summary_pattern = re.compile(
            r"\b\d+\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)\b",
            re.IGNORECASE,
        )

        for line in reversed(combined):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("=") and stripped.endswith("="):
                return stripped
            if summary_pattern.search(stripped):
                return stripped

        return "Unknown"

    def format_and_lint(self) -> tuple[bool, str]:
        """Run make format and make lint.

        Returns:
            Tuple of (passed, output summary)
        """
        try:
            # Run format
            format_result = subprocess.run(
                ["make", "format"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if format_result.returncode != 0:
                return False, f"Format failed: {format_result.stderr[:100]}"

            # Run lint
            lint_result = subprocess.run(
                ["make", "lint"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if lint_result.returncode != 0:
                return False, f"Lint failed: {lint_result.stderr[:100]}"

            return True, "Format and lint passed"

        except subprocess.TimeoutExpired:
            return False, "Timeout during format/lint"
        except FileNotFoundError:
            return False, "make not found"


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: git_helper.py <command> [args]")
        print("Commands:")
        print("  branch <fix_id> <description>  - Create fix branch")
        print("  list                           - List fix branches")
        print("  status <branch>                - Show branch status")
        print("  merge <branch> [target]        - Merge fix branch")
        print("  delete <branch> [--force]      - Delete branch")
        print("  revert <commit>                - Revert a commit")
        print("  test [path]                    - Run tests")
        print("  lint                           - Run format and lint")
        sys.exit(1)

    command = sys.argv[1]
    helper = GitHelper()

    if command == "branch":
        if len(sys.argv) < 4:
            print("Usage: git_helper.py branch <fix_id> <description>")
            sys.exit(1)

        fix_id = sys.argv[2]
        description = " ".join(sys.argv[3:])
        info = helper.create_fix_branch(fix_id, description)
        print(f"Branch: {info.name}")
        print(f"Fix ID: {info.fix_id}")

    elif command == "list":
        branches = helper.get_fix_branches()
        if branches:
            print("Fix branches:")
            for b in branches:
                print(f"  {b.fix_id}: {b.name}")
        else:
            print("No fix branches found")

    elif command == "status":
        if len(sys.argv) < 3:
            branch = helper.get_current_branch()
        else:
            branch = sys.argv[2]

        info = helper.get_branch_status(branch)
        print(f"Branch: {info.name}")
        print(f"Ahead: {info.ahead} commits")
        print(f"Behind: {info.behind} commits")

    elif command == "merge":
        if len(sys.argv) < 3:
            print("Usage: git_helper.py merge <branch> [target]")
            sys.exit(1)

        branch = sys.argv[2]
        target = sys.argv[3] if len(sys.argv) > 3 else None

        result = helper.merge_fix_branch(branch, target)
        if result.success:
            print(f"OK: {result.message}")
            if result.commit_sha:
                print(f"Commit: {result.commit_sha}")
        else:
            print(f"FAILED: {result.message}")
            if result.conflicts:
                print("Conflicts:")
                for f in result.conflicts:
                    print(f"  - {f}")
            sys.exit(1)

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: git_helper.py delete <branch> [--force]")
            sys.exit(1)

        branch = sys.argv[2]
        force = "--force" in sys.argv

        if helper.delete_branch(branch, force):
            print(f"Deleted: {branch}")
        else:
            print(f"Failed to delete: {branch}")
            sys.exit(1)

    elif command == "revert":
        if len(sys.argv) < 3:
            print("Usage: git_helper.py revert <commit>")
            sys.exit(1)

        commit = sys.argv[2]
        result = helper.revert_commit(commit)

        if result.success:
            print(f"OK: {result.message}")
            if result.commit_sha:
                print(f"Revert commit: {result.commit_sha}")
        else:
            print(f"FAILED: {result.message}")
            if result.conflicts:
                print("Conflicts:")
                for f in result.conflicts:
                    print(f"  - {f}")
            sys.exit(1)

    elif command == "test":
        test_path = sys.argv[2] if len(sys.argv) > 2 else "tests/"
        passed, summary = helper.run_tests(test_path)
        print(f"Tests: {'PASS' if passed else 'FAIL'}")
        print(f"Summary: {summary}")
        sys.exit(0 if passed else 1)

    elif command == "lint":
        passed, summary = helper.format_and_lint()
        print(f"Lint: {'PASS' if passed else 'FAIL'}")
        print(f"Summary: {summary}")
        sys.exit(0 if passed else 1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
