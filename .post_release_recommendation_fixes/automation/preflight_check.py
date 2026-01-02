#!/usr/bin/env python3
"""Pre-Flight Check for Post-Release Recommendation Fixes.

Verifies prerequisites before starting a fix session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from versioning import (
    ensure_within_base,
    read_tracking_version,
    resolve_base_path,
    resolve_version,
)


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        status = "PASS" if self.passed else self.severity.upper()
        return f"[{status}] {self.name}: {self.message}"


class PreflightChecker:
    """Run pre-flight checks before starting a fix session."""

    def __init__(
        self,
        base_path: str | Path = ".post_release_recommendation_fixes",
        source_todo: str | Path | None = None,
        version: str | None = None,
    ) -> None:
        """Initialize checker.

        Args:
            base_path: Base path for fix workflow
            source_todo: Path to source POST_RELEASE_TODO.md
            version: Release version override
        """
        self.root_path = Path(base_path)
        self.version = resolve_version(version)
        self.base_path = resolve_base_path(self.root_path, self.version)
        self.source_todo = Path(source_todo) if source_todo else None

    def _read_text_in_base(self, path: Path) -> str:
        """Read text from a path validated under the base path."""
        ensure_within_base(self.base_path, path)
        return path.read_text()

    def run_all_checks(self) -> list[CheckResult]:
        """Run all pre-flight checks.

        Returns:
            List of CheckResult objects
        """
        checks = [
            self.check_git_repo(),
            self.check_git_clean(),
            self.check_git_branch(),
            self.check_python_version(),
            self.check_venv(),
            self.check_pytest(),
            self.check_make_commands(),
            self.check_tracking_file(),
            self.check_release_version(),
            self.check_evidence_validation(),
            self.check_source_todo(),
            self.check_pre_release_tracking(),
            self.check_no_active_session(),
        ]
        return checks

    def check_git_repo(self) -> CheckResult:
        """Check if current directory is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return CheckResult(
                    name="Git Repository",
                    passed=True,
                    message="Inside git repository",
                )
            return CheckResult(
                name="Git Repository",
                passed=False,
                message="Not inside a git repository",
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return CheckResult(
                name="Git Repository",
                passed=False,
                message="Git not available",
            )

    def check_git_clean(self) -> CheckResult:
        """Check if working directory is clean (no uncommitted changes)."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and not result.stdout.strip():
                return CheckResult(
                    name="Git Clean",
                    passed=True,
                    message="Working directory is clean",
                )

            # Count modified files
            changes = len([l for l in result.stdout.strip().split("\n") if l])
            return CheckResult(
                name="Git Clean",
                passed=False,
                message=f"{changes} uncommitted changes. Commit or stash first.",
                severity="warning",
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return CheckResult(
                name="Git Clean",
                passed=False,
                message="Could not check git status",
                severity="warning",
            )

    def check_git_branch(self) -> CheckResult:
        """Check current git branch."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
                if branch in ["main", "master"]:
                    return CheckResult(
                        name="Git Branch",
                        passed=True,
                        message=f"On branch: {branch}",
                    )
                return CheckResult(
                    name="Git Branch",
                    passed=True,
                    message=f"On branch: {branch} (not main/master)",
                    severity="info",
                )
            return CheckResult(
                name="Git Branch",
                passed=False,
                message="Could not determine current branch",
                severity="warning",
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return CheckResult(
                name="Git Branch",
                passed=False,
                message="Git not available",
            )

    def check_python_version(self) -> CheckResult:
        """Check Python version is 3.8+."""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            return CheckResult(
                name="Python Version",
                passed=True,
                message=f"Python {version.major}.{version.minor}.{version.micro}",
            )
        return CheckResult(
            name="Python Version",
            passed=False,
            message=f"Python {version.major}.{version.minor} (need 3.8+)",
        )

    def check_venv(self) -> CheckResult:
        """Check if virtual environment exists and has python."""
        venv_paths = [
            Path(".venv/bin/python"),       # Unix
            Path(".venv/Scripts/python"),   # Windows
            Path("venv/bin/python"),        # Alternative name
        ]

        for venv_python in venv_paths:
            if venv_python.exists():
                try:
                    result = subprocess.run(
                        [str(venv_python), "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        return CheckResult(
                            name="Virtual Environment",
                            passed=True,
                            message=f"{venv_python.parent.parent}: {version}",
                        )
                except (subprocess.TimeoutExpired, OSError):
                    continue

        return CheckResult(
            name="Virtual Environment",
            passed=False,
            message="No venv found. Use: .venv/bin/python for scripts",
            severity="warning",
        )

    def check_pytest(self) -> CheckResult:
        """Check if pytest is available (checks venv first via python -m)."""
        # Try venv python with -m pytest first (more reliable than scripts)
        # IMPORTANT: Don't resolve() - keep using venv path to use venv packages
        python_locations = [
            ".venv/bin/python",       # Local venv (Unix)
            ".venv/Scripts/python",   # Local venv (Windows)
            "venv/bin/python",        # Alternative venv name
        ]

        for python_cmd in python_locations:
            python_path = Path(python_cmd)
            if python_path.exists():
                try:
                    # Use the venv path directly, not resolved
                    result = subprocess.run(
                        [python_cmd, "-m", "pytest", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        version_line = result.stdout.strip().split("\n")[0]
                        return CheckResult(
                            name="Pytest",
                            passed=True,
                            message=f"{version_line} (venv)",
                        )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue

        # Try system pytest via shutil.which
        pytest_system = shutil.which("pytest")
        if pytest_system:
            try:
                result = subprocess.run(
                    [pytest_system, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    version_line = result.stdout.strip().split("\n")[0]
                    return CheckResult(
                        name="Pytest",
                        passed=True,
                        message=f"{version_line} (system)",
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # None found
        return CheckResult(
            name="Pytest",
            passed=False,
            message="pytest not found. Run: pip install pytest",
        )

    def check_make_commands(self) -> CheckResult:
        """Check if make format and make lint are available."""
        makefile = Path("Makefile")
        if not makefile.exists():
            return CheckResult(
                name="Make Commands",
                passed=False,
                message="No Makefile found",
                severity="warning",
            )

        content = makefile.read_text()
        has_format = "format:" in content
        has_lint = "lint:" in content

        if has_format and has_lint:
            return CheckResult(
                name="Make Commands",
                passed=True,
                message="make format and make lint available",
            )

        missing = []
        if not has_format:
            missing.append("format")
        if not has_lint:
            missing.append("lint")

        return CheckResult(
            name="Make Commands",
            passed=False,
            message=f"Missing targets: {', '.join(missing)}",
            severity="warning",
        )

    def check_tracking_file(self) -> CheckResult:
        """Check if TRACKING.md exists and has content."""
        tracking_path = self.base_path / "TRACKING.md"

        if not tracking_path.exists():
            return CheckResult(
                name="Tracking File",
                passed=False,
                message="TRACKING.md not found. Run todo_importer.py first.",
            )

        content = self._read_text_in_base(tracking_path)

        # Check if it has actual items (not just template)
        if "| Pending |" in content or "| Complete |" in content:
            return CheckResult(
                name="Tracking File",
                passed=True,
                message="TRACKING.md exists with items",
            )

        return CheckResult(
            name="Tracking File",
            passed=False,
            message="TRACKING.md exists but has no items. Run todo_importer.py.",
            severity="warning",
        )

    def check_release_version(self) -> CheckResult:
        """Check release version alignment between env and tracking file."""
        tracking_path = self.base_path / "TRACKING.md"
        tracking_version = read_tracking_version(tracking_path, self.base_path)
        env_version = self.version

        if not env_version and not tracking_version:
            return CheckResult(
                name="Release Version",
                passed=False,
                message="No release version set (RR_VERSION) and none found in tracking",
                severity="warning",
            )

        if tracking_version and env_version and tracking_version != env_version:
            return CheckResult(
                name="Release Version",
                passed=False,
                message=f"Mismatch: tracking={tracking_version}, RR_VERSION={env_version}",
                severity="error",
            )

        version = tracking_version or env_version or "UNKNOWN"
        return CheckResult(
            name="Release Version",
            passed=True,
            message=f"Version: {version}",
        )

    def check_evidence_validation(self) -> CheckResult:
        """Check that tracking evidence is machine-valid JSON."""
        tracking_path = self.base_path / "TRACKING.md"
        validator_path = Path(".release_review/automation/evidence_validator.py")

        if not tracking_path.exists():
            return CheckResult(
                name="Evidence Validation",
                passed=True,
                message="Tracking file not found (skip evidence validation)",
                severity="info",
            )

        if not validator_path.exists():
            return CheckResult(
                name="Evidence Validation",
                passed=False,
                message="Evidence validator not found at .release_review/automation/evidence_validator.py",
                severity="warning",
            )

        python_cmd = self._resolve_python_command()
        try:
            result = subprocess.run(
                [python_cmd, str(validator_path), "--file", str(tracking_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return CheckResult(
                name="Evidence Validation",
                passed=False,
                message="Failed to run evidence validator",
                severity="warning",
            )

        if result.returncode == 0:
            return CheckResult(
                name="Evidence Validation",
                passed=True,
                message="Tracking evidence is machine-valid JSON",
            )

        error = result.stderr.strip() or result.stdout.strip() or "Validation failed"
        return CheckResult(
            name="Evidence Validation",
            passed=False,
            message=error,
            severity="error",
        )

    def check_source_todo(self) -> CheckResult:
        """Check if source POST_RELEASE_TODO.md exists."""
        if not self.source_todo:
            # Try to find one
            release_review = Path(".release_review")
            if release_review.exists():
                if self.version:
                    candidate = release_review / self.version / "POST_RELEASE_TODO.md"
                    if candidate.exists():
                        return CheckResult(
                            name="Source TODO",
                            passed=True,
                            message=f"Found: {candidate}",
                            severity="info",
                        )
                todos = list(release_review.glob("*/POST_RELEASE_TODO.md"))
                if todos:
                    return CheckResult(
                        name="Source TODO",
                        passed=True,
                        message=f"Found: {todos[0]}",
                        severity="info",
                    )
            return CheckResult(
                name="Source TODO",
                passed=True,
                message="No source specified (optional)",
                severity="info",
            )

        if self.source_todo.exists():
            return CheckResult(
                name="Source TODO",
                passed=True,
                message=f"Found: {self.source_todo}",
            )

        return CheckResult(
            name="Source TODO",
            passed=False,
            message=f"Not found: {self.source_todo}",
        )

    def check_pre_release_tracking(self) -> CheckResult:
        """Check for pre-release tracking file presence."""
        if not self.version:
            return CheckResult(
                name="Pre-Release Tracking",
                passed=True,
                message="RR_VERSION not set (skip pre-release tracking check)",
                severity="info",
            )

        tracking = Path(".release_review") / self.version / "PRE_RELEASE_REVIEW_TRACKING.md"
        if tracking.exists():
            return CheckResult(
                name="Pre-Release Tracking",
                passed=True,
                message=f"Found: {tracking}",
            )

        return CheckResult(
            name="Pre-Release Tracking",
            passed=False,
            message=f"Missing: {tracking}",
            severity="warning",
        )

    def _resolve_python_command(self) -> str:
        """Resolve python path for tooling."""
        python_candidates = [
            ".venv/bin/python",
            ".venv/Scripts/python",
            "venv/bin/python",
        ]
        for candidate in python_candidates:
            if Path(candidate).exists():
                return candidate
        return sys.executable

    def check_no_active_session(self) -> CheckResult:
        """Check if there's an active (in-progress) session."""
        sessions_dir = self.base_path / "sessions"
        if not sessions_dir.exists():
            return CheckResult(
                name="Active Session",
                passed=True,
                message="No sessions directory",
                severity="info",
            )

        # Look for sessions with "In progress" in PROGRESS.md
        active_sessions = []
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir() or session_dir.name == "TEMPLATE":
                continue

            progress_file = session_dir / "PROGRESS.md"
            if progress_file.exists():
                content = self._read_text_in_base(progress_file)
                if "**Ended**: (In progress)" in content:
                    active_sessions.append(session_dir.name)

        if active_sessions:
            return CheckResult(
                name="Active Session",
                passed=True,
                message=f"Active: {', '.join(active_sessions)}. Consider resuming.",
                severity="warning",
            )

        return CheckResult(
            name="Active Session",
            passed=True,
            message="No active sessions",
        )

    def print_report(self, results: list[CheckResult]) -> bool:
        """Print pre-flight check report.

        Args:
            results: List of check results

        Returns:
            True if all critical checks passed
        """
        print("=" * 60)
        print("PRE-FLIGHT CHECK REPORT")
        print("=" * 60)
        print()

        errors = []
        warnings = []

        for result in results:
            print(result)
            if not result.passed:
                if result.severity == "error":
                    errors.append(result)
                elif result.severity == "warning":
                    warnings.append(result)

        print()
        print("-" * 60)

        if errors:
            print(f"ERRORS: {len(errors)} critical issues found")
            for err in errors:
                print(f"  - {err.name}: {err.message}")
            print()
            print("Fix these issues before starting a session.")
            return False

        if warnings:
            print(f"WARNINGS: {len(warnings)} non-critical issues")
            for warn in warnings:
                print(f"  - {warn.name}: {warn.message}")
            print()

        print("All critical checks passed. Ready to start session.")
        return True


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight check for fix sessions"
    )
    parser.add_argument(
        "--source",
        help="Path to source POST_RELEASE_TODO.md",
    )
    parser.add_argument(
        "--version",
        help="Release version override (defaults to RR_VERSION)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output if checks fail",
    )

    args = parser.parse_args()

    checker = PreflightChecker(source_todo=args.source, version=args.version)
    results = checker.run_all_checks()

    if args.quiet:
        errors = [r for r in results if not r.passed and r.severity == "error"]
        if errors:
            for err in errors:
                print(f"ERROR: {err.name}: {err.message}")
            sys.exit(1)
        sys.exit(0)

    passed = checker.print_report(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
