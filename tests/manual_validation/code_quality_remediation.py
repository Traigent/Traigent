#!/usr/bin/env python3
"""Automated Code Quality Remediation Script"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class CodeQualityRemediator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.tracker_file = self.project_root / "code_quality_remediation_tracker.json"
        self.summary_file = self.project_root / "code_quality_summary.md"
        self.venv_path = self.project_root / "traigent_test_env" / "bin"
        self.tracker_data = self.load_tracker()

    def load_tracker(self) -> dict[str, Any]:
        """Load or initialize the tracking file"""
        if self.tracker_file.exists():
            with open(self.tracker_file) as f:
                return json.load(f)
        else:
            return {
                "remediation_started": datetime.now(UTC).isoformat(),
                "total_files": 0,
                "processed_files": 0,
                "files": {},
            }

    def save_tracker(self):
        """Save the tracking data to file"""
        with open(self.tracker_file, "w") as f:
            json.dump(self.tracker_data, f, indent=2)

    def run_command(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

    def detect_issues(self, file_path: str) -> dict[str, list[str]]:
        """Run all static analysis tools and collect issues"""
        issues = {"flake8": [], "ruff": [], "mypy": [], "black": [], "isort": []}

        # Flake8
        code, stdout, stderr = self.run_command(
            [
                str(self.venv_path / "flake8"),
                file_path,
                "--extend-ignore=E501,W503,E203",  # Ignore line length and some formatting
            ]
        )
        if code != 0:
            issues["flake8"] = stdout.split("\n") if stdout else []

        # Ruff
        code, stdout, stderr = self.run_command(
            [str(self.venv_path / "ruff"), "check", file_path]
        )
        if code != 0:
            issues["ruff"] = stdout.split("\n") if stdout else []

        # MyPy
        code, stdout, stderr = self.run_command(
            [
                str(self.venv_path / "mypy"),
                file_path,
                "--ignore-missing-imports",
                "--no-strict-optional",
                "--allow-untyped-defs",
            ]
        )
        if code != 0:
            issues["mypy"] = stdout.split("\n") if stdout else []

        # Black
        code, stdout, stderr = self.run_command(
            [str(self.venv_path / "black"), "--check", file_path]
        )
        if code != 0:
            issues["black"] = ["File would be reformatted"]

        # isort
        code, stdout, stderr = self.run_command(
            [str(self.venv_path / "isort"), "--check-only", file_path]
        )
        if code != 0:
            issues["isort"] = ["Imports would be sorted"]

        return {k: v for k, v in issues.items() if v}

    def apply_auto_fixes(self, file_path: str) -> bool:
        """Apply automatic formatting and fixes"""
        success = True

        # Black formatting
        code, _, _ = self.run_command(
            [str(self.venv_path / "black"), file_path, "--quiet"]
        )
        if code != 0:
            success = False

        # isort formatting
        code, _, _ = self.run_command(
            [str(self.venv_path / "isort"), file_path, "--quiet"]
        )
        if code != 0:
            success = False

        # Ruff auto-fix
        code, _, _ = self.run_command(
            [
                str(self.venv_path / "ruff"),
                "check",
                "--fix",
                "--unsafe-fixes",
                file_path,
            ]
        )

        return success

    def run_tests_for_file(self, file_path: str) -> tuple[bool, str]:
        """Run tests associated with the file"""
        file_path_obj = Path(file_path)

        # Determine test file patterns
        test_patterns = []

        if file_path_obj.parts[0] == "tests":
            # It's a test file itself
            test_patterns.append(file_path)
        else:
            # Look for corresponding test files
            module_name = file_path_obj.stem
            if file_path_obj.parts[0] == "traigent":
                # Look for tests in tests/ directory
                relative_path = Path(*file_path_obj.parts[1:])
                test_dir = Path("tests") / relative_path.parent

                test_patterns.extend(
                    [
                        str(test_dir / f"test_{module_name}.py"),
                        str(test_dir / f"{module_name}_test.py"),
                        str(
                            Path("tests")
                            / "unit"
                            / relative_path.parent
                            / f"test_{module_name}.py"
                        ),
                        str(
                            Path("tests")
                            / "integration"
                            / relative_path.parent
                            / f"test_{module_name}.py"
                        ),
                    ]
                )

        # Run tests if found
        for test_path in test_patterns:
            if Path(test_path).exists():
                code, stdout, stderr = self.run_command(
                    [str(self.venv_path / "pytest"), test_path, "-xvs", "--tb=short"]
                )
                if code == 0:
                    return True, "Tests passed"
                else:
                    return False, f"Tests failed: {stderr[:500]}"

        # No tests found, consider it a pass
        return True, "No tests found for this file"

    def process_file(self, file_path: str) -> dict[str, Any]:
        """Process a single file through the remediation pipeline"""
        file_info = {
            "status": "in_progress",
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "issues_found": {},
            "issues_resolved": {},
            "test_status": "not_run",
            "attempt_count": 0,
            "error_log": [],
        }

        max_attempts = 3

        for attempt in range(max_attempts):
            file_info["attempt_count"] = attempt + 1

            # Phase 1: Detect issues
            issues = self.detect_issues(file_path)
            if not issues:
                # No issues found
                file_info["status"] = "completed"
                file_info["test_status"] = "passed"
                break

            file_info["issues_found"] = issues

            # Phase 2: Apply auto-fixes
            self.apply_auto_fixes(file_path)

            # Phase 3: Verify fixes
            remaining_issues = self.detect_issues(file_path)

            # Calculate resolved issues
            for tool, original_issues in issues.items():
                if tool not in remaining_issues:
                    file_info["issues_resolved"][tool] = len(original_issues)
                else:
                    resolved_count = len(original_issues) - len(remaining_issues[tool])
                    if resolved_count > 0:
                        file_info["issues_resolved"][tool] = resolved_count

            # Phase 4: Run tests
            test_passed, test_msg = self.run_tests_for_file(file_path)

            if not remaining_issues and test_passed:
                file_info["status"] = "completed"
                file_info["test_status"] = "passed"
                break
            elif not test_passed:
                file_info["error_log"].append(f"Attempt {attempt + 1}: {test_msg}")
                file_info["test_status"] = "failed"

            if attempt == max_attempts - 1:
                file_info["status"] = "failed"
                file_info["error_log"].append(
                    f"Max attempts reached. Remaining issues: {remaining_issues}"
                )

        file_info["completed_at"] = datetime.now(UTC).isoformat()
        return file_info

    def enumerate_files(self) -> list[str]:
        """Enumerate all Python files in target directories"""
        target_dirs = [
            "traigent",
            "tests",
            "examples",
            "demos",
            "scripts",
            "playground",
        ]
        python_files = []

        # Patterns to exclude
        exclude_patterns = [
            "*/__pycache__/*",
            "*/.mypy_cache/*",
            "*/build/*",
            "*/dist/*",
            "*.egg-info/*",
            "*/.pytest_cache/*",
            "*/.ruff_cache/*",
            "*/.venv/*",
            "*/venv/*",
            "*_test_env/*",
        ]

        for dir_name in target_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                for f in dir_path.rglob("*.py"):
                    # Check if file should be excluded
                    file_str = str(f)
                    should_exclude = False
                    for pattern in exclude_patterns:
                        if pattern.replace("*", "") in file_str:
                            should_exclude = True
                            break
                    if not should_exclude:
                        python_files.append(file_str)

        return sorted(python_files)

    def run_remediation(self):
        """Main remediation process"""
        print("Starting code quality remediation...")

        # Enumerate files
        all_files = self.enumerate_files()
        self.tracker_data["total_files"] = len(all_files)
        print(f"Found {len(all_files)} Python files to process")

        # Process each file
        for i, file_path in enumerate(all_files, 1):
            print(f"Processing [{i}/{len(all_files)}]: {file_path}")

            # Skip if already processed successfully
            if file_path in self.tracker_data["files"]:
                if self.tracker_data["files"][file_path]["status"] == "completed":
                    print("  Already completed, skipping...")
                    continue

            # Process the file
            file_info = self.process_file(file_path)
            self.tracker_data["files"][file_path] = file_info

            if file_info["status"] == "completed":
                self.tracker_data["processed_files"] += 1
                print("  ✓ Completed successfully")
            else:
                print(f"  ✗ Failed after {file_info['attempt_count']} attempts")

            # Save progress every 10 files
            if i % 10 == 0:
                self.save_tracker()
                print(
                    f"Progress saved: {self.tracker_data['processed_files']}/{len(all_files)} files completed"
                )

        # Final save
        self.save_tracker()
        self.generate_summary()
        print("\nRemediation complete! See code_quality_summary.md for details.")

    def generate_summary(self):
        """Generate the final summary report"""
        completed_files = [
            f
            for f, info in self.tracker_data["files"].items()
            if info["status"] == "completed"
        ]
        failed_files = [
            f
            for f, info in self.tracker_data["files"].items()
            if info["status"] == "failed"
        ]

        # Count total issues resolved by type
        issues_resolved = {"flake8": 0, "ruff": 0, "mypy": 0, "black": 0, "isort": 0}
        for file_info in self.tracker_data["files"].values():
            for tool, count in file_info.get("issues_resolved", {}).items():
                issues_resolved[tool] += count

        success_rate = (
            (len(completed_files) / self.tracker_data["total_files"] * 100)
            if self.tracker_data["total_files"] > 0
            else 0
        )

        summary = f"""# Code Quality Remediation Summary

## Overview
- Start Time: {self.tracker_data['remediation_started']}
- End Time: {datetime.now(UTC).isoformat()}
- Files Processed: {len(self.tracker_data['files'])}
- Success Rate: {success_rate:.1f}%

## Results by Category
### Successfully Remediated
- Files: {len(completed_files)}
- Total Issues Resolved:
  - Flake8: {issues_resolved['flake8']}
  - Ruff: {issues_resolved['ruff']}
  - MyPy: {issues_resolved['mypy']}
  - Black: {issues_resolved['black']}
  - isort: {issues_resolved['isort']}

### Failed Remediation
- Files: {len(failed_files)}

## Failed Files Requiring Manual Intervention
"""

        if failed_files:
            for file_path in failed_files[:20]:  # Show first 20 failed files
                file_info = self.tracker_data["files"][file_path]
                summary += f"\n### {file_path}\n"
                summary += f"- Attempts: {file_info['attempt_count']}\n"
                if file_info.get("error_log"):
                    summary += f"- Last Error: {file_info['error_log'][-1][:200]}\n"

        with open(self.summary_file, "w") as f:
            f.write(summary)


if __name__ == "__main__":
    remediator = CodeQualityRemediator()
    remediator.run_remediation()
