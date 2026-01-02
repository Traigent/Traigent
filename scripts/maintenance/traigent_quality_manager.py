#!/usr/bin/env python3
"""
Traigent Quality Manager - Unified Code Quality Tool
===================================================

A comprehensive, safe tool for code quality reporting and automated fixing.
Combines quality reporting with safe automated fixes and proper validation.

Safety Features:
- Dry-run mode for all operations
- Automatic backup creation before modifications
- Syntax validation before applying changes
- Rollback capability
- Interactive confirmation for destructive operations
- Comprehensive logging

Usage:
    python traigent_quality_manager.py --report                    # Generate quality report
    python traigent_quality_manager.py --fix --dry-run             # Preview fixes
    python traigent_quality_manager.py --fix                       # Apply fixes (interactive)
    python traigent_quality_manager.py --fix --auto-yes            # Apply fixes (non-interactive)
    python traigent_quality_manager.py --rollback <backup_id>      # Rollback changes
"""

import argparse
import ast
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from traigent.utils.secure_path import validate_path

class BackupManager:
    """Manages backups for safe rollback capability."""

    def __init__(self, base_path: Path):
        self.base_path = validate_path(base_path, Path.cwd(), must_exist=True)
        self.backup_dir = validate_path(
            self.base_path / "scripts" / "maintenance" / "backups",
            self.base_path,
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, files: List[Path]) -> str:
        """Create a backup of specified files."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_id = f"quality_fix_{timestamp}"
        backup_path = validate_path(self.backup_dir / backup_id, self.backup_dir)
        backup_path.mkdir(exist_ok=True)

        backup_manifest = {"backup_id": backup_id, "timestamp": timestamp, "files": []}

        for file_path in files:
            if file_path.exists():
                safe_path = validate_path(file_path, self.base_path, must_exist=True)
                relative_path = safe_path.relative_to(self.base_path)
                backup_file = validate_path(backup_path / relative_path, backup_path)
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(safe_path, backup_file)
                backup_manifest["files"].append(str(relative_path))

        # Save manifest
        manifest_path = validate_path(backup_path / "manifest.json", backup_path)
        with open(manifest_path, "w") as f:
            json.dump(backup_manifest, f, indent=2)

        return backup_id

    def restore_backup(self, backup_id: str) -> bool:
        """Restore files from backup."""
        backup_path = validate_path(self.backup_dir / backup_id, self.backup_dir)
        manifest_file = validate_path(backup_path / "manifest.json", backup_path)

        if not manifest_file.exists():
            return False

        with open(manifest_file) as f:
            manifest = json.load(f)

        for relative_path in manifest["files"]:
            backup_file = validate_path(backup_path / relative_path, backup_path)
            original_file = validate_path(self.base_path / relative_path, self.base_path)

            if backup_file.exists():
                original_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_file, original_file)

        return True

    def list_backups(self) -> List[Dict]:
        """List available backups."""
        backups = []
        for backup_dir in self.backup_dir.glob("quality_fix_*"):
            manifest_file = validate_path(
                backup_dir / "manifest.json",
                backup_dir,
            )
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    backups.append(manifest)
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)


class QualityChecker:
    """Enhanced quality checker with safety features."""

    def __init__(self, base_path: Path, dry_run: bool = False):
        self.base_path = base_path
        self.dry_run = dry_run
        self.python_exe = sys.executable
        self.backup_manager = BackupManager(base_path)

        # Setup logging
        log_dir = base_path / "scripts" / "maintenance" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    log_dir
                    / f"quality_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def validate_python_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax without executing."""
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def run_flake8(self) -> Dict:
        """Run flake8 and return results."""
        try:
            # Check if flake8 is available
            check_result = subprocess.run(
                [self.python_exe, "-c", "import flake8; print('available')"],
                capture_output=True,
                text=True,
                cwd=self.base_path,
            )

            if check_result.returncode != 0:
                return {"error": "flake8 not installed", "available": False}

            result = subprocess.run(
                [
                    self.python_exe,
                    "-m",
                    "flake8",
                    "--max-line-length=100",
                    "--extend-ignore=E203,W503",
                    "--exclude=__pycache__,venv,archive,scripts/archive",
                    "--format=json",
                    "traigent/",
                ],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=60,
            )

            if result.returncode in [0, 1]:  # 0 = no errors, 1 = found errors
                try:
                    if result.stdout.strip():
                        return {"issues": json.loads(result.stdout), "available": True}
                    return {"issues": {}, "available": True}
                except json.JSONDecodeError:
                    # Fallback to stderr or return empty
                    return {
                        "issues": {},
                        "available": True,
                        "raw_output": result.stdout,
                    }
            else:
                return {
                    "error": f"flake8 failed with code {result.returncode}",
                    "stderr": result.stderr,
                }

        except subprocess.TimeoutExpired:
            return {"error": "flake8 timed out"}
        except Exception as e:
            return {"error": f"Unexpected error running flake8: {e}"}

    def run_ruff(self) -> List[Dict]:
        """Run ruff and return results."""
        try:
            # Check if ruff is available
            check_result = subprocess.run(
                [self.python_exe, "-c", "import ruff; print('available')"],
                capture_output=True,
                text=True,
                cwd=self.base_path,
            )

            if check_result.returncode != 0:
                # Try direct ruff command
                try:
                    subprocess.run(
                        ["ruff", "--version"], capture_output=True, check=True
                    )
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return [{"error": "ruff not installed"}]

            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    "traigent/",
                    "--output-format=json",
                    "--exclude=__pycache__,venv,archive",
                ],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=60,
            )

            if result.returncode in [0, 1]:  # 0 = no errors, 1 = found errors
                if result.stdout.strip():
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        return []
                return []
            else:
                return [{"error": f"ruff failed with code {result.returncode}"}]

        except subprocess.TimeoutExpired:
            return [{"error": "ruff timed out"}]
        except Exception as e:
            return [{"error": f"Unexpected error running ruff: {e}"}]

    def run_mypy(self) -> str:
        """Run mypy and return results."""
        try:
            result = subprocess.run(
                [
                    self.python_exe,
                    "-m",
                    "mypy",
                    "traigent/",
                    "--ignore-missing-imports",
                    "--no-error-summary",
                    "--show-error-codes",
                ],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=120,
            )

            output = result.stdout + result.stderr
            if result.returncode not in [0, 1]:
                return f"mypy failed with code {result.returncode}\\n{output}"

            return output if output.strip() else "No mypy issues found"

        except subprocess.TimeoutExpired:
            return "mypy timed out"
        except FileNotFoundError:
            return "mypy not installed"
        except Exception as e:
            return f"Unexpected error running mypy: {e}"

    def check_model_name_typos(self) -> List[Dict]:
        """Check for model name typos (from original check_code_quality.py)."""
        issues = []
        files_to_check = [
            "traigent/api/decorators.py",
            "traigent/config/types.py",
            "traigent/cloud/models.py",
            "traigent/agents/platforms.py",
            "traigent/cli/main.py",
            "traigent/integrations/framework_override.py",
        ]

        for file_path_str in files_to_check:
            file_path = self.base_path / file_path_str
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    if "o4-mini" in content:
                        issues.append(
                            {
                                "type": "model_name_typo",
                                "file": file_path_str,
                                "issue": "Found 'o4-mini' typo",
                                "line": None,
                            }
                        )
                except Exception as e:
                    issues.append(
                        {
                            "type": "read_error",
                            "file": file_path_str,
                            "issue": f"Could not read file: {e}",
                            "line": None,
                        }
                    )
        return issues

    def check_debug_prints(self) -> List[Dict]:
        """Check for debug print statements."""
        issues = []
        excluded_dirs = [
            "__pycache__",
            "venv",
            "archive",
            "scripts/archive",
            "tests",
            "examples",
        ]

        for py_file in self.base_path.rglob("traigent/**/*.py"):
            if any(exc_dir in str(py_file) for exc_dir in excluded_dirs):
                continue

            try:
                lines = py_file.read_text().splitlines()
                for line_no, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if (
                        stripped.startswith("print(")
                        and not stripped.startswith('print(f"')
                        and "logger" not in line
                    ):
                        issues.append(
                            {
                                "type": "debug_print",
                                "file": str(py_file.relative_to(self.base_path)),
                                "line": line_no,
                                "issue": "Debug print statement",
                                "content": line.strip(),
                            }
                        )
            except Exception:
                continue

        return issues

    def generate_report(self) -> str:
        """Generate comprehensive quality report."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        reports_dir = self.base_path / "reports" / "code-quality"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"quality_report_{timestamp}.md"

        self.logger.info("🔍 Running comprehensive code quality analysis...")

        # Run all checks
        flake8_results = self.run_flake8()
        ruff_results = self.run_ruff()
        mypy_output = self.run_mypy()
        model_typos = self.check_model_name_typos()
        debug_prints = self.check_debug_prints()

        # Count issues
        flake8_available = flake8_results.get("available", False)
        ruff_available = not (
            isinstance(ruff_results, list)
            and ruff_results
            and "error" in ruff_results[0]
        )
        mypy_available = (
            "not installed" not in mypy_output
            and "mypy not installed" not in mypy_output
        )

        flake8_count = len(flake8_results.get("issues", {})) if flake8_available else 0
        ruff_count = len(ruff_results) if ruff_available else 0
        model_typo_count = len(model_typos)
        debug_print_count = len(debug_prints)

        # Generate report content
        report_content = f"""# Traigent Code Quality Report
Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Summary
- **Flake8 Issues**: {flake8_count} {'✅' if flake8_count == 0 else '⚠️'}
- **Ruff Issues**: {ruff_count} {'✅' if ruff_count == 0 else '⚠️'}
- **Model Name Typos**: {model_typo_count} {'✅' if model_typo_count == 0 else '❌'}
- **Debug Print Statements**: {debug_print_count} {'✅' if debug_print_count == 0 else '⚠️'}
- **MyPy**: {'✅ Available' if mypy_available else '❌ Not installed'}

## 🛠️ Tool Availability
- **Flake8**: {'✅ Available' if flake8_available else '❌ Not available'}
- **Ruff**: {'✅ Available' if ruff_available else '❌ Not available'}
- **MyPy**: {'✅ Available' if mypy_available else '❌ Not available'}

## 🔧 Quick Fix
To automatically fix safe issues:
```bash
python scripts/maintenance/traigent_quality_manager.py --fix --dry-run  # Preview fixes
python scripts/maintenance/traigent_quality_manager.py --fix             # Apply fixes
```

"""

        # Add detailed sections
        if flake8_count > 0:
            report_content += "## 🔍 Flake8 Issues\n"
            if isinstance(flake8_results.get("issues"), dict):
                for filename, file_issues in flake8_results["issues"].items():
                    report_content += f"### {filename}\\n"
                    for issue in file_issues:
                        report_content += f"- Line {issue.get('line', 'N/A')}: {issue.get('message', 'Unknown issue')}\\n"
            report_content += "\\n"

        if ruff_count > 0:
            report_content += "## 🦀 Ruff Issues\n"
            for issue in ruff_results[:10]:  # Limit to first 10
                if "error" not in issue:
                    report_content += f"- **{issue.get('filename', 'Unknown file')}** "
                    report_content += (
                        f"(Line {issue.get('location', {}).get('row', 'N/A')}): "
                    )
                    report_content += f"{issue.get('message', 'Unknown issue')}\\n"
            if ruff_count > 10:
                report_content += f"... and {ruff_count - 10} more issues\\n"
            report_content += "\\n"

        if model_typo_count > 0:
            report_content += "## ❌ Model Name Typos\n"
            for issue in model_typos:
                report_content += f"- **{issue['file']}**: {issue['issue']}\\n"
            report_content += "\\n"

        if debug_print_count > 0:
            report_content += "## 🐛 Debug Print Statements\n"
            for issue in debug_prints[:10]:  # Limit to first 10
                report_content += f"- **{issue['file']}** (Line {issue['line']}): `{issue['content']}`\\n"
            if debug_print_count > 10:
                report_content += f"... and {debug_print_count - 10} more issues\\n"
            report_content += "\\n"

        if (
            mypy_available
            and mypy_output.strip()
            and "No mypy issues found" not in mypy_output
        ):
            report_content += (
                "## 🐍 MyPy Analysis\n```\n" + mypy_output[:2000] + "\n```\n\\n"
            )

        report_content += """## 🚀 Next Steps
1. Review the issues above
2. Run with `--fix --dry-run` to preview automatic fixes
3. Apply safe automatic fixes with `--fix`
4. Address remaining issues manually
5. Run tests to verify fixes: `python -m pytest`

---
*Generated by Traigent Quality Manager*
"""

        # Write report
        with open(report_file, "w") as f:
            f.write(report_content)

        self.logger.info(f"📋 Quality report saved to: {report_file}")
        return str(report_file)

    def fix_model_typos(self, files_with_issues: List[str]) -> List[Tuple[str, str]]:
        """Fix model name typos safely."""
        fixes_applied = []

        for file_path_str in files_with_issues:
            file_path = self.base_path / file_path_str
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text()
                original_content = content

                # Fix the typo
                content = content.replace("o4-mini", "gpt-4o-mini")

                if content != original_content:
                    # Validate syntax
                    is_valid, error = self.validate_python_syntax(content)
                    if not is_valid:
                        self.logger.error(
                            f"Syntax validation failed for {file_path}: {error}"
                        )
                        continue

                    if not self.dry_run:
                        file_path.write_text(content)

                    fixes_applied.append(
                        (file_path_str, "Fixed o4-mini -> gpt-4o-mini")
                    )

            except Exception as e:
                self.logger.error(f"Error fixing {file_path}: {e}")

        return fixes_applied

    def apply_fixes(self, auto_confirm: bool = False) -> Dict:
        """Apply safe automated fixes."""
        self.logger.info("🔧 Starting safe automated fixes...")

        # Collect issues that can be fixed
        model_typos = self.check_model_name_typos()

        files_to_modify = []
        planned_fixes = []

        # Plan model typo fixes
        for issue in model_typos:
            if issue["type"] == "model_name_typo":
                files_to_modify.append(self.base_path / issue["file"])
                planned_fixes.append(f"Fix model typo in {issue['file']}")

        if not planned_fixes:
            self.logger.info("✅ No fixable issues found!")
            return {"fixes_applied": 0, "issues": "No fixable issues found"}

        # Show planned fixes
        print("\\n🔧 Planned Fixes:")
        for fix in planned_fixes:
            print(f"  • {fix}")

        if not auto_confirm and not self.dry_run:
            response = input("\\n❓ Apply these fixes? [y/N]: ").lower().strip()
            if response != "y":
                self.logger.info("❌ Fixes cancelled by user")
                return {"fixes_applied": 0, "issues": "Cancelled by user"}

        # Create backup
        if not self.dry_run and files_to_modify:
            backup_id = self.backup_manager.create_backup(files_to_modify)
            self.logger.info(f"💾 Backup created: {backup_id}")
        else:
            backup_id = "dry-run"

        # Apply fixes
        total_fixes = 0
        fix_results = []

        # Fix model typos
        typo_files = [
            issue["file"] for issue in model_typos if issue["type"] == "model_name_typo"
        ]
        if typo_files:
            fixes = self.fix_model_typos(typo_files)
            fix_results.extend(fixes)
            total_fixes += len(fixes)

        result = {
            "fixes_applied": total_fixes,
            "fix_details": fix_results,
            "backup_id": backup_id,
            "dry_run": self.dry_run,
        }

        if self.dry_run:
            self.logger.info(f"🔍 DRY RUN: Would apply {total_fixes} fixes")
        else:
            self.logger.info(f"✅ Applied {total_fixes} fixes successfully!")
            if backup_id != "dry-run":
                self.logger.info(
                    f"💡 To rollback: python traigent_quality_manager.py --rollback {backup_id}"
                )

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Traigent Quality Manager - Unified Code Quality Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--report", action="store_true", help="Generate quality report")
    parser.add_argument("--fix", action="store_true", help="Apply automated fixes")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    parser.add_argument(
        "--auto-yes", action="store_true", help="Auto-confirm all fixes"
    )
    parser.add_argument("--rollback", metavar="BACKUP_ID", help="Rollback to backup")
    parser.add_argument(
        "--list-backups", action="store_true", help="List available backups"
    )

    args = parser.parse_args()

    base_path = Path(__file__).parent.parent.parent  # Go up from scripts/maintenance/
    quality_manager = QualityChecker(base_path, dry_run=args.dry_run)

    if args.list_backups:
        backups = quality_manager.backup_manager.list_backups()
        if backups:
            print("\\n📦 Available Backups:")
            for backup in backups:
                print(f"  • {backup['backup_id']} ({backup['timestamp']})")
                print(f"    Files: {len(backup['files'])}")
        else:
            print("No backups found.")
        return

    if args.rollback:
        print(f"🔄 Rolling back to backup: {args.rollback}")
        success = quality_manager.backup_manager.restore_backup(args.rollback)
        if success:
            print("✅ Rollback completed successfully!")
        else:
            print("❌ Rollback failed - backup not found.")
        return

    if args.fix:
        result = quality_manager.apply_fixes(auto_confirm=args.auto_yes)
        print(f"\\n🎯 Result: {result['fixes_applied']} fixes applied")
        if result.get("fix_details"):
            for file_path, description in result["fix_details"]:
                print(f"  • {file_path}: {description}")
    elif args.report or not any([args.fix, args.rollback, args.list_backups]):
        # Default to generating report
        report_path = quality_manager.generate_report()
        print(f"\\n📋 Quality report generated: {report_path}")
        print("\\n💡 Next steps:")
        print("  1. Review the report")
        print("  2. Run with --fix --dry-run to preview fixes")
        print("  3. Run with --fix to apply safe automated fixes")


if __name__ == "__main__":
    main()
