#!/usr/bin/env python3
"""
Safe Cleanup Manager - Consolidated Project Cleanup Tool
=======================================================

A comprehensive, safe tool for project cleanup operations.
Consolidates functionality from multiple cleanup scripts with enhanced safety.

Safety Features:
- Interactive confirmation for all destructive operations
- Dry-run mode to preview changes
- Automatic backup creation before deletion
- Detailed logging of all operations
- Undo capability
- Size and age thresholds for safety

Usage:
    python safe_cleanup_manager.py --analyze                   # Analyze what can be cleaned
    python safe_cleanup_manager.py --cleanup --dry-run         # Preview cleanup
    python safe_cleanup_manager.py --cleanup --interactive     # Interactive cleanup
    python safe_cleanup_manager.py --cleanup --auto            # Automated cleanup
    python safe_cleanup_manager.py --undo <backup_id>          # Undo cleanup
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from traigent.utils.secure_path import safe_open, validate_path

class CleanupBackupManager:
    """Manages backups for cleanup operations with undo capability."""

    def __init__(self, base_path: Path):
        self.base_path = validate_path(base_path, Path.cwd(), must_exist=True)
        self.backup_dir = validate_path(
            self.base_path / "scripts" / "maintenance" / "cleanup_backups",
            self.base_path,
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_cleanup_backup(
        self, items_to_remove: List[Path], description: str
    ) -> str:
        """Create a backup before cleanup operations."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_id = f"cleanup_{timestamp}"
        backup_path = validate_path(self.backup_dir / backup_id, self.backup_dir)
        backup_path.mkdir(exist_ok=True)

        backup_manifest = {
            "backup_id": backup_id,
            "timestamp": timestamp,
            "description": description,
            "items": [],
        }

        total_size = 0
        for item_path in items_to_remove:
            if not item_path.exists():
                continue

                relative_path = str(item_path.relative_to(self.base_path))
                backup_item_path = validate_path(
                    backup_path / relative_path,
                    backup_path,
                )

            try:
                if item_path.is_file():
                    backup_item_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item_path, backup_item_path)
                    size = item_path.stat().st_size
                elif item_path.is_dir():
                    shutil.copytree(item_path, backup_item_path, dirs_exist_ok=True)
                    size = sum(
                        f.stat().st_size for f in item_path.rglob("*") if f.is_file()
                    )
                else:
                    continue

                backup_manifest["items"].append(
                    {
                        "path": relative_path,
                        "type": "file" if item_path.is_file() else "directory",
                        "size": size,
                    }
                )
                total_size += size

            except Exception as e:
                logging.warning(f"Could not backup {item_path}: {e}")

        backup_manifest["total_size"] = total_size

        # Save manifest
        manifest_path = validate_path(backup_path / "manifest.json", backup_path)
        with safe_open(manifest_path, backup_path, mode="w", encoding="utf-8") as f:
            json.dump(backup_manifest, f, indent=2)

        return backup_id

    def restore_cleanup_backup(self, backup_id: str) -> bool:
        """Restore items from cleanup backup."""
        backup_path = validate_path(self.backup_dir / backup_id, self.backup_dir)
        manifest_file = validate_path(backup_path / "manifest.json", backup_path)

        if not manifest_file.exists():
            return False

        with safe_open(manifest_file, backup_path, mode="r", encoding="utf-8") as f:
            manifest = json.load(f)

        restored_count = 0
        for item_info in manifest["items"]:
            backup_item = validate_path(
                backup_path / item_info["path"],
                backup_path,
            )
            original_item = validate_path(
                self.base_path / item_info["path"],
                self.base_path,
            )

            if not backup_item.exists():
                continue

            try:
                if item_info["type"] == "file":
                    original_item.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_item, original_item)
                else:  # directory
                    if original_item.exists():
                        shutil.rmtree(original_item)
                    shutil.copytree(backup_item, original_item)
                restored_count += 1
            except Exception as e:
                logging.error(f"Could not restore {item_info['path']}: {e}")

        logging.info(f"Restored {restored_count} items from backup {backup_id}")
        return restored_count > 0

    def list_cleanup_backups(self) -> List[Dict]:
        """List available cleanup backups."""
        backups = []
        for backup_dir in self.backup_dir.glob("cleanup_*"):
            manifest_file = validate_path(
                backup_dir / "manifest.json",
                backup_dir,
            )
            if manifest_file.exists():
                with safe_open(
                    manifest_file, backup_dir, mode="r", encoding="utf-8"
                ) as f:
                    manifest = json.load(f)
                    backups.append(manifest)
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)


class SafeCleanupAnalyzer:
    """Analyzes project for cleanup opportunities with safety checks."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.safe_extensions = {
            ".pyc",
            ".pyo",
            ".pyd",
            "__pycache__",
            ".DS_Store",
            "Thumbs.db",
            ".tmp",
            ".temp",
            ".cache",
            ".log",
            ".pid",
            ".lock",
            ".coverage",
            "~",
            ".bak",
            ".old",
            ".orig",
            ".rej",
        }
        self.safe_directories = {
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
            ".vscode",
            ".idea",
            "dist",
            "build",
            "*.egg-info",
        }
        # Minimum age before considering files for cleanup (to avoid cleaning recent work)
        self.min_age_days = 7

    def analyze_for_cleanup(self) -> Dict:
        """Analyze project for safe cleanup opportunities."""
        analysis = {
            "safe_to_remove": {"files": [], "directories": [], "total_size": 0},
            "review_manually": {"files": [], "directories": [], "total_size": 0},
            "old_reports": {"files": [], "total_size": 0},
            "duplicate_files": {"groups": [], "total_size": 0},
        }

        # Find safe-to-remove items
        self._find_safe_removals(analysis)

        # Find old reports and logs
        self._find_old_reports(analysis)

        # Find potential duplicates
        self._find_duplicates(analysis)

        # Find items that need manual review
        self._find_manual_review_items(analysis)

        return analysis

    def _find_safe_removals(self, analysis: Dict) -> None:
        """Find files and directories that are safe to remove."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.min_age_days)

        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)

            # Skip certain directories entirely
            if any(
                skip in str(root_path) for skip in [".git", "venv", ".venv", "archive"]
            ):
                dirs.clear()  # Don't recurse into these directories
                continue

            # Check directories for safe removal
            dirs_to_remove = []
            for dirname in dirs[:]:  # Create a copy to safely modify during iteration
                dir_path = root_path / dirname

                if dirname in self.safe_directories:
                    try:
                        # Check if directory is old enough
                        dir_mtime = datetime.fromtimestamp(
                            dir_path.stat().st_mtime, tz=timezone.utc
                        )
                        if dir_mtime < cutoff_date:
                            size = sum(
                                f.stat().st_size
                                for f in dir_path.rglob("*")
                                if f.is_file()
                            )
                            analysis["safe_to_remove"]["directories"].append(
                                {
                                    "path": str(dir_path.relative_to(self.base_path)),
                                    "size": size,
                                    "modified": dir_mtime.isoformat(),
                                    "reason": f"Safe directory type: {dirname}",
                                }
                            )
                            analysis["safe_to_remove"]["total_size"] += size
                            dirs_to_remove.append(dirname)
                    except Exception:
                        continue

            # Remove directories we've marked for cleanup from further traversal
            for dirname in dirs_to_remove:
                if dirname in dirs:
                    dirs.remove(dirname)

            # Check files for safe removal
            for filename in files:
                file_path = root_path / filename

                try:
                    file_stat = file_path.stat()
                    file_mtime = datetime.fromtimestamp(
                        file_stat.st_mtime, tz=timezone.utc
                    )

                    # Check if file is old enough
                    if file_mtime < cutoff_date:
                        # Check if file extension is safe to remove
                        if any(
                            filename.endswith(ext) for ext in self.safe_extensions
                        ) or any(
                            pattern in filename for pattern in self.safe_extensions
                        ):

                            analysis["safe_to_remove"]["files"].append(
                                {
                                    "path": str(file_path.relative_to(self.base_path)),
                                    "size": file_stat.st_size,
                                    "modified": file_mtime.isoformat(),
                                    "reason": "Safe file type",
                                }
                            )
                            analysis["safe_to_remove"][
                                "total_size"
                            ] += file_stat.st_size

                except Exception:
                    continue

    def _find_old_reports(self, analysis: Dict) -> None:
        """Find old report and log files."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(
            days=30
        )  # Reports older than 30 days

        report_dirs = [
            self.base_path / "reports",
            self.base_path / "logs",
            self.base_path / "scripts" / "maintenance" / "logs",
        ]

        for report_dir in report_dirs:
            if not report_dir.exists():
                continue

            for report_file in report_dir.rglob("*"):
                if not report_file.is_file():
                    continue

                try:
                    file_stat = report_file.stat()
                    file_mtime = datetime.fromtimestamp(
                        file_stat.st_mtime, tz=timezone.utc
                    )

                    if file_mtime < cutoff_date:
                        analysis["old_reports"]["files"].append(
                            {
                                "path": str(report_file.relative_to(self.base_path)),
                                "size": file_stat.st_size,
                                "modified": file_mtime.isoformat(),
                                "reason": "Report/log older than 30 days",
                            }
                        )
                        analysis["old_reports"]["total_size"] += file_stat.st_size

                except Exception:
                    continue

    def _find_duplicates(self, analysis: Dict) -> None:
        """Find potential duplicate files (simplified detection)."""
        file_hashes = {}

        # Only check certain file types for duplicates
        check_extensions = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}

        for file_path in self.base_path.rglob("*"):
            if (
                not file_path.is_file()
                or file_path.suffix not in check_extensions
                or any(
                    skip in str(file_path)
                    for skip in [".git", "venv", ".venv", "archive", "__pycache__"]
                )
            ):
                continue

            try:
                # Calculate file hash using SHA-256 for collision resistance
                with safe_open(file_path, self.base_path, mode="rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                file_size = file_path.stat().st_size

                if file_hash not in file_hashes:
                    file_hashes[file_hash] = []

                file_hashes[file_hash].append(
                    {
                        "path": str(file_path.relative_to(self.base_path)),
                        "size": file_size,
                    }
                )

            except Exception:
                continue

        # Find groups with more than one file (potential duplicates)
        for file_hash, files in file_hashes.items():
            if len(files) > 1:
                group_size = sum(file_info["size"] for file_info in files)
                analysis["duplicate_files"]["groups"].append(
                    {
                        "hash": file_hash,
                        "files": files,
                        "count": len(files),
                        "total_size": group_size,
                    }
                )
                analysis["duplicate_files"]["total_size"] += group_size

    def _find_manual_review_items(self, analysis: Dict) -> None:
        """Find items that need manual review before cleanup."""
        # Look for files that might be important but haven't been accessed recently
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)

        suspicious_files = []

        for file_path in self.base_path.rglob("*"):
            if not file_path.is_file() or any(
                skip in str(file_path) for skip in [".git", "venv", ".venv", "archive"]
            ):
                continue

            try:
                file_stat = file_path.stat()
                file_mtime = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)

                # Check for files that haven't been modified in 90 days
                # but might be important (config files, scripts, etc.)
                if (
                    file_mtime < cutoff_date
                    and file_path.suffix
                    in {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
                    and file_stat.st_size > 1024
                ):  # At least 1KB

                    suspicious_files.append(
                        {
                            "path": str(file_path.relative_to(self.base_path)),
                            "size": file_stat.st_size,
                            "modified": file_mtime.isoformat(),
                            "reason": "Old but potentially important file",
                        }
                    )

            except Exception:
                continue

        analysis["review_manually"]["files"] = suspicious_files
        analysis["review_manually"]["total_size"] = sum(
            f["size"] for f in suspicious_files
        )


class SafeCleanupManager:
    """Main cleanup manager with safety features."""

    def __init__(self, base_path: Path, dry_run: bool = False):
        self.base_path = base_path
        self.dry_run = dry_run
        self.backup_manager = CleanupBackupManager(base_path)
        self.analyzer = SafeCleanupAnalyzer(base_path)

        # Setup logging
        log_dir = base_path / "scripts" / "maintenance" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    log_dir
                    / f"cleanup_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def analyze_cleanup_opportunities(self) -> Dict:
        """Analyze and report cleanup opportunities."""
        self.logger.info("🔍 Analyzing cleanup opportunities...")
        return self.analyzer.analyze_for_cleanup()

    def perform_cleanup(
        self, interactive: bool = True, auto_confirm: bool = False
    ) -> Dict:
        """Perform cleanup operations with safety checks."""
        analysis = self.analyze_cleanup_opportunities()

        if not any(
            [
                analysis["safe_to_remove"]["files"],
                analysis["safe_to_remove"]["directories"],
                analysis["old_reports"]["files"],
            ]
        ):
            self.logger.info("✅ No cleanup needed - project is already clean!")
            return {"items_removed": 0, "size_freed": 0}

        # Show what will be cleaned
        total_items = (
            len(analysis["safe_to_remove"]["files"])
            + len(analysis["safe_to_remove"]["directories"])
            + len(analysis["old_reports"]["files"])
        )

        total_size = (
            analysis["safe_to_remove"]["total_size"]
            + analysis["old_reports"]["total_size"]
        )

        print("\\n🧹 Cleanup Plan:")
        print(f"  📁 Files to remove: {len(analysis['safe_to_remove']['files'])}")
        print(
            f"  📂 Directories to remove: {len(analysis['safe_to_remove']['directories'])}"
        )
        print(f"  📋 Old reports to remove: {len(analysis['old_reports']['files'])}")
        print(f"  💾 Total size to free: {self.format_size(total_size)}")

        if self.dry_run:
            print("\\n🔍 DRY RUN - No files will be removed")
            return {
                "items_removed": total_items,
                "size_freed": total_size,
                "dry_run": True,
            }

        # Get confirmation
        if interactive and not auto_confirm:
            response = input("\\n❓ Proceed with cleanup? [y/N]: ").lower().strip()
            if response != "y":
                self.logger.info("❌ Cleanup cancelled by user")
                return {"items_removed": 0, "size_freed": 0, "cancelled": True}

        # Create backup
        items_to_remove = []

        # Collect all items to remove
        for item_info in analysis["safe_to_remove"]["files"]:
            items_to_remove.append(self.base_path / item_info["path"])

        for item_info in analysis["safe_to_remove"]["directories"]:
            items_to_remove.append(self.base_path / item_info["path"])

        for item_info in analysis["old_reports"]["files"]:
            items_to_remove.append(self.base_path / item_info["path"])

        if items_to_remove:
            backup_id = self.backup_manager.create_cleanup_backup(
                items_to_remove,
                f"Cleanup of {total_items} items freeing {self.format_size(total_size)}",
            )
            self.logger.info(f"💾 Backup created: {backup_id}")

        # Perform cleanup
        items_removed = 0
        size_freed = 0

        for item_path in items_to_remove:
            if not item_path.exists():
                continue

            try:
                if item_path.is_file():
                    size_freed += item_path.stat().st_size
                    item_path.unlink()
                elif item_path.is_dir():
                    size_freed += sum(
                        f.stat().st_size for f in item_path.rglob("*") if f.is_file()
                    )
                    shutil.rmtree(item_path)

                items_removed += 1

            except Exception as e:
                self.logger.error(f"Could not remove {item_path}: {e}")

        self.logger.info(
            f"✅ Cleanup completed: {items_removed} items removed, {self.format_size(size_freed)} freed"
        )
        if items_to_remove:
            self.logger.info(
                f"💡 To undo: python safe_cleanup_manager.py --undo {backup_id}"
            )

        return {
            "items_removed": items_removed,
            "size_freed": size_freed,
            "backup_id": backup_id if items_to_remove else None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Safe Cleanup Manager - Consolidated Project Cleanup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--analyze", action="store_true", help="Analyze cleanup opportunities"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Perform cleanup operations"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview cleanup without removing files"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive cleanup with confirmations",
    )
    parser.add_argument(
        "--auto", action="store_true", help="Automated cleanup without confirmations"
    )
    parser.add_argument("--undo", metavar="BACKUP_ID", help="Undo cleanup using backup")
    parser.add_argument(
        "--list-backups", action="store_true", help="List available cleanup backups"
    )

    args = parser.parse_args()

    base_path = Path(__file__).parent.parent.parent
    cleanup_manager = SafeCleanupManager(base_path, dry_run=args.dry_run)

    if args.list_backups:
        backups = cleanup_manager.backup_manager.list_cleanup_backups()
        if backups:
            print("\\n📦 Available Cleanup Backups:")
            for backup in backups:
                size_str = cleanup_manager.format_size(backup["total_size"])
                print(f"  • {backup['backup_id']} ({backup['timestamp']})")
                print(f"    Description: {backup['description']}")
                print(f"    Items: {len(backup['items'])}, Size: {size_str}")
        else:
            print("No cleanup backups found.")
        return

    if args.undo:
        print(f"🔄 Undoing cleanup: {args.undo}")
        success = cleanup_manager.backup_manager.restore_cleanup_backup(args.undo)
        if success:
            print("✅ Undo completed successfully!")
        else:
            print("❌ Undo failed - backup not found.")
        return

    if args.cleanup:
        result = cleanup_manager.perform_cleanup(
            interactive=args.interactive or not args.auto, auto_confirm=args.auto
        )

        if not result.get("cancelled"):
            print("\\n🎯 Cleanup Result:")
            print(f"  Items removed: {result['items_removed']}")
            print(f"  Size freed: {cleanup_manager.format_size(result['size_freed'])}")
            if result.get("backup_id"):
                print(f"  Backup ID: {result['backup_id']}")

    elif args.analyze or not any([args.cleanup, args.undo, args.list_backups]):
        # Default to analysis
        analysis = cleanup_manager.analyze_cleanup_opportunities()

        print("\\n🔍 Cleanup Analysis Report:")
        print("\\n🗑️  Safe to Remove:")
        print(f"  📁 Files: {len(analysis['safe_to_remove']['files'])}")
        print(f"  📂 Directories: {len(analysis['safe_to_remove']['directories'])}")
        print(
            f"  💾 Size: {cleanup_manager.format_size(analysis['safe_to_remove']['total_size'])}"
        )

        print("\\n📋 Old Reports/Logs:")
        print(f"  📄 Files: {len(analysis['old_reports']['files'])}")
        print(
            f"  💾 Size: {cleanup_manager.format_size(analysis['old_reports']['total_size'])}"
        )

        if analysis["duplicate_files"]["groups"]:
            print("\\n📎 Potential Duplicates:")
            print(f"  🔗 Groups: {len(analysis['duplicate_files']['groups'])}")
            print(
                f"  💾 Size: {cleanup_manager.format_size(analysis['duplicate_files']['total_size'])}"
            )

        if analysis["review_manually"]["files"]:
            print("\\n⚠️  Manual Review Needed:")
            print(f"  📄 Files: {len(analysis['review_manually']['files'])}")
            print(
                f"  💾 Size: {cleanup_manager.format_size(analysis['review_manually']['total_size'])}"
            )

        total_cleanable = (
            analysis["safe_to_remove"]["total_size"]
            + analysis["old_reports"]["total_size"]
        )

        print(f"\\n💡 Total cleanable: {cleanup_manager.format_size(total_cleanable)}")
        print("\\n🚀 Next steps:")
        print("  1. Run with --cleanup --dry-run to preview cleanup")
        print("  2. Run with --cleanup --interactive for safe cleanup")


if __name__ == "__main__":
    main()
