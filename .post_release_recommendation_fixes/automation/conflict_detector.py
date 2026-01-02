#!/usr/bin/env python3
"""Conflict Detector for Post-Release Recommendation Fixes.

Detects potential conflicts between fixes that might be assigned in parallel.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from versioning import resolve_base_path, resolve_version


@dataclass
class FileScope:
    """File scope for a fix."""

    fix_id: str
    files: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)


@dataclass
class Conflict:
    """A detected conflict between two fixes."""

    fix_a: str
    fix_b: str
    conflict_type: str  # file, directory, module, dependency
    details: str
    severity: str = "warning"  # warning, error

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.fix_a} <-> {self.fix_b}: {self.conflict_type} - {self.details}"


class ConflictDetector:
    """Detect conflicts between fixes based on file scope and dependencies."""

    def __init__(
        self,
        tracking_path: str | Path = ".post_release_recommendation_fixes/TRACKING.md",
        version: str | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            tracking_path: Path to TRACKING.md
            version: Release version override
        """
        resolved_version = resolve_version(version)
        if tracking_path == ".post_release_recommendation_fixes/TRACKING.md":
            base = resolve_base_path(".post_release_recommendation_fixes", resolved_version)
            self.tracking_path = base / "TRACKING.md"
        else:
            self.tracking_path = Path(tracking_path)

    def get_fix_scopes(self) -> dict[str, FileScope]:
        """Extract file scopes for all fixes from TRACKING.md.

        Returns:
            Dictionary mapping fix ID to FileScope
        """
        if not self.tracking_path.exists():
            return {}

        content = self.tracking_path.read_text()
        scopes: dict[str, FileScope] = {}

        current_fix_id = None
        in_details = False

        for line in content.split("\n"):
            if "## Item Details" in line:
                in_details = True
                continue

            if not in_details:
                continue

            # Parse fix header
            match = re.match(r"^###\s+(\d+):", line.strip())
            if match:
                current_fix_id = match.group(1)
                scopes[current_fix_id] = FileScope(fix_id=current_fix_id)
                continue

            # Parse location field
            if current_fix_id and line.startswith("- **Location**:"):
                location = line.split(":", 1)[1].strip()
                scope = scopes[current_fix_id]

                # Extract file paths
                files = self._extract_files(location)
                scope.files.extend(files)

                # Extract directories
                dirs = self._extract_directories(files)
                scope.directories.extend(dirs)

                # Extract module names
                modules = self._extract_modules(files)
                scope.modules.extend(modules)

        return scopes

    def _extract_files(self, location: str) -> list[str]:
        """Extract file paths from location string.

        Args:
            location: Location string (may contain markdown links)

        Returns:
            List of file paths
        """
        files = []

        # Match markdown links [text](path)
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        for match in link_pattern.finditer(location):
            path = match.group(2)
            if self._is_supported_path(path):
                files.append(path)

        # Match bare file paths
        path_pattern = re.compile(
            r"[\w./-]+\.(py|md|yaml|yml|json|toml|ini|cfg|txt|sql|sh)"
        )
        for match in path_pattern.finditer(location):
            path = match.group(0)
            if path not in files:
                files.append(path)

        return files

    def _is_supported_path(self, path: str) -> bool:
        """Check if a path looks like a tracked file type."""
        return bool(
            re.search(r"\.(py|md|yaml|yml|json|toml|ini|cfg|txt|sql|sh)$", path)
        )

    def _extract_directories(self, files: list[str]) -> list[str]:
        """Extract directory paths from file list.

        Args:
            files: List of file paths

        Returns:
            List of unique directory paths
        """
        dirs = set()
        for f in files:
            # Get parent directory
            parts = f.rsplit("/", 1)
            if len(parts) > 1:
                dirs.add(parts[0])
        return list(dirs)

    def _extract_modules(self, files: list[str]) -> list[str]:
        """Extract Python module names from file paths.

        Args:
            files: List of file paths

        Returns:
            List of module names
        """
        modules = []
        for f in files:
            # Convert path to module name
            if f.endswith(".py"):
                module = f[:-3].replace("/", ".")
                modules.append(module)
        return modules

    def detect_conflicts(
        self,
        fix_ids: list[str] | None = None,
    ) -> list[Conflict]:
        """Detect conflicts between fixes.

        Args:
            fix_ids: Optional list of fix IDs to check (default: all)

        Returns:
            List of detected conflicts
        """
        scopes = self.get_fix_scopes()

        if fix_ids:
            scopes = {k: v for k, v in scopes.items() if k in fix_ids}

        conflicts = []
        fix_list = list(scopes.keys())

        # Compare each pair of fixes
        for i, fix_a in enumerate(fix_list):
            for fix_b in fix_list[i + 1:]:
                scope_a = scopes[fix_a]
                scope_b = scopes[fix_b]

                # Check file conflicts (exact same file)
                file_conflicts = set(scope_a.files) & set(scope_b.files)
                for f in file_conflicts:
                    conflicts.append(Conflict(
                        fix_a=fix_a,
                        fix_b=fix_b,
                        conflict_type="file",
                        details=f"Both modify: {f}",
                        severity="error",
                    ))

                # Check directory conflicts (same directory, different files)
                dir_conflicts = set(scope_a.directories) & set(scope_b.directories)
                if dir_conflicts and not file_conflicts:
                    for d in dir_conflicts:
                        conflicts.append(Conflict(
                            fix_a=fix_a,
                            fix_b=fix_b,
                            conflict_type="directory",
                            details=f"Both touch directory: {d}",
                            severity="warning",
                        ))

                # Check module import conflicts
                module_conflicts = self._check_module_dependencies(
                    scope_a.modules, scope_b.modules
                )
                for mc in module_conflicts:
                    conflicts.append(Conflict(
                        fix_a=fix_a,
                        fix_b=fix_b,
                        conflict_type="module",
                        details=mc,
                        severity="warning",
                    ))

        return conflicts

    def _check_module_dependencies(
        self,
        modules_a: list[str],
        modules_b: list[str],
    ) -> list[str]:
        """Check for module dependency conflicts.

        Args:
            modules_a: Modules from fix A
            modules_b: Modules from fix B

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # Check if modules are in the same package
        for mod_a in modules_a:
            pkg_a = mod_a.rsplit(".", 1)[0] if "." in mod_a else mod_a
            for mod_b in modules_b:
                pkg_b = mod_b.rsplit(".", 1)[0] if "." in mod_b else mod_b

                if pkg_a == pkg_b and mod_a != mod_b:
                    conflicts.append(
                        f"Same package ({pkg_a}): {mod_a.split('.')[-1]} and {mod_b.split('.')[-1]}"
                    )

        return conflicts

    def can_run_parallel(self, fix_a: str, fix_b: str) -> tuple[bool, list[Conflict]]:
        """Check if two fixes can safely run in parallel.

        Args:
            fix_a: First fix ID
            fix_b: Second fix ID

        Returns:
            Tuple of (can_parallel, list of conflicts)
        """
        conflicts = self.detect_conflicts([fix_a, fix_b])

        # Error-level conflicts prevent parallel execution
        errors = [c for c in conflicts if c.severity == "error"]

        return len(errors) == 0, conflicts

    def suggest_batches(self, fix_ids: list[str]) -> list[list[str]]:
        """Suggest parallel batches of non-conflicting fixes.

        Args:
            fix_ids: List of fix IDs to batch

        Returns:
            List of batches (each batch can run in parallel)
        """
        if not fix_ids:
            return []

        # Build conflict graph
        conflicts_map: dict[str, set[str]] = {f: set() for f in fix_ids}
        all_conflicts = self.detect_conflicts(fix_ids)

        for conflict in all_conflicts:
            if conflict.severity == "error":
                conflicts_map[conflict.fix_a].add(conflict.fix_b)
                conflicts_map[conflict.fix_b].add(conflict.fix_a)

        # Greedy batch assignment
        batches: list[list[str]] = []
        remaining = set(fix_ids)

        while remaining:
            batch: list[str] = []
            for fix_id in list(remaining):
                # Check if fix conflicts with any in current batch
                can_add = True
                for existing in batch:
                    if fix_id in conflicts_map.get(existing, set()):
                        can_add = False
                        break

                if can_add:
                    batch.append(fix_id)
                    remaining.remove(fix_id)

            if batch:
                batches.append(batch)
            else:
                # Shouldn't happen, but handle to prevent infinite loop
                batches.append([remaining.pop()])

        return batches

    def generate_report(self, fix_ids: list[str] | None = None) -> str:
        """Generate conflict detection report.

        Args:
            fix_ids: Optional list of fix IDs to analyze

        Returns:
            Markdown report
        """
        conflicts = self.detect_conflicts(fix_ids)
        scopes = self.get_fix_scopes()

        if fix_ids:
            scopes = {k: v for k, v in scopes.items() if k in fix_ids}

        lines = [
            "# Conflict Detection Report",
            "",
            f"**Fixes Analyzed**: {len(scopes)}",
            f"**Conflicts Found**: {len(conflicts)}",
            "",
        ]

        if not conflicts:
            lines.extend([
                "No conflicts detected. All fixes can run in parallel.",
                "",
            ])
        else:
            errors = [c for c in conflicts if c.severity == "error"]
            warnings = [c for c in conflicts if c.severity == "warning"]

            lines.extend([
                f"- **Errors**: {len(errors)} (blocking)",
                f"- **Warnings**: {len(warnings)} (caution)",
                "",
                "## Conflict Details",
                "",
            ])

            for conflict in conflicts:
                lines.append(f"- {conflict}")

            lines.extend([
                "",
                "## Suggested Batches",
                "",
            ])

            batches = self.suggest_batches(list(scopes.keys()))
            for i, batch in enumerate(batches, 1):
                lines.append(f"**Batch {i}**: {', '.join(batch)}")

        lines.extend([
            "",
            "## Fix Scopes",
            "",
            "| Fix ID | Files | Directories |",
            "|--------|-------|-------------|",
        ])

        for fix_id, scope in sorted(scopes.items()):
            files = ", ".join(scope.files[:3])
            if len(scope.files) > 3:
                files += f" (+{len(scope.files) - 3})"
            dirs = ", ".join(scope.directories[:2])
            lines.append(f"| {fix_id} | {files} | {dirs} |")

        return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: conflict_detector.py <command> [fix_ids...]")
        print("Commands:")
        print("  check [ids]   - Check for conflicts (all or specific fixes)")
        print("  batch [ids]   - Suggest parallel batches")
        print("  report        - Generate full report")
        print("  pair <a> <b>  - Check if two fixes can run in parallel")
        sys.exit(1)

    command = sys.argv[1]
    detector = ConflictDetector()

    if command == "check":
        fix_ids = sys.argv[2:] if len(sys.argv) > 2 else None
        conflicts = detector.detect_conflicts(fix_ids)

        if conflicts:
            print(f"Found {len(conflicts)} conflicts:")
            for conflict in conflicts:
                print(f"  {conflict}")
            sys.exit(1)
        else:
            print("No conflicts detected")

    elif command == "batch":
        fix_ids = sys.argv[2:] if len(sys.argv) > 2 else None
        if not fix_ids:
            scopes = detector.get_fix_scopes()
            fix_ids = list(scopes.keys())

        batches = detector.suggest_batches(fix_ids)
        print(f"Suggested {len(batches)} batches:")
        for i, batch in enumerate(batches, 1):
            print(f"  Batch {i}: {', '.join(batch)}")

    elif command == "report":
        print(detector.generate_report())

    elif command == "pair":
        if len(sys.argv) < 4:
            print("Usage: conflict_detector.py pair <fix_a> <fix_b>")
            sys.exit(1)

        fix_a, fix_b = sys.argv[2], sys.argv[3]
        can_parallel, conflicts = detector.can_run_parallel(fix_a, fix_b)

        if can_parallel:
            print(f"OK: {fix_a} and {fix_b} can run in parallel")
            if conflicts:
                print("  Warnings:")
                for c in conflicts:
                    print(f"    - {c.details}")
        else:
            print(f"BLOCKED: {fix_a} and {fix_b} have conflicts:")
            for c in conflicts:
                print(f"  - {c}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
