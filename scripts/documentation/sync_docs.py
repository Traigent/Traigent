#!/usr/bin/env python3
"""
Documentation Synchronization Script for TraiGent SDK

Automatically synchronizes documentation across multiple files to ensure consistency.
Updates version numbers, feature lists, and API references.

Usage:
    python scripts/sync_docs.py                    # Sync all documentation
    python scripts/sync_docs.py --version 1.0.0    # Update version everywhere
    python scripts/sync_docs.py --dry-run          # Preview changes without applying
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DocumentationSynchronizer:
    """Synchronizes documentation across the project."""

    def __init__(self, root_path: Path = PROJECT_ROOT, dry_run: bool = False):
        self.root_path = root_path
        self.dry_run = dry_run
        self.changes_made = []

    def sync_all(self) -> bool:
        """Run all synchronization tasks."""
        print("🔄 Starting documentation synchronization...\n")

        tasks = [
            ("Version Numbers", self.sync_versions),
            ("Feature Lists", self.sync_features),
            ("API References", self.sync_api_references),
            ("Status Information", self.sync_status),
            ("Examples", self.sync_examples),
        ]

        success = True
        for name, task_func in tasks:
            print(f"📋 Syncing {name}...")
            try:
                task_func()
                print("   ✅ Completed\n")
            except Exception as e:
                print(f"   ❌ Failed: {e}\n")
                success = False

        self._print_summary()
        return success

    def sync_versions(self, new_version: Optional[str] = None) -> None:
        """Synchronize version numbers across all files."""
        # Get current version from pyproject.toml or setup.py
        current_version = self._get_current_version()
        target_version = new_version or current_version

        if not target_version:
            print("   ⚠️  No version found to synchronize")
            return

        print(f"   Target version: {target_version}")

        # Files to update
        version_files = [
            (
                self.root_path / "pyproject.toml",
                r'version\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)',
            ),
            (
                self.root_path / "setup.py",
                r'version\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)',
            ),
            (
                self.root_path / "traigent" / "__init__.py",
                r'__version__\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)',
            ),
            (
                self.root_path / "docs" / "DOCUMENTATION_MANIFEST.yaml",
                r"version:\s*([0-9]+\.[0-9]+\.[0-9]+)",
            ),
        ]

        for file_path, pattern in version_files:
            if file_path.exists():
                self._update_file_version(file_path, pattern, target_version)

    def _get_current_version(self) -> Optional[str]:
        """Get the current version from project files."""
        # Try pyproject.toml first
        pyproject_path = self.root_path / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            match = re.search(r'version\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)', content)
            if match:
                return match.group(1)

        # Try setup.py
        setup_path = self.root_path / "setup.py"
        if setup_path.exists():
            content = setup_path.read_text()
            match = re.search(r'version\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)', content)
            if match:
                return match.group(1)

        return None

    def _update_file_version(
        self, file_path: Path, pattern: str, new_version: str
    ) -> None:
        """Update version in a specific file."""
        content = file_path.read_text()

        def replacer(match):
            return match.group(0).replace(match.group(1), new_version)

        new_content = re.sub(pattern, replacer, content)

        if new_content != content:
            if not self.dry_run:
                file_path.write_text(new_content)
            self.changes_made.append(
                f"Updated version in {file_path.name} to {new_version}"
            )

    def sync_features(self) -> None:
        """Synchronize feature lists between README and CURRENT_STATUS."""
        readme_path = self.root_path / "README.md"
        status_path = self.root_path / "docs" / "CURRENT_STATUS.md"

        if not (readme_path.exists() and status_path.exists()):
            return

        # Extract completed features from CURRENT_STATUS.md
        status_content = status_path.read_text()
        completed_features = re.findall(
            r"✅ Complete[d]?\s*\|\s*`([^`]+)`", status_content
        )

        if completed_features:
            print(f"   Found {len(completed_features)} completed features")
            # This would update README feature list based on CURRENT_STATUS
            # For now, we just log what would be synced
            if self.dry_run:
                print(f"   Would sync {len(completed_features)} features to README")

    def sync_api_references(self) -> None:
        """Synchronize API references from docstrings to documentation."""
        # Extract API signatures from code
        api_signatures = self._extract_api_signatures()

        # Update API documentation files
        api_doc_path = self.root_path / "docs" / "generated" / "API_REFERENCE.md"
        if not api_doc_path.parent.exists():
            api_doc_path.parent.mkdir(parents=True, exist_ok=True)

        if api_signatures and not self.dry_run:
            # Generate basic API reference
            self._generate_api_reference(api_signatures, api_doc_path)
            self.changes_made.append(
                f"Updated API reference with {len(api_signatures)} entries"
            )

    def _extract_api_signatures(self) -> Dict[str, str]:
        """Extract API signatures from Python files."""
        import ast

        signatures = {}
        traigent_path = self.root_path / "traigent"

        if not traigent_path.exists():
            return signatures

        for py_file in traigent_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith("_"):
                            module_path = str(py_file.relative_to(self.root_path))[
                                :-3
                            ].replace("/", ".")
                            key = f"{module_path}.{node.name}"
                            signatures[key] = (
                                ast.get_docstring(node) or "No documentation"
                            )
            except:
                pass

        return signatures

    def _generate_api_reference(
        self, signatures: Dict[str, str], output_path: Path
    ) -> None:
        """Generate API reference document."""
        lines = [
            "# API Reference",
            "",
            "*Auto-generated from source code*",
            "",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Group by module
        modules = {}
        for key, doc in signatures.items():
            module = ".".join(key.split(".")[:-1])
            if module not in modules:
                modules[module] = []
            modules[module].append((key.split(".")[-1], doc))

        for module in sorted(modules.keys()):
            lines.append(f"## {module}")
            lines.append("")

            for func_name, doc in sorted(modules[module]):
                lines.append(f"### {func_name}")
                lines.append("")
                if doc and len(doc) > 10:
                    # Take first paragraph of docstring
                    first_para = doc.split("\n\n")[0]
                    lines.append(first_para)
                else:
                    lines.append("*No documentation available*")
                lines.append("")

        if not self.dry_run:
            output_path.write_text("\n".join(lines))

    def sync_status(self) -> None:
        """Synchronize status information across documentation."""
        # This would sync completion status, metrics, etc.
        # For now, just check consistency
        readme_path = self.root_path / "README.md"
        status_path = self.root_path / "docs" / "CURRENT_STATUS.md"

        if readme_path.exists() and status_path.exists():
            readme_content = readme_path.read_text()
            status_content = status_path.read_text()

            # Check if README mentions features that CURRENT_STATUS says are incomplete
            if "Coming Soon" in readme_content and "✅ Complete" in status_content:
                print("   ⚠️  Potential inconsistency between README and CURRENT_STATUS")

    def sync_examples(self) -> None:
        """Ensure examples are consistent across documentation."""
        # Check that examples in different docs don't contradict
        example_patterns = []

        for md_file in self.root_path.rglob("*.md"):
            if any(part in str(md_file) for part in ["venv", "env", "node_modules"]):
                continue

            content = md_file.read_text()
            python_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

            for block in python_blocks:
                # Look for common patterns
                if "@traigent.optimize" in block:
                    example_patterns.append((md_file.name, block))

        if len(example_patterns) > 1:
            print(
                f"   Found {len(example_patterns)} optimization examples to check for consistency"
            )

    def _print_summary(self) -> None:
        """Print synchronization summary."""
        print("\n" + "=" * 60)
        print("📊 SYNCHRONIZATION SUMMARY")
        print("=" * 60)

        if self.dry_run:
            print("\n🔍 DRY RUN - No changes were made")

        if self.changes_made:
            print(f"\n✅ Changes made ({len(self.changes_made)}):")
            for change in self.changes_made:
                print(f"   • {change}")
        else:
            print("\n✅ All documentation is already synchronized")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Synchronize TraiGent documentation")
    parser.add_argument("--version", help="Update version number everywhere")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    parser.add_argument(
        "--features", action="store_true", help="Sync feature lists only"
    )
    parser.add_argument("--api", action="store_true", help="Sync API references only")
    parser.add_argument(
        "--status", action="store_true", help="Sync status information only"
    )
    parser.add_argument("--examples", action="store_true", help="Sync examples only")

    args = parser.parse_args()

    synchronizer = DocumentationSynchronizer(dry_run=args.dry_run)

    # Run specific sync or all
    if args.version:
        synchronizer.sync_versions(args.version)
    elif args.features:
        synchronizer.sync_features()
    elif args.api:
        synchronizer.sync_api_references()
    elif args.status:
        synchronizer.sync_status()
    elif args.examples:
        synchronizer.sync_examples()
    else:
        synchronizer.sync_all()

    sys.exit(0)


if __name__ == "__main__":
    main()
