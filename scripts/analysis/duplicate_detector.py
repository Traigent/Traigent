#!/usr/bin/env python3
"""
Code duplication detection utility for Traigent SDK.

This script helps identify common patterns of code duplication
and suggests refactoring opportunities.
"""

import argparse
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from traigent.utils.secure_path import PathTraversalError, safe_read_text, validate_path


class DuplicationDetector:
    """Detects code duplication patterns in Python files."""

    def __init__(self, min_lines: int = 5):
        """
        Initialize detector.

        Args:
            min_lines: Minimum lines to consider for duplication
        """
        self.min_lines = min_lines
        self.code_blocks: Dict[str, List[Tuple[Path, int]]] = {}
        self._base_dir: Path | None = None

    def analyze_directory(self, directory: Path) -> Dict[str, List[Tuple[Path, int]]]:
        """
        Analyze directory for code duplication.

        Args:
            directory: Directory to analyze

        Returns:
            Dictionary mapping code hashes to file locations
        """
        self._base_dir = directory.resolve()
        python_files = list(directory.rglob("*.py"))

        for file_path in python_files:
            # Skip test files and generated files
            if any(
                part in str(file_path)
                for part in ["test_", "__pycache__", ".venv", "venv"]
            ):
                continue

            self._analyze_file(file_path)

        # Filter to only duplicated code
        duplicates = {
            code_hash: locations
            for code_hash, locations in self.code_blocks.items()
            if len(locations) > 1
        }

        return duplicates

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for code blocks."""
        try:
            base_dir = self._base_dir or file_path.parent.resolve()
            validated_path = validate_path(file_path, base_dir, must_exist=True)
            content = safe_read_text(validated_path, base_dir)
            lines = content.splitlines(keepends=True)
        except (OSError, UnicodeDecodeError):
            return

        # Extract function and class definitions
        try:
            tree = ast.parse("".join(lines))
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    self._extract_code_block(file_path, lines, node)
        except SyntaxError:
            pass

    def _extract_code_block(
        self, file_path: Path, lines: List[str], node: ast.AST
    ) -> None:
        """Extract and hash a code block."""
        start_line = node.lineno - 1
        end_line = getattr(node, "end_lineno", start_line + self.min_lines) - 1

        if end_line - start_line < self.min_lines:
            return

        # Extract code block
        code_block = "".join(lines[start_line : end_line + 1])

        # Normalize the code (remove whitespace variations)
        normalized_code = self._normalize_code(code_block)

        # Create hash
        code_hash = hashlib.sha256(normalized_code.encode()).hexdigest()[:12]

        if code_hash not in self.code_blocks:
            self.code_blocks[code_hash] = []

        self.code_blocks[code_hash].append((file_path, start_line + 1))

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove comments and normalize whitespace
        lines = []
        for line in code.split("\n"):
            # Strip comments but preserve structure
            if "#" in line:
                line = line.split("#")[0].rstrip()
            # Normalize whitespace but preserve indentation structure
            if line.strip():
                lines.append(line.rstrip())

        return "\n".join(lines)


def generate_report(
    duplicates: Dict[str, List[Tuple[Path, int]]],
    output_file: str | Path | None = None,
) -> str:
    """Generate duplication report."""
    report_lines = [
        "# Code Duplication Analysis Report",
        "",
        f"Found {len(duplicates)} instances of duplicated code blocks.",
        "",
    ]

    for i, (code_hash, locations) in enumerate(duplicates.items(), 1):
        report_lines.extend(
            [
                f"## Duplicate Block #{i} (Hash: {code_hash})",
                f"Found in {len(locations)} locations:",
                "",
            ]
        )

        for file_path, line_num in locations:
            report_lines.append(f"- `{file_path}:{line_num}`")

        report_lines.append("")

        # Add suggestion for refactoring
        if len(locations) > 2:
            report_lines.extend(
                [
                    "**Refactoring Suggestion:**",
                    "- Extract common functionality into a shared utility function",
                    "- Consider using inheritance or composition patterns",
                    "- Look for opportunities to create reusable components",
                    "",
                ]
            )

    # Add summary and recommendations
    report_lines.extend(
        [
            "## Summary",
            "",
            f"**Total Duplicate Blocks**: {len(duplicates)}",
            f"**Files Affected**: {len({loc[0] for locs in duplicates.values() for loc in locs})}",
            "",
            "## Recommendations",
            "",
            "1. **High Priority**: Blocks duplicated >3 times should be extracted to utilities",
            "2. **Medium Priority**: Similar patterns suggest architectural improvements",
            "3. **Low Priority**: Minor duplications may be acceptable for readability",
            "",
            "## Next Steps",
            "",
            "1. Review each duplicate block manually",
            "2. Identify common functionality that can be abstracted",
            "3. Create utility functions or base classes",
            "4. Update all instances to use the new shared code",
            "5. Add tests for the new shared components",
        ]
    )

    report_content = "\n".join(report_lines)

    if output_file:
        # nosec - developer-facing CLI; output_file is the user's --output arg
        # and they own the destination by definition.
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Report saved to {output_file}")

    return report_content


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect code duplication in Python projects"
    )
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument(
        "--min-lines", type=int, default=5, help="Minimum lines for duplication"
    )
    parser.add_argument("--output", "-o", help="Output file for report")

    args = parser.parse_args()

    base_dir = Path.cwd()
    try:
        directory = validate_path(args.directory, base_dir, must_exist=True)
    except (PathTraversalError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1

    detector = DuplicationDetector(min_lines=args.min_lines)
    duplicates = detector.analyze_directory(directory)

    if not duplicates:
        print("No code duplication found!")
        return 0

    output_file = validate_path(
        args.output or f"duplication_report_{directory.name}.md",
        base_dir,
    )
    generate_report(duplicates, output_file)

    print(f"Analysis complete. Found {len(duplicates)} duplicate code blocks.")
    print(f"Report: {output_file}")


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)
