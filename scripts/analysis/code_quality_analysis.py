#!/usr/bin/env python3
"""Code quality analysis script for import hygiene and defensive patterns."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class Issue:
    category: str
    severity: str
    line: int
    description: str
    suggestion: str


class CodeQualityAnalyzer:
    """Analyzes Python files for import hygiene and defensive patterns."""

    def __init__(self) -> None:
        self.issues: Dict[Path, List[Issue]] = {}

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[warn] Unable to read {file_path}: {exc}")
            return

        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            self._record_issue(
                file_path,
                Issue(
                    category="syntax",
                    severity="High",
                    line=exc.lineno or 0,
                    description=f"Unable to parse file: {exc.msg}",
                    suggestion="Fix syntax errors so static analysis can run.",
                ),
            )
            return

        issues: List[Issue] = []

        # Pass 1: Import Hygiene Analysis
        issues.extend(self._analyze_import_hygiene(tree))

        # Pass 2: Defensive Patterns Analysis
        issues.extend(self._analyze_defensive_patterns(tree))

        if issues:
            self.issues[file_path] = issues

    def _analyze_import_hygiene(self, tree: ast.Module) -> List[Issue]:
        """Analyze import hygiene issues."""
        issues: List[Issue] = []

        body = list(tree.body)
        docstring_offset = 0
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                docstring_offset = 1

        imports_seen = False
        code_before_import = None
        code_seen_after_imports = False

        for node in body[docstring_offset:]:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_seen = True
                if code_before_import is not None:
                    issues.append(
                        Issue(
                            category="import_hygiene",
                            severity="Medium",
                            line=code_before_import.lineno,
                            description="Executable statements appear before imports.",
                            suggestion="Place imports directly below the module docstring.",
                        )
                    )
                if code_seen_after_imports:
                    issues.append(
                        Issue(
                            category="import_hygiene",
                            severity="Medium",
                            line=node.lineno,
                            description="Import found after executable code.",
                            suggestion="Move late imports to the top of the file or justify the lazy import.",
                        )
                    )
                continue

            if not imports_seen and code_before_import is None:
                code_before_import = node
            elif imports_seen:
                code_seen_after_imports = True

        # Check for duplicate imports
        import_statements: List[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_statements.append((node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_statements.append((node.lineno, f"from {module} import {alias.name}"))

        seen: Dict[str, int] = {}
        for lineno, stmt in import_statements:
            if stmt in seen:
                issues.append(
                    Issue(
                        category="import_hygiene",
                        severity="Medium",
                        line=lineno,
                        description=f"Duplicate import: {stmt} (first seen on line {seen[stmt]}).",
                        suggestion="Consolidate duplicate imports.",
                    )
                )
            else:
                seen[stmt] = lineno

        # Check for try/except wrapping imports that swallow errors
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            if not any(isinstance(stmt, (ast.Import, ast.ImportFrom)) for stmt in node.body):
                continue

            for handler in node.handlers:
                if not self._is_broad_exception(handler.type):
                    continue
                if any(isinstance(stmt, ast.Raise) for stmt in handler.body):
                    continue
                issues.append(
                    Issue(
                        category="import_hygiene",
                        severity="Medium",
                        line=handler.lineno or node.lineno,
                        description="Import guarded by a broad exception handler without a fallback.",
                        suggestion="Handle optional imports explicitly or re-raise after logging.",
                    )
                )

        return issues

    def _analyze_defensive_patterns(self, tree: ast.Module) -> List[Issue]:
        """Analyze defensive programming patterns."""
        issues: List[Issue] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            for handler in node.handlers:
                if not self._is_broad_exception(handler.type):
                    continue
                if self._handler_has_meaningful_action(handler.body):
                    continue
                issues.append(
                    Issue(
                        category="defensive_pattern",
                        severity="High",
                        line=handler.lineno or node.lineno,
                        description="Broad exception handler swallows errors without remediation.",
                        suggestion="Catch specific exceptions or re-raise after logging relevant context.",
                    )
                )

        return issues

    @staticmethod
    def _is_broad_exception(handler_type: ast.expr | None) -> bool:
        return handler_type is None or (
            isinstance(handler_type, ast.Name) and handler_type.id in {"Exception", "BaseException"}
        )

    @staticmethod
    def _handler_has_meaningful_action(body: Sequence[ast.stmt]) -> bool:
        if any(isinstance(stmt, ast.Raise) for stmt in body):
            return True
        # Allow return statements that propagate an error indicator
        if any(isinstance(stmt, ast.Return) and stmt.value is not None for stmt in body):
            return True
        # Consider bare passes and docstring-like expressions as non-meaningful
        return not all(isinstance(stmt, (ast.Pass, ast.Expr)) for stmt in body)

    def _record_issue(self, file_path: Path, issue: Issue) -> None:
        self.issues.setdefault(file_path, []).append(issue)

    def analyze_directory(self, directory: Path) -> None:
        """Analyze all Python files in a directory recursively."""
        for file_path in directory.rglob("*.py"):
            if any(part.startswith(".") for part in file_path.parts):
                continue
            self.analyze_file(file_path)

    def analyze_paths(self, paths: Iterable[Path]) -> None:
        for path in paths:
            if path.is_file():
                self.analyze_file(path)
            elif path.is_dir():
                self.analyze_directory(path)

    def print_report(self) -> None:
        """Print the analysis report."""
        if not self.issues:
            print("No code quality issues found.")
            return

        total_files = len(self.issues)
        total_issues = sum(len(file_issues) for file_issues in self.issues.values())
        print(f"Code Quality Analysis Report - {total_issues} issues across {total_files} files\n")

        for file_path, file_issues in sorted(self.issues.items(), key=lambda item: str(item[0])):
            print(str(file_path))
            print("-" * len(str(file_path)))

            import_issues = [i for i in file_issues if i.category == "import_hygiene"]
            defensive_issues = [i for i in file_issues if i.category == "defensive_pattern"]
            syntax_issues = [i for i in file_issues if i.category == "syntax"]

            for label, issues in (
                ("Syntax", syntax_issues),
                ("Import hygiene", import_issues),
                ("Defensive patterns", defensive_issues),
            ):
                if not issues:
                    continue
                print(f"{label}:")
                for issue in issues:
                    print(f"  {issue.severity}: line {issue.line} - {issue.description}")
                    print(f"    Suggestion: {issue.suggestion}")
            print()


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Static checks for import hygiene and defensive programming patterns."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["traigent", "tests", "examples"],
        help="Files or directories to scan. Relative paths are resolved from the repository root.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    targets: List[Path] = []
    for raw_path in args.paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        if candidate.exists():
            targets.append(candidate)
        else:
            print(f"[warn] Skipping missing path: {candidate}")

    if not targets:
        print("[error] No valid files or directories to analyze.")
        return

    analyzer = CodeQualityAnalyzer()
    analyzer.analyze_paths(targets)
    analyzer.print_report()


if __name__ == "__main__":
    main()
