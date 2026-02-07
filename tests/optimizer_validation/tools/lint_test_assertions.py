#!/usr/bin/env python3
"""CI lint rule for test assertion quality.

This script checks that tests follow assertion best practices:
1. Tests calling result_validator must assert validation.passed
2. Tests should have explicit behavior assertions (not just exception checks)
3. Tests should not rely solely on vacuous assertions
4. Tests should not use 'assert not isinstance(result, Exception)' as sole check
5. Tests should not use 'assert hasattr' without checking the attribute value
6. ExpectedResult should specify at least one validation field

Usage:
    python -m tests.optimizer_validation.tools.lint_test_assertions
    python -m tests.optimizer_validation.tools.lint_test_assertions --strict
    python -m tests.optimizer_validation.tools.lint_test_assertions --fix

Exit codes:
    0: All checks pass
    1: Lint errors found
    2: Lint warnings found (only with --strict)
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LintIssue:
    file: str
    line: int
    column: int
    severity: Severity
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.severity.value}[{self.code}] {self.message}"


class TestAssertionLinter(ast.NodeVisitor):
    """AST-based linter for test assertions."""

    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.source_lines = source.split("\n")
        self.issues: list[LintIssue] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test_"):
            self._check_test_function(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("test_"):
            self._check_test_function(node)
        self.generic_visit(node)

    def _check_test_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Check a test function for assertion issues."""
        source = "\n".join(
            self.source_lines[node.lineno - 1 : node.end_lineno or node.lineno + 50]
        )

        # Rule T001: result_validator called without asserting validation.passed
        if "result_validator" in source:
            if "validation.passed" not in source and ".passed" not in source:
                # Check if validator result is assigned
                if "validation = result_validator" not in source:
                    self.issues.append(
                        LintIssue(
                            file=self.file_path,
                            line=node.lineno,
                            column=0,
                            severity=Severity.ERROR,
                            code="T001",
                            message=f"Test '{node.name}' calls result_validator without asserting validation.passed",
                        )
                    )

        # Rule T002: Test has only exception check (no behavior assertions)
        assertions = self._find_assertions(node)
        has_behavior_assertion = False

        behavior_patterns = [
            "trials",
            "stop_reason",
            "best_config",
            "best_score",
            "metrics",
            "config",
            "trial_count",
        ]

        for assertion in assertions:
            code = assertion.get("code", "")
            if any(p in code for p in behavior_patterns):
                has_behavior_assertion = True

        if not has_behavior_assertion and "result_validator" in source:
            self.issues.append(
                LintIssue(
                    file=self.file_path,
                    line=node.lineno,
                    column=0,
                    severity=Severity.WARNING,
                    code="T002",
                    message=f"Test '{node.name}' has no explicit behavior assertions",
                )
            )

        # Rule T003: Vacuous assertion detected
        for assertion in assertions:
            code = assertion.get("code", "")
            if "len(" in code and ">= 0" in code:
                self.issues.append(
                    LintIssue(
                        file=self.file_path,
                        line=assertion.get("line", node.lineno),
                        column=0,
                        severity=Severity.WARNING,
                        code="T003",
                        message="Vacuous assertion 'len(x) >= 0' is always true",
                    )
                )
            if code.strip() == "assert True":
                self.issues.append(
                    LintIssue(
                        file=self.file_path,
                        line=assertion.get("line", node.lineno),
                        column=0,
                        severity=Severity.ERROR,
                        code="T003",
                        message="Vacuous assertion 'assert True'",
                    )
                )

        # Rule T004: 'assert not isinstance(result, Exception)' as sole check
        self._check_isinstance_sole_assertion(node, assertions, behavior_patterns)

        # Rule T005: 'assert hasattr(obj, X)' without subsequent value check
        self._check_hasattr_without_value(node, assertions, source)

    def _check_isinstance_sole_assertion(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        assertions: list[dict],
        behavior_patterns: list[str],
    ) -> None:
        """T004: Flag tests where 'not isinstance(x, Exception)' is the only real check."""
        isinstance_assertions = []
        non_isinstance_behavior = []

        for assertion in assertions:
            code = assertion.get("code", "")
            # Match: assert not isinstance(result, Exception)
            # Also match: assert not isinstance(result, BaseException)
            if re.search(
                r"assert\s+not\s+isinstance\s*\(.+,\s*(Exception|BaseException)\s*\)",
                code,
            ):
                isinstance_assertions.append(assertion)
            elif any(p in code for p in behavior_patterns):
                non_isinstance_behavior.append(assertion)

        # Only flag if isinstance check exists AND no behavior assertions follow
        if isinstance_assertions and not non_isinstance_behavior:
            self.issues.append(
                LintIssue(
                    file=self.file_path,
                    line=isinstance_assertions[0].get("line", node.lineno),
                    column=0,
                    severity=Severity.WARNING,
                    code="T004",
                    message=(
                        f"Test '{node.name}' uses 'assert not isinstance(result, Exception)' "
                        "as sole behavioral check — add explicit assertions on trials/config/metrics"
                    ),
                )
            )

    def _check_hasattr_without_value(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        assertions: list[dict],
        source: str,
    ) -> None:
        """T005: Flag 'assert hasattr(obj, attr)' without subsequent value access."""
        for assertion in assertions:
            code = assertion.get("code", "")
            match = re.search(
                r"assert\s+hasattr\s*\(\s*(\w+)\s*,\s*[\"'](\w+)[\"']\s*\)", code
            )
            if not match:
                continue

            obj_name = match.group(1)
            attr_name = match.group(2)

            # Check if the attribute is accessed/asserted elsewhere in the test
            # Look for patterns like: obj.attr, getattr(obj, 'attr')
            access_pattern = rf"{re.escape(obj_name)}\.{re.escape(attr_name)}"
            # Exclude the hasattr line itself from the search
            lines_after = source[source.find(code) + len(code) :]
            if not re.search(access_pattern, lines_after):
                self.issues.append(
                    LintIssue(
                        file=self.file_path,
                        line=assertion.get("line", node.lineno),
                        column=0,
                        severity=Severity.WARNING,
                        code="T005",
                        message=(
                            f"Test '{node.name}' asserts hasattr({obj_name}, '{attr_name}') "
                            "without checking the attribute value"
                        ),
                    )
                )

    def _find_assertions(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[dict]:
        """Find all assertions in a function."""
        assertions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                line = child.lineno
                code = (
                    self.source_lines[line - 1]
                    if line <= len(self.source_lines)
                    else ""
                )
                assertions.append({"line": line, "code": code.strip()})
        return assertions


class ExpectedResultLinter(ast.NodeVisitor):
    """AST-based linter for ExpectedResult instantiation quality."""

    # Fields that make an ExpectedResult non-vacuous
    VALIDATION_FIELDS = {
        "expected_stop_reason",
        "best_score_range",
        "required_metrics",
    }

    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.source_lines = source.split("\n")
        self.issues: list[LintIssue] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Check ExpectedResult(...) calls for missing validation fields."""
        # Match: ExpectedResult(...)
        func = node.func
        if isinstance(func, ast.Name) and func.id == "ExpectedResult":
            self._check_expected_result(node)
        elif isinstance(func, ast.Attribute) and func.attr == "ExpectedResult":
            self._check_expected_result(node)
        self.generic_visit(node)

    def _check_expected_result(self, node: ast.Call) -> None:
        """T006: Flag ExpectedResult missing all validation fields."""
        # Check if this is a failure scenario (outcome=FAILURE) — skip those
        for kw in node.keywords:
            if kw.arg == "outcome":
                # If outcome is set to FAILURE, skip — failure tests don't need
                # stop_reason or best_score_range
                val = kw.value
                if isinstance(val, ast.Attribute) and "FAILURE" in val.attr:
                    return
                if isinstance(val, ast.Name) and "FAILURE" in val.id:
                    return

        # Collect all keyword argument names
        kwarg_names = {kw.arg for kw in node.keywords if kw.arg is not None}

        # Check if ANY validation field is specified
        has_validation = bool(kwarg_names & self.VALIDATION_FIELDS)

        if not has_validation:
            self.issues.append(
                LintIssue(
                    file=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    severity=Severity.INFO,
                    code="T006",
                    message=(
                        "ExpectedResult() missing all validation fields — "
                        "add expected_stop_reason, best_score_range, or required_metrics"
                    ),
                )
            )


def lint_file(file_path: Path) -> list[LintIssue]:
    """Lint a single file."""
    source = file_path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    linter = TestAssertionLinter(str(file_path), source)
    linter.visit(tree)

    # Also check ExpectedResult instantiations
    er_linter = ExpectedResultLinter(str(file_path), source)
    er_linter.visit(tree)
    linter.issues.extend(er_linter.issues)

    return linter.issues


def lint_directory(directory: Path) -> list[LintIssue]:
    """Lint all test files in a directory."""
    all_issues = []
    for file_path in directory.rglob("test_*.py"):
        if "tools" in file_path.parts:
            continue
        issues = lint_file(file_path)
        all_issues.extend(issues)
    return all_issues


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Lint test assertions")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (not just errors)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        default=Path("tests/optimizer_validation"),
        help="Directory to lint",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "github"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    issues = lint_directory(args.directory)

    # Print issues grouped by severity
    errors = [i for i in issues if i.severity == Severity.ERROR]
    warnings = [i for i in issues if i.severity == Severity.WARNING]
    infos = [i for i in issues if i.severity == Severity.INFO]

    if args.format == "github":
        # GitHub Actions annotation format
        for issue in issues:
            if issue.severity == Severity.INFO:
                continue  # Skip INFO in GitHub format
            level = "error" if issue.severity == Severity.ERROR else "warning"
            print(
                f"::{level} file={issue.file},line={issue.line},col={issue.column}::{issue.message}"
            )
    else:
        for issue in issues:
            print(issue)

    print()
    print(
        f"Found {len(errors)} errors, {len(warnings)} warnings, "
        f"and {len(infos)} info"
    )

    if errors:
        return 1
    if args.strict and warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
