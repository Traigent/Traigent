#!/usr/bin/env python3
"""CI lint rule for pricing table consistency.

Detects hardcoded LLM pricing tables outside the canonical source
(traigent/utils/cost_calculator.py). Prevents pricing drift where
different files maintain independent rate tables that silently diverge.

Detection rules:
  P001 - Dict with pricing-pattern keys (prompt/completion/input/output
         + cost context) containing numeric values
  P002 - Dict with model-name keys (gpt-*, claude-*, gemini-*, o1-*, etc.)
         whose values are numeric (floats, tuples, or nested dicts)
  P003 - Variable assignment with pricing-pattern name (cost_per_*,
         *_rates, *_pricing, *_COST_*) containing numeric dict/tuple

Usage:
    python -m tests.optimizer_validation.tools.lint_pricing_consistency
    python -m tests.optimizer_validation.tools.lint_pricing_consistency --strict
    python -m tests.optimizer_validation.tools.lint_pricing_consistency --format github

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
        return (
            f"{self.file}:{self.line}:{self.column}: "
            f"{self.severity.value}[{self.code}] {self.message}"
        )


# --- Allowlist: files that may legitimately contain pricing data ---
ALLOWLISTED_FILES = {
    # Canonical source of truth for LLM pricing
    "traigent/utils/cost_calculator.py",
    # Billing tier rates (not LLM model pricing)
    "traigent/cloud/billing.py",
}

# Path segments that exclude entire subtrees.
# Matched as exact directory components (not substrings).
ALLOWLISTED_PATH_SEGMENTS = {"tests", "experimental"}

# --- Detection patterns ---
_PRICING_KEYS = {
    "prompt",
    "completion",
    "input_cost_per_token",
    "output_cost_per_token",
}

# When paired together, "input" and "output" are pricing-related
_PAIRED_PRICING_KEYS = {"input", "output"}

_MODEL_NAME_RE = re.compile(
    r"^(gpt-|claude-|gemini-|o1-|o3-|o4-|llama|mistral)", re.IGNORECASE
)

_PRICING_VAR_RE = re.compile(
    r"(cost_per_|_rates$|_pricing$|_COST_|COST_PER_|per_1k|per_1m|TOKEN_COST)",
    re.IGNORECASE,
)


def _is_numeric_node(node: ast.expr) -> bool:
    """Check if an AST node represents a numeric literal or container of numerics."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_numeric_node(node.operand)
    if isinstance(node, ast.Tuple):
        return all(_is_numeric_node(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        # For dicts: require at least half the values to be numeric.
        # This catches mixed tables (pricing + metadata) without over-triggering.
        total = len(node.values)
        numeric = sum(1 for v in node.values if v is not None and _is_numeric_leaf(v))
        return numeric > 0 and numeric * 2 >= total
    return False


def _is_numeric_leaf(node: ast.expr) -> bool:
    """Check if a single AST node is a numeric literal (not recursive into containers)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_numeric_leaf(node.operand)
    if isinstance(node, ast.Tuple):
        return all(_is_numeric_leaf(elt) for elt in node.elts)
    return False


def _get_string_keys(node: ast.Dict) -> list[str]:
    """Extract string keys from an AST Dict node."""
    keys = []
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)
    return keys


class PricingLinter(ast.NodeVisitor):
    """AST-based linter for hardcoded pricing tables."""

    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.issues: list[LintIssue] = []
        # Track P002 hit lines so P001 can suppress child-dict noise
        self._p002_lines: set[int] = set()

    def visit_Dict(self, node: ast.Dict) -> None:
        """Check dict literals for pricing patterns."""
        str_keys = _get_string_keys(node)
        if str_keys:
            self._check_p002(node, str_keys)
            self._check_p001(node, str_keys)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check variable assignments for pricing-pattern names."""
        self._check_p003_from_value(
            node.value, node.targets, node.lineno, node.col_offset
        )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check annotated assignments for pricing-pattern names."""
        if node.value is not None and node.target is not None:
            self._check_p003_from_value(
                node.value, [node.target], node.lineno, node.col_offset
            )
        self.generic_visit(node)

    def _check_p001(self, node: ast.Dict, str_keys: list[str]) -> None:
        """P001: Dict with pricing-pattern keys containing numeric values.

        Suppressed when the parent dict already triggered P002 (avoids noisy
        duplicates for each sub-dict like {"input": 0.01, "output": 0.03}
        inside a model table).
        """
        # If this dict is a child value of a P002 dict, skip
        if node.lineno in self._p002_lines:
            return

        key_set = set(str_keys)

        # Direct pricing keys: prompt, completion, input_cost_per_token, etc.
        has_pricing_key = bool(key_set & _PRICING_KEYS)

        # Paired keys: "input" + "output" together indicate pricing
        has_paired = _PAIRED_PRICING_KEYS.issubset(key_set)

        if not has_pricing_key and not has_paired:
            return

        # Check if values are numeric
        has_numeric_values = any(
            _is_numeric_leaf(v) or (isinstance(v, ast.Tuple) and _is_numeric_node(v))
            for v in node.values
            if v is not None
        )
        if not has_numeric_values:
            return

        matched_keys = sorted(key_set & (_PRICING_KEYS | _PAIRED_PRICING_KEYS))
        self.issues.append(
            LintIssue(
                file=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                severity=Severity.ERROR,
                code="P001",
                message=(
                    f"Hardcoded pricing dict with keys {matched_keys}. "
                    "Use traigent.utils.cost_calculator as the single source of truth."
                ),
            )
        )

    def _check_p002(self, node: ast.Dict, str_keys: list[str]) -> None:
        """P002: Dict with model-name keys and numeric-ish values."""
        model_keys = [k for k in str_keys if _MODEL_NAME_RE.match(k)]
        if len(model_keys) < 2:
            return

        # Check if values are numeric (floats, tuples, or nested dicts with numerics)
        numeric_value_count = sum(
            1 for v in node.values if v is not None and _is_numeric_node(v)
        )
        if numeric_value_count < 2:
            return

        # Record this line + all child value lines to suppress P001 noise
        self._p002_lines.add(node.lineno)
        for v in node.values:
            if v is not None:
                self._p002_lines.add(v.lineno)

        self.issues.append(
            LintIssue(
                file=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                severity=Severity.ERROR,
                code="P002",
                message=(
                    f"Model pricing table with {len(model_keys)} model keys "
                    f"({', '.join(model_keys[:3])}{'...' if len(model_keys) > 3 else ''}). "
                    "Use traigent.utils.cost_calculator as the single source of truth."
                ),
            )
        )

    def _check_p003_from_value(
        self,
        value: ast.expr,
        targets: list[ast.expr],
        lineno: int,
        col_offset: int,
    ) -> None:
        """P003: Variable with pricing-pattern name assigned a numeric dict/tuple."""
        for target in targets:
            name = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr

            if (
                name
                and _PRICING_VAR_RE.search(name)
                and isinstance(value, (ast.Dict, ast.Tuple))
                and _is_numeric_node(value)
            ):
                self.issues.append(
                    LintIssue(
                        file=self.file_path,
                        line=lineno,
                        column=col_offset,
                        severity=Severity.WARNING,
                        code="P003",
                        message=(
                            f"Variable '{name}' looks like a pricing table. "
                            "Ensure it derives from "
                            "traigent.utils.cost_calculator.ESTIMATION_MODEL_PRICING."
                        ),
                    )
                )


def _is_allowlisted(file_path: str) -> bool:
    """Check if a file is in the pricing data allowlist.

    Uses path-anchored matching: ALLOWLISTED_FILES checks suffixes,
    ALLOWLISTED_PATH_SEGMENTS checks exact directory components.
    """
    normalized = Path(file_path).as_posix()
    for allowed in ALLOWLISTED_FILES:
        if normalized.endswith(allowed):
            return True
    parts = Path(normalized).parts
    for segment in ALLOWLISTED_PATH_SEGMENTS:
        if segment in parts:
            return True
    return False


def lint_file(file_path: Path) -> list[LintIssue]:
    """Lint a single file for pricing consistency issues."""
    if _is_allowlisted(str(file_path)):
        return []

    source = file_path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    linter = PricingLinter(str(file_path), source)
    linter.visit(tree)
    return linter.issues


def lint_directory(directory: Path) -> list[LintIssue]:
    """Lint all Python files in a directory (excluding allowlisted)."""
    all_issues = []
    for file_path in sorted(directory.rglob("*.py")):
        issues = lint_file(file_path)
        all_issues.extend(issues)
    return all_issues


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Lint for hardcoded pricing tables outside canonical source"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (not just errors)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        default=Path("traigent"),
        help="Directory to lint (default: traigent/)",
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

    errors = [i for i in issues if i.severity == Severity.ERROR]
    warnings = [i for i in issues if i.severity == Severity.WARNING]
    infos = [i for i in issues if i.severity == Severity.INFO]

    if args.format == "github":
        for issue in issues:
            if issue.severity == Severity.INFO:
                continue
            level = "error" if issue.severity == Severity.ERROR else "warning"
            print(
                f"::{level} file={issue.file},line={issue.line},"
                f"col={issue.column}::{issue.message}"
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
