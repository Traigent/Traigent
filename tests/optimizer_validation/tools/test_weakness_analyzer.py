#!/usr/bin/env python3
"""Static analysis tool for detecting weak tests in the optimizer validation suite.

This tool detects three primary anti-patterns:
1. IT-VRO: Assertion Abstraction Leakage (Validator Reliance Only)
2. IT-CBM: Condition-Behavior Mismatch
3. IT-VTA: Vacuous Truth Assertions

Usage:
    python -m tests.optimizer_validation.tools.test_weakness_analyzer
    python -m tests.optimizer_validation.tools.test_weakness_analyzer --output json
    python -m tests.optimizer_validation.tools.test_weakness_analyzer --fix
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class IssueType(Enum):
    """Issue type codes for test weaknesses."""

    IT_VRO = "IT-VRO"  # Validator Reliance Only
    IT_CBM = "IT-CBM"  # Condition-Behavior Mismatch
    IT_VTA = "IT-VTA"  # Vacuous Truth Assertions
    IT_NTV = "IT-NTV"  # No Trial Verification
    IT_NCV = "IT-NCV"  # No Config Verification
    IT_NSR = "IT-NSR"  # No Stop Reason Check
    IT_NRV = "IT-NRV"  # No Range Validation
    IT_NEM = "IT-NEM"  # No Error Message Check


class RootCause(Enum):
    """Root cause codes for test weaknesses."""

    RC_MA = "RC-MA"  # Missing Assertions
    RC_WA = "RC-WA"  # Weak Assertions
    RC_GB = "RC-GB"  # Guard Bypass
    RC_MV = "RC-MV"  # Missing Validation
    RC_SM = "RC-SM"  # Scenario Mismatch


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestIssue:
    """Represents a detected test weakness."""

    issue_type: IssueType
    root_cause: RootCause
    severity: Severity
    file_path: str
    test_name: str
    line_number: int
    description: str
    evidence: list[str] = field(default_factory=list)
    suggested_fix: str | None = None


@dataclass
class AnalysisResult:
    """Result of analyzing a test file."""

    file_path: str
    total_tests: int
    issues: list[TestIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "total_tests": self.total_tests,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "issue_type": issue.issue_type.value,
                    "root_cause": issue.root_cause.value,
                    "severity": issue.severity.value,
                    "test_name": issue.test_name,
                    "line_number": issue.line_number,
                    "description": issue.description,
                    "evidence": issue.evidence,
                    "suggested_fix": issue.suggested_fix,
                }
                for issue in self.issues
            ],
        }


class TestASTVisitor(ast.NodeVisitor):
    """AST visitor for analyzing test functions."""

    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.source_lines = source.split("\n")
        self.issues: list[TestIssue] = []
        self.test_count = 0

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions (async test functions)."""
        if node.name.startswith("test_"):
            self.test_count += 1
            self._analyze_test_function(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        if node.name.startswith("test_"):
            self.test_count += 1
            self._analyze_test_function(node)
        self.generic_visit(node)

    def _analyze_test_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Analyze a test function for weaknesses."""
        analysis = TestFunctionAnalysis(node, self.source_lines)

        # Check for IT-VRO: Validator Reliance Only
        vro_issues = self._check_validator_reliance_only(node, analysis)
        self.issues.extend(vro_issues)

        # Check for IT-VTA: Vacuous Truth Assertions
        vta_issues = self._check_vacuous_assertions(node, analysis)
        self.issues.extend(vta_issues)

        # Check for IT-CBM: Condition-Behavior Mismatch
        cbm_issues = self._check_condition_behavior_mismatch(node, analysis)
        self.issues.extend(cbm_issues)

        # Check for missing specific verifications
        missing_issues = self._check_missing_verifications(node, analysis)
        self.issues.extend(missing_issues)

    def _check_validator_reliance_only(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        analysis: "TestFunctionAnalysis",
    ) -> list[TestIssue]:
        """Check for IT-VRO: tests that only rely on result_validator."""
        issues = []

        if analysis.has_validator_call and not analysis.has_validation_passed_assert:
            # Validator called but validation.passed not asserted
            issues.append(
                TestIssue(
                    issue_type=IssueType.IT_VRO,
                    root_cause=RootCause.RC_MA,
                    severity=Severity.HIGH,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=node.lineno,
                    description="result_validator called but validation.passed not asserted",
                    evidence=analysis.validator_call_lines,
                    suggested_fix="Add: validation = result_validator(scenario, result); assert validation.passed, validation.summary()",
                )
            )

        if (
            analysis.has_validator_call
            and analysis.has_validation_passed_assert
            and not analysis.has_explicit_behavior_assertions
        ):
            # Validator called with assert, but no behavior-specific assertions
            # This is a softer form of VRO
            issues.append(
                TestIssue(
                    issue_type=IssueType.IT_VRO,
                    root_cause=RootCause.RC_MA,
                    severity=Severity.MEDIUM,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=node.lineno,
                    description="Test relies only on validator without explicit behavior assertions",
                    evidence=["Only assertions: exception check + validator"],
                    suggested_fix="Add explicit assertions for trials, configs, stop_reason, or metrics",
                )
            )

        return issues

    def _check_vacuous_assertions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        analysis: "TestFunctionAnalysis",
    ) -> list[TestIssue]:
        """Check for IT-VTA: Vacuous Truth Assertions."""
        issues = []

        for assertion in analysis.vacuous_assertions:
            issues.append(
                TestIssue(
                    issue_type=IssueType.IT_VTA,
                    root_cause=RootCause.RC_WA,
                    severity=Severity.MEDIUM,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=assertion["line"],
                    description=f"Vacuous assertion: {assertion['pattern']}",
                    evidence=[assertion["code"]],
                    suggested_fix=assertion.get("fix"),
                )
            )

        return issues

    def _check_condition_behavior_mismatch(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        analysis: "TestFunctionAnalysis",
    ) -> list[TestIssue]:
        """Check for IT-CBM: Condition-Behavior Mismatch."""
        issues = []

        # Check if max_trials exceeds config space cardinality for grid search
        if analysis.is_grid_search:
            config_cardinality = analysis.compute_config_cardinality()
            max_trials = analysis.get_max_trials()

            if max_trials and config_cardinality and max_trials > config_cardinality:
                issues.append(
                    TestIssue(
                        issue_type=IssueType.IT_CBM,
                        root_cause=RootCause.RC_GB,
                        severity=Severity.HIGH,
                        file_path=self.file_path,
                        test_name=node.name,
                        line_number=node.lineno,
                        description=f"max_trials ({max_trials}) > config space cardinality ({config_cardinality}), stop condition cannot trigger",
                        evidence=[
                            f"Config space: {config_cardinality} combinations",
                            f"max_trials: {max_trials}",
                        ],
                        suggested_fix="Increase config space or reduce max_trials",
                    )
                )

        # Check stop condition tests without expected_stop_reason
        if (
            analysis.test_name_suggests_stop_condition
            and not analysis.has_stop_reason_check
        ):
            issues.append(
                TestIssue(
                    issue_type=IssueType.IT_CBM,
                    root_cause=RootCause.RC_MV,
                    severity=Severity.MEDIUM,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=node.lineno,
                    description="Test name suggests stop condition but no stop_reason assertion",
                    evidence=[f"Test name: {node.name}"],
                    suggested_fix="Add expected_stop_reason to ExpectedResult or assert stop_reason",
                )
            )

        # Check failure tests without proper error verification
        if analysis.expects_failure and not analysis.has_error_verification:
            issues.append(
                TestIssue(
                    issue_type=IssueType.IT_CBM,
                    root_cause=RootCause.RC_MV,
                    severity=Severity.HIGH,
                    file_path=self.file_path,
                    test_name=node.name,
                    line_number=node.lineno,
                    description="Test expects failure but doesn't verify error type or message",
                    evidence=["ExpectedOutcome.FAILURE without error checks"],
                    suggested_fix="Add error_type and/or error_message_contains to ExpectedResult",
                )
            )

        return issues

    def _check_missing_verifications(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        analysis: "TestFunctionAnalysis",
    ) -> list[TestIssue]:
        """Check for missing specific verifications."""
        issues = []

        # Test with trials but no trial count assertion
        if analysis.accesses_trials and not analysis.has_trial_count_assertion:
            # Only flag if the test seems to expect trials
            if not analysis.expects_failure and not analysis.has_exception_check:
                pass  # Lower priority, handled by VRO

        return issues


class TestFunctionAnalysis:
    """Detailed analysis of a test function."""

    def __init__(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: list[str]
    ) -> None:
        self.node = node
        self.source_lines = source_lines
        self._analyze()

    def _analyze(self) -> None:
        """Run all analysis checks."""
        self._extract_assertions()
        self._check_validator_usage()
        self._check_scenario_config()
        self._check_expected_result()

    def _extract_assertions(self) -> None:
        """Extract and categorize all assertions."""
        self.assertions: list[dict] = []
        self.vacuous_assertions: list[dict] = []

        for child in ast.walk(self.node):
            if isinstance(child, ast.Assert):
                assertion_code = self._get_source_line(child.lineno)
                assertion_info = {
                    "line": child.lineno,
                    "code": assertion_code.strip(),
                    "node": child,
                }
                self.assertions.append(assertion_info)

                # Check for vacuous patterns
                vacuous = self._check_vacuous_pattern(child, assertion_code)
                if vacuous:
                    vacuous["line"] = child.lineno
                    vacuous["code"] = assertion_code.strip()
                    self.vacuous_assertions.append(vacuous)

    def _check_vacuous_pattern(self, assert_node: ast.Assert, code: str) -> dict | None:
        """Check if assertion matches vacuous patterns."""
        code_lower = code.lower()

        # Pattern: assert True
        if re.search(r"assert\s+True\s*(,|$|#)", code):
            return {
                "pattern": "assert True",
                "fix": "Remove or replace with meaningful assertion",
            }

        # Pattern: assert len(x) >= 0
        if re.search(r"len\([^)]+\)\s*>=\s*0", code):
            return {
                "pattern": "len(x) >= 0 (always true)",
                "fix": "Use >= 1 or specific expected count",
            }

        # Pattern: hasattr only check without value check
        if (
            re.search(r"assert\s+hasattr\(", code)
            and "assert" not in code[code.find("hasattr") + 10 :]
        ):
            # Check if this is the only assertion for this attribute
            if not any(
                attr in code
                for attr in [".passed", ".trials", ".config", ".stop_reason"]
            ):
                return {
                    "pattern": "hasattr only (no value check)",
                    "fix": "Also assert the attribute value",
                }

        # Pattern: isinstance(result, Exception) without type/message check
        if "isinstance" in code and "Exception" in code:
            if not any(
                check in code
                for check in [
                    "error_type",
                    "error_message",
                    "str(result)",
                    "type(result)",
                ]
            ):
                return {
                    "pattern": "isinstance Exception (no type check)",
                    "fix": "Check specific exception type",
                }

        return None

    def _check_validator_usage(self) -> None:
        """Analyze result_validator usage patterns."""
        self.has_validator_call = False
        self.has_validation_passed_assert = False
        self.validator_call_lines: list[str] = []

        source_block = "\n".join(
            self.source_lines[
                self.node.lineno - 1 : self.node.end_lineno or self.node.lineno + 50
            ]
        )

        if "result_validator" in source_block:
            self.has_validator_call = True
            # Find lines with validator calls
            for i in range(
                self.node.lineno - 1,
                min(
                    self.node.end_lineno or len(self.source_lines),
                    len(self.source_lines),
                ),
            ):
                line = self.source_lines[i]
                if "result_validator" in line:
                    self.validator_call_lines.append(f"L{i+1}: {line.strip()}")

        if "validation.passed" in source_block or ".passed" in source_block:
            self.has_validation_passed_assert = True

    def _check_scenario_config(self) -> None:
        """Extract scenario configuration for feasibility analysis."""
        source_block = "\n".join(
            self.source_lines[
                self.node.lineno - 1 : self.node.end_lineno or self.node.lineno + 50
            ]
        )

        # Check if grid search
        self.is_grid_search = '"grid"' in source_block or "'grid'" in source_block

        # Extract max_trials
        max_trials_match = re.search(r"max_trials\s*=\s*(\d+)", source_block)
        self._max_trials = int(max_trials_match.group(1)) if max_trials_match else None

        # Extract config_space (simplified - counts list items)
        self._config_space_sizes: list[int] = []
        # Find lists in config_space
        config_space_match = re.search(
            r"config_space\s*=\s*\{([^}]+)\}", source_block, re.DOTALL
        )
        if config_space_match:
            config_content = config_space_match.group(1)
            # Count items in each list
            list_matches = re.findall(r"\[([^\]]+)\]", config_content)
            for list_match in list_matches:
                items = [x.strip() for x in list_match.split(",") if x.strip()]
                self._config_space_sizes.append(len(items))

    def _check_expected_result(self) -> None:
        """Extract expected result configuration."""
        source_block = "\n".join(
            self.source_lines[
                self.node.lineno - 1 : self.node.end_lineno or self.node.lineno + 50
            ]
        )

        self.expects_failure = "ExpectedOutcome.FAILURE" in source_block
        self.has_error_verification = (
            "error_type" in source_block or "error_message" in source_block
        )

        self.has_stop_reason_check = (
            "expected_stop_reason" in source_block
            or "stop_reason" in source_block.lower()
        )

        # Check for behavior assertions
        behavior_patterns = [
            "len(result.trials)",
            "result.trials",
            "trial.config",
            "result.stop_reason",
            "result.best_config",
            "result.best_score",
            "trial.metrics",
        ]
        self.has_explicit_behavior_assertions = any(
            pattern in source_block for pattern in behavior_patterns
        )

        # Additional checks
        self.accesses_trials = "trials" in source_block
        self.has_trial_count_assertion = (
            re.search(r"len\([^)]*trials[^)]*\)\s*(>=|==|<=|>|<)\s*\d+", source_block)
            is not None
        )
        self.has_exception_check = (
            "isinstance(result, Exception)" in source_block
            or "not isinstance(result, Exception)" in source_block
        )

    @property
    def test_name_suggests_stop_condition(self) -> bool:
        """Check if test name suggests it tests a stop condition."""
        stop_keywords = ["timeout", "max_trials", "stop", "halt", "terminate", "limit"]
        return any(kw in self.node.name.lower() for kw in stop_keywords)

    def compute_config_cardinality(self) -> int | None:
        """Compute the cardinality of the config space."""
        if not self._config_space_sizes:
            return None
        result = 1
        for size in self._config_space_sizes:
            result *= size
        return result

    def get_max_trials(self) -> int | None:
        """Get max_trials value."""
        return self._max_trials

    def _get_source_line(self, lineno: int) -> str:
        """Get source line by number."""
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1]
        return ""


def analyze_file(file_path: Path) -> AnalysisResult:
    """Analyze a single test file."""
    source = file_path.read_text()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return AnalysisResult(file_path=str(file_path), total_tests=0, issues=[])

    visitor = TestASTVisitor(str(file_path), source)
    visitor.visit(tree)

    return AnalysisResult(
        file_path=str(file_path),
        total_tests=visitor.test_count,
        issues=visitor.issues,
    )


def analyze_directory(directory: Path) -> list[AnalysisResult]:
    """Analyze all test files in a directory."""
    results = []

    for file_path in directory.rglob("test_*.py"):
        # Skip tools directory
        if "tools" in file_path.parts:
            continue
        result = analyze_file(file_path)
        results.append(result)

    return results


def generate_report(results: list[AnalysisResult], format: str = "text") -> str:
    """Generate analysis report."""
    total_tests = sum(r.total_tests for r in results)
    total_issues = sum(len(r.issues) for r in results)

    if format == "json":
        report = {
            "summary": {
                "total_files": len(results),
                "total_tests": total_tests,
                "total_issues": total_issues,
                "by_issue_type": {},
                "by_severity": {},
            },
            "files": [r.to_dict() for r in results],
        }

        # Aggregate by type
        for result in results:
            for issue in result.issues:
                type_key = issue.issue_type.value
                if type_key not in report["summary"]["by_issue_type"]:
                    report["summary"]["by_issue_type"][type_key] = 0
                report["summary"]["by_issue_type"][type_key] += 1

                sev_key = issue.severity.value
                if sev_key not in report["summary"]["by_severity"]:
                    report["summary"]["by_severity"][sev_key] = 0
                report["summary"]["by_severity"][sev_key] += 1

        return json.dumps(report, indent=2)

    # Text format
    lines = [
        "=" * 80,
        "TEST WEAKNESS ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Total Files Analyzed: {len(results)}",
        f"Total Tests Found: {total_tests}",
        f"Total Issues Found: {total_issues}",
        "",
    ]

    # Group by severity
    by_severity: dict[Severity, list[TestIssue]] = {}
    for result in results:
        for issue in result.issues:
            if issue.severity not in by_severity:
                by_severity[issue.severity] = []
            by_severity[issue.severity].append(issue)

    for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
        if severity in by_severity:
            issues = by_severity[severity]
            lines.append(f"\n{severity.value.upper()} Issues ({len(issues)}):")
            lines.append("-" * 40)
            for issue in issues[:10]:  # Show first 10
                lines.append(f"  [{issue.issue_type.value}] {issue.test_name}")
                lines.append(f"    File: {issue.file_path}:{issue.line_number}")
                lines.append(f"    {issue.description}")
                if issue.suggested_fix:
                    lines.append(f"    Fix: {issue.suggested_fix}")
                lines.append("")
            if len(issues) > 10:
                lines.append(f"  ... and {len(issues) - 10} more")

    # Summary by type
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY BY ISSUE TYPE")
    lines.append("=" * 80)

    by_type: dict[IssueType, int] = {}
    for result in results:
        for issue in result.issues:
            by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1

    for issue_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        pct = (count / total_tests * 100) if total_tests > 0 else 0
        lines.append(f"  {issue_type.value}: {count} ({pct:.1f}% of tests)")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze test files for weaknesses")
    parser.add_argument(
        "--output", "-o", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        default=Path("tests/optimizer_validation"),
        help="Directory to analyze",
    )
    parser.add_argument("--save", "-s", type=Path, help="Save report to file")
    args = parser.parse_args()

    results = analyze_directory(args.directory)
    report = generate_report(results, args.output)

    if args.save:
        args.save.write_text(report)
        print(f"Report saved to {args.save}")
    else:
        print(report)


if __name__ == "__main__":
    main()
