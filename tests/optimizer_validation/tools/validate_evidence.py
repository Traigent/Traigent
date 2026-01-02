#!/usr/bin/env python3
"""Validate test evidence against the JSON schema.

This tool checks that all tests emit complete evidence with required fields.
It can validate:
1. A pytest JSON report file
2. Live test output
3. The viewer's data file

Usage:
    python -m tests.optimizer_validation.tools.validate_evidence report.json
    python -m tests.optimizer_validation.tools.validate_evidence --viewer-data
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


@dataclass
class ValidationIssue:
    """A validation issue found in test evidence."""

    test_name: str
    section: str
    field: str
    issue_type: str  # "missing", "invalid_type", "invalid_value", "empty"
    message: str
    severity: str  # "error", "warning"


@dataclass
class ValidationReport:
    """Summary of validation results."""

    total_tests: int
    tests_with_evidence: int
    tests_without_evidence: int
    issues: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        """Check if all tests pass validation."""
        errors = [i for i in self.issues if i.severity == "error"]
        return len(errors) == 0 and self.tests_without_evidence == 0

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Total tests: {self.total_tests}",
            f"With evidence: {self.tests_with_evidence}",
            f"Without evidence: {self.tests_without_evidence}",
            f"Issues found: {len(self.issues)}",
        ]
        return "\n".join(lines)


# Required sections and their mandatory fields
REQUIRED_SECTIONS = {
    "scenario": {
        "required": ["name", "config_space", "injection_mode", "max_trials"],
        "recommended": ["description", "execution_mode", "objectives"],
    },
    "expected": {
        "required": ["outcome"],
        "recommended": ["min_trials", "max_trials"],
    },
    "actual": {
        "required": ["type"],
        "recommended": ["trial_count", "stop_reason"],
    },
    "validation_checks": {
        "required": [],  # Array must exist but items have their own requirements
        "recommended": [],
    },
}

# Validation check item requirements (supports both "name" and "check" field names)
VALIDATION_CHECK_REQUIRED = ["passed"]  # "name" or "check" is checked separately

SCHEMA_PATH = Path(__file__).parent.parent / "specs" / "evidence_schema.json"
_SCHEMA_CACHE: dict[str, Any] | None = None


def _load_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        with open(SCHEMA_PATH) as f:
            _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE


def _format_schema_path(path: list[Any]) -> tuple[str, str]:
    if not path:
        return "root", ""
    section = str(path[0])
    field = ""
    for part in path[1:]:
        if isinstance(part, int):
            field += f"[{part}]"
        else:
            field = f"{field}.{part}" if field else str(part)
    return section, field


def _validate_against_schema(
    evidence: dict[str, Any], test_name: str
) -> list[ValidationIssue]:
    schema = _load_schema()
    validator = Draft7Validator(schema)
    issues: list[ValidationIssue] = []

    for error in sorted(validator.iter_errors(evidence), key=lambda e: list(e.path)):
        section, field = _format_schema_path(list(error.path))
        issues.append(
            ValidationIssue(
                test_name=test_name,
                section=section,
                field=field,
                issue_type="schema",
                message=error.message,
                severity="error",
            )
        )
    return issues


def validate_config_space(config_space: Any, test_name: str) -> list[ValidationIssue]:
    """Validate config_space structure."""
    issues = []

    if not isinstance(config_space, dict):
        issues.append(
            ValidationIssue(
                test_name=test_name,
                section="scenario",
                field="config_space",
                issue_type="invalid_type",
                message=f"config_space must be object, got {type(config_space).__name__}",
                severity="error",
            )
        )
        return issues

    # Check for at least one parameter (excluding _summary)
    params = [k for k in config_space.keys() if k != "_summary"]
    if not params:
        issues.append(
            ValidationIssue(
                test_name=test_name,
                section="scenario",
                field="config_space",
                issue_type="empty",
                message="config_space has no parameters",
                severity="error",
            )
        )

    # Validate each parameter
    for param, spec in config_space.items():
        if param == "_summary":
            continue

        if not isinstance(spec, dict):
            issues.append(
                ValidationIssue(
                    test_name=test_name,
                    section="scenario",
                    field=f"config_space.{param}",
                    issue_type="invalid_type",
                    message=f"Parameter spec must be object, got {type(spec).__name__}",
                    severity="error",
                )
            )
            continue

        if "type" not in spec:
            issues.append(
                ValidationIssue(
                    test_name=test_name,
                    section="scenario",
                    field=f"config_space.{param}.type",
                    issue_type="missing",
                    message="Parameter missing 'type' field",
                    severity="error",
                )
            )

    return issues


def validate_validation_checks(checks: Any, test_name: str) -> list[ValidationIssue]:
    """Validate validation_checks array."""
    issues = []

    if not isinstance(checks, list):
        issues.append(
            ValidationIssue(
                test_name=test_name,
                section="validation_checks",
                field="",
                issue_type="invalid_type",
                message=f"validation_checks must be array, got {type(checks).__name__}",
                severity="error",
            )
        )
        return issues

    if len(checks) == 0:
        issues.append(
            ValidationIssue(
                test_name=test_name,
                section="validation_checks",
                field="",
                issue_type="empty",
                message="validation_checks is empty",
                severity="warning",
            )
        )

    for i, check in enumerate(checks):
        if not isinstance(check, dict):
            issues.append(
                ValidationIssue(
                    test_name=test_name,
                    section="validation_checks",
                    field=f"[{i}]",
                    issue_type="invalid_type",
                    message=f"Check must be object, got {type(check).__name__}",
                    severity="error",
                )
            )
            continue

        # Check for name identifier (can be "name" or "check")
        if "name" not in check and "check" not in check:
            issues.append(
                ValidationIssue(
                    test_name=test_name,
                    section="validation_checks",
                    field=f"[{i}]",
                    issue_type="missing",
                    message="Check missing identifier ('name' or 'check')",
                    severity="error",
                )
            )

        for field in VALIDATION_CHECK_REQUIRED:
            if field not in check:
                issues.append(
                    ValidationIssue(
                        test_name=test_name,
                        section="validation_checks",
                        field=f"[{i}].{field}",
                        issue_type="missing",
                        message=f"Check missing required field '{field}'",
                        severity="error",
                    )
                )

    return issues


def validate_evidence(
    evidence: dict[str, Any], test_name: str
) -> list[ValidationIssue]:
    """Validate a single test's evidence."""
    issues: list[ValidationIssue] = []
    issues.extend(_validate_against_schema(evidence, test_name))

    # Validate scenario section for recommended fields and config shape
    scenario = evidence.get("scenario", {})
    if isinstance(scenario, dict):
        if "config_space" in scenario:
            issues.extend(validate_config_space(scenario["config_space"], test_name))

        for field in REQUIRED_SECTIONS["scenario"]["recommended"]:
            if field not in scenario or scenario[field] is None:
                issues.append(
                    ValidationIssue(
                        test_name=test_name,
                        section="scenario",
                        field=field,
                        issue_type="missing",
                        message=f"Recommended field 'scenario.{field}' is missing",
                        severity="warning",
                    )
                )

    expected = evidence.get("expected", {})
    if isinstance(expected, dict):
        for field in REQUIRED_SECTIONS["expected"]["recommended"]:
            if field not in expected or expected[field] is None:
                issues.append(
                    ValidationIssue(
                        test_name=test_name,
                        section="expected",
                        field=field,
                        issue_type="missing",
                        message=f"Recommended field 'expected.{field}' is missing",
                        severity="warning",
                    )
                )

    actual = evidence.get("actual", {})
    if isinstance(actual, dict):
        for field in REQUIRED_SECTIONS["actual"]["recommended"]:
            if field not in actual or actual[field] is None:
                issues.append(
                    ValidationIssue(
                        test_name=test_name,
                        section="actual",
                        field=field,
                        issue_type="missing",
                        message=f"Recommended field 'actual.{field}' is missing",
                        severity="warning",
                    )
                )

    if "validation_checks" in evidence:
        issues.extend(
            validate_validation_checks(evidence["validation_checks"], test_name)
        )

    # Validate trials if present
    trials = evidence.get("trials", [])
    if isinstance(trials, list):
        for i, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            for field in ["index", "config", "status"]:
                if field not in trial:
                    issues.append(
                        ValidationIssue(
                            test_name=test_name,
                            section="trials",
                            field=f"[{i}].{field}",
                            issue_type="missing",
                            message=f"Trial missing required field '{field}'",
                            severity="error",
                        )
                    )

    return issues


def extract_evidence_from_report(report_path: Path) -> dict[str, dict[str, Any] | None]:
    """Extract evidence from pytest-json-report output."""
    tests: dict[str, dict[str, Any] | None] = {}

    with open(report_path) as f:
        report = json.load(f)

    for test in report.get("tests", []):
        nodeid = test.get("nodeid", "unknown")
        test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid

        # Look for evidence in captured output
        evidence = None
        call_data = test.get("call", {})
        stdout = call_data.get("stdout", "")

        for line in stdout.split("\n"):
            if "TEST_EVIDENCE" in line and line.strip().startswith("{"):
                try:
                    data = json.loads(line.strip())
                    if data.get("type") == "TEST_EVIDENCE":
                        evidence = data
                        break
                except json.JSONDecodeError:
                    continue

        tests[test_name] = evidence

    return tests


def extract_evidence_from_viewer_data(
    viewer_dir: Path,
) -> dict[str, dict[str, Any] | None]:
    """Extract evidence from viewer's data.js file."""
    tests: dict[str, dict[str, Any] | None] = {}

    data_file = viewer_dir / "data.js"
    if not data_file.exists():
        return tests

    content = data_file.read_text()

    # Parse the JavaScript data
    # Format: window.TEST_DATA = { ... }
    if "window.TEST_DATA" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(content[start:end])
                for _file_path, file_data in data.items():
                    if isinstance(file_data, dict):
                        for test_name, test_data in file_data.get("tests", {}).items():
                            result = test_data.get("result", {})
                            evidence = result.get("evidence")
                            tests[test_name] = evidence
            except json.JSONDecodeError:
                pass

    return tests


def validate_all_tests(
    tests: dict[str, dict[str, Any] | None],
) -> ValidationReport:
    """Validate all tests and generate a report."""
    issues: list[ValidationIssue] = []
    tests_with_evidence = 0
    tests_without_evidence = 0

    for test_name, evidence in tests.items():
        if evidence is None:
            tests_without_evidence += 1
            issues.append(
                ValidationIssue(
                    test_name=test_name,
                    section="root",
                    field="",
                    issue_type="missing",
                    message="No evidence captured for this test",
                    severity="error",
                )
            )
        else:
            tests_with_evidence += 1
            issues.extend(validate_evidence(evidence, test_name))

    return ValidationReport(
        total_tests=len(tests),
        tests_with_evidence=tests_with_evidence,
        tests_without_evidence=tests_without_evidence,
        issues=issues,
    )


def print_report(report: ValidationReport, verbose: bool = False) -> None:
    """Print validation report to stdout."""
    print("\n" + "=" * 60)
    print("TEST EVIDENCE VALIDATION REPORT")
    print("=" * 60)
    print(report.summary())
    print()

    if report.is_valid:
        print("✓ All tests have valid evidence")
    else:
        print("✗ Validation failed")

    # Group issues by test
    issues_by_test: dict[str, list[ValidationIssue]] = {}
    for issue in report.issues:
        if issue.test_name not in issues_by_test:
            issues_by_test[issue.test_name] = []
        issues_by_test[issue.test_name].append(issue)

    # Print errors first
    errors = [i for i in report.issues if i.severity == "error"]
    warnings = [i for i in report.issues if i.severity == "warning"]

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        print("-" * 40)
        for test_name, issues in sorted(issues_by_test.items()):
            test_errors = [i for i in issues if i.severity == "error"]
            if test_errors:
                print(f"\n  {test_name}:")
                for issue in test_errors:
                    location = (
                        f"{issue.section}.{issue.field}"
                        if issue.field
                        else issue.section
                    )
                    print(f"    • [{issue.issue_type}] {location}: {issue.message}")

    if warnings and verbose:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        print("-" * 40)
        for test_name, issues in sorted(issues_by_test.items()):
            test_warnings = [i for i in issues if i.severity == "warning"]
            if test_warnings:
                print(f"\n  {test_name}:")
                for issue in test_warnings:
                    location = (
                        f"{issue.section}.{issue.field}"
                        if issue.field
                        else issue.section
                    )
                    print(f"    • {location}: {issue.message}")

    print("\n" + "=" * 60)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate test evidence against schema"
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to pytest JSON report file",
    )
    parser.add_argument(
        "--viewer-data",
        action="store_true",
        help="Validate from viewer data.js instead of report",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show warnings in addition to errors",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text",
    )

    args = parser.parse_args()

    # Determine source
    if args.viewer_data:
        viewer_dir = Path(__file__).parent.parent / "viewer"
        tests = extract_evidence_from_viewer_data(viewer_dir)
        if not tests:
            print("No tests found in viewer data")
            return 1
    elif args.report:
        report_path = Path(args.report)
        if not report_path.exists():
            print(f"Report file not found: {report_path}")
            return 1
        tests = extract_evidence_from_report(report_path)
    else:
        parser.print_help()
        return 1

    # Validate
    report = validate_all_tests(tests)

    # Output
    if args.json:
        output = {
            "summary": {
                "total_tests": report.total_tests,
                "tests_with_evidence": report.tests_with_evidence,
                "tests_without_evidence": report.tests_without_evidence,
                "is_valid": report.is_valid,
            },
            "issues": [
                {
                    "test_name": i.test_name,
                    "section": i.section,
                    "field": i.field,
                    "issue_type": i.issue_type,
                    "message": i.message,
                    "severity": i.severity,
                }
                for i in report.issues
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
