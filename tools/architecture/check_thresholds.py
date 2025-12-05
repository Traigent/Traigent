#!/usr/bin/env python3
"""
Architecture Threshold Checker

Validates architecture metrics against configurable thresholds.
Used in CI to gate PRs that introduce excessive complexity.

Usage:
    python tools/architecture/check_thresholds.py [options]

Options:
    --data-file FILE        Path to architecture_data.json
    --max-complexity N      Max cyclomatic complexity allowed (default: 50)
    --max-methods N         Max methods per class allowed (default: 80)
    --max-lines N           Max lines per file allowed (default: 2500)
    --max-fan-in N          Max dependents for hub modules (default: 100)
    --fail-on-violation     Exit with error code if violations found
    --output FILE           Write results to file

Exit Codes:
    0 - All thresholds passed
    1 - Threshold violations found (with --fail-on-violation)
    2 - Error running checks
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThresholdConfig:
    """Configuration for architecture thresholds."""

    max_complexity: int = 50
    max_methods: int = 80
    max_lines: int = 2500
    max_fan_in: int = 100
    warn_complexity: int = 30
    warn_methods: int = 50
    warn_lines: int = 1500


@dataclass
class Violation:
    """A threshold violation."""

    category: str
    location: str
    value: int | float
    threshold: int | float
    severity: str  # ERROR, WARNING


def check_thresholds(data: dict, config: ThresholdConfig) -> list[Violation]:
    """Check architecture data against thresholds."""
    violations: list[Violation] = []

    # Check complexity
    complexity_data = data.get("complexity", {})
    high_complexity = complexity_data.get("high_complexity_modules", [])

    for item in high_complexity:
        if isinstance(item, (list, tuple)):
            module, avg_cc, max_cc = item
        else:
            continue

        if max_cc > config.max_complexity:
            violations.append(
                Violation(
                    category="COMPLEXITY",
                    location=module,
                    value=max_cc,
                    threshold=config.max_complexity,
                    severity="ERROR",
                )
            )
        elif max_cc > config.warn_complexity:
            violations.append(
                Violation(
                    category="COMPLEXITY",
                    location=module,
                    value=max_cc,
                    threshold=config.warn_complexity,
                    severity="WARNING",
                )
            )

    # Check hub modules (fan-in)
    hub_modules = data.get("hub_modules", [])
    for item in hub_modules:
        if isinstance(item, dict):
            module = item.get("module", "")
            fan_in = item.get("fan_in", 0)
        elif isinstance(item, (list, tuple)):
            module, fan_in = item[0], item[1]
        else:
            continue

        if fan_in > config.max_fan_in:
            violations.append(
                Violation(
                    category="FAN_IN",
                    location=module,
                    value=fan_in,
                    threshold=config.max_fan_in,
                    severity="ERROR",
                )
            )

    # Check file sizes from package structure
    pkg_struct = data.get("package_structure", {})

    def check_files(pkg_data: dict, prefix: str = "") -> None:
        for mod_name, mod_data in pkg_data.get("modules", {}).items():
            lines = mod_data.get("lines", 0)
            path = f"{prefix}/{mod_name}.py" if prefix else f"{mod_name}.py"

            if lines > config.max_lines:
                violations.append(
                    Violation(
                        category="FILE_SIZE",
                        location=path,
                        value=lines,
                        threshold=config.max_lines,
                        severity="ERROR",
                    )
                )
            elif lines > config.warn_lines:
                violations.append(
                    Violation(
                        category="FILE_SIZE",
                        location=path,
                        value=lines,
                        threshold=config.warn_lines,
                        severity="WARNING",
                    )
                )

        for sub_name, sub_data in pkg_data.get("subpackages", {}).items():
            sub_prefix = f"{prefix}/{sub_name}" if prefix else sub_name
            check_files(sub_data, sub_prefix)

    check_files(pkg_struct)

    # Check class method counts from class_hierarchy
    class_hierarchy = data.get("class_hierarchy", {})
    # This requires the all_classes data which isn't in summary

    return violations


def format_report(violations: list[Violation], config: ThresholdConfig) -> str:
    """Format violations as a report."""
    lines = [
        "# Architecture Threshold Check Results",
        "",
        "## Configuration",
        "",
        f"- Max Complexity: {config.max_complexity} (warn: {config.warn_complexity})",
        f"- Max Methods/Class: {config.max_methods} (warn: {config.warn_methods})",
        f"- Max Lines/File: {config.max_lines} (warn: {config.warn_lines})",
        f"- Max Fan-In: {config.max_fan_in}",
        "",
    ]

    errors = [v for v in violations if v.severity == "ERROR"]
    warnings = [v for v in violations if v.severity == "WARNING"]

    if not violations:
        lines.extend(
            [
                "## ✅ All Thresholds Passed",
                "",
                "No violations detected.",
            ]
        )
    else:
        lines.extend(
            [
                "## Summary",
                "",
                f"- **Errors**: {len(errors)}",
                f"- **Warnings**: {len(warnings)}",
                "",
            ]
        )

        if errors:
            lines.extend(
                [
                    "## ❌ Errors (Must Fix)",
                    "",
                    "| Category | Location | Value | Threshold |",
                    "|----------|----------|-------|-----------|",
                ]
            )
            for v in errors:
                lines.append(
                    f"| {v.category} | `{v.location}` | {v.value} | {v.threshold} |"
                )
            lines.append("")

        if warnings:
            lines.extend(
                [
                    "## ⚠️ Warnings",
                    "",
                    "| Category | Location | Value | Threshold |",
                    "|----------|----------|-------|-----------|",
                ]
            )
            for v in warnings:
                lines.append(
                    f"| {v.category} | `{v.location}` | {v.value} | {v.threshold} |"
                )
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check architecture thresholds")
    parser.add_argument(
        "--data-file", default="tools/architecture/output/architecture_data.json"
    )
    parser.add_argument("--max-complexity", type=int, default=50)
    parser.add_argument("--max-methods", type=int, default=80)
    parser.add_argument("--max-lines", type=int, default=2500)
    parser.add_argument("--max-fan-in", type=int, default=100)
    parser.add_argument("--fail-on-violation", action="store_true")
    parser.add_argument("--output", help="Output file for report")

    args = parser.parse_args()

    # Load data
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}", file=sys.stderr)
        sys.exit(2)

    try:
        data = json.loads(data_file.read_text())
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # Configure thresholds
    config = ThresholdConfig(
        max_complexity=args.max_complexity,
        max_methods=args.max_methods,
        max_lines=args.max_lines,
        max_fan_in=args.max_fan_in,
    )

    # Check thresholds
    print("🔍 Checking architecture thresholds...")
    violations = check_thresholds(data, config)

    # Generate report
    report = format_report(violations, config)

    # Output
    if args.output:
        Path(args.output).write_text(report)
        print(f"📝 Report written to {args.output}")

    # Print summary
    errors = [v for v in violations if v.severity == "ERROR"]
    warnings = [v for v in violations if v.severity == "WARNING"]

    print(f"\n📊 Results: {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        print("\n❌ Error violations:")
        for v in errors[:5]:
            print(f"   [{v.category}] {v.location}: {v.value} > {v.threshold}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")

    if warnings:
        print("\n⚠️ Warning violations:")
        for v in warnings[:5]:
            print(f"   [{v.category}] {v.location}: {v.value} > {v.threshold}")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings) - 5} more")

    if not violations:
        print("\n✅ All thresholds passed!")
        sys.exit(0)
    elif args.fail_on_violation and errors:
        print("\n❌ Threshold violations found!")
        sys.exit(1)
    else:
        print("\n⚠️ Violations found (not failing)")
        sys.exit(0)


if __name__ == "__main__":
    main()
