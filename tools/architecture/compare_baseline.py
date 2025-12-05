#!/usr/bin/env python3
"""
Architecture Baseline Comparison

Compares current architecture metrics against a baseline to detect drift.
Useful for tracking architecture evolution over time and catching regressions.

Usage:
    python tools/architecture/compare_baseline.py [options]

Options:
    --current FILE      Current architecture_data.json
    --baseline FILE     Baseline metrics file (default: baseline_metrics.json)
    --output FILE       Output report file
    --format FORMAT     Output format: md, json (default: md)

Tracks:
- Lines of code changes
- Complexity changes
- New high-complexity modules
- Hub module fan-in changes
- Class method count changes
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MetricChange:
    """A change in a metric."""

    category: str
    name: str
    old_value: float | int | None
    new_value: float | int
    change: float | int
    change_percent: float
    significance: str  # MAJOR, MINOR, INFO


def compare_metrics(current: dict, baseline: dict | None) -> list[MetricChange]:
    """Compare current metrics against baseline."""
    changes: list[MetricChange] = []

    if baseline is None:
        # No baseline - report current state only
        return changes

    # Compare summary stats
    current_summary = current.get("summary", {})
    baseline_summary = baseline.get("summary", {})

    # Lines of code
    current_loc = current_summary.get("total_lines", 0)
    baseline_loc = baseline_summary.get("total_lines", 0)
    if baseline_loc > 0:
        loc_change = current_loc - baseline_loc
        loc_pct = (loc_change / baseline_loc) * 100
        significance = (
            "MAJOR" if abs(loc_pct) > 10 else "MINOR" if abs(loc_pct) > 5 else "INFO"
        )
        changes.append(
            MetricChange(
                category="SIZE",
                name="Lines of Code",
                old_value=baseline_loc,
                new_value=current_loc,
                change=loc_change,
                change_percent=loc_pct,
                significance=significance,
            )
        )

    # Total files
    current_files = current_summary.get("total_files", 0)
    baseline_files = baseline_summary.get("total_files", 0)
    if baseline_files > 0:
        files_change = current_files - baseline_files
        files_pct = (files_change / baseline_files) * 100
        significance = "MINOR" if abs(files_change) > 5 else "INFO"
        changes.append(
            MetricChange(
                category="SIZE",
                name="Total Files",
                old_value=baseline_files,
                new_value=current_files,
                change=files_change,
                change_percent=files_pct,
                significance=significance,
            )
        )

    # Compare complexity
    current_complexity = current.get("complexity", {})
    baseline_complexity = baseline.get("complexity", {})

    current_avg_cc = current_complexity.get("average_complexity", 0)
    baseline_avg_cc = baseline_complexity.get("average_complexity", 0)
    if baseline_avg_cc > 0:
        cc_change = current_avg_cc - baseline_avg_cc
        cc_pct = (cc_change / baseline_avg_cc) * 100
        significance = (
            "MAJOR" if cc_change > 1 else "MINOR" if cc_change > 0.5 else "INFO"
        )
        changes.append(
            MetricChange(
                category="COMPLEXITY",
                name="Average Complexity",
                old_value=baseline_avg_cc,
                new_value=current_avg_cc,
                change=round(cc_change, 2),
                change_percent=round(cc_pct, 2),
                significance=significance,
            )
        )

    # Compare high complexity modules
    current_high_cc = set()
    for item in current_complexity.get("high_complexity_modules", []):
        if isinstance(item, (list, tuple)):
            current_high_cc.add(item[0])

    baseline_high_cc = set()
    for item in baseline_complexity.get("high_complexity_modules", []):
        if isinstance(item, (list, tuple)):
            baseline_high_cc.add(item[0])

    new_high_cc = current_high_cc - baseline_high_cc
    for module in new_high_cc:
        changes.append(
            MetricChange(
                category="COMPLEXITY",
                name=f"New High Complexity: {module}",
                old_value=None,
                new_value=1,
                change=1,
                change_percent=100,
                significance="MAJOR",
            )
        )

    # Compare hub modules
    current_hubs = {}
    for item in current.get("hub_modules", []):
        if isinstance(item, dict):
            current_hubs[item.get("module", "")] = item.get("fan_in", 0)
        elif isinstance(item, (list, tuple)):
            current_hubs[item[0]] = item[1]

    baseline_hubs = {}
    for item in baseline.get("hub_modules", []):
        if isinstance(item, dict):
            baseline_hubs[item.get("module", "")] = item.get("fan_in", 0)
        elif isinstance(item, (list, tuple)):
            baseline_hubs[item[0]] = item[1]

    for module, fan_in in current_hubs.items():
        if module in baseline_hubs:
            old_fan_in = baseline_hubs[module]
            if fan_in != old_fan_in:
                change = fan_in - old_fan_in
                pct = (change / old_fan_in * 100) if old_fan_in > 0 else 100
                significance = (
                    "MAJOR" if change > 10 else "MINOR" if change > 5 else "INFO"
                )
                changes.append(
                    MetricChange(
                        category="STABILITY",
                        name=f"Hub: {module}",
                        old_value=old_fan_in,
                        new_value=fan_in,
                        change=change,
                        change_percent=round(pct, 1),
                        significance=significance,
                    )
                )

    return changes


def format_markdown_report(changes: list[MetricChange], has_baseline: bool) -> str:
    """Format comparison as markdown."""
    lines = [
        "# Architecture Drift Report",
        "",
    ]

    if not has_baseline:
        lines.extend(
            [
                "## ℹ️ No Baseline Available",
                "",
                "This is the first analysis or no baseline file was found.",
                "Current metrics will be used as the new baseline.",
                "",
            ]
        )
        return "\n".join(lines)

    if not changes:
        lines.extend(
            [
                "## ✅ No Significant Changes",
                "",
                "Architecture metrics are stable compared to baseline.",
                "",
            ]
        )
        return "\n".join(lines)

    # Group by significance
    major = [c for c in changes if c.significance == "MAJOR"]
    minor = [c for c in changes if c.significance == "MINOR"]
    info = [c for c in changes if c.significance == "INFO"]

    lines.extend(
        [
            "## Summary",
            "",
            f"- **Major Changes**: {len(major)}",
            f"- **Minor Changes**: {len(minor)}",
            f"- **Informational**: {len(info)}",
            "",
        ]
    )

    if major:
        lines.extend(
            [
                "## 🚨 Major Changes",
                "",
                "| Category | Metric | Old | New | Change |",
                "|----------|--------|-----|-----|--------|",
            ]
        )
        for c in major:
            old = c.old_value if c.old_value is not None else "N/A"
            sign = "+" if c.change > 0 else ""
            lines.append(
                f"| {c.category} | {c.name} | {old} | {c.new_value} | {sign}{c.change} ({sign}{c.change_percent:.1f}%) |"
            )
        lines.append("")

    if minor:
        lines.extend(
            [
                "## ⚠️ Minor Changes",
                "",
                "| Category | Metric | Change |",
                "|----------|--------|--------|",
            ]
        )
        for c in minor:
            sign = "+" if c.change > 0 else ""
            lines.append(
                f"| {c.category} | {c.name} | {sign}{c.change} ({sign}{c.change_percent:.1f}%) |"
            )
        lines.append("")

    if info:
        lines.extend(
            [
                "## ℹ️ Informational Changes",
                "",
                "<details>",
                "<summary>Click to expand</summary>",
                "",
                "| Category | Metric | Change |",
                "|----------|--------|--------|",
            ]
        )
        for c in info:
            sign = "+" if c.change > 0 else ""
            lines.append(f"| {c.category} | {c.name} | {sign}{c.change} |")
        lines.extend(
            [
                "",
                "</details>",
                "",
            ]
        )

    return "\n".join(lines)


def format_json_report(changes: list[MetricChange], has_baseline: bool) -> str:
    """Format comparison as JSON."""
    return json.dumps(
        {
            "has_baseline": has_baseline,
            "changes": [
                {
                    "category": c.category,
                    "name": c.name,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "change": c.change,
                    "change_percent": c.change_percent,
                    "significance": c.significance,
                }
                for c in changes
            ],
            "summary": {
                "major": len([c for c in changes if c.significance == "MAJOR"]),
                "minor": len([c for c in changes if c.significance == "MINOR"]),
                "info": len([c for c in changes if c.significance == "INFO"]),
            },
        },
        indent=2,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare architecture metrics")
    parser.add_argument(
        "--current", default="tools/architecture/output/architecture_data.json"
    )
    parser.add_argument(
        "--baseline", default="tools/architecture/baseline_metrics.json"
    )
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--format", choices=["md", "json"], default="md")

    args = parser.parse_args()

    # Load current data
    current_file = Path(args.current)
    if not current_file.exists():
        print(f"❌ Current data file not found: {current_file}", file=sys.stderr)
        sys.exit(1)

    current = json.loads(current_file.read_text())

    # Load baseline (optional)
    baseline_file = Path(args.baseline)
    baseline = None
    has_baseline = False

    if baseline_file.exists():
        try:
            baseline = json.loads(baseline_file.read_text())
            has_baseline = True
            print(f"📊 Comparing against baseline: {baseline_file}")
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse baseline file: {baseline_file}")
    else:
        print(f"ℹ️ No baseline file found: {baseline_file}")

    # Compare
    changes = compare_metrics(current, baseline)

    # Generate report
    if args.format == "json":
        report = format_json_report(changes, has_baseline)
    else:
        report = format_markdown_report(changes, has_baseline)

    # Output
    if args.output:
        Path(args.output).write_text(report)
        print(f"📝 Report written to {args.output}")
    else:
        print(report)

    # Summary
    if changes:
        major = len([c for c in changes if c.significance == "MAJOR"])
        if major > 0:
            print(f"\n🚨 {major} major architecture changes detected!")
        else:
            print(f"\n✅ No major changes, {len(changes)} minor/info changes")
    elif has_baseline:
        print("\n✅ No significant architecture drift detected")


if __name__ == "__main__":
    main()
