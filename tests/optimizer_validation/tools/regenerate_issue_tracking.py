#!/usr/bin/env python3
"""Regenerate issue_tracking.json from current codebase state.

This script scans all test files and generates a fresh issue tracking
file based on the current state of assertions and patterns.

Usage:
    python -m tests.optimizer_validation.tools.regenerate_issue_tracking
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .test_weakness_analyzer import (
    IssueType,
    RootCause,
    Severity,
    analyze_directory,
)


def regenerate_issue_tracking(output_path: Path | None = None) -> dict:
    """Regenerate issue tracking from current codebase."""
    test_dir = Path("tests/optimizer_validation")
    results = analyze_directory(test_dir)

    # Build issue tracking structure
    tracking = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_issues": 0,
            "files_analyzed": len(results),
            "issue_types": {
                it.value: it.name.replace("_", " ").title() for it in IssueType
            },
            "root_causes": {
                rc.value: rc.name.replace("_", " ").title() for rc in RootCause
            },
            "fixes_applied_count": 0,
            "issues_remaining": 0,
        },
        "summary": {
            "by_issue_type": {},
            "by_root_cause": {},
            "by_severity": {},
            "by_file": {},
        },
        "issues": [],
    }

    # Process all issues
    for result in results:
        file_name = Path(result.file_path).name

        for issue in result.issues:
            issue_entry = {
                "issue_id": f"ISS-{len(tracking['issues']) + 1:04d}",
                "test_id": f"{result.file_path}::{issue.test_name}",
                "issue_type": issue.issue_type.value,
                "root_cause": issue.root_cause.value,
                "severity": issue.severity.value,
                "file": result.file_path,
                "test_name": issue.test_name,
                "line_number": issue.line_number,
                "description": issue.description,
                "evidence": issue.evidence,
                "suggested_fix": issue.suggested_fix,
                "status": "open",
            }
            tracking["issues"].append(issue_entry)

            # Update summaries
            it = issue.issue_type.value
            if it not in tracking["summary"]["by_issue_type"]:
                tracking["summary"]["by_issue_type"][it] = {"count": 0, "tests": []}
            tracking["summary"]["by_issue_type"][it]["count"] += 1
            tracking["summary"]["by_issue_type"][it]["tests"].append(
                issue_entry["test_id"]
            )

            rc = issue.root_cause.value
            if rc not in tracking["summary"]["by_root_cause"]:
                tracking["summary"]["by_root_cause"][rc] = {"count": 0, "tests": []}
            tracking["summary"]["by_root_cause"][rc]["count"] += 1
            tracking["summary"]["by_root_cause"][rc]["tests"].append(
                issue_entry["test_id"]
            )

            sev = issue.severity.value
            if sev not in tracking["summary"]["by_severity"]:
                tracking["summary"]["by_severity"][sev] = {"count": 0}
            tracking["summary"]["by_severity"][sev]["count"] += 1

            if file_name not in tracking["summary"]["by_file"]:
                tracking["summary"]["by_file"][file_name] = {"count": 0, "tests": []}
            tracking["summary"]["by_file"][file_name]["count"] += 1
            tracking["summary"]["by_file"][file_name]["tests"].append(issue.test_name)

    # Update totals
    tracking["metadata"]["total_issues"] = len(tracking["issues"])
    tracking["metadata"]["issues_remaining"] = len(tracking["issues"])

    # Add percentages
    total = len(tracking["issues"])
    for it_data in tracking["summary"]["by_issue_type"].values():
        it_data["percentage"] = (
            round(it_data["count"] / total * 100, 1) if total > 0 else 0
        )

    if output_path:
        output_path.write_text(json.dumps(tracking, indent=2))

    return tracking


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate issue tracking")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tests/optimizer_validation/issue_tracking.json"),
        help="Output file path",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print to stdout instead of writing file",
    )
    args = parser.parse_args()

    tracking = regenerate_issue_tracking(None if args.dry_run else args.output)

    if args.dry_run:
        print(json.dumps(tracking, indent=2))
    else:
        print(f"Regenerated issue tracking: {args.output}")
        print(f"Total issues: {tracking['metadata']['total_issues']}")
        print("By issue type:")
        for it, data in tracking["summary"]["by_issue_type"].items():
            print(f"  {it}: {data['count']} ({data['percentage']}%)")


if __name__ == "__main__":
    main()
