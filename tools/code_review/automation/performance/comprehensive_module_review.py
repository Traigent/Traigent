#!/usr/bin/env python3
"""
Comprehensive manual-style review of ALL 179 performance review modules.
Reviews each module's issues, checks, and functions with detailed analysis.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class ModuleReviewSummary:
    """Summary of module review."""

    module: str
    total_functions: int
    functions_ok: int
    functions_needs_followup: int
    total_checks: int
    checks_pass: int
    checks_fail: int
    checks_needs_followup: int
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    review_status: str  # clean, minor_issues, major_issues, critical_issues
    key_concerns: List[str]
    manual_review_notes: str


def analyze_function_status(functions: List[Dict]) -> Dict[str, int]:
    """Analyze function review status."""
    status_counts = {"ok": 0, "needs_followup": 0, "other": 0}
    for func in functions:
        status = func.get("status", "unknown")
        if status == "ok":
            status_counts["ok"] += 1
        elif status == "needs_followup":
            status_counts["needs_followup"] += 1
        else:
            status_counts["other"] += 1
    return status_counts


def analyze_checks(checks: List[Dict]) -> Dict[str, int]:
    """Analyze check results."""
    result_counts = {"pass": 0, "fail": 0, "needs_followup": 0, "other": 0}
    for check in checks:
        result = check.get("result", "unknown")
        if result == "pass":
            result_counts["pass"] += 1
        elif result == "fail":
            result_counts["fail"] += 1
        elif result == "needs_followup":
            result_counts["needs_followup"] += 1
        else:
            result_counts["other"] += 1
    return result_counts


def analyze_issues(issues: List[Dict]) -> Dict[str, int]:
    """Analyze issue severity."""
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in issues:
        severity = issue.get("severity", "unknown").lower()
        if severity in severity_counts:
            severity_counts[severity] += 1
    return severity_counts


def determine_review_status(checks: Dict[str, int], issues: Dict[str, int]) -> str:
    """Determine overall review status."""
    if issues["critical"] > 0:
        return "critical_issues"
    elif issues["high"] > 0 or checks["fail"] > 0:
        return "major_issues"
    elif issues["medium"] > 0 or checks["needs_followup"] > 2:
        return "minor_issues"
    else:
        return "clean"


def extract_key_concerns(review: Dict) -> List[str]:
    """Extract key concerns from review."""
    concerns = []

    # Check for failed checks
    for check in review.get("checks", []):
        if check.get("result") == "fail":
            concerns.append(f"FAIL: {check.get('name', 'unknown')}")

    # Check for critical/high issues
    for issue in review.get("issues", []):
        severity = issue.get("severity", "").lower()
        if severity in ["critical", "high"]:
            concerns.append(f"{severity.upper()}: {issue.get('title', 'unknown')}")

    return concerns[:5]  # Top 5 concerns


def review_module(review_path: Path) -> ModuleReviewSummary:
    """Perform comprehensive review of a single module."""
    with open(review_path, "r") as f:
        review = json.load(f)

    module = review.get("module", "unknown")
    functions = review.get("functions", [])
    checks = review.get("checks", [])
    issues = review.get("issues", [])

    # Analyze components
    func_status = analyze_function_status(functions)
    check_results = analyze_checks(checks)
    issue_severity = analyze_issues(issues)

    # Determine status
    review_status = determine_review_status(check_results, issue_severity)
    key_concerns = extract_key_concerns(review)

    # Generate manual review notes
    notes_parts = []

    # Summary note
    if review_status == "clean":
        notes_parts.append("✅ Module is clean with no significant performance issues.")
    elif review_status == "minor_issues":
        notes_parts.append(
            "⚠️ Module has minor performance concerns that should be addressed."
        )
    elif review_status == "major_issues":
        notes_parts.append(
            "🔶 Module has major performance issues requiring attention."
        )
    elif review_status == "critical_issues":
        notes_parts.append(
            "🚨 Module has CRITICAL performance issues requiring immediate action!"
        )

    # Add specific notes
    if func_status["needs_followup"] > 0:
        notes_parts.append(
            f"{func_status['needs_followup']} functions need performance review."
        )

    if check_results["fail"] > 0:
        notes_parts.append(f"{check_results['fail']} performance checks FAILED.")

    if issue_severity["critical"] > 0:
        notes_parts.append(f"⚠️ {issue_severity['critical']} CRITICAL issues found!")

    if issue_severity["high"] > 0:
        notes_parts.append(f"{issue_severity['high']} HIGH priority issues identified.")

    manual_review_notes = " ".join(notes_parts)

    return ModuleReviewSummary(
        module=module,
        total_functions=len(functions),
        functions_ok=func_status["ok"],
        functions_needs_followup=func_status["needs_followup"],
        total_checks=len(checks),
        checks_pass=check_results["pass"],
        checks_fail=check_results["fail"],
        checks_needs_followup=check_results["needs_followup"],
        total_issues=len(issues),
        critical_issues=issue_severity["critical"],
        high_issues=issue_severity["high"],
        medium_issues=issue_severity["medium"],
        low_issues=issue_severity["low"],
        review_status=review_status,
        key_concerns=key_concerns,
        manual_review_notes=manual_review_notes,
    )


def main():
    """Review all 179 modules."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    review_dir = project_root / "out" / "performance"

    print("=" * 100)
    print("COMPREHENSIVE PERFORMANCE REVIEW - ALL 179 MODULES")
    print("=" * 100)
    print()

    # Find all review files
    review_files = sorted(review_dir.glob("**/*.review.json"))

    all_summaries = []

    # Category counters
    critical_modules = []
    major_issues_modules = []
    minor_issues_modules = []
    clean_modules = []

    print(f"Reviewing {len(review_files)} modules...\n")

    for i, review_file in enumerate(review_files, 1):
        try:
            summary = review_module(review_file)
            all_summaries.append(summary)

            # Categorize
            if summary.review_status == "critical_issues":
                critical_modules.append(summary)
            elif summary.review_status == "major_issues":
                major_issues_modules.append(summary)
            elif summary.review_status == "minor_issues":
                minor_issues_modules.append(summary)
            else:
                clean_modules.append(summary)

            # Print progress every 20 modules
            if i % 20 == 0:
                print(f"Progress: {i}/{len(review_files)} modules reviewed...")

        except Exception as e:
            print(f"❌ Error reviewing {review_file}: {e}")
            continue

    print(f"\n✅ Completed review of {len(all_summaries)} modules\n")

    # Generate comprehensive report
    print("=" * 100)
    print("REVIEW SUMMARY BY CATEGORY")
    print("=" * 100)
    print()

    # Critical Issues
    if critical_modules:
        print(f"🚨 CRITICAL ISSUES ({len(critical_modules)} modules)")
        print("-" * 100)
        for summary in critical_modules:
            print(f"\n📍 {summary.module}")
            print(
                f"   Issues: {summary.critical_issues} critical, {summary.high_issues} high, "
                f"{summary.medium_issues} medium, {summary.low_issues} low"
            )
            print(f"   Notes: {summary.manual_review_notes}")
            if summary.key_concerns:
                print(f"   Key Concerns:")
                for concern in summary.key_concerns:
                    print(f"      • {concern}")
        print()

    # Major Issues
    if major_issues_modules:
        print(f"\n🔶 MAJOR ISSUES ({len(major_issues_modules)} modules)")
        print("-" * 100)
        for summary in major_issues_modules:
            print(f"\n📍 {summary.module}")
            print(
                f"   Issues: {summary.high_issues} high, {summary.medium_issues} medium, {summary.low_issues} low"
            )
            print(
                f"   Checks: {summary.checks_fail} failed, {summary.checks_needs_followup} need followup"
            )
            if summary.key_concerns:
                print(f"   Key Concerns:")
                for concern in summary.key_concerns[:3]:  # Top 3
                    print(f"      • {concern}")
        print()

    # Minor Issues
    print(f"\n⚠️  MINOR ISSUES ({len(minor_issues_modules)} modules)")
    print("-" * 100)
    for summary in minor_issues_modules[:10]:  # Show first 10
        print(
            f"   • {summary.module}: {summary.medium_issues}M/{summary.low_issues}L issues"
        )
    if len(minor_issues_modules) > 10:
        print(f"   ... and {len(minor_issues_modules) - 10} more")
    print()

    # Clean Modules
    print(f"\n✅ CLEAN MODULES ({len(clean_modules)} modules)")
    print("-" * 100)
    print(f"   {len(clean_modules)} modules with no significant performance issues")
    print()

    # Overall Statistics
    print("\n" + "=" * 100)
    print("OVERALL STATISTICS")
    print("=" * 100)

    total_issues = sum(s.total_issues for s in all_summaries)
    total_critical = sum(s.critical_issues for s in all_summaries)
    total_high = sum(s.high_issues for s in all_summaries)
    total_medium = sum(s.medium_issues for s in all_summaries)
    total_low = sum(s.low_issues for s in all_summaries)

    total_checks = sum(s.total_checks for s in all_summaries)
    total_checks_fail = sum(s.checks_fail for s in all_summaries)

    print(f"\n📊 Modules Reviewed: {len(all_summaries)}")
    print(f"   • Critical Issues: {len(critical_modules)} modules")
    print(f"   • Major Issues: {len(major_issues_modules)} modules")
    print(f"   • Minor Issues: {len(minor_issues_modules)} modules")
    print(
        f"   • Clean: {len(clean_modules)} modules ({len(clean_modules)/len(all_summaries)*100:.1f}%)"
    )

    print(f"\n📋 Total Issues Found: {total_issues}")
    print(
        f"   • Critical: {total_critical} ({total_critical/total_issues*100:.1f}%)"
        if total_issues > 0
        else "   • Critical: 0"
    )
    print(
        f"   • High: {total_high} ({total_high/total_issues*100:.1f}%)"
        if total_issues > 0
        else "   • High: 0"
    )
    print(
        f"   • Medium: {total_medium} ({total_medium/total_issues*100:.1f}%)"
        if total_issues > 0
        else "   • Medium: 0"
    )
    print(
        f"   • Low: {total_low} ({total_low/total_issues*100:.1f}%)"
        if total_issues > 0
        else "   • Low: 0"
    )

    print(f"\n🔍 Performance Checks: {total_checks} total")
    print(f"   • Failed: {total_checks_fail}")
    print(
        f"   • Pass Rate: {(1 - total_checks_fail/total_checks)*100:.1f}%"
        if total_checks > 0
        else "   • Pass Rate: N/A"
    )

    # Save detailed results
    output_file = (
        project_root
        / "reports"
        / "development"
        / "code_debt"
        / "comprehensive_module_review.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "summary": {
            "total_modules": len(all_summaries),
            "critical_modules": len(critical_modules),
            "major_issues_modules": len(major_issues_modules),
            "minor_issues_modules": len(minor_issues_modules),
            "clean_modules": len(clean_modules),
            "total_issues": total_issues,
            "total_critical": total_critical,
            "total_high": total_high,
            "total_medium": total_medium,
            "total_low": total_low,
            "total_checks": total_checks,
            "total_checks_fail": total_checks_fail,
        },
        "modules": [asdict(s) for s in all_summaries],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Detailed results saved to: {output_file}")

    # Save markdown report
    md_file = (
        project_root
        / "reports"
        / "development"
        / "code_debt"
        / "COMPREHENSIVE_MODULE_REVIEW.md"
    )

    with open(md_file, "w") as f:
        f.write("# Comprehensive Performance Review - All 179 Modules\n\n")
        f.write(f"**Date**: 2025-10-13\n")
        f.write(f"**Modules Reviewed**: {len(all_summaries)}/179\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Modules**: {len(all_summaries)}\n")
        f.write(f"- **Critical Issues**: {len(critical_modules)} modules\n")
        f.write(f"- **Major Issues**: {len(major_issues_modules)} modules\n")
        f.write(f"- **Minor Issues**: {len(minor_issues_modules)} modules\n")
        f.write(
            f"- **Clean Modules**: {len(clean_modules)} modules ({len(clean_modules)/len(all_summaries)*100:.1f}%)\n\n"
        )

        f.write("## Critical Issues Modules\n\n")
        for summary in critical_modules:
            f.write(f"### {summary.module}\n\n")
            f.write(f"**Status**: 🚨 CRITICAL\n\n")
            f.write(
                f"- **Issues**: {summary.critical_issues} critical, {summary.high_issues} high, "
                f"{summary.medium_issues} medium, {summary.low_issues} low\n"
            )
            f.write(f"- **Notes**: {summary.manual_review_notes}\n\n")
            if summary.key_concerns:
                f.write("**Key Concerns**:\n")
                for concern in summary.key_concerns:
                    f.write(f"- {concern}\n")
            f.write("\n---\n\n")

        f.write("## Major Issues Modules\n\n")
        for summary in major_issues_modules:
            f.write(f"### {summary.module}\n\n")
            f.write(
                f"- **Issues**: {summary.high_issues}H / {summary.medium_issues}M / {summary.low_issues}L\n"
            )
            f.write(f"- **Failed Checks**: {summary.checks_fail}\n")
            if summary.key_concerns:
                f.write(f"- **Top Concerns**: {', '.join(summary.key_concerns[:2])}\n")
            f.write("\n")

        f.write("\n## Statistics\n\n")
        f.write(f"- Total Issues: {total_issues}\n")
        f.write(f"- Critical: {total_critical}\n")
        f.write(f"- High: {total_high}\n")
        f.write(f"- Medium: {total_medium}\n")
        f.write(f"- Low: {total_low}\n")

    print(f"✅ Markdown report saved to: {md_file}\n")

    print("=" * 100)
    print("REVIEW COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
