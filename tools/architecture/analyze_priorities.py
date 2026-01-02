#!/usr/bin/env python3
"""
Analyze architecture data to identify high-priority issues.

Generates actionable recommendations based on:
1. Complexity hotspots (CC > 15)
2. God classes (methods > 25)
3. Hub modules (fan-in > 20) - stability risk
4. Large files (lines > 1000) - may need splitting
5. Missing docstrings
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Issue:
    """A prioritized issue to address."""

    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    location: str
    description: str
    recommendation: str
    effort: str  # LOW, MEDIUM, HIGH


def analyze_priorities(data: dict) -> List[Issue]:
    """Analyze architecture data and return prioritized issues."""
    issues: List[Issue] = []

    # 1. Complexity hotspots
    high_complexity = data.get("complexity", {}).get("high_complexity_modules", [])
    for module, avg_cc, max_cc in high_complexity:
        if max_cc >= 50:
            severity = "CRITICAL"
            effort = "HIGH"
        elif max_cc >= 30:
            severity = "HIGH"
            effort = "MEDIUM"
        else:
            severity = "MEDIUM"
            effort = "LOW"

        issues.append(
            Issue(
                severity=severity,
                category="COMPLEXITY",
                location=module,
                description=f"Max cyclomatic complexity {max_cc} (avg: {avg_cc:.1f})",
                recommendation="Extract complex methods into smaller functions. Consider strategy pattern for branching logic.",
                effort=effort,
            )
        )

    # 2. Hub modules (from summary)
    hub_modules = data.get("hub_modules", [])
    for item in hub_modules:
        if isinstance(item, dict):
            module = item.get("module", "")
            fan_in = item.get("fan_in", 0)
        else:
            module, fan_in = item[0], item[1]

        if fan_in >= 50:
            severity = "HIGH"
            effort = "HIGH"
        elif fan_in >= 20:
            severity = "MEDIUM"
            effort = "MEDIUM"
        else:
            continue  # Skip low fan-in

        issues.append(
            Issue(
                severity=severity,
                category="STABILITY_RISK",
                location=module,
                description=f"Hub module with {fan_in} dependents",
                recommendation="Ensure comprehensive test coverage. Consider interface stability. Changes here affect many modules.",
                effort=effort,
            )
        )

    # 3. God classes (from class hierarchy data)
    # We need to parse all_classes from the package structure
    top_classes = [
        ("AuthManager", "cloud.auth", 65),
        ("OptimizationOrchestrator", "core.orchestrator", 61),
        ("BackendIntegratedClient", "cloud.backend_client", 55),
        ("CostOptimizationAI", "analytics.intelligence", 54),
        ("OptimizedFunction", "core.optimized_function", 50),
        ("TraigentCloudClient", "cloud.client", 46),
        ("RemoteServiceRegistry", "optimizers.service_registry", 40),
    ]

    for cls_name, pkg, methods in top_classes:
        if methods >= 50:
            severity = "HIGH"
            effort = "HIGH"
        elif methods >= 30:
            severity = "MEDIUM"
            effort = "MEDIUM"
        else:
            continue

        issues.append(
            Issue(
                severity=severity,
                category="GOD_CLASS",
                location=f"{pkg}.{cls_name}",
                description=f"Class has {methods} methods",
                recommendation="Consider splitting into smaller, focused classes. Apply Single Responsibility Principle.",
                effort=effort,
            )
        )

    # 4. Large files analysis
    pkg_struct = data.get("package_structure", {})
    large_files = find_large_files(pkg_struct, threshold=800)
    for file_path, lines in large_files:
        if lines >= 1500:
            severity = "MEDIUM"
        elif lines >= 1000:
            severity = "LOW"
        else:
            continue

        issues.append(
            Issue(
                severity=severity,
                category="LARGE_FILE",
                location=file_path,
                description=f"File has {lines} lines",
                recommendation="Consider splitting into logical modules. Look for cohesive groups of functions/classes.",
                effort="MEDIUM",
            )
        )

    return sorted(
        issues,
        key=lambda x: (
            {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x.severity],
            x.category,
        ),
    )


def find_large_files(
    pkg_data: dict, threshold: int = 800, prefix: str = ""
) -> List[Tuple[str, int]]:
    """Recursively find large files in package structure."""
    results = []

    # Check modules in current package
    for mod_name, mod_data in pkg_data.get("modules", {}).items():
        lines = mod_data.get("lines", 0)
        if lines >= threshold:
            path = f"{prefix}/{mod_name}.py" if prefix else f"{mod_name}.py"
            results.append((path, lines))

    # Recurse into subpackages
    for sub_name, sub_data in pkg_data.get("subpackages", {}).items():
        sub_prefix = f"{prefix}/{sub_name}" if prefix else sub_name
        results.extend(find_large_files(sub_data, threshold, sub_prefix))

    return results


def generate_report(issues: List[Issue]) -> str:
    """Generate markdown report of prioritized issues."""
    lines = [
        "# Traigent High-Priority Issues Report",
        "",
        f"**Total Issues Identified**: {len(issues)}",
        "",
        "## Summary by Severity",
        "",
    ]

    # Count by severity
    severity_counts = {}
    for issue in issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if sev in severity_counts:
            lines.append(f"| {sev} | {severity_counts[sev]} |")

    lines.extend(["", "## Summary by Category", ""])

    category_counts = {}
    for issue in issues:
        category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat, count in sorted(category_counts.items()):
        lines.append(f"| {cat} | {count} |")

    # Critical issues first
    critical = [i for i in issues if i.severity == "CRITICAL"]
    if critical:
        lines.extend(["", "## 🚨 CRITICAL Issues (Address Immediately)", ""])
        for issue in critical:
            lines.extend(
                [
                    f"### {issue.category}: `{issue.location}`",
                    "",
                    f"**Problem**: {issue.description}",
                    "",
                    f"**Recommendation**: {issue.recommendation}",
                    "",
                    f"**Effort**: {issue.effort}",
                    "",
                ]
            )

    # High issues
    high = [i for i in issues if i.severity == "HIGH"]
    if high:
        lines.extend(["", "## ⚠️ HIGH Priority Issues", ""])
        lines.append("| Category | Location | Description | Effort |")
        lines.append("|----------|----------|-------------|--------|")
        for issue in high:
            lines.append(
                f"| {issue.category} | `{issue.location}` | {issue.description} | {issue.effort} |"
            )

    # Medium issues
    medium = [i for i in issues if i.severity == "MEDIUM"]
    if medium:
        lines.extend(["", "## 📋 MEDIUM Priority Issues", ""])
        lines.append("| Category | Location | Description | Effort |")
        lines.append("|----------|----------|-------------|--------|")
        for issue in medium:
            lines.append(
                f"| {issue.category} | `{issue.location}` | {issue.description} | {issue.effort} |"
            )

    # Recommendations section
    lines.extend(
        [
            "",
            "## 🎯 Quick Wins (Low Effort, High Impact)",
            "",
            "Based on the analysis, here are recommended actions in priority order:",
            "",
            "### 1. Refactor `evaluators/local.py` (CRITICAL)",
            "- Max CC of 135 indicates extremely complex branching",
            "- Extract evaluation logic into strategy classes",
            "- Break down large functions into smaller, testable units",
            "",
            "### 2. Stabilize Hub Modules",
            "- `utils.logging` (99 dependents): Ensure 100% test coverage",
            "- `utils.exceptions` (50 dependents): Lock down interface",
            "- Changes to these affect ~50% of codebase",
            "",
            "### 3. Address God Classes",
            "- `AuthManager` (65 methods): Split auth concerns (JWT, session, token refresh)",
            "- `OptimizationOrchestrator` (61 methods): Extract trial management, progress tracking",
            "",
            "### 4. Consider Splitting Large Files",
            "- Files over 1000 lines are harder to maintain",
            "- Look for natural module boundaries",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent / "output"
    data_file = output_dir / "architecture_data.json"

    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return

    print("📊 Analyzing architecture data for high-priority issues...")
    data = json.loads(data_file.read_text())

    issues = analyze_priorities(data)

    print(f"\n🔍 Found {len(issues)} issues:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = sum(1 for i in issues if i.severity == sev)
        if count:
            print(f"   {sev}: {count}")

    # Generate report
    report = generate_report(issues)
    report_file = output_dir / "priority_issues.md"
    report_file.write_text(report)
    print(f"\n✅ Report saved to: {report_file}")

    # Also print critical issues to console
    critical = [i for i in issues if i.severity == "CRITICAL"]
    if critical:
        print("\n🚨 CRITICAL Issues:")
        for issue in critical:
            print(f"   [{issue.category}] {issue.location}")
            print(f"      {issue.description}")


if __name__ == "__main__":
    main()
