#!/usr/bin/env python3
"""
Find deprecated patterns in the codebase after the decorator->attribute refactoring
"""

import csv
import datetime
import re
from pathlib import Path
from typing import Dict, List

# Patterns to search for
DEPRECATED_PATTERNS = [
    # Direct references to old class name
    (r"DecoratorBasedProvider", "Replace with AttributeBasedProvider", "high"),
    # Injection mode references
    (
        r'injection_mode\s*=\s*["\'"]decorator["\'"]',
        'Replace with injection_mode="attribute"',
        "high",
    ),
    (
        r'injection_mode\s*:\s*["\'"]decorator["\'"]',
        'Replace with injection_mode: "attribute"',
        "high",
    ),
    # Documentation references (less critical)
    (r"decorator\s+injection\s+mode", 'Update to "attribute injection mode"', "medium"),
    (r"decorator-based\s+injection", 'Update to "attribute-based injection"', "medium"),
    # Import statements
    (
        r"from\s+.*\s+import\s+.*DecoratorBasedProvider",
        "Update import to AttributeBasedProvider",
        "high",
    ),
    # Deprecated markers
    (r"@deprecated", "Review deprecation - may be unrelated", "low"),
    (r"#\s*TODO.*decorator", "Review TODO related to decorator pattern", "low"),
    # Old file patterns
    (r"_old\.py$", "Old file - consider removal", "medium"),
    (r"_backup\.py$", "Backup file - consider removal", "medium"),
    (r"_copy\.py$", "Copy file - consider removal", "medium"),
    (r"\.bak$", "Backup file - consider removal", "medium"),
]

# File extensions to scan
SCAN_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml"}

# Directories to skip
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "venv",
    "traigent_test_env",
    "node_modules",
    ".venv",
    "env",
    ".env",
    "htmlcov",
    ".coverage",
    "dist",
    "build",
    ".egg-info",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".hypothesis",
    "project_review",
}


def should_scan_file(file_path: Path) -> bool:
    """Check if file should be scanned"""
    # Skip if in excluded directory
    for part in file_path.parts:
        if part in SKIP_DIRS:
            return False

    # Check extension
    return file_path.suffix in SCAN_EXTENSIONS or file_path.name in {
        "Makefile",
        "Dockerfile",
    }


def scan_file(file_path: Path) -> List[Dict[str, any]]:
    """Scan a single file for deprecated patterns"""
    findings = []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()
    except Exception:
        # Skip files that can't be read
        return findings

    # Check filename patterns
    for pattern, description, severity in DEPRECATED_PATTERNS:
        if re.search(pattern, str(file_path)):
            findings.append(
                {
                    "file": str(file_path),
                    "line": 0,
                    "pattern": pattern,
                    "match": file_path.name,
                    "description": description,
                    "severity": severity,
                    "context": f"Filename: {file_path.name}",
                }
            )

    # Check file content
    for line_num, line in enumerate(lines, 1):
        for pattern, description, severity in DEPRECATED_PATTERNS:
            matches = re.finditer(pattern, line)
            for match in matches:
                findings.append(
                    {
                        "file": str(file_path),
                        "line": line_num,
                        "pattern": pattern,
                        "match": match.group(0),
                        "description": description,
                        "severity": severity,
                        "context": line.strip(),
                    }
                )

    return findings


def scan_project(project_root: Path) -> List[Dict[str, any]]:
    """Scan entire project for deprecated patterns"""
    all_findings = []
    files_scanned = 0

    for file_path in project_root.rglob("*"):
        if file_path.is_file() and should_scan_file(file_path):
            findings = scan_file(file_path)
            all_findings.extend(findings)
            files_scanned += 1

            if files_scanned % 100 == 0:
                print(f"Scanned {files_scanned} files...")

    return all_findings


def write_report(findings: List[Dict[str, any]], output_dir: Path):
    """Write findings to CSV and markdown reports"""
    # Group by severity
    by_severity = {"high": [], "medium": [], "low": []}
    for finding in findings:
        by_severity[finding["severity"]].append(finding)

    # Write CSV
    csv_path = output_dir / "deprecated_patterns.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if findings:
            writer = csv.DictWriter(f, fieldnames=findings[0].keys())
            writer.writeheader()
            writer.writerows(findings)

    # Write Markdown report
    md_path = output_dir / "deprecated_patterns_report.md"
    with open(md_path, "w") as f:
        f.write("# Deprecated Patterns Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"Total findings: {len(findings)}\n\n")

        # Summary by severity
        f.write("## Summary by Severity\n\n")
        f.write("| Severity | Count |\n")
        f.write("|----------|-------|\n")
        for severity in ["high", "medium", "low"]:
            count = len(by_severity[severity])
            f.write(f"| {severity.upper()} | {count} |\n")

        # Detailed findings by severity
        for severity in ["high", "medium", "low"]:
            findings_list = by_severity[severity]
            if findings_list:
                f.write(f"\n## {severity.upper()} Severity Findings\n\n")

                # Group by file
                by_file = {}
                for finding in findings_list:
                    file_path = finding["file"]
                    if file_path not in by_file:
                        by_file[file_path] = []
                    by_file[file_path].append(finding)

                for file_path, file_findings in sorted(by_file.items()):
                    f.write(f"\n### {file_path}\n\n")
                    for finding in file_findings:
                        f.write(
                            f"- **Line {finding['line']}**: {finding['description']}\n"
                        )
                        f.write(f"  - Match: `{finding['match']}`\n")
                        f.write(f"  - Context: `{finding['context']}`\n")

    return csv_path, md_path


def main():
    """Main execution"""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "project_review"
    output_dir.mkdir(exist_ok=True)

    print("Scanning project for deprecated patterns...")
    print(f"Project root: {project_root}")

    # Scan project
    findings = scan_project(project_root)

    print(f"\nFound {len(findings)} deprecated patterns")

    # Write reports
    csv_path, md_path = write_report(findings, output_dir)

    print("\nReports written:")
    print(f"- CSV: {csv_path}")
    print(f"- Markdown: {md_path}")

    # Show summary
    by_severity = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        by_severity[finding["severity"]] += 1

    print("\nSummary:")
    for severity, count in by_severity.items():
        if count > 0:
            print(f"- {severity.upper()}: {count} findings")


if __name__ == "__main__":
    main()
