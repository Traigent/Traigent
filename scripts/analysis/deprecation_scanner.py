#!/usr/bin/env python3
"""
Comprehensive deprecation and technical debt scanner for all Python files.
Generates a detailed tracking table with parallel processing.
"""

import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Patterns to search for
PATTERNS = {
    "deprecated": [
        r"@deprecated",
        r"DEPRECATED",
        r"deprecat",
        r"#\s*deprecated",
    ],
    "backward_compat": [
        r"backward.?compat",
        r"legacy",
        r"old.?api",
        r"local_mode\(",
        r"TRAIGENT_BACKEND_URL",
        r"TRAIGENT_API_KEY",
    ],
    "conditional_import": [
        r"try:\s*import",
        r"try:\s*from",
        r"except\s*Import",
        r"except\s*ModuleNotFound",
    ],
    "security": [
        r"TODO.*auth",
        r"TODO.*security",
        r"FIXME.*auth",
        r"FIXME.*security",
        r"auth.*TODO",
        r"security.*TODO",
        r"credential",
        r"password",
        r"secret",
        r"token",
    ],
    "critical": [
        r"CRITICAL",
        r"SECURITY",
        r"VULNERABILITY",
        r"XXX",
        r"HACK",
        r"FIXME",
    ],
    "hardcoded": [
        r"localhost:\d+",
        r"127\.0\.0\.1:\d+",
        r"http://localhost",
        r"DEFAULT_LOCAL_URL",
    ],
    "todo": [
        r"\bTODO\b",
        r"\bFIXME\b",
        r"\bHACK\b",
        r"\bXXX\b",
    ],
}


def analyze_file(file_path: str) -> Dict:
    """Analyze a single Python file for deprecation patterns."""
    result = {
        "path": file_path,
        "has_deprecated": False,
        "has_backward": False,
        "has_conditional": False,
        "has_security": False,
        "has_critical": False,
        "has_hardcoded": False,
        "has_todo": False,
        "findings": [],
        "line_count": 0,
        "size_kb": 0,
    }

    try:
        # Get file stats
        stat = os.stat(file_path)
        result["size_kb"] = round(stat.st_size / 1024, 2)

        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
            result["line_count"] = len(lines)

            # Check each pattern category
            for category, patterns in PATTERNS.items():
                for pattern in patterns:
                    if re.search(
                        pattern,
                        content,
                        re.IGNORECASE if category != "conditional_import" else 0,
                    ):
                        if category == "deprecated":
                            result["has_deprecated"] = True
                        elif category == "backward_compat":
                            result["has_backward"] = True
                        elif category == "conditional_import":
                            result["has_conditional"] = True
                        elif category == "security":
                            result["has_security"] = True
                        elif category == "critical":
                            result["has_critical"] = True
                        elif category == "hardcoded":
                            result["has_hardcoded"] = True
                        elif category == "todo":
                            result["has_todo"] = True

                        # Find specific occurrences
                        for i, line in enumerate(lines, 1):
                            if re.search(
                                pattern,
                                line,
                                (
                                    re.IGNORECASE
                                    if category != "conditional_import"
                                    else 0
                                ),
                            ):
                                finding = f"Line {i}: {pattern} - {line.strip()[:100]}"
                                if finding not in result["findings"]:
                                    result["findings"].append(finding)
                                    if (
                                        len(result["findings"]) >= 5
                                    ):  # Limit findings per file
                                        break
    except Exception as e:
        result["findings"].append(f"Error analyzing file: {str(e)}")

    return result


def determine_risk_level(result: Dict) -> str:
    """Determine risk level based on findings."""
    if result["has_critical"] or result["has_security"]:
        return "🔴 Critical"
    elif result["has_deprecated"] and result["has_backward"]:
        return "🟡 High"
    elif result["has_deprecated"] or result["has_backward"] or result["has_hardcoded"]:
        return "🟠 Medium"
    else:
        return "🟢 Low"


def determine_status(file_path: str, result: Dict) -> str:
    """Determine file status."""
    if "deprecated" in file_path.lower() or "archive" in file_path.lower():
        return "📦 Archive"
    elif "test" in file_path.lower():
        return "🧪 Test"
    elif "example" in file_path.lower():
        return "📝 Example"
    else:
        return "✅ Active"


def scan_directory(directory: str) -> List[Dict]:
    """Scan all Python files in a directory."""
    results = []
    py_files = list(Path(directory).rglob("*.py"))

    print(f"Scanning {len(py_files)} files in {directory}...")

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_file = {
            executor.submit(analyze_file, str(f)): str(f) for f in py_files
        }

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                result["risk"] = determine_risk_level(result)
                result["status"] = determine_status(file_path, result)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return results


def main():
    """Main scanning function."""
    directories = [
        "traigent",
        "tests",
        "examples",
        "scripts",
        "playground",
        "walkthrough",
        "deprecated_auth_backup",
    ]

    all_results = []

    print("Starting comprehensive deprecation scan...")
    print("=" * 80)

    for directory in directories:
        if os.path.exists(directory):
            results = scan_directory(directory)
            all_results.extend(results)
            print(f"✓ {directory}: {len(results)} files scanned")

    print("=" * 80)
    print(f"Total files scanned: {len(all_results)}")

    # Sort by risk level
    risk_order = {"🔴 Critical": 0, "🟡 High": 1, "🟠 Medium": 2, "🟢 Low": 3}
    all_results.sort(key=lambda x: (risk_order.get(x["risk"], 4), x["path"]))

    # Generate CSV report
    csv_file = "reports/development/deprecation_scan_results.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "path",
            "status",
            "risk",
            "has_deprecated",
            "has_backward",
            "has_conditional",
            "has_security",
            "has_critical",
            "has_hardcoded",
            "has_todo",
            "line_count",
            "size_kb",
            "findings",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            result["findings"] = "; ".join(
                result["findings"][:3]
            )  # Limit findings in CSV
            writer.writerow(result)

    print(f"\n✅ Results saved to {csv_file}")

    # Generate summary statistics
    stats = {
        "total_files": len(all_results),
        "critical_risk": sum(1 for r in all_results if r["risk"] == "🔴 Critical"),
        "high_risk": sum(1 for r in all_results if r["risk"] == "🟡 High"),
        "medium_risk": sum(1 for r in all_results if r["risk"] == "🟠 Medium"),
        "low_risk": sum(1 for r in all_results if r["risk"] == "🟢 Low"),
        "with_deprecated": sum(1 for r in all_results if r["has_deprecated"]),
        "with_backward": sum(1 for r in all_results if r["has_backward"]),
        "with_conditional": sum(1 for r in all_results if r["has_conditional"]),
        "with_security": sum(1 for r in all_results if r["has_security"]),
        "with_critical": sum(1 for r in all_results if r["has_critical"]),
        "with_hardcoded": sum(1 for r in all_results if r["has_hardcoded"]),
        "with_todo": sum(1 for r in all_results if r["has_todo"]),
    }

    print("\n📊 Summary Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Save summary
    with open("reports/development/deprecation_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
