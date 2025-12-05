#!/usr/bin/env python3
"""
Find old, backup, and potentially obsolete files in the codebase.
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path

# Patterns that indicate old/backup files
OLD_FILE_PATTERNS = [
    r"_old\.py$",
    r"_backup\.py$",
    r"_copy\.py$",
    r"\.bak$",
    r"\.old$",
    r"\.backup$",
    r"\.copy$",
    r"_v\d+\.py$",  # Version numbered files like _v1.py, _v2.py
    r"\.orig$",
    r"~$",  # Emacs backup files
    r"\.tmp$",
    r"\.temp$",
    r"test_.*_old\.py$",
    r"old_.*\.py$",
    r"backup_.*\.py$",
    r"copy_of_.*\.py$",
]

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


def should_check_file(file_path: Path) -> bool:
    """Check if file should be scanned for old file patterns."""
    # Skip if in excluded directory
    for part in file_path.parts:
        if part in SKIP_DIRS:
            return False
    return True


def find_old_files(project_root: Path) -> list:
    """Find all files matching old/backup patterns."""
    old_files = []

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)

        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        # Skip if current directory should be excluded
        if not should_check_file(root_path):
            continue

        # Check each file
        for file in files:
            file_path = root_path / file

            # Check against patterns
            for pattern in OLD_FILE_PATTERNS:
                if re.search(pattern, file, re.IGNORECASE):
                    try:
                        stat = file_path.stat()
                        old_files.append(
                            {
                                "path": str(file_path.relative_to(project_root)),
                                "pattern": pattern,
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                                "age_days": (
                                    datetime.now()
                                    - datetime.fromtimestamp(stat.st_mtime)
                                ).days,
                            }
                        )
                        break  # Don't match multiple patterns for same file
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    return old_files


def generate_report(old_files: list, output_dir: Path):
    """Generate report of old files."""
    # Sort by pattern and size
    old_files.sort(key=lambda x: (x["pattern"], -x["size"]))

    # Write CSV
    csv_path = output_dir / "old_backup_files.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if old_files:
            writer = csv.DictWriter(f, fieldnames=old_files[0].keys())
            writer.writeheader()
            writer.writerows(old_files)

    # Write Markdown report
    md_path = output_dir / "old_backup_files_report.md"
    with open(md_path, "w") as f:
        f.write("# Old and Backup Files Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total old/backup files found: {len(old_files)}\n\n")

        # Summary by pattern
        pattern_counts = {}
        total_size = 0
        for file in old_files:
            pattern = file["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total_size += file["size"]

        f.write("## Summary by Pattern\n\n")
        f.write("| Pattern | Count | Description |\n")
        f.write("|---------|-------|--------------|\n")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            desc = get_pattern_description(pattern)
            f.write(f"| `{pattern}` | {count} | {desc} |\n")

        f.write(f"\n**Total size**: {total_size / (1024*1024):.2f} MB\n")

        # Files by age
        f.write("\n## Files by Age\n\n")
        age_buckets = {
            "Very old (>365 days)": [],
            "Old (180-365 days)": [],
            "Recent (90-180 days)": [],
            "New (<90 days)": [],
        }

        for file in old_files:
            age = file["age_days"]
            if age > 365:
                age_buckets["Very old (>365 days)"].append(file)
            elif age > 180:
                age_buckets["Old (180-365 days)"].append(file)
            elif age > 90:
                age_buckets["Recent (90-180 days)"].append(file)
            else:
                age_buckets["New (<90 days)"].append(file)

        for bucket, files in age_buckets.items():
            if files:
                f.write(f"\n### {bucket} ({len(files)} files)\n\n")
                for file in sorted(files, key=lambda x: -x["age_days"])[
                    :10
                ]:  # Show top 10
                    f.write(
                        f"- `{file['path']}` ({file['age_days']} days old, {file['size']/1024:.1f} KB)\n"
                    )
                if len(files) > 10:
                    f.write(f"- ... and {len(files) - 10} more\n")

        # Largest files
        f.write("\n## Largest Old/Backup Files\n\n")
        f.write("| File | Size | Age (days) | Pattern |\n")
        f.write("|------|------|------------|----------|\n")
        for file in sorted(old_files, key=lambda x: -x["size"])[:20]:
            size_mb = file["size"] / (1024 * 1024)
            f.write(
                f"| `{file['path']}` | {size_mb:.2f} MB | {file['age_days']} | `{file['pattern']}` |\n"
            )

        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write(
            "1. **Very old files (>365 days)**: Consider deletion if no longer needed\n"
        )
        f.write("2. **Version numbered files**: Keep only the latest version\n")
        f.write("3. **Backup files**: Move to proper backup location or delete\n")
        f.write("4. **Large files**: Prioritize for cleanup to save space\n")
        f.write("5. **Test files**: Verify if old test files are still relevant\n")

        f.write("\n## Cleanup Commands\n\n")
        f.write("```bash\n")
        f.write("# Preview files that would be deleted (dry run)\n")
        f.write(
            "find . -name '*_old.py' -o -name '*_backup.py' -o -name '*.bak' | head -20\n\n"
        )
        f.write("# Delete specific pattern (use with caution!)\n")
        f.write("# find . -name '*.bak' -type f -delete\n\n")
        f.write("# Interactive deletion (safer)\n")
        f.write("# find . -name '*_old.py' -type f -exec rm -i {} \\;\n")
        f.write("```\n")


def get_pattern_description(pattern: str) -> str:
    """Get human-readable description for pattern."""
    descriptions = {
        r"_old\.py$": "Python files ending with _old",
        r"_backup\.py$": "Python files ending with _backup",
        r"_copy\.py$": "Python files ending with _copy",
        r"\.bak$": "Backup files with .bak extension",
        r"\.old$": "Files with .old extension",
        r"\.backup$": "Files with .backup extension",
        r"\.copy$": "Files with .copy extension",
        r"_v\d+\.py$": "Version numbered Python files",
        r"\.orig$": "Original files (often from merges)",
        r"~$": "Editor backup files",
        r"\.tmp$": "Temporary files",
        r"\.temp$": "Temporary files",
        r"test_.*_old\.py$": "Old test files",
        r"old_.*\.py$": "Python files starting with old_",
        r"backup_.*\.py$": "Python files starting with backup_",
        r"copy_of_.*\.py$": "Python files starting with copy_of_",
    }
    return descriptions.get(pattern, "Other old/backup pattern")


def main():
    """Main execution."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "project_review"
    output_dir.mkdir(exist_ok=True)

    print(f"Scanning for old/backup files in: {project_root}")

    # Find old files
    old_files = find_old_files(project_root)

    print(f"Found {len(old_files)} old/backup files")

    if old_files:
        # Generate report
        generate_report(old_files, output_dir)

        print("\nReports generated:")
        print(f"- CSV: {output_dir / 'old_backup_files.csv'}")
        print(f"- Markdown: {output_dir / 'old_backup_files_report.md'}")

        # Show summary
        total_size = sum(f["size"] for f in old_files)
        print(f"\nTotal size of old/backup files: {total_size / (1024*1024):.2f} MB")
        print(
            f"Average age: {sum(f['age_days'] for f in old_files) / len(old_files):.0f} days"
        )


if __name__ == "__main__":
    main()
